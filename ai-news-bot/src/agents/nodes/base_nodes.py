"""
LangGraph 기본 노드 클래스들

이 모듈은 LangGraph 워크플로우에서 사용되는 기본 노드 클래스들을 정의합니다.
각 노드 타입별로 추상 클래스를 제공하여 일관된 인터페이스를 보장합니다.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging
from langchain_core.documents import Document


class BaseRetrieverNode(ABC):
    """검색 노드의 기본 클래스"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"RetrieverNode.{name}")
    
    @abstractmethod
    def retrieve(self, state: Dict[str, Any]) -> List[Document]:
        """검색 수행 - 하위 클래스에서 구현해야 함"""
        pass
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """노드 실행 - 공통 로직 처리"""
        print(f"=== {self.name} 노드 실행 ===")
        
        try:
            # 검색 수행
            new_docs = self.retrieve(state)
            
            # 기존 컨텍스트와 병합
            prev_context = state.get("context", [])
            prev_context = [doc if hasattr(doc, 'page_content') else Document(page_content=str(doc)) for doc in prev_context]
            new_docs = [doc if hasattr(doc, 'page_content') else Document(page_content=str(doc)) for doc in new_docs]
            
            # 중복 제거
            all_docs = self._deduplicate_documents(prev_context + new_docs)
            
            # 상태 업데이트
            new_state = dict(state)
            new_state["context"] = all_docs
            new_state["step"] = self.name
            new_state.setdefault("trace", []).append((self.name, [doc.page_content[:80] + ' ...' for doc in new_docs]))
            
            # 로깅
            self.logger.info(f"{len(new_docs)}개 문서 검색됨")
            for i, doc in enumerate(new_docs[:3]):
                content = getattr(doc, 'page_content', str(doc))
                print(f"  [{i}] {content[:100]} ...")
            
            return new_state
            
        except Exception as e:
            self.logger.error(f"검색 중 오류 발생: {e}")
            return state
    
    def _deduplicate_documents(self, docs: List[Document]) -> List[Document]:
        """문서 중복 제거"""
        seen = set()
        unique_docs = []
        for doc in docs:
            key = getattr(doc, "page_content", str(doc))
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)
        return unique_docs
    
    def _get_user_question(self, state: Dict[str, Any]) -> str:
        """상태에서 사용자 질문 추출"""
        for msg in reversed(state["messages"]):
            if getattr(msg, "type", None) == "human":
                return msg.content
        raise ValueError("user 메시지가 없습니다.")


class BaseEvaluatorNode(ABC):
    """평가 노드의 기본 클래스"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"EvaluatorNode.{name}")
    
    @abstractmethod
    def evaluate(self, state: Dict[str, Any]) -> str:
        """평가 수행 - 하위 클래스에서 구현해야 함"""
        pass
    
    def __call__(self, state: Dict[str, Any]) -> str:
        """노드 실행 - 공통 로직 처리"""
        print(f"=== {self.name} 노드 실행 ===")
        try:
            result = self.evaluate(state)
            self.logger.info(f"평가 결과: {result}")
            return result
        except Exception as e:
            self.logger.error(f"평가 중 오류 발생: {e}")
            return "insufficient"  # 오류 시 기본값
    
    def _get_user_question(self, state: Dict[str, Any]) -> str:
        """상태에서 사용자 질문 추출"""
        for msg in reversed(state["messages"]):
            if getattr(msg, "type", None) == "human":
                return msg.content
        raise ValueError("user 메시지가 없습니다.")


class BaseGeneratorNode(ABC):
    """생성 노드의 기본 클래스"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"GeneratorNode.{name}")
    
    @abstractmethod
    def generate(self, state: Dict[str, Any]) -> str:
        """생성 수행 - 하위 클래스에서 구현해야 함"""
        pass
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """노드 실행 - 공통 로직 처리"""
        print(f"=== {self.name} 노드 실행 ===")
        try:
            result = self.generate(state)
            
            new_state = dict(state)
            new_state["answer"] = result
            new_state.setdefault("messages", []).append({"role": "assistant", "content": result})
            
            self.logger.info("생성 완료")
            return new_state
            
        except Exception as e:
            self.logger.error(f"생성 중 오류 발생: {e}")
            error_answer = "죄송합니다. 답변 생성 중 오류가 발생했습니다. 다시 시도해주세요."
            new_state = dict(state)
            new_state["answer"] = error_answer
            new_state.setdefault("messages", []).append({"role": "assistant", "content": error_answer})
            return new_state
    
    def _get_user_question(self, state: Dict[str, Any]) -> str:
        """상태에서 사용자 질문 추출"""
        for msg in reversed(state["messages"]):
            if getattr(msg, "type", None) == "human":
                return msg.content
        raise ValueError("user 메시지가 없습니다.")


class BaseClarificationNode:
    def __init__(self, name="clarification"):
        self.name = name
    def clarify(self, state):
        raise NotImplementedError

class BaseRecommendNode:
    def __init__(self, name="recommend_articles"):
        self.name = name
    def recommend(self, state):
        raise NotImplementedError 