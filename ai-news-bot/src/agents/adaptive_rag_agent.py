import pickle
from langgraph.graph import StateGraph, START, END
from typing import List, Tuple, Dict, Callable, Optional, Any
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langgraph.graph import MessagesState
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import logging
from langchain_core.documents import Document
import re
from abc import ABC, abstractmethod

# 개선된 Grader 프롬프트
GRADE_PROMPT = """
당신은 사용자의 질문에 대해 주어진 컨텍스트가 충분한 정보를 제공하는지 평가하는 전문가입니다.

[사용자 질문]
{question}

[제공된 컨텍스트]
{context}

평가 기준:
1. 컨텍스트가 질문과 직접적으로 관련된 구체적인 정보를 포함하는가?
2. 컨텍스트만으로 질문에 대한 완전하고 정확한 답변이 가능한가?
3. 컨텍스트의 정보가 최신이고 신뢰할 수 있는가?

다음 중 하나로만 답변하세요:
- "SUFFICIENT": 컨텍스트가 충분한 정보를 제공함
- "INSUFFICIENT": 컨텍스트가 부족하거나 관련성이 낮음

답변: """

# 검색 쿼리 최적화 프롬프트
QUERY_OPTIMIZATION_PROMPT = """
사용자의 질문을 인터넷 검색에 최적화된 키워드로 변환해주세요.

[원본 질문]
{original_question}

[현재 컨텍스트 정보]
{context_info}

지침:
1. 검색 키워드는 구체적이고 명확해야 합니다
2. 질문의 핵심 키워드를 포함해야 합니다
3. 너무 일반적이거나 모호한 용어는 피하세요
4. 최신 정보를 찾기 위해 연도나 "최신", "2024" 등의 키워드를 추가하세요
5. 10단어 이내로 제한하세요

최적화된 검색 쿼리: """

# ==================== 추상 클래스 및 기본 노드들 ====================

class BaseRetrieverNode(ABC):
    """검색 노드의 기본 클래스"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"RetrieverNode.{name}")
    
    @abstractmethod
    def retrieve(self, state: Dict[str, Any]) -> List[Document]:
        """검색 수행"""
        pass
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """노드 실행"""
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

class BaseEvaluatorNode(ABC):
    """평가 노드의 기본 클래스"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"EvaluatorNode.{name}")
    
    @abstractmethod
    def evaluate(self, state: Dict[str, Any]) -> str:
        """평가 수행"""
        pass
    
    def __call__(self, state: Dict[str, Any]) -> str:
        """노드 실행"""
        print(f"=== {self.name} 노드 실행 ===")
        try:
            result = self.evaluate(state)
            self.logger.info(f"평가 결과: {result}")
            return result
        except Exception as e:
            self.logger.error(f"평가 중 오류 발생: {e}")
            return "insufficient"  # 오류 시 기본값

class BaseGeneratorNode(ABC):
    """생성 노드의 기본 클래스"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"GeneratorNode.{name}")
    
    @abstractmethod
    def generate(self, state: Dict[str, Any]) -> str:
        """생성 수행"""
        pass
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """노드 실행"""
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

# ==================== 구체적인 노드 구현 ====================

class ArticleSearchNode(BaseRetrieverNode):
    """특정 기사 검색 노드"""
    
    def __init__(self, ensemble_retriever: EnsembleRetriever):
        super().__init__("article_search")
        self.ensemble_retriever = ensemble_retriever
    
    def retrieve(self, state: Dict[str, Any]) -> List[Document]:
        user_question = self._get_user_question(state)
        article_id = state.get("article_id")
        
        if not article_id:
            print("[article_search_node] article_id가 없어 빈 컨텍스트 반환")
            return []
        
        # 특정 기사에서만 검색
        faiss_retriever = self.ensemble_retriever.retrievers[0]
        all_docs = faiss_retriever.vectorstore.docstore._dict.values()
        article_docs = [doc for doc in all_docs if doc.metadata.get("article_id") == article_id]
        
        if article_docs:
            # 해당 기사만으로 임시 retriever 생성
            faiss_vectorstore, bm25_retriever = create_vectorstores(article_docs, faiss_retriever.vectorstore.embedding_function)
            temp_ensemble = build_ensemble_retriever(faiss_vectorstore, bm25_retriever)
            context = temp_ensemble.invoke(user_question)
            print(f"[article_search_node] 특정 기사에서 {len(context)}개 문서 검색됨")
            return context
        else:
            print(f"[article_search_node] 해당 기사를 찾을 수 없음: {article_id}")
            return []
    
    def _get_user_question(self, state: Dict[str, Any]) -> str:
        for msg in reversed(state["messages"]):
            if getattr(msg, "type", None) == "human":
                return msg.content
        raise ValueError("user 메시지가 없습니다.")

class AllSearchNode(BaseRetrieverNode):
    """전체 기사 검색 노드"""
    
    def __init__(self, ensemble_retriever: EnsembleRetriever):
        super().__init__("all_search")
        self.ensemble_retriever = ensemble_retriever
    
    def retrieve(self, state: Dict[str, Any]) -> List[Document]:
        user_question = self._get_user_question(state)
        context = self.ensemble_retriever.invoke(user_question)
        print(f"[all_search_node] 전체 기사에서 {len(context)}개 문서 검색됨")
        return context
    
    def _get_user_question(self, state: Dict[str, Any]) -> str:
        for msg in reversed(state["messages"]):
            if getattr(msg, "type", None) == "human":
                return msg.content
        raise ValueError("user 메시지가 없습니다.")

class WebSearchNode(BaseRetrieverNode):
    """인터넷 검색 노드"""
    
    def __init__(self):
        super().__init__("web_search")
    
    def retrieve(self, state: Dict[str, Any]) -> List[Document]:
        user_question = self._get_user_question(state)
        
        # 현재 컨텍스트 정보 추출 (검색 쿼리 최적화용)
        current_context = state.get("context", [])
        context_info = ""
        if current_context:
            context_texts = [getattr(doc, 'page_content', str(doc))[:200] for doc in current_context[:2]]
            context_info = " ".join(context_texts)
        
        # 검색 쿼리 최적화
        optimized_query = optimize_search_query(user_question, context_info)
        print(f"[web_search_node] 최적화된 검색 쿼리: {optimized_query}")
        
        context = web_search_func(optimized_query)
        print(f"[web_search_node] 웹 검색 결과: {len(context)}개")
        return context
    
    def _get_user_question(self, state: Dict[str, Any]) -> str:
        for msg in reversed(state["messages"]):
            if getattr(msg, "type", None) == "human":
                return msg.content
        raise ValueError("user 메시지가 없습니다.")

class ContextSufficiencyEvaluator(BaseEvaluatorNode):
    """컨텍스트 충분성 평가 노드"""
    
    def __init__(self):
        super().__init__("context_sufficiency_evaluator")
    
    def evaluate(self, state: Dict[str, Any]) -> str:
        user_question = self._get_user_question(state)
        context = state.get("context", [])
        
        # context가 비어있으면 무조건 insufficient
        if not context:
            print("[context_sufficiency_evaluator] context가 비어있음 -> insufficient")
            return "insufficient"
        
        if context and hasattr(context[0], "page_content"):
            context_text = "\n\n".join([doc.page_content for doc in context])
        else:
            context_text = "\n\n".join([str(c) for c in context])
        
        print(f"[context_sufficiency_evaluator] context 길이: {len(context_text)}")
        print(f"[context_sufficiency_evaluator] context 미리보기: {context_text[:200]} ...")
        
        # context가 너무 짧으면 insufficient로 판단
        if len(context_text.strip()) < 100:
            print("[context_sufficiency_evaluator] context가 너무 짧음 -> insufficient")
            return "insufficient"
        
        # LLM을 통한 엄격한 평가
        if context and context_sufficient_llm(user_question, context_text):
            print("[context_sufficiency_evaluator] LLM 판정: sufficient")
            return "sufficient"
        else:
            print("[context_sufficiency_evaluator] LLM 판정: insufficient")
            return "insufficient"
    
    def _get_user_question(self, state: Dict[str, Any]) -> str:
        for msg in reversed(state["messages"]):
            if getattr(msg, "type", None) == "human":
                return msg.content
        raise ValueError("user 메시지가 없습니다.")

class AnswerGenerator(BaseGeneratorNode):
    """답변 생성 노드"""
    
    def __init__(self):
        super().__init__("answer_generator")
    
    def generate(self, state: Dict[str, Any]) -> str:
        user_question = self._get_user_question(state)
        context = state.get("context", [])
        
        # context_text만 사용
        if context and hasattr(context[0], "page_content"):
            context_text = "\n\n".join([doc.page_content for doc in context])
        else:
            context_text = "\n\n".join([str(c) for c in context])
        
        print(f"[answer_generator] context_text 길이: {len(context_text)}")
        print(f"[answer_generator] context_text 미리보기: {context_text[:100]} ...")
        print(f"[answer_generator] user_question: {user_question}")
        
        # 컨텍스트가 없는 경우 처리
        if not context_text.strip() or len(context_text.strip()) < 50:
            prompt = f"""
[질문]
{user_question}

죄송합니다. 이 질문에 답변하기에 충분한 정보를 찾지 못했습니다. 
다른 질문을 해주시거나, 더 구체적으로 질문해주시면 도움을 드릴 수 있습니다.
"""
        else:
            prompt = f"""
[참고 기사/문서 내용]
{context_text}

[질문]
{user_question}

위 참고 내용을 바탕으로, 사용자의 질문에 대해 정확하고 친절하게 답변해 주세요.
답변 시 다음 지침을 따라주세요:
1. 참고 내용에 있는 정보만을 사용하여 답변하세요
2. 확실하지 않은 정보는 언급하지 마세요
3. 필요하다면 이모지를 사용해서 항목을 구분해주세요
4. 마크다운 문법(*, -, #)은 사용하지 마세요
"""
        
        print(f"[answer_generator] LLM 프롬프트 미리보기: {prompt[:100]} ...")
        
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text
    
    def _get_user_question(self, state: Dict[str, Any]) -> str:
        for msg in reversed(state["messages"]):
            if getattr(msg, "type", None) == "human":
                return msg.content
        raise ValueError("user 메시지가 없습니다.")

# ==================== 유틸리티 함수들 ====================

def context_sufficient_llm(question: str, context: str) -> bool:
    """개선된 컨텍스트 충분성 평가 함수"""
    grader_model = genai.GenerativeModel('models/gemini-2.0-flash')
    prompt = GRADE_PROMPT.format(question=question, context=context)
    
    print(f"[context_sufficient_llm] 질문: {question}")
    print(f"[context_sufficient_llm] 컨텍스트 길이: {len(context)}")
    print(f"[context_sufficient_llm] 컨텍스트 미리보기: {context[:200]} ...")
    
    try:
    response = grader_model.generate_content(prompt)
        answer = response.text.strip().upper()
    print(f"[context_sufficient_llm] LLM 판정 결과: '{answer}'")
    
        is_sufficient = "SUFFICIENT" in answer
    print(f"[context_sufficient_llm] 최종 판정: {is_sufficient}")
    return is_sufficient
    except Exception as e:
        print(f"[context_sufficient_llm] 오류 발생: {e}")
        return False

def optimize_search_query(original_question: str, context_info: str = "") -> str:
    """검색 쿼리 최적화 함수"""
    try:
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        prompt = QUERY_OPTIMIZATION_PROMPT.format(
            original_question=original_question,
            context_info=context_info
        )
        
        response = model.generate_content(prompt)
        optimized_query = response.text.strip()
        
        # 불필요한 문자 제거
        optimized_query = re.sub(r'["""]', '', optimized_query)
        optimized_query = optimized_query.strip()
        
        print(f"[optimize_search_query] 원본: {original_question}")
        print(f"[optimize_search_query] 최적화: {optimized_query}")
        
        return optimized_query
    except Exception as e:
        print(f"[optimize_search_query] 오류 발생: {e}")
        return original_question

def build_ensemble_retriever(faiss_vectorstore, bm25_retriever):
    """앙상블 리트리버 생성 함수"""
    faiss_retriever = faiss_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})
    bm25_retriever.k = 8
    return EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.6, 0.4]  # FAISS에 더 높은 가중치
    )

def create_vectorstores(documents, embedding_model):
    """벡터스토어 생성 함수"""
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,  # 더 작은 청크로 분할
        chunk_overlap=150,
        model_name="gpt-4"
    )
    chunked_documents = []
    for doc in documents:
        chunked = text_splitter.split_documents([doc])
        chunked_documents.extend(chunked)
    faiss_vectorstore = FAISS.from_documents(
        documents=chunked_documents,
        embedding=embedding_model
    )
    bm25_retriever = BM25Retriever.from_documents(chunked_documents)
    return faiss_vectorstore, bm25_retriever

def web_search_func(query):
    """개선된 웹 검색 함수"""
    try:
    search = DuckDuckGoSearchAPIWrapper()
        results = search.results(query=query, max_results=8)
        
    if not results:
        return [Document(page_content=f"검색 결과 없음: {query}")]
        
        # 검색 결과를 더 구조화된 형태로 변환
        documents = []
        for item in results:
            title = item.get('title', '')
            snippet = item.get('snippet', '')
            link = item.get('link', '')
            
            # 제목과 내용을 결합하여 더 풍부한 컨텍스트 생성
            content = f"제목: {title}\n내용: {snippet}\n출처: {link}"
            documents.append(Document(page_content=content))
        
        return documents
    except Exception as e:
        print(f"[web_search_func] 검색 오류: {e}")
        return [Document(page_content=f"검색 중 오류 발생: {str(e)}")]

# ==================== 그래프 빌더 ====================

def build_adaptive_rag_graph(ensemble_retriever, article_id=None):
    """모듈화된 Adaptive RAG 그래프 빌더"""
    
    # 노드 인스턴스 생성
    article_search_node = ArticleSearchNode(ensemble_retriever)
    all_search_node = AllSearchNode(ensemble_retriever)
    web_search_node = WebSearchNode()
    context_evaluator = ContextSufficiencyEvaluator()
    answer_generator = AnswerGenerator()
    
    # 그래프 생성
    workflow = StateGraph(dict)
    
    # 노드 등록
    workflow.add_node("article_search", article_search_node)
    workflow.add_node("all_search", all_search_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("context_evaluator", context_evaluator)
    workflow.add_node("generate_answer", answer_generator)
    
    # 엣지 연결
    workflow.add_edge(START, "article_search")
    
    # article_search 후 분기
    workflow.add_conditional_edges(
        "article_search",
        context_evaluator,
        {
            "sufficient": "generate_answer", 
            "insufficient": "all_search"
        }
    )
    
    # all_search 후 분기
    workflow.add_conditional_edges(
        "all_search",
        context_evaluator,
        {
            "sufficient": "generate_answer", 
            "insufficient": "web_search"
        }
    )
    
    # web_search는 항상 generate_answer로
    workflow.add_edge("web_search", "generate_answer")
    workflow.add_edge("generate_answer", END)
    
    # 컴파일
    return workflow.compile()

# ==================== 메인 에이전트 클래스 ====================

class AdaptiveRAGAgent:
    """
    모듈화된 강화된 Adaptive RAG Agent for news QA.
    
    주요 개선사항:
    1. 모듈화된 노드 구조
    2. 재사용 가능한 컴포넌트
    3. 더 엄격한 컨텍스트 충분성 평가
    4. 검색 쿼리 최적화
    5. 단계별 검색 전략 개선
    6. 오류 처리 강화
    """
    def __init__(self, embedding_model=None):
        if embedding_model is None:
            embedding_model = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                task_type="retrieval_document"
            )
        self.embedding_model = embedding_model
        self.logger = logging.getLogger(__name__)

    def answer(self, user_question: str, documents: List[Document], article_id: Optional[str] = None) -> str:
        """
        사용자 질문에 대한 답변을 생성합니다.
        
        Args:
            user_question: 사용자 질문
            documents: 검색할 문서 목록
            article_id: 특정 기사 ID (선택사항)
            
        Returns:
            str: 생성된 답변
        """
        try:
            self.logger.info(f"질문 처리 시작: {user_question[:50]}...")
            self.logger.info(f"문서 수: {len(documents)}")
            if article_id:
                self.logger.info(f"특정 기사 ID: {article_id}")
            
        # 1. 문서 chunking 및 vectorstore/retriever 생성 (메모리 내)
        faiss_vectorstore, bm25_retriever = create_vectorstores(
            documents=documents,
            embedding_model=self.embedding_model
        )
        ensemble_retriever = build_ensemble_retriever(
            faiss_vectorstore=faiss_vectorstore,
            bm25_retriever=bm25_retriever
        )
            
        # 2. 그래프 생성
        graph = build_adaptive_rag_graph(
            ensemble_retriever=ensemble_retriever,
            article_id=article_id
        )
            
        # 3. 답변 생성
        from langchain_core.messages import convert_to_messages
        state = {
            "messages": convert_to_messages([
                {"role": "user", "content": user_question}
            ]),
            "context": [],
                "trace": [],
                "article_id": article_id  # article_id를 상태에 추가
        }
            
        answer = None
            final_state = None
            
            # 그래프 실행 및 상태 추적
        for chunk in graph.stream(state):
            for node, update in chunk.items():
                    self.logger.info(f"노드 실행: {node}")
                if "answer" in update:
                    answer = update["answer"]
                        final_state = update
                    if "trace" in update:
                        self.logger.info(f"검색 추적: {update['trace']}")
            
            if answer:
                self.logger.info("답변 생성 완료")
                return answer
            else:
                self.logger.warning("답변 생성 실패")
                return "죄송합니다. 답변을 생성하지 못했습니다. 다시 시도해주세요."
                
        except Exception as e:
            self.logger.error(f"답변 생성 중 오류 발생: {e}", exc_info=True)
            return f"죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}"

    def debug_run(self, user_question: str, documents: List[Document], article_id: Optional[str] = None):
        """
        디버깅용 실행 함수
        """
        print(f"=== AdaptiveRAGAgent 디버그 실행 ===")
        print(f"질문: {user_question}")
        print(f"문서 수: {len(documents)}")
        if article_id:
            print(f"특정 기사 ID: {article_id}")
        
        return self.answer(user_question, documents, article_id)

# ==================== 테스트 코드 ====================

if __name__ == "__main__":
    # 테스트 코드
    user_question = "빅테크들이 ai 기술로 해고한 사례가 있으면 설명해줘."
    
    # 테스트용 문서 생성
    test_documents = [
        Document(
            page_content="구글은 AI 기술 발전으로 일부 직원들을 해고했다. 특히 검색 엔진 최적화 팀에서 큰 변화가 있었다.",
            metadata={"article_id": "test1", "title": "구글 AI 해고"}
        ),
        Document(
            page_content="마이크로소프트도 AI 도입으로 조직 개편을 진행했다. 하지만 대부분의 직원들은 재배치되었다.",
            metadata={"article_id": "test2", "title": "마이크로소프트 AI 조직개편"}
        )
    ]
    
    agent = AdaptiveRAGAgent()
    result = agent.debug_run(user_question, test_documents)
    print(f"\n=== 최종 결과 ===")
    print(result)