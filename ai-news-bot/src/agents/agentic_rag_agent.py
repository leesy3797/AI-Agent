"""
Agentic RAG Agent

이 모듈은 LangGraph를 활용한 모듈화된 Agentic RAG 에이전트를 제공합니다.
각 노드가 독립적으로 관리되어 유지보수성과 확장성이 뛰어납니다.
"""

from typing import List, Optional
import logging
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

from .nodes.retriever_nodes import ArticleSearchNode, AllSearchNode, WebSearchNode
from .nodes.evaluator_nodes import ContextSufficiencyEvaluator
from .nodes.generator_nodes import AnswerGenerator
from .utils.vectorstore_utils import create_vectorstores, build_ensemble_retriever


def build_agentic_rag_graph(ensemble_retriever, article_id=None):
    """
    모듈화된 Agentic RAG 그래프 빌더
    
    Args:
        ensemble_retriever: 앙상블 리트리버
        article_id: 특정 기사 ID (선택사항)
        
    Returns:
        CompiledStateGraph: 컴파일된 그래프
    """
    
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


class AgenticRAGAgent:
    """
    모듈화된 강화된 Agentic RAG Agent for news QA.
    
    주요 특징:
    1. 모듈화된 노드 구조
    2. 재사용 가능한 컴포넌트
    3. 더 엄격한 컨텍스트 충분성 평가
    4. 검색 쿼리 최적화
    5. 단계별 검색 전략 개선
    6. 오류 처리 강화
    
    사용 예시:
        agent = AgenticRAGAgent()
        answer = agent.answer(
            user_question="질문",
            documents=document_list,
            article_id="optional_article_id"
        )
    """
    
    def __init__(self, embedding_model=None):
        """
        AgenticRAGAgent 초기화
        
        Args:
            embedding_model: 임베딩 모델 (기본값: GoogleGenerativeAIEmbeddings)
        """
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
            graph = build_agentic_rag_graph(
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
        
        Args:
            user_question: 사용자 질문
            documents: 검색할 문서 목록
            article_id: 특정 기사 ID (선택사항)
            
        Returns:
            str: 생성된 답변
        """
        print(f"=== AgenticRAGAgent 디버그 실행 ===")
        print(f"질문: {user_question}")
        print(f"문서 수: {len(documents)}")
        if article_id:
            print(f"특정 기사 ID: {article_id}")
        
        return self.answer(user_question, documents, article_id) 