"""
검색 노드 구현

이 모듈은 다양한 검색 전략을 구현한 노드들을 제공합니다.
각 노드는 특정 검색 방식에 특화되어 있습니다.
"""

from typing import Dict, Any, List
from langchain_core.documents import Document
from langchain.retrievers.ensemble import EnsembleRetriever

from .base_nodes import BaseRetrieverNode
from ..utils.vectorstore_utils import create_vectorstores, build_ensemble_retriever
from ..utils.search_utils import web_search_func, optimize_search_query


class ArticleSearchNode(BaseRetrieverNode):
    """특정 기사 검색 노드"""
    
    def __init__(self, ensemble_retriever: EnsembleRetriever):
        super().__init__("article_search")
        self.ensemble_retriever = ensemble_retriever
    
    def retrieve(self, state: Dict[str, Any]) -> List[Document]:
        """특정 기사에서만 검색 수행"""
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
            faiss_vectorstore, bm25_retriever = create_vectorstores(article_docs, faiss_retriever.vectorstore.embedding_function)
            temp_ensemble = build_ensemble_retriever(faiss_vectorstore, bm25_retriever)
            # 항상 앙상블 리트리버 사용
            context = temp_ensemble.invoke(user_question)
            print(f"[article_search_node] 앙상블 리트리버로 {len(context)}개 문서 반환")
            return context
        else:
            print(f"[article_search_node] 해당 기사를 찾을 수 없음: {article_id}")
            return []


class AllSearchNode(BaseRetrieverNode):
    """전체 기사 검색 노드"""
    
    def __init__(self, ensemble_retriever: EnsembleRetriever):
        super().__init__("all_search")
        self.ensemble_retriever = ensemble_retriever
    
    def retrieve(self, state: Dict[str, Any]) -> List[Document]:
        """전체 기사에서 검색 수행"""
        user_question = self._get_user_question(state)
        # 질의 길이에 따라 가중치 조정
        if len(user_question.strip()) < 15:
            context = self.ensemble_retriever.retrievers[1].get_relevant_documents(user_question)  # BM25
        else:
            context = self.ensemble_retriever.retrievers[0].similarity_search(user_question)  # semantic
        # 두 결과 합치기 (중복 제거)
        doc_ids = set()
        merged = []
        for doc in context:
            doc_id = getattr(doc, 'metadata', {}).get('doc_id', id(doc))
            if doc_id not in doc_ids:
                merged.append(doc)
                doc_ids.add(doc_id)
        print(f"[all_search_node] 최종 {len(merged)}개 문서 반환")
        return merged


class WebSearchNode(BaseRetrieverNode):
    """인터넷 검색 노드"""
    
    def __init__(self):
        super().__init__("web_search")
    
    def retrieve(self, state: Dict[str, Any]) -> List[Document]:
        """웹 검색 수행"""
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