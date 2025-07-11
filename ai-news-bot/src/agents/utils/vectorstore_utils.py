"""
벡터스토어 유틸리티 함수들

이 모듈은 벡터스토어 생성 및 관리와 관련된 유틸리티 함수들을 제공합니다.
"""

from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def create_vectorstores(documents: List[Document], embedding_model):
    """
    문서로부터 벡터스토어 생성
    
    Args:
        documents: 처리할 문서 목록
        embedding_model: 임베딩 모델
        
    Returns:
        tuple: (faiss_vectorstore, bm25_retriever)
    """
    improved_chunks = []
    for doc in documents:
        text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        meta = getattr(doc, 'metadata', {})
        # 문단 단위로 우선 분할
        for para in text.split('\n\n'):
            para = para.strip()
            if not para:
                continue
            # 너무 긴 문단은 문장 단위로 추가 분할
            if len(para) > 500:
                for sent in para.split('. '):
                    sent = sent.strip()
                    if sent:
                        improved_chunks.append(Document(page_content=sent, metadata=meta))
            else:
                improved_chunks.append(Document(page_content=para, metadata=meta))
    
    faiss_vectorstore = FAISS.from_documents(
        documents=improved_chunks,
        embedding=embedding_model
    )
    
    bm25_retriever = BM25Retriever.from_documents(improved_chunks)
    
    return faiss_vectorstore, bm25_retriever


def build_ensemble_retriever(faiss_vectorstore, bm25_retriever):
    """
    앙상블 리트리버 생성
    
    Args:
        faiss_vectorstore: FAISS 벡터스토어
        bm25_retriever: BM25 리트리버
        
    Returns:
        EnsembleRetriever: 앙상블 리트리버
    """
    faiss_retriever = faiss_vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 8}
    )
    bm25_retriever.k = 8
    
    return EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.3, 0.7]  # FAISS 30%, BM25 70%
    ) 