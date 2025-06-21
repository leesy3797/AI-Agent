"""
검색 유틸리티 함수들

이 모듈은 웹 검색 및 검색 쿼리 최적화와 관련된 유틸리티 함수들을 제공합니다.
"""

import re
import google.generativeai as genai
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.documents import Document

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


def optimize_search_query(original_question: str, context_info: str = "") -> str:
    """
    사용자 질문을 인터넷 검색에 최적화된 쿼리로 변환
    
    Args:
        original_question: 원본 사용자 질문
        context_info: 현재 컨텍스트 정보 (선택사항)
        
    Returns:
        str: 최적화된 검색 쿼리
    """
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


def web_search_func(query: str):
    """
    웹 검색 수행
    
    Args:
        query: 검색 쿼리
        
    Returns:
        List[Document]: 검색 결과 문서 목록
    """
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