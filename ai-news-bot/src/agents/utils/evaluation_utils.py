"""
평가 유틸리티 함수들

이 모듈은 컨텍스트 충분성 평가와 관련된 유틸리티 함수들을 제공합니다.
"""

import google.generativeai as genai

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


def context_sufficient_llm(question: str, context: str) -> bool:
    """
    LLM을 사용한 컨텍스트 충분성 평가
    
    Args:
        question: 사용자 질문
        context: 평가할 컨텍스트
        
    Returns:
        bool: 컨텍스트가 충분하면 True, 부족하면 False
    """
    grader_model = genai.GenerativeModel('models/gemini-2.0-flash')
    prompt = GRADE_PROMPT.format(question=question, context=context)
    
    print(f"[context_sufficient_llm] 질문: {question}")
    print(f"[context_sufficient_llm] 컨텍스트 길이: {len(context)}")
    print(f"[context_sufficient_llm] 컨텍스트 미리보기: {context[:200]} ...")
    
    try:
        response = grader_model.generate_content(prompt)
        answer = response.text.strip().upper()
        print(f"[context_sufficient_llm] LLM 판정 결과: '{answer}'")
        
        is_sufficient = answer == "SUFFICIENT"
        print(f"[context_sufficient_llm] 최종 판정: {is_sufficient}")
        return is_sufficient
    except Exception as e:
        print(f"[context_sufficient_llm] 오류 발생: {e}")
        return False 