"""
생성 노드 구현

이 모듈은 다양한 생성 로직을 구현한 노드들을 제공합니다.
각 노드는 특정 방식으로 콘텐츠를 생성합니다.
"""

from typing import Dict, Any
import google.generativeai as genai
from .base_nodes import BaseGeneratorNode


class AnswerGenerator(BaseGeneratorNode):
    """답변 생성 노드"""
    
    def __init__(self):
        super().__init__("answer_generator")
    
    def generate(self, state: Dict[str, Any]) -> str:
        """답변 생성"""
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
        
        # 출처 정보 추출 (중복 제거)
        sources = set()
        for doc in context:
            meta = getattr(doc, 'metadata', {})
            title = meta.get('title', '')
            date = meta.get('date', '')
            url = meta.get('url', '')
            if title or url:
                # 출처의 고유성을 title, date, url로 판단하여 set에 추가
                sources.add(f"- {title} {url}")
        sources_text = '\n'.join(sorted(sources))
        
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
3. 필요하다면 이모지를 사용해서 항목을 구분해 주세요
4. 답변 마지막에 아래와 같이 참고 기사 출처를 명시하세요:

[참고 기사 출처]
{sources_text}
"""
        
        print(f"[answer_generator] LLM 프롬프트 미리보기: {prompt[:100]} ...")
        
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text 