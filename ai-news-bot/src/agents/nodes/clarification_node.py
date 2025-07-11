"""
질의 명확화 노드 구현
"""
from typing import Dict, Any
from .base_nodes import BaseClarificationNode
import google.generativeai as genai

class ClarificationNode(BaseClarificationNode):
    """
    질의 명확화 노드: 질문이 너무 짧거나 모호할 때 LLM을 통해 보완/확장
    """
    def __init__(self):
        super().__init__("clarification")
    def clarify(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get("messages", [])
        if messages and hasattr(messages[0], "content"):
            user_question = messages[0].content
        elif messages and isinstance(messages[0], dict):
            user_question = messages[0].get("content", "")
        else:
            user_question = ""
        # 너무 짧거나 모호한 질문 판단
        if len(user_question.strip()) < 10:
            prompt = f"""
[질문]
{user_question}

위 질문이 너무 짧거나 모호합니다. 사용자가 궁금한 점을 더 명확하게 알 수 있도록, 다음 중 하나를 수행하세요:
- 질문을 더 구체적으로 보완
- 추가로 필요한 정보를 요청

명확한 질문 형태로 제안해 주세요.
"""
            model = genai.GenerativeModel('models/gemini-2.0-flash')
            response = model.generate_content(prompt)
            clarified = response.text.strip()
            state["clarified_question"] = clarified
            return state
        else:
            state["clarified_question"] = user_question
            return state
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self.clarify(state) 