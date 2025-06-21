"""
평가 노드 구현

이 모듈은 다양한 평가 로직을 구현한 노드들을 제공합니다.
각 노드는 특정 기준에 따라 상태를 평가합니다.
"""

from typing import Dict, Any
from .base_nodes import BaseEvaluatorNode
from ..utils.evaluation_utils import context_sufficient_llm


class ContextSufficiencyEvaluator(BaseEvaluatorNode):
    """컨텍스트 충분성 평가 노드"""
    
    def __init__(self):
        super().__init__("context_sufficiency_evaluator")
    
    def evaluate(self, state: Dict[str, Any]) -> str:
        """컨텍스트 충분성 평가"""
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