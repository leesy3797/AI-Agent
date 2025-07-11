"""
추천 기사 노드 구현
"""
from typing import Dict, Any, List
from .base_nodes import BaseRecommendNode

class RecommendArticlesNode(BaseRecommendNode):
    """
    답변 후 관련 기사 추천 노드
    """
    def __init__(self):
        super().__init__("recommend_articles")
    def recommend(self, state: Dict[str, Any]) -> List[Dict[str, str]]:
        # context에서 기사 메타데이터 추출, 유사도/최근성 기준 상위 2~3개 추천
        context = state.get("context", [])
        seen = set()
        recommendations = []
        for doc in context:
            meta = getattr(doc, 'metadata', {})
            title = meta.get('title', '')
            url = meta.get('url', '')
            if title and url and url not in seen:
                recommendations.append({"title": title, "url": url})
                seen.add(url)
            if len(recommendations) >= 3:
                break
        state["recommendations"] = recommendations
        return recommendations
    def __call__(self, state: Dict[str, Any]) -> list:
        return self.recommend(state) 