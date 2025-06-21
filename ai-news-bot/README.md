# AI 뉴스 봇 - 모듈화된 Agentic RAG Agent

AI 뉴스 기사에 대한 지능형 대화형 봇입니다. LangGraph를 활용한 모듈화된 Agentic RAG (Retrieval-Augmented Generation) 기술을 사용하여 정확하고 관련성 높은 답변을 제공합니다.

## 🚀 주요 특징

### 1. 모듈화된 아키텍처
- **독립적인 노드 구조**: 각 기능이 독립적인 노드로 분리되어 유지보수성 향상
- **재사용 가능한 컴포넌트**: 노드와 유틸리티 함수들이 다른 프로젝트에서도 재사용 가능
- **명확한 책임 분리**: 검색, 평가, 생성 기능이 각각 별도 모듈로 관리

### 2. 강화된 검색 전략
- **단계별 검색**: 특정 기사 → 전체 기사 → 인터넷 검색 순서로 진행
- **검색 쿼리 최적화**: LLM을 활용한 지능형 검색 쿼리 변환
- **앙상블 리트리버**: FAISS + BM25 조합으로 검색 정확도 향상

### 3. 엄격한 품질 관리
- **컨텍스트 충분성 평가**: LLM 기반 지능형 평가로 답변 품질 보장
- **다단계 검증**: 길이, 내용, 관련성 다각도 검증
- **오류 처리 강화**: 각 단계별 견고한 예외 처리

## 🏗️ 모듈 구조

```
src/agents/
├── __init__.py                 # 패키지 초기화
├── agentic_rag_agent.py       # 메인 에이전트 클래스
├── nodes/                     # LangGraph 노드들
│   ├── __init__.py
│   ├── base_nodes.py          # 기본 노드 클래스들
│   ├── retriever_nodes.py     # 검색 노드들
│   ├── evaluator_nodes.py     # 평가 노드들
│   └── generator_nodes.py     # 생성 노드들
└── utils/                     # 유틸리티 함수들
    ├── __init__.py
    ├── vectorstore_utils.py   # 벡터스토어 관련
    ├── search_utils.py        # 검색 관련
    └── evaluation_utils.py    # 평가 관련
```

## 🔄 워크플로우

```
사용자 질문
    ↓
특정 기사 검색 (ArticleSearchNode)
    ↓
컨텍스트 충분성 평가 (ContextSufficiencyEvaluator)
    ↓
충분함? → 답변 생성 (AnswerGenerator)
    ↓
부족함? → 전체 기사 검색 (AllSearchNode)
    ↓
컨텍스트 충분성 평가 (ContextSufficiencyEvaluator)
    ↓
충분함? → 답변 생성 (AnswerGenerator)
    ↓
부족함? → 인터넷 검색 (WebSearchNode)
    ↓
답변 생성 (AnswerGenerator)
```

## 📦 설치 및 설정

### 1. 환경 설정
```bash
# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일에 다음 정보 입력:
# TELEGRAM_BOT_TOKEN=your_telegram_bot_token
# GOOGLE_API_KEY=your_google_api_key
```

### 2. 데이터베이스 초기화
```bash
python -m src.database.initialize
```

### 3. 봇 실행
```bash
python bot.py
```

## 🧪 테스트

모듈화된 AgenticRAGAgent를 테스트하려면:

```bash
python test_agentic_rag.py
```

### 테스트 케이스
1. **모듈화된 컴포넌트**: 각 노드와 유틸리티 함수의 독립적 작동 확인
2. **특정 기사 검색**: 특정 기사에서만 정보를 찾는 기능
3. **불충분한 컨텍스트**: 관련 없는 정보에 대한 처리
4. **웹 검색 최적화**: 인터넷 검색 쿼리 최적화 기능

## 🔧 주요 컴포넌트

### AgenticRAGAgent 클래스
```python
from src.agents.agentic_rag_agent import AgenticRAGAgent

agent = AgenticRAGAgent()

# 기본 사용법
answer = agent.answer(
    user_question="질문",
    documents=document_list,
    article_id="optional_article_id"
)

# 디버깅용
answer = agent.debug_run(
    user_question="질문",
    documents=document_list,
    article_id="optional_article_id"
)
```

### 노드 컴포넌트들
```python
# 검색 노드들
from src.agents.nodes.retriever_nodes import ArticleSearchNode, AllSearchNode, WebSearchNode

# 평가 노드들
from src.agents.nodes.evaluator_nodes import ContextSufficiencyEvaluator

# 생성 노드들
from src.agents.nodes.generator_nodes import AnswerGenerator
```

### 유틸리티 함수들
```python
# 벡터스토어 관련
from src.agents.utils.vectorstore_utils import create_vectorstores, build_ensemble_retriever

# 검색 관련
from src.agents.utils.search_utils import web_search_func, optimize_search_query

# 평가 관련
from src.agents.utils.evaluation_utils import context_sufficient_llm
```

## 📊 성능 개선 사항

### 검색 정확도 향상
- **FAISS 가중치 증가**: 0.5 → 0.6 (더 정확한 벡터 검색)
- **청크 크기 최적화**: 1000 → 800 (더 세밀한 정보 분할)
- **검색 결과 수 증가**: 5 → 8 (더 많은 후보 검색)

### 컨텍스트 품질 향상
- **최소 길이 임계값**: 50 → 100 (더 충분한 정보 요구)
- **엄격한 평가 기준**: "yes/no" → "SUFFICIENT/INSUFFICIENT"
- **다단계 검증**: 길이 + 내용 + 관련성 검증

### 웹 검색 개선
- **구조화된 결과**: 제목 + 내용 + 출처 포함
- **최적화된 쿼리**: LLM 기반 쿼리 변환
- **컨텍스트 활용**: 기존 정보를 활용한 검색 개선

## 🎯 사용 예시

### 텔레그램 봇 명령어
- `/start` - 봇 시작
- `/news` - 최신 AI 뉴스 보기
- `/help` - 도움말 보기

### 대화 기능
1. 뉴스 목록에서 기사 선택
2. 기사 요약 확인
3. "이 기사로 대화하기" 선택
4. AI와 자연스러운 대화 진행

## 🔍 문제 해결

### 자주 발생하는 문제들

1. **검색 결과가 부정확한 경우**
   - 더 구체적인 질문을 해보세요
   - 다른 기사를 선택해보세요

2. **답변이 부족한 경우**
   - 시스템이 자동으로 인터넷 검색을 수행합니다
   - 잠시 기다린 후 다시 시도해보세요

3. **API 오류 발생**
   - Google API 키가 올바르게 설정되었는지 확인
   - 네트워크 연결 상태 확인

## 🛠️ 개발 가이드

### 새로운 노드 추가하기
```python
from src.agents.nodes.base_nodes import BaseRetrieverNode

class CustomSearchNode(BaseRetrieverNode):
    def __init__(self):
        super().__init__("custom_search")
    
    def retrieve(self, state):
        # 검색 로직 구현
        return documents
```

### 새로운 평가 노드 추가하기
```python
from src.agents.nodes.base_nodes import BaseEvaluatorNode

class CustomEvaluator(BaseEvaluatorNode):
    def __init__(self):
        super().__init__("custom_evaluator")
    
    def evaluate(self, state):
        # 평가 로직 구현
        return "sufficient" or "insufficient"
```

## 📈 향후 개선 계획

- [ ] 멀티모달 검색 지원 (이미지, 영상)
- [ ] 실시간 뉴스 업데이트 알림
- [ ] 사용자 맞춤형 뉴스 추천
- [ ] 다국어 지원 확장
- [ ] 성능 모니터링 대시보드
- [ ] 노드별 성능 메트릭 수집

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 문의

프로젝트에 대한 문의사항이 있으시면 이슈를 생성해주세요. 