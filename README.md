# AI-Agent 하위 프로젝트 네비게이션 🧭

LangGraph와 n8n으로 구현한 다양한 AI 에이전트·자동화 워크플로우를 모아둔 모놀리포입니다. 아래 가이드를 통해 원하는 하위 프로젝트로 바로 이동하세요.

### 1. [[Langgraph] AI 뉴스 요약/브리핑 에이전트 (ai-news-bot)](ai-news-bot/) 📰
- **주요 내용**: 뉴스 수집 → 요약/정리 → 벡터스토어 적재 → 알림 파이프라인
- **빠른 시작**: [README 보기](ai-news-bot/README.md)
- **활용 기술**: Python, LangGraph, LangChain, FAISS, BM25, Google Programmable Search, Telegram Bot API, SQLite

### 2. [[N8N] 유튜브 자동화 봇 (youtube-bot)](youtube-bot/) ▶️
- **주요 내용**: 유튜브 스크립트 추출/요약, 하이라이트 생성, 메타데이터 보조
- **빠른 시작**: [README 보기](youtube-bot/README.md)
- **활용 기술**: n8n, YouTube Data API, Telegram Bot API, OpenAI API, JavaScript(Code 노드)

### 3. [[N8N] n8n 워크플로우 모음 (n8n-workflow)](n8n-workflow/) ⚙️
- **주요 내용**: n8n에서 바로 Import하여 실행 가능한 워크플로우 JSON 모음
- **빠른 시작**: n8n UI → Import from file → 원하는 `.json` 선택
- **대표 워크플로우**:
  - `n8n_emergency_realtime_data_scraper.json`: 공공데이터 수집 (응급의료기관 관련 데이터) → 전처리 → 구글 시트 적재
  - `n8n_metro_datascraper.json`: 지하철 도착 정보 데이터 수집 → 구글 시트 적재
  - `n8n_openwebui_ai_agent_chatbot.json`: OpenWebUI 연동 에이전트 챗봇 플로우 → 질의응답/요약 자동화
  - `n8n_youtube_summarization.json`: 유튜브 자막 추출 → 요약 → 텔레그램/채널로 전송 자동화
- **활용 기술**: n8n, HTTP Request, OpenAI API, Telegram Bot API, Google Sheets API, YouTube Data API, RSS/웹 스크래핑

### 4. [[LangGraph] 랭그래프 튜토리얼 (langgraph-tutorial)](langgraph-tutorial/) 🧩
- **주요 내용**: LangGraph 중심의 에이전트/워크플로 학습 자료 및 튜토리얼
- **빠른 시작**: [README 보기](langgraph-tutorial/readme.md)
- **추천 노트북**: `agent development/run_an_agent.ipynb`, `agent development/workflow_agent.ipynb`
- **활용 기술**: Python, LangGraph, Jupyter Notebook, LLM API(OpenAI/Anthropic 호환)

---

#### 사용 방법 🚀
1) 위 섹션에서 관심 프로젝트를 선택합니다.
2) 각 프로젝트의 `README.md` 또는 예제 노트북을 참고해 실행합니다.
3) 필요 시 `requirements.txt` 설치와 환경 변수 설정을 진행합니다. 
