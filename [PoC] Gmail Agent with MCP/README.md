# Gmail 에이전트 with LangGraph

LangGraph를 활용한 지능형 Gmail 관리 에이전트입니다. Telegram을 통해 실시간으로 이메일을 모니터링하고 관리할 수 있습니다.

## 주요 기능

- 실시간 이메일 알림 및 요약
- AI 기반 이메일 분석 및 작업 추출
- Telegram을 통한 이메일 관리
- 자동 답장 생성 및 전송
- 작업 관리 및 추적

## 설치 방법

1. 저장소 클론:
```bash
git clone [repository-url]
cd gmail-agent
```

2. 가상환경 생성 및 활성화:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

3. 의존성 설치:
```bash
pip install -r requirements.txt
```

4. 환경 변수 설정:
`.env` 파일을 생성하고 다음 변수들을 설정합니다:
```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
GEMINI_API_KEY=your_gemini_api_key
```

5. Gmail API 설정:
- [Google Cloud Console](https://console.cloud.google.com)에서 프로젝트를 생성합니다.
- Gmail API를 활성화합니다.
- OAuth 2.0 클라이언트 ID를 생성하고 `credentials.json` 파일을 다운로드합니다.
- `credentials.json` 파일을 프로젝트 루트 디렉토리에 저장합니다.

6. Gemini API 키 설정:
- [Google AI Studio](https://makersuite.google.com/app/apikey)에서 API 키를 생성합니다.
- 생성된 API 키를 `.env` 파일의 `GEMINI_API_KEY`에 설정합니다.

## 사용 방법

1. Telegram 봇 생성:
- [@BotFather](https://t.me/botfather)를 통해 새로운 봇을 생성합니다.
- 발급받은 토큰을 `.env` 파일의 `TELEGRAM_BOT_TOKEN`에 설정합니다.

2. 에이전트 실행:
```bash
python -m src.main
```

3. Telegram 봇 사용:
- 봇을 시작하려면 `/start` 명령어를 입력합니다.
- 새로운 이메일이 도착하면 자동으로 알림이 전송됩니다.
- 이메일 알림에서 제공되는 버튼을 통해 다양한 작업을 수행할 수 있습니다:
  - 📝 답장: AI가 생성한 답장 초안을 확인하고 전송
  - ✅ 읽음: 이메일을 읽음으로 표시
  - 🗑️ 삭제: 이메일 삭제
  - 📋 작업 추가: 이메일에서 추출된 작업을 작업 목록에 추가

## 프로젝트 구조

```
gmail-agent/
├── src/
│   ├── __init__.py
│   ├── main.py              # 메인 애플리케이션
│   ├── config.py            # 설정 관리
│   ├── gmail_client.py      # Gmail API 클라이언트
│   ├── telegram_client.py   # Telegram 봇 클라이언트
│   ├── agent_state.py       # 에이전트 상태 관리
│   └── agent_nodes.py       # LangGraph 노드 정의
├── requirements.txt         # 의존성 목록
├── .env                     # 환경 변수
└── README.md               # 프로젝트 문서
```

## 개발 로드맵

1. MVP (현재)
   - 기본 이메일 알림 및 요약
   - 간단한 이메일 관리 기능

2. 향후 계획
   - 고급 이메일 분석 및 분류
   - 작업 관리 시스템 개선
   - 다국어 지원
   - 사용자 설정 및 커스터마이징
   - 성능 최적화 및 확장성 개선

## 라이선스

MIT License