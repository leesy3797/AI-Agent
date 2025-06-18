# AI News Telegram Bot

AI Times의 최신 뉴스를 자동으로 수집하고 요약하여 텔레그램으로 제공하는 봇입니다.

## 주요 기능

- 매일 오전 9시 정기 뉴스 브리핑
- 사용자 요청 기반 뉴스 제공
- AI 기반 기사 요약
- LangGraph를 활용한 대화형 챗봇

## 설치 방법

1. Python 3.9 이상 설치

2. 가상환경 생성 및 활성화
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

4. 환경 변수 설정
- `.env.example` 파일을 `.env`로 복사
- 필요한 API 키와 설정값 입력:
  - TELEGRAM_BOT_TOKEN: 텔레그램 봇 토큰
  - GOOGLE_API_KEY: Google Gemini API 키
  - DB_*: PostgreSQL 데이터베이스 설정
  - 기타 설정값

5. 데이터베이스 설정
- PostgreSQL 설치
- 데이터베이스 생성
- pgvector 확장 설치

## 실행 방법

```bash
python src/main.py
```

## 프로젝트 구조

```
[LangGraph] AI News Bot/
├── src/                   # 소스 코드
│   ├── bot/              # 텔레그램 봇
│   ├── crawler/          # 웹 크롤링
│   ├── database/         # 데이터베이스
│   ├── agent/            # LangGraph 에이전트
│   └── utils/            # 유틸리티
└── tests/                # 테스트 코드
```

## 개발 단계

1. 기반 구축
   - 텔레그램 봇 연동
   - 스크래핑 로직
   - DB 저장 기능

2. 핵심 기능 구현
   - 정기/요청 기반 알림
   - 페이지네이션

3. AI 기능 연동
   - 기사 요약
   - VectorDB
   - LangGraph 에이전트

4. 고도화 및 배포
   - 대화 메모리
   - 오류 처리
   - 서버 배포 