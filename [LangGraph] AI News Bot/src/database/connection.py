"""
데이터베이스 연결 및 세션 관리를 위한 모듈
"""
import os
from contextlib import contextmanager
from typing import Generator
from dotenv import dotenv_values # dotenv를 이용하여 직접 값을 읽어올 때 사용
from urllib.parse import quote_plus # URL 인코딩을 위한 모듈 추가

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from .models import Base

# .env 파일 로드
# dotenv_values를 사용하여 .env 파일의 모든 값을 딕셔너리로 로드하고 명시적으로 인코딩을 지정합니다.
# 일반적으로 .env 파일은 UTF-8로 작성되므로 'utf-8'을 지정합니다.
config = dotenv_values(encoding="utf-8") 

# 데이터베이스 연결 설정
# config 딕셔너리에서 값을 가져오며, 기본값 처리
DB_HOST = config.get("DB_HOST", "localhost")
DB_PORT = config.get("DB_PORT", "5432")
DB_NAME = config.get("DB_NAME", "ai_news_bot")
DB_USER = config.get("DB_USER", "postgres")
DB_PASSWORD = config.get("DB_PASSWORD")

if not DB_PASSWORD:
    raise ValueError("DB_PASSWORD environment variable is not set in .env file")

# 연결 문자열 구성 요소를 하나씩 URL 인코딩합니다.
# psycopg3도 URL에서 특수문자를 올바르게 처리할 수 있도록 quote_plus를 유지하는 것이 좋습니다.
encoded_db_user = quote_plus(DB_USER)
encoded_db_password = quote_plus(DB_PASSWORD)
encoded_db_host = quote_plus(DB_HOST)
encoded_db_port = quote_plus(DB_PORT)
encoded_db_name = quote_plus(DB_NAME)

# 최종 연결 문자열 구성
# psycopg3를 사용하도록 'postgresql+psycopg' 스키마를 명시합니다.
DATABASE_URL = (
    f"postgresql+psycopg://{encoded_db_user}:{encoded_db_password}@"
    f"{encoded_db_host}:{encoded_db_port}/{encoded_db_name}"
)

# 생성된 DATABASE_URL을 출력하여 디버깅에 활용합니다.
print(f"Generated DATABASE_URL for psycopg3: {DATABASE_URL}")

# 엔진 생성
# DATABASE_URL에 이미 psycopg3 스키마가 포함되어 있습니다.
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    echo=True,  # SQL 쿼리 로깅 활성화
    connect_args={
        'client_encoding': 'utf8', # 명시적으로 클라이언트 인코딩을 UTF-8로 설정
        # 'options': '-c client_encoding=utf8 -c timezone=Asia/Seoul' # 이 부분은 여전히 필요 없을 가능성이 높습니다.
    }
)

# 세션 팩토리 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    데이터베이스 세션을 제공하는 컨텍스트 매니저
    
    Yields:
        Session: 데이터베이스 세션
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db() -> None:
    """
    데이터베이스 테이블을 생성합니다.
    """
    try:
        # 연결 테스트
        # 여기서 오류가 발생한다면, 연결 문자열이나 PostgreSQL 설정 문제입니다.
        with engine.connect() as conn:
            print("데이터베이스 연결 성공")
        
        # 테이블 생성
        Base.metadata.create_all(bind=engine)
        print("데이터베이스 테이블이 성공적으로 생성되었습니다.")
    except Exception as e:
        print(f"데이터베이스 테이블 생성 중 오류 발생: {str(e)}")
        raise

def get_db_session() -> Session:
    """
    데이터베이스 세션을 반환합니다.
    
    Returns:
        Session: 데이터베이스 세션
    """
    return SessionLocal()