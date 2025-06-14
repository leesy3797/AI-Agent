from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Gmail API 설정
    GMAIL_CREDENTIALS_FILE: str = "credentials.json"
    GMAIL_TOKEN_FILE: str = "token.json"
    
    # Telegram Bot 설정
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    
    # Gemini API 설정
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # 데이터베이스 설정
    DATABASE_URL: str = "sqlite:///gmail_agent.db"
    
    # 기타 설정
    POLLING_INTERVAL: int = 60  # Gmail 폴링 간격 (초)
    MAX_EMAILS_PER_POLL: int = 10  # 한 번에 처리할 최대 이메일 수
    
    class Config:
        env_file = ".env"

settings = Settings() 