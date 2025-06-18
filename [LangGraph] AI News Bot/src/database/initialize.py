# 예시: src/database/initialize.py
from .connection import init_db

if __name__ == "__main__":
    init_db()
    print("DB 테이블 생성 완료")