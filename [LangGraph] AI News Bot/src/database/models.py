import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import Column, String, DateTime, ForeignKey, Boolean, Integer, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, declarative_base
from pydantic import BaseModel

Base = declarative_base()

class Article(Base):
    """뉴스 기사 테이블"""
    __tablename__ = "articles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    url = Column(String(500), unique=True, nullable=False)
    content = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)  # Gemini 요약 결과 저장
    published_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 관계 설정
    user_reads = relationship("UserRead", back_populates="article")

class User(Base):
    """사용자 모델"""
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    telegram_id = Column(String(100), unique=True, nullable=False)
    username = Column(String(100))
    first_name = Column(String(100))
    last_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    notification_enabled = Column(Boolean, default=True)
    last_active = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 관계 설정
    read_articles = relationship("UserRead", back_populates="user")

class UserRead(Base):
    """사용자별 읽은 기사 모델"""
    __tablename__ = "user_reads"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    article_id = Column(UUID(as_uuid=True), ForeignKey("articles.id"), nullable=False)
    read_at = Column(DateTime, default=datetime.utcnow)

    # 관계 설정
    user = relationship("User", back_populates="read_articles")
    article = relationship("Article", back_populates="user_reads")

# Pydantic 모델
class ArticleBase(BaseModel):
    title: str
    url: str
    content: str
    published_at: datetime

class ArticleCreate(ArticleBase):
    pass

class ArticleResponse(ArticleBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class UserBase(BaseModel):
    telegram_id: str
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_active: bool = True
    notification_enabled: bool = True

class UserCreate(UserBase):
    pass

class UserResponse(UserBase):
    id: uuid.UUID
    last_active: datetime
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True 