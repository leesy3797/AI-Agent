from datetime import datetime
from typing import List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from .models import Article, User, UserRead, ArticleCreate, UserCreate

# Article CRUD
def create_article(db: Session, article: ArticleCreate) -> Article:
    """새로운 기사를 생성합니다."""
    db_article = Article(
        title=article.title,
        url=article.url,
        content=article.content,
        published_at=article.published_at
    )
    db.add(db_article)
    db.commit()
    db.refresh(db_article)
    return db_article

def get_article_by_url(db: Session, url: str) -> Optional[Article]:
    """URL로 기사를 조회합니다."""
    return db.query(Article).filter(Article.url == url).first()

def get_latest_articles(db: Session, skip: int = 0, limit: int = 20) -> List[Article]:
    """최신 기사 목록을 조회합니다."""
    return db.query(Article)\
        .order_by(Article.published_at.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()

def get_unread_articles(db: Session, user_id: UUID, skip: int = 0, limit: int = 20) -> List[Article]:
    """사용자가 읽지 않은 기사 목록을 조회합니다."""
    return db.query(Article)\
        .outerjoin(UserRead, (UserRead.article_id == Article.id) & (UserRead.user_id == user_id))\
        .filter(UserRead.id.is_(None))\
        .order_by(Article.published_at.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()

def get_article_by_id(db: Session, article_id: int) -> Optional[Article]:
    """ID로 기사를 조회합니다."""
    return db.query(Article).filter(Article.id == article_id).first()

# User CRUD
def create_user(db: Session, user: UserCreate) -> User:
    """새로운 사용자를 생성합니다."""
    db_user = User(
        telegram_id=user.telegram_id,
        username=user.username,
        full_name=user.full_name,
        notification_enabled=user.notification_enabled
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user_by_telegram_id(db: Session, telegram_id: int) -> Optional[User]:
    """텔레그램 ID로 사용자를 조회합니다."""
    return db.query(User).filter(User.telegram_id == telegram_id).first()

def update_user_last_active(db: Session, user_id: UUID) -> None:
    """사용자의 마지막 활동 시간을 업데이트합니다."""
    db.query(User).filter(User.id == user_id).update({"last_active": datetime.utcnow()})
    db.commit()

# UserRead CRUD
def mark_article_as_read(db: Session, user_id: UUID, article_id: UUID) -> UserRead:
    """기사를 읽음으로 표시합니다."""
    user_read = UserRead(user_id=user_id, article_id=article_id)
    db.add(user_read)
    db.commit()
    db.refresh(user_read)
    return user_read

def get_user_read_articles(db: Session, user_id: UUID, skip: int = 0, limit: int = 20) -> List[Article]:
    """사용자가 읽은 기사 목록을 조회합니다."""
    return db.query(Article)\
        .join(UserRead, UserRead.article_id == Article.id)\
        .filter(UserRead.user_id == user_id)\
        .order_by(UserRead.read_at.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()

def get_users_for_notification(db: Session) -> List[User]:
    """알림을 받을 사용자 목록을 조회합니다."""
    return db.query(User)\
        .filter(User.is_active == True, User.notification_enabled == True)\
        .all()

def create_user_read(db: Session, user_id: int, article_id: int) -> UserRead:
    """사용자의 읽은 기사 기록을 생성합니다."""
    user_read = UserRead(user_id=user_id, article_id=article_id)
    db.add(user_read)
    db.commit()
    db.refresh(user_read)
    return user_read 