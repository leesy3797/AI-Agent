import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel

from ..database.connection import get_db
from ..database.crud import create_article, get_article_by_url
from ..database.models import ArticleCreate, Article

logger = logging.getLogger(__name__)

class NewsArticle(BaseModel):
    """뉴스 기사 데이터 모델"""
    title: str
    url: str
    published_at: datetime
    content: str

class AITimesCrawler:
    """AITimes 웹사이트 크롤러"""
    
    BASE_URL = "https://www.aitimes.com"
    NEWS_LIST_URL = f"{BASE_URL}/news/articleList.html?view_type=sm"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
    
    def _get_soup(self, url: str) -> BeautifulSoup:
        """URL에서 BeautifulSoup 객체를 가져옵니다."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": self.BASE_URL,
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    
    def _parse_date(self, date_str: str) -> datetime:
        """날짜 문자열을 datetime 객체로 변환합니다."""
        try:
            # "2025.06.18 10:30" 형식의 날짜 문자열을 파싱
            return datetime.strptime(date_str.strip(), "%Y.%m.%d %H:%M")
        except ValueError:
            logger.warning(f"Could not parse date: {date_str}")
            return datetime.now()
    
    def _extract_article_content(self, article_url: str) -> str:
        """기사 본문을 추출합니다."""
        logger = logging.getLogger("aitimes_crawler")
        soup = self._get_soup(article_url)
        
        content_selectors = [
            "article#article-view-content-div",
            "div#article-view-content-div",
            "div.article-view-content",
            "div.article-body",
            "div#articleBodyContents"
        ]
        for selector in content_selectors:
            article_body = soup.select_one(selector)
            if article_body:
                for element in article_body.select("script, style, .article-photo, .ad, figure, figcaption, button"):
                    element.decompose()
                content = article_body.get_text(separator="\n", strip=True)
                if content:
                    return content
        # 본문 셀렉터 실패 시 HTML 앞부분 로그
        logger.warning(f"[크롤러] 본문 셀렉터 실패: {article_url}")
        logger.debug(soup.prettify()[:1000])
        return ""
    
    def _save_article_to_db(self, article: NewsArticle) -> bool:
        """기사를 DB에 저장합니다."""
        try:
            with get_db() as db:
                # 중복 체크
                existing_article = get_article_by_url(db, article.url)
                if existing_article:
                    logger.info(f"Article already exists: {article.url}")
                    return False
                
                # 새 기사 저장
                article_create = ArticleCreate(
                    title=article.title,
                    url=article.url,
                    content=article.content,
                    published_at=article.published_at
                )
                create_article(db, article_create)
                logger.info(f"Article saved to DB: {article.url}")
                return True
        except Exception as e:
            logger.error(f"Error saving article to DB: {e}")
            return False
    
    def get_latest_news(self, max_articles: int = 50) -> List[NewsArticle]:
        """최신 뉴스 목록을 가져옵니다."""
        try:
            # 현재 시간
            now = datetime.now()
            # 48시간 전 시간
            forty_eight_hours_ago = now - timedelta(hours=48)
            
            # DB에서 48시간 이내의 기사 가져오기
            with get_db() as db:
                db_articles = db.query(Article)\
                    .filter(Article.published_at >= forty_eight_hours_ago)\
                    .order_by(Article.published_at.desc())\
                    .limit(max_articles)\
                    .all()
                
                logger.info(f"DB에서 가져온 48시간 이내 기사 수: {len(db_articles)}")
                
                # NewsArticle 객체로 변환
                articles = []
                for db_article in db_articles:
                    article = NewsArticle(
                        title=db_article.title,
                        url=db_article.url,
                        content=db_article.content,
                        published_at=db_article.published_at
                    )
                    articles.append(article)
                
                return articles
            
        except Exception as e:
            logger.error(f"Error fetching latest news: {e}")
            return []
    
    def get_article_by_url(self, url: str) -> Optional[NewsArticle]:
        """특정 URL의 기사를 가져옵니다."""
        try:
            # 먼저 DB에서 확인
            with get_db() as db:
                db_article = get_article_by_url(db, url)
                if db_article:
                    logger.info(f"Article already exists in DB: {url}")
                    return NewsArticle(
                        title=db_article.title,
                        url=db_article.url,
                        content=db_article.content,
                        published_at=db_article.published_at
                    )
            
            # DB에 없으면 크롤링
            soup = self._get_soup(url)
            if not soup:
                return None
                
            # # 본문 div 추출
            # article_div = soup.select_one('article#article-view-content-div')
            # if not article_div:
            #     logger.warning(f"[Article Div] Could not find article content at {url}")
            #     return None
                
            # 제목 추출
            title = soup.select_one('#articleViewCon > article > header > h3')
            if not title:
                logger.warning(f"[Title] Could not find article title at {url}")
                return None
            title = title.get_text(strip=True)
            
            # 날짜 추출
            date_str = soup.select_one('#articleViewCon > article > header > div.info-group > article:nth-child(1) > ul > li:nth-child(2) > i')
            if not date_str:
                logger.warning(f"[Date] Could not find article date at {url}")
                return None
            date_str = date_str.get_text(strip=True)
            published_at = self._parse_date(date_str)
            
            # 본문 추출
            content = self._extract_article_content(url)
            if not content:
                logger.warning(f"[Content] Could not extract article content at {url}")
                return None
            
            article = NewsArticle(
                title=title,
                url=url,
                content=content,
                published_at=published_at
            )
            
            # DB에 저장
            if self._save_article_to_db(article):
                logger.info(f"Article saved to DB: {url}")
                return article
            return None
            
        except Exception as e:
            logger.error(f"Error fetching article {url}: {str(e)}")
            return None

    def get_news_since(self, latest_time: datetime, max_articles: int = 50):
        """
        latest_time 이후에 발행된 기사만 크롤링해서 반환
        (실제 구현은 사이트 구조에 따라 다를 수 있음)
        """
        all_articles = self.get_latest_news(max_articles=max_articles)
        # published_at이 latest_time보다 큰 기사만 반환
        new_articles = [article for article in all_articles if hasattr(article, 'published_at') and article.published_at > latest_time]
        return new_articles

    def crawl_and_save_latest_news(self, max_articles: int = 50) -> List[NewsArticle]:
        """
        웹에서 최신 기사 목록을 실시간으로 크롤링해서 DB에 없는 기사만 저장하고,
        새로 저장된 기사 목록을 반환한다.
        """
        logger = logging.getLogger("aitimes_crawler")
        saved_articles = []
        try:
            soup = self._get_soup(self.NEWS_LIST_URL)
            article_links = soup.select('h4.titles > a')
            logger.info(f"[크롤러] 기사 링크 개수: {len(article_links)}")
            for link in article_links[:max_articles]:
                article_url = urljoin(self.BASE_URL, link.get('href'))
                logger.info(f"[크롤러] 기사 링크: {article_url}")
                # DB에 이미 있는지 확인
                with get_db() as db:
                    if get_article_by_url(db, article_url):
                        logger.info(f"[크롤러] 이미 DB에 있음: {article_url}")
                        continue
                # 상세 크롤링
                article = self.get_article_by_url(article_url)
                if article:
                    logger.info(f"[크롤러] 크롤링 성공: {article.title} / {article.published_at}")
                    if article.content:
                        logger.info(f"[크롤러] 본문 길이: {len(article.content)} 글자")
                    else:
                        logger.warning(f"[크롤러] 본문이 비어 있음: {article_url}")
                    saved_articles.append(article)
                else:
                    logger.warning(f"[크롤러] 크롤링 실패: {article_url}")
        except Exception as e:
            logger.error(f"[크롤러] 예외 발생: {e}")
        return saved_articles

if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    crawler = AITimesCrawler()
    
    # DB에서 가장 최신 기사 published_at 가져오기
    from datetime import datetime
    from src.database.connection import get_db
    from src.database.models import Article
    with get_db() as db:
        latest_article = db.query(Article).order_by(Article.published_at.desc()).first()
        if latest_article:
            latest_time = latest_article.published_at
        else:
            latest_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    articles = crawler.get_news_since(latest_time, max_articles=5)
    for article in articles:
        print(f"\n제목: {article.title}")
        print(f"URL: {article.url}")
        print(f"날짜: {article.published_at}")
        print(f"본문 길이: {len(article.content)} 글자") 