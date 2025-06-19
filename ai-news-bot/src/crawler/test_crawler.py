import logging
from datetime import datetime
from typing import List, Optional

from src.crawler.aitimes import AITimesCrawler
from src.database.crud import get_article_by_url
from src.database.connection import get_db

# 로깅 설정
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_crawler():
    """크롤러 테스트"""
    try:
        # 크롤러 인스턴스 생성
        crawler = AITimesCrawler()
        logger.info("크롤러 인스턴스 생성 완료")

        # 최신 뉴스 크롤링 테스트
        logger.info("최신 뉴스 크롤링 시작...")
        articles = crawler.get_latest_news(max_articles=3)  # 3개만 테스트
        
        if not articles:
            logger.error("뉴스를 가져오지 못했습니다.")
            return
        
        logger.info(f"크롤링된 뉴스 수: {len(articles)}")
        
        # 각 기사 상세 정보 출력
        for i, article in enumerate(articles, 1):
            print(f"\n{'='*50}")
            print(f"기사 {i}")
            print(f"제목: {article.title}")
            print(f"URL: {article.url}")
            print(f"발행일: {article.published_at}")
            print(f"본문 길이: {len(article.content)} 글자")
            print(f"본문 미리보기: {article.content[:200]}...")
            print(f"{'='*50}\n")

        # 특정 기사 크롤링 테스트
        if articles:
            test_url = articles[0].url
            logger.info(f"특정 기사 크롤링 테스트: {test_url}")
            article = crawler.get_article_by_url(test_url)
            
            if article:
                print("\n특정 기사 크롤링 결과:")
                print(f"제목: {article.title}")
                print(f"URL: {article.url}")
                print(f"발행일: {article.published_at}")
                print(f"본문 길이: {len(article.content)} 글자")
                print(f"본문 미리보기: {article.content[:200]}...")
            else:
                logger.error("특정 기사 크롤링 실패")

    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {e}")

if __name__ == "__main__":
    test_crawler() 