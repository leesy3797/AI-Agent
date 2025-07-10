import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Optional, List
import pickle
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

from src.crawler.aitimes import AITimesCrawler, NewsArticle
from src.database.connection import get_db
from src.database.crud import get_user_by_telegram_id, mark_article_as_read, get_unread_articles, get_article_by_id, create_user_read, get_article_by_url, create_article
from src.agents.agentic_rag_agent import AgenticRAGAgent

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# SQLAlchemy 로그 비활성화 (SQL 쿼리 로그 출력 방지)
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)


# 텔레그램 봇 토큰
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN이 설정되지 않았습니다.")

# Gemini API 키 설정
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Gemini 모델 설정
model = genai.GenerativeModel('gemini-1.5-flash')

# 크롤러 인스턴스 생성
crawler = AITimesCrawler()

# 외부 검색 함수 예시
def web_search(query):
    # 실제 구현 필요 (Google/Bing API 등)
    return []

# 벡터스토어 로드 및 앙상블 리트리버 생성
def load_vectorstore():
    """벡터DB 로드 또는 생성"""
    # FAISS 인덱스 디렉토리 생성
    faiss_dir = Path("src/vectorstore")
    faiss_dir.mkdir(exist_ok=True)
    
    try:
        # FAISS 로드
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        
        # FAISS 인덱스 파일 존재 확인
        if not (faiss_dir / "index.faiss").exists():
            raise FileNotFoundError("FAISS 인덱스 파일이 없습니다.")
            
        faiss_all = FAISS.load_local(
            "src/vectorstore", 
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # BM25 로드
        if not os.path.exists('src/vectorstore/bm25_all.pkl'):
            raise FileNotFoundError("BM25 파일이 없습니다.")
            
        with open('src/vectorstore/bm25_all.pkl', 'rb') as f:
            bm25_all = pickle.load(f)
            
        ensemble = build_ensemble_retriever(faiss_all, bm25_all)
        return ensemble
    except FileNotFoundError as e:
        print(f"벡터DB 파일이 없습니다. 새로 생성합니다... ({str(e)})")
        # 임베딩 모델 초기화
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        
        # 빈 FAISS 인덱스 생성
        faiss_all = FAISS.from_texts(
            ["초기화 텍스트"],
            embeddings,
            metadatas=[{"article_id": "init"}]
        )
        
        # 빈 BM25 인덱스 생성
        bm25_all = BM25Retriever.from_texts(
            ["초기화 텍스트"],
            metadatas=[{"article_id": "init"}]
        )
        
        # 파일로 저장
        faiss_all.save_local("src/vectorstore")
        with open('src/vectorstore/bm25_all.pkl', 'wb') as f:
            pickle.dump(bm25_all, f)
            
        ensemble = build_ensemble_retriever(faiss_all, bm25_all)
        return ensemble

# 앙상블 리트리버 생성 함수 (AgenticRAGAgent에서 사용)
def build_ensemble_retriever(faiss_vectorstore, bm25_retriever):
    faiss_retriever = faiss_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})
    bm25_retriever.k = 8
    from langchain.retrievers.ensemble import EnsembleRetriever
    return EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.6, 0.4]  # FAISS에 더 높은 가중치
    )

# embedding_model은 한 번만 생성해서 재사용
rag_agent = AgenticRAGAgent()

# 자동 크롤링 (스케줄러에서 호출)
async def crawl_new_articles():
    """새로운 기사를 크롤링하고 DB에 저장합니다."""
    try:
        logger.info("새로운 기사 크롤링 시작")
        new_articles = crawler.crawl_and_save_latest_news(max_articles=50)
        logger.info(f"자동 크롤링: {len(new_articles)}개 새 기사 저장")
    except Exception as e:
        logger.error(f"크롤링 중 오류 발생: {e}")

async def get_user_news(user_id: str) -> List[NewsArticle]:
    """사용자에게 보여줄 뉴스 목록을 가져옵니다."""
    # DB에서 12시간 이내의 기사 가져오기
    articles = crawler.get_latest_news(max_articles=20)
    logger.info(f"DB에서 가져온 총 기사 수: {len(articles)}")
    
    # DB에서 사용자가 읽지 않은 기사만 필터링
    with get_db() as db:
        user = get_user_by_telegram_id(db, user_id)
        if not user:
            logger.info(f"사용자를 찾을 수 없음: {user_id}")
            return articles
        
        unread_articles = get_unread_articles(db, user.id)
        unread_urls = {article.url for article in unread_articles}
        logger.info(f"읽지 않은 기사 수: {len(unread_urls)}")
        
        filtered_articles = [article for article in articles if article.url in unread_urls]
        logger.info(f"필터링 후 남은 기사 수: {len(filtered_articles)}")
        return filtered_articles

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """사용자가 /start 명령어를 보냈을 때 환영 메시지를 보냅니다."""
    user = update.effective_user
    welcome_message = (
        f"안녕하세요, {user.full_name}님! AI 뉴스 봇에 오신 것을 환영합니다.\n\n"
        "사용 가능한 명령어:\n"
        "/news - 최신 AI 뉴스 보기\n"
        "/help - 도움말 보기"
    )
    await update.message.reply_html(welcome_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """도움말을 보여줍니다."""
    help_text = (
        "�� AI 뉴스 봇 사용 방법\n\n"
        "1. /news - 최신 AI 뉴스 목록을 보여줍니다.\n"
        "2. 뉴스 제목을 클릭하면 요약을 볼 수 있습니다.\n"
        "3. 요약 후 '이 기사로 대화하기'를 선택하면 AI와 대화할 수 있습니다.\n\n"
        "매일 오전 9시에 새로운 뉴스 알림을 받으실 수 있습니다."
    )
    await update.message.reply_text(help_text)

def create_news_keyboard(articles: List[NewsArticle], page: int = 0) -> InlineKeyboardMarkup:
    """뉴스 목록을 위한 인라인 키보드를 생성합니다."""
    items_per_page = 5
    start_idx = page * items_per_page
    end_idx = start_idx + items_per_page
    current_items = articles[start_idx:end_idx]
    
    keyboard = []
    for idx, article in enumerate(current_items):
        marker = f'({idx + 1 + start_idx})'
        keyboard.append([InlineKeyboardButton(f"{marker} {article.title}", callback_data=f"news_{article.url}")])
    
    # 페이지네이션 버튼
    nav_buttons = []
    if page > 0:
        nav_buttons.append(InlineKeyboardButton("⬅️ 이전", callback_data=f"page_{page-1}"))
    if end_idx < len(articles):
        nav_buttons.append(InlineKeyboardButton("다음 ➡️", callback_data=f"page_{page+1}"))
    if nav_buttons:
        keyboard.append(nav_buttons)
    
    return InlineKeyboardMarkup(keyboard)

# DB에서 가장 최근 기사 시간 가져오기
async def get_latest_article_time():
    from src.database.models import Article
    with get_db() as db:
        latest = db.query(Article).order_by(Article.published_at.desc()).first()
        if latest:
            return latest.published_at
        else:
            return None

# /news 명령어: 최신 기사 크롤링 후 뉴스 목록 보여주기
async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE, page: int = 0, do_crawl: bool = True) -> None:
    context.user_data['chat_article_id'] = None  # 기사 대화 세션 초기화
    try:
        user_id = str(update.effective_user.id)
        if do_crawl:
            # 최초 /news 명령어에서만 크롤링
            new_articles = crawler.crawl_and_save_latest_news(max_articles=10) # 50)
            logger.info(f"실시간 크롤링: {len(new_articles)}개 새 기사 저장")
        # DB에서 기사 목록만 페이징
        articles = await get_user_news(user_id)
        if not articles:
            if update.message:
                await update.message.reply_text("새로운 뉴스가 없습니다. 나중에 다시 확인해주세요.")
            elif update.callback_query:
                await update.callback_query.message.reply_text("새로운 뉴스가 없습니다. 나중에 다시 확인해주세요.")
            return
        keyboard = create_news_keyboard(articles, page)
        if update.message:
            await update.message.reply_text(
                "📰 새로운 AI 뉴스 목록입니다. 관심 있는 기사를 선택해주세요:",
                reply_markup=keyboard
            )
        elif update.callback_query:
            await update.callback_query.message.edit_text(
                "📰 새로운 AI 뉴스 목록입니다. 관심 있는 기사를 선택해주세요:",
                reply_markup=keyboard
            )
    except Exception as e:
        import traceback
        logger.error(f"Error in news_command: {e}\n{traceback.format_exc()}")
        if update.message:
            await update.message.reply_text("뉴스를 가져오는 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
        elif update.callback_query:
            await update.callback_query.message.reply_text("뉴스를 가져오는 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")

async def summarize_article(article):
    """Gemini를 사용하여 기사를 요약합니다. (DB 캐싱)"""
    try:
        # DB에서 summary가 있으면 바로 반환
        with get_db() as db:
            db_article = get_article_by_url(db, article.url)
            if db_article and db_article.summary:
                return f"📝 {article.title}\n\n{db_article.summary}"
        # summary가 없으면 Gemini로 요약
        prompt = f"""
주어진 AI 뉴스 기사의 핵심 내용을 가장 정확하고 충실하게 요약해주세요.

요약 시 다음 지침을 엄수해주세요:

1.  **기사의 핵심 정보와 주요 사실만을 간결하게 요약합니다.**
2.  **개인적인 의견, 해석, 또는 기사에 없는 내용은 절대 추가하지 않습니다.** 오직 기사 내용에 기반한 객관적인 정보를 제공해야 합니다.
3.  요약은 3-4문장으로 구성하며, AI 및 기술 분야에 관심 있는 독자들이 기사의 가장 중요한 내용을 신속하게 파악할 수 있도록 합니다.

---

제목: {article.title}

내용:
{article.content}
"""
        
        # Gemini로 요약 생성
        response = await model.generate_content_async(prompt)
        summary = response.text
        # DB에 저장
        with get_db() as db:
            db_article = get_article_by_url(db, article.url)
            if db_article:
                db_article.summary = summary
                db.commit()
        return f"📝 {article.title}\n\n{summary}"
    except Exception as e:
        logger.error(f"기사 요약 중 오류 발생: {str(e)}")
        return f"📝 {article.title}\n\n{article.content[:200]}..."  # 오류 발생시 기존 방식으로 대체

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """인라인 키보드 버튼 콜백을 처리합니다."""
    query = update.callback_query
    await query.answer()
    
    try:
        if query.data.startswith("article_"):
            article_id = int(query.data.split("_")[1])
            with get_db() as db:
                article = get_article_by_id(db, article_id)
                if article:
                    # Gemini로 요약
                    summary = await summarize_article(article)
                    
                    # 요약 후 액션 버튼
                    keyboard = [
                        [
                            InlineKeyboardButton("🔗 원문 기사 보기", url=article.url),
                            InlineKeyboardButton("💬 이 기사로 대화하기", callback_data=f"chat_{article.url}")
                        ],
                        [
                            InlineKeyboardButton("목록으로 돌아가기", callback_data="back_to_list")
                        ]
                    ]
                    
                    await query.message.reply_text(
                        summary,
                        reply_markup=InlineKeyboardMarkup(keyboard)
                    )
                    
                    # 사용자의 읽은 기사 기록
                    user = get_user_by_telegram_id(db, update.effective_user.id)
                    if user:
                        create_user_read(db, user.id, article_id)
                else:
                    await query.message.reply_text("기사를 찾을 수 없습니다.")
        
        elif query.data.startswith("news_"):
            # 뉴스 아이템 선택 처리
            article_url = query.data[5:]  # "news_" 제거
            with get_db() as db:
                article = get_article_by_url(db, article_url)
                if article:
                    # 기사를 읽음으로 표시
                    user = get_user_by_telegram_id(db, str(query.from_user.id))
                    if user:
                        mark_article_as_read(db, user.id, article.id)
                    
                    # Gemini로 요약
                    summary = await summarize_article(article)
                    
                    # 요약 후 액션 버튼
                    keyboard = [
                        [
                            InlineKeyboardButton("🔗 원문 기사 보기", url=article.url),
                            InlineKeyboardButton("💬 이 기사로 대화하기", callback_data=f"chat_{article.url}")
                        ],
                        [
                            InlineKeyboardButton("목록으로 돌아가기", callback_data="back_to_list")
                        ]
                    ]
                    await query.message.reply_text(
                        summary,
                        reply_markup=InlineKeyboardMarkup(keyboard)
                    )
                else:
                    await query.message.reply_text("죄송합니다. 기사를 가져오는 중 오류가 발생했습니다.")
        
        elif query.data.startswith("page_"):
            # 페이지네이션 처리 (크롤링 없이)
            page = int(query.data.split("_")[1])
            await news_command(update, context, page, do_crawl=False)
        
        elif query.data.startswith("chat_"):
            # 채팅 시작 처리
            article_url = query.data[5:]  # "chat_" 제거
            context.user_data['chat_article_id'] = article_url  # 항상 최신 기사로 덮어쓰기
            await start_chat(update, context, article_url)
        
        elif query.data == "back_to_list":
            # 항상 첫 페이지 뉴스 목록으로 이동
            await news_command(update, context, page=0)
    
    except Exception as e:
        logger.error(f"버튼 콜백 처리 중 오류 발생: {str(e)}")
        await query.message.reply_text("죄송합니다. 처리 중 오류가 발생했습니다.")

async def error_handler(update: Optional[Update], context: ContextTypes.DEFAULT_TYPE) -> None:
    """에러를 처리하고 로깅합니다."""
    logger.error(f"Update {update} caused error {context.error}")
    if update and update.effective_message:
        await update.effective_message.reply_text(
            "죄송합니다. 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        )

# 챗봇 세션 핸들러 예시
async def start_chat(update: Update, context: ContextTypes.DEFAULT_TYPE, article_url: str):
    user_id = str(update.effective_user.id)
    article_id = article_url  # URL을 ID로 사용
    await update.callback_query.message.reply_text("이 기사에 대해 궁금한 점을 입력해 주세요!")
    # 다음 메시지를 챗봇 세션으로 연결
    context.user_data['chat_article_id'] = article_id  # article.url로 저장

async def handle_user_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_question = update.message.text
        articles = await get_user_news(str(update.effective_user.id))
        documents = [
            Document(page_content=article.content, metadata={"article_id": article.url, "title": getattr(article, "title", None)})
            for article in articles
        ]
        article_id = context.user_data.get('chat_article_id')
        answer = rag_agent.answer(user_question, documents, article_id=article_id)
        await update.message.reply_text(answer)
    except Exception as e:
        logger.error(f"handle_user_message error: {e}", exc_info=True)
        # 중복 에러 메시지 전송 방지: 사용자에게는 에러 메시지 전송하지 않음

def main():
    # 봇 시작
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    # 명령어 핸들러 등록
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("news", lambda update, context: news_command(update, context, page=0, do_crawl=True)))
    application.add_handler(CallbackQueryHandler(button_callback, pattern="^news_|^page_|^chat_|^article_|^back_to_list$"))
    
    # 에러 핸들러 등록
    application.add_error_handler(error_handler)
    
    # 스케줄러는 BackgroundScheduler로 실행
    scheduler = BackgroundScheduler()
    scheduler.add_job(crawl_new_articles, 'cron', hour='*')
    scheduler.start()

    # 핸들러 등록 예시 (main 함수 내)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_message))

    logger.info("텔레그램 봇이 시작되었습니다. (Ctrl+C로 종료)")
    application.run_polling(timeout=60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("봇이 종료되었습니다.")