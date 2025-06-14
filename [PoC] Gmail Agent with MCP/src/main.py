import asyncio
import logging
from typing import Dict, List
from .gmail_client import GmailClient
from .telegram_client import TelegramClient
from .agent_state import AgentState, EmailData, create_initial_state
from .agent_nodes import create_workflow
from .config import settings

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GmailAgent:
    def __init__(self):
        self.gmail_client = GmailClient()
        self.telegram_client = TelegramClient()
        self.workflow = create_workflow(self.gmail_client, self.telegram_client)
        self._setup_telegram_handlers()
    
    def _setup_telegram_handlers(self):
        """Telegram 봇 핸들러를 설정합니다."""
        # 시작 명령어 핸들러
        self.telegram_client.register_command("start", self._handle_start)
        
        # 콜백 핸들러
        self.telegram_client.register_callback("reply", self._handle_reply)
        self.telegram_client.register_callback("mark_read", self._handle_mark_read)
        self.telegram_client.register_callback("delete", self._handle_delete)
        self.telegram_client.register_callback("add_task", self._handle_add_task)
        self.telegram_client.register_callback("send_reply", self._handle_send_reply)
        self.telegram_client.register_callback("edit_reply", self._handle_edit_reply)
        self.telegram_client.register_callback("cancel_reply", self._handle_cancel_reply)
    
    async def _handle_start(self, update, context):
        """봇 시작 명령을 처리합니다."""
        chat_id = update.effective_chat.id
        await self.telegram_client.send_message(
            chat_id=chat_id,
            text="Gmail 에이전트가 시작되었습니다. 새로운 이메일이 도착하면 알려드리겠습니다."
        )
    
    async def _handle_reply(self, update, context):
        """답장 요청을 처리합니다."""
        query = update.callback_query
        email_id = query.data.split(":")[1]
        
        # 상태 업데이트
        state = create_initial_state()
        state["current_email"] = EmailData(
            id=email_id,
            sender="",  # Gmail API에서 가져올 예정
            subject="",
            body=""
        )
        state["user_command"] = "reply"
        state["telegram_chat_id"] = query.message.chat_id
        
        # 워크플로우 실행
        await self.workflow.ainvoke(state)
    
    async def _handle_mark_read(self, update, context):
        """읽음 표시 요청을 처리합니다."""
        query = update.callback_query
        email_id = query.data.split(":")[1]
        
        state = create_initial_state()
        state["current_email"] = EmailData(
            id=email_id,
            sender="",
            subject="",
            body=""
        )
        state["user_command"] = "mark_read"
        state["telegram_chat_id"] = query.message.chat_id
        
        await self.workflow.ainvoke(state)
    
    async def _handle_delete(self, update, context):
        """이메일 삭제 요청을 처리합니다."""
        query = update.callback_query
        email_id = query.data.split(":")[1]
        
        state = create_initial_state()
        state["current_email"] = EmailData(
            id=email_id,
            sender="",
            subject="",
            body=""
        )
        state["user_command"] = "delete"
        state["telegram_chat_id"] = query.message.chat_id
        
        await self.workflow.ainvoke(state)
    
    async def _handle_add_task(self, update, context):
        """작업 추가 요청을 처리합니다."""
        query = update.callback_query
        email_id = query.data.split(":")[1]
        
        state = create_initial_state()
        state["current_email"] = EmailData(
            id=email_id,
            sender="",
            subject="",
            body=""
        )
        state["user_command"] = "add_task"
        state["telegram_chat_id"] = query.message.chat_id
        
        await self.workflow.ainvoke(state)
    
    async def _handle_send_reply(self, update, context):
        """답장 전송 요청을 처리합니다."""
        query = update.callback_query
        email_id = query.data.split(":")[1]
        
        state = create_initial_state()
        state["current_email"] = EmailData(
            id=email_id,
            sender="",
            subject="",
            body=""
        )
        state["user_command"] = "send_reply"
        state["telegram_chat_id"] = query.message.chat_id
        
        await self.workflow.ainvoke(state)
    
    async def _handle_edit_reply(self, update, context):
        """답장 수정 요청을 처리합니다."""
        query = update.callback_query
        email_id = query.data.split(":")[1]
        
        await self.telegram_client.send_message(
            chat_id=query.message.chat_id,
            text="수정할 답장 내용을 입력해주세요."
        )
    
    async def _handle_cancel_reply(self, update, context):
        """답장 취소 요청을 처리합니다."""
        query = update.callback_query
        await query.message.edit_text("답장이 취소되었습니다.")
    
    async def check_new_emails(self):
        """새로운 이메일을 확인하고 처리합니다."""
        while True:
            try:
                # 읽지 않은 이메일 가져오기
                unread_emails = self.gmail_client.get_unread_emails(
                    max_results=settings.MAX_EMAILS_PER_POLL
                )
                
                for email in unread_emails:
                    # 상태 초기화
                    state = create_initial_state()
                    state["current_email"] = EmailData(**email)
                    
                    # 워크플로우 실행
                    await self.workflow.ainvoke(state)
                
                # 대기
                await asyncio.sleep(settings.POLLING_INTERVAL)
            
            except Exception as e:
                logger.error(f"이메일 확인 중 오류 발생: {str(e)}")
                await asyncio.sleep(settings.POLLING_INTERVAL)
    
    def run(self):
        """에이전트를 실행합니다."""
        # Telegram 봇 시작
        self.telegram_client.run()
        
        # 이메일 확인 루프 시작
        asyncio.run(self.check_new_emails())

if __name__ == "__main__":
    agent = GmailAgent()
    agent.run() 