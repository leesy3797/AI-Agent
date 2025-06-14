from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from typing import Dict, List, Optional, Callable
import asyncio
from .config import settings

class TelegramClient:
    def __init__(self):
        self.application = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()
        self.command_handlers: Dict[str, Callable] = {}
        self.callback_handlers: Dict[str, Callable] = {}
        self._setup_handlers()
    
    def _setup_handlers(self):
        """기본 핸들러를 설정합니다."""
        self.application.add_handler(CommandHandler("start", self._start_command))
        self.application.add_handler(CallbackQueryHandler(self._handle_callback))
    
    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """봇 시작 명령을 처리합니다."""
        await update.message.reply_text(
            "안녕하세요! Gmail 에이전트 봇입니다. "
            "새로운 이메일이 도착하면 알려드리겠습니다."
        )
    
    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """콜백 쿼리를 처리합니다."""
        query = update.callback_query
        await query.answer()
        
        if query.data in self.callback_handlers:
            await self.callback_handlers[query.data](update, context)
    
    def register_command(self, command: str, handler: Callable):
        """새로운 명령 핸들러를 등록합니다."""
        self.command_handlers[command] = handler
        self.application.add_handler(CommandHandler(command, handler))
    
    def register_callback(self, callback_data: str, handler: Callable):
        """새로운 콜백 핸들러를 등록합니다."""
        self.callback_handlers[callback_data] = handler
    
    async def send_email_notification(
        self,
        chat_id: int,
        email_data: Dict,
        summary: str,
        actions: List[Dict[str, str]]
    ):
        """이메일 알림을 전송합니다."""
        # 인라인 키보드 생성
        keyboard = []
        for action in actions:
            keyboard.append([
                InlineKeyboardButton(
                    action["text"],
                    callback_data=f"{action['action']}:{email_data['id']}"
                )
            ])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # 메시지 텍스트 구성
        message_text = (
            f"📧 새 이메일 도착!\n\n"
            f"📨 보낸 사람: {email_data['sender']}\n"
            f"📝 제목: {email_data['subject']}\n\n"
            f"📋 요약:\n{summary}"
        )
        
        # 메시지 전송
        await self.application.bot.send_message(
            chat_id=chat_id,
            text=message_text,
            reply_markup=reply_markup
        )
    
    async def send_confirmation(
        self,
        chat_id: int,
        message: str,
        confirm_action: str,
        cancel_action: str
    ):
        """확인 메시지를 전송합니다."""
        keyboard = [
            [
                InlineKeyboardButton("확인", callback_data=confirm_action),
                InlineKeyboardButton("취소", callback_data=cancel_action)
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await self.application.bot.send_message(
            chat_id=chat_id,
            text=message,
            reply_markup=reply_markup
        )
    
    async def send_reply_draft(
        self,
        chat_id: int,
        draft_text: str,
        message_id: str
    ):
        """답장 초안을 전송합니다."""
        keyboard = [
            [
                InlineKeyboardButton("전송", callback_data=f"send_reply:{message_id}"),
                InlineKeyboardButton("수정", callback_data=f"edit_reply:{message_id}"),
                InlineKeyboardButton("취소", callback_data=f"cancel_reply:{message_id}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await self.application.bot.send_message(
            chat_id=chat_id,
            text=f"📝 답장 초안:\n\n{draft_text}",
            reply_markup=reply_markup
        )
    
    def run(self):
        """봇을 실행합니다."""
        self.application.run_polling()
    
    async def stop(self):
        """봇을 중지합니다."""
        await self.application.stop() 