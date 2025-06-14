from typing import Optional, List, Dict, TypedDict
from pydantic import BaseModel

class EmailData(BaseModel):
    id: str
    sender: str
    subject: str
    body: str

class Task(BaseModel):
    description: str
    due_date: Optional[str] = None
    priority: Optional[str] = None
    status: str = "pending"

class AgentState(TypedDict):
    # 현재 처리 중인 이메일 정보
    current_email: Optional[EmailData]
    
    # 이메일 분석 결과
    email_summary: Optional[str]
    email_intent: Optional[str]
    extracted_tasks: List[Task]
    
    # 사용자 상호작용
    user_command: Optional[str]
    user_reply_text: Optional[str]
    llm_draft_response: Optional[str]
    
    # 작업 상태
    action_status: Optional[str]
    action_error: Optional[str]
    
    # Telegram 관련
    telegram_chat_id: Optional[int]
    telegram_message_id: Optional[int]
    
    # 배치 처리
    last_processed_email_ids: List[str]

def create_initial_state() -> AgentState:
    """초기 에이전트 상태를 생성합니다."""
    return {
        "current_email": None,
        "email_summary": None,
        "email_intent": None,
        "extracted_tasks": [],
        "user_command": None,
        "user_reply_text": None,
        "llm_draft_response": None,
        "action_status": None,
        "action_error": None,
        "telegram_chat_id": None,
        "telegram_message_id": None,
        "last_processed_email_ids": []
    } 