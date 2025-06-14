from typing import Dict, List, Tuple, Any
from langgraph.graph import StateGraph, END
import google.generativeai as genai
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from .agent_state import AgentState, EmailData, Task
from .gmail_client import GmailClient
from .telegram_client import TelegramClient
from .config import settings

# Gemini 초기화
genai.configure(api_key=settings.GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# 이메일 분석 프롬프트
EMAIL_ANALYSIS_PROMPT = """당신은 이메일을 분석하고 요약하는 AI 어시스턴트입니다.
다음 정보를 추출해주세요:
1. 이메일의 주요 내용 요약
2. 발신자의 의도 (정보 요청, 작업 지시, 홍보 등)
3. 이메일에서 추출된 작업 목록 (있는 경우)

응답은 다음 형식으로 해주세요:
{
    "summary": "이메일 요약",
    "intent": "발신자의 의도",
    "tasks": [
        {
            "description": "작업 설명",
            "due_date": "마감일 (있는 경우)",
            "priority": "우선순위 (높음/중간/낮음)"
        }
    ]
}

이메일 내용:
{email_content}"""

# 답장 생성 프롬프트
REPLY_DRAFT_PROMPT = """당신은 전문적인 이메일 답장을 작성하는 AI 어시스턴트입니다.
다음 정보를 바탕으로 적절한 답장을 작성해주세요:
1. 원본 이메일의 내용과 의도
2. 사용자가 원하는 답장의 톤과 스타일

답장은 전문적이고 명확해야 하며, 원본 이메일의 모든 질문이나 요청사항에 대해 적절히 대응해야 합니다.

원본 이메일:
{original_email}

사용자 지시사항:
{user_instructions}"""

class EmailAnalysisNode:
    def __init__(self, gmail_client: GmailClient, telegram_client: TelegramClient):
        self.gmail_client = gmail_client
        self.telegram_client = telegram_client
    
    async def __call__(self, state: AgentState) -> AgentState:
        """이메일을 분석하고 요약합니다."""
        if not state["current_email"]:
            return state
        
        # 이메일 내용 준비
        email_content = f"""
        제목: {state['current_email'].subject}
        발신자: {state['current_email'].sender}
        내용: {state['current_email'].body}
        """
        
        # Gemini로 분석
        try:
            response = model.generate_content(
                EMAIL_ANALYSIS_PROMPT.format(email_content=email_content)
            )
            analysis = response.text
            
            # JSON 파싱
            import json
            analysis_dict = json.loads(analysis)
            
            state["email_summary"] = analysis_dict["summary"]
            state["email_intent"] = analysis_dict["intent"]
            state["extracted_tasks"] = [
                Task(**task) for task in analysis_dict["tasks"]
            ]
        except Exception as e:
            state["action_error"] = f"이메일 분석 실패: {str(e)}"
        
        return state

class TelegramNotificationNode:
    def __init__(self, telegram_client: TelegramClient):
        self.telegram_client = telegram_client
    
    async def __call__(self, state: AgentState) -> AgentState:
        """분석된 이메일 정보를 Telegram으로 전송합니다."""
        if not state["current_email"] or not state["email_summary"]:
            return state
        
        # 기본 액션 버튼
        actions = [
            {"text": "📝 답장", "action": "reply"},
            {"text": "✅ 읽음", "action": "mark_read"},
            {"text": "🗑️ 삭제", "action": "delete"},
            {"text": "📋 작업 추가", "action": "add_task"}
        ]
        
        # Telegram으로 알림 전송
        await self.telegram_client.send_email_notification(
            chat_id=state["telegram_chat_id"],
            email_data=state["current_email"].dict(),
            summary=state["email_summary"],
            actions=actions
        )
        
        return state

class ReplyDraftingNode:
    def __init__(self, gmail_client: GmailClient):
        self.gmail_client = gmail_client
    
    async def __call__(self, state: AgentState) -> AgentState:
        """이메일 답장 초안을 생성합니다."""
        if not state["current_email"] or not state["user_command"] == "reply":
            return state
        
        # 답장 생성
        try:
            response = model.generate_content(
                REPLY_DRAFT_PROMPT.format(
                    original_email=state["current_email"].dict(),
                    user_instructions=state["user_reply_text"] or "전문적이고 정중한 톤으로 답장해주세요."
                )
            )
            state["llm_draft_response"] = response.text
        except Exception as e:
            state["action_error"] = f"답장 생성 실패: {str(e)}"
        
        return state

class ActionRouterNode:
    def __init__(self, gmail_client: GmailClient, telegram_client: TelegramClient):
        self.gmail_client = gmail_client
        self.telegram_client = telegram_client
    
    async def __call__(self, state: AgentState) -> AgentState:
        """사용자 명령에 따라 적절한 액션을 수행합니다."""
        if not state["user_command"]:
            return state
        
        command = state["user_command"].split(":")[0]
        email_id = state["current_email"].id if state["current_email"] else None
        
        try:
            if command == "reply":
                # 답장 초안 생성 및 전송
                await self.telegram_client.send_reply_draft(
                    chat_id=state["telegram_chat_id"],
                    draft_text=state["llm_draft_response"],
                    message_id=email_id
                )
            elif command == "send_reply":
                # 답장 전송
                success = self.gmail_client.send_reply(
                    email_id,
                    state["llm_draft_response"]
                )
                state["action_status"] = "success" if success else "failed"
            elif command == "mark_read":
                # 읽음 표시
                success = self.gmail_client.mark_as_read(email_id)
                state["action_status"] = "success" if success else "failed"
            elif command == "delete":
                # 이메일 삭제
                success = self.gmail_client.delete_email(email_id)
                state["action_status"] = "success" if success else "failed"
            
            # 작업 상태 알림
            if state["action_status"]:
                status_text = "성공적으로 처리되었습니다." if state["action_status"] == "success" else "처리 중 오류가 발생했습니다."
                await self.telegram_client.send_confirmation(
                    chat_id=state["telegram_chat_id"],
                    message=status_text,
                    confirm_action="continue",
                    cancel_action="end"
                )
        
        except Exception as e:
            state["action_error"] = str(e)
            state["action_status"] = "failed"
        
        return state

def create_workflow(
    gmail_client: GmailClient,
    telegram_client: TelegramClient
) -> StateGraph:
    """LangGraph 워크플로우를 생성합니다."""
    # 노드 초기화
    email_analysis = EmailAnalysisNode(gmail_client, telegram_client)
    telegram_notification = TelegramNotificationNode(telegram_client)
    reply_drafting = ReplyDraftingNode(gmail_client)
    action_router = ActionRouterNode(gmail_client, telegram_client)
    
    # 워크플로우 그래프 생성
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("analyze_email", email_analysis)
    workflow.add_node("notify_telegram", telegram_notification)
    workflow.add_node("draft_reply", reply_drafting)
    workflow.add_node("route_action", action_router)
    
    # 엣지 추가
    workflow.add_edge("analyze_email", "notify_telegram")
    workflow.add_edge("notify_telegram", "route_action")
    workflow.add_edge("draft_reply", "route_action")
    
    # 조건부 엣지
    workflow.add_conditional_edges(
        "route_action",
        lambda x: "draft_reply" if x["user_command"] == "reply" else END
    )
    
    # 시작 노드 설정
    workflow.set_entry_point("analyze_email")
    
    return workflow 