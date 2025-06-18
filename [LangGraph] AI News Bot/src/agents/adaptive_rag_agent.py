import pickle
from typing import List, Tuple, Dict, Callable, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import logging

# 로깅 설정
logger = logging.getLogger("hybrid_rag_agent")
logger.setLevel(logging.INFO)

# Grader 프롬프트
GRADE_PROMPT = """
아래는 사용자의 질문과 검색된 문서(또는 context)입니다.

[질문]
{question}

[검색된 문서/컨텍스트]
{context}

이 context만으로 사용자의 질문에 충분히 답변할 수 있으면 'yes', 부족하면 'no'로만 답변하세요.
"""

class AdaptiveRAGAgent:
    def __init__(self, ensemble_retriever, web_search_func: Callable[[str], List[str]]):
        self.ensemble_retriever = ensemble_retriever
        self.web_search_func = web_search_func
        self.user_sessions = {}
        self.grader_model = genai.GenerativeModel('models/gemini-2.0-flash')

    def context_sufficient_llm(self, question: str, context: str) -> bool:
        prompt = GRADE_PROMPT.format(question=question, context=context)
        logger.info(f"[Grader] 질문: {question}\n컨텍스트 길이: {len(context)}")
        response = self.grader_model.generate_content(prompt)
        answer = response.text.strip().lower()
        logger.info(f"[Grader] LLM 판정 결과: {answer}")
        return answer.startswith("yes")

    def agentic_rag_pipeline(self, user_question: str, article_id: Optional[str] = None, chat_history: List[Tuple[str, str]] = None) -> Dict:
        state = {
            "user_question": user_question,
            "context": [],
            "step": "article_search",
            "trace": []
        }
        # 1단계: 기사별 필터 검색
        logger.info("[RAG] 1단계: 기사별 필터 검색 시작")
        filter_dict = {"article_id": article_id} if article_id else None
        context = self.ensemble_retriever.invoke(user_question, filter=filter_dict)
        state["context"] = context
        state["trace"].append(("article_search", [doc.page_content[:100] for doc in context]))
        if context and self.context_sufficient_llm(user_question, "\n\n".join([doc.page_content for doc in context])):
            logger.info("[RAG] 기사 context로 충분, 답변 생성")
            state["step"] = "article_search"
            return state
        # 2단계: 전체 검색 (필터 없이)
        logger.info("[RAG] 2단계: 전체 기사 검색 시작")
        context = self.ensemble_retriever.invoke(user_question)
        state["context"] = context
        state["trace"].append(("all_search", [doc.page_content[:100] for doc in context]))
        if context and self.context_sufficient_llm(user_question, "\n\n".join([doc.page_content for doc in context])):
            logger.info("[RAG] 전체 기사 context로 충분, 답변 생성")
            state["step"] = "all_search"
            return state
        # 3단계: 외부 검색
        logger.info("[RAG] 3단계: 외부 검색 시작")
        context = self.web_search_func(user_question)
        state["context"] = context
        state["trace"].append(("web_search", context))
        state["step"] = "web_search"
        return state

    def build_prompt(self, context: List, user_question: str, chat_history: List[Tuple[str, str]]) -> str:
        context_text = "\n\n".join([doc.page_content for doc in context])
        history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in chat_history])
        prompt = f"""
[대화 내역]
{history_text}

[참고 기사/문서 내용]
{context_text}

[질문]
{user_question}

위 참고 내용과 대화 내역을 바탕으로, 사용자의 질문에 대해 정확하고 친절하게 답변해 주세요.
"""
        return prompt

    def generate_with_gemini(self, prompt: str) -> str:
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text

    def chat_with_agent(self, user_id: str, article_id: Optional[str], user_question: str):
        session_key = (user_id, article_id)
        chat_history = self.user_sessions.get(session_key, [])
        rag_result = self.agentic_rag_pipeline(user_question, article_id, chat_history)
        context_docs = rag_result["context"]
        prompt = self.build_prompt(context_docs, user_question, chat_history)
        answer = self.generate_with_gemini(prompt)
        chat_history.append((user_question, answer))
        self.user_sessions[session_key] = chat_history
        # 트래킹 로그 출력
        logger.info(f"[RAG Trace] 단계별 검색 결과: {rag_result['trace']}")
        logger.info(f"[RAG Trace] 최종 답변 단계: {rag_result['step']}")
        return answer

# 앙상블 리트리버 생성 함수 (유틸)
def build_ensemble_retriever(faiss_vectorstore, bm25_retriever):
    faiss_retriever = faiss_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    bm25_retriever.k = 5
    return EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

def create_and_save_vectorstores(documents, faiss_path, bm25_path, embedding_model=None):
    # 문서에 article_id 메타데이터가 포함되어 있어야 함
    faiss_vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embedding_model
    )
    with open(faiss_path, "wb") as f:
        pickle.dump(faiss_vectorstore, f)
    bm25_retriever = BM25Retriever.from_documents(documents)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
    return faiss_vectorstore, bm25_retriever 