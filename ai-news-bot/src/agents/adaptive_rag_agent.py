import pickle
from langgraph.graph import StateGraph, START, END
from typing import List, Tuple, Dict, Callable, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langgraph.graph import MessagesState
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import logging
from langchain_core.documents import Document

# Grader 프롬프트
GRADE_PROMPT = """
아래는 사용자의 질문과 검색된 문서(또는 context)입니다.

[질문]
{question}

[검색된 문서/컨텍스트]
{context}

이 context만으로 사용자의 질문에 충분히 답변할 수 있으면 'yes', 부족하면 'no'로만 답변하세요.

답변 기준:
- context에 질문과 관련된 구체적인 정보가 있으면 'yes'
- context가 비어있거나 질문과 관련 없는 내용이면 'no'
- context가 있지만 질문에 대한 구체적인 답변이 불가능하면 'no'

답변: """

# Gemini2.0-flash LLM을 이용해 context가 답변에 충분한지 평가하는 함수
# (VectorDB에서 Retrieve된 문서가 user query에 대해 답변하기에 적절/충분한지 판정)
def context_sufficient_llm(question: str, context: str) -> bool:
    """
    Gemini2.0-flash grader_model을 사용하여, 주어진 context(문서 집합)가
    user query(질문)에 대해 답변하기에 충분한지 평가합니다.
    - 충분하면 'yes', 부족하면 'no'를 반환하도록 프롬프트 설계
    - 'yes'로 시작하면 sufficient, 아니면 insufficient으로 간주
    """
    grader_model = genai.GenerativeModel('models/gemini-2.0-flash')
    prompt = GRADE_PROMPT.format(question=question, context=context)
    print(f"[context_sufficient_llm] 질문: {question}")
    print(f"[context_sufficient_llm] 컨텍스트 길이: {len(context)}")
    print(f"[context_sufficient_llm] 컨텍스트 미리보기: {context[:200]} ...")
    
    response = grader_model.generate_content(prompt)
    answer = response.text.strip().lower()
    print(f"[context_sufficient_llm] LLM 판정 결과: '{answer}'")
    
    is_sufficient = answer.startswith("yes")
    print(f"[context_sufficient_llm] 최종 판정: {is_sufficient}")
    return is_sufficient


# 앙상블 리트리버 생성 함수 (유틸)
def build_ensemble_retriever(faiss_vectorstore, bm25_retriever):
    faiss_retriever = faiss_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    bm25_retriever.k = 5
    return EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

def create_vectorstores(documents, embedding_model):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=100,
        model_name="gpt-4"
    )
    chunked_documents = []
    for doc in documents:
        chunked = text_splitter.split_documents([doc])
        chunked_documents.extend(chunked)
    faiss_vectorstore = FAISS.from_documents(
        documents=chunked_documents,
        embedding=embedding_model
    )
    bm25_retriever = BM25Retriever.from_documents(chunked_documents)
    return faiss_vectorstore, bm25_retriever

def deduplicate_documents(docs):
    # page_content 기준으로 중복 제거
    seen = set()
    unique_docs = []
    for doc in docs:
        key = getattr(doc, "page_content", str(doc))
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    return unique_docs

def get_user_question_from_state(state):
    for msg in reversed(state["messages"]):
        # MessagesState의 메시지는 HumanMessage/AIMessage 등 객체임
        if getattr(msg, "type", None) == "human":
            return msg.content
    raise ValueError("user 메시지가 없습니다.")

def article_search_node(state, ensemble_retriever, article_id=None):
    print('단일 기사 검색 노드')
    user_question = get_user_question_from_state(state)
    # context에 사용할 문서만 추출
    if article_id:
        # faiss_retriever에서 전체 문서 추출
        faiss_retriever = ensemble_retriever.retrievers[0]
        all_docs = faiss_retriever.vectorstore.docstore._dict.values()
        article_docs = [doc for doc in all_docs if doc.metadata.get("article_id") == article_id]
        # 임시로 해당 기사만으로 retriever 생성
        if article_docs:
            faiss_vectorstore, bm25_retriever = create_vectorstores(article_docs, faiss_retriever.vectorstore.embedding_function)
            temp_ensemble = build_ensemble_retriever(faiss_vectorstore, bm25_retriever)
            context = temp_ensemble.invoke(user_question)
        else:
            context = []
    else:
        # article_id가 없으면 빈 context 반환 (전체 검색은 all_search_node에서 수행)
        context = []
    print(f"[article_search_node] context 개수: {len(context)}")
    for i, doc in enumerate(context):
        print(f"  [{i}] {getattr(doc, 'page_content', str(doc))[:80]} ...")
    state = dict(state)
    prev_context = state.get("context", [])
    # str이 섞여 있으면 Document로 변환
    prev_context = [doc if hasattr(doc, 'page_content') else Document(page_content=str(doc)) for doc in prev_context]
    context = [doc if hasattr(doc, 'page_content') else Document(page_content=str(doc)) for doc in context]
    new_context = deduplicate_documents(prev_context + context)
    state["context"] = new_context
    state["step"] = "article_search"
    state.setdefault("trace", []).append(("article_search", [doc.page_content[:80] + ' ...' for doc in context]))
    return state

def all_search_node(state, ensemble_retriever):
    print('전체 기사 검색 노드')
    user_question = get_user_question_from_state(state)
    context = ensemble_retriever.invoke(user_question)
    print(f"[all_search_node] context 개수: {len(context)}")
    for i, doc in enumerate(context):
        print(f"  [{i}] {getattr(doc, 'page_content', str(doc))[:80]} ...")
    state = dict(state)
    prev_context = state.get("context", [])
    prev_context = [doc if hasattr(doc, 'page_content') else Document(page_content=str(doc)) for doc in prev_context]
    context = [doc if hasattr(doc, 'page_content') else Document(page_content=str(doc)) for doc in context]
    new_context = deduplicate_documents(prev_context + context)
    state["context"] = new_context
    state["step"] = "all_search"
    state.setdefault("trace", []).append(("all_search", [doc.page_content[:80] + ' ...' for doc in context]))
    return state

def web_search_func(query):
    search = DuckDuckGoSearchAPIWrapper()
    results = search.results(query=query, max_results=10)
    if not results:
        return [Document(page_content=f"검색 결과 없음: {query}")]
    return [Document(page_content=f"{item['title']} - {item.get('snippet','')}") for item in results]

def web_search_node(state):
    print('인터넷 검색 노드')
    user_question = get_user_question_from_state(state)
    context = web_search_func(user_question)
    print(f"[web_search_node] context 개수: {len(context)}")
    for i, doc in enumerate(context):
        print(f"  [{i}] {getattr(doc, 'page_content', str(doc))[:80]} ...")
    state = dict(state)
    prev_context = state.get("context", [])
    prev_context = [doc if hasattr(doc, 'page_content') else Document(page_content=str(doc)) for doc in prev_context]
    context = [doc if hasattr(doc, 'page_content') else Document(page_content=str(doc)) for doc in context]
    new_context = deduplicate_documents(prev_context + context)
    state["context"] = new_context
    state["step"] = "web_search"
    state.setdefault("trace", []).append(("web_search", [getattr(doc, 'page_content', str(doc))[:80] + ' ...' for doc in context]))
    return state

def context_sufficiency_condition(state) -> str:
    print('LLM 문맥 평가 분기')
    user_question = get_user_question_from_state(state)
    context = state.get("context", [])
    
    # context가 비어있으면 무조건 insufficient
    if not context:
        print("[context_sufficiency_condition] context가 비어있음 -> insufficient")
        return "insufficient"
    
    if context and hasattr(context[0], "page_content"):
        context_text = "\n\n".join([doc.page_content for doc in context])
    else:
        context_text = "\n\n".join([str(c) for c in context])
    
    print(f"[context_sufficiency_condition] context 길이: {len(context_text)}")
    print(f"[context_sufficiency_condition] context 미리보기: {context_text[:200]} ...")
    
    # context가 너무 짧으면 insufficient로 판단
    if len(context_text.strip()) < 50:
        print("[context_sufficiency_condition] context가 너무 짧음 -> insufficient")
        return "insufficient"
    
    if context and context_sufficient_llm(user_question, context_text):
        print("[context_sufficiency_condition] LLM 판정: sufficient")
        return "sufficient"
    else:
        print("[context_sufficiency_condition] LLM 판정: insufficient")
        return "insufficient"
    
def build_prompt(context: List, user_question: str, chat_history: List[Tuple[str, str]]) -> str:
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

def generate_with_gemini_node(state):
    print("state keys in [generate_with_gemini_node]:", list(state.keys()))
    user_question = get_user_question_from_state(state)
    context = state.get("context", [])
    # context_text만 사용
    if context and hasattr(context[0], "page_content"):
        context_text = "\n\n".join([doc.page_content for doc in context])
    else:
        context_text = "\n\n".join([str(c) for c in context])
    print(f"[generate_with_gemini_node] context_text 길이: {len(context_text)}")
    print(f"[generate_with_gemini_node] context_text 미리보기: {context_text[:80]} ...")
    print(f"[generate_with_gemini_node] user_question: {user_question}")
    prompt = f"""
[참고 기사/문서 내용]
{context_text}

[질문]
{user_question}

위 참고 내용만 바탕으로, 사용자의 질문에 대해 정확하고 친절하게 답변해 주세요.
절대 마크다운 문법(*, -, #)을 사용하지 말고, 필요하다면 이모지를 사용해서 항목을 구분해주세요.
"""
    print(f"[generate_with_gemini_node] LLM 프롬프트 미리보기: {prompt[:80]} ...")
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    response = model.generate_content(prompt)
    state = dict(state)
    state["answer"] = response.text
    # messages에 답변 누적(선택)
    state.setdefault("messages", []).append({"role": "assistant", "content": response.text})
    return state

# 그래프 빌더 함수
def build_adaptive_rag_graph(ensemble_retriever, article_id=None):
    workflow = StateGraph(dict)
    # 노드 등록
    workflow.add_node("article_search", lambda state: article_search_node(state, ensemble_retriever, article_id))
    workflow.add_node("all_search", lambda state: all_search_node(state, ensemble_retriever))
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate_answer", generate_with_gemini_node)
    
    # 엣지 연결
    workflow.add_edge(START, "article_search")
    
    # article_search 후 분기
    workflow.add_conditional_edges(
        "article_search",
        context_sufficiency_condition,
        {
            "sufficient": "generate_answer", 
            "insufficient": "all_search"
        }
    )
    
    # all_search 후 분기
    workflow.add_conditional_edges(
        "all_search",
        context_sufficiency_condition,
        {
            "sufficient": "generate_answer", 
            "insufficient": "web_search"
        }
    )
    
    # web_search는 항상 generate_answer로
    workflow.add_edge("web_search", "generate_answer")
    workflow.add_edge("generate_answer", END)
    
    # 컴파일
    return workflow.compile()

# 1. 임베딩 모델
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    task_type="retrieval_document"
)

# AdaptiveRAGAgent 클래스 수정: 문서 리스트를 받아서 RAG를 수행
class AdaptiveRAGAgent:
    """
    Adaptive RAG Agent for news QA.
    - user_question: str (질문)
    - documents: List[Document] (기사 전체)
    - article_id: Optional[str] (특정 기사 우선 답변용)
    """
    def __init__(self, embedding_model=None):
        if embedding_model is None:
            embedding_model = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                task_type="retrieval_document"
            )
        self.embedding_model = embedding_model

    def answer(self, user_question: str, documents: List[Document], article_id: Optional[str] = None) -> str:
        # 1. 문서 chunking 및 vectorstore/retriever 생성 (메모리 내)
        faiss_vectorstore, bm25_retriever = create_vectorstores(
            documents=documents,
            embedding_model=self.embedding_model
        )
        ensemble_retriever = build_ensemble_retriever(
            faiss_vectorstore=faiss_vectorstore,
            bm25_retriever=bm25_retriever
        )
        # 2. 그래프 생성
        graph = build_adaptive_rag_graph(
            ensemble_retriever=ensemble_retriever,
            article_id=article_id
        )
        # 3. 답변 생성
        from langchain_core.messages import convert_to_messages
        state = {
            "messages": convert_to_messages([
                {"role": "user", "content": user_question}
            ]),
            "context": [],
            "trace": []
        }
        answer = None
        for chunk in graph.stream(state):
            for node, update in chunk.items():
                if "answer" in update:
                    answer = update["answer"]
        return answer or "답변을 생성하지 못했습니다."

if __name__ == "__main__":
    user_question = "빅테크들이 ai 기술로 해고한 사례가 있으면 설명해줘."
    debug_run_graph(graph, user_question)

def debug_run_graph(graph, user_question):
    """디버깅용 함수"""
    from langchain_core.messages import convert_to_messages
    state = {
        "messages": convert_to_messages([
            {"role": "user", "content": user_question}
        ]),
        "context": [],
        "trace": []
    }
    
    print(f"=== Graph 실행 시작 ===")
    print(f"질문: {user_question}")
    
    for chunk in graph.stream(state):
        for node, update in chunk.items():
            print(f"\n=== {node} 노드 실행 ===")
            print(f"업데이트된 키들: {list(update.keys())}")
            if "context" in update:
                print(f"컨텍스트 개수: {len(update['context'])}")
            if "answer" in update:
                print(f"답변: {update['answer']}")
            if "trace" in update:
                print(f"트레이스: {update['trace']}")
    
    print(f"\n=== Graph 실행 완료 ===")