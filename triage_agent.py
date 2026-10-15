
import os
from typing import TypedDict, Literal

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

SYSTEM_PROMPT = (
    "You are a customer support agent. Be concise, professional, and empathetic. "
    "Never promise a specific refund amount or timeline you are not given. "
    "If information is missing, ask a clarifying question instead of guessing. "
    "When asked to write or improve a reply, output only the reply text itself — "
    "no preamble, no meta-commentary about the reply, no phrases like 'here is the reply'."
)

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = DirectoryLoader("docs", glob="**/*.txt", loader_cls=TextLoader)
raw_docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
FAQ_DOCS = splitter.split_documents(raw_docs)

vectordb = Chroma.from_documents(FAQ_DOCS, FastEmbedEmbeddings())
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Conversation so far:\n{history}\n\nAnswer the customer using only this context:\n{context}\n\nLatest message: {query}"),
])
rag_chain = rag_prompt | llm

def get_order_status(order_id: str) -> str:

    return f"Order {order_id} is currently in transit, expected in 2 days."

def generate_title(query: str) -> str:
    """Short topic summary for the sidebar, not a full echo of the message."""
    prompt = (
        "Summarize this customer message as a short topic label, 4 to 5 words, "
        "no punctuation, no quotes, title case.\n"
        f"Message: {query}"
    )
    title = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]).content.strip()
    return title.strip('"\'')

MAX_RETRIES = 2


class State(TypedDict):
    query: str
    history: str  # recent prior turns, formatted as plain text
    category: Literal["faq", "technical", "billing"]
    draft: str
    approved: bool
    refund_amount: float
    retries: int

def classify_node(state: State) -> State:
    history = state.get("history", "")
    prompt = (
        f"Conversation so far:\n{history}\n\n"
        f"Classify the latest customer message as exactly one word: faq, technical, or billing.\n"
        f"Latest message: {state['query']}"
    )
    category = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]).content.strip().lower()
    if category not in ("faq", "technical", "billing"):
        category = "faq"
    return {**state, "category": category, "retries": 0}

def route_query(state: State) -> str:
    return state["category"]

def faq_node(state: State) -> State:
    docs = retriever.invoke(state["query"])
    context = "\n".join(d.page_content for d in docs)
    draft = rag_chain.invoke({"context": context, "query": state["query"], "history": state.get("history", "")}).content
    return {**state, "draft": draft}

def technical_node(state: State) -> State:
    status = get_order_status("12345")  # TODO: extract real order id from query/session
    prompt = (
        f"Conversation so far:\n{state.get('history', '')}\n\n"
        f"Order status: {status}\nWrite a reply to the latest message: {state['query']}"
    )
    draft = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]).content
    return {**state, "draft": draft}

def billing_node(state: State) -> State:
    prompt = (
        f"Conversation so far:\n{state.get('history', '')}\n\n"
        f"Draft a short refund reply to the latest message: {state['query']}"
    )
    draft = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]).content
    # TODO: replace with a real refund amount lookup tied to the actual order. Hardcoded for demo purposes only.
    return {**state, "draft": draft, "refund_amount": 75.0}

def qa_node(state: State) -> State:
    verdict = llm.invoke(
        f"Does this draft properly answer the query? Reply only 'pass' or 'fail'.\n"
        f"Query: {state['query']}\nDraft: {state['draft']}"
    ).content.strip().lower()
    return {**state, "approved": verdict.startswith("pass")}

def qa_router(state: State) -> str:
    if not state["approved"]:
        if state.get("retries", 0) >= MAX_RETRIES:
            return "send"  # avoid infinite retry loop; ship best-effort draft
        return "retry"
    if state["category"] == "billing" and state.get("refund_amount", 0) > 50:
        return "human_review"
    return "send"

def rewrite_node(state: State) -> State:
    draft = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Improve this reply to better answer '{state['query']}':\n{state['draft']}\n\n"
            "Return only the improved reply text itself. Do not include any preamble, "
            "commentary about the rewrite, or phrases like 'here is the improved version'."
        )),
    ]).content.strip()
    return {**state, "draft": draft, "retries": state.get("retries", 0) + 1}

def human_review_node(state: State) -> State:
    # interrupt() pauses execution here until resumed with human input
    from langgraph.types import interrupt
    decision = interrupt({"draft": state["draft"], "refund_amount": state["refund_amount"]})
    return {**state, "approved": decision == "approve"}

def review_router(state: State) -> str:
    return "send" if state.get("approved") else "hold"

def hold_node(state: State) -> State:
    # Reviewer rejected the draft: never send it. Replace it with a neutral
    # holding message that makes no refund promise.
    print(f"[log] draft rejected by reviewer, not sent (category={state.get('category')}):\n{state['draft']}\n")
    holding = (
        "Thank you for your patience. Your request needs further review by our support team, "
        "and a team member will follow up with you directly."
    )
    return {**state, "draft": holding}

def send_node(state: State) -> State:
    print(f"[log] draft sent to customer (category={state.get('category')}):\n{state['draft']}\n")
    return state

# ---------- Build graph ----------
graph = StateGraph(State)
graph.add_node("classify", classify_node)
graph.add_node("faq", faq_node)
graph.add_node("technical", technical_node)
graph.add_node("billing", billing_node)
graph.add_node("qa", qa_node)
graph.add_node("rewrite", rewrite_node)
graph.add_node("human_review", human_review_node)
graph.add_node("hold", hold_node)
graph.add_node("send", send_node)

graph.set_entry_point("classify")
graph.add_conditional_edges("classify", route_query, {
    "faq": "faq", "technical": "technical", "billing": "billing"
})
graph.add_edge("faq", "qa")
graph.add_edge("technical", "qa")
graph.add_edge("billing", "qa")
graph.add_conditional_edges("qa", qa_router, {
    "retry": "rewrite", "human_review": "human_review", "send": "send"
})
graph.add_edge("rewrite", "qa")
graph.add_conditional_edges("human_review", review_router, {"send": "send", "hold": "hold"})
graph.add_edge("hold", END)
graph.add_edge("send", END)

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
app = graph.compile(checkpointer=SqliteSaver(conn))

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    result = app.invoke({"query": "I want a refund for my order, it never arrived"}, config)
    if "__interrupt__" in result:
        print("Paused for human approval:", result["__interrupt__"])
        # resume with: app.invoke(Command(resume="approve"), config)
