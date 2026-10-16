
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

from orders import get_order, extract_order_id, init_orders_db

init_orders_db()

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
REFUND_REVIEW_THRESHOLD = 50  # refunds above this amount need a human reviewer


class State(TypedDict):
    query: str
    history: str  # recent prior turns, formatted as plain text
    user_id: str  # authenticated customer; scopes order lookups
    category: Literal["faq", "technical", "billing"]
    draft: str
    approved: bool
    refund_amount: float
    review_reason: str  # why a human must review: "refund_over_limit" | "qa_failed" | ""
    retries: int


def lookup_order(state: State):
    """Find the customer's order referenced in the conversation.

    Returns (order or None, text describing the result for the LLM). The text never
    contains order details unless the order belongs to this customer.
    """
    order_id = extract_order_id(state["query"], state.get("history", ""))
    if not order_id:
        return None, "No order number was given. Ask the customer for their order number. Do not guess any order details."
    order = get_order(order_id, state.get("user_id", ""))
    if order is None:
        return None, (
            f"No order '{order_id}' was found on this customer's account. Say you could not find it "
            "and ask them to check the number. Do not guess any order details."
        )
    return order, (
        f"Order {order['order_id']}: status {order['status']}, "
        f"expected delivery {order['eta'] or 'unknown'}, total ${order['total']:.2f}."
    )


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
    # State persists across turns in a thread, so reset per-turn fields.
    return {**state, "category": category, "retries": 0, "approved": False,
            "refund_amount": 0.0, "review_reason": ""}

def route_query(state: State) -> str:
    return state["category"]

def faq_node(state: State) -> State:
    docs = retriever.invoke(state["query"])
    context = "\n".join(d.page_content for d in docs)
    draft = rag_chain.invoke({"context": context, "query": state["query"], "history": state.get("history", "")}).content
    return {**state, "draft": draft}

def technical_node(state: State) -> State:
    _, order_info = lookup_order(state)
    prompt = (
        f"Conversation so far:\n{state.get('history', '')}\n\n"
        f"Order information: {order_info}\nWrite a reply to the latest message: {state['query']}"
    )
    draft = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]).content
    return {**state, "draft": draft}

def billing_node(state: State) -> State:
    order, order_info = lookup_order(state)
    prompt = (
        f"Conversation so far:\n{state.get('history', '')}\n\n"
        f"Order information: {order_info}\n"
        f"Draft a short refund reply to the latest message: {state['query']}"
    )
    draft = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]).content
    # The refund under review is the customer's own order total; 0 when no order was identified.
    return {**state, "draft": draft, "refund_amount": order["total"] if order else 0.0}

def qa_node(state: State) -> State:
    verdict = llm.invoke(
        f"Does this draft properly answer the query? Reply only 'pass' or 'fail'.\n"
        f"Query: {state['query']}\nDraft: {state['draft']}"
    ).content.strip().lower()
    approved = verdict.startswith("pass")
    reason = ""
    if not approved and state.get("retries", 0) >= MAX_RETRIES:
        reason = "qa_failed"  # out of retries: a human decides, the draft is not sent as-is
    elif approved and state["category"] == "billing" and state.get("refund_amount", 0) > REFUND_REVIEW_THRESHOLD:
        reason = "refund_over_limit"
    return {**state, "approved": approved, "review_reason": reason}

def qa_router(state: State) -> str:
    if state.get("review_reason"):
        return "human_review"
    if not state["approved"]:
        return "retry"
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
    decision = interrupt({
        "draft": state["draft"],
        "refund_amount": state.get("refund_amount", 0.0),
        "reason": state.get("review_reason", ""),
    })
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
    result = app.invoke(
        {"query": "I want a refund for order 12345, it never arrived", "user_id": "demo@example.com"}, config
    )
    if "__interrupt__" in result:
        print("Paused for human approval:", result["__interrupt__"])
        # resume with: app.invoke(Command(resume="approve"), config)
