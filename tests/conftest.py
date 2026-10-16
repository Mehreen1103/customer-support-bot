"""Shared test fixtures. Tests run in a temp working dir so no real DB files are touched."""

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

APP_MODULES = ("chat_store", "triage_agent", "orders", "ui", "reviewer")


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Isolated cwd with fresh imports of the app modules (their DB paths are relative)."""
    monkeypatch.chdir(tmp_path)
    for name in APP_MODULES:
        sys.modules.pop(name, None)
    yield tmp_path
    for name in APP_MODULES:
        sys.modules.pop(name, None)


class FakeGraphApp:
    """Stand-in for the compiled LangGraph app, for UI-level tests."""

    def __init__(self):
        self.calls = []

    def invoke(self, payload, config=None):
        self.calls.append((payload, config))
        return {"draft": "stub reply", "category": "faq"}


@pytest.fixture
def stub_agent(workdir, monkeypatch):
    """Replace triage_agent (heavy: LLM, vector store) with a light stub so ui.py can be imported."""
    mod = types.ModuleType("triage_agent")
    mod.app = FakeGraphApp()
    mod.generate_title = lambda q: "Test title"
    monkeypatch.setitem(sys.modules, "triage_agent", mod)
    return mod


# ---------------------------------------------------------------------------
# Real LangGraph graph with the LLM / embeddings / vector store stubbed out.
# ---------------------------------------------------------------------------

class Brain:
    """Controls what the fake LLM says and records every prompt it receives."""

    def __init__(self):
        self.category = "faq"
        self.qa_script = ["pass"]  # verdicts consumed in order; the last one repeats
        self.prompts = []
        self._qa_calls = 0

    def next_qa(self):
        verdict = self.qa_script[min(self._qa_calls, len(self.qa_script) - 1)]
        self._qa_calls += 1
        return verdict

    @property
    def qa_calls(self):
        return self._qa_calls


def _to_text(inp):
    if isinstance(inp, str):
        return inp
    if hasattr(inp, "to_string"):
        return inp.to_string()
    return "\n".join(getattr(m, "content", str(m)) for m in inp)


@pytest.fixture
def agent(workdir, monkeypatch):
    from langchain_core.documents import Document
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    brain = Brain()

    def fake_llm(inp):
        text = _to_text(inp)
        brain.prompts.append(text)
        if "Classify the latest customer message" in text:
            out = brain.category
        elif "Does this draft properly answer" in text:
            out = brain.next_qa()
        elif "Summarize this customer message" in text:
            out = "Test Title"
        else:
            out = "Draft reply"
        return AIMessage(content=out)

    monkeypatch.setenv("GROQ_API_KEY", "test")
    groq = types.ModuleType("langchain_groq")
    groq.ChatGroq = lambda **kw: RunnableLambda(fake_llm)
    monkeypatch.setitem(sys.modules, "langchain_groq", groq)

    class _Retriever:
        def invoke(self, query):
            return [Document(page_content="Shipping takes 3-5 business days.")]

    class _Chroma:
        @classmethod
        def from_documents(cls, docs, embeddings):
            return cls()

        def as_retriever(self, **kwargs):
            return _Retriever()

    vectorstores = types.ModuleType("langchain_community.vectorstores")
    vectorstores.Chroma = _Chroma
    embeddings = types.ModuleType("langchain_community.embeddings")
    embeddings.FastEmbedEmbeddings = lambda: None
    monkeypatch.setitem(sys.modules, "langchain_community.vectorstores", vectorstores)
    monkeypatch.setitem(sys.modules, "langchain_community.embeddings", embeddings)

    (workdir / "docs").mkdir()
    (workdir / "docs" / "faq.txt").write_text("Q: How long does shipping take?\nA: 3-5 business days.\n")

    import orders
    import triage_agent

    def add_order(user_id, order_id, total, status="in transit", eta="in 2 days"):
        import sqlite3

        conn = sqlite3.connect(orders.DB_PATH)
        conn.execute(
            "INSERT OR REPLACE INTO orders (order_id, user_id, status, eta, total) VALUES (?, ?, ?, ?, ?)",
            (order_id, user_id, status, eta, total),
        )
        conn.commit()
        conn.close()

    return types.SimpleNamespace(mod=triage_agent, app=triage_agent.app, brain=brain, add_order=add_order)
