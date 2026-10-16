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
