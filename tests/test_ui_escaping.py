"""Bug 4: user/LLM text and URL-supplied identity must be HTML-escaped before unsafe_allow_html."""

import pytest
from streamlit.testing.v1 import AppTest

from conftest import ROOT

PAYLOAD = '<img src=x onerror="alert(1)">'
SCRIPT = "<script>alert(1)</script>"


def run_ui(uid="alice@example.com", name="Alice", tid=None):
    at = AppTest.from_file(str(ROOT / "ui.py"), default_timeout=30)
    at.query_params["uid"] = uid
    at.query_params["name"] = name
    if tid:
        at.query_params["tid"] = tid
    at.run()
    assert not at.exception, at.exception
    return at


def all_markdown(at):
    return [m.value for m in at.markdown]


def test_name_from_url_is_escaped(stub_agent):
    at = run_ui(name=PAYLOAD)
    rendered = [v for v in all_markdown(at) if "onerror" in v]
    assert rendered, "name should still be displayed"
    for v in rendered:
        assert "<img" not in v
        assert "&lt;img" in v


def test_email_from_url_is_escaped(stub_agent):
    at = run_ui(uid=SCRIPT)
    rendered = [v for v in all_markdown(at) if "alert(1)" in v]
    assert rendered
    for v in rendered:
        assert "<script>" not in v


@pytest.mark.parametrize("payload", [SCRIPT, PAYLOAD])
def test_stored_message_is_escaped(stub_agent, payload):
    import chat_store

    chat_store.init_db()
    chat_store.save_message("alice@example.com", "t1", "user", payload)
    chat_store.save_message("alice@example.com", "t1", "assistant", payload, "<b>faq</b>")

    at = run_ui(tid="t1")
    rendered = [v for v in all_markdown(at) if "alert(1)" in v]
    assert len(rendered) == 2, "both user and assistant bubbles should be shown"
    for v in rendered:
        assert "<script>" not in v and "<img" not in v
    # category label is escaped too
    assert not any("<b>faq</b>" in v for v in all_markdown(at))


def test_newlines_still_render_as_line_breaks(stub_agent):
    import chat_store

    chat_store.init_db()
    chat_store.save_message("alice@example.com", "t1", "user", "line one\nline two")
    at = run_ui(tid="t1")
    assert any("line one<br>line two" in v for v in all_markdown(at))
