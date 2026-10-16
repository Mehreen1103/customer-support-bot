"""Bugs 2 and 3: reviews are persisted, customers cannot approve, only the reviewer console can."""

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from conftest import ROOT

ALICE = "alice@example.com"
PASSWORD = "s3cret-pass"


@pytest.fixture(autouse=True)
def _fresh_throttle():
    st.cache_resource.clear()  # the login throttle is process-wide
    yield
    st.cache_resource.clear()


def customer(uid=ALICE, tid="t1"):
    at = AppTest.from_file(str(ROOT / "ui.py"), default_timeout=30)
    at.query_params["uid"] = uid
    at.query_params["name"] = "Alice"
    at.query_params["tid"] = tid
    at.run()
    assert not at.exception, at.exception
    return at


def reviewer_app(monkeypatch, password=PASSWORD):
    if password is None:
        monkeypatch.delenv("REVIEWER_PASSWORD", raising=False)
    else:
        monkeypatch.setenv("REVIEWER_PASSWORD", password)
    at = AppTest.from_file(str(ROOT / "reviewer.py"), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def sign_in(at, name="Rita", password=PASSWORD):
    at.text_input(key="reviewer_name").set_value(name)
    at.text_input(key="reviewer_password").set_value(password)
    [b for b in at.button if b.label == "Sign in"][0].click().run()
    assert not at.exception, at.exception
    return at


def labels(at):
    return [b.label.lower() for b in at.button]


def page_text(at):
    return " ".join([m.value for m in at.markdown] + [t.value for t in at.text])


@pytest.fixture
def pending(agent):
    """A customer asks for a $75 refund; the graph pauses for review (real graph, stubbed LLM)."""
    agent.brain.category = "billing"
    agent.add_order(ALICE, "12345", 75.0)
    at = customer()
    at.chat_input[0].set_value("I want a refund for order 12345, it never arrived").run()
    assert not at.exception, at.exception
    return agent, at


# ---------- storage layer ----------

def test_review_queue_lifecycle_and_scoping(workdir):
    import chat_store as cs

    cs.init_db()
    cs.save_message(ALICE, "t1", "user", "refund please")
    rid = cs.create_pending_review(ALICE, "t1", "draft", 75.0, "refund_over_limit")

    assert cs.create_pending_review(ALICE, "t1", "draft again", 75.0) == rid, "one open review per thread"
    assert cs.get_open_review("t1", ALICE)["id"] == rid
    assert cs.get_open_review("t1", "bob@example.com") is None, "customers only see their own"
    assert [r["id"] for r in cs.list_open_reviews()] == [rid]

    assert cs.claim_review(rid) is True
    assert cs.claim_review(rid) is False, "double click / second reviewer must not claim it again"
    cs.release_review(rid)
    assert cs.claim_review(rid) is True
    cs.complete_review(rid, "approve", "Rita")
    assert cs.get_open_review("t1", ALICE) is None
    assert cs.list_open_reviews() == []
    assert cs.claim_review(rid) is False, "a decided review cannot be claimed"

    conn = sqlite3.connect(cs.DB_PATH)
    status, by = conn.execute("SELECT status, decided_by FROM pending_reviews WHERE id=?", (rid,)).fetchone()
    assert (status, by) == ("approved", "Rita")


def test_stale_claim_can_be_retaken(workdir):
    import chat_store as cs

    cs.init_db()
    rid = cs.create_pending_review(ALICE, "t1", "draft")
    assert cs.claim_review(rid)
    old = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=cs.REVIEW_CLAIM_TIMEOUT_SECONDS + 5)).isoformat()
    conn = sqlite3.connect(cs.DB_PATH)
    conn.execute("UPDATE pending_reviews SET claimed_at=? WHERE id=?", (old, rid))
    conn.commit()
    assert cs.claim_review(rid), "a reviewer that crashed mid-decision must not block the thread forever"


def test_cannot_queue_a_review_on_someone_elses_thread(workdir):
    import chat_store as cs

    cs.init_db()
    cs.save_message(ALICE, "t1", "user", "hi")
    with pytest.raises(PermissionError):
        cs.create_pending_review("bob@example.com", "t1", "draft")


def test_deleting_a_thread_cancels_its_open_review(workdir):
    import chat_store as cs

    cs.init_db()
    cs.save_message(ALICE, "t1", "user", "hi")
    rid = cs.create_pending_review(ALICE, "t1", "draft")
    cs.delete_thread("t1", ALICE)
    assert cs.claim_review(rid) is False
    assert cs.list_open_reviews() == []


# ---------- bug 2: the customer cannot approve ----------

def test_customer_ui_has_no_approval_code_path():
    source = (ROOT / "ui.py").read_text()
    assert "Command" not in source and "resume" not in source


def test_customer_sees_only_under_review_and_no_controls(pending):
    agent, at = pending
    assert not any("approve" in l or "reject" in l for l in labels(at))
    assert at.chat_input[0].disabled, "no new messages while the thread is awaiting review"
    text = page_text(at)
    assert "passed to our support team for review" in text
    assert "Draft reply" not in text, "the unapproved draft must not be shown to the customer"


# ---------- bug 3: pending reviews survive a refresh ----------

def test_pending_review_survives_a_new_session(pending):
    import chat_store as cs

    agent, _ = pending
    assert cs.get_open_review("t1", ALICE) is not None
    refreshed = customer()  # brand-new Streamlit session: nothing in session_state
    assert "passed to our support team for review" in page_text(refreshed)
    assert refreshed.chat_input[0].disabled
    assert not any("approve" in l or "reject" in l for l in labels(refreshed))


# ---------- reviewer console: end to end ----------

def test_reviewer_approves_and_customer_receives_the_reply(pending, monkeypatch, capsys):
    import chat_store as cs

    agent, _ = pending
    at = sign_in(reviewer_app(monkeypatch))
    text = page_text(at)
    assert f"Customer: {ALICE}" in text and "$75.00" in text
    assert "Refund over the review threshold" in text

    capsys.readouterr()
    at.button(key="approve_1").click().run()
    assert not at.exception, at.exception

    assert "draft sent to customer" in capsys.readouterr().out
    last = cs.get_messages("t1", ALICE)[-1]
    assert last["role"] == "assistant" and last["content"] == "Draft reply"
    assert cs.get_open_review("t1", ALICE) is None
    assert cs.claim_review(1) is False, "an approved review cannot be approved again"

    after = customer()
    assert not after.chat_input[0].disabled, "customer can continue the conversation"
    assert any("Draft reply" in m.value for m in after.markdown)


def test_reviewer_rejects_and_customer_gets_a_holding_message(pending, monkeypatch, capsys):
    import chat_store as cs

    at = sign_in(reviewer_app(monkeypatch))
    capsys.readouterr()
    at.button(key="reject_1").click().run()
    assert not at.exception, at.exception
    out = capsys.readouterr().out
    assert "draft sent to customer" not in out and "rejected by reviewer" in out
    last = cs.get_messages("t1", ALICE)[-1]["content"]
    assert "further review" in last and "75" not in last


# ---------- reviewer console: access control ----------

def test_console_is_disabled_when_no_password_is_configured(pending, monkeypatch):
    at = reviewer_app(monkeypatch, password=None)
    assert any("REVIEWER_PASSWORD is not set" in e.value for e in at.error)
    assert not any("approve" in l for l in labels(at))
    assert ALICE not in page_text(at)


def test_queue_is_hidden_until_signed_in_and_wrong_password_is_refused(pending, monkeypatch):
    at = reviewer_app(monkeypatch)
    assert ALICE not in page_text(at) and not any("approve" in l for l in labels(at))
    sign_in(at, password="wrong")
    assert any("Incorrect password" in e.value for e in at.error)
    assert ALICE not in page_text(at) and not any("approve" in l for l in labels(at))


def test_repeated_failures_lock_out_even_the_right_password(pending, monkeypatch):
    at = reviewer_app(monkeypatch)
    for _ in range(5):
        sign_in(at, password="wrong")
    sign_in(at, password=PASSWORD)
    assert any("Too many failed attempts" in e.value for e in at.error)
    assert ALICE not in page_text(at)


def test_reviewer_must_give_a_name(pending, monkeypatch):
    at = sign_in(reviewer_app(monkeypatch), name="   ")
    assert ALICE not in page_text(at)
