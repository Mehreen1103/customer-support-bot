"""
Reviewer console: approve or reject drafts held for human review (large refunds, drafts that failed QA).

Run it separately from the customer app, on an internal port, from the same directory
so both share the same databases:

    REVIEWER_PASSWORD=... streamlit run reviewer.py --server.port 8502

The customer app has no approval code path at all. Access here is a single shared
password, which is a stopgap: put this behind your SSO / VPN for production.
"""

import hmac
import os
import time

import streamlit as st
from langgraph.types import Command

from triage_agent import app as graph_app
from chat_store import (
    init_db, list_open_reviews, claim_review, complete_review, release_review, get_messages, save_message,
)

st.set_page_config(page_title="Support Review Console", layout="wide")
init_db()

REASON_LABELS = {
    "refund_over_limit": "Refund over the review threshold",
    "qa_failed": "Draft failed automated QA",
}
MAX_FAILED_LOGINS = 5
LOCKOUT_SECONDS = 60


@st.cache_resource
def _login_throttle() -> dict:
    """Process-wide (not per-session) so opening a new browser session does not reset it."""
    return {"failures": 0, "locked_until": 0.0}


def check_password(candidate: str, expected: str) -> bool:
    return hmac.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))


def login_screen(expected_password: str):
    throttle = _login_throttle()
    st.title("Support review console")
    with st.form("reviewer_login"):
        name = st.text_input("Your name", key="reviewer_name", max_chars=80)
        password = st.text_input("Password", type="password", key="reviewer_password", max_chars=200)
        submitted = st.form_submit_button("Sign in", type="primary")
    if not submitted:
        return
    now = time.time()
    if now < throttle["locked_until"]:
        st.error("Too many failed attempts. Try again in a minute.")
    elif not name.strip():
        st.error("Enter your name so decisions can be attributed.")
    elif check_password(password, expected_password):
        throttle["failures"] = 0
        st.session_state.reviewer = name.strip()
        st.rerun()
    else:
        throttle["failures"] += 1
        if throttle["failures"] >= MAX_FAILED_LOGINS:
            throttle["locked_until"] = now + LOCKOUT_SECONDS
            throttle["failures"] = 0
        st.error("Incorrect password.")


def decide(review: dict, decision: str, reviewer: str):
    """Resume the paused graph with the reviewer's decision and put the outcome in the customer's thread."""
    if not claim_review(review["id"]):
        st.warning("This review was already handled, or is being handled by someone else.")
        return
    try:
        final = graph_app.invoke(
            Command(resume=decision), {"configurable": {"thread_id": review["thread_id"]}}
        )
        save_message(review["user_id"], review["thread_id"], "assistant", final["draft"], final.get("category"))
        complete_review(review["id"], decision, reviewer)
    except Exception as exc:  # return it to the queue instead of losing it
        release_review(review["id"])
        st.error(f"Could not process that decision; the review is back in the queue. ({type(exc).__name__})")
        return
    st.rerun()


def console(reviewer: str):
    st.title("Support review console")
    st.caption(f"Signed in as {reviewer}")  # plain text: never raw HTML
    top_left, top_right = st.columns([1, 1])
    with top_left:
        if st.button("Refresh"):
            st.rerun()
    with top_right:
        if st.button("Sign out"):
            st.session_state.pop("reviewer", None)
            st.rerun()

    reviews = list_open_reviews()
    if not reviews:
        st.info("No drafts are waiting for review.")
        return

    for review in reviews:
        with st.container(border=True):
            st.subheader(f"Review #{review['id']}")
            # Everything below is customer- or LLM-controlled: show it as plain text, never as HTML/markdown.
            st.text(f"Customer: {review['user_id']}")
            st.text(f"Reason: {REASON_LABELS.get(review['reason'], review['reason'] or 'Unspecified')}")
            st.text(f"Refund amount under review: ${review['refund_amount']:.2f}")
            st.text(f"Waiting since (UTC): {review['created_at']}")
            if review["status"] == "processing":
                st.text("Status: being handled by a reviewer")

            st.text("Recent conversation:")
            for msg in get_messages(review["thread_id"], review["user_id"])[-6:]:
                speaker = "Customer" if msg["role"] == "user" else "Agent"
                st.text(f"{speaker}: {msg['content']}")

            st.text_area("Draft reply", value=review["draft"], disabled=True, height=140, key=f"draft_{review['id']}")

            approve_col, reject_col = st.columns([1, 1])
            with approve_col:
                approve = st.button("Approve and send", key=f"approve_{review['id']}", type="primary")
            with reject_col:
                reject = st.button("Reject (send holding message)", key=f"reject_{review['id']}")
            if approve or reject:
                decide(review, "approve" if approve else "reject", reviewer)


expected_password = os.getenv("REVIEWER_PASSWORD", "")
if not expected_password:
    # Fail closed: never run an approval console without a password.
    st.error("REVIEWER_PASSWORD is not set, so the review console is disabled.")
    st.stop()

if not st.session_state.get("reviewer"):
    login_screen(expected_password)
    st.stop()

console(st.session_state.reviewer)
