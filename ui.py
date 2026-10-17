"""
Chatbot UI for the Customer Support Triage Agent.
Run with: streamlit run ui.py
"""

from datetime import datetime, date
import html
import textwrap

import streamlit as st
from triage_agent import app as graph_app, generate_title
from chat_store import init_db, new_thread_id, save_message, get_threads, get_messages, delete_thread, set_thread_title, user_can_access_thread, create_pending_review, get_open_review

st.set_page_config(page_title="Customer Support Bot", layout="wide")
init_db()

st.markdown(
    textwrap.dedent(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif !important;
    }

    .stApp, section[data-testid="stSidebar"], div[data-testid="stHeader"], div[data-testid="stChatInput"] {
        background-color: #ffffff;
    }

    section[data-testid="stSidebar"] { border-right: 1px solid #e6f4ea; }

    h1, h2, h3 { font-weight: 600; color: #14532d; letter-spacing: -0.3px; }

    .center-title {
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        color: #14532d;
        margin-bottom: 0.2rem;
    }

    .center-welcome {
        text-align: center;
        font-size: 0.95rem;
        color: #4b7a5a;
        margin-bottom: 1.2rem;
    }

    .subtitle { color: #6b6b6b; font-size: 0.9rem; margin-bottom: 1.2rem; }

    .disclaimer {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 0.6rem 0.9rem;
        font-size: 0.78rem;
        color: #166534;
        margin-bottom: 1rem;
        line-height: 1.4;
    }

    .sidebar-section-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #6b9b7a;
        margin: 1rem 0 0.4rem 0;
    }

    .thread-time { font-size: 0.7rem; color: #9ec4ab; padding-left: 0.3rem; }

    .stButton button {
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
        border-radius: 8px;
        border: 1px solid #16a34a;
        background-color: #ffffff;
        color: #16a34a;
    }
    .stButton button:hover { background-color: #f0fdf4; border-color: #15803d; }

    button[kind="primary"] {
        background-color: #16a34a !important;
        color: #ffffff !important;
        border: 1px solid #16a34a !important;
    }
    button[kind="primary"]:hover { background-color: #15803d !important; }

    /* Chat bubbles */
    .msg-row { display: flex; margin-bottom: 0.9rem; }
    .msg-row.user { justify-content: flex-end; }
    .msg-row.assistant { justify-content: flex-start; }

    .bubble {
        max-width: 68%;
        padding: 0.7rem 1rem;
        border-radius: 16px;
        font-size: 0.93rem;
        line-height: 1.5;
    }
    .bubble.user {
        background-color: #16a34a;
        color: #ffffff;
        border-bottom-right-radius: 4px;
    }
    .bubble.assistant {
        background-color: #f0fdf4;
        color: #14320f;
        border: 1px solid #bbf7d0;
        border-bottom-left-radius: 4px;
    }

    .category-tag {
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 600;
        padding: 0.12rem 0.55rem;
        border-radius: 20px;
        margin-bottom: 0.35rem;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        background-color: #dcfce7;
        color: #166534;
    }

    .msg-time { font-size: 0.68rem; opacity: 0.7; margin-top: 0.3rem; }
    .bubble.user .msg-time { color: #e6f4ea; }
    .bubble.assistant .msg-time { color: #4b7a5a; }

    .status-pending { color: #a16207; font-weight: 500; font-size: 0.85rem; margin-top: 0.5rem; }

    .empty-state { text-align: center; padding: 2.5rem 1rem; color: #6b9b7a; }

    div[data-testid="stForm"] {
        padding: 0 0.5rem;
    }
    div[data-testid="stForm"] input {
        border-radius: 24px !important;
        border: 1px solid #d1e7d9 !important;
        padding: 0.6rem 1.1rem !important;
        background-color: #f8fdfa !important;
    }
    div[data-testid="stForm"] button {
        border-radius: 24px !important;
    }

    .example-chip button { border-radius: 20px !important; font-size: 0.85rem !important; padding: 0.4rem 1rem !important; }

    .landing-wrap { max-width: 760px; margin: 3.5rem auto 0 auto; }
    .landing-wrap div[data-testid="stHorizontalBlock"] {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(22, 163, 74, 0.12);
        border: 1px solid #e6f4ea;
        gap: 0 !important;
    }
    .landing-left {
        background-color: #16a34a;
        color: #ffffff;
        padding: 3rem 2rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        min-height: 420px;
        height: 100%;
    }
    .landing-left-title { font-size: 1.7rem; font-weight: 700; margin-bottom: 0.5rem; }
    .landing-left-sub { font-size: 0.9rem; color: #dcfce7; line-height: 1.5; max-width: 240px; }
    .landing-right {
        padding: 3rem 1.7rem 0 1.7rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
        background-color: #ffffff;
    }
    .landing-title {
        text-align: left;
        font-size: 1.5rem;
        font-weight: 700;
        color: #14532d;
        margin-bottom: 1.5rem;
    }
    .landing-note {
        font-size: 0.72rem;
        color: #9ec4ab;
        margin-top: 0.8rem;
        text-align: left;
    }
    </style>
    """
    ),
    unsafe_allow_html=True,
)

CATEGORY_LABELS = {"faq": "FAQ", "technical": "Technical", "billing": "Billing"}


def format_time(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str)
    except (ValueError, TypeError):
        return ""
    if dt.date() == date.today():
        return dt.strftime("%I:%M %p").lstrip("0")
    return dt.strftime("%b %d, %I:%M %p").lstrip("0")


def build_history_text(thread_id: str, user_id: str, limit: int = 6) -> str:
    msgs = get_messages(thread_id, user_id)[-limit:]
    lines = []
    for m in msgs:
        speaker = "Customer" if m["role"] == "user" else "Agent"
        lines.append(f"{speaker}: {m['content']}")
    return "\n".join(lines)


def render_bubble(role: str, content: str, category: str = None, timestamp: str = None):
    cls = "user" if role == "user" else "assistant"
    # Message text comes from users and the LLM: escape before it is injected into HTML.
    safe_content = html.escape(content).replace("\n", "<br>")
    tag_html = (
        f"<span class='category-tag'>{html.escape(str(CATEGORY_LABELS.get(category, category)))}</span><br>"
        if category else ""
    )
    time_html = f"<div class='msg-time'>{format_time(timestamp)}</div>" if timestamp else ""
    bubble_html = f'<div class="msg-row {cls}"><div class="bubble {cls}">{tag_html}{safe_content}{time_html}</div></div>'
    st.markdown(bubble_html, unsafe_allow_html=True)


# ---------- Session state ----------
params = st.query_params
if "user_id" not in st.session_state:
    st.session_state.user_id = params.get("uid", "")
if "customer_name" not in st.session_state:
    st.session_state.customer_name = params.get("name", "")
if "thread_id" not in st.session_state:
    st.session_state.thread_id = params.get("tid", None)
if "search" not in st.session_state:
    st.session_state.search = ""

# ---------- Landing screen: collect name + email ----------
if not st.session_state.user_id or not st.session_state.customer_name:
    st.markdown('<div class="landing-wrap">', unsafe_allow_html=True)
    left_col, right_col = st.columns([1, 1.2], gap="small")

    with left_col:
        st.markdown(
            '<div class="landing-left">'
            '<div class="landing-left-title">Hello, Welcome!</div>'
            '<div class="landing-left-sub">Tell us who you are so we can pick up right where you left off.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown(
            '<div class="landing-right"><div class="landing-title">Get started</div></div>',
            unsafe_allow_html=True,
        )

        with st.form("landing_form"):
            name_input = st.text_input("Name", placeholder="Your full name", label_visibility="collapsed", max_chars=80)
            email_input = st.text_input("Email", placeholder="you@example.com", label_visibility="collapsed", max_chars=254)
            submitted = st.form_submit_button("Enter", use_container_width=True, type="primary")

        st.markdown(
            '<div class="landing-note">This identity is remembered for this session only and is not a verified login.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        if not name_input.strip():
            st.error("Please enter your name.")
        elif "@" not in email_input or "." not in email_input:
            st.error("Please enter a valid email address.")
        else:
            st.session_state.customer_name = name_input.strip()
            st.session_state.user_id = email_input.strip().lower()
            st.query_params["uid"] = st.session_state.user_id
            st.query_params["name"] = st.session_state.customer_name
            st.rerun()

    st.stop()

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown(f"<div class='sidebar-section-label'>Signed in as</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-weight:600;color:#14532d;'>{html.escape(st.session_state.customer_name)}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.78rem;color:#6b9b7a;margin-bottom:0.8rem;'>{html.escape(st.session_state.user_id)}</div>", unsafe_allow_html=True)

    if st.button("Switch user", use_container_width=True):
        st.session_state.user_id = ""
        st.session_state.customer_name = ""
        st.session_state.thread_id = None
        st.query_params.clear()
        st.rerun()

    if st.button("New chat", use_container_width=True, type="primary"):
        new_id = new_thread_id()
        st.session_state.thread_id = new_id
        st.query_params["tid"] = new_id
        st.rerun()

    st.markdown("<div class='sidebar-section-label'>History</div>", unsafe_allow_html=True)
    threads = get_threads(st.session_state.user_id)

    if len(threads) > 5:
        st.session_state.search = st.text_input(
            "Search", value=st.session_state.search, placeholder="Search conversations",
            label_visibility="collapsed",
        )

    query = st.session_state.search.lower().strip()
    for thread_id, title, last_active in threads:
        if query and query not in title.lower():
            continue
        is_active = thread_id == st.session_state.thread_id
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(
                title or "New conversation",
                key=f"thread_{thread_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.thread_id = thread_id
                st.query_params["tid"] = thread_id
                st.rerun()
        with col2:
            if st.button("Del", key=f"del_{thread_id}"):
                delete_thread(thread_id, st.session_state.user_id)
                if st.session_state.thread_id == thread_id:
                    st.session_state.thread_id = None
                st.rerun()
        st.markdown(f"<div class='thread-time'>{format_time(last_active)}</div>", unsafe_allow_html=True)

# ---------- Main panel ----------
if st.session_state.thread_id and not user_can_access_thread(st.session_state.thread_id, st.session_state.user_id):
    st.session_state.thread_id = None

if st.session_state.thread_id is None:
    st.session_state.thread_id = new_thread_id()
    st.query_params["tid"] = st.session_state.thread_id

st.markdown("<div class='center-title'>Customer Support Bot</div>", unsafe_allow_html=True)
st.markdown(f"<div class='center-welcome'>Welcome, {html.escape(st.session_state.customer_name)}</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='disclaimer'>Responses are AI-drafted and unverified.</div>",
    unsafe_allow_html=True,
)

thread_id = st.session_state.thread_id
config = {"configurable": {"thread_id": thread_id}}
messages = get_messages(thread_id, st.session_state.user_id)
pending_review = get_open_review(thread_id, st.session_state.user_id)

example_clicked = None
if not messages and not pending_review:
    render_bubble(
        "assistant",
        f"Hello {st.session_state.customer_name}, I am your customer support bot. How can I assist you?",
    )
    examples = [
        "How long does shipping take?",
        "My order tracking isn't updating.",
        "I want a refund, my order never arrived.",
    ]
    cols = st.columns(len(examples))
    for c, ex in zip(cols, examples):
        with c:
            st.markdown("<div class='example-chip'>", unsafe_allow_html=True)
            if st.button(ex, key=f"ex_{ex}"):
                example_clicked = ex
            st.markdown("</div>", unsafe_allow_html=True)

for msg in messages:
    render_bubble(msg["role"], msg["content"], msg["category"], msg["created_at"])

if pending_review:
    # The customer only sees that the reply is under review: not the unapproved draft, and no approve/reject controls.
    render_bubble(
        "assistant",
        "Your request has been passed to our support team for review. Their reply will appear here once it is ready.",
    )
    if st.button("Check for update", key="refresh_review"):
        st.rerun()

user_query = st.chat_input("Type your message...", disabled=bool(pending_review))
if example_clicked:
    user_query = example_clicked

if user_query:
    is_first_message = len(messages) == 0
    save_message(st.session_state.user_id, thread_id, "user", user_query)
    render_bubble("user", user_query)

    if is_first_message:
        title = generate_title(user_query)
        set_thread_title(thread_id, st.session_state.user_id, title)

    history_text = build_history_text(thread_id, st.session_state.user_id)

    with st.spinner("Thinking..."):
        result = graph_app.invoke(
            {"query": user_query, "history": history_text, "user_id": st.session_state.user_id}, config
        )

    if "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0].value
        create_pending_review(
            st.session_state.user_id,
            thread_id,
            interrupt_data["draft"],
            interrupt_data.get("refund_amount", 0.0),
            interrupt_data.get("reason", ""),
        )
    else:
        save_message(st.session_state.user_id, thread_id, "assistant", result["draft"], result.get("category"))

    st.rerun()