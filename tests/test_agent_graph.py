"""Bugs 1, 6, 7: graph routing, escalation on failed QA, and customer-scoped order data."""

import pytest
from langgraph.types import Command

from conftest import ROOT  # noqa: F401  (ensures sys.path is set up)


def cfg(tid="t1"):
    return {"configurable": {"thread_id": tid}}


def ask(agent, query, user="alice@example.com", tid="t1"):
    return agent.app.invoke({"query": query, "history": "", "user_id": user}, cfg(tid))


def interrupt_payload(result):
    return result["__interrupt__"][0].value


# ---------- bug 6: failed QA must not be sent ----------

def test_qa_failure_after_max_retries_escalates_and_is_not_sent(agent, capsys):
    agent.brain.qa_script = ["fail"]
    result = ask(agent, "How long does shipping take?")
    assert "__interrupt__" in result, "draft that never passed QA must go to a human"
    payload = interrupt_payload(result)
    assert payload["reason"] == "qa_failed"
    assert payload["refund_amount"] == 0.0  # non-billing turn: no KeyError, no stale amount
    assert agent.brain.qa_calls == agent.mod.MAX_RETRIES + 1
    assert "draft sent to customer" not in capsys.readouterr().out


def test_escalated_qa_failure_can_be_approved_or_rejected(agent, capsys):
    agent.brain.qa_script = ["fail"]
    ask(agent, "How long does shipping take?", tid="a")
    ask(agent, "How long does shipping take?", tid="r")
    capsys.readouterr()

    approved = agent.app.invoke(Command(resume="approve"), cfg("a"))
    assert approved["draft"] == "Draft reply"
    assert "draft sent to customer" in capsys.readouterr().out

    rejected = agent.app.invoke(Command(resume="reject"), cfg("r"))
    assert "further review" in rejected["draft"]
    out = capsys.readouterr().out
    assert "draft sent to customer" not in out and "rejected by reviewer" in out


def test_passing_draft_is_sent_without_review(agent, capsys):
    result = ask(agent, "How long does shipping take?")
    assert "__interrupt__" not in result
    assert "draft sent to customer" in capsys.readouterr().out


def test_failed_then_passing_draft_is_retried_then_sent(agent, capsys):
    agent.brain.qa_script = ["fail", "pass"]
    result = ask(agent, "How long does shipping take?")
    assert "__interrupt__" not in result
    assert agent.brain.qa_calls == 2
    assert "draft sent to customer" in capsys.readouterr().out


# ---------- bug 1 (regression): rejected refund is never sent ----------

def test_rejected_refund_is_held_not_sent(agent, capsys):
    agent.brain.category = "billing"
    agent.add_order("alice@example.com", "12345", 75.0)
    result = ask(agent, "Refund order 12345 please")
    assert interrupt_payload(result)["reason"] == "refund_over_limit"
    capsys.readouterr()
    final = agent.app.invoke(Command(resume="reject"), cfg())
    assert "draft sent to customer" not in capsys.readouterr().out
    assert "75" not in final["draft"]


# ---------- bug 7: real, customer-scoped order data ----------

def test_refund_amount_comes_from_the_customers_own_order(agent):
    agent.brain.category = "billing"
    agent.add_order("alice@example.com", "12345", 75.0)
    payload = interrupt_payload(ask(agent, "I want a refund for order 12345, it never arrived"))
    assert payload["refund_amount"] == 75.0
    assert any("Order 12345: status in transit" in p and "$75.00" in p for p in agent.brain.prompts)


def test_small_refund_is_not_escalated(agent):
    agent.brain.category = "billing"
    agent.add_order("alice@example.com", "20002", 30.0, status="delivered", eta=None)
    result = ask(agent, "Refund order 20002")
    assert "__interrupt__" not in result


def test_another_customers_order_is_not_disclosed(agent):
    agent.add_order("bob@example.com", "12345", 75.0, status="SECRET-STATUS")
    for category in ("technical", "billing"):
        agent.brain.category = category
        result = ask(agent, "Where is order 12345?", user="alice@example.com", tid=category)
        assert "__interrupt__" not in result, "no refund should be escalated from someone else's order"
    joined = "\n".join(agent.brain.prompts)
    assert "SECRET-STATUS" not in joined and "$75.00" not in joined
    assert "No order '12345' was found on this customer's account" in joined


def test_no_order_number_asks_for_it_instead_of_guessing(agent):
    agent.brain.category = "billing"
    result = ask(agent, "I want a refund, my order never arrived.")
    assert "__interrupt__" not in result
    assert any("No order number was given" in p for p in agent.brain.prompts)


def test_refund_amount_does_not_leak_into_the_next_turn(agent):
    agent.brain.category = "billing"
    agent.add_order("alice@example.com", "12345", 75.0)
    first = ask(agent, "Refund order 12345")
    assert interrupt_payload(first)["refund_amount"] == 75.0
    agent.app.invoke(Command(resume="approve"), cfg())

    agent.brain.category = "faq"
    agent.brain.qa_script = ["fail"]
    agent.brain._qa_calls = 0
    second = ask(agent, "How long does shipping take?")
    assert interrupt_payload(second)["refund_amount"] == 0.0


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Where is order 12345?", "12345"),
        ("ORDER ID: 99887766", "99887766"),
        ("my order #ab-1234 is late", "AB-1234"),
        ("order number A1B2C3 hasn't moved", "A1B2C3"),
        ("see #XY99001", "XY99001"),
        ("I want a refund, my order never arrived.", None),
        ("My tracking isn't updating", None),
        ("", None),
    ],
)
def test_extract_order_id(workdir, text, expected):
    import orders

    assert orders.extract_order_id(text) == expected
