"""Deterministic user-visible response templates for CAR-bench agents."""

from __future__ import annotations


def clean_user_content(content: str) -> str:
    """Normalize text before sending it to the benchmark-visible user."""

    content = content.replace("\u200b", "").replace("\xa0", " ")
    return "\n".join(line.strip() for line in content.splitlines() if line.strip())


def render_done(summary: str) -> str:
    summary = clean_user_content(summary).rstrip(".")
    if not summary:
        return "Done."
    return f"Done, {summary}."


def render_question(question: str) -> str:
    question = clean_user_content(question).rstrip("?.")
    return f"{question}?"


def render_confirmation(statement: str, question: str) -> str:
    statement = clean_user_content(statement).rstrip(".")
    question = clean_user_content(question).rstrip("?.")
    return f"{statement}. {question}?"


def render_missing_capability() -> str:
    return "I don't currently have that capability, so I can't complete this request."


def render_missing_required_information() -> str:
    return "I need more information before I can complete this request."


def render_unavailable_control() -> str:
    return "I can't complete this request with the available controls."


def render_malformed_action() -> str:
    return "I couldn't safely interpret the requested action, so I can't complete it."


def render_malformed_tool_call() -> str:
    return "I couldn't safely interpret the tool action, so I can't complete it."


def render_malformed_tool_arguments() -> str:
    return "I couldn't safely interpret the tool details, so I can't complete it."
