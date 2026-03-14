"""Convert raw GitHub issue JSON into a stable internal structure."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def summarize_issue(issue: Dict[str, Any]) -> Dict[str, Any]:
    """Return a structured representation of a GitHub issue."""

    labels = [lbl.get("name") for lbl in issue.get("labels", []) if isinstance(lbl, dict)]

    return {
        "number": issue.get("number"),
        "title": issue.get("title") or "",
        "body": issue.get("body") or "",
        "url": issue.get("html_url") or issue.get("url"),
        "labels": labels,
        "created_at": issue.get("created_at"),
        "user": (issue.get("user") or {}).get("login"),
    }


def format_issue_text(issue: Dict[str, Any]) -> str:
    """Format issue content into a single prompt text for the LM."""

    parts: List[str] = []
    title = issue.get("title")
    body = issue.get("body")

    if title:
        parts.append(f"TITLE: {title.strip()}")
    if body:
        parts.append("BODY:\n" + body.strip())

    labels = issue.get("labels")
    if labels:
        parts.append("LABELS: " + ", ".join(labels))

    return "\n\n".join(parts).strip()
