"""Simple LM-based issue advisor using a next-token predictor."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from transformers import pipeline


LOG = logging.getLogger(__name__)


class IssueAdvisor:
    # Use a small frontier model available on the Hugging Face Hub.
    # Default to Qwen 3.1.7b for reasonable capabilities on modest hardware.
    DEFAULT_MODEL = "Qwen/Qwen3-1.7b"
    CANDIDATE_ACTIONS = ["comment", "pr"]

    PROMPT_TEMPLATE = (
        "You are an assistant that reads a GitHub issue and recommends an action.\n"
        "Only use one of these actions: comment, pr.\n\n"
        "Reply in the following exact format (no extra text):\n"
        "Action: <comment|pr>\n"
        "Detail: <if action=comment, write a comment; if action=pr, write the branch name>\n"
        "Research notebook: <path to a notebook file>\n\n"
        "ISSUE:\n{issue_text}\n\n"
        "RESPONSE:\n"
    )

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("MODEL_NAME") or self.DEFAULT_MODEL
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device=-1,
                trust_remote_code=True,
            )
        except Exception as e:
            LOG.error("Failed to load model %s: %s", self.model_name, str(e))
            raise

    def advise(self, issue_text: str, issue_number: Optional[int] = None) -> Dict[str, str]:
        """Return an action, detail, and a research notebook path."""

        prompt = self.PROMPT_TEMPLATE.format(issue_text=issue_text.strip())

        out = self.pipeline(
            prompt,
            max_new_tokens=64,
            temperature=0.0,
            do_sample=False,
        )

        raw = out[0]["generated_text"][len(prompt) :].strip()

        action = "comment"
        detail = ""
        research_notebook = ""

        # Parse best-effort lines; allow missing fields.
        for line in raw.splitlines():
            if line.lower().startswith("action:"):
                action = line.split(":", 1)[1].strip().lower()
            elif line.lower().startswith("detail:"):
                detail = line.split(":", 1)[1].strip()
            elif line.lower().startswith("research notebook:"):
                research_notebook = line.split(":", 1)[1].strip()

        if action not in self.CANDIDATE_ACTIONS:
            action = "comment"

        # If the model didn’t emit a detail line, use any remaining text as the detail.
        if not detail and raw:
            lines = [l for l in raw.splitlines() if not l.lower().startswith("action:")]
            detail = "\n".join(lines).strip()

        # Ensure we always return something useful.
        if not detail:
            defaults = {
                "comment": "Add a short comment outlining what you found and next steps.",
                "pr": "Suggest a branch name or PR summary.",
            }
            detail = defaults.get(action, "No detail provided.")

        # Research notebook is required; default to a sane path if the model omits it.
        if not research_notebook:
            if issue_number is not None:
                research_notebook = f"research/issue-{issue_number}.ipynb"
            else:
                research_notebook = "research/notebook.ipynb"

        return {
            "action": action,
            "detail": detail,
            "research_notebook": research_notebook,
        }