"""Simple LM-based issue advisor using a next-token predictor."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from transformers import pipeline


LOG = logging.getLogger(__name__)


class IssueAdvisor:
    """Recommend a label and an initial comment for a GitHub issue."""

    # Use a small frontier model available on the Hugging Face Hub.
    # `distilgpt2` is a lightweight, open model that works well for simple prompt tasks.
    DEFAULT_MODEL = "distilgpt2"
    CANDIDATE_LABELS = ["comment", "pr", "ignore", "needs-research"]

    PROMPT_TEMPLATE = (
        "You are an assistant that reads a GitHub issue and provides an initial label and comment." "\n"
        "Only use one of these labels: comment, pr, ignore, needs-research.\n\n"
        "Reply in the following exact format (no extra text):\n"
        "Label: <one of comment|pr|ignore|needs-research>\n"
        "Comment: <a short actionable note about next steps>\n\n"
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

    def advise(self, issue_text: str) -> Dict[str, str]:
        """Return a label, a short comment, and a pseudo-confidence score."""

        prompt = self.PROMPT_TEMPLATE.format(issue_text=issue_text.strip())

        out = self.pipeline(
            prompt,
            max_new_tokens=64,
            temperature=0.0,
            do_sample=False,
        )

        raw = out[0]["generated_text"][len(prompt) :].strip()

        label = "needs-research"
        comment = ""

        # Parse best-effort lines; allow missing Comment line.
        for line in raw.splitlines():
            if line.lower().startswith("label:"):
                label = line.split(":", 1)[1].strip().lower()
            elif line.lower().startswith("comment:"):
                comment = line.split(":", 1)[1].strip()

        if label not in self.CANDIDATE_LABELS:
            label = "needs-research"

        # If the model didn’t emit a comment line, use any remaining text as the comment.
        if not comment and raw:
            # Remove potential label line and treat the rest as comment
            lines = [l for l in raw.splitlines() if not l.lower().startswith("label:")]
            comment = "\n".join(lines).strip()

        # If we still don’t have a comment, provide a safe default
        if not comment:
            defaults = {
                "comment": "Add a short comment summarizing what you found and next steps.",
                "pr": "Consider opening a PR with a fix or reproduction case.",
                "ignore": "No action needed; this looks already handled or out of scope.",
                "needs-research": "Try reproducing the issue locally and gather logs / a minimal repro.",
            }
            comment = defaults.get(label, "No comment available.")

        # The model doesn't provide a real confidence; synthesize a simple score.
        score = 1.0 if label in self.CANDIDATE_LABELS else 0.0
        return {"label": label, "comment": comment, "score": float(score)}
