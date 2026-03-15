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
    CANDIDATE_LABELS = ["comment", "pr", "research-folder"]

    PROMPT_TEMPLATE = (
        "You are an assistant that reads a GitHub issue and recommends an action plus next steps." "\n"
        "Only use one of these actions: comment, pr, research-folder.\n\n"
        "Reply in the following exact format (no extra text):\n"
        "Action: <one of comment|pr|research-folder>\n"
        "Next steps: <a short actionable suggestion — e.g., comment text, a research-folder plan, or a PR idea>\n\n"
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

        action = "comment"
        comment = ""

        # Parse best-effort lines; allow missing Next steps line.
        for line in raw.splitlines():
            if line.lower().startswith("action:"):
                action = line.split(":", 1)[1].strip().lower()
            elif line.lower().startswith("next steps:"):
                comment = line.split(":", 1)[1].strip()

        if action not in self.CANDIDATE_LABELS:
            action = "comment"

        # If the model didn’t emit a Next steps line, use any remaining text as the suggestion.
        if not comment and raw:
            # Remove potential action line and treat the rest as the suggestion
            lines = [l for l in raw.splitlines() if not l.lower().startswith("action:")]
            comment = "\n".join(lines).strip()

        # If we still don’t have a next-steps suggestion, provide a safe default
        if not comment:
            defaults = {
                "comment": "Add a short comment outlining what you found and next steps.",
                "pr": "Consider opening a PR with a fix or reproduction case.",
                "research-folder": "Create a research folder with repro steps, logs, and minimal examples.",
            }
            comment = defaults.get(action, "No next steps available.")

        # The model doesn't provide a real confidence; synthesize a simple score.
        score = 1.0 if action in self.CANDIDATE_LABELS else 0.0
        return {"action": action, "next_steps": comment, "score": float(score)}

        # The model doesn't provide a real confidence; synthesize a simple score.
        score = 1.0 if label in self.CANDIDATE_LABELS else 0.0
        return {"label": label, "comment": comment, "score": float(score)}
