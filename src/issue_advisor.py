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

    PROMPT_TEMPLATE = (
        "You are an assistant that reads a GitHub issue and provides guidance.\n\n"
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

    def advise(self, issue_text: str, issue_number: Optional[int] = None) -> str:
        """Return the raw model output for the given issue text."""

        prompt = self.PROMPT_TEMPLATE.format(issue_text=issue_text.strip())

        out = self.pipeline(
            prompt,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,
        )

        # The pipeline includes the prompt; remove it to return only the model response.
        return out[0]["generated_text"][len(prompt) :].strip()

