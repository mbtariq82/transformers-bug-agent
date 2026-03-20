"""Simple LM-based issue advisor using a next-token predictor."""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
from typing import Dict, List, Optional

from transformers import pipeline


LOG = logging.getLogger(__name__)


class IssueAdvisor:
    # Use a small frontier model available on the Hugging Face Hub.
    DEFAULT_MODEL = "Qwen/Qwen3-1.7b"

    PROMPT_TEMPLATE = (
        "You are an assistant that reads a GitHub issue and provides guidance.\n"
        "If you need to run terminal commands to investigate, output a JSON object like:\n"
        "{{\"commands\": [\"ls -la\", \"pwd\"]}}\n"
        "Otherwise, just provide text guidance.\n\n"
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
            do_sample=False,
        )

        # The pipeline includes the prompt; remove it to return only the model response.
        raw_response = out[0]["generated_text"][len(prompt) :].strip()

        # Try to parse as JSON for commands
        try:
            data = json.loads(raw_response)
            if "commands" in data and isinstance(data["commands"], list):
                return self._execute_commands(data["commands"])
        except json.JSONDecodeError:
            pass  # Not JSON, treat as text

        return raw_response  # Fallback to text

    def _execute_commands(self, commands: List[str]) -> str:
        results = []
        allowed_commands = {"ls", "pwd", "echo", "cat", "head", "tail"}  # Whitelist for safety
        for cmd in commands:
            base_cmd = shlex.split(cmd)[0]
            if base_cmd not in allowed_commands:
                results.append(f"Blocked unsafe command: {cmd}")
                continue
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
                results.append(f"$ {cmd}\n{result.stdout}\n{result.stderr}")
            except Exception as e:
                results.append(f"Error running {cmd}: {e}")
        return "\n\n".join(results)

