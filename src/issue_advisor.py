"""
Advanced issue advisor using SmolAgents framework with CodeAgent and TransformersModel.

This module provides intelligent analysis of GitHub issues by leveraging large language models
to understand issue content, identify potential root causes, and suggest actionable next steps
for maintainers and contributors. The advisor uses a structured approach with tool-augmented
reasoning capabilities to provide comprehensive guidance on bug reports and feature requests.
"""

from __future__ import annotations

import json
import logging
import os
import traceback
from typing import Dict, List, Optional

from smolagents import CodeAgent, TransformersModel


LOG = logging.getLogger(__name__)


class IssueAdvisor:
    # Default base model for the advisor
    DEFAULT_MODEL = "HuggingFaceTB/SmolLM-1.7B"

    SYSTEM_PROMPT = (
        "You are an assistant that reads a GitHub issue and provides guidance.\n"
        "Provide clear, actionable advice based on the issue content.\n"
    )

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("MODEL_NAME") or self.DEFAULT_MODEL
        try:
            # Many text models (like gpt2) do not provide a built-in chat template,
            # and TransformersModel.apply_chat_template requires one.
            # For a non-chat LM, we set a simple template covering user/assistant turns.
            template = """{% for message in messages %}"""
            template += """
{% if message.role == 'system' %}SYSTEM: {{ message.content }}\n
"""
            template += """
{% elif message.role == 'user' %}USER: {{ message.content }}\n
"""
            template += """
{% elif message.role == 'assistant' %}ASSISTANT: {{ message.content }}\n
"""
            template += """
{% endif %}
{% endfor %}
"""
            self.model = TransformersModel(
                model_id=self.model_name,
            )
            # For small models like gpt2, flatten_messages_as_text must be False to support string content correctly.
            self.model.flatten_messages_as_text = False
            # Ensure the tokenizer does not lose the template and uses it for chat formatting.
            self.model.tokenizer.chat_template = template

            if self.model.tokenizer.pad_token_id is None:
                self.model.tokenizer.pad_token = self.model.tokenizer.eos_token

            self.agent = CodeAgent(
                tools=[],
                model=self.model,
                max_steps=1,  # Critical: reduce token-expansion from multi-step reasoning
            )
        except Exception as e:
            LOG.error("Failed to initialize SmolAgents with model %s: %s", self.model_name, str(e))
            traceback.print_exc()
            raise

    def _generate_direct(self, issue_text: str) -> str:
        """Generate an advisory response directly from the underlying model."""
        try:
            model_max = getattr(self.model.tokenizer, "model_max_length", 1024)
            inputs = self.model.tokenizer(
                issue_text,
                truncation=True,
                max_length=model_max,
                return_tensors="pt",
                padding="longest",
            )
            pad_token_id = self.model.tokenizer.pad_token_id or self.model.tokenizer.eos_token_id
            attention_mask = inputs.get("attention_mask")

            if getattr(self.model, "model", None) is not None:
                output_ids = self.model.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=attention_mask,
                    max_new_tokens=256,
                    pad_token_id=pad_token_id,
                    eos_token_id=self.model.tokenizer.eos_token_id,
                )
                raw = self.model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                return raw.strip()

            result = self.model.generate(
                [{"role": "user", "content": issue_text}],
                max_new_tokens=256,
                stop_sequences=["\n\n"],
            )
            if hasattr(result, "content"):
                return result.content.strip()
            return str(result).strip()
        except Exception as e:
            LOG.error("Direct model generation failed: %s", str(e))
            return f"Error analyzing issue: {e}"

    def advise(self, issue_text: str, issue_number: Optional[int] = None) -> str:
        """Return the agent's response for the given issue text."""

        cleaned_issue = issue_text.strip()
        if not cleaned_issue:
            return "No issue text provided for analysis."

        # Guard against very long issues exceeding model context length.
        max_model_tokens = getattr(self.model.tokenizer, "model_max_length", 1024)
        max_issue_tokens = int(max_model_tokens * 0.5)  # stronger hard cap for agent overhead
        try:
            tokenized_issue = self.model.tokenizer(cleaned_issue, add_special_tokens=False)
            issue_token_len = len(tokenized_issue["input_ids"])
            if issue_token_len > max_issue_tokens:
                LOG.warning(
                    "Issue text is too long (%d tokens), truncating to %d tokens for model context",
                    issue_token_len,
                    max_issue_tokens,
                )
                front_ids = tokenized_issue["input_ids"][: max_issue_tokens // 2]
                back_ids = tokenized_issue["input_ids"][-max_issue_tokens // 2 :]
                truncated_ids = front_ids + back_ids
                cleaned_issue = self.model.tokenizer.decode(truncated_ids, skip_special_tokens=True)
                # For large inputs, avoid CodeAgent expansion by running direct generation path.
                return self._generate_direct(cleaned_issue)
        except Exception as e:
            LOG.warning("Unable to compute token length for truncation: %s", e)

        prompt = f"Please analyze this GitHub issue and provide guidance:\n\n{cleaned_issue}"
        token_length = len(self.model.tokenizer(cleaned_issue, add_special_tokens=False)["input_ids"])
        LOG.debug("Using cleaned issue length %d tokens before CodeAgent run", token_length)

        try:
            response = self.agent.run(prompt)
            return str(response)
        except Exception as e:
            LOG.error("Error running CodeAgent: %s", str(e))
            raise RuntimeError(f"CodeAgent failed: {e}") from e

