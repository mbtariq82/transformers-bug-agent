"""Simple LM-based issue advisor using SmolAgents."""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import traceback
from typing import Dict, List, Optional

from smolagents import CodeAgent, TransformersModel, tool


LOG = logging.getLogger(__name__)


@tool
def execute_safe_command(command: str) -> str:
    """
    Execute a safe terminal command and return the output.
    
    Args:
        command: The command to execute
        
    Returns:
        The command output or error message
    """
    allowed_commands = {"ls", "pwd", "echo", "cat", "head", "tail"}  # Whitelist for safety
    base_cmd = shlex.split(command)[0]
    if base_cmd not in allowed_commands:
        return f"Blocked unsafe command: {command}"
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR: {result.stderr}"
        return output.strip()
    except Exception as e:
        return f"Error running {command}: {e}"


class IssueAdvisor:
    # Default base model for the advisor
    DEFAULT_MODEL = "HuggingFaceTB/SmolLM-1.7B"

    SYSTEM_PROMPT = (
        "You are an assistant that reads a GitHub issue and provides guidance.\n"
        "You can use the execute_safe_command tool to run terminal commands for investigation.\n"
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
                apply_chat_template_kwargs={"chat_template": template},
            )
            # For small models like gpt2, flatten_messages_as_text must be False to support string content correctly.
            self.model.flatten_messages_as_text = False
            # Ensure the tokenizer does not lose the template and uses it for chat formatting.
            self.model.tokenizer.chat_template = template

            if self.model.tokenizer.pad_token_id is None:
                self.model.tokenizer.pad_token = self.model.tokenizer.eos_token

            self.agent = CodeAgent(
                tools=[execute_safe_command],
                model=self.model,
                max_steps=5,  # Limit steps to avoid infinite loops
            )
        except Exception as e:
            LOG.error("Failed to initialize SmolAgents with model %s: %s", self.model_name, str(e))
            traceback.print_exc()
            raise

    def _generate_direct(self, issue_text: str) -> str:
        """Generate an advisory response directly from the underlying model."""
        try:
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
        max_issue_tokens = max_model_tokens - 256  # reserve room for instructions + response
        try:
            tokenized_issue = self.model.tokenizer(cleaned_issue, add_special_tokens=False)
            issue_token_len = len(tokenized_issue["input_ids"])
            if issue_token_len > max_issue_tokens:
                LOG.warning(
                    "Issue text is too long (%d tokens), truncating to %d tokens for model context",
                    issue_token_len,
                    max_issue_tokens,
                )
                truncated_ids = tokenized_issue["input_ids"][-max_issue_tokens:]
                cleaned_issue = self.model.tokenizer.decode(truncated_ids, skip_special_tokens=True)
        except Exception as e:
            LOG.warning("Unable to compute token length for truncation: %s", e)

        prompt = f"Please analyze this GitHub issue and provide guidance:\n\n{cleaned_issue}"
        token_length = len(self.model.tokenizer(cleaned_issue, add_special_tokens=False)["input_ids"])
        LOG.debug("Using cleaned issue length %d tokens before CodeAgent run", token_length)

        # Use direct generation for models with narrow context windows to avoid large CodeAgent prompt overhead.
        if getattr(self.model.tokenizer, "model_max_length", 1024) < 2048:
            LOG.info("Model has small context window (%d tokens), using direct generation path", self.model.tokenizer.model_max_length)
            return self._generate_direct(cleaned_issue)

        try:
            response = self.agent.run(prompt)
            return str(response)
        except Exception as e:
            LOG.error("Error running CodeAgent: %s", str(e))
            return self._generate_direct(cleaned_issue)

