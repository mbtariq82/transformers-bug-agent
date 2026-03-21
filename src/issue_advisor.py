"""Simple LM-based issue advisor using SmolAgents."""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
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
    # Use a small, well-supported model for smolagents
    DEFAULT_MODEL = "HuggingFaceTB/SmolLM-1.7B"

    SYSTEM_PROMPT = (
        "You are an assistant that reads a GitHub issue and provides guidance.\n"
        "You can use the execute_safe_command tool to run terminal commands for investigation.\n"
        "Provide clear, actionable advice based on the issue content.\n"
    )

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("MODEL_NAME") or self.DEFAULT_MODEL
        try:
            model = TransformersModel(model_id=self.model_name)
            self.agent = CodeAgent(
                tools=[execute_safe_command],
                model=model,
                system_prompt=self.SYSTEM_PROMPT,
                max_steps=5,  # Limit steps to avoid infinite loops
            )
        except Exception as e:
            LOG.error("Failed to initialize SmolAgents with model %s: %s", self.model_name, str(e))
            raise

    def advise(self, issue_text: str, issue_number: Optional[int] = None) -> str:
        """Return the agent's response for the given issue text."""
        
        prompt = f"Please analyze this GitHub issue and provide guidance:\n\n{issue_text.strip()}"
        
        try:
            response = self.agent.run(prompt)
            return str(response)
        except Exception as e:
            LOG.error("Error running agent: %s", str(e))
            return f"Error analyzing issue: {e}"

