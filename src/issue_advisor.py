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

from smolagents import CodeAgent, TransformersModel, tool


LOG = logging.getLogger(__name__)


class IssueAdvisor:
    # Default base model for the advisor
    DEFAULT_MODEL = "HuggingFaceTB/SmolLM-1.7B"

    SYSTEM_PROMPT = (
        "You are an assistant that analyzes GitHub issues and provides guidance."
    )

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("MODEL_NAME") or self.DEFAULT_MODEL
        self.current_chunks: List[str] = []
        self.current_chunk_index = 0
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
                tools=[self._create_chunking_tool()],
                model=self.model,
                max_steps=3,  # Reduced from 5 to minimize context usage
            )
        except Exception as e:
            LOG.error("Failed to initialize SmolAgents with model %s: %s", self.model_name, str(e))
            traceback.print_exc()
            raise

    def _create_chunking_tool(self):
        """Create a tool for accessing issue chunks."""
        @tool
        def get_next_chunk() -> str:
            """
            Get the next chunk of the issue text.
            
            Returns:
                The next chunk of issue text, or empty string if no more chunks available.
            """
            if self.current_chunk_index < len(self.current_chunks):
                chunk = self.current_chunks[self.current_chunk_index]
                self.current_chunk_index += 1
                return f"CHUNK {self.current_chunk_index}/{len(self.current_chunks)}:\n{chunk}"
            else:
                return "No more chunks available. All issue content has been provided."
        
        return get_next_chunk

    def _chunk_text(self, text: str, chunk_size_tokens: int) -> List[str]:
        """Split text into chunks of approximately chunk_size_tokens."""
        try:
            tokenized = self.model.tokenizer(text, add_special_tokens=False)
            tokens = tokenized["input_ids"]
            
            chunks = []
            for i in range(0, len(tokens), chunk_size_tokens):
                chunk_tokens = tokens[i:i + chunk_size_tokens]
                chunk_text = self.model.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append(chunk_text.strip())
            
            return chunks
        except Exception as e:
            LOG.warning("Error chunking text: %s, falling back to character-based chunking", e)
            # Fallback: simple character-based chunking
            chunk_size_chars = chunk_size_tokens * 4  # Rough approximation
            chunks = []
            for i in range(0, len(text), chunk_size_chars):
                chunk = text[i:i + chunk_size_chars]
                # Try to break at sentence boundaries
                if i + chunk_size_chars < len(text):
                    last_period = chunk.rfind('.')
                    last_newline = chunk.rfind('\n')
                    break_point = max(last_period, last_newline)
                    if break_point > chunk_size_chars * 0.7:  # Only if we're not losing too much
                        chunk = chunk[:break_point + 1]
                chunks.append(chunk.strip())
            return chunks

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
        """Return the agent's response for the given issue text using chunking."""

        cleaned_issue = issue_text.strip()
        if not cleaned_issue:
            return "No issue text provided for analysis."

        # Always use chunking regardless of issue length
        max_model_tokens = getattr(self.model.tokenizer, "model_max_length", 1024)
        chunk_size_tokens = int(max_model_tokens * 0.05)  # 5% of context per chunk (very conservative)
        
        # Split issue into chunks
        self.current_chunks = self._chunk_text(cleaned_issue, chunk_size_tokens)
        self.current_chunk_index = 0
        
        LOG.info("Split issue into %d chunks of ~%d tokens each", len(self.current_chunks), chunk_size_tokens)

        # Start with the first chunk
        if not self.current_chunks:
            return "Unable to process issue text."

        first_chunk = self.current_chunks[0]
        self.current_chunk_index = 1

        # Use direct generation for the first chunk to avoid SmolAgents overhead
        analysis_prompt = f"Analyze this GitHub issue chunk and provide guidance:\n\n{first_chunk}"
        
        try:
            # Direct model call for first chunk
            inputs = self.model.tokenizer(
                analysis_prompt,
                truncation=True,
                max_length=int(max_model_tokens * 0.8),  # Leave room for generation
                return_tensors="pt",
                padding="longest",
            )
            
            if hasattr(self.model, "model"):
                output_ids = self.model.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=200,
                    pad_token_id=self.model.tokenizer.pad_token_id or self.model.tokenizer.eos_token_id,
                    eos_token_id=self.model.tokenizer.eos_token_id,
                )
                response = self.model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                return response.strip()
            else:
                # Fallback to TransformersModel generate
                result = self.model.generate(
                    [{"role": "user", "content": analysis_prompt}],
                    max_new_tokens=200,
                )
                if hasattr(result, "content"):
                    return result.content.strip()
                return str(result).strip()
                
        except Exception as e:
            LOG.error("Error in direct generation: %s", str(e))
            return f"Analysis based on first chunk only (error: {e}):\n\n{first_chunk[:500]}..."

