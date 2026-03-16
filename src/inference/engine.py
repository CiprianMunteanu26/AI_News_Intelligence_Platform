"""
Inference engine — handles model generation calls.
Manages tokenization, generation parameters, and device placement.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.utils.logger import get_logger

logger = get_logger(__name__)


class InferenceEngine:
    """
    Orchestrates the LLM generation process.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

        # Determine device
        self.device = next(model.parameters()).device
        logger.debug("Inference engine initialized on device: %s", self.device)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text for a given prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Override defaults with kwargs if provided
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        gen_kwargs.update(kwargs)

        with torch.no_grad():
            output_tokens = self.model.generate(**inputs, **gen_kwargs)

        # Get only the newly generated tokens (skip the prompt)
        new_tokens = output_tokens[0][inputs["input_ids"].shape[1] :]

        # Decode and clean
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()

    def batch_generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        """
        Generate responses for a batch of prompts.
        Note: Basic implementation, handles one by one for now to avoid padding complexity.
        """
        # TODO: Implement proper batching with padding and attention masks if needed for performance.
        responses = []
        for p in prompts:
            responses.append(self.generate(p, **kwargs))
        return responses
