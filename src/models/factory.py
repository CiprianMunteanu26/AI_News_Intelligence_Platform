"""
Model factory — handles loading LLMs and tokenizers.
Support for Hugging Face models, quantization, and device placement.
"""

from __future__ import annotations

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelFactory:
    """
    Factory for loading and configuring LLMs.
    """

    @staticmethod
    def load_model(
        model_name: str,
        device_map: str = "auto",
        use_half_precision: bool = True,
        load_in_4bit: bool = False,
        trust_remote_code: bool = True,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a model and tokenizer from Hugging Face.

        Parameters
        ----------
        model_name : str
            Repo ID or local path.
        device_map : str
            Device placement strategy.
        use_half_precision : bool
            Whether to use float16/bfloat16.
        load_in_4bit : bool
            Whether to use bitsandbytes 4-bit quantization.

        Returns
        -------
        tuple[PreTrainedModel, PreTrainedTokenizer]
        """
        logger.info("Loading model: %s", model_name)

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)

        # Ensure pad_token is set (needed for some models like Mistral/Llama)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Model loading kwargs
        kwargs = {
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }

        if load_in_4bit:
            # Requires bitsandbytes
            kwargs["load_in_4bit"] = True
            logger.info("Using 4-bit quantization")
        elif use_half_precision:
            # Check for bfloat16 support, fallback to float16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                kwargs["torch_dtype"] = torch.bfloat16
                logger.debug("Using bfloat16")
            else:
                kwargs["torch_dtype"] = torch.float16
                logger.debug("Using float16")

        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            logger.info("Model loaded successfully")
            return model, tokenizer
        except Exception as e:
            logger.error("Failed to load model %s: %s", model_name, e)
            raise

    @staticmethod
    def load_tiny_fallback() -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a tiny model for local testing without large VRAM requirements.
        Defaults to a small GPT-2 or similar.
        """
        # Using a very small model as default fallback
        tiny_model_name = "sshleifer/tiny-gpt2"
        logger.warning("Loading tiny fallback model: %s", tiny_model_name)
        return ModelFactory.load_model(tiny_model_name, device_map="cpu", use_half_precision=False)
