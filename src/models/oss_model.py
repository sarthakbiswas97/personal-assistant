"""Open-source model backend using Hugging Face transformers."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from threading import Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

from src.models.base import BaseModel, Message

logger = logging.getLogger(__name__)


class OSSModel(BaseModel):
    """Qwen2.5-0.5B-Instruct backend via Hugging Face transformers.

    Runs inference in a background thread to avoid blocking the
    async event loop, and exposes an async streaming interface
    via TextIteratorStreamer.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> None:
        self._model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self._model.eval()
        logger.info("Loaded OSS model: %s", model_name)

    async def generate(self, messages: list[Message]) -> str:
        """Generate a complete response by collecting all streamed tokens."""
        chunks: list[str] = []
        async for token in self.stream(messages):
            chunks.append(token)
        return "".join(chunks)

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """Stream response tokens using TextIteratorStreamer in a background thread."""
        input_ids = self._prepare_input(messages)

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": 512,
            "streamer": streamer,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
        }

        thread = Thread(target=self._run_generation, args=(generation_kwargs,))
        thread.start()

        loop = asyncio.get_running_loop()
        for token in streamer:
            if token:
                yield token
            # Yield control back to the event loop between tokens
            await asyncio.sleep(0)

        await loop.run_in_executor(None, thread.join)

    async def cleanup(self) -> None:
        """Release model from memory."""
        del self._model
        del self._tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleaned up OSS model resources")

    def _prepare_input(self, messages: list[Message]) -> torch.Tensor:
        """Format messages using the model's chat template and tokenize."""
        message_dicts = [m.to_dict() for m in messages]
        text = self._tokenizer.apply_chat_template(
            message_dicts,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self._tokenizer([text], return_tensors="pt").input_ids.to(
            self._model.device
        )

    @torch.inference_mode()
    def _run_generation(self, kwargs: dict) -> None:
        """Run model.generate in a thread (called by streamer pattern)."""
        self._model.generate(**kwargs)
