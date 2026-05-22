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
        self._device = self._select_device()
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        ).to(self._device)
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

        # Read tokens from the streamer's queue without blocking the event loop.
        # streamer.text_queue.get() is blocking, so we offload each call to
        # a thread executor. This lets other async tasks (e.g. the frontier
        # model in arena mode) run while we wait for the next token.
        loop = asyncio.get_running_loop()
        while True:
            token = await loop.run_in_executor(None, streamer.text_queue.get)
            if token is streamer.stop_signal:
                break
            if token:
                yield token

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
            self._device
        )

    @staticmethod
    def _select_device() -> str:
        """Select the best available device, avoiding MPS due to compatibility issues."""
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @torch.inference_mode()
    def _run_generation(self, kwargs: dict) -> None:
        """Run model.generate in a thread (called by streamer pattern)."""
        self._model.generate(**kwargs)
