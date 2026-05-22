"""Gradio web UI for the AI personal assistant."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator

import gradio as gr
from dotenv import load_dotenv

from src.config import load_settings
from src.guardrails import check_input, check_output
from src.memory import ConversationMemory
from src.models.base import BaseModel, Message

logger = logging.getLogger(__name__)

load_dotenv()

# -- Model registry --

_models: dict[str, BaseModel] = {}


def _get_or_create_model(model_key: str) -> BaseModel:
    """Lazy-load and cache model backends."""
    if model_key in _models:
        return _models[model_key]

    settings = load_settings()

    if model_key == "Frontier (OpenAI)":
        from src.models.frontier_model import FrontierModel

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for the frontier model.")
        _models[model_key] = FrontierModel(
            api_key=settings.openai_api_key,
            model_name=settings.frontier_model_name,
        )
    elif model_key == "OSS (Qwen2.5-0.5B)":
        from src.models.oss_model import OSSModel

        _models[model_key] = OSSModel(model_name=settings.oss_model_name)
    else:
        raise ValueError(f"Unknown model: {model_key}")

    logger.info("Loaded model backend: %s", model_key)
    return _models[model_key]


# -- Per-session memory --

_memories: dict[str, ConversationMemory] = {}

SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI assistant. "
    "Answer questions clearly and concisely. "
    "If you don't know something, say so rather than guessing."
)


def _get_memory(session_id: str) -> ConversationMemory:
    """Get or create conversation memory for a session."""
    if session_id not in _memories:
        settings = load_settings()
        _memories[session_id] = ConversationMemory(
            max_turns=settings.max_conversation_turns,
            system_prompt=SYSTEM_PROMPT,
        )
    return _memories[session_id]


# -- Chat handler --


async def _stream_response(
    model: BaseModel, messages: list[Message]
) -> AsyncIterator[str]:
    """Stream model response with output guardrail check."""
    full_response = ""
    async for token in model.stream(messages):
        full_response += token
        yield token

    # Check complete response against output guardrails
    output_check = check_output(full_response)
    if output_check.is_blocked:
        yield f"\n\n[Response filtered: {output_check.reason}]"


async def respond(
    message: str,
    history: list[dict],
    model_choice: str,
) -> AsyncIterator[str]:
    """Handle a chat message with guardrails, memory, and streaming.

    Args:
        message: User's input text.
        history: Gradio-managed conversation history.
        model_choice: Selected model backend key.

    Yields:
        Streamed response text chunks.
    """
    # Input guardrails
    input_check = check_input(message)
    if input_check.is_blocked:
        yield input_check.reason
        return

    # Build session key from model choice (simple approach for single-user)
    session_id = f"default_{model_choice}"
    memory = _get_memory(session_id)

    # Record user message and get snapshot for model
    memory.add_user_message(message)
    snapshot = memory.get_snapshot()
    messages = snapshot.to_message_list()

    # Get model and stream response
    model = _get_or_create_model(model_choice)
    full_response = ""

    async for chunk in _stream_response(model, messages):
        full_response += chunk
        yield full_response

    # Record assistant response in memory
    memory.add_assistant_message(full_response)


# -- App factory --


def create_app() -> gr.Blocks:
    """Build and return the Gradio application."""
    model_choices = ["OSS (Qwen2.5-0.5B)", "Frontier (OpenAI)"]

    # Default to OSS if no API key is set
    default_model = (
        "Frontier (OpenAI)"
        if os.getenv("OPENAI_API_KEY")
        else "OSS (Qwen2.5-0.5B)"
    )

    with gr.Blocks(title="AI Personal Assistant") as app:
        gr.Markdown("# AI Personal Assistant\nCompare OSS and Frontier models side by side.")

        model_selector = gr.Dropdown(
            choices=model_choices,
            value=default_model,
            label="Model",
            interactive=True,
        )

        gr.ChatInterface(
            fn=respond,
            additional_inputs=[model_selector],
            type="messages",
        )

    return app


def main() -> None:
    """Launch the Gradio app."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    app = create_app()
    app.launch()


if __name__ == "__main__":
    main()
