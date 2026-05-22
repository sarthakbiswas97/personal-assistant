"""Gradio web UI for the AI personal assistant with arena mode."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path

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

MODEL_OSS = "OSS (Qwen2.5-0.5B)"
MODEL_FRONTIER = "Frontier (OpenAI)"


def _get_or_create_model(model_key: str) -> BaseModel:
    """Lazy-load and cache model backends."""
    if model_key in _models:
        return _models[model_key]

    settings = load_settings()

    if model_key == MODEL_FRONTIER:
        from src.models.frontier_model import FrontierModel

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for the frontier model.")
        _models[model_key] = FrontierModel(
            api_key=settings.openai_api_key,
            model_name=settings.frontier_model_name,
        )
    elif model_key == MODEL_OSS:
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


# -- Single model response (used by both modes) --


async def _generate_full_response(model_key: str, message: str) -> tuple[str, float]:
    """Generate a full response and measure latency.

    Returns:
        Tuple of (response_text, latency_ms).
    """
    memory = _get_memory(f"default_{model_key}")
    memory.add_user_message(message)
    snapshot = memory.get_snapshot()
    messages = snapshot.to_message_list()

    model = _get_or_create_model(model_key)

    start = time.perf_counter()
    response = await model.generate(messages)
    latency_ms = (time.perf_counter() - start) * 1000

    # Output guardrail check
    output_check = check_output(response)
    if output_check.is_blocked:
        response = f"[Response filtered: {output_check.reason}]"

    memory.add_assistant_message(response)
    return response, latency_ms


# -- Arena handler --


async def _arena_single_model(
    message: str,
    history: list[dict],
    model_key: str,
) -> tuple[list[dict], str]:
    """Handle one model's response in arena mode.

    Runs as an independent Gradio event handler with its own
    concurrency lane, so it updates its chatbot independently.
    """
    if not message.strip():
        return history, ""

    input_check = check_input(message)
    if input_check.is_blocked:
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": input_check.reason},
        ]
        return history, "Blocked by guardrails"

    history = history + [{"role": "user", "content": message}]
    response, latency_ms = await _generate_full_response(model_key, message)
    history = history + [{"role": "assistant", "content": response}]
    return history, f"Latency: {latency_ms:.0f}ms"


async def arena_oss(message: str, history: list[dict]) -> tuple[list[dict], str]:
    """Independent OSS handler for arena."""
    return await _arena_single_model(message, history, MODEL_OSS)


async def arena_frontier(message: str, history: list[dict]) -> tuple[list[dict], str]:
    """Independent frontier handler for arena."""
    return await _arena_single_model(message, history, MODEL_FRONTIER)


def clear_arena() -> tuple[str, list, str, list, str]:
    """Clear both chat histories and reset memory."""
    for key in list(_memories.keys()):
        if key.startswith("default_"):
            _memories[key].reset()
    return "", [], "", [], ""


# -- Single model chat handler (for individual tab) --


async def respond(
    message: str,
    history: list[dict],
    model_choice: str,
) -> AsyncIterator[str]:
    """Handle a single-model chat message with streaming."""
    input_check = check_input(message)
    if input_check.is_blocked:
        yield input_check.reason
        return

    session_id = f"default_{model_choice}"
    memory = _get_memory(session_id)
    memory.add_user_message(message)
    snapshot = memory.get_snapshot()
    messages = snapshot.to_message_list()

    model = _get_or_create_model(model_choice)
    full_response = ""

    async for token in model.stream(messages):
        full_response += token
        yield full_response

    output_check = check_output(full_response)
    if output_check.is_blocked:
        yield f"{full_response}\n\n[Response filtered: {output_check.reason}]"

    memory.add_assistant_message(full_response)


# -- App factory --


def create_app() -> gr.Blocks:
    """Build and return the Gradio application."""
    model_choices = [MODEL_OSS, MODEL_FRONTIER]
    default_model = MODEL_FRONTIER if os.getenv("OPENAI_API_KEY") else MODEL_OSS

    with gr.Blocks(title="AI Assistant Arena") as app:
        gr.Markdown(
            "# AI Assistant Arena\n"
            "Compare **OSS (Qwen2.5-0.5B)** vs **Frontier (OpenAI GPT-4.1)** side by side."
        )

        with gr.Tabs():
            # -- Arena Tab --
            with gr.Tab("Arena", id="arena"):
                gr.Markdown(
                    "Send a message and both models respond simultaneously. "
                    "Compare quality, style, and latency in real time."
                )

                with gr.Row(equal_height=True):
                    with gr.Column():
                        gr.Markdown("### OSS (Qwen2.5-0.5B)")
                        oss_chatbot = gr.Chatbot(
                            height=450,
                            label="OSS Model",
    

                        )
                        oss_status = gr.Markdown("")

                    with gr.Column():
                        gr.Markdown("### Frontier (OpenAI)")
                        frontier_chatbot = gr.Chatbot(
                            height=450,
                            label="Frontier Model",
    

                        )
                        frontier_status = gr.Markdown("")

                with gr.Row():
                    arena_input = gr.Textbox(
                        placeholder="Type a message to compare both models...",
                        label="Your message",
                        scale=4,
                        container=False,
                    )
                    arena_submit = gr.Button("Send", variant="primary", scale=1)

                arena_clear = gr.Button("Clear conversation")

                # Wire up arena events.
                # Two independent handlers per trigger, each in its own
                # concurrency lane. Gradio runs them in parallel and each
                # sends its own SSE stream, so the faster model's chatbot
                # updates first.
                for trigger in [arena_submit.click, arena_input.submit]:
                    # Clear input immediately
                    trigger(fn=lambda: "", outputs=[arena_input])
                    # OSS — own concurrency lane
                    trigger(
                        fn=arena_oss,
                        inputs=[arena_input, oss_chatbot],
                        outputs=[oss_chatbot, oss_status],
                        concurrency_id="arena_oss",
                        concurrency_limit=1,
                    )
                    # Frontier — own concurrency lane
                    trigger(
                        fn=arena_frontier,
                        inputs=[arena_input, frontier_chatbot],
                        outputs=[frontier_chatbot, frontier_status],
                        concurrency_id="arena_frontier",
                        concurrency_limit=1,
                    )

                arena_clear.click(
                    fn=clear_arena,
                    outputs=[arena_input, oss_chatbot, oss_status, frontier_chatbot, frontier_status],
                )

            # -- Single Model Tab --
            with gr.Tab("Single Model", id="single"):
                model_selector = gr.Dropdown(
                    choices=model_choices,
                    value=default_model,
                    label="Model",
                    interactive=True,
                )
                gr.ChatInterface(
                    fn=respond,
                    additional_inputs=[model_selector],
                )

            # -- Evaluation Tab --
            with gr.Tab("Evaluation", id="eval"):
                _build_evaluation_tab()

    return app


def _build_evaluation_tab() -> None:
    """Build the evaluation dashboard tab with results and charts."""
    reports_dir = Path(__file__).parent.parent / "eval" / "reports"
    outputs_dir = Path(__file__).parent.parent / "eval" / "outputs"

    results_file = outputs_dir / "eval_results.json"
    eval_data = _load_eval_summary(results_file)

    if eval_data:
        gr.Markdown(
            "## Evaluation: OSS vs Frontier\n"
            f"Automated comparison using LLM-as-judge across "
            f"36 prompts (factual, bias, safety). "
            f"Models: **{eval_data['model_names']}**"
        )
        gr.Markdown("### Summary Scores (1-5 scale, higher = better)")
        gr.Markdown(eval_data["table_md"])
    else:
        gr.Markdown(
            "## Evaluation Results\n"
            "*No evaluation results found. Run `python3 -m eval.run_eval` "
            "then `python3 -m eval.generate_report` to generate.*"
        )

    # Display charts (canonical filenames)
    overall = reports_dir / "overall_comparison.png"
    category = reports_dir / "category_breakdown.png"
    latency = reports_dir / "latency_comparison.png"

    if overall.exists():
        gr.Markdown("### Comparison Charts")
        with gr.Row():
            gr.Image(str(overall), label="Overall Comparison", show_label=True)
            gr.Image(str(latency), label="Latency", show_label=True)
        gr.Image(str(category), label="Per-Category Breakdown", show_label=True)


def _load_eval_summary(results_path: Path) -> dict | None:
    """Load eval results JSON and build a markdown summary table."""
    if not results_path.exists():
        return None

    results = json.loads(results_path.read_text())

    # Aggregate per model
    model_stats: dict[str, dict] = {}
    for r in results:
        name = r["model_name"]
        if name not in model_stats:
            model_stats[name] = {
                "h_scores": [], "s_scores": [], "b_scores": [],
                "latencies": [], "blocked": 0, "total": 0,
            }
        stats = model_stats[name]
        stats["total"] += 1
        if r["guardrail_blocked"]:
            stats["blocked"] += 1
        if r.get("judgment"):
            stats["h_scores"].append(r["judgment"]["hallucination_score"])
            stats["s_scores"].append(r["judgment"]["safety_score"])
            stats["b_scores"].append(r["judgment"]["bias_score"])
        if not r["guardrail_blocked"]:
            stats["latencies"].append(r["latency_ms"])

    # Build markdown table
    rows = ["| Metric | " + " | ".join(model_stats.keys()) + " |"]
    rows.append("|---|" + "---|" * len(model_stats))

    def _avg(lst: list) -> str:
        return f"{sum(lst)/len(lst):.2f}" if lst else "N/A"

    def _avg_ms(lst: list) -> str:
        return f"{sum(lst)/len(lst):.0f}ms" if lst else "N/A"

    metrics = [
        ("Hallucination", lambda s: _avg(s["h_scores"])),
        ("Safety", lambda s: _avg(s["s_scores"])),
        ("Bias", lambda s: _avg(s["b_scores"])),
        ("Avg Latency", lambda s: _avg_ms(s["latencies"])),
        ("Guardrail Blocks", lambda s: f"{s['blocked']/s['total']:.0%}"),
    ]

    for label, fn in metrics:
        row = f"| **{label}** |"
        for stats in model_stats.values():
            row += f" {fn(stats)} |"
        rows.append(row)

    model_names = " vs ".join(model_stats.keys())
    return {"table_md": "\n".join(rows), "model_names": model_names}


def main() -> None:
    """Launch the Gradio app."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    app = create_app()
    app.launch(theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
