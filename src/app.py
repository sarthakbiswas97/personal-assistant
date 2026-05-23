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
from src.memory.manager import MemoryManager
from src.memory.persistence import RedisSessionStore
from src.memory.summarizer import ConversationSummarizer
from src.memory.working import WorkingMemory
from src.models.base import BaseModel, Message
from src.observability import MetricsCollector
from src.tools.calculator import CalculatorTool
from src.tools.registry import ToolRegistry
from src.tools.router import ToolRouter
from src.tools.web_search import WebSearchTool
from src.tools.wikipedia import WikipediaTool

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


# -- Per-session memory (tiered context management) --

_managers: dict[str, MemoryManager] = {}
_store: RedisSessionStore | None = None
_summarizer: ConversationSummarizer | None = None
_metrics: MetricsCollector | None = None
_registry: ToolRegistry | None = None

SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI assistant. "
    "Answer questions clearly and concisely. "
    "If you don't know something, say so rather than guessing."
)


def _get_store() -> RedisSessionStore:
    """Lazy-init shared Redis store."""
    global _store
    if _store is None:
        settings = load_settings()
        _store = RedisSessionStore(redis_url=settings.redis_url)
    return _store


def _get_summarizer() -> ConversationSummarizer:
    """Lazy-init shared summarizer."""
    global _summarizer
    if _summarizer is None:
        settings = load_settings()
        _summarizer = ConversationSummarizer(
            api_key=settings.openai_api_key,
            model_name=settings.frontier_model_name,
        )
    return _summarizer


def _get_metrics() -> MetricsCollector:
    """Lazy-init shared metrics collector."""
    global _metrics
    if _metrics is None:
        settings = load_settings()
        _metrics = MetricsCollector(redis_url=settings.redis_url)
    return _metrics


def _get_registry() -> ToolRegistry:
    """Lazy-init shared tool registry."""
    global _registry
    if _registry is None:
        settings = load_settings()
        router = ToolRouter(
            api_key=settings.openai_api_key,
            model_name=settings.frontier_model_name,
        )
        _registry = ToolRegistry(router)
        _registry.register(WebSearchTool())
        _registry.register(WikipediaTool())
        _registry.register(CalculatorTool())
        logger.info("Tool registry initialized: %s", _registry.tool_names)
    return _registry


def _get_memory(session_id: str) -> MemoryManager:
    """Get or create a MemoryManager for a session."""
    if session_id not in _managers:
        settings = load_settings()
        working = WorkingMemory(
            max_turns=settings.max_conversation_turns,
            system_prompt=SYSTEM_PROMPT,
        )
        _managers[session_id] = MemoryManager(
            session_id=session_id,
            working=working,
            summarizer=_get_summarizer(),
            store=_get_store(),
            metrics=_get_metrics(),
        )
    return _managers[session_id]


# -- Single model response (used by both modes) --


async def _generate_full_response(
    model_key: str, message: str, tool_context: str = ""
) -> tuple[str, float]:
    """Generate a full response and measure latency.

    Args:
        model_key: Which model to use.
        message: User's message.
        tool_context: Pre-computed tool results to inject into context.

    Returns:
        Tuple of (response_text, latency_ms).
    """
    memory = _get_memory(f"default_{model_key}")
    await memory.add_user_message(message)
    snapshot = memory.get_snapshot()
    messages = snapshot.to_message_list()

    # Inject tool context before the latest user message
    if tool_context:
        messages.insert(-1, Message(role="system", content=tool_context))

    model = _get_or_create_model(model_key)

    start = time.perf_counter()
    response = await model.generate(messages)
    latency_ms = (time.perf_counter() - start) * 1000

    # Output guardrail check
    output_check = check_output(response)
    if output_check.is_blocked:
        response = f"[Response filtered: {output_check.reason}]"

    await memory.add_assistant_message(response)

    # Record metrics
    asyncio.create_task(_get_metrics().record_request(model_key, latency_ms))

    return response, latency_ms


# -- Arena handler --


async def arena_respond(
    message: str,
    oss_history: list[dict],
    frontier_history: list[dict],
) -> tuple[str, list[dict], str, list[dict], str]:
    """Handle arena mode: both models respond concurrently.

    Fires both requests simultaneously via asyncio.create_task.
    Both responses appear together with independent latency measurements.
    """
    if not message.strip():
        return "", oss_history, "", frontier_history, ""

    input_check = check_input(message)
    if input_check.is_blocked:
        asyncio.create_task(_get_metrics().record_guardrail_block())
        blocked_msg = input_check.reason
        oss_history = oss_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": blocked_msg},
        ]
        frontier_history = frontier_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": blocked_msg},
        ]
        return (
            "", oss_history, "Blocked by guardrails",
            frontier_history, "Blocked by guardrails",
        )

    oss_history = oss_history + [{"role": "user", "content": message}]
    frontier_history = frontier_history + [{"role": "user", "content": message}]

    # Tools execute ONCE, results shared by both models
    tool_context = await _get_registry().route_and_execute(message)

    # Log tool calls to observability
    if tool_context:
        for result in _get_registry()._last_results:
            asyncio.create_task(
                _get_metrics().record_tool_call(result.tool_name, result.success)
            )

    # Fire both concurrently — latency measured independently inside each
    oss_task = asyncio.create_task(
        _generate_full_response(MODEL_OSS, message, tool_context)
    )
    frontier_task = asyncio.create_task(
        _generate_full_response(MODEL_FRONTIER, message, tool_context)
    )

    try:
        oss_response, oss_latency = await oss_task
    except ValueError as e:
        oss_response, oss_latency = f"Error: {e}", 0
        asyncio.create_task(_get_metrics().record_error())

    try:
        frontier_response, frontier_latency = await frontier_task
    except ValueError as e:
        frontier_response, frontier_latency = f"Error: {e}", 0
        asyncio.create_task(_get_metrics().record_error())

    oss_history = oss_history + [{"role": "assistant", "content": oss_response}]
    frontier_history = frontier_history + [
        {"role": "assistant", "content": frontier_response}
    ]

    return (
        "",
        oss_history, f"Latency: {oss_latency:.0f}ms",
        frontier_history, f"Latency: {frontier_latency:.0f}ms",
    )


async def clear_arena() -> tuple[str, list, str, list, str]:
    """Clear both chat histories and reset memory."""
    for key in list(_managers.keys()):
        if key.startswith("default_"):
            await _managers[key].reset()
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
        asyncio.create_task(_get_metrics().record_guardrail_block())
        yield input_check.reason
        return

    try:
        model = _get_or_create_model(model_choice)
    except ValueError as e:
        asyncio.create_task(_get_metrics().record_error())
        yield f"Error: {e}"
        return

    # Tool execution
    tool_context = await _get_registry().route_and_execute(message)
    for result in _get_registry()._last_results:
        asyncio.create_task(
            _get_metrics().record_tool_call(result.tool_name, result.success)
        )

    session_id = f"default_{model_choice}"
    memory = _get_memory(session_id)
    await memory.add_user_message(message)
    snapshot = memory.get_snapshot()
    messages = snapshot.to_message_list()

    if tool_context:
        messages.insert(-1, Message(role="system", content=tool_context))

    full_response = ""

    async for token in model.stream(messages):
        full_response += token
        yield full_response

    output_check = check_output(full_response)
    if output_check.is_blocked:
        yield f"{full_response}\n\n[Response filtered: {output_check.reason}]"

    await memory.add_assistant_message(full_response)


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

                # Single handler fires both models concurrently.
                # Both responses appear together with independent latency.
                arena_outputs = [
                    arena_input, oss_chatbot, oss_status,
                    frontier_chatbot, frontier_status,
                ]
                arena_submit.click(
                    fn=arena_respond,
                    inputs=[arena_input, oss_chatbot, frontier_chatbot],
                    outputs=arena_outputs,
                )
                arena_input.submit(
                    fn=arena_respond,
                    inputs=[arena_input, oss_chatbot, frontier_chatbot],
                    outputs=arena_outputs,
                )
                arena_clear.click(
                    fn=clear_arena,
                    outputs=arena_outputs,
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

            # -- Observability Tab --
            with gr.Tab("Observability", id="obs"):
                _build_observability_tab()

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


def _build_observability_tab() -> None:
    """Build the live observability dashboard tab."""

    async def _fetch_metrics() -> str:
        """Fetch current metrics from Redis and format as markdown."""
        collector = _get_metrics()
        summary = await collector.get_summary()

        if summary is None:
            return "*Redis not configured. Set REDIS_URL to enable observability.*"

        lines = ["## Live Metrics", ""]

        # Request counts
        lines.append("### Requests")
        if summary.requests:
            lines.append("| Model | Count | Avg Latency | P50 | P95 |")
            lines.append("|---|---|---|---|---|")
            for model, count in summary.requests.items():
                avg = f"{summary.latency_avg.get(model, 0):.0f}ms"
                p50 = f"{summary.latency_p50.get(model, 0):.0f}ms"
                p95 = f"{summary.latency_p95.get(model, 0):.0f}ms"
                lines.append(f"| {model} | {count} | {avg} | {p50} | {p95} |")
        else:
            lines.append("*No requests yet.*")

        # Tool metrics
        if summary.tool_calls:
            lines.append("")
            lines.append("### Tools")
            lines.append("| Tool | Calls | Failures |")
            lines.append("|---|---|---|")
            for tool, count in summary.tool_calls.items():
                failures = summary.tool_failures.get(tool, 0)
                lines.append(f"| {tool} | {count} | {failures} |")

        # System metrics
        lines.append("")
        lines.append("### System")
        lines.append(f"- **Guardrail blocks:** {summary.guardrail_blocks}")
        lines.append(f"- **Context summarizations:** {summary.summarizations}")
        lines.append(f"- **Session restores (from Redis):** {summary.session_restores}")
        lines.append(f"- **Errors:** {summary.errors}")

        return "\n".join(lines)

    gr.Markdown(
        "## Observability\n"
        "Live runtime metrics collected via Redis. Click Refresh to update."
    )
    metrics_display = gr.Markdown("*Click Refresh to load metrics.*")
    refresh_btn = gr.Button("Refresh Metrics", variant="secondary")
    refresh_btn.click(fn=_fetch_metrics, outputs=[metrics_display])


def main() -> None:
    """Launch the Gradio app."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    app = create_app()
    app.launch(theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
