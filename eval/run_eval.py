"""Evaluation runner: runs both models against all prompt sets and judges responses."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from eval.judge import JudgmentScore, LLMJudge
from src.config import load_settings
from src.guardrails import check_input
from src.models.base import BaseModel, Message

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"
OUTPUTS_DIR = Path(__file__).parent / "outputs"

SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI assistant. "
    "Answer questions clearly and concisely. "
    "If you don't know something, say so rather than guessing."
)


@dataclass(frozen=True)
class EvalResult:
    """Immutable result of a single evaluation run."""

    prompt_id: str
    model_name: str
    category: str
    prompt: str
    response: str
    latency_ms: float
    guardrail_blocked: bool
    judgment: JudgmentScore | None = None


async def _run_single_prompt(
    model: BaseModel,
    model_name: str,
    prompt_data: dict,
    category: str,
) -> EvalResult:
    """Run a single prompt against a model and measure latency."""
    prompt = prompt_data["prompt"]

    # Check guardrails first
    input_check = check_input(prompt)
    if input_check.is_blocked:
        return EvalResult(
            prompt_id=prompt_data["id"],
            model_name=model_name,
            category=category,
            prompt=prompt,
            response=f"[GUARDRAIL BLOCKED] {input_check.reason}",
            latency_ms=0.0,
            guardrail_blocked=True,
        )

    messages = [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=prompt),
    ]

    start = time.perf_counter()
    response = await model.generate(messages)
    latency_ms = (time.perf_counter() - start) * 1000

    return EvalResult(
        prompt_id=prompt_data["id"],
        model_name=model_name,
        category=category,
        prompt=prompt,
        response=response,
        latency_ms=latency_ms,
        guardrail_blocked=False,
    )


async def run_evaluation(
    models: dict[str, BaseModel],
    judge: LLMJudge,
) -> list[EvalResult]:
    """Run full stress test evaluation across all models.

    Handles single-turn and multi-turn (context) prompts.
    Uses the stress_test.json prompt set (50 prompts).
    """
    stress_file = PROMPTS_DIR / "stress_test.json"
    prompts = json.loads(stress_file.read_text())
    logger.info("Running %d stress test prompts", len(prompts))

    all_results: list[EvalResult] = []

    for prompt_data in prompts:
        category = prompt_data.get("category", "unknown")

        for model_name, model in models.items():
            # Run the primary prompt
            result = await _run_single_prompt(
                model, model_name, prompt_data, category
            )

            # For multi-turn context prompts, run follow-up
            if "follow_up" in prompt_data and not result.guardrail_blocked:
                follow_up_data = {
                    "id": prompt_data["id"].replace("a", "b"),
                    "prompt": prompt_data["follow_up"],
                }
                # Build messages with context from turn 1
                messages = [
                    Message(role="system", content=SYSTEM_PROMPT),
                    Message(role="user", content=prompt_data["prompt"]),
                    Message(role="assistant", content=result.response),
                    Message(role="user", content=prompt_data["follow_up"]),
                ]
                start = time.perf_counter()
                follow_response = await model.generate(messages)
                follow_latency = (time.perf_counter() - start) * 1000

                follow_result = EvalResult(
                    prompt_id=follow_up_data["id"],
                    model_name=model_name,
                    category="context_follow_up",
                    prompt=prompt_data["follow_up"],
                    response=follow_response,
                    latency_ms=follow_latency,
                    guardrail_blocked=False,
                )

                # Judge follow-up with context
                expected_fu = prompt_data.get("follow_up_expected", "")
                judgment_fu = await judge.judge(
                    prompt_id=follow_up_data["id"],
                    model_name=model_name,
                    category="context_follow_up",
                    prompt=prompt_data["follow_up"],
                    response=follow_response,
                    expected=f"Context: user said '{prompt_data['prompt']}'. Expected: {expected_fu}",
                )
                follow_result = EvalResult(
                    prompt_id=follow_result.prompt_id,
                    model_name=follow_result.model_name,
                    category=follow_result.category,
                    prompt=follow_result.prompt,
                    response=follow_result.response,
                    latency_ms=follow_result.latency_ms,
                    guardrail_blocked=False,
                    judgment=judgment_fu,
                )
                all_results.append(follow_result)
                logger.info(
                    "[context_follow_up] %s | %s | latency=%.0fms",
                    model_name, follow_up_data["id"], follow_latency,
                )

            # Judge primary prompt
            expected = prompt_data.get("expected", "")
            judgment = await judge.judge(
                prompt_id=result.prompt_id,
                model_name=model_name,
                category=category,
                prompt=result.prompt,
                response=result.response,
                expected=expected,
            )

            result = EvalResult(
                prompt_id=result.prompt_id,
                model_name=result.model_name,
                category=result.category,
                prompt=result.prompt,
                response=result.response,
                latency_ms=result.latency_ms,
                guardrail_blocked=result.guardrail_blocked,
                judgment=judgment,
            )

            all_results.append(result)
            logger.info(
                "[%s] %s | %s | latency=%.0fms",
                category, model_name, result.prompt_id, result.latency_ms,
            )

    return all_results


def save_results(results: list[EvalResult], filename: str = "eval_results.json") -> Path:
    """Save evaluation results to JSON."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / filename

    serializable = []
    for r in results:
        d = asdict(r)
        serializable.append(d)

    output_path.write_text(json.dumps(serializable, indent=2))
    logger.info("Saved %d results to %s", len(results), output_path)
    return output_path


async def main() -> None:
    """Run the full evaluation pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    settings = load_settings()

    if not settings.openai_api_key:
        logger.error("OPENAI_API_KEY is required for evaluation")
        return

    # Initialize models
    from src.models.frontier_model import FrontierModel
    from src.models.oss_model import OSSModel

    frontier_label = f"Frontier ({settings.frontier_model_name})"
    models: dict[str, BaseModel] = {
        "OSS (Qwen2.5-0.5B)": OSSModel(model_name=settings.oss_model_name),
        frontier_label: FrontierModel(
            api_key=settings.openai_api_key,
            model_name=settings.frontier_model_name,
        ),
    }

    judge = LLMJudge(api_key=settings.openai_api_key)

    try:
        results = await run_evaluation(models, judge)
        save_results(results)
        logger.info("Evaluation complete: %d total results", len(results))
    finally:
        for model in models.values():
            await model.cleanup()
        await judge.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
