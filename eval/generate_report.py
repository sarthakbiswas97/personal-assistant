"""Generate evaluation report with comparison infographics."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).parent / "outputs"
REPORTS_DIR = Path(__file__).parent / "reports"


@dataclass(frozen=True)
class ModelScores:
    """Aggregated scores for a single model."""

    model_name: str
    avg_hallucination: float
    avg_safety: float
    avg_bias: float
    avg_latency_ms: float
    guardrail_block_rate: float
    category_scores: dict[str, dict[str, float]]


def _aggregate_scores(results: list[dict]) -> dict[str, ModelScores]:
    """Aggregate per-model scores from raw evaluation results."""
    model_data: dict[str, list[dict]] = {}
    for r in results:
        name = r["model_name"]
        if name not in model_data:
            model_data[name] = []
        model_data[name].append(r)

    aggregated = {}
    for model_name, entries in model_data.items():
        judgments = [e for e in entries if e.get("judgment")]
        blocked = [e for e in entries if e["guardrail_blocked"]]

        h_scores = [e["judgment"]["hallucination_score"] for e in judgments]
        s_scores = [e["judgment"]["safety_score"] for e in judgments]
        b_scores = [e["judgment"]["bias_score"] for e in judgments]
        latencies = [e["latency_ms"] for e in entries if not e["guardrail_blocked"]]

        # Per-category breakdown
        categories: dict[str, list[dict]] = {}
        for e in judgments:
            cat = e["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(e)

        category_scores = {}
        for cat, cat_entries in categories.items():
            category_scores[cat] = {
                "hallucination": np.mean([e["judgment"]["hallucination_score"] for e in cat_entries]),
                "safety": np.mean([e["judgment"]["safety_score"] for e in cat_entries]),
                "bias": np.mean([e["judgment"]["bias_score"] for e in cat_entries]),
            }

        aggregated[model_name] = ModelScores(
            model_name=model_name,
            avg_hallucination=np.mean(h_scores) if h_scores else 0.0,
            avg_safety=np.mean(s_scores) if s_scores else 0.0,
            avg_bias=np.mean(b_scores) if b_scores else 0.0,
            avg_latency_ms=np.mean(latencies) if latencies else 0.0,
            guardrail_block_rate=len(blocked) / len(entries) if entries else 0.0,
            category_scores=category_scores,
        )

    return aggregated


def _plot_overall_comparison(scores: dict[str, ModelScores], output_dir: Path) -> Path:
    """Generate grouped bar chart comparing overall scores."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(scores.keys())
    metrics = ["Hallucination\n(higher=better)", "Safety\n(higher=better)", "Bias\n(higher=better)"]
    x = np.arange(len(metrics))
    width = 0.35

    for i, model_name in enumerate(models):
        s = scores[model_name]
        values = [s.avg_hallucination, s.avg_safety, s.avg_bias]
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=model_name)
        ax.bar_label(bars, fmt="%.2f", fontsize=9)

    ax.set_ylabel("Score (1-5)")
    ax.set_title("Overall Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 5.5)
    ax.legend()
    fig.tight_layout()

    path = output_dir / "overall_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_category_breakdown(scores: dict[str, ModelScores], output_dir: Path) -> Path:
    """Generate per-category radar/bar chart."""
    sns.set_theme(style="whitegrid")
    categories = set()
    for s in scores.values():
        categories.update(s.category_scores.keys())
    categories = sorted(categories)

    fig, axes = plt.subplots(1, len(categories), figsize=(6 * len(categories), 5))
    if len(categories) == 1:
        axes = [axes]

    models = list(scores.keys())

    for idx, cat in enumerate(categories):
        ax = axes[idx]
        metric_names = ["Hallucination", "Safety", "Bias"]
        x = np.arange(len(metric_names))
        width = 0.35

        for i, model_name in enumerate(models):
            cat_scores = scores[model_name].category_scores.get(cat, {})
            values = [
                cat_scores.get("hallucination", 0),
                cat_scores.get("safety", 0),
                cat_scores.get("bias", 0),
            ]
            offset = (i - (len(models) - 1) / 2) * width
            bars = ax.bar(x + offset, values, width, label=model_name)
            ax.bar_label(bars, fmt="%.1f", fontsize=8)

        ax.set_title(f"Category: {cat.title()}")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, fontsize=9)
        ax.set_ylim(0, 5.5)
        ax.set_ylabel("Score (1-5)")
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Per-Category Score Breakdown", fontsize=14, y=1.02)
    fig.tight_layout()

    path = output_dir / "category_breakdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_latency_comparison(scores: dict[str, ModelScores], output_dir: Path) -> Path:
    """Generate latency comparison bar chart."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    models = list(scores.keys())
    latencies = [scores[m].avg_latency_ms for m in models]

    bars = ax.bar(models, latencies, color=sns.color_palette("muted", len(models)))
    ax.bar_label(bars, fmt="%.0f ms", fontsize=10)

    ax.set_ylabel("Average Latency (ms)")
    ax.set_title("Response Latency Comparison")
    fig.tight_layout()

    path = output_dir / "latency_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_report(results_file: str = "eval_results.json") -> None:
    """Generate all infographics from evaluation results.

    Args:
        results_file: Name of the JSON results file in eval/outputs/.
    """
    results_path = OUTPUTS_DIR / results_file
    if not results_path.exists():
        logger.error("Results file not found: %s", results_path)
        return

    results = json.loads(results_path.read_text())
    scores = _aggregate_scores(results)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    overall_path = _plot_overall_comparison(scores, REPORTS_DIR)
    logger.info("Generated: %s", overall_path)

    category_path = _plot_category_breakdown(scores, REPORTS_DIR)
    logger.info("Generated: %s", category_path)

    latency_path = _plot_latency_comparison(scores, REPORTS_DIR)
    logger.info("Generated: %s", latency_path)

    # Print summary table
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for model_name, s in scores.items():
        print(f"\n{model_name}:")
        print(f"  Hallucination:      {s.avg_hallucination:.2f}/5")
        print(f"  Safety:             {s.avg_safety:.2f}/5")
        print(f"  Bias:               {s.avg_bias:.2f}/5")
        print(f"  Avg Latency:        {s.avg_latency_ms:.0f}ms")
        print(f"  Guardrail Blocks:   {s.guardrail_block_rate:.0%}")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    generate_report()
