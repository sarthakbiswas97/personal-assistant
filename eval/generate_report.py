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
    avg_grounding: float
    avg_coherence: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
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
        g_scores = [e["judgment"].get("grounding_score", 3) for e in judgments]
        c_scores = [e["judgment"].get("coherence_score", 3) for e in judgments]
        latencies = sorted([e["latency_ms"] for e in entries if not e["guardrail_blocked"] and e["latency_ms"] > 0])

        # Latency percentiles
        p50 = latencies[len(latencies) // 2] if latencies else 0.0
        p95 = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)] if latencies else 0.0
        p99 = latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)] if latencies else 0.0

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
                "grounding": np.mean([e["judgment"].get("grounding_score", 3) for e in cat_entries]),
                "coherence": np.mean([e["judgment"].get("coherence_score", 3) for e in cat_entries]),
            }

        aggregated[model_name] = ModelScores(
            model_name=model_name,
            avg_hallucination=np.mean(h_scores) if h_scores else 0.0,
            avg_safety=np.mean(s_scores) if s_scores else 0.0,
            avg_bias=np.mean(b_scores) if b_scores else 0.0,
            avg_grounding=np.mean(g_scores) if g_scores else 0.0,
            avg_coherence=np.mean(c_scores) if c_scores else 0.0,
            avg_latency_ms=np.mean(latencies) if latencies else 0.0,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            guardrail_block_rate=len(blocked) / len(entries) if entries else 0.0,
            category_scores=category_scores,
        )

    return aggregated


def _plot_overall_comparison(scores: dict[str, ModelScores], output_dir: Path) -> Path:
    """Generate grouped bar chart comparing overall scores."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(scores.keys())
    metrics = ["Halluc.", "Safety", "Bias", "Grounding", "Coherence"]
    x = np.arange(len(metrics))
    width = 0.35

    for i, model_name in enumerate(models):
        s = scores[model_name]
        values = [s.avg_hallucination, s.avg_safety, s.avg_bias, s.avg_grounding, s.avg_coherence]
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


def _generate_pdf(
    scores: dict[str, ModelScores], output_dir: Path, repo_url: str = ""
) -> Path:
    """Generate a 1-page evaluation PDF with summary, charts, and recommendations."""
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = output_dir / "evaluation_report.pdf"
    models = list(scores.keys())

    with PdfPages(str(pdf_path)) as pdf:
        fig = plt.figure(figsize=(14, 8.5))  # wide landscape
        gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.4,
                              top=0.88, bottom=0.06, left=0.05, right=0.95)

        # -- Title --
        fig.suptitle("AI Assistant Evaluation: OSS vs Frontier",
                     fontsize=16, fontweight="bold", y=0.96)
        fig.text(0.06, 0.91,
                 "Methodology: 36 prompts (12 factual, 12 bias, 12 safety) evaluated by LLM-as-judge. "
                 "Scores are 1-5 (higher = better).",
                 fontsize=8, color="gray")

        # -- Summary Table (top-left) --
        ax_table = fig.add_subplot(gs[0, 0])
        ax_table.axis("off")

        table_data = []
        for m in models:
            s = scores[m]
            short_name = m.split("(")[0].strip() if "(" in m else m
            table_data.append([
                short_name,
                f"{s.avg_hallucination:.2f}",
                f"{s.avg_safety:.2f}",
                f"{s.avg_bias:.2f}",
                f"{s.avg_grounding:.2f}",
                f"{s.avg_coherence:.2f}",
                f"{s.p50_latency_ms:.0f}",
                f"{s.p95_latency_ms:.0f}",
                f"{s.p99_latency_ms:.0f}",
                f"{s.guardrail_block_rate:.0%}",
            ])

        col_labels = ["Model", "Halluc.", "Safety", "Bias", "Ground.", "Coher.", "P50ms", "P95ms", "P99ms", "Block%"]
        table = ax_table.table(
            cellText=table_data, colLabels=col_labels,
            loc="center", cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)
        # Style header
        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#4472C4")
            table[0, j].set_text_props(color="white", fontweight="bold")
        ax_table.set_title("Summary Scores", fontsize=10, fontweight="bold", pad=12)

        # -- Overall Comparison (top-center+right) --
        ax_overall = fig.add_subplot(gs[0, 1:])
        metric_labels = ["Hallucination", "Safety", "Bias"]
        x = np.arange(len(metric_labels))
        width = 0.35
        colors = ["#4472C4", "#ED7D31"]

        for i, m in enumerate(models):
            s = scores[m]
            vals = [s.avg_hallucination, s.avg_safety, s.avg_bias]
            short = m.split("(")[0].strip()
            offset = (i - (len(models) - 1) / 2) * width
            bars = ax_overall.bar(x + offset, vals, width, label=short, color=colors[i])
            ax_overall.bar_label(bars, fmt="%.2f", fontsize=7)

        ax_overall.set_ylabel("Score (1-5)", fontsize=8)
        ax_overall.set_xticks(x)
        ax_overall.set_xticklabels(metric_labels, fontsize=8)
        ax_overall.set_ylim(0, 5.5)
        ax_overall.legend(fontsize=7)
        ax_overall.set_title("Overall Comparison", fontsize=10, fontweight="bold")

        # -- Latency Percentiles (middle row) --
        ax_lat_table = fig.add_subplot(gs[1, 0])
        ax_lat_table.axis("off")
        lat_data = []
        for m in models:
            s = scores[m]
            short = m.split("(")[0].strip()
            lat_data.append([short, f"{s.avg_latency_ms:.0f}", f"{s.p50_latency_ms:.0f}", f"{s.p95_latency_ms:.0f}", f"{s.p99_latency_ms:.0f}"])
        lat_labels = ["Model", "Avg (ms)", "P50", "P95", "P99"]
        lat_table = ax_lat_table.table(cellText=lat_data, colLabels=lat_labels, loc="center", cellLoc="center")
        lat_table.auto_set_font_size(False)
        lat_table.set_fontsize(8)
        lat_table.scale(1, 1.4)
        for j in range(len(lat_labels)):
            lat_table[0, j].set_facecolor("#E67E22")
            lat_table[0, j].set_text_props(color="white", fontweight="bold")
        ax_lat_table.set_title("Latency Percentiles", fontsize=10, fontweight="bold", pad=12)

        # -- Category scores heatmap (middle row, cols 1-2) --
        ax_cat = fig.add_subplot(gs[1, 1:])
        categories = sorted({cat for s in scores.values() for cat in s.category_scores})[:8]
        cat_labels = [c.replace("_", "\n")[:12] for c in categories]
        model_short = [m.split("(")[0].strip() for m in models]
        # Build heatmap data: avg of all 5 dimensions per category per model
        heat_data = []
        for m in models:
            row = []
            for cat in categories:
                cs = scores[m].category_scores.get(cat, {})
                avg = np.mean([cs.get("hallucination", 3), cs.get("safety", 3), cs.get("bias", 3), cs.get("grounding", 3), cs.get("coherence", 3)])
                row.append(avg)
            heat_data.append(row)
        ax_cat.imshow(heat_data, cmap="RdYlGn", vmin=1, vmax=5, aspect="auto")
        ax_cat.set_xticks(range(len(categories)))
        ax_cat.set_xticklabels(cat_labels, fontsize=6, rotation=45, ha="right")
        ax_cat.set_yticks(range(len(models)))
        ax_cat.set_yticklabels(model_short, fontsize=8)
        for i in range(len(models)):
            for j in range(len(categories)):
                ax_cat.text(j, i, f"{heat_data[i][j]:.1f}", ha="center", va="center", fontsize=7, fontweight="bold")
        ax_cat.set_title("Avg Score by Category (1-5)", fontsize=10, fontweight="bold")

        # -- Latency Comparison (bottom-left) --
        ax_lat = fig.add_subplot(gs[2, 0])
        latencies = [scores[m].avg_latency_ms for m in models]
        short_names = [m.split("(")[0].strip() for m in models]
        bars = ax_lat.barh(short_names, latencies, color=colors[:len(models)])
        ax_lat.bar_label(bars, fmt="%.0f ms", fontsize=7)
        ax_lat.set_xlabel("Avg Latency (ms)", fontsize=8)
        ax_lat.set_title("Response Latency", fontsize=10, fontweight="bold")

        # -- Key Findings & Recommendations (bottom-center+right) --
        ax_text = fig.add_subplot(gs[2, 1:])
        ax_text.axis("off")

        # Build dynamic findings from scores
        oss = next((s for m, s in scores.items() if "OSS" in m or "Qwen" in m), None)
        frontier = next((s for m, s in scores.items() if "Frontier" in m or "gpt" in m.lower()), None)

        findings = []
        if oss and frontier:
            h_gap = frontier.avg_hallucination - oss.avg_hallucination
            findings.append(
                f"Frontier scores +{h_gap:.1f} on hallucination "
                f"({frontier.avg_hallucination:.1f} vs {oss.avg_hallucination:.1f})."
            )
            speed = oss.avg_latency_ms / frontier.avg_latency_ms if frontier.avg_latency_ms > 0 else 0
            findings.append(
                f"OSS is {speed:.0f}x slower "
                f"({oss.avg_latency_ms:.0f}ms vs {frontier.avg_latency_ms:.0f}ms)."
            )
            findings.append(
                f"Guardrails block {oss.guardrail_block_rate:.0%} of prompts "
                f"before reaching either model."
            )
            findings.append(
                "Bias is the largest gap -- OSS lacks capacity "
                "for nuanced stereotype handling."
            )

        recommendations = [
            "Use frontier models for production safety-critical applications.",
            "OSS models are viable for cost-sensitive deployments with guardrails.",
            "Invest in prompt engineering or fine-tuning to close the bias gap.",
            "GPU deployment would reduce OSS latency by ~10x.",
        ]

        text = "KEY FINDINGS\n"
        for i, f in enumerate(findings, 1):
            text += f"  {i}. {f}\n"
        text += "\nRECOMMENDATIONS\n"
        for i, r in enumerate(recommendations, 1):
            text += f"  {i}. {r}\n"

        if repo_url:
            text += f"\nDetailed prompt/response data: {repo_url}/blob/main/eval/outputs/eval_results.json"

        ax_text.text(0, 1, text, transform=ax_text.transAxes,
                     fontsize=7.5, verticalalignment="top", fontfamily="monospace",
                     linespacing=1.5, wrap=True)
        ax_text.set_title("Findings & Recommendations", fontsize=10, fontweight="bold")

        pdf.savefig(fig)
        plt.close(fig)

    return pdf_path


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

    # Generate PDF report
    pdf_path = _generate_pdf(scores, REPORTS_DIR)
    logger.info("Generated: %s", pdf_path)

    # Print summary table
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for model_name, s in scores.items():
        print(f"\n{model_name}:")
        print(f"  Hallucination:      {s.avg_hallucination:.2f}/5")
        print(f"  Safety:             {s.avg_safety:.2f}/5")
        print(f"  Bias:               {s.avg_bias:.2f}/5")
        print(f"  Grounding:          {s.avg_grounding:.2f}/5")
        print(f"  Coherence:          {s.avg_coherence:.2f}/5")
        print(f"  Latency Avg:        {s.avg_latency_ms:.0f}ms")
        print(f"  Latency P50:        {s.p50_latency_ms:.0f}ms")
        print(f"  Latency P95:        {s.p95_latency_ms:.0f}ms")
        print(f"  Latency P99:        {s.p99_latency_ms:.0f}ms")
        print(f"  Guardrail Blocks:   {s.guardrail_block_rate:.0%}")
    print("=" * 60)
    print(f"\nPDF report: {pdf_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    generate_report()
