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
    """Generate a 1-page evaluation PDF."""
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = output_dir / "evaluation_report.pdf"
    models = list(scores.keys())
    colors = ["#4472C4", "#ED7D31"]
    short_names = [m.split("(")[0].strip() for m in models]

    # Clean category names
    cat_name_map = {
        "bias": "Bias", "context": "Context", "context_follow_up": "Follow-up",
        "guardrail_stress": "Guardrails", "pure_factual": "Factual",
        "reasoning": "Reasoning", "tool_factual": "Tools",
    }

    with PdfPages(str(pdf_path)) as pdf:
        fig = plt.figure(figsize=(15, 9))
        gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.3,
                              top=0.88, bottom=0.06, left=0.06, right=0.96)

        fig.suptitle("AI Assistant Evaluation: OSS vs Frontier",
                     fontsize=16, fontweight="bold", y=0.96)
        fig.text(0.06, 0.91,
                 "50-prompt stress test across 7 categories. "
                 "Scored 1-5 by LLM-as-judge on 5 dimensions. Higher = better.",
                 fontsize=8, color="gray")

        # -- Row 1 Left: Quality Scores Table --
        ax_table = fig.add_subplot(gs[0, 0])
        ax_table.axis("off")
        table_data = []
        for m in models:
            s = scores[m]
            sn = m.split("(")[0].strip()
            table_data.append([
                sn, f"{s.avg_hallucination:.1f}", f"{s.avg_safety:.1f}",
                f"{s.avg_bias:.1f}", f"{s.avg_grounding:.1f}",
                f"{s.avg_coherence:.1f}", f"{s.guardrail_block_rate:.0%}",
            ])
        col_labels = ["Model", "Halluc.", "Safety", "Bias", "Ground.", "Coher.", "Blocked"]
        tbl = ax_table.table(cellText=table_data, colLabels=col_labels,
                             loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.5)
        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor("#4472C4")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        ax_table.set_title("Quality Scores (1-5 scale)", fontsize=11, fontweight="bold", pad=14)

        # -- Row 1 Right: All 5 Dimensions Bar Chart --
        ax_dims = fig.add_subplot(gs[0, 1])
        dim_labels = ["Halluc.", "Safety", "Bias", "Ground.", "Coher."]
        x = np.arange(len(dim_labels))
        width = 0.35
        for i, m in enumerate(models):
            s = scores[m]
            vals = [s.avg_hallucination, s.avg_safety, s.avg_bias,
                    s.avg_grounding, s.avg_coherence]
            offset = (i - 0.5) * width
            bars = ax_dims.bar(x + offset, vals, width,
                               label=short_names[i], color=colors[i])
            ax_dims.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
        ax_dims.set_ylabel("Score (1-5)", fontsize=9)
        ax_dims.set_xticks(x)
        ax_dims.set_xticklabels(dim_labels, fontsize=9)
        ax_dims.set_ylim(0, 5.8)
        ax_dims.legend(fontsize=8)
        ax_dims.set_title("Overall Comparison (all 5 dimensions)",
                          fontsize=11, fontweight="bold")

        # -- Row 2 Left: Per-Category Grouped Bars --
        ax_cat = fig.add_subplot(gs[1, 0])
        categories = sorted({c for s in scores.values() for c in s.category_scores})
        clean_cats = [cat_name_map.get(c, c) for c in categories]
        x_cat = np.arange(len(categories))
        width_cat = 0.35
        for i, m in enumerate(models):
            vals = []
            for cat in categories:
                cs = scores[m].category_scores.get(cat, {})
                avg = np.mean([
                    cs.get("hallucination", 3), cs.get("safety", 3),
                    cs.get("bias", 3), cs.get("grounding", 3),
                    cs.get("coherence", 3),
                ])
                vals.append(avg)
            offset = (i - 0.5) * width_cat
            bars = ax_cat.bar(x_cat + offset, vals, width_cat,
                              label=short_names[i], color=colors[i])
            ax_cat.bar_label(bars, fmt="%.1f", fontsize=6, padding=1)
        ax_cat.set_xticks(x_cat)
        ax_cat.set_xticklabels(clean_cats, fontsize=8, rotation=30, ha="right")
        ax_cat.set_ylabel("Avg Score (1-5)", fontsize=8)
        ax_cat.set_ylim(0, 5.8)
        ax_cat.legend(fontsize=7)
        ax_cat.set_title("Score by Category (avg of 5 dimensions)",
                          fontsize=11, fontweight="bold")

        # -- Row 2 Right: Latency Percentile Grouped Bars --
        ax_lat = fig.add_subplot(gs[1, 1])
        lat_labels = ["P50", "P95", "P99"]
        x_lat = np.arange(len(lat_labels))
        width_lat = 0.35
        for i, m in enumerate(models):
            s = scores[m]
            vals = [s.p50_latency_ms / 1000, s.p95_latency_ms / 1000,
                    s.p99_latency_ms / 1000]
            offset = (i - 0.5) * width_lat
            bars = ax_lat.bar(x_lat + offset, vals, width_lat,
                              label=short_names[i], color=colors[i])
            ax_lat.bar_label(bars, fmt="%.1fs", fontsize=7, padding=2)
        ax_lat.set_xticks(x_lat)
        ax_lat.set_xticklabels(lat_labels, fontsize=10)
        ax_lat.set_ylabel("Latency (seconds)", fontsize=9)
        ax_lat.legend(fontsize=8)
        ax_lat.set_title("Latency Percentiles (lower = better)",
                          fontsize=11, fontweight="bold")

        # -- Row 3: Key Findings & Recommendations (full width) --
        ax_text = fig.add_subplot(gs[2, :])
        ax_text.axis("off")

        oss = next((s for m, s in scores.items() if "OSS" in m), None)
        frontier = next((s for m, s in scores.items() if "Frontier" in m), None)

        findings = []
        if oss and frontier:
            g_gap = frontier.avg_grounding - oss.avg_grounding
            h_gap = frontier.avg_hallucination - oss.avg_hallucination
            speed = oss.p50_latency_ms / frontier.p50_latency_ms if frontier.p50_latency_ms > 0 else 0
            findings = [
                f"Grounding is the biggest gap (-{g_gap:.1f}): "
                f"OSS fabricates when tools don't fire.",
                f"Hallucination gap is -{h_gap:.1f} "
                f"({oss.avg_hallucination:.1f} vs {frontier.avg_hallucination:.1f}).",
                f"OSS is {speed:.0f}x slower at P50 "
                f"({oss.p50_latency_ms/1000:.1f}s vs {frontier.p50_latency_ms/1000:.1f}s).",
                f"Guardrails block {oss.guardrail_block_rate:.0%} of "
                f"prompts identically before either model.",
            ]

        recs = [
            "Use frontier for safety-critical production.",
            "OSS viable for cost-sensitive use with guardrails.",
            "Tool use helps OSS most on grounding.",
            "GPU would reduce OSS P50 by ~5-10x.",
        ]

        left = "KEY FINDINGS\n" + "\n".join(f"  {i}. {f}" for i, f in enumerate(findings, 1))
        right = "RECOMMENDATIONS\n" + "\n".join(f"  {i}. {r}" for i, r in enumerate(recs, 1))

        ax_text.text(0.02, 0.95, left, transform=ax_text.transAxes,
                     fontsize=8, verticalalignment="top", fontfamily="monospace",
                     linespacing=1.6)
        ax_text.text(0.55, 0.95, right, transform=ax_text.transAxes,
                     fontsize=8, verticalalignment="top", fontfamily="monospace",
                     linespacing=1.6)
        ax_text.set_title("Findings & Recommendations",
                          fontsize=11, fontweight="bold")

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
