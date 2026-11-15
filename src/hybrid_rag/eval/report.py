"""Markdown report generator for evaluation results."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from hybrid_rag.models import EvalResult


def generate_markdown_report(
    summary_data: list[dict[str, Any]],
    bootstrap_comparisons: dict[str, dict[str, Any]] | None = None,
    chunking_stats: list[dict[str, Any]] | None = None,
    latency_stats: dict[str, dict[str, int]] | None = None,
    judge_stats: dict[str, Any] | None = None,
) -> str:
    """Generate a GitHub Flavored Markdown evaluation report."""
    today = datetime.now(UTC).strftime("%Y-%m-%d")

    lines: list[str] = [
        f"# Evaluation Report — Hybrid RAG ({today})",
        "",
        "> Benchmark results comparing dense, BM25, hybrid (RRF), and hybrid + rerank retrieval pipelines.",
        "",
        "## Summary Metrics",
        "",
        "| Mode | Hit@5 | Recall@5 | MRR | nDCG@10 |",
        "|---|---|---|---|---|",
    ]

    for row in summary_data:
        m = row.get("mode", "")
        h5 = f"{row.get('hit_rate_at_5', 0.0):.4f}"
        rec = f"{row.get('recall_at_5', 0.0):.4f}"
        mrr = f"{row.get('mrr', 0.0):.4f}"
        ndcg = f"{row.get('ndcg_at_10', 0.0):.4f}"
        lines.append(f"| {m} | {h5} | {rec} | {mrr} | {ndcg} |")

    if bootstrap_comparisons:
        lines.append("")
        lines.append("## Paired Bootstrap Comparisons (vs Dense Only, 95% CI)")
        lines.append("")
        lines.append("| Mode Comparison | Metric | Diff Mean | 95% CI | p-value | Significant |")
        lines.append("|---|---|---|---|---|---|")

        for key, comp in bootstrap_comparisons.items():
            diff = comp.get("diff_mean", 0.0)
            ci_low = comp.get("ci_lower", 0.0)
            ci_high = comp.get("ci_upper", 0.0)
            pval = comp.get("p_value", 1.0)
            sig = "✓ Yes" if comp.get("significant", False) else "✗ No"
            lines.append(
                f"| {key} | MRR | {diff:+.4f} | [{ci_low:+.4f}, {ci_high:+.4f}] | {pval:.4f} | {sig} |"
            )

    if chunking_stats:
        lines.append("")
        lines.append("## Chunking Strategy Comparison")
        lines.append("")
        lines.append("| Strategy | Chunks | Avg Chars | Dedup Dropped |")
        lines.append("|---|---|---|---|")
        for cs in chunking_stats:
            strat = cs.get("strategy", "")
            cnt = cs.get("total_chunks", 0)
            avg = cs.get("avg_chunk_size", 0)
            drp = cs.get("dedup_dropped", 0)
            lines.append(f"| {strat} | {cnt} | {avg} | {drp} |")

    if latency_stats:
        lines.append("")
        lines.append("## Latency Profile (ms)")
        lines.append("")
        lines.append("| Mode | p50 (ms) | p95 (ms) |")
        lines.append("|---|---|---|")
        for mode, lat in latency_stats.items():
            p50 = lat.get("p50", 0)
            p95 = lat.get("p95", 0)
            lines.append(f"| {mode} | {p50} | {p95} |")

    if judge_stats:
        lines.append("")
        lines.append("## LLM Judge Calibration")
        lines.append("")
        lines.append(f"- **Cohen's Kappa**: {judge_stats.get('kappa', 0.0):.4f}")
        lines.append(f"- **Agreement Rate**: {judge_stats.get('agreement_rate', 0.0) * 100:.1f}%")
        lines.append(f"- **Sample Size**: {judge_stats.get('n_samples', 0)}")

    lines.append("")
    lines.append("---")
    lines.append("*Report generated automatically by `hybrid_rag.eval.run_eval`.*")

    return "\n".join(lines)


def save_eval_report(
    report_text: str,
    results: list[EvalResult],
    out_markdown: Path,
) -> None:
    """Save markdown report and accompanying CSV to disk."""
    out_markdown.parent.mkdir(parents=True, exist_ok=True)
    out_markdown.write_text(report_text, encoding="utf-8")

    # Save CSV
    csv_path = out_markdown.with_suffix(".csv")
    records = [r.model_dump() for r in results]
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
