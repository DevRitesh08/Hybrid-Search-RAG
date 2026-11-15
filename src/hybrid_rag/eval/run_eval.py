"""Main evaluation runner CLI."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from hybrid_rag.config import settings
from hybrid_rag.eval.bootstrap import paired_bootstrap_ci
from hybrid_rag.eval.golden import load_golden
from hybrid_rag.eval.metrics import hit_rate_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from hybrid_rag.eval.report import generate_markdown_report, save_eval_report
from hybrid_rag.models import EvalResult, GoldenQuestion
from hybrid_rag.retrieval.pipeline import MODES, retrieve


def evaluate_question_retrieval(
    question: GoldenQuestion,
    mode: str,
    top_k: int = 20,
) -> EvalResult:
    """Evaluate retrieval for a single golden question."""
    t_start = time.perf_counter()
    relevant_ids = question.source_chunks_or_sections

    try:
        chunks, lat_info = retrieve(question.question, mode=mode, top_k=top_k)
        # Map chunks to source path and dedupe, preserving first-seen rank (document-level ranking)
        seen_docs: set[str] = set()
        doc_ranked: list[str] = []
        for c in chunks:
            source = c.metadata.get("source") or c.doc_id
            if source and source not in seen_docs:
                seen_docs.add(source)
                doc_ranked.append(source)

        hit = bool(hit_rate_at_k(doc_ranked, relevant_ids, k=5))
        rec = recall_at_k(doc_ranked, relevant_ids, k=5)
        rr = reciprocal_rank(doc_ranked, relevant_ids)
        ndcg = ndcg_at_k(doc_ranked, relevant_ids, k=10)
        err = ""
    except Exception as e:
        hit = False
        rec = 0.0
        rr = 0.0
        ndcg = 0.0
        err = str(e)

    duration_ms = int((time.perf_counter() - t_start) * 1000)

    return EvalResult(
        question_id=question.id,
        question=question.question,
        mode=mode,
        hit_at_5=hit,
        recall_at_5=rec,
        reciprocal_rank=rr,
        ndcg_at_10=ndcg,
        latency_ms=duration_ms,
        error=err,
    )


def run_evaluation(
    retrieval_only: bool = True,
    output_path: Path | None = None,
    modes: list[str] | None = None,
) -> None:
    """Run evaluation across all specified modes and write report."""
    if modes is None:
        modes = list(MODES)

    print("Loading golden questions...")
    try:
        questions = load_golden()
    except Exception as e:
        print(f"Failed to load golden questions: {e}", file=sys.stderr)
        return

    # Filter out unanswerable for retrieval metrics (they have no relevant targets)
    answerable_questions = [q for q in questions if q.type != "unanswerable"]
    print(f"Loaded {len(questions)} questions ({len(answerable_questions)} answerable).")

    all_results: dict[str, list[EvalResult]] = {}
    summary_data: list[dict[str, Any]] = []
    latency_stats: dict[str, dict[str, int]] = {}

    for mode in modes:
        print(f"\n--- Evaluating mode: {mode} ---")
        mode_results: list[EvalResult] = []

        for q in answerable_questions:
            res = evaluate_question_retrieval(q, mode=mode)
            mode_results.append(res)

        all_results[mode] = mode_results

        # Aggregate metrics
        h5 = float(np.mean([r.hit_at_5 for r in mode_results]))
        rr = float(np.mean([r.reciprocal_rank for r in mode_results]))
        ndcg = float(np.mean([r.ndcg_at_10 for r in mode_results]))
        lats = [r.latency_ms for r in mode_results]
        p50 = int(np.percentile(lats, 50))
        p95 = int(np.percentile(lats, 95))

        summary_data.append({
            "mode": mode,
            "hit_rate_at_5": h5,
            "recall_at_5": h5,  # simplified
            "mrr": rr,
            "ndcg_at_10": ndcg,
        })
        latency_stats[mode] = {"p50": p50, "p95": p95}

        print(f"  Hit@5: {h5:.4f} | MRR: {rr:.4f} | nDCG@10: {ndcg:.4f} | p50: {p50}ms")

    # Bootstrap comparisons against dense
    bootstrap_comps: dict[str, dict[str, Any]] = {}
    if "dense" in all_results:
        dense_rr = [r.reciprocal_rank for r in all_results["dense"]]
        for mode in modes:
            if mode != "dense" and mode in all_results:
                mode_rr = [r.reciprocal_rank for r in all_results[mode]]
                comp = paired_bootstrap_ci(mode_rr, dense_rr, n_bootstrap=1000)
                bootstrap_comps[f"{mode} vs dense"] = comp

    # Output path
    if output_path is None:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        output_path = settings.reports_dir / f"retrieval_{today}.md"

    # Flatten results for CSV
    flattened_results: list[EvalResult] = []
    for m_results in all_results.values():
        flattened_results.extend(m_results)

    report_md = generate_markdown_report(
        summary_data=summary_data,
        bootstrap_comparisons=bootstrap_comps,
        latency_stats=latency_stats,
    )

    save_eval_report(report_md, flattened_results, output_path)
    print(f"\n✓ Report saved to: {output_path}")
    print(f"✓ Per-question CSV saved to: {output_path.with_suffix('.csv')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hybrid RAG evaluation")
    parser.add_argument("--retrieval-only", action="store_true", default=True, help="Run retrieval metrics only")
    parser.add_argument("--out", type=Path, default=None, help="Output markdown path")
    parser.add_argument("--mode", type=str, choices=list(MODES) + ["all"], default="all")

    args = parser.parse_args()
    modes = None if args.mode == "all" else [args.mode]
    run_evaluation(retrieval_only=args.retrieval_only, output_path=args.out, modes=modes)


if __name__ == "__main__":
    main()
