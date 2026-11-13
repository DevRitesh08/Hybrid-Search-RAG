"""Command-line interface for Hybrid RAG."""

from __future__ import annotations

import argparse
import sys

from hybrid_rag.config import settings
from hybrid_rag.generation.answer_pipeline import run_generation_pipeline
from hybrid_rag.retrieval.pipeline import retrieve


def cmd_smoke_test(args: argparse.Namespace) -> None:
    """Run a quick smoke test on the retrieval and generation pipeline."""
    test_question = "What is the AutoModel class in Transformers?"
    print(f"Smoke test question: '{test_question}'")
    print(f"Retrieval mode: {settings.retrieval_mode}")

    try:
        chunks, lat = retrieve(test_question, mode=settings.retrieval_mode, top_k=5)
        print(f"✓ Retrieved {len(chunks)} chunks in {lat.get('retrieve_ms', 0)}ms.")
        if chunks:
            print(f"  Top chunk ID: {chunks[0].chunk_id} (Score: {chunks[0].score:.4f})")
            print(f"  Top source: {chunks[0].metadata.get('source', 'unknown')}")
        else:
            print("⚠ No chunks retrieved (is the index seeded? Run `make seed`)")

        response = run_generation_pipeline(test_question, chunks, mode=settings.retrieval_mode)
        print(f"\nResponse Refused: {response.refused}")
        print(f"Composite Confidence: {response.confidence.composite:.2f}")
        print(f"Answer snippet: {response.answer[:200]}...")
        print("\n✓ Smoke test completed successfully.")
    except Exception as e:
        print(f"✗ Smoke test failed: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_ask(args: argparse.Namespace) -> None:
    """Ask a question from the command line."""
    question = args.question
    mode = args.mode or settings.retrieval_mode
    top_k = args.top_k or settings.rerank_top_k

    print(f"Querying [{mode}]: '{question}'")
    chunks, lat = retrieve(question, mode=mode, top_k=top_k)
    response = run_generation_pipeline(question, chunks, mode=mode)

    print("\n" + "=" * 60)
    if response.refused:
        print(f"[REFUSED] {response.refusal_reason}\n")
    print(response.answer)
    print("=" * 60)

    if response.citations:
        print(f"\nCitations ({len(response.citations)}):")
        for c in response.citations:
            verified = "✓" if c.verified else "✗"
            print(f"  [{c.n}] {verified} {c.source} ({c.heading})")

    print(f"\nConfidence: composite={response.confidence.composite:.2f} "
          f"(retrieval={response.confidence.retrieval:.2f}, "
          f"coverage={response.confidence.coverage:.2f}, "
          f"completeness={response.confidence.completeness:.2f})")
    print(f"Latency: {response.latency.total_ms}ms")


def cmd_seed(args: argparse.Namespace) -> None:
    """Download corpus, normalize, and build index."""
    import subprocess

    print("Step 1/3: Downloading corpus...")
    subprocess.run([sys.executable, "scripts/download_corpus.py"], check=True)

    print("\nStep 2/3: Normalizing raw documents...")
    from hybrid_rag.ingest.normalize import main as normalize_main
    normalize_main()

    print(f"\nStep 3/3: Building indexes with strategy: {args.strategy}...")
    from hybrid_rag.index.build import build_indexes
    build_indexes(strategy=args.strategy)

    print("\n✓ Seed pipeline completed successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # smoke-test
    p_smoke = subparsers.add_parser("smoke-test", help="Run quick smoke test")
    p_smoke.set_defaults(func=cmd_smoke_test)

    # seed
    p_seed = subparsers.add_parser("seed", help="Download, normalize, and index corpus")
    p_seed.add_argument("--strategy", default="structural", choices=["fixed", "structural", "semantic"])
    p_seed.set_defaults(func=cmd_seed)

    # ask
    p_ask = subparsers.add_parser("ask", help="Ask a question")
    p_ask.add_argument("question", type=str, help="Question text")
    p_ask.add_argument("--mode", default=None, choices=["dense", "bm25", "hybrid", "hybrid_rerank"])
    p_ask.add_argument("--top-k", type=int, default=None)
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
