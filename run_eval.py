
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_eval.evaluator import (
    RagasRagEvaluator,
    default_local_llm_and_embeddings,
    load_items_from_json,
)
from rag_eval.local_models import default_run_config


def main() -> int:
    p = argparse.ArgumentParser(
        description="Ragas evaluation: faithfulness, answer relevancy, context precision "
        "(via context_utilization), and hallucination flags.",
        epilog="Examples:  run_eval.py data.json report.json\n"
        "           run_eval.py data.json -o report.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input_json", type=Path, help="JSON file with items to evaluate")
    p.add_argument(
        "output_json",
        type=Path,
        nargs="?",
        default=None,
        help="Optional output file (same as -o). If omitted, print report to stdout.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write report JSON here (overrides positional output if both are given)",
    )
    p.add_argument(
        "--faithfulness-threshold",
        type=float,
        default=0.5,
        help="Below this faithfulness score, row is flagged as hallucination risk (default: 0.5)",
    )
    p.add_argument(
        "--relevancy-threshold",
        type=float,
        default=0.5,
        help="Warn when answer_relevancy is below this (default: 0.5)",
    )
    p.add_argument(
        "--context-precision-threshold",
        type=float,
        default=0.5,
        help="Warn when context utilization score is below this (default: 0.5)",
    )
    p.add_argument(
        "--ragas-timeout",
        type=int,
        default=600,
        metavar="SEC",
        help="Ragas per-step timeout in seconds (default: 600; local Ollama is slow)",
    )
    p.add_argument(
        "--ragas-max-workers",
        type=int,
        default=1,
        metavar="N",
        help="Parallel Ragas metric jobs (default: 1; use 1 for a single Ollama server)",
    )
    args = p.parse_args()

    out_path: Path | None = args.output if args.output is not None else args.output_json

    items = load_items_from_json(args.input_json)
    llm, embeddings = default_local_llm_and_embeddings()

    run_cfg = default_run_config(
        timeout_sec=args.ragas_timeout,
        max_workers=args.ragas_max_workers,
    )
    evaluator = RagasRagEvaluator(
        faithfulness_threshold=args.faithfulness_threshold,
        answer_relevancy_threshold=args.relevancy_threshold,
        context_precision_threshold=args.context_precision_threshold,
        llm=llm,
        embeddings=embeddings,
        run_config=run_cfg,
    )
    report = evaluator.evaluate(items)
    text = report.to_json(indent=2)

    if out_path:
        out_path.write_text(text, encoding="utf-8")
    else:
        print(text)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ConnectionError, RuntimeError, OSError) as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1)
