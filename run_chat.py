"""
Browser chat: PDFs as context, grounded answers, Ragas scores on each reply.
Uses local Ollama only (no cloud APIs). See README for `ollama pull` models.

  .venv\\Scripts\\python run_chat.py

Open http://127.0.0.1:7860 — upload PDFs, click **Load PDFs**, then chat.
"""

from __future__ import annotations

import argparse
import sys

from rag_eval.gradio_chat import launch_app


def main() -> int:
    p = argparse.ArgumentParser(description="PDF RAG chat with Ragas evaluation in the browser.")
    p.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    p.add_argument("--share", action="store_true", help="Create a temporary public Gradio link")
    p.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Number of PDF chunks to retrieve per question (default: 6)",
    )
    args = p.parse_args()

    try:
        launch_app(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            top_k_chunks=args.top_k,
        )
    except (ConnectionError, RuntimeError, OSError) as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
