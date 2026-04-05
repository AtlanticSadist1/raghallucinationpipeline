from __future__ import annotations

import io
import typing as t
from pathlib import Path

from pypdf import PdfReader


def extract_text_from_pdf(source: str | Path | bytes) -> str:
    """
    Extract plain text from a PDF file path, ``Path``, or raw bytes.
    Pages are concatenated with blank lines; empty pages are skipped.
    """
    if isinstance(source, bytes):
        reader = PdfReader(io.BytesIO(source))
    else:
        reader = PdfReader(str(source))
    parts: list[str] = []
    for page in reader.pages:
        try:
            txt = page.extract_text()
        except Exception:
            txt = ""
        if txt:
            parts.append(txt)
    return "\n\n".join(parts).strip()


def chunk_text(
    text: str,
    *,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[str]:
    """Split text into overlapping character windows (simple RAG chunks)."""
    text = text.strip()
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size - 1))
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = end - overlap
    return chunks


def contexts_from_pdf_paths(
    paths: t.Sequence[str | Path],
    *,
    max_chars_per_pdf: int = 80_000,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> list[str]:
    """
    Load PDFs and return a list of context strings suitable for Ragas.

    Each PDF is capped at ``max_chars_per_pdf`` (then chunked). Every chunk
    becomes one context string so metrics can treat them like retrieved passages.
    """
    out: list[str] = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")
        raw = extract_text_from_pdf(path)
        if not raw:
            continue
        if len(raw) > max_chars_per_pdf:
            raw = raw[:max_chars_per_pdf]
        out.extend(chunk_text(raw, chunk_size=chunk_size, overlap=chunk_overlap))
    return out
