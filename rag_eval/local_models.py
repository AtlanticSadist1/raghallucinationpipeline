"""
Local-only inference via **Ollama** (no cloud API keys).

Requires ``ollama serve`` running and models pulled, e.g.::

    ollama pull llama3.2
    ollama pull nomic-embed-text

Environment (optional overrides):

- ``OLLAMA_BASE_URL`` — default ``http://127.0.0.1:11434``
- ``OLLAMA_CHAT_MODEL`` — chat model name (default ``llama3.2``)
- ``OLLAMA_EMBED_MODEL`` — embedding model (default ``nomic-embed-text``)
- ``OLLAMA_HTTP_TIMEOUT`` — httpx timeout seconds for each Ollama request (default ``600``)
- ``RAGAS_TIMEOUT_SEC`` — Ragas ``asyncio`` timeout per metric sub-task (default ``600``)
- ``RAGAS_MAX_WORKERS`` — parallel Ragas jobs (default ``1``; use ``1`` for local Ollama)
- ``OLLAMA_NUM_CTX`` — optional chat context window, e.g. ``8192`` (helps long-context Ragas prompts)
"""

from __future__ import annotations

import json
import os
import typing as t
import urllib.error
import urllib.request

from ragas.run_config import RunConfig

if t.TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseLanguageModel


def ollama_base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


def ollama_chat_model() -> str:
    return os.environ.get("OLLAMA_CHAT_MODEL", "llama3.2")


def ollama_embed_model() -> str:
    return os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def ollama_http_timeout() -> float:
    """Seconds for Ollama HTTP client (long generations need a high value)."""
    return float(os.environ.get("OLLAMA_HTTP_TIMEOUT", "600"))


def default_run_config(
    *,
    timeout_sec: int | None = None,
    max_workers: int | None = None,
) -> RunConfig:
    """
    Ragas runtime defaults tuned for **local Ollama**: one job at a time and long timeouts.

    Without this, Ragas uses ``max_workers=16`` and a 180s cap per metric step, which often
    causes ``TimeoutError`` when several metrics compete for a single GPU/CPU Ollama queue.
    """
    ts = (
        timeout_sec
        if timeout_sec is not None
        else int(os.environ.get("RAGAS_TIMEOUT_SEC", "600"))
    )
    mw = (
        max_workers
        if max_workers is not None
        else int(os.environ.get("RAGAS_MAX_WORKERS", "1"))
    )
    return RunConfig(timeout=max(ts, 60), max_workers=max(mw, 1))


def check_ollama_running(timeout: float = 5.0) -> None:
    """Raises ``ConnectionError`` if the Ollama server is not reachable."""
    url = f"{ollama_base_url()}/api/tags"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                raise ConnectionError(f"Ollama returned HTTP {resp.status} from {url}")
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Cannot reach Ollama at {ollama_base_url()}. "
            "Start it with `ollama serve` and ensure models are pulled "
            f"(`ollama pull {ollama_chat_model()}`, "
            f"`ollama pull {ollama_embed_model()}`)."
        ) from e


def _ollama_has_model(name: str, timeout: float = 5.0) -> bool:
    url = f"{ollama_base_url()}/api/tags"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, json.JSONDecodeError, OSError):
        return False
    models = data.get("models") or []
    req_base = name.split(":")[0]
    for m in models:
        n = (m.get("name") or "").strip()
        if not n:
            continue
        if n == name or n.split(":")[0] == req_base:
            return True
    return False


def check_models_available() -> None:
    """Raises ``RuntimeError`` if chat or embed models are missing locally."""
    chat, emb = ollama_chat_model(), ollama_embed_model()
    if not _ollama_has_model(chat):
        raise RuntimeError(
            f"Ollama chat model '{chat}' not found. Run: ollama pull {chat}"
        )
    if not _ollama_has_model(emb):
        raise RuntimeError(
            f"Ollama embedding model '{emb}' not found. Run: ollama pull {emb}"
        )


def default_local_llm_and_embeddings() -> tuple[BaseLanguageModel, Embeddings]:
    """
    Build LangChain chat + embedding clients pointing at local Ollama.

    Verifies the server is up and models are present before returning.
    """
    check_ollama_running()
    check_models_available()

    from langchain_ollama import ChatOllama, OllamaEmbeddings

    base = ollama_base_url()
    timeout = ollama_http_timeout()
    client_kw: dict[str, t.Any] = {"timeout": timeout}
    num_ctx = os.environ.get("OLLAMA_NUM_CTX")
    llm_kw: dict[str, t.Any] = {
        "base_url": base,
        "model": ollama_chat_model(),
        "temperature": 0,
        "client_kwargs": client_kw,
    }
    if num_ctx and num_ctx.isdigit():
        llm_kw["num_ctx"] = int(num_ctx)
    llm = ChatOllama(**llm_kw)
    embeddings = OllamaEmbeddings(
        base_url=base,
        model=ollama_embed_model(),
        client_kwargs=client_kw,
    )
    return llm, embeddings
