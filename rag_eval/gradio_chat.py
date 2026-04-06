from __future__ import annotations

import typing as t

import numpy as np

from rag_eval.evaluator import EvalItem, RagasRagEvaluator, default_local_llm_and_embeddings
from rag_eval.guardrails import guard_text
from rag_eval.pdf_text import chunk_text, extract_text_from_pdf

if t.TYPE_CHECKING:
    pass


def _cosine_top_k(
    query_vec: list[float],
    doc_matrix: np.ndarray,
    k: int,
) -> list[int]:
    q = np.asarray(query_vec, dtype=np.float64)
    if doc_matrix.size == 0:
        return []
    norms = np.linalg.norm(doc_matrix, axis=1)
    qn = np.linalg.norm(q)
    if qn < 1e-12:
        return []
    sims = (doc_matrix @ q) / (norms * qn + 1e-12)
    k = min(k, len(sims))
    return list(np.argsort(-sims)[:k])


def _build_doc_state(
    pdf_paths: list[str],
    embeddings_model,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> dict[str, t.Any]:
    all_chunks: list[str] = []
    for p in pdf_paths:
        text = extract_text_from_pdf(p)
        if not text:
            continue
        all_chunks.extend(chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap))
    if not all_chunks:
        return {"ready": False, "error": "No text could be extracted from the PDFs."}
    batch_size = 32
    embs: list[list[float]] = []
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        embs.extend(embeddings_model.embed_documents(batch))
    mat = np.asarray(embs, dtype=np.float64)
    return {
        "ready": True,
        "chunks": all_chunks,
        "matrix": mat.tolist(),
        "error": None,
    }


def _answer_from_context(
    llm,
    question: str,
    contexts: list[str],
) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    ctx_block = "\n\n---\n\n".join(contexts)
    sys = SystemMessage(
        content=(
            "You are a careful assistant. Answer using ONLY the context below. "
            "If the context does not contain enough information, say you cannot "
            "find that in the documents. Be concise.\n\n"
            f"Context:\n{ctx_block}"
        )
    )
    human = HumanMessage(content=question)
    resp = llm.invoke([sys, human])
    return getattr(resp, "content", str(resp)) or ""


def _format_scores_block(report) -> str:
    if not report.rows:
        return ""
    r = report.rows[0]
    lines = [
        "",
        "---",
        "**Ragas (last reply)**",
        f"- Faithfulness: {r.faithfulness if r.faithfulness is not None else 'n/a'}",
        f"- Answer relevancy: {r.answer_relevancy if r.answer_relevancy is not None else 'n/a'}",
        f"- Context precision (utilization): {r.context_precision if r.context_precision is not None else 'n/a'}",
        f"- Hallucination risk (low faithfulness): **{'yes' if r.hallucination_flag else 'no'}**",
        f"- Flags: {', '.join(r.flags) if r.flags else 'none'}",
    ]
    return "\n".join(lines)


def launch_app(
    *,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
    top_k_chunks: int = 6,
) -> None:
    """Start the Gradio chat UI in the browser."""
    import gradio as gr

    llm, embeddings = default_local_llm_and_embeddings()
    evaluator = RagasRagEvaluator(llm=llm, embeddings=embeddings)

    def ingest(files: list | str | None) -> tuple[str, dict]:
        if not files:
            return "Upload at least one PDF, then click **Load PDFs**.", {"ready": False}
        if isinstance(files, str):
            files = [files]
        paths: list[str] = []
        for f in files:
            if hasattr(f, "name"):
                paths.append(f.name)
            else:
                paths.append(str(f))
        try:
            state = _build_doc_state(paths, embeddings)
        except Exception as e:
            return f"Error reading PDFs: {e}", {"ready": False}
        if not state["ready"]:
            return state.get("error") or "Failed to index PDFs.", state
        n = len(state["chunks"])
        return f"Indexed **{n}** chunks from {len(paths)} file(s). You can chat below.", state

    def respond(user_text: str, history: list | None, state: dict) -> tuple[str, list, dict]:
        """Gradio 6+ chat history: list of ``{\"role\": \"user\"|\"assistant\", \"content\": str}``."""
        history = list(history or [])
        if not user_text or not str(user_text).strip():
            return "", history, state
        q = user_text.strip()

        if not state.get("ready"):
            history.append({"role": "user", "content": q})
            history.append(
                {
                    "role": "assistant",
                    "content": "Please load PDFs first (upload files and click **Load PDFs**).",
                }
            )
            return "", history, state

        allowed, reason = guard_text(q)
        if not allowed:
            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": reason or "This request cannot be processed."})
            return "", history, state

        chunks: list[str] = state["chunks"]
        matrix: np.ndarray = np.asarray(state["matrix"], dtype=np.float64)
        q_emb = embeddings.embed_query(q)
        idxs = _cosine_top_k(q_emb, matrix, top_k_chunks)
        picked = [chunks[i] for i in idxs] if idxs else chunks[:top_k_chunks]

        try:
            answer = _answer_from_context(llm, q, picked)
        except Exception as e:
            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": f"Model error: {e}"})
            return "", history, state

        try:
            report = evaluator.evaluate(
                [EvalItem(question=q, contexts=picked, response=answer)],
                show_progress=False,
            )
            suffix = _format_scores_block(report)
        except Exception as e:
            suffix = f"\n\n---\n*(Ragas evaluation failed: {e})*"

        full_reply = answer + suffix
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": full_reply})
        return "", history, state

    with gr.Blocks(title="PDF RAG + Ragas") as demo:
        gr.Markdown(
            "## PDF-grounded chat\n"
            "Upload PDFs, click **Load PDFs**, then ask questions. "
            "Each assistant reply includes **Ragas** scores (faithfulness, relevancy, context precision) "
            "and a hallucination hint based on faithfulness."
        )
        files = gr.File(
            label="Context PDFs",
            file_count="multiple",
            file_types=[".pdf"],
            type="filepath",
        )
        load_btn = gr.Button("Load PDFs into knowledge base", variant="primary")
        status = gr.Markdown("")
        state = gr.State({"ready": False})

        load_btn.click(fn=ingest, inputs=[files], outputs=[status, state])

        chatbot = gr.Chatbot(label="Chat", height=480)
        msg = gr.Textbox(label="Your message", placeholder="Ask something about the PDFs…")
        clear = gr.Button("Clear chat")

        msg.submit(respond, [msg, chatbot, state], [msg, chatbot, state])
        clear.click(lambda: [], outputs=[chatbot])

    demo.launch(server_name=server_name, server_port=server_port, share=share)
