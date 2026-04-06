from __future__ import annotations

import json
import math
import typing as t
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.run_config import RunConfig
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_utilization
from rag_eval.guardrails import guard_text
from ragas.metrics._faithfulness import faithfulness

if t.TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseLanguageModel


@dataclass
class EvalItem:
    """One user question, retrieved source strings, and the model response."""

    question: str
    contexts: list[str]
    response: str
    id: str | None = None


@dataclass
class RowEvaluation:
    id: str | None
    question: str
    response: str
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    hallucination_flag: bool
    flags: list[str] = field(default_factory=list)


@dataclass
class EvaluationReport:
    rows: list[RowEvaluation]
    aggregate_faithfulness: float | None
    aggregate_answer_relevancy: float | None
    aggregate_context_precision: float | None
    hallucination_count: int

    def to_dict(self) -> dict[str, t.Any]:
        return {
            "rows": [asdict(r) for r in self.rows],
            "aggregate": {
                "faithfulness": self.aggregate_faithfulness,
                "answer_relevancy": self.aggregate_answer_relevancy,
                "context_precision": self.aggregate_context_precision,
            },
            "hallucination_count": self.hallucination_count,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def _nan_to_none(x: t.Any) -> float | None:
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(xf):
        return None
    return xf


def _mean(vals: list[float | None]) -> float | None:
    nums = [v for v in vals if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


class RagasRagEvaluator:
    """
    Runs Ragas metrics on (question, retrieved contexts, response) tuples:

    - **faithfulness** — claims in the answer supported by contexts (groundedness).
    - **answer_relevancy** — alignment between the answer and the question (embedding + LLM).
    - **context precision** — from Ragas ``context_utilization``: whether each retrieved
      chunk was useful for producing the given answer (no gold reference required).

    **Hallucination heuristic:** ``faithfulness`` below a threshold marks likely
    ungrounded / hallucinated content; additional warnings come from low relevancy
    or low context precision.
    """

    def __init__(
        self,
        *,
        faithfulness_threshold: float = 0.5,
        answer_relevancy_threshold: float = 0.5,
        context_precision_threshold: float = 0.5,
        llm: BaseLanguageModel | None = None,
        embeddings: Embeddings | None = None,
        run_config: RunConfig | None = None,
    ) -> None:
        self.faithfulness_threshold = faithfulness_threshold
        self.answer_relevancy_threshold = answer_relevancy_threshold
        self.context_precision_threshold = context_precision_threshold
        self.llm = llm
        self.embeddings = embeddings
        if run_config is None:
            from rag_eval.local_models import default_run_config

            run_config = default_run_config()
        self._run_config = run_config
        self._metrics = [faithfulness, answer_relevancy, context_utilization]

    def evaluate(
        self, items: list[EvalItem], *, show_progress: bool = True
    ) -> EvaluationReport:
        if not items:
            return EvaluationReport(
                rows=[],
                aggregate_faithfulness=None,
                aggregate_answer_relevancy=None,
                aggregate_context_precision=None,
                hallucination_count=0,
            )

        samples: list[SingleTurnSample] = []
        for it in items:
            if not it.contexts:
                raise ValueError(
                    f"Sample {it.id or '(no id)'}: at least one context string is required."
                )
            samples.append(
                SingleTurnSample(
                    user_input=it.question,
                    retrieved_contexts=list(it.contexts),
                    response=it.response,
                )
            )

        dataset = EvaluationDataset(samples=samples)
        result = evaluate(
            dataset,
            metrics=self._metrics,
            llm=self.llm,
            embeddings=self.embeddings,
            show_progress=show_progress,
            run_config=self._run_config,
        )

        rows_out: list[RowEvaluation] = []
        f_vals: list[float | None] = []
        ar_vals: list[float | None] = []
        cp_vals: list[float | None] = []

        for idx, score_row in enumerate(result.scores):
            it = items[idx]
            f = _nan_to_none(score_row.get("faithfulness"))
            ar = _nan_to_none(score_row.get("answer_relevancy"))
            cp = _nan_to_none(score_row.get("context_utilization"))
            f_vals.append(f)
            ar_vals.append(ar)
            cp_vals.append(cp)

            flags: list[str] = []
            if f is None:
                flags.append("faithfulness_unavailable")
            elif f < self.faithfulness_threshold:
                flags.append("low_faithfulness")
            if ar is None:
                flags.append("answer_relevancy_unavailable")
            elif ar < self.answer_relevancy_threshold:
                flags.append("low_answer_relevancy")
            if cp is None:
                flags.append("context_precision_unavailable")
            elif cp < self.context_precision_threshold:
                flags.append("low_context_precision")

            hallucination = f is not None and f < self.faithfulness_threshold

            rows_out.append(
                RowEvaluation(
                    id=it.id,
                    question=it.question,
                    response=it.response,
                    faithfulness=f,
                    answer_relevancy=ar,
                    context_precision=cp,
                    hallucination_flag=hallucination,
                    flags=flags,
                )
            )

        return EvaluationReport(
            rows=rows_out,
            aggregate_faithfulness=_mean(f_vals),
            aggregate_answer_relevancy=_mean(ar_vals),
            aggregate_context_precision=_mean(cp_vals),
            hallucination_count=sum(1 for r in rows_out if r.hallucination_flag),
        )


def evaluate_batch(
    items: list[EvalItem],
    *,
    faithfulness_threshold: float = 0.5,
    answer_relevancy_threshold: float = 0.5,
    context_precision_threshold: float = 0.5,
    llm: BaseLanguageModel | None = None,
    embeddings: Embeddings | None = None,
    run_config: RunConfig | None = None,
    show_progress: bool = True,
) -> EvaluationReport:
    """Convenience wrapper around :class:`RagasRagEvaluator`."""
    return RagasRagEvaluator(
        faithfulness_threshold=faithfulness_threshold,
        answer_relevancy_threshold=answer_relevancy_threshold,
        context_precision_threshold=context_precision_threshold,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    ).evaluate(items, show_progress=show_progress)


def load_items_from_json(path: str | Path) -> list[EvalItem]:
    """
    Load evaluation items from JSON.

    Expected shape::

        {
          "items": [
            {
              "id": "optional",
              "question": "...",
              "contexts": ["source chunk 1", "..."],
              "context_pdfs": ["relative/or/absolute/path.pdf"],
              "response": "model answer"
            }
          ]
        }

    Aliases: ``query`` for ``question``, ``sources`` or ``documents`` for ``contexts``,
    ``answer`` for ``response``.

    PDF paths in ``context_pdfs`` are resolved relative to the JSON file's directory
    when not absolute. Extracted text is appended as context strings (see
    :func:`rag_eval.pdf_text.contexts_from_pdf_paths`).
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        entries = raw
    else:
        entries = raw.get("items") or raw.get("samples") or raw.get("data")
    if not isinstance(entries, list):
        raise ValueError("JSON must be a list or an object with an 'items' array.")

    items: list[EvalItem] = []
    for i, row in enumerate(entries):
        if not isinstance(row, dict):
            raise ValueError(f"Entry {i} must be an object.")
        q = row.get("question") or row.get("query") or row.get("user_input")
        ctx = row.get("contexts") or row.get("sources") or row.get("documents") or []
        pdf_specs = row.get("context_pdfs") or row.get("pdf_contexts") or []
        resp = row.get("response") or row.get("answer")
        if not q or resp is None:
            raise ValueError(
                f"Entry {i}: need question (or query) and response (or answer)."
            )
        allowed_q, reason_q = guard_text(str(q))
        if not allowed_q:
            raise ValueError(f"Entry {i}: unsafe question blocked: {reason_q}")
        allowed_r, reason_r = guard_text(str(resp))
        if not allowed_r:
            raise ValueError(f"Entry {i}: unsafe response blocked: {reason_r}")
        if not ctx and not pdf_specs:
            raise ValueError(
                f"Entry {i}: provide at least one of contexts (or sources) or context_pdfs."
            )
        if isinstance(ctx, str):
            ctx = [ctx]
        ctx_list = [str(c) for c in ctx]
        if pdf_specs:
            from rag_eval.pdf_text import contexts_from_pdf_paths

            if isinstance(pdf_specs, str):
                pdf_specs = [pdf_specs]
            resolved = []
            for p in pdf_specs:
                pp = Path(p)
                if not pp.is_absolute():
                    pp = (path.parent / pp).resolve()
                resolved.append(pp)
            ctx_list.extend(contexts_from_pdf_paths(resolved))
        items.append(
            EvalItem(
                id=row.get("id"),
                question=str(q),
                contexts=ctx_list,
                response=str(resp),
            )
        )
    return items


def default_local_llm_and_embeddings():
    """Local Ollama chat + embeddings (see :mod:`rag_eval.local_models`)."""
    from rag_eval.local_models import default_local_llm_and_embeddings as _factory

    return _factory()
