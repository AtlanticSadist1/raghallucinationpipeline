# RAG hallucination evaluation pipeline

Evaluate LLM answers against source text with **Ragas** (faithfulness, answer relevancy, context precision via context utilization) and optional **hallucination flags**. You can run a **batch job from JSON** (including **PDF** sources) or a **browser chat** that ingests PDFs, answers from retrieved chunks, and scores every reply.

**Everything runs locally** via **Ollama** — no OpenAI or other cloud LLM/embeddings APIs are used by this project.

---

## Quick start (checklist)

Do these **once** (or when you set up a new machine):

1. Install **[Ollama](https://ollama.com)** and start it (the app usually keeps the server running in the background).
2. Pull the default models (terminal):

   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

3. Install **Python 3.12**, create a venv in this repo, and install dependencies (see [One-time Python setup](#one-time-python-setup) below).

Before **each** terminal session where you run the tools:

4. **Activate the virtual environment** (commands differ on Windows vs macOS/Linux — see below).
5. Confirm Ollama is reachable (optional but useful):

   ```bash
   ollama list
   ```

   You should see `llama3.2` and `nomic-embed-text` (or whatever you set in `OLLAMA_CHAT_MODEL` / `OLLAMA_EMBED_MODEL`).

Then choose **either** [batch evaluation](#how-to-run-batch-evaluation-json) **or** [browser chat](#how-to-run-the-browser-chat).

---

## One-time Python setup

Use the folder that contains `requirements.txt` and `rag_eval/` as the project root.

### Windows (PowerShell)

```powershell
cd C:\path\to\raghallucinationpipeline
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # only if Activate.ps1 is blocked
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux (bash/zsh)

```bash
cd /path/to/raghallucinationpipeline
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Activate the venv later (every new terminal)

| OS | Command |
|----|---------|
| **Windows PowerShell** | `cd` to project root, then `.\.venv\Scripts\Activate.ps1` |
| **Windows CMD** | `cd` to project root, then `.venv\Scripts\activate.bat` |
| **macOS / Linux** | `cd` to project root, then `source .venv/bin/activate` |

Your prompt should show `(.venv)` when the venv is active. All `python` / `pip` commands below assume the venv is **activated** (or you use the full path to `.venv\Scripts\python` on Windows).

---

## How to run batch evaluation (JSON)

**What it does:** Reads a JSON file of rows (each with a question, model response, and text/PDF context). Runs Ragas and writes a JSON report with scores and hallucination flags.

### Steps

1. **Ollama** is running and models are pulled (see [Quick start](#quick-start-checklist)).
2. **Activate** your venv and `cd` to the project root.
3. Run:

   **Windows (PowerShell)** — example using the sample file (second argument is the output path; `-o` works too):

   ```powershell
   python run_eval.py examples\sample_batch.json results.json
   ```

   **macOS / Linux:**

   ```bash
   python run_eval.py examples/sample_batch.json results.json
   ```

   Equivalent: `python run_eval.py examples/sample_batch.json -o results.json`

   If you prefer not to activate the venv, call the interpreter explicitly, e.g. Windows:

   ```powershell
   .\.venv\Scripts\python.exe run_eval.py examples\sample_batch.json results.json
   ```

4. **Output:** `results.json` in the current directory (or the path you passed to `-o`). If you omit `-o`, the report is printed to the terminal.

### Optional CLI flags

| Flag | Purpose |
|------|--------|
| `--faithfulness-threshold` | Below this faithfulness score, a row is flagged as hallucination risk (default `0.5`). |
| `--relevancy-threshold` | Warn when answer relevancy is below this (default `0.5`). |
| `--context-precision-threshold` | Warn when context utilization is below this (default `0.5`). |
| `--ragas-timeout SEC` | Seconds Ragas waits per metric sub-step (default `600`). Raise this if you still see timeouts on a slow PC. |
| `--ragas-max-workers N` | Parallel Ragas jobs (default `1`). Keep **`1`** for a single local Ollama server; higher values overload the queue and cause `TimeoutError`. |

Example:

```bash
python run_eval.py examples/sample_batch.json -o results.json --faithfulness-threshold 0.6
```

Slow machine or very large contexts:

```bash
python run_eval.py examples/sample_batch.json results.json --ragas-timeout 900 --ragas-max-workers 1
```

### JSON format (reference)

Each item needs a **question**, a **response**, and at least one of **text contexts** or **PDF paths**.

```json
{
  "items": [
    {
      "id": "optional-id",
      "question": "What is the refund window?",
      "contexts": ["Returns within 30 days..."],
      "response": "You have 30 days to return the product."
    }
  ]
}
```

**Field aliases:** `query` → question; `sources` / `documents` → contexts; `answer` → response.

### PDFs as context (batch)

Add **`context_pdfs`** (or `pdf_contexts`): a list of paths. Paths **relative to the JSON file** are resolved from that file’s folder.

```json
{
  "items": [
    {
      "question": "Summarize the policy.",
      "context_pdfs": ["policies/handbook.pdf"],
      "response": "The model answer to evaluate goes here."
    }
  ]
}
```

On Windows in JSON you can use doubled backslashes, e.g. `"policies\\handbook.pdf"`. You can combine `contexts` and `context_pdfs`.

---

## How to run the browser chat

**What it does:** Starts a local web UI (Gradio). You upload PDFs, index them, then chat. Each reply is grounded on retrieved chunks and ends with Ragas scores and a hallucination-risk line.

### Steps

1. **Ollama** is running and models are pulled.
2. **Activate** the venv and `cd` to the project root.
3. Start the app:

   ```bash
   python run_chat.py
   ```

   Windows without activating venv:

   ```powershell
   .\.venv\Scripts\python.exe run_chat.py
   ```

4. Open a browser to **http://127.0.0.1:7860** (unless you changed host/port).
5. In the UI:
   - Click **Context PDFs** and select one or more `.pdf` files.
   - Click **Load PDFs into knowledge base** and wait until the status says indexing finished.
   - Type a question in **Your message** and submit. The assistant answer includes a **Ragas** block at the bottom.
6. Stop the server with **Ctrl+C** in the terminal.

### Optional CLI flags

| Flag | Example | Purpose |
|------|---------|---------|
| `--host` | `python run_chat.py --host 0.0.0.0` | Listen on all interfaces (default `127.0.0.1`). |
| `--port` | `python run_chat.py --port 8080` | Port (default `7860`). |
| `--share` | `python run_chat.py --share` | Gradio temporary public URL (use with care). |
| `--top-k` | `python run_chat.py --top-k 8` | How many PDF chunks to retrieve per question (default `6`). |

Full example:

```bash
python run_chat.py --host 127.0.0.1 --port 7860 --top-k 6
```

---

## Environment variables (optional)

Override Ollama defaults without editing code:

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama API base URL |
| `OLLAMA_CHAT_MODEL` | `llama3.2` | Chat + Ragas judge LLM |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embeddings for retrieval and answer relevancy |
| `OLLAMA_HTTP_TIMEOUT` | `600` | Max seconds for a single Ollama HTTP request (generation can be long) |
| `OLLAMA_NUM_CTX` | _(unset)_ | Optional context length, e.g. `8192`, for the chat model |
| `RAGAS_TIMEOUT_SEC` | `600` | Ragas asyncio timeout per metric step (overridden by `--ragas-timeout`) |
| `RAGAS_MAX_WORKERS` | `1` | Parallel Ragas tasks (overridden by `--ragas-max-workers`) |

**Windows PowerShell**

```powershell
$env:OLLAMA_CHAT_MODEL = "llama3.1:8b"
python run_eval.py examples\sample_batch.json -o results.json
```

**macOS / Linux**

```bash
export OLLAMA_CHAT_MODEL=llama3.1:8b
python run_eval.py examples/sample_batch.json -o results.json
```

---

## Project layout (quick reference)

| Path | Role |
|------|------|
| `rag_eval/evaluator.py` | `RagasRagEvaluator`, `load_items_from_json`, `EvalItem`, `default_local_llm_and_embeddings` |
| `rag_eval/local_models.py` | Ollama URL/model env vars and health checks |
| `rag_eval/pdf_text.py` | PDF extraction and chunking |
| `rag_eval/gradio_chat.py` | Browser UI (`launch_app`) |
| `run_eval.py` | Batch CLI |
| `run_chat.py` | Browser chat CLI |
| `examples/sample_batch.json` | Example batch file |

---

## Using the library in Python

```python
from rag_eval import EvalItem, RagasRagEvaluator, default_local_llm_and_embeddings

llm, embeddings = default_local_llm_and_embeddings()
ev = RagasRagEvaluator(llm=llm, embeddings=embeddings)
report = ev.evaluate(
    [
        EvalItem(
            question="What is the capital of France?",
            contexts=["Paris is the capital of France."],
            response="Paris.",
        )
    ]
)
print(report.to_json())
```

You can also pass your own LangChain-compatible `llm` / `embeddings` into `RagasRagEvaluator`.

---

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| **Cannot reach Ollama** | Start Ollama; on Linux you may run `ollama serve`. Check `OLLAMA_BASE_URL`. |
| **Model not found** | `ollama pull <name>` for both chat and embed models (defaults: `llama3.2`, `nomic-embed-text`). |
| **`python` not found** | Use `py -3.12` (Windows) or `python3.12` (macOS/Linux), or the full path `.venv\Scripts\python.exe`. |
| **Wrong / no packages** | Activate the venv first, then `pip install -r requirements.txt`. |
| **Activate.ps1 disabled** | PowerShell: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| **Import or Pydantic errors on Python 3.14** | Recreate the venv with **Python 3.12**. |
| **Empty PDF text** | Scanned/image-only PDFs need OCR elsewhere first. |
| **Odd Ragas scores** | Try a larger local chat model via `OLLAMA_CHAT_MODEL`. |
| **`TimeoutError` during `run_eval`** | Defaults are already tuned (`--ragas-max-workers 1`, `--ragas-timeout 600`). Increase `--ragas-timeout` and ensure `--ragas-max-workers` stays `1` unless you run multiple Ollama instances. Increase `OLLAMA_HTTP_TIMEOUT` if the client drops long runs. |
| **“No statements were generated from the answer”** | The faithfulness step could not parse the model output. Try a stronger chat model, set `OLLAMA_NUM_CTX=8192`, or shorten the evaluated `response` text in JSON. |

---

## Privacy note

Inference stays on your machine (Ollama). Pulling models uses the network when you `ollama pull`; normal chat and evaluation traffic goes to your local Ollama server only, not to cloud LLM APIs from this codebase.
