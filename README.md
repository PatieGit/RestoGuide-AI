# RestoGuide AI

An AI knowledge assistant for restaurant staff. Ask it about any procedure, recipe, or
customer preference in the restaurant's manual and get an answer grounded in that manual —
or a plain "not found" when the manual doesn't cover it.

Built as a domain-specific RAG system: 16 curated question/answer records, embedded with
`gemini-embedding-001`, retrieved from a local ChromaDB store, and answered by
`gemini-3.5-flash-lite` under strict guardrails.

**Benchmark results** (30 pairs, `eval/run_eval.py`):

| Metric | Target | Result |
| --- | --- | --- |
| Retrieval Hit Rate @ k=3 | ≥ 85 % | **100 %** |
| Zero-hallucination / refusal rate | 100 % | **100 %** |
| Query latency | < 3 s | median **2.3 s**, mean 2.0–2.9 s |

Latency is quoted as a range because it varies with Gemini's response time, not with anything
in this repo — measured over five consecutive runs. Individual queries occasionally stall well
past 3 s regardless of load.

---

## Setup

Requires Python 3.11+ (developed and verified on 3.14.5) and a Google AI Studio API key.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item .env.example .env
# then open .env and paste your key from https://aistudio.google.com/apikey
```

Verify the environment:

```powershell
python -c "import langchain, chromadb, streamlit, langchain_google_genai; print('deps ok')"
python -c "import os; from dotenv import load_dotenv; load_dotenv(); assert os.getenv('GOOGLE_API_KEY'); print('env ok')"
```

## Running it

The system is two independent pipelines. Build the knowledge base first, then query it.

### 1. Ingestion — build the index

```powershell
python -m src.ingestion.run_ingest --dry-run   # preview all 16 chunks, makes NO API calls
python -m src.ingestion.run_ingest             # embed and persist to ./chroma_db
```

Run this once, and again whenever `data/data.json` changes. Every run is a full rebuild, so
there is no stale state to reconcile.

Useful flags: `--limit N` (embed only the first N records), `--strategy {question_only,question_answer}`,
`--data PATH`.

### 2. Query — ask questions

```powershell
python -m src.query.run_query "How do I prep the beef wet fry?"
python -m src.query.run_query "what temperature should cold food be kept at?" --show-context
```

`--show-context` prints every retrieved chunk with its similarity score and whether the
relevance floor kept or dropped it.

### 3. The app

```powershell
streamlit run app.py
```

Quick-action buttons for common tasks sit in the sidebar, answers cite their source records,
and each citation expands to the exact manual text behind it.

Follow-up questions work: ask "what about for fish?" after a question about deliveries and it
resolves against the conversation rather than starting fresh. When a follow-up is rewritten,
the app shows what it actually searched for, so a surprising answer is always traceable.

> **Stop the app with Ctrl+C.** Force-killing the process while ChromaDB holds its database
> open can corrupt the store. If that happens the app tells you so — rebuild with
> `python -m src.ingestion.run_ingest`.

## Evaluation

```powershell
python eval/run_eval.py                      # full benchmark: hit rate, refusals, latency
python eval/run_eval.py --no-generate        # retrieval metrics only, no LLM calls
python eval/run_eval.py --floor-mode flat    # compare retrieval floor configurations
python eval/run_eval_multiturn.py            # conversational follow-ups
python eval/run_eval_multiturn.py --no-condense   # the stateless baseline, for comparison
```

Results are written to `eval/results.<strategy>.<floor_mode>.json`. `eval/benchmark.json`
holds the 30 ground-truth pairs: 12 direct, 12 paraphrased/informal, and 6 out-of-scope
queries that must be refused.

## Tests

```powershell
pytest -q                     # everything (129 tests)
pytest -q -m "not live"       # offline only, no API calls
pytest tests/test_isolation.py -q
```

## How it works

```
data/data.json ──► src/ingestion/ ──► ChromaDB (./chroma_db) ──► src/query/ ──► app.py
                   (writes only)        the contract           (reads only)
```

The two pipelines are **strictly isolated** — neither imports the other, and
`tests/test_isolation.py` proves it by importing each one with the other package physically
deleted. They communicate only through the persisted collection, whose exact shape is
documented in `IMPLEMENTATION.md` §3. `app.py` imports only `src.query`, so the interface
cannot write to the index even by accident.

Two details worth knowing:

- **Chunking is record-based.** One JSON record becomes exactly one chunk — never split by
  length, because each record is a self-contained procedure and cutting one in half leaves
  fragments that are meaningless when retrieved alone.
- **Embedding is question-only.** The vector is built from the record's *question*; the
  answer travels alongside as metadata and is reassembled into the prompt at query time.
  This diverges deliberately from `proposal.md` and `technical.md`; see the "Documented
  divergence" section of `IMPLEMENTATION.md`. The alternative (`question_answer`) is fully
  implemented and switchable via `EMBEDDING_STRATEGY` in `.env`.

**Two independent guardrails** stand between a question and an invented answer: a relevance
floor that refuses locally without ever calling the LLM, and a system prompt that instructs
refusal when the retrieved context doesn't contain the answer. Both were verified at 100 %
on the out-of-scope benchmark set.

## Project layout

```
data/data.json         the knowledge base — 16 question/answer records
src/config.py          models, paths, thresholds (shared declarations only)
src/schema.py          record validation and text formatting
src/ingestion/         loader, categorizer, chunker, embeddings, indexer, CLI
src/query/             embeddings, retriever, prompts, generator, engine, CLI
app.py                 Streamlit employee view
eval/                  benchmark, harness, committed results
tests/                 129 tests, live ones marked
IMPLEMENTATION.md      full build plan, pipeline contract, and every design decision
```

## Configuration

All settings live in `.env` (see `.env.example`). The ones that change behaviour:

| Setting | Default | Effect |
| --- | --- | --- |
| `EMBEDDING_STRATEGY` | `question_only` | What gets vectorised. Changing it requires a re-ingest. |
| `FLOOR_MODE` | `relative` | `relative` gates the top hit and keeps others near it; `flat` judges each chunk alone. |
| `MIN_RELEVANCE` | `0.75` | The floor itself. Calibrated on the benchmark; re-tune after a strategy change. |
| `TOP_K` | `3` | Chunks retrieved per query. |

## Not included (deliberate scope decisions)

These were considered and deferred, not overlooked:

- **Manager panel and role-based access.** The proposal describes a PIN-protected view for
  managers to add recipes and customer preferences. This MVP ships the employee view only;
  the manager panel, incremental (non-rebuild) ingestion, and an audit trail are specified
  as Phase 6 in `IMPLEMENTATION.md`.
- **Multi-format ingestion.** Only `data.json` is read. PDFs and spreadsheets are excluded
  on purpose — mis-parsed layouts silently corrupt a knowledge base, and every indexed chunk
  here is traceable to a reviewed record.
- **Voice input and POS integration**, both out of scope in the proposal.
