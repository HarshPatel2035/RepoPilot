# RepoPilot

Ask questions about any public GitHub repository and get answers cited to the exact files and lines.

RepoPilot clones a repository, splits its source into line-numbered chunks, embeds them into
Postgres with pgvector, and answers questions with **hybrid retrieval** (vector + keyword search,
merged with Reciprocal Rank Fusion) and an LLM that must cite `[path:start-end]` for every claim.

> Status: **Phase 3 of 6** — RAG API plus a LangGraph agent that investigates code and drafts
> GitHub issue replies, with a retrieval quality gate in CI. See [Roadmap](#roadmap).

## Evaluation

Measured on a hand-labelled golden set of 26 questions about `pallets/itsdangerous@2.2.0`,
at k = 8 (full method, metrics and limits in [evals/README.md](evals/README.md)):

| Retrieval mode | Hit rate | MRR | p50 latency |
|---|---|---|---|
| vector only | 0.846 | 0.519 | 11 ms |
| keyword only | 0.769 | 0.382 | 12 ms |
| **hybrid (default)** | **0.962** | 0.580 | 12 ms |
| hybrid + rerank | 0.923 | **0.708** | 2284 ms |

The eval found that Postgres's `simple` text config made hybrid search *worse* than vector
alone, and switching to `english` fixed it. It also showed the reranker is not worth ~2.3 s
per question when the LLM reads all 8 chunks anyway. CI fails the build if hybrid hit rate
drops below 0.90 or MRR below 0.50.

## Architecture

```mermaid
flowchart LR
    U[Client] -->|POST /repos| API[FastAPI]
    API -->|background task| IDX[Indexer]
    IDX --> CL[git clone --depth 1]
    CL --> F[Filter: code & docs only]
    F --> CH[Line chunker 40 lines, 10 overlap]
    CH --> EM[bge-small embeddings, ONNX]
    EM --> PG[(Postgres + pgvector)]
    U -->|POST /repos/id/ask| API
    API --> HS[Hybrid search: vector + full-text, RRF]
    HS --> PG
    HS --> LLM[Groq Llama 3.3 70B]
    LLM -->|answer + cited sources| U
```

## Quickstart (Docker)

```bash
cp .env.example .env          # then add your GROQ_API_KEY (free at console.groq.com)
docker compose up --build
```

The API docs are at http://localhost:8000/docs.

```bash
# 1. Index a repository (returns immediately, indexing runs in the background)
curl -X POST localhost:8000/repos -H 'content-type: application/json' \
  -d '{"url": "https://github.com/pallets/itsdangerous"}'

# 2. Poll until status is "ready"
curl localhost:8000/repos/1

# 3. Ask a question
curl -X POST localhost:8000/repos/1/ask -H 'content-type: application/json' \
  -d '{"question": "How are signatures verified?"}'
```

Response shape (sources are real retrieval results for this question; answer text abbreviated):

```json
{
  "answer": "... [src/itsdangerous/signer.py:61-100] ...",
  "sources": [
    {"path": "README.md", "start_line": 1, "end_line": 40, "score": 0.0320},
    {"path": "src/itsdangerous/signer.py", "start_line": 61, "end_line": 100, "score": 0.0299}
  ]
}
```

## API

| Method | Path | What it does |
|---|---|---|
| `GET` | `/health` | Checks the database connection |
| `POST` | `/repos` | Queue a GitHub repo for indexing (`202`). Re-posting re-indexes it |
| `GET` | `/repos` | List repos and their status (`pending`, `indexing`, `ready`, `failed`) |
| `GET` | `/repos/{id}` | One repo's status, chunk count and error |
| `POST` | `/repos/{id}/ask` | Ask a question; returns the answer and its sources |
| `POST` | `/repos/{id}/agent` | Ask the agent; it searches and reads files over several steps and returns the answer, sources and every tool call it made |
| `POST` | `/repos/{id}/issues/{n}/draft-reply` | Read GitHub issue `n` and draft a reply for a maintainer to review. Never posts anything |

## Agent

`/ask` does one search and one LLM call. The agent (`/agent`, `/draft-reply`) is a
[LangGraph](https://langchain-ai.github.io/langgraph/) state machine that decides what to look
at next, with three read-only tools:

| Tool | What it does |
|---|---|
| `search_code(query)` | Hybrid search, returns the top chunks labelled `[path:start-end]` |
| `list_files(prefix)` | Lists indexed file paths, to explore the layout |
| `read_file(path, start_line, end_line)` | Reads up to 150 numbered lines of a stored file |

```mermaid
flowchart LR
    S([start]) --> A[agent: LLM with tools]
    A -->|tool calls| T[tools]
    T -->|tool_rounds + 1| A
    A -->|answer| E([end])
    A -.->|budget used up: tool_choice=none| E
```

- **Every run ends with an answer.** The graph counts tool rounds in its state; when the budget
  (`AGENT_MAX_TOOL_ROUNDS`, default 6) is spent, the model must answer from the evidence it has,
  with tool use switched off, instead of crashing into LangGraph's recursion limit.
- **Traceable.** Each tool records exactly what it showed the model. The response lists those
  sources and every tool call, so you can audit how an answer was reached.
- **Safe by design.** Tools are read-only. There is no "run the tests" tool on purpose: running
  code from an arbitrary public repo needs a real sandbox first. Repository files and issue text
  are fenced off as untrusted data in the prompt, and issue replies are drafts that a person
  reviews; RepoPilot never posts to GitHub, which limits what a prompt injection could do.

```bash
curl -X POST localhost:8000/repos/1/agent -H 'content-type: application/json' \
  -d '{"question": "Where is the signature checked, and which keys does it try?"}'
curl -X POST localhost:8000/repos/1/issues/42/draft-reply
```

## Design decisions

- **Postgres + pgvector instead of a separate vector DB.** Repo metadata, chunks, vectors and
  full-text index live in one transactional store. Re-indexing swaps chunks in a single
  transaction, so search never sees a half-built index.
- **Hybrid search with Reciprocal Rank Fusion.** Embeddings find code by meaning ("where are
  passwords checked?"); Postgres full-text search finds exact identifiers (`LRUCache`) that
  embeddings often miss. RRF merges both rankings without having to tune score scales.
  Full-text search uses the `english` config (stopwords removed, words stemmed), chosen
  because the eval showed `simple` made hybrid worse than vector search alone.
- **Reranking is optional.** A cross-encoder (`RERANK=true`) ranks the best chunk first more
  often, but the eval measured ~2.3 s extra per question on CPU, so it is off by default.
- **Line-based chunks with the file path prepended before embedding.** Every chunk keeps
  1-based line numbers so answers can cite `file:start-end`; prefixing the path lets a question
  about "auth" match code inside `auth.py`.
- **fastembed (ONNX) for embeddings.** Runs `BAAI/bge-small-en-v1.5` on CPU without PyTorch,
  which keeps the Docker image small.
- **Strict URL validation before `git clone`.** Only `https://github.com/<owner>/<repo>` is
  accepted, which blocks `file://`, SSH, other hosts and argument injection. Symlinks inside
  the clone are never read.
- **Errors are safe to show.** Known problems (private repo, no source files) are reported on
  the repo; unexpected errors are logged server-side and shown as a generic message.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Unit tests run without a database; set TEST_DATABASE_URL to run the store and API tests too.
export TEST_DATABASE_URL=postgresql://repopilot:repopilot@localhost:5432/repopilot_test
pytest
ruff check . && ruff format --check .

# Run the API against your own Postgres (needs the pgvector extension available)
uvicorn --factory repopilot.api:create_app --reload
```

Tests use a deterministic hash embedder and a fake LLM, so they run offline and fast. CI runs
lint, the full test suite against a real pgvector service container, the retrieval quality
gate with the real embedding model, and a Docker build.

## Project layout

```
repopilot/
  api.py         FastAPI app and request/response schemas
  indexer.py     clone -> filter -> chunk -> embed -> store pipeline
  ingest.py      URL validation, git clone, file filtering, line chunker
  store.py       Postgres schema, hybrid search (vector + full-text + RRF), stored files
  embeddings.py  fastembed wrapper
  llm.py         prompt building and Groq client
  retrieval.py   search modes + optional cross-encoder reranking
  agent/         LangGraph agent graph, read-only tools, GitHub issue client
  evals/         golden-set loader, metrics, LLM judge, eval runner and CLI
evals/           golden question set, results and evaluation write-up
tests/           unit tests + database-backed store, API and eval tests
```

## Roadmap

1. **RAG core** — FastAPI, pgvector hybrid search, Docker, tests, CI ✅
2. **Evaluation** — golden question set, retrieval metrics, LLM-as-judge, citation checks,
   eval gate in CI ✅
3. **Agent** — LangGraph agent with search, list and read tools; drafts GitHub issue replies ✅
   (a sandboxed "run tests" tool is deferred until there is a safe sandbox)
4. **Issue classifier** — scikit-learn baseline vs LoRA-tuned small model, tracked in MLflow
5. **Frontend + observability** — React UI, tracing, latency and cost dashboard
6. **Deploy** — Azure Container Apps, CD from GitHub Actions
