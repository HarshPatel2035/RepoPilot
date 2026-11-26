# Evaluations

RepoPilot is measured against a hand-labelled **golden set**: 26 questions about
[`pallets/itsdangerous`](https://github.com/pallets/itsdangerous) pinned to tag `2.2.0`, so
the labelled line numbers never drift. Each question records where the answer lives
(file + line range) and a short reference answer. See [`itsdangerous.json`](itsdangerous.json).

The questions mix plain-English phrasing ("Why should different parts of an app use different
salts?") with identifier-heavy ones ("What does NoneAlgorithm do?"), and cover source code
and docs.

## Retrieval eval (no API key needed, runs in CI)

A retrieved chunk is relevant if it is in the same file as a labelled span and their line
ranges overlap. Each labelled span counts once, so overlapping chunks cannot inflate scores.

| Metric | Meaning |
|---|---|
| Hit rate @k | Share of questions where at least one relevant chunk was retrieved |
| Recall @k | Share of all labelled spans that were retrieved |
| MRR | Mean of 1 / rank of the first relevant chunk (rewards ranking it first) |
| nDCG @k | Ranking quality over all relevant chunks, 1.0 = perfect order |

Results at k = 8 (the API default), CPU, `BAAI/bge-small-en-v1.5` embeddings:

| Mode | Hit rate | Recall | MRR | nDCG | p50 latency |
|---|---|---|---|---|---|
| vector only | 0.846 | 0.801 | 0.519 | 0.571 | 11 ms |
| keyword only (Postgres full-text) | 0.769 | 0.686 | 0.382 | 0.429 | 12 ms |
| **hybrid (RRF) — production default** | **0.962** | **0.917** | 0.580 | 0.635 | 12 ms |
| hybrid + cross-encoder rerank | 0.923 | 0.859 | **0.708** | **0.712** | 2284 ms |

Raw per-question results: [`results/retrieval.json`](results/retrieval.json).

### What the eval changed

1. **Keyword search config.** The first run used Postgres's `simple` text config. Keyword
   search scored hit@5 = 0.42 and dragged hybrid *below* vector-only (0.69 vs 0.81), because
   words like "how" and "the" matched every chunk. Switching to the `english` config
   (stopwords + stemming) raised keyword hit@5 to 0.69 and hybrid to 0.85, above vector-only.
2. **Reranker stays off by default.** The cross-encoder improves MRR by 22% (0.580 to 0.708),
   but costs ~2.3 s per question on CPU, about 190x the hybrid search time and longer than
   the LLM call. The LLM reads all 8 chunks, so whether the right chunk is *in* the set
   (hybrid's 0.962 hit rate) matters more than whether it is ranked first. Set `RERANK=true`
   to turn it on.

### Limits

- 26 questions: one question moves hit rate by 3.8 points, so small differences are noise.
- One small repository (~2,800 lines). Results on large or non-Python repos may differ.
- Labels were written by one person.

## Generation eval (needs `GROQ_API_KEY`)

Each question is answered by the full pipeline, then scored:

| Metric | How |
|---|---|
| Faithfulness (1-5) | LLM judge: is every claim supported by the retrieved chunks? |
| Correctness (1-5) | LLM judge: does the answer agree with the reference answer? |
| Citation precision | Deterministic: share of `[path:start-end]` citations that point at a chunk the model was actually given (catches invented sources) |
| Answers citing | Share of answers with at least one citation |

The judge defaults to a different model family (`openai/gpt-oss-120b`) from the answering
model (`llama-3.3-70b-versatile`), so a model is not grading its own answers. Malformed judge
output is recorded and excluded rather than guessed. This eval is a manual report, not a CI
gate, because it costs API calls and LLM output varies run to run.

Results: not run yet (needs an API key).

## Running

```bash
# Needs Postgres with pgvector (see the main README). First run downloads the models.
python -m repopilot.evals retrieval                       # all four modes, k=8
python -m repopilot.evals --k 5 retrieval --modes hybrid  # one mode, different k
python -m repopilot.evals retrieval --min-hit-rate 0.90 --min-mrr 0.50   # exits 1 if below
python -m repopilot.evals generation --mode hybrid        # needs GROQ_API_KEY
```

In CI, the `retrieval-eval` job runs on every push and fails the build if hybrid drops below
hit rate 0.90 or MRR 0.50. The `generation-eval` workflow runs on demand from the Actions tab
once a `GROQ_API_KEY` repository secret is added.
