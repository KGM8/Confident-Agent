# Confident Agent — South African History

Agent-based validation pipeline for reducing LLM hallucinations on South African
history questions, built for COMP301 (Phase 4 — Design a Confident Agent).

## Architecture (as designed, one change explained below)

```
QUESTION
   │
   ▼
QUERY ANALYSER  (rule-based, no LLM call — see note below)
   │
   ▼
RETRIEVAL AGENT ──► VECTOR STORE (TF-IDF over your SAHO chunks)
   │
   ▼
RETRIEVAL VERIFIER ── POOR ──► QUERY REFORMULATION ──► back to RETRIEVAL AGENT (max 1 retry)
   │ GOOD
   ▼
ANSWER AGENT (LLM, grounded in retrieved chunks, must cite)
   │
   ▼
CRITIC AGENT (LLM) ── FAIL ──► REVISE ANSWER ──► CRITIC AGENT (max 1 retry)
   │ PASS
   ▼
CONFIDENCE CALCULATOR (combines retrieval score + critic verdict + revision count)
   │
   ▼
FINAL RESPONSE + CITATIONS + CONFIDENCE  (ANSWER or ABSTAIN)
```

### One deliberate change from the Initial architecture version

The original diagram implies the Query Analyser is its own agent (i.e. its own LLM
call). I implemented it as a **cheap rule-based step** (strip question words, pull
capitalised terms/years as search terms) instead of a 6th LLM call per question.

Why: you're running Llama-2-7B locally (per your baseline report) for 300+ questions,
maybe more once you add MCQ + open-ended + retries. Every extra LLM call in the
pipeline multiplies your total runtime and is a real engineering cost the brief
explicitly wants you to acknowledge (§4.8, "computational cost, latency... should be
acknowledged"). Query analysis is the one step that genuinely doesn't need an LLM to
do a decent job, so cutting it there is the cheapest place to save calls without
touching the parts that actually decide correctness/confidence. Worth a sentence in
your report's Threats to Validity / trade-offs section.

Everything else in the diagram (retrieval verifier, critic, confidence calculator,
retry loops) is implemented as designed.

## What's real vs. what you need to plug in

- **Pipeline logic, retrieval, agents, evaluation harness, metrics: fully implemented
  and runnable.**
- **`data/chunks/`: empty.** I don't have your scraped SAHO corpus (only the
  benchmark Q&A pairs, which reference `source_document` filenames but not the
  document text). Drop your chunked `.md`/`.txt` files in there — see
  `data/chunks/README.md` for the expected format. Until you do, the vector store
  has nothing to retrieve from.
- **LLM endpoint: configured for LM Studio's local OpenAI-compatible server**
  (`http://localhost:1234/v1`), matching your baseline setup. Change `config/config.yaml`
  if you're pointing at something else.
- **Retrieval: TF-IDF (scikit-learn), not embeddings.** No external model download is
  needed (works fully offline, which matters since you're already fighting with local
  models), and it's a perfectly defensible retrieval strategy to report and justify
  experimentally. If you want to swap in sentence-transformer embeddings later, only
  `retrieval/vector_store.py` needs to change — the interface stays the same.

## Setup

```bash
pip install -r requirements.txt
```

Start LM Studio, load your model, start the local server (default port 1234).

Drop your chunked SAHO documents into `data/chunks/` (one file per chunk, or a single
`chunks.jsonl` — see `data/chunks/README.md`).

## Running

Single question, ad hoc:
```bash
python main.py ask "In which year did massive national school boycotts erupt in the townships?"
```

Full benchmark run (produces results in the same format as your baseline, for a fair
Phase 5 comparison):
```bash
python main.py benchmark --mcq ../benchmark/mcq_benchmark_300.jsonl --out results/mcq_agent_results.jsonl
python main.py benchmark --open ../benchmark/open_benchmark_300.jsonl --out results/open_agent_results.jsonl
```

Then:
```bash
python -m evaluation.metrics results/mcq_agent_results.jsonl --type mcq
python -m evaluation.metrics results/open_agent_results.jsonl --type open
```

## About your existing MCQ benchmark file

Heads up (separate from this build): `benchmark_300.jsonl` / `mcq_benchmark_300.jsonl`
has 286/300 gold answers = "A", and your baseline parser silently defaulted
unparseable model output to "A" too — together they made the reported 82.3% baseline
meaningless (a same-questions "always guess A" strategy scores 95.3%). I've included
`tools/fix_mcq_bias.py`, which shuffles each question's option order (and updates the
gold letter to match) without changing any question text or correct answer content —
so it doesn't count as altering the benchmark's content, just de-biasing its format.
Re-run your baseline on the fixed file before doing the Phase 5 comparison, or the
agent-vs-baseline comparison will inherit the same flaw.
