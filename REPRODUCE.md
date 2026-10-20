# Confident Agent (COMP301, 2026): artefact manifest and reproduction guide

Put this file in the project root. Everything listed must be in the submission for the results to be reproducible.

## Environment
- Python 3.12; `pip install -r requirements.txt`
- Baseline also needs llama-cpp-python 0.3.35 (CPU wheel):
  `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`
- Model: `llama-2-7b-chat.Q4_K_S.gguf` (TheBloke/Llama-2-7B-Chat-GGUF on Hugging Face). Not included (about 4 GB).
- Agent: LM Studio local server at http://localhost:1234/v1, model `llama-2-7b-chat`, context 4096. Settings in `config/config.yaml`.
- On an 8 GB machine, close LM Studio before running the baseline.

## Artefacts
| Artefact | Purpose | Produced by |
|---|---|---|
| `saho_dataset/` (zip) | Raw SAHO scrape and cleaned manifests: the frozen source collection | Group scraper |
| `data/chunks/` (13,557 .md files) | Source documents used by the generator and the retriever | Group conversion step |
| `quick_benchmark.py` | Generates the benchmark questions | Group |
| `benchmark_provenance/benchmark_300.jsonl`, `open_benchmark_300.jsonl` | Original benchmarks, unmodified | `quick_benchmark.py` |
| `benchmark_300_options_rebalanced.jsonl` | MCQ with option order reshuffled (seed 42) | `tools/fix_mcq_bias.py` |
| `mcq_sample30.jsonl`, `open_sample30.jsonl` | 30-question evaluation samples (seed 42) | `tools/sample_benchmark.py` |
| `baseline_*_sample30_results.jsonl` | Baseline results (original condition) | `run_eval.py` |
| `baseline_mcq_sample30_v2_results.jsonl` | MCQ baseline with 40-token cap | `run_eval_v2.py` |
| `results/mcq_agent_results.jsonl`, `results/open_agent_results.jsonl` | Agent results (original condition, unchanged) | `main.py benchmark` |
| `results/*_source_manifest.jsonl` | URL, retrieval date, SHA-256 hash, supporting sentence per question | `tools/build_source_manifest.py` |
| `results/final_checks_output.txt` | Parser audit, scoring symmetry, sensitivity analysis | `tools/final_checks.py` |
| `results/critic_audit.jsonl`, `_reparsed.jsonl`, `_summary.txt` | Release-stage audit with raw critic replies | `tools/critic_audit.py` |
| `results/retrieval_diagnostics.jsonl`, `_summary.txt` | Whether the gold string reached the critic | `tools/retrieval_diagnostics.py` |
| `results/critic_controls.jsonl`, `_summary.txt` | Positive and negative critic controls | `tools/critic_controls.py` |
| `Rubric_Scoring_Workbook_final.xlsx` | Two independent human raters, agreed categories, reasons | Group members |

## Commands
See Appendix B of the report for the full ordered list. Quick check without the model: `python -m tests.test_pipeline_smoke` (10 tests).

## Seeds and determinism
Option shuffle and sampling: seed 42. Rubric workbook shuffle: seed 301. All model calls use temperature 0 (greedy).
