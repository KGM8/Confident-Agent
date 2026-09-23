import argparse
import json
import os

import yaml

from llm_client import LLMClient
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from pipeline.confident_agent import ConfidentAgent
from evaluation.benchmark import run_mcq_benchmark, run_open_benchmark


def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_agent(cfg):
    llm = LLMClient(
        base_url=cfg["llm"]["base_url"],
        api_key=cfg["llm"]["api_key"],
        model=cfg["llm"]["model"],
    )
    store = VectorStore(cfg["retrieval"]["chunks_dir"])
    if store.is_empty():
        print(f"WARNING: no chunks found in {cfg['retrieval']['chunks_dir']}. "
              "The agent will abstain on everything until you add your SAHO "
              "chunks there. See data/chunks/README.md.")
    retriever = Retriever(
        store,
        top_k=cfg["retrieval"]["top_k"],
        good_threshold=cfg["retrieval"]["good_retrieval_threshold"],
        max_reformulations=cfg["retrieval"]["max_reformulations"],
    )
    return ConfidentAgent(llm, retriever, cfg)


def cmd_ask(args, cfg):
    agent = build_agent(cfg)
    resp = agent.answer(args.question)
    print(json.dumps(resp.__dict__, indent=2))


def cmd_benchmark(args, cfg):
    agent = build_agent(cfg)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.mcq:
        run_mcq_benchmark(agent, args.mcq, args.out)
    elif args.open:
        run_open_benchmark(agent, args.open, args.out)
    else:
        raise SystemExit("Provide --mcq or --open with a benchmark file path.")
    print(f"Results written to {args.out}")


def main():
    parser = argparse.ArgumentParser(description="Confident Agent CLI")
    parser.add_argument("--config", default="config/config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    ask_p = sub.add_parser("ask", help="Ask a single question")
    ask_p.add_argument("question")
    ask_p.set_defaults(func=cmd_ask)

    bench_p = sub.add_parser("benchmark", help="Run a benchmark file through the agent")
    bench_p.add_argument("--mcq", help="Path to MCQ benchmark jsonl")
    bench_p.add_argument("--open", help="Path to open-ended benchmark jsonl")
    bench_p.add_argument("--out", required=True, help="Where to write results jsonl")
    bench_p.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    cfg = load_config(args.config)
    args.func(args, cfg)


if __name__ == "__main__":
    main()
