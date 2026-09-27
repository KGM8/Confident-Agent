#!/usr/bin/env python3
"""
run_eval.py

Evaluates a local GGUF model (e.g. Llama 2 7B Chat) against benchmark_300.jsonl,
entirely in-process via llama-cpp-python -- no server needed.

Setup (run once, inside your activated venv):
    uv pip install llama-cpp-python

Usage:
    python run_eval.py --model-path /path/to/llama-2-7b-chat.Q4_K_M.gguf

Optional:
    python run_eval.py --model-path ... --benchmark benchmark_300.jsonl \
        --output results.jsonl --n-ctx 4096 --limit 300
"""

import argparse
import json
import re
import string
import sys
import time

SYSTEM_PROMPT_MCQ = (
    "You are taking a multiple-choice exam on South African history and "
    "public figures. For each question, respond with ONLY the single "
    "letter (A, B, C, or D) of the correct answer. Do not explain, do not "
    "repeat the question, output nothing but the letter."
)

SYSTEM_PROMPT_OPEN = (
    "You are taking a short-answer exam on South African history and "
    "public figures. For each question, respond with ONLY the short "
    "factual answer (a few words). Do not explain, do not repeat the "
    "question, do not write a full sentence."
)

# Llama 2 chat prompt format
PROMPT_TEMPLATE = "[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"


def build_user_message_mcq(item):
    lines = [item["question"], ""]
    for letter in ("A", "B", "C", "D"):
        lines.append(f"{letter}. {item['choices'][letter]}")
    lines.append("\nAnswer:")
    return "\n".join(lines)


def build_user_message_open(item):
    return f"{item['question']}\n\nAnswer (a few words only):"


def extract_letter(generated_text):
    match = re.search(r"\b([ABCD])\b", generated_text.strip().upper())
    if match:
        return match.group(1)
    # fallback: first character that's A-D
    for ch in generated_text.strip().upper():
        if ch in "ABCD":
            return ch
    return None


_STOPWORDS = {"a", "an", "the"}


def normalize_text(s):
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    words = [w for w in s.split() if w not in _STOPWORDS]
    return " ".join(words)


def short_answer_correct(gold, generated):
    # take just the first line of the generation, in case the model rambles
    generated = generated.strip().split("\n")[0]
    g = normalize_text(gold)
    p = normalize_text(generated)
    if not g or not p:
        return False
    return g in p or p in g


def load_benchmark(path, limit=None):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    if limit:
        items = items[:limit]
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to the .gguf file")
    parser.add_argument("--benchmark", default="benchmark_300.jsonl")
    parser.add_argument("--output", default="results.jsonl")
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-gpu-layers", type=int, default=0, help="Set >0 to offload layers to GPU if you have one set up")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N questions (useful for a quick test run)")
    args = parser.parse_args()

    try:
        from llama_cpp import Llama
    except ImportError:
        print("ERROR: llama-cpp-python is not installed. Run: uv pip install llama-cpp-python", file=sys.stderr)
        sys.exit(1)

    print(f"Loading benchmark from {args.benchmark} ...")
    items = load_benchmark(args.benchmark, args.limit)
    print(f"  {len(items)} questions loaded")

    print(f"Loading model from {args.model_path} (this can take a minute)...")
    llm = Llama(
        model_path=args.model_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        verbose=False,
    )

    correct = 0
    unparseable = 0
    start = time.time()

    mode = "mcq" if items and "choices" in items[0] else "open"
    print(f"Detected benchmark format: {mode}")

    with open(args.output, "w", encoding="utf-8") as out_f:
        for i, item in enumerate(items, 1):
            if mode == "mcq":
                user_msg = build_user_message_mcq(item)
                system_prompt = SYSTEM_PROMPT_MCQ
            else:
                user_msg = build_user_message_open(item)
                system_prompt = SYSTEM_PROMPT_OPEN

            prompt = PROMPT_TEMPLATE.format(system=system_prompt, user=user_msg)
            response = llm(
                prompt,
                max_tokens=10 if mode == "mcq" else 30,
                temperature=0.0,
                stop=["</s>", "[INST]"],
            )
            generated = response["choices"][0]["text"]
            gold = item["answer"]

            if mode == "mcq":
                predicted = extract_letter(generated)
                is_correct = predicted == gold
                if predicted is None:
                    unparseable += 1
            else:
                predicted = generated.strip().split("\n")[0]
                is_correct = short_answer_correct(gold, generated)

            if is_correct:
                correct += 1

            record = {
                "id": item.get("id", i),
                "question": item["question"],
                "gold": gold,
                "predicted": predicted,
                "raw_output": generated.strip(),
                "correct": is_correct,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            if i % 20 == 0 or i == len(items):
                elapsed = time.time() - start
                running_acc = correct / i
                print(f"  [{i}/{len(items)}] running accuracy: {running_acc:.1%}  ({elapsed:.0f}s elapsed)")

    accuracy = correct / len(items) if items else 0.0
    print("\n" + "=" * 50)
    print(f"Model:    {args.model_path}")
    print(f"Format:   {mode}")
    print(f"Total:    {len(items)}")
    print(f"Correct:  {correct}")
    print(f"Accuracy: {accuracy:.1%}")
    if mode == "mcq" and unparseable:
        print(f"Note: {unparseable} responses didn't contain a clear A/B/C/D and were marked incorrect.")
    if mode == "open":
        print("Note: open-ended scoring uses lenient substring matching after normalization —")
        print("      spot-check a sample of results.jsonl to sanity-check the matching.")
    print(f"Per-question results written to {args.output}")
    print("=" * 50)


if __name__ == "__main__":
    main()
