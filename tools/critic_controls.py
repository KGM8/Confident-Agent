"""
critic_controls.py -- positive and negative controls for the critic.

The audit showed the critic never returned PASS, even for correct answers whose supporting
text was in the retrieved excerpts. That is consistent with two very different explanations:
  (a) the critic (prompt + model) cannot certify ANYTHING, even a perfectly supported answer, or
  (b) it can, but real answers are messy (paraphrase, preamble, source disagreement) and it balks.
Controls separate them. For questions whose supporting text really is in the retrieved excerpts:
  POSITIVE control: the answer is the GOLD text itself (verbatim in the excerpts). A working
                    verifier should return PASS.
  NEGATIVE control: the answer is a value that does NOT appear in the excerpts (a wrong option,
                    or another question's gold). A working verifier should NOT return PASS.
The critic prompt and settings are exactly those used in the pipeline. Nothing is tuned.

Needs LM Studio running with the model loaded (about 40 critic calls at the default --n 10,
roughly 1.5 hours on a CPU laptop). Safe to Ctrl+C and re-run: finished controls are skipped.

    python tools/critic_controls.py
    python tools/critic_controls.py --n 2          (smoke test)
    python tools/critic_controls.py --summarize-only
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import critic_audit as ca  # noqa: E402
from agents.verifier import (SYSTEM_PROMPT, build_critic_prompt, parse_verdict_legacy,  # noqa: E402
                             parse_verdict_robust)


def build_controls(agent, items, n_per_kind):
    """Pick items whose gold text is inside a retrieved chunk of the right document, and make
    one positive and (where possible) one negative control for each."""
    all_golds = [(it["choices"][it["gold"]] if it["kind"] == "mcq" else it["gold"]) for it in items]
    controls, taken = [], collections.Counter()
    for idx, it in enumerate(items):
        if taken[it["kind"]] >= n_per_kind:
            continue
        chunks = agent.retriever.retrieve(it["question"]).chunks
        expected = (it.get("source_document") or "").replace(".md", "")
        gold_text = it["choices"][it["gold"]] if it["kind"] == "mcq" else it["gold"]
        g = ca.norm(gold_text)
        if not g or not any(c.chunk.source_id == expected and g in ca.norm(c.chunk.text) for c in chunks):
            continue                                   # supporting text not in the excerpts: not a fair control
        taken[it["kind"]] += 1
        context = ca.norm(" ".join(c.chunk.text for c in chunks))
        if it["kind"] == "mcq":
            pos = f"{it['gold']}: {gold_text}"
            wrong = [(k, v) for k, v in it["choices"].items() if k != it["gold"] and ca.norm(v) not in context]
            neg = f"{wrong[0][0]}: {wrong[0][1]}" if wrong else None
        else:
            pos = gold_text
            others = [x for j, x in enumerate(all_golds) if j != idx and ca.norm(x) and ca.norm(x) not in context]
            neg = others[0] if others else None
        controls.append({"kind": it["kind"], "id": it["id"], "type": "positive", "question": it["question"], "answer": pos})
        if neg:
            controls.append({"kind": it["kind"], "id": it["id"], "type": "negative", "question": it["question"], "answer": neg})
    return controls


def run_controls(agent, controls, out_path, max_tokens, resume=True):
    done = set()
    if resume and os.path.exists(out_path):
        done = {(r["kind"], r["id"], r["type"]) for r in ca.load_jsonl(out_path)}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        from tqdm import tqdm
        it_iter = tqdm(controls, desc="critic controls")
    except ImportError:
        it_iter = controls
    with open(out_path, "a" if resume else "w", encoding="utf-8") as f:
        for c in it_iter:
            if (c["kind"], c["id"], c["type"]) in done:
                continue
            chunks = agent.retriever.retrieve(c["question"]).chunks
            raw = agent.llm.complete(SYSTEM_PROMPT, build_critic_prompt(c["question"], chunks, c["answer"]),
                                     temperature=0.0, max_tokens=max_tokens)
            lv, _ = parse_verdict_legacy(raw)
            rv, rs = parse_verdict_robust(raw)
            rec = dict(c, critic_raw=raw, legacy_verdict=lv, robust_verdict=rv, robust_status=rs)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()


def summarize(recs):
    L = ["=" * 78, "CRITIC CONTROLS: does the critic pass the gold answer when its string is in the excerpts?", "=" * 78]
    verdicts = ["PASS", "FAIL", "CONFLICT", "UNPARSEABLE"]
    L.append(f"{'control':26s}" + "".join(f"{v:>13s}" for v in verdicts) + f"{'total':>8s}")
    for typ, label in (("positive", "POSITIVE (gold string present)"), ("negative", "NEGATIVE (value absent)")):
        for scope, sub in [("all", [r for r in recs if r["type"] == typ])] + \
                          [(k, [r for r in recs if r["type"] == typ and r["kind"] == k]) for k in ("mcq", "open")]:
            if not sub:
                continue
            c = collections.Counter(r["robust_verdict"] for r in sub)
            L.append(f"{label + ' ' + scope:26s}" + "".join(f"{c.get(v, 0):>13d}" for v in verdicts) + f"{len(sub):>8d}")
    pos = [r for r in recs if r["type"] == "positive"]
    neg = [r for r in recs if r["type"] == "negative"]
    if pos and neg:
        pp = sum(r["robust_verdict"] == "PASS" for r in pos) / len(pos)
        np_ = sum(r["robust_verdict"] == "PASS" for r in neg) / len(neg)
        pu = sum(r["robust_verdict"] == "UNPARSEABLE" for r in pos) / len(pos)
        L.append(f"\nPASS rate: positive controls {pp:.0%}, negative controls {np_:.0%}.  Unparseable replies on positive controls: {pu:.0%}.")
        if pp == 0:
            L.append("Reading: the critic never passes the gold answer even when the gold string is in its excerpts. String presence is not proof of support; check support by hand (tools/control_support_check.py).")
        elif pp > np_:
            L.append("Reading: the critic CAN pass clean supported answers and discriminates; its failures on real answers come from messy answers.")
        else:
            L.append("Reading: the critic passes unsupported answers as readily as supported ones -> no discrimination.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--sample-mcq", default="mcq_sample30.jsonl")
    ap.add_argument("--sample-open", default="open_sample30.jsonl")
    ap.add_argument("--agent-mcq", default="results/mcq_agent_results.jsonl")
    ap.add_argument("--agent-open", default="results/open_agent_results.jsonl")
    ap.add_argument("--out", default="results/critic_controls.jsonl")
    ap.add_argument("--n", type=int, default=10, help="controls per benchmark (each gives a positive and a negative)")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    args.only, args.limit = None, None

    if not args.summarize_only:
        from main import build_agent, load_config
        cfg = load_config(args.config)
        agent = build_agent(cfg)
        controls = build_controls(agent, ca.collect_items(args), args.n)
        print(f"{len(controls)} control cases ({sum(c['type'] == 'positive' for c in controls)} positive, "
              f"{sum(c['type'] == 'negative' for c in controls)} negative)")
        run_controls(agent, controls, args.out, cfg["llm"]["max_tokens_critic"])
    text = summarize(ca.load_jsonl(args.out))
    print("\n" + text)
    with open(os.path.splitext(args.out)[0] + "_summary.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
