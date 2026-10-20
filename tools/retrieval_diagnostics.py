"""
retrieval_diagnostics.py -- was there evidence for the critic to verify?

A verifier can only release an answer that the retrieved excerpts SUPPORT. If the
retriever did not surface the supporting passage, a correct answer was either a lucky
guess or came from the model's own memory, and a grounded critic is RIGHT to withhold
it. This tool measures how often that is the case, using the agent's own retriever and
settings. No LLM calls (LM Studio can be closed); takes a minute or two.

For each of the 60 evaluation questions it records:
  doc_hit              the question's source document is among the top-4 retrieved chunks
  gold_in_right_doc    the gold answer text appears inside a retrieved chunk OF THAT DOCUMENT
                       (the strongest sign that supporting evidence reached the critic)
  gold_in_context      the gold answer text appears anywhere in the retrieved excerpts
and, if results/critic_audit.jsonl exists, joins the critic's verdict to those flags.

    python tools/retrieval_diagnostics.py
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
from pipeline.confident_agent import _parse_mcq_answer  # noqa: E402


def diagnose(agent, items):
    recs = []
    for it in items:
        chunks = agent.retriever.retrieve(it["question"]).chunks
        expected = (it.get("source_document") or "").replace(".md", "")
        gold_text = it["choices"][it["gold"]] if it["kind"] == "mcq" else it["gold"]
        g = ca.norm(gold_text)
        if it["kind"] == "mcq":
            letter = _parse_mcq_answer(it["answer"], it["choices"])
            correct = letter is not None and letter == it["gold"]
        else:
            correct = bool(g) and g in ca.norm(it["answer"])
        recs.append({
            "kind": it["kind"], "id": it["id"], "content_correct": correct,
            "doc_hit": expected in [c.chunk.source_id for c in chunks],
            "gold_in_right_doc": bool(g) and any(c.chunk.source_id == expected and g in ca.norm(c.chunk.text) for c in chunks),
            "gold_in_context": bool(g) and g in ca.norm(" ".join(c.chunk.text for c in chunks)),
            "top_score": round(chunks[0].score, 3) if chunks else 0.0,
            "retrieved_sources": [c.chunk.source_id for c in chunks],
        })
    return recs


def summarize(recs, audit=None):
    n = len(recs)
    kinds = sorted({r["kind"] for r in recs})
    L = ["=" * 78, "RETRIEVAL DIAGNOSTICS: did the gold answer string reach the critic? (string presence, not verified support)", "=" * 78]

    def rate(flag, subset):
        return f"{sum(r[flag] for r in subset)}/{len(subset)}"
    for flag, label in (("doc_hit", "right source document among the retrieved chunks"),
                        ("gold_in_right_doc", "gold string inside a retrieved chunk of that document"),
                        ("gold_in_context", "gold string anywhere in the retrieved excerpts")):
        per = ", ".join(f"{k} {rate(flag, [r for r in recs if r['kind'] == k])}" for k in kinds)
        L.append(f"  {label:58s} {rate(flag, recs):>7s}   ({per})")
    corr = [r for r in recs if r["content_correct"]]
    wrong = [r for r in recs if not r["content_correct"]]
    with_ev = [r for r in corr if r["gold_in_right_doc"]]
    L.append(f"\nAmong the {len(corr)} content-correct answers:")
    L.append(f"  gold string WAS in the retrieved excerpts: {len(with_ev)}    right document retrieved: {sum(r['doc_hit'] for r in corr)}")
    L.append(f"  correct but gold string NOT retrieved: {len(corr) - len(with_ev)}  <- right by luck or memory; a grounded critic should withhold these")
    L.append(f"  => at most {len(with_ev)} of {len(corr)} correct answers had the gold string available (an upper bound: presence is not support).")
    L.append(f"Among the {len(wrong)} content-wrong answers: gold string in the excerpts for {sum(r['gold_in_right_doc'] for r in wrong)} "
             f"(the answer was wrong although the gold string was available)")
    if audit:
        verdicts = ["PASS", "FAIL", "CONFLICT", "UNPARSEABLE"]
        key = {(a["kind"], a["id"]): a for a in audit}
        L.append("\nCritic verdict (robust reader) by whether the gold string was in the excerpts:")
        L.append(f"   {'':34s}" + "".join(f"{v:>13s}" for v in verdicts) + f"{'total':>8s}")
        for label, flag in (("gold string in excerpts", True), ("gold string not in excerpts", False)):
            sub = [r for r in recs if r["gold_in_right_doc"] == flag and (r["kind"], r["id"]) in key]
            c = collections.Counter(key[(r["kind"], r["id"])]["robust_verdict"] for r in sub)
            L.append(f"   {label:34s}" + "".join(f"{c.get(v, 0):>13d}" for v in verdicts) + f"{len(sub):>8d}")
    L.append("\nThe content-correct answers one by one:")
    for r in corr:
        v = ""
        if audit:
            a = {(x["kind"], x["id"]): x for x in audit}.get((r["kind"], r["id"]))
            v = f"  critic: {a['robust_verdict']}" if a else ""
        L.append(f"   [{r['kind']} #{r['id']}] right doc retrieved: {str(r['doc_hit']):5s}  gold string in excerpts: {str(r['gold_in_right_doc']):5s}{v}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--sample-mcq", default="mcq_sample30.jsonl")
    ap.add_argument("--sample-open", default="open_sample30.jsonl")
    ap.add_argument("--agent-mcq", default="results/mcq_agent_results.jsonl")
    ap.add_argument("--agent-open", default="results/open_agent_results.jsonl")
    ap.add_argument("--audit", default="results/critic_audit.jsonl")
    ap.add_argument("--out", default="results/retrieval_diagnostics.jsonl")
    ap.add_argument("--only", choices=["mcq", "open"])
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    from main import build_agent, load_config
    agent = build_agent(load_config(args.config))
    items = ca.collect_items(args)
    recs = diagnose(agent, items)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    audit = ca.load_jsonl(args.audit) if os.path.exists(args.audit) else None
    if audit:
        # Re-read every stored critic reply with the CURRENT robust reader, so this summary always
        # agrees with `critic_audit.py --summarize-only --reparse`. The verdicts saved in
        # critic_audit.jsonl came from the first version of the reader, which mistook an echoed
        # option list ("PASS, CONFLICT, or FAIL") for a PASS verdict.
        from agents.verifier import parse_verdict_robust
        changed = 0
        for a in audit:
            v, st = parse_verdict_robust(a.get("critic_raw", ""))
            changed += v != a.get("robust_verdict")
            a["robust_verdict"], a["robust_status"] = v, st
        print(f"(critic verdicts re-read from the stored raw replies with the current reader; {changed} changed)\n")
    text = summarize(recs, audit)
    print(text)
    with open(os.path.splitext(args.out)[0] + "_summary.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
