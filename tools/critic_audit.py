"""
critic_audit.py -- release-stage audit of the Confident Agent's critic.

Question being answered: the agent abstained on all 60 evaluation questions and
the critic returned zero PASS verdicts. Is that (a) an interface/parsing defect,
(b) a prompt problem, (c) an over-restrictive release policy, or (d) a genuine
inability of the critic to establish support?

Method: the 60 saved answers are held FIXED (30 MCQ + 30 open-ended, including
the ones whose content was correct). For each, the critic is called once with
exactly the prompt the pipeline uses, its RAW reply is logged, and that single
reply is read by BOTH parsers:
    legacy  = the original condition (verdict only if a line starts "VERDICT:")
    robust  = the corrected condition (label found anywhere, explicit status)
Because only the verdict reader differs, any change in release decisions is
attributable to the parser alone. No confidence threshold is touched.

Outputs a JSONL of raw replies and a summary with, for each parser, the release
confusion matrix against content-correctness (released/withheld x correct/incorrect).

Run from the project root with LM Studio running (about 1 critic call per item,
60 calls; expect a couple of hours on a CPU laptop). Safe to Ctrl+C and re-run:
finished items are skipped.

    python tools/critic_audit.py
    python tools/critic_audit.py --limit 4          (quick smoke test)
    python tools/critic_audit.py --summarize-only   (re-print the summary)
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.verifier import (SYSTEM_PROMPT, build_critic_prompt, find_ungrounded_years,  # noqa: E402
                             parse_verdict_legacy, parse_verdict_robust)
from pipeline.confident_agent import _parse_mcq_answer  # noqa: E402


def norm(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def collect_items(args):
    items = []
    for kind, sample_path, res_path in (("mcq", args.sample_mcq, args.agent_mcq),
                                         ("open", args.sample_open, args.agent_open)):
        if args.only and kind != args.only:
            continue
        if not (os.path.exists(sample_path) and os.path.exists(res_path)):
            print(f"WARNING: missing {sample_path} or {res_path}; skipping {kind}")
            continue
        sample = {r["id"]: r for r in load_jsonl(sample_path)}
        res = {r["id"]: r for r in load_jsonl(res_path)}
        for i in sorted(set(sample) & set(res)):
            s, r = sample[i], res[i]
            items.append({"kind": kind, "id": i, "question": s["question"], "gold": r["gold"],
                          "answer": r["raw_output"], "choices": s.get("choices"),
                          "source_document": s.get("source_document"),
                          "saved_predicted": r.get("predicted"),
                          "original_decision": r.get("decision"),
                          "original_verdict": r.get("critic_verdict")})
    if args.limit:
        by_kind = collections.defaultdict(list)
        for it in items:
            by_kind[it["kind"]].append(it)
        items = [it for k in by_kind for it in by_kind[k][:args.limit]]
    return items


def audit_one(agent, it, max_tokens):
    retrieval = agent.retriever.retrieve(it["question"])
    if it["kind"] == "mcq":
        letter = _parse_mcq_answer(it["answer"], it["choices"])
        critic_input = f"{letter}: {it['choices'][letter]}" if letter else it["answer"]
        content_correct = letter is not None and letter == it["gold"]
    else:
        critic_input = it["answer"]
        content_correct = bool(norm(it["gold"])) and norm(it["gold"]) in norm(it["answer"])

    rec = {"kind": it["kind"], "id": it["id"], "gold": it["gold"], "critic_input": critic_input,
           "content_correct": content_correct, "original_decision": it["original_decision"],
           "original_verdict": it["original_verdict"]}
    if not retrieval.chunks:
        rec.update(critic_raw="", legacy_verdict="FAIL", legacy_status="no_chunks",
                   robust_verdict="UNPARSEABLE", robust_status="no_chunks",
                   ungrounded_years=[], release_legacy=False, release_robust=False)
        return rec
    raw = agent.llm.complete(SYSTEM_PROMPT, build_critic_prompt(it["question"], retrieval.chunks, critic_input),
                             temperature=0.0, max_tokens=max_tokens)
    lv, ls = parse_verdict_legacy(raw)
    rv, rs = parse_verdict_robust(raw)
    ung = find_ungrounded_years(critic_input, retrieval.chunks)
    rec.update(critic_raw=raw, legacy_verdict=lv, legacy_status=ls, robust_verdict=rv, robust_status=rs,
               ungrounded_years=ung, release_legacy=(lv == "PASS" and not ung),
               release_robust=(rv == "PASS" and not ung))
    return rec


def run_audit(agent, items, out_path, max_tokens, resume=True):
    done = set()
    if resume and os.path.exists(out_path):
        done = {(r["kind"], r["id"]) for r in load_jsonl(out_path)}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        from tqdm import tqdm
        it_iter = tqdm(items, desc="critic audit")
    except ImportError:
        it_iter = items
    with open(out_path, "a" if resume else "w", encoding="utf-8") as f:
        for it in it_iter:
            if (it["kind"], it["id"]) in done:
                continue
            f.write(json.dumps(audit_one(agent, it, max_tokens), ensure_ascii=False) + "\n")
            f.flush()


def _matrix(recs, key):
    ta = sum(1 for r in recs if r[key] and r["content_correct"])
    fa = sum(1 for r in recs if r[key] and not r["content_correct"])
    fr = sum(1 for r in recs if not r[key] and r["content_correct"])
    tr = sum(1 for r in recs if not r[key] and not r["content_correct"])
    return ta, fa, fr, tr


def summarize(recs):
    n = len(recs)
    lines = ["=" * 78, "CRITIC RELEASE-STAGE AUDIT (saved answers held fixed; only the verdict reader differs)", "=" * 78]
    kinds = collections.Counter(r["kind"] for r in recs)
    correct = sum(r["content_correct"] for r in recs)
    lines.append(f"items audited: {n}  ({dict(kinds)})   content-correct answers among them: {correct}")
    status = collections.Counter(r["robust_status"] for r in recs)
    prefixed = sum(1 for r in recs if r["robust_status"] in ("tolerant", "loose", "ambiguous", "unparseable"))
    lines.append(f"\nHow the critic's raw reply was laid out (robust reader): {dict(status)}")
    lines.append(f"  replies where the VERDICT label was NOT at the very start: {prefixed}/{n}"
                 f"  (the legacy reader can only see a verdict at the very start of a line)")
    lines.append(f"\nVerdicts  legacy: {dict(collections.Counter(r['legacy_verdict'] for r in recs))}")
    lines.append(f"          robust: {dict(collections.Counter(r['robust_verdict'] for r in recs))}")
    lines.append(f"          original batch run (for reference): {dict(collections.Counter(r['original_verdict'] for r in recs))}")
    for label, key in (("LEGACY reader (original condition)", "release_legacy"),
                       ("ROBUST reader (corrected condition)", "release_robust")):
        lines.append(f"\nRelease decisions, {label}  [release = critic PASS and no ungrounded year]")
        for scope, subset in [("all", recs)] + [(k, [r for r in recs if r["kind"] == k]) for k in sorted(kinds)]:
            ta, fa, fr, tr = _matrix(subset, key)
            prec = f"{ta / (ta + fa):.0%}" if ta + fa else "n/a (nothing released)"
            rec_ = f"{ta / (ta + fr):.0%}" if ta + fr else "n/a"
            lines.append(f"  {scope:5s} released-correct {ta:2d} | released-INCORRECT {fa:2d} | withheld-correct {fr:2d} | "
                         f"withheld-incorrect {tr:2d}   precision of released: {prec}; correct answers released: {rec_}")
    pc = [r for r in recs if r["content_correct"]]
    pi = [r for r in recs if not r["content_correct"]]
    if pc and pi:
        a = sum(r["robust_verdict"] == "PASS" for r in pc) / len(pc)
        b = sum(r["robust_verdict"] == "PASS" for r in pi) / len(pi)
        lines.append(f"\nDiscrimination (robust reader): PASS on {a:.0%} of content-correct answers vs {b:.0%} of incorrect ones.")
    lines.append("\nHow to read this: if legacy shows 0 PASS but robust shows PASS on some items, the zero was at least "
                 "partly a parsing/interface defect. If robust also shows ~0 PASS even on correct answers, the critic "
                 "(prompt + model) genuinely fails to verify. If robust PASSes correct and incorrect answers alike, "
                 "it releases without discriminating (false accepts).")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--sample-mcq", default="mcq_sample30.jsonl")
    ap.add_argument("--sample-open", default="open_sample30.jsonl")
    ap.add_argument("--agent-mcq", default="results/mcq_agent_results.jsonl")
    ap.add_argument("--agent-open", default="results/open_agent_results.jsonl")
    ap.add_argument("--out", default="results/critic_audit.jsonl")
    ap.add_argument("--only", choices=["mcq", "open"])
    ap.add_argument("--limit", type=int, help="items per benchmark (smoke test)")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--reparse", action="store_true",
                    help="with --summarize-only: re-read the stored raw critic replies with the current parsers (no LLM calls)")
    args = ap.parse_args()

    if not args.summarize_only:
        from main import build_agent, load_config
        cfg = load_config(args.config)
        agent = build_agent(cfg)
        items = collect_items(args)
        print(f"{len(items)} saved answers to audit")
        run_audit(agent, items, args.out, cfg["llm"]["max_tokens_critic"])
    recs = load_jsonl(args.out)
    if args.reparse:
        for r in recs:
            r["legacy_verdict"], r["legacy_status"] = parse_verdict_legacy(r.get("critic_raw", ""))
            r["robust_verdict"], r["robust_status"] = parse_verdict_robust(r.get("critic_raw", ""))
            r["release_legacy"] = r["legacy_verdict"] == "PASS" and not r["ungrounded_years"]
            r["release_robust"] = r["robust_verdict"] == "PASS" and not r["ungrounded_years"]
        with open(os.path.splitext(args.out)[0] + "_reparsed.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    text = summarize(recs)
    print("\n" + text)
    with open(os.path.splitext(args.out)[0] + "_summary.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
