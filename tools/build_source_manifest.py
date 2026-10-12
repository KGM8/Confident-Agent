"""
Builds a per-question SOURCE MANIFEST for a benchmark file (version 2: matches
against the RAW scrape, because the benchmark was generated from raw data).

For every question it records: source URL, retrieval date, SHA-256 hash of the
source document's full text, and the sentence that establishes the gold answer.

How the join works: chunk filenames (e.g. "biography-0102-biography-102.md") do
NOT reliably identify a source record, so each chunk's TEXT is located inside
the raw documents instead. Documents fetched on or before --generated-on (the
original scrape the questions were generated from) are searched first; the
later re-fetch (Oct 2026) is only used as a fallback and is labelled as such.

Hash note: where the raw files carry a document_hash it equals
SHA-256(full_text); for rows without one the same hash is computed here.

Usage (from the project root; --raw-dir is the saho_dataset folder that holds
article.jsonl, biography.jsonl, etc., NOT the clean/ subfolder):

    python tools/build_source_manifest.py --benchmark open_sample30.jsonl --chunks-dir data/chunks --raw-dir "C:/path/to/saho_dataset" --out results/open_sample30_source_manifest.jsonl
"""
import argparse
import bisect
import collections
import hashlib
import html
import json
import os
import re
import time

RAW_TYPES = ["article", "biography", "place", "archive", "event"]


def norm(text: str) -> str:
    """Lowercase, keep only a-z0-9: immune to HTML entities, curly quotes,
    odd replacement characters, and whitespace/newline differences."""
    return re.sub(r"[^a-z0-9]", "", html.unescape(text).lower())


def doc_name(r):
    j = r.get("jsonld")
    if isinstance(j, list) and j and isinstance(j[0], dict):
        return j[0].get("headline") or j[0].get("name")
    return r.get("title")


class Index:
    def __init__(self):
        self.meta, self.parts, self.starts, self.pos = [], [], [], 0
        self.big = ""

    def add(self, n, meta):
        self.starts.append(self.pos)
        self.parts.append(n)
        self.meta.append(meta)
        self.pos += len(n) + 1

    def finalize(self):
        self.big = "|".join(self.parts)
        self.parts = []

    def doc_norm(self, i):
        end = self.starts[i + 1] - 1 if i + 1 < len(self.starts) else len(self.big)
        return self.big[self.starts[i]:end]

    def find_docs(self, probe, max_hits=20):
        hits, idx = [], self.big.find(probe)
        while idx != -1 and len(hits) < max_hits:
            hits.append(bisect.bisect_right(self.starts, idx) - 1)
            idx = self.big.find(probe, idx + 1)
        return hits


def load_raw(raw_dir, cutoff):
    early, late = Index(), Index()
    late_by_hash = collections.defaultdict(set)
    hash_checked = hash_bad = 0
    for t in RAW_TYPES:
        path = os.path.join(raw_dir, f"{t}.jsonl")
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping {t}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if t == "event":
                    text, url = r.get("body") or "", r.get("url")
                else:
                    text, url = r.get("full_text") or "", r.get("_source_url")
                n = norm(text)
                if not n:
                    continue
                h = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if r.get("document_hash") and t != "event":
                    hash_checked += 1
                    hash_bad += (r["document_hash"] != h)
                fetched = r.get("_fetched_at") or ""
                meta = {"source_url": url, "retrieval_date": fetched, "document_hash": h,
                        "doc_name": doc_name(r), "type": r.get("_type_label") or t}
                if fetched[:10] > cutoff:
                    late_by_hash[h].add(fetched[:10])
                    late.add(n, meta)
                else:
                    early.add(n, meta)
    early.finalize()
    late.finalize()
    return early, late, late_by_hash, hash_checked, hash_bad


def locate(n, idx):
    if len(n) < 40:
        return "too_short", []
    L, P = len(n), min(60, len(n))
    positions = sorted({max(0, min(L - P, int(L * f))) for f in (0.2, 0.5, 0.8)})
    votes = collections.Counter()
    for p in positions:
        for di in set(idx.find_docs(n[p:p + P])):
            votes[di] += 1
    if not votes:
        return "unmatched", []
    best = max(votes.values())
    return "found", sorted(di for di, v in votes.items() if v == best)


def resolve(cands, idx):
    """Group candidates by text hash. Distinct texts -> ambiguous. Identical text
    under one URL (e.g. fetched twice) -> matched. Identical text under several
    URLs -> matched_duplicates. Earliest fetch is chosen."""
    groups = collections.defaultdict(list)
    for c in cands:
        groups[idx.meta[c]["document_hash"]].append(c)
    if len(groups) > 1:
        return "ambiguous", cands
    group = sorted(next(iter(groups.values())),
                   key=lambda c: (idx.meta[c]["retrieval_date"], idx.meta[c]["source_url"] or ""))
    urls = {idx.meta[c]["source_url"] for c in group}
    return ("matched" if len(urls) == 1 else "matched_duplicates"), group


def supporting_passage(text, gold_norm):
    for s in re.split(r"(?<=[.!?])\s+|\n+", html.unescape(text)):
        if gold_norm and gold_norm in norm(s):
            return s.strip()[:500]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--chunks-dir", required=True)
    ap.add_argument("--raw-dir", required=True, help="folder containing article.jsonl, biography.jsonl, ...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--generated-on", default="2026-09-12",
                    help="date the benchmark questions were generated (YYYY-MM-DD)")
    args = ap.parse_args()

    t0 = time.time()
    print("Loading raw scrape (this can take 1-2 minutes)...")
    early, late, late_by_hash, hc, hb = load_raw(args.raw_dir, args.generated_on)
    print(f"  original scrape (fetched on/before {args.generated_on}): {len(early.meta)} documents")
    print(f"  later re-fetch (after {args.generated_on}):               {len(late.meta)} documents")
    print(f"  hash check on rows that carry a document_hash: {hc - hb}/{hc} equal SHA-256(full_text)")
    print(f"  loaded in {time.time() - t0:.0f}s")

    with open(args.benchmark, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]

    out_rows, status_counts = [], collections.Counter()
    gold_in_chunk = gold_in_doc = late_only = 0
    quality = collections.Counter()

    for r in rows:
        is_mcq = "choices" in r
        gold_text = r["choices"][r["answer"]] if is_mcq else r["answer"]
        gold_n = norm(gold_text)
        rec = {
            "id": r["id"], "question": r["question"], "gold": gold_text,
            "source_document": r["source_document"], "match_status": None,
            "source_url": None, "retrieval_date": None, "document_hash": None,
            "doc_name": None, "manifest_type": None,
            "gold_in_chunk": None, "gold_in_matched_doc": None,
            "weak_gold_match": len(gold_n) < 4,
            "matched_in_later_refetch_only": False,
            "match_quality": None, "same_text_refetched_on": None,
            "supporting_passage": None, "candidate_urls": None,
        }
        fname = r["source_document"]
        path = os.path.join(args.chunks_dir, fname if fname.endswith(".md") else fname + ".md")
        if not os.path.exists(path):
            rec["match_status"] = "chunk_file_missing"
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as cf:
                chunk = re.sub(r"^#.*\n+", "", cf.read(), count=1)
            rec["gold_in_chunk"] = bool(gold_n) and gold_n in norm(chunk)
            gold_in_chunk += bool(rec["gold_in_chunk"])

            chunk_n = norm(chunk)
            idx = early
            status, cands = locate(chunk_n, early)
            if status == "unmatched" and late.meta:
                status, cands = locate(chunk_n, late)
                if status == "found":
                    idx, rec["matched_in_later_refetch_only"] = late, True
                    late_only += 1
            if status == "found":
                # chunk files are whole documents, so a candidate whose full text
                # equals the chunk is the true source; others merely share passages
                exact = [c for c in cands if idx.doc_norm(c) == chunk_n]
                rec["match_quality"] = "exact" if exact else "partial"
                status, group = resolve(exact or cands, idx)
                m = idx.meta[group[0]]
                if status != "ambiguous":
                    rec.update(source_url=m["source_url"], retrieval_date=m["retrieval_date"],
                               document_hash=m["document_hash"], doc_name=m["doc_name"],
                               manifest_type=m["type"])
                    rec["same_text_refetched_on"] = sorted(late_by_hash.get(m["document_hash"], []))
                    rec["gold_in_matched_doc"] = bool(gold_n) and gold_n in idx.doc_norm(group[0])
                    gold_in_doc += bool(rec["gold_in_matched_doc"])
                rec["supporting_passage"] = supporting_passage(chunk, gold_n)
                if status != "matched":
                    rec["candidate_urls"] = sorted({(idx.meta[c]["source_url"], idx.meta[c]["retrieval_date"][:10])
                                                    for c in group}, key=str)
            rec["match_status"] = status
        status_counts[rec["match_status"]] += 1
        quality[rec["match_quality"]] += 1
        out_rows.append(rec)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in out_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(rows)
    print(f"\nSource manifest written to {args.out}  ({n} questions)")
    for k, v in status_counts.most_common():
        print(f"  {k:22s} {v}")
    print(f"  chunk text identical to the source document: {quality['exact']}  (partial overlap only: {quality['partial']})")
    print(f"  gold answer found in chunk text:          {gold_in_chunk}/{n}")
    print(f"  gold answer found in matched raw document: {gold_in_doc}/{n}")
    print(f"  matched only in the later (Oct) re-fetch:  {late_only}")
    print("\nReview any row where match_status is not matched/matched_duplicates, "
          "gold_in_chunk is false, or weak_gold_match is true -- those need a human look.")


if __name__ == "__main__":
    main()