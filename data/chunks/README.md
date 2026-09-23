# Expected chunk format

Drop your scraped/chunked SAHO documents here in **either** form:

## Option A — one file per chunk (matches your benchmark's `source_document` field)
```
data/chunks/biography-0102-biography-102.md
data/chunks/article-1125-article-1125.md
...
```
Plain text/markdown content. The filename (minus extension) is used as the chunk id
and citation — this is what lets the pipeline's citations line up with your
benchmark's `source_document` field for scoring/traceability.

## Option B — a single JSONL file
```
data/chunks/chunks.jsonl
```
One JSON object per line:
```json
{"id": "biography-0102-biography-102", "text": "..."}
```

Either format works — `retrieval/vector_store.py` auto-detects which one is present
and loads accordingly. If both are present, the JSONL file takes priority.
