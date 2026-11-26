"""Evaluation of the rankings (tfidf, bm25, custom, custom_title) on the queries with human labels.

Labels are merged from:
- validation_labels.csv (Part 2 validation queries, query text in VALIDATION_QUERIES),
- test_queries_results.csv (Part 2 test queries),
- docs/unlabelled_for_review.csv (rows we label by hand; empty labels are ignored).

For every query and method the top 10 of search_ranked (Part 1 index, AND search, as in Part 3) is
evaluated: P@5, P@10, AP@10 (mean = MAP@10) and nDCG@10, with src/evaluation.py. Results without a
label count as not relevant and are listed in docs/unlabelled_for_review.csv to be labelled. Rows already
in that file are always kept (they hold our labels); only new (query, pid) pairs are appended.
n_relevant of a query = number of results labelled relevant for it (all sources), as in Part 2.

Run from the repository root: python scripts/eval_rankings.py
"""

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.engine import load_raw_documents  # noqa: E402
from src.evaluation import avg_precision_at_k, ndcg_at_k, precision_at_k  # noqa: E402
from src.indexing import build_search_data, load_corpus  # noqa: E402
from src.ranking import CUSTOM_TITLE_WEIGHT, CUSTOM_WEIGHTS, search_ranked  # noqa: E402

VALIDATION_QUERIES = {1: "women full sleeve sweatshirt cotton", 2: "men slim jeans blue"}  # as in Part 2
METHODS = ["tfidf", "bm25", "custom", "custom_title"]
K = 10
VALIDATION_PATH = os.path.join(ROOT, "validation_labels.csv")
TEST_PATH = os.path.join(ROOT, "test_queries_results.csv")
REVIEW_PATH = os.path.join(ROOT, "docs", "unlabelled_for_review.csv")
METRICS_PATH = os.path.join(ROOT, "docs", "part4_ranking_metrics.md")
REVIEW_COLUMNS = ["query", "pid", "rank", "method", "title", "brand", "category", "link", "label"]


def load_labels():
    """Returns ({(query, pid): 0/1}, ordered list of queries, number of conflicting labels).

    Labels from the original files win over the review file if both label the same (query, pid).
    """
    validation = pd.read_csv(VALIDATION_PATH, dtype={"pid": str}, encoding="utf-8-sig")
    validation["query"] = validation["query_id"].map(VALIDATION_QUERIES)
    test = pd.read_csv(TEST_PATH, dtype={"pid": str}, encoding="utf-8-sig")
    original = pd.concat([validation.rename(columns={"labels": "label"})[["query", "pid", "label"]],
                          test[["query", "pid", "label"]]])
    labels = {(q, p): int(l) for q, p, l in original.itertuples(index=False)}
    queries = list(dict.fromkeys(list(test["query"]) + list(VALIDATION_QUERIES.values())))

    conflicts = 0
    review = load_review()
    for q, p, l in review[["query", "pid", "label"]].itertuples(index=False):
        if pd.isna(l) or str(l).strip() == "":
            continue
        value = int(float(l))
        if value not in (0, 1):
            raise ValueError(f"Label must be 0 or 1, got {l!r} for ({q}, {p})")
        if (q, p) in labels:
            conflicts += labels[(q, p)] != value
            continue
        labels[(q, p)] = value
    return labels, queries, conflicts


def load_review():
    """The review CSV (empty table if it does not exist yet)."""
    if not os.path.exists(REVIEW_PATH):
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    return pd.read_csv(REVIEW_PATH, dtype={"pid": str, "label": str}, keep_default_na=False, encoding="utf-8-sig")


def evaluate(data, labels, queries):
    """Per query and method: metrics, top-10 pids and unlabelled results. Returns (rows, unlabelled)."""
    n_relevant = {q: sum(l for (lq, _), l in labels.items() if lq == q) for q in queries}
    rows, unlabelled = [], []
    for query in queries:
        for method in METHODS:
            top = [pid for pid, _ in search_ranked(query, method, data, top_k=K)]
            rels = [labels.get((query, pid), 0) for pid in top]
            missing = [(rank, pid) for rank, pid in enumerate(top, start=1) if (query, pid) not in labels]
            unlabelled += [(query, pid, rank, method) for rank, pid in missing]
            rels += [0] * (K - len(rels))  # fewer than K results: the missing positions are not relevant
            rows.append({
                "query": query, "method": method, "results": len(top), "unlabelled": len(missing),
                "n_relevant": n_relevant[query],
                "P@5": precision_at_k(rels, 5), "P@10": precision_at_k(rels, 10),
                "AP@10": avg_precision_at_k(rels, K, n_relevant[query]),
                "nDCG@10": ndcg_at_k(rels, K, n_relevant[query]),
            })
    return pd.DataFrame(rows), unlabelled


def update_review(unlabelled, documents):
    """Appends the new unlabelled (query, pid) pairs to the review CSV, keeping every existing row.

    A pid returned by several methods gets one row with the methods and ranks joined by ";".
    Returns the number of rows added.
    """
    review = load_review()
    known = set(zip(review["query"], review["pid"]))
    grouped = {}
    for query, pid, rank, method in unlabelled:
        grouped.setdefault((query, pid), []).append((method, rank))
    new_rows = []
    for (query, pid), hits in grouped.items():
        if (query, pid) in known:
            continue
        doc = documents.get(pid, {})
        new_rows.append({
            "query": query, "pid": pid,
            "rank": ";".join(str(r) for _, r in hits), "method": ";".join(m for m, _ in hits),
            "title": " ".join((doc.get("title") or "").split()), "brand": doc.get("brand") or "",
            "category": " / ".join(x for x in (doc.get("category"), doc.get("sub_category")) if x),
            "link": doc.get("url") or "", "label": "",
        })
    if new_rows:
        review = pd.concat([review, pd.DataFrame(new_rows, columns=REVIEW_COLUMNS)], ignore_index=True)
    os.makedirs(os.path.dirname(REVIEW_PATH), exist_ok=True)
    review[REVIEW_COLUMNS].to_csv(REVIEW_PATH, index=False, encoding="utf-8-sig")
    return len(new_rows)


def markdown_table(df, float_cols):
    """Markdown table of a DataFrame (floats with 3 decimals, numbers right aligned)."""
    cols = list(df.columns)
    align = ["---:" if pd.api.types.is_numeric_dtype(df[c]) else "---" for c in cols]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(align) + " |"]
    for row in df.itertuples(index=False):
        cells = [f"{v:.3f}" if c in float_cols else str(v) for c, v in zip(cols, row)]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_metrics(rows, n_queries, n_labels, conflicts):
    """Writes docs/part4_ranking_metrics.md: summary per method and the per-query values."""
    metrics = ["P@5", "P@10", "AP@10", "nDCG@10"]
    summary = rows.groupby("method", sort=False).agg(
        **{"P@5": ("P@5", "mean"), "P@10": ("P@10", "mean"), "MAP@10": ("AP@10", "mean"),
           "nDCG@10": ("nDCG@10", "mean"), "unlabelled in top 10": ("unlabelled", "sum"),
           "results in top 10": ("results", "sum")}).reset_index()
    per_query = rows[["query", "method", "results", "unlabelled", "n_relevant"] + metrics]

    lines = ["# Part 4: ranking metrics on the labelled queries", "",
             "Generated by `scripts/eval_rankings.py`.", "",
             f"- Queries: {n_queries} (5 Part 2 test queries + 2 validation queries). Labels used: {n_labels}"
             f" (conflicting labels in the review file ignored: {conflicts}).",
             "- Rankings: `search_ranked` on the Part 1 index (AND search), top 10.",
             f"- custom: weights {dict((k, v) for k, v in CUSTOM_WEIGHTS.items())}; custom_title: same with "
             f"title weight {CUSTOM_TITLE_WEIGHT}.",
             "- Unlabelled results count as not relevant; they are listed in `docs/unlabelled_for_review.csv`.",
             "- AP@10 and nDCG@10 use n_relevant = number of results labelled relevant for the query (Part 2).",
             "", "## Mean over the queries", ""]
    lines += markdown_table(summary, {"P@5", "P@10", "MAP@10", "nDCG@10"})
    lines += ["", "## Per query", ""]
    lines += markdown_table(per_query, set(metrics))
    with open(METRICS_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return summary


def main():
    """Evaluates every method, updates the review CSV and writes the metrics table."""
    data = build_search_data(load_corpus(os.path.join(ROOT, "processed_corpus.json")))
    documents = load_raw_documents(os.path.join(ROOT, "fashion_products_dataset.json"))
    labels, queries, conflicts = load_labels()
    rows, unlabelled = evaluate(data, labels, queries)
    added = update_review(unlabelled, documents)
    summary = write_metrics(rows, len(queries), len(labels), conflicts)
    print(summary.round(3).to_string(index=False))
    print(f"Unlabelled (query, pid) pairs added to the review file: {added}")
    print("Saved", METRICS_PATH, "and", REVIEW_PATH)


if __name__ == "__main__":
    main()
