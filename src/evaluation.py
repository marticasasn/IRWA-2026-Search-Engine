"""Evaluation metrics. Each ranking is a list of 0/1 labels in ranked order."""

import numpy as np
import pandas as pd


def precision_at_k(labels, k):
    """Fraction of the top k documents that are relevant."""
    return sum(labels[:k]) / k


def recall_at_k(labels, k, n_relevant=None):
    """Fraction of all the relevant documents found in the top k."""
    n_relevant = sum(labels) if n_relevant is None else n_relevant
    return sum(labels[:k]) / n_relevant if n_relevant > 0 else 0.0


def avg_precision_at_k(labels, k, n_relevant=None):
    """Mean of P@i at the positions i <= k of the relevant documents, divided by all relevant docs."""
    n_relevant = sum(labels) if n_relevant is None else n_relevant
    hits, total = 0, 0.0
    for i, rel in enumerate(labels[:k], start=1):
        if rel:
            hits += 1
            total += hits / i
    return total / n_relevant if n_relevant > 0 else 0.0


def f1_at_k(labels, k, n_relevant=None):
    """Harmonic mean of P@k and R@k."""
    p = precision_at_k(labels, k)
    r = recall_at_k(labels, k, n_relevant)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def reciprocal_rank(labels, k=None):
    """1 / position of the first relevant document (0 if there is none in the top k)."""
    for i, rel in enumerate(labels[:k], start=1):
        if rel:
            return 1 / i
    return 0.0


def dcg_at_k(labels, k):
    """Discounted cumulative gain of the top k documents."""
    labels = np.asarray(labels[:k], dtype=float)
    return float(np.sum((2 ** labels - 1) / np.log2(np.arange(2, len(labels) + 2))))


def ndcg_at_k(labels, k, n_relevant=None):
    """DCG@k divided by the DCG@k of the ideal ranking."""
    n_relevant = sum(labels) if n_relevant is None else n_relevant
    ideal = dcg_at_k([1] * n_relevant, k)
    return dcg_at_k(labels, k) / ideal if ideal > 0 else 0.0


def mean_avg_precision(rankings, k, n_relevants=None):
    """MAP: mean of AP@k over several queries."""
    n_relevants = n_relevants or [None] * len(rankings)
    return float(np.mean([avg_precision_at_k(l, k, n) for l, n in zip(rankings, n_relevants)]))


def mean_reciprocal_rank(rankings, k=None):
    """MRR: mean of the reciprocal rank over several queries."""
    return float(np.mean([reciprocal_rank(l, k) for l in rankings]))


def evaluate_query(labels, k, n_relevant=None):
    """All the per-query metrics at cut-off k, in a dict."""
    return {
        "P@K": precision_at_k(labels, k),
        "R@K": recall_at_k(labels, k, n_relevant),
        "AP@K": avg_precision_at_k(labels, k, n_relevant),
        "F1@K": f1_at_k(labels, k, n_relevant),
        "RR": reciprocal_rank(labels, k),
        "NDCG@K": ndcg_at_k(labels, k, n_relevant),
    }


def evaluate_rankings(rankings, k, n_relevants=None):
    """Table with the metrics of every query plus MAP and MRR (rankings: {query_id: labels})."""
    n_relevants = n_relevants or {}
    rows = {qid: evaluate_query(labels, k, n_relevants.get(qid)) for qid, labels in rankings.items()}
    table = pd.DataFrame(rows).T
    table.loc["mean"] = table.mean()
    table = table.rename(index={"mean": "mean (MAP / MRR)"})
    return table.round(3)


def evaluate_labelled_csv(path, k=20):
    """Reads a results CSV (query_id, rank, label) filled by hand and evaluates it."""
    df = pd.read_csv(path, encoding="utf-8-sig").sort_values(["query_id", "rank"])
    if df["label"].isna().any():
        raise ValueError("There are empty labels in the CSV, fill them first.")
    rankings = {qid: g["label"].astype(int).tolist() for qid, g in df.groupby("query_id")}
    return evaluate_rankings(rankings, k)
