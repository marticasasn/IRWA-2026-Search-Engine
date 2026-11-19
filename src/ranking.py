"""Ranking of the AND search results: TF-IDF cosine similarity and BM25."""

import math
from collections import Counter

from src.indexing import conjunctive_search
from src.preprocessing import build_terms


def rank_tfidf(query, pids, index, idf, doc_norms):
    """Ranks the given pids by cosine similarity of tf-idf vectors; returns [(pid, score)]."""
    query_tf = Counter(t for t in build_terms(query) if t in index)
    query_vec = {t: tf * idf[t] for t, tf in query_tf.items()}
    query_norm = math.sqrt(sum(w ** 2 for w in query_vec.values()))

    scores = []
    for pid in pids:
        dot = sum(w * index[t].get(pid, 0) * idf[t] for t, w in query_vec.items())
        denom = query_norm * doc_norms.get(pid, 0)
        scores.append((pid, dot / denom if denom > 0 else 0.0))
    # Highest score first, pid breaks ties so the order is always the same
    return sorted(scores, key=lambda x: (-x[1], x[0]))


def rank_bm25(query, pids, index, idf_bm25, doc_lens, avg_len, k1=1.2, b=0.75):
    """Ranks the given pids by BM25 score; returns [(pid, score)]."""
    terms = set(t for t in build_terms(query) if t in index)

    scores = []
    for pid in pids:
        # Length normalisation: long documents need a higher tf to get the same score
        norm = k1 * (1 - b + b * doc_lens.get(pid, 0) / avg_len)
        score = 0.0
        for t in terms:
            tf = index[t].get(pid, 0)
            score += idf_bm25[t] * tf * (k1 + 1) / (tf + norm)
        scores.append((pid, score))
    return sorted(scores, key=lambda x: (-x[1], x[0]))


def search(query, index, idf, doc_norms, top_k=None):
    """AND search followed by tf-idf ranking; returns the top_k [(pid, score)]."""
    pids = conjunctive_search(query, index)
    ranking = rank_tfidf(query, pids, index, idf, doc_norms)
    return ranking[:top_k] if top_k else ranking
