"""TF-IDF ranking with cosine similarity between the query and the documents."""

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


def search(query, index, idf, doc_norms, top_k=None):
    """AND search followed by tf-idf ranking; returns the top_k [(pid, score)]."""
    pids = conjunctive_search(query, index)
    ranking = rank_tfidf(query, pids, index, idf, doc_norms)
    return ranking[:top_k] if top_k else ranking
