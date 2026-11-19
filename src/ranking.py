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


# Default weights of the custom ranking (see rank_custom)
CUSTOM_WEIGHTS = {
    "rating": 0.5,          # weight of the normalised rating
    "discount": 0.2,        # weight of the normalised discount
    "out_of_stock": 0.5,    # multiplier applied to out of stock products
    "neutral_rating": 3.0,  # rating given to products without rating (middle of the 1-5 scale)
}


def quality_factor(rating, discount, out_of_stock, weights=None):
    """Quality factor of a product: (1 + w_r * rating + w_d * discount) * stock penalty.

    rating is mapped from [1, 5] to [0, 1] and discount from [0, 100] to [0, 1].
    """
    w = {**CUSTOM_WEIGHTS, **(weights or {})}
    if rating is None:
        rating = w["neutral_rating"]
    rating_norm = (rating - 1) / 4
    discount_norm = (discount or 0) / 100
    factor = 1 + w["rating"] * rating_norm + w["discount"] * discount_norm
    return factor * w["out_of_stock"] if out_of_stock else factor


def rank_custom(query, pids, index, idf_bm25, doc_lens, avg_len, product_info, weights=None,
                k1=1.2, b=0.75):
    """BM25 score min-max normalised over the results times the quality factor; returns [(pid, score)]."""
    bm25 = rank_bm25(query, pids, index, idf_bm25, doc_lens, avg_len, k1, b)
    if not bm25:
        return []
    high, low = bm25[0][1], bm25[-1][1]
    scores = []
    for pid, score in bm25:
        # With a single score (or all equal) every document gets 1, only the quality decides
        relevance = (score - low) / (high - low) if high > low else 1.0
        scores.append((pid, relevance * quality_factor(*product_info[pid], weights)))
    return sorted(scores, key=lambda x: (-x[1], x[0]))


def search(query, index, idf, doc_norms, top_k=None):
    """AND search followed by tf-idf ranking; returns the top_k [(pid, score)]."""
    pids = conjunctive_search(query, index)
    ranking = rank_tfidf(query, pids, index, idf, doc_norms)
    return ranking[:top_k] if top_k else ranking


def search_ranked(query, method, data, top_k=20, weights=None):
    """AND search followed by the chosen ranking ("tfidf", "bm25" or "custom"); returns the top_k [(pid, score)].

    data is the dict returned by build_search_data. weights is only used by "custom".
    """
    d = data
    pids = conjunctive_search(query, d["index"])
    if method == "tfidf":
        ranking = rank_tfidf(query, pids, d["index"], d["idf"], d["doc_norms"])
    elif method == "bm25":
        ranking = rank_bm25(query, pids, d["index"], d["idf_bm25"], d["doc_lens"], d["avg_len"])
    elif method == "custom":
        ranking = rank_custom(query, pids, d["index"], d["idf_bm25"], d["doc_lens"], d["avg_len"],
                              d["product_info"], weights)
    else:
        raise ValueError(f"Unknown ranking method: {method}")
    return ranking[:top_k] if top_k else ranking
