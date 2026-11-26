"""Inverted index over the all_terms field of the processed corpus."""

import math
from collections import Counter, defaultdict

import pandas as pd

from src.preprocessing import build_terms


def load_corpus(path="processed_corpus.json"):
    """Loads the processed corpus (JSON Lines) into a DataFrame."""
    return pd.read_json(path, lines=True, dtype={"pid": str})


def build_index(corpus, field="all_terms"):
    """Builds the inverted index: term -> {pid: term frequency}."""
    index = defaultdict(dict)
    for pid, terms in zip(corpus["pid"], corpus[field]):
        for term, count in Counter(terms).items():
            index[term][pid] = count
    return dict(index)


def compute_idf(index, n_docs):
    """Inverse document frequency of every term: log(N / df)."""
    return {term: math.log(n_docs / len(postings)) for term, postings in index.items()}


def compute_doc_norms(index, idf):
    """Length of the tf-idf vector of every document (used for the cosine)."""
    norms = defaultdict(float)
    for term, postings in index.items():
        for pid, tf in postings.items():
            norms[pid] += (tf * idf[term]) ** 2
    return {pid: math.sqrt(value) for pid, value in norms.items()}


def compute_doc_lengths(corpus, field="all_terms"):
    """Number of terms of every document (used for the BM25 length normalisation)."""
    return {pid: len(terms) for pid, terms in zip(corpus["pid"], corpus[field])}


def compute_bm25_idf(index, n_docs):
    """BM25 idf of every term: log((N - df + 0.5) / (df + 0.5) + 1), always positive."""
    return {term: math.log((n_docs - len(postings) + 0.5) / (len(postings) + 0.5) + 1)
            for term, postings in index.items()}


def get_product_info(corpus):
    """Rating, discount and stock of every product: pid -> (rating, discount, out_of_stock).

    Missing ratings and discounts are stored as None.
    """
    info = {}
    for pid, rating, discount, oos in zip(corpus["pid"], corpus["average_rating_num"],
                                          corpus["discount_num"], corpus["out_of_stock"]):
        info[pid] = (None if pd.isna(rating) else float(rating),
                     None if pd.isna(discount) else float(discount),
                     bool(oos))
    return info


def compute_mean_rating(product_info):
    """Mean rating of the products that have one (neutral value for the unrated ones)."""
    ratings = [rating for rating, _, _ in product_info.values() if rating is not None]
    return sum(ratings) / len(ratings)


def build_search_data(corpus, field="all_terms"):
    """Everything the rankers need, computed once: index, idfs, norms, lengths and product info."""
    index = build_index(corpus, field)
    idf = compute_idf(index, len(corpus))
    doc_lens = compute_doc_lengths(corpus, field)
    product_info = get_product_info(corpus)
    return {
        "index": index,
        "idf": idf,
        "doc_norms": compute_doc_norms(index, idf),
        "idf_bm25": compute_bm25_idf(index, len(corpus)),
        "doc_lens": doc_lens,
        "avg_len": sum(doc_lens.values()) / len(doc_lens),
        "product_info": product_info,
        "mean_rating": compute_mean_rating(product_info),
    }


def conjunctive_search(query, index, analyzer=build_terms):
    """AND search: returns the set of pids that contain all the query terms.

    analyzer turns the query into terms; it must be the one used to build the index.
    """
    terms = set(analyzer(query))
    if not terms or any(t not in index for t in terms):
        return set()
    # Start with the shortest posting list so the intersection is fast
    terms = sorted(terms, key=lambda t: len(index[t]))
    result = set(index[terms[0]])
    for term in terms[1:]:
        result &= index[term].keys()
    return result
