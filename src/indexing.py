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


def conjunctive_search(query, index):
    """AND search: returns the set of pids that contain all the query terms."""
    terms = set(build_terms(query))
    if not terms or any(t not in index for t in terms):
        return set()
    # Start with the shortest posting list so the intersection is fast
    terms = sorted(terms, key=lambda t: len(index[t]))
    result = set(index[terms[0]])
    for term in terms[1:]:
        result &= index[term].keys()
    return result
