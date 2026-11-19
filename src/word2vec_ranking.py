"""Word2Vec ranking with cosine similarity between averaged query and document vectors."""

import numpy as np
from gensim.models import Word2Vec

from src.indexing import conjunctive_search
from src.preprocessing import build_terms


def train_word2vec(corpus, field="all_terms", seed=42):
    """Trains Word2Vec on the token lists of the corpus (single worker so runs are reproducible)."""
    return Word2Vec(
        sentences=list(corpus[field]),
        vector_size=100,
        window=5,
        min_count=2,
        seed=seed,
        workers=1,
    )


def text_vector(tokens, model):
    """Average of the word vectors of the tokens, skipping OOV words; zero vector if none."""
    vectors = [model.wv[t] for t in tokens if t in model.wv]
    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0)


def compute_doc_vectors(corpus, model, field="all_terms"):
    """Precomputes the averaged vector of every document: pid -> vector."""
    return {pid: text_vector(terms, model) for pid, terms in zip(corpus["pid"], corpus[field])}


def rank_word2vec(query, pids, model, doc_vectors):
    """Ranks the given pids by cosine similarity of word2vec vectors; returns [(pid, score)]."""
    query_vec = text_vector(build_terms(query), model)
    query_norm = np.linalg.norm(query_vec)

    scores = []
    for pid in pids:
        doc_vec = doc_vectors[pid]
        denom = query_norm * np.linalg.norm(doc_vec)
        scores.append((pid, float(query_vec @ doc_vec / denom) if denom > 0 else 0.0))
    # Highest score first, pid breaks ties so the order is always the same
    return sorted(scores, key=lambda x: (-x[1], x[0]))


def search_word2vec(query, index, model, doc_vectors, top_k=None):
    """AND search followed by word2vec ranking; returns the top_k [(pid, score)]."""
    pids = conjunctive_search(query, index)
    ranking = rank_word2vec(query, pids, model, doc_vectors)
    return ranking[:top_k] if top_k else ranking
