"""Search engine used by the web app: loads the corpus once and answers ranked queries."""

import json
import os
import pickle
import time

from src.indexing import build_search_data, load_corpus
from src.ranking import search_ranked

# Repository root, so the default paths work whatever the current directory is
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Bump this when the cached structure changes so old pickles are rebuilt
CACHE_VERSION = 1

# Ranking methods accepted by search (same names as search_ranked)
METHODS = ("custom", "bm25", "tfidf")

# Raw dataset fields shown in the results list
DISPLAY_FIELDS = ("pid", "title", "description", "brand", "category", "sub_category",
                  "selling_price", "actual_price", "discount", "average_rating",
                  "out_of_stock", "url", "images")


def _resolve(path):
    """Returns path unchanged if it is absolute, otherwise relative to the repository root."""
    return path if os.path.isabs(path) else os.path.join(ROOT_DIR, path)


def _file_signature(path):
    """Size and modification time of a file, used to detect when the cache is out of date."""
    stat = os.stat(path)
    return stat.st_size, int(stat.st_mtime)


def load_raw_documents(path):
    """Loads the raw dataset (a JSON array) into a dict pid -> product record."""
    with open(path, encoding="utf-8") as f:
        return {doc["pid"]: doc for doc in json.load(f)}


class SearchEngine:
    """Ranked search over the fashion products corpus.

    The search data (index, idfs, norms, ...) is built once at startup and cached in a pickle
    so later restarts only need to load it.
    """

    def __init__(self, processed_path="processed_corpus.json",
                 raw_path="fashion_products_dataset.json",
                 cache_path="cache/search_data.pkl"):
        """Loads the search data and the raw documents, from the cache if it is up to date."""
        self.processed_path = _resolve(processed_path)
        self.raw_path = _resolve(raw_path)
        self.cache_path = _resolve(cache_path)

        start = time.perf_counter()
        self.data, self.documents, self.from_cache = self._load()
        self.startup_time = time.perf_counter() - start

    def _signature(self):
        """Identifies the input files and cache format; a different value means rebuild."""
        return (CACHE_VERSION, _file_signature(self.processed_path), _file_signature(self.raw_path))

    def _load(self):
        """Returns (search_data, documents, from_cache), building and caching them if needed."""
        signature = self._signature()
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("signature") == signature:
                    return cached["data"], cached["documents"], True
            except Exception as e:  # corrupt or incompatible cache: rebuild it
                print(f"Ignoring search cache ({e})")

        corpus = load_corpus(self.processed_path)
        data = build_search_data(corpus)
        documents = load_raw_documents(self.raw_path)
        self._save_cache({"signature": signature, "data": data, "documents": documents})
        return data, documents, False

    def _save_cache(self, payload):
        """Writes the cache atomically (temporary file + rename) so a crash never leaves it half written."""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        tmp_path = self.cache_path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, self.cache_path)

    def search(self, query, method="custom", top_k=20):
        """Ranks the products for query with the given method; returns a list of result dicts.

        Each dict has the display fields of the raw dataset plus rank (1-based) and score.
        """
        if method not in METHODS:
            raise ValueError(f"Unknown ranking method: {method}")
        ranking = search_ranked(query, method, self.data, top_k=top_k)
        results = []
        for rank, (pid, score) in enumerate(ranking, start=1):
            doc = self.documents.get(pid, {})
            result = {field: doc.get(field) for field in DISPLAY_FIELDS}
            result["pid"] = pid
            result["rank"] = rank
            result["score"] = score
            results.append(result)
        return results

    def get_document(self, pid):
        """Returns a copy of all the fields of one product, or None if the pid is unknown."""
        doc = self.documents.get(pid)
        return dict(doc) if doc is not None else None
