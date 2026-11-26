"""Search engine used by the web app: loads the corpus once and answers ranked queries.

On top of the Part 3 rankers (search_ranked) the web engine adds, each one switchable per query:
- protected_words: an index built with build_terms_web, where words like "leggings" or "shorts" are not stemmed.
- field_boost: products whose title or category/sub_category contain the query terms get a higher score.
- or_fallback: if the AND search finds nothing, an OR search is ranked instead (and flagged).
- collapse_duplicates: near-duplicates (same brand + normalized title) are collapsed to the best ranked one.
With all four switched off the results are exactly search_ranked on the Part 1 index.
"""

import json
import os
import pickle
import re
import time
from functools import lru_cache

from src.indexing import build_search_data, conjunctive_search, load_corpus
from src.preprocessing import build_terms, build_terms_web
from src.ranking import rank_bm25, rank_custom, rank_tfidf

# Repository root, so the default paths work whatever the current directory is
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Bump this when the cached structure changes so old pickles are rebuilt
CACHE_VERSION = 2

# Ranking methods accepted by search (same names as search_ranked)
METHODS = ("custom", "bm25", "tfidf")

# Raw dataset fields shown in the results list
DISPLAY_FIELDS = ("pid", "title", "description", "brand", "category", "sub_category",
                  "selling_price", "actual_price", "discount", "average_rating",
                  "out_of_stock", "url", "images")

# Product details excluded from text indexing (same list as the Part 1 notebook)
EXCLUDED_DETAIL_KEYS = {
    "Style Code", "Model Name", "Pack of", "Reversible", "Sales Package",
    "Fabric Care", "Country of Origin", "Size", "Other Details",
}

# Default weights of the field boost: score * (1 + title * share_in_title + category * share_in_category)
FIELD_BOOST_WEIGHTS = {"title": 0.5, "category": 0.25}

# Query analyzer of each index variant ("base" = Part 1 index, "web" = protected words)
ANALYZERS = {"base": build_terms, "web": build_terms_web}


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


def details_to_text(details):
    """Joins the useful values of product_details (a list of one-key dicts) into one text, as in Part 1."""
    merged = {}
    for item in details or []:
        if isinstance(item, dict):
            merged.update(item)
    return " ".join(str(v) for k, v in merged.items() if k not in EXCLUDED_DETAIL_KEYS)


def normalize_title(title):
    """Lowercase title with only letters and digits, used to detect near-duplicates."""
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def duplicate_key(doc):
    """Two products with the same key (brand + normalized title) are considered near-duplicates."""
    return (doc.get("brand") or "").strip().lower(), normalize_title(doc.get("title"))


def web_terms(corpus):
    """Re-processes the text fields of the corpus with build_terms_web.

    Returns (all_terms, title_terms, category_terms) lists aligned with the corpus rows; all_terms uses
    the same fields as Part 1 (title + description + details + brand).
    """
    analyze = lru_cache(maxsize=None)(build_terms_web)  # many descriptions are repeated
    all_terms, title_terms, category_terms = [], [], []
    for row in corpus.itertuples(index=False):
        title = analyze(row.title)
        all_terms.append(title + analyze(row.description) + analyze(details_to_text(row.product_details))
                         + analyze(row.brand))
        title_terms.append(title)
        category_terms.append(analyze(row.category) + analyze(row.sub_category))
    return all_terms, title_terms, category_terms


def or_search(query, index, analyzer):
    """OR search: returns the set of pids that contain at least one query term."""
    pids = set()
    for term in set(analyzer(query)):
        pids |= index.get(term, {}).keys()
    return pids


def rank_pids(method, query, pids, data, analyzer):
    """Ranks pids with the chosen Part 3 ranker ("tfidf", "bm25" or "custom"); returns [(pid, score)]."""
    if method == "tfidf":
        return rank_tfidf(query, pids, data["index"], data["idf"], data["doc_norms"], analyzer=analyzer)
    if method == "bm25":
        return rank_bm25(query, pids, data["index"], data["idf_bm25"], data["doc_lens"], data["avg_len"],
                         analyzer=analyzer)
    if method == "custom":
        return rank_custom(query, pids, data["index"], data["idf_bm25"], data["doc_lens"], data["avg_len"],
                           data["product_info"], mean_rating=data["mean_rating"], analyzer=analyzer)
    raise ValueError(f"Unknown ranking method: {method}")


def apply_field_boost(ranking, query_terms, title_terms, category_terms, product_info, method, weights=None):
    """Multiplies every score by 1 + w_title * share of query terms in the title + w_category * share in
    category/sub_category, then sorts again (custom keeps out of stock products last)."""
    w = {**FIELD_BOOST_WEIGHTS, **(weights or {})}
    if not query_terms:
        return ranking
    n = len(query_terms)
    boosted = []
    for pid, score in ranking:
        in_title = len(query_terms & title_terms.get(pid, frozenset())) / n
        in_category = len(query_terms & category_terms.get(pid, frozenset())) / n
        boosted.append((pid, score * (1 + w["title"] * in_title + w["category"] * in_category)))
    if method == "custom":
        return sorted(boosted, key=lambda x: (product_info[x[0]][2], -x[1], x[0]))
    return sorted(boosted, key=lambda x: (-x[1], x[0]))


class SearchEngine:
    """Ranked search over the fashion products corpus.

    The search data (index, idfs, norms, ...) of both index variants is built once at startup and cached
    in a pickle so later restarts only need to load it.
    """

    def __init__(self, processed_path="processed_corpus.json",
                 raw_path="fashion_products_dataset.json",
                 cache_path="cache/search_data.pkl"):
        """Loads the search data and the raw documents, from the cache if it is up to date."""
        self.processed_path = _resolve(processed_path)
        self.raw_path = _resolve(raw_path)
        self.cache_path = _resolve(cache_path)

        start = time.perf_counter()
        payload, self.from_cache = self._load()
        self.indexes = payload["indexes"]
        self.documents = payload["documents"]
        self.duplicate_keys = payload["duplicate_keys"]
        self.startup_time = time.perf_counter() - start

    def _signature(self):
        """Identifies the input files and cache format; a different value means rebuild."""
        return (CACHE_VERSION, _file_signature(self.processed_path), _file_signature(self.raw_path))

    def _load(self):
        """Returns (payload, from_cache), building and caching the payload if needed."""
        signature = self._signature()
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("signature") == signature:
                    return cached, True
            except Exception as e:  # corrupt or incompatible cache: rebuild it
                print(f"Ignoring search cache ({e})")

        payload = self._build()
        payload["signature"] = signature
        self._save_cache(payload)
        return payload, False

    def _build(self):
        """Builds both index variants, the raw documents and the near-duplicate keys."""
        corpus = load_corpus(self.processed_path)
        base = {
            "data": build_search_data(corpus),
            "title_terms": dict(zip(corpus["pid"], map(frozenset, corpus["title_terms"]))),
            "category_terms": {pid: frozenset(c + s) for pid, c, s in
                               zip(corpus["pid"], corpus["category_terms"], corpus["sub_category_terms"])},
        }
        all_terms, title_terms, category_terms = web_terms(corpus)
        corpus["web_terms"] = all_terms
        web = {
            "data": build_search_data(corpus, field="web_terms"),
            "title_terms": dict(zip(corpus["pid"], map(frozenset, title_terms))),
            "category_terms": dict(zip(corpus["pid"], map(frozenset, category_terms))),
        }
        documents = load_raw_documents(self.raw_path)
        return {
            "indexes": {"base": base, "web": web},
            "documents": documents,
            "duplicate_keys": {pid: duplicate_key(doc) for pid, doc in documents.items()},
        }

    def _save_cache(self, payload):
        """Writes the cache atomically (temporary file + rename) so a crash never leaves it half written."""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        tmp_path = self.cache_path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, self.cache_path)

    def search_with_info(self, query, method="custom", top_k=20, protected_words=True, field_boost=True,
                         or_fallback=True, collapse_duplicates=True, boost_weights=None):
        """Ranks the products for query; returns a dict with the results and how they were obtained.

        Keys: results (list of result dicts, see search), total (number of matching products before
        top_k and collapsing), fallback (True if the results come from the OR fallback), hidden (number
        of near-duplicates hidden behind the shown results) and query_terms.
        """
        if method not in METHODS:
            raise ValueError(f"Unknown ranking method: {method}")
        variant = "web" if protected_words else "base"
        bundle = self.indexes[variant]
        analyzer = ANALYZERS[variant]
        data = bundle["data"]

        pids = conjunctive_search(query, data["index"], analyzer=analyzer)
        fallback = False
        if not pids and or_fallback:
            pids = or_search(query, data["index"], analyzer)
            fallback = bool(pids)

        ranking = rank_pids(method, query, pids, data, analyzer)
        query_terms = set(analyzer(query))
        if field_boost:
            ranking = apply_field_boost(ranking, query_terms, bundle["title_terms"], bundle["category_terms"],
                                        data["product_info"], method, boost_weights)

        shown, hidden_counts = self._select(ranking, top_k, collapse_duplicates)
        results = []
        for rank, (pid, score) in enumerate(shown, start=1):
            doc = self.documents.get(pid, {})
            result = {field: doc.get(field) for field in DISPLAY_FIELDS}
            result.update(pid=pid, rank=rank, score=score, similar_hidden=hidden_counts.get(pid, 0))
            results.append(result)
        return {
            "results": results,
            "total": len(ranking),
            "fallback": fallback,
            "hidden": sum(hidden_counts.values()),
            "query_terms": sorted(query_terms),
        }

    def _select(self, ranking, top_k, collapse_duplicates):
        """Takes the top_k of the ranking; if collapsing, keeps only the best ranked product of each
        near-duplicate group. Returns (shown [(pid, score)], {shown pid: number of hidden duplicates})."""
        if not collapse_duplicates:
            return (ranking[:top_k] if top_k else ranking), {}
        shown, first_of_group, hidden = [], {}, {}
        for pid, score in ranking:
            key = self.duplicate_keys.get(pid, (pid,))
            if key in first_of_group:
                best = first_of_group[key]
                if best is not None:
                    hidden[best] = hidden.get(best, 0) + 1
                continue
            if top_k and len(shown) >= top_k:
                # Groups that start after the top_k are not shown, only remembered to skip their copies
                first_of_group[key] = None
                continue
            first_of_group[key] = pid
            shown.append((pid, score))
        return shown, hidden

    def search(self, query, method="custom", top_k=20, **options):
        """Ranks the products for query; returns a list of result dicts.

        Each dict has the display fields of the raw dataset plus rank (1-based), score and similar_hidden.
        options are the switches of search_with_info (protected_words, field_boost, or_fallback,
        collapse_duplicates, boost_weights).
        """
        return self.search_with_info(query, method, top_k, **options)["results"]

    def get_document(self, pid):
        """Returns a copy of all the fields of one product, or None if the pid is unknown."""
        doc = self.documents.get(pid)
        return dict(doc) if doc is not None else None
