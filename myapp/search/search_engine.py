import time
from urllib.parse import urlencode

from src.engine import METHODS, SearchEngine as CoreSearchEngine


# Search switches of src.engine.SearchEngine.search_with_info shown in the UI (all on by default)
SEARCH_OPTIONS = {
    "protected_words": "Protected words (no stemming of leggings, shorts, ...)",
    "field_boost": "Boost title / category matches",
    "or_fallback": "OR fallback when no exact match",
    "collapse_duplicates": "Hide near-duplicates",
}


def parse_options(form):
    """Reads the search switches from a submitted form; if the form has no options block, all are on."""
    if form.get("options-form") is None:
        return {name: True for name in SEARCH_OPTIONS}
    return {name: form.get(name) == "on" for name in SEARCH_OPTIONS}


def details_url(pid, search_id, rank):
    """Url of our details page; keeps pid, search_id and rank for the click analytics."""
    return "/doc_details?" + urlencode({"pid": pid, "search_id": search_id, "rank": rank})


class SearchEngine:
    """Class that implements the search engine logic (web layer over src.engine.SearchEngine)"""

    methods = METHODS

    def __init__(self, raw_path="fashion_products_dataset.json"):
        """Loads our engine once; raw_path is the raw dataset used for the display fields."""
        self.engine = CoreSearchEngine(raw_path=raw_path)
        source = "cache" if self.engine.from_cache else "corpus (cache rebuilt)"
        print(f"Search engine ready in {self.engine.startup_time:.2f}s, loaded from {source}")

    def search(self, search_query, search_id, method="custom", top_k=20, options=None):
        """Ranked search; returns the info dict of src.engine.SearchEngine.search_with_info
        (results, total, fallback, hidden, query_terms). Every result dict also gets search_id and the
        details_url to our page. options are the search switches (see SEARCH_OPTIONS)."""
        print("Search query:", search_query, "| method:", method, "| options:", options)
        start = time.perf_counter()
        info = self.engine.search_with_info(search_query, method=method, top_k=top_k, **(options or {}))
        info["time_ms"] = (time.perf_counter() - start) * 1000
        print(f"Search took {info['time_ms']:.1f} ms, {len(info['results'])} shown of {info['total']}")

        for result in info["results"]:
            result["search_id"] = search_id
            result["details_url"] = details_url(result["pid"], search_id, result["rank"])
        return info

    def get_document(self, pid):
        """All the fields of one product, or None if the pid is unknown."""
        return self.engine.get_document(pid)
