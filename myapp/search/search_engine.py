import time
from urllib.parse import urlencode

from src.engine import METHODS, SearchEngine as CoreSearchEngine


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

    def search(self, search_query, search_id, method="custom", top_k=20):
        """Ranked search; every result dict also gets search_id and the details_url to our page."""
        print("Search query:", search_query, "| method:", method)
        start = time.perf_counter()
        results = self.engine.search(search_query, method=method, top_k=top_k)
        print(f"Search took {(time.perf_counter() - start) * 1000:.1f} ms, {len(results)} results")

        for result in results:
            result["search_id"] = search_id
            result["details_url"] = details_url(result["pid"], search_id, result["rank"])
        return results

    def get_document(self, pid):
        """All the fields of one product, or None if the pid is unknown."""
        return self.engine.get_document(pid)
