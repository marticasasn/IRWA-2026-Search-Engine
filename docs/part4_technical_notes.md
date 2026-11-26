# Part 4: technical notes

List of what was implemented and where. Paths are relative to the repository root.

## Web app integration

- Course template (https://github.com/trokhymovych/irwa-search-engine) copied: `web_app.py`, `myapp/`, `templates/`,
  `static/`. Line endings normalised to LF.
- Secrets: `.env` gitignored; `.env.example` with `SECRET_KEY`, `DEBUG`, `SESSION_COOKIE_NAME`, `DATA_FILE_PATH`,
  `GROQ_API_KEY`, `GROQ_MODEL`, optional `ANALYTICS_FILE`.
- `web_app.py` changes to the template:
  - `app.config["SESSION_COOKIE_NAME"]` instead of `app.session_cookie_name` (ignored by Flask 3).
  - `DEBUG` parsed as a boolean (`"False"` is false).
  - stdout/stderr reconfigured with `errors="replace"` (prints of non-ASCII text cannot fail a request).
  - The template's pydantic corpus load (`myapp/search/load_corpus.py`) is no longer used; `/stats` and
    `/dashboard` get products from `SearchEngine.get_document`.
- Routes: `/` (search box, method select, search options), `/search` (POST), `/doc_details` (GET `pid`,
  `search_id`, `rank`), `/rag_summary` (GET `search_id`, `mode`), `/dwell` (POST beacon), `/stats`, `/dashboard`,
  `/plot_number_of_views` (template), `/favicon.ico` (`static/favicon.ico`, cropped from the template's UPF logo).
- Pages: `templates/index.html`, `templates/results.html`, `templates/doc_details.html`,
  `templates/_search_options.html`, `templates/dashboard.html`; styles in `static/styles/custom.css`.

## Search engine for the web (`src/engine.py`)

- `SearchEngine(processed_path="processed_corpus.json", raw_path="fashion_products_dataset.json",
  cache_path="cache/search_data.pkl")`.
  - Builds two index variants once and caches them with pickle (`cache/`, gitignored), together with the raw
    documents and the near-duplicate keys. Cache key: `CACHE_VERSION` + size and mtime of both data files.
  - `base`: `build_search_data` on the Part 1 `all_terms` (same as Part 3).
  - `web`: `all_terms` recomputed with `build_terms_web` (title + description + product details without
    `EXCLUDED_DETAIL_KEYS` + brand, the Part 1 fields).
  - Attributes: `startup_time`, `from_cache`.
- `search(query, method="custom", top_k=20, **options)`: list of result dicts (pid, title, description, brand,
  category, sub_category, selling_price, actual_price, discount, average_rating, out_of_stock, url, images, rank,
  score, similar_hidden).
- `search_with_info(query, method, top_k, protected_words=True, field_boost=True, or_fallback=True,
  collapse_duplicates=True, boost_weights=None)`: dict with `results`, `total`, `fallback`, `hidden`, `query_terms`.
- `get_document(pid)`: all the raw fields of a product.
- `METHODS = ("custom", "custom_title", "bm25", "tfidf")`.
- With all four options off, results are identical to `src.ranking.search_ranked` on the Part 1 index (checked for
  the 5 test queries and every method).

### Search options

| Option | Implementation |
|---|---|
| `protected_words` | `PROTECTED_TERMS` in `src/preprocessing.py` (leggings/legging -> legging, jeggings/jegging -> jegging, shorts, tights, chinos/chino -> chino, dress/dresses -> dress, dressed, dressing, suited) are mapped instead of stemmed; `build_terms(text, protected=None)`, `build_terms_web(text)`. Selects the `web` index; off = `base` index |
| `field_boost` | `apply_field_boost`: score * (1 + 0.5 * share of query terms in the title + 0.25 * share in category/sub_category) (`FIELD_BOOST_WEIGHTS`); re-sorted, custom keeps out of stock last |
| `or_fallback` | `or_search`: if the AND search returns nothing, the union of the postings of the query terms is ranked; `fallback=True` |
| `collapse_duplicates` | `duplicate_key` = (lowercased brand, `normalize_title`); only the best ranked product of each key is shown; `similar_hidden` per result, `hidden` total |

- Query analyzer: optional `analyzer=build_terms` parameter added to `conjunctive_search` (`src/indexing.py`),
  `rank_tfidf`, `rank_bm25`, `rank_custom` (`src/ranking.py`); the default keeps the Part 1-3 behaviour: the full
  rankings of the 7 labelled queries x 3 methods are bit-for-bit identical to the Part 3 code, and the 15 rankings
  of `part3_rankings.csv` are reproduced (same pids, scores equal up to the CSV float precision).
- Comparison: `scripts/search_before_after.py` -> `docs/part4_search_before_after.md` (top 5, 5 test queries,
  3 methods, all options off vs on).

## Custom score with title coverage (`src/ranking.py`)

- `score = bm25_norm * (1 + w_r * rating_norm + w_d * discount_norm + w_t * title_coverage)`.
- `title_coverage(query_terms, title_terms)`: fraction of the distinct query terms present in the title terms.
- `CUSTOM_WEIGHTS = {"rating": 0.3, "discount": 0.1, "title": 0.0}`; `w_t = 0` gives the Part 3 custom score.
- `quality_factor(..., title_coverage=0.0)`, `rank_custom(..., title_terms=None)`.
- Method `custom_title` in `search_ranked` and the web engine: `CUSTOM_TITLE_WEIGHT = 0.3`.
- `build_search_data` returns an extra key `title_terms` (pid -> frozenset) when the corpus has `title_terms`.

## Ranking evaluation (`scripts/eval_rankings.py`)

- Labels: `validation_labels.csv` (2 queries: "women full sleeve sweatshirt cotton", "men slim jeans blue"),
  `test_queries_results.csv` (5 test queries), `docs/unlabelled_for_review.csv` (filled by hand).
- Top 10 of `search_ranked` (Part 1 index) for tfidf, bm25, custom, custom_title; metrics from `src/evaluation.py`:
  P@5, P@10, AP@10 (mean = MAP@10), nDCG@10; `n_relevant` = labelled relevant results of the query (Part 2).
- Unlabelled results count as not relevant and are appended to `docs/unlabelled_for_review.csv`
  (query, pid, rank, method, title, brand, category, link, label); existing rows are never removed.
- Output: `docs/part4_ranking_metrics.md` (mean per method + per query, with unlabelled counts).

## RAG (`myapp/generation/`)

- Baseline: template `RAGGenerator` (`rag.py`), prompt and logic unchanged; two compatibility changes: results are
  read as dicts (`res['pid']`), and no API call is made when `GROQ_API_KEY` is empty.
- `ImprovedRAGGenerator` (`rag_improved.py`), constructor switches and parameters:

| Parameter | Default | Effect |
|---|---|---|
| `rich_context` | True | `format_product_rich`: [pid] title, brand, category / sub_category, price, MRP, discount, rating, stock, description cut at `description_chars` (200) |
| `filter_results` | True | `select_products`: score >= `relative_threshold` (0.5) * best score, optional absolute `min_score` (None), at most `max_products` (6); out of stock marked, or excluded with `exclude_out_of_stock` (False) |
| `handle_no_good` | True | no LLM pick for 0 results (`no_results`), OR fallback results (`fallback`), nothing left after filtering (`below_threshold`) or an LLM answer `NO_GOOD_MATCH` (`no_match`); `suggest_rephrase` (True) asks the LLM for rephrasing tips |
| `strict_prompt` | True | `STRICT_PROMPT`: only the given products, cite by [PID], price and rating in the justification, fixed 3-line format; temperature 0 |
| `verify_citations` | True | `check_citations`: pids in the answer (`PID_PATTERN`) must be among the given ones; otherwise the answer is discarded (`invalid_citation`); no pid cited -> `unverified` |

- `generate_response(user_query, retrieved_results, fallback=False)` returns a dict: `text`, `status`, `products`,
  `cited`, `invalid` (and `discarded_text` when discarded).
- Web: `/rag_summary?search_id=...&mode=improved|baseline` re-runs the last search of the session (stored in the
  session, not counted as a new query) and returns JSON; `results.html` loads it with `fetch`, shows it on top with a
  baseline/improved select (default improved, choice kept in the session).
- Comparison: `scripts/rag_compare.py` -> `docs/rag_comparison.md` (5 test queries + "something nice to wear",
  "laptop charger", "cheap black tshirt under 300"; retrieved products and both answers side by side).
  Generated with `GROQ_MODEL=openai/gpt-oss-20b`, the model of both generators (the template's
  `llama-3.1-8b-instant` is no longer available on Groq).

## Web analytics (`myapp/analytics/`)

- `AnalyticsData(path=None, clock=None)` in `analytics_data.py`; file `data/analytics.json` (gitignored), path
  overridable with `ANALYTICS_FILE`; `load()`, `save(force)`, `flush()`, `reset()`. Writes are atomic
  (temporary file + rename):
  - after every query, click and dwell event (`flush()` in `web_app.py`);
  - request rows: at most every `SAVE_INTERVAL_SECONDS` (5);
  - at exit: `atexit` -> `flush()` (Ctrl+C; SIGTERM and SIGBREAK are turned into a normal exit). `flush()` only
    writes if the process has unsaved changes, so the Flask reloader's watcher process never overwrites the file.
  - Tested with a real console Ctrl+C on `python web_app.py` (DEBUG=True) right after 2 searches and 1 click:
    exit code 0, all events and the last request row in the file.
- Star schema:

| Table | Fields |
|---|---|
| `dim_session` | session_id, start, end, user_agent, browser, os, device (mobile/desktop), ip, country, n_queries |
| `fact_request` | timestamp, session_id, path, method, status, response_ms |
| `fact_query` | query_id (= search_id), session_id, timestamp, query, n_terms, query_terms, method, n_results, fallback, position_in_session, mission_id, options |
| `fact_click` | click_id, query_id, session_id, pid, rank, timestamp, dwell_seconds, dwell_source (beacon / next_request) |

- Sessions: id stored in the Flask session cookie (`sid`, `last_seen`); new session after
  `SESSION_TIMEOUT_MINUTES` (30) without requests (`touch_session`).
- User agent: `parse_user_agent` (httpagentparser; OS from `dist` when present, `OS_NAMES` normalisation; device by
  `MOBILE_PATTERN`). Country: `country_of` returns "local" for loopback/private addresses, otherwise "unknown"
  (no geolocation service).
- Requests: `before_request` / `after_request` hooks in `web_app.py` (`start_request`, `log_request`).
- Queries: `save_query_terms` (next integer id) + `record_query`; missions: consecutive queries of a session that
  share a processed term get the same `mission_id`.
- Clicks: `/doc_details` with a `search_id` -> `record_click` (also updates the template's `fact_clicks` counter).
- Dwell time: `doc_details.html` sends `navigator.sendBeacon("/dwell")` on `pagehide` / `visibilitychange`
  (`record_dwell_beacon`, keeps the largest value); fallback `fill_dwell_from_next_request` on the next request of
  the session (ignoring `/static/`, `/dwell`, `/favicon.ico`), only within the session timeout.
- Dashboard (`dashboard.py`, `templates/dashboard.html`, Vega-Lite via jsdelivr): `compute_kpis` (sessions, queries,
  clicks, CTR = share of queries with at least one click, mean rank of clicked docs, mean dwell time, queries with
  0 results) and `build_charts` (queries per hour of day, top queries, clicked ranks, most clicked products,
  browser/OS/device, query length, ranking method usage, dwell time), each with a caption.
- `scripts/simulate_traffic.py [--users 40] [--seed 7]`: drives the app with the Flask test client and a fake
  clock; writes only `data/analytics_simulated.json` (refuses the real file).

## Dependencies added (`requirements.txt`)

Flask, python-dotenv, groq, altair, httpagentparser, pydantic.
