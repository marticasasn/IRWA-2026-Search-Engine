import atexit
import os
import signal
import sys
import time
from json import JSONEncoder

import httpagentparser  # for getting the user agent as json
from flask import Flask, abort, g, jsonify, redirect, render_template, send_from_directory, session, url_for
from flask import request

from myapp.analytics.analytics_data import AnalyticsData, ClickedDoc
from myapp.analytics.dashboard import build_charts, compute_kpis
from myapp.search.objects import Document, StatsDocument
from myapp.search.search_engine import SEARCH_OPTIONS, SearchEngine, parse_options
from myapp.generation.rag import RAGGenerator
from myapp.generation.rag_improved import ImprovedRAGGenerator
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env

# Product titles and LLM answers contain non-ASCII characters; never let a print fail a request
# on consoles with a limited encoding (e.g. cp1252 on Windows)
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")


# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)
_default.default = JSONEncoder().default
JSONEncoder.default = _default
# end lines ***for using method to_json in objects ***


# instantiate the Flask application
app = Flask(__name__)

# random 'secret_key' is used for persisting data in secure cookie
app.secret_key = os.getenv("SECRET_KEY")
# open browser dev tool to see the cookies (Flask 3 reads the name from the config)
app.config["SESSION_COOKIE_NAME"] = os.getenv("SESSION_COOKIE_NAME", "session")
# instantiate our search engine (loads the corpus and the index once, cached in cache/)
search_engine = SearchEngine(raw_path=os.getenv("DATA_FILE_PATH", "fashion_products_dataset.json"))
# instantiate our in memory persistence, saved to a JSON file (gitignored); ANALYTICS_FILE overrides the
# path, e.g. to look at the simulated traffic of scripts/simulate_traffic.py
ANALYTICS_FILE = os.getenv("ANALYTICS_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                          "data", "analytics.json"))
analytics_data = AnalyticsData(path=ANALYTICS_FILE)
# Unsaved analytics are written when the app stops (Ctrl+C raises KeyboardInterrupt and exits normally);
# SIGTERM and Ctrl+Break (Windows) are turned into a normal exit so the atexit save runs too
atexit.register(analytics_data.flush)


def _exit_on_signal(signum, frame):
    """Normal exit on a termination signal, so the atexit handlers run."""
    sys.exit(0)


for _name in ("SIGTERM", "SIGBREAK"):
    if hasattr(signal, _name):
        signal.signal(getattr(signal, _name), _exit_on_signal)
print("Analytics data file:", ANALYTICS_FILE)
# instantiate the RAG generators: the template's one (baseline) and ours (improved)
rag_generators = {"improved": ImprovedRAGGenerator(), "baseline": RAGGenerator()}
DEFAULT_RAG = "improved"

DEFAULT_METHOD = "custom"
METHOD_LABELS = {"custom": "Custom (BM25 + rating + discount)",
                 "custom_title": "Custom + title coverage", "bm25": "BM25", "tfidf": "TF-IDF cosine"}


# Requests that do not count as "the next request" for the dwell time fallback
DWELL_IGNORED_PREFIXES = ("/static/", "/dwell", "/favicon.ico")


def get_doc_object(pid):
    """Product as the template's Document object (used by the stats and dashboard pages)."""
    return Document(**search_engine.get_document(pid))


@app.before_request
def start_request():
    """Starts the request timer and resolves the physical analytics session of the user."""
    g.start_time = time.perf_counter()
    session_id = analytics_data.touch_session(session.get("sid"), session.get("last_seen"),
                                              request.headers.get("User-Agent"), request.remote_addr)
    session["sid"] = session_id
    session["last_seen"] = analytics_data.now().isoformat()
    g.session_id = session_id
    if not request.path.startswith(DWELL_IGNORED_PREFIXES):
        analytics_data.fill_dwell_from_next_request(session_id)


@app.after_request
def log_request(response):
    """Stores the request in fact_request and saves the analytics file (at most every few seconds)."""
    elapsed_ms = (time.perf_counter() - g.get("start_time", time.perf_counter())) * 1000
    analytics_data.record_request(g.get("session_id"), request.path, request.method, response.status_code, elapsed_ms)
    analytics_data.save(force=False)
    return response


# Home URL "/"
@app.route('/')
def index():
    print("starting home url /...")

    # flask server creates a session by persisting a cookie in the user's browser.
    # the 'session' object keeps data between multiple requests. Example:
    session['some_var'] = "Some value that is kept in session"

    user_agent = request.headers.get('User-Agent')
    print("Raw user browser:", user_agent)

    user_ip = request.remote_addr
    agent = httpagentparser.detect(user_agent)

    print("Remote IP: {} - JSON user browser {}".format(user_ip, agent))
    print(session)
    return render_template('index.html', page_title="Welcome", methods=METHOD_LABELS,
                           selected_method=session.get('last_method', DEFAULT_METHOD),
                           options=session.get('last_options', {name: True for name in SEARCH_OPTIONS}),
                           option_labels=SEARCH_OPTIONS)


@app.route('/search', methods=['POST'])
def search_form_post():
    """Runs the query with the selected ranking method and shows the results page."""
    search_query = request.form.get('search-query', '').strip()
    if not search_query:
        return redirect(url_for('index'))
    method = request.form.get('method', DEFAULT_METHOD)
    if method not in METHOD_LABELS:
        method = DEFAULT_METHOD

    options = parse_options(request.form)

    session['last_search_query'] = search_query
    session['last_method'] = method
    session['last_options'] = options

    search_id = analytics_data.save_query_terms(search_query)

    info = search_engine.search(search_query, search_id, method=method, options=options)
    results = info["results"]
    # The RAG summary is loaded by the results page from /rag_summary, so the results show at once
    session['last_search'] = {"search_id": search_id, "query": search_query, "method": method, "options": options}

    found_count = info["total"]
    session['last_found_count'] = found_count
    analytics_data.record_query(search_id, g.session_id, search_query, method, found_count, info["fallback"],
                                info["query_terms"], options)
    analytics_data.flush()  # queries, clicks and dwell times are written at once

    print(session)

    return render_template('results.html', results_list=results, page_title="Results", found_counter=found_count,
                           query=search_query, method=method, methods=METHOD_LABELS, info=info, options=options,
                           option_labels=SEARCH_OPTIONS, search_id=search_id,
                           rag_mode=session.get('rag_mode', DEFAULT_RAG), rag_modes=list(rag_generators))


@app.route('/rag_summary', methods=['GET'])
def rag_summary():
    """AI summary of the last search of this session, as JSON (mode = "improved" or "baseline").

    The search is run again with the same parameters (it is deterministic and fast) instead of keeping
    the results in the session cookie; it is not counted as a new query.
    """
    last = session.get('last_search')
    if not last or str(last["search_id"]) != request.args.get("search_id", ""):
        abort(404)
    mode = request.args.get("mode", DEFAULT_RAG)
    if mode not in rag_generators:
        mode = DEFAULT_RAG
    session['rag_mode'] = mode

    info = search_engine.engine.search_with_info(last["query"], method=last["method"], **last["options"])
    if mode == "improved":
        answer = rag_generators[mode].generate_response(last["query"], info["results"], fallback=info["fallback"])
    else:
        answer = {"text": rag_generators[mode].generate_response(last["query"], info["results"]), "status": "baseline"}
    answer["mode"] = mode
    print("RAG response ({}): {}".format(mode, answer["text"]))
    return jsonify(answer)


@app.route('/doc_details', methods=['GET'])
def doc_details():
    """
    Show the details page of one product. The url keeps pid, search_id and rank for the click analytics.
    """

    # getting request parameters:
    # user = request.args.get('user')
    print("doc details session: ")
    print(session)

    res = session.get("some_var")
    print("recovered var from session:", res)

    # get the query string parameters from request
    clicked_doc_id = request.args.get("pid", "")
    search_id = request.args.get("search_id")
    rank = request.args.get("rank")
    print("click in id={} (search_id={}, rank={})".format(clicked_doc_id, search_id, rank))

    doc = search_engine.get_document(clicked_doc_id)
    if doc is None:
        abort(404)

    # store the click in fact_click (it also updates the template's counter table fact_clicks);
    # only clicks coming from a results page (with a search_id) are counted
    click_id = None
    if search_id:
        click_id = analytics_data.record_click(search_id, g.session_id, clicked_doc_id, rank)
        analytics_data.flush()

    print("fact_clicks count for id={} is {}".format(clicked_doc_id, analytics_data.fact_clicks.get(clicked_doc_id, 0)))
    return render_template('doc_details.html', page_title="Product details", doc=doc, search_id=search_id, rank=rank,
                           query=session.get('last_search_query'), click_id=click_id)


@app.route('/favicon.ico')
def favicon():
    """Site icon (UPF logo mark) for browsers that request /favicon.ico directly."""
    return send_from_directory(app.static_folder, "favicon.ico", mimetype="image/vnd.microsoft.icon")


@app.route('/dwell', methods=['POST'])
def dwell():
    """Receives the dwell time beacon of the details page (click_id, dwell_ms)."""
    try:
        stored = analytics_data.record_dwell_beacon(request.form.get("click_id", ""), g.session_id,
                                                    float(request.form.get("dwell_ms", "")))
    except ValueError:
        stored = False
    analytics_data.flush()
    return ("", 204) if stored else ("", 400)


@app.route('/stats', methods=['GET'])
def stats():
    """
    Show simple statistics example. ### Replace with yourdashboard ###
    :return:
    """

    docs = []
    for doc_id in analytics_data.fact_clicks:
        row: Document = get_doc_object(doc_id)
        count = analytics_data.fact_clicks[doc_id]
        doc = StatsDocument(pid=row.pid, title=row.title, description=row.description, url=row.url, count=count)
        docs.append(doc)
    
    # simulate sort by ranking
    docs.sort(key=lambda doc: doc.count, reverse=True)
    return render_template('stats.html', clicks_data=docs)


@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Analytics dashboard: KPIs and Altair charts built from the star schema tables."""
    sessions, _, queries, clicks = analytics_data.dataframes()

    def title_of(pid):
        """Title of a product (None if unknown)."""
        doc = search_engine.get_document(pid)
        return doc["title"] if doc else None

    return render_template('dashboard.html', page_title="Dashboard", kpis=compute_kpis(sessions, queries, clicks),
                           charts=build_charts(sessions, queries, clicks, title_of), data_file=ANALYTICS_FILE)


# New route added for generating an examples of basic Altair plot (used for dashboard)
@app.route('/plot_number_of_views', methods=['GET'])
def plot_number_of_views():
    return analytics_data.plot_number_of_views()


if __name__ == "__main__":
    # os.getenv returns a string, so "False" must not count as true
    debug = os.getenv("DEBUG", "False").strip().lower() in ("1", "true", "yes")
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=debug)
