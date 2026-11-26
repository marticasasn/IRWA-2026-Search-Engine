import os
from json import JSONEncoder

import httpagentparser  # for getting the user agent as json
from flask import Flask, abort, redirect, render_template, session, url_for
from flask import request

from myapp.analytics.analytics_data import AnalyticsData, ClickedDoc
from myapp.search.objects import Document, StatsDocument
from myapp.search.search_engine import SEARCH_OPTIONS, SearchEngine, parse_options
from myapp.generation.rag import RAGGenerator
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env


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
# instantiate our in memory persistence
analytics_data = AnalyticsData()
# instantiate RAG generator
rag_generator = RAGGenerator()

DEFAULT_METHOD = "custom"
METHOD_LABELS = {"custom": "Custom (BM25 + rating + discount)", "bm25": "BM25", "tfidf": "TF-IDF cosine"}


def get_doc_object(pid):
    """Product as the template's Document object (used by the stats and dashboard pages)."""
    return Document(**search_engine.get_document(pid))


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

    # generate RAG response based on user query and retrieved results
    rag_response = rag_generator.generate_response(search_query, results)
    print("RAG response:", rag_response)

    found_count = info["total"]
    session['last_found_count'] = found_count

    print(session)

    return render_template('results.html', results_list=results, page_title="Results", found_counter=found_count,
                           rag_response=rag_response, query=search_query, method=method, methods=METHOD_LABELS,
                           info=info, options=options, option_labels=SEARCH_OPTIONS)


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

    # store data in statistics table 1
    if clicked_doc_id in analytics_data.fact_clicks.keys():
        analytics_data.fact_clicks[clicked_doc_id] += 1
    else:
        analytics_data.fact_clicks[clicked_doc_id] = 1

    print("fact_clicks count for id={} is {}".format(clicked_doc_id, analytics_data.fact_clicks[clicked_doc_id]))
    print(analytics_data.fact_clicks)
    return render_template('doc_details.html', page_title="Product details", doc=doc, search_id=search_id, rank=rank,
                           query=session.get('last_search_query'))


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
    visited_docs = []
    for doc_id in analytics_data.fact_clicks.keys():
        d: Document = get_doc_object(doc_id)
        doc = ClickedDoc(doc_id, d.description, analytics_data.fact_clicks[doc_id])
        visited_docs.append(doc)

    # simulate sort by ranking
    visited_docs.sort(key=lambda doc: doc.counter, reverse=True)

    for doc in visited_docs: print(doc)
    return render_template('dashboard.html', visited_docs=visited_docs)


# New route added for generating an examples of basic Altair plot (used for dashboard)
@app.route('/plot_number_of_views', methods=['GET'])
def plot_number_of_views():
    return analytics_data.plot_number_of_views()


if __name__ == "__main__":
    # os.getenv returns a string, so "False" must not count as true
    debug = os.getenv("DEBUG", "False").strip().lower() in ("1", "true", "yes")
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=debug)
