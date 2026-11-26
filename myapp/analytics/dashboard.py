"""KPIs and Altair charts of the analytics dashboard, computed from AnalyticsData."""

import altair as alt
import pandas as pd

CHART_WIDTH = "container"  # the chart takes the width of its div; axes and legend are fitted inside it
TOP_N = 10
LABEL_CHARS = 32           # longer axis labels are cut, the full text is in the tooltip


def truncate(text, max_chars=LABEL_CHARS):
    """Text cut to max_chars characters, with "…" when it was longer."""
    text = " ".join(str(text or "").split())
    return text if len(text) <= max_chars else text[:max_chars - 1].rstrip() + "…"


def unique_labels(labels):
    """Labels made unique by adding " (2)", " (3)"... to repeats, so different items never share a bar."""
    seen, result = {}, []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        result.append(label if seen[label] == 1 else f"{label} ({seen[label]})")
    return result


def _fit(chart, width):
    """Sets the size; with width "container" the whole chart (labels included) fits the div width."""
    if width == "container":
        return chart.properties(width=width, autosize=alt.AutoSizeParams(type="fit", contains="padding"))
    return chart.properties(width=width)


def _bar(df, x, y, x_title, y_title, sort=None, horizontal=False, width=CHART_WIDTH, height=220, tooltip=None):
    """Simple bar chart; horizontal puts the categories on the y axis. tooltip defaults to [x, y]."""
    tooltip = tooltip or [x, y]
    if horizontal:
        encoding = dict(y=alt.Y(f"{x}:N", title=x_title, sort=sort or "-x"),
                        x=alt.X(f"{y}:Q", title=y_title), tooltip=tooltip)
    else:
        encoding = dict(x=alt.X(f"{x}:O", title=x_title, sort=sort),
                        y=alt.Y(f"{y}:Q", title=y_title), tooltip=tooltip)
    chart = alt.Chart(df).mark_bar(color="#c8102e").encode(**encoding).properties(height=height)
    return _fit(chart, width)


def _counts(series, name):
    """Value counts of a series as a DataFrame with columns name and "count"."""
    counts = series.value_counts()
    return pd.DataFrame({name: counts.index.astype(str), "count": counts.values})


def compute_kpis(sessions, queries, clicks):
    """Headline numbers of the dashboard (None when there is no data to compute them)."""
    n_queries, n_clicks = len(queries), len(clicks)
    clicked_queries = clicks["query_id"].dropna().nunique() if n_clicks else 0
    ranks = clicks["rank"].dropna() if n_clicks else pd.Series(dtype=float)
    dwell = clicks["dwell_seconds"].dropna() if n_clicks else pd.Series(dtype=float)
    zero = int((queries["n_results"] == 0).sum()) if n_queries else 0
    return {
        "Sessions": len(sessions),
        "Queries": n_queries,
        "Clicks": n_clicks,
        "CTR (queries with a click)": f"{100 * clicked_queries / n_queries:.1f}%" if n_queries else None,
        "Mean rank of clicked docs": f"{ranks.mean():.2f}" if len(ranks) else None,
        "Mean dwell time": f"{dwell.mean():.1f} s" if len(dwell) else None,
        "Queries with 0 results": f"{zero} ({100 * zero / n_queries:.1f}%)" if n_queries else None,
    }


def build_charts(sessions, queries, clicks, title_of):
    """List of {title, caption, spec (Vega-Lite JSON) or None, optional specs (list of JSON shown side by
    side)} for the dashboard.

    title_of maps a pid to its product title.
    """
    charts = []

    def add(title, caption, chart_fn, has_data):
        """Appends a chart, or an empty placeholder if there is no data for it."""
        charts.append({"title": title, "caption": caption, "spec": chart_fn().to_json() if has_data else None})

    has_queries, has_clicks, has_sessions = len(queries) > 0, len(clicks) > 0, len(sessions) > 0

    def queries_per_hour():
        hours = pd.to_datetime(queries["timestamp"].str.slice(0, 19)).dt.hour
        df = pd.DataFrame({"hour": range(24)})
        df["queries"] = df["hour"].map(hours.value_counts()).fillna(0).astype(int)
        return _bar(df, "hour", "queries", "Hour of the day", "Queries")
    add("Queries per hour of the day",
        "Number of queries submitted in each hour of the day (server local time), all days added together.",
        queries_per_hour, has_queries)

    def top_queries():
        normalized = queries["query"].str.lower().str.split().str.join(" ")
        df = _counts(normalized, "query").head(TOP_N)
        df["label"] = unique_labels([truncate(q) for q in df["query"]])
        return _bar(df, "label", "count", "Query", "Times submitted", horizontal=True,
                    tooltip=[alt.Tooltip("query:N", title="Query"), alt.Tooltip("count:Q", title="Times submitted")])
    add("Top queries",
        f"The {TOP_N} most frequent queries (lowercased, repeated spaces removed) and how many times each was submitted.",
        top_queries, has_queries)

    def clicked_ranks():
        df = _counts(clicks["rank"].dropna().astype(int), "rank")
        df["rank"] = df["rank"].astype(int)
        return _bar(df, "rank", "count", "Rank of the clicked result", "Clicks", sort="ascending")
    add("Rank of the clicked results",
        "Number of clicks by the position (rank) that the clicked product had in the results list.",
        clicked_ranks, has_clicks and clicks["rank"].notna().any())

    def most_clicked():
        df = _counts(clicks["pid"], "pid").head(TOP_N)
        df["title"] = [" ".join((title_of(p) or p).split()) for p in df["pid"]]
        df["label"] = unique_labels([truncate(t) for t in df["title"]])
        return _bar(df, "label", "count", "Product", "Clicks", horizontal=True,
                    tooltip=[alt.Tooltip("title:N", title="Product"), alt.Tooltip("pid:N", title="pid"),
                             alt.Tooltip("count:Q", title="Clicks")])
    add("Most clicked products",
        f"The {TOP_N} products whose details page was opened most often from the results.",
        most_clicked, has_clicks)

    # Three separate charts (one per column of the page grid) instead of one concatenated chart, so each one
    # can take its column width: side by side on wide screens, one under the other on phones
    environment = [
        _bar(_counts(sessions[col], col), col, "count", label, "Sessions", sort="-y", height=180).to_json()
        for col, label in (("browser", "Browser"), ("os", "Operating system"), ("device", "Device type"))
    ] if has_sessions else None
    charts.append({"title": "Browser, operating system and device",
                   "caption": "Number of sessions by browser, operating system and device type, parsed from the "
                              "user agent.",
                   "spec": None, "specs": environment})

    def query_length():
        df = _counts(queries["n_terms"], "words")
        df["words"] = df["words"].astype(int)
        return _bar(df, "words", "count", "Words in the query", "Queries", sort="ascending")
    add("Query length",
        "Number of queries by number of words in the query as typed.",
        query_length, has_queries)

    def method_usage():
        return _bar(_counts(queries["method"], "method"), "method", "count", "Ranking method", "Queries", sort="-y")
    add("Ranking method usage",
        "Number of queries submitted with each ranking method.",
        method_usage, has_queries)

    def dwell_times():
        df = clicks.dropna(subset=["dwell_seconds"])[["dwell_seconds", "dwell_source"]]
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("dwell_seconds:Q", bin=alt.Bin(maxbins=20), title="Dwell time (seconds)"),
            y=alt.Y("count():Q", title="Clicks"),
            color=alt.Color("dwell_source:N", title="Measured by"),
            tooltip=[alt.Tooltip("dwell_source:N", title="Measured by"), alt.Tooltip("count():Q", title="Clicks")],
        ).properties(height=200)
        return _fit(chart, CHART_WIDTH)
    add("Dwell time",
        "Number of clicks by time spent on the details page; measured by the page beacon or, if missing, "
        "by the time until the next request of the session.",
        dwell_times, has_clicks and clicks["dwell_seconds"].notna().any())

    return charts
