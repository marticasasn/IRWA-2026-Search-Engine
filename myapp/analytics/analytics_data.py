"""Web analytics: an in-memory star schema saved to a JSON file.

Tables (all keyed or listed as plain dicts, so they serialise to JSON directly):
- dim_session: session_id -> start, end, user agent, browser, OS, device type, IP, country, query count.
- fact_request: list of every HTTP request (timestamp, session_id, path, method, status, response time).
- fact_query: query_id -> session_id, timestamp, raw query, number of terms, method, number of results,
  OR fallback, position of the query in the session, mission_id.
- fact_click: click_id -> query_id, session_id, pid, rank, timestamp, dwell time and how it was measured.

A physical session is our own id stored in the Flask session cookie; it is replaced after
SESSION_TIMEOUT_MINUTES of inactivity. Consecutive queries of a session that share a processed term
belong to the same "mission".
"""

import ipaddress
import json
import os
import re
import time
import uuid
from datetime import datetime, timedelta

import altair as alt
import httpagentparser
import pandas as pd

SESSION_TIMEOUT_MINUTES = 30
SAVE_INTERVAL_SECONDS = 5  # the file is written at most this often (and always on demand / at exit)
MOBILE_PATTERN = re.compile(r"mobile|android|iphone|ipad|ipod|windows phone", re.IGNORECASE)
# Device names that httpagentparser reports as the OS, mapped to the OS name
OS_NAMES = {"iPhone": "iOS", "iPad": "iOS", "iPod": "iOS", "Macintosh": "macOS", "Mac OS": "macOS"}


def parse_user_agent(user_agent):
    """Browser, OS and device type ("mobile" or "desktop") of a user agent string."""
    user_agent = user_agent or ""
    info = httpagentparser.detect(user_agent) if user_agent else {}
    browser = info.get("browser", {}).get("name") or "Unknown"
    # "dist" holds e.g. Android or iOS, which httpagentparser otherwise reports as Linux / Mac OS
    os_name = ((info.get("dist") or {}).get("name") or (info.get("os") or info.get("platform") or {}).get("name")
               or "Unknown")
    device = "mobile" if MOBILE_PATTERN.search(user_agent) else "desktop"
    return browser, OS_NAMES.get(os_name, os_name), device


def country_of(ip):
    """"local" for loopback and private addresses; "unknown" otherwise (no geolocation service is used)."""
    try:
        address = ipaddress.ip_address(ip)
    except (ValueError, TypeError):
        return "unknown"
    return "local" if address.is_loopback or address.is_private else "unknown"


class AnalyticsData:
    """
    An in memory persistence object.
    Declare more variables to hold analytics tables.
    """

    def __init__(self, path=None, clock=None):
        """Creates the tables and loads them from path (JSON) if the file exists.

        clock returns the current datetime (injectable so the traffic simulator can control time).
        """
        self.path = path
        self.clock = clock or (lambda: datetime.now().astimezone())
        self.dim_session = {}
        self.fact_request = []
        self.fact_query = {}
        self.fact_click = {}
        # Example of statistics table (template): key = doc id | value = click counter
        self.fact_clicks = {}
        self._last_save = 0.0
        self._dirty = False
        if path and os.path.exists(path):
            self.load()

    # ---------- persistence ----------

    def to_dict(self):
        """All the tables as one JSON-serialisable dict."""
        return {"dim_session": self.dim_session, "fact_request": self.fact_request,
                "fact_query": self.fact_query, "fact_click": self.fact_click}

    def load(self):
        """Loads the tables from self.path and rebuilds the template's click counters."""
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        self.dim_session = data.get("dim_session", {})
        self.fact_request = data.get("fact_request", [])
        self.fact_query = data.get("fact_query", {})
        self.fact_click = data.get("fact_click", {})
        self.fact_clicks = {}
        for click in self.fact_click.values():
            self.fact_clicks[click["pid"]] = self.fact_clicks.get(click["pid"], 0) + 1

    def save(self, force=True):
        """Writes the tables to self.path atomically. With force=False only if they changed and the
        last save is older than SAVE_INTERVAL_SECONDS."""
        if not self.path:
            return
        if not force and (not self._dirty or time.time() - self._last_save < SAVE_INTERVAL_SECONDS):
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)
        os.replace(tmp_path, self.path)
        self._last_save = time.time()
        self._dirty = False

    def reset(self):
        """Empties every table (the file is overwritten on the next save)."""
        self.dim_session, self.fact_request, self.fact_query, self.fact_click, self.fact_clicks = {}, [], {}, {}, {}
        self._dirty = True

    def now(self):
        """Current time from the clock."""
        return self.clock()

    # ---------- sessions and requests ----------

    def touch_session(self, session_id, last_seen, user_agent, ip):
        """Returns the physical session id for a request, opening a new session if there is none or the
        last request is older than SESSION_TIMEOUT_MINUTES. Updates the session end time."""
        now = self.now()
        expired = (last_seen is None or now - datetime.fromisoformat(last_seen)
                   > timedelta(minutes=SESSION_TIMEOUT_MINUTES))
        if session_id is None or session_id not in self.dim_session or expired:
            session_id = uuid.uuid4().hex
            browser, os_name, device = parse_user_agent(user_agent)
            self.dim_session[session_id] = {
                "session_id": session_id, "start": now.isoformat(), "end": now.isoformat(),
                "user_agent": user_agent or "", "browser": browser, "os": os_name, "device": device,
                "ip": ip, "country": country_of(ip), "n_queries": 0,
            }
        self.dim_session[session_id]["end"] = now.isoformat()
        self._dirty = True
        return session_id

    def record_request(self, session_id, path, method, status, response_ms):
        """Appends one row to fact_request."""
        self.fact_request.append({
            "timestamp": self.now().isoformat(), "session_id": session_id, "path": path,
            "method": method, "status": status, "response_ms": round(response_ms, 2),
        })
        self._dirty = True

    # ---------- queries ----------

    def save_query_terms(self, terms: str) -> int:
        """Returns a new query id (search_id); the query itself is stored with record_query."""
        existing = [int(qid) for qid in self.fact_query if str(qid).isdigit()]
        return max(existing, default=0) + 1

    def record_query(self, query_id, session_id, query, method, n_results, fallback, query_terms, options=None):
        """Adds a row to fact_query, with its position in the session and its mission."""
        session = self.dim_session.get(session_id, {})
        session["n_queries"] = session.get("n_queries", 0) + 1
        previous = [q for q in self.fact_query.values() if q["session_id"] == session_id]
        previous = previous[-1] if previous else None
        if previous and set(previous["query_terms"]) & set(query_terms):
            mission_id = previous["mission_id"]
        else:
            mission_id = f"{session_id[:8]}-{session['n_queries']}"
        self.fact_query[str(query_id)] = {
            "query_id": str(query_id), "session_id": session_id, "timestamp": self.now().isoformat(),
            "query": query, "n_terms": len(query.split()), "query_terms": list(query_terms),
            "method": method, "n_results": n_results, "fallback": bool(fallback),
            "position_in_session": session["n_queries"], "mission_id": mission_id,
            "options": options or {},
        }
        self._dirty = True

    # ---------- clicks and dwell time ----------

    def record_click(self, query_id, session_id, pid, rank):
        """Adds a row to fact_click (dwell time filled later) and returns its click_id."""
        click_id = uuid.uuid4().hex[:16]
        self.fact_click[click_id] = {
            "click_id": click_id, "query_id": str(query_id) if query_id else None, "session_id": session_id,
            "pid": pid, "rank": int(rank) if str(rank or "").isdigit() else None,
            "timestamp": self.now().isoformat(), "dwell_seconds": None, "dwell_source": None,
        }
        self.fact_clicks[pid] = self.fact_clicks.get(pid, 0) + 1
        self._dirty = True
        return click_id

    def record_dwell_beacon(self, click_id, session_id, dwell_ms):
        """Dwell time sent by the details page (sendBeacon). Several beacons of the same page view
        report the time since the page loaded, so the largest one is kept. Returns True if stored."""
        click = self.fact_click.get(click_id)
        if click is None or click["session_id"] != session_id:
            return False
        seconds = round(float(dwell_ms) / 1000, 2)
        if click["dwell_source"] != "beacon" or seconds > (click["dwell_seconds"] or 0):
            click["dwell_seconds"], click["dwell_source"] = seconds, "beacon"
            self._dirty = True
        return True

    def fill_dwell_from_next_request(self, session_id):
        """Fallback: clicks of the session without dwell time get the time until this request
        (only if it is within the session timeout)."""
        now = self.now()
        for click in self.fact_click.values():
            if click["session_id"] == session_id and click["dwell_seconds"] is None:
                seconds = (now - datetime.fromisoformat(click["timestamp"])).total_seconds()
                if 0 <= seconds <= SESSION_TIMEOUT_MINUTES * 60:
                    click["dwell_seconds"], click["dwell_source"] = round(seconds, 2), "next_request"
                    self._dirty = True

    # ---------- tables for the dashboard ----------

    def dataframes(self):
        """The tables as pandas DataFrames (sessions, requests, queries, clicks)."""
        return (pd.DataFrame(list(self.dim_session.values())), pd.DataFrame(self.fact_request),
                pd.DataFrame(list(self.fact_query.values())), pd.DataFrame(list(self.fact_click.values())))

    def plot_number_of_views(self):
        """Template example chart: number of views per document (HTML)."""
        data = [{'Document ID': doc_id, 'Number of Views': count} for doc_id, count in self.fact_clicks.items()]
        df = pd.DataFrame(data, columns=['Document ID', 'Number of Views'])
        chart = alt.Chart(df).mark_bar().encode(
            x='Document ID',
            y='Number of Views'
        ).properties(
            title='Number of Views per Document'
        )
        return chart.to_html()


class ClickedDoc:
    def __init__(self, doc_id, description, counter):
        self.doc_id = doc_id
        self.description = description
        self.counter = counter

    def to_json(self):
        return self.__dict__

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)
