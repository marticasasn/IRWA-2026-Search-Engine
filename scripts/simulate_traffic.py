"""Simulated traffic, ONLY to test the analytics pipeline (sessions, queries, clicks, dwell time, dashboard).

It drives the real Flask app through its test client, with a fake clock, and writes to
data/analytics_simulated.json. It never writes to the real analytics file (data/analytics.json).
To look at the result: ANALYTICS_FILE=data/analytics_simulated.json python web_app.py, then open /dashboard.

Run from the repository root: python scripts/simulate_traffic.py [--users 40] [--seed 7]
"""

import argparse
import contextlib
import io
import os
import random
import re
import sys
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_FILE = os.path.join(ROOT, "data", "analytics.json")
SIMULATED_FILE = os.path.join(ROOT, "data", "analytics_simulated.json")

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Mobile Safari/537.36",
]
IPS = ["127.0.0.1", "192.168.1.20", "10.0.0.7", "203.0.113.5", "198.51.100.23"]  # private and documentation ranges
# Each mission is a list of queries a user may type one after the other (refinements share terms)
MISSIONS = [
    ["women cotton kurta", "women cotton kurta blue", "printed cotton kurta"],
    ["men slim fit chinos", "men chinos beige"],
    ["hooded full sleeve sweatshirt", "hooded sweatshirt men"],
    ["blue denim shorts", "denim shorts women"],
    ["women black leggings", "black tights women"],
    ["running shoes"], ["men formal shirt white"], ["laptop charger"], ["cheap black tshirt under 300"],
    ["winter jacket men", "men jacket hooded"], ["socks"], ["women saree silk"],
]
METHOD_WEIGHTS = {"custom": 0.55, "custom_title": 0.15, "bm25": 0.2, "tfidf": 0.1}


class FakeClock:
    """Clock that only moves when advance is called."""

    def __init__(self, start):
        """Starts the clock at start (aware datetime)."""
        self.current = start

    def __call__(self):
        """Current fake time."""
        return self.current

    def advance(self, seconds):
        """Moves the clock forward."""
        self.current += timedelta(seconds=seconds)


def quiet(fn, *args, **kwargs):
    """Calls fn hiding the app prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def simulate_user(app, clock, rng):
    """One user: 1-3 sessions (separated by more than 30 minutes) of 1-3 missions each."""
    client = app.test_client()
    headers = {"User-Agent": rng.choice(USER_AGENTS)}
    environ = {"REMOTE_ADDR": rng.choice(IPS)}
    get = lambda url: quiet(client.get, url, headers=headers, environ_base=environ)  # noqa: E731
    post = lambda url, data: quiet(client.post, url, data=data, headers=headers, environ_base=environ)  # noqa: E731

    for _ in range(rng.randint(1, 3)):
        get("/")
        for mission in rng.sample(MISSIONS, rng.randint(1, 3)):
            for query in mission[:rng.randint(1, len(mission))]:
                clock.advance(rng.uniform(5, 40))
                method = rng.choices(list(METHOD_WEIGHTS), weights=list(METHOD_WEIGHTS.values()))[0]
                page = post("/search", {"search-query": query, "method": method}).get_data(as_text=True)
                links = re.findall(r'href="(/doc_details\?[^"]+)"', page)
                for _ in range(min(len(links), rng.choice([0, 0, 1, 1, 1, 2, 3]))):
                    # Top results are clicked more often
                    position = min(int(rng.expovariate(0.35)), len(links) - 1)
                    clock.advance(rng.uniform(3, 20))
                    details = get(links[position].replace("&amp;", "&")).get_data(as_text=True)
                    match = re.search(r'data.append\("click_id", "(\w+)"\)', details)
                    clock.advance(rng.uniform(5, 180))
                    if match and rng.random() < 0.7:  # the beacon is not always received
                        post("/dwell", {"click_id": match.group(1),
                                        "dwell_ms": str(int(rng.uniform(5, 180) * 1000))})
        get("/stats") if rng.random() < 0.2 else None
        clock.advance(rng.uniform(31, 300) * 60)  # inactivity: the next visit is a new physical session


def main():
    """Runs the simulation and saves it to SIMULATED_FILE."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--users", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if os.path.abspath(SIMULATED_FILE) == os.path.abspath(REAL_FILE):
        raise SystemExit("Refusing to write simulated traffic to the real analytics file.")
    if os.path.exists(SIMULATED_FILE):
        os.remove(SIMULATED_FILE)  # every run starts from an empty simulated file
    os.environ["ANALYTICS_FILE"] = SIMULATED_FILE
    os.environ["GROQ_API_KEY"] = ""  # no LLM calls (load_dotenv does not override it)
    sys.path.insert(0, ROOT)
    web_app = quiet(__import__, "web_app")
    analytics = web_app.analytics_data
    if os.path.abspath(analytics.path) != os.path.abspath(SIMULATED_FILE):
        raise SystemExit("The app is not using the simulated analytics file, stopping.")

    rng = random.Random(args.seed)
    clock = FakeClock(datetime.now().astimezone().replace(hour=8, minute=0, second=0, microsecond=0)
                      - timedelta(days=2))
    analytics.clock = clock
    for _ in range(args.users):
        clock.advance(rng.uniform(0, 3) * 3600 / max(1, args.users / 10))
        simulate_user(web_app.app, clock, rng)
    analytics.save()

    sessions, requests, queries, clicks = analytics.dataframes()
    print(f"Simulated {args.users} users: {len(sessions)} sessions, {len(requests)} requests, "
          f"{len(queries)} queries, {len(clicks)} clicks")
    print("Saved to", SIMULATED_FILE)


if __name__ == "__main__":
    main()
