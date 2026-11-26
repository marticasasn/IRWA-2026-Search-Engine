"""Improved RAG generator. The template's RAGGenerator (rag.py) is kept as the baseline.

Improvements, each one switchable in the constructor:
1. rich_context: structured product lines (pid, title, brand, category, prices, discount, rating, stock,
   truncated description) instead of only pid and title.
2. filter_results: only the top max_products results above a relative score threshold; out of stock
   products are marked (or excluded with exclude_out_of_stock).
3. handle_no_good: with 0 results, OR fallback results or nothing above the threshold the LLM is not
   asked to pick a product; an honest message is returned (optionally with rephrasing tips from the LLM).
4. strict_prompt: the prompt only allows the given products, requires citing them by pid with price and
   rating, fixes a short output format, and the call uses temperature 0.
   verify_citations: every pid cited in the answer must be one of the products given to the LLM.
"""

import os
import re

from dotenv import load_dotenv
from groq import Groq

from myapp.generation.rag import RAGGenerator

load_dotenv()  # take environment variables from .env

# Pids of the dataset: 16 upper case letters or digits
PID_PATTERN = re.compile(r"\b[A-Z0-9]{16}\b")

# Answer the LLM must give when no product fits (handled in code)
NO_MATCH_TOKEN = "NO_GOOD_MATCH"

UNAVAILABLE_MESSAGE = "RAG is not available. Check your credentials (.env file) or account limits."
NO_RESULTS_MESSAGE = "No products match this search, so there is no product to recommend."
FALLBACK_MESSAGE = ("No product contains all the words of the search; the results only match some of them, "
                    "so no product is recommended.")
BELOW_THRESHOLD_MESSAGE = "None of the retrieved products is a close enough match to recommend one."
NO_MATCH_MESSAGE = "None of the retrieved products fits the request well enough to recommend one."
INVALID_CITATION_MESSAGE = ("The generated answer cited products that were not in the search results, "
                            "so it was discarded.")

STRICT_PROMPT = """You are a shopping assistant of a fashion e-commerce site.
Use ONLY the products listed below. Never invent products, prices, ratings or features.

## User request
{user_query}

## Products (one per line, the PID is in square brackets)
{products}

## Rules
- Pick the single product that best fits the request and cite it by its PID in square brackets, exactly as written.
- Justify the choice in one or two sentences that mention its price and its rating.
- Optionally give one alternative, also cited by PID with its price and rating.
- Respect any constraint of the request (price limit, colour, gender, size). Out of stock products should not be the best product.
- If no product fits the request, answer exactly {no_match} and nothing else.

## Output format (exactly these three lines, no other text)
Best product: [PID] title
Why: one or two sentences with price and rating
Alternative: [PID] title - short reason (or "Alternative: none")
"""

REPHRASE_PROMPT = """A user searched a fashion e-commerce site (clothing, footwear, bags and accessories) for:
"{user_query}"
The search found no good matching products. In at most two short sentences, suggest how the user could
rephrase the search (for example more general words or a product type that this kind of shop sells).
Do not recommend specific products."""


def _price(value):
    """Price text with the rupee sign, or 'n/a' if missing."""
    return f"Rs {value}" if value not in (None, "") else "n/a"


def format_product_rich(result, description_chars=200):
    """One structured line with the main fields of a result dict (improvement 1)."""
    description = " ".join((result.get("description") or "").split())
    if len(description) > description_chars:
        description = description[:description_chars].rsplit(" ", 1)[0] + "..."
    category = " / ".join(x for x in (result.get("category"), result.get("sub_category")) if x)
    rating = result.get("average_rating")
    parts = [
        f"[{result['pid']}] {' '.join((result.get('title') or '').split())}",
        f"Brand: {result.get('brand') or 'n/a'}",
        f"Category: {category or 'n/a'}",
        f"Price: {_price(result.get('selling_price'))} (MRP {_price(result.get('actual_price'))}, "
        f"discount {result.get('discount') or 'none'})",
        f"Rating: {rating + '/5' if rating else 'no rating'}",
        "OUT OF STOCK" if result.get("out_of_stock") else "In stock",
        f"Description: {description or 'none'}",
    ]
    return " | ".join(parts)


def format_product_basic(result):
    """The baseline line: only pid and title."""
    return f"- PID: {result['pid']}, Title: {result['title']}"


def check_citations(text, allowed_pids):
    """Returns (cited pids, cited pids that are not in allowed_pids), in order of appearance."""
    cited = list(dict.fromkeys(PID_PATTERN.findall(text or "")))
    return cited, [pid for pid in cited if pid not in allowed_pids]


class ImprovedRAGGenerator:
    """RAG summary with richer context, result filtering, no-good handling and a strict, verified prompt."""

    def __init__(self, rich_context=True, filter_results=True, handle_no_good=True, strict_prompt=True,
                 verify_citations=True, max_products=6, relative_threshold=0.5, min_score=None,
                 exclude_out_of_stock=False, suggest_rephrase=True, description_chars=200):
        """Every improvement can be switched off; the numbers configure the filtering (see the module doc).

        relative_threshold keeps results with score >= relative_threshold * best score; min_score
        (optional) is an absolute minimum score.
        """
        self.rich_context = rich_context
        self.filter_results = filter_results
        self.handle_no_good = handle_no_good
        self.strict_prompt = strict_prompt
        self.verify_citations = verify_citations
        self.max_products = max_products
        self.relative_threshold = relative_threshold
        self.min_score = min_score
        self.exclude_out_of_stock = exclude_out_of_stock
        self.suggest_rephrase = suggest_rephrase
        self.description_chars = description_chars

    def select_products(self, results):
        """Improvement 2: top max_products results above the score thresholds (out of stock excluded
        if exclude_out_of_stock). Without filter_results, the first 20 results as in the baseline."""
        if not self.filter_results:
            return list(results[:20])
        candidates = [r for r in results if not (self.exclude_out_of_stock and r.get("out_of_stock"))]
        if not candidates:
            return []
        best = max(r["score"] for r in candidates)
        selected = [r for r in candidates
                    if r["score"] >= self.relative_threshold * best
                    and (self.min_score is None or r["score"] >= self.min_score)]
        return selected[:self.max_products]

    def build_prompt(self, user_query, products):
        """Prompt for the LLM with the selected products (strict or baseline template)."""
        if self.rich_context:
            lines = [format_product_rich(p, self.description_chars) for p in products]
        else:
            lines = [format_product_basic(p) for p in products]
        if self.strict_prompt:
            return STRICT_PROMPT.format(user_query=user_query, products="\n".join(lines), no_match=NO_MATCH_TOKEN)
        return RAGGenerator.PROMPT_TEMPLATE.format(retrieved_results="\n".join(lines), user_query=user_query)

    def _complete(self, prompt, temperature=None):
        """Calls the Groq chat API; returns the generated text. Raises if no key is configured."""
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key or api_key.startswith("<"):
            raise RuntimeError("GROQ_API_KEY is not set")
        client = Groq(api_key=api_key)
        kwargs = {"temperature": temperature} if temperature is not None else {}
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
            **kwargs,
        )
        return completion.choices[0].message.content.strip()

    def _no_good(self, status, message, user_query):
        """Answer for the no-good cases (improvement 3), optionally with rephrasing tips from the LLM."""
        text = message
        if self.suggest_rephrase:
            try:
                tips = self._complete(REPHRASE_PROMPT.format(user_query=user_query), temperature=0)
                if tips:
                    text += "\n" + tips
            except Exception as e:
                print(f"Error during RAG rephrase suggestion: {e}")
        return {"text": text, "status": status, "products": [], "cited": [], "invalid": []}

    def generate_response(self, user_query, retrieved_results, fallback=False):
        """Generates the summary for the retrieved results (list of result dicts of our engine).

        fallback tells whether the results come from the engine's OR fallback. Returns a dict with
        text, status ("ok", "no_results", "fallback", "below_threshold", "no_match", "invalid_citation",
        "unverified", "unavailable"), products (pids given to the LLM), cited and invalid (cited pids that
        were not given).
        """
        if self.handle_no_good:
            if not retrieved_results:
                return self._no_good("no_results", NO_RESULTS_MESSAGE, user_query)
            if fallback:
                return self._no_good("fallback", FALLBACK_MESSAGE, user_query)

        products = self.select_products(retrieved_results)
        if not products:
            if self.handle_no_good:
                return self._no_good("below_threshold", BELOW_THRESHOLD_MESSAGE, user_query)
            return {"text": NO_RESULTS_MESSAGE, "status": "no_results", "products": [], "cited": [], "invalid": []}

        pids = [p["pid"] for p in products]
        try:
            text = self._complete(self.build_prompt(user_query, products),
                                  temperature=0 if self.strict_prompt else None)
        except Exception as e:
            print(f"Error during RAG generation: {e}")
            return {"text": UNAVAILABLE_MESSAGE, "status": "unavailable", "products": pids, "cited": [], "invalid": []}

        if self.handle_no_good and NO_MATCH_TOKEN in text:
            return {"text": NO_MATCH_MESSAGE, "status": "no_match", "products": pids, "cited": [], "invalid": []}

        cited, invalid = check_citations(text, set(pids))
        status = "ok"
        if self.verify_citations:
            if invalid:
                return {"text": INVALID_CITATION_MESSAGE, "status": "invalid_citation", "products": pids,
                        "cited": cited, "invalid": invalid, "discarded_text": text}
            if not cited:
                status = "unverified"  # no pid cited, so nothing could be checked
        return {"text": text, "status": status, "products": pids, "cited": cited, "invalid": invalid}
