"""Text preprocessing, copied from the Part 1 notebook (same behaviour)."""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

stemmer = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))

# Normalize common spelling variations in fashion products
NORMALIZATION_RULES = [
    (r"\bt[\s-]?shirts?\b", "tshirt"),
    (r"\btees?\b", "tshirt"),
    (r"\bmen'?s\b", "men"),
    (r"\bwomen'?s\b", "women"),
    (r"\bboy'?s\b", "boys"),
    (r"\bgirl'?s\b", "girls"),
    (r"\b(slim|regular|skinny|relaxed|loose)fit\b", r"\1 fit"),
    (r"\bpolycotton\b", "poly cotton"),
]

# Web engine only (Part 4): words that Porter stemming merges with unrelated words, e.g.
# "leggings" -> "leg", "shorts" -> "short" (short sleeve), "dressed" -> "dress".
# Each word maps to the term stored in the index instead of its stem. Not used by default,
# so the Part 1-3 results stay reproducible.
PROTECTED_TERMS = {
    "leggings": "legging", "legging": "legging",
    "jeggings": "jegging", "jegging": "jegging",
    "shorts": "shorts",
    "tights": "tights",
    "chinos": "chino", "chino": "chino",
    "dress": "dress", "dresses": "dress",
    "dressed": "dressed", "dressing": "dressing",
    "suited": "suited",
}


# Clean the text before tokenization
def clean_text(text):
    """Steps 1-4: lowercase, remove URLs and non-ASCII, normalize variants, remove punctuation."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.encode("ascii", errors="ignore").decode()
    for pattern, replacement in NORMALIZATION_RULES:
        text = re.sub(pattern, replacement, text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    return text


# Apply the preprocessing steps and return the processed words
def build_terms(text, protected=None):
    """Full pre-processing pipeline: returns the list of processed terms of a text.

    protected (optional) maps words to the term used instead of their stem (see PROTECTED_TERMS);
    with the default None the output is exactly the Part 1-3 one.
    """
    if not isinstance(text, str) or text.strip() == "":
        return []
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and len(t) > 1]
    tokens = [t for t in tokens if t not in STOP_WORDS]
    if protected:
        return [protected.get(t) or stemmer.stem(t) for t in tokens]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens


def build_terms_web(text):
    """Pre-processing of the web engine: build_terms with the PROTECTED_TERMS not stemmed."""
    return build_terms(text, PROTECTED_TERMS)
