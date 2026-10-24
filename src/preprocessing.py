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
def build_terms(text):
    """Full pre-processing pipeline: returns the list of processed terms of a text."""
    if not isinstance(text, str) or text.strip() == "":
        return []
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and len(t) > 1]
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens
