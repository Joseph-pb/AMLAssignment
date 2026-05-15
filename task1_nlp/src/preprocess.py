"""Text preprocessing for Task 1 (sentiment NLP).

Implements a toggleable preprocessing pipeline grounded in the
lecture material allowed for Task 1:

- W02_L03 motivates keeping emphasis punctuation ("!", "?") as tokens
  because they are sentiment carriers.
- W02_L04 / W05_L09 motivate min_df-style vocabulary handling downstream.
- W03_L05 motivates negation-aware stopword removal (we keep "not", "no",
  "never", etc. -- removing them would erase sentiment polarity).

Pipeline order, each step toggleable via ``config``:

    1. lowercase
    2. number normalisation on the RAW string (replace any run of digits
       with the literal token ``<NUM>``). Note that the tokeniser regex
       ``[a-zA-Z']+|[!?]`` would drop digits anyway, so this toggle
       effectively chooses between *dropping* and *substituting* numbers.
    3. tokenisation with NLTK ``RegexpTokenizer(r"[a-zA-Z']+|[!?]")``
       which preserves "!" and "?" as standalone tokens.
    4. stopword removal using NLTK English stopwords MINUS a hand-curated
       negation list (so "not bad" survives as ["not", "bad"]).
    5. lemmatisation with NLTK ``WordNetLemmatizer`` using POS="v"
       (verbs are the high-leverage POS for sentiment).

``porter_stem`` is exposed separately for the stemming-vs-lemmatisation
comparison required by section 3.2 of the revised plan.
"""

from __future__ import annotations

import re
import sys
from typing import Optional

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

for _pkg in ("punkt", "stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{_pkg}" if _pkg != "punkt" else f"tokenizers/{_pkg}")
    except LookupError:
        try:
            nltk.download(_pkg, quiet=True)
        except Exception as e:  # noqa: BLE001
            print(f"[preprocess] NLTK download of '{_pkg}' failed: {e}", file=sys.stderr)

from nltk.corpus import stopwords as _stopwords  # noqa: E402

NEGATION_WORDS: set[str] = {
    "not", "no", "nor", "never", "none", "nobody", "nothing",
    "neither", "nowhere", "cannot",
}

_BASE_STOPWORDS: set[str] = set(_stopwords.words("english")) - NEGATION_WORDS

DEFAULT_CONFIG: dict = {
    "lowercase": True,
    "replace_digits": True,
    "keep_emphasis_punct": True,
    "remove_stopwords": True,
    "lemmatise": True,
}

_TOKENIZER_KEEP_PUNCT = RegexpTokenizer(r"[a-zA-Z']+|[!?]")
_TOKENIZER_NO_PUNCT = RegexpTokenizer(r"[a-zA-Z']+")
_DIGIT_RE = re.compile(r"\d+")
_LEMMATISER = WordNetLemmatizer()
_STEMMER = PorterStemmer()


def preprocess(text: str, config: Optional[dict] = None) -> list[str]:
    """Apply the configured preprocessing pipeline to ``text``."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    if cfg["lowercase"]:
        text = text.lower()

    if cfg["replace_digits"]:
        text = _DIGIT_RE.sub(" <NUM> ", text)

    tokenizer = _TOKENIZER_KEEP_PUNCT if cfg["keep_emphasis_punct"] else _TOKENIZER_NO_PUNCT
    tokens: list[str] = tokenizer.tokenize(text)

    if cfg["replace_digits"]:
        tokens = ["<NUM>" if t.lower() == "num" else t for t in tokens]

    if cfg["remove_stopwords"]:
        tokens = [t for t in tokens if t not in _BASE_STOPWORDS]

    if cfg["lemmatise"]:
        tokens = [_LEMMATISER.lemmatize(t, pos="v") if t not in {"!", "?", "<NUM>"} else t
                  for t in tokens]

    return tokens


def identity_analyzer(tokens: list[str]) -> list[str]:
    """Pass-through analyzer for ``TfidfVectorizer`` when input is already tokenised.

    Defined at module level (not as a lambda) so the fitted vectoriser is picklable.
    """
    return tokens


def porter_stem(tokens: list[str]) -> list[str]:
    """Porter-stem a token list (for the stemming-vs-lemmatisation comparison)."""
    return [_STEMMER.stem(t) if t not in {"!", "?", "<NUM>"} else t for t in tokens]


if __name__ == "__main__":
    samples = [
        ("positive", "I absolutely loved this film! The acting was brilliant."),
        ("negative", "This movie was not bad, actually -- but the ending? Terrible."),
        ("spam",     "Subject: Enron deal closing 12/03. Please unsubscribe if not interested."),
    ]
    for tag, s in samples:
        print(f"--- {tag} ---")
        print("IN :", s)
        print("OUT:", preprocess(s))
        print()
