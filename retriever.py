'''retriever.py
=================

Utility module that implements a lightweight BM25 based *Retriever* used by the FastAPI
service.  The responsibilities are:

1. Load CSV files (via :func:`load_csv_documents`) and turn each row into a plain‑text
   document.  A custom ``row_to_text`` callable can be supplied; otherwise the row values
   are concatenated with spaces.
2. Tokenise text using a very small regular‑expression based tokenizer – this keeps the
   dependency footprint tiny while still providing decent tokenisation for English‑like
   data.
3. Build a BM25 index (``rank_bm25.BM25Okapi``) over the tokenised documents.
4. Expose a :class:`Retriever` class with a :meth:`Retriever.retrieve` method that returns
   the *top‑k* most relevant documents together with their BM25 scores.
5. Provide a lightweight :func:`review_code` helper that analyses a Python source string
   and returns a rating, pros, cons and technology‑upgrade suggestions.  This is useful
   for the "review the code" feature request.

The implementation is deliberately self‑contained, type‑annotated and includes basic
error handling and logging so that it can be used both in development and production
environments.
'''"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

import pandas as pd
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Logging configuration – the host application can override the level.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Attach a simple console handler only if the application hasn't configured logging.
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%({asctime}s) - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Tokenisation utilities
# ---------------------------------------------------------------------------
_TOKEN_PATTERN = re.compile(r"\w+")
# Backward‑compatible alias expected by some older code paths.
_TOKEN_PATT = _TOKEN_PATTERN


def _tokenise(text: str) -> List[str]:
    """Return a list of lower‑cased word tokens extracted from *text*.

    The function uses a simple ``\w+`` regular expression which works well for the
    typical CSV‑based datasets used in this repository.  It deliberately avoids any
    heavy NLP libraries to keep the container image small.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    tokens = _TOKEN_PATTERN.findall(text.lower())
    logger.debug(
        "Tokenised %d characters into %d tokens", len(text), len(tokens)
    )
    return tokens


def tokenize(text: str) -> List[str]:
    """Public tokenisation helper used by external modules.

    This thin wrapper forwards to the internal ``_tokenise`` implementation.  It
    exists to preserve the original public API that some callers (including earlier
    versions of ``main.py``) rely on.
    """
    return _tokenise(text)


# ---------------------------------------------------------------------------
# CSV loading utilities
# ---------------------------------------------------------------------------
def load_csv_documents(
    csv_path: Path | str,
    *,
    row_to_text: Callable[[pd.Series], str] | None = None,
    encoding: str = "utf-8",
) -> List[str]:
    """Load a CSV file and return a list of plain‑text documents.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.  ``str`` values are converted to :class:`Path`.
    row_to_text:
        Optional callable that receives a :class:`pandas.Series` representing a row
        and returns the textual representation.  If omitted, the row values are
        concatenated with a single space.
    encoding:
        File encoding – defaults to UTF‑8.

    Returns
    -------
    List[str]
        One document per CSV row.
    """
    path = Path(csv_path)
    if not path.is_file():
        logger.error("CSV file not found: %s", path)
        raise FileNotFoundError(f"CSV file not found: {path}")

    logger.info("Loading CSV file %s", path)
    df = pd.read_csv(path, encoding=encoding)
    logger.debug("CSV contains %d rows and %d columns", df.shape[0], df.shape[1])

    if row_to_text is None:
        def default_row_to_text(row: pd.Series) -> str:
            # Convert all values to string, ignore NaN, and join with spaces.
            return " ".join(str(v) for v in row.values if pd.notna(v))

        row_to_text = default_row_to_text

    documents = []
    for idx, row in df.iterrows():
        try:
            doc = row_to_text(row)
            if not isinstance(doc, str):
                raise ValueError("row_to_text must return a string")
            documents.append(doc)
        except Exception as exc:
            logger.warning(
                "Failed to convert row %d to text: %s", idx, exc, exc_info=False
            )
    logger.info("Loaded %d documents from %s", len(documents), path)
    return documents


# ---------------------------------------------------------------------------
# Retriever implementation
# ---------------------------------------------------------------------------
class Retriever:
    """Simple BM25 based retriever.

    The class is deliberately lightweight – it stores the original documents, their
    tokenised form and a :class:`rank_bm25.BM25Okapi` index.  Retrieval is performed by
    tokenising the query, scoring against the BM25 model and returning the top‑k
    results.
    """

    def __init__(
        self,
        documents: Sequence[str] | None = None,
        *,
        bm25: BM25Okapi | None = None,
    ) -> None:
        """Create a new :class:`Retriever`.

        Parameters
        ----------
        documents:
            Optional iterable of raw document strings.  If omitted the instance can be
            populated later via :meth:`add_documents`.
        bm25:
            An existing BM25 index – mainly useful for testing or when the index is
            built elsewhere.
        """
        self._documents: List[str] = []
        self._tokenised: List[List[str]] = []
        self._bm25: BM25Okapi | None = None

        if documents is not None:
            self.add_documents(documents)
        if bm25 is not None:
            self._bm25 = bm25
        logger.debug("Retriever initialised with %d documents", len(self._documents))

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def add_documents(self, docs: Iterable[str]) -> None:
        """Add one or more documents to the retriever and rebuild the BM25 index.

        The method tokenises each document, stores the raw text and then (re)creates a
        :class:`BM25Okapi` instance.
        """
        new_docs = list(docs)
        if not new_docs:
            logger.info("No new documents supplied to add_documents")
            return
        logger.info("Adding %d documents to the Retriever", len(new_docs))
        for doc in new_docs:
            if not isinstance(doc, str):
                raise TypeError("All documents must be strings")
            self._documents.append(doc)
            self._tokenised.append(_tokenise(doc))
        # Re‑build BM25 index – this is cheap for the modest data sizes used here.
        self._bm25 = BM25Okapi(self._tokenised)
        logger.debug("BM25 index rebuilt with %d documents", len(self._documents))

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Return the *k* most relevant documents for *query*.

        Parameters
        ----------
        query:
            The user query string.
        k:
            Number of top results to return (default 5).  If ``k`` exceeds the number of
            indexed documents it is capped automatically.

        Returns
        -------
        List[Tuple[str, float]]
            A list of ``(document, score)`` tuples ordered by descending relevance.
        """
        if self._bm25 is None:
            logger.error("Attempted to retrieve without any indexed documents")
            raise RuntimeError("Retriever has no indexed documents")
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        query_tokens = _tokenise(query)
        logger.debug("Query tokenised into %d tokens", len(query_tokens))
        scores = self._bm25.get_scores(query_tokens)
        # Pair each document with its score and sort.
        doc_score_pairs = list(zip(self._documents, scores))
        doc_score_pairs.sort(key=lambda pair: pair[1], reverse=True)
        top_k = doc_score_pairs[: min(k, len(doc_score_pairs))]
        logger.info(
            "Retrieved %d results for query (requested %d)", len(top_k), k
        )
        return top_k

    # ---------------------------------------------------------------------
    # Helper properties (read‑only)
    # ---------------------------------------------------------------------
    @property
    def documents(self) -> Tuple[str, ...]:
        """Immutable view of the stored documents."""
        return tuple(self._documents)

    @property
    def index_size(self) -> int:
        """Number of documents currently indexed."""
        return len(self._documents)


# ---------------------------------------------------------------------------
# Simple code‑review helper (feature request implementation)
# ---------------------------------------------------------------------------
def review_code(source: str) -> dict:
    """Analyse *source* code and return a lightweight review.

    The analysis is heuristic and aims to provide a quick rating (1‑5), a list of
    pros, cons and suggestions for technology upgrades.  It does **not** execute the
    code – it only inspects the text.

    Returns
    -------
    dict
        ``{"rating": int, "pros": List[str], "cons": List[str], "suggestions": List[str]}``
    """
    if not isinstance(source, str):
        raise TypeError("source must be a string containing Python code")

    lines = source.splitlines()
    total_lines = len(lines)
    comment_lines = sum(1 for l in lines if l.strip().startswith("#"))
    docstring_present = any('"""' in l or "'''" in l for l in lines)
    type_hints = any('->' in l or ':' in l for l in lines if '(' in l and ')' in l)
    logging_used = "logging" in source
    async_used = "async def" in source or "await " in source

    pros = []
    cons = []
    suggestions = []

    # Pros based on simple heuristics
    if docstring_present:
        pros.append("Docstrings are present, improving readability.")
    if type_hints:
        pros.append("Type hints are used throughout the module.")
    if logging_used:
        pros.append("Structured logging is configured.")
    if total_lines < 300:
        pros.append("File size is modest, easy to maintain.")

    # Cons
    if comment_lines / max(total_lines, 1) < 0.05:
        cons.append("Few inline comments – consider adding more explanations.")
    if not async_used:
        cons.append("No async I/O – could benefit from async HTTP calls in a web service.")
    if "pandas" in source and "read_csv" in source and "chunksize" not in source:
        cons.append("CSV loading reads the whole file into memory; consider streaming for large files.")

    # Suggestions / technology upgrades
    if async_used is False:
        suggestions.append("Introduce async HTTP client (e.g., httpx) for external API calls.")
    if "pandas" in source:
        suggestions.append("For very large CSVs, evaluate using polars or Dask for better performance.")
    suggestions.append("Consider using a more modern tokenizer like spaCy or nltk for richer tokenisation.")
    suggestions.append("If the service scales, replace in‑memory BM25 with an external search engine such as Elasticsearch or Typesense.")

    # Simple rating algorithm (1‑5)
    score = 0
    score += 1 if docstring_present else 0
    score += 1 if type_hints else 0
    score += 1 if logging_used else 0
    score += 1 if async_used else 0
    rating = max(1, min(5, score))

    review = {
        "rating": rating,
        "pros": pros,
        "cons": cons,
        "suggestions": suggestions,
    }
    logger.debug("Code review generated: %s", review)
    return review


__all__ = [
    "tokenize",
    "load_csv_documents",
    "Retriever",
    "review_code",
]
