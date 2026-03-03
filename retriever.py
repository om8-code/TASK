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

The implementation is deliberately self‑contained, type‑annotated and includes basic
error handling and logging so that it can be used both in development and production
environments.
'''

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
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
    logger.debug("Tokenised %d characters into %d tokens", len(text), len(tokens))
    return tokens
+
+
+def tokenize(text: str) -> List[str]:
+    """Public tokenisation helper used by external modules.
+
+    This thin wrapper forwards to the internal ``_tokenise`` implementation.
+    It exists to preserve the original public API that some callers (including
+    earlier versions of ``main.py``) expect.
+    """
+    return _tokenise(text)
*** End Patch ***

# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------
def load_csv_documents(
    csv_path: str | Path,
    *,
    row_to_text: Callable[[pd.Series], str] | None = None,
    encoding: str = "utf-8",
) -> List[str]:
    """Load a CSV file and convert each row into a plain‑text document.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.
    row_to_text:
        Optional callable that receives a :class:`pandas.Series` representing a row
        and returns a string.  If omitted, the default implementation concatenates all
        column values separated by a single space.
    encoding:
        File encoding – defaults to UTF‑8.

    Returns
    -------
    List[str]
        A list where each element corresponds to a row in the CSV file.
    """
    path = Path(csv_path)
    if not path.is_file():
        logger.error("CSV file not found: %s", path)
        raise FileNotFoundError(f"CSV file not found: {path}")

    logger.info("Loading CSV data from %s", path)
    df = pd.read_csv(path, encoding=encoding)
    logger.debug("CSV contains %d rows and %d columns", df.shape[0], df.shape[1])

    if row_to_text is None:
        def _default_row_to_text(row: pd.Series) -> str:
            # Convert every value to string, replace NaN with empty string.
            return " ".join(str(v) for v in row.fillna("").values)
        row_to_text = _default_row_to_text

    documents = []
    for idx, row in df.iterrows():
        try:
            doc = row_to_text(row)
            if not isinstance(doc, str):
                raise ValueError("row_to_text must return a string")
            documents.append(doc)
        except Exception as exc:  # pragma: no cover – defensive programming
            logger.warning("Failed to convert row %d to text: %s", idx, exc)
    logger.info("Converted %d rows into documents", len(documents))
    return documents
*** End Patch ***

# ---------------------------------------------------------------------------
# Retriever implementation
# ---------------------------------------------------------------------------
class Retriever:
    """BM25 based retriever for a static collection of plain‑text documents.

    The class is deliberately immutable after construction – adding new documents
    requires creating a new instance (or using :meth:`add_documents` which rebuilds the
    underlying BM25 model).  This design keeps the internal state simple and thread‑safe
    for the typical FastAPI usage pattern where a single instance is created at
    application start‑up.
    """

    def __init__(self, documents: Sequence[str]):
        if not documents:
            raise ValueError("Retriever requires at least one document")
        self._documents: List[str] = list(documents)
        logger.debug("Initialising Retriever with %d documents", len(self._documents))
        self._tokenised_corpus: List[List[str]] = [_tokenise(doc) for doc in self._documents]
        self._bm25 = BM25Okapi(self._tokenised_corpus)
        logger.info("BM25 index built for %d documents", len(self._documents))
*** End Patch ***

*** End Patch ***
    
*** End Patch ***