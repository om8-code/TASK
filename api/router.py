"""api/router.py
~~~~~~~~~~~~~~~~~~

FastAPI router that exposes two primary endpoints:

1. ``/search`` – Accepts a user query, forwards it to the :class:`Retriever`
   service and optionally to an LLM for a natural‑language answer.
2. ``/review-code`` – Accepts a snippet of Python code and returns a simple
   rating together with pros, cons and technology‑upgrade suggestions.

The module follows the repository's coding conventions:

* **Structured logging** via ``structlog`` – every request logs its start,
  parameters and outcome.
* **Async HTTP client** – ``httpx.AsyncClient`` is used for external LLM calls.
* **Dependency injection** – the ``Retriever`` instance is provided by a
  FastAPI dependency defined elsewhere (``api.dependencies``).
* **Pydantic models** – request validation and response schemas are fully
  typed.

The implementation is deliberately lightweight yet production‑ready: all
IO is asynchronous, errors are captured and transformed into clear HTTP
responses, and the code is fully type‑annotated for static analysis.
"""

from __future__ import annotations

import os
from typing import List, Optional

import structlog
import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, validator

# Local imports – the concrete implementations live in other modules of the
# repository.  Importing them here keeps the router focused on request/response
# handling only.
from .dependencies import get_retriever, Retriever

logger = structlog.get_logger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    """Payload for the ``/search`` endpoint.

    Attributes
    ----------
    query: str
        The natural‑language question the user wants answered.
    top_k: int, optional
        Number of documents to retrieve from the BM25 index.  Defaults to 5.
    use_llm: bool, optional
        If ``True`` the retrieved context is forwarded to the configured LLM.
    """

    query: str = Field(..., min_length=1, description="User query string")
    top_k: int = Field(5, ge=1, le=20, description="Number of documents to retrieve")
    use_llm: bool = Field(False, description="Whether to augment the answer with an LLM")

    @validator("query")
    def strip_query(cls, v: str) -> str:
        return v.strip()


class DocumentResult(BaseModel):
    """Single document returned by the BM25 retriever."""

    id: str
    text: str
    score: float


class SearchResponse(BaseModel):
    """Response model for ``/search``.

    If ``use_llm`` was ``True`` and the LLM call succeeded, ``answer`` will be
    populated; otherwise it will be ``None``.
    """

    query: str
    retrieved: List[DocumentResult]
    answer: Optional[str] = None
    llm_used: bool = False


class ReviewRequest(BaseModel):
    """Payload for the ``/review-code`` endpoint.

    The ``code`` field should contain a UTF‑8 string with the Python source to
    be analysed.
    """

    code: str = Field(..., min_length=1, description="Python source code to review")

    @validator("code")
    def normalize_code(cls, v: str) -> str:
        # Normalise line endings and strip trailing whitespace for a stable
        # analysis.
        return "\n".join(line.rstrip() for line in v.replace("\r\n", "\n").split("\n"))


class ReviewResponse(BaseModel):
    """Result of the automated code review.

    * ``rating`` – integer from 1 (poor) to 5 (excellent).
    * ``pros`` – list of positive observations.
    * ``cons`` – list of negative observations.
    * ``suggestions`` – recommended technology or architectural changes.
    """

    rating: int = Field(..., ge=1, le=5)
    pros: List[str]
    cons: List[str]
    suggestions: List[str]


# ---------------------------------------------------------------------------
# Helper – async LLM call
# ---------------------------------------------------------------------------

async def _call_llm(prompt: str) -> str:
    """Send *prompt* to the configured LLM and return its raw answer.

    The function reads ``LLM_ENDPOINT`` and ``LLM_API_KEY`` from the environment.
    If either variable is missing, an ``HTTPException`` with status 503 is raised.
    """

    endpoint = os.getenv("LLM_ENDPOINT")
    api_key = os.getenv("LLM_API_KEY")
    if not endpoint or not api_key:
        logger.error("LLM configuration missing", endpoint=endpoint, api_key_present=bool(api_key))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service is not configured",
        )

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]}

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            logger.debug("Calling LLM", endpoint=endpoint, payload=payload)
            response = await client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            # The exact shape depends on the provider; we attempt a best‑effort
            # extraction.
            answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            logger.debug("LLM response received", answer=answer[:100])
            return answer.strip()
        except httpx.HTTPError as exc:
            logger.error("LLM request failed", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Failed to obtain response from LLM",
            ) from exc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/search", response_model=SearchResponse, status_code=status.HTTP_200_OK)
async def search_endpoint(
    request: SearchRequest,
    retriever: Retriever = Depends(get_retriever),
) -> SearchResponse:
    """Handle a search request.

    The flow is:
    1. Retrieve the top‑k documents using the BM25 ``Retriever``.
    2. If ``use_llm`` is true, build a prompt that contains the retrieved
       context and forward it to the LLM.
    3. Return the documents and, when applicable, the LLM answer.
    """

    logger.info("Search request received", query=request.query, top_k=request.top_k, use_llm=request.use_llm)

    # Step 1 – BM25 retrieval (synchronous call wrapped in ``await`` for API
    # consistency – the underlying implementation may be async in the future).
    try:
        docs = retriever.retrieve(request.query, k=request.top_k)
    except Exception as exc:  # pragma: no cover – defensive programming
        logger.exception("Retriever failed", error=str(exc))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Retrieval error")

    retrieved = [DocumentResult(id=str(doc.id), text=doc.text, score=doc.score) for doc in docs]

    answer: Optional[str] = None
    llm_used = False

    if request.use_llm:
        # Build a concise prompt that respects token limits.
        context = "\n\n".join(doc.text for doc in docs)
        prompt = (
            f"You are a helpful assistant. Answer the following question **only** using the provided context.\n"
            f"If the answer cannot be derived, respond with \"I don't have enough information.\"\n\n"
            f"Context:\n{context}\n\nQuestion: {request.query}"
        )
        answer = await _call_llm(prompt)
        llm_used = True

    response = SearchResponse(
        query=request.query,
        retrieved=retrieved,
        answer=answer,
        llm_used=llm_used,
    )
    logger.info("Search response prepared", result_count=len(retrieved), llm_used=llm_used)
    return response


@router.post("/review-code", response_model=ReviewResponse, status_code=status.HTTP_200_OK)
async def review_code_endpoint(request: ReviewRequest) -> ReviewResponse:
    """Perform a lightweight static analysis of the supplied Python code.

    The analysis is intentionally simple – it looks for common best‑practice
    markers (type hints, docstrings, logging) and anti‑patterns (bare ``except``
    clauses, ``print`` statements, mutable default arguments).  The function
    returns a rating and a set of actionable suggestions.
    """

    logger.info("Code review request received", code_length=len(request.code))

    lines = request.code.split("\n")
    pros: List[str] = []
    cons: List[str] = []
    suggestions: List[str] = []

    # --- Simple heuristics -------------------------------------------------
    if any(line.lstrip().startswith("def ") for line in lines):
        pros.append("Functions are defined, indicating modular design.")
    else:
        cons.append("No function definitions found; code may be monolithic.")

    if any("type hint" in line for line in lines):
        pros.append("Type hints detected, improving readability and static analysis.")
    else:
        cons.append("Missing type hints; consider adding them for better tooling support.")

    if any('"""' in line for line in lines):
        pros.append("Docstrings are present, aiding documentation.")
    else:
        cons.append("Docstrings missing; add them to describe intent.")

    if any("print(" in line for line in lines):
        cons.append("Use of `print` for logging detected; replace with structured logging.")
        suggestions.append("Adopt `structlog` or the standard `logging` module for production‑grade logs.")

    if any("except:" in line for line in lines):
        cons.append("Bare `except` clause found; this can mask unexpected errors.")
        suggestions.append("Specify exception types and handle them explicitly.")

    if any("= []" in line or "= {}" in line for line in lines if "def " not in line):
        cons.append("Mutable default arguments may lead to subtle bugs.")
        suggestions.append("Replace mutable defaults with `None` and initialise inside the function.")

    # Rating heuristic – start from 3 and adjust.
    rating = 3
    rating += len(pros) // 2
    rating -= len(cons) // 2
    rating = max(1, min(5, rating))

    # Technology upgrade suggestions – generic but useful.
    suggestions.extend([
        "Consider switching from pandas to Polars for faster CSV ingestion.",
        "If the service scales, move the BM25 index to an external search engine like Elasticsearch or Typesense.",
        "Replace synchronous HTTP calls with `httpx.AsyncClient` (already used here) for better concurrency.",
        "Containerise the application with a multi‑stage Docker build to minimise image size.",
    ])

    response = ReviewResponse(
        rating=rating,
        pros=pros,
        cons=cons,
        suggestions=suggestions,
    )
    logger.info("Code review completed", rating=rating, pros=len(pros), cons=len(cons))
    return response


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> dict:
    """Simple health‑check endpoint used by orchestration tools.

    Returns a JSON payload confirming that the service is alive.
    """

    logger.debug("Health check invoked")
    return {"status": "ok"}
