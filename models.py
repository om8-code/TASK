'''models.py
=================
This module defines all Pydantic models used by the FastAPI service.
It includes:

* **Settings** – a ``BaseSettings`` subclass that reads configuration from
  environment variables (or a ``.env`` file).  The service currently needs the
  optional OpenRouter/Groq API keys and a default ``top_k`` value for the
  BM25 retriever.
* **Document** – a lightweight representation of a CSV row that has been
  transformed into plain‑text.  It is returned to the client as part of the
  retrieval payload.
* **QueryRequest** – the payload accepted by the ``/search`` endpoint.  It
  contains the user query and an optional ``top_k`` that overrides the global
  default.
* **RetrievalResponse** – the response model for the ``/search`` endpoint.
  It returns the original query together with the list of most‑relevant
  ``Document`` objects.
* **AnswerRequest** – the payload accepted by the optional ``/answer``
  endpoint when an LLM is configured.  It re‑uses ``QueryRequest`` but is kept
  separate for clarity and future extensibility.
* **AnswerResponse** – the response model for the LLM‑augmented endpoint.  It
  contains the generated answer, the list of source documents that were fed to
  the model and an optional ``message`` field that explains why no answer could
  be produced.
* **CodeReviewRequest** – a new model that accepts raw source code (and an
  optional language hint) for automated review.
* **CodeReviewResponse** – a new model that returns a rating, pros, cons,
  suggestions, and an optional alternative technology recommendation.

All models are deliberately simple, type‑annotated and include helpful
validation where appropriate.  They are ready for production use and can be
imported from any part of the code‑base without causing circular imports.
'''  # noqa: D400, D401

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, BaseSettings, Field, validator


class Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    The service can operate in two modes:

    * **Pure retrieval** – only the BM25 retriever is used.  In this case no
      API keys are required.
    * **Retrieval + LLM** – if either ``OPENROUTER_API_KEY`` or ``GROQ_API_KEY``
      is present the request can be forwarded to the corresponding LLM provider.
    """

    openrouter_api_key: Optional[str] = Field(
        default=None, env="OPENROUTER_API_KEY", description="API key for OpenRouter"
    )
    groq_api_key: Optional[str] = Field(
        default=None, env="GROQ_API_KEY", description="API key for Groq"
    )
    top_k: int = Field(
        default=5,
        env="TOP_K",
        ge=1,
        le=100,
        description="Default number of documents to retrieve for a query",
    )

    class Config:
        env_file = ".env"
        case_sensitive = False
        env_file_encoding = "utf-8"

    @property
    def llm_enabled(self) -> bool:
        """Return ``True`` when an LLM integration key is available."""
        return bool(self.openrouter_api_key or self.groq_api_key)


class Document(BaseModel):
    """A single document derived from a CSV row.

    Attributes
    ----------
    id: str
        Unique identifier – typically the original CSV row index.
    text: str
        Plain‑text representation of the row, suitable for BM25 indexing.
    """

    id: str = Field(..., description="Unique document identifier")
    text: str = Field(..., description="Plain‑text content of the document")

    class Config:
        orm_mode = True


class QueryRequest(BaseModel):
    """Payload for the ``/search`` endpoint.

    The ``top_k`` field overrides the global default defined in ``Settings``.
    """

    query: str = Field(..., min_length=1, description="User search query")
    top_k: Optional[int] = Field(
        None,
        ge=1,
        le=100,
        description="Number of documents to retrieve; falls back to Settings.top_k",
    )

    @validator("top_k")
    def _positive_top_k(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError("top_k must be at least 1")
        return v


class RetrievalResponse(BaseModel):
    """Response model for the ``/search`` endpoint."""

    query: str = Field(..., description="The original user query")
    results: List[Document] = Field(..., description="List of retrieved documents")

    class Config:
        orm_mode = True


class AnswerRequest(QueryRequest):
    """Payload for the optional ``/answer`` endpoint.

    Inherits from ``QueryRequest`` to keep the API surface consistent while
    allowing future extensions specific to LLM‑driven answering.
    """

    # No additional fields for now – placeholder for future use.
    pass


class AnswerResponse(BaseModel):
    """Response model for the LLM‑augmented endpoint.

    If the LLM cannot produce a confident answer, ``answer`` may be ``None`` and
    ``message`` will contain an explanatory note.
    """

    answer: Optional[str] = Field(
        None, description="Generated answer text, if available"
    )
    sources: List[Document] = Field(
        ..., description="Documents that were supplied to the LLM as context"
    )
    message: Optional[str] = Field(
        None,
        description="Human‑readable explanation when no answer could be generated",
    )

    class Config:
        orm_mode = True


# ---------------------------------------------------------------------------
# New models for automated code review functionality
# ---------------------------------------------------------------------------

class CodeReviewRequest(BaseModel):
    """Request model for submitting source code to be reviewed.

    Attributes
    ----------
    code: str
        The raw source code that should be analysed.
    language: str, optional
        Programming language hint (e.g., ``"python"``).  Defaults to ``"python"``
        because the repository primarily contains Python code.
    """

    code: str = Field(..., min_length=1, description="Source code to review")
    language: str = Field(
        "python",
        description="Programming language of the supplied code; defaults to Python",
    )

    @validator("code")
    def _non_empty_code(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("code must contain non‑whitespace characters")
        return v


class CodeReviewResponse(BaseModel):
    """Response model containing a structured review of submitted code.

    The service evaluates the code and returns:

    * ``rating`` – an integer from 1 (poor) to 5 (excellent).
    * ``pros`` – a list of positive aspects.
    * ``cons`` – a list of drawbacks or potential issues.
    * ``suggestions`` – actionable recommendations for improvement.
    * ``alternative_technology`` – optional suggestion of a different tech
      stack or library that could better solve the problem.
    """

    rating: int = Field(
        ..., ge=1, le=5, description="Overall quality rating on a 1‑5 scale"
    )
    pros: List[str] = Field(..., description="Positive aspects of the code")
    cons: List[str] = Field(..., description="Negative aspects or risks")
    suggestions: List[str] = Field(
        ..., description="Actionable improvement recommendations"
    )
    alternative_technology: Optional[str] = Field(
        None,
        description="Suggested alternative technology, library, or framework",
    )

    @validator("pros", "cons", "suggestions", each_item=True)
    def _non_empty_strings(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("list items must be non‑empty strings")
        return v.strip()

    @validator("rating")
    def _rating_range(cls, v: int) -> int:
        if not (1 <= v <= 5):
            raise ValueError("rating must be between 1 and 5")
        return v

    class Config:
        orm_mode = True


__all__ = [
    "Settings",
    "Document",
    "QueryRequest",
    "RetrievalResponse",
    "AnswerRequest",
    "AnswerResponse",
    "CodeReviewRequest",
    "CodeReviewResponse",
]
