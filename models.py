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
        Unique identifier – typically the row index or a composite key.
    text: str
        The plain‑text representation of the row.
    source: Optional[str]
        Human‑readable source name (e.g., the CSV filename) – useful for the
        client to display where the information originated.
    """

    id: str = Field(..., description="Unique identifier for the document")
    text: str = Field(..., description="Full text content of the document")
    source: Optional[str] = Field(
        default=None, description="Optional source label (e.g., filename)"
    )

    class Config:
        orm_mode = True


class QueryRequest(BaseModel):
    """Payload for the search endpoint.

    ``top_k`` is optional; when omitted the global default from ``Settings``
    will be used by the service implementation.
    """

    query: str = Field(..., min_length=1, description="User query string")
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Number of documents to retrieve (overrides global default)",
    )

    @validator("query")
    def _strip_query(cls, v: str) -> str:
        return v.strip()


class RetrievalResponse(BaseModel):
    """Response model for the ``/search`` endpoint.

    It echoes the original query and returns the list of most relevant
    ``Document`` objects.
    """

    query: str = Field(..., description="The query that was processed")
    documents: List[Document] = Field(
        ..., description="List of documents ordered by relevance"
    )

    class Config:
        orm_mode = True


class AnswerRequest(QueryRequest):
    """Payload for the optional LLM‑augmented answer endpoint.

    Inherits all fields from :class:`QueryRequest`.  Keeping a separate subclass
    makes the OpenAPI schema clearer and leaves room for future LLM‑specific
    parameters (e.g., temperature, model name).
    """

    pass


class AnswerResponse(BaseModel):
    """Response model when an LLM is used to generate an answer.

    Attributes
    ----------
    answer: str
        The text generated by the LLM.  If the model could not find a grounded
        answer, a predefined fallback message is returned.
    sources: List[Document]
        Documents that were supplied to the LLM as context.
    message: Optional[str]
        Additional information for the client – for example, a warning that the
        answer is based on limited data.
    """

    answer: str = Field(..., description="Answer generated by the LLM")
    sources: List[Document] = Field(
        ..., description="Documents that were used as context for the answer"
    )
    message: Optional[str] = Field(
        default=None,
        description="Optional explanatory message (e.g., fallback notice)",
    )

    class Config:
        orm_mode = True


__all__ = [
    "Settings",
    "Document",
    "QueryRequest",
    "RetrievalResponse",
    "AnswerRequest",
    "AnswerResponse",
]
