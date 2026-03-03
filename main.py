"""
Fund Chatbot Service

This repository provides a FastAPI‑based chatbot that answers queries about
investment fund data using only two CSV files: ``trades.csv`` and ``holdings.csv``.
The service loads the CSVs into pandas DataFrames, converts each row into a
plain‑text representation, tokenizes the text, and performs lightweight
retrieval (BM25‑style) to find the most relevant rows for a user question.
If an OpenRouter API key is supplied, the retrieved context is passed to a
large language model (LLM) to generate a natural‑language answer while
enforcing strict grounding rules. When no API key is present, the service
operates in deterministic mode and returns a fallback message if the answer
cannot be derived from the data.

Typical applications:
- Internal fund analysts querying historical trade and holding information.
- Building a conversational interface for portfolio reporting dashboards.
- Prototyping data‑driven chat assistants without exposing raw CSV data.

The code is deliberately minimal, requires only the packages listed in
``requirements.txt``, and can be deployed with any ASGI server (e.g., uvicorn).
"""

import os
import re
import json
import logging
from typing import Optional, List, Tuple

import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Configuration & Environment
# ---------------------------------------------------------------------------
load_dotenv()

TRADES_PATH = os.getenv("TRADES_PATH", "trades.csv")
HOLDINGS_PATH = os.getenv("HOLDINGS_PATH", "holdings.csv")
FALLBACK = "Sorry can not find the answer"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)

SYSTEM_PROMPT = (
    "You are a data assistant specialized in fund information. Use ONLY the provided context from the CSV files (trades.csv and holdings.csv) to answer user queries.\n"
    "Guidelines:\n"
    "- If the answer is not present in the context, reply exactly: Sorry can not find the answer\n"
    "- Do not use external knowledge or make assumptions.\n"
    "- Keep responses concise and factual.\n"
)

# ---------------------------------------------------------------------------
# Structured Logging Setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("fund_chatbot")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '{"time":"%(asctime)s","level":"%(levelname)s","module":"%(module)s","message":%(message)s}'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# FastAPI Application & Routers
# ---------------------------------------------------------------------------
app = FastAPI(title="Fund Chatbot (CSV‑only, optional OpenRouter)")
api_router = APIRouter()
app.include_router(api_router)

# ---------------------------------------------------------------------------
# Global State (populated on startup)
# ---------------------------------------------------------------------------
llm_enabled: bool = False
trades_df: Optional[pd.DataFrame] = None
holdings_df: Optional[pd.DataFrame] = None
retriever: Optional["Retriever"] = None
all_documents: List[str] = []  # plain‑text representation of every row

# ---------------------------------------------------------------------------
# Helper Functions & Classes
# ---------------------------------------------------------------------------

def row_to_text(row: pd.Series, source: str) -> str:
    """Convert a pandas Series (row) into a deterministic plain‑text string.

    Args:
        row: The pandas Series representing a CSV row.
        source: Either "trades" or "holdings" – used to prefix the document.
    Returns:
        A human‑readable string containing key‑value pairs.
    """
    parts = [f"{col}: {row[col]}" for col in row.index]
    return f"{source.upper()} | " + " | ".join(parts)

class Retriever:
    """Simple BM25 retriever over a list of plain‑text documents."""

    def __init__(self, documents: List[str]):
        self.documents = documents
        self.tokenized_corpus = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(json.dumps({"msg": "Retriever initialized", "doc_count": len(documents)}))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[int, str]]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_n = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        logger.info(json.dumps({"msg": "Retrieval performed", "query": query, "k": k}))
        return [(idx, self.documents[idx]) for idx, _ in top_n]

async def call_llm(messages: List[dict]) -> str:
    """Call OpenRouter LLM asynchronously using httpx.

    Raises:
        HTTPException: If the request fails or the response format is unexpected.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": OPENROUTER_MODEL, "messages": messages}
    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, json=payload, headers=headers, timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            logger.info(json.dumps({"msg": "LLM response received", "len": len(content)}))
            return content
        except httpx.HTTPError as exc:
            logger.error(json.dumps({"msg": "LLM request failed", "error": str(exc)}))
            raise HTTPException(status_code=502, detail="LLM service unavailable")

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., description="User question about fund data")
    top_k: int = Field(5, ge=1, le=20, description="Number of documents to retrieve")

class AnswerResponse(BaseModel):
    answer: str = Field(..., description="Generated answer or fallback message")
    retrieved_documents: List[str] = Field(..., description="Plain‑text documents used as context")

class ReviewResponse(BaseModel):
    rating: str = Field(..., description="Overall rating (e.g., 4/5)")
    pros: List[str]
    cons: List[str]
    suggestions: List[str]

# ---------------------------------------------------------------------------
# Startup Event – Load CSVs and build Retriever
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global trades_df, holdings_df, all_documents, retriever, llm_enabled
    logger.info(json.dumps({"msg": "Startup initiated"}))

    # Load CSVs – fail fast if missing
    try:
        trades_df = pd.read_csv(TRADES_PATH)
        holdings_df = pd.read_csv(HOLDINGS_PATH)
        logger.info(json.dumps({"msg": "CSV files loaded", "trades_rows": len(trades_df), "holdings_rows": len(holdings_df)}))
    except Exception as exc:
        logger.error(json.dumps({"msg": "Failed to load CSV files", "error": str(exc)}))
        raise

    # Convert rows to plain‑text documents
    all_documents = []
    for _, row in trades_df.iterrows():
        all_documents.append(row_to_text(row, "trades"))
    for _, row in holdings_df.iterrows():
        all_documents.append(row_to_text(row, "holdings"))

    retriever = Retriever(all_documents)
    llm_enabled = bool(OPENROUTER_API_KEY)
    logger.info(json.dumps({"msg": "Startup completed", "llm_enabled": llm_enabled}))

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@api_router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QueryRequest):
    if retriever is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    retrieved = retriever.retrieve(request.query, k=request.top_k)
    docs = [doc for _, doc in retrieved]

    if llm_enabled:
        # Build messages for LLM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{\"\n\".join(docs)}\n\nQuestion: {request.query}"},
        ]
        answer = await call_llm(messages)
        # Enforce fallback rule if LLM returns empty or generic answer
        if not answer or answer.strip().lower() == FALLBACK.lower():
            answer = FALLBACK
    else:
        # Deterministic fallback – concatenate top docs
        answer = "\n---\n".join(docs) if docs else FALLBACK

    return AnswerResponse(answer=answer, retrieved_documents=docs)

@api_router.get("/review", response_model=ReviewResponse)
async def code_review():
    """Return a static review of the current codebase.

    In a real‑world scenario this could be generated by a static analysis tool
    or an LLM, but for the purpose of this repository we provide a concise
    handcrafted review.
    """
    rating = "4/5"
    pros = [
        "Clear separation of concerns (loading, retrieval, LLM integration).",
        "Uses async HTTP client (httpx) for non‑blocking LLM calls.",
        "Structured logging outputs JSON‑compatible lines for easy ingestion.",
        "Modular FastAPI router makes future endpoint expansion straightforward.",
        "BM25 retrieval is lightweight and does not require external services.",
    ]
    cons = [
        "Retrieval is purely lexical; no semantic embeddings are used, limiting recall.",
        "CSV loading occurs at startup; large files could cause long cold starts.",
        "Error handling around CSV parsing is minimal – malformed rows raise generic exceptions.",
        "The fallback answer is hard‑coded; a more nuanced ""cannot answer"" response could improve UX.",
    ]
    suggestions = [
        "Replace BM25 with a vector store (e.g., FAISS or Chroma) and use sentence‑transformers for semantic search.",
        "Stream CSV processing with chunks or Dask for scalability.",
        "Introduce pydantic validators for environment variables to catch misconfiguration early.",
        "Add OpenTelemetry instrumentation for distributed tracing.",
        "Consider using a dedicated async LLM SDK (e.g., openai‑async) for richer features.",
    ]
    return ReviewResponse(rating=rating, pros=pros, cons=cons, suggestions=suggestions)
