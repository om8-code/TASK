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
from typing import Optional, List, Tuple

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# rank_bm25 is used for BM25 retrieval
from rank_bm25 import BM25Okapi

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
TRADES_PATH = os.getenv("TRADES_PATH", "trades.csv")
HOLDINGS_PATH = os.getenv("HOLDINGS_PATH", "holdings.csv")
FALLBACK = "Sorry can not find the answer"

# OpenRouter configuration – optional LLM integration
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
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(title="Fund Chatbot (CSV‑only, optional OpenRouter)")

# Global state – populated in the startup event
llm_enabled: bool = False
trades_df: Optional[pd.DataFrame] = None
holdings_df: Optional[pd.DataFrame] = None
retriever: Optional["Retriever"] = None
all_documents: List[str] = []  # plain‑text representation of every row

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def row_to_text(row: pd.Series) -> str:
    """Convert a pandas Series (a CSV row) into a readable plain‑text string.

    Example output: "date: 2023-01-01; ticker: AAPL; quantity: 100; price: 150.0"
    """
    parts = []
    for col, val in row.items():
        # Convert NaN to empty string for readability
        if pd.isna(val):
            continue
        parts.append(f"{col}: {val}")
    return "; ".join(parts)


def tokenize(text: str) -> List[str]:
    """Simple whitespace‑agnostic tokenizer returning lower‑cased word tokens.
    Non‑alphanumeric characters are stripped.
    """
    return [tok.lower() for tok in re.findall(r"\w+", text)]

# ---------------------------------------------------------------------------
# Retriever implementation using BM25
# ---------------------------------------------------------------------------
class Retriever:
    def __init__(self, documents: List[str]):
        self.documents = documents
        tokenized_corpus = [tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def get_top_k(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Return the top *k* documents and their BM25 scores for *query*.
        The return value is a list of tuples ``(document_text, score)`` sorted
        by descending relevance.
        """
        if not query:
            return []
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        # Pair each document with its score
        scored_docs = list(zip(self.documents, scores))
        # Sort by score descending and take top k
        top = sorted(scored_docs, key=lambda x: x[1], reverse=True)[:k]
        # Filter out zero‑score entries (no relevance)
        return [(doc, float(score)) for doc, score in top if score > 0]

# ---------------------------------------------------------------------------
# Pydantic models for request/response validation
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str = Field(..., description="User question about fund data")
    top_k: int = Field(5, ge=1, le=20, description="Number of retrieved rows to consider")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="Generated answer or fallback message")
    sources: List[str] = Field(
        default_factory=list,
        description="Plain‑text rows that were used as context for the answer",
    )

# ---------------------------------------------------------------------------
# LLM interaction (OpenRouter) – optional
# ---------------------------------------------------------------------------
def call_openrouter_llm(context: str, question: str) -> str:
    """Send *context* and *question* to OpenRouter and return the model's answer.
    If the request fails or the model returns an empty answer, the fallback
    message is used.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OpenRouter API key is not configured")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        # OpenRouter follows the OpenAI schema
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return content.strip() or FALLBACK
    except Exception as exc:
        # Log the exception in a real‑world scenario; here we just fallback
        return FALLBACK

# ---------------------------------------------------------------------------
# Application lifecycle events
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup_event():
    global llm_enabled, trades_df, holdings_df, retriever, all_documents

    # Load CSV files – raise a clear error if they cannot be read
    try:
        trades_df = pd.read_csv(TRADES_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load trades CSV from '{TRADES_PATH}': {e}")

    try:
        holdings_df = pd.read_csv(HOLDINGS_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load holdings CSV from '{HOLDINGS_PATH}': {e}")

    # Convert each row of both dataframes into a plain‑text document
    all_documents = []
    for _, row in trades_df.iterrows():
        all_documents.append(row_to_text(row))
    for _, row in holdings_df.iterrows():
        all_documents.append(row_to_text(row))

    if not all_documents:
        raise RuntimeError("No documents were generated from the CSV files.")

    # Initialise the BM25 retriever
    retriever = Retriever(all_documents)

    # Determine if LLM integration should be active
    llm_enabled = bool(OPENROUTER_API_KEY)

# ---------------------------------------------------------------------------
# Core endpoint – chat
# ---------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if retriever is None:
        raise HTTPException(status_code=500, detail="Retriever not initialized.")

    # Retrieve the most relevant documents
    top_docs = retriever.get_top_k(request.query, k=request.top_k)
    sources = [doc for doc, _ in top_docs]

    # If no relevant documents were found, immediately return fallback
    if not sources:
        return ChatResponse(answer=FALLBACK, sources=[])

    # Concatenate sources to form the context for the LLM (or for deterministic fallback)
    context = "\n\n".join(sources)

    if llm_enabled:
        answer = call_openrouter_llm(context, request.query)
    else:
        # Deterministic mode – simply echo the retrieved rows as the answer.
        # In a real implementation you might implement rule‑based extraction.
        answer = context if context else FALLBACK

    # Ensure the answer respects the fallback rule
    if not answer or answer.strip() == "":
        answer = FALLBACK

    return ChatResponse(answer=answer, sources=sources)
