'''\
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
'''\

import os
import re
import json
from typing import Optional, List, Tuple

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

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
# Fixed the incomplete environment variable line
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)

SYSTEM_PROMPT = (
    "You are a data assistant that must answer ONLY using the provided context from two CSV files: "
    "trades.csv and holdings.csv.\n"
    "Rules:\n"
    "- If the answer is not present or cannot be derived from the given context, respond with exactly:\n"
    "  Sorry can not find the answer\n"
    "- Do NOT use outside knowledge.\n"
    "- Do NOT guess.\n"
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

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question about the fund data")

class LLMAnswerResponse(BaseModel):
    answer: str
    model: str = Field(default=OPENROUTER_MODEL, description="Model used for generation")

class DocsResponse(BaseModel):
    documents: List[str]

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    """Very small tokenizer – extracts alphanumeric tokens, lower‑cased."""
    return re.findall(r"[A-Za-z0-9_\.-]+", str(text).lower())

def row_to_text(row: pd.Series) -> str:
    """Convert a pandas Series (a CSV row) into a deterministic plain‑text string."""
    parts = []
    for k, v in row.items():
        if pd.isna(v):
            continue
        parts.append(f"{k}: {v}")
    return " | ".join(parts)

# ---------------------------------------------------------------------------
# Retriever implementation (simple overlap scoring – lightweight BM25 analogue)
# ---------------------------------------------------------------------------
class Retriever:
    def __init__(self, trades: pd.DataFrame, holdings: pd.DataFrame):
        self.trade_docs = [row_to_text(trades.iloc[i]) for i in range(len(trades))]
        self.hold_docs = [row_to_text(holdings.iloc[i]) for i in range(len(holdings))]
        self.trade_tok = [set(tokenize(d)) for d in self.trade_docs]
        self.hold_tok = [set(tokenize(d)) for d in self.hold_docs]

    def _score(self, qtok: set, doc_tok: set) -> float:
        """Simple overlap score – number of shared tokens.
        In a real system you would use BM25; this keeps dependencies minimal.
        """
        return len(qtok.intersection(doc_tok))

    def retrieve(
        self, query: str, k: int = 8, min_score: float = 2.0
    ) -> List[Tuple[str, int, float, str]]:
        """Return a list of up to *k* most relevant documents.
        Each entry is a tuple: (document_text, index, score, source_type).
        """
        qtok = set(tokenize(query))
        if not qtok:
            return []

        results: List[Tuple[str, int, float, str]] = []
        # Score trade documents
        for idx, doc_tok in enumerate(self.trade_tok):
            sc = self._score(qtok, doc_tok)
            if sc >= min_score:
                results.append((self.trade_docs[idx], idx, sc, "trade"))
        # Score holdings documents
        for idx, doc_tok in enumerate(self.hold_tok):
            sc = self._score(qtok, doc_tok)
            if sc >= min_score:
                results.append((self.hold_docs[idx], idx, sc, "holding"))

        # Sort by score descending and keep top *k*
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:k]

# ---------------------------------------------------------------------------
# OpenRouter LLM call (optional)
# ---------------------------------------------------------------------------
def call_openrouter(messages: List[dict]) -> str:
    """Send a chat completion request to OpenRouter and return the generated text.
    Raises HTTPException on failure.
    """
    url = f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}")

# ---------------------------------------------------------------------------
# Application lifecycle events
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global llm_enabled, trades_df, holdings_df, retriever
    # Load CSVs – fail fast if files are missing or malformed
    try:
        trades_df = pd.read_csv(TRADES_PATH)
        holdings_df = pd.read_csv(HOLDINGS_PATH)
    except Exception as exc:
        raise RuntimeError(f"Failed to load CSV data: {exc}")

    retriever = Retriever(trades_df, holdings_df)
    llm_enabled = bool(OPENROUTER_API_KEY)

# ---------------------------------------------------------------------------
# API endpoint – /query
# ---------------------------------------------------------------------------
@app.post("/query", response_model=LLMAnswerResponse, responses={200: {"model": DocsResponse}})
async def query_endpoint(request: AskRequest):
    """Handle a user question.
    * Retrieve the most relevant rows.
    * If LLM integration is enabled, forward the context to the model and return the answer.
    * Otherwise, return the raw documents.
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    retrieved = retriever.retrieve(request.question, k=8)
    if not retrieved:
        # No relevant docs – either fallback answer or empty list
        if llm_enabled:
            return LLMAnswerResponse(answer=FALLBACK, model=OPENROUTER_MODEL)
        else:
            return DocsResponse(documents=[])

    # Build a concise context string (join top docs)
    context = "\n".join([doc for doc, _, _, _ in retrieved])

    if llm_enabled:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.question}"},
        ]
        answer = call_openrouter(messages)
        # Ensure we never return an empty answer – enforce fallback rule
        if not answer or answer.strip() == "":
            answer = FALLBACK
        return LLMAnswerResponse(answer=answer, model=OPENROUTER_MODEL)
    else:
        # Deterministic mode – just return the retrieved documents
        return DocsResponse(documents=[doc for doc, _, _, _ in retrieved])
