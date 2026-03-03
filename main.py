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

# ---------------------------------------------------------------------------
