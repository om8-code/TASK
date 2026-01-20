import os
import re
import json
import difflib
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# CONFIG
# -----------------------------
TRADES_PATH = os.getenv("TRADES_PATH", "trades.csv")
HOLDINGS_PATH = os.getenv("HOLDINGS_PATH", "holdings.csv")

FALLBACK = "Sorry can not find the answer"

# -----------------------------
# LLM (optional - OpenRouter)
# -----------------------------
# If OPENROUTER_API_KEY is NOT set, the app runs in deterministic mode (no-LLM).
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

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

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Fund Chatbot (CSV-only, optional OpenRouter)")

# Initialized at startup
llm_enabled: bool = False
trades: Optional[pd.DataFrame] = None
holdings: Optional[pd.DataFrame] = None
retriever = None
resolve_fund = None
available_years: List[int] = []


# -----------------------------
# Schemas
# -----------------------------
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class AskResponse(BaseModel):
    answer: str
    model: str


# -----------------------------
# Retrieval (lightweight)
# -----------------------------
def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_\.\-]+", str(text).lower())


def row_to_text(row: pd.Series) -> str:
    parts = []
    for k, v in row.items():
        if pd.isna(v):
            continue
        parts.append(f"{k}: {v}")
    return " | ".join(parts)


class Retriever:
    def __init__(self, trades_df: pd.DataFrame, holdings_df: pd.DataFrame):
        self.trade_docs = [row_to_text(trades_df.iloc[i]) for i in range(len(trades_df))]
        self.hold_docs = [row_to_text(holdings_df.iloc[i]) for i in range(len(holdings_df))]
        self.trade_tok = [set(tokenize(d)) for d in self.trade_docs]
        self.hold_tok = [set(tokenize(d)) for d in self.hold_docs]

    def retrieve(self, query: str, k: int = 8, min_score: float = 2.0) -> List[Tuple[str, int, float, str]]:
        qtok = set(tokenize(query))
        if not qtok:
            return []

        results: List[Tuple[str, int, float, str]] = []

        def score(doc_tok: set) -> float:
            return float(len(qtok.intersection(doc_tok)))

        # trades
        scores = [score(t) for t in self.trade_tok]
        idxs = np.argsort(scores)[::-1][:k]
        for i in idxs:
            s = float(scores[int(i)])
            if s >= min_score:
                results.append(("trades", int(i), s, self.trade_docs[int(i)]))

        # holdings
        scores = [score(t) for t in self.hold_tok]
        idxs = np.argsort(scores)[::-1][:k]
        for i in idxs:
            s = float(scores[int(i)])
            if s >= min_score:
                results.append(("holdings", int(i), s, self.hold_docs[int(i)]))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:k]


# -----------------------------
# Fund resolver (robust)
# -----------------------------
def normalize_fund_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip()).lower()


def build_fund_resolver(trades_df: pd.DataFrame, holdings_df: pd.DataFrame):
    trades_names = sorted({str(x).strip() for x in trades_df.get("PortfolioName", pd.Series([])).dropna().unique()})
    holdings_names = sorted({str(x).strip() for x in holdings_df.get("PortfolioName", pd.Series([])).dropna().unique()})

    trades_norm = {normalize_fund_name(x): x for x in trades_names}
    holdings_norm = {normalize_fund_name(x): x for x in holdings_names}
    all_norm = sorted(set(trades_norm) | set(holdings_norm))

    def _best_match(ut_norm: str, preferred: Optional[str]) -> Optional[str]:
        # 1) substring match
        subs = [n for n in all_norm if n and n in ut_norm]
        if subs:
            best = sorted(subs, key=len, reverse=True)[0]
            if preferred == "holdings" and best in holdings_norm:
                return holdings_norm[best]
            if preferred == "trades" and best in trades_norm:
                return trades_norm[best]
            return holdings_norm.get(best) or trades_norm.get(best)

        # 2) fuzzy match
        close = difflib.get_close_matches(ut_norm, all_norm, n=1, cutoff=0.65)
        if close:
            best = close[0]
            if preferred == "holdings" and best in holdings_norm:
                return holdings_norm[best]
            if preferred == "trades" and best in trades_norm:
                return trades_norm[best]
            return holdings_norm.get(best) or trades_norm.get(best)

        return None

    def _resolve(user_text: str, preferred: Optional[str] = None) -> Optional[str]:
        ut_norm = normalize_fund_name(user_text)
        ut_norm2 = re.sub(r"\b(fund|portfolio)\b", "", ut_norm).strip()
        return _best_match(ut_norm, preferred) or (_best_match(ut_norm2, preferred) if ut_norm2 else None)

    return _resolve


# -----------------------------
# Analytics
# -----------------------------
def extract_year(text: str) -> Optional[int]:
    m = re.search(r"(19\d{2}|20\d{2})", text)
    return int(m.group(1)) if m else None


def count_trades_for_fund(trades_df: pd.DataFrame, fund: str) -> int:
    if "PortfolioName" not in trades_df.columns:
        return 0
    return int((trades_df["PortfolioName"] == fund).sum())


def count_holdings_for_fund(holdings_df: pd.DataFrame, fund: str, asof=None) -> int:
    if "PortfolioName" not in holdings_df.columns:
        return 0
    df = holdings_df[holdings_df["PortfolioName"] == fund].copy()
    if df.empty:
        return 0
    if "AsOfDate" in df.columns:
        if asof is None:
            asof = df["AsOfDate"].max()
        df = df[df["AsOfDate"] == asof]
    return int(len(df))


def year_end_pl_for_fund(holdings_df: pd.DataFrame, fund: str, year: int) -> Optional[float]:
    required = {"PortfolioName", "AsOfDate", "PL_YTD"}
    if not required.issubset(set(holdings_df.columns)):
        return None

    df = holdings_df[holdings_df["PortfolioName"] == fund].copy()
    df = df[df["AsOfDate"].dt.year == year]
    if df.empty:
        return None

    last_date = df["AsOfDate"].max()
    df_last = df[df["AsOfDate"] == last_date]
    return float(df_last["PL_YTD"].sum())


def rank_funds_by_yearly_pl(holdings_df: pd.DataFrame, year: int, top_n: int = 10):
    required = {"PortfolioName", "AsOfDate", "PL_YTD"}
    if not required.issubset(set(holdings_df.columns)):
        return []

    out = []
    for fund in sorted(holdings_df["PortfolioName"].dropna().unique()):
        val = year_end_pl_for_fund(holdings_df, fund, year)
        if val is not None:
            out.append((fund, val))

    out.sort(key=lambda x: x[1], reverse=True)
    return out[:top_n]


# -----------------------------
# Optional OpenRouter call
# -----------------------------
def openrouter_chat(user_message: str, context: str) -> str:
    if not llm_enabled:
        raise RuntimeError("LLM disabled")

    url = f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    # Optional OpenRouter headers
    if os.getenv("OPENROUTER_HTTP_REFERER"):
        headers["HTTP-Referer"] = os.getenv("OPENROUTER_HTTP_REFERER")
    if os.getenv("OPENROUTER_APP_NAME"):
        headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME")

    prompt = (
        "Answer the QUESTION using ONLY the CONTEXT provided. "
        "If the answer is not in the context and cannot be derived from it, respond with found=false. "
        "Return STRICT JSON with keys: found (boolean), answer (string).\n\n"
        f"QUESTION:\n{user_message}\n\nCONTEXT:\n{context}"
    )

    payload = {
        "model": OPENROUTER_MODEL,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


# -----------------------------
# Main QA
# -----------------------------
def answer_question(question: str) -> str:
    assert trades is not None and holdings is not None and retriever is not None and resolve_fund is not None

    q = question.strip()

    # Intent: total number of trades
    if re.search(r"\b(trade|trades)\b", q, flags=re.I) and re.search(r"\b(count|number|total)\b", q, flags=re.I):
        fund = resolve_fund(q, preferred="trades")
        if fund:
            n = count_trades_for_fund(trades, fund)
            return f"Total trades for {fund}: {n}"

    # Intent: total number of holdings
    if re.search(r"\b(holding|holdings)\b", q, flags=re.I) and re.search(r"\b(count|number|total)\b", q, flags=re.I):
        fund = resolve_fund(q, preferred="holdings")
        if fund:
            latest = None
            if "AsOfDate" in holdings.columns:
                latest = holdings.loc[holdings["PortfolioName"] == fund, "AsOfDate"].max()
            n = count_holdings_for_fund(holdings, fund, asof=latest)
            if pd.notna(latest):
                return f"Total holdings for {fund} as of {latest.date()}: {n}"
            return f"Total holdings for {fund}: {n}"

    # Intent: rank funds by yearly P&L
    if re.search(r"\b(performed|performance|better|best|top)\b", q, flags=re.I) and re.search(
        r"\b(pl|p\&l|profit|loss|ytd)\b", q, flags=re.I
    ):
        year = extract_year(q)
        if year is None and available_years:
            year = max(available_years)

        if year in available_years:
            top = rank_funds_by_yearly_pl(holdings, year, top_n=10)
            if top:
                lines = [f"Top funds by year-end PL_YTD in {year} (sum at last AsOfDate in {year}):"]
                for f, v in top:
                    lines.append(f"- {f}: {v:,.2f}")
                return "\n".join(lines)

    # Intent: specific fund year-end P&L
    if re.search(r"\b(pl|p\&l|profit|loss|ytd)\b", q, flags=re.I):
        year = extract_year(q)
        fund = resolve_fund(q, preferred="holdings")
        if fund and year and year in available_years:
            val = year_end_pl_for_fund(holdings, fund, year)
            if val is not None:
                return f"Year-end PL_YTD for {fund} in {year}: {val:,.2f}"

    # Retrieval fallback (extractive or LLM)
    hits = retriever.retrieve(q, k=8, min_score=1.5)
    if not hits:
        return FALLBACK

    ctx_lines = []
    for src, idx, score, doc in hits[:6]:
        ctx_lines.append(f"[{src} row {idx} | score={score:.2f}] {doc}")

    # If no LLM key: return top relevant rows (still CSV-only)
    if not llm_enabled:
        return "\n".join(ctx_lines[:3])

    raw = openrouter_chat(q, "\n".join(ctx_lines))
    try:
        obj = json.loads(raw)
        if not obj.get("found"):
            return FALLBACK
        ans = str(obj.get("answer", "")).strip()
    except Exception:
        return FALLBACK

    # Hard enforcement: must reference at least one provided row tag
    if not re.search(r"\[(trades|holdings) row\s+\d+", ans, flags=re.I):
        return FALLBACK
    if ans.strip() == FALLBACK:
        return FALLBACK
    return ans


# -----------------------------
# Startup: load CSVs locally
# -----------------------------
@app.on_event("startup")
def startup():
    global trades, holdings, retriever, resolve_fund, available_years, llm_enabled

    llm_enabled = bool(OPENROUTER_API_KEY)

    if not os.path.exists(TRADES_PATH):
        raise RuntimeError(f"Missing {TRADES_PATH}. Set TRADES_PATH or place trades.csv next to main.py")
    if not os.path.exists(HOLDINGS_PATH):
        raise RuntimeError(f"Missing {HOLDINGS_PATH}. Set HOLDINGS_PATH or place holdings.csv next to main.py")

    trades = pd.read_csv(TRADES_PATH)
    holdings = pd.read_csv(HOLDINGS_PATH)

    # trades dates (these appear time-like in your file; keep coercion)
    for col in ["TradeDate", "SettleDate"]:
        if col in trades.columns:
            trades[col] = pd.to_datetime(trades[col], errors="coerce")

    # holdings dates are day-first in your file (e.g., 01/08/23 = 1 Aug 2023)
    for col in ["AsOfDate", "OpenDate", "CloseDate"]:
        if col in holdings.columns:
            holdings[col] = pd.to_datetime(holdings[col], dayfirst=True, errors="coerce")

    retriever = Retriever(trades, holdings)
    resolve_fund = build_fund_resolver(trades, holdings)

    if "AsOfDate" in holdings.columns:
        available_years = sorted([int(y) for y in holdings["AsOfDate"].dropna().dt.year.unique()])
    else:
        available_years = []


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": OPENROUTER_MODEL if llm_enabled else "(no-llm)",
        "trades_loaded": trades is not None,
        "holdings_loaded": holdings is not None,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question is empty.")
    try:
        ans = answer_question(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return AskResponse(answer=ans, model=(OPENROUTER_MODEL if llm_enabled else "(no-llm)"))
