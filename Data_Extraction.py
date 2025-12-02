import os
import time
import requests
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from docx import Document



FMP_API_KEY = ""
START_DATE = "2021-01-01"
EVENT_START = pd.Timestamp("2021-01-01")
EVENT_END   = pd.Timestamp("2025-12-31") 



SECTORS = {
    "Tech": ["AAPL", "MSFT", "GOOGL"],
    "Financials": ["JPM", "BAC", "UBS"],
    "HealthCare": ["JNJ", "PFE", "UNH"]
}
EARNINGS_TIMING = {
    "AAPL": "after",
    "MSFT": "after",
    "GOOGL": "after",
    "AMZN": "after",
    "TSLA": "after",
    "HD": "before",
    "JPM": "before",
    "BAC": "before",
    "GS": "before",
    "JNJ": "before",
    "PFE": "before",
    "UNH": "before",
    "XOM": "before",
    "CAT": "before",
    "HON": "before",
}
SYMBOL_TO_NAME = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google",          
    "JPM": "JPMorgan",
    "BAC": "Bank of America",
    "UBS": "UBS Bank",
    "JNJ": "Johnson & Johnson",
    "PFE": "Pfizer",
    "UNH": "UnitedHealth"
}




FINBERT_MODEL = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
finbert_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
finbert_model.eval() 

def finbert_sentiment_score(text: str, max_chars: int = 3000):
    if text.strip() == "":
        return None
    text = text[:max_chars] 

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = finbert_model(**inputs)
        logits = outputs.logits

    probs = torch.softmax(logits, dim=-1)[0].numpy()

    id2label = finbert_model.config.id2label
    label_probs = {}
    for key, label in id2label.items():
        idx = int(key)
        label_probs[label.lower()] = float(probs[idx])

    pos = label_probs.get("positive", 0.0)
    neg = label_probs.get("negative", 0.0)

    score_minus1_1 = pos - neg
    score_0_10 = (score_minus1_1 + 1.0) * 5.0
    return score_0_10






BASE_URL = "https://financialmodelingprep.com/api/v3"

def fmp_get(path: str, params: dict):

    url = BASE_URL + path
    params = dict(params) if params else {}
    params["apikey"] = FMP_API_KEY

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


def get_prices(symbol: str, start_date: str = START_DATE):
    data = fmp_get(f"/historical-price-full/{symbol}", {
        "from": start_date,
        "serietype": "line",
    })
    hist = data.get("historical", [])
    if not hist:
        return pd.DataFrame()

    df = pd.DataFrame(hist)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df[["close"]]


def get_earnings(symbol: str, limit: int = 40):

    data = fmp_get(f"/historical/earning_calendar/{symbol}", {"limit": limit})

    if not isinstance(data, list) or len(data) == 0:
        print(str(data)[:300])
        return pd.DataFrame()

    df = pd.DataFrame(data)

    eps_candidates = ["eps", "reportedEPS", "epsActual"]
    eps_col = next((c for c in eps_candidates if c in df.columns), None)

    est_candidates = ["estimatedEPS", "epsEstimated", "epsEstimate"]
    est_col = next((c for c in est_candidates if c in df.columns), None)

    if eps_col is not None and eps_col != "eps":
        df.rename(columns={eps_col: "eps"}, inplace=True)
    if est_col is not None and est_col != "estimatedEPS":
        df.rename(columns={est_col: "estimatedEPS"}, inplace=True)

    keep_cols = [
        c for c in
        ["date", "symbol", "eps", "estimatedEPS", "revenue", "estimatedRevenue", "revenueEstimated"]
        if c in df.columns
    ]
    df = df[keep_cols]

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    if "eps" in df.columns and "estimatedEPS" in df.columns:
        df["eps_surprise"] = df["eps"] - df["estimatedEPS"]
    else:
        df["eps_surprise"] = np.nan

    return df



def get_quarter_from_date(dt):
    dt = pd.to_datetime(dt)
    return (dt.month - 1) // 3 + 1


def get_event_trading_day(trading_index: pd.DatetimeIndex,
                          calendar_date: pd.Timestamp,
                          timing: str):
    
    trading_days = trading_index.sort_values()
    calendar_date = pd.to_datetime(calendar_date)

    if timing == "before":
        if calendar_date in trading_days:
            return calendar_date
        future = trading_days[trading_days > calendar_date]
        return future[0] if len(future) > 0 else None

    elif timing == "after":
        future = trading_days[trading_days > calendar_date]
        return future[0] if len(future) > 0 else None

    else:
        raise ValueError(f"Unknown timing: {timing}")


def get_transcript_local(symbol: str, year: int, quarter: int):

    symbol_up = symbol.upper()
    company_name = SYMBOL_TO_NAME.get(symbol_up, symbol_up)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "transcripts")

    base_names = [
        f"{company_name} Q{quarter} {year} Transcript",
        f"{company_name} Q{quarter} {year} transcript",
        f"{symbol_up} Q{quarter} {year} Transcript",
        f"{symbol_up} Q{quarter} {year} transcript",
    ]

    for base in base_names:
        path_txt = os.path.join(base_dir, base + ".txt")
        if os.path.exists(path_txt):
            with open(path_txt, "r", encoding="utf-8") as f:
                return f.read()

    for base in base_names:
        path_docx = os.path.join(base_dir, base + ".docx")
        if os.path.exists(path_docx):
            try:
                doc = Document(path_docx)
                paragraphs = [p.text for p in doc.paragraphs]
                text = "\n".join(paragraphs)
                return text
            except Exception as e:
                return None

    return None
