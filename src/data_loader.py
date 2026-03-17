import os
import json
import time
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Paths
ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "processed"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cache TTLs (in days) 
TTL = {
    "prices":       1,
    "fundamentals": 7,
    "fred":         1,
    "fama_french":  30,
    "sp500":        7,
}

FRED_API_KEY = os.getenv("FRED_API_KEY", "")


# Internal cache helpers

def _cache_key(name: str, params: dict) -> str:
    raw = json.dumps(params, sort_keys=True, default=str)
    digest = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"{name}_{digest}"


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.parquet"


def _meta_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.meta.json"


def _is_cache_valid(key: str, ttl_days: int) -> bool:
    meta = _meta_path(key)
    if not meta.exists() or not _cache_path(key).exists():
        return False
    try:
        with open(meta) as f:
            saved_at = datetime.fromisoformat(json.load(f)["saved_at"])
        return datetime.utcnow() - saved_at < timedelta(days=ttl_days)
    except Exception:
        return False


def _load_cache(key: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_parquet(_cache_path(key))
    except Exception as e:
        log.warning(f"Cache read failed for {key}: {e}")
        return None


def _save_cache(key: str, df: pd.DataFrame) -> None:
    try:
        df.to_parquet(_cache_path(key))
        with open(_meta_path(key), "w") as f:
            json.dump({"saved_at": datetime.utcnow().isoformat()}, f)
        log.info(f"Cached {key} ({len(df)} rows)")
    except Exception as e:
        log.warning(f"Cache write failed for {key}: {e}")



def get_price_data(
    tickers: List[str],
    start: str,
    end: str,
    as_of_date: Optional[str] = None,
) -> pd.DataFrame:

    tickers = sorted(set(tickers))
    key = _cache_key("prices", {"tickers": tickers, "start": start, "end": end})

    if _is_cache_valid(key, TTL["prices"]):
        log.info(f"Loading prices from cache: {key}")
        df = _load_cache(key)
    else:
        log.info(f"Downloading prices for {len(tickers)} tickers from yfinance...")
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            # yfinance returns MultiIndex columns when >1 ticker
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw["Close"]
            else:
                df = raw[["Close"]].rename(columns={"Close": tickers[0]})

            df.dropna(how="all", inplace=True)
            _save_cache(key, df)
        except Exception as e:
            log.error(f"yfinance download failed: {e}")
            df = _load_cache(key)
            if df is None:
                raise RuntimeError(f"Cannot fetch price data and no cache exists: {e}")

    assert df is not None
    if as_of_date:
        df = df[df.index <= pd.Timestamp(as_of_date)]

    return df


def get_fundamentals(
    tickers: List[str],
    as_of_date: Optional[str] = None,
) -> pd.DataFrame:
    tickers = sorted(set(tickers))
    key = _cache_key("fundamentals", {"tickers": tickers})

    if _is_cache_valid(key, TTL["fundamentals"]):
        log.info(f"Loading fundamentals from cache: {key}")
        cached = _load_cache(key)
        assert cached is not None
        return cached

    log.info(f"Fetching fundamentals for {len(tickers)} tickers...")
    records = []

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            time.sleep(0.1)  # gentle rate limiting

            # Valuation 
            pb  = info.get("priceToBook", np.nan)
            pe  = info.get("trailingPE", np.nan)

            # FCF yield = freeCashflow / marketCap
            fcf  = info.get("freeCashflow", np.nan)
            mcap = info.get("marketCap", np.nan)
            fcf_yield = (fcf / mcap) if (fcf is not None and mcap is not None
                                         and not np.isnan(fcf) and not np.isnan(mcap)
                                         and mcap != 0) else np.nan

            # Quality 
            roe            = info.get("returnOnEquity", np.nan)
            profit_margin  = info.get("profitMargins", np.nan)
            current_ratio  = info.get("currentRatio", np.nan)
            debt_to_equity = info.get("debtToEquity", np.nan)

            # Earnings quality: operating CF / net income (>1 = low accruals = high quality)
            op_cf      = info.get("operatingCashflow", np.nan)
            net_income = info.get("netIncomeToCommon", np.nan)
            earnings_quality = (
                op_cf / abs(net_income)
                if (op_cf is not None and net_income is not None
                    and not np.isnan(op_cf) and not np.isnan(net_income)
                    and net_income != 0)
                else np.nan
            )

            # FCF quality: FCF / net income
            fcf_quality = (
                fcf / abs(net_income)
                if (fcf is not None and net_income is not None
                    and not np.isnan(fcf) and not np.isnan(net_income)
                    and net_income != 0)
                else np.nan
            )

            records.append({
                "ticker":           ticker,
                "pb_ratio":         pb,
                "pe_ratio":         pe,
                "fcf_yield":        fcf_yield,
                "roe":              roe,
                "profit_margin":    profit_margin,
                "current_ratio":    current_ratio,
                "debt_to_equity":   debt_to_equity,
                "earnings_quality": earnings_quality,
                "fcf_quality":      fcf_quality,
                "market_cap":       mcap,
                "sector":           info.get("sector", "Unknown"),
                "fetched_at":       datetime.utcnow().isoformat(),
            })
            log.info(f"  {ticker}: ok")

        except Exception as e:
            log.warning(f"  {ticker}: failed ({e})")
            records.append({"ticker": ticker})

    df = pd.DataFrame(records).set_index("ticker")
    _save_cache(key, df)
    return df


def get_sector_map(tickers: List[str]) -> dict:
    """Return {ticker: sector} dict using cached fundamentals data."""
    fundamentals = get_fundamentals(tickers)
    sector_map = {}
    for t in tickers:
        if t in fundamentals.index and "sector" in fundamentals.columns:
            sector_map[t] = fundamentals.loc[t, "sector"] if pd.notna(fundamentals.loc[t, "sector"]) else "Unknown"
        else:
            sector_map[t] = "Unknown"
    return sector_map


def get_fred_data(
    series_ids: Optional[List[str]] = None,
    start: str = "2000-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    if series_ids is None:
        series_ids = ["FEDFUNDS", "DGS10", "DGS2", "T10Y2Y"]

    end = end or datetime.utcnow().strftime("%Y-%m-%d")
    key = _cache_key("fred", {"series": sorted(series_ids), "start": start, "end": end})

    if _is_cache_valid(key, TTL["fred"]):
        log.info(f"Loading FRED data from cache: {key}")
        cached = _load_cache(key)
        assert cached is not None
        return cached

    log.info(f"Fetching FRED series: {series_ids}")

    try:
        import pandas_datareader.data as web

        frames = {}
        for sid in series_ids:
            try:
                s = web.DataReader(sid, "fred", start, end)
                frames[sid] = s[sid]
                log.info(f"  {sid}: {len(s)} rows")
            except Exception as e:
                log.warning(f"  {sid}: failed ({e})")

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        _save_cache(key, df)
        return df

    except Exception as e:
        log.error(f"FRED fetch failed: {e}")
        cached = _load_cache(key)
        if cached is not None:
            log.info("Using stale FRED cache as fallback")
            return cached
        raise RuntimeError(f"Cannot fetch FRED data and no cache: {e}")


def get_fama_french_factors(
    start: str = "2000-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    end = end or datetime.utcnow().strftime("%Y-%m-%d")
    key = _cache_key("fama_french", {"start": start, "end": end})

    if _is_cache_valid(key, TTL["fama_french"]):
        log.info("Loading Fama-French from cache")
        cached = _load_cache(key)
        assert cached is not None
        return cached

    log.info("Fetching Fama-French 3-factor data...")
    try:
        import pandas_datareader.data as web

        df = web.DataReader("F-F_Research_Data_Factors", "famafrench", start, end)[0]
        df = df / 100.0  # convert from percent to decimal
        df.index = pd.to_datetime(df.index.to_timestamp())
        df = df[df.index >= start]
        if end:
            df = df[df.index <= end]

        _save_cache(key, df)
        log.info(f"Fama-French: {len(df)} monthly rows")
        return df

    except Exception as e:
        log.error(f"Fama-French fetch failed: {e}")
        cached = _load_cache(key)
        if cached is not None:
            return cached
        raise RuntimeError(f"Cannot fetch Fama-French data and no cache: {e}")


def get_sp500_tickers() -> List[str]:
    key = _cache_key("sp500", {})

    if _is_cache_valid(key, TTL["sp500"]):
        log.info("Loading S&P 500 tickers from cache")
        df = _load_cache(key)
        assert df is not None
        return df["ticker"].tolist()

    log.info("Scraping S&P 500 tickers from Wikipedia...")
    try:
        import requests
        from io import StringIO

        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0 (portfolio-risk-tool/1.0)"},
            timeout=15,
        )
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        df = tables[0][["Symbol", "GICS Sector"]].copy()
        df.columns = ["ticker", "sector"]
        # Wikipedia uses periods; yfinance uses hyphens (e.g. BRK.B to BRK-B)
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        df.sort_values("ticker", inplace=True)
        df.reset_index(drop=True, inplace=True)

        _save_cache(key, df)
        log.info(f"S&P 500: {len(df)} tickers")
        return df["ticker"].tolist()

    except Exception as e:
        log.error(f"Wikipedia scrape failed: {e}")
        cached = _load_cache(key)
        if cached is not None:
            return cached["ticker"].tolist()
        raise RuntimeError(f"Cannot fetch S&P 500 tickers and no cache: {e}")



if __name__ == "__main__":
    TEST_TICKERS = ["AAPL", "MSFT", "JPM", "JNJ", "XOM"]

    print("\n=== Price Data ===")
    prices = get_price_data(TEST_TICKERS, start="2023-01-01", end="2024-01-01")
    print(prices.tail())

    print("\n=== Fundamentals ===")
    fundamentals = get_fundamentals(TEST_TICKERS)
    print(fundamentals[["pb_ratio", "pe_ratio", "roe", "profit_margin"]])

    print("\n=== FRED Data ===")
    fred = get_fred_data()
    print(fred.tail())

    print("\n=== Fama-French ===")
    ff = get_fama_french_factors(start="2020-01-01")
    print(ff.tail())

    print("\n=== S&P 500 Tickers (first 10) ===")
    sp500 = get_sp500_tickers()
    print(sp500[:10])
