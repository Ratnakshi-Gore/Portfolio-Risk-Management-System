import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mstats

log = logging.getLogger(__name__)

# Earnings lag constant (days)
# Never use quarterly fundamental data until 45 days after quarter end.
# This prevents lookahead bias in backtesting.
EARNINGS_LAG_DAYS = 45

DEFAULT_WEIGHTS = {"value": 0.33, "momentum": 0.34, "quality": 0.33}


def _winsorize(series: pd.Series, limits: Tuple[float, float] = (0.01, 0.01)) -> pd.Series:
    """Clip outliers at 1st and 99th percentile."""
    clean = series.dropna()
    if len(clean) < 5:
        return series
    winsorized = mstats.winsorize(clean, limits=limits)
    result = series.copy()
    result[series.notna()] = winsorized
    return result


def _zscore(series: pd.Series) -> pd.Series:
    """Cross-sectional z-score: (x - mean) / std. Returns NaN for constant series."""
    std = series.std()
    if std == 0 or np.isnan(std):
        return pd.Series(np.nan, index=series.index)
    return (series - series.mean()) / std


def _safe_score(raw: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """
    Winsorize → z-score a raw metric series.
    If higher_is_better=False (e.g. P/E, P/B, debt), flip the sign so that
    a high score always means 'more attractive'.
    """
    processed = _zscore(_winsorize(raw))
    return processed if higher_is_better else -processed


def _check_earnings_lag(as_of_date: Optional[str]) -> Optional[datetime]:
    """
    Return the latest fundamental data date we're allowed to use.
    Fundamentals are considered available only 45 days after quarter end.
    """
    if as_of_date is None:
        return None
    cutoff = datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=EARNINGS_LAG_DAYS)
    return cutoff


# Factor 1: Value Score

def compute_value_score(
    fundamentals: pd.DataFrame,
    as_of_date: Optional[str] = None,
) -> pd.Series:
  
    if as_of_date:
        log.info(f"Value score as of {as_of_date} (earnings lag: {EARNINGS_LAG_DAYS}d)")

    required = ["pb_ratio", "pe_ratio", "fcf_yield"]
    missing = [c for c in required if c not in fundamentals.columns]
    if missing:
        raise ValueError(f"fundamentals missing columns: {missing}")

    # Lower P/B and P/E = cheaper = better → flip sign
    pb_score  = _safe_score(fundamentals["pb_ratio"],  higher_is_better=False)
    pe_score  = _safe_score(fundamentals["pe_ratio"],  higher_is_better=False)
    fcf_score = _safe_score(fundamentals["fcf_yield"], higher_is_better=True)

    value = (
        0.40 * pb_score.fillna(0)
        + 0.35 * pe_score.fillna(0)
        + 0.25 * fcf_score.fillna(0)
    )

    # Final cross-sectional z-score of the composite
    value = _zscore(value)
    value.name = "value_score"
    log.info(f"Value scores computed for {value.notna().sum()} tickers")
    return value


# Factor 2: Momentum Score

def compute_momentum_score(
    price_df: pd.DataFrame,
    as_of_date: Optional[str] = None,
) -> pd.Series:
  
    if as_of_date:
        price_df = price_df[price_df.index <= pd.Timestamp(as_of_date)]

    if price_df.empty:
        log.warning("Empty price_df — returning empty momentum scores")
        return pd.Series(dtype=float)

    if len(price_df) < 252:
        log.warning("Less than 252 trading days of price data — momentum may be unreliable")

    # Most recent available date
    end_date   = price_df.index[-1]
    # Use position-based indexing for trading days (not calendar days)
    # 1 month ≈ 21 trading days, 12 months ≈ 252 trading days
    one_month_ago  = price_df.index[price_df.index <= end_date - timedelta(days=30)]
    twelve_months_ago = price_df.index[price_df.index <= end_date - timedelta(days=365)]

    if len(one_month_ago) == 0 or len(twelve_months_ago) == 0:
        log.warning("Insufficient history for momentum calculation")
        return pd.Series(dtype=float)

    t_minus_1  = one_month_ago[-1]
    t_minus_12 = twelve_months_ago[-1]

    price_now   = price_df.loc[t_minus_1]   
    price_12m   = price_df.loc[t_minus_12]   
    price_end   = price_df.loc[end_date]     # latest price

    # 12-month return (from t-12 to t-1, skipping most recent month)
    ret_12m = (price_now - price_12m) / price_12m

    # 1-month return (what we're subtracting out)
    ret_1m  = (price_end - price_now) / price_now

    # Momentum = 12m return - 1m return
    momentum_raw = ret_12m - ret_1m

    momentum = _safe_score(momentum_raw, higher_is_better=True)
    momentum.name = "momentum_score"
    log.info(f"Momentum scores computed for {momentum.notna().sum()} tickers")
    return momentum


# Factor 3: Quality Score

def compute_quality_score(
    fundamentals: pd.DataFrame,
    as_of_date: Optional[str] = None,
) -> pd.Series:

    if as_of_date:
        log.info(f"Quality score as of {as_of_date}")

    metric_config = [
        ("roe",              0.20, True),
        ("profit_margin",    0.20, True),
        ("earnings_quality", 0.20, True),
        ("fcf_quality",      0.15, True),
        ("current_ratio",    0.15, True),
        ("debt_to_equity",   0.10, False),  # lower debt = better → flip
    ]

    composite = pd.Series(0.0, index=fundamentals.index)
    total_weight_used = 0.0

    for col, weight, higher_is_better in metric_config:
        if col not in fundamentals.columns:
            log.warning(f"Quality metric '{col}' not found, skipping")
            continue
        scored = _safe_score(fundamentals[col], higher_is_better=higher_is_better)
        composite += weight * scored.fillna(0)
        total_weight_used += weight

    if total_weight_used < 0.5:
        log.warning("Less than 50% of quality metrics available — scores may be unreliable")

    quality = _zscore(composite)
    quality.name = "quality_score"
    log.info(f"Quality scores computed for {quality.notna().sum()} tickers")
    return quality


# Composite Score

def compute_composite_score(
    value: pd.Series,
    momentum: pd.Series,
    quality: pd.Series,
    weights: Optional[Dict[str, float]] = None,
) -> pd.Series:

    if weights is None:
        weights = DEFAULT_WEIGHTS

    total = sum(weights.values())
    if abs(total - 1.0) > 0.01:
        log.warning(f"Factor weights sum to {total:.3f}, not 1.0 — normalizing")
        weights = {k: v / total for k, v in weights.items()}

    # Align all three on the same tickers
    tickers = value.index.union(momentum.index).union(quality.index)
    v = value.reindex(tickers).fillna(0)
    m = momentum.reindex(tickers).fillna(0)
    q = quality.reindex(tickers).fillna(0)

    composite = (
        weights["value"]    * v
        + weights["momentum"] * m
        + weights["quality"]  * q
    )

    # Final z-score pass
    composite = _zscore(composite)
    composite.name = "composite_score"
    composite.sort_values(ascending=False, inplace=True)

    log.info(
        f"Composite score | weights: V={weights['value']:.2f} "
        f"M={weights['momentum']:.2f} Q={weights['quality']:.2f}"
    )
    return composite


# Stock Ranking

def rank_stocks(
    composite_scores: pd.Series,
    top_n: int = 20,
) -> pd.DataFrame:

    scores = composite_scores.dropna().sort_values(ascending=False)

    n = len(scores)
    percentile = pd.Series(
        [(n - i) / n * 100 for i in range(n)],
        index=scores.index,
        name="percentile_rank",
    )

    result = pd.DataFrame({
        "composite_score": scores,
        "percentile_rank": percentile,
    })

    top = result.head(top_n).copy()
    log.info(f"Top {top_n} stocks selected from universe of {n}")
    return top



def run_factor_pipeline(
    price_df: pd.DataFrame,
    fundamentals: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    as_of_date: Optional[str] = None,
    top_n: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    tickers = list(fundamentals.index)
    price_subset = price_df[[t for t in tickers if t in price_df.columns]]

    value    = compute_value_score(fundamentals, as_of_date=as_of_date)
    momentum = compute_momentum_score(price_subset, as_of_date=as_of_date)
    quality  = compute_quality_score(fundamentals, as_of_date=as_of_date)
    composite = compute_composite_score(value, momentum, quality, weights=weights)

    all_tickers = composite.index
    scores_df = pd.DataFrame({
        "value_score":     value.reindex(all_tickers),
        "momentum_score":  momentum.reindex(all_tickers),
        "quality_score":   quality.reindex(all_tickers),
        "composite_score": composite,
    })

    n = len(scores_df.dropna(subset=["composite_score"]))
    scores_df["percentile_rank"] = scores_df["composite_score"].rank(pct=True) * 100
    scores_df.sort_values("composite_score", ascending=False, inplace=True)

    top_df = scores_df.head(top_n).copy()
    return scores_df, top_df



if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from src.data_loader import get_price_data, get_fundamentals

    TEST_TICKERS = ["AAPL", "MSFT", "JPM", "JNJ", "XOM", "GOOGL", "AMZN", "META", "BRK-B", "V"]

    print("Loading data...")
    prices = get_price_data(TEST_TICKERS, start="2022-01-01", end="2024-01-01")
    fundamentals = get_fundamentals(TEST_TICKERS)

    print("\n=== Running factor pipeline ===")
    scores_df, top_df = run_factor_pipeline(
        price_df=prices,
        fundamentals=fundamentals,
        weights={"value": 0.33, "momentum": 0.34, "quality": 0.33},
    )

    print("\nAll factor scores:")
    print(scores_df.round(3))

    print(f"\nTop 5 stocks:")
    print(top_df.head(5).round(3))
