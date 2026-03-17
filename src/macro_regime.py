import logging
from enum import Enum
from typing import Dict, Optional
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# Regime definitions
class Regime(str, Enum):
    HIGH_RISING = "HIGH_RISING"
    RISING      = "RISING"
    FALLING     = "FALLING"
    STABLE      = "STABLE"


# Factor weights per regime
# Keys: value, momentum, quality — must sum to 1.0
REGIME_WEIGHTS: Dict[str, Dict[str, float]] = {
    Regime.HIGH_RISING: {"value": 0.20, "momentum": 0.30, "quality": 0.50},
    Regime.RISING:      {"value": 0.25, "momentum": 0.35, "quality": 0.40},
    Regime.FALLING:     {"value": 0.25, "momentum": 0.45, "quality": 0.30},
    Regime.STABLE:      {"value": 0.33, "momentum": 0.34, "quality": 0.33},
}

REGIME_DESCRIPTIONS = {
    Regime.HIGH_RISING: (
        "HIGH & RISING rates — Fed is tightening aggressively. "
        "Quality stocks (strong balance sheets, low debt) tend to outperform. "
        "Factor weights tilted heavily toward Quality."
    ),
    Regime.RISING: (
        "RISING rates — Fed is in a tightening cycle. "
        "Quality and Momentum both rewarded. "
        "Factor weights balanced with Quality tilt."
    ),
    Regime.FALLING: (
        "FALLING rates — Fed is cutting or easing. "
        "Momentum and growth stocks tend to outperform. "
        "Factor weights tilted toward Momentum."
    ),
    Regime.STABLE: (
        "STABLE rates — Fed is on hold. "
        "No strong regime signal. "
        "Factor weights equally distributed."
    ),
}

# Threshold: 'high' rates = Fed funds rate above this level
HIGH_RATE_THRESHOLD = 4.0  # percent

# Threshold for 'rising' vs 'falling': 3-month trend in basis points
TREND_THRESHOLD_BPS = 10.0  # 0.10 percentage points


# Core regime detection

def get_current_regime(
    fred_data: pd.DataFrame,
    as_of_date: Optional[str] = None,
) -> Regime:
    if "FEDFUNDS" not in fred_data.columns:
        log.warning("FEDFUNDS not in fred_data — defaulting to STABLE regime")
        return Regime.STABLE

    df = fred_data.copy()
    if as_of_date:
        df = df[df.index <= pd.Timestamp(as_of_date)]

    # Drop NaN Fed funds rows (weekends/holidays show NaN for monthly series)
    fed = df["FEDFUNDS"].dropna()

    if len(fed) < 3:
        log.warning("Insufficient FEDFUNDS history — defaulting to STABLE")
        return Regime.STABLE

    current_rate = fed.iloc[-1]

    # 12-month average (up to 12 months of monthly data)
    avg_12m = fed.tail(12).mean()

    # 3-month trend: difference between most recent and 3 months ago
    if len(fed) >= 3:
        rate_3m_ago = fed.iloc[-3]
        trend_3m = current_rate - rate_3m_ago  # positive = rising
    else:
        trend_3m = 0.0

    is_high   = current_rate > HIGH_RATE_THRESHOLD
    is_rising = trend_3m > (TREND_THRESHOLD_BPS / 100)
    is_falling = trend_3m < -(TREND_THRESHOLD_BPS / 100)

    if is_high and is_rising:
        regime = Regime.HIGH_RISING
    elif is_rising:
        regime = Regime.RISING
    elif is_falling:
        regime = Regime.FALLING
    else:
        regime = Regime.STABLE

    log.info(
        f"Regime: {regime} | Fed rate: {current_rate:.2f}% | "
        f"12m avg: {avg_12m:.2f}% | 3m trend: {trend_3m:+.2f}%"
    )
    return regime


# Factor weights

def get_factor_weights(regime: Regime) -> Dict[str, float]:

    weights = REGIME_WEIGHTS[regime]
    log.info(
        f"Factor weights for {regime}: "
        f"V={weights['value']:.2f} M={weights['momentum']:.2f} Q={weights['quality']:.2f}"
    )
    return weights.copy()


# Yield curve multiplier

def get_yield_curve_multiplier(
    fred_data: pd.DataFrame,
    as_of_date: Optional[str] = None,
) -> float:
    df = fred_data.copy()
    if as_of_date:
        df = df[df.index <= pd.Timestamp(as_of_date)]

    if "T10Y2Y" in df.columns:
        spread_series = df["T10Y2Y"].dropna()
    elif "DGS10" in df.columns and "DGS2" in df.columns:
        spread_series = (df["DGS10"] - df["DGS2"]).dropna()
    else:
        log.warning("No yield curve data available — using multiplier=1.0")
        return 1.0

    if len(spread_series) == 0:
        log.warning("Empty yield curve series — using multiplier=1.0")
        return 1.0

    spread = spread_series.iloc[-1]

    if spread >= 1.0:
        multiplier = 1.0
    elif spread >= 0.0:
        # Linear scale: 1.0% spread → 1.0 multiplier; 0% spread → 0.75
        multiplier = 0.75 + (spread / 1.0) * 0.25
    elif spread >= -1.0:
        # Linear scale: 0% spread → 0.75 multiplier; -1% spread → 0.5
        multiplier = 0.75 + (spread / 1.0) * 0.25
    else:
        multiplier = 0.5

    multiplier = round(float(np.clip(multiplier, 0.5, 1.0)), 3)
    log.info(f"Yield curve spread: {spread:.3f}% → position multiplier: {multiplier:.3f}")
    return multiplier


def describe_regime(regime: Regime) -> str:
    return REGIME_DESCRIPTIONS.get(regime, "Unknown regime")


def run_macro_overlay(
    fred_data: pd.DataFrame,
    as_of_date: Optional[str] = None,
) -> Dict:
    regime = get_current_regime(fred_data, as_of_date=as_of_date)
    weights = get_factor_weights(regime)
    multiplier = get_yield_curve_multiplier(fred_data, as_of_date=as_of_date)
    description = describe_regime(regime)

    df = fred_data.copy()
    if as_of_date:
        df = df[df.index <= pd.Timestamp(as_of_date)]

    # Safe extraction — guard against empty series after date filter / dropna
    fed_series = df["FEDFUNDS"].dropna() if "FEDFUNDS" in df.columns else pd.Series(dtype=float)
    fed_rate = float(fed_series.iloc[-1]) if len(fed_series) > 0 else None

    if "T10Y2Y" in df.columns and len(df["T10Y2Y"].dropna()) > 0:
        spread = float(df["T10Y2Y"].dropna().iloc[-1])
    elif "DGS10" in df.columns and "DGS2" in df.columns:
        spread_s = (df["DGS10"] - df["DGS2"]).dropna()
        spread = float(spread_s.iloc[-1]) if len(spread_s) > 0 else None
    else:
        spread = None

    return {
        "regime":           regime,
        "factor_weights":   weights,
        "yield_multiplier": multiplier,
        "description":      description,
        "fed_rate":         fed_rate,
        "spread_10y2y":     spread,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from src.data_loader import get_fred_data

    print("Fetching FRED data...")
    fred = get_fred_data()

    print("\n=== Macro Overlay ===")
    result = run_macro_overlay(fred)

    print(f"Regime          : {result['regime']}")
    print(f"Fed Funds Rate  : {result['fed_rate']:.2f}%")
    print(f"10Y-2Y Spread   : {result['spread_10y2y']:.3f}%")
    print(f"Factor Weights  : {result['factor_weights']}")
    print(f"Yield Multiplier: {result['yield_multiplier']}")
    print(f"\nDescription:\n{result['description']}")

    print("\n=== Historical regime at 2022-12-31 (rate hike cycle) ===")
    result_2022 = run_macro_overlay(fred, as_of_date="2022-12-31")
    print(f"Regime: {result_2022['regime']} | Fed rate: {result_2022['fed_rate']:.2f}%")

    print("\n=== Historical regime at 2020-06-30 (COVID cuts) ===")
    result_2020 = run_macro_overlay(fred, as_of_date="2020-06-30")
    print(f"Regime: {result_2020['regime']} | Fed rate: {result_2020['fed_rate']:.2f}%")
