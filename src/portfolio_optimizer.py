import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

log = logging.getLogger(__name__)

# Constants
TRADING_DAYS = 252          # annualisation factor
MAX_WEIGHT   = 0.15         # max 15% in any single stock
MIN_WEIGHT   = 0.0          # long-only
RISK_FREE    = 0.05         # 5% risk-free rate for Sharpe calculation
FACTOR_BLEND = 0.3          # 30% factor tilt in expected returns, 70% historical

# Expected returns

def compute_expected_returns(
    price_df: pd.DataFrame,
    factor_scores: Optional[pd.Series] = None,
    lookback_days: int = 252,
) -> pd.Series:
    if price_df.empty:
        raise ValueError("price_df is empty")

    # Using only recent lookback window
    recent = price_df.tail(lookback_days)
    daily_returns = recent.pct_change().dropna()

    # Annualise
    hist_returns = daily_returns.mean() * TRADING_DAYS
    hist_returns.name = "expected_return"

    if factor_scores is None or factor_scores.empty:
        log.info("No factor scores provided — using historical returns only")
        return hist_returns

    common = hist_returns.index.intersection(factor_scores.index)
    if len(common) == 0:
        log.warning("No common tickers between price data and factor scores")
        return hist_returns

    # Cross-sectional z-score of factor scores convert to return adjustment
    fs = factor_scores.loc[common]
    fs_std = fs.std()
    if fs_std == 0:
        log.warning("Factor scores have zero std — using historical returns only")
        return hist_returns

    fs_zscore = (fs - fs.mean()) / fs_std

    # Scale: 1 std in factor score → 2% additional annual return (empirically calibrated)
    factor_adjustment = fs_zscore * 0.02

    # Blend: 70% historical + 30% factor-tilted
    blended = hist_returns.copy()
    blended.loc[common] = (
        (1 - FACTOR_BLEND) * hist_returns.loc[common]
        + FACTOR_BLEND * (hist_returns.loc[common] + factor_adjustment)
    )

    log.info(
        f"Expected returns: min={blended.min():.3f} "
        f"max={blended.max():.3f} mean={blended.mean():.3f}"
    )
    return blended


# Covariance matrix

def compute_covariance_matrix(
    price_df: pd.DataFrame,
    lookback_days: int = 252,
) -> pd.DataFrame:
    recent = price_df.tail(lookback_days)
    daily_returns = recent.pct_change().dropna()

    # Drop tickers with any remaining NaN (e.g. newly listed stocks with short history)
    daily_returns = daily_returns.dropna(axis=1, how="any")

    tickers = daily_returns.columns.tolist()
    n = len(tickers)

    if n < 2:
        raise ValueError(f"Need at least 2 tickers, got {n}")

    lw = LedoitWolf()
    lw.fit(daily_returns.values)
    cov_daily = lw.covariance_

    # Annualise
    cov_annual = cov_daily * TRADING_DAYS

    cov_df = pd.DataFrame(cov_annual, index=tickers, columns=tickers)

    log.info(
        f"Covariance matrix: shape={cov_df.shape} | "
        f"shrinkage={lw.shrinkage_:.3f} | "
        f"condition_number={np.linalg.cond(cov_annual):.1f}"
    )
    return cov_df


# Core optimizer

def _effective_max_weight(n: int, budget: float, requested_max: float) -> float:
    generous = budget / n + 0.10
    return min(budget, max(requested_max, generous))


def _neg_sharpe(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free: float = RISK_FREE,
) -> float:
    port_return   = weights @ expected_returns
    port_variance = weights @ cov_matrix @ weights + 1e-8   # ridge for smoothness
    port_vol      = np.sqrt(port_variance)
    return -(port_return - risk_free) / port_vol


def _portfolio_variance(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> float:
    return float(weights @ cov_matrix @ weights)


def optimize_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    yield_multiplier: float = 1.0,
    objective: str = "sharpe",
    max_weight: float = MAX_WEIGHT,
) -> Dict:
    tickers = expected_returns.index.tolist()
    n = len(tickers)

    if n < 2:
        raise ValueError(f"Need at least 2 tickers, got {n}")

    # Align cov_matrix with expected_returns
    cov_aligned = cov_matrix.loc[tickers, tickers].values
    mu = expected_returns.values

    # Effective budget after yield curve multiplier
    budget = float(np.clip(yield_multiplier, 0.5, 1.0))

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - budget},  # weights sum to budget
    ]

    # Relax cap for small universes so constraints stay feasible
    eff_max = _effective_max_weight(n, budget, max_weight)
    bounds = [(MIN_WEIGHT, eff_max) for _ in range(n)]
    if eff_max > max_weight:
        log.info(f"Relaxed max_weight {max_weight:.2f}→{eff_max:.2f} (n={n} tickers)")

    # Objective
    if objective == "sharpe":
        obj_fn = lambda w: _neg_sharpe(w, mu, cov_aligned)
    else:
        obj_fn = lambda w: _portfolio_variance(w, cov_aligned)

    # Multiple random starting points — keep the best converged result
    N_STARTS  = 8
    best_result = None
    best_val    = np.inf
    rng = np.random.default_rng(42)

    start_points = [np.full(n, budget / n)]                        # 1: equal weight
    for _ in range(N_STARTS - 1):                                  # rest: random
        r = rng.dirichlet(np.ones(n)) * budget
        r = np.clip(r, MIN_WEIGHT, eff_max)
        r = r / r.sum() * budget
        start_points.append(r)

    for w0 in start_points:
        res = minimize(
            obj_fn,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-8},
        )
        if res.success and res.fun < best_val:
            best_val    = res.fun
            best_result = res

    if best_result is None:
        log.warning("No starting point converged — using equal weights.")
        w_opt = np.full(n, budget / n)
    else:
        w_opt = best_result.x
        log.info(f"Optimizer converged (best of {N_STARTS} starts, fun={best_val:.6f})")

    # Clean up tiny weights (below 0.1% set to 0)
    w_opt[w_opt < 0.001] = 0.0
    if w_opt.sum() > 0:
        w_opt = w_opt / w_opt.sum() * budget  # re-normalise to budget
    else:
        w_opt = np.full(n, budget / n)  # fallback to equal weights

    weights_series = pd.Series(w_opt, index=tickers).round(6)

    port_return   = float(weights_series.values @ mu)
    port_variance = float(weights_series.values @ cov_aligned @ weights_series.values)
    port_vol      = float(np.sqrt(port_variance))
    sharpe        = (port_return - RISK_FREE) / port_vol if port_vol > 0 else 0.0
    cash_weight   = round(1.0 - float(weights_series.sum()), 6)

    log.info(
        f"Optimised: return={port_return:.3f} vol={port_vol:.3f} "
        f"Sharpe={sharpe:.3f} cash={cash_weight:.3f} "
        f"(yield_mult={yield_multiplier:.2f})"
    )

    return {
        "weights":         weights_series,
        "expected_return": port_return,
        "volatility":      port_vol,
        "sharpe_ratio":    sharpe,
        "cash_weight":     cash_weight,
        "converged":       best_result is not None,
    }


# Efficient frontier

def get_efficient_frontier(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_points: int = 50,
) -> pd.DataFrame:
    tickers = expected_returns.index.tolist()
    n = len(tickers)
    cov = cov_matrix.loc[tickers, tickers].values
    mu  = expected_returns.values

    min_ret = float(mu.min())
    max_ret = float(mu.max())
    target_returns = np.linspace(min_ret, max_ret, n_points)

    # Relax cap so frontier constraints are feasible for small universes
    eff_max = _effective_max_weight(n, 1.0, MAX_WEIGHT)
    bounds  = [(0.0, eff_max)] * n
    rng     = np.random.default_rng(0)

    results = []
    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
        ]

        # Try equal-weight start then one random start
        best_res = None
        for w0 in [np.full(n, 1.0 / n),
                   rng.dirichlet(np.ones(n))]:
            w0 = np.clip(w0, 0.0, eff_max)
            w0 = w0 / w0.sum()
            res = minimize(
                _portfolio_variance,
                w0,
                args=(cov,),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-8},
            )
            if res.success:
                best_res = res
                break

        if best_res is not None:
            vol = float(np.sqrt(max(best_res.fun, 0.0)))
            sharpe = (target - RISK_FREE) / vol if vol > 1e-9 else 0.0
            results.append({
                "target_return": round(target, 5),
                "volatility":    round(vol, 5),
                "sharpe_ratio":  round(sharpe, 4),
            })

    frontier_df = pd.DataFrame(results)
    log.info(f"Efficient frontier: {len(frontier_df)} valid points computed")
    return frontier_df



if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    from src.data_loader import get_price_data, get_fundamentals, get_fred_data
    from src.factor_model import (
        compute_value_score, compute_momentum_score,
        compute_quality_score, compute_composite_score,
    )
    from src.macro_regime import run_macro_overlay

    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    TICKERS = ["AAPL", "MSFT", "JPM", "JNJ", "XOM"]
    START   = "2023-01-01"
    END     = "2024-12-31"

    print("Loading data...")
    prices = get_price_data(TICKERS, START, END)
    fundamentals = get_fundamentals(TICKERS)
    fred = get_fred_data()

    print("\nComputing factor scores...")
    value    = compute_value_score(fundamentals)
    momentum = compute_momentum_score(prices)
    quality  = compute_quality_score(fundamentals)

    macro = run_macro_overlay(fred)
    weights_map = macro["factor_weights"]
    composite = compute_composite_score(value, momentum, quality, weights_map)

    print(f"\nMacro regime : {macro['regime']}")
    print(f"Factor weights: {weights_map}")
    print(f"Yield multiplier: {macro['yield_multiplier']}")

    print("\nComputing expected returns & covariance...")
    exp_ret = compute_expected_returns(prices, composite)
    cov     = compute_covariance_matrix(prices)

    print("\n=== Expected Returns ===")
    print(exp_ret.round(4))

    print("\n=== Covariance Matrix (annualised) ===")
    print(cov.round(4))

    print("\n=== Optimised Portfolio (Max Sharpe) ===")
    result = optimize_portfolio(
        exp_ret, cov,
        yield_multiplier=macro["yield_multiplier"],
        objective="sharpe",
    )
    print(f"Weights:\n{result['weights'].round(4)}")
    print(f"Expected Return : {result['expected_return']:.2%}")
    print(f"Volatility      : {result['volatility']:.2%}")
    print(f"Sharpe Ratio    : {result['sharpe_ratio']:.3f}")
    print(f"Cash Weight     : {result['cash_weight']:.2%}")
    print(f"Converged       : {result['converged']}")

    print("\n=== Efficient Frontier (first 5 points) ===")
    frontier = get_efficient_frontier(exp_ret, cov, n_points=20)
    print(frontier.head())
