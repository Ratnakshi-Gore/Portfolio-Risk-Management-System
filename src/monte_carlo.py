import logging
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TRADING_DAYS = 252


def run_monte_carlo(
    weights: pd.Series,
    price_df: pd.DataFrame,
    n_simulations: int = 10_000,
    horizon_days: int = 252,
    initial_value: float = 100_000.0,
    seed: int = 42,
) -> np.ndarray:
   
    # Align tickers between weights and prices
    active = weights[weights > 1e-6]
    tickers = active.index.tolist()
    w = active.values

    if len(tickers) < 1:
        raise ValueError("No active positions (all weights are zero)")

    # Filter prices to active tickers and drop NaN columns
    prices = price_df[tickers].dropna(axis=1, how="any")
    tickers = prices.columns.tolist()
    w = weights.reindex(tickers).fillna(0).values
    w = w / w.sum()  # re-normalise after potential ticker drops

    daily_returns = prices.pct_change().dropna()

    if len(daily_returns) < 30:
        raise ValueError(f"Need at least 30 days of returns, got {len(daily_returns)}")

    mu_daily = daily_returns.mean().values                  # (n_assets,)
    cov_daily = daily_returns.cov().values                  # (n_assets, n_assets)

    # Portfolio-level daily parameters (weighted)
    port_mu = float(w @ mu_daily)
    port_var = float(w @ cov_daily @ w)
    port_sigma = np.sqrt(port_var)

    dt = 1.0  # 1 trading day

    log.info(
        f"Monte Carlo: {n_simulations} sims × {horizon_days} days | "
        f"port_mu={port_mu*252:.3f} ann | port_sigma={port_sigma*np.sqrt(252):.3f} ann"
    )

    # Generate paths using GBM 
    rng = np.random.default_rng(seed)

    # Drift term: (mu - 0.5 * sigma^2) * dt
    drift = (port_mu - 0.5 * port_sigma**2) * dt
    # Diffusion term: sigma * sqrt(dt) * Z
    diffusion = port_sigma * np.sqrt(dt)

    # Random shocks: shape (n_simulations, horizon_days)
    Z = rng.standard_normal((n_simulations, horizon_days))

    # Log returns for each day
    log_returns = drift + diffusion * Z  # (n_sims, horizon_days)

    # Cumulative sum of log returns → cumulative product of gross returns
    cum_log_returns = np.cumsum(log_returns, axis=1)

    # Portfolio value paths (prepend initial value at t=0)
    paths = np.zeros((n_simulations, horizon_days + 1))
    paths[:, 0] = initial_value
    paths[:, 1:] = initial_value * np.exp(cum_log_returns)

    log.info(
        f"Simulation complete | "
        f"median final: ${np.median(paths[:, -1]):,.0f} | "
        f"5th pctl: ${np.percentile(paths[:, -1], 5):,.0f} | "
        f"95th pctl: ${np.percentile(paths[:, -1], 95):,.0f}"
    )

    return paths


# VaR and CVaR

def compute_var_cvar(
    simulated_paths: np.ndarray,
    confidence: float = 0.95,
    initial_value: Optional[float] = None,
) -> Dict[str, float]:
    if initial_value is None:
        initial_value = simulated_paths[0, 0]

    final_values = simulated_paths[:, -1]
    total_returns = (final_values - initial_value) / initial_value

    # VaR: the (1-confidence) quantile of returns (a negative number = loss)
    var_pct = float(np.percentile(total_returns, (1 - confidence) * 100))
    var_dollar = var_pct * initial_value

    # CVaR: mean of all returns below the VaR threshold
    tail_returns = total_returns[total_returns <= var_pct]
    if len(tail_returns) > 0:
        cvar_pct = float(np.mean(tail_returns))
    else:
        cvar_pct = var_pct
    cvar_dollar = cvar_pct * initial_value

    result = {
        "var_pct":       round(var_pct, 4),
        "var_dollar":    round(var_dollar, 2),
        "cvar_pct":      round(cvar_pct, 4),
        "cvar_dollar":   round(cvar_dollar, 2),
        "confidence":    confidence,
        "median_return": round(float(np.median(total_returns)), 4),
        "mean_return":   round(float(np.mean(total_returns)), 4),
    }

    log.info(
        f"VaR({confidence:.0%}): {var_pct:.2%} (${abs(var_dollar):,.0f}) | "
        f"CVaR: {cvar_pct:.2%} (${abs(cvar_dollar):,.0f}) | "
        f"Median return: {result['median_return']:.2%}"
    )
    return result


# Summary statistics

def get_simulation_summary(
    simulated_paths: np.ndarray,
    initial_value: Optional[float] = None,
) -> Dict:
    if initial_value is None:
        initial_value = simulated_paths[0, 0]

    final_values = simulated_paths[:, -1]
    total_returns = (final_values - initial_value) / initial_value

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pctl_values = {
        f"p{p}": round(float(np.percentile(final_values, p)), 2)
        for p in percentiles
    }
    pctl_returns = {
        f"p{p}_return": round(float(np.percentile(total_returns, p)), 4)
        for p in percentiles
    }

    prob_profit = float(np.mean(total_returns > 0))
    prob_loss_10 = float(np.mean(total_returns < -0.10))
    prob_gain_20 = float(np.mean(total_returns > 0.20))

    summary = {
        "n_simulations":  simulated_paths.shape[0],
        "horizon_days":   simulated_paths.shape[1] - 1,
        "initial_value":  initial_value,
        "final_mean":     round(float(np.mean(final_values)), 2),
        "final_median":   round(float(np.median(final_values)), 2),
        "final_std":      round(float(np.std(final_values)), 2),
        "prob_profit":    round(prob_profit, 4),
        "prob_loss_10pct": round(prob_loss_10, 4),
        "prob_gain_20pct": round(prob_gain_20, 4),
        **pctl_values,
        **pctl_returns,
    }

    log.info(
        f"Summary: P(profit)={prob_profit:.1%} | "
        f"P(loss>10%)={prob_loss_10:.1%} | P(gain>20%)={prob_gain_20:.1%}"
    )
    return summary



if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    from src.data_loader import get_price_data, get_fundamentals, get_fred_data
    from src.factor_model import (
        compute_value_score, compute_momentum_score,
        compute_quality_score, compute_composite_score,
    )
    from src.macro_regime import run_macro_overlay
    from src.portfolio_optimizer import (
        compute_expected_returns, compute_covariance_matrix, optimize_portfolio,
    )

    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    TICKERS = ["AAPL", "MSFT", "JPM", "JNJ", "XOM"]
    START   = "2023-01-01"
    END     = "2024-12-31"

    print("Loading data & optimising portfolio...")
    prices       = get_price_data(TICKERS, START, END)
    fundamentals = get_fundamentals(TICKERS)
    fred         = get_fred_data()

    value    = compute_value_score(fundamentals)
    momentum = compute_momentum_score(prices)
    quality  = compute_quality_score(fundamentals)

    macro     = run_macro_overlay(fred)
    composite = compute_composite_score(value, momentum, quality, macro["factor_weights"])

    exp_ret = compute_expected_returns(prices, composite)
    cov     = compute_covariance_matrix(prices)
    opt     = optimize_portfolio(exp_ret, cov, yield_multiplier=macro["yield_multiplier"])

    print(f"\nPortfolio weights:\n{opt['weights'].round(4)}")
    print(f"Expected return: {opt['expected_return']:.2%} | Vol: {opt['volatility']:.2%}")

    print("\n=== Running Monte Carlo (10,000 simulations × 252 days) ===")
    paths = run_monte_carlo(
        weights=opt["weights"],
        price_df=prices,
        n_simulations=10_000,
        horizon_days=252,
        initial_value=100_000,
    )

    print(f"\nPaths shape: {paths.shape}")
    print(f"Starting value: ${paths[0, 0]:,.0f}")
    print(f"Final median:   ${np.median(paths[:, -1]):,.0f}")

    print("\n=== VaR / CVaR ===")
    var_95 = compute_var_cvar(paths, confidence=0.95)
    var_99 = compute_var_cvar(paths, confidence=0.99)
    print(f"95% VaR: {var_95['var_pct']:.2%} (${abs(var_95['var_dollar']):,.0f})")
    print(f"95% CVaR: {var_95['cvar_pct']:.2%} (${abs(var_95['cvar_dollar']):,.0f})")
    print(f"99% VaR: {var_99['var_pct']:.2%} (${abs(var_99['var_dollar']):,.0f})")
    print(f"99% CVaR: {var_99['cvar_pct']:.2%} (${abs(var_99['cvar_dollar']):,.0f})")

    print("\n=== Simulation Summary ===")
    summary = get_simulation_summary(paths)
    print(f"P(profit):    {summary['prob_profit']:.1%}")
    print(f"P(loss>10%):  {summary['prob_loss_10pct']:.1%}")
    print(f"P(gain>20%):  {summary['prob_gain_20pct']:.1%}")
    print(f"5th pctl:     ${summary['p5']:,.0f} ({summary['p5_return']:.2%})")
    print(f"50th pctl:    ${summary['p50']:,.0f} ({summary['p50_return']:.2%})")
    print(f"95th pctl:    ${summary['p95']:,.0f} ({summary['p95_return']:.2%})")
