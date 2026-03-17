import logging
from typing import Dict, Optional
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

log = logging.getLogger(__name__)

TRADING_DAYS = 252


def compute_drawdown_series(portfolio_returns: pd.Series) -> pd.Series:

    cum = (1 + portfolio_returns).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1.0
    drawdown.name = "drawdown"
    return drawdown


def _max_drawdown_info(portfolio_returns: pd.Series) -> Dict:
    # Compute max drawdown percentage and duration (in trading days).
    dd = compute_drawdown_series(portfolio_returns)
    max_dd = float(dd.min())

    # Duration: longest stretch from peak to recovery
    cum = (1 + portfolio_returns).cumprod()
    running_max = cum.cummax()
    is_in_drawdown = cum < running_max

    if not is_in_drawdown.any():
        return {"max_drawdown": 0.0, "max_dd_duration_days": 0}

    # Find stretches of consecutive drawdown days
    groups = (~is_in_drawdown).cumsum()
    dd_groups = groups[is_in_drawdown]
    if dd_groups.empty:
        return {"max_drawdown": max_dd, "max_dd_duration_days": 0}

    longest = dd_groups.value_counts().max()
    return {
        "max_drawdown": round(max_dd, 6),
        "max_dd_duration_days": int(longest),
    }


def compute_all_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.05,
) -> Dict:

    n_days = len(portfolio_returns)
    if n_days < 2:
        raise ValueError(f"Need at least 2 return observations, got {n_days}")

    rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    excess = portfolio_returns - rf_daily

    # Return metrics 
    total_return = float((1 + portfolio_returns).prod() - 1)
    ann_return = float((1 + total_return) ** (TRADING_DAYS / n_days) - 1)
    ann_vol = float(portfolio_returns.std() * np.sqrt(TRADING_DAYS))

    # Sharpe
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

    # Sortino, uses only downside deviation
    negative_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = float(negative_returns.std() * np.sqrt(TRADING_DAYS)) if len(negative_returns) > 1 else 0.0
    sortino = (ann_return - risk_free_rate) / downside_std if downside_std > 0 else 0.0

    # Drawdown
    dd_info = _max_drawdown_info(portfolio_returns)
    max_dd = dd_info["max_drawdown"]

    # Calmar
    calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-9 else 0.0

    # Tail risk
    var_95 = float(np.percentile(portfolio_returns, 5))
    var_99 = float(np.percentile(portfolio_returns, 1))
    cvar_95 = float(portfolio_returns[portfolio_returns <= var_95].mean()) if (portfolio_returns <= var_95).any() else var_95
    cvar_99 = float(portfolio_returns[portfolio_returns <= var_99].mean()) if (portfolio_returns <= var_99).any() else var_99

    # Win rate / profit factor
    win_rate = float((portfolio_returns > 0).mean())
    pos_sum = float(portfolio_returns[portfolio_returns > 0].sum())
    neg_sum = float(portfolio_returns[portfolio_returns < 0].sum())
    profit_factor = pos_sum / abs(neg_sum) if abs(neg_sum) > 1e-12 else float("inf")

    # Skew / kurtosis
    skewness = float(portfolio_returns.skew())
    kurtosis = float(portfolio_returns.kurtosis())  # excess kurtosis

    metrics = {
        "total_return":       round(total_return, 6),
        "ann_return":         round(ann_return, 6),
        "ann_volatility":     round(ann_vol, 6),
        "sharpe_ratio":       round(sharpe, 4),
        "sortino_ratio":      round(sortino, 4),
        "calmar_ratio":       round(calmar, 4),
        "max_drawdown":       round(max_dd, 6),
        "max_dd_duration":    dd_info["max_dd_duration_days"],
        "var_95_daily":       round(var_95, 6),
        "var_99_daily":       round(var_99, 6),
        "cvar_95_daily":      round(cvar_95, 6),
        "cvar_99_daily":      round(cvar_99, 6),
        "win_rate":           round(win_rate, 4),
        "profit_factor":      round(profit_factor, 4),
        "skewness":           round(skewness, 4),
        "excess_kurtosis":    round(kurtosis, 4),
        "n_trading_days":     n_days,
    }

    # Benchmark-relative metrics 
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        # Align dates
        common = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common) > 10:
            p = portfolio_returns.loc[common]
            b = benchmark_returns.loc[common]

            # Benchmark annualised return
            bench_total = float((1 + b).prod() - 1)
            bench_ann = float((1 + bench_total) ** (TRADING_DAYS / len(b)) - 1)

            # Beta = cov(p, b) / var(b)
            cov_pb = float(np.cov(p.values, b.values)[0, 1])
            var_b = float(b.var())
            beta = cov_pb / var_b if var_b > 1e-12 else 1.0

            # CAPM Alpha
            alpha = ann_return - risk_free_rate - beta * (bench_ann - risk_free_rate)

            # R-squared
            corr = float(np.corrcoef(p.values, b.values)[0, 1])
            r_squared = corr ** 2

            # Information ratio = excess return / tracking error
            tracking_diff = p - b
            tracking_error = float(tracking_diff.std() * np.sqrt(TRADING_DAYS))
            info_ratio = (ann_return - bench_ann) / tracking_error if tracking_error > 1e-9 else 0.0

            metrics.update({
                "benchmark_total_return": round(bench_total, 6),
                "benchmark_ann_return":   round(bench_ann, 6),
                "beta":                   round(beta, 4),
                "alpha_capm":             round(alpha, 6),
                "r_squared":              round(r_squared, 4),
                "information_ratio":      round(info_ratio, 4),
                "tracking_error":         round(tracking_error, 6),
                "correlation":            round(corr, 4),
            })
        else:
            log.warning(f"Only {len(common)} common dates with benchmark — skipping relative metrics")

    log.info(
        f"Metrics: Sharpe={metrics['sharpe_ratio']:.3f} | "
        f"Sortino={metrics['sortino_ratio']:.3f} | "
        f"MaxDD={metrics['max_drawdown']:.2%} | "
        f"Beta={metrics.get('beta', 'N/A')}"
    )
    return metrics



def compute_rolling_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    window: int = 252,
) -> pd.DataFrame:
    rf_daily = (1 + 0.05) ** (1 / TRADING_DAYS) - 1
    result = pd.DataFrame(index=portfolio_returns.index)

    # Rolling volatility (annualised)
    result["rolling_volatility"] = portfolio_returns.rolling(window).std() * np.sqrt(TRADING_DAYS)

    # Rolling Sharpe
    rolling_mean = portfolio_returns.rolling(window).mean() * TRADING_DAYS
    result["rolling_sharpe"] = (rolling_mean - 0.05) / result["rolling_volatility"]

    # Rolling beta and alpha 
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        common = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common) > window:
            p = portfolio_returns.loc[common]
            b = benchmark_returns.loc[common]

            # Rolling covariance / variance
            rolling_cov = p.rolling(window).cov(b)
            rolling_var = b.rolling(window).var()
            result.loc[common, "rolling_beta"] = rolling_cov / rolling_var

            # Rolling alpha (annualised)
            p_ann = p.rolling(window).mean() * TRADING_DAYS
            b_ann = b.rolling(window).mean() * TRADING_DAYS
            beta_col = result.loc[common, "rolling_beta"]
            result.loc[common, "rolling_alpha"] = p_ann - 0.05 - beta_col * (b_ann - 0.05)

    log.info(f"Rolling metrics computed (window={window} days, {len(result)} rows)")
    return result


# Fama-French 3-Factor alpha

def compute_fama_french_alpha(
    portfolio_returns: pd.Series,
    ff_factors: pd.DataFrame,
) -> Dict:

    # FF factors from Ken French are in percentage points, converting to decimal
    ff = ff_factors.copy()
    for col in ["Mkt-RF", "SMB", "HML", "RF"]:
        if col in ff.columns and ff[col].abs().max() > 1.0:
            ff[col] = ff[col] / 100.0

    # FF data is monthly, resampling daily portfolio returns to monthly
    # Compound daily returns within each month: (1+r1)*(1+r2)*...*(1+rn) - 1
    monthly_port = portfolio_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    # Convert FF PeriodIndex to month-end DatetimeIndex for alignment
    if hasattr(ff.index, 'to_timestamp'):
        ff.index = ff.index.to_timestamp(how='end')
    # Normalise both to month-end for matching
    monthly_port.index = monthly_port.index.to_period("M").to_timestamp(how="end")
    ff.index = ff.index.to_period("M").to_timestamp(how="end")

    # Align dates
    common = monthly_port.index.intersection(ff.index)
    if len(common) < 12:
        log.warning(f"Only {len(common)} common months for FF regression — need at least 12")
        return {"error": "insufficient data", "n_obs": len(common)}

    p = monthly_port.loc[common].values
    rf = ff.loc[common, "RF"].values if "RF" in ff.columns else np.zeros(len(common))
    excess_p = p - rf

    # Build X matrix: [Mkt-RF, SMB, HML, intercept(alpha)]
    X = np.column_stack([
        ff.loc[common, "Mkt-RF"].values,
        ff.loc[common, "SMB"].values,
        ff.loc[common, "HML"].values,
        np.ones(len(common)),  # intercept = alpha
    ])
    y = excess_p

    # OLS via least squares
    coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)

    beta_mkt = float(coeffs[0])
    beta_smb = float(coeffs[1])
    beta_hml = float(coeffs[2])
    alpha_monthly = float(coeffs[3])
    alpha_annual = alpha_monthly * 12  # monthly to annual

    # Residuals and R-squared
    y_hat = X @ coeffs
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Standard error of alpha for t-test
    n_obs = len(y)
    n_params = X.shape[1]
    dof = n_obs - n_params
    mse = ss_res / dof if dof > 0 else 0.0

    # (X'X)^{-1} diagonal gives variance of each coefficient
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
        se_alpha = float(np.sqrt(mse * xtx_inv[-1, -1]))
    except np.linalg.LinAlgError:
        se_alpha = 0.0

    t_stat = alpha_monthly / se_alpha if se_alpha > 1e-12 else 0.0
    p_value = float(2 * sp_stats.t.sf(abs(t_stat), dof)) if dof > 0 else 1.0

    result = {
        "alpha_monthly":  round(alpha_monthly, 8),
        "alpha_annual":   round(alpha_annual, 6),
        "beta_mkt":       round(beta_mkt, 4),
        "beta_smb":       round(beta_smb, 4),
        "beta_hml":       round(beta_hml, 4),
        "r_squared":      round(r_squared, 4),
        "t_stat_alpha":   round(t_stat, 4),
        "p_value_alpha":  round(p_value, 6),
        "n_observations": n_obs,
        "significant_5pct": p_value < 0.05,
    }

    log.info(
        f"FF3 Alpha: {alpha_annual:.4f} ann ({alpha_monthly:.6f} monthly) | "
        f"t={t_stat:.3f} p={p_value:.4f} | "
        f"Betas: Mkt={beta_mkt:.3f} SMB={beta_smb:.3f} HML={beta_hml:.3f} | "
        f"R²={r_squared:.3f}"
    )
    return result



def get_metrics_summary(metrics: Dict) -> pd.DataFrame:
    """
    Format the metrics dict into a clean 2-column DataFrame for display.
    Groups metrics by category for readability.
    """
    display_order = [
        ("Return", "total_return", "Total Return", "{:.2%}"),
        ("Return", "ann_return", "Annualised Return", "{:.2%}"),
        ("Return", "ann_volatility", "Annualised Volatility", "{:.2%}"),
        ("Ratios", "sharpe_ratio", "Sharpe Ratio", "{:.3f}"),
        ("Ratios", "sortino_ratio", "Sortino Ratio", "{:.3f}"),
        ("Ratios", "calmar_ratio", "Calmar Ratio", "{:.3f}"),
        ("Drawdown", "max_drawdown", "Max Drawdown", "{:.2%}"),
        ("Drawdown", "max_dd_duration", "Max DD Duration (days)", "{}"),
        ("Tail Risk", "var_95_daily", "Daily VaR (95%)", "{:.4%}"),
        ("Tail Risk", "var_99_daily", "Daily VaR (99%)", "{:.4%}"),
        ("Tail Risk", "cvar_95_daily", "Daily CVaR (95%)", "{:.4%}"),
        ("Tail Risk", "cvar_99_daily", "Daily CVaR (99%)", "{:.4%}"),
        ("Distribution", "win_rate", "Win Rate", "{:.1%}"),
        ("Distribution", "profit_factor", "Profit Factor", "{:.2f}"),
        ("Distribution", "skewness", "Skewness", "{:.3f}"),
        ("Distribution", "excess_kurtosis", "Excess Kurtosis", "{:.3f}"),
        ("Benchmark", "benchmark_ann_return", "Benchmark Ann. Return", "{:.2%}"),
        ("Benchmark", "beta", "Beta", "{:.3f}"),
        ("Benchmark", "alpha_capm", "CAPM Alpha", "{:.2%}"),
        ("Benchmark", "r_squared", "R-Squared", "{:.3f}"),
        ("Benchmark", "information_ratio", "Information Ratio", "{:.3f}"),
        ("Benchmark", "tracking_error", "Tracking Error", "{:.2%}"),
        ("Benchmark", "correlation", "Correlation", "{:.3f}"),
    ]

    rows = []
    for category, key, label, fmt in display_order:
        if key in metrics:
            val = metrics[key]
            rows.append({
                "Category": category,
                "Metric": label,
                "Value": fmt.format(val),
            })

    return pd.DataFrame(rows)



if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from src.backtester import run_backtest
    from src.data_loader import get_fama_french_factors

    TICKERS = ["AAPL", "MSFT", "JPM", "JNJ", "XOM"]

    print("=" * 60)
    print("RUNNING BACKTEST (to get return series)")
    print("=" * 60)

    bt = run_backtest(
        tickers=TICKERS,
        start_date="2021-01-01",
        end_date="2024-12-31",
        rebalance_freq="M",
        initial_value=100_000,
    )

    port_ret = bt["daily_returns"]
    bench_ret = bt["benchmark_returns"]

    print("\n" + "=" * 60)
    print("RISK METRICS")
    print("=" * 60)

    metrics = compute_all_metrics(port_ret, bench_ret, risk_free_rate=0.05)

    summary = get_metrics_summary(metrics)
    print(summary.to_string(index=False))

    print("\n=== Drawdown Series (first 5) ===")
    dd = compute_drawdown_series(port_ret)
    print(dd.head())
    print(f"Max drawdown: {dd.min():.2%}")

    print("\n=== Rolling Metrics (last 5 rows) ===")
    rolling = compute_rolling_metrics(port_ret, bench_ret, window=252)
    print(rolling.dropna().tail().round(4))

    print("\n=== Fama-French 3-Factor Alpha ===")
    try:
        ff = get_fama_french_factors()
        if ff is not None and not ff.empty:
            ff3 = compute_fama_french_alpha(port_ret, ff)
            print(f"Alpha (annual): {ff3['alpha_annual']:.4f}")
            print(f"t-stat:         {ff3['t_stat_alpha']:.3f}")
            print(f"p-value:        {ff3['p_value_alpha']:.4f}")
            print(f"Significant:    {ff3['significant_5pct']}")
            print(f"Beta (Mkt):     {ff3['beta_mkt']:.3f}")
            print(f"Beta (SMB):     {ff3['beta_smb']:.3f}")
            print(f"Beta (HML):     {ff3['beta_hml']:.3f}")
            print(f"R-squared:      {ff3['r_squared']:.3f}")
            print(f"Observations:   {ff3['n_observations']}")
        else:
            print("Fama-French data not available — skipping")
    except Exception as e:
        print(f"FF3 regression failed: {e}")
