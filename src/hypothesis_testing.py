import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

log = logging.getLogger(__name__)

TRADING_DAYS = 252


# 1. Information Coefficient (IC) Significance

def _compute_ic_series(
    factor_scores: pd.Series,
    price_df: pd.DataFrame,
    forward_months: int = 1,
) -> pd.Series:
    tickers = factor_scores.dropna().index.tolist()
    common = [t for t in tickers if t in price_df.columns]

    if len(common) < 3:
        return np.nan

    # Forward return: from the last available date in price_df
    last_date = price_df.index[-1]
    forward_days = forward_months * 21  # approximate trading days per month

    # We need enough history to compute forward return
    if len(price_df) < forward_days + 1:
        return np.nan

    # Use the price at end vs price at (end - forward_days)
    end_prices = price_df[common].iloc[-1]
    start_prices = price_df[common].iloc[-(forward_days + 1)]

    forward_returns = (end_prices / start_prices) - 1.0
    forward_returns = forward_returns.dropna()

    common_final = factor_scores.reindex(forward_returns.index).dropna().index
    common_final = common_final.intersection(forward_returns.dropna().index)

    if len(common_final) < 3:
        return np.nan

    # Spearman rank correlation
    ic, _ = sp_stats.spearmanr(
        factor_scores.loc[common_final].values,
        forward_returns.loc[common_final].values,
    )
    return float(ic)


def test_ic_significance(
    price_df: pd.DataFrame,
    fundamentals: pd.DataFrame,
    periods: List[int] = [1, 3, 6, 12],
) -> Dict:
    from src.factor_model import (
        compute_value_score, compute_momentum_score,
        compute_quality_score,
    )

    # Generate quarterly evaluation dates (need enough history before and after)
    all_dates = price_df.index
    if len(all_dates) < TRADING_DAYS:
        return {"error": "Need at least 1 year of price data"}

    # Quarterly snapshots starting 1 year in, ending 1 year before the end
    min_date = all_dates[TRADING_DAYS]
    max_date = all_dates[-TRADING_DAYS] if len(all_dates) > 2 * TRADING_DAYS else all_dates[-1]
    quarterly_dates = pd.date_range(min_date, max_date, freq="QE")
    quarterly_dates = [d for d in quarterly_dates if d in all_dates or
                       len(all_dates[all_dates <= d]) > 0]

    # Map to actual trading days
    eval_dates = []
    for d in quarterly_dates:
        mask = all_dates <= d
        if mask.any():
            eval_dates.append(all_dates[mask][-1])
    eval_dates = sorted(set(eval_dates))

    if len(eval_dates) < 4:
        return {"error": f"Only {len(eval_dates)} evaluation dates — need at least 4"}

    log.info(f"IC test: {len(eval_dates)} evaluation dates, periods={periods}")

    # Compute IC at each evaluation date for each factor
    factors = ["value", "momentum", "quality"]
    results = {f: {p: [] for p in periods} for f in factors}

    for eval_date in eval_dates:
        as_of = eval_date.strftime("%Y-%m-%d")
        prices_up_to = price_df[price_df.index <= eval_date]

        if len(prices_up_to) < 60:
            continue

        # Compute factor scores at this point in time
        try:
            value_scores = compute_value_score(fundamentals)
            momentum_scores = compute_momentum_score(prices_up_to, as_of_date=as_of)
            quality_scores = compute_quality_score(fundamentals)
        except Exception as e:
            log.debug(f"Factor computation failed at {as_of}: {e}")
            continue

        score_map = {
            "value": value_scores,
            "momentum": momentum_scores,
            "quality": quality_scores,
        }

        for period in periods:
            # Forward prices from eval_date
            forward_days = period * 21
            future_mask = price_df.index > eval_date
            future_prices = price_df[future_mask]

            if len(future_prices) < forward_days:
                continue

            # Forward return for each ticker
            start_p = prices_up_to.iloc[-1]
            end_p = future_prices.iloc[min(forward_days - 1, len(future_prices) - 1)]
            fwd_ret = (end_p / start_p) - 1.0
            fwd_ret = fwd_ret.dropna()

            for factor_name, scores in score_map.items():
                common = scores.dropna().index.intersection(fwd_ret.dropna().index)
                if len(common) < 3:
                    continue

                ic, _ = sp_stats.spearmanr(
                    scores.loc[common].values,
                    fwd_ret.loc[common].values,
                )
                if not np.isnan(ic):
                    results[factor_name][period].append(ic)

    # Compute summary statistics
    summary = {}
    for factor_name in factors:
        summary[factor_name] = {}
        for period in periods:
            ic_values = results[factor_name][period]
            n = len(ic_values)

            if n < 2:
                summary[factor_name][period] = {
                    "ic_mean": np.nan, "ic_std": np.nan, "icir": np.nan,
                    "t_stat": np.nan, "p_value": np.nan, "n_obs": n,
                    "significant": False,
                }
                continue

            ic_arr = np.array(ic_values)
            ic_mean = float(np.mean(ic_arr))
            ic_std = float(np.std(ic_arr, ddof=1))
            icir = ic_mean / ic_std if ic_std > 1e-9 else 0.0

            # t-test: H0: mean IC = 0
            t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 1e-9 else 0.0
            p_value = float(2 * sp_stats.t.sf(abs(t_stat), n - 1))

            summary[factor_name][period] = {
                "ic_mean":     round(ic_mean, 4),
                "ic_std":      round(ic_std, 4),
                "icir":        round(icir, 4),
                "t_stat":      round(t_stat, 4),
                "p_value":     round(p_value, 6),
                "n_obs":       n,
                "significant": p_value < 0.05,
            }

            log.info(
                f"IC [{factor_name}] {period}M: mean={ic_mean:.4f} std={ic_std:.4f} "
                f"ICIR={icir:.3f} t={t_stat:.3f} p={p_value:.4f} (n={n})"
            )

    return summary


# 2. Strategy Significance (Bootstrap)

def test_strategy_significance(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> Dict:
    # Convert to monthly returns for bootstrap (more independent samples)
    monthly_port = portfolio_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_port = monthly_port.dropna()

    n_months = len(monthly_port)
    if n_months < 12:
        return {"error": "Need at least 12 months for bootstrap test",
                "n_months": n_months}

    # Actual Sharpe (annualised from monthly)
    rf_monthly = (1 + 0.05) ** (1/12) - 1
    actual_mean = float(monthly_port.mean())
    actual_std = float(monthly_port.std())
    actual_sharpe = (actual_mean - rf_monthly) / actual_std * np.sqrt(12) if actual_std > 1e-9 else 0.0

    # Bootstrap
    rng = np.random.default_rng(seed)
    monthly_arr = monthly_port.values
    bootstrap_sharpes = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(monthly_arr, size=n_months, replace=True)
        s_mean = sample.mean()
        s_std = sample.std(ddof=1)
        if s_std > 1e-9:
            bootstrap_sharpes[i] = (s_mean - rf_monthly) / s_std * np.sqrt(12)
        else:
            bootstrap_sharpes[i] = 0.0

    # p-value: fraction of bootstrap Sharpes <= 0 (one-sided test)
    p_value_sharpe = float(np.mean(bootstrap_sharpes <= 0))

    result = {
        "actual_sharpe":          round(actual_sharpe, 4),
        "bootstrap_mean_sharpe":  round(float(np.mean(bootstrap_sharpes)), 4),
        "bootstrap_std_sharpe":   round(float(np.std(bootstrap_sharpes)), 4),
        "bootstrap_5th_pctl":     round(float(np.percentile(bootstrap_sharpes, 5)), 4),
        "bootstrap_95th_pctl":    round(float(np.percentile(bootstrap_sharpes, 95)), 4),
        "p_value_sharpe":         round(p_value_sharpe, 6),
        "significant_5pct":       p_value_sharpe < 0.05,
        "n_months":               n_months,
        "n_bootstrap":            n_bootstrap,
    }

    log.info(
        f"Bootstrap: actual Sharpe={actual_sharpe:.3f} | "
        f"bootstrap mean={np.mean(bootstrap_sharpes):.3f} ± {np.std(bootstrap_sharpes):.3f} | "
        f"p(Sharpe≤0)={p_value_sharpe:.4f}"
    )

    # Excess return test (if benchmark provided)
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        monthly_bench = benchmark_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        common = monthly_port.index.intersection(monthly_bench.index)

        if len(common) >= 12:
            excess = monthly_port.loc[common].values - monthly_bench.loc[common].values
            actual_excess_mean = float(excess.mean()) * 12  # annualised

            # Bootstrap excess returns
            bootstrap_excess = np.zeros(n_bootstrap)
            for i in range(n_bootstrap):
                sample = rng.choice(excess, size=len(excess), replace=True)
                bootstrap_excess[i] = sample.mean() * 12

            p_value_excess = float(np.mean(bootstrap_excess <= 0))

            result.update({
                "actual_excess_return":     round(actual_excess_mean, 4),
                "bootstrap_mean_excess":    round(float(np.mean(bootstrap_excess)), 4),
                "p_value_excess":           round(p_value_excess, 6),
                "excess_significant_5pct":  p_value_excess < 0.05,
            })

            log.info(
                f"Excess return test: actual={actual_excess_mean:.3f} ann | "
                f"p(excess≤0)={p_value_excess:.4f}"
            )

    return result


# 3. Factor Independence

def test_factor_independence(
    value_scores: pd.Series,
    momentum_scores: pd.Series,
    quality_scores: pd.Series,
) -> Dict:
    # Align to common tickers
    common = (value_scores.dropna().index
              .intersection(momentum_scores.dropna().index)
              .intersection(quality_scores.dropna().index))

    if len(common) < 3:
        return {"error": f"Only {len(common)} common tickers — need at least 3"}

    v = value_scores.loc[common].values
    m = momentum_scores.loc[common].values
    q = quality_scores.loc[common].values

    # Pairwise Spearman correlations
    corr_vm, p_vm = sp_stats.spearmanr(v, m)
    corr_vq, p_vq = sp_stats.spearmanr(v, q)
    corr_mq, p_mq = sp_stats.spearmanr(m, q)

    # Independence check: all pairwise |corr| < 0.5 is good
    max_abs_corr = max(abs(corr_vm), abs(corr_vq), abs(corr_mq))
    independent = max_abs_corr < 0.5

    result = {
        "value_momentum": {
            "correlation": round(float(corr_vm), 4),
            "p_value":     round(float(p_vm), 6),
        },
        "value_quality": {
            "correlation": round(float(corr_vq), 4),
            "p_value":     round(float(p_vq), 6),
        },
        "momentum_quality": {
            "correlation": round(float(corr_mq), 4),
            "p_value":     round(float(p_mq), 6),
        },
        "max_abs_correlation": round(float(max_abs_corr), 4),
        "factors_independent": independent,
        "n_tickers": len(common),
        "verdict": (
            "GOOD: Factors are sufficiently independent (max |corr| < 0.5)"
            if independent else
            "WARNING: High factor correlation detected — some redundancy"
        ),
    }

    log.info(
        f"Factor independence: V-M={corr_vm:.3f} V-Q={corr_vq:.3f} M-Q={corr_mq:.3f} | "
        f"max|corr|={max_abs_corr:.3f} | {'PASS' if independent else 'WARN'}"
    )
    return result


# Combined validation report

def generate_validation_report(
    ic_results: Dict,
    bootstrap_results: Dict,
    independence_results: Dict,
) -> Dict:
    verdicts = []

    # IC verdict: at least one factor should have significant IC at some horizon
    ic_sig_count = 0
    ic_total = 0
    if "error" not in ic_results:
        for factor in ic_results:
            for period in ic_results[factor]:
                stats = ic_results[factor][period]
                if isinstance(stats, dict) and "significant" in stats:
                    ic_total += 1
                    if stats["significant"]:
                        ic_sig_count += 1

    if ic_sig_count > 0:
        verdicts.append("PASS")
        ic_verdict = f"PASS: {ic_sig_count}/{ic_total} factor-horizon combos are significant"
    elif ic_total > 0:
        verdicts.append("PARTIAL")
        ic_verdict = f"PARTIAL: 0/{ic_total} factor-horizon combos significant (may need more data)"
    else:
        verdicts.append("FAIL")
        ic_verdict = "FAIL: Could not compute IC tests"

    # Bootstrap verdict
    if "error" not in bootstrap_results:
        if bootstrap_results.get("significant_5pct", False):
            verdicts.append("PASS")
            boot_verdict = f"PASS: Strategy Sharpe significantly > 0 (p={bootstrap_results['p_value_sharpe']:.4f})"
        else:
            verdicts.append("PARTIAL")
            boot_verdict = f"PARTIAL: Sharpe positive but not significant (p={bootstrap_results['p_value_sharpe']:.4f})"
    else:
        verdicts.append("FAIL")
        boot_verdict = "FAIL: Could not run bootstrap test"

    # Independence verdict
    if "error" not in independence_results:
        if independence_results.get("factors_independent", False):
            verdicts.append("PASS")
            indep_verdict = independence_results["verdict"]
        else:
            verdicts.append("PARTIAL")
            indep_verdict = independence_results["verdict"]
    else:
        verdicts.append("FAIL")
        indep_verdict = "FAIL: Could not compute factor independence"

    # Overall
    if all(v == "PASS" for v in verdicts):
        overall = "PASS"
    elif any(v == "FAIL" for v in verdicts):
        overall = "FAIL"
    else:
        overall = "PARTIAL"

    report = {
        "overall_verdict":       overall,
        "ic_verdict":            ic_verdict,
        "bootstrap_verdict":     boot_verdict,
        "independence_verdict":  indep_verdict,
        "ic_significant_count":  ic_sig_count,
        "ic_total_tests":        ic_total,
    }

    log.info(f"Validation report: {overall} | IC: {ic_verdict} | Bootstrap: {boot_verdict}")
    return report


# Display helper

def get_ic_summary_table(ic_results: Dict) -> pd.DataFrame:
    """Convert IC results to a clean DataFrame for display."""
    rows = []
    if "error" in ic_results:
        return pd.DataFrame()

    for factor in ic_results:
        for period in ic_results[factor]:
            stats = ic_results[factor][period]
            if isinstance(stats, dict) and "ic_mean" in stats:
                rows.append({
                    "Factor":      factor.title(),
                    "Horizon (M)": period,
                    "IC Mean":     stats["ic_mean"],
                    "IC Std":      stats["ic_std"],
                    "ICIR":        stats["icir"],
                    "t-stat":      stats["t_stat"],
                    "p-value":     stats["p_value"],
                    "N":           stats["n_obs"],
                    "Sig?":        "Yes" if stats["significant"] else "No",
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from src.data_loader import get_price_data, get_fundamentals, get_fred_data
    from src.factor_model import (
        compute_value_score, compute_momentum_score,
        compute_quality_score, compute_composite_score,
    )
    from src.macro_regime import run_macro_overlay
    from src.portfolio_optimizer import (
        compute_expected_returns, compute_covariance_matrix, optimize_portfolio,
    )
    from src.backtester import run_backtest

    TICKERS = ["AAPL", "MSFT", "JPM", "JNJ", "XOM"]
    START   = "2021-01-01"
    END     = "2024-12-31"

    # Run backtest first (for bootstrap test)
    print("=" * 60)
    print("RUNNING BACKTEST")
    print("=" * 60)

    bt = run_backtest(
        tickers=TICKERS,
        start_date=START,
        end_date=END,
        rebalance_freq="M",
        initial_value=100_000,
    )
    port_ret = bt["daily_returns"]
    bench_ret = bt["benchmark_returns"]

    #  Load data for IC and independence tests 
    print("\n" + "=" * 60)
    print("LOADING DATA FOR HYPOTHESIS TESTS")
    print("=" * 60)

    prices       = get_price_data(TICKERS, START, END)
    fundamentals = get_fundamentals(TICKERS)

    # Test 1: IC Significance 
    print("\n" + "=" * 60)
    print("TEST 1: INFORMATION COEFFICIENT (IC) SIGNIFICANCE")
    print("=" * 60)

    ic_results = test_ic_significance(prices, fundamentals, periods=[1, 3, 6, 12])

    ic_table = get_ic_summary_table(ic_results)
    if not ic_table.empty:
        print(ic_table.to_string(index=False))
    else:
        print(f"IC test result: {ic_results}")

    # Test 2: Bootstrap Strategy Significance
    print("\n" + "=" * 60)
    print("TEST 2: BOOTSTRAP STRATEGY SIGNIFICANCE")
    print("=" * 60)

    boot_results = test_strategy_significance(port_ret, bench_ret, n_bootstrap=10_000)

    print(f"Actual Sharpe:          {boot_results.get('actual_sharpe', 'N/A')}")
    print(f"Bootstrap Mean Sharpe:  {boot_results.get('bootstrap_mean_sharpe', 'N/A')}")
    print(f"Bootstrap Std:          {boot_results.get('bootstrap_std_sharpe', 'N/A')}")
    print(f"Bootstrap 5th pctl:     {boot_results.get('bootstrap_5th_pctl', 'N/A')}")
    print(f"Bootstrap 95th pctl:    {boot_results.get('bootstrap_95th_pctl', 'N/A')}")
    print(f"p-value (Sharpe > 0):   {boot_results.get('p_value_sharpe', 'N/A')}")
    print(f"Significant (5%):       {boot_results.get('significant_5pct', 'N/A')}")

    if "actual_excess_return" in boot_results:
        print(f"\nExcess Return (ann):    {boot_results['actual_excess_return']:.4f}")
        print(f"p-value (excess > 0):   {boot_results['p_value_excess']:.4f}")
        print(f"Excess Significant:     {boot_results['excess_significant_5pct']}")

    # Test 3: Factor Independence 
    print("\n" + "=" * 60)
    print("TEST 3: FACTOR INDEPENDENCE")
    print("=" * 60)

    value_scores    = compute_value_score(fundamentals)
    momentum_scores = compute_momentum_score(prices)
    quality_scores  = compute_quality_score(fundamentals)

    indep_results = test_factor_independence(value_scores, momentum_scores, quality_scores)

    print(f"Value ↔ Momentum:  {indep_results['value_momentum']['correlation']:.4f} "
          f"(p={indep_results['value_momentum']['p_value']:.4f})")
    print(f"Value ↔ Quality:   {indep_results['value_quality']['correlation']:.4f} "
          f"(p={indep_results['value_quality']['p_value']:.4f})")
    print(f"Momentum ↔ Quality:{indep_results['momentum_quality']['correlation']:.4f} "
          f"(p={indep_results['momentum_quality']['p_value']:.4f})")
    print(f"Max |corr|:        {indep_results['max_abs_correlation']:.4f}")
    print(f"Verdict:           {indep_results['verdict']}")

    # Combined Report
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    report = generate_validation_report(ic_results, boot_results, indep_results)

    print(f"Overall:      {report['overall_verdict']}")
    print(f"IC Tests:     {report['ic_verdict']}")
    print(f"Bootstrap:    {report['bootstrap_verdict']}")
    print(f"Independence: {report['independence_verdict']}")
