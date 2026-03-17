import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TRADING_DAYS = 252


# Lookahead prevention

def prevent_lookahead(df: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    cutoff = pd.Timestamp(as_of_date)
    return df[df.index <= cutoff].copy()


def get_rebalance_dates(
    price_df: pd.DataFrame,
    freq: str = "M",
) -> List[pd.Timestamp]:
    dates = price_df.resample(freq).last().index
    # Keep only dates that are in the actual price data
    valid = [d for d in dates if d in price_df.index or
             len(price_df.index[price_df.index <= d]) > 0]

    # Map to the nearest actual trading day on or before
    actual_dates = []
    for d in valid:
        mask = price_df.index <= d
        if mask.any():
            actual_dates.append(price_df.index[mask][-1])

    # Deduplicate and sort
    actual_dates = sorted(set(actual_dates))

    # Need at least 252 days of lookback for the first rebalance
    if len(price_df) > 252:
        min_date = price_df.index[252]
        actual_dates = [d for d in actual_dates if d >= min_date]

    log.info(f"Rebalance dates: {len(actual_dates)} ({freq} frequency)")
    return actual_dates



def compute_benchmark_returns(
    price_df: pd.DataFrame,
    benchmark: str = "SPY",
) -> pd.Series:
    import yfinance as yf

    start = price_df.index[0].strftime("%Y-%m-%d")
    end   = price_df.index[-1].strftime("%Y-%m-%d")

    try:
        raw = yf.download(benchmark, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            log.warning(f"No benchmark data for {benchmark}")
            return pd.Series(dtype=float)

        if isinstance(raw.columns, pd.MultiIndex):
            bench_prices = raw["Close"][benchmark]
        else:
            bench_prices = raw["Close"]

        bench_returns = bench_prices.pct_change().dropna()
        bench_returns.name = benchmark
        return bench_returns

    except Exception as e:
        log.warning(f"Failed to fetch benchmark {benchmark}: {e}")
        return pd.Series(dtype=float)



def run_backtest(
    tickers: List[str],
    start_date: str,
    end_date: str,
    rebalance_freq: str = "M",
    initial_value: float = 100_000.0,
    benchmark: str = "SPY",
) -> Dict:
    from src.data_loader import get_price_data, get_fundamentals, get_fred_data
    from src.factor_model import (
        compute_value_score, compute_momentum_score,
        compute_quality_score, compute_composite_score,
    )
    from src.macro_regime import run_macro_overlay
    from src.portfolio_optimizer import (
        compute_expected_returns, compute_covariance_matrix, optimize_portfolio,
    )

    log.info(f"Starting backtest: {start_date} → {end_date} | "
             f"{len(tickers)} tickers | rebalance={rebalance_freq}")

    # Load full price history (we'll filter with as_of_date at each step) 
    prices = get_price_data(tickers, start_date, end_date)
    if prices.empty:
        raise ValueError("No price data available for the given tickers/dates")

    # Get rebalance dates
    rebalance_dates = get_rebalance_dates(prices, freq=rebalance_freq)
    if len(rebalance_dates) < 2:
        raise ValueError("Need at least 2 rebalance dates for a backtest")

    # Fetch fundamentals and FRED data (full history, filtered per step)
    fundamentals = get_fundamentals(tickers)
    fred = get_fred_data()

    # Walk-forward loop 
    weights_history = []
    turnover_list   = []
    prev_weights    = pd.Series(0.0, index=tickers)

    daily_returns_list = []

    for i, rebal_date in enumerate(rebalance_dates):
        as_of = rebal_date.strftime("%Y-%m-%d")

        try:
            # 1. Filter prices to only data available on this date
            prices_filtered = prevent_lookahead(prices, as_of)

            if len(prices_filtered) < 60:
                log.warning(f"Skipping {as_of}: only {len(prices_filtered)} days of data")
                continue

            # 2. Compute factor scores using only past data
            value    = compute_value_score(fundamentals)
            momentum = compute_momentum_score(prices_filtered, as_of_date=as_of)
            quality  = compute_quality_score(fundamentals)

            # 3. Macro regime using only past FRED data
            macro = run_macro_overlay(fred, as_of_date=as_of)

            # 4. Composite score with regime-adjusted weights
            composite = compute_composite_score(
                value, momentum, quality, macro["factor_weights"]
            )

            # 5. Expected returns and covariance from past prices only
            exp_ret = compute_expected_returns(prices_filtered, composite)
            cov     = compute_covariance_matrix(prices_filtered)

            # 6. Optimize portfolio
            opt = optimize_portfolio(
                exp_ret, cov,
                yield_multiplier=macro["yield_multiplier"],
            )
            new_weights = opt["weights"]

        except Exception as e:
            log.warning(f"Rebalance {as_of} failed: {e}. Keeping previous weights.")
            new_weights = prev_weights

        # Compute turnover
        turnover = float((new_weights - prev_weights).abs().sum()) / 2.0
        turnover_list.append(turnover)

        # Track weights
        weights_history.append((rebal_date, new_weights.to_dict()))

        # Compute daily returns until next rebalance
        if i < len(rebalance_dates) - 1:
            next_rebal = rebalance_dates[i + 1]
        else:
            next_rebal = prices.index[-1]

        # Daily returns in the holding period
        # Include rebal_date as anchor so pct_change captures the first day's return
        period_prices = prices[(prices.index >= rebal_date) & (prices.index <= next_rebal)]
        if len(period_prices) >= 2:
            period_returns = period_prices.pct_change().dropna()

            # Portfolio daily return = weighted sum of individual returns
            active_tickers = new_weights[new_weights > 1e-6].index.tolist()
            for date, row in period_returns.iterrows():
                port_ret = 0.0
                for t in active_tickers:
                    if t in row.index and not np.isnan(row[t]):
                        port_ret += new_weights[t] * row[t]
                daily_returns_list.append((date, port_ret))

        prev_weights = new_weights

    # Build NAV series 
    if not daily_returns_list:
        raise ValueError("No daily returns computed — backtest period too short")

    dates, rets = zip(*daily_returns_list)
    daily_returns = pd.Series(rets, index=pd.DatetimeIndex(dates), name="portfolio")
    daily_returns = daily_returns[~daily_returns.index.duplicated(keep="first")]

    nav = (1 + daily_returns).cumprod() * initial_value
    nav.name = "NAV"

    bench_returns = compute_benchmark_returns(prices, benchmark)
    if not bench_returns.empty:
        # Align benchmark to portfolio dates
        common_dates = daily_returns.index.intersection(bench_returns.index)
        bench_returns = bench_returns.loc[common_dates]

    # Summary stats 
    total_return = float(nav.iloc[-1] / initial_value - 1)
    ann_return   = float((1 + total_return) ** (252 / len(daily_returns)) - 1)
    ann_vol      = float(daily_returns.std() * np.sqrt(252))
    sharpe       = (ann_return - 0.05) / ann_vol if ann_vol > 0 else 0.0
    max_dd       = float((nav / nav.cummax() - 1).min())
    avg_turnover = float(np.mean(turnover_list)) if turnover_list else 0.0

    log.info(
        f"Backtest complete | Return={total_return:.2%} | "
        f"Ann={ann_return:.2%} | Vol={ann_vol:.2%} | "
        f"Sharpe={sharpe:.3f} | MaxDD={max_dd:.2%} | "
        f"Avg turnover={avg_turnover:.2%}"
    )

    return {
        "nav":               nav,
        "daily_returns":     daily_returns,
        "benchmark_returns": bench_returns,
        "weights_history":   weights_history,
        "turnover":          turnover_list,
        "rebalance_dates":   rebalance_dates,
        "metadata": {
            "tickers":         tickers,
            "start_date":      start_date,
            "end_date":        end_date,
            "rebalance_freq":  rebalance_freq,
            "initial_value":   initial_value,
            "benchmark":       benchmark,
            "total_return":    round(total_return, 4),
            "ann_return":      round(ann_return, 4),
            "ann_volatility":  round(ann_vol, 4),
            "sharpe_ratio":    round(sharpe, 3),
            "max_drawdown":    round(max_dd, 4),
            "avg_turnover":    round(avg_turnover, 4),
            "n_rebalances":    len(rebalance_dates),
        },
    }



if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    TICKERS = ["AAPL", "MSFT", "JPM", "JNJ", "XOM"]

    print("=" * 60)
    print("RUNNING WALK-FORWARD BACKTEST")
    print("=" * 60)

    result = run_backtest(
        tickers=TICKERS,
        start_date="2021-01-01",
        end_date="2024-12-31",
        rebalance_freq="M",
        initial_value=100_000,
    )

    meta = result["metadata"]
    print(f"\n=== Backtest Results ===")
    print(f"Period:          {meta['start_date']} → {meta['end_date']}")
    print(f"Tickers:         {meta['tickers']}")
    print(f"Rebalances:      {meta['n_rebalances']} ({meta['rebalance_freq']})")
    print(f"Total Return:    {meta['total_return']:.2%}")
    print(f"Ann. Return:     {meta['ann_return']:.2%}")
    print(f"Ann. Volatility: {meta['ann_volatility']:.2%}")
    print(f"Sharpe Ratio:    {meta['sharpe_ratio']:.3f}")
    print(f"Max Drawdown:    {meta['max_drawdown']:.2%}")
    print(f"Avg Turnover:    {meta['avg_turnover']:.2%}")

    print(f"\nNAV (first 5 days):")
    print(result["nav"].head())
    print(f"\nNAV (last 5 days):")
    print(result["nav"].tail())

    print(f"\nFinal NAV: ${result['nav'].iloc[-1]:,.0f}")

    # Show first 3 weight snapshots
    print(f"\n=== Weight History (first 3 rebalances) ===")
    for date, weights in result["weights_history"][:3]:
        active = {k: round(v, 3) for k, v in weights.items() if v > 0.001}
        print(f"  {date.strftime('%Y-%m-%d')}: {active}")
