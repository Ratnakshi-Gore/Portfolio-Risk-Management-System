import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)



STRESS_SCENARIOS: Dict[str, Dict] = {
    "2008 Financial Crisis": {
        "start":       "2008-09-01",
        "end":         "2008-12-31",
        "description": "Lehman Brothers collapse, global banking crisis. "
                       "S&P 500 fell ~38% in 4 months.",
        "market_return": -0.38,   # fallback if ticker has no data
    },
    "COVID Crash (2020)": {
        "start":       "2020-02-19",
        "end":         "2020-03-23",
        "description": "Global pandemic panic. S&P 500 fell ~34% in 33 days — "
                       "the fastest bear market in history.",
        "market_return": -0.34,
    },
    "2022 Rate Shock": {
        "start":       "2022-01-03",
        "end":         "2022-12-30",
        "description": "Fed raised rates from 0% to 4.5%. Both stocks and bonds fell. "
                       "S&P 500 dropped ~19%.",
        "market_return": -0.19,
    },
    "Dot-com Bust": {
        "start":       "2000-03-24",
        "end":         "2002-10-09",
        "description": "Tech bubble burst. NASDAQ lost 78%, S&P 500 fell ~49% "
                       "over 2.5 years.",
        "market_return": -0.49,
    },
}


# Fetch actual historical returns during a crisis

def _get_crisis_returns(
    tickers: List[str],
    start: str,
    end: str,
    market_return: float,
) -> pd.Series:
    returns = {}

    try:
        # Download all tickers at once for efficiency
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

        if raw.empty:
            log.warning(f"No data for period {start} to {end} — using market fallback")
            return pd.Series({t: market_return for t in tickers})

        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]]
            prices.columns = [tickers[0]] if len(tickers) == 1 else prices.columns

        for ticker in tickers:
            if ticker in prices.columns:
                series = prices[ticker].dropna()
                if len(series) >= 2:
                    total_return = (series.iloc[-1] / series.iloc[0]) - 1.0
                    returns[ticker] = float(total_return)
                else:
                    returns[ticker] = market_return
                    log.debug(f"{ticker}: insufficient data for {start}–{end}, using market fallback")
            else:
                returns[ticker] = market_return
                log.debug(f"{ticker}: no data for {start}–{end}, using market fallback")

    except Exception as e:
        log.warning(f"Failed to fetch crisis data: {e} — using market fallback for all")
        return pd.Series({t: market_return for t in tickers})

    return pd.Series(returns)



def run_stress_test(
    weights: pd.Series,
    scenario_name: str,
    initial_value: float = 100_000.0,
) -> Dict:
    if scenario_name not in STRESS_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}. "
                        f"Valid: {list(STRESS_SCENARIOS.keys())}")

    scenario = STRESS_SCENARIOS[scenario_name]
    active = weights[weights > 1e-6]
    tickers = active.index.tolist()

    if not tickers:
        return {
            "scenario": scenario_name,
            "description": scenario["description"],
            "period": f"{scenario['start']} to {scenario['end']}",
            "portfolio_return": 0.0,
            "dollar_loss": 0.0,
            "ticker_returns": {},
            "worst_ticker": None,
            "best_ticker": None,
        }

    # Fetch actual returns during the crisis
    crisis_returns = _get_crisis_returns(
        tickers,
        scenario["start"],
        scenario["end"],
        scenario["market_return"],
    )

    # Weighted portfolio return
    w = active.values
    r = crisis_returns.reindex(tickers).values
    portfolio_return = float(w @ r)
    dollar_loss = portfolio_return * initial_value

    # Individual ticker breakdown
    ticker_returns = {t: round(float(crisis_returns.get(t, scenario["market_return"])), 4)
                      for t in tickers}

    worst = min(ticker_returns, key=ticker_returns.get)
    best  = max(ticker_returns, key=ticker_returns.get)

    log.info(
        f"Stress [{scenario_name}]: portfolio={portfolio_return:.2%} "
        f"(${dollar_loss:+,.0f}) | worst={worst} ({ticker_returns[worst]:.2%})"
    )

    return {
        "scenario":         scenario_name,
        "description":      scenario["description"],
        "period":           f"{scenario['start']} to {scenario['end']}",
        "portfolio_return": round(portfolio_return, 4),
        "dollar_loss":      round(dollar_loss, 2),
        "ticker_returns":   ticker_returns,
        "worst_ticker":     worst,
        "best_ticker":      best,
    }



def run_all_scenarios(
    weights: pd.Series,
    initial_value: float = 100_000.0,
) -> List[Dict]:
    results = []
    for name in STRESS_SCENARIOS:
        result = run_stress_test(weights, name, initial_value)
        results.append(result)

    # Sort by portfolio return (worst first)
    results.sort(key=lambda x: x["portfolio_return"])

    log.info(f"Ran {len(results)} stress scenarios | "
             f"worst: {results[0]['scenario']} ({results[0]['portfolio_return']:.2%})")

    return results


def get_scenario_summary(results: List[Dict]) -> pd.DataFrame:
    """Convert stress test results to a clean DataFrame for display."""
    rows = []
    for r in results:
        rows.append({
            "Scenario":         r["scenario"],
            "Period":           r["period"],
            "Portfolio Return": r["portfolio_return"],
            "Dollar P&L":      r["dollar_loss"],
            "Worst Ticker":    r["worst_ticker"],
            "Best Ticker":     r["best_ticker"],
        })
    return pd.DataFrame(rows)



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

    print("\n" + "=" * 60)
    print("STRESS TEST RESULTS")
    print("=" * 60)

    results = run_all_scenarios(opt["weights"], initial_value=100_000)

    for r in results:
        print(f"\n--- {r['scenario']} ---")
        print(f"Period:           {r['period']}")
        print(f"Portfolio Return: {r['portfolio_return']:.2%}")
        print(f"Dollar P&L:       ${r['dollar_loss']:+,.0f}")
        print(f"Worst ticker:     {r['worst_ticker']} ({r['ticker_returns'][r['worst_ticker']]:.2%})")
        print(f"Best ticker:      {r['best_ticker']} ({r['ticker_returns'][r['best_ticker']]:.2%})")
        print(f"All returns:      {r['ticker_returns']}")

    print("\n=== Summary Table ===")
    summary = get_scenario_summary(results)
    print(summary.to_string(index=False))
