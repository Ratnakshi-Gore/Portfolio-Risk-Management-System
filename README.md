# Portfolio Risk Management System

A multi-factor equity risk management platform that scores stocks using Value, Momentum, and Quality signals, overlays macroeconomic regime detection, optimizes portfolios via Markowitz mean-variance optimization, and validates performance through backtesting, Monte Carlo simulation, stress testing, and statistical hypothesis testing.

Built with both a **Streamlit web UI** and an **Excel (.xlsm) frontend** powered by win32com COM automation.


---

## Architecture

```
User Input (tickers, backtest period)
        |
        v
+------------------+     +-------------------+
|   data_loader    | --> |   factor_model    |
| (yfinance, FRED, |     | (Value, Momentum, |
|  Fama-French)    |     |  Quality z-scores)|
+------------------+     +-------------------+
        |                         |
        v                         v
+------------------+     +-------------------+
|  macro_regime    | --> | portfolio_optimizer|
| (Fed rate regime,|     | (Markowitz MVO,   |
|  yield curve)    |     |  Ledoit-Wolf cov) |
+------------------+     +-------------------+
        |                         |
        v                         v
+------------------+     +-------------------+
|   backtester     |     |   monte_carlo     |
| (walk-forward,   |     | (10k GBM paths,   |
|  monthly rebal)  |     |  VaR/CVaR)        |
+------------------+     +-------------------+
        |                         |
        v                         v
+------------------+     +-------------------+
|  risk_metrics    |     |   stress_test     |
| (Sharpe, Sortino,|     | (2008, COVID,     |
|  FF3 alpha, etc) |     |  2022, Dot-com)   |
+------------------+     +-------------------+
        |
        v
+------------------+
|hypothesis_testing|
| (IC tests,       |
|  bootstrap,      |
|  factor indep.)  |
+------------------+
        |
        v
+------------------+     +-------------------+
|  Streamlit App   |     |  Excel Interface  |
| (4-tab web UI)   |     | (win32com live    |
|                  |     |  writeback)       |
+------------------+     +-------------------+
```

---

## Factor Model

Three cross-sectional factor signals, each winsorized at 1st/99th percentile and converted to z-scores:

| Factor | Components | Weight (Stable Regime) |
|--------|-----------|----------------------|
| **Value** | P/B (40%), P/E (35%), FCF Yield (25%) | 33% |
| **Momentum** | 12-month return minus 1-month (skip reversal) | 34% |
| **Quality** | ROE, profit margin, earnings quality, FCF quality, current ratio, debt-to-equity | 33% |

A 45-day earnings lag is applied to all quarterly fundamentals to prevent lookahead bias.

---

## Macro Regime Overlay

Detects the Fed interest rate environment using FRED data (Federal Funds Rate, 10Y-2Y spread) and adjusts factor weights accordingly:

| Regime | Value | Momentum | Quality |
|--------|-------|----------|---------|
| HIGH_RISING | 20% | 30% | 50% |
| RISING | 25% | 35% | 40% |
| FALLING | 25% | 45% | 30% |
| STABLE | 33% | 34% | 33% |

A yield curve multiplier (0.5-1.0) scales down portfolio exposure when the curve inverts (recession signal).

---

## Portfolio Optimization

- **Expected returns:** 70% historical mean + 30% factor-score tilt
- **Covariance matrix:** Ledoit-Wolf shrinkage estimator (scikit-learn)
- **Objective:** Maximize Sharpe ratio via SLSQP
- **Constraints:** Long-only, weights sum to 1, max 15% per stock
- **Efficient frontier:** 50-point curve for visualization

---

## Risk Analysis

### Monte Carlo Simulation
- 10,000 forward paths using Geometric Brownian Motion
- Cholesky decomposition for correlated random shocks
- Reports VaR/CVaR at 95% and 99% confidence levels

### Stress Testing
Historical crisis scenarios applied to current portfolio weights:
- **2008 Financial Crisis** (Sep-Dec 2008)
- **COVID Crash** (Feb-Mar 2020)
- **2022 Rate Shock** (Jan-Dec 2022)
- **Dot-com Bust** (Mar 2000 - Oct 2002)

Uses actual historical returns from yfinance for each ticker during the crisis period.

### Risk Metrics
Sharpe, Sortino, Calmar, Max Drawdown, VaR/CVaR, Beta, CAPM Alpha, Information Ratio, Win Rate, Profit Factor, Skewness, Kurtosis, Fama-French 3-Factor Alpha.

---

## Backtesting

Walk-forward backtest with monthly rebalancing:
- At each rebalance date, only data available on that date is used
- 45-day earnings lag prevents look-ahead on quarterly fundamentals
- Tracks portfolio NAV, benchmark (SPY) comparison, turnover, and weight history

---

## Statistical Validation

1. **Information Coefficient (IC):** Spearman rank correlation between factor scores and forward returns at 1/3/6/12 month horizons. Tests H0: mean IC = 0.
2. **Bootstrap Test:** 10,000 bootstrap resamples of monthly returns. Reports p-value that observed Sharpe ratio is not due to chance.
3. **Factor Independence:** Pairwise Spearman correlations between Value, Momentum, and Quality scores. Low correlation confirms factors capture different signals.

---

### Installation

```bash
git clone https://github.com/yourusername/portfolio-risk-management.git
cd portfolio-risk-management
pip install -r requirements.txt
```

For the Excel interface (Windows only):
```bash
pip install pywin32
```

---

## Usage

### Streamlit Web App

```bash
streamlit run streamlit_app.py
```

Opens at http://localhost:8501. Enter tickers in the sidebar, adjust parameters, and click Run Analysis. Results are displayed across 4 tabs: Portfolio, Risk, Backtest, and Statistics.

### Excel Interface (Windows)

1. Open `excel/PortfolioRiskTool.xlsm` in Excel
2. Enable macros when prompted
3. Open the VBA editor (Alt+F11), insert a Module, paste the macro from `excel/VBA_MACRO.txt`
4. Add a Form Control button on the Analysis sheet and assign the `RunAnalysis` macro
5. Click the button - it prompts for tickers and backtest years via dialog boxes
6. Results populate LIVE across 3 sheets: Analysis, Backtest, Raw Data

The Python script connects to the running Excel instance via COM automation and writes results in real-time.

### Jupyter Notebook

```bash
cd notebooks
jupyter notebook exploration.ipynb
```

Interactive walkthrough of data loading, factor analysis, optimization, backtesting, and statistical validation.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Parquet caching with TTL | Avoids repeated API calls; different TTLs for price (1 day) vs fundamentals (7 days) vs FF factors (30 days) |
| Ledoit-Wolf shrinkage | Sample covariance is noisy with small stock universes; shrinkage improves out-of-sample stability |
| 45-day earnings lag | Quarterly financials aren't available on filing date; 45 days is conservative estimate for SEC processing |
| 12M-1M momentum | Standard academic approach; skipping the most recent month avoids short-term reversal effect |
| Win32COM over xlwings | xlwings VBA reference wouldn't load on target machine; win32com provides direct COM access without add-in dependencies |
| Max 15% per stock | Prevents excessive concentration while allowing meaningful positions |

---
