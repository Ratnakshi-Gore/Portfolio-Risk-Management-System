import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="Portfolio Risk Management",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.sidebar.title("Portfolio Risk Management")
st.sidebar.markdown("---")

tickers_input = st.sidebar.text_input(
    "Tickers (comma-separated)",
    value="AAPL, MSFT, JPM, JNJ, XOM",
    help="Enter stock tickers separated by commas",
)

backtest_years = st.sidebar.slider(
    "Backtest Period (years)", min_value=2, max_value=10, value=3,
    help="Minimum 3 years recommended for meaningful results",
)

mc_simulations = st.sidebar.number_input(
    "Monte Carlo Simulations", min_value=100, max_value=250000,
    value=1000, step=500,
)

run_button = st.sidebar.button("Run Full Analysis", type="primary", use_container_width=True)

st.sidebar.markdown("---")


@st.cache_data(show_spinner=False, ttl=3600)
def run_full_pipeline(tickers_str, years, n_sims):
    """Run the complete pipeline and return all results."""
    from src.backtester import run_backtest
    from src.risk_metrics import (
        compute_all_metrics, get_metrics_summary,
        compute_fama_french_alpha, compute_drawdown_series,
        compute_rolling_metrics,
    )
    from src.monte_carlo import run_monte_carlo, get_simulation_summary
    from src.stress_test import run_all_scenarios
    from src.data_loader import (
        get_price_data, get_fama_french_factors, get_fred_data,
        get_fundamentals,
    )
    from src.macro_regime import run_macro_overlay
    from src.factor_model import (
        compute_value_score, compute_momentum_score, compute_quality_score,
    )
    from src.hypothesis_testing import (
        test_strategy_significance, test_factor_independence,
    )

    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = f"{datetime.now().year - years}-01-01"

    results = {"tickers": tickers, "start_date": start_date, "end_date": end_date}

    # 1. Backtest
    bt = run_backtest(tickers=tickers, start_date=start_date, end_date=end_date,
                      rebalance_freq="M", initial_value=100_000)
    results["bt"] = bt
    results["sector_map"] = bt.get("sector_map", {})

    # 2. Risk metrics
    metrics = compute_all_metrics(bt["daily_returns"], bt["benchmark_returns"])
    results["metrics"] = metrics
    results["summary"] = get_metrics_summary(metrics)

    # 3. Drawdown & rolling
    results["drawdown"] = compute_drawdown_series(bt["daily_returns"])
    try:
        results["rolling"] = compute_rolling_metrics(
            bt["daily_returns"], bt["benchmark_returns"]
        )
    except Exception:
        results["rolling"] = None

    # 4. Fama-French
    try:
        ff = get_fama_french_factors()
        results["ff"] = compute_fama_french_alpha(bt["daily_returns"], ff)
    except Exception:
        results["ff"] = None

    # 5. Macro regime
    try:
        fred = get_fred_data()
        results["regime"] = run_macro_overlay(fred_data=fred)
    except Exception:
        results["regime"] = None

    # 6. Monte Carlo
    try:
        latest_w = _get_weights(bt["weights_history"], tickers)
        prices = get_price_data(tickers, start_date, end_date)
        paths = run_monte_carlo(latest_w, prices, n_simulations=n_sims)
        results["mc_paths"] = paths
        results["mc_summary"] = get_simulation_summary(paths, initial_value=100_000)
    except Exception:
        results["mc_paths"] = None
        results["mc_summary"] = None

    # 7. Stress tests
    try:
        latest_w = _get_weights(bt["weights_history"], tickers)
        results["stress"] = run_all_scenarios(latest_w, initial_value=100_000)
    except Exception:
        results["stress"] = None

    # 8. Factor scores
    try:
        fundamentals = get_fundamentals(tickers)
        prices = get_price_data(tickers, start_date, end_date)
        v = compute_value_score(fundamentals)
        m = compute_momentum_score(prices)
        q = compute_quality_score(fundamentals)
        scores = pd.DataFrame({"Value": v, "Momentum": m, "Quality": q})
        _, wts = bt["weights_history"][-1] if bt["weights_history"] else (None, {})
        scores["Weight"] = pd.Series(wts)
        scores = scores.fillna(0)
        results["scores"] = scores
    except Exception:
        results["scores"] = None

    # 9. Hypothesis testing
    try:
        results["bootstrap"] = test_strategy_significance(
            bt["daily_returns"], bt["benchmark_returns"]
        )
    except Exception:
        results["bootstrap"] = None

    try:
        if results.get("scores") is not None:
            s = results["scores"]
            results["independence"] = test_factor_independence(
                s["Value"], s["Momentum"], s["Quality"]
            )
        else:
            results["independence"] = None
    except Exception:
        results["independence"] = None

    return results


def _get_weights(weights_history, tickers):
    if not weights_history:
        n = len(tickers)
        return pd.Series({t: 1.0 / n for t in tickers})
    _, wts = weights_history[-1]
    return pd.Series(wts)

if run_button:
    with st.spinner("Running full analysis pipeline... (this takes ~2 minutes)"):
        try:
            results = run_full_pipeline(tickers_input, backtest_years, mc_simulations)
            st.session_state["results"] = results
            st.session_state["run_complete"] = True
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.session_state["run_complete"] = False

# Check if we have results
if not st.session_state.get("run_complete"):
    st.title("Portfolio Risk Management System")
    st.stop()

R = st.session_state["results"]
bt = R["bt"]
metrics = R["metrics"]

tab1, tab2, tab3, tab4 = st.tabs([
    "Portfolio", "Risk", "Backtest", "Statistics"
])


with tab1:
    st.header("Portfolio Overview")

    # Top-level KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return", f"{metrics['total_return']:.2%}")
    col2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")
    col3.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
    col4.metric("Beta", f"{metrics['beta']:.3f}")

    st.markdown("---")

    # Weights pie chart, sector allocation, and regime info
    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        st.subheader("Current Portfolio Weights")
        scores = R.get("scores")
        if scores is not None:
            weights = scores["Weight"]
            weights = weights[weights > 0]
            fig_pie = go.Figure(data=[go.Pie(
                labels=weights.index,
                values=weights.values,
                hole=0.4,
                textinfo="label+percent",
                marker=dict(colors=px.colors.qualitative.Set2),
            )])
            fig_pie.update_layout(height=350, margin=dict(t=20, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.subheader("Sector Allocation")
        sector_map = R.get("sector_map", {})
        if scores is not None and sector_map:
            w = scores["Weight"]
            w = w[w > 0]
            sector_weights = {}
            for ticker, wt in w.items():
                sec = sector_map.get(ticker, "Unknown")
                sector_weights[sec] = sector_weights.get(sec, 0) + wt
            if sector_weights:
                fig_sector = go.Figure(data=[go.Pie(
                    labels=list(sector_weights.keys()),
                    values=list(sector_weights.values()),
                    hole=0.4,
                    textinfo="label+percent",
                    marker=dict(colors=px.colors.qualitative.Pastel),
                )])
                fig_sector.update_layout(height=350, margin=dict(t=20, b=20))
                st.plotly_chart(fig_sector, use_container_width=True)
        else:
            st.info("Sector data not available")

    with c3:
        st.subheader("Macro Regime")
        regime = R.get("regime")
        if regime:
            regime_name = str(regime.get("regime", "UNKNOWN"))
            desc = regime.get("description", "")

            regime_colors = {
                "STABLE": "green", "RISING": "orange",
                "HIGH_RISING": "red", "FALLING": "blue",
            }
            color = regime_colors.get(regime_name.split(".")[-1], "grey")

            st.markdown(
                f'<div style="background-color:{color};color:white;padding:15px;'
                f'border-radius:10px;text-align:center;font-size:24px;font-weight:bold;">'
                f'{regime_name}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"*{desc}*")

            factor_wts = regime.get("factor_weights", {})
            if factor_wts:
                st.markdown("**Factor Weights:**")
                fw_df = pd.DataFrame([factor_wts]).T
                fw_df.columns = ["Weight"]
                st.dataframe(fw_df.style.format("{:.2%}"), use_container_width=True)
        else:
            st.info("Regime detection not available")

    # Factor scores table
    st.markdown("---")
    st.subheader("Factor Scores (Latest Rebalance)")
    if scores is not None:
        styled = scores.style.format({
            "Value": "{:.4f}", "Momentum": "{:.4f}",
            "Quality": "{:.4f}", "Weight": "{:.2%}",
        }).background_gradient(subset=["Value", "Momentum", "Quality"], cmap="RdYlGn")
        st.dataframe(styled, use_container_width=True)


with tab2:
    st.header("Risk Analysis")

    # VaR/CVaR metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("VaR (95%)", f"{metrics['var_95_daily']:.2%}")
    col2.metric("CVaR (95%)", f"{metrics['cvar_95_daily']:.2%}")
    col3.metric("VaR (99%)", f"{metrics['var_99_daily']:.2%}")
    col4.metric("CVaR (99%)", f"{metrics['cvar_99_daily']:.2%}")

    st.markdown("---")

    # Monte Carlo
    st.subheader("Monte Carlo Simulation (1-Year Forward)")
    mc_paths = R.get("mc_paths")
    mc_summary = R.get("mc_summary")

    if mc_paths is not None:
        c1, c2 = st.columns([2, 1])

        with c1:
            n_paths = min(200, mc_paths.shape[0])
            fig_mc = go.Figure()

            # Sample paths (faint)
            for i in range(n_paths):
                fig_mc.add_trace(go.Scatter(
                    y=mc_paths[i], mode="lines",
                    line=dict(color="rgba(100,149,237,0.05)", width=0.5),
                    showlegend=False, hoverinfo="skip",
                ))

            # Percentile bands
            p5 = np.percentile(mc_paths, 5, axis=0)
            p50 = np.percentile(mc_paths, 50, axis=0)
            p95 = np.percentile(mc_paths, 95, axis=0)

            fig_mc.add_trace(go.Scatter(y=p95, mode="lines", name="95th %ile",
                                        line=dict(color="green", dash="dash")))
            fig_mc.add_trace(go.Scatter(y=p50, mode="lines", name="Median",
                                        line=dict(color="navy", width=2)))
            fig_mc.add_trace(go.Scatter(y=p5, mode="lines", name="5th %ile",
                                        line=dict(color="red", dash="dash")))

            fig_mc.update_layout(
                title="Simulated Portfolio Value (1 Year)",
                xaxis_title="Trading Days",
                yaxis_title="Portfolio Value ($)",
                yaxis_tickformat="$,.0f",
                height=400,
            )
            st.plotly_chart(fig_mc, use_container_width=True)

        with c2:
            if mc_summary:
                st.markdown("**Simulation Summary**")
                display_keys = [
                    ("prob_profit", "P(Profit)", "{:.1%}"),
                    ("prob_loss_10pct", "P(Loss > 10%)", "{:.1%}"),
                    ("prob_gain_20pct", "P(Gain > 20%)", "{:.1%}"),
                    ("final_median", "Median Value", "${:,.0f}"),
                    ("p5", "5th Percentile", "${:,.0f}"),
                    ("p95", "95th Percentile", "${:,.0f}"),
                ]
                for key, label, fmt in display_keys:
                    val = mc_summary.get(key, 0)
                    if val is not None:
                        st.metric(label, fmt.format(val))

    # Stress tests
    st.markdown("---")
    st.subheader("Stress Test Results")
    stress = R.get("stress")
    if stress:
        stress_df = pd.DataFrame(stress)
        fig_stress = go.Figure(data=[go.Bar(
            y=stress_df["scenario"],
            x=stress_df["portfolio_return"] * 100,
            orientation="h",
            marker_color=["#d32f2f" if r < -0.2 else "#ff9800" if r < -0.1
                          else "#4caf50" for r in stress_df["portfolio_return"]],
            text=[f"{r:.1%}" for r in stress_df["portfolio_return"]],
            textposition="outside",
        )])
        fig_stress.update_layout(
            title="Portfolio Impact Under Historical Crises",
            xaxis_title="Portfolio Return (%)",
            height=300,
            margin=dict(l=200),
        )
        st.plotly_chart(fig_stress, use_container_width=True)

        # Detail table
        detail_df = stress_df[["scenario", "portfolio_return", "dollar_loss", "worst_ticker"]].copy()
        detail_df.columns = ["Scenario", "Return", "Dollar Loss", "Worst Ticker"]
        detail_df["Return"] = detail_df["Return"].apply(lambda x: f"{x:.2%}")
        detail_df["Dollar Loss"] = detail_df["Dollar Loss"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(detail_df, use_container_width=True, hide_index=True)


with tab3:
    st.header("Backtest Results")

    # KPIs
    meta = bt["metadata"]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Return", f"{meta['total_return']:.2%}")
    col2.metric("Sharpe", f"{meta['sharpe_ratio']:.3f}")
    col3.metric("Rebalances", meta['n_rebalances'])
    col4.metric("Avg Turnover", f"{meta['avg_turnover']:.2%}")
    col5.metric("Calmar", f"{metrics['calmar_ratio']:.3f}")

    st.markdown("---")

    # NAV chart
    st.subheader("Portfolio NAV vs Benchmark (SPY)")
    nav = bt["nav"]
    bench_nav = (1 + bt["benchmark_returns"]).cumprod() * meta.get("initial_value", 100_000)

    fig_nav = go.Figure()
    fig_nav.add_trace(go.Scatter(
        x=nav.index, y=nav.values, name="Portfolio",
        line=dict(color="navy", width=2),
    ))
    fig_nav.add_trace(go.Scatter(
        x=bench_nav.index, y=bench_nav.values, name="Benchmark (SPY)",
        line=dict(color="grey", width=1.5, dash="dash"),
    ))
    fig_nav.update_layout(
        xaxis_title="Date", yaxis_title="Portfolio Value ($)",
        yaxis_tickformat="$,.0f", height=400,
        legend=dict(x=0.02, y=0.98),
    )
    st.plotly_chart(fig_nav, use_container_width=True)

    # Drawdown chart
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Drawdown")
        dd = R.get("drawdown")
        if dd is not None:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=dd.index, y=dd.values * 100, fill="tozeroy",
                line=dict(color="red", width=1),
                fillcolor="rgba(255,0,0,0.2)",
            ))
            fig_dd.update_layout(
                xaxis_title="Date", yaxis_title="Drawdown (%)",
                height=300,
            )
            st.plotly_chart(fig_dd, use_container_width=True)

    with c2:
        st.subheader("Rolling Sharpe (252-day)")
        rolling = R.get("rolling")
        if rolling is not None and "rolling_sharpe" in rolling.columns:
            fig_rs = go.Figure()
            rs = rolling["rolling_sharpe"].dropna()
            fig_rs.add_trace(go.Scatter(
                x=rs.index, y=rs.values,
                line=dict(color="navy", width=1.5),
            ))
            fig_rs.add_hline(y=0, line_dash="dash", line_color="grey")
            fig_rs.update_layout(
                xaxis_title="Date", yaxis_title="Rolling Sharpe",
                height=300,
            )
            st.plotly_chart(fig_rs, use_container_width=True)

    # Full metrics table
    st.markdown("---")
    st.subheader("Full Performance Metrics")
    summary = R.get("summary")
    if summary is not None:
        st.dataframe(summary, use_container_width=True, hide_index=True)

    # Fama-French
    ff = R.get("ff")
    if ff:
        st.markdown("---")
        st.subheader("Fama-French 3-Factor Regression")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("FF3 Alpha (ann.)", f"{ff['alpha_annual']:.4%}")
        c2.metric("p-value", f"{ff['p_value_alpha']:.4f}")
        c3.metric("R-squared", f"{ff['r_squared']:.4f}")
        c4.metric("Observations", ff['n_observations'])

        st.markdown("**Factor Betas:**")
        beta_df = pd.DataFrame({
            "Factor": ["Market (Mkt-RF)", "Size (SMB)", "Value (HML)"],
            "Beta": [ff["beta_mkt"], ff["beta_smb"], ff["beta_hml"]],
        })
        st.dataframe(beta_df, hide_index=True)


with tab4:
    st.header("Statistical Validation")

    # Bootstrap test
    st.subheader("Strategy Significance (Bootstrap Test)")
    bootstrap = R.get("bootstrap")
    if bootstrap:
        c1, c2, c3 = st.columns(3)
        c1.metric("Sharpe Ratio", f"{bootstrap.get('actual_sharpe', 0):.3f}")
        c2.metric("Bootstrap p-value", f"{bootstrap.get('p_value_sharpe', 0):.4f}")
        c3.metric("Excess Return p-value", f"{bootstrap.get('p_value_excess', 0):.4f}")

        sig = bootstrap.get("p_value_sharpe", 1)
        if sig < 0.05:
            st.success("Strategy Sharpe is statistically significant at 5% level")
        elif sig < 0.10:
            st.warning("Strategy Sharpe is marginally significant (5-10% level)")
        else:
            st.info(
                f"Strategy Sharpe not significant at 10% level (p={sig:.3f}). "
                "This is common with small samples (< 60 months)."
            )
    else:
        st.info("Bootstrap test not available")

    st.markdown("---")

    # Factor independence
    st.subheader("Factor Independence")
    independence = R.get("independence")
    if independence:
        pairs = independence.get("pairs", [])
        if pairs:
            ind_df = pd.DataFrame(pairs)
            ind_df.columns = ["Factor 1", "Factor 2", "Correlation",
                              "p-value", "Independent?"]
            st.dataframe(ind_df, hide_index=True, use_container_width=True)

            # Correlation heatmap
            factors = sorted(set(
                [p[0] for p in pairs] + [p[1] for p in pairs]
            ))
            n = len(factors)
            corr_matrix = np.eye(n)
            idx = {f: i for i, f in enumerate(factors)}
            for p in pairs:
                i, j = idx[p[0]], idx[p[1]]
                corr_matrix[i, j] = p[2]
                corr_matrix[j, i] = p[2]

            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix, x=factors, y=factors,
                colorscale="RdBu_r", zmin=-1, zmax=1,
                text=np.round(corr_matrix, 3),
                texttemplate="%{text}",
                hoverinfo="skip",
            ))
            fig_corr.update_layout(
                title="Factor Correlation Matrix",
                height=350, width=400,
            )
            st.plotly_chart(fig_corr)

    st.markdown("---")

    # Weights history over time
    st.subheader("Portfolio Weights Over Time")
    wh = bt.get("weights_history")
    if wh:
        all_tickers = sorted(set(t for _, w in wh for t in w.keys()))
        wh_data = []
        for date, wts in wh:
            row = {"Date": date}
            for t in all_tickers:
                row[t] = wts.get(t, 0.0)
            wh_data.append(row)
        wh_df = pd.DataFrame(wh_data).set_index("Date")

        fig_wh = go.Figure()
        colors = px.colors.qualitative.Set2
        for i, ticker in enumerate(all_tickers):
            fig_wh.add_trace(go.Scatter(
                x=wh_df.index, y=wh_df[ticker],
                name=ticker, stackgroup="one",
                line=dict(width=0),
                fillcolor=colors[i % len(colors)],
            ))
        fig_wh.update_layout(
            yaxis_title="Weight", yaxis_tickformat=".0%",
            height=400, legend=dict(x=1.02, y=1),
        )
        st.plotly_chart(fig_wh, use_container_width=True)
