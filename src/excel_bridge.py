import sys
import os
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# Win32COM helpers

def rgb(r: int, g: int, b: int) -> int:
    # Convert RGB to BGR integer for win32com Excel colors.
    return r + (g * 256) + (b * 65536)

# Color constants
NAVY = rgb(0, 51, 102)
WHITE = rgb(255, 255, 255)
GREEN = rgb(0, 128, 0)
RED = rgb(200, 0, 0)
BLUE = rgb(0, 0, 200)
LIGHT_BLUE = rgb(217, 225, 242)   # D9E1F2
HEADER_FILL = rgb(0, 51, 102)     # dark navy
INPUT_YELLOW = rgb(255, 255, 204) # FFFFCC
GREY = rgb(102, 102, 102)


def connect_to_excel(workbook_name: str = "PortfolioRiskTool"):
    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()

    try:
        app = win32com.client.GetActiveObject("Excel.Application")
    except Exception:
        raise RuntimeError(
            "Excel is not running. Please open PortfolioRiskTool.xlsm first."
        )

    # Find our workbook among all open workbooks
    wb = None
    for i in range(1, app.Workbooks.Count + 1):
        name = app.Workbooks(i).Name
        if workbook_name.lower() in name.lower():
            wb = app.Workbooks(i)
            break

    if wb is None:
        raise RuntimeError(
            f"Workbook '{workbook_name}' not found in open Excel workbooks.\n"
            f"Open workbooks: {[app.Workbooks(i+1).Name for i in range(app.Workbooks.Count)]}"
        )

    return app, wb


def get_or_create_sheet(wb, name: str):
    # Get an existing sheet by name or create a new one.
    for i in range(1, wb.Worksheets.Count + 1):
        if wb.Worksheets(i).Name == name:
            return wb.Worksheets(i)

    # Create new sheet at the end
    new_sheet = wb.Worksheets.Add(After=wb.Worksheets(wb.Worksheets.Count))
    new_sheet.Name = name
    return new_sheet


def clear_sheet(ws, keep_rows: int = 0):
    # Clear sheet content below keep_rows.
    used = ws.UsedRange
    if used is None:
        return
    last_row = used.Row + used.Rows.Count - 1
    last_col = used.Column + used.Columns.Count - 1
    if last_row > keep_rows and last_col > 0:
        ws.Range(
            ws.Cells(keep_rows + 1, 1),
            ws.Cells(last_row, last_col)
        ).ClearContents()
        ws.Range(
            ws.Cells(keep_rows + 1, 1),
            ws.Cells(last_row, last_col)
        ).ClearFormats()


def write_status(ws, msg: str, color=None):
    # Write a status message to cell D2 (visible in real-time).
    ws.Cells(2, 4).Value = msg
    if color:
        ws.Cells(2, 4).Font.Color = color
    else:
        ws.Cells(2, 4).Font.Color = BLUE
    ws.Cells(2, 4).Font.Italic = True


def set_cell(ws, row, col, value, bold=False, font_color=None,
             fill_color=None, number_format=None, font_size=None,
             italic=False):
    # Helper to set a cell value with formatting.
    cell = ws.Cells(row, col)
    cell.Value = value
    if bold:
        cell.Font.Bold = True
    if font_color is not None:
        cell.Font.Color = font_color
    if fill_color is not None:
        cell.Interior.Color = fill_color
    if number_format:
        cell.NumberFormat = number_format
    if font_size:
        cell.Font.Size = font_size
    if italic:
        cell.Font.Italic = True


# Sheet writers

def write_analysis_sheet(ws, metrics_summary, metadata, regime_info, ff_results,
                         tickers, backtest_years):
    # Populate the Analysis sheet with dashboard data.
    clear_sheet(ws)

    # Title
    set_cell(ws, 1, 1, "Portfolio Risk Management Tool",
             bold=True, font_size=16, font_color=NAVY)
    set_cell(ws, 2, 1, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             italic=True, font_color=GREY)

    # Input cells
    set_cell(ws, 3, 1, "Tickers:", bold=True)
    set_cell(ws, 3, 2, ", ".join(tickers), fill_color=INPUT_YELLOW)
    set_cell(ws, 4, 1, "Backtest Years:", bold=True)
    set_cell(ws, 4, 2, backtest_years, fill_color=INPUT_YELLOW)

    # Backtest parameters
    row = 6
    set_cell(ws, row, 1, "Backtest Parameters", bold=True, font_size=12, font_color=NAVY)
    row += 1
    params = {
        "Period": f"{metadata.get('start_date', '')} -> {metadata.get('end_date', '')}",
        "Rebalance Freq": metadata.get("rebalance_freq", "M"),
        "Initial Value": f"${metadata.get('initial_value', 100000):,.0f}",
        "# Rebalances": str(metadata.get("n_rebalances", "")),
        "Total Return": f"{metadata.get('total_return', 0):.2%}",
        "Annualised Return": f"{metadata.get('ann_return', 0):.2%}",
        "Annualised Vol": f"{metadata.get('ann_volatility', 0):.2%}",
        "Avg Turnover": f"{metadata.get('avg_turnover', 0):.2%}",
    }
    for k, v in params.items():
        set_cell(ws, row, 1, k, bold=True)
        set_cell(ws, row, 2, str(v))
        row += 1

    # Macro Regime
    row += 1
    set_cell(ws, row, 1, "Current Macro Regime", bold=True, font_size=12, font_color=NAVY)
    row += 1
    set_cell(ws, row, 1, regime_info)
    row += 2

    # Performance metrics table
    set_cell(ws, row, 1, "Performance & Risk Metrics", bold=True, font_size=12, font_color=NAVY)
    row += 1

    # Column headers
    for j, col_name in enumerate(metrics_summary.columns):
        set_cell(ws, row, 1 + j, col_name, bold=True,
                 font_color=WHITE, fill_color=HEADER_FILL)
    row += 1

    # Data rows
    for _, data_row in metrics_summary.iterrows():
        for j, col_name in enumerate(metrics_summary.columns):
            set_cell(ws, row, 1 + j, data_row[col_name])
        row += 1

    # Fama-French
    if ff_results:
        row += 1
        set_cell(ws, row, 1, "Fama-French 3-Factor Regression",
                 bold=True, font_size=12, font_color=NAVY)
        row += 1
        ff_display = {
            "FF3 Alpha (annualised)": f"{ff_results.get('alpha_annual', 0):.4%}",
            "Alpha t-stat": f"{ff_results.get('t_stat_alpha', 0):.3f}",
            "Alpha p-value": f"{ff_results.get('p_value_alpha', 0):.4f}",
            "R-squared": f"{ff_results.get('r_squared', 0):.4f}",
            "# Observations": str(ff_results.get("n_observations", "")),
            "Beta (Mkt-RF)": f"{ff_results.get('beta_mkt', 0):.4f}",
            "Beta (SMB)": f"{ff_results.get('beta_smb', 0):.4f}",
            "Beta (HML)": f"{ff_results.get('beta_hml', 0):.4f}",
        }
        for k, v in ff_display.items():
            set_cell(ws, row, 1, k, bold=True)
            set_cell(ws, row, 2, v)
            row += 1

    # Auto-fit columns
    ws.Columns("A:D").AutoFit()


def write_backtest_sheet(ws, nav, daily_returns, benchmark_returns,
                         weights_history, metadata):
    # Populate the Backtest sheet with NAV and weights data.
    clear_sheet(ws)

    # Title
    set_cell(ws, 1, 1, "Backtest Results - Daily NAV",
             bold=True, font_size=14, font_color=NAVY)

    # NAV table headers
    for j, h in enumerate(["Date", "Portfolio NAV", "Benchmark NAV"]):
        set_cell(ws, 2, 1 + j, h, bold=True, font_color=WHITE, fill_color=HEADER_FILL)

    # NAV data
    bench_nav = (1 + benchmark_returns).cumprod() * metadata.get("initial_value", 100_000)

    for i, (date, value) in enumerate(nav.items()):
        r = 3 + i
        set_cell(ws, r, 1, date.strftime("%Y-%m-%d"))
        set_cell(ws, r, 2, round(float(value), 2), number_format="#,##0.00")
        bench_val = bench_nav.get(date)
        if bench_val is not None and not np.isnan(bench_val):
            set_cell(ws, r, 3, round(float(bench_val), 2), number_format="#,##0.00")

    # Weights history
    if weights_history:
        col_start = 5  # column E
        set_cell(ws, 1, col_start, "Portfolio Weights (at each rebalance)",
                 bold=True, font_size=14, font_color=NAVY)

        all_tickers = sorted(set(t for _, w in weights_history for t in w.keys()))

        # Header
        set_cell(ws, 2, col_start, "Date", bold=True,
                 font_color=WHITE, fill_color=HEADER_FILL)
        for j, ticker in enumerate(all_tickers):
            set_cell(ws, 2, col_start + 1 + j, ticker, bold=True,
                     font_color=WHITE, fill_color=HEADER_FILL)

        # Data
        for i, (date, wts) in enumerate(weights_history):
            r = 3 + i
            if hasattr(date, "strftime"):
                set_cell(ws, r, col_start, date.strftime("%Y-%m-%d"))
            else:
                set_cell(ws, r, col_start, str(date))
            for j, ticker in enumerate(all_tickers):
                val = wts.get(ticker, 0.0)
                set_cell(ws, r, col_start + 1 + j, round(val, 4),
                         number_format="0.00%")

    # Auto-fit
    ws.Columns("A:C").AutoFit()
    if weights_history:
        end_col_letter = chr(ord('E') + len(all_tickers))
        ws.Columns(f"E:{end_col_letter}").AutoFit()


def write_raw_data_sheet(ws, factor_scores, stress_results, mc_summary):
    # Populate the Raw Data sheet with factor scores, stress tests, Monte Carlo.
    clear_sheet(ws)
    row = 1

    # Factor scores 
    if factor_scores is not None and not factor_scores.empty:
        set_cell(ws, row, 1, "Factor Scores (latest rebalance)",
                 bold=True, font_size=14, font_color=NAVY)
        row += 1

        # Headers
        cols = ["Ticker"] + list(factor_scores.columns)
        for j, c in enumerate(cols):
            set_cell(ws, row, 1 + j, c, bold=True,
                     font_color=WHITE, fill_color=HEADER_FILL)
        row += 1

        # Data
        for ticker, data_row in factor_scores.iterrows():
            set_cell(ws, row, 1, str(ticker), bold=True)
            for j, col in enumerate(factor_scores.columns):
                val = float(data_row[col])
                fmt = "0.00%" if col == "Weight" else "0.0000"
                set_cell(ws, row, 2 + j, round(val, 4), number_format=fmt)
            row += 1
        row += 2

    # Stress tests 
    if stress_results:
        set_cell(ws, row, 1, "Stress Test Results",
                 bold=True, font_size=14, font_color=NAVY)
        row += 1

        # Headers
        stress_headers = ["Scenario", "Description", "Portfolio Return",
                          "Dollar Loss", "Worst Ticker"]
        for j, h in enumerate(stress_headers):
            set_cell(ws, row, 1 + j, h, bold=True,
                     font_color=WHITE, fill_color=HEADER_FILL)
        row += 1

        for s in stress_results:
            set_cell(ws, row, 1, s.get("scenario", ""), bold=True)
            set_cell(ws, row, 2, s.get("description", ""))
            set_cell(ws, row, 3, round(s.get("portfolio_return", 0), 4),
                     number_format="0.00%")
            set_cell(ws, row, 4, round(s.get("dollar_loss", 0), 2),
                     number_format="$#,##0")
            set_cell(ws, row, 5, s.get("worst_ticker", ""))
            row += 1
        row += 2

    # Monte Carlo 
    if mc_summary:
        set_cell(ws, row, 1, "Monte Carlo Simulation (1-Year Forward)",
                 bold=True, font_size=14, font_color=NAVY)
        row += 1

        for k, v in mc_summary.items():
            set_cell(ws, row, 1, str(k), bold=True)
            if isinstance(v, (int, float)):
                if abs(v) < 1:
                    set_cell(ws, row, 2, v, number_format="0.00%")
                else:
                    set_cell(ws, row, 2, v, number_format="$#,##0.00")
            else:
                set_cell(ws, row, 2, str(v))
            row += 1

    # Auto-fit
    ws.Columns("A:F").AutoFit()


# Pipeline runner (reuses existing modules)

def run_pipeline(tickers, start_date, end_date):

    from src.backtester import run_backtest
    from src.risk_metrics import (
        compute_all_metrics, get_metrics_summary,
        compute_fama_french_alpha,
    )
    from src.monte_carlo import run_monte_carlo, get_simulation_summary
    from src.stress_test import run_all_scenarios
    from src.data_loader import get_price_data, get_fama_french_factors, get_fred_data
    from src.macro_regime import run_macro_overlay
    from src.factor_model import (
        compute_value_score, compute_momentum_score, compute_quality_score,
    )
    from src.data_loader import get_fundamentals

    results = {}

    # Step 1: Backtest
    print("[1/6] Running backtest...")
    bt = run_backtest(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        rebalance_freq="M",
        initial_value=100_000,
    )
    results["bt"] = bt
    print(f"  Total Return: {bt['metadata']['total_return']:.2%}")

    # Step 2: Risk metrics
    print("[2/6] Computing risk metrics...")
    metrics = compute_all_metrics(bt["daily_returns"], bt["benchmark_returns"])
    summary = get_metrics_summary(metrics)
    results["metrics"] = metrics
    results["summary"] = summary

    # Step 3: Fama-French
    print("[3/6] Fama-French regression...")
    ff_results = None
    try:
        ff = get_fama_french_factors()
        ff_results = compute_fama_french_alpha(bt["daily_returns"], ff)
        print(f"  FF3 Alpha: {ff_results['alpha_annual']:.4%}")
    except Exception as e:
        print(f"  FF3 failed: {e}")
    results["ff_results"] = ff_results

    # Step 4: Macro regime
    print("[4/6] Detecting macro regime...")
    regime_info = "N/A"
    try:
        fred = get_fred_data()
        regime_result = run_macro_overlay(fred_data=fred)
        regime_info = (
            f"{regime_result.get('regime', 'UNKNOWN')} - "
            f"{regime_result.get('description', '')}"
        )
    except Exception as e:
        regime_info = f"Detection failed: {e}"
    results["regime_info"] = regime_info

    # Step 5: Monte Carlo
    print("[5/6] Monte Carlo simulation...")
    mc_sum = None
    try:
        latest_w = _get_latest_weights(bt["weights_history"], tickers)
        prices = get_price_data(tickers, start_date, end_date)
        paths = run_monte_carlo(latest_w, prices, n_simulations=1_000)
        mc_sum = get_simulation_summary(paths, initial_value=100_000)
    except Exception as e:
        print(f"  Monte Carlo failed: {e}")
    results["mc_sum"] = mc_sum

    # Step 6: Stress testing
    print("[6/6] Stress testing...")
    stress = None
    try:
        latest_w = _get_latest_weights(bt["weights_history"], tickers)
        stress = run_all_scenarios(latest_w, initial_value=100_000)
    except Exception as e:
        print(f"  Stress test failed: {e}")
    results["stress"] = stress

    # Factor scores
    print("  Computing factor scores...")
    scores = None
    try:
        fundamentals = get_fundamentals(tickers)
        prices = get_price_data(tickers, start_date, end_date)
        value = compute_value_score(fundamentals)
        momentum = compute_momentum_score(prices)
        quality = compute_quality_score(fundamentals)
        scores = pd.DataFrame({
            "Value": value, "Momentum": momentum, "Quality": quality,
        })
        _, latest_wts = bt["weights_history"][-1] if bt["weights_history"] else (None, {})
        scores["Weight"] = pd.Series(latest_wts)
        scores = scores.fillna(0)
    except Exception as e:
        print(f"  Factor scores failed: {e}")
    results["scores"] = scores
    results["tickers"] = tickers

    return results


def _get_latest_weights(weights_history, tickers):
    """Extract the most recent weights as a pd.Series."""
    if not weights_history:
        n = len(tickers)
        return pd.Series({t: 1.0 / n for t in tickers})
    _, latest_wts = weights_history[-1]
    return pd.Series(latest_wts)



def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Parse arguments
    tickers_str = sys.argv[1] if len(sys.argv) > 1 else "AAPL, MSFT, JPM, JNJ, XOM"
    backtest_years = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    workbook_name = sys.argv[3] if len(sys.argv) > 3 else "PortfolioRiskTool"

    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    if not tickers:
        print("ERROR: No tickers provided")
        sys.exit(1)

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_year = datetime.now().year - backtest_years
    start_date = f"{start_year}-01-01"

    print("=" * 60)
    print("EXCEL BRIDGE - Win32COM Live Writeback")
    print("=" * 60)
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Period:  {start_date} -> {end_date}")
    print(f"  Workbook: {workbook_name}")
    print()

    # Connect to Excel
    app = None
    wb = None
    ws_analysis = None
    use_com = True

    try:
        app, wb = connect_to_excel(workbook_name)
        ws_analysis = get_or_create_sheet(wb, "Analysis")
        print("✓ Connected to Excel via COM")

        # Set lock cell
        ws_analysis.Cells(3, 4).Value = "RUNNING"
        ws_analysis.Cells(3, 4).Font.Color = RED
        write_status(ws_analysis, "Pipeline starting...")

    except Exception as e:
        print(f"⚠ Could not connect to Excel: {e}")
        print("  Will run pipeline and save results via openpyxl instead.")
        use_com = False

    # Run pipeline
    try:
        if use_com:
            write_status(ws_analysis, "Step 1/6: Running backtest...")

        results = run_pipeline(tickers, start_date, end_date)

        if use_com:
            # Write to open Excel via COM
            bt = results["bt"]
            metadata = bt["metadata"]

            write_status(ws_analysis, "Writing results to Analysis sheet...")
            write_analysis_sheet(
                ws=ws_analysis,
                metrics_summary=results["summary"],
                metadata=metadata,
                regime_info=results["regime_info"],
                ff_results=results["ff_results"],
                tickers=tickers,
                backtest_years=backtest_years,
            )

            write_status(ws_analysis, "Writing Backtest sheet...")
            ws_backtest = get_or_create_sheet(wb, "Backtest")
            write_backtest_sheet(
                ws=ws_backtest,
                nav=bt["nav"],
                daily_returns=bt["daily_returns"],
                benchmark_returns=bt["benchmark_returns"],
                weights_history=bt["weights_history"],
                metadata=metadata,
            )

            write_status(ws_analysis, "Writing Raw Data sheet...")
            ws_raw = get_or_create_sheet(wb, "Raw Data")
            write_raw_data_sheet(
                ws=ws_raw,
                factor_scores=results["scores"],
                stress_results=results["stress"],
                mc_summary=results["mc_sum"],
            )

            write_status(ws_analysis,
                         f"✓ Analysis complete - {datetime.now().strftime('%H:%M:%S')} | "
                         f"{len(tickers)} tickers | {metadata['n_rebalances']} rebalances",
                         color=GREEN)

            ws_analysis.Cells(3, 4).Value = ""

            print("\n✓ Results written to Excel successfully!")
            print("  Check all 3 sheets: Analysis, Backtest, Raw Data")

        else:
            # Fallback: write to .xlsx file via openpyxl 
            from src.excel_app import write_results_to_excel
            output = str(Path(PROJECT_ROOT) / "excel" / "PortfolioRiskTool_results.xlsx")
            write_results_to_excel(results, output)
            print(f"\n✓ Results saved to: {output}")
            print("  Open this file in Excel to see all 3 sheets.")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        traceback.print_exc()

        if use_com and ws_analysis:
            try:
                write_status(ws_analysis, f"ERROR: {e}", color=RED)
                ws_analysis.Cells(3, 4).Value = ""  # clear lock
            except Exception:
                pass  # Excel might be gone

        # Try openpyxl fallback
        if "results" in locals() and results.get("bt"):
            try:
                from src.excel_app import write_results_to_excel
                output = str(Path(PROJECT_ROOT) / "excel" / "PortfolioRiskTool_error_fallback.xlsx")
                write_results_to_excel(results, output)
                print(f"  Partial results saved to: {output}")
            except Exception:
                pass

        sys.exit(1)

    finally:
        # Clean up COM
        try:
            import pythoncom
            pythoncom.CoUninitialize()
        except Exception:
            pass

    print("\n" + "=" * 60)
    print("DONE - You can close this window.")
    print("=" * 60)

    # Keep console open so user can see output
    input("\nPress Enter to close...")


if __name__ == "__main__":
    main()
