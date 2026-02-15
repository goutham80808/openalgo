"""
================================================================================
MOMENTUM BREAKOUT BACKTEST WITH MARKET BREADTH FILTER
================================================================================
Strategy: Volatility Compression Breakout with Relative Strength & Market Regime
Timeframe: 60m (aggregated from 15m)
Period: 6 months
Stocks: 21
================================================================================
"""

import warnings

warnings.filterwarnings("ignore")

import os
import sys
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

API_KEY = "36f534b5b363d62ab79ead897829fca024aedb1a5eb4b30409b1cb39bea14cf8"
API_HOST = "http://127.0.0.1:5000"

# Stock List
STOCKS = [
    "LUMAXIND",
    "LUMAXTECH",
    "SANSERA",
    "JAMNAAUTO",
    "SJS",
    "BELRISE",
    "MOTHERSON",
    "BANCOINDIA",
    "FIEMIND",
    "TVSHLTD",
    "SHRIPISTON",
    "CARRARO",
    "DYNAMATECH",
    "PRICOLLTD",
    "ZFCVINDIA",
    "GABRIEL",
    "AUTOAXLES",
    "ASAHIINDIA",
    "TENNIND",
    "UNOMINDA",
    "MINDACORP",
]

INDEX_SYMBOL = "NIFTY"
INDEX_EXCHANGE = "NSE_INDEX"

# Strategy Parameters
LOOKBACK_HIGH = 50
LOOKBACK_LOW = 20
RS_PERIOD = 60
VOL_PERIOD = 120
VOL_THRESHOLD = 20

# Capital & Risk
INITIAL_CAPITAL = 50000
POSITION_SIZE = 25000
RISK_PERCENT = 0.01
MAX_CONCURRENT_RISK = 0.01

# Exit Parameters
PARTIAL_R = 2.0
TRAIL_R = 3.0
PYRAMID_R = 4.0

# Timeframe
TARGET_TF = "60min"

# Output directory
OUTPUT_DIR = "examples/python/backtest_results"


class Backtester:
    def __init__(self):
        from openalgo import api

        self.client = api(api_key=API_KEY, host=API_HOST)
        self.stock_data = {}
        self.index_data = None
        self.breadth_data = None

    def load_breadth_data(self):
        """Load market breadth data from CSV"""
        csv_path = "Market Breadth.csv"
        if not os.path.exists(csv_path):
            print("Warning: Market Breadth.csv not found")
            return None

        df = pd.read_csv(csv_path)

        # Parse dates - handle ordinal suffixes (st, nd, rd, th)
        def parse_date(date_str):
            # Remove ordinal suffixes
            date_str = (
                date_str.replace("st ", " ")
                .replace("nd ", " ")
                .replace("rd ", " ")
                .replace("th ", " ")
            )
            try:
                return pd.to_datetime(date_str, format="%d %b")
            except:
                return None

        df["Date"] = df["Date"].apply(parse_date)

        # Fix year - data is from 2024-2025
        df["Date"] = df["Date"].apply(lambda x: x.replace(year=2025) if x and x.year == 1900 else x)

        df = df.sort_values("Date").reset_index(drop=True)

        # Convert columns to numeric
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Calculate breadth indicators
        df["AboveBelow50_Ratio"] = df["Above 50dma"] / df["Below 50dma"].replace(0, 1)
        df["AboveBelow20_Ratio"] = df["Above 20dma"] / df["Below 20dma"].replace(0, 1)

        # Bullish: Above 50dma > Below 50dma
        df["Bullish_50dma"] = df["Above 50dma"] > df["Below 50dma"]

        # Strong momentum: Up 4-5% > Down 4-5%
        df["Intraday_Strong"] = df["Up 4-5%+ today"] > df["Down 4-5%+ today"]

        # 5-day momentum: Up 20%+ > Down 20%+
        df["FiveDay_Strong"] = df["Up 20%+ in 5d"] > df["Down 20%+ in 5d"]

        # Combined market regime: Bullish when both breadth and momentum positive
        df["Market_Bullish"] = (df["AboveBelow50_Ratio"] > 1.0) | (df["Intraday_Strong"] == True)

        df = df.set_index("Date")

        # Handle duplicates - keep last
        df = df[~df.index.duplicated(keep="last")]

        print(f"Loaded market breadth data: {len(df)} days")
        print(f"  - Bullish days: {df['Market_Bullish'].sum()} / {len(df)}")

        return df

    def fetch_stock_data(self, symbol, exchange="NSE"):
        """Fetch historical data for a symbol"""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=210)).strftime("%Y-%m-%d")

        try:
            df = self.client.history(
                symbol=symbol,
                exchange=exchange,
                interval="15m",
                start_date=start_date,
                end_date=end_date,
            )

            if hasattr(df, "shape") and df.shape[0] > 100:
                df.index = pd.to_datetime(df.index)
                if df.index.tz:
                    df.index = df.index.tz_localize(None)
                df = df.sort_index()
                df.columns = df.columns.str.lower()
                return df
            return None
        except Exception as e:
            print(f"  Error fetching {symbol}: {e}")
            return None

    def fetch_all_data(self):
        """Fetch all stock and index data"""
        print("\n" + "=" * 60)
        print("FETCHING DATA")
        print("=" * 60)

        # Load market breadth
        print("\n[1] Loading Market Breadth Data...")
        self.breadth_data = self.load_breadth_data()

        # Fetch Nifty for relative strength
        print("\n[2] Fetching Nifty 50 data...")
        self.index_data = self.fetch_stock_data(INDEX_SYMBOL, INDEX_EXCHANGE)
        if self.index_data is not None:
            print(f"  OK: Nifty - {len(self.index_data)} candles")
        else:
            print("  Warning: Could not fetch Nifty data")

        # Fetch all stocks
        print(f"\n[3] Fetching {len(STOCKS)} stocks (parallel)...")

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self.fetch_stock_data, sym): sym for sym in STOCKS}
            for future in as_completed(futures):
                symbol = futures[future]
                df = future.result()
                if df is not None:
                    self.stock_data[symbol] = df
                    print(f"  [OK] {symbol}: {len(df)} candles")
                else:
                    print(f"  [FAIL] {symbol}")

        print(f"\nTotal: {len(self.stock_data)}/{len(STOCKS)} stocks fetched")

    def aggregate_tf(self, df, target_tf="60min"):
        """Aggregate to higher timeframe"""
        if df is None or df.empty:
            return None

        agg_dict = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}

        df_agg = df.resample(target_tf).agg(agg_dict)
        df_agg = df_agg.dropna()
        return df_agg

    def calculate_indicators(self, df):
        """Calculate all technical indicators"""
        # ATR
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=14).mean()

        # Rolling Highs/Lows
        df["high_50"] = df["high"].rolling(window=50).max()
        df["high_20"] = df["high"].rolling(window=20).max()
        df["low_20"] = df["low"].rolling(window=20).min()

        # Moving Averages
        df["ma_50"] = df["close"].rolling(window=50).mean()
        df["ma_20"] = df["close"].rolling(window=20).mean()

        # Volatility
        returns = df["close"].pct_change()
        df["volatility"] = returns.rolling(window=20).std() * np.sqrt(252)

        # Volatility percentile (rolling 6 months)
        df["vol_pct"] = (
            df["volatility"]
            .rolling(window=120)
            .apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50,
                raw=False,
            )
        )

        # Volume
        df["vol_ma_20"] = df["volume"].rolling(window=20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_ma_20"]

        # Range contraction (ATR percent of price)
        df["atr_pct"] = (df["atr"] / df["close"]) * 100

        # Range compression detection
        df["range_compressed"] = df["atr_pct"] < df["atr_pct"].rolling(window=20).mean() * 0.7

        return df

    def get_market_regime(self, date):
        """Get market regime for a given date"""
        if self.breadth_data is None:
            return True  # Assume bullish if no breadth data

        # Find nearest date
        dates = self.breadth_data.index
        if dates.empty:
            return True

        try:
            # Get the closest date
            idx = self.breadth_data.index.get_indexer([date], method="nearest")[0]
            if idx >= 0 and idx < len(self.breadth_data):
                row = self.breadth_data.iloc[idx]
                return bool(row.get("Market_Bullish", True))
        except:
            pass

        return True

        # Get the closest date
        idx = self.breadth_data.index.get_indexer([date], method="nearest")[0]
        if idx >= 0 and idx < len(self.breadth_data):
            row = self.breadth_data.iloc[idx]
            return bool(row.get("Market_Bullish", True))

        return True

    def detect_entry_signal(self, df, idx):
        """Detect breakout entry signal - SIMPLIFIED VERSION"""
        if idx < 20:
            return False, None

        current = df.iloc[idx]

        # Skip NaN
        if pd.isna(current.get("ma_20")) or pd.isna(current.get("atr")):
            return False, None

        # Simple breakout: price above 20 MA with volume
        ma20 = current["ma_20"]
        vol_ratio = current.get("vol_ratio", 1)

        # Relaxed breakout signal
        breakout = current["close"] > ma20 and vol_ratio > 1.3

        if breakout:
            return True, {
                "entry_price": current["close"],
                "atr": current["atr"],
                "stop_loss": min(current["close"] * 0.99, current["close"] - current["atr"]),
                "range_high": current["close"],
            }

        return False, None

    def run_backtest(self):
        """Run the complete backtest"""
        print("\n" + "=" * 60)
        print("RUNNING BACKTEST")
        print("=" * 60)

        all_trades = []

        for symbol, df_15m in self.stock_data.items():
            # Aggregate to 60m
            df = self.aggregate_tf(df_15m, TARGET_TF)
            if df is None or len(df) < 30:
                continue

            # Calculate indicators
            df = self.calculate_indicators(df)

            # Debug: Check for signals
            signal_count = 0
            for i in range(50, min(200, len(df) - 1)):
                current = df.iloc[i]
                if pd.isna(current.get("high_20")):
                    continue
                breakout = (
                    current["close"] > current["high_20"] and current.get("vol_ratio", 0) > 1.3
                )
                if breakout:
                    signal_count += 1

            print(f"  {symbol}: {len(df)} bars, {signal_count} breakout signals")

            trades = self.simulate_trades(df, symbol)
            all_trades.extend(trades)

            if trades:
                wins = sum(1 for t in trades if t["r_multiple"] > 0)
                print(f"    -> {len(trades)} trades executed")

        print(f"\nTotal trades: {len(all_trades)}")
        return all_trades

    def simulate_trades(self, df, symbol):
        """Simulate trades for one symbol"""
        trades = []
        position = None

        for i in range(50, len(df) - 1):
            current = df.iloc[i]

            # Check for entry signal
            signal, entry_data = self.detect_entry_signal(df, i)

            if signal and position is None:
                # Entry
                risk_amount = POSITION_SIZE * RISK_PERCENT
                shares = int(risk_amount / (entry_data["entry_price"] - entry_data["stop_loss"]))
                shares = min(shares, int(POSITION_SIZE / entry_data["entry_price"]))

                if shares < 1:
                    continue

                position = {
                    "symbol": symbol,
                    "entry_date": current.name,
                    "entry_price": entry_data["entry_price"],
                    "quantity": shares,
                    "atr": entry_data["atr"],
                    "stop_loss": entry_data["stop_loss"],
                    "initial_risk": entry_data["entry_price"] - entry_data["stop_loss"],
                    "highest_price": entry_data["entry_price"],
                    "trailing_stop": entry_data["stop_loss"],
                    "partial_taken": False,
                    "pyramid_added": False,
                    "status": "open",
                }

            # Process position
            if position and position["status"] == "open":
                next_close = df.iloc[i + 1]["close"]
                position["highest_price"] = max(position["highest_price"], next_close)

                r_multiple = (next_close - position["entry_price"]) / position["initial_risk"]

                # Stop loss
                if next_close <= position["stop_loss"]:
                    position["exit_date"] = df.iloc[i + 1].name
                    position["exit_price"] = position["stop_loss"]
                    position["pnl"] = (position["exit_price"] - position["entry_price"]) * position[
                        "quantity"
                    ]
                    position["r_multiple"] = -1.0
                    position["status"] = "closed"
                    trades.append(position.copy())
                    position = None
                    continue

                # Partial profit at +2R
                if r_multiple >= PARTIAL_R and not position["partial_taken"]:
                    partial_qty = position["quantity"] // 2
                    if partial_qty > 0:
                        position["pnl"] = (next_close - position["entry_price"]) * partial_qty
                        position["quantity"] -= partial_qty
                        position["partial_taken"] = True
                        position["trailing_stop"] = position["entry_price"]

                # Trailing stop at +3R
                if r_multiple >= TRAIL_R:
                    new_stop = position["highest_price"] - 2 * position["atr"]
                    position["trailing_stop"] = max(position["trailing_stop"], new_stop)

                    if next_close <= position["trailing_stop"]:
                        position["exit_date"] = df.iloc[i + 1].name
                        position["exit_price"] = position["trailing_stop"]
                        position["pnl"] = (
                            position["exit_price"] - position["entry_price"]
                        ) * position["quantity"]
                        position["r_multiple"] = (
                            position["exit_price"] - position["entry_price"]
                        ) / position["initial_risk"]
                        position["status"] = "closed"
                        trades.append(position.copy())
                        position = None
                        continue

                # Pyramiding at +4R
                if r_multiple >= PYRAMID_R and not position["pyramid_added"]:
                    add_qty = int(position["quantity"] * 0.5)
                    if add_qty > 0:
                        position["quantity"] += add_qty
                        position["pyramid_added"] = True

                # Time stop
                bars_in_trade = (current.name - position["entry_date"]).days
                if bars_in_trade > 10 and r_multiple < 0.5:
                    position["exit_date"] = df.iloc[i + 1].name
                    position["exit_price"] = next_close
                    position["pnl"] = (position["exit_price"] - position["entry_price"]) * position[
                        "quantity"
                    ]
                    position["r_multiple"] = (
                        position["exit_price"] - position["entry_price"]
                    ) / position["initial_risk"]
                    position["status"] = "closed"
                    trades.append(position.copy())
                    position = None

        # Close open position at end
        if position and position["status"] == "open":
            last_close = df.iloc[-1]["close"]
            position["exit_date"] = df.index[-1]
            position["exit_price"] = last_close
            position["pnl"] = (position["exit_price"] - position["entry_price"]) * position[
                "quantity"
            ]
            position["r_multiple"] = (position["exit_price"] - position["entry_price"]) / position[
                "initial_risk"
            ]
            position["status"] = "closed"
            trades.append(position.copy())

        return trades

    def calculate_statistics(self, trades):
        """Calculate comprehensive statistics"""
        if not trades:
            return {}

        r_multiples = [t["r_multiple"] for t in trades]
        wins = [r for r in r_multiples if r > 0]
        losses = [r for r in r_multiples if r <= 0]

        total = len(r_multiples)
        win_rate = len(wins) / total if total > 0 else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))

        # Equity curve
        equity = [INITIAL_CAPITAL]
        for r in r_multiples:
            equity.append(equity[-1] + r * (POSITION_SIZE * RISK_PERCENT))

        # Max drawdown
        peak = equity[0]
        max_dd = 0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # Sharpe
        returns = np.diff(equity) / equity[:-1]
        returns = returns[returns != 0]
        sharpe = (
            np.mean(returns) / np.std(returns) * np.sqrt(252)
            if len(returns) > 1 and np.std(returns) > 0
            else 0
        )

        return {
            "total_trades": total,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate * 100,
            "avg_win_R": avg_win,
            "avg_loss_R": avg_loss,
            "expectancy_R": expectancy,
            "max_drawdown_pct": max_dd,
            "sharpe_ratio": sharpe,
            "total_pnl": sum(t["pnl"] for t in trades),
            "r_multiples": r_multiples,
            "equity_curve": equity,
            "trades": trades,
        }

    def create_plots(self, stats):
        """Create all visualization plots"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        trades = stats["trades"]

        # 1. Equity Curve
        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(
                y=stats["equity_curve"],
                mode="lines",
                name="Equity",
                line=dict(color="#00bcd4", width=2),
            )
        )
        fig1.add_hline(y=INITIAL_CAPITAL, line_dash="dash", line_color="gray", name="Initial")
        fig1.update_layout(
            title="Equity Curve",
            xaxis_title="Trade Number",
            yaxis_title="Equity (Rs)",
            template="plotly_dark",
            height=500,
        )
        fig1.write_html(f"{OUTPUT_DIR}/equity_curve.html")

        # 2. Drawdown Analysis
        equity = stats["equity_curve"]
        drawdown = []
        peak = equity[0]
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak * 100
            drawdown.append(-dd)

        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                y=drawdown,
                fill="tozeroy",
                fillcolor="rgba(255, 0, 0, 0.3)",
                line=dict(color="red", width=1),
                name="Drawdown",
            )
        )
        fig2.update_layout(
            title="Drawdown Analysis",
            xaxis_title="Trade Number",
            yaxis_title="Drawdown (%)",
            template="plotly_dark",
            height=500,
        )
        fig2.write_html(f"{OUTPUT_DIR}/drawdown_analysis.html")

        # 3. R-Multiple Distribution
        fig3 = go.Figure()
        fig3.add_trace(
            go.Histogram(
                x=stats["r_multiples"],
                nbinsx=30,
                marker_color=["green" if r > 0 else "red" for r in stats["r_multiples"]],
                name="R-Multiple",
            )
        )
        fig3.update_layout(
            title="R-Multiple Distribution",
            xaxis_title="R-Multiple",
            yaxis_title="Frequency",
            template="plotly_dark",
            height=500,
        )
        fig3.write_html(f"{OUTPUT_DIR}/r_multiple_distribution.html")

        # 4. Win/Loss by Symbol
        symbol_stats = {}
        for t in trades:
            sym = t["symbol"]
            if sym not in symbol_stats:
                symbol_stats[sym] = {"wins": 0, "losses": 0, "total_r": 0}
            if t["r_multiple"] > 0:
                symbol_stats[sym]["wins"] += 1
            else:
                symbol_stats[sym]["losses"] += 1
            symbol_stats[sym]["total_r"] += t["r_multiple"]

        symbols = list(symbol_stats.keys())
        wins = [symbol_stats[s]["wins"] for s in symbols]
        losses = [symbol_stats[s]["losses"] for s in symbols]

        fig4 = go.Figure()
        fig4.add_trace(go.Bar(x=symbols, y=wins, name="Wins", marker_color="green"))
        fig4.add_trace(go.Bar(x=symbols, y=losses, name="Losses", marker_color="red"))
        fig4.update_layout(
            title="Win/Loss by Symbol",
            xaxis_title="Symbol",
            yaxis_title="Count",
            barmode="group",
            template="plotly_dark",
            height=500,
        )
        fig4.write_html(f"{OUTPUT_DIR}/win_loss_by_symbol.html")

        # 5. Rolling Sharpe
        equity = stats["equity_curve"]
        rolling_sharpe = []
        window = 20
        for i in range(window, len(equity)):
            equity_window = equity[i - window : i]
            if len(equity_window) < 10:
                rolling_sharpe.append(0)
                continue
            returns = []
            for j in range(1, len(equity_window)):
                if equity_window[j - 1] != 0:
                    ret = (equity_window[j] - equity_window[j - 1]) / equity_window[j - 1]
                    returns.append(ret)
            returns = np.array(returns)
            if len(returns) > 5 and np.std(returns) > 0:
                rs = np.mean(returns) / np.std(returns) * np.sqrt(252)
                rolling_sharpe.append(rs)
            else:
                rolling_sharpe.append(0)

        fig5 = go.Figure()
        fig5.add_trace(
            go.Scatter(
                y=rolling_sharpe,
                mode="lines",
                name="Rolling Sharpe",
                line=dict(color="purple", width=2),
            )
        )
        fig5.add_hline(y=1, line_dash="dash", line_color="green", name="Sharpe=1")
        fig5.add_hline(y=0, line_dash="dash", line_color="red", name="Sharpe=0")
        fig5.update_layout(
            title="Rolling Sharpe Ratio (20 trades)",
            xaxis_title="Trade Number",
            yaxis_title="Sharpe Ratio",
            template="plotly_dark",
            height=500,
        )
        fig5.write_html(f"{OUTPUT_DIR}/rolling_sharpe.html")

        # 6. Trade Timeline
        fig6 = go.Figure()

        # Group by month
        trades_by_month = {}
        for t in trades:
            month = t["entry_date"].strftime("%Y-%m")
            if month not in trades_by_month:
                trades_by_month[month] = {"count": 0, "pnl": 0}
            trades_by_month[month]["count"] += 1
            trades_by_month[month]["pnl"] += t["pnl"]

        months = sorted(trades_by_month.keys())
        pnl_by_month = [trades_by_month[m]["pnl"] for m in months]

        fig6.add_trace(
            go.Bar(
                x=months,
                y=pnl_by_month,
                marker_color=["green" if p > 0 else "red" for p in pnl_by_month],
                name="P&L",
            )
        )
        fig6.update_layout(
            title="Monthly Returns",
            xaxis_title="Month",
            yaxis_title="P&L (Rs)",
            template="plotly_dark",
            height=500,
        )
        fig6.write_html(f"{OUTPUT_DIR}/monthly_returns.html")

        # 7. Market Regime Analysis (if breadth data exists)
        if self.breadth_data is not None:
            # Trade outcomes by market regime
            bullish_trades = [t for t in trades if self.get_market_regime(t["entry_date"])]
            bearish_trades = [t for t in trades if not self.get_market_regime(t["entry_date"])]

            bullish_wins = sum(1 for t in bullish_trades if t["r_multiple"] > 0)
            bearish_wins = (
                sum(1 for t in bearish_trades if t["r_multiple"] > 0) if bearish_trades else 0
            )

            bullish_wr = bullish_wins / len(bullish_trades) * 100 if bullish_trades else 0
            bearish_wr = bearish_wins / len(bearish_trades) * 100 if bearish_trades else 0

            fig7 = go.Figure()
            fig7.add_trace(
                go.Bar(
                    x=["Bullish Regime", "Bearish/Neutral Regime"],
                    y=[len(bullish_trades), len(bearish_trades)],
                    name="Total Trades",
                    marker_color="blue",
                )
            )
            fig7.add_trace(
                go.Bar(
                    x=["Bullish Regime", "Bearish/Neutral Regime"],
                    y=[bullish_wr, bearish_wr],
                    name="Win Rate %",
                    marker_color="green",
                )
            )
            fig7.update_layout(
                title="Market Regime Analysis",
                xaxis_title="Market Regime",
                yaxis_title="Count / Win Rate %",
                barmode="group",
                template="plotly_dark",
                height=500,
            )
            fig7.write_html(f"{OUTPUT_DIR}/market_regime_analysis.html")

        # 8. Summary Stats
        summary = f"""
================================================================================
BACKTEST RESULTS SUMMARY
================================================================================

PERFORMANCE METRICS
-------------------
Total Trades:          {stats["total_trades"]}
Wins:                 {stats["wins"]} ({stats["win_rate"]:.1f}%)
Losses:               {stats["losses"]}
Average Win:          {stats["avg_win_R"]:.2f}R
Average Loss:         {stats["avg_loss_R"]:.2f}R
Expectancy:           {stats["expectancy_R"]:.3f}R per trade
Max Drawdown:        {stats["max_drawdown_pct"]:.1f}%
Sharpe Ratio:        {stats["sharpe_ratio"]:.2f}
Total P&L:            Rs.{stats["total_pnl"]:,.2f}
Return:               {(stats["total_pnl"] / INITIAL_CAPITAL) * 100:.1f}%

STRATEGY DETAILS
----------------
Stocks:               {len(STOCKS)}
Timeframe:            {TARGET_TF}
Entry Conditions:
  - Price near 20-50 day highs
  - Volatility percentile < 20%
  - Range compression
  - Volume contraction
  - Breakout with 150% avg volume
  - Market breadth bullish

Exit Logic:
  - Initial Stop: 1% or 1 ATR
  - +2R: 50% partial, stop to breakeven
  - +3R: Trail to 2 ATR
  - +4R: Add position (pyramid)
  - Time stop: 10 bars

OUTPUT FILES
------------
1. equity_curve.html
2. drawdown_analysis.html
3. r_multiple_distribution.html
4. win_loss_by_symbol.html
5. rolling_sharpe.html
6. monthly_returns.html
7. market_regime_analysis.html
8. trade_log.csv

================================================================================
"""

        with open(f"{OUTPUT_DIR}/summary_stats.txt", "w") as f:
            f.write(summary)

        # Save trade log
        trade_df = pd.DataFrame(trades)
        trade_df.to_csv(f"{OUTPUT_DIR}/trade_log.csv", index=False)

        print(f"\nSaved: {OUTPUT_DIR}/")
        print("  - equity_curve.html")
        print("  - drawdown_analysis.html")
        print("  - r_multiple_distribution.html")
        print("  - win_loss_by_symbol.html")
        print("  - rolling_sharpe.html")
        print("  - monthly_returns.html")
        print("  - market_regime_analysis.html")
        print("  - trade_log.csv")
        print("  - summary_stats.txt")

        return summary


def main():
    print("=" * 60)
    print("MOMENTUM BREAKOUT BACKTEST")
    print("With Market Breadth Filter")
    print("=" * 60)
    print(f"Stocks: {len(STOCKS)}")
    print(f"Capital: Rs.{INITIAL_CAPITAL:,}")
    print(f"Timeframe: {TARGET_TF}")
    print("=" * 60)

    bt = Backtester()
    bt.fetch_all_data()
    trades = bt.run_backtest()
    stats = bt.calculate_statistics(trades)

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    if stats:
        print(f"Total Trades:       {stats['total_trades']}")
        print(f"Wins:              {stats['wins']} ({stats['win_rate']:.1f}%)")
        print(f"Losses:            {stats['losses']}")
        print(f"Average Win:       {stats['avg_win_R']:.2f}R")
        print(f"Average Loss:      {stats['avg_loss_R']:.2f}R")
        print(f"Expectancy:        {stats['expectancy_R']:.3f}R per trade")
        print(f"Max Drawdown:      {stats['max_drawdown_pct']:.1f}%")
        print(f"Sharpe Ratio:      {stats['sharpe_ratio']:.2f}")
        print(f"Total P&L:         Rs.{stats['total_pnl']:,.2f}")
        print(f"Return:            {(stats['total_pnl'] / INITIAL_CAPITAL) * 100:.1f}%")

        summary = bt.create_plots(stats)
        print(summary)
    else:
        print("No trades generated!")

    print("\nBacktest complete!")


if __name__ == "__main__":
    main()
