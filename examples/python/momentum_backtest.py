"""
Momentum Breakout Backtest with R-Multiple Exit Strategy
=========================================================
Strategy: Volatility compression breakout with relative strength filter
Timeframe: 15m (aggregated to 60m for signals)
Period: 6 months
"""

import warnings

warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openalgo import api

API_KEY = "36f534b5b363d62ab79ead897829fca024aedb1a5eb4b30409b1cb39bea14cf8"
API_HOST = "http://127.0.0.1:5000"

STOCKS = [
    "LUMAXIND",
    "LUMAXTECH",
    "SANSERA",
    "JAMNAAUTO",
    "BELRISE",
    "MOTHERSON",
    "FIEMIND",
    "TVSHLTD",
    "PRICOLLTD",
    "ZFCVINDIA",
    "GABRIEL",
    "MINDACORP",
]

INDEX_SYMBOL = "NIFTY"  # Will use Nifty 50 for relative strength
INDEX_EXCHANGE = "NSE_INDEX"  # Exchange for Nifty

# Strategy Parameters
LOOKBACK_HIGH = 50  # Days for high
LOOKBACK_LOW = 20  # Days for low
RS_PERIOD = 60  # 3 months for relative strength
VOL_PERIOD = 120  # 6 months for volatility percentile
VOL_THRESHOLD = 20  # Bottom 20% volatility

# Capital & Risk
INITIAL_CAPITAL = 50000
POSITION_SIZE = 25000
RISK_PERCENT = 0.01  # 1% of position
MAX_CONCURRENT_RISK = 0.01  # 1% equity max

# Exit Parameters
PARTIAL_R = 2.0  # Take partial at +2R
TRAIL_R = 3.0  # Trail stop at +3R
PYRAMID_R = 4.0  # Add position at +4R

client = api(api_key=API_KEY, host=API_HOST)


def fetch_stock_data(symbol, interval="15m", days=210, exchange="NSE"):
    """Fetch historical data for a symbol"""
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    try:
        df = client.history(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
        )

        if hasattr(df, "shape") and df.shape[0] > 100:
            df.index = pd.to_datetime(df.index)
            if df.index.tz:
                df.index = df.index.tz_localize(None)
            df = df.sort_index()
            df.columns = df.columns.str.lower()
            return symbol, df
        else:
            return symbol, None
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return symbol, None


def aggregate_to_higher_tf(df, target_tf="60min"):
    """Aggregate 15m data to higher timeframe"""
    if df is None or df.empty:
        return None

    agg_dict = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}

    # Resample to target timeframe
    df_hf = df.resample(target_tf).agg(agg_dict)
    df_hf = df_hf.dropna()
    return df_hf


def calculate_atr(df, period=14):
    """Calculate ATR"""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr


def calculate_volatility_percentile(df, period=20):
    """Calculate rolling volatility percentile over 6 months"""
    returns = df["close"].pct_change()
    volatility = returns.rolling(window=period).std() * np.sqrt(252)  # Annualized

    # Rolling 6-month percentile
    vol_percentile = volatility.rolling(window=VOL_PERIOD).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )

    return vol_percentile


def calculate_highs_lows(df, lookback_high=50, lookback_low=20):
    """Calculate rolling highs and lows"""
    highs = df["high"].rolling(window=lookback_high).max()
    lows = df["low"].rolling(window=lookback_low).min()
    return highs, lows


def calculate_relative_strength(stock_df, index_df, period=60):
    """Calculate relative strength vs index"""
    if stock_df is None or index_df is None:
        return None

    # Align dates
    common_dates = stock_df.index.intersection(index_df.index)
    if len(common_dates) < period:
        return None

    stock_ret = stock_df.loc[common_dates, "close"].pct_change()
    index_ret = index_df.loc[common_dates, "close"].pct_change()

    rs = (1 + stock_ret).rolling(window=period).apply(lambda x: (1 + x).prod() - 1, raw=False) - (
        1 + index_ret
    ).rolling(window=period).apply(lambda x: (1 + x).prod() - 1, raw=False)

    return rs


def calculate_volume_contraction(df, lookback=20):
    """Detect volume contraction (low volume during base)"""
    avg_volume = df["volume"].rolling(window=lookback).mean()
    volume_ratio = df["volume"] / avg_volume
    return volume_ratio < 0.7  # Bottom 30% volume


def detect_breakout(df, idx, highs, lows, vol_pct, rs, vol_contraction):
    """Detect if current bar is a breakout setup - SIMPLIFIED VERSION"""
    if idx < 20:
        return False

    current = df.iloc[idx]
    prev = df.iloc[idx - 1]

    # Simple breakout: price breaking 20-day high with volume
    range_high = highs.iloc[idx]
    range_low = lows.iloc[idx]

    # Near 20-day high (relaxed from 50-day)
    near_high = current["close"] >= highs.iloc[idx] * 0.95

    # Positive relative strength (or neutral)
    rs_positive = rs.iloc[idx] > -0.05 if rs is not None else True

    # Low volatility (relaxed)
    low_vol = vol_pct.iloc[idx] <= 40 if vol_pct is not None else True

    # Breakout: close above 20-day high with volume
    breakout = (
        current["close"] > range_high
        and current["volume"] > df["volume"].iloc[idx - 10 : idx].mean() * 1.2
    )

    return near_high and rs_positive and breakout


class Trade:
    def __init__(self, symbol, entry_date, entry_price, quantity, atr, stop_loss):
        self.symbol = symbol
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.quantity = quantity
        self.atr = atr
        self.stop_loss = stop_loss
        self.initial_risk = abs(entry_price - stop_loss)

        # Position tracking
        self.positions = [
            {"entry_price": entry_price, "quantity": quantity, "entry_date": entry_date}
        ]

        # Exit tracking
        self.exit_date = None
        self.exit_price = None
        self.pnl = 0
        self.r_multiple = 0
        self.status = "open"  # open, closed
        self.partial_taken = False
        self.pyramid_added = False

        # For trailing stop
        self.highest_price = entry_price
        self.trailing_stop = stop_loss


def run_backtest(stock_df, index_df, symbol):
    """Run backtest on a single stock"""
    if stock_df is None or stock_df.empty or len(stock_df) < 100:
        return []

    # Aggregate to 60m for signals
    df = aggregate_to_higher_tf(stock_df, "60min")
    if df is None or len(df) < 50:
        return []

    # Calculate indicators - simplified
    df["atr"] = calculate_atr(df).fillna(df["close"] * 0.02)
    df["highs"], df["lows"] = calculate_highs_lows(df, 20, 10)
    df["rs"] = 0  # Skip RS calculation for now

    trades = []
    open_trades = []

    # Calculate simple MA for trend
    df["ma20"] = df["close"].rolling(window=20).mean()

    trades = []
    open_trades = []

    for i in range(20, len(df) - 1):
        current = df.iloc[i]

        # Skip if missing data
        if pd.isna(current["atr"]) or pd.isna(current["highs"]):
            continue

        # Simple breakout detection - relaxed to find more signals
        ma20 = df["ma20"].iloc[i]

        # Breakout: close above 20-day MA with volume
        vol_ma = df["volume"].iloc[max(0, i - 10) : i].mean()

        # Signal: price breaks above 20-day MA with volume - momentum entry
        if current["close"] > ma20 and current["volume"] > vol_ma * 1.5:
            # Check if we already have a position in this stock
            if any(t.symbol == symbol and t.status == "open" for t in open_trades):
                continue

        # Simple breakout detection - use MA
        ma20 = df["ma20"].iloc[i]

        # Breakout: close above 20-day MA with volume
        vol_ma = df["volume"].iloc[max(0, i - 10) : i].mean()

        if current["close"] > ma20 and current["volume"] > vol_ma * 1.5:
            # Check if we already have a position in this stock
            if any(t.symbol == symbol and t.status == "open" for t in open_trades):
                continue

            # Entry
            entry_price = current["close"]
            atr = current["atr"]

            # Stop: 1% or 1 ATR
            stop_loss = min(entry_price * 0.99, entry_price - atr)

            # Calculate position size
            risk_amount = POSITION_SIZE * RISK_PERCENT  # 250
            shares = int(risk_amount / (entry_price - stop_loss))
            shares = min(shares, int(POSITION_SIZE / entry_price))

            if shares < 1:
                continue

            trade = Trade(
                symbol=symbol,
                entry_date=current.name,
                entry_price=entry_price,
                quantity=shares,
                atr=atr,
                stop_loss=stop_loss,
            )
            open_trades.append(trade)

        # Process open trades
        for trade in open_trades[:]:
            if trade.status == "closed":
                continue

            current_price = current["close"]
            trade.highest_price = max(trade.highest_price, current_price)

            # Calculate current P&L in R
            price_change = current_price - trade.entry_price
            r_multiple = price_change / trade.initial_risk if trade.initial_risk > 0 else 0

            # Check stop loss
            if current_price <= trade.stop_loss:
                trade.exit_date = current.name
                trade.exit_price = trade.stop_loss
                trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity
                trade.r_multiple = -1.0
                trade.status = "closed"
                trades.append(trade)
                open_trades.remove(trade)
                continue

            # Check partial profit at +2R
            if r_multiple >= PARTIAL_R and not trade.partial_taken:
                # Sell half
                partial_qty = trade.quantity // 2
                if partial_qty > 0:
                    trade.pnl += (current_price - trade.entry_price) * partial_qty
                    trade.quantity -= partial_qty
                    trade.partial_taken = True
                    # Move stop to breakeven
                    trade.trailing_stop = trade.entry_price

            # Update trailing stop at +3R
            if r_multiple >= TRAIL_R:
                # Trail to 2 ATR or prior swing low
                new_stop = trade.highest_price - 2 * trade.atr
                trade.trailing_stop = max(trade.trailing_stop, new_stop)

                if current_price <= trade.trailing_stop:
                    trade.exit_date = current.name
                    trade.exit_price = trade.trailing_stop
                    trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity
                    trade.r_multiple = (trade.exit_price - trade.entry_price) / trade.initial_risk
                    trade.status = "closed"
                    trades.append(trade)
                    open_trades.remove(trade)
                    continue

            # Pyramiding at +4R (only if not already added)
            if r_multiple >= PYRAMID_R and not trade.pyramid_added:
                # Add 50% more position using locked-in profit
                add_qty = int(trade.quantity * 0.5)
                if add_qty > 0:
                    trade.positions.append(
                        {
                            "entry_price": current_price,
                            "quantity": add_qty,
                            "entry_date": current.name,
                        }
                    )
                    trade.quantity += add_qty
                    trade.pyramid_added = True

            # Time stop: exit if no expansion after 10 bars
            bars_in_trade = (current.name - trade.entry_date).days
            if bars_in_trade > 10 and r_multiple < 0.5:
                trade.exit_date = current.name
                trade.exit_price = current_price
                trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity
                trade.r_multiple = (trade.exit_price - trade.entry_price) / trade.initial_risk
                trade.status = "closed"
                trades.append(trade)
                open_trades.remove(trade)

    # Close remaining trades at last price
    if len(df) > 0:
        last_price = df.iloc[-1]["close"]
        for trade in open_trades:
            trade.exit_date = df.index[-1]
            trade.exit_price = last_price
            trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity
            trade.r_multiple = (trade.exit_price - trade.entry_price) / trade.initial_risk
            trade.status = "closed"
            trades.append(trade)

    return trades


def calculate_statistics(all_trades):
    """Calculate comprehensive statistics"""
    if not all_trades:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "avg_win_R": 0,
            "avg_loss_R": 0,
            "expectancy_R": 0,
            "max_drawdown_pct": 0,
            "risk_of_ruin": 0,
            "sharpe_ratio": 0,
            "total_pnl": 0,
            "r_multiples": [],
            "equity_curve": [INITIAL_CAPITAL],
            "trades": [],
        }

    r_multiples = [t.r_multiple for t in all_trades]
    wins = [r for r in r_multiples if r > 0]
    losses = [r for r in r_multiples if r <= 0]

    total_trades = len(r_multiples)
    win_rate = len(wins) / total_trades if total_trades > 0 else 0

    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    # Expectancy
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))

    # Max drawdown (simple)
    equity = INITIAL_CAPITAL
    equity_curve = [equity]
    for r in r_multiples:
        equity += r * (POSITION_SIZE * RISK_PERCENT)
        equity_curve.append(equity)

    peak = equity_curve[0]
    max_dd = 0
    for e in equity_curve:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Risk of ruin (simplified)
    consecutive_losses = 0
    max_consecutive = 0
    for r in r_multiples:
        if r < 0:
            consecutive_losses += 1
            max_consecutive = max(max_consecutive, consecutive_losses)
        else:
            consecutive_losses = 0

    # Approximate risk of ruin
    if avg_loss != 0 and win_rate > 0:
        loss_prob = (INITIAL_CAPITAL * MAX_CONCURRENT_RISK) / (
            abs(avg_loss) * POSITION_SIZE * RISK_PERCENT
        )
        risk_of_ruin = (loss_prob ** (1 / win_rate)) if win_rate > 0 else 1
    else:
        risk_of_ruin = 0

    # Sharpe-like ratio
    returns = np.diff(equity_curve) / equity_curve[:-1]
    returns = returns[returns != 0]
    sharpe = (
        np.mean(returns) / np.std(returns) * np.sqrt(252)
        if len(returns) > 1 and np.std(returns) > 0
        else 0
    )

    return {
        "total_trades": total_trades,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate * 100,
        "avg_win_R": avg_win,
        "avg_loss_R": avg_loss,
        "expectancy_R": expectancy,
        "max_drawdown_pct": max_dd,
        "risk_of_ruin": min(risk_of_ruin, 100),
        "sharpe_ratio": sharpe,
        "total_pnl": sum(t.pnl for t in all_trades),
        "r_multiples": r_multiples,
        "equity_curve": equity_curve,
        "trades": all_trades,
    }


def plot_results(stats, symbol_stats):
    """Create comprehensive plots"""

    # R-Multiple Distribution
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "R-Multiple Distribution",
            "Equity Curve",
            "Win/Loss by Symbol",
            "Cumulative Returns",
        ],
        specs=[
            [{"type": "histogram"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "scatter"}],
        ],
    )

    # Histogram of R-multiples
    fig.add_trace(
        go.Histogram(
            x=stats["r_multiples"], nbinsx=30, name="R-Multiples", marker_color="steelblue"
        ),
        row=1,
        col=1,
    )

    # Equity curve
    fig.add_trace(
        go.Scatter(
            y=stats["equity_curve"], mode="lines", name="Equity", line=dict(color="green", width=2)
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=INITIAL_CAPITAL, line_dash="dash", line_color="gray", row=1, col=2)

    # Win/Loss by symbol
    symbol_pnl = {}
    for symbol, trades in symbol_stats.items():
        wins = sum(1 for t in trades if t.r_multiple > 0)
        losses = len(trades) - wins
        symbol_pnl[symbol] = {"wins": wins, "losses": losses}

    symbols = list(symbol_pnl.keys())[:15]  # Top 15
    fig.add_trace(
        go.Bar(
            x=symbols, y=[symbol_pnl[s]["wins"] for s in symbols], name="Wins", marker_color="green"
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=symbols,
            y=[symbol_pnl[s]["losses"] for s in symbols],
            name="Losses",
            marker_color="red",
        ),
        row=2,
        col=1,
    )

    # Cumulative returns
    sorted_returns = sorted(stats["r_multiples"])
    cumulative = np.cumsum(sorted_returns)
    fig.add_trace(
        go.Scatter(
            y=cumulative, mode="lines", name="Cumulative R", line=dict(color="purple", width=2)
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=900, width=1400, title_text="Momentum Breakout Backtest Results", showlegend=True
    )

    fig.write_html("backtest_results.html")
    print("Saved: backtest_results.html")

    # Individual symbol performance
    fig2 = go.Figure()
    symbol_returns = {}
    for symbol, trades in symbol_stats.items():
        total_r = sum(t.r_multiple for t in trades)
        symbol_returns[symbol] = total_r

    sorted_sym = sorted(symbol_returns.items(), key=lambda x: x[1], reverse=True)

    fig2.add_trace(
        go.Bar(
            x=[s[0] for s in sorted_sym],
            y=[s[1] for s in sorted_sym],
            marker_color=["green" if s[1] > 0 else "red" for s in sorted_sym],
        )
    )

    fig2.update_layout(
        title="Total R-Multiple by Symbol",
        xaxis_title="Symbol",
        yaxis_title="Total R",
        height=500,
        width=1400,
    )

    fig2.write_html("symbol_performance.html")
    print("Saved: symbol_performance.html")


def main():
    print("=" * 60)
    print("MOMENTUM BREAKOUT BACKTEST")
    print("=" * 60)
    print(f"Stocks: {len(STOCKS)}")
    print(f"Period: 6 months")
    print(f"Capital: Rs.{INITIAL_CAPITAL:,}")
    print(f"Position Size: Rs.{POSITION_SIZE:,}")
    print(f"Risk per Trade: {RISK_PERCENT * 100}%")
    print("=" * 60)

    # Fetch Nifty 50 data first (for relative strength)
    print("\n[1/3] Fetching Nifty 50 data...")
    index_symbol, index_df = fetch_stock_data(INDEX_SYMBOL, exchange=INDEX_EXCHANGE)
    if index_df is None:
        print("Warning: Could not fetch Nifty data, using fallback")
        index_df = None
    else:
        print(f"  Nifty fetched: {len(index_df)} candles")

    # Fetch all stock data in parallel
    print("\n[2/3] Fetching stock data (parallel)...")
    stock_data = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_stock_data, symbol): symbol for symbol in STOCKS}
        for future in as_completed(futures):
            symbol, df = future.result()
            if df is not None:
                stock_data[symbol] = df
                print(f"  [OK] {symbol}: {len(df)} candles")
            else:
                print(f"  [X] {symbol}: Failed")

    print(f"\nSuccessfully fetched: {len(stock_data)}/{len(STOCKS)} stocks")

    # Run backtest on each stock
    print("\n[3/3] Running backtests...")
    all_trades = []
    symbol_stats = {}

    for symbol, df in stock_data.items():
        trades = run_backtest(df, index_df, symbol)
        if trades:
            all_trades.extend(trades)
            symbol_stats[symbol] = trades
            wins = sum(1 for t in trades if t.r_multiple > 0)
            print(f"  {symbol}: {len(trades)} trades, {wins} wins")

    print(f"\nTotal trades: {len(all_trades)}")

    # Calculate statistics
    stats = calculate_statistics(all_trades)

    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Total Trades:       {stats['total_trades']}")
    print(f"Wins:               {stats['wins']} ({stats['win_rate']:.1f}%)")
    print(f"Losses:             {stats['losses']}")
    print(f"Average Win:        {stats['avg_win_R']:.2f}R")
    print(f"Average Loss:       {stats['avg_loss_R']:.2f}R")
    print(f"Expectancy:         {stats['expectancy_R']:.3f}R per trade")
    print(f"Max Drawdown:       {stats['max_drawdown_pct']:.1f}%")
    print(f"Risk of Ruin:      {stats['risk_of_ruin']:.1f}%")
    print(f"Sharpe Ratio:      {stats['sharpe_ratio']:.2f}")
    print(f"Total P&L:          Rs.{stats['total_pnl']:,.2f}")
    print("=" * 60)

    # Save detailed trade log
    if all_trades:
        trade_df = pd.DataFrame(
            [
                {
                    "Symbol": t.symbol,
                    "Entry Date": t.entry_date,
                    "Entry Price": t.entry_price,
                    "Exit Date": t.exit_date,
                    "Exit Price": t.exit_price,
                    "Quantity": t.quantity,
                    "P&L": t.pnl,
                    "R-Multiple": t.r_multiple,
                    "Partial Taken": t.partial_taken,
                    "Pyramid Added": t.pyramid_added,
                }
                for t in all_trades
            ]
        )
        trade_df.to_csv("trade_log.csv", index=False)
        print("Saved: trade_log.csv")

    # Create plots
    print("\nGenerating plots...")
    plot_results(stats, symbol_stats)

    print("\n[OK] Backtest complete!")


if __name__ == "__main__":
    main()
