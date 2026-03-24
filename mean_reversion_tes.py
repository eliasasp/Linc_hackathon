import numpy as np
import pandas as pd
from trading_simulator import TradingSimulator

# ===== LOAD DATA =====
prices = pd.read_csv("prices.csv", index_col="Date", parse_dates=True)

BENCHMARK = "Idx_01"
OTHER     = "Idx_03"

print(f"Loaded {len(prices)} trading days | Pair: {BENCHMARK} / {OTHER}")

# ===== ASSET METADATA =====
equity_ccy = {
    'Stock_01': 'Crncy_03', 'Stock_02': 'Crncy_04', 'Stock_03': 'Crncy_04',
    'Stock_04': 'Crncy_02', 'Stock_05': 'Crncy_03', 'Stock_06': 'Crncy_02',
    'Stock_07': 'Crncy_03', 'Stock_08': 'Crncy_02', 'Stock_09': 'Crncy_04',
    'Stock_10': 'Crncy_03', 'Stock_11': 'Crncy_01', 'Stock_12': 'Crncy_04',
    'Stock_13': 'Crncy_01', 'Stock_14': 'Crncy_01', 'Stock_15': 'Crncy_01',
}

fx_pairs_map = {
    'FX_01': ('Crncy_02', 'Crncy_01'),
    'FX_02': ('Crncy_04', 'Crncy_02'),
    'FX_03': ('Crncy_04', 'Crncy_03'),
    'FX_04': ('Crncy_02', 'Crncy_03'),
    'FX_05': ('Crncy_01', 'Crncy_03'),
    'FX_06': ('Crncy_04', 'Crncy_01'),
}

# ===== STRATEGY PARAMETERS =====
ROLL_WIN       = 30
Z_ENTRY        = 1.0
Z_EXIT         = 0.5
VOL_WINDOW     = 30       # lookback for realised volatility
VOL_TARGET     = 0.01     # target 1% daily vol per leg
MAX_NOTIONAL   = 100_000  # cap per leg to avoid oversizing in low-vol regimes

# ===== PRECOMPUTE SIGNALS =====
def rolling_beta(y, x, window):
    betas = pd.Series(index=y.index, dtype=float)
    for i in range(window, len(y)):
        yi = y.iloc[i-window:i].values
        xi = x.iloc[i-window:i].values
        X  = np.column_stack([np.ones(window), xi])
        b  = np.linalg.lstsq(X, yi, rcond=None)[0]
        betas.iloc[i] = b[1]
    return betas

benchmark = prices[BENCHMARK]
other     = prices[OTHER]

beta   = rolling_beta(benchmark, other, ROLL_WIN)
spread = benchmark - beta * other
mu     = spread.rolling(ROLL_WIN).mean()
sigma  = spread.rolling(ROLL_WIN).std()
z      = (spread - mu) / sigma.replace(0, np.nan)

# Realised daily volatility (annualised not needed — we use daily vol directly)
ret_bench = benchmark.pct_change()
ret_other = other.pct_change()
vol_bench = ret_bench.rolling(VOL_WINDOW).std()
vol_other = ret_other.rolling(VOL_WINDOW).std()

# ===== STATE =====
signal = 0  # +1 = long Idx_01 / short Idx_03, -1 = short Idx_01 / long Idx_03

def strategy(row_pos, cash, portfolio, signal_prices, data):
    global signal
    orders = []
    date   = data.index[row_pos]

    if date not in z.index or pd.isna(z[date]) or pd.isna(beta[date]):
        return orders

    z_val = z[date]
    prev  = signal

    # Entry / exit
    if prev == 0:
        if z_val < -Z_ENTRY:
            signal =  1
        elif z_val > Z_ENTRY:
            signal = -1
    elif prev ==  1 and z_val > -Z_EXIT:
        signal = 0
    elif prev == -1 and z_val <  Z_EXIT:
        signal = 0

    if signal == prev:
        return orders

    # Volatility-scaled position sizing:
    # shares = VOL_TARGET / daily_vol / price  → each leg risks VOL_TARGET of portfolio per day
    vb = vol_bench[date]
    vo = vol_other[date]

    if pd.isna(vb) or pd.isna(vo) or vb == 0 or vo == 0:
        return orders

    bench_notional = min(VOL_TARGET / vb * signal_prices[BENCHMARK], MAX_NOTIONAL)
    other_notional = min(VOL_TARGET / vo * signal_prices[OTHER],     MAX_NOTIONAL)

    bench_tgt = int(bench_notional / signal_prices[BENCHMARK] * signal)
    other_tgt = int(-other_notional / signal_prices[OTHER] * beta[date] * signal)

    for ticker, tgt in [(BENCHMARK, bench_tgt), (OTHER, other_tgt)]:
        curr  = portfolio.get(ticker, 0)
        delta = tgt - curr
        if delta != 0:
            orders.append(('BUY' if delta > 0 else 'SELL', ticker, abs(delta)))

    return orders

# ===== RUN SIMULATION =====
simulator = TradingSimulator(
    assets              = list(prices.columns),
    initial_cash        = 100_000,
    equity_currency_map = equity_ccy,
    fx_pairs_map        = fx_pairs_map,
)
simulator.run(strategy, prices, prices)
simulator.save_results(
    orders_file    = "orders_meanrev.csv",
    portfolio_file = "portfolio_meanrev.csv",
)


"""
test to variate the position sizes depending on the volatility
increase amount of winning bets

Regime filter:
Only trade when the spread is actually mean-reverting, not trending. Compute the half-life of the spread over a rolling window and skip trading if it's above e.g. 30 days — that signals the relationship has broken down temporarily
Alternatively, only trade when the rolling correlation between Idx_01 and Idx_03 is above a threshold (e.g. 0.7), since low correlation periods are when pair trades blow up

Think more about risk, can't have the same bet size
"""