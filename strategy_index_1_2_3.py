import numpy as np
import pandas as pd
from trading_simulator import TradingSimulator

# ===== LOAD DATA =====
prices = pd.read_csv("prices.csv", index_col="Date", parse_dates=True)

BENCHMARK = "Idx_01"
OTHERS    = ["Idx_02", "Idx_03"]

print(f"Loaded {len(prices)} trading days | Pairs: {BENCHMARK} vs {OTHERS}")

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
ROLL_WIN      = 30
Z_ENTRY       = 1.5
Z_EXIT        = 0.75
POSITION_SIZE = 100

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

pair_data = {}
for other in OTHERS:
    x      = prices[other]
    beta   = rolling_beta(benchmark, x, ROLL_WIN)
    spread = benchmark - beta * x
    mu     = spread.rolling(ROLL_WIN).mean()
    sigma  = spread.rolling(ROLL_WIN).std()
    z      = (spread - mu) / sigma.replace(0, np.nan)
    pair_data[other] = {"beta": beta, "z": z}

# ===== STATE =====
signals = {other: 0 for other in OTHERS}

def strategy(row_pos, cash, portfolio, signal_prices, data):
    orders = []
    date   = data.index[row_pos]

    desired = {BENCHMARK: 0, **{o: 0 for o in OTHERS}}

    for other in OTHERS:
        z    = pair_data[other]["z"]
        beta = pair_data[other]["beta"]

        if date not in z.index or pd.isna(z[date]) or pd.isna(beta[date]):
            continue

        z_val = z[date]
        prev  = signals[other]
        new   = prev

        if prev == 0:
            if z_val < -Z_ENTRY:
                new =  1
            elif z_val > Z_ENTRY:
                new = -1
        elif prev ==  1 and z_val > -Z_EXIT:
            new = 0
        elif prev == -1 and z_val <  Z_EXIT:
            new = 0

        signals[other] = new

        bench_tgt = int(POSITION_SIZE * new)
        other_tgt = int(-POSITION_SIZE * beta[date] * new)

        desired[BENCHMARK] += bench_tgt
        desired[other]     += other_tgt

    for ticker, tgt in desired.items():
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
simulator.plot_performance(prices, save_file="meanrev_1.5,0.75.png")