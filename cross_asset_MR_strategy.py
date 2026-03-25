import numpy as np
import pandas as pd
from pathlib import Path
from trading_simulator import TradingSimulator
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster

# ===== LOAD DATA =====
prices = pd.read_csv("prices.csv", index_col="Date", parse_dates=True)

print(f"Loaded {len(prices)} trading days, {len(prices.columns)} assets")

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

# ===== PARAMETERS =====
CORR_WINDOW   = 20
CORR_THRESH   = 0.75

ZSCORE_WINDOW = 20
ENTRY_Z       = 2
EXIT_Z        = 0

TOP_N_PAIRS   = 10
N_CLUSTERS    = 4

VOL_WINDOW    = 100
RISK_WINDOW   = 100


# ===== CLUSTER FUNCTION =====
def compute_clusters(returns_window):
    corr = returns_window.corr()
    dist = 1 - corr

    dist_condensed = dist.values[np.triu_indices_from(dist, k=1)]
    Z = linkage(dist_condensed, method='average')

    labels = fcluster(Z, t=N_CLUSTERS, criterion='maxclust')
    return dict(zip(corr.index, labels))


# ===== STRATEGY =====
def compute_positions(prices):
    returns = prices.pct_change()
    assets  = list(prices.columns)
    n       = len(assets)

    positions = pd.DataFrame(0.0, index=prices.index, columns=assets)

    rolling_corr = returns.rolling(CORR_WINDOW).corr()
    log_prices   = np.log(prices)

    active_trades = {}

    for t, date in enumerate(prices.index):

        if t < max(CORR_WINDOW, ZSCORE_WINDOW):
            continue

        # ===== COMPUTE CLUSTERS (rolling) =====
        cluster_map = compute_clusters(
            returns.iloc[t-CORR_WINDOW:t]
        )

        new_active_trades = {}
        pair_signals = []

        for i in range(n):
            for j in range(i + 1, n):

                asset_i = assets[i]
                asset_j = assets[j]
                pair_id = (asset_i, asset_j)

                # ===== CLUSTER FILTER =====
                if cluster_map.get(asset_i) != cluster_map.get(asset_j):
                    continue

                try:
                    corr = rolling_corr.loc[date].loc[asset_i, asset_j]
                except:
                    continue

                if pd.isna(corr) or corr < CORR_THRESH:
                    continue

                spread = log_prices[asset_i] - log_prices[asset_j]
                window_spread = spread.iloc[t - ZSCORE_WINDOW:t]

                if window_spread.isna().any():
                    continue

                mean = window_spread.mean()
                std  = window_spread.std()

                if std == 0 or pd.isna(std):
                    continue

                zscore = (spread.iloc[t] - mean) / std

                # ===== HANDLE EXISTING TRADE =====
                if pair_id in active_trades:
                    direction = active_trades[pair_id]["direction"]
                    weight    = active_trades[pair_id]["weight"]

                    # EXIT
                    if abs(zscore) < EXIT_Z:
                        continue

                    # HOLD
                    new_active_trades[pair_id] = {
                        "direction": direction,
                        "weight": weight
                    }

                    if direction == 1:
                        positions.loc[date, asset_i] += weight
                        positions.loc[date, asset_j] -= weight
                    else:
                        positions.loc[date, asset_i] -= weight
                        positions.loc[date, asset_j] += weight

                    continue

                # ===== NEW SIGNAL =====
                if abs(zscore) > ENTRY_Z:
                    pair_signals.append({
                        "pair": pair_id,
                        "asset_i": asset_i,
                        "asset_j": asset_j,
                        "zscore": zscore,
                        "corr": corr
                    })

        # ===== SELECT TOP-N =====
        pair_signals = sorted(
            pair_signals,
            key=lambda x: abs(x["zscore"]),
            reverse=True
        )

        selected = pair_signals[:TOP_N_PAIRS]

        # ===== OPEN NEW TRADES =====
        for signal in selected:
            pair_id = signal["pair"]
            asset_i = signal["asset_i"]
            asset_j = signal["asset_j"]
            zscore  = signal["zscore"]
            corr    = signal["corr"]

            weight = corr ** 2

            if zscore > 0:
                direction = -1
                positions.loc[date, asset_i] -= weight
                positions.loc[date, asset_j] += weight
            else:
                direction = 1
                positions.loc[date, asset_i] += weight
                positions.loc[date, asset_j] -= weight

            new_active_trades[pair_id] = {
                "direction": direction,
                "weight": weight
            }

        active_trades = new_active_trades

    # ===== LAG =====
    positions = positions.shift(1)

    # ===== VOL SCALING =====
    volatility = np.sqrt(
        (returns ** 2).rolling(
            VOL_WINDOW, min_periods=VOL_WINDOW // 2
        ).sum()
    )

    vol_scaled_pos = positions / volatility.replace(0, np.nan)

    # ===== PORTFOLIO RISK =====
    portfolio_daily_pnl = (
        vol_scaled_pos.shift(2) * returns
    ).dropna(how='all').sum(axis=1)

    portfolio_risk = portfolio_daily_pnl.rolling(
        RISK_WINDOW, min_periods=20
    ).std()

    final_positions = vol_scaled_pos.div(portfolio_risk, axis=0)

    return final_positions


# ===== PRECOMPUTE =====
target_shares_df = compute_positions(prices)


# ===== EXECUTION =====
def strategy(row_pos, cash, portfolio, signal_prices, data):
    orders = []
    date   = data.index[row_pos]

    if date not in target_shares_df.index:
        return orders

    targets = target_shares_df.loc[date]

    if targets.isna().all():
        return orders

    for ticker in targets.index:
        tgt = targets.get(ticker)

        if pd.isna(tgt):
            continue

        tgt   = int(round(tgt))
        delta = tgt - portfolio.get(ticker, 0)

        if delta != 0:
            orders.append(
                ('BUY' if delta > 0 else 'SELL', ticker, abs(delta))
            )

    return orders


# ===== RUN =====
simulator = TradingSimulator(
    assets              = list(prices.columns),
    initial_cash        = 100_000,
    equity_currency_map = equity_ccy,
    fx_pairs_map        = fx_pairs_map,
)

simulator.run(strategy, prices, prices)

simulator.save_results(
    orders_file    = "orders.csv",
    portfolio_file = "portfolio.csv",
)

simulator.plot_performance(
    prices,
    save_file="performance_plot.png"
)


