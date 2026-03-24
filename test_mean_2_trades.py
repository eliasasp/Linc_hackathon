import numpy as np
import pandas as pd
from trading_simulator import TradingSimulator

# ===== LOAD =====
prices = pd.read_csv("prices.csv", index_col="Date", parse_dates=True)

BENCHMARK = "Idx_01"
OTHER     = "Idx_03"

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

traded = False

def strategy(row_pos, cash, portfolio, signal_prices, data):
    global traded
    orders = []

    if row_pos == 0 and not traded:
        orders.append(('BUY',  OTHER,     100))
        orders.append(('SELL', BENCHMARK, 100))
        traded = True

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
    orders_file    = "orders_direction_test.csv",
    portfolio_file = "portfolio_direction_test.csv",
)
simulator.plot_performance(prices, save_file="performance_direction_test.png")