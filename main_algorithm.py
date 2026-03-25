import pandas as pd
import numpy as np
from trading_simulator import TradingSimulator
from volatility_sisr import IncrementalVolatilityFilter, estimate_sv_parameters
from commodities_trading import advanced_commodity_strategy # Din bevisat fungerande Schwartz-modell

# ==========================================
# 1. LADDA PRISDATA OCH HITTA ALLA TILLGÅNGAR
# ==========================================
print("Laddar prisdata...")
prices = pd.read_csv("prices.csv", index_col="Date", parse_dates=True)

# Vi väljer nu ALLA kolumner som börjar på 'Idx_' och 'Comm_'
ASSETS_TO_TRADE = [c for c in prices.columns if c.startswith("Idx_") or c.startswith("Comm_")]

print(f"Hittade totalt {len(ASSETS_TO_TRADE)} tillgångar att handla:")
print(f"Index: {[c for c in ASSETS_TO_TRADE if 'Idx' in c]}")
print(f"Råvaror: {[c for c in ASSETS_TO_TRADE if 'Comm' in c]}")

# ==========================================
# 2. PRE-KALKYLERA FILTER & HISTORIK (Ditt state)
# ==========================================
print("\nInitierar partikelfilter och historik för varje tillgång...")
my_filters = {}
my_history = {}

for ticker in ASSETS_TO_TRADE:
    historik = prices[ticker].dropna()
    returns = np.log(historik / historik.shift(1)).dropna().values
    
    # Uppskatta parametrar (SISR-partikelfilter)
    mu, phi, sigma_eta = estimate_sv_parameters(returns)
    my_filters[ticker] = IncrementalVolatilityFilter(mu, phi, sigma_eta, n_particles=1000)
    
    # Initiera tom historik
    my_history[ticker] = []

# Paketera data
strategy_data = {
    'filters': my_filters,
    'history': my_history
}

# ==========================================
# 3. KÖR BACKTESTING-SIMULATORN
# ==========================================
print("\nStartar backtest på hela portföljen...")
simulator = TradingSimulator(
    assets       = ASSETS_TO_TRADE,
    initial_cash = 100_000 # Du kan överväga att öka kassan om du handlar väldigt många tillgångar
)

# Vi kör din "aggressiva" Schwartz-modell som gav Sharpe 0.75
simulator.run(advanced_commodity_strategy, prices, strategy_data)

# Spara resultat
simulator.save_results(
    orders_file    = "portfolio_orders.csv",
    portfolio_file = "portfolio_performance.csv",
)

# Visa dashboarden
simulator.plot_performance(prices, save_file="full_portfolio_plot.png")