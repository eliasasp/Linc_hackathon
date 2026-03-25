import numpy as np
import pandas as pd
from pathlib import Path

# --- IMPORTERA DINA FUNKTIONER HÄR ---
from volatility_sisr import run_volatility_filter_on_prices
from correlation_sisr import particle_filter_correlation
from trading_simulator import TradingSimulator

def find_best_pairs(prices_df, n_pairs=5):
    """
    Skannar snabbt igenom all data för att hitta de par med starkast 
    positiv och negativ historisk korrelation.
    """
    print("Skannar marknaden efter de bästa paren...")
    returns_df = np.log(prices_df / prices_df.shift(1)).dropna()
    corr_matrix = returns_df.corr()
    
    pairs = []
    columns = corr_matrix.columns
    
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            asset_A = columns[i]
            asset_B = columns[j]
            corr = corr_matrix.iloc[i, j]
            
            if abs(corr) > 0.5: 
                pairs.append((asset_A, asset_B, corr))
                
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    best_pairs = pairs[:n_pairs]
    
    print("\n--- TOPP-PAR HITTADE ---")
    for a, b, c in best_pairs:
        print(f"{a} & {b} (Korrelation: {c:.2f})")
    print("------------------------\n")
    
    return [(p[0], p[1]) for p in best_pairs]

# ==========================================
# 3. PRE-COMPUTE: SKAPA MÅLPORTFÖLJEN
# ==========================================
def compute_pf_pairs_positions(prices_df):
    target_df = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    
    # Kör skannern bara på aktierna för att hitta de 10 bästa paren
    PAIRS_TO_TRADE = find_best_pairs(prices_df[STOCK_COLS], n_pairs=10)
    
    base_shares = 100 
    
    for asset_A, asset_B in PAIRS_TO_TRADE:
        print(f"Kör partikelfilter för par: {asset_A} & {asset_B}...")

        ret_A = np.log(prices_df[asset_A] / prices_df[asset_A].shift(1)).dropna().values
        ret_B = np.log(prices_df[asset_B] / prices_df[asset_B].shift(1)).dropna().values
        
        # Kör de tunga partikelfiltren
        vol_A = run_volatility_filter_on_prices(prices_df[asset_A])
        vol_B = run_volatility_filter_on_prices(prices_df[asset_B])
        dyn_corr = particle_filter_correlation(ret_A, ret_B, vol_A, vol_B)
        
        # Känn av om paret i snitt rör sig med eller emot varandra
        mean_corr = np.mean(dyn_corr)
        is_positive_corr = mean_corr > 0
        
        if is_positive_corr:
            spread = np.log(prices_df[asset_A]) - np.log(prices_df[asset_B])
        else:
            spread = np.log(prices_df[asset_A]) + np.log(prices_df[asset_B])
            
        z_score = (spread - spread.rolling(20).mean()) / spread.rolling(20).std()
        
        positions_A = np.zeros(len(prices_df))
        positions_B = np.zeros(len(prices_df))
        current_pos = 0
        
        for i in range(20, len(prices_df)):
            z = z_score.iloc[i]
            corr = dyn_corr[i-1] 
            
            # Entry-signaler
            if current_pos == 0 and abs(corr) > 0.6:
                if z < -2.0: current_pos = 1   
                elif z > 2.0: current_pos = -1 
            
            # Exit-signaler
            elif current_pos != 0 and abs(z) <= 0.5:
                current_pos = 0 
                
            # Tilldela positioner beroende på korrelationsriktning
            if is_positive_corr:
                positions_A[i] = current_pos
                positions_B[i] = -current_pos
            else:
                positions_A[i] = current_pos
                positions_B[i] = current_pos
            
        target_df[asset_A] += positions_A * base_shares
        target_df[asset_B] += positions_B * base_shares
        
    return target_df

target_shares_df = compute_pf_pairs_positions(prices)

# ==========================================
# 4. STRATEGI & ORDERHANTERING
# ==========================================

def strategy(row_pos, cash, portfolio, signal_prices, data):
    orders = []
    date = data.index[row_pos]

    if date not in target_shares_df.index: return orders

    targets = target_shares_df.loc[date]
    if targets.isna().all(): return orders

    for ticker in STOCK_COLS:
        tgt = targets.get(ticker)
        if pd.isna(tgt): continue
        tgt = int(round(tgt))
        
        current_pos = portfolio.get(ticker, 0)
        delta = tgt - current_pos
        
        if delta != 0:
            action = 'BUY' if delta > 0 else 'SELL'
            orders.append((action, ticker, abs(delta)))

    # (FX hedging utelämnat för tillfället, men kan läggas in här om ni vill valutasäkra)
    return orders

# ==========================================
# 5. KÖR SIMULATORN
# ==========================================
simulator = TradingSimulator(
    assets              = list(prices.columns),
    initial_cash        = 100_000,
    equity_currency_map = equity_ccy,
    fx_pairs_map        = fx_pairs_map,
)

print("\nKör simulering...")
simulator.run(strategy, prices, prices)

print("Sparar resultat och plottar...")
simulator.save_results(orders_file="orders.csv", portfolio_file="portfolio.csv")
simulator.plot_performance(prices, save_file="performance_plot.png")