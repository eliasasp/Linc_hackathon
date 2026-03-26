import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

def plot_pairs_trade(prices_file, orders_file, asset1, asset2, window=60, z_entry=1.4, z_exit=0.5):
    print(f"Ritar upp Pairs Trade-analys för {asset1} och {asset2}...")
    
    # 1. Ladda data
    prices_df = pd.read_csv(prices_file, index_col='Date', parse_dates=True)
    try:
        orders_df = pd.read_csv(orders_file, parse_dates=['Date'])
    except FileNotFoundError:
        print(f"Kunde inte hitta {orders_file}. Kör backtestet först!")
        return

    # Hantera kolumnnamn (Ticker vs Asset)
    col_ticker = 'Ticker' if 'Ticker' in orders_df.columns else 'Asset'
    
    # 2. Beräkna den historiska spreaden och Z-score (exakt som i algoritmen)
    p1 = prices_df[asset1]
    p2 = prices_df[asset2]
    
    # Log-spread
    spread = np.log(p1) - np.log(p2)
    
    # Rullande Z-score
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    z_score = (spread - rolling_mean) / rolling_std

    # 3. Hitta de dagar då BÅDA tillgångarna handlades (våra entry/exits)
    o1 = orders_df[orders_df[col_ticker] == asset1]
    o2 = orders_df[orders_df[col_ticker] == asset2]
    
    # Slå ihop dem på datum för att hitta par-transaktionerna
    pair_trades = pd.merge(o1, o2, on='Date', suffixes=('_1', '_2'))
    
    # -- RITA GRAFERNA --
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # --- ÖVRE GRAFEN: Normaliserade priser ---
    # Vi delar med första giltiga priset så att båda börjar på 1.0 (enklare att jämföra)
    ax1.plot(prices_df.index, p1 / p1.iloc[window], label=f'{asset1}', color='blue', linewidth=1.5)
    ax1.plot(prices_df.index, p2 / p2.iloc[window], label=f'{asset2}', color='orange', linewidth=1.5)
    ax1.set_title(f'Pairs Trading Analys: {asset1} vs {asset2}', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Normaliserat Pris', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- UNDRE GRAFEN: Z-score och signaler ---
    ax2.plot(prices_df.index, z_score, label='Z-score (Spread)', color='purple', linewidth=1.5)
    
    # Rita ut våra trigger-linjer
    ax2.axhline(z_entry, color='red', linestyle='--', alpha=0.6, label=f'Entry (+{z_entry})')
    ax2.axhline(-z_entry, color='green', linestyle='--', alpha=0.6, label=f'Entry (-{z_entry})')
    ax2.axhline(z_exit, color='gray', linestyle=':', alpha=0.6, label=f'Exit (+{z_exit})')
    ax2.axhline(-z_exit, color='gray', linestyle=':', alpha=0.6, label=f'Exit (-{z_exit})')
    ax2.axhline(0, color='black', linewidth=1)

    # Lägg ut markörer för tradesen
    for _, trade in pair_trades.iterrows():
        date = trade['Date']
        act1 = trade['Action_1']
        
        # Hämta Z-scoret för just denna dag
        z_val = z_score.loc[date]
        
        # Logik för att färga pilarna:
        if act1 == 'SELL' and z_val > 1.0:
            # Vi blankade Asset 1 för att Z var högt (Övervärderad)
            ax2.scatter(date, z_val, color='red', marker='v', s=150, zorder=5, edgecolors='black')
        elif act1 == 'BUY' and z_val < -1.0:
            # Vi köpte Asset 1 för att Z var lågt (Undervärderad)
            ax2.scatter(date, z_val, color='green', marker='^', s=150, zorder=5, edgecolors='black')
        else:
            # Om Z är nära 0, är det en stängning (Exit)
            ax2.scatter(date, z_val, color='yellow', marker='X', s=150, zorder=5, edgecolors='black')

    ax2.set_ylabel('Z-Score', fontsize=12)
    ax2.set_xlabel('Datum', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()