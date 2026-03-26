import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_debug_dashboard():
    print("Laddar in simulatorns resultat för felsökning...")
    
    # 1. Ladda data
    prices = pd.read_csv('prices.csv', index_col='Date', parse_dates=True)
    
    # Byt ut namnet här om er output-fil heter något annat
    portfolio = pd.read_csv('momentum_portfolio.csv', index_col='Date', parse_dates=True)
    
    # Hitta vilka kolumner som faktiskt är tillgångar (filtrera bort 'Cash' eller 'Total' om de finns)
    asset_cols = [c for c in portfolio.columns if c in prices.columns]
    
    # 2. Räkna ut det faktiska värdet i kronor på alla positioner (Antal * Dagens Pris)
    pos_values = portfolio[asset_cols] * prices[asset_cols]
    pos_values = pos_values.fillna(0)
    
    # Totalt investerat värde på marknaden
    total_invested = pos_values.sum(axis=1)
    
    # Hitta den tillgång vi har investerat överlägset mest i för att detaljstudera den
    most_held_asset = pos_values.sum(axis=0).idxmax()

    # ==========================================
    # BYGG GRAFEN (3 Subplots)
    # ==========================================
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # --- GRAF 1: KAPITALALLOKERING (Stacked Area) ---
    # Filtrera bort tillgångar vi aldrig rört så legend:en inte blir för lång
    active_assets = pos_values.columns[(pos_values > 0).any()]
    
    axes[0].stackplot(pos_values.index, pos_values[active_assets].T, labels=active_assets)
    axes[0].set_title('1. Vilka tillgångar ligger algoritmen i? (Kapitalallokering)', fontweight='bold')
    axes[0].set_ylabel('Investerat Värde (SEK)')
    axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small', title="Aktiva Innehav")
    axes[0].grid(True, alpha=0.3)

    # --- GRAF 2: RISK-EXPONERING (Investerat vs Kassa) ---
    axes[1].plot(total_invested.index, total_invested, label='Investerat på marknaden', color='blue', linewidth=2)
    
    # Kolla om er simulator sparar Cash-kolumnen, rita den i så fall
    if 'Cash' in portfolio.columns:
        axes[1].plot(portfolio.index, portfolio['Cash'], label='Kontanter (Säkerhet)', color='green', linewidth=2)
        
    axes[1].set_title('2. Säkerhetsspärren: Ligger vi exponerade eller i säkerhet?', fontweight='bold')
    axes[1].set_ylabel('Värde (SEK)')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)

    # --- GRAF 3: DETALJSTUDIE AV EN AKTIE (Pris + EWMA + Innehav) ---
    # Vi räknar ut EWMA just för den mest ägda aktien för att se varför boten agerade (Antar span=60 här, ändra om ni använde annat)
    ret = np.log(prices[most_held_asset] / prices[most_held_asset].shift(1)).fillna(0)
    ewma_drift = ret.ewm(span=60, adjust=False).mean()
    
    ax3 = axes[2]
    ax3_twin = ax3.twinx() # Skapa en extra Y-axel för driften
    
    # Rita priset
    ax3.plot(prices.index, prices[most_held_asset], color='black', label=f'Pris ({most_held_asset})')
    
    # Rita driften på höger Y-axel
    ax3_twin.plot(ewma_drift.index, ewma_drift, color='orange', label='EWMA Drift Signal', linestyle='--')
    ax3_twin.axhline(0, color='red', linestyle=':', label='Noll-linjen (Säljsignal)')
    
    # Färglägg bakgrunden GRÖN under de dagar som algoritmen faktiskt ÄGDE aktien
    held_mask = pos_values[most_held_asset] > 0
    ax3.fill_between(pos_values.index, prices[most_held_asset].min(), prices[most_held_asset].max(),
                     where=held_mask, color='green', alpha=0.2, label='Dagar vi ägde aktien')
                     
    ax3.set_title(f'3. Maskinens logik studerad i mikroskop: {most_held_asset}', fontweight='bold')
    ax3.set_ylabel('Pris', color='black')
    ax3_twin.set_ylabel('EWMA Drift', color='orange')
    
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('debug_dashboard.png', dpi=150)
    print("Dashboard genererad! Öppna 'debug_dashboard.png'.")

if __name__ == "__main__":
    generate_debug_dashboard()