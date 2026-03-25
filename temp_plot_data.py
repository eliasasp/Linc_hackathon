import matplotlib.pyplot as plt
import pandas as pd
prices = pd.read_csv("prices.csv", index_col="Date", parse_dates=True)
STOCK_COLS = [c for c in prices.columns if c.startswith("Stock_")]
FX_COLS    = [c for c in prices.columns if c.startswith("FX_")]

print(f"Laddade {len(prices)} handelsdagar, {len(prices.columns)} tillgångar")

equity_ccy = {
    'Stock_01': 'Crncy_03', 'Stock_02': 'Crncy_04', 'Stock_03': 'Crncy_04',
    'Stock_04': 'Crncy_02', 'Stock_05': 'Crncy_03', 'Stock_06': 'Crncy_02',
    'Stock_07': 'Crncy_03', 'Stock_08': 'Crncy_02', 'Stock_09': 'Crncy_04',
    'Stock_10': 'Crncy_03', 'Stock_11': 'Crncy_01', 'Stock_12': 'Crncy_04',
    'Stock_13': 'Crncy_01', 'Stock_14': 'Crncy_01', 'Stock_15': 'Crncy_01',
}

fx_pairs_map = {
    'FX_01': ('Crncy_02', 'Crncy_01'), 'FX_02': ('Crncy_04', 'Crncy_02'),
    'FX_03': ('Crncy_04', 'Crncy_03'), 'FX_04': ('Crncy_02', 'Crncy_03'),
    'FX_05': ('Crncy_01', 'Crncy_03'), 'FX_06': ('Crncy_04', 'Crncy_01'),
}
# Sätt stil
plt.style.use('default')

# Skapa en 2x2 figur för de olika tillgångsklasserna
fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)

# Lista över titlar och de prefix som identifierar tillgångarna i CSV-filen
asset_groups = [
    ("Stocks", "Stock_"),
    ("Indices", "Idx_"),
    ("Commodities", "Comm_"),
    ("FX", "FX_")
]

for ax, (title, prefix) in zip(axes.flat, asset_groups):
    # Filtrera ut kolumner som matchar prefixet
    cols = [c for c in prices.columns if c.startswith(prefix)]
    
    if len(cols) > 0:
        # Normalisera priserna (indexera till 100 vid start) för bättre jämförelse
        normalized_data = (prices[cols] / prices[cols].iloc[0] * 100)
        normalized_data.plot(ax=ax, linewidth=1.0, legend=True)
        
        ax.set_title(f"{title} (Indexerat till 100)", fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, ncol=3, loc='upper left')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_ylabel("Normaliserat Pris")
    else:
        ax.set_title(f"Ingen data hittades för {title}")

plt.tight_layout()
plt.show()