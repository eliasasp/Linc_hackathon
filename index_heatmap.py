import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load ──────────────────────────────────────────────────────────────────────
prices = pd.read_csv("prices.csv", index_col="Date", parse_dates=True)
idx_cols = [c for c in prices.columns if c.startswith("Idx_")]

returns = prices[idx_cols].pct_change().dropna()
corr    = returns.corr()

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    vmin=-1, vmax=1,
    linewidths=0.5,
    linecolor="white",
    square=True,
    ax=ax,
)

ax.set_title("Index Return Correlations", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("index_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()