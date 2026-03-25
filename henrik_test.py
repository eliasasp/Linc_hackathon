import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH = "prices.csv"

GROUPS = {
    "Stocks":      [f"Stock_{i:02d}" for i in range(1, 16)],
    "Commodities": [f"Comm_{i:02d}"  for i in range(1, 7)],
    "Indices":     ["Idx_01", "Idx_02", "Idx_03", "Idx_04"],
    "FX":          [f"FX_{i:02d}"    for i in range(1, 7)],
}

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 8))
axes = axes.flatten()

for ax, (title, cols) in zip(axes, GROUPS.items()):
    cols = [c for c in cols if c in df.columns]

    for col in cols:
        ax.plot(df["Date"], df[col], linewidth=1.2, label=col)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=7, ncol=2, loc="upper left", framealpha=0.6)
    ax.grid(True, linestyle="--", alpha=0.4)

fig.suptitle("Asset Price Dashboard", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
#plt.savefig("asset_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()


