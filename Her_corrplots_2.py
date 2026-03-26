import numpy as np
import pandas as pd
from pathlib import Path
from trading_simulator import TradingSimulator
import matplotlib.pyplot as plt
import seaborn as sns


# ===== LOAD DATA =====
prices = pd.read_csv("prices.csv", index_col="Date", parse_dates=True)


# ==== Stock correlation heatmap ====
def plot_stock_FX_correlation_heatmap(prices):
    # Select stock columns
    STOCK_COLS = [c for c in prices.columns if c.startswith("Stock_")]
    FX_COLS = [c for c in prices.columns if c.startswith("FX_")]
    # Compute returns
    cols = STOCK_COLS + FX_COLS
    
    returns = prices[cols].pct_change().dropna()

    # Correlation matrix
    corr = returns.corr()

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )

    plt.title("Stock,FX Return Correlation Heatmap")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

plot_stock_FX_correlation_heatmap(prices)




# ==== Stock, Idx correlation heatmap ====
def plot_stock_fx_correlation_heatmap(prices):
    STOCK_COLS = [c for c in prices.columns if c.startswith("Stock_")]
    IDX_COLS = [c for c in prices.columns if c.startswith("Idx_")]

    cols = STOCK_COLS +  IDX_COLS

    returns = prices[cols].pct_change().dropna()

    corr = returns.corr()

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )

    plt.title("Stock,Idx Return Correlation Heatmap")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

plot_stock_fx_correlation_heatmap(prices)



# ==== Stock commodity correlation heatmap ====
def plot_stock_comm_correlation_heatmap(prices):
    STOCK_COLS = [c for c in prices.columns if c.startswith("Stock_")]
    COMM_COLS = [c for c in prices.columns if c.startswith("Comm_")]

    cols = STOCK_COLS + COMM_COLS

    returns = prices[cols].pct_change().dropna()

    corr = returns.corr()

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )

    plt.title("Stock-Commodity Return Correlation Heatmap")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

plot_stock_comm_correlation_heatmap(prices)