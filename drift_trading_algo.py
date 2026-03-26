import pandas as pd
import numpy as np
from trading_simulator import TradingSimulator

# =========================================================
# 1. VERKTYG: EWMA DRIFT
# =========================================================

def calculate_all_ewma_drifts(prices_df, span=20):
    """Beräknar exponentiellt viktad drift (trend) för alla tillgångar."""
    daily_returns = np.log(prices_df / prices_df.shift(1)).fillna(0)
    ewma_drift = daily_returns.ewm(span=span, adjust=False).mean()
    return ewma_drift

# =========================================================
# 2. HUVUDALGORITMEN (Renodlad Trend-Momentum)
# =========================================================

def macro_algorithm(row_pos, cash, portfolio, signal_prices, data):
    orders = []
    
    all_assets = data['assets']
    params = data['params']
    
    data['step_count'] += 1
    
    # Vänta på att EWMA ska få lite historik
    if data['step_count'] < params['warmup_period']:
        return []

    # ---------------------------------------------------------
    # STEG 1: LÄS AV DAGENS TRENDER
    # ---------------------------------------------------------
    current_date = data['ewma_drift'].index[row_pos]
    drifts_today = data['ewma_drift'].loc[current_date]
    
    macro_targets = {asset: 0 for asset in all_assets}
    investable_cash = cash * 0.95 # Vi investerar 95% av kassan
    
    # ---------------------------------------------------------
    # STEG 2: ALLOKERA KAPITALET (Bara de starkaste tillgångarna)
    # ---------------------------------------------------------
    # Sortera tillgångarna från högst till lägst trend
    sorted_assets = drifts_today.sort_values(ascending=False)
    
    # Plocka ut de topp X bästa
    top_trending = sorted_assets.head(params['num_assets_to_hold']).index
    
    capital_per_asset = investable_cash / len(top_trending)
    
    for asset in top_trending:
        # Säkerhetsspärr: Köp BARA om driften faktiskt är positiv.
        # Om hela marknaden störtdyker ligger vi hellre i 100% cash.
        if drifts_today[asset] > 0: 
            price = signal_prices[asset]
            macro_targets[asset] = int(capital_per_asset / price)

    # ---------------------------------------------------------
    # STEG 3: SKAPA ORDRAR (Rebalansering)
    # ---------------------------------------------------------
    for asset in all_assets:
        target_qty = macro_targets[asset]
        current_qty = data['inventory'][asset]
        
        delta = target_qty - current_qty
        
        if delta > 0:
            orders.append(('BUY', asset, delta))
        elif delta < 0:
            orders.append(('SELL', asset, abs(delta)))
            
        # Uppdatera vårt minne av vad vi äger
        data['inventory'][asset] = target_qty
        
    return orders

# =========================================================
# 3. KÖR BACKTEST
# =========================================================

def run_macro_backtest():
    print("Laddar prisdata för Momentum-strategin...")
    prices = pd.read_csv('prices.csv', index_col='Date', parse_dates=['Date'])
    
    # Eftersom vi handlar ren momentum, kan vi ta med alla aktier och index!
    # (Jag tog bort valutor och råvaror då de ofta är sämre för ren trendföljning)
    selected_assets = [c for c in prices.columns if c.startswith('Stock_') or c.startswith('Idx_')]
    prices = prices[selected_assets]
    all_assets = prices.columns.tolist()
        
    print("Beräknar EWMA-drift för hela datasetet...")
    # Testa att skruva på 'span' här. 
    # 20 = snabb (1 månad), 60 = medium (1 kvartal), 120 = trög (ett halvår)
    span_val = 60 
    ewma_drift_df = calculate_all_ewma_drifts(prices, span=span_val) 
    
    strategy_data = {
        'assets': all_assets,
        'step_count': 0,
        'ewma_drift': ewma_drift_df,
        'inventory': {asset: 0 for asset in all_assets}, 
        
        'params': {
            'warmup_period': span_val,   
            'num_assets_to_hold': 5      # Sprid risken över de 5 absolut starkaste aktierna
        }
    }
    
    print("Startar backtesting-simulatorn (Ren EWMA Momentum)...")
    simulator = TradingSimulator(
        assets=all_assets,
        initial_cash=100_000
    )
    
    simulator.run(
        strategy_fn=macro_algorithm, 
        prices_df=prices, 
        data=strategy_data
    )
    
    simulator.save_results("momentum_orders.csv", "momentum_portfolio.csv")
    simulator.plot_performance(prices, save_file="momentum_performance.png")

if __name__ == "__main__":
    run_macro_backtest()