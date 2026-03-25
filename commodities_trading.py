import numpy as np
from sklearn.linear_model import LinearRegression
import volatility_sisr
import pandas as pd

def calibrate_drift_only(prices_window):
    """
    Räknar BARA ut kappa och alpha.
    """
    # Gör om till numpy array ifall vi skickar in en vanlig lista
    prices = np.array(prices_window)
    log_S = np.log(prices)
    X_t_minus_1 = log_S[:-1].reshape(-1, 1)
    X_t = log_S[1:]
    lr = LinearRegression()
    lr.fit(X_t_minus_1, X_t)
    b = np.clip(lr.coef_[0], 1e-5, 0.9999) 
    a = lr.intercept_
    kappa = -np.log(b)
    alpha = a / (1 - b)
    return kappa, alpha

def advanced_commodity_strategy(row_pos, cash, portfolio, signal_prices, data):
    orders = []
    
    # Packa upp vårt "minne" från data-variabeln
    filters = data['filters']
    history = data['history']
    
    window_size = 200
    days_ahead = 200
    n_paths = 1000
    dt = 1.0

    for ticker in signal_prices:
        current_price = signal_prices.get(ticker)
        if pd.isna(current_price) or ticker not in filters:
            continue
            
        # 1. Uppdatera det rullande fönstret
        history[ticker].append(current_price)
        if len(history[ticker]) > window_size:
            history[ticker].pop(0) 
            
        # 2. Uppdatera partikelfiltret med dagens pris
        sigma_t = filters[ticker].update(current_price)
        
        # Handla inte förrän vi har 60 dagars data
        if len(history[ticker]) < window_size:
            continue
            
        # 3. Kalibrera driften (Schwartz)
        kappa, alpha = calibrate_drift_only(history[ticker])
        current_log_price = np.log(current_price)
        
        # 4. Monte Carlo Simulering
        paths = np.full(n_paths, current_log_price)
        for _ in range(days_ahead):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            drift = kappa * (alpha - paths) * dt
            diffusion = sigma_t * dW
            paths = paths + drift + diffusion
            
        prob_up = np.sum(paths > current_log_price) / n_paths
        
        # =========================================================
        # 5. DYNAMISK TRADINGLOGIK (Ny Position Sizing!)
        # =========================================================
        current_position = portfolio.get(ticker, 0)
        
        # 1. KORTSIKTIGT MOMENTUM (Skydd mot fallande knivar)
        # Vi kräver att priset åtminstone ligger över sitt 5-dagars snitt för att köpa.
        short_sma = np.mean(history[ticker][-5:])
        is_bouncing_up = current_price > short_sma
        is_crashing_down = current_price < short_sma
        
        # A. Grundpott: Max 10% risk per trade
        max_notional = cash * 0.10 
        base_shares = max_notional / current_price
        
        # B. Skalning
        conviction = abs(prob_up - 0.5) * 2 
        vol_scalar = 0.015 / max(sigma_t, 0.001) 
        vol_scalar = min(vol_scalar, 1.5) 
        
        target_position = int(base_shares * conviction * vol_scalar)
        
        # --- EXEKVERA ORDRAR ---
        
        # 1. KÖP: Monte Carlo säger Upp OCH priset har faktiskt börjat vända upp
        if prob_up > 0.85 and is_bouncing_up:
            delta = target_position - current_position
            if delta > (target_position * 0.20):
                orders.append(('BUY', ticker, delta))
                
        # 2. BLANKA: Monte Carlo säger Ner OCH priset är i en tydlig kortsiktig nedgång
        elif prob_up < 0.15 and is_crashing_down:
            target_short = -target_position
            delta = target_short - current_position 
            if delta < -(target_position * 0.20):
                orders.append(('SELL', ticker, abs(delta)))
                
        # 3. EXIT & DEN RIKTIGA STOPP-LOSSEN
        elif current_position != 0:
            
            # Exit: Vi har nått vårt mål och jämvikten är återställd (50/50 chans)
            edge_lost = (0.45 < prob_up < 0.55)
            
            # Riktig Stop-Loss: Har vi brutit 10-dagars lägsta/högsta?
            lowest_10d = np.min(history[ticker][-10:])
            highest_10d = np.max(history[ticker][-10:])
            
            # Om vi köpt, men priset gör nya bottnar -> SÄLJ DIREKT
            stop_loss_long = (current_position > 0 and current_price < lowest_10d)
            # Om vi blankat, men priset gör nya toppar -> KÖP TILLBAKA DIREKT
            stop_loss_short = (current_position < 0 and current_price > highest_10d)
            
            if edge_lost or stop_loss_long or stop_loss_short:
                # Stäng positionen
                if current_position > 0:
                    orders.append(('SELL', ticker, current_position))
                else:
                    orders.append(('BUY', ticker, abs(current_position)))

    return orders

