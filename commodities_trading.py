import numpy as np
from sklearn.linear_model import LinearRegression
import volatility_sisr

def calibrate_drift_only(prices_window):
    """
    Räknar BARA ut kappa och alpha. Vi struntar i volatiliteten här.
    """
    log_S = np.log(prices_window).dropna().values
    X_t_minus_1 = log_S[:-1].reshape(-1, 1)
    X_t = log_S[1:]
    
    lr = LinearRegression()
    lr.fit(X_t_minus_1, X_t)
    
    b = np.clip(lr.coef_[0], 1e-5, 0.9999) 
    a = lr.intercept_
    
    kappa = -np.log(b)
    alpha = a / (1 - b)
    
    return kappa, alpha

def generate_advanced_schwartz_signals(prices_series, window=60, n_paths=1000):
    """
    Kombinerar regression (drift), partikelfilter (volatilitet) och Monte Carlo (framtid).
    """
    signals = np.zeros(len(prices_series))
    dt = 1.0
    days_ahead = 10
    # 2. hämta volatiliteten från partikelfiltret.
    sigma_list = volatility_sisr.run_volatility_filter_on_prices(prices_series)
    
    for t in range(window, len(prices_series)):
        # 1. Hämta de historiska priserna för att hitta jämvikten
        window_prices = prices_series.iloc[t-window:t]
        kappa, alpha = calibrate_drift_only(window_prices)
        
        current_log_price = np.log(prices_series.iloc[t])
        
        # 2. hämta volatiliteten från partikelfiltret.
        sigma_t = sigma_list[t]
        
        # 3. MONTE CARLO SIMULERING (Spela upp 1000 framtider)
        paths = np.zeros(n_paths)
        paths[:] = current_log_price
        
        for step in range(days_ahead):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            drift = kappa * (alpha - paths) * dt
            
            # Använd partikelfiltrets dynamiska sigma!
            diffusion = sigma_t * dW 
            paths = paths + drift + diffusion
            
        # 4. SANNOLIKHET OCH SIGNAL
        prob_up = np.sum(paths > current_log_price) / n_paths
        
        # Handla bara om vi har en enorm statistisk fördel (över 90% chans)
        if prob_up > 0.90:
            signals[t] = 1   # Extremt hög chans för uppgång
        elif prob_up < 0.10:
            signals[t] = -1  # Extremt hög chans för nedgång (kortning)
        else:
            signals[t] = 0   # Inget tydligt edge, avvakta
            
    return signals