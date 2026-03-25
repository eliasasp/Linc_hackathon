import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def estimate_sv_parameters(returns):
    """
    Snabbuppskattning av mu, phi och sigma_eta direkt från data. Vill ha stängningspris (inte log avkastning)
    """
    # Ta bort nollor för att undvika log(0)-krascher
    ret_clean = returns[returns != 0]
    if len(ret_clean) == 0:
        return -0.5, 0.95, 0.2 # Fallback om datan är trasig
        
    # 1. Estimera mu (Långsiktigt snitt av log-variansen)
    var_returns = np.var(ret_clean)
    mu_est = np.log(var_returns)
    
    # För att estimera phi och sigma_eta använder vi logaritmen av kvadrerade avkastningar
    # Vi lägger till en liten konstant (1e-6) för att inte få -inf
    log_sq_ret = np.log(ret_clean**2 + 1e-6)
    
    # 2. Estimera phi (AR(1) koefficienten) via kovarians
    x = log_sq_ret[:-1]
    y = log_sq_ret[1:]
    covariance = np.cov(x, y)[0, 1]
    variance = np.var(x)
    phi_est = covariance / variance
    
    # Klipp phi så det håller sig inom rimliga gränser för dagsdata (0.5 till 0.99)
    phi_est = np.clip(phi_est, 0.50, 0.99)
    
    # 3. Estimera sigma_eta (bruset i volatiliteten)
    # Residualerna från vår AR(1) proxy
    residuals = y - (mu_est * (1 - phi_est) + phi_est * x)
    # Eftersom log(chi^2) bruset är enormt, skalar vi ner det empiriskt för filtret
    sigma_eta_est = np.std(residuals) * 0.15 
    sigma_eta_est = np.clip(sigma_eta_est, 0.05, 0.5)
    
    return mu_est, phi_est, sigma_eta_est

def run_volatility_filter_on_prices(prices_series, n_particles=1000):
    """
    All-in-one funktion: Tar in priser, estimerar parametrar, kör partikelfiltret och returnerar volatiliteten.
    """
    # 1. Räkna ut log-avkastningen
    returns = np.log(prices_series / prices_series.shift(1)).dropna().values
    
    # 2. Uppskatta parametrarna automatiskt
    mu, phi, sigma_eta = estimate_sv_parameters(returns)
    print(f"Automatiska parametrar -> mu: {mu:.2f}, phi: {phi:.2f}, sigma_eta: {sigma_eta:.2f}")
    
    # 3. Kör partikelfiltret (samma logik som tidigare)
    n_steps = len(returns)
    estimated_volatility = np.zeros(n_steps)
    
    particles_h = np.random.normal(mu, sigma_eta / np.sqrt(1 - phi**2), n_particles)
    weights = np.ones(n_particles) / n_particles

    for t in range(n_steps):
        y_t = returns[t]
        
        # Predict
        noise = np.random.normal(0, sigma_eta, n_particles)
        particles_h = mu + phi * (particles_h - mu) + noise
        
        # Update / Weight
        particle_std_devs = np.exp(particles_h / 2)
        weights = norm.pdf(y_t, loc=0, scale=particle_std_devs)
        weights += 1e-300 
        weights /= np.sum(weights)
        
        # Estimate
        estimated_volatility[t] = np.sum(weights * particle_std_devs)
        
        # Resample
        indices = np.random.choice(n_particles, size=n_particles, p=weights)
        particles_h = particles_h[indices]

    return estimated_volatility

def test_auto_volatility_filter():
    print("Startar test av automatiskt partikelfilter...")
    
    # 1. Läs in er hackathon-data
    try:
        prices = pd.read_csv('prices.csv', index_col='Date', parse_dates=['Date'])
    except FileNotFoundError:
        print("Kunde inte hitta 'prices.csv'. Se till att filen ligger i samma mapp som skriptet.")
        return

    # 2. Välj vilken tillgång du vill testa på (Prova gärna att byta till 'FX_01' eller 'Comm_01' sen!)
    asset = 'Stock_01'
    stock_prices = prices[asset]
    
    # 3. Kör "All-in-one"-funktionen!
    # Den sköter nu allt: beräknar avkastning, estimerar parametrar och kör filtret.
    dold_volatilitet = run_volatility_filter_on_prices(stock_prices)

    # 4. Rita upp resultatet (Pris överst, Volatilitet underst)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Övre grafen: Aktiens pris
    ax1.plot(stock_prices.index, stock_prices, color='blue', label=f'Prisutveckling {asset}')
    ax1.set_title(f'SISR Partikelfilter med auto-parametrar för {asset}')
    ax1.set_ylabel('Normaliserat Pris')
    ax1.grid(True)
    ax1.legend()

    # Undre grafen: Estimerad volatilitet
    # Notera att volatilitets-arrayen är 1 steg kortare pga log-avkastningen, så vi skippar första datumet
    ax2.plot(stock_prices.index[1:], dold_volatilitet, color='red', label='Dold Volatilitet (\u03C3)')
    ax2.set_ylabel('Volatilitet')
    ax2.set_xlabel('Datum')
    ax2.grid(True)
    ax2.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Kör funktionen
#test_auto_volatility_filter()