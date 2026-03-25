import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class IncrementalVolatilityFilter:
    def __init__(self, mu, phi, sigma_eta, n_particles=500):
        self.mu = mu
        self.phi = phi
        self.sigma_eta = sigma_eta
        self.n_particles = n_particles
        
        # Initiera partiklar för dag 0
        self.particles_h = np.random.normal(
            mu, 
            sigma_eta / np.sqrt(1 - phi**2), 
            n_particles
        )
        self.last_price = None

    def update(self, current_price):
        """
        Matas med dagens pris. Uppdaterar partiklarna ETT steg och 
        returnerar dagens uppskattade volatilitet.
        """
        if self.last_price is None:
            self.last_price = current_price
            # Kan inte beräkna avkastning första dagen, returnera ett neutralt startvärde
            return np.exp(self.mu / 2) 
        
        # 1. Beräkna dagens log-avkastning
        y_t = np.log(current_price / self.last_price)
        self.last_price = current_price
        
        if y_t == 0:
            y_t = 1e-6 # Undvik division med noll/krascher
            
        # 2. Predict (Flytta partiklarna framåt)
        noise = np.random.normal(0, self.sigma_eta, self.n_particles)
        self.particles_h = self.mu + self.phi * (self.particles_h - self.mu) + noise
        
        # 3. Update (Väg partiklarna baserat på hur väl de förklarar dagens avkastning)
        particle_std_devs = np.exp(self.particles_h / 2)
        weights = norm.pdf(y_t, loc=0, scale=particle_std_devs)
        weights += 1e-300 
        weights /= np.sum(weights)
        
        # 4. Estimate (Dagens volatilitet)
        estimated_volatility = np.sum(weights * particle_std_devs)
        
        # 5. Resample (Byt ut dåliga partiklar mot bra)
        indices = np.random.choice(self.n_particles, size=self.n_particles, p=weights)
        self.particles_h = self.particles_h[indices]

        return estimated_volatility


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


def test_incremental_volatility_filter():
    print("Startar test av det INKREMENTELLA partikelfiltret...")
    
    # 1. Läs in data (skapa en låtsas-serie om du inte har filen än)
    try:
        prices = pd.read_csv('prices.csv', index_col='Date', parse_dates=['Date'])
        stock_prices = prices['Stock_01']
    except FileNotFoundError:
        print("Kunde inte hitta 'prices.csv'. Skapar fejk-data för att testa!")
        # Fejk-data för att bevisa att filtret snurrar
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 500)
        stock_prices = pd.Series(100 * np.exp(np.cumsum(returns)))
    
    # 2. Räkna ut log-avkastningen för att estimera parametrarna
    # (Mindre felskrivning i din docstring: funktionen VILL HA avkastning, inte stängningspris, 
    # och det är exakt det du ger den här, så det är helt rätt!)
    returns = np.log(stock_prices / stock_prices.shift(1)).dropna().values
    mu, phi, sigma_eta = estimate_sv_parameters(returns)
    print(f"Estimerade parametrar -> mu: {mu:.2f}, phi: {phi:.2f}, sigma_eta: {sigma_eta:.2f}")

    # 3. Skapa en instans av ditt NYA filter
    my_filter = IncrementalVolatilityFilter(mu, phi, sigma_eta, n_particles=1000)

    # 4. Mata filtret steg-för-steg (precis som simulatorn kommer göra)
    estimerad_vol = []
    for price in stock_prices:
        vol = my_filter.update(price)
        estimerad_vol.append(vol)

    # 5. Plotta
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(stock_prices.values, color='blue', label='Prisutveckling')
    ax1.set_title('Inkrementellt Partikelfilter (Objektorienterat)')
    ax1.set_ylabel('Pris')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(estimerad_vol, color='red', label='Dold Volatilitet (\u03C3)')
    ax2.set_ylabel('Volatilitet')
    ax2.set_xlabel('Dagar')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

# Testa den nya klassen!
#test_incremental_volatility_filter()