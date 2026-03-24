import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import volatility_sisr

def particle_filter_correlation(returns_x, returns_y, vol_x, vol_y, n_particles=1000, sigma_z=0.05):
    """
    SISR Partikelfilter för dynamisk korrelation.
    Nu uppdaterad för att ta in dynamisk volatilitet (vol_x, vol_y) från ert andra filter!
    """
    n_steps = len(returns_x)
    estimated_correlation = np.zeros(n_steps)
    
    # --- HÄR ÄR DEN STORA ÄNDRINGEN ---
    # Istället för att dela med np.std (en fast siffra), delar vi varje dags 
    # avkastning med just den dagens estimerade volatilitet!
    # Vi drar också av snittavkastningen (som oftast är nära 0) för att centrera bruset.
    x_norm = (returns_x - np.mean(returns_x)) / vol_x
    y_norm = (returns_y - np.mean(returns_y)) / vol_y
    # ----------------------------------
    
    # 1. Initiera partiklar i Fisher Z-rymden
    particles_z = np.random.normal(0, 0.5, n_particles)
    weights = np.ones(n_particles) / n_particles

    for t in range(n_steps):
        x_t = x_norm[t]
        y_t = y_norm[t]
        
        # 2. PREDICT: Slumpvandring i Z-rymden
        noise = np.random.normal(0, sigma_z, n_particles)
        particles_z = particles_z + noise
        particles_rho = np.tanh(particles_z)
        particles_rho = np.clip(particles_rho, -0.999, 0.999)
        
        # 3. WEIGHT: Bivariat normalfördelningens täthetsfunktion
        det = 1 - particles_rho**2
        exponent = - (x_t**2 - 2 * particles_rho * x_t * y_t + y_t**2) / (2 * det)
        weights = (1 / np.sqrt(det)) * np.exp(exponent)
        
        weights += 1e-300
        weights /= np.sum(weights)
        
        # 4. ESTIMATE
        current_rho_estimate = np.sum(weights * particles_rho)
        estimated_correlation[t] = current_rho_estimate
        
        # 5. RESAMPLE
        indices = np.random.choice(n_particles, size=n_particles, p=weights)
        particles_z = particles_z[indices]

    return estimated_correlation

def test_correlation_filter():
    print("Startar test av korrelationsfiltret...")

    prices = pd.read_csv('prices.csv', index_col='Date', parse_dates=['Date'])

    # Välj två tillgångar
    asset_x = 'Stock_01'
    asset_y = 'Idx_01'

    # Beräkna log-avkastningar
    returns_x = np.log(prices[asset_x] / prices[asset_x].shift(1)).dropna().values
    returns_y = np.log(prices[asset_y] / prices[asset_y].shift(1)).dropna().values

    # STEG 1: Kör Volatilitetsfiltret för Tillgång X
    # (Här använder vi den automatiska funktionen vi skrev tidigare)
    print(f"Beräknar dynamisk volatilitet för {asset_x}...")
    vol_x = volatility_sisr.run_volatility_filter_on_prices(prices[asset_x])

    # STEG 2: Kör Volatilitetsfiltret för Tillgång Y
    print(f"Beräknar dynamisk volatilitet för {asset_y}...")
    vol_y = volatility_sisr.run_volatility_filter_on_prices(prices[asset_y])

    # STEG 3: Kör Korrelationsfiltret och skicka in allt!
    print("Beräknar den volatilitetsjusterade korrelationen...")
    dynamisk_korrelation = particle_filter_correlation(returns_x, returns_y, vol_x, vol_y)
    
    # -- RITA UPP GRAFERNA --
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Övre grafen: Prisutvecklingen (Normaliserad så de startar på samma ställe)
    ax1.plot(prices.index, prices[asset_x] / prices[asset_x].iloc[0], label=asset_x, color='blue')
    ax1.plot(prices.index, prices[asset_y] / prices[asset_y].iloc[0], label=asset_y, color='orange')
    ax1.set_title(f'Prisutveckling: {asset_x} vs {asset_y}')
    ax1.set_ylabel('Normaliserat Pris')
    ax1.legend()
    ax1.grid(True)
    
    # Undre grafen: Dynamisk Korrelation
    dates = prices.index[1:]
    ax2.plot(dates, dynamisk_korrelation, label='Estimerad Dynamisk Korrelation (\u03C1)', color='purple')
    
    # Lägg in linjer för noll och genomsnittlig korrelation för kontext
    ax2.axhline(0, color='black', linewidth=1, linestyle='--')
    ax2.axhline(np.mean(dynamisk_korrelation), color='red', linewidth=1, linestyle=':', label='Snittkorrelation')
    
    ax2.set_ylabel('Korrelation (-1 till 1)')
    ax2.set_xlabel('Datum')
    ax2.set_ylim(-1.1, 1.1)
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Kör testet
test_correlation_filter()