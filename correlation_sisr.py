import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import volatility_sisr
from itertools import combinations

class DynamicCorrelationMatrix:
    def __init__(self, assets, n_particles=500, sigma_z=0.02):
        self.assets = assets
        self.n_assets = len(assets)
        self.asset_to_idx = {asset: i for i, asset in enumerate(assets)}
        
        # Skapa filter för alla unika par (kombinationer)
        # Vi sparar dem i en dictionary med nyckeln "AssetA_AssetB"
        self.pair_filters = {}
        for a1, a2 in combinations(assets, 2):
            pair_key = tuple(sorted((a1, a2)))
            self.pair_filters[pair_key] = IncrementalCorrelationFilter(
                n_particles=n_particles, 
                sigma_z=sigma_z
            )
            
        # Initiera själva matrisen (ettor i diagonalen)
        self.matrix = np.eye(self.n_assets)

    def update(self, returns_dict, vol_dict):
        """
        returns_dict: {'Stock_01': 0.02, 'Stock_02': -0.01, ...}
        vol_dict: {'Stock_01': 0.015, 'Stock_02': 0.025, ...}
        """
        for (a1, a2), filter_obj in self.pair_filters.items():
            # Hämta dagsdata för paret
            ret_x, ret_y = returns_dict[a1], returns_dict[a2]
            v_x, v_y = vol_dict[a1], vol_dict[a2]
            
            # Uppdatera det specifika partikelfiltret
            rho = filter_obj.update(ret_x, ret_y, v_x, v_y)
            
            # Sätt in värdet i matrisen (symmetriskt)
            i, j = self.asset_to_idx[a1], self.asset_to_idx[a2]
            self.matrix[i, j] = rho
            self.matrix[j, i] = rho
            
        return self.matrix
import numpy as np
from scipy.stats import norm

class IncrementalCorrelationFilter:
    def __init__(self, n_particles=500, sigma_z=0.02):
        self.n_particles = n_particles
        self.sigma_z = sigma_z
        
        # Initiera partiklar i Fisher Z-rymden (ger stabilare konvergens än direkt i Rho)
        # Z = 0 motsvarar en korrelation på 0.
        self.particles_z = np.random.normal(0, 0.5, n_particles)
        
        # Vi håller koll på ett rullande medelvärde för att centrera avkastningen utan "fusk"
        self.mean_x = 0
        self.mean_y = 0
        self.t = 0

    def update(self, ret_x, ret_y, vol_x, vol_y):
        """
        Uppdaterar korrelationen baserat på dagens avkastning och 
        den dynamiska volatiliteten från ditt SISR-volatilitetsfilter.
        """
        self.t += 1
        
        # 1. Uppdatera rullande medelvärden (online estimation)
        self.mean_x += (ret_x - self.mean_x) / self.t
        self.mean_y += (ret_y - self.mean_y) / self.t
        
        # 2. Normalisera dagens avkastning med dynamisk volatilitet
        # Detta gör att vi mäter korrelationen på "standardiserade" chocker
        x_norm = (ret_x - self.mean_x) / vol_x
        y_norm = (ret_y - self.mean_y) / vol_y
        
        # 3. Predict: Slumpvandring i Z-rymden
        noise = np.random.normal(0, self.sigma_z, self.n_particles)
        self.particles_z += noise
        
        # Konvertera Z till Rho (korrelation -1 till 1) för att beräkna vikter
        particles_rho = np.tanh(self.particles_z)
        particles_rho = np.clip(particles_rho, -0.999, 0.999)
        
        # 4. Weight: Bivariat normalfördelning
        det = 1 - particles_rho**2
        exponent = - (x_norm**2 - 2 * particles_rho * x_norm * y_norm + y_norm**2) / (2 * det)
        weights = (1 / np.sqrt(det)) * np.exp(exponent)
        
        weights += 1e-300 # Undvik division med noll
        weights /= np.sum(weights)
        
        # 5. Estimate: Väg samman partiklarna till dagens rho
        current_rho = np.sum(weights * particles_rho)
        
        # 6. Resample: Systematisk resampling för att behålla partikel-mångfald
        indices = np.random.choice(self.n_particles, size=self.n_particles, p=weights)
        self.particles_z = self.particles_z[indices]

        return current_rho

def test_full_matrix_step_by_step():
    print("--- STARTAR TEST FÖR ALLA TILLGÅNGAR (DAG 1-10) ---")
    
    # 1. Ladda data
    prices = pd.read_csv('prices.csv', index_col='Date', parse_dates=['Date'])
    
    # Hämta ALLA tillgångar från filen
    all_assets = prices.columns.tolist()
    print(f"Hittade {len(all_assets)} tillgångar i filen.")
    
    # Inställningar för att Pandas ska visa hela matrisen i terminalen
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', None)

    # 2. Initiera filter för alla tillgångar
    corr_manager = DynamicCorrelationMatrix(all_assets)
    vol_filters = {}
    
    print("Initierar volatilitetsfilter för alla tillgångar...")
    for asset in all_assets:
        # Beräkna parametrar baserat på historiken
        rets_full = np.log(prices[asset] / prices[asset].shift(1)).dropna().values
        mu, phi, sigma_eta = volatility_sisr.estimate_sv_parameters(rets_full)
        vol_filters[asset] = volatility_sisr.IncrementalVolatilityFilter(mu, phi, sigma_eta)

    # 3. Loopen för de första 10 dagarna
    for t in range(1, 1001):
        date = prices.index[t]
        returns_today = {}
        vols_today = {}
        
        # Beräkna dagsdata för varje enskild tillgång
        for asset in all_assets:
            p_now = prices[asset].iloc[t]
            p_yesterday = prices[asset].iloc[t-1]
            
            # Räkna ut log-return (avkastning)
            ret = np.log(p_now / p_yesterday)
            returns_today[asset] = ret
            
            # Uppdatera volatilitet (SISR)
            vols_today[asset] = vol_filters[asset].update(p_now)
            
        # 4. Uppdatera den globala korrelationsmatrisen
        # Detta kör nu N*(N-1)/2 partikelfilter i bakgrunden!
        current_matrix = corr_manager.update(returns_today, vols_today)
        
        # Hämta de 10 högsta
        top_10 = get_top_correlations(current_matrix, all_assets, top_n=10)
        
        print(f"\n--- TOPP 10 KORRELATIONER DAG {t} ---")
        for i, item in enumerate(top_10, 1):
            print(f"{i}. {item['pair']}: {item['value']:.4f}")

        # 5. Printa resultatet
        print(f"\n--- DAG {t} ({date.date()}) ---")
        df_matrix = pd.DataFrame(current_matrix, index=all_assets, columns=all_assets)
        #print(df_matrix.round(3)) # 3 decimaler räcker för en stor matris


def get_top_correlations(matrix, assets, top_n=10):
    """
    Extraherar de högsta unika korrelationerna från en matris.
    """
    correlations = []
    n = len(assets)
    
    # Vi använder np.triu_indices med k=1 för att bara titta på 
    # den övre triangeln ovanför diagonalen.
    row_idx, col_idx = np.triu_indices(n, k=1)
    
    for i, j in zip(row_idx, col_idx):
        correlations.append({
            'pair': f"{assets[i]} & {assets[j]}",
            'value': matrix[i, j]
        })
    
    # Sortera listan baserat på värdet (fallande)
    sorted_corr = sorted(correlations, key=lambda x: x['value'], reverse=True)
    
    return sorted_corr[:top_n]


# Kör testet för hela portföljen
# test_full_matrix_step_by_step()