import pandas as pd
import numpy as np
from trading_simulator import TradingSimulator
import volatility_sisr
import correlation_sisr

def calculate_zscore(hist1, hist2):
    """Beräknar Z-score för spreaden mellan två historiska pris-serier"""
    # Vi använder log-priser för att få procentuell spridning
    spread = np.log(np.array(hist1)) - np.log(np.array(hist2))
    mu = np.mean(spread)#kolla på om man vill begränsa fönsterstorleken
    sigma = np.std(spread)#kolla på att ändra till partikelfilter
    
    if sigma == 0: 
        return 0.0
    return (spread[-1] - mu) / sigma

def main_algorithm(row_pos, cash, portfolio, signal_prices, data):
    orders = []
    
    # Packa upp våra verktyg och parametrar
    all_assets = data['assets']
    corr_manager = data['corr_manager']
    vol_filters = data['vol_filters']
    last_prices = data['last_prices']
    history = data['history']
    open_pairs = data['open_pairs']
    params = data['params']
    
    returns_today = {}
    vols_today = {}
    
    # ---------------------------------------------------------
    # 1. UPPDATERA DATA (Volatilitet, Priser & Historik)
    # ---------------------------------------------------------
    for asset in all_assets:
        p_today = signal_prices.get(asset)
        if pd.isna(p_today):
            continue
            
        # Historik (för Z-score beräkningen)
        history[asset].append(p_today)
        if len(history[asset]) > params['window']:
            history[asset].pop(0)
            
        p_yesterday = last_prices[asset]
        if p_yesterday > 0 and p_today > 0:
            ret = np.log(p_today / p_yesterday)
        else:
            ret = 0.0
            
        returns_today[asset] = ret
        vols_today[asset] = vol_filters[asset].update(p_today)
        last_prices[asset] = p_today

    # ---------------------------------------------------------
    # 2. UPPDATERA MATRISEN (SISR Korrelation)
    # ---------------------------------------------------------
    current_matrix = corr_manager.update(returns_today, vols_today)
    data['step_count'] += 1
    
    # Vänta tills vi fyllt historiken innan vi börjar handla
    if data['step_count'] < params['window']:
        return []

    # ---------------------------------------------------------
    # 3. HANTERA BEFINTLIGA POSITIONER (Exit & Stop-Loss)
    # ---------------------------------------------------------
    pairs_to_close = []
    
    for pair_key, p_data in open_pairs.items():
        a1, a2 = pair_key
        z = calculate_zscore(history[a1], history[a2])
        direction = p_data['direction']
        
        exit_trade = False
        
        # Om vi köpte a1 och blankade a2 (direction == 1, vi väntade på att Z skulle gå UPP)
        if direction == 1:
            # Exit vinst (närmar sig 0) ELLER Exit förlust (fortsätter störtdyka)
            if z > -params['z_exit'] or z < -params['z_stop']: 
                exit_trade = True
                
        # Om vi blankade a1 och köpte a2 (direction == -1, vi väntade på att Z skulle gå NER)
        elif direction == -1:
            if z < params['z_exit'] or z > params['z_stop']: 
                exit_trade = True
                
        if exit_trade:
            # Stäng positionen genom att göra tvärtom
            if direction == 1:
                orders.append(('SELL', a1, p_data['q1']))
                orders.append(('BUY', a2, p_data['q2']))
            else:
                orders.append(('BUY', a1, p_data['q1']))
                orders.append(('SELL', a2, p_data['q2']))
            
            pairs_to_close.append(pair_key)
            
    # Städa bort de stängda paren från minnet
    for pk in pairs_to_close:
        del open_pairs[pk]

    # ---------------------------------------------------------
    # 4. HITTA OCH ÖPPNA NYA PAIRS
    # ---------------------------------------------------------
    # Hämta de topp 10 mest korrelerade paren idag
    top_10 = correlation_sisr.get_top_correlations(current_matrix, all_assets, top_n=10)
    
    for item in top_10:
        rho = item['value']
        
        # Filtrera: Bara handla på urstarka samband
        if rho < params['corr_thresh']:
            continue
            
        # Dela upp stringen för att få fram tillgångarna (ex "Comm_01 & Comm_02")
        a1, a2 = item['pair'].split(' & ')
        pair_key = (a1, a2)
        
        # Kolla så vi inte redan har en öppen trade för detta par
        if pair_key in open_pairs or (a2, a1) in open_pairs: #kanske ta bort denna sen!
            continue
            
        # Beräkna prisspridningen i standardavvikelser
        z = calculate_zscore(history[a1], history[a2])
        
        # Handla endast om spridningen är onormalt stor
        if abs(z) > params['z_entry']:
            
            # Insats: 5% av totala kassan per "ben" i paret
            notional = cash * 0.05
            q1 = int(notional / signal_prices[a1])
            q2 = int(notional / signal_prices[a2])
            
            if q1 == 0 or q2 == 0:
                continue
                
            if z > params['z_entry']:
                # Z är hög. Det betyder att a1 är övervärderad i förhållande till a2.
                orders.append(('SELL', a1, q1)) # Blanka a1
                orders.append(('BUY', a2, q2))  # Köp a2
                open_pairs[pair_key] = {'direction': -1, 'q1': q1, 'q2': q2, 'entry_z': z}
                
            elif z < -params['z_entry']:
                # Z är låg. Det betyder att a1 är undervärderad.
                orders.append(('BUY', a1, q1))  # Köp a1
                orders.append(('SELL', a2, q2)) # Blanka a2
                open_pairs[pair_key] = {'direction': 1, 'q1': q1, 'q2': q2, 'entry_z': z}

    return orders

def run_backtest():
    print("Laddar prisdata...")
    prices = pd.read_csv('prices.csv', index_col='Date', parse_dates=['Date'])
    all_assets = prices.columns.tolist()
    
    print(f"Initierar korrelationsmatris och volatilitetsfilter för {len(all_assets)} tillgångar...")
    corr_manager = correlation_sisr.DynamicCorrelationMatrix(all_assets)
    vol_filters = {}
    
    for asset in all_assets:
        rets_full = np.log(prices[asset] / prices[asset].shift(1)).dropna().values
        mu, phi, sigma_eta = volatility_sisr.estimate_sv_parameters(rets_full)
        vol_filters[asset] = volatility_sisr.IncrementalVolatilityFilter(mu, phi, sigma_eta)
    
    # ---------------------------------------------------------
    # STATE OCH PARAMETRAR
    # ---------------------------------------------------------
    strategy_data = {
        'assets': all_assets,
        'corr_manager': corr_manager,
        'vol_filters': vol_filters,
        'last_prices': {asset: prices[asset].iloc[0] for asset in all_assets},
        'step_count': 0,
        
        # Nya fält för Pairs Trading
        'history': {asset: [] for asset in all_assets},
        'open_pairs': {}, 
        'params': {
            'window': 60,         # Dagar för rullande medelvärde
            'corr_thresh': 0.75,  # Lägsta korrelation för att överväga paret
            'z_entry': 2.0,       # Starta bettet vid 2 standardavvikelser
            'z_exit': 0.5,        # Ta vinst när Z går under 0.5
            'z_stop': 4.0         # Panik-sälj om spridningen fortsätter till 4.0
        }
    }
    
    print("Startar backtesting-simulatorn (Pairs Trading)...")
    simulator = TradingSimulator(
        assets=all_assets,
        initial_cash=100_000
    )
    
    simulator.run(
        strategy_fn=main_algorithm, 
        prices_df=prices, 
        data=strategy_data
    )
    
    # Spara och kolla hur mycket pengar vi tjänade
    simulator.save_results("pairs_orders.csv", "pairs_portfolio.csv")
    simulator.plot_performance(prices, save_file="pairs_performance.png")

if __name__ == "__main__":
    run_backtest()