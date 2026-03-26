import pandas as pd
import numpy as np
from trading_simulator import TradingSimulator
import volatility_sisr
import correlation_sisr
from plt_pairtrade import plot_pairs_trade

def calculate_zscore(hist1, hist2, current_sigma):
    """Beräknar Z-score för spreaden mellan två historiska pris-serier"""
    # Vi använder log-priser för att få procentuell spridning
    spread = np.log(np.array(hist1)) - np.log(np.array(hist2))
    mu = np.mean(spread)#kolla på om man vill begränsa fönsterstorleken
    #kolla på att ändra till partikelfilter
    
    if current_sigma == 0: 
        return 0.0
    return (spread[-1] - mu) / current_sigma





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

    
    # +++ NYTT: BERÄKNA ADAPTIVT ENTRY-KRAV +++
    # Vi mäter marknadens totala stress genom snittvolatiliteten för alla assets
    avg_market_vol = np.mean(list(vols_today.values()))
    
    # Om marknadsvol är högre än params['base_vol'] (t.ex. 0.01), höjer vi ribban för Z-score
    # Detta stoppar oss från att "jaga brus" under kriser som 1994
    base_vol = params.get('base_vol', 0.01) 
    vol_multiplier = max(1.0, avg_market_vol / base_vol)
    adaptive_z_entry = params['z_entry_lo'] * vol_multiplier


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

        a1_idx = corr_manager.asset_to_idx[a1]
        a2_idx = corr_manager.asset_to_idx[a2]

        rho_today = current_matrix[a1_idx, a2_idx]
        vol_today = vols_today[a1] + vols_today[a2] + 2 * vols_today[a1] * vols_today[a2] * rho_today # stdavv för skillnaden mellan tillgångarna som går in i zscore

        z = calculate_zscore(history[a1], history[a2], vol_today)
        direction = p_data['direction']
        
        exit_trade = False

        # ===== CORRELATION GUARD ( NYTT BLOCK) =====
        if rho_today < params['min_active_corr']:
            exit_trade = True

        # ===== **TIME STOP** (NYTT BLOCK) =====
        time_in_trade = data['step_count'] - p_data['entry_step']

        if time_in_trade > params['time_stop'] and abs(z) > 0.6:
            exit_trade = True
        
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
    # Hämta de topp (...) mest korrelerade paren idag
    top_N = correlation_sisr.get_top_correlations(current_matrix, all_assets, top_n=params['top_N'])
    
    for item in top_N:
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
            
        
        vol_today = vols_today[a1] + vols_today[a2] + 2 * vols_today[a1] * vols_today[a2] * rho
        # Beräkna prisspridningen i standardavvikelser
        
        # Dynamisk "Target Volatility": Vi vill att paret ska svänga max t.ex. 1% per dag
        target_vol = 0.01 
        risk_scaler = (target_vol / max(vol_today, 0.0001))*40
        # Begränsa så vi inte satsar mer än 17% TOTALT I PARET oavsett hur låg vol är
        position_size = min(0.20, 0.05 * risk_scaler)

        # **vikta traden mot den asset som har högre volatilitet <-> den vi tror reagerar starkast på divergensbeteendet
        w_1 = 1 / ( (vols_today[a2]/vols_today[a1])**3 + 1 )
        w_2 = 1 - w_1

        position_size_1 = w_1 * position_size
        position_size_2 = w_2 * position_size
        
        z = calculate_zscore(history[a1], history[a2], vol_today)
        
        # Handla endast om spridningen är onormalt stor
        if abs(z) > adaptive_z_entry:
            
            # Insats: viktad efter varje tillgångs nuvarande volatilitet
            notional_1 = cash * position_size_1
            notional_2 = cash * position_size_2


            q1 = int(notional_1 / signal_prices[a1])
            q2 = int(notional_2 / signal_prices[a2])
            
            if q1 == 0 or q2 == 0:
                continue
                
            if z > adaptive_z_entry:
                # Z är hög. Det betyder att a1 är övervärderad i förhållande till a2.
                orders.append(('SELL', a1, q1)) # Blanka a1
                orders.append(('BUY', a2, q2))  # Köp a2
                open_pairs[pair_key] = {'direction': -1, 'q1': q1, 'q2': q2, 'entry_z': z, 'entry_step': data['step_count']}
                
            elif z < -adaptive_z_entry:
                # Z är låg. Det betyder att a1 är undervärderad.
                orders.append(('BUY', a1, q1))  # Köp a1
                orders.append(('SELL', a2, q2)) # Blanka a2
                open_pairs[pair_key] = {'direction': 1, 'q1': q1, 'q2': q2, 'entry_z': z, 'entry_step': data['step_count']}

    return orders















def run_backtest():
    print("Laddar prisdata...")
    prices = pd.read_csv('prices.csv', index_col='Date', parse_dates=['Date'])
    
    #selected_assets = ['Stock_01', 'Stock_02', 'Stock_03', 'Stock_04' , 'Stock_05' ,'Stock_06', 'Stock_07', 'Stock_08', 'Stock_09', 'Stock_10', 'Stock_11', 'Stock_12', 'Stock_13', 'Stock_14', 'Stock_15', 'Idx_04']
    #selected_assets = ['Comm_01', 'Comm_02', 'Comm_03', 'Comm_04', 'Comm_05', 'Comm_06']
    selected_assets = ['Idx_02', 'Idx_03', 'Idx_04', 'Stock_02', 'Stock_03', 'Stock_06', 'Stock_09']

    prices = prices[selected_assets]

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
            'corr_thresh': 0.4,  # Lägsta korrelation för att överväga paret
            'z_entry_lo': 1.4,       # Starta bettet om z > z_entry_lo
            #'z_entry_hi': 3,        # .. och om z < z_entry hi
            'base_vol': 0.012,       #(Normal dags-volatilitet)
            'z_exit': 0.5,        # Ta vinst när Z går under [...]
            'z_stop': 9,          # Panik-sälj om spridningen fortsätter 
            'time_stop': 20,
            'min_active_corr': 0.3,   # ⭐ NYTT (exit om korrelation dör)
            'top_N':15

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
 

    '''plot_pairs_trade('prices.csv', 'pairs_orders.csv', 'Idx_04', 'Stock_02')
    plot_pairs_trade('prices.csv', 'pairs_orders.csv', 'Idx_04', 'Stock_03')
    plot_pairs_trade('prices.csv', 'pairs_orders.csv', 'Idx_04', 'Stock_09')
    plot_pairs_trade('prices.csv', 'pairs_orders.csv', 'Idx_02', 'Idx_03')
    plot_pairs_trade('prices.csv', 'pairs_orders.csv', 'Stock_02', 'Stock_03')'''





#imorrn: kolla om vi kan implementera win/lose ratio och på något sätt se om vi kan bestämma position size utifrån det
#verkade funka bra att vikta varje ben, kan vi vikta starkare? tex **2
