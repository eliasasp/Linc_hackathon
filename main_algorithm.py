import pandas as pd
import numpy as np
from trading_simulator import TradingSimulator
import volatility_sisr
import correlation_sisr
import matplotlib.pyplot as plt 

# BYT UT MOT DENNA:
def calculate_zscore(hist1, hist2, vol1, vol2, rho):
    """Beräknar Z-score för spreaden mellan två historiska pris-serier"""
    # Vi använder log-priser för att få procentuell spridning
    spread = np.log(np.array(hist1)) - np.log(np.array(hist2))
    mu = np.mean(spread) 
    
    # RÄTTNING: Volatilitet (sigma) måste kvadreras för att bli varians
    var1 = vol1**2
    var2 = vol2**2
    
    # RÄTTNING: Variansen för en differens (A - B) beräknas med ett MINUSTECKEN före kovariansen
    variance_spread = var1 + var2 - 2 * vol1 * vol2 * rho
    sigma = np.sqrt(max(variance_spread, 1e-8))
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
        idx1 = corr_manager.asset_to_idx[a1]
        idx2 = corr_manager.asset_to_idx[a2]
        rho_today = current_matrix[idx1, idx2]
        z = calculate_zscore(history[a1], history[a2], vols_today[a1], vols_today[a2], rho_today)
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
    top_10 = correlation_sisr.get_top_correlations(current_matrix, all_assets, top_n=1)
    
    for item in top_10:
        rho = item['value']
        
        # Filtrera: Bara handla på urstarka samband
        if rho < params['corr_thresh']:
            continue
            
        # Dela upp stringen för att få fram tillgångarna (ex "Comm_01 & Comm_02")
        a1, a2 = item['pair'].split(' & ')
        pair_key = (a1, a2)

        # NY KOD: Översätt namn till index först
        idx1 = corr_manager.asset_to_idx[a1]
        idx2 = corr_manager.asset_to_idx[a2]
        rho_today = current_matrix[idx1, idx2]
        z = calculate_zscore(history[a1], history[a2], vols_today[a1], vols_today[a2], rho_today)
        
        # Handla endast om spridningen är onormalt stor
        if abs(z) > params['z_entry']:
            
            # Insats: 1% av totala kassan per "ben" i paret
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
    # ---------------------------------------------------------
    # GAMMAL KOD: 
    all_assets = prices.columns.tolist()
    # ---------------------------------------------------------
    # NY KOD: Välj exakt vilka två tillgångar du vill handla
    #all_assets = ['Stock_09', 'Idx_04']  # Byt ut mot de du vill testa!
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
            'window': 55,         # Dagar för rullande medelvärde
            'corr_thresh': 0.75,  # Lägsta korrelation för att överväga paret
            'z_entry': 4.0,       # Starta bettet vid 2 standardavvikelser
            'z_exit': 0.01,        # Ta vinst när Z går under 0.5
            'z_stop': 8.0         # Panik-sälj om spridningen fortsätter till 4.0
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


def plot_pairs_trade(prices_file, orders_file, asset1, asset2, window=60, z_entry=2.0, z_exit=0.5):
    print(f"Ritar upp Pairs Trade-analys för {asset1} och {asset2}...")
    
    # 1. Ladda data
    prices_df = pd.read_csv(prices_file, index_col='Date', parse_dates=True)
    try:
        orders_df = pd.read_csv(orders_file, parse_dates=['Date'])
    except FileNotFoundError:
        print(f"Kunde inte hitta {orders_file}. Kör backtestet först!")
        return

    # Hantera kolumnnamn (Ticker vs Asset)
    col_ticker = 'Ticker' if 'Ticker' in orders_df.columns else 'Asset'
    
    # 2. Beräkna den historiska spreaden och Z-score (exakt som i algoritmen)
    p1 = prices_df[asset1]
    p2 = prices_df[asset2]
    
    # Log-spread
    spread = np.log(p1) - np.log(p2)
    
    # Rullande Z-score
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    z_score = (spread - rolling_mean) / rolling_std

    # 3. Hitta de dagar då BÅDA tillgångarna handlades (våra entry/exits)
    o1 = orders_df[orders_df[col_ticker] == asset1]
    o2 = orders_df[orders_df[col_ticker] == asset2]
    
    # Slå ihop dem på datum för att hitta par-transaktionerna
    pair_trades = pd.merge(o1, o2, on='Date', suffixes=('_1', '_2'))
    
    # -- RITA GRAFERNA --
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # --- ÖVRE GRAFEN: Normaliserade priser ---
    # Vi delar med första giltiga priset så att båda börjar på 1.0 (enklare att jämföra)
    ax1.plot(prices_df.index, p1 / p1.iloc[window], label=f'{asset1}', color='blue', linewidth=1.5)
    ax1.plot(prices_df.index, p2 / p2.iloc[window], label=f'{asset2}', color='orange', linewidth=1.5)
    ax1.set_title(f'Pairs Trading Analys: {asset1} vs {asset2}', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Normaliserat Pris', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- UNDRE GRAFEN: Z-score och signaler ---
    ax2.plot(prices_df.index, z_score, label='Z-score (Spread)', color='purple', linewidth=1.5)
    
    # Rita ut våra trigger-linjer
    ax2.axhline(z_entry, color='red', linestyle='--', alpha=0.6, label=f'Entry (+{z_entry})')
    ax2.axhline(-z_entry, color='green', linestyle='--', alpha=0.6, label=f'Entry (-{z_entry})')
    ax2.axhline(z_exit, color='gray', linestyle=':', alpha=0.6, label=f'Exit (+{z_exit})')
    ax2.axhline(-z_exit, color='gray', linestyle=':', alpha=0.6, label=f'Exit (-{z_exit})')
    ax2.axhline(0, color='black', linewidth=1)

    # Lägg ut markörer för tradesen
    for _, trade in pair_trades.iterrows():
        date = trade['Date']
        act1 = trade['Action_1']
        
        # Hämta Z-scoret för just denna dag
        z_val = z_score.loc[date]
        
        # Logik för att färga pilarna:
        if act1 == 'SELL' and z_val > 1.0:
            # Vi blankade Asset 1 för att Z var högt (Övervärderad)
            ax2.scatter(date, z_val, color='red', marker='v', s=150, zorder=5, edgecolors='black')
        elif act1 == 'BUY' and z_val < -1.0:
            # Vi köpte Asset 1 för att Z var lågt (Undervärderad)
            ax2.scatter(date, z_val, color='green', marker='^', s=150, zorder=5, edgecolors='black')
        else:
            # Om Z är nära 0, är det en stängning (Exit)
            ax2.scatter(date, z_val, color='yellow', marker='X', s=150, zorder=5, edgecolors='black')

    ax2.set_ylabel('Z-Score', fontsize=12)
    ax2.set_xlabel('Datum', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

import pandas as pd

def print_trade_history(orders_file='pairs_orders.csv'):
    print("\n" + "="*65)
    print(" 📊 TRADE HISTORIK (PAIRS TRADING)")
    print("="*65)
    
    try:
        orders_df = pd.read_csv(orders_file, parse_dates=['Date'])
    except FileNotFoundError:
        print(f"Kunde inte hitta filen '{orders_file}'. Har backtestet körts?")
        return
        
    if orders_df.empty:
        print("Orderboken är tom! Algoritmen gjorde inga trades.")
        return

    # Hantera kolumnnamn
    col_ticker = 'Ticker' if 'Ticker' in orders_df.columns else 'Asset'
    
    # Gruppera ordrarna per datum (Eftersom vi alltid köper och säljer samtidigt)
    grouped_orders = orders_df.groupby('Date')
    
    open_positions = {}
    completed_trades = []
    
    for date, group in grouped_orders:
        # En pairs trade innebär att vi har exakt 2 rader per datum
        if len(group) == 2:
            rows = group.to_dict('records')
            
            # Sortera alfabetiskt så vi alltid får samma ordning (T.ex. Comm_01 före Comm_02)
            rows_sorted = sorted(rows, key=lambda x: x[col_ticker])
            
            asset1 = rows_sorted[0][col_ticker]
            asset2 = rows_sorted[1][col_ticker]
            action1 = rows_sorted[0]['Action']
            pair_key = (asset1, asset2)
            
            # Om vi INTE har detta par i minnet, är det en ENTRY
            if pair_key not in open_positions:
                # Kolla om vi är Långa A1 eller Korta A1
                direction = f"KÖP {asset1} / BLANKA {asset2}" if action1 == 'BUY' else f"BLANKA {asset1} / KÖP {asset2}"
                
                open_positions[pair_key] = {
                    'entry_date': date,
                    'direction': direction
                }
                
            # Om vi REDAN HAR paret i minnet, är detta en EXIT (stängning)
            else:
                entry_data = open_positions.pop(pair_key)
                days_held = (date - entry_data['entry_date']).days
                
                completed_trades.append({
                    'pair': f"{asset1} & {asset2}",
                    'direction': entry_data['direction'],
                    'entry': entry_data['entry_date'].strftime('%Y-%m-%d'),
                    'exit': date.strftime('%Y-%m-%d'),
                    'duration': days_held
                })
                
    # --- SKRIV UT RESULTATET ---
    if not completed_trades:
        print("Inga stängda trades hittades.")
        
    for i, t in enumerate(completed_trades, 1):
        print(f"Trade #{i}: {t['pair']}")
        print(f"  Riktning:   {t['direction']}")
        print(f"  Tidsfönster:{t['entry']}  -->  {t['exit']} ({t['duration']} dagar)")
        print("-" * 65)
        
    # Om algoritmen låg kvar i trades när 8-årsperioden tog slut
    if open_positions:
        print("\n⚠️ ÖPPNA POSITIONER VID BACKTESTETS SLUT:")
        for pair_key, data in open_positions.items():
            print(f"  {pair_key[0]} & {pair_key[1]} | {data['direction']} | Sedan: {data['entry_date'].strftime('%Y-%m-%d')}")
        print("="*65)

# ==========================================
# TESTA FUNKTIONEN (Lägg detta längst ner i din fil)
# ==========================================

if __name__ == "__main__":
    run_backtest()
    print_trade_history('pairs_orders.csv')
    
    # Byt ut 'Comm_03' och 'Comm_04' mot de två tillgångar du vill granska.
    # Kolla i din utskrift från backtestet vilket par som hade bäst korrelation!
    #plot_pairs_trade('prices.csv', 'pairs_orders.csv', 'Stock_09', 'Idx_04')