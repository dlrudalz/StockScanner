import requests
import pandas as pd
import numpy as np
from hmmlearn import hmm
import time
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from config import POLYGON_API_KEY

# Configuration
EXCHANGES = ["XNYS", "XNAS", "XASE"]  # NYSE, NASDAQ, AMEX
MAX_TICKERS_PER_EXCHANGE = 200
RATE_LIMIT = 0.001  # More conservative rate limiting
MIN_DAYS_DATA = 200
N_STATES = 3
SECTOR_SAMPLE_SIZE = 20  # Reduced for efficiency
TRANSITION_WINDOW = 30

# Global Cache
DATA_CACHE = {
    'tickers': None,
    'sector_mappings': None,
    'last_updated': None,
    'stock_data': {},
    'last_regime': None
}

def get_all_tickers():
    """Fetch tickers from all exchanges with multiple fallback strategies"""
    all_tickers = []
    
    for exchange in EXCHANGES:
        url = "https://api.polygon.io/v3/reference/tickers"
        params = {
            "exchange": exchange,
            "market": "stocks",
            "active": "true",
            "limit": MAX_TICKERS_PER_EXCHANGE,
            "apiKey": POLYGON_API_KEY
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            if 'results' in data and data['results']:
                tickers = sorted(
                    [(t['ticker'], t.get('market_cap', 0)) for t in data['results']],
                    key=lambda x: x[1],
                    reverse=True
                )
                all_tickers.extend([t[0] for t in tickers])
        except Exception as e:
            print(f"Error fetching {exchange} tickers: {str(e)}")
            all_tickers.extend(get_index_components(exchange))
    
    return list(set(all_tickers))[:MAX_TICKERS_PER_EXCHANGE * len(EXCHANGES)]

def get_index_components(exchange):
    """Fallback to index components"""
    try:
        if exchange == "XNYS":
            tables = pd.read_html("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")
            return tables[1]['Symbol'].tolist()
        elif exchange == "XNAS":
            tables = pd.read_html("https://en.wikipedia.org/wiki/NASDAQ-100")
            return tables[4]['Ticker'].tolist()
        elif exchange == "XASE":
            return ["SPY", "DIA", "QQQ"]  # ETFs as last resort
    except:
        return ["SPY", "DIA", "QQQ"]

def fetch_stock_data(symbol, days=365):
    """Get historical data with caching"""
    if symbol in DATA_CACHE['stock_data']:
        cached = DATA_CACHE['stock_data'][symbol]
        if datetime.now() - cached['timestamp'] < timedelta(days=1):
            return cached['data']
    
    time.sleep(RATE_LIMIT)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": POLYGON_API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            results = response.json().get('results', [])
            if results:
                df = pd.DataFrame(results)
                df['date'] = pd.to_datetime(df['t'], unit='ms')
                result = df.set_index('date')['c']
                DATA_CACHE['stock_data'][symbol] = {
                    'data': result,
                    'timestamp': datetime.now()
                }
                return result
    except Exception as e:
        print(f"Error fetching {symbol}: {str(e)}")
    return None

def get_market_cap(symbol):
    """Fetch current market capitalization"""
    time.sleep(RATE_LIMIT)
    url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
    params = {"apiKey": POLYGON_API_KEY}
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json().get('results', {}).get('market_cap', 0)
    except:
        return 0

class MarketRegimeAnalyzer:
    def __init__(self, n_states=3):
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=2000,
            tol=1e-4,
            random_state=42
        )
        self.state_labels = {}
        self.feature_scaler = StandardScaler()
        
    def prepare_market_data(self, tickers, sample_size=100):
        prices_data = []
        valid_tickers = []
        
        for symbol in tqdm(tickers[:sample_size]):
            prices = fetch_stock_data(symbol)
            if prices is not None and len(prices) >= MIN_DAYS_DATA:
                prices_data.append(prices)
                valid_tickers.append(symbol)
        
        if not prices_data:
            raise ValueError("Insufficient data for market composite")
        
        return pd.concat(prices_data, axis=1).mean(axis=1).dropna()

    def analyze_regime(self, index_data, n_states=None):
        n_states = n_states or self.model.n_components
        
        log_returns = np.log(index_data).diff().dropna()
        features = pd.DataFrame({
            'returns': log_returns,
            'volatility': log_returns.rolling(21).std(),
            'momentum': log_returns.rolling(14).mean()
        }).dropna()
        
        if len(features) < 60:
            raise ValueError("Insufficient feature data")
        
        scaled_features = self.feature_scaler.fit_transform(features)
        
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=2000,
            tol=1e-4,
            random_state=42
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(scaled_features)
        
        # Label states
        state_means = sorted([(i, np.mean(model.means_[i])) for i in range(model.n_components)],
                            key=lambda x: x[1])
        
        if n_states == 3:
            state_labels = {
                state_means[0][0]: 'Bear',
                state_means[1][0]: 'Neutral',
                state_means[2][0]: 'Bull'
            }
        elif n_states == 4:
            state_labels = {
                state_means[0][0]: 'Severe Bear',
                state_means[1][0]: 'Mild Bear',
                state_means[2][0]: 'Mild Bull',
                state_means[3][0]: 'Strong Bull'
            }
        else:
            state_labels = {i: f'State {i+1}' for i in range(n_states)}
        
        states = model.predict(scaled_features)
        
        return {
            'model': model,
            'regimes': [state_labels[s] for s in states],
            'probabilities': model.predict_proba(scaled_features),
            'features': features,
            'index_data': index_data[features.index[0]:],
            'state_labels': state_labels
        }

class SectorRegimeSystem:
    def __init__(self):
        self.sector_mappings = {}
        self.sector_composites = {}
        self.sector_analyzers = {}
        self.overall_analyzer = MarketRegimeAnalyzer()
        self.transition_performance = {}
        self.sector_weights = {}
        self.sector_scores = {}
        
    def map_tickers_to_sectors(self, tickers):
        """Proper sector mapping with multiple fallback strategies"""
        print("Mapping tickers to sectors...")
        self.sector_mappings = {}
        
        sector_map = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'],
            'Financial Services': ['JPM', 'BAC', 'GS', 'MS', 'V', 'MA'],
            'Healthcare': ['PFE', 'MRK', 'JNJ', 'UNH', 'ABT', 'TMO'],
            'Consumer Cyclical': ['HD', 'MCD', 'NKE', 'TSLA', 'AMZN'],
            'Communication Services': ['GOOGL', 'META', 'DIS', 'NFLX']
        }
        
        for symbol in tqdm(tickers):
            # First try manual mapping
            sector = next((k for k, v in sector_map.items() if symbol in v), None)
            
            if not sector:
                # Fallback to Polygon API
                url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
                params = {"apiKey": POLYGON_API_KEY}
                time.sleep(RATE_LIMIT)
                
                try:
                    response = requests.get(url, params=params)
                    if response.status_code == 200:
                        data = response.json().get('results', {})
                        sector = data.get('sic_description', 
                                       data.get('industry', 
                                              data.get('primary_exchange', 'Unknown')))
                except:
                    sector = "Unknown"
            
            if sector not in ['XNYS', 'XNAS', 'XASE']:  # Exclude exchanges
                self.sector_mappings.setdefault(sector, []).append(symbol)
        
        # Remove small sectors
        self.sector_mappings = {k: v for k, v in self.sector_mappings.items() 
                              if len(v) > 5 and k != 'Unknown'}
        return self.sector_mappings
    
    def calculate_sector_weights(self):
        """Compute market-cap based sector weights"""
        total_mcap = 0
        sector_mcaps = {}
        
        for sector, tickers in self.sector_mappings.items():
            sector_mcap = 0
            for symbol in tickers[:100]:  # Limit to top 100 per sector
                mcap = get_market_cap(symbol)
                sector_mcap += mcap if mcap else 0
            sector_mcaps[sector] = sector_mcap
            total_mcap += sector_mcap
        
        if total_mcap == 0:  # Fallback equal weighting
            total_mcap = len(sector_mcaps)
            sector_mcaps = {k: 1 for k in sector_mcaps}
        
        self.sector_weights = {sector: mcap/total_mcap 
                             for sector, mcap in sector_mcaps.items()}
        return self.sector_weights
    
    def build_sector_composites(self, sample_size=20):
        """Create sector composite indexes"""
        print("\nBuilding sector composites...")
        self.sector_composites = {}
        
        for sector, tickers in tqdm(self.sector_mappings.items()):
            prices_data = []
            
            for symbol in tickers[:sample_size]:
                prices = fetch_stock_data(symbol)
                if prices is not None and len(prices) >= MIN_DAYS_DATA:
                    prices_data.append(prices)
            
            if prices_data:
                self.sector_composites[sector] = pd.concat(prices_data, axis=1).mean(axis=1).dropna()
    
    def analyze_sector_regimes(self, n_states=3):
        """Run regime analysis for each sector"""
        print("\nAnalyzing sector regimes...")
        self.sector_analyzers = {}
        
        for sector, composite in tqdm(self.sector_composites.items()):
            analyzer = MarketRegimeAnalyzer()
            try:
                results = analyzer.analyze_regime(composite, n_states=n_states)
                self.sector_analyzers[sector] = {
                    'results': results,
                    'composite': composite,
                    'volatility': composite.pct_change().std()
                }
            except Exception as e:
                print(f"Error analyzing {sector}: {str(e)}")
    
    def calculate_sector_scores(self, market_regime):
        """Generate sector scores"""
        self.sector_scores = {}
        
        for sector, data in self.sector_analyzers.items():
            try:
                probs = data['results']['probabilities'][-1]
                labels = data['results']['state_labels']
                
                bull_prob = sum(probs[i] for i, label in labels.items() if 'Bull' in label)
                bear_prob = sum(probs[i] for i, label in labels.items() if 'Bear' in label)
                momentum = data['composite'].pct_change(21).iloc[-1] if len(data['composite']) > 21 else 0
                
                base_score = bull_prob - bear_prob + (momentum * 10)
                
                # Apply regime adjustments
                if market_regime == "Bull":
                    base_score *= 1 + (data['volatility'] / 0.02)
                elif market_regime == "Bear":
                    base_score *= 1 + (0.04 - min(data['volatility'], 0.04)) / 0.02
                
                # Apply weighting
                weight = self.sector_weights.get(sector, 0)
                self.sector_scores[sector] = base_score * (1 + weight * 0.5)
            except:
                self.sector_scores[sector] = 0
        
        return pd.Series(self.sector_scores).sort_values(ascending=False)
    
    def generate_entry_signals(self, market_regime):
        """Generate sector entry signals"""
        entry_signals = {}
        
        for sector, score in self.sector_scores.items():
            try:
                sector_data = self.sector_analyzers[sector]
                sector_regime = sector_data['results']['regimes'][-1]
                
                bull_aligned = 'Bull' in market_regime and 'Bull' in sector_regime
                bear_aligned = 'Bear' in market_regime and 'Bear' in sector_regime
                
                if bull_aligned:
                    if score > 0.6: entry_signals[sector] = "STRONG BUY"
                    elif score > 0.4: entry_signals[sector] = "BUY"
                    else: entry_signals[sector] = "NEUTRAL"
                elif bear_aligned:
                    if score < -0.6: entry_signals[sector] = "STRONG SHORT"
                    elif score < -0.4: entry_signals[sector] = "SHORT"
                    else: entry_signals[sector] = "NEUTRAL"
                else:
                    entry_signals[sector] = "NEUTRAL"
            except:
                entry_signals[sector] = "NEUTRAL"
        
        return entry_signals
    
    def plot_sector_performance(self):
        """Save sector performance plot"""
        if not self.sector_scores:
            return
            
        try:
            scores = pd.Series(self.sector_scores).sort_values()
            signals = self.generate_entry_signals(DATA_CACHE.get('last_regime', "UNKNOWN"))
            
            plt.figure(figsize=(14, 10))
            colors = {
                "STRONG BUY": "darkgreen",
                "BUY": "lightgreen",
                "NEUTRAL": "gray",
                "SHORT": "lightcoral",
                "STRONG SHORT": "darkred"
            }
            bar_colors = [colors.get(signals.get(s, "NEUTRAL"), "gray") for s in scores.index]
            
            ax = scores.plot(kind='barh', color=bar_colors)
            ax.set_title("Sector Performance")
            ax.axvline(0, color='black', linestyle='--')
            
            for i, (score, sector) in enumerate(zip(scores, scores.index)):
                ax.text(score, i, f" {signals.get(sector, 'NEUTRAL')}", 
                       va='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f"sector_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.close()
        except Exception as e:
            print(f"Plotting error: {str(e)}")

def continuous_market_analysis():
    """Main analysis loop"""
    sector_system = SectorRegimeSystem()
    last_refresh = datetime.min
    params = {'N_STATES': 3, 'SAMPLE_SIZE': 20, 'MAX_TICKERS': 150}
    
    while True:
        try:
            print(f"\n=== Analysis Cycle {datetime.now()} ===")
            
            # Refresh data weekly
            if (datetime.now() - last_refresh) > timedelta(days=7):
                tickers = get_all_tickers()
                DATA_CACHE['tickers'] = tickers
                sector_system.map_tickers_to_sectors(tickers)
                sector_system.calculate_sector_weights()
                last_refresh = datetime.now()
            
            # Build and analyze
            sector_system.build_sector_composites(params['SAMPLE_SIZE'])
            market_data = sector_system.overall_analyzer.prepare_market_data(
                DATA_CACHE['tickers'][:params['MAX_TICKERS']])
            market_results = sector_system.overall_analyzer.analyze_regime(
                market_data, params['N_STATES'])
            sector_system.analyze_sector_regimes(params['N_STATES'])
            
            # Generate outputs
            current_regime = market_results['regimes'][-1]
            DATA_CACHE['last_regime'] = current_regime
            sector_scores = sector_system.calculate_sector_scores(current_regime)
            entry_signals = sector_system.generate_entry_signals(current_regime)
            
            print(f"\nMarket Regime: {current_regime}")
            print("\nTop Sectors:")
            print(sector_scores.head(10))
            print("\nSignals:")
            print("\n".join(f"{k}: {v}" for k, v in entry_signals.items() if v != "NEUTRAL"))
            
            # Save plots
            sector_system.plot_sector_performance()
            
            # Adjust parameters
            if market_results['features']['volatility'].iloc[-1] > 0.04:
                params['N_STATES'] = 4
            
            time.sleep(3600)  # Run hourly
            
        except KeyboardInterrupt:
            print("\nStopping analysis...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            time.sleep(300)

if __name__ == "__main__":
    continuous_market_analysis()