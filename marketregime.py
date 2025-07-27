import requests
import pandas as pd
import numpy as np
from hmmlearn import hmm
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from config import POLYGON_API_KEY  # Import from config.py

# Configuration
EXCHANGES = ["XNYS", "XNAS", "XASE"]  # NYSE, NASDAQ, AMEX
MAX_TICKERS_PER_EXCHANGE = 200  # Reduced per exchange to stay within limits
RATE_LIMIT = 0.001  # seconds between requests
MIN_DAYS_DATA = 200  # Minimum days of data required for analysis
N_STATES = 3  # Bull/Neutral/Bear regimes
SECTOR_SAMPLE_SIZE = 50  # Stocks per sector for composite
TRANSITION_WINDOW = 30  # Days to analyze around regime transitions

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
        # Primary method - exchange listing
        url = "https://api.polygon.io/v3/reference/tickers"
        params = {
            "exchange": exchange,
            "market": "stocks",
            "active": "true",
            "limit": MAX_TICKERS_PER_EXCHANGE,
            "apiKey": POLYGON_API_KEY
        }
        
        try:
            print(f"Fetching {exchange} tickers...")
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'results' in data and data['results']:
                # Get most liquid stocks (sorted by descending market cap)
                tickers = sorted(
                    [(t['ticker'], t.get('market_cap', 0)) for t in data['results']],
                    key=lambda x: x[1],
                    reverse=True
                )
                exchange_tickers = [t[0] for t in tickers]
                all_tickers.extend(exchange_tickers)
                print(f"Found {len(exchange_tickers)} tickers for {exchange}")
        except Exception as e:
            print(f"API error for {exchange}: {str(e)}")
            # Fallback to index components if primary method fails
            print(f"Using index components as proxy for {exchange}")
            all_tickers.extend(get_index_components(exchange))
    
    # Remove duplicates and limit total tickers
    unique_tickers = list(set(all_tickers))
    return unique_tickers[:MAX_TICKERS_PER_EXCHANGE * len(EXCHANGES)]

def get_index_components(exchange):
    """Fallback: Get index components based on exchange"""
    try:
        if exchange == "XNYS":
            tables = pd.read_html("https://en.wikipedia.org/wiki/NYSE_Composite")
            return tables[2]['Symbol'].tolist()[:MAX_TICKERS_PER_EXCHANGE]
        elif exchange == "XNAS":
            tables = pd.read_html("https://en.wikipedia.org/wiki/NASDAQ-100")
            return tables[4]['Ticker'].tolist()[:MAX_TICKERS_PER_EXCHANGE]
        elif exchange == "XASE":
            tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_American_Stock_Exchange_companies")
            return tables[0]['Symbol'].tolist()[:MAX_TICKERS_PER_EXCHANGE]
    except:
        return ["DIA", "SPY", "QQQ"]  # Broad market ETFs as last resort

def fetch_stock_data(symbol, days=365):
    """Get historical data with caching and retry logic"""
    if RATE_LIMIT:
        time.sleep(RATE_LIMIT)
    
    # Check cache first
    if symbol in DATA_CACHE['stock_data']:
        cached_data = DATA_CACHE['stock_data'][symbol]
        if datetime.now() - cached_data['timestamp'] < timedelta(days=1):
            return cached_data['data']
    
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
                # Update cache
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
    url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
    params = {"apiKey": POLYGON_API_KEY}
    time.sleep(RATE_LIMIT)
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json().get('results', {}).get('market_cap', 0)
    except:
        return 0
    return 0

class MarketRegimeAnalyzer:
    def __init__(self, n_states=3):
        """Initialize HMM model for regime detection"""
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=2000,
            tol=1e-4,
            init_params='stmc',
            params='stmc',
            random_state=42
        )
        self.state_labels = {}
        self.feature_scaler = StandardScaler()
        
    def prepare_market_data(self, tickers, sample_size=100):
        """Create composite market index from sample of stocks"""
        prices_data = []
        valid_tickers = []
        
        print("\nBuilding market composite from multiple exchanges...")
        for symbol in tqdm(tickers[:sample_size]):
            prices = fetch_stock_data(symbol)
            if prices is not None and len(prices) >= MIN_DAYS_DATA:
                prices_data.append(prices)
                valid_tickers.append(symbol)
        
        if not prices_data:
            raise ValueError("Insufficient data to create market composite")
        
        composite = pd.concat(prices_data, axis=1)
        composite.columns = valid_tickers
        return composite.mean(axis=1).dropna()

    def analyze_regime(self, index_data, n_states=None):
        """Analyze regime using HMM with adaptive states"""
        if n_states is None:
            n_states = self.model.n_components
        
        # Calculate features
        log_returns = np.log(index_data).diff().dropna()
        features = pd.DataFrame({
            'returns': log_returns,
            'volatility': log_returns.rolling(21).std(),
            'momentum': log_returns.rolling(14).mean()
        }).dropna()
        
        if len(features) < 60:
            raise ValueError(f"Only {len(features)} days of feature data")
        
        # Scale features
        scaled_features = self.feature_scaler.fit_transform(features)
        
        # Create model with potentially different state count
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=2000,
            tol=1e-4,
            random_state=42
        )
        
        # Fit model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(scaled_features)
        
        # Label states
        state_means = sorted([
            (i, np.mean(model.means_[i])) 
            for i in range(model.n_components)
        ], key=lambda x: x[1])
        
        state_labels = {}
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
            for i in range(n_states):
                state_labels[i] = f'State {i+1}'
        
        # Predict regimes
        states = model.predict(scaled_features)
        state_probs = model.predict_proba(scaled_features)
        
        return {
            'model': model,
            'regimes': [state_labels[s] for s in states],
            'probabilities': state_probs,
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
        """Assign tickers to sectors using Polygon API with robust error handling"""
        print("Mapping tickers to sectors...")
        self.sector_mappings = {}
        
        for symbol in tqdm(tickers):
            url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
            params = {"apiKey": POLYGON_API_KEY}
            time.sleep(RATE_LIMIT)
            
            try:
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json().get('results', {})
                    sector = data.get('sic_description', 'Unknown')
                    if sector == 'Unknown':
                        sector = data.get('primary_exchange', 'Unknown')
                    self.sector_mappings.setdefault(sector, []).append(symbol)
            except Exception as e:
                print(f"Sector mapping failed for {symbol}: {str(e)}")
        
        # Remove unknown sectors and small sectors
        self.sector_mappings = {k: v for k, v in self.sector_mappings.items() 
                              if k != 'Unknown' and len(v) > 10}
        return self.sector_mappings
    
    def calculate_sector_weights(self):
        """Compute market-cap based sector weights with error handling"""
        total_mcap = 0
        sector_mcaps = {}
        
        for sector, tickers in self.sector_mappings.items():
            sector_mcap = 0
            for symbol in tickers[:100]:  # Limit to top 100 per sector
                try:
                    mcap = get_market_cap(symbol)
                    sector_mcap += mcap if mcap else 0
                except Exception as e:
                    print(f"Error getting market cap for {symbol}: {str(e)}")
            
            sector_mcaps[sector] = sector_mcap
            total_mcap += sector_mcap
        
        self.sector_weights = {sector: mcap/total_mcap if total_mcap > 0 else 1/len(sector_mcaps)
                             for sector, mcap in sector_mcaps.items()}
        return self.sector_weights
    
    def build_sector_composites(self, sample_size=50):
        """Create sector composite indexes with adjustable sample size"""
        print("\nBuilding sector composites...")
        self.sector_composites = {}
        
        for sector, tickers in tqdm(self.sector_mappings.items()):
            prices_data = []
            valid_tickers = []
            
            for symbol in tickers[:sample_size]:
                try:
                    prices = fetch_stock_data(symbol)
                    if prices is not None and len(prices) >= MIN_DAYS_DATA:
                        prices_data.append(prices)
                        valid_tickers.append(symbol)
                except Exception as e:
                    print(f"Error processing {symbol}: {str(e)}")
            
            if prices_data:
                composite = pd.concat(prices_data, axis=1)
                composite.columns = valid_tickers
                self.sector_composites[sector] = composite.mean(axis=1).dropna()
    
    def analyze_sector_regimes(self, n_states=3):
        """Run regime analysis for each sector with robust error handling"""
        print("\nAnalyzing sector regimes...")
        self.sector_analyzers = {}
        
        for sector, composite in tqdm(self.sector_composites.items()):
            try:
                analyzer = MarketRegimeAnalyzer()
                results = analyzer.analyze_regime(composite, n_states=n_states)
                
                self.sector_analyzers[sector] = {
                    'results': results,
                    'composite': composite,
                    'volatility': composite.pct_change().std()
                }
            except Exception as e:
                print(f"Error analyzing {sector}: {str(e)}")
                continue
    
    def calculate_sector_scores(self, market_regime):
        """Generate sector scores with comprehensive error handling"""
        self.sector_scores = {}
        
        if not self.sector_analyzers:
            print("Warning: No sector analyzers available")
            return pd.Series()
        
        for sector, data in self.sector_analyzers.items():
            try:
                if 'results' not in data or 'probabilities' not in data['results']:
                    continue
                    
                current_probs = data['results']['probabilities'][-1]
                state_labels = data['results'].get('state_labels', {})
                
                # Calculate bull/bear probabilities
                bull_prob = sum(current_probs[i] for i, label in state_labels.items() 
                              if 'Bull' in label) if state_labels else 0
                bear_prob = sum(current_probs[i] for i, label in state_labels.items() 
                              if 'Bear' in label) if state_labels else 0
                
                # Calculate momentum
                momentum = data['composite'].pct_change(21).iloc[-1] if len(data['composite']) > 21 else 0
                
                # Base score
                base_score = bull_prob - bear_prob + (momentum * 10)
                
                # Apply regime-specific adjustments
                if market_regime == "Bull":
                    beta_factor = 1 + (data['volatility'] / 0.02)
                    base_score *= beta_factor
                elif market_regime == "Bear":
                    volatility_factor = 1 + (0.04 - min(data['volatility'], 0.04)) / 0.02
                    base_score *= volatility_factor
                
                # Apply market cap weighting
                weight = self.sector_weights.get(sector, 0)
                self.sector_scores[sector] = base_score * (1 + weight * 0.5)
                
            except Exception as e:
                print(f"Error calculating score for {sector}: {str(e)}")
                self.sector_scores[sector] = 0
        
        return pd.Series(self.sector_scores).sort_values(ascending=False)
    
    def detect_regime_transitions(self, market_results):
        """Track sector performance during regime shifts with error handling"""
        self.transition_performance = {}
        
        if 'regimes' not in market_results or 'index_data' not in market_results:
            return self.transition_performance
            
        market_regimes = market_results['regimes']
        transitions = []
        
        # Identify transition points
        for i in range(1, len(market_regimes)):
            if market_regimes[i] != market_regimes[i-1]:
                transitions.append(market_results['index_data'].index[i])
        
        # Analyze sector performance around transitions
        for transition_date in transitions[-5:]:  # Last 5 transitions
            transition_perf = {}
            
            for sector, data in self.sector_analyzers.items():
                try:
                    sector_index = data['composite']
                    if transition_date in sector_index.index:
                        idx = sector_index.index.get_loc(transition_date)
                        
                        if idx > TRANSITION_WINDOW and idx + TRANSITION_WINDOW < len(sector_index):
                            pre_perf = sector_index.iloc[idx] / sector_index.iloc[idx-TRANSITION_WINDOW] - 1
                            post_perf = sector_index.iloc[idx+TRANSITION_WINDOW] / sector_index.iloc[idx] - 1
                            transition_perf[sector] = {
                                'pre_period': pre_perf,
                                'post_period': post_perf,
                                'outperformance': post_perf - pre_perf
                            }
                except Exception as e:
                    print(f"Error analyzing transition for {sector}: {str(e)}")
                    continue
            
            if transition_perf:
                self.transition_performance[transition_date] = pd.DataFrame(transition_perf).T
        
        return self.transition_performance
    
    def generate_entry_signals(self, market_regime):
        """Generate sector entry signals with comprehensive error handling"""
        entry_signals = {}
        
        if not self.sector_scores:
            print("Warning: No sector scores available - returning neutral signals")
            return {sector: "NEUTRAL" for sector in self.sector_mappings.keys()}
        
        for sector, score in self.sector_scores.items():
            try:
                if sector not in self.sector_analyzers:
                    entry_signals[sector] = "NEUTRAL"
                    continue
                    
                sector_data = self.sector_analyzers[sector]
                sector_regime = sector_data['results'].get('regimes', [""])[-1] if 'results' in sector_data else "UNKNOWN"
                
                # Determine thresholds based on market regime confidence
                if market_regime == "UNKNOWN":
                    bull_threshold = 0.8
                    bear_threshold = -0.8
                else:
                    bull_threshold = 0.6 if 'Bull' in market_regime else 0.4
                    bear_threshold = -0.4 if 'Bear' in market_regime else -0.6
                
                # Generate signals
                if 'Bull' in market_regime and 'Bull' in sector_regime:
                    if score > bull_threshold:
                        entry_signals[sector] = "STRONG BUY"
                    elif score > (bull_threshold - 0.2):
                        entry_signals[sector] = "BUY"
                    else:
                        entry_signals[sector] = "NEUTRAL"
                elif 'Bear' in market_regime and 'Bear' in sector_regime:
                    if score < bear_threshold:
                        entry_signals[sector] = "STRONG SHORT"
                    elif score < (bear_threshold + 0.2):
                        entry_signals[sector] = "SHORT"
                    else:
                        entry_signals[sector] = "NEUTRAL"
                else:
                    entry_signals[sector] = "NEUTRAL"
                    
            except Exception as e:
                print(f"Error generating signal for {sector}: {str(e)}")
                entry_signals[sector] = "NEUTRAL"
        
        return entry_signals
    
    def plot_sector_performance(self):
        """Save sector performance plot to file with robust error handling"""
        if not self.sector_scores:
            print("No sector scores available to plot")
            return
            
        try:
            scores = pd.Series(self.sector_scores).sort_values()
            
            # Safely get current regime
            current_regime = "UNKNOWN"
            if (hasattr(self.overall_analyzer, 'features') and 
                hasattr(self.overall_analyzer, 'model') and
                hasattr(self.overall_analyzer, 'state_labels')):
                try:
                    if not self.overall_analyzer.features.empty:
                        current_state = self.overall_analyzer.model.predict(
                            self.overall_analyzer.feature_scaler.transform(
                                self.overall_analyzer.features.iloc[-1:].values
                            )
                        )[0]
                        current_regime = self.overall_analyzer.state_labels.get(current_state, "UNKNOWN")
                except:
                    pass
                    
            signals = self.generate_entry_signals(current_regime)
            
            # Create plot
            plt.figure(figsize=(14, 10))
            colors = []
            for sector in scores.index:
                signal = signals.get(sector, "NEUTRAL")
                if signal == "STRONG BUY":
                    colors.append("darkgreen")
                elif signal == "BUY":
                    colors.append("lightgreen")
                elif signal == "STRONG SHORT":
                    colors.append("darkred")
                elif signal == "SHORT":
                    colors.append("lightcoral")
                else:
                    colors.append("gray")
            
            ax = scores.plot(kind='barh', color=colors)
            ax.set_title(f"Sector Scores with Entry Signals (Market: {current_regime})")
            ax.set_xlabel("Composite Score")
            ax.axvline(0, color='black', linestyle='--')
            
            # Add labels
            for i, (score, sector) in enumerate(zip(scores, scores.index)):
                ax.text(score, i, f" {signals.get(sector, 'NEUTRAL')}", 
                       va='center', fontweight='bold')
            
            plt.tight_layout()
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sector_performance_{timestamp}.png"
            plt.savefig(filename)
            print(f"Sector performance plot saved to {filename}")
            plt.close()
            
        except Exception as e:
            print(f"Error generating performance plot: {str(e)}")
            if 'plt' in locals():
                plt.close()
    
    def plot_transition_performance(self):
        """Save transition performance plots to files with error handling"""
        if not self.transition_performance:
            return
            
        try:
            for date, perf_df in self.transition_performance.items():
                perf_df = perf_df.sort_values('outperformance', ascending=True)
                
                plt.figure(figsize=(14, 8))
                perf_df['outperformance'].plot(kind='barh', color='steelblue')
                plt.title(f"Sector Outperformance During {date.strftime('%Y-%m-%d')} Transition")
                plt.xlabel("Post-Transition Outperformance vs Pre-Transition")
                plt.axvline(0, color='black', linestyle='--')
                plt.tight_layout()
                
                # Save to file
                filename = f"transition_performance_{date.strftime('%Y%m%d')}.png"
                plt.savefig(filename)
                print(f"Transition performance plot saved to {filename}")
                plt.close()
                
        except Exception as e:
            print(f"Error generating transition plot: {str(e)}")
            if 'plt' in locals():
                plt.close()

def adjust_parameters(system, market_results, current_params):
    """Dynamically adjust parameters based on market conditions"""
    new_params = current_params.copy()
    volatility = market_results['features']['volatility'].iloc[-1]
    
    print("\nAdjusting parameters based on market conditions...")
    
    # Adjust state complexity based on volatility
    if volatility > 0.04:
        new_params['N_STATES'] = 4  # More states during high volatility
        print(f"Increased states to 4 (Volatility: {volatility:.4f})")
    else:
        new_params['N_STATES'] = 3
        print(f"Using normal 3-state model (Volatility: {volatility:.4f})")
    
    # Adjust sample size based on regime stability
    if DATA_CACHE.get('last_regime') == market_results['regimes'][-1]:
        new_params['SECTOR_SAMPLE_SIZE'] = min(10, new_params['SECTOR_SAMPLE_SIZE'] + 1)
        print(f"Increased sector sample to {new_params['SECTOR_SAMPLE_SIZE']} (Stable regime)")
    else:
        new_params['SECTOR_SAMPLE_SIZE'] = max(3, new_params['SECTOR_SAMPLE_SIZE'] - 1)
        print(f"Decreased sector sample to {new_params['SECTOR_SAMPLE_SIZE']} (Regime change detected)")
    
    # Adjust ticker count based on recent errors
    if system.overall_analyzer.model.monitor_.history[-1]['converged']:
        new_params['MAX_TICKERS'] = min(200, new_params['MAX_TICKERS'] + 10)
        print(f"Increased ticker limit to {new_params['MAX_TICKERS']} (Model converged)")
    else:
        new_params['MAX_TICKERS'] = max(50, new_params['MAX_TICKERS'] - 10)
        print(f"Decreased ticker limit to {new_params['MAX_TICKERS']} (Convergence issues)")
    
    return new_params

def continuous_market_analysis():
    """Continuously running analysis with dynamic adjustments"""
    # Initialize system
    sector_system = SectorRegimeSystem()
    last_ticker_refresh = datetime.min
    last_param_adjust = datetime.now()
    param_update_interval = timedelta(hours=12)
    
    # Dynamic parameters
    current_params = {
        'N_STATES': 3,
        'SECTOR_SAMPLE_SIZE': 5,
        'MAX_TICKERS': 100
    }
    
    while True:
        try:
            print(f"\n{'='*40}\nStarting new analysis cycle at {datetime.now()}\n{'='*40}")
            
            # Refresh tickers periodically (weekly)
            if (datetime.now() - last_ticker_refresh) > timedelta(days=7):
                print("Refreshing ticker list...")
                tickers = get_all_tickers()
                DATA_CACHE['tickers'] = tickers
                last_ticker_refresh = datetime.now()
            else:
                tickers = DATA_CACHE['tickers'] or get_all_tickers()
            
            # Apply current parameter limits
            tickers = tickers[:current_params['MAX_TICKERS']]
            
            # Update sector mappings
            if not DATA_CACHE['sector_mappings']:
                print("Mapping sectors...")
                sector_system.map_tickers_to_sectors(tickers)
                DATA_CACHE['sector_mappings'] = sector_system.sector_mappings
            else:
                sector_system.sector_mappings = DATA_CACHE['sector_mappings']
            
            # Update sector weights
            sector_system.calculate_sector_weights()
            
            # Build composites with current sample size
            print("Building composites...")
            sector_system.build_sector_composites(
                sample_size=current_params['SECTOR_SAMPLE_SIZE']
            )
            
            # Analyze overall market with current state count
            market_index = sector_system.overall_analyzer.prepare_market_data(tickers)
            market_results = sector_system.overall_analyzer.analyze_regime(
                market_index, 
                n_states=current_params['N_STATES']
            )
            
            # Analyze sectors
            sector_system.analyze_sector_regimes(
                n_states=current_params['N_STATES']
            )
            
            # Generate outputs
            current_regime = market_results['regimes'][-1]
            sector_scores = sector_system.calculate_sector_scores(current_regime)
            entry_signals = sector_system.generate_entry_signals(current_regime)
            
            # Print current status
            print(f"\nCurrent Market Regime: {current_regime}")
            print("\nTop Sectors by Score:")
            print(sector_scores.head(10))
            print("\nEntry Signals:")
            for sector, signal in entry_signals.items():
                if signal != "NEUTRAL":
                    print(f"{sector}: {signal}")
            
            # Dynamic parameter adjustment
            if (datetime.now() - last_param_adjust) > param_update_interval:
                current_params = adjust_parameters(
                    sector_system,
                    market_results,
                    current_params
                )
                last_param_adjust = datetime.now()
                
            # Generate plots (will save to files automatically)
            sector_system.plot_sector_performance()
            sector_system.plot_transition_performance()
            
            # Cache current regime for next comparison
            DATA_CACHE['last_regime'] = current_regime
            
            # Sleep until next cycle (1 hour)
            print(f"\nCycle complete. Sleeping for 1 hour...")
            time.sleep(3600)
            
        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt. Shutting down gracefully...")
            break
        except Exception as e:
            print(f"Critical error: {str(e)}")
            print("Restarting in 5 minutes...")
            time.sleep(300)

if __name__ == "__main__":
    continuous_market_analysis()  # Start continuous loop