import requests
import pandas as pd
import numpy as np
from hmmlearn import hmm
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bars
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from config import POLYGON_API_KEY  # Import from config.py

# Configuration
EXCHANGES = ["XNYS", "XNAS", "XASE"]  # NYSE, NASDAQ, AMEX
MAX_TICKERS_PER_EXCHANGE = 200  # Reduced per exchange to stay within limits
RATE_LIMIT = .001  # seconds between requests
MIN_DAYS_DATA = 200  # Minimum days of data required for analysis
N_STATES = 3  # Bull/Neutral/Bear regimes

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
                return df.set_index('date')['c']
    except Exception as e:
        print(f"Error fetching {symbol}: {str(e)}")
    return None

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

    def analyze_regime(self, market_index):
        """Analyze market regime using HMM"""
        # Calculate features
        log_returns = np.log(market_index).diff().dropna()
        features = pd.DataFrame({
            'returns': log_returns,
            'volatility': log_returns.rolling(21).std(),
            'momentum': log_returns.rolling(14).mean()
        }).dropna()
        
        if len(features) < 60:
            raise ValueError(f"Only {len(features)} days of feature data")
        
        # Scale features and fit model
        scaled_features = self.feature_scaler.fit_transform(features)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model.fit(scaled_features)
        
        # Label states
        state_means = sorted([
            (i, np.mean(self.model.means_[i])) 
            for i in range(self.model.n_components)
        ], key=lambda x: x[1])
        
        self.state_labels = {
            state_means[0][0]: 'Bear',
            state_means[1][0]: 'Neutral',
            state_means[2][0]: 'Bull'
        }
        
        # Predict regimes
        states = self.model.predict(scaled_features)
        state_probs = self.model.predict_proba(scaled_features)
        
        return {
            'regimes': [self.state_labels[s] for s in states],
            'probabilities': state_probs,
            'features': features,
            'market_index': market_index[features.index[0]:]
        }

def analyze_market_regime():
    """Main function to analyze market regime"""
    # Get tickers
    tickers = get_all_tickers()
    print(f"Found {len(tickers)} total tickers for analysis")
    
    # Create analyzer instance
    analyzer = MarketRegimeAnalyzer(n_states=3)
    
    try:
        # Prepare market data
        market_index = analyzer.prepare_market_data(tickers)
        if len(market_index) < MIN_DAYS_DATA:
            raise ValueError(f"Need at least {MIN_DAYS_DATA} days of data")
            
        # Analyze regime
        results = analyzer.analyze_regime(market_index)
        
        # Display results
        current_regime = results['regimes'][-1]
        current_probs = results['probabilities'][-1]
        
        print(f"\nCurrent Market Regime: {current_regime}")
        print("Current Probabilities:")
        for i, prob in enumerate(current_probs):
            print(f"{analyzer.state_labels[i]}: {prob:.1%}")
        
        print("\nRegime History (last 20 periods):")
        print(pd.Series(results['regimes'][-20:]).value_counts())
        
        # Plot results
        plt.figure(figsize=(14, 8))
        
        # Price plot
        ax1 = plt.subplot(2, 1, 1)
        results['market_index'][-250:].plot(ax=ax1, color='black')
        ax1.set_title('Market Index with Regime Shading')
        
        # Regime shading
        regime_colors = {'Bull': 'green', 'Neutral': 'yellow', 'Bear': 'red'}
        for i in range(-250, 0):
            if i < -len(results['regimes']):
                continue
            ax1.axvspan(
                results['market_index'].index[i],
                results['market_index'].index[i+1] if i < -1 else results['market_index'].index[-1],
                color=regime_colors.get(results['regimes'][i], 'gray'),
                alpha=0.1
            )
        
        # Probabilities plot
        ax2 = plt.subplot(2, 1, 2)
        prob_df = pd.DataFrame(
            results['probabilities'][-250:],
            index=results['market_index'].index[-250:],
            columns=[analyzer.state_labels[i] for i in range(3)]
        )
        prob_df.plot(ax=ax2, color=['red', 'yellow', 'green'])
        ax2.set_title('Regime Probabilities')
        
        plt.tight_layout()
        plt.show()
        
        return results
        
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        return None

if __name__ == "__main__":
    analyze_market_regime()