import os
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from threading import Thread, Lock
from queue import Queue
from websocket import create_connection, WebSocketConnectionClosedException
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ratelimit import limits, sleep_and_retry

# Import API key from config.py
try:
    from config import POLYGON_API_KEY
except ImportError:
    print("Error: config.py file not found. Please create it with your Polygon API key")
    exit(1)
except AttributeError:
    print("Error: POLYGON_API_KEY not defined in config.py")
    exit(1)

# Validate API key
if not POLYGON_API_KEY or POLYGON_API_KEY == "YOUR_API_KEY_HERE":
    print("ERROR: Invalid Polygon API key. Please set your actual API key in config.py")
    exit(1)

# Configuration
EXCHANGES = ["XNYS", "XNAS", "XASE"]
MAX_TICKERS = 500
N_STATES = 3
SECTOR_SAMPLE_SIZE = 20
TREND_LOOKBACK = 365
MIN_DAYS_DATA = 200
RATE_LIMIT = 0.001

# Global state with thread safety
class GlobalState:
    def __init__(self):
        self.lock = Lock()
        self.ticker_cache = pd.DataFrame()
        self.last_refresh = datetime.min.replace(tzinfo=timezone.utc)
        self.stock_data = {}
        self.market_regime = None
        self.sector_scores = {}
        self.top_tickers = []
        
    def update_tickers(self, new_tickers):
        with self.lock:
            self.ticker_cache = new_tickers
            self.last_refresh = datetime.now(timezone.utc)
            
    def get_tickers(self):
        with self.lock:
            return self.ticker_cache.copy()

GLOBAL_STATE = GlobalState()

# Ticker Scanner Module
class PolygonTickerScanner:
    def __init__(self, api_key=POLYGON_API_KEY):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v3/reference"
        self.websocket_url = "wss://socket.polygon.io/stocks"
        self.active = False
        self.event_queue = Queue(maxsize=10000)
        self.ws_thread = None
        self.process_thread = None

    @sleep_and_retry
    @limits(calls=5, period=60)
    def _call_polygon_api(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                print("ERROR: Invalid Polygon API key. Please verify your credentials in config.py")
            elif response.status_code == 429:
                print("WARNING: API rate limit exceeded. Sleeping before retry...")
                time.sleep(60)
            else:
                print(f"API error: {e}")
            return None
        except Exception as e:
            print(f"Network error: {e}")
            return None

    def refresh_tickers(self):
        all_results = []
        for exchange in EXCHANGES:
            url = f"{self.base_url}/tickers?market=stocks&exchange={exchange}&active=true&limit=1000&apiKey={self.api_key}"
            while url:
                data = self._call_polygon_api(url)
                if not data:
                    break
                    
                all_results.extend(data.get("results", []))
                url = data.get("next_url", None)
                if url:
                    url += f"&apiKey={self.api_key}"
                    time.sleep(0.2)

        if all_results:
            df = pd.DataFrame(all_results)
            tickers = df[["ticker", "name", "primary_exchange", "last_updated_utc"]].copy()
            GLOBAL_STATE.update_tickers(tickers)
            print(f"Refreshed {len(tickers)} tickers")
            return True
        else:
            print("Warning: No tickers fetched. Using previous data")
            return False

    def _websocket_listener(self):
        while self.active:
            try:
                ws = create_connection(self.websocket_url, timeout=10)
                ws.send(json.dumps({"action": "auth", "params": self.api_key}))
                ws.send(json.dumps({"action": "subscribe", "params": "T.*,A.*,AM.*"}))
                
                last_ping = time.time()
                while self.active:
                    try:
                        data = ws.recv()
                        if data:
                            self.event_queue.put(json.loads(data))
                            
                        if time.time() - last_ping > 30:
                            ws.ping()
                            last_ping = time.time()
                            
                    except (WebSocketConnectionClosedException, ConnectionResetError):
                        break
            except Exception as e:
                print(f"WebSocket error: {e}")
                time.sleep(5)
            finally:
                if 'ws' in locals():
                    try:
                        ws.close()
                    except:
                        pass

    def _process_events(self):
        while self.active:
            if not self.event_queue.empty():
                event = self.event_queue.get()
                if isinstance(event, list):
                    for e in event:
                        self._handle_event(e)
                else:
                    self._handle_event(event)
            time.sleep(0.1)

    def _handle_event(self, event):
        if event.get("ev") == "T":  # Trade event
            ticker = event["sym"]
            current_tickers = GLOBAL_STATE.get_tickers()
            if ticker not in current_tickers["ticker"].values:
                print(f"New ticker detected: {ticker}. Refreshing...")
                self.refresh_tickers()

    def start(self):
        if not self.active:
            self.active = True
            if not self.refresh_tickers():  # Initial refresh
                # Fallback to sample tickers if API fails
                sample_tickers = pd.DataFrame({
                    "ticker": ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "JPM", "JNJ", "V", "WMT"],
                    "name": ["Apple", "Microsoft", "Alphabet", "Amazon", "Meta", "Tesla", "JPMorgan", "Johnson&Johnson", "Visa", "Walmart"],
                    "primary_exchange": ["XNAS"]*6 + ["XNYS"]*4,
                    "last_updated_utc": [datetime.now(timezone.utc)]*10
                })
                GLOBAL_STATE.update_tickers(sample_tickers)
                print("Using sample tickers as fallback")
                
            self.ws_thread = Thread(target=self._websocket_listener, daemon=True)
            self.process_thread = Thread(target=self._process_events, daemon=True)
            self.ws_thread.start()
            self.process_thread.start()
            print("Ticker scanner started")

    def stop(self):
        self.active = False
        if self.ws_thread:
            self.ws_thread.join(timeout=2)
        if self.process_thread:
            self.process_thread.join(timeout=2)
        print("Ticker scanner stopped")

# Market Analysis Module
class MarketRegimeAnalyzer:
    def __init__(self, n_states=3):
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=2000,
            tol=1e-4,
            random_state=42
        )
        self.feature_scaler = StandardScaler()

    def prepare_market_data(self, tickers):
        prices_data = []
        for symbol in tickers[:min(MAX_TICKERS, len(tickers))]:
            prices = self.fetch_stock_data(symbol)
            if prices is not None and len(prices) >= MIN_DAYS_DATA:
                prices_data.append(prices)
                
        if not prices_data:
            print("ERROR: No valid price data found. Using SPY as fallback")
            return self.fetch_stock_data("SPY") or pd.Series()
            
        return pd.concat(prices_data, axis=1).mean(axis=1).dropna()

    def fetch_stock_data(self, symbol):
        # Check cache first
        if symbol in GLOBAL_STATE.stock_data:
            cached = GLOBAL_STATE.stock_data[symbol]
            if datetime.now(timezone.utc) - cached['timestamp'] < timedelta(days=1):
                return cached['data']
        
        # Fetch new data
        time.sleep(RATE_LIMIT)
        end_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        start_date = (datetime.now(timezone.utc) - timedelta(days=TREND_LOOKBACK)).strftime('%Y-%m-%d')
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY}
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                if results:
                    df = pd.DataFrame(results)
                    df['date'] = pd.to_datetime(df['t'], unit='ms')
                    result = df.set_index('date')['c']
                    GLOBAL_STATE.stock_data[symbol] = {
                        'data': result,
                        'timestamp': datetime.now(timezone.utc)
                    }
                    return result
                else:
                    print(f"No data returned for {symbol}")
            else:
                print(f"Error fetching {symbol}: HTTP {response.status_code}")
        except Exception as e:
            print(f"Error fetching {symbol}: {str(e)}")
        return None

    def analyze_regime(self, index_data):
        if len(index_data) < 60:
            print("Insufficient data for regime analysis. Returning neutral")
            return "Neutral"
            
        log_returns = np.log(index_data).diff().dropna()
        features = pd.DataFrame({
            'returns': log_returns,
            'volatility': log_returns.rolling(21).std(),
            'momentum': log_returns.rolling(14).mean()
        }).dropna()
        
        if len(features) < 60:
            print("Insufficient feature data. Returning neutral")
            return "Neutral"
            
        scaled_features = self.feature_scaler.fit_transform(features)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model.fit(scaled_features)
        
        # Label states
        state_means = sorted([(i, np.mean(self.model.means_[i])) for i in range(self.model.n_components)])
        state_labels = {state_means[0][0]: 'Bear', state_means[1][0]: 'Neutral', state_means[2][0]: 'Bull'}
        
        states = self.model.predict(scaled_features)
        return [state_labels[s] for s in states][-1]

# Sector Analysis Module
class SectorAnalyzer:
    def __init__(self):
        self.sector_mappings = {}
        self.sector_composites = {}
        self.sector_weights = {}

    def map_tickers_to_sectors(self, tickers):
        sector_map = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'],
            'Financial Services': ['JPM', 'BAC', 'GS', 'MS', 'V', 'MA'],
            'Healthcare': ['PFE', 'MRK', 'JNJ', 'UNH', 'ABT', 'TMO'],
            'Consumer Cyclical': ['HD', 'MCD', 'NKE', 'TSLA', 'AMZN'],
            'Communication Services': ['GOOGL', 'META', 'DIS', 'NFLX']
        }
        
        for symbol in tickers:
            sector = next((k for k, v in sector_map.items() if symbol in v), "Other")
            self.sector_mappings.setdefault(sector, []).append(symbol)
        
        # Remove small sectors
        self.sector_mappings = {k: v for k, v in self.sector_mappings.items() if len(v) > 5}

    def calculate_sector_weights(self):
        self.sector_weights = {sector: 1/len(self.sector_mappings) for sector in self.sector_mappings}
        return self.sector_weights

    def build_sector_composites(self, market_data_fetcher):
        for sector, tickers in self.sector_mappings.items():
            prices_data = []
            for symbol in tickers[:SECTOR_SAMPLE_SIZE]:
                prices = market_data_fetcher(symbol)
                if prices is not None and len(prices) >= MIN_DAYS_DATA:
                    prices_data.append(prices)
            if prices_data:
                self.sector_composites[sector] = pd.concat(prices_data, axis=1).mean(axis=1).dropna()
            else:
                print(f"WARNING: No data for {sector} sector")

    def calculate_sector_scores(self, market_regime):
        sector_scores = {}
        for sector, composite in self.sector_composites.items():
            try:
                if composite.empty:
                    continue
                    
                returns = composite.pct_change()
                momentum = composite.pct_change(14).iloc[-1] if len(composite) > 14 else 0
                volatility = returns.std()
                
                # Simple scoring
                bull_score = 1 if returns.iloc[-1] > 0 else -1
                base_score = bull_score + momentum * 10
                
                # Regime adjustments
                if market_regime == "Bull":
                    base_score *= 1 + (volatility / 0.02)
                elif market_regime == "Bear":
                    base_score *= 1 + (0.04 - min(volatility, 0.04)) / 0.02
                
                # Apply weighting
                weight = self.sector_weights.get(sector, 0)
                sector_scores[sector] = base_score * (1 + weight * 0.5)
            except Exception as e:
                print(f"Error scoring {sector}: {e}")
                sector_scores[sector] = 0
        
        return pd.Series(sector_scores).sort_values(ascending=False)

# Trend Following Strategy
class TrendFollowingStrategy:
    def __init__(self):
        self.trend_scores = {}

    def calculate_trend_score(self, prices):
        if len(prices) < 50:
            return 0
            
        # Moving averages
        ma50 = prices.rolling(50).mean()
        ma200 = prices.rolling(200).mean()
        ma_cross = 1 if ma50.iloc[-1] > ma200.iloc[-1] and ma50.iloc[-2] <= ma200.iloc[-2] else 0.5 if ma50.iloc[-1] > ma200.iloc[-1] else 0

        # Momentum
        momentum = prices.pct_change(30).iloc[-1] if len(prices) > 30 else 0

        # Composite score
        return ma_cross * 0.6 + momentum * 0.4

    def rank_tickers(self, tickers, market_data_fetcher):
        scores = []
        for symbol in tickers:
            prices = market_data_fetcher(symbol)
            if prices is not None and len(prices) >= 50:
                score = self.calculate_trend_score(prices)
                scores.append((symbol, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

# Weekly Analysis Scheduler
def weekly_analysis():
    scanner = PolygonTickerScanner()
    market_analyzer = MarketRegimeAnalyzer()
    sector_analyzer = SectorAnalyzer()
    trend_strategy = TrendFollowingStrategy()
    
    scanner.start()
    
    try:
        while True:
            now = datetime.now(timezone.utc)
            
            # Run every Sunday at 18:00 UTC (market close)
            if now.weekday() == 6 and now.hour == 18 and now.minute < 10:
                print("\n" + "="*50)
                print(f"Starting weekly analysis at {now}")
                print("="*50)
                
                # Step 1: Get fresh tickers
                current_tickers = GLOBAL_STATE.get_tickers()["ticker"].tolist()
                print(f"Loaded {len(current_tickers)} tickers")
                
                # Step 2: Market regime analysis
                market_data = market_analyzer.prepare_market_data(current_tickers)
                if not market_data.empty:
                    market_regime = market_analyzer.analyze_regime(market_data)
                    GLOBAL_STATE.market_regime = market_regime
                    print(f"\nCurrent Market Regime: {market_regime}")
                else:
                    print("Skipping market regime analysis - no data")
                    market_regime = "Neutral"
                
                # Step 3: Sector analysis
                sector_analyzer.map_tickers_to_sectors(current_tickers)
                sector_analyzer.calculate_sector_weights()
                sector_analyzer.build_sector_composites(market_analyzer.fetch_stock_data)
                
                if sector_analyzer.sector_composites:
                    sector_scores = sector_analyzer.calculate_sector_scores(market_regime)
                    GLOBAL_STATE.sector_scores = sector_scores
                    
                    print("\nSector Scores:")
                    print(sector_scores.head(10))
                    
                    # Step 4: Get top sectors
                    top_sectors = sector_scores.head(3).index.tolist()
                    print(f"\nTop Sectors: {top_sectors}")
                    
                    # Step 5: Get tickers in top sectors
                    sector_tickers = []
                    for sector in top_sectors:
                        sector_tickers.extend(sector_analyzer.sector_mappings.get(sector, []))
                    
                    # Step 6: Apply trend following strategy
                    if sector_tickers:
                        ranked_tickers = trend_strategy.rank_tickers(sector_tickers, market_analyzer.fetch_stock_data)
                        top_10 = [ticker for ticker, score in ranked_tickers[:10]]
                        GLOBAL_STATE.top_tickers = top_10
                        
                        print("\nTop 10 Tickers for Investment:")
                        for i, (ticker, score) in enumerate(ranked_tickers[:10], 1):
                            print(f"{i}. {ticker} (Score: {score:.2f})")
                    else:
                        print("No tickers found in top sectors")
                else:
                    print("Skipping sector analysis - no composites built")
                
                # Save results
                timestamp = now.strftime("%Y%m%d")
                if hasattr(GLOBAL_STATE, 'top_tickers') and GLOBAL_STATE.top_tickers:
                    with open(f"weekly_portfolio_{timestamp}.txt", "w") as f:
                        f.write("\n".join(GLOBAL_STATE.top_tickers))
                
                print("\nAnalysis complete. Sleeping until next week...")
                time.sleep(60 * 60 * 24)  # Sleep 24 hours after completion
            
            else:
                # Print status every hour
                if now.minute == 0:
                    print(f"{now.strftime('%Y-%m-%d %H:%M')} - Waiting for Sunday 18:00 UTC")
                time.sleep(60)  # Check every minute
                
    except KeyboardInterrupt:
        print("\nStopping scanner...")
    finally:
        scanner.stop()

if __name__ == "__main__":
    weekly_analysis()