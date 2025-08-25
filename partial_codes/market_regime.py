import numpy as np
import pandas as pd
import requests
import time
import os
import logging
import json
import talib
import joblib
import math
from hmmlearn import hmm
from websocket import create_connection, WebSocketConnectionClosedException
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread, Lock
from queue import Queue, Empty
from tqdm import tqdm
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings

# ======================== CONFIGURATION SETTINGS ======================== #
try:
    from config import POLYGON_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY, DISCORD_WEBHOOK_URL
except ImportError:
    # Fallback if config.py doesn't exist
    POLYGON_API_KEY = "ld1Poa63U6t4Y2MwOCA2JeKQyHVrmyg8"
    ALPACA_API_KEY = "YOUR_ALPACA_API_KEY"
    ALPACA_SECRET_KEY = "YOUR_ALPACA_SECRET_KEY"
    DISCORD_WEBHOOK_URL = "YOUR_DISCORD_WEBHOOK_URL"

# -------------------- ADJUSTABLE PARAMETERS -------------------- #
# Ticker Scanner Settings
EXCHANGES = ["XNAS", "XNYS", "XASE"]  # Stock exchanges to monitor
TICKER_CACHE_FILE = "ticker_cache.parquet"
MISSING_TICKERS_FILE = "missing_tickers.json"
METADATA_FILE = "scanner_metadata.json"
SCANNER_LOG_LEVEL = logging.INFO
SCANNER_MAX_WORKERS = 20  # Increased worker threads
SCANNER_REFRESH_INTERVAL = 3600  # 1 hour
WS_RECONNECT_DELAY = 5
MAX_RECONNECT_ATTEMPTS = 10
MIN_CACHE_REFRESH_INTERVAL = 300  # 5 minutes minimum between refreshes

# Market Analysis Settings
N_STATES = 4  # Number of market regimes (3 or 4 recommended)
MARKET_COMPOSITE_SAMPLE_SIZE = 100
MIN_DAYS_DATA = 200
SECTOR_SAMPLE_SIZE = 30
SECTOR_MIN_TICKERS = 10
ANALYSIS_INTERVAL = 3600 * 6  # 6 hours
MIN_TICKERS_FOR_ANALYSIS = 500

# Trading Recommendations
ASSET_ALLOCATIONS = {
    "Bear": {
        "bonds": 60,
        "defensive_stocks": 30,
        "cash": 10
    },
    "Severe Bear": {
        "bonds": 40,
        "gold": 30,
        "cash": 30
    },
    "Bull": {
        "growth_stocks": 70,
        "tech": 20,
        "cash": 10
    },
    "Strong Bull": {
        "growth_stocks": 80,
        "small_caps": 15,
        "cash": 5
    },
    "Neutral": {
        "value_stocks": 50,
        "dividend_stocks": 30,
        "cash": 20
    }
}

# Suppress warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ======================== TICKER SCANNER (ENHANCED) ======================== #
class PolygonTickerScanner:
    def __init__(self, api_key=POLYGON_API_KEY, exchanges=EXCHANGES, 
                 cache_file=TICKER_CACHE_FILE, log_level=SCANNER_LOG_LEVEL,
                 max_workers=SCANNER_MAX_WORKERS, refresh_interval=SCANNER_REFRESH_INTERVAL):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v3/reference"
        self.websocket_url = "wss://socket.polygon.io/stocks"
        self.cache_file = cache_file
        self.missing_file = MISSING_TICKERS_FILE
        self.metadata_file = METADATA_FILE
        self.exchanges = exchanges
        self.event_queue = Queue(maxsize=10000)
        self.active = False
        self.ws_reconnect_delay = WS_RECONNECT_DELAY
        self.max_reconnect_attempts = MAX_RECONNECT_ATTEMPTS
        self.current_reconnect_attempts = 0
        self.cache_lock = Lock()
        self.known_missing_tickers = set()
        self.max_workers = max_workers
        self.refresh_interval = refresh_interval
        self.last_refresh_time = 0
        
        # Configure logging
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=log_level
        )
        self.logger = logging.getLogger("PolygonTickerScanner")
        
        # Initialize cache
        self._init_cache()

    def _init_cache(self):
        """Initialize cache with metadata and fast lookup structures"""
        # Load metadata
        metadata = self._load_metadata()
        self.last_refresh_time = metadata.get('last_refresh_time', 0)
        
        cache_exists = os.path.exists(self.cache_file)
        cache_valid = False
        
        # Load main cache if exists
        if cache_exists:
            try:
                self.ticker_cache = pd.read_parquet(self.cache_file)
                # Handle legacy cache without 'type' column
                if 'type' not in self.ticker_cache.columns:
                    self.ticker_cache['type'] = 'CS'  # Assume existing are stocks
                    self.logger.warning("Legacy cache detected. Added 'type' column with default 'CS'")
                # Check if cache is non-empty
                if not self.ticker_cache.empty:
                    self.logger.info(f"Loaded cache with {len(self.ticker_cache)} tickers")
                    cache_valid = True
                else:
                    self.logger.warning("Cache file is empty - treating as invalid")
            except Exception as e:
                self.logger.error(f"Error loading cache: {e}")
        
        # Initialize empty cache if doesn't exist or invalid
        if not cache_exists or not cache_valid:
            self.ticker_cache = pd.DataFrame(columns=["ticker", "name", "primary_exchange", "last_updated_utc", "type"])
            self.logger.info("No valid cache found - initializing empty cache")
        
        # Create fast lookup set
        self.current_tickers_set = set(self.ticker_cache['ticker'].tolist()) if not self.ticker_cache.empty else set()
        
        # Load missing tickers
        if os.path.exists(self.missing_file):
            try:
                with open(self.missing_file, 'r') as f:
                    self.known_missing_tickers = set(json.load(f))
            except Exception as e:
                self.logger.error(f"Error loading missing tickers: {e}")
                self.known_missing_tickers = set()
        
        # FORCE REFRESH ON FIRST RUN (NO VALID CACHE)
        if not cache_valid or self.ticker_cache.empty:
            self.logger.info("First run detected - forcing full scan")
            self._refresh_all_tickers()  # BLOCKING initial refresh
        # Regular stale cache check
        elif time.time() - self.last_refresh_time > self.refresh_interval:
            self.logger.info("Cache is stale, starting background refresh")
            Thread(target=self._refresh_all_tickers, daemon=True).start()
    
    def _load_metadata(self):
        """Load scanner metadata"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save scanner metadata"""
        metadata = {'last_refresh_time': self.last_refresh_time}
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f)
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
    
    def _save_missing_tickers(self):
        """Persist known missing tickers"""
        try:
            with open(self.missing_file, 'w') as f:
                json.dump(list(self.known_missing_tickers), f)
        except Exception as e:
            self.logger.error(f"Error saving missing tickers: {e}")

    def _call_polygon_api(self, url):
        """Call Polygon API with optimized error handling"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if response.status_code == 429:
                self.logger.warning("Rate limit exceeded, retrying after delay")
                time.sleep(2)
                return self._call_polygon_api(url)
            self.logger.error(f"API request failed: {e}")
            return None

    def _fetch_exchange_page(self, url):
        """Fetch a single API page"""
        try:
            data = self._call_polygon_api(url)
            if not data:
                return [], None
                
            results = data.get("results", [])
            next_url = data.get("next_url")
            if next_url:
                next_url += f"&apiKey={self.api_key}"
                
            return results, next_url
        except Exception as e:
            self.logger.error(f"Error fetching page: {e}")
            return [], None

    def _fetch_exchange_tickers(self, exchange):
        """Fetch all tickers for an exchange with parallel pagination"""
        self.logger.info(f"Starting ticker fetch for {exchange}")
        base_url = f"{self.base_url}/tickers?market=stocks&exchange={exchange}&active=true&limit=1000&apiKey={self.api_key}"
        
        all_results = []
        next_urls = [base_url]
        page_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while next_urls:
                futures = {executor.submit(self._fetch_exchange_page, url): url for url in next_urls}
                next_urls = []
                
                for future in as_completed(futures):
                    results, next_url = future.result()
                    if results:
                        stock_results = [r for r in results if r.get('type') == 'CS']
                        all_results.extend(stock_results)
                        page_count += 1
                    if next_url:
                        next_urls.append(next_url)
        
        self.logger.info(f"Completed {exchange}: {len(all_results)} stocks")
        return all_results

    def _refresh_all_tickers(self):
        """Refresh all tickers with delta updates"""
        # Rate limit refreshes
        if time.time() - self.last_refresh_time < MIN_CACHE_REFRESH_INTERVAL:
            self.logger.warning("Refresh skipped - too soon after last refresh")
            return
            
        start_time = time.time()
        self.logger.info("Starting full ticker refresh")
        
        # Get current state before refresh
        with self.cache_lock:
            old_tickers = set(self.current_tickers_set)
        
        # Fetch new data
        with ThreadPoolExecutor(max_workers=len(self.exchanges)) as executor:
            futures = {executor.submit(self._fetch_exchange_tickers, exch): exch for exch in self.exchanges}
            all_results = []
            for future in tqdm(as_completed(futures), total=len(self.exchanges), desc="Exchanges"):
                all_results.extend(future.result())
        
        if not all_results:
            self.logger.warning("No stocks fetched during refresh")
            return
            
        # Process new data
        new_df = pd.DataFrame(all_results)[["ticker", "name", "primary_exchange", "last_updated_utc", "type"]]
        new_tickers = set(new_df['ticker'].tolist())
        
        # Calculate changes
        added = new_tickers - old_tickers
        removed = old_tickers - new_tickers
        
        # Update cache
        with self.cache_lock:
            self.ticker_cache = new_df
            self.ticker_cache.to_parquet(self.cache_file)
            self.current_tickers_set = new_tickers
            
            # Clean up missing tickers that now exist
            rediscovered = added & self.known_missing_tickers
            if rediscovered:
                self.known_missing_tickers -= rediscovered
                self._save_missing_tickers()
        
        # Update metadata
        self.last_refresh_time = time.time()
        self._save_metadata()
        
        # Log results
        elapsed = time.time() - start_time
        self.logger.info(f"Refresh completed in {elapsed:.2f}s")
        self.logger.info(f"Total stocks: {len(new_df)} | Added: {len(added)} | Removed: {len(removed)}")
        if added:
            self.logger.info(f"New tickers: {', '.join(list(added)[:5])}{'...' if len(added)>5 else ''}")
        if removed:
            self.logger.info(f"Removed tickers: {', '.join(list(removed)[:5])}{'...' if len(removed)>5 else ''}")

    def _update_single_ticker(self, ticker):
        """Update a single ticker efficiently"""
        if ticker in self.known_missing_tickers:
            return
            
        url = f"{self.base_url}/tickers/{ticker}?apiKey={self.api_key}"
        try:
            data = self._call_polygon_api(url)
            if not data or not data.get("results"):
                self.known_missing_tickers.add(ticker)
                self._save_missing_tickers()
                return
                
            ticker_data = data["results"]
            if ticker_data.get("type") != "CS":
                self.known_missing_tickers.add(ticker)
                self._save_missing_tickers()
                return
                
            new_row = {
                "ticker": ticker,
                "name": ticker_data.get("name", ""),
                "primary_exchange": ticker_data.get("primary_exchange", ""),
                "last_updated_utc": ticker_data.get("last_updated_utc", ""),
                "type": "CS"
            }
            
            with self.cache_lock:
                # Update DataFrame
                if ticker in self.current_tickers_set:
                    idx = self.ticker_cache.index[self.ticker_cache["ticker"] == ticker]
                    self.ticker_cache.loc[idx, list(new_row)] = list(new_row.values())
                else:
                    self.ticker_cache = pd.concat([
                        self.ticker_cache, 
                        pd.DataFrame([new_row])
                    ], ignore_index=True)
                    self.current_tickers_set.add(ticker)
                
                # Update cache file
                self.ticker_cache.to_parquet(self.cache_file)
                
        except Exception as e:
            self.logger.error(f"Failed to update {ticker}: {e}")

    def _websocket_listener(self):
        """WebSocket listener with exponential backoff"""
        while self.active:
            try:
                delay = min(self.ws_reconnect_delay * (2 ** self.current_reconnect_attempts), 30)
                self.logger.info(f"WebSocket reconnect in {delay}s (attempt {self.current_reconnect_attempts+1})")
                time.sleep(delay)
                
                ws = create_connection(self.websocket_url, timeout=10)
                self.logger.info("WebSocket connected")
                self.current_reconnect_attempts = 0
                
                # Authenticate and subscribe
                ws.send(json.dumps({"action": "auth", "params": self.api_key}))
                ws.send(json.dumps({"action": "subscribe", "params": "T.*"}))
                
                # Connection maintenance
                last_ping = time.time()
                while self.active:
                    try:
                        data = ws.recv()
                        if data:
                            self.event_queue.put(json.loads(data))
                            
                        # Send keepalive
                        if time.time() - last_ping > 30:
                            ws.ping()
                            last_ping = time.time()
                    except (WebSocketConnectionClosedException, ConnectionResetError):
                        break
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
                self.current_reconnect_attempts += 1
                if self.current_reconnect_attempts >= self.max_reconnect_attempts:
                    self.logger.error("Max reconnect attempts reached")
                    self.active = False

    def _process_events(self):
        """Process events with efficient ticker checks"""
        while self.active:
            try:
                event = self.event_queue.get(timeout=1)
                events = event if isinstance(event, list) else [event]
                
                for e in events:
                    try:
                        if e.get("ev") == "T":  # Trade event
                            ticker = e["sym"]
                            
                            # Fast path check
                            with self.cache_lock:
                                if ticker in self.current_tickers_set or ticker in self.known_missing_tickers:
                                    continue
                            
                            self.logger.info(f"New ticker detected: {ticker}")
                            Thread(target=self._update_single_ticker, args=(ticker,), daemon=True).start()
                    except KeyError:
                        continue
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Event processing error: {e}")

    def _background_refresher(self):
        """Background refresh with interval management"""
        while self.active:
            time.sleep(self.refresh_interval)
            if not self.active:
                break
            self.logger.info("Starting background refresh")
            self._refresh_all_tickers()

    def start(self):
        """Start scanner services"""
        if not self.active:
            self.active = True
            Thread(target=self._websocket_listener, daemon=True).start()
            Thread(target=self._process_events, daemon=True).start()
            Thread(target=self._background_refresher, daemon=True).start()
            self.logger.info("Ticker scanner started")
        else:
            self.logger.warning("Scanner already running")

    def stop(self):
        """Stop scanner services"""
        self.active = False
        self.logger.info("Scanner stopped")

    def get_current_tickers(self):
        """Get current ticker list"""
        with self.cache_lock:
            return self.ticker_cache.copy()


# ======================== MARKET REGIME ANALYZER ======================== #
class MarketRegimeAnalyzer:
    def __init__(self, n_states=N_STATES, polygon_api_key=POLYGON_API_KEY):
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=2000,
            tol=1e-4,
            init_params="se",
            params="stmc",
            random_state=42,
        )
        self.state_labels = {}
        self.feature_scaler = StandardScaler()
        self.polygon_api_key = polygon_api_key
        self.data_cache = {}
        os.makedirs("data_cache", exist_ok=True)

    def prepare_market_data(self, tickers, sample_size=MARKET_COMPOSITE_SAMPLE_SIZE, min_days_data=MIN_DAYS_DATA):
        prices_data = []
        valid_tickers = []
        mcaps = {}  # For market cap weighting

        # Create progress bar for market composite building
        total_steps = min(sample_size, len(tickers)) * 2
        progress_bar = tqdm(total=total_steps, desc="Building Market Composite")

        # First pass: Get market caps for weighting
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {
                executor.submit(self.get_market_cap, ticker): ticker 
                for ticker in tickers[:sample_size]
            }
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    mcaps[ticker] = future.result() or 1  # Default to 1 if None
                except:
                    mcaps[ticker] = 1
                finally:
                    progress_bar.update(1)

        total_mcap = sum(mcaps.values())
        
        # Second pass: Get price data with market cap weighting
        def fetch_and_validate(symbol):
            prices = self.fetch_stock_data(symbol)
            if prices is not None and len(prices) >= min_days_data:
                weight = mcaps.get(symbol, 1) / total_mcap
                return symbol, prices * weight  # Apply market cap weighting
            return None

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(fetch_and_validate, symbol): symbol
                for symbol in tickers[:sample_size]
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    if result:
                        valid_tickers.append(result[0])
                        prices_data.append(result[1])
                except Exception as e:
                    print(f"Error processing {symbol}: {str(e)}")
                finally:
                    progress_bar.update(1)

        # Close progress bar
        progress_bar.close()
        
        if not prices_data:
            raise ValueError("Insufficient data to create market composite")
            
        # Align and combine data
        composite = pd.concat(prices_data, axis=1)
        composite.columns = valid_tickers
        composite = composite.fillna(method='ffill').fillna(method='bfill')
        return composite.sum(axis=1).dropna()  # Sum of weighted prices

    def analyze_regime(self, index_data, n_states=None):
        if n_states is None:
            n_states = self.model.n_components

        # Calculate advanced features
        log_returns = np.log(index_data).diff().dropna()
        features = pd.DataFrame({
            "returns": log_returns,
            "volatility": log_returns.rolling(21).std(),
            "momentum": log_returns.rolling(14).mean(),
            "rsi": talib.RSI(index_data, timeperiod=14),
            "macd": talib.MACD(index_data)[0],  # MACD line
            "adx": talib.ADX(index_data, index_data, index_data, timeperiod=14)
        }).dropna()

        if len(features) < 100:  # Increased minimum for better model stability
            raise ValueError(f"Only {len(features)} days of feature data")
            
        # Scale features
        scaled_features = self.feature_scaler.fit_transform(features)

        # Create and fit model
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=2000,
            tol=1e-4,
            init_params="se",  # Critical fix
            params="stmc",
            random_state=42,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(scaled_features)

        # Label states based on volatility and returns
        state_stats = []
        for i in range(model.n_components):
            state_return = model.means_[i][0]  # Returns feature
            state_vol = model.means_[i][1]    # Volatility feature
            state_stats.append((i, state_return, state_vol))
            
        # Sort by return then volatility
        state_stats.sort(key=lambda x: (x[1], -x[2]))
        
        # Create labels based on sorted states
        if n_states == 3:
            state_labels = {
                state_stats[0][0]: "Bear",
                state_stats[1][0]: "Neutral",
                state_stats[2][0]: "Bull",
            }
        elif n_states == 4:
            state_labels = {
                state_stats[0][0]: "Severe Bear",
                state_stats[1][0]: "Mild Bear",
                state_stats[2][0]: "Mild Bull",
                state_stats[3][0]: "Strong Bull",
            }
        else:
            state_labels = {i: f"State {i+1}" for i in range(n_states)}

        # Predict regimes
        states = model.predict(scaled_features)
        state_probs = model.predict_proba(scaled_features)
        
        # Calculate state durations
        state_durations = self.calculate_state_durations(states)

        return {
            "model": model,
            "regimes": [state_labels[s] for s in states],
            "probabilities": state_probs,
            "features": features,
            "index_data": index_data[features.index[0] :],
            "state_labels": state_labels,
            "state_durations": state_durations
        }
        
    def calculate_state_durations(self, states):
        """Calculate average duration per state"""
        durations = {state: [] for state in set(states)}
        current_state = states[0]
        current_duration = 1
        
        for i in range(1, len(states)):
            if states[i] == current_state:
                current_duration += 1
            else:
                durations[current_state].append(current_duration)
                current_state = states[i]
                current_duration = 1
                
        durations[current_state].append(current_duration)
        
        # Calculate average durations
        avg_durations = {}
        for state, durs in durations.items():
            avg_durations[state] = sum(durs) / len(durs) if durs else 0
            
        return avg_durations

    def fetch_stock_data(self, symbol, days=365):
        # Check cache first
        cache_file = f"data_cache/{symbol}_{days}.pkl"
        if os.path.exists(cache_file):
            return pd.read_pickle(cache_file)
            
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.polygon_api_key,
        }

        try:
            time.sleep(0.15)  # Respect rate limits (6-7 requests/sec)
            response = requests.get(url, params=params, timeout=15)
            
            # Handle rate limits
            if response.status_code == 429:
                time.sleep(30)
                return self.fetch_stock_data(symbol, days)
                
            if response.status_code != 200:
                return None

            results = response.json().get("results", [])
            if not results:
                return None
                
            df = pd.DataFrame(results)
            df["date"] = pd.to_datetime(df["t"], unit="ms")
            result = df.set_index("date")["c"]
            
            # Cache result
            result.to_pickle(cache_file)
            return result
            
        except requests.exceptions.Timeout:
            print(f"Timeout fetching {symbol}, skipping")
            return None
        except Exception as e:
            print(f"Error fetching {symbol}: {str(e)}")
            return None
            
    def get_market_cap(self, symbol):
        cache_file = f"data_cache/mcap_{symbol}.pkl"
        if os.path.exists(cache_file):
            return joblib.load(cache_file)
            
        url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
        params = {"apiKey": self.polygon_api_key}
        
        try:
            time.sleep(0.15)  # Rate limiting
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 429:
                time.sleep(30)
                return self.get_market_cap(symbol)
                
            if response.status_code == 200:
                data = response.json().get("results", {})
                mcap = data.get("market_cap", 0)
                joblib.dump(mcap, cache_file)
                return mcap
        except Exception:
            pass
            
        return 0


# ======================== SECTOR REGIME SYSTEM ======================== #
class SectorRegimeSystem:
    def __init__(self, polygon_api_key=POLYGON_API_KEY):
        self.sector_mappings = {}
        self.sector_composites = {}
        self.sector_analyzers = {}
        self.overall_analyzer = MarketRegimeAnalyzer(polygon_api_key=polygon_api_key)
        self.sector_weights = {}
        self.sector_scores = {}
        self.polygon_api_key = polygon_api_key
        self.current_regime = None

    def map_tickers_to_sectors(self, tickers):
        self.sector_mappings = {}
        cache_file = "data_cache/sector_mappings.pkl"
        
        # Try loading from cache
        if os.path.exists(cache_file):
            self.sector_mappings = joblib.load(cache_file)
            return self.sector_mappings

        # Parallel sector mapping with progress bar
        def map_single_ticker(symbol):
            cache_file = f"data_cache/sector_{symbol}.pkl"
            if os.path.exists(cache_file):
                return symbol, joblib.load(cache_file)
                
            url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
            params = {"apiKey": self.polygon_api_key}
            try:
                time.sleep(0.15)
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 429:
                    time.sleep(30)
                    return map_single_ticker(symbol)
                    
                if response.status_code == 200:
                    data = response.json().get("results", {})
                    sector = data.get("sic_description", "Unknown")
                    if sector == "Unknown":
                        sector = data.get("primary_exchange", "Unknown")
                    
                    # Cache result
                    joblib.dump(sector, cache_file)
                    return symbol, sector
            except Exception as e:
                print(f"Sector mapping failed for {symbol}: {str(e)}")
            return symbol, "Unknown"

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(map_single_ticker, symbol): symbol for symbol in tickers}

            for future in tqdm(as_completed(futures), total=len(tickers), desc="Mapping Sectors"):
                try:
                    symbol, sector = future.result()
                    if sector != "Unknown":
                        self.sector_mappings.setdefault(sector, []).append(symbol)
                except Exception as e:
                    print(f"Error processing sector mapping: {str(e)}")

        # Remove unknown sectors and small sectors
        self.sector_mappings = {
            k: v for k, v in self.sector_mappings.items() 
            if k != "Unknown" and len(v) > SECTOR_MIN_TICKERS
        }
        
        # Save to cache
        joblib.dump(self.sector_mappings, cache_file)
        return self.sector_mappings

    def calculate_sector_weights(self):
        total_mcap = 0
        sector_mcaps = {}
        
        # Create progress bar for sector weights
        total_sectors = len(self.sector_mappings)
        progress_bar = tqdm(total=total_sectors, desc="Calculating Sector Weights")
        
        # Use cached data if available
        for sector, tickers in self.sector_mappings.items():
            sector_mcap = 0
            # Get market caps for first 30 tickers in sector
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_ticker = {
                    executor.submit(self.overall_analyzer.get_market_cap, symbol): symbol
                    for symbol in tickers[:30]
                }
                for future in as_completed(future_to_ticker):
                    symbol = future_to_ticker[future]
                    try:
                        mcap = future.result()
                        sector_mcap += mcap if mcap else 0
                    except:
                        pass

            sector_mcaps[sector] = sector_mcap
            total_mcap += sector_mcap
            progress_bar.update(1)
        
        # Close progress bar
        progress_bar.close()

        self.sector_weights = {
            sector: mcap / total_mcap if total_mcap > 0 else 1 / len(sector_mcaps)
            for sector, mcap in sector_mcaps.items()
        }
        return self.sector_weights

    def build_sector_composites(self, sample_size=SECTOR_SAMPLE_SIZE):
        print("\nBuilding sector composites...")
        self.sector_composites = {}
        cache_file = "data_cache/sector_composites.pkl"
        
        # Try loading from cache
        if os.path.exists(cache_file):
            self.sector_composites = joblib.load(cache_file)
            return self.sector_composites

        # Create progress bar for sector composites
        total_sectors = len(self.sector_mappings)
        progress_bar = tqdm(total=total_sectors, desc="Building Sector Composites")

        # Process each sector
        for sector, tickers in self.sector_mappings.items():
            prices_data = []
            mcaps = {}
            
            # Get market caps first
            for symbol in tickers[:sample_size]:
                mcaps[symbol] = self.overall_analyzer.get_market_cap(symbol) or 1
            total_mcap = sum(mcaps.values())
            
            # Get price data with weighting
            for symbol in tickers[:sample_size]:
                try:
                    prices = self.overall_analyzer.fetch_stock_data(symbol)
                    if prices is not None and len(prices) >= 200:
                        weight = mcaps[symbol] / total_mcap
                        prices_data.append(prices * weight)
                except Exception as e:
                    print(f"Error processing {symbol}: {str(e)}")
            
            if prices_data:
                composite = pd.concat(prices_data, axis=1)
                composite = composite.fillna(method='ffill').fillna(method='bfill')
                self.sector_composites[sector] = composite.sum(axis=1).dropna()
            
            progress_bar.update(1)
        
        # Close progress bar
        progress_bar.close()
        
        # Save to cache
        joblib.dump(self.sector_composites, cache_file)
        return self.sector_composites

    def analyze_sector_regimes(self, n_states=N_STATES):
        print("\nAnalyzing sector regimes...")
        self.sector_analyzers = {}
        
        # Create progress bar for sector analysis
        total_sectors = len(self.sector_composites)
        progress_bar = tqdm(total=total_sectors, desc="Analyzing Sector Regimes")
        
        # Get overall market regime first
        if not hasattr(self, 'market_composite'):
            tickers = [t for sublist in self.sector_mappings.values() for t in sublist]
            self.market_composite = self.overall_analyzer.prepare_market_data(tickers[:100])
        market_result = self.overall_analyzer.analyze_regime(self.market_composite)
        self.current_regime = market_result["regimes"][-1]
        
        # Process each sector
        for sector, composite in self.sector_composites.items():
            try:
                analyzer = MarketRegimeAnalyzer(polygon_api_key=self.polygon_api_key)
                results = analyzer.analyze_regime(composite, n_states=n_states)
                self.sector_analyzers[sector] = {
                    "results": results,
                    "composite": composite,
                    "volatility": composite.pct_change().std(),
                    "analyzer": analyzer
                }
            except Exception as e:
                print(f"Error analyzing {sector}: {str(e)}")
            finally:
                progress_bar.update(1)
                
        # Close progress bar
        progress_bar.close()
        return self.sector_analyzers

    def calculate_sector_scores(self):
        self.sector_scores = {}
        if not self.sector_analyzers:
            return pd.Series()

        for sector, data in self.sector_analyzers.items():
            try:
                if "results" not in data:
                    continue
                    
                # Get latest probabilities
                current_probs = data["results"]["probabilities"][-1]
                state_labels = data["results"].get("state_labels", {})
                
                # Calculate sector momentum
                momentum = data["composite"].pct_change(21).iloc[-1] if len(data["composite"]) > 21 else 0
                
                # Calculate bull/bear probabilities
                bull_prob = sum(
                    current_probs[i] 
                    for i, label in state_labels.items() 
                    if "Bull" in label
                )
                bear_prob = sum(
                    current_probs[i] 
                    for i, label in state_labels.items() 
                    if "Bear" in label
                )
                
                # Base score (simplified)
                base_score = bull_prob - bear_prob
                
                # Apply momentum adjustment
                momentum_factor = 1 + (momentum * 5)  # Amplify momentum effect
                adjusted_score = base_score * momentum_factor
                
                # Apply sector weight
                weight = self.sector_weights.get(sector, 0.01)
                self.sector_scores[sector] = adjusted_score * (1 + weight)
                
            except Exception as e:
                print(f"Error calculating score for {sector}: {str(e)}")
                self.sector_scores[sector] = 0

        return pd.Series(self.sector_scores).sort_values(ascending=False)


# ======================== UNIFIED MARKET SYSTEM ======================== #
class UnifiedMarketSystem:
    def __init__(self, polygon_api_key=POLYGON_API_KEY):
        self.ticker_scanner = PolygonTickerScanner(api_key=polygon_api_key)
        self.sector_system = SectorRegimeSystem(polygon_api_key=polygon_api_key)
        self.active = False
        self.analysis_thread = None
        self.lock = Lock()
        self.current_regime = None
        self.sector_scores = {}
        self.last_analysis_time = 0
        self.analysis_interval = ANALYSIS_INTERVAL
        self.min_tickers = MIN_TICKERS_FOR_ANALYSIS
        
        # Configure logging
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger("UnifiedMarketSystem")
    
    def start(self):
        """Start the unified market system"""
        if not self.active:
            self.active = True
            # Start ticker scanner
            self.ticker_scanner.start()
            self.logger.info("Ticker scanner started")
            
            # Start analysis thread
            self.analysis_thread = Thread(target=self._run_periodic_analysis, daemon=True)
            self.analysis_thread.start()
            self.logger.info("Market regime analysis thread started")
        else:
            self.logger.warning("System is already running")
    
    def stop(self):
        """Stop the system"""
        self.active = False
        self.ticker_scanner.stop()
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(5)
        self.logger.info("System stopped")
    
    def _run_periodic_analysis(self):
        """Periodically analyze market regime and sector scores"""
        while self.active:
            try:
                # Wait for next analysis cycle
                time.sleep(self.analysis_interval)
                
                # Get current tickers
                tickers_df = self.ticker_scanner.get_current_tickers()
                if tickers_df is None or tickers_df.empty or len(tickers_df) < self.min_tickers:
                    self.logger.warning(f"Not enough tickers for analysis: {len(tickers_df) if tickers_df is not None else 0}")
                    continue
                
                tickers = tickers_df['ticker'].tolist()
                self.logger.info(f"Starting market regime analysis with {len(tickers)} tickers")
                
                # Update sector mappings
                self.sector_system.map_tickers_to_sectors(tickers)
                self.logger.info(f"Mapped tickers to {len(self.sector_system.sector_mappings)} sectors")
                
                # Calculate sector weights
                self.sector_system.calculate_sector_weights()
                self.logger.info("Calculated sector weights")
                
                # Build sector composites
                self.sector_system.build_sector_composites()
                self.logger.info("Built sector composites")
                
                # Analyze sector regimes
                self.sector_system.analyze_sector_regimes()
                self.logger.info("Analyzed sector regimes")
                
                # Calculate sector scores
                scores = self.sector_system.calculate_sector_scores()
                
                # Update shared state
                with self.lock:
                    self.current_regime = self.sector_system.current_regime
                    self.sector_scores = scores.to_dict()
                    self.last_analysis_time = time.time()
                
                self.logger.info(f"Analysis completed. Current regime: {self.current_regime}")
                self.logger.info("Top sectors:")
                for sector, score in sorted(self.sector_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
                    self.logger.info(f"  {sector}: {score:.4f}")
                
                # Send Discord notification
                self.send_discord_notification()
                
            except Exception as e:
                self.logger.error(f"Analysis failed: {e}")
    
    def send_discord_notification(self):
        """Send market analysis summary to Discord"""
        if not DISCORD_WEBHOOK_URL:
            return
            
        try:
            regime = self.get_market_regime()
            sector_scores = self.get_sector_scores()
            
            if not regime or not sector_scores:
                return
                
            # Prepare message content
            content = f"**Market Analysis Update**\nCurrent Regime: **{regime}**"
            
            # Prepare fields for embed
            fields = []
            
            # Asset allocation
            allocation = ASSET_ALLOCATIONS.get(regime, {})
            alloc_text = "\n".join([f"- {asset.replace('_', ' ').title()}: {pct}%" for asset, pct in allocation.items()])
            fields.append({"name": "Recommended Allocation", "value": alloc_text, "inline": True})
            
            # Top sectors
            top_sectors = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            sectors_text = "\n".join([f"- {sector}: {score:.2f}" for sector, score in top_sectors])
            fields.append({"name": "Top Performing Sectors", "value": sectors_text, "inline": True})
            
            # Create embed
            embed = {
                "title": "Market Analysis Report",
                "color": 0x3498db,
                "fields": fields,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Send to Discord
            payload = {
                "content": content,
                "embeds": [embed]
            }
            
            headers = {"Content-Type": "application/json"}
            response = requests.post(DISCORD_WEBHOOK_URL, data=json.dumps(payload), headers=headers)
            
            if response.status_code != 204:
                self.logger.warning(f"Discord notification failed: {response.status_code} {response.text}")
            else:
                self.logger.info("Sent market analysis to Discord")
                
        except Exception as e:
            self.logger.error(f"Discord notification failed: {e}")
    
    def get_current_tickers(self):
        """Get current ticker list"""
        return self.ticker_scanner.get_current_tickers()
    
    def get_market_regime(self):
        """Get current market regime"""
        with self.lock:
            return self.current_regime
    
    def get_sector_scores(self):
        """Get current sector scores"""
        with self.lock:
            return self.sector_scores.copy()
    
    def get_trading_recommendations(self):
        """Get trading recommendations based on current regime"""
        regime = self.get_market_regime()
        if not regime:
            return {}
        
        # Get top 3 sectors by score
        sector_scores = self.get_sector_scores()
        if not sector_scores:
            return {}
            
        top_sectors = sorted(
            sector_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        # Create recommendations
        regime_rec = ASSET_ALLOCATIONS.get(regime, {})
        return {
            "regime": regime,
            "asset_allocation": regime_rec,
            "recommended_sectors": [s[0] for s in top_sectors],
            "last_updated": datetime.fromtimestamp(self.last_analysis_time).strftime("%Y-%m-%d %H:%M:%S")
        }


# ======================== MAIN EXECUTION ======================== #
if __name__ == "__main__":
    market_system = UnifiedMarketSystem()
    
    try:
        market_system.start()
        print("Unified market system running. Press Ctrl+C to stop.")
        
        # Example: Print recommendations every hour
        while True:
            time.sleep(3600)
            rec = market_system.get_trading_recommendations()
            if rec:
                print("\n===== MARKET ANALYSIS REPORT =====")
                print(f"Current Market Regime: {rec['regime']}")
                print("\nRecommended Asset Allocation:")
                for asset, alloc in rec['asset_allocation'].items():
                    print(f"  {asset.replace('_', ' ').title()}: {alloc}%")
                print("\nTop Performing Sectors:")
                for i, sector in enumerate(rec['recommended_sectors'], 1):
                    print(f"  {i}. {sector}")
                print(f"\nLast Updated: {rec['last_updated']}")
                print("=" * 40)
            else:
                print("Waiting for initial market analysis...")
        
    except KeyboardInterrupt:
        print("\nStopping unified market system...")
    finally:
        market_system.stop()