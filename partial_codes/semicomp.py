import os
import time
import json
import logging
import requests
import numpy as np
import pandas as pd
import joblib
import talib
import concurrent.futures
from websocket import create_connection, WebSocketConnectionClosedException
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread, Lock
from queue import Queue, Empty
from hmmlearn import hmm
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings

# ================ GLOBAL CONFIGURATION ================
class GlobalConfig:
    # Ticker Scanner Settings
    TICKER_CACHE_FILE = "ticker_cache.parquet"
    TICKER_REFRESH_INTERVAL = 3600  # Seconds between full ticker refreshes (1 hour)
    MAX_WORKERS_TICKER_FETCH = 15   # Threads for parallel ticker fetching
    EXCHANGES = ["XNAS", "XNYS", "XASE"]  # Stock exchanges to monitor
    
    # Market Regime Analyzer Settings
    MIN_TICKERS_FOR_COMPOSITE = 500  # Minimum tickers for reliable market composite
    MIN_DAYS_DATA = 200              # Minimum days of data required per stock
    N_STATES = 3                     # Default HMM states (3 or 4)
    VALIDATION_WINDOWS = 36          # Windows for model validation
    HMM_RESTARTS = 5                 # Number of HMM training restarts
    HMM_MAX_ITER = 5000              # Maximum iterations for HMM training
    MIN_FEATURE_DAYS = 100           # Minimum days after feature calculation
    ADAPTIVE_TOL = True              # Use adaptive tolerance for HMM
    
    # Sector Analysis Settings
    SECTOR_COMPOSITE_SAMPLE_SIZE = 30  # Tickers per sector for composite
    MIN_SECTOR_TICKERS = 10            # Minimum tickers to form a sector
    
    # Data Caching
    DATA_CACHE_DIR = "data_cache"
    CACHE_EXPIRY_DAYS = 7            # Days before cache is considered stale
    
    # API Configuration
    POLYGON_API_KEY = None            # Will be initialized below
    API_RATE_LIMIT_DELAY = 0.15       # Seconds between API calls
    
    # System Behavior
    LOG_LEVEL = logging.INFO
    MAX_RECONNECT_ATTEMPTS = 10
    WS_RECONNECT_DELAY = 5

    @classmethod
    def init(cls):
        # Initialize API key (from config.py or environment)
        try:
            from config import POLYGON_API_KEY as api_key
            cls.POLYGON_API_KEY = api_key
        except ImportError:
            cls.POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', 'your_api_key_here')
        
        # Create cache directory
        os.makedirs(cls.DATA_CACHE_DIR, exist_ok=True)

# Initialize the configuration
GlobalConfig.init()

# Suppress warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=GlobalConfig.LOG_LEVEL
)
logger = logging.getLogger(__name__)

# Reduce hmmlearn logging level
hmm_logger = logging.getLogger("hmmlearn")
hmm_logger.setLevel(logging.ERROR)

class PolygonTickerScanner:
    def __init__(self, api_key=GlobalConfig.POLYGON_API_KEY, 
                 exchanges=GlobalConfig.EXCHANGES, 
                 cache_file=GlobalConfig.TICKER_CACHE_FILE,
                 log_level=GlobalConfig.LOG_LEVEL,
                 max_workers=GlobalConfig.MAX_WORKERS_TICKER_FETCH, 
                 refresh_interval=GlobalConfig.TICKER_REFRESH_INTERVAL):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v3/reference"
        self.websocket_url = "wss://socket.polygon.io/stocks"
        self.cache_file = cache_file
        self.exchanges = exchanges
        self.event_queue = Queue(maxsize=10000)
        self.active = False
        self.ws_reconnect_delay = GlobalConfig.WS_RECONNECT_DELAY
        self.max_reconnect_attempts = GlobalConfig.MAX_RECONNECT_ATTEMPTS
        self.current_reconnect_attempts = 0
        self.cache_lock = Lock()
        self.known_missing_tickers = set()
        self.max_workers = max_workers
        self.refresh_interval = refresh_interval
        self.last_refresh_time = 0
        
        self.logger = logging.getLogger("PolygonTickerScanner")
        self.logger.setLevel(log_level)
        
        self._init_cache()

    def _init_cache(self):
        """Initialize ticker cache with thread-safe locking"""
        with self.cache_lock:
            if os.path.exists(self.cache_file):
                try:
                    self.ticker_cache = pd.read_parquet(self.cache_file)
                    self.logger.info(f"Loaded cache with {len(self.ticker_cache)} tickers")
                    
                    if time.time() - self.last_refresh_time > self.refresh_interval:
                        Thread(target=self._refresh_all_tickers, daemon=True).start()
                except Exception as e:
                    self.logger.error(f"Error loading cache: {e}")
                    self.ticker_cache = pd.DataFrame(columns=["ticker", "name", "primary_exchange", "last_updated_utc"])
                    self._refresh_all_tickers()
            else:
                self.ticker_cache = pd.DataFrame(columns=["ticker", "name", "primary_exchange", "last_updated_utc"])
                self._refresh_all_tickers()

    def _call_polygon_api(self, url):
        """Call Polygon API with rate limiting"""
        try:
            time.sleep(GlobalConfig.API_RATE_LIMIT_DELAY)
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            return None
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON response from: {url}")
            return None

    def _fetch_exchange_page(self, exchange, url):
        """Fetch a single page for an exchange"""
        try:
            self.logger.debug(f"Fetching page: {url}")
            data = self._call_polygon_api(url)
            if not data:
                return []
                
            results = data.get("results", [])
            next_url = data.get("next_url", None)
            if next_url:
                next_url += f"&apiKey={self.api_key}"
                
            return results, next_url
        except Exception as e:
            self.logger.error(f"Error fetching page for {exchange}: {e}")
            return [], None

    def _fetch_exchange_tickers(self, exchange):
        """Fetch all tickers for a specific exchange using parallel page fetching"""
        self.logger.info(f"Starting ticker fetch for exchange: {exchange}")
        base_url = f"{self.base_url}/tickers?market=stocks&exchange={exchange}&active=true&limit=1000&apiKey={self.api_key}"
        
        all_results = []
        next_urls = [base_url]
        page_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while next_urls:
                futures = []
                current_urls = next_urls.copy()
                next_urls = []
                
                for url in current_urls:
                    futures.append(executor.submit(self._fetch_exchange_page, exchange, url))
                
                for future in concurrent.futures.as_completed(futures):
                    results, next_url = future.result()
                    if results:
                        all_results.extend(results)
                        page_count += 1
                        
                    if next_url:
                        next_urls.append(next_url)
                        
                self.logger.debug(f"Processed {len(current_urls)} pages for {exchange}. Total pages: {page_count}")
        
        self.logger.info(f"Completed fetch for {exchange}: {len(all_results)} tickers across {page_count} pages")
        return all_results

    def _refresh_all_tickers(self):
        """Fetch all tickers from Polygon using parallel execution"""
        start_time = time.time()
        self.logger.info("Starting full ticker refresh")
        
        with self.cache_lock:
            current_count = len(self.ticker_cache)
            self.logger.info(f"Current tickers in cache: {current_count}")
        
        with ThreadPoolExecutor(max_workers=len(self.exchanges)) as exchange_executor:
            exchange_futures = {exchange_executor.submit(self._fetch_exchange_tickers, exchange): exchange 
                                for exchange in self.exchanges}
            
            exchange_counts = {}
            all_results = []
            for future in concurrent.futures.as_completed(exchange_futures):
                exchange = exchange_futures[future]
                try:
                    exchange_results = future.result()
                    all_results.extend(exchange_results)
                    exchange_counts[exchange] = len(exchange_results)
                    self.logger.debug(f"Finished {exchange}: {len(exchange_results)} tickers")
                except Exception as e:
                    self.logger.error(f"Error processing {exchange}: {e}")
                    exchange_counts[exchange] = 0
        
        self.logger.info("Ticker refresh summary:")
        total_tickers = 0
        for exchange in self.exchanges:
            count = exchange_counts.get(exchange, 0)
            self.logger.info(f"  {exchange}: {count} tickers")
            total_tickers += count
        
        if all_results:
            df = pd.DataFrame(all_results)
            
            if not df.empty:
                new_cache = df[["ticker", "name", "primary_exchange", "last_updated_utc"]].copy()
            else:
                new_cache = pd.DataFrame(columns=["ticker", "name", "primary_exchange", "last_updated_utc"])
            
            with self.cache_lock:
                self.ticker_cache = new_cache
                self.ticker_cache.to_parquet(self.cache_file)
                new_count = len(self.ticker_cache)
            
            elapsed = time.time() - start_time
            self.logger.info(f"Total tickers: {total_tickers}")
            self.logger.info(f"Refresh completed in {elapsed:.2f} seconds")
            self.last_refresh_time = time.time()
        else:
            self.logger.warning("No tickers fetched during refresh")
            with self.cache_lock:
                current_count = len(self.ticker_cache)
            self.logger.info(f"Total tickers remains: {current_count}")

    def _update_single_ticker(self, ticker):
        """Fetch and update a single ticker in cache"""
        if ticker in self.known_missing_tickers:
            return
        
        url = f"{self.base_url}/tickers/{ticker}?apiKey={self.api_key}"
        try:
            data = self._call_polygon_api(url)
            if data and data.get("status") == "OK" and data.get("results"):
                ticker_data = data["results"]
                new_row = {
                    "ticker": ticker,
                    "name": ticker_data.get("name", ""),
                    "primary_exchange": ticker_data.get("primary_exchange", ""),
                    "last_updated_utc": ticker_data.get("last_updated_utc", "")
                }
                
                with self.cache_lock:
                    if ticker in self.ticker_cache["ticker"].values:
                        idx = self.ticker_cache.index[self.ticker_cache["ticker"] == ticker].tolist()
                        for col in new_row:
                            self.ticker_cache.at[idx[0], col] = new_row[col]
                    else:
                        self.ticker_cache = pd.concat([
                            self.ticker_cache, 
                            pd.DataFrame([new_row])
                        ], ignore_index=True)
                    
                    self.ticker_cache.to_parquet(self.cache_file)
                    new_count = len(self.ticker_cache)
                
                self.logger.info(f"Updated ticker: {ticker}")
                self.logger.info(f"Total tickers after update: {new_count}")
            else:
                self.known_missing_tickers.add(ticker)
                self.logger.warning(f"Ticker not found: {ticker}")
        except Exception as e:
            self.logger.error(f"Failed to update ticker {ticker}: {e}")

    def _websocket_listener(self):
        """Handle real-time WebSocket data with exponential backoff"""
        while self.active:
            try:
                delay = min(self.ws_reconnect_delay * (2 ** self.current_reconnect_attempts), 300)
                self.logger.info(f"Connecting to WebSocket in {delay:.1f}s (attempt {self.current_reconnect_attempts + 1}/{self.max_reconnect_attempts})")
                time.sleep(delay)
                
                ws = create_connection(
                    self.websocket_url,
                    timeout=15,
                    enable_multithread=True
                )
                self.logger.info("WebSocket connected")
                self.current_reconnect_attempts = 0
                
                auth_msg = json.dumps({"action": "auth", "params": self.api_key})
                ws.send(auth_msg)
                
                sub_msg = json.dumps({
                    "action": "subscribe",
                    "params": "T.*,A.*,AM.*"
                })
                ws.send(sub_msg)
                
                last_ping = time.time()
                while self.active:
                    try:
                        data = ws.recv()
                        if data:
                            self.event_queue.put(json.loads(data))
                        
                        if time.time() - last_ping > 30:
                            ws.ping()
                            last_ping = time.time()
                            
                    except (WebSocketConnectionClosedException, ConnectionResetError) as e:
                        self.logger.warning(f"WebSocket error: {e}")
                        break
                    except Exception as e:
                        self.logger.error(f"Unexpected WebSocket error: {e}", exc_info=True)
                        break
                        
            except Exception as e:
                self.logger.error(f"Connection failed: {e}")
                self.current_reconnect_attempts += 1
                if self.current_reconnect_attempts >= self.max_reconnect_attempts:
                    self.logger.error("Max reconnection attempts reached. Stopping scanner.")
                    self.active = False
            finally:
                try:
                    if 'ws' in locals():
                        ws.close()
                except:
                    pass

    def _process_events(self):
        """Process real-time events from queue in batches"""
        batch_size = 100
        batch_timeout = 0.5
        
        while self.active:
            try:
                batch = []
                start_time = time.time()
                
                while len(batch) < batch_size and (time.time() - start_time) < batch_timeout:
                    try:
                        event = self.event_queue.get(timeout=0.1)
                        batch.append(event)
                    except Empty:
                        continue
                
                for event in batch:
                    if isinstance(event, list):
                        for e in event:
                            self._handle_single_event(e)
                    else:
                        self._handle_single_event(event)
                
                if not batch:
                    time.sleep(0.5)
                    
            except Exception as e:
                self.logger.error(f"Event processing error: {e}", exc_info=True)

    def _handle_single_event(self, event):
        """Handle individual WebSocket events"""
        try:
            if event.get("ev") == "T":
                ticker = event["sym"]
                
                with self.cache_lock:
                    exists = ticker in self.ticker_cache["ticker"].values
                
                if not exists and ticker not in self.known_missing_tickers:
                    self.logger.info(f"New ticker detected: {ticker}")
                    Thread(target=self._update_single_ticker, args=(ticker,), daemon=True).start()
                    
        except KeyError as e:
            self.logger.warning(f"Missing key in event: {e}")
        except Exception as e:
            self.logger.error(f"Event handling error: {e}", exc_info=True)

    def _background_refresher(self):
        """Periodically refresh tickers in the background"""
        while self.active:
            try:
                time.sleep(self.refresh_interval)
                if not self.active:
                    break
                    
                self.logger.info("Starting background ticker refresh...")
                start_time = time.time()
                self._refresh_all_tickers()
                elapsed = time.time() - start_time
                self.logger.info(f"Background refresh completed in {elapsed:.2f} seconds")
            except Exception as e:
                self.logger.error(f"Background refresh failed: {e}")

    def start(self):
        """Start the real-time scanner"""
        if not self.active:
            self.active = True
            Thread(target=self._websocket_listener, daemon=True).start()
            Thread(target=self._process_events, daemon=True).start()
            Thread(target=self._background_refresher, daemon=True).start()
            self.logger.info("Real-time scanner started")
        else:
            self.logger.warning("Scanner is already running")

    def stop(self):
        """Stop the scanner"""
        self.active = False
        self.logger.info("Scanner stopped")

    def get_current_tickers(self):
        """Get current ticker list"""
        with self.cache_lock:
            return self.ticker_cache.copy()

    def check_for_new_listings(self):
        """Check for new listings by comparing with cache"""
        with self.cache_lock:
            old_tickers = set(self.ticker_cache["ticker"])
            old_count = len(old_tickers)
        
        self.logger.info(f"Checking for new listings. Current tickers: {old_count}")
        self._refresh_all_tickers()
        
        with self.cache_lock:
            new_tickers = set(self.ticker_cache["ticker"])
            new_count = len(new_tickers)
        
        added = new_tickers - old_tickers
        removed = old_tickers - new_tickers
        
        if added:
            self.logger.info(f"New tickers detected: {len(added)}")
            self.logger.debug(f"New tickers: {', '.join(sorted(added)[:10])}{'...' if len(added) > 10 else ''}")
        if removed:
            self.logger.info(f"Delisted tickers: {len(removed)}")
            self.logger.debug(f"Delisted tickers: {', '.join(sorted(removed)[:10])}{'...' if len(removed) > 10 else ''}")
        
        self.logger.info(f"Total tickers after check: {new_count}")
        return added, removed

class MarketRegimeAnalyzer:
    def __init__(self, n_states=GlobalConfig.N_STATES, 
                 polygon_api_key=GlobalConfig.POLYGON_API_KEY, 
                 min_tickers=GlobalConfig.MIN_TICKERS_FOR_COMPOSITE, 
                 min_days_data=GlobalConfig.MIN_DAYS_DATA):
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
        self.min_tickers = min_tickers  # Minimum tickers for reliable composite
        self.min_days_data = min_days_data  # Minimum days of data required
        self.logger = logging.getLogger("MarketRegimeAnalyzer")

    def prepare_market_data(self, tickers):
        """Build market composite from qualified tickers only"""
        prices_data = []
        qualified_tickers = []
        mcaps = {}
        self.logger.info("Filtering tickers with sufficient data...")

        # Step 1: Identify tickers with sufficient data
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {
                executor.submit(self.validate_ticker_data, ticker): ticker 
                for ticker in tickers
            }
            
            with logging_redirect_tqdm():
                pbar = tqdm(as_completed(future_to_ticker), total=len(tickers), desc="Validating tickers")
                for future in pbar:
                    ticker = future_to_ticker[future]
                    try:
                        result = future.result()
                        if result:
                            qualified_tickers.append(ticker)
                            mcaps[ticker] = result["mcap"]
                    except:
                        pass
                    pbar.set_postfix_str(ticker)
        
        # Ensure we meet minimum ticker requirement
        if len(qualified_tickers) < self.min_tickers:
            self.logger.warning(f"Only {len(qualified_tickers)} qualified tickers found (min: {self.min_tickers})")
            if len(qualified_tickers) < 100:
                raise ValueError("Insufficient qualified tickers for composite")
        
        # Use top 500 by market cap or all available if less than 500
        selected_tickers = sorted(qualified_tickers, key=lambda t: mcaps[t], reverse=True)[:self.min_tickers]
        total_mcap = sum(mcaps[t] for t in selected_tickers)
        
        # Step 2: Build composite from qualified tickers
        prices_data = []
        self.logger.info(f"Building composite from {len(selected_tickers)} qualified tickers")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self.fetch_stock_data, symbol): symbol
                for symbol in selected_tickers
            }

            with logging_redirect_tqdm():
                pbar = tqdm(as_completed(futures), total=len(selected_tickers), desc="Fetching qualified data")
                for future in pbar:
                    symbol = futures[future]
                    try:
                        prices = future.result()
                        if prices is not None:
                            weight = mcaps.get(symbol, 1) / total_mcap
                            prices_data.append(prices * weight)
                    except Exception as e:
                        self.logger.error(f"Error processing {symbol}: {str(e)}")
                    pbar.set_postfix_str(symbol)

        if not prices_data:
            raise ValueError("Insufficient data to create market composite")
            
        # Create composite index
        composite = pd.concat(prices_data, axis=1)
        composite.columns = selected_tickers
        composite = composite.ffill().bfill()
        return composite.sum(axis=1).dropna()

    def validate_ticker_data(self, ticker):
        """Check if ticker has sufficient data and return market cap"""
        cache_file = os.path.join(GlobalConfig.DATA_CACHE_DIR, f"validation_{ticker}.pkl")
        if os.path.exists(cache_file):
            # Check cache expiration
            mod_time = os.path.getmtime(cache_file)
            if (time.time() - mod_time) < (GlobalConfig.CACHE_EXPIRY_DAYS * 86400):
                return joblib.load(cache_file)
        
        # Fetch price data
        prices = self.fetch_stock_data(ticker, days=400)  # Extra buffer
        if prices is None or len(prices) < self.min_days_data:
            return None
            
        # Fetch market cap
        mcap = self.get_market_cap(ticker) or 1
        
        # Cache validation result
        result = {"prices": prices, "mcap": mcap}
        joblib.dump(result, cache_file)
        return result

    def analyze_regime(self, index_data, n_states=None):
        """Enhanced HMM analysis with robust convergence handling"""
        if n_states is None:
            n_states = self.model.n_components

        # Calculate features
        log_returns = np.log(index_data).diff().dropna()
        features = pd.DataFrame({
            "returns": log_returns,
            "volatility": log_returns.rolling(21).std(),
            "momentum": log_returns.rolling(14).mean(),
            "rsi": talib.RSI(index_data, timeperiod=14),
            "macd": talib.MACD(index_data)[0],
            "adx": talib.ADX(index_data, index_data, index_data, timeperiod=14)
        }).dropna()

        # Validate feature data
        if len(features) < GlobalConfig.MIN_FEATURE_DAYS:
            self.logger.error(f"Insufficient feature data: {len(features)} days (min: {GlobalConfig.MIN_FEATURE_DAYS})")
            raise ValueError(f"Only {len(features)} days of feature data")
            
        # Scale features and check for constant columns
        scaled_features = self.feature_scaler.fit_transform(features)
        
        # Handle NaNs/Infs
        if np.isnan(scaled_features).any() or np.isinf(scaled_features).any():
            self.logger.warning("Features contain NaNs/Infs - filling with 0")
            scaled_features = np.nan_to_num(scaled_features)
        
        # Check for constant features
        non_constant_mask = (scaled_features.std(axis=0) > 1e-8)
        if not non_constant_mask.all():
            constant_cols = features.columns[~non_constant_mask].tolist()
            self.logger.warning(f"Dropping constant columns: {constant_cols}")
            scaled_features = scaled_features[:, non_constant_mask]
        
        if scaled_features.shape[1] == 0:
            raise ValueError("No non-constant features available after filtering")
            
        # Multiple restarts with adaptive tolerance
        n_restarts = GlobalConfig.HMM_RESTARTS
        best_model = None
        best_log_likelihood = -np.inf
        best_seed = 0

        for seed in range(n_restarts):
            try:
                # Adaptive tolerance that increases with each restart
                current_tol = 1e-6 * (1.5 ** seed) if GlobalConfig.ADAPTIVE_TOL else 1e-6
                
                current_model = hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type="diag",
                    n_iter=GlobalConfig.HMM_MAX_ITER,
                    tol=current_tol,
                    init_params="se",
                    params="stmc",
                    random_state=seed,
                )
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    current_model.fit(scaled_features)
                
                log_likelihood = current_model.score(scaled_features)
                if log_likelihood > best_log_likelihood:
                    best_model = current_model
                    best_log_likelihood = log_likelihood
                    best_seed = seed
                    
                self.logger.debug(f"Restart {seed+1}/{n_restarts}: log-likelihood={log_likelihood:.2f}, tol={current_tol:.1e}")
                    
            except Exception as e:
                self.logger.warning(f"Restart {seed+1} failed: {str(e)}")

        if best_model is None:
            raise RuntimeError("All restarts failed for HMM training")
            
        self.logger.info(f"Selected model from restart {best_seed} with log-likelihood: {best_log_likelihood:.2f}")
        model = best_model

        # Determine state labels based on return and volatility
        state_stats = []
        for i in range(model.n_components):
            state_return = model.means_[i][0]
            state_vol = model.means_[i][1]
            state_stats.append((i, state_return, state_vol))
            
        state_stats.sort(key=lambda x: (x[1], -x[2]))
        
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

        states = model.predict(scaled_features)
        state_probs = model.predict_proba(scaled_features)
        
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
        
        avg_durations = {}
        for state, durs in durations.items():
            avg_durations[state] = sum(durs) / len(durs) if durs else 0
            
        return avg_durations

    def fetch_stock_data(self, symbol, days=365):
        cache_file = os.path.join(GlobalConfig.DATA_CACHE_DIR, f"{symbol}_{days}.pkl")
        if os.path.exists(cache_file):
            # Check cache expiration
            mod_time = os.path.getmtime(cache_file)
            if (time.time() - mod_time) < (GlobalConfig.CACHE_EXPIRY_DAYS * 86400):
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
            time.sleep(GlobalConfig.API_RATE_LIMIT_DELAY)
            response = requests.get(url, params=params, timeout=15)
            
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
            
            # Ensure no zero or negative prices
            result = result.replace(0, np.nan).ffill().bfill()
            result = np.maximum(result, 0.01)
            
            result.to_pickle(cache_file)
            return result
            
        except requests.exceptions.Timeout:
            self.logger.warning(f"Timeout fetching {symbol}, skipping")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching {symbol}: {str(e)}")
            return None
            
    def get_market_cap(self, symbol):
        cache_file = os.path.join(GlobalConfig.DATA_CACHE_DIR, f"mcap_{symbol}.pkl")
        if os.path.exists(cache_file):
            # Check cache expiration
            mod_time = os.path.getmtime(cache_file)
            if (time.time() - mod_time) < (GlobalConfig.CACHE_EXPIRY_DAYS * 86400):
                return joblib.load(cache_file)
            
        url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
        params = {"apiKey": self.polygon_api_key}
        
        try:
            time.sleep(GlobalConfig.API_RATE_LIMIT_DELAY)
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 429:
                time.sleep(30)
                return self.get_market_cap(symbol)
                
            if response.status_code == 200:
                data = response.json().get("results", {})
                mcap = data.get("market_cap", 0)
                joblib.dump(mcap, cache_file)
                return mcap
        except Exception as e:
            self.logger.error(f"Error getting market cap for {symbol}: {str(e)}")
            
        return 0


class SectorRegimeSystem:
    def __init__(self, polygon_api_key=GlobalConfig.POLYGON_API_KEY):
        self.sector_mappings = {}
        self.sector_composites = {}
        self.sector_analyzers = {}
        self.overall_analyzer = MarketRegimeAnalyzer(polygon_api_key=polygon_api_key)
        self.sector_weights = {}
        self.sector_scores = {}
        self.polygon_api_key = polygon_api_key
        self.current_regime = None
        self.logger = logging.getLogger("SectorRegimeSystem")

    def map_tickers_to_sectors(self, tickers):
        self.sector_mappings = {}
        cache_file = os.path.join(GlobalConfig.DATA_CACHE_DIR, "sector_mappings.pkl")
        
        if os.path.exists(cache_file):
            # Check cache expiration
            mod_time = os.path.getmtime(cache_file)
            if (time.time() - mod_time) < (GlobalConfig.CACHE_EXPIRY_DAYS * 86400):
                self.sector_mappings = joblib.load(cache_file)
                return self.sector_mappings

        def map_single_ticker(symbol):
            cache_file = os.path.join(GlobalConfig.DATA_CACHE_DIR, f"sector_{symbol}.pkl")
            if os.path.exists(cache_file):
                # Check cache expiration
                mod_time = os.path.getmtime(cache_file)
                if (time.time() - mod_time) < (GlobalConfig.CACHE_EXPIRY_DAYS * 86400):
                    return symbol, joblib.load(cache_file)
                
            url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
            params = {"apiKey": self.polygon_api_key}
            try:
                time.sleep(GlobalConfig.API_RATE_LIMIT_DELAY)
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 429:
                    time.sleep(30)
                    return map_single_ticker(symbol)
                    
                if response.status_code == 200:
                    data = response.json().get("results", {})
                    sector = data.get("sic_description", "Unknown")
                    if sector == "Unknown":
                        sector = data.get("primary_exchange", "Unknown")
                    
                    joblib.dump(sector, cache_file)
                    return symbol, sector
            except Exception as e:
                self.logger.warning(f"Sector mapping failed for {symbol}: {str(e)}")
            return symbol, "Unknown"

        total_tickers = len(tickers)
        self.logger.info(f"Mapping {total_tickers} tickers to sectors")
        
        with logging_redirect_tqdm(), ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(map_single_ticker, symbol): symbol for symbol in tickers}
            
            pbar = tqdm(total=total_tickers, desc="Mapping sectors", unit="ticker")
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    symbol, sector = future.result()
                    if sector != "Unknown":
                        self.sector_mappings.setdefault(sector, []).append(symbol)
                except Exception as e:
                    self.logger.warning(f"Error processing {symbol}: {str(e)}")
                finally:
                    pbar.update(1)
                    pbar.set_postfix_str(symbol)
            
            pbar.close()

        self.sector_mappings = {
            k: v for k, v in self.sector_mappings.items() 
            if k != "Unknown" and len(v) > GlobalConfig.MIN_SECTOR_TICKERS
        }
        
        joblib.dump(self.sector_mappings, cache_file)
        self.logger.info(f"Mapped to {len(self.sector_mappings)} sectors")
        return self.sector_mappings

    def calculate_sector_weights(self):
        total_mcap = 0
        sector_mcaps = {}
        
        self.logger.info("Calculating sector weights...")
        
        for sector, tickers in self.sector_mappings.items():
            sector_mcap = 0
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_ticker = {
                    executor.submit(self.overall_analyzer.get_market_cap, symbol): symbol
                    for symbol in tickers[:30]
                }
                
                with logging_redirect_tqdm():
                    pbar = tqdm(as_completed(future_to_ticker), total=min(30, len(tickers)), desc=f"Processing {sector[:15]}...")
                    for future in pbar:
                        symbol = future_to_ticker[future]
                        try:
                            mcap = future.result()
                            sector_mcap += mcap if mcap else 0
                        except:
                            pass
                        pbar.set_postfix_str(symbol)

            sector_mcaps[sector] = sector_mcap
            total_mcap += sector_mcap

        self.sector_weights = {
            sector: mcap / total_mcap if total_mcap > 0 else 1 / len(sector_mcaps)
            for sector, mcap in sector_mcaps.items()
        }
        self.logger.info("Sector weights calculated")
        return self.sector_weights

    def build_sector_composites(self, sample_size=GlobalConfig.SECTOR_COMPOSITE_SAMPLE_SIZE):
        self.logger.info("\nBuilding sector composites...")
        self.sector_composites = {}
        cache_file = os.path.join(GlobalConfig.DATA_CACHE_DIR, "sector_composites.pkl")
        
        if os.path.exists(cache_file):
            # Check cache expiration
            mod_time = os.path.getmtime(cache_file)
            if (time.time() - mod_time) < (GlobalConfig.CACHE_EXPIRY_DAYS * 86400):
                self.sector_composites = joblib.load(cache_file)
                return self.sector_composites

        total_sectors = len(self.sector_mappings)
        self.logger.info(f"Building composites for {total_sectors} sectors")
        
        # Step 1: Precompute all required market caps in parallel
        all_tickers = []
        ticker_to_sector = {}
        for sector, tickers in self.sector_mappings.items():
            for ticker in tickers[:sample_size]:
                all_tickers.append(ticker)
                ticker_to_sector[ticker] = sector
        
        # Fetch all market caps in parallel
        with ThreadPoolExecutor(max_workers=15) as executor:
            market_caps = list(tqdm(
                executor.map(self.overall_analyzer.get_market_cap, all_tickers),
                total=len(all_tickers),
                desc="Fetching Market Caps"
            ))
        
        # Create a dictionary of ticker to market cap
        ticker_mcaps = dict(zip(all_tickers, market_caps))
        
        # Calculate total market cap per sector
        sector_total_mcaps = {}
        for ticker, mcap in ticker_mcaps.items():
            if mcap is None or mcap <= 0:
                continue
            sector = ticker_to_sector[ticker]
            sector_total_mcaps.setdefault(sector, 0)
            sector_total_mcaps[sector] += mcap

        # Step 2: Process sectors in parallel
        def process_sector(sector):
            prices_data = []
            sector_tickers = self.sector_mappings[sector][:sample_size]
            total_mcap = sector_total_mcaps.get(sector, 1)
            
            # Fetch price data in parallel for this sector
            with ThreadPoolExecutor(max_workers=10) as sector_executor:
                futures = {}
                for symbol in sector_tickers:
                    weight = (ticker_mcaps.get(symbol, 1) / total_mcap)
                    futures[sector_executor.submit(
                        self.overall_analyzer.fetch_stock_data, symbol
                    )] = (symbol, weight)
                
                for future in as_completed(futures):
                    symbol, weight = futures[future]
                    try:
                        prices = future.result()
                        if prices is not None and len(prices) >= 200:
                            prices_data.append(prices * weight)
                    except Exception as e:
                        self.logger.warning(f"Error processing {symbol}: {str(e)}")
            
            if prices_data:
                composite = pd.concat(prices_data, axis=1)
                composite = composite.ffill().bfill()
                composite = composite.sum(axis=1).dropna()
                
                # Clean composite data
                composite = composite.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                composite = np.maximum(composite, 0.01)  # Ensure positive prices
                
                if len(composite) < GlobalConfig.MIN_DAYS_DATA:
                    self.logger.warning(f"Insufficient data for {sector}: {len(composite)} days")
                    return sector, None
                    
                return sector, composite
            return sector, None

        # Process all sectors in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:  # Limit sector parallelism
            futures = {
                executor.submit(process_sector, sector): sector
                for sector in self.sector_mappings
            }
            
            with logging_redirect_tqdm():
                pbar = tqdm(as_completed(futures), total=len(futures), desc="Building Composites")
                for future in pbar:
                    sector = futures[future]
                    try:
                        sector, composite = future.result()
                        if composite is not None:
                            self.sector_composites[sector] = composite
                        else:
                            self.logger.warning(f"Composite creation failed for {sector}")
                    except Exception as e:
                        self.logger.error(f"Error processing sector {sector}: {str(e)}")
                    pbar.set_postfix_str(sector[:15])
        
        joblib.dump(self.sector_composites, cache_file)
        self.logger.info(f"Built {len(self.sector_composites)} sector composites")
        return self.sector_composites

    def analyze_sector_regimes(self, n_states=GlobalConfig.N_STATES):
        self.logger.info("\nAnalyzing sector regimes...")
        self.sector_analyzers = {}
        
        # Build market composite from qualified tickers
        tickers = [t for sublist in self.sector_mappings.values() for t in sublist]
        self.market_composite = self.overall_analyzer.prepare_market_data(tickers)
        market_result = self.overall_analyzer.analyze_regime(self.market_composite)
        self.current_regime = market_result["regimes"][-1]
        self.logger.info(f"Overall market regime: {self.current_regime}")
        
        total_sectors = len(self.sector_composites)
        self.logger.info(f"Analyzing {total_sectors} sectors")
        
        with logging_redirect_tqdm():
            pbar = tqdm(total=total_sectors, desc="Analyzing sectors", unit="sector")
            
            for sector, composite in self.sector_composites.items():
                try:
                    # Pre-check composite data
                    if len(composite) < GlobalConfig.MIN_FEATURE_DAYS:
                        self.logger.warning(f"Skipping {sector}: insufficient data ({len(composite)} days)")
                        continue
                        
                    # Ensure data quality
                    if np.isnan(composite).any() or np.isinf(composite).any():
                        self.logger.warning(f"{sector} has NaN/Inf values - cleaning")
                        composite = composite.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                        self.sector_composites[sector] = composite
                        
                    analyzer = MarketRegimeAnalyzer(polygon_api_key=self.polygon_api_key)
                    results = analyzer.analyze_regime(composite, n_states=n_states)
                    self.sector_analyzers[sector] = {
                        "results": results,
                        "composite": composite,
                        "volatility": composite.pct_change().std(),
                        "analyzer": analyzer
                    }
                except Exception as e:
                    self.logger.error(f"Error analyzing {sector}: {str(e)}")
                    # Provide specific diagnostics for common issues
                    if "days of feature data" in str(e):
                        self.logger.error(f"Insufficient feature data for {sector}: "
                                        f"{len(composite)} days total")
                    elif "converge" in str(e).lower():
                        self.logger.error(f"Convergence failed for {sector} - consider increasing HMM_RESTARTS")
                
                pbar.update(1)
                pbar.set_postfix_str(sector[:15])
            
            pbar.close()
                
        self.logger.info("Sector regime analysis completed")
        return self.sector_analyzers

    def calculate_sector_scores(self):
        self.sector_scores = {}
        if not self.sector_analyzers:
            return pd.Series()

        self.logger.info("Calculating sector scores...")
        
        for sector, data in self.sector_analyzers.items():
            try:
                if "results" not in data:
                    continue
                    
                current_probs = data["results"]["probabilities"][-1]
                state_labels = data["results"].get("state_labels", {})
                
                momentum = data["composite"].pct_change(21).iloc[-1] if len(data["composite"]) > 21 else 0
                
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
                
                base_score = bull_prob - bear_prob
                momentum_factor = 1 + (momentum * 5)
                adjusted_score = base_score * momentum_factor
                weight = self.sector_weights.get(sector, 0.01)
                self.sector_scores[sector] = adjusted_score * (1 + weight)
                
            except Exception as e:
                self.logger.error(f"Error calculating score for {sector}: {str(e)}")
                self.sector_scores[sector] = 0

        self.logger.info("Sector scores calculated")
        return pd.Series(self.sector_scores).sort_values(ascending=False)

    def get_regime_aware_screener(self):
        if not self.current_regime:
            return {}
            
        screening_rules = {
            "Bull": {
                "filters": [
                    {"indicator": "RSI", "min": 50, "max": 70},
                    {"indicator": "MACD", "signal": "positive"},
                    {"indicator": "Volume", "min": 1.5, "lookback": 21}
                ],
                "types": ["Breakouts", "Momentum"]
            },
            "Bear": {
                "filters": [
                    {"indicator": "RSI", "min": 30, "max": 50},
                    {"indicator": "Volatility", "max": 0.4},
                    {"indicator": "ShortFloat", "min": 0.1}
                ],
                "types": ["ShortSqueeze", "Oversold"]
            },
            "Severe Bear": {
                "filters": [
                    {"indicator": "RSI", "max": 30},
                    {"indicator": "DebtToEquity", "max": 0.7}
                ],
                "types": ["Oversold", "Fundamentals"]
            },
            "Strong Bull": {
                "filters": [
                    {"indicator": "RSI", "max": 80},
                    {"indicator": "Volume", "min": 2.0, "lookback": 50},
                    {"indicator": "EarningsGrowth", "min": 0.2}
                ],
                "types": ["Momentum", "Growth"]
            }
        }
        
        return screening_rules.get(self.current_regime, {})

    def validate_model(self, data, windows=GlobalConfig.VALIDATION_WINDOWS, min_test_size=10):
        self.logger.info("\nRunning model validation...")
        accuracy = []
        state_transitions = []
        
        for i in range(windows, len(data) - min_test_size):
            train_data = data.iloc[:i]
            test_data = data.iloc[i:i+min_test_size]
            
            analyzer = MarketRegimeAnalyzer()
            train_result = analyzer.analyze_regime(train_data)
            
            test_features = analyzer.feature_scaler.transform(
                analyzer.prepare_features(test_data))
            test_states = analyzer.model.predict(test_features)
            
            full_result = analyzer.analyze_regime(pd.concat([train_data, test_data]))
            actual_states = full_result["regimes"][-min_test_size:]
            
            predicted = [train_result["state_labels"][s] for s in test_states]
            acc = sum(p == a for p, a in zip(predicted, actual_states)) / min_test_size
            accuracy.append(acc)
            
            state_transitions.append((train_result["regimes"][-1], predicted[0]))
        
        avg_accuracy = np.mean(accuracy) if accuracy else 0
        transition_matrix = pd.crosstab(
            [t[0] for t in state_transitions],
            [t[1] for t in state_transitions],
            normalize='index'
        )
        
        self.logger.info(f"Model Validation Accuracy: {avg_accuracy:.2%}")
        self.logger.info("\nState Transition Probabilities:")
        self.logger.info(transition_matrix)
        
        return {
            "accuracy": avg_accuracy,
            "transition_matrix": transition_matrix
        }
        
    def prepare_features(self, price_data):
        log_returns = np.log(price_data).diff().dropna()
        features = pd.DataFrame({
            "returns": log_returns,
            "volatility": log_returns.rolling(21).std(),
            "momentum": log_returns.rolling(14).mean(),
            "rsi": talib.RSI(price_data, timeperiod=14),
            "macd": talib.MACD(price_data)[0]
        }).dropna()
        return features


class MarketScanner:
    def __init__(self, api_key=GlobalConfig.POLYGON_API_KEY):
        self.ticker_scanner = PolygonTickerScanner(api_key=api_key)
        self.regime_system = SectorRegimeSystem(polygon_api_key=api_key)
        self.regime_results = None
        self.sector_scores = None
        self.active = False
        self.analysis_lock = Lock()
        self.logger = logging.getLogger("MarketScanner")

    def start(self):
        """Start the integrated market scanning system"""
        if not self.active:
            self.ticker_scanner.start()
            self.active = True
            
            # Run initial analysis immediately
            Thread(target=self.run_full_regime_analysis, daemon=True).start()
            
            # Start periodic analysis scheduler
            Thread(target=self._periodic_regime_analysis, daemon=True).start()
            self.logger.info("Market Scanner started with integrated regime analysis")
        else:
            self.logger.warning("Scanner is already running")

    def stop(self):
        """Stop all components"""
        self.ticker_scanner.stop()
        self.active = False
        self.logger.info("Market Scanner stopped")

    def _periodic_regime_analysis(self):
        """Run regime analysis at regular intervals with immediate first run"""
        # First run was already done in start(), now schedule daily runs
        while self.active:
            try:
                now = datetime.now()
                next_run = datetime(now.year, now.month, now.day) + timedelta(days=1, hours=16)
                wait_seconds = (next_run - now).total_seconds()
                
                if wait_seconds > 0:
                    self.logger.info(f"Next regime analysis scheduled at {next_run}")
                    time.sleep(wait_seconds)
                
                if not self.active:
                    break
                    
                self.run_full_regime_analysis()
                
            except Exception as e:
                self.logger.error(f"Periodic regime analysis failed: {e}")
                time.sleep(3600)

    def run_full_regime_analysis(self):
        """Run complete market regime analysis (with locking to prevent concurrent runs)"""
        if not self.analysis_lock.acquire(blocking=False):
            self.logger.info("Analysis already in progress. Skipping duplicate run.")
            return
            
        try:
            original_log_level = logging.getLogger().level
            logging.getLogger().setLevel(logging.WARNING)
            
            with logging_redirect_tqdm():
                self.logger.info("Starting full market regime analysis")
                
                tickers_df = self.ticker_scanner.get_current_tickers()
                tickers = tickers_df['ticker'].tolist()
                self.logger.info(f"Loaded {len(tickers)} tickers for analysis")
                
                self.regime_system.map_tickers_to_sectors(tickers)
                self.logger.info(f"Mapped tickers to {len(self.regime_system.sector_mappings)} sectors")
                
                self.regime_system.calculate_sector_weights()
                self.logger.info("Calculated sector weights")
                
                self.regime_system.build_sector_composites()
                self.logger.info("Built sector composites")
                
                self.regime_system.analyze_sector_regimes()
                self.logger.info("Completed sector regime analysis")
                
                self.sector_scores = self.regime_system.calculate_sector_scores()
                self.logger.info("Calculated sector scores")
                
                current_regime = self.regime_system.current_regime
                self.logger.info(f"Current market regime: {current_regime}")
                
                self.regime_results = {
                    "market_regime": current_regime,
                    "sector_scores": self.sector_scores,
                    "screening_rules": self.regime_system.get_regime_aware_screener(),
                    "timestamp": datetime.now().isoformat()
                }
            
            self.logger.info("Market regime analysis completed")
            return self.regime_results
        except Exception as e:
            self.logger.error(f"Error during regime analysis: {str(e)}", exc_info=True)
            return None
        finally:
            logging.getLogger().setLevel(original_log_level)
            self.analysis_lock.release()

    def get_current_recommendations(self):
        """Get current market recommendations"""
        if not self.regime_results:
            return {"status": "No analysis available"}
        
        return {
            "market_regime": self.regime_results["market_regime"],
            "top_sectors": self.regime_results["sector_scores"].head(5).to_dict(),
            "screening_rules": self.regime_results["screening_rules"],
            "last_updated": self.regime_results["timestamp"]
        }


if __name__ == "__main__":
    scanner = MarketScanner()
    
    try:
        scanner.start()
        logger.info("Market Scanner running. Press Ctrl+C to stop.")
        
        while True:
            time.sleep(60)
            if int(time.time()) % 300 == 0:
                logger.info("\nCurrent Market Recommendations:")
                logger.info(json.dumps(scanner.get_current_recommendations(), indent=2))
                
    except KeyboardInterrupt:
        logger.info("\nStopping scanner...")
    finally:
        scanner.stop()