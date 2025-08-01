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
    POLYGON_API_KEY = "YOUR_POLYGON_API_KEY"
    ALPACA_API_KEY = "YOUR_ALPACA_API_KEY"
    ALPACA_SECRET_KEY = "YOUR_ALPACA_SECRET_KEY"
    DISCORD_WEBHOOK_URL = "YOUR_DISCORD_WEBHOOK_URL"

# -------------------- ADJUSTABLE PARAMETERS -------------------- #
# Ticker Scanner Settings
EXCHANGES = ["XNAS", "XNYS", "XASE"]  # Stock exchanges to monitor
TICKER_CACHE_FILE = "ticker_cache.parquet"
SCANNER_LOG_LEVEL = logging.INFO
SCANNER_MAX_WORKERS = 15
SCANNER_REFRESH_INTERVAL = 3600  # 1 hour
WS_RECONNECT_DELAY = 5
MAX_RECONNECT_ATTEMPTS = 10

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

# ======================== TICKER SCANNER ======================== #
class PolygonTickerScanner:
    def __init__(self, api_key=POLYGON_API_KEY, exchanges=EXCHANGES, 
                 cache_file=TICKER_CACHE_FILE, log_level=SCANNER_LOG_LEVEL,
                 max_workers=SCANNER_MAX_WORKERS, refresh_interval=SCANNER_REFRESH_INTERVAL):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v3/reference"
        self.websocket_url = "wss://socket.polygon.io/stocks"
        self.cache_file = cache_file
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
        """Initialize ticker cache with thread-safe locking"""
        with self.cache_lock:
            if os.path.exists(self.cache_file):
                try:
                    self.ticker_cache = pd.read_parquet(self.cache_file)
                    # Handle legacy cache without 'type' column
                    if 'type' not in self.ticker_cache.columns:
                        self.ticker_cache['type'] = 'CS'  # Assume existing are stocks
                        self.logger.warning("Legacy cache detected. Added 'type' column with default 'CS'")
                    self.logger.info(f"Loaded cache with {len(self.ticker_cache)} tickers")
                    
                    # Start background refresh if needed
                    if time.time() - self.last_refresh_time > self.refresh_interval:
                        Thread(target=self._refresh_all_tickers, daemon=True).start()
                except Exception as e:
                    self.logger.error(f"Error loading cache: {e}")
                    self.ticker_cache = pd.DataFrame(columns=["ticker", "name", "primary_exchange", "last_updated_utc", "type"])
                    self._refresh_all_tickers()
            else:
                self.ticker_cache = pd.DataFrame(columns=["ticker", "name", "primary_exchange", "last_updated_utc", "type"])
                self._refresh_all_tickers()

    def _call_polygon_api(self, url):
        """Call Polygon API without rate limits"""
        try:
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
                
                # Submit current batch of URLs
                for url in current_urls:
                    futures.append(executor.submit(self._fetch_exchange_page, exchange, url))
                
                # Process results
                for future in as_completed(futures):
                    results, next_url = future.result()
                    if results:
                        # FILTER: Only include common stocks (type "CS")
                        stock_results = [r for r in results if r.get('type') == 'CS']
                        all_results.extend(stock_results)
                        page_count += 1
                        
                    if next_url:
                        next_urls.append(next_url)
                        
                self.logger.debug(f"Processed {len(current_urls)} pages for {exchange}. Total pages: {page_count}")
        
        self.logger.info(f"Completed fetch for {exchange}: {len(all_results)} STOCKS across {page_count} pages")
        return all_results

    def _refresh_all_tickers(self):
        """Fetch all tickers from Polygon using parallel execution"""
        start_time = time.time()
        self.logger.info("Starting full ticker refresh (STOCKS ONLY)")
        
        # Get current ticker count
        with self.cache_lock:
            current_count = len(self.ticker_cache)
        self.logger.info(f"Current STOCKS in cache: {current_count}")
        
        # Fetch all exchanges in parallel
        with ThreadPoolExecutor(max_workers=len(self.exchanges)) as exchange_executor:
            exchange_futures = {exchange_executor.submit(self._fetch_exchange_tickers, exchange): exchange 
                                for exchange in self.exchanges}
            
            exchange_counts = {}
            all_results = []
            for future in as_completed(exchange_futures):
                exchange = exchange_futures[future]
                try:
                    exchange_results = future.result()
                    all_results.extend(exchange_results)
                    exchange_counts[exchange] = len(exchange_results)
                    self.logger.debug(f"Finished {exchange}: {len(exchange_results)} stocks")
                except Exception as e:
                    self.logger.error(f"Error processing {exchange}: {e}")
                    exchange_counts[exchange] = 0
        
        # Log summary of tickers per exchange
        self.logger.info("Stock refresh summary:")
        total_stocks = 0
        for exchange in self.exchanges:
            count = exchange_counts.get(exchange, 0)
            self.logger.info(f"  {exchange}: {count} stocks")
            total_stocks += count
        
        # Process and cache results
        if all_results:
            df = pd.DataFrame(all_results)
            
            # Filter to only necessary columns + type
            if not df.empty:
                new_cache = df[["ticker", "name", "primary_exchange", "last_updated_utc", "type"]].copy()
            else:
                new_cache = pd.DataFrame(columns=["ticker", "name", "primary_exchange", "last_updated_utc", "type"])
            
            with self.cache_lock:
                # Replace cache completely for this refresh
                self.ticker_cache = new_cache
                self.ticker_cache.to_parquet(self.cache_file)
                new_count = len(self.ticker_cache)
            
            elapsed = time.time() - start_time
            self.logger.info(f"Total STOCKS: {total_stocks}")
            self.logger.info(f"Refresh completed in {elapsed:.2f} seconds")
            self.last_refresh_time = time.time()
        else:
            self.logger.warning("No stocks fetched during refresh")
            with self.cache_lock:
                current_count = len(self.ticker_cache)
            self.logger.info(f"Total stocks remains: {current_count}")

    def _update_single_ticker(self, ticker):
        """Fetch and update a single ticker in cache (stocks only)"""
        if ticker in self.known_missing_tickers:
            return
        
        url = f"{self.base_url}/tickers/{ticker}?apiKey={self.api_key}"
        try:
            data = self._call_polygon_api(url)
            if data and data.get("status") == "OK" and data.get("results"):
                ticker_data = data["results"]
                
                # FILTER: Only process common stocks (type "CS")
                if ticker_data.get("type") != "CS":
                    self.logger.warning(f"Ignoring non-stock security: {ticker} (type={ticker_data.get('type')})")
                    self.known_missing_tickers.add(ticker)
                    
                    # Remove from cache if exists
                    with self.cache_lock:
                        if ticker in self.ticker_cache["ticker"].values:
                            self.ticker_cache = self.ticker_cache[self.ticker_cache["ticker"] != ticker]
                            self.ticker_cache.to_parquet(self.cache_file)
                            self.logger.info(f"Removed non-stock: {ticker}")
                    return
                
                # Update stock in cache
                new_row = {
                    "ticker": ticker,
                    "name": ticker_data.get("name", ""),
                    "primary_exchange": ticker_data.get("primary_exchange", ""),
                    "last_updated_utc": ticker_data.get("last_updated_utc", ""),
                    "type": "CS"  # Explicitly mark as stock
                }
                
                with self.cache_lock:
                    # Update in place without full cache reload
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
                
                self.logger.info(f"Updated stock: {ticker}")
                self.logger.info(f"Total stocks after update: {new_count}")
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
                self.current_reconnect_attempts = 0  # Reset on success
                
                # Authentication
                auth_msg = json.dumps({"action": "auth", "params": self.api_key})
                ws.send(auth_msg)
                
                # Subscription
                sub_msg = json.dumps({
                    "action": "subscribe",
                    "params": "T.*,A.*,AM.*"  # Trades, Aggregates, Minute Aggs
                })
                ws.send(sub_msg)
                
                # Connection maintenance
                last_ping = time.time()
                while self.active:
                    try:
                        # Set timeout to allow periodic ping checks
                        data = ws.recv()
                        if data:
                            self.event_queue.put(json.loads(data))
                        
                        # Send ping every 30 seconds to maintain connection
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
        batch_timeout = 0.5  # seconds
        
        while self.active:
            try:
                batch = []
                start_time = time.time()
                
                # Collect events with timeout
                while len(batch) < batch_size and (time.time() - start_time) < batch_timeout:
                    try:
                        event = self.event_queue.get(timeout=0.1)
                        batch.append(event)
                    except Empty:
                        continue
                
                # Process batch
                for event in batch:
                    if isinstance(event, list):  # Handle batched events
                        for e in event:
                            self._handle_single_event(e)
                    else:
                        self._handle_single_event(event)
                
                # Sleep if no events processed
                if not batch:
                    time.sleep(0.5)
                    
            except Exception as e:
                self.logger.error(f"Event processing error: {e}", exc_info=True)

    def _handle_single_event(self, event):
        """Handle individual WebSocket events"""
        try:
            if event.get("ev") == "T":  # Trade event
                ticker = event["sym"]
                
                # Check if ticker exists in cache
                with self.cache_lock:
                    exists = ticker in self.ticker_cache["ticker"].values
                
                if not exists and ticker not in self.known_missing_tickers:
                    self.logger.info(f"New ticker detected: {ticker}")
                    # Use thread pool for parallel updates if needed
                    Thread(target=self._update_single_ticker, args=(ticker,), daemon=True).start()
                    
                # Add your custom processing logic here
                # price = event["p"]
                # print(f"Trade: {ticker} @ {price}")
                
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
            # Start WebSocket listener thread
            Thread(target=self._websocket_listener, daemon=True).start()
            # Start event processor thread
            Thread(target=self._process_events, daemon=True).start()
            # Start background refresh thread
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

        print("\nBuilding market composite from multiple exchanges...")

        # First pass: Get market caps for weighting
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {
                executor.submit(self.get_market_cap, ticker): ticker 
                for ticker in tickers[:sample_size]
            }
            for future in tqdm(as_completed(future_to_ticker), total=min(sample_size, len(tickers)), desc="Fetching Market Caps"):
                ticker = future_to_ticker[future]
                try:
                    mcaps[ticker] = future.result() or 1  # Default to 1 if None
                except:
                    mcaps[ticker] = 1

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

            for future in tqdm(as_completed(futures), total=min(sample_size, len(tickers)), desc="Fetching Price Data"):
                symbol = futures[future]
                try:
                    result = future.result()
                    if result:
                        valid_tickers.append(result[0])
                        prices_data.append(result[1])
                except Exception as e:
                    print(f"Error processing {symbol}: {str(e)}")

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

        # Parallel sector mapping
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

        # Process each sector
        for sector, tickers in tqdm(self.sector_mappings.items(), desc="Sectors"):
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
        
        # Save to cache
        joblib.dump(self.sector_composites, cache_file)
        return self.sector_composites

    def analyze_sector_regimes(self, n_states=N_STATES):
        print("\nAnalyzing sector regimes...")
        self.sector_analyzers = {}
        
        # Get overall market regime first
        if not hasattr(self, 'market_composite'):
            tickers = [t for sublist in self.sector_mappings.values() for t in sublist]
            self.market_composite = self.overall_analyzer.prepare_market_data(tickers[:100])
        market_result = self.overall_analyzer.analyze_regime(self.market_composite)
        self.current_regime = market_result["regimes"][-1]
        
        # Process each sector
        for sector, composite in tqdm(self.sector_composites.items(), desc="Analyzing Sectors"):
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