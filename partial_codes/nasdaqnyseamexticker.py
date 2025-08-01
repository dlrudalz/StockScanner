import requests
import pandas as pd
from websocket import create_connection, WebSocketConnectionClosedException
import json
import time
import concurrent.futures
from threading import Thread, Lock
from queue import Queue, Empty
import os
import logging
from config import POLYGON_API_KEY

class PolygonTickerScanner:
    def __init__(self, api_key=POLYGON_API_KEY, exchanges=["XNAS", "XNYS", "XASE"], 
                 cache_file="ticker_cache.parquet", log_level=logging.INFO,
                 max_workers=15, refresh_interval=3600):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v3/reference"
        self.websocket_url = "wss://socket.polygon.io/stocks"
        self.cache_file = cache_file
        self.exchanges = exchanges
        self.event_queue = Queue(maxsize=10000)
        self.active = False
        self.ws_reconnect_delay = 5
        self.max_reconnect_attempts = 10
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
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while next_urls:
                futures = []
                current_urls = next_urls.copy()
                next_urls = []
                
                # Submit current batch of URLs
                for url in current_urls:
                    futures.append(executor.submit(self._fetch_exchange_page, exchange, url))
                
                # Process results
                for future in concurrent.futures.as_completed(futures):
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
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.exchanges)) as exchange_executor:
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


if __name__ == "__main__":
    scanner = PolygonTickerScanner(
        max_workers=15,  # Higher since no rate limits
        refresh_interval=3600,  # Refresh every hour
        log_level=logging.INFO
    )
    
    try:
        scanner.start()
        scanner.logger.info("Scanner running. Press Ctrl+C to stop.")
        
        # Run until keyboard interrupt
        while scanner.active:
            time.sleep(1)
        
    except KeyboardInterrupt:
        scanner.logger.info("\nReceived keyboard interrupt. Shutting down...")
    finally:
        scanner.stop()