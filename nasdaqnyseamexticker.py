import requests
import pandas as pd
from websocket import create_connection, WebSocketConnectionClosedException
import json
import time
from threading import Thread
from queue import Queue
import os
from ratelimit import limits, sleep_and_retry
from config import POLYGON_API_KEY  # Import from config.py

class PolygonTickerScanner:
    def __init__(self, api_key=POLYGON_API_KEY):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v3/reference"
        self.websocket_url = "wss://socket.polygon.io/stocks"
        self.cache_file = "ticker_cache.parquet"
        self.exchanges = ["XNAS", "XNYS", "XASE"]  # NASDAQ, NYSE, AMEX
        self.event_queue = Queue(maxsize=10000)
        self.active = False
        self.ws_reconnect_delay = 5  # seconds between reconnection attempts
        self.max_reconnect_attempts = 5
        self.current_reconnect_attempts = 0
        
        # Initialize cache
        if os.path.exists(self.cache_file):
            self.ticker_cache = pd.read_parquet(self.cache_file)
        else:
            self.ticker_cache = pd.DataFrame(columns=["ticker", "name", "primary_exchange", "last_updated_utc"])
            self._refresh_all_tickers()

    @sleep_and_retry
    @limits(calls=5, period=60)  # Polygon rate limit
    def _call_polygon_api(self, url):
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def _refresh_all_tickers(self):
        """Fetch all tickers from Polygon and cache them"""
        all_results = []
        
        for exchange in self.exchanges:
            url = f"{self.base_url}/tickers?market=stocks&exchange={exchange}&active=true&limit=1000&apiKey={self.api_key}"
            
            while url:
                try:
                    data = self._call_polygon_api(url)
                    all_results.extend(data["results"])
                    url = data.get("next_url", None)
                    if url:
                        url += f"&apiKey={self.api_key}"
                        time.sleep(0.2)  # Respect rate limits
                except Exception as e:
                    print(f"Error fetching tickers: {e}")
                    break
        
        # Process and cache
        if all_results:
            df = pd.DataFrame(all_results)
            self.ticker_cache = df[["ticker", "name", "primary_exchange", "last_updated_utc"]].copy()
            self.ticker_cache.to_parquet(self.cache_file)
            print(f"Cached {len(self.ticker_cache)} tickers")
        else:
            print("Warning: No tickers were fetched")

    def _websocket_listener(self):
        """Handle real-time WebSocket data with improved reconnection logic"""
        while self.active:
            try:
                print(f"Attempting WebSocket connection (attempt {self.current_reconnect_attempts + 1}/{self.max_reconnect_attempts})")
                ws = create_connection(
                    self.websocket_url,
                    timeout=10,  # Connection timeout
                    enable_multithread=True
                )
                print("WebSocket connected")
                self.current_reconnect_attempts = 0  # Reset on successful connection
                
                # Authentication
                auth_msg = json.dumps({
                    "action": "auth",
                    "params": self.api_key
                })
                ws.send(auth_msg)
                
                # Subscription
                sub_msg = json.dumps({
                    "action": "subscribe",
                    "params": "T.*,A.*,AM.*"  # Trades, Aggregates, Minute Aggs
                })
                ws.send(sub_msg)
                
                # Keep connection alive
                last_ping = time.time()
                while self.active:
                    try:
                        # Set a timeout for recv to allow periodic ping checks
                        data = ws.recv()
                        if data:
                            self.event_queue.put(json.loads(data))
                        
                        # Send ping every 30 seconds to keep connection alive
                        if time.time() - last_ping > 30:
                            try:
                                ws.ping()
                                last_ping = time.time()
                            except Exception as e:
                                print(f"Ping failed: {e}")
                                raise  # Will trigger reconnection
                                
                    except (WebSocketConnectionClosedException, ConnectionResetError, TimeoutError) as e:
                        print(f"WebSocket receive error: {e}")
                        break
                    except Exception as e:
                        print(f"Unexpected WebSocket error: {e}")
                        break
                        
            except Exception as e:
                print(f"WebSocket connection error: {e}")
                self.current_reconnect_attempts += 1
                if self.current_reconnect_attempts >= self.max_reconnect_attempts:
                    print("Max reconnection attempts reached. Stopping scanner.")
                    self.active = False
                    break
            finally:
                if 'ws' in locals():
                    try:
                        ws.close()
                    except:
                        pass
                if self.active and self.current_reconnect_attempts < self.max_reconnect_attempts:
                    print(f"Attempting to reconnect in {self.ws_reconnect_delay} seconds...")
                    time.sleep(self.ws_reconnect_delay)

    def _process_events(self):
        """Process real-time events from queue"""
        while self.active:
            try:
                events = []
                while not self.event_queue.empty():
                    events.append(self.event_queue.get())
                
                if events:
                    # Process in bulk for efficiency
                    for event in events:
                        if isinstance(event, list):
                            for e in event:
                                self._handle_single_event(e)
                        else:
                            self._handle_single_event(event)
                
                time.sleep(0.1)  # Prevent CPU overuse
                
            except Exception as e:
                print(f"Processing error: {e}")

    def _handle_single_event(self, event):
        """Handle individual WebSocket events"""
        if event.get("ev") == "T":  # Trade event
            ticker = event["sym"]
            price = event["p"]
            print(f"Trade: {ticker} @ {price}")
            # Update cache if needed
            if ticker not in self.ticker_cache["ticker"].values:
                print(f"New ticker detected: {ticker}. Refreshing cache...")
                self._refresh_all_tickers()

    def start(self):
        """Start the real-time scanner"""
        if not self.active:
            self.active = True
            # Start WebSocket listener thread
            Thread(target=self._websocket_listener, daemon=True).start()
            # Start event processor thread
            Thread(target=self._process_events, daemon=True).start()
            print("Real-time scanner started")
        else:
            print("Scanner is already running")

    def stop(self):
        """Stop the scanner"""
        self.active = False
        print("Scanner stopped")

    def get_current_tickers(self):
        """Get current ticker list"""
        return self.ticker_cache.copy()

    def check_for_new_listings(self):
        """Check for new listings by comparing with cache"""
        old_tickers = set(self.ticker_cache["ticker"])
        self._refresh_all_tickers()
        new_tickers = set(self.ticker_cache["ticker"])
        
        added = new_tickers - old_tickers
        removed = old_tickers - new_tickers
        
        if added:
            print(f"New tickers detected: {added}")
        if removed:
            print(f"Delisted tickers: {removed}")
        
        return added, removed


# Example Usage
if __name__ == "__main__":
    scanner = PolygonTickerScanner()
    
    try:
        scanner.start()
        
        # Run for 5 minutes (or until keyboard interrupt)
        while scanner.active:
            time.sleep(1)
        
        # Check for new listings periodically
        scanner.check_for_new_listings()
        
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt. Shutting down...")
    finally:
        scanner.stop()