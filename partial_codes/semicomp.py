import numpy as np
import pandas as pd
import requests
import math
import time
import os
import logging
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread, Lock
from queue import Queue, Empty
from tqdm import tqdm
from websocket import create_connection, WebSocketConnectionClosedException

# ======================== CONFIGURATION SETTINGS ======================== #
POLYGON_API_KEY = "ld1Poa63U6t4Y2MwOCA2JeKQyHVrmyg8"

# Scanner parameters
EXCHANGES = ["XNAS", "XNYS", "XASE"]
TICKER_CACHE_FILE = "ticker_cache.parquet"
MISSING_TICKERS_FILE = "missing_tickers.json"
METADATA_FILE = "scanner_metadata.json"
SCANNER_LOG_LEVEL = logging.INFO
SCANNER_MAX_WORKERS = 20
SCANNER_REFRESH_INTERVAL = 3600
WS_RECONNECT_DELAY = 5
MAX_RECONNECT_ATTEMPTS = 10
MIN_CACHE_REFRESH_INTERVAL = 300

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=SCANNER_LOG_LEVEL
)
logger = logging.getLogger("PolygonTickerScanner")

# ======================== TICKER SCANNER ======================== #
class PolygonTickerScanner:
    def __init__(self):
        self.api_key = POLYGON_API_KEY
        self.base_url = "https://api.polygon.io/v3/reference"
        self.websocket_url = "wss://socket.polygon.io/stocks"
        self.cache_file = TICKER_CACHE_FILE
        self.missing_file = MISSING_TICKERS_FILE
        self.metadata_file = METADATA_FILE
        self.exchanges = EXCHANGES
        self.event_queue = Queue(maxsize=10000)
        self.active = False
        self.ws_reconnect_delay = WS_RECONNECT_DELAY
        self.max_reconnect_attempts = MAX_RECONNECT_ATTEMPTS
        self.current_reconnect_attempts = 0
        self.cache_lock = Lock()
        self.known_missing_tickers = set()
        self.max_workers = SCANNER_MAX_WORKERS
        self.refresh_interval = SCANNER_REFRESH_INTERVAL
        self.last_refresh_time = 0
        self._init_cache()
        
    def _init_cache(self):
        metadata = self._load_metadata()
        self.last_refresh_time = metadata.get('last_refresh_time', 0)
        
        cache_exists = os.path.exists(self.cache_file)
        cache_valid = False
        
        if cache_exists:
            try:
                self.ticker_cache = pd.read_parquet(self.cache_file)
                if not self.ticker_cache.empty:
                    logger.info(f"Loaded cache with {len(self.ticker_cache)} tickers")
                    cache_valid = True
                else:
                    logger.warning("Cache file is empty - treating as invalid")
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
        
        if not cache_exists or not cache_valid:
            self.ticker_cache = pd.DataFrame(columns=["ticker", "name", "primary_exchange", "last_updated_utc", "type"])
            logger.info("No valid cache found - initializing empty cache")
        
        self.current_tickers_set = set(self.ticker_cache['ticker'].tolist()) if not self.ticker_cache.empty else set()
        
        if os.path.exists(self.missing_file):
            try:
                with open(self.missing_file, 'r') as f:
                    self.known_missing_tickers = set(json.load(f))
            except Exception as e:
                logger.error(f"Error loading missing tickers: {e}")
                self.known_missing_tikers = set()
        
        if not cache_valid or self.ticker_cache.empty:
            logger.info("First run detected - forcing full scan")
            self._refresh_all_tickers()
        elif time.time() - self.last_refresh_time > self.refresh_interval:
            logger.info("Cache is stale, starting background refresh")
            Thread(target=self._refresh_all_tickers, daemon=True).start()
    
    def _load_metadata(self):
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
        return {}
    
    def _save_metadata(self):
        metadata = {'last_refresh_time': self.last_refresh_time}
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def _save_missing_tickers(self):
        try:
            with open(self.missing_file, 'w') as f:
                json.dump(list(self.known_missing_tickers), f)
        except Exception as e:
            logger.error(f"Error saving missing tickers: {e}")

    def _call_polygon_api(self, url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if response.status_code == 429:
                logger.warning("Rate limit exceeded, retrying after delay")
                time.sleep(2)
                return self._call_polygon_api(url)
            logger.error(f"API request failed: {e}")
            return None

    def _fetch_exchange_page(self, url):
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
            logger.error(f"Error fetching page: {e}")
            return [], None

    def _fetch_exchange_tickers(self, exchange):
        logger.info(f"Starting ticker fetch for {exchange}")
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
        
        logger.info(f"Completed {exchange}: {len(all_results)} stocks")
        return all_results

    def _refresh_all_tickers(self):
        if time.time() - self.last_refresh_time < MIN_CACHE_REFRESH_INTERVAL:
            logger.warning("Refresh skipped - too soon after last refresh")
            return
            
        start_time = time.time()
        logger.info("Starting full ticker refresh")
        
        with self.cache_lock:
            old_tickers = set(self.current_tickers_set)
        
        with ThreadPoolExecutor(max_workers=len(self.exchanges)) as executor:
            futures = {executor.submit(self._fetch_exchange_tickers, exch): exch for exch in self.exchanges}
            all_results = []
            for future in tqdm(as_completed(futures), total=len(self.exchanges), desc="Exchanges"):
                all_results.extend(future.result())
        
        if not all_results:
            logger.warning("No stocks fetched during refresh")
            return
            
        new_df = pd.DataFrame(all_results)[["ticker", "name", "primary_exchange", "last_updated_utc", "type"]]
        new_tickers = set(new_df['ticker'].tolist())
        
        added = new_tickers - old_tickers
        removed = old_tickers - new_tickers
        
        with self.cache_lock:
            self.ticker_cache = new_df
            self.ticker_cache.to_parquet(self.cache_file)
            self.current_tickers_set = new_tickers
            
            rediscovered = added & self.known_missing_tickers
            if rediscovered:
                self.known_missing_tickers -= rediscovered
                self._save_missing_tickers()
        
        self.last_refresh_time = time.time()
        self._save_metadata()
        
        elapsed = time.time() - start_time
        logger.info(f"Refresh completed in {elapsed:.2f}s")
        logger.info(f"Total stocks: {len(new_df)} | Added: {len(added)} | Removed: {len(removed)}")
        if added:
            logger.info(f"New tickers: {', '.join(list(added)[:5])}{'...' if len(added)>5 else ''}")
        if removed:
            logger.info(f"Removed tickers: {', '.join(list(removed)[:5])}{'...' if len(removed)>5 else ''}")

    def start(self):
        if not self.active:
            self.active = True
            Thread(target=self._websocket_listener, daemon=True).start()
            Thread(target=self._background_refresher, daemon=True).start()
            logger.info("Ticker scanner started")
        else:
            logger.warning("Scanner already running")

    def stop(self):
        self.active = False
        logger.info("Scanner stopped")

    def get_current_tickers(self):
        with self.cache_lock:
            return self.ticker_cache.copy()

    def get_current_tickers_list(self, max_tickers=None):
        with self.cache_lock:
            tickers = self.ticker_cache['ticker'].tolist()
            return tickers[:max_tickers] if max_tickers else tickers

# ======================== BREAKOUT STRATEGY ======================== #
class EnhancedBreakoutStrategy:
    """Breakout detection system with dynamic stop calculation"""
    def __init__(self, entry_price, atr, adx, activation_percent=0.05, base_multiplier=1.5):
        self.entry = entry_price
        self.initial_atr = atr
        self.base_adx = adx
        self.activation_percent = activation_percent
        self.base_multiplier = base_multiplier
        self.activated = False
        self.highest_high = entry_price
        self.current_stop = entry_price - (base_multiplier * atr)
        self.growth_potential = 1.0
        self.consecutive_confirmations = 0
        self.last_direction = "up"
        self.previous_close = entry_price
        
    def update(self, current_bar):
        high = current_bar['High']
        low = current_bar['Low']
        close = current_bar['Close']
        
        # Update highest high
        if high > self.highest_high:
            self.highest_high = high
            self.consecutive_confirmations = 0
            
            # Activate if we've moved enough from entry
            if not self.activated and (high > self.entry * (1 + self.activation_percent)):
                self.activated = True
                
        # Check for confirmation
        if close > self.previous_close:
            if self.last_direction == "up":
                self.consecutive_confirmations += 1
            else:
                self.consecutive_confirmations = 1
            self.last_direction = "up"
        else:
            if self.last_direction == "down":
                self.consecutive_confirmations += 1
            else:
                self.consecutive_confirmations = 1
            self.last_direction = "down"
        
        # Calculate dynamic stop
        if self.activated:
            # Dynamic multiplier based on ADX strength
            adx_factor = min(2.0, max(0.5, self.base_adx / 20.0))
            multiplier = self.base_multiplier * adx_factor
            
            # Calculate potential stop level
            new_stop = self.highest_high - (multiplier * self.initial_atr)
            
            # Only move stop up, never down
            if new_stop > self.current_stop:
                self.current_stop = new_stop
                
            # Trailing growth potential calculation
            self.growth_potential = min(3.0, (close - self.entry) / (self.entry - self.current_stop))
        else:
            # Static stop before activation
            self.current_stop = self.entry - (self.base_multiplier * self.initial_atr)
        
        self.previous_close = close
        return self.current_stop

class PolygonBreakoutScanner:
    def __init__(self, ticker_scanner, max_tickers=500):
        self.api_key = POLYGON_API_KEY
        self.base_url = "https://api.polygon.io/v2"
        self.ticker_scanner = ticker_scanner
        self.max_tickers = max_tickers
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        self.spy_data = self.get_polygon_data("SPY")
        self.spy_3mo_return = self.calculate_3mo_return(self.spy_data) if self.spy_data is not None else 0

    def get_polygon_data(self, ticker):
        """Fetch historical data for a ticker"""
        url = f"{self.base_url}/aggs/ticker/{ticker}/range/1/day/{self.start_date}/{self.end_date}?adjusted=true&sort=asc&limit=500&apiKey={self.api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if data['status'] == 'OK' and data['resultsCount'] > 0:
                df = pd.DataFrame(data['results'])
                df.rename(columns={
                    'v': 'Volume',
                    'o': 'Open',
                    'c': 'Close',
                    'h': 'High',
                    'l': 'Low',
                    't': 'Timestamp',
                    'n': 'TransactionCount'
                }, inplace=True)
                df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
                return df
            return None
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return None

    def calculate_3mo_return(self, df):
        if len(df) < 63:
            return 0
        return ((df['Close'].iloc[-1] / df['Close'].iloc[-63]) - 1) * 100

    def calculate_rsi(self, prices, window=14):
        delta = prices.diff().dropna()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_indicators(self, df):
        if df is None or len(df) < 200:
            return None
            
        try:
            latest = df.iloc[-1].copy()
            high = latest['High']
            low = latest['Low']
            close = latest['Close']
            volume = latest['Volume']
            
            # Moving averages
            sma_50 = df['Close'].rolling(50).mean().iloc[-1]
            sma_200 = df['Close'].rolling(200).mean().iloc[-1]
            distance_sma50 = ((close - sma_50) / sma_50) * 100
            distance_sma200 = ((close - sma_200) / sma_200) * 100
            
            # Volatility metrics
            df['Range'] = df['High'] - df['Low']
            current_range = high - low
            avg_range_10d = df['Range'].rolling(10).mean().iloc[-1]
            volatility_ratio = current_range / avg_range_10d if avg_range_10d > 0 else 1.0
            
            # ATR calculation
            df['prev_close'] = df['Close'].shift(1)
            df['H-L'] = df['High'] - df['Low']
            df['H-PC'] = abs(df['High'] - df['prev_close'])
            df['L-PC'] = abs(df['Low'] - df['prev_close'])
            df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            atr = df['TR'].rolling(14).mean().iloc[-1]
            
            # Simplified ADX calculation
            plus_dm = (df['High'] - df['High'].shift(1)).where(
                (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), 0)
            minus_dm = (df['Low'].shift(1) - df['Low']).where(
                (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), 0)
            
            tr_smooth = df['TR'].ewm(alpha=1/14, min_periods=14).mean()
            plus_di = 100 * (plus_dm.ewm(alpha=1/14, min_periods=14).mean() / tr_smooth)
            minus_di = 100 * (minus_dm.ewm(alpha=1/14, min_periods=14).mean() / tr_smooth)
            dx = 100 * abs((plus_di - minus_di) / (plus_di + minus_di))
            adx = dx.ewm(alpha=1/14, min_periods=14).mean().iloc[-1]
            
            # Momentum indicators
            rsi = self.calculate_rsi(df['Close']).iloc[-1]
            ten_day_change = ((close / df['Close'].iloc[-10]) - 1) * 100 if len(df) >= 10 else 0
                
            # Volume analysis
            avg_volume = df['Volume'].rolling(30).mean().iloc[-1]
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            
            # Breakout detection
            consolidation_high = df['High'].rolling(20).max().iloc[-2]
            breakout_confirmed = close > consolidation_high * 1.03
            
            # Relative strength
            stock_3mo_return = self.calculate_3mo_return(df)
            relative_strength = stock_3mo_return - self.spy_3mo_return
            
            return {
                'Close': float(close),
                'High': float(high),
                'Low': float(low),
                'Volume': float(volume),
                'SMA_50': float(sma_50),
                'SMA_200': float(sma_200),
                'Distance_SMA50': float(distance_sma50),
                'Distance_SMA200': float(distance_sma200),
                'ATR': float(atr),
                'ADX': float(adx),
                'RSI': float(rsi),
                '10D_Change': float(ten_day_change),
                'AvgVolume': float(avg_volume),
                'Volume_Ratio': float(volume_ratio),
                'Consolidation_High': float(consolidation_high),
                'Breakout_Confirmed': breakout_confirmed,
                'Current_Range': float(current_range),
                'Volatility_Ratio': float(volatility_ratio),
                'Relative_Strength': float(relative_strength)
            }
            
        except Exception as e:
            print(f"Indicator calculation error: {str(e)}")
            return None

    def calculate_upside_score(self, indicators):
        breakout_score = 30 if indicators['Breakout_Confirmed'] else 0
        rs_score = min(25, max(0, indicators['Relative_Strength'] * 0.5))
        
        vol_ratio = indicators['Volume_Ratio']
        volume_score = min(20, max(0, (vol_ratio - 1.0) * 10)) if vol_ratio > 1.0 else 0
        
        vol_contraction = indicators['Volatility_Ratio']
        volatility_score = 15 * max(0, 1 - min(1, vol_contraction/0.7))
        
        rsi = indicators['RSI']
        rsi_score = 10 * (1 - abs(rsi - 60)/20) if 40 <= rsi <= 80 else 0
        
        return min(100, breakout_score + rs_score + volume_score + volatility_score + rsi_score)

    def scan_tickers(self):
        results = []
        tickers = self.ticker_scanner.get_current_tickers_list(self.max_tickers)
        
        for i, ticker in enumerate(tickers):
            if i > 0 and i % 5 == 0:
                time.sleep(1)
                
            print(f"Processing {ticker} ({i+1}/{len(tickers)})...", end='\r')
            data = self.get_polygon_data(ticker)
            if data is None:
                continue
                
            indicators = self.calculate_indicators(data)
            if indicators is None:
                continue
                
            try:
                # Core breakout conditions
                above_sma50 = indicators['Close'] > indicators['SMA_50']
                above_sma200 = indicators['Close'] > indicators['SMA_200']
                strong_adx = indicators['ADX'] > 25
                volume_surge = indicators['Volume_Ratio'] > 1.5
                volatility_contraction = indicators['Volatility_Ratio'] < 0.7
                valid_rsi = 40 <= indicators['RSI'] <= 80
                
                if not (above_sma50 and above_sma200 and strong_adx and volume_surge 
                        and volatility_contraction and valid_rsi):
                    continue
                
                upside_score = self.calculate_upside_score(indicators)
                
                breakout_system = EnhancedBreakoutStrategy(
                    entry_price=indicators['Close'],
                    atr=indicators['ATR'],
                    adx=indicators['ADX']
                )
                
                risk_per_share = indicators['Close'] - breakout_system.current_stop
                risk_percent = (risk_per_share / indicators['Close']) * 100
                
                profit_target = indicators['Close'] + (3 * indicators['ATR'])
                reward_risk_ratio = (profit_target - indicators['Close']) / risk_per_share
                
                results.append({
                    'Ticker': ticker,
                    'Upside_Score': round(upside_score, 1),
                    'Price': round(indicators['Close'], 2),
                    'Breakout_Level': round(indicators['Consolidation_High'], 2),
                    'Breakout_Confirmed': indicators['Breakout_Confirmed'],
                    'ADX': round(indicators['ADX'], 1),
                    'RSI': round(indicators['RSI'], 1),
                    'Relative_Strength': round(indicators['Relative_Strength'], 1),
                    'Vol_Ratio': round(indicators['Volatility_Ratio'], 2),
                    'Volume_Ratio': round(indicators['Volume_Ratio'], 1),
                    'SMA50_Dist%': round(indicators['Distance_SMA50'], 1),
                    'ATR': round(indicators['ATR'], 2),
                    '10D_Change%': round(indicators['10D_Change'], 1),
                    'Profit_Target': round(profit_target, 2),
                    'Reward_Risk_Ratio': round(reward_risk_ratio, 1),
                    'Upside_Potential%': round((profit_target/indicators['Close'] - 1)*100, 1)
                })
                    
            except Exception as e:
                print(f"Error evaluating {ticker}: {str(e)}")
                continue
        
        if results:
            df_results = pd.DataFrame(results)
            df_results['Rank'] = df_results['Upside_Score'].rank(ascending=False, method='min').astype(int)
            return df_results.sort_values('Upside_Score', ascending=False)
        return pd.DataFrame()
    
    def save_results(self, df, filename='breakout_signals.csv'):
        df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")

# ======================== MAIN EXECUTION ======================== #
def main():
    # Initialize and start ticker scanner
    ticker_scanner = PolygonTickerScanner()
    ticker_scanner.start()
    
    # Initialize breakout scanner with ticker scanner reference
    breakout_scanner = PolygonBreakoutScanner(ticker_scanner, max_tickers=500)
    
    try:
        print("Ticker scanner running. Press Ctrl+C to stop.")
        last_scan_time = 0
        scan_interval = 3600  # 1 hour
        
        while True:
            current_time = time.time()
            
            # Run scan once per interval
            if current_time - last_scan_time > scan_interval:
                print("\nStarting breakout scan...")
                results = breakout_scanner.scan_tickers()
                
                if not results.empty:
                    print("\nTop Breakout Candidates:")
                    print(results.head(20))
                    date_str = datetime.now().strftime("%Y%m%d_%H%M")
                    breakout_scanner.save_results(results, filename=f"breakout_signals_{date_str}.csv")
                else:
                    print("\nNo breakout opportunities found.")
                
                last_scan_time = current_time
                print(f"Next scan in {scan_interval//60} minutes")
            
            # Periodic status update
            time.sleep(60)
            tickers = ticker_scanner.get_current_tickers()
            print(f"\nCurrent tickers: {len(tickers)} | Last refresh: {time.ctime(ticker_scanner.last_refresh_time)}", end='\r')
            
    except KeyboardInterrupt:
        print("\nStopping scanners...")
    finally:
        ticker_scanner.stop()

if __name__ == "__main__":
    main()