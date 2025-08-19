import numpy as np
import pandas as pd
import asyncio
import aiohttp
import time
import os
import logging
import json
from datetime import datetime, timedelta
from threading import Lock, Event, Thread
from tqdm import tqdm
import signal
from urllib.parse import urlencode
import sys
import queue

# PyQt5 imports for UI
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QTableView, QTextEdit, QLabel, QHeaderView, QProgressBar,
                            QAction, QMenu, QFileDialog, QStatusBar, QMessageBox, QDateEdit)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QAbstractTableModel, QTimer
from PyQt5.QtGui import QFont, QColor, QIcon

# ======================== CONSOLIDATED CONFIGURATION ======================== #
class Config:
    # API Configuration
    POLYGON_API_KEY = "ld1Poa63U6t4Y2MwOCA2JeKQyHVrmyg8"
    
    # Scanner Behavior
    EXCHANGES = ["XNAS", "XNYS", "XASE"]
    MAX_TICKERS = 5000
    MAX_CONCURRENT_REQUESTS = 50
    RATE_LIMIT_DELAY = 0.05  # 50ms between requests (premium can handle this)
    SCAN_INTERVAL = 3600  # 1 hour between full scans
    MIN_CACHE_REFRESH_INTERVAL = 300  # 5 minutes
    
    # File Management
    TICKER_CACHE_FILE = "ticker_cache.parquet"
    MISSING_TICKERS_FILE = "missing_tickers.json"
    METADATA_FILE = "scanner_metadata.json"
    RESULTS_FILE_PREFIX = "breakout_signals_"
    FAILURE_LOG_PREFIX = "breakout_failures_"
    
    # Technical Parameters
    ENTRY_ACTIVATION_PERCENT = 0.05
    BASE_MULTIPLIER = 1.5
    RSI_WINDOW = 14
    ADX_THRESHOLD = 25
    VOLUME_SURGE_RATIO = 1.5
    VOLATILITY_CONTRACTION_RATIO = 0.7
    RSI_MIN = 40
    RSI_MAX = 80
    CONSOLIDATION_BREAKOUT_FACTOR = 1.03
    REWARD_RISK_MULTIPLIER = 3.0
    
    # Position Sizing
    MAX_RISK_PERCENT = 0.02  # 2% of capital per trade
    
    # Stoploss Parameters
    VOLATILITY_SCALING_FACTOR = 0.8  # Min volatility multiplier
    TIME_DECAY_FACTOR = 0.02  # 2% tightening per day after 5 days
    MIN_STOP_MULTIPLIER = 0.7  # Minimum stop multiplier
    MAX_STOP_MULTIPLIER = 2.0  # Maximum stop multiplier
    
    # Logging Configuration
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Initialize configuration
config = Config()

# ======================== LOGGING SETUP ======================== #
def setup_logging():
    """Configure logging with file and console handlers"""
    # Create logs directory if not exists
    os.makedirs("logs", exist_ok=True)
    
    # Create main logger
    logger = logging.getLogger("PolygonStockScanner")
    logger.setLevel(config.LOG_LEVEL)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(config.LOG_FORMAT)
    
    # File handler - all logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/scanner_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(config.LOG_LEVEL)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler - info and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(config.LOG_LEVEL)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# ======================== GRACEFUL SHUTDOWN HANDLER ======================== #
class GracefulShutdown:
    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.original_sigint_handler = signal.getsignal(signal.SIGINT)
        
    def init(self):
        signal.signal(signal.SIGINT, self._shutdown_signal_handler)
        
    def _shutdown_signal_handler(self, signum, frame):
        logger.info("Shutdown signal received, initiating graceful shutdown...")
        self.shutdown_event.set()
        
    async def wait_for_shutdown(self):
        await self.shutdown_event.wait()

# ======================== TICKER SCANNER ======================== #
class PolygonTickerScanner:
    def __init__(self):
        self.api_key = config.POLYGON_API_KEY
        self.base_url = "https://api.polygon.io/v3/reference/tickers"
        self.exchanges = config.EXCHANGES
        self.active = False
        self.cache_lock = Lock()
        self.refresh_lock = Lock()
        self.known_missing_tickers = set()
        self.initial_refresh_complete = Event()
        self.last_refresh_time = 0
        self.ticker_cache = pd.DataFrame(columns=[
            "ticker", "name", "primary_exchange", "last_updated_utc", "type"
        ])
        self.current_tickers_set = set()
        self.historical_tickers = {}  # Cache for historical tickers
        self.last_historical_fetch_time = 0  # For rate limiting
        
    def _init_cache(self):
        """Initialize or load ticker cache from disk"""
        metadata = self._load_metadata()
        self.last_refresh_time = metadata.get('last_refresh_time', 0)
        
        if os.path.exists(config.TICKER_CACHE_FILE):
            try:
                self.ticker_cache = pd.read_parquet(config.TICKER_CACHE_FILE)
                logger.info(f"Loaded ticker cache with {len(self.ticker_cache)} symbols")
            except Exception as e:
                logger.error(f"Cache load error: {e}")
                self.ticker_cache = pd.DataFrame(columns=[
                    "ticker", "name", "primary_exchange", "last_updated_utc", "type"
                ])
        
        self.current_tickers_set = set(self.ticker_cache['ticker'].tolist()) if not self.ticker_cache.empty else set()
        
        if os.path.exists(config.MISSING_TICKERS_FILE):
            try:
                with open(config.MISSING_TICKERS_FILE, 'r') as f:
                    self.known_missing_tickers = set(json.load(f))
            except Exception as e:
                logger.error(f"Missing tickers load error: {e}")
        
        # Always start with refresh complete cleared
        self.initial_refresh_complete.clear()

    def _load_metadata(self):
        if os.path.exists(config.METADATA_FILE):
            try:
                with open(config.METADATA_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Metadata load error: {e}")
        return {}

    def _save_metadata(self):
        metadata = {'last_refresh_time': self.last_refresh_time}
        try:
            with open(config.METADATA_FILE, 'w') as f:
                json.dump(metadata, f)
        except Exception as e:
            logger.error(f"Metadata save error: {e}")

    def _save_missing_tickers(self):
        try:
            with open(config.MISSING_TICKERS_FILE, 'w') as f:
                json.dump(list(self.known_missing_tickers), f)
        except Exception as e:
            logger.error(f"Missing tickers save error: {e}")

    async def _call_polygon_api(self, session, url):
        try:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', 1))
                    logger.warning(f"Rate limit hit, retrying after {retry_after} seconds")
                    await asyncio.sleep(retry_after)
                    return await self._call_polygon_api(session, url)
                else:
                    logger.error(f"API request failed: {response.status}")
                    return None
        except asyncio.CancelledError:
            logger.debug("API request cancelled")
            raise
        except Exception as e:
            logger.error(f"API request exception: {e}")
            return None

    async def _fetch_exchange_tickers(self, session, exchange, date=None):
        """Fetch tickers for an exchange, optionally for a specific date"""
        logger.info(f"Fetching tickers for {exchange} on {date if date else 'current date'}")
        all_results = []
        next_url = None
        
        params = {
            "market": "stocks",
            "exchange": exchange,
            "active": "true",
            "limit": 1000,
            "apiKey": self.api_key
        }
        
        # Add date parameter if provided
        if date:
            params["date"] = date.strftime('%Y-%m-%d')
        
        # Initial URL construction
        url = f"{self.base_url}?{urlencode(params)}"
        
        while True:
            # For subsequent pages, use next_url with API key
            if next_url:
                url = f"{next_url}&apiKey={self.api_key}"
            
            data = await self._call_polygon_api(session, url)
            if not data:
                break
                
            results = data.get("results", [])
            # Filter for common stocks only
            stock_results = [r for r in results if r.get('type', '').upper() == 'CS']
            all_results.extend(stock_results)
            
            next_url = data.get("next_url", None)
            if not next_url:
                break
                
            # Minimal delay for premium API access
            await asyncio.sleep(0.05)
        
        logger.info(f"Completed {exchange}: {len(all_results)} stocks")
        return all_results

    async def _refresh_all_tickers_async(self):
        if time.time() - self.last_refresh_time < config.MIN_CACHE_REFRESH_INTERVAL:
            return
            
        start_time = time.time()
        logger.info("Starting full ticker refresh")
        
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_exchange_tickers(session, exch) for exch in self.exchanges]
            all_results = []
            for future in asyncio.as_completed(tasks):
                try:
                    results = await future
                    all_results.extend(results)
                except asyncio.CancelledError:
                    logger.info("Ticker refresh cancelled")
                    return
        
        if not all_results:
            logger.warning("Refresh fetched no results")
            return
            
        new_df = pd.DataFrame(all_results)[["ticker", "name", "primary_exchange", "last_updated_utc", "type"]]
        new_tickers = set(new_df['ticker'].tolist())
        
        with self.cache_lock:
            old_tickers = set(self.current_tickers_set)
            added = new_tickers - old_tickers
            removed = old_tickers - new_tickers
            
            self.ticker_cache = new_df
            self.ticker_cache.to_parquet(config.TICKER_CACHE_FILE)
            self.current_tickers_set = new_tickers
            
            rediscovered = added & self.known_missing_tickers
            if rediscovered:
                self.known_missing_tickers -= rediscovered
                self._save_missing_tickers()
        
        self.last_refresh_time = time.time()
        self._save_metadata()
        
        elapsed = time.time() - start_time
        logger.info(f"Ticker refresh completed in {elapsed:.2f}s")
        logger.info(f"Total: {len(new_df)} | Added: {len(added)} | Removed: {len(removed)}")

    def _refresh_all_tickers(self):
        if time.time() - self.last_refresh_time < config.MIN_CACHE_REFRESH_INTERVAL:
            return
            
        with self.refresh_lock:
            if time.time() - self.last_refresh_time < config.MIN_CACHE_REFRESH_INTERVAL:
                return
                
            asyncio.run(self._refresh_all_tickers_async())

    def start(self):
        if not self.active:
            self.active = True
            # Initialize cache only when started
            self._init_cache()
            
            # Start initial refresh if needed
            if self.ticker_cache.empty or time.time() - self.last_refresh_time > config.MIN_CACHE_REFRESH_INTERVAL:
                logger.info("Starting initial ticker refresh")
                Thread(target=self._initial_refresh, daemon=True).start()
            else:
                logger.info("Using existing ticker cache")
                self.initial_refresh_complete.set()
                # Start background refresher
                Thread(target=self._background_refresher, daemon=True).start()
                
            logger.info("Ticker scanner started")

    def _initial_refresh(self):
        try:
            self._refresh_all_tickers()
            self.initial_refresh_complete.set()
            # Start background refresher after initial refresh
            Thread(target=self._background_refresher, daemon=True).start()
        finally:
            self.initial_refresh_complete.set()

    def _background_refresher(self):
        logger.info("Background refresher started")
        while self.active:
            current_time = time.time()
            if current_time - self.last_refresh_time > config.SCAN_INTERVAL:
                try:
                    self._refresh_all_tickers()
                except Exception as e:
                    logger.error(f"Background refresh failed: {e}")
            time.sleep(60)

    def stop(self):
        self.active = False
        logger.info("Ticker scanner stopped")

    def get_current_tickers_list(self):
        with self.cache_lock:
            return self.ticker_cache['ticker'].tolist()
            
    async def fetch_all_stocks_for_range(self, start_date, end_date):
        """Fetch all active stocks for a date range"""
        logger.info(f"Fetching all stocks for date range: {start_date} to {end_date}")
        all_tickers = set()
        current_date = start_date
        date_cache_key = f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
        
        # Check if we have this range cached
        if date_cache_key in self.historical_tickers:
            return self.historical_tickers[date_cache_key]
            
        # Rate limit protection - 5 requests per minute
        now = time.time()
        if now - self.last_historical_fetch_time < 12:  # 12 seconds = 5 requests per minute
            wait_time = 12 - (now - self.last_historical_fetch_time)
            logger.info(f"Rate limit protection: Waiting {wait_time:.1f} seconds")
            await asyncio.sleep(wait_time)
        
        self.last_historical_fetch_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            while current_date <= end_date:
                formatted_date = current_date.strftime('%Y-%m-%d')
                
                # Skip weekends
                if current_date.weekday() >= 5:  # Saturday=5, Sunday=6
                    current_date += timedelta(days=1)
                    continue
                
                # Fetch tickers for this date
                tasks = [self._fetch_exchange_tickers(session, exch, current_date) for exch in self.exchanges]
                for future in asyncio.as_completed(tasks):
                    try:
                        results = await future
                        all_tickers.update([ticker['ticker'] for ticker in results])
                    except Exception as e:
                        logger.error(f"Error fetching tickers for {current_date}: {str(e)}")
                
                # Move to next day
                current_date += timedelta(days=1)
        
        ticker_list = list(all_tickers)
        self.historical_tickers[date_cache_key] = ticker_list
        logger.info(f"Found {len(ticker_list)} unique stocks in date range")
        return ticker_list

# ======================== STOPLOSS MANAGER ======================== #
class StoplossManager:
    def __init__(self, entry_price, initial_atr, base_adx):
        self.entry = entry_price
        self.initial_atr = initial_atr
        self.current_atr = initial_atr
        self.base_adx = base_adx
        self.highest_high = entry_price
        self.current_stop = entry_price - (config.BASE_MULTIPLIER * initial_atr)
        self.previous_close = entry_price
        self.days_since_entry = 0
        
    def update(self, current_bar):
        """Update stoploss based on new market data"""
        high = current_bar['High']
        low = current_bar['Low']
        close = current_bar['Close']
        
        # Update days counter
        self.days_since_entry += 1
        
        # Update ATR for volatility scaling
        current_range = high - low
        self.current_atr = 0.7 * self.current_atr + 0.3 * current_range
        
        # Track highest high
        if high > self.highest_high:
            self.highest_high = high
            
        # Calculate stoploss factors
        adx_factor = self._calculate_adx_factor()
        volatility_factor = self._calculate_volatility_factor()
        time_factor = self._calculate_time_factor()
        
        # Combine all factors
        multiplier = config.BASE_MULTIPLIER * adx_factor * volatility_factor * time_factor
        new_stop = self.highest_high - (multiplier * self.initial_atr)
        
        # Only move stop up, never down
        if new_stop > self.current_stop:
            self.current_stop = new_stop
        
        self.previous_close = close
        return self.current_stop
        
    def _calculate_adx_factor(self):
        """Calculate ADX-based stoploss adjustment"""
        return min(config.MAX_STOP_MULTIPLIER, 
                 max(config.MIN_STOP_MULTIPLIER, self.base_adx / 20.0))
                
    def _calculate_volatility_factor(self):
        """Calculate volatility-based stoploss adjustment"""
        volatility_ratio = self.current_atr / self.initial_atr
        return max(config.VOLATILITY_SCALING_FACTOR, 
                 min(1.2, volatility_ratio))
                 
    def _calculate_time_factor(self):
        """Calculate time-based stoploss adjustment"""
        if self.days_since_entry <= 5:
            return 1.0
            
        # Tighten stop after 5 days
        return max(config.MIN_STOP_MULTIPLIER, 
                 1 - (self.days_since_entry - 5) * config.TIME_DECAY_FACTOR)

# ======================== BREAKOUT STRATEGY ======================== #
class EnhancedBreakoutStrategy:
    def __init__(self, entry_price, atr, adx, entry_volume, avg_volume):
        self.entry = entry_price
        self.activated = False
        self.entry_volume = entry_volume
        self.avg_volume = avg_volume
        
        # Create stoploss manager
        self.stoploss_manager = StoplossManager(entry_price, atr, adx)
        self.current_stop = self.stoploss_manager.current_stop
        
    def update(self, current_bar):
        """Update strategy state based on new market data"""
        high = current_bar['High']
        volume = current_bar['Volume']
        
        # Activation check with volume confirmation
        if not self.activated:
            price_activation = high > self.entry * (1 + config.ENTRY_ACTIVATION_PERCENT)
            volume_confirmation = volume > self.avg_volume * config.VOLUME_SURGE_RATIO
            
            if price_activation and volume_confirmation:
                self.activated = True
                logger.debug(f"Stop activated: Price={high:.2f}, Volume={volume:.0f} (Avg={self.avg_volume:.0f})")
                
        # Update stoploss if activated
        if self.activated:
            self.current_stop = self.stoploss_manager.update(current_bar)
        else:
            # Update manager for state consistency but don't change current stop
            self.stoploss_manager.update(current_bar)
        
        return self.current_stop

# ======================== BREAKOUT SCANNER ======================== #
class PolygonBreakoutScanner:
    def __init__(self, ticker_scanner, end_date=None, start_date=None):
        self.ticker_scanner = ticker_scanner
        self.session = None
        self.spy_data = None
        self.spy_3mo_return = 0
        # Use provided dates or current dates
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.start_date = start_date or (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        self.shutting_down = False
        self.failure_logger = None
        self.spy_sma_50 = None
        self.spy_sma_200 = None

    async def async_init(self):
        self.session = aiohttp.ClientSession()
        self.spy_data = await self.async_get_polygon_data("SPY")
        if self.spy_data is not None:
            self.spy_3mo_return = self.calculate_3mo_return(self.spy_data)
            
            # Calculate SPY moving averages for market regime filter
            if len(self.spy_data) >= 200:
                self.spy_sma_50 = self.spy_data['Close'].rolling(50).mean().iloc[-1]
                self.spy_sma_200 = self.spy_data['Close'].rolling(200).mean().iloc[-1]

    async def async_close(self):
        if self.session:
            self.shutting_down = True
            await self.session.close()
            logger.debug("HTTP session closed")
        
        if self.failure_logger:
            for handler in self.failure_logger.handlers[:]:
                handler.close()
                self.failure_logger.removeHandler(handler)

    async def async_get_polygon_data(self, ticker):
        """Enhanced data fetching with better error handling"""
        if self.shutting_down:
            return None
            
        if not self.session or self.session.closed:
            return None
            
        # Build parameters with proper formatting
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,  # Increased limit to ensure we get all data
            "apiKey": config.POLYGON_API_KEY
        }
        
        # Construct URL with encoded parameters
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{self.start_date}/{self.end_date}?{urlencode(params)}"
        
        try:
            async with self.session.get(url, timeout=15) as response:  # Increased timeout
                # Handle rate limits first
                if response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', 1))
                    logger.warning(f"Rate limit hit for {ticker}, retrying after {retry_after} seconds")
                    await asyncio.sleep(retry_after)
                    return await self.async_get_polygon_data(ticker)
                
                # Handle other errors
                if response.status != 200:
                    error_msg = await response.text()
                    logger.warning(f"API request for {ticker} failed: {response.status} - {error_msg}")
                    return pd.DataFrame()  # Return empty DataFrame
                
                data = await response.json()
                
                # Handle empty results more gracefully
                if data.get('resultsCount', 0) == 0:
                    logger.debug(f"No results found for {ticker}")
                    return pd.DataFrame()  # Return empty DataFrame
                    
                df = pd.DataFrame(data['results'])
                
                # Validate we have required columns
                required_columns = {'v', 'o', 'c', 'h', 'l', 't'}
                if not required_columns.issubset(df.columns):
                    logger.warning(f"Missing columns in response for {ticker}")
                    return pd.DataFrame()
                    
                df.rename(columns={
                    'v': 'Volume', 'o': 'Open', 'c': 'Close',
                    'h': 'High', 'l': 'Low', 't': 'Timestamp'
                }, inplace=True)
                
                # Handle missing/invalid timestamps
                try:
                    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
                except Exception as e:
                    logger.warning(f"Invalid timestamp for {ticker}: {e}")
                    # Generate date range as fallback
                    df['Date'] = pd.date_range(end=datetime.now(), periods=len(df), freq='D')
                    
                return df
                
        except asyncio.TimeoutError:
            logger.warning(f"Request for {ticker} timed out")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return pd.DataFrame()

    def calculate_3mo_return(self, df):
        if df is None or len(df) < 63: 
            return 0
        return ((df['Close'].iloc[-1] / df['Close'].iloc[-63]) - 1) * 100

    def calculate_rsi(self, prices):
        if prices is None or len(prices) < config.RSI_WINDOW + 1:
            return pd.Series([np.nan] * len(prices))
            
        delta = prices.diff().dropna()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/config.RSI_WINDOW, min_periods=config.RSI_WINDOW).mean()
        avg_loss = loss.ewm(alpha=1/config.RSI_WINDOW, min_periods=config.RSI_WINDOW).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_indicators(self, df):
        if df is None or len(df) < 200: 
            return None
            
        try:
            latest = df.iloc[-1].copy()
            close = latest['Close']
            high = latest['High']
            low = latest['Low']
            volume = latest['Volume']
            
            sma_50 = df['Close'].rolling(50).mean().iloc[-1]
            sma_200 = df['Close'].rolling(200).mean().iloc[-1]
            
            df['Range'] = df['High'] - df['Low']
            current_range = high - low
            avg_range_10d = df['Range'].rolling(10).mean().iloc[-1]
            volatility_ratio = current_range / avg_range_10d if avg_range_10d > 0 else 1.0
            
            df['prev_close'] = df['Close'].shift(1)
            df['H-L'] = df['High'] - df['Low']
            df['H-PC'] = abs(df['High'] - df['prev_close'])
            df['L-PC'] = abs(df['Low'] - df['prev_close'])
            df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            atr = df['TR'].rolling(14).mean().iloc[-1]
            
            plus_dm = (df['High'] - df['High'].shift(1)).clip(lower=0)
            minus_dm = (df['Low'].shift(1) - df['Low']).clip(lower=0)
            tr_smooth = df['TR'].ewm(alpha=1/14, min_periods=14).mean()
            plus_di = 100 * (plus_dm.ewm(alpha=1/14, min_periods=14).mean() / tr_smooth)
            minus_di = 100 * (minus_dm.ewm(alpha=1/14, min_periods=14).mean() / tr_smooth)
            dx = 100 * abs((plus_di - minus_di) / (plus_di + minus_di))
            adx = dx.ewm(alpha=1/14, min_periods=14).mean().iloc[-1]
            
            rsi = self.calculate_rsi(df['Close']).iloc[-1]
            ten_day_change = ((close / df['Close'].iloc[-10]) - 1) * 100 if len(df) >= 10 else 0
                
            avg_volume = df['Volume'].rolling(30).mean().iloc[-1]
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            
            consolidation_high = df['High'].rolling(20).max().iloc[-2] if len(df) >= 20 else 0
            breakout_confirmed = close > consolidation_high * config.CONSOLIDATION_BREAKOUT_FACTOR if consolidation_high > 0 else False
            
            stock_3mo_return = self.calculate_3mo_return(df)
            relative_strength = stock_3mo_return - self.spy_3mo_return
            
            return {
                'Close': close, 'High': high, 'Low': low, 'Volume': volume,
                'SMA_50': sma_50, 'SMA_200': sma_200,
                'ATR': atr, 'ADX': adx, 'RSI': rsi,
                '10D_Change': ten_day_change, 'AvgVolume': avg_volume,
                'Volume_Ratio': volume_ratio, 'Consolidation_High': consolidation_high,
                'Breakout_Confirmed': breakout_confirmed,
                'Volatility_Ratio': volatility_ratio,
                'Relative_Strength': relative_strength
            }
        except Exception as e:
            logger.error(f"Indicator calculation error: {str(e)}")
            return None

    def calculate_upside_score(self, indicators):
        if indicators is None:
            return 0
            
        breakout_score = 30 if indicators.get('Breakout_Confirmed', False) else 0
        rs_score = min(25, max(0, indicators.get('Relative_Strength', 0) * 0.5))
        vol_ratio = indicators.get('Volume_Ratio', 1.0)
        volume_score = min(20, max(0, (vol_ratio - 1.0) * 10)) if vol_ratio > 1.0 else 0
        vol_contraction = indicators.get('Volatility_Ratio', 1.0)
        volatility_score = 15 * max(0, 1 - min(1, vol_contraction/0.7))
        rsi = indicators.get('RSI', 50)
        rsi_score = 10 * (1 - abs(rsi - 60)/20) if config.RSI_MIN <= rsi <= config.RSI_MAX else 0
        return min(100, breakout_score + rs_score + volume_score + volatility_score + rsi_score)

    async def process_ticker(self, ticker):
        if self.shutting_down:
            return None, ["Shutdown in progress"]
        
        try:
            data = await self.async_get_polygon_data(ticker)
            
            # Skip if we have insufficient data
            if data is None or data.empty or len(data) < 200:
                return None, ["Insufficient historical data"]
                
            # More robust indicator calculation
            try:
                indicators = self.calculate_indicators(data)
                if indicators is None: 
                    return None, ["Indicator calculation failed"]
            except Exception as e:
                logger.error(f"Indicator error for {ticker}: {str(e)}")
                return None, [f"Indicator error: {str(e)}"]
            
            failure_reasons = []
            
            # Check each condition individually and collect failure reasons
            conditions = [
                ("Above SMA50", indicators['Close'] > indicators['SMA_50']),
                ("Above SMA200", indicators['Close'] > indicators['SMA_200']),
                (f"ADX > {config.ADX_THRESHOLD}", indicators['ADX'] > config.ADX_THRESHOLD),
                (f"Volume Ratio > {config.VOLUME_SURGE_RATIO}", indicators['Volume_Ratio'] > config.VOLUME_SURGE_RATIO),
                (f"Volatility Ratio < {config.VOLATILITY_CONTRACTION_RATIO}", 
                indicators['Volatility_Ratio'] < config.VOLATILITY_CONTRACTION_RATIO),
                (f"RSI between {config.RSI_MIN}-{config.RSI_MAX}", 
                config.RSI_MIN <= indicators['RSI'] <= config.RSI_MAX)
            ]
            
            # Check all conditions
            for name, condition in conditions:
                if not condition:
                    failure_reasons.append(name)
            
            # If any condition failed, return with reasons
            if failure_reasons:
                return None, failure_reasons
            
            # Create breakout strategy with volume confirmation
            breakout_system = EnhancedBreakoutStrategy(
                indicators['Close'], 
                indicators['ATR'], 
                indicators['ADX'],
                indicators['Volume'],
                indicators['AvgVolume']
            )
            
            # Update with current bar (simulates breakout day)
            current_bar = {
                'High': indicators['High'],
                'Low': indicators['Low'],
                'Close': indicators['Close'],
                'Volume': indicators['Volume']
            }
            current_stop = breakout_system.update(current_bar)
            
            # Calculate risk metrics
            risk_per_share = indicators['Close'] - current_stop
            profit_target = indicators['Close'] + (config.REWARD_RISK_MULTIPLIER * indicators['ATR'])
            reward_risk_ratio = (profit_target - indicators['Close']) / risk_per_share
            
            # Position sizing per $10,000 capital
            position_size = (10000 * config.MAX_RISK_PERCENT) / risk_per_share if risk_per_share > 0 else 0
            
            return {
                'Ticker': ticker,
                'Upside_Score': round(indicators['Relative_Strength'], 1),  # Use relative strength for score
                'Price': round(indicators['Close'], 2),
                'Stop_Loss': round(current_stop, 2),
                'Risk_Share': round(risk_per_share, 2),
                'Position_Size': round(position_size, 0),
                'Profit_Target': round(profit_target, 2),
                'Reward_Risk_Ratio': round(reward_risk_ratio, 1),
                'Upside_Potential%': round((profit_target/indicators['Close'] - 1)*100, 1),
                'ADX': round(indicators['ADX'], 1),
                'RSI': round(indicators['RSI'], 1),
                'Volume_Ratio': round(indicators['Volume_Ratio'], 1),
                'ATR': round(indicators['ATR'], 2)
            }, None
        except Exception as e:
            logger.exception(f"Critical error processing {ticker}")
            return None, [f"Critical error: {str(e)}"]

    def setup_failure_logger(self, scan_start_time):
        """Create a dedicated logger for failure details"""
        if not os.path.exists("failure_logs"):
            os.makedirs("failure_logs")
            
        timestamp = scan_start_time.strftime("%Y%m%d_%H%M%S")
        log_file = f"failure_logs/{config.FAILURE_LOG_PREFIX}{timestamp}.log"
        
        failure_logger = logging.getLogger(f"FailureLogger_{timestamp}")
        failure_logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        if failure_logger.hasHandlers():
            failure_logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        failure_logger.addHandler(file_handler)
        
        failure_logger.propagate = False
        return failure_logger

    def log_failure_details(self, ticker, reasons):
        """Log detailed failure reasons to dedicated file"""
        if self.failure_logger:
            reason_str = ", ".join(reasons)
            self.failure_logger.info(f"{ticker}: {reason_str}")

    async def scan_tickers_async(self, shutdown_event, tickers=None):
        scan_start_time = datetime.now()
        self.failure_logger = self.setup_failure_logger(scan_start_time)
        
        # Market regime filter - skip bear markets
        if self.spy_sma_50 and self.spy_sma_200 and self.spy_sma_50 < self.spy_sma_200:
            logger.warning("Bear market detected (SPY 50d < 200d), skipping scan")
            self.log_failure_details("MARKET", ["Bear market detected"])
            return pd.DataFrame()
        
        results = []
        failure_summary = {}
        
        # Use provided tickers or get current list
        if tickers is None:
            tickers = self.ticker_scanner.get_current_tickers_list()
            
        if not tickers:
            logger.warning("No tickers available for scanning")
            return pd.DataFrame()
        
        scan_tickers = tickers[:config.MAX_TICKERS]
        total_tickers = len(scan_tickers)
        logger.info(f"Starting scan of {total_tickers} stocks")
        
        semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
        
        async def process_with_semaphore(ticker):
            try:
                async with semaphore:
                    if shutdown_event.is_set():
                        return None, ["Shutdown"]
                    result, reasons = await self.process_ticker(ticker)
                    if shutdown_event.is_set():
                        return None, ["Shutdown"]
                    await asyncio.sleep(config.RATE_LIMIT_DELAY)
                    return ticker, result, reasons
            except Exception as e:
                logger.error(f"Task error for {ticker}: {str(e)}")
                return ticker, None, [f"Task error: {str(e)}"]
        
        # Create tasks properly
        tasks = [asyncio.create_task(process_with_semaphore(t)) for t in scan_tickers]
        
        # Process with progress bar
        pbar = tqdm(total=len(tasks), desc="Scanning stocks", disable=shutdown_event.is_set())
        
        try:
            for future in asyncio.as_completed(tasks):
                if shutdown_event.is_set():
                    # Cancel all remaining tasks
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    # Wait for tasks to handle cancellation
                    await asyncio.gather(*tasks, return_exceptions=True)
                    break
                    
                try:
                    ticker, result, reasons = await future
                    if result: 
                        results.append(result)
                    elif reasons:
                        # Log detailed failure
                        self.log_failure_details(ticker, reasons)
                        
                        # Update failure summary
                        for reason in reasons:
                            failure_summary[reason] = failure_summary.get(reason, 0) + 1
                except asyncio.CancelledError:
                    logger.debug("Scan task cancelled")
                except Exception as e:
                    if not shutdown_event.is_set():
                        logger.error(f"Error in scan task: {e}")
                
                # Update progress
                pbar.update(1)
        
        finally:
            pbar.close()
        
        # Log failure statistics
        if failure_summary:
            logger.info("Breakout condition failure summary:")
            total_failures = sum(failure_summary.values())
            for reason, count in sorted(failure_summary.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_failures) * 100
                logger.info(f"  {reason}: {count} stocks ({percentage:.1f}%)")
        else:
            logger.info("No stocks failed conditions (all passed or data errors)")
        
        if results:
            df = pd.DataFrame(results)
            df['Rank'] = df['Upside_Score'].rank(ascending=False).astype(int)
            return df.sort_values('Upside_Score', ascending=False)
        return pd.DataFrame()
    
    def save_results(self, df):
        if df is None or df.empty:
            logger.warning("No results to save")
            return None
            
        filename = f"{config.RESULTS_FILE_PREFIX}{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"Results saved to {filename}")
        return filename

# ======================== BACKTESTING FUNCTIONALITY ======================== #
class BacktestingThread(QThread):
    backtest_completed = pyqtSignal(pd.DataFrame)
    backtest_progress = pyqtSignal(int)
    backtest_status = pyqtSignal(str)
    backtest_log = pyqtSignal(str)
    backtest_finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.start_date = None
        self.end_date = None
        self.ticker_scanner = PolygonTickerScanner()
        self.is_running = False
        self.results = []
        self.total_days = 0
        self.processed_days = 0
        self.all_tickers = []  # Stores all tickers for the date range

    def set_date_range(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def run(self):
        if not self.start_date or not self.end_date:
            self.backtest_log.emit("Error: Date range not set")
            return
            
        self.is_running = True
        self.backtest_status.emit("Initializing backtesting...")
        self.backtest_log.emit(f"Backtesting from {self.start_date} to {self.end_date}")
        
        try:
            # Initialize ticker scanner
            self.ticker_scanner.start()
            self.ticker_scanner.initial_refresh_complete.wait()
            
            # Convert to datetime objects
            start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
            
            # Fetch all stocks for the date range
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.all_tickers = loop.run_until_complete(
                self.ticker_scanner.fetch_all_stocks_for_range(start_dt, end_dt)
            )
            self.backtest_log.emit(f"Found {len(self.all_tickers)} stocks active during backtest period")
            
            # Calculate date range
            self.total_days = (end_dt - start_dt).days + 1
            self.processed_days = 0
            
            # Run backtest for each day in range
            current_dt = start_dt
            while current_dt <= end_dt and self.is_running:
                self.backtest_status.emit(f"Processing {current_dt.strftime('%Y-%m-%d')}")
                self.backtest_log.emit(f"Scanning for {current_dt.strftime('%Y-%m-%d')}...")
                
                # Create scanner for this specific date
                scanner = PolygonBreakoutScanner(self.ticker_scanner)
                scanner.end_date = current_dt.strftime('%Y-%m-%d')
                scanner.start_date = (current_dt - timedelta(days=365)).strftime('%Y-%m-%d')
                
                # Run the scan for this date with all tickers
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(scanner.async_init())
                results = loop.run_until_complete(
                    scanner.scan_tickers_async(asyncio.Event(), tickers=self.all_tickers)
                )
                loop.run_until_complete(scanner.async_close())
                loop.close()
                
                # Process results
                if results is not None and not results.empty:
                    results['Date'] = current_dt.strftime('%Y-%m-%d')
                    self.results.append(results)
                    self.backtest_log.emit(f"Found {len(results)} opportunities for {current_dt.strftime('%Y-%m-%d')}")
                else:
                    self.backtest_log.emit(f"No opportunities found for {current_dt.strftime('%Y-%m-%d')}")
                
                # Update progress
                self.processed_days += 1
                progress = int((self.processed_days / self.total_days) * 100)
                self.backtest_progress.emit(progress)
                
                # Move to next day
                current_dt += timedelta(days=1)
            
            # Combine all results
            if self.results:
                all_results = pd.concat(self.results)
                self.backtest_completed.emit(all_results)
            else:
                self.backtest_log.emit("No opportunities found during backtest period")
            
        except Exception as e:
            self.backtest_log.emit(f"Backtesting error: {str(e)}")
        finally:
            self.is_running = False
            self.backtest_finished.emit()
            self.backtest_status.emit("Backtesting completed")

    def stop(self):
        self.is_running = False
        self.backtest_log.emit("Backtesting stopped by user")


class BacktestingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        layout = QVBoxLayout(self)
        
        # Date range selection
        date_layout = QHBoxLayout()
        date_layout.addWidget(QLabel("Start Date:"))
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDate(datetime.now().date() - timedelta(days=365))
        date_layout.addWidget(self.start_date_edit)
        
        date_layout.addSpacing(20)
        date_layout.addWidget(QLabel("End Date:"))
        self.end_date_edit = QDateEdit()
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDate(datetime.now().date())
        date_layout.addWidget(self.end_date_edit)
        
        layout.addLayout(date_layout)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Backtest")
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        self.start_btn.clicked.connect(self.start_backtest)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Backtest")
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white;")
        self.stop_btn.clicked.connect(self.stop_backtest)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        
        self.export_btn = QPushButton("Export Results")
        self.export_btn.setStyleSheet("background-color: #2196F3; color: white;")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        btn_layout.addWidget(self.export_btn)
        
        layout.addLayout(btn_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready to start backtesting")
        self.status_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.status_label)
        
        # Results table
        self.results_table = QTableView()
        self.results_table.setSortingEnabled(True)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        layout.addWidget(self.results_table)
        
        # Log view
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view)
        
        # Backtesting thread
        self.backtest_thread = BacktestingThread()
        self.backtest_thread.backtest_completed.connect(self.show_results)
        self.backtest_thread.backtest_progress.connect(self.update_progress)
        self.backtest_thread.backtest_status.connect(self.update_status)
        self.backtest_thread.backtest_log.connect(self.update_log)
        self.backtest_thread.backtest_finished.connect(self.backtest_finished)
        
        self.current_results = None

    def start_backtest(self):
        start_date = self.start_date_edit.date().toString("yyyy-MM-dd")
        end_date = self.end_date_edit.date().toString("yyyy-MM-dd")
        
        if start_date >= end_date:
            QMessageBox.warning(self, "Invalid Date Range", "End date must be after start date")
            return
            
        self.backtest_thread.set_date_range(start_date, end_date)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.log_view.clear()
        self.backtest_thread.start()

    def stop_backtest(self):
        self.backtest_thread.stop()
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Stopping backtest...")

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_status(self, message):
        self.status_label.setText(message)

    def update_log(self, message):
        self.log_view.append(message)

    def show_results(self, results_df):
        self.current_results = results_df
        model = PandasModel(results_df)
        self.results_table.setModel(model)
        self.results_table.resizeColumnsToContents()
        self.export_btn.setEnabled(True)
        self.update_log(f"Backtest completed with {len(results_df)} total opportunities")

    def backtest_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def export_results(self):
        if self.current_results is None or self.current_results.empty:
            QMessageBox.warning(self, "Export Error", "No results to export")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                self.current_results.to_csv(file_path, index=False)
                self.update_log(f"Exported results to {file_path}")
                self.status_label.setText(f"Results exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export results: {str(e)}")

# ======================== PYQT UI COMPONENTS ======================== #
class ScannerThread(QThread):
    scan_completed = pyqtSignal(pd.DataFrame)
    status_update = pyqtSignal(str)
    scan_finished = pyqtSignal()
    results_saved = pyqtSignal(str)
    log_message = pyqtSignal(str)  # New signal for log messages
    thread_finished = pyqtSignal()  # Signal to notify when thread is completely done

    def __init__(self, parent=None):
        super().__init__(parent)
        self.shutdown_event = asyncio.Event()
        self.is_running = False
        self.scanner = None
        self.ticker_scanner = PolygonTickerScanner()
        self.log_queue = queue.Queue()
        self.log_timer = QTimer()
        self.log_timer.setSingleShot(False)
        self.log_timer.timeout.connect(self.process_log_queue)

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.is_running = True
        self.status_update.emit("Starting scanner...")
        self.log_message.emit("Initializing ticker database...")
        
        try:
            # Start log timer in main thread context
            self.log_timer.start(100)
            
            # Initialize ticker scanner
            self.ticker_scanner.start()
            self.ticker_scanner.initial_refresh_complete.wait()
            num_tickers = len(self.ticker_scanner.get_current_tickers_list())
            self.log_message.emit(f"Ticker refresh complete: {num_tickers} symbols")
            
            # Initialize breakout scanner
            self.scanner = PolygonBreakoutScanner(self.ticker_scanner)
            loop.run_until_complete(self.scanner.async_init())
            self.status_update.emit("Scanner initialized")
            
            # Main scanning loop
            last_scan_time = 0
            while self.is_running:
                current_time = time.time()
                
                if current_time - last_scan_time > config.SCAN_INTERVAL:
                    self.status_update.emit("Starting scan...")
                    self.log_message.emit("Scanning for breakout opportunities...")
                    results = loop.run_until_complete(
                        self.scanner.scan_tickers_async(self.shutdown_event)
                    )
                    
                    if results is not None and not results.empty:
                        self.scan_completed.emit(results)
                        self.status_update.emit(f"Found {len(results)} opportunities")
                        filename = self.scanner.save_results(results)
                        if filename:
                            self.results_saved.emit(filename)
                    else:
                        self.status_update.emit("No opportunities found")
                        self.log_message.emit("No breakout opportunities found in this scan")
                    
                    last_scan_time = time.time()
                    next_scan = datetime.fromtimestamp(
                        last_scan_time + config.SCAN_INTERVAL).strftime('%H:%M:%S')
                    self.status_update.emit(f"Next scan at {next_scan}")
                    self.scan_finished.emit()
                
                # Check every second if we need to stop
                for _ in range(10):
                    if not self.is_running:
                        break
                    time.sleep(0.1)
            
        except Exception as e:
            self.log_message.emit(f"ERROR: {str(e)}")
        finally:
            self.status_update.emit("Shutting down...")
            self.log_message.emit("Cleaning up resources...")
            
            # Clean up asyncio tasks properly
            if self.scanner:
                try:
                    loop.run_until_complete(self.scanner.async_close())
                except Exception as e:
                    logger.error(f"Error closing scanner: {e}")
            
            self.ticker_scanner.stop()
            
            # Cancel all pending tasks
            tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
            if tasks:
                for task in tasks:
                    task.cancel()
                # Wait for tasks to handle cancellation
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            
            loop.close()
            self.is_running = False
            self.status_update.emit("Scanner stopped")
            self.log_timer.stop()
            self.thread_finished.emit()

    def stop(self):
        self.is_running = False
        self.shutdown_event.set()
        
    def process_log_queue(self):
        """Process any pending log messages in the queue"""
        while not self.log_queue.empty():
            try:
                msg = self.log_queue.get_nowait()
                self.log_message.emit(msg)
            except queue.Empty:
                break

class PandasModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
            
        if role == Qt.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
            
        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter
            
        if role == Qt.BackgroundRole:
            # Color coding based on columns
            col_name = self._data.columns[index.column()]
            
            if col_name == 'Upside_Score':
                try:
                    score = float(self._data.iloc[index.row(), index.column()])
                    if score > 80:
                        return QColor(144, 238, 144)  # Light green
                    elif score > 60:
                        return QColor(173, 216, 230)  # Light blue
                except:
                    pass
                    
            elif col_name == 'Reward_Risk_Ratio':
                try:
                    ratio = float(self._data.iloc[index.row(), index.column()])
                    if ratio > 3.0:
                        return QColor(144, 238, 144)  # Light green
                    elif ratio > 2.0:
                        return QColor(173, 216, 230)  # Light blue
                except:
                    pass
                    
            elif col_name == 'Position_Size':
                return QColor(230, 230, 250)  # Lavender
                
            elif col_name == 'Stop_Loss':
                return QColor(255, 228, 225)  # Misty rose
                
        return None

    def headerData(self, section, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[section]
        return None

class ScannerUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Polygon Stock Scanner")
        self.setGeometry(100, 100, 1200, 800)
        self.current_results = None
        self.scanner_thread = None
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create menu bar
        self.create_menu()
        
        # Control panel
        control_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Scanner")
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.start_btn.clicked.connect(self.start_scanner)
        self.start_btn.setFixedHeight(40)
        
        self.stop_btn = QPushButton("Stop Scanner")
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self.stop_btn.clicked.connect(self.stop_scanner)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setFixedHeight(40)
        
        self.refresh_btn = QPushButton("Refresh Now")
        self.refresh_btn.setStyleSheet("background-color: #2196F3; color: white;")
        self.refresh_btn.clicked.connect(self.force_refresh)
        self.refresh_btn.setEnabled(True)
        self.refresh_btn.setFixedHeight(40)
        
        self.status_label = QLabel("Status: Not running")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.refresh_btn)
        control_layout.addSpacing(20)
        control_layout.addWidget(self.status_label)
        control_layout.addStretch()
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(20)
        
        # Stats label
        self.stats_label = QLabel("Tickers: 0 | Last Refresh: Never")
        self.stats_label.setStyleSheet("font-size: 10pt;")
        
        # Tab widget
        self.tab_widget = QTabWidget()
        
        # Results tab
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        
        self.results_table = QTableView()
        self.results_table.setSortingEnabled(True)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QTableView.SelectRows)
        self.results_table.setSelectionMode(QTableView.SingleSelection)
        
        results_layout.addWidget(QLabel("Breakout Opportunities:"))
        results_layout.addWidget(self.results_table)
        
        # Logs tab
        logs_tab = QWidget()
        logs_layout = QVBoxLayout(logs_tab)
        
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Courier New", 10))
        self.log_view.setStyleSheet("background-color: #f0f0f0;")
        
        logs_layout.addWidget(QLabel("Scanner Logs:"))
        logs_layout.addWidget(self.log_view)
        
        # Create backtesting tab
        self.backtesting_tab = BacktestingTab(self)
        self.tab_widget.addTab(self.backtesting_tab, "Backtesting")
        
        # Add tabs
        self.tab_widget.addTab(results_tab, "Results")
        self.tab_widget.addTab(logs_tab, "Logs")
        
        # Add widgets to main layout
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.stats_label)
        main_layout.addWidget(self.tab_widget)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Timer for stats updates
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(5000)  # Update every 5 seconds
        
        # Initial log message
        self.update_log("Application started. Click 'Start Scanner' to begin.")
        self.update_log(f"Using Polygon API key: {config.POLYGON_API_KEY[:6]}...")
        self.update_log(f"Stoploss configuration: Base Multiplier={config.BASE_MULTIPLIER}, ADX Scaling=[{config.MIN_STOP_MULTIPLIER}-{config.MAX_STOP_MULTIPLIER}]")
        
    def create_menu(self):
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("&File")
        
        export_action = QAction("Export Results", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def start_scanner(self):
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.refresh_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        # Create scanner thread only when starting
        self.scanner_thread = ScannerThread()
        self.scanner_thread.scan_completed.connect(self.update_results)
        self.scanner_thread.status_update.connect(self.update_status)
        self.scanner_thread.scan_finished.connect(self.scan_finished)
        self.scanner_thread.results_saved.connect(self.results_saved)
        self.scanner_thread.log_message.connect(self.update_log)
        self.scanner_thread.thread_finished.connect(self.thread_finished)
        
        self.scanner_thread.start()
        self.status_bar.showMessage("Scanner started")

    def stop_scanner(self):
        if self.scanner_thread and self.scanner_thread.isRunning():
            self.stop_btn.setEnabled(False)
            self.scanner_thread.stop()
            self.status_bar.showMessage("Stopping scanner...")

    def force_refresh(self):
        if self.scanner_thread and self.scanner_thread.isRunning():
            # Access the ticker scanner through the thread
            self.scanner_thread.ticker_scanner._refresh_all_tickers()
            self.update_log("Forced ticker refresh initiated")
            self.status_bar.showMessage("Refreshing ticker data...")
        else:
            self.update_log("Scanner not running - cannot refresh")
        
    def update_log(self, message):
        self.log_view.append(message)
        # Auto-scroll to bottom
        self.log_view.verticalScrollBar().setValue(
            self.log_view.verticalScrollBar().maximum()
        )
        self.status_bar.showMessage(message.split(" - ")[-1] if " - " in message else message)

    def update_results(self, results_df):
        self.current_results = results_df
        model = PandasModel(results_df)
        self.results_table.setModel(model)
        self.results_table.resizeColumnsToContents()
        self.update_log(f"Displaying {len(results_df)} breakout opportunities")

    def update_status(self, message):
        self.status_label.setText(f"Status: {message}")

    def scan_finished(self):
        self.progress_bar.setValue(0)
        self.refresh_btn.setEnabled(True)
        self.status_bar.showMessage("Scan completed")

    def results_saved(self, filename):
        self.status_bar.showMessage(f"Results saved to {filename}")

    def thread_finished(self):
        """Handle thread completion"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_bar.showMessage("Scanner stopped")

    def export_results(self):
        if self.current_results is None or self.current_results.empty:
            QMessageBox.warning(self, "Export Error", "No results to export")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                self.current_results.to_csv(file_path, index=False)
                self.update_log(f"Exported results to {file_path}")
                self.status_bar.showMessage(f"Results exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export results: {str(e)}")

    def show_about(self):
        about_text = (
            "<b>Polygon Stock Scanner</b><br><br>"
            "This application scans for breakout opportunities using technical analysis indicators.<br><br>"
            "Version: 3.0 (Professional Edition with Backtesting)<br>"
            "Data Provider: Polygon.io<br><br>"
            "Features:<br>"
            "- Real-time scanning of US stock markets<br>"
            "- Historical backtesting with custom date ranges<br>"
            "- Professional-grade trailing stoploss with ADX/volatility scaling<br>"
            "- Position sizing based on risk management<br>"
            "- Market regime detection (bull/bear markets)<br>"
            "- Automatic hourly scanning<br>"
            "- Results ranking with Upside Score<br><br>"
            "© 2023 Polygon Stock Scanner"
        )
        QMessageBox.about(self, "About Polygon Stock Scanner", about_text)

    def update_stats(self):
        """Update statistics periodically"""
        if self.scanner_thread and self.scanner_thread.isRunning() and hasattr(self.scanner_thread, 'ticker_scanner'):
            tickers = self.scanner_thread.ticker_scanner.get_current_tickers_list()
            last_refresh = datetime.fromtimestamp(
                self.scanner_thread.ticker_scanner.last_refresh_time
            ).strftime('%Y-%m-%d %H:%M') if self.scanner_thread.ticker_scanner.last_refresh_time > 0 else "Never"
            
            self.stats_label.setText(
                f"Tickers: {len(tickers)} | Last Refresh: {last_refresh}"
            )

    def closeEvent(self, event):
        if self.scanner_thread and self.scanner_thread.isRunning():
            self.scanner_thread.stop()
            self.scanner_thread.wait(5000)  # Wait up to 5 seconds for clean exit
            
        event.accept()

# ======================== MAIN APPLICATION ======================== #
if __name__ == "__main__":
    # Windows event loop policy
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern UI style
    
    # Set application font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = ScannerUI()
    window.show()
    sys.exit(app.exec_())