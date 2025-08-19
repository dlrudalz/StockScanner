import numpy as np
import pandas as pd
import asyncio
import aiohttp
import time
import os
import logging
import json
from datetime import datetime, timedelta, time as dt_time, date
import pytz
from threading import Lock, Event, Thread
from tqdm import tqdm
import signal
from urllib.parse import urlencode
import sys
import queue
import pickle
import traceback
import platform
import psutil
import requests

# PyQt5 imports for UI
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QTableView, QTextEdit, QLabel, QHeaderView, QProgressBar,
                            QAction, QMenu, QFileDialog, QStatusBar, QMessageBox, QTableWidget, 
                            QTableWidgetItem, QSplitter, QAbstractItemView)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QAbstractTableModel, QTimer
from PyQt5.QtGui import QFont, QColor, QIcon

# ======================== CONSOLIDATED CONFIGURATION ======================== #
class Config:
    # API Configuration
    POLYGON_API_KEY = "ld1Poa63U6t4Y2MwOCA2JeKQyHVrmyg8"
    BROKER_API_KEY = "YOUR_BROKER_API_KEY"  # Replace with actual broker API
    
    # Scanner Behavior
    EXCHANGES = ["XNAS", "XNYS", "XASE"]
    MAX_TICKERS = 5000
    MAX_CONCURRENT_REQUESTS = 50
    RATE_LIMIT_DELAY = 0.05  # 50ms between requests
    
    # File Management
    TICKER_CACHE_FILE = "ticker_cache.parquet"
    MISSING_TICKERS_FILE = "missing_tickers.json"
    METADATA_FILE = "scanner_metadata.json"
    STATE_FILE = "trading_state.pkl"
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
    ACCOUNT_VALUE = 10000  # Starting account value
    
    # Stoploss Parameters
    VOLATILITY_SCALING_FACTOR = 0.8  # Min volatility multiplier
    TIME_DECAY_FACTOR = 0.02  # 2% tightening per day after 5 days
    MIN_STOP_MULTIPLIER = 0.7  # Minimum stop multiplier
    MAX_STOP_MULTIPLIER = 2.0  # Maximum stop multiplier
    
    # Trading Hours (Eastern Time)
    PRE_MARKET_START = dt_time(4, 0)    # 4:00 AM ET
    MARKET_OPEN = dt_time(9, 30)        # 9:30 AM ET
    MARKET_CLOSE = dt_time(16, 0)        # 4:00 PM ET
    POST_MARKET_END = dt_time(20, 0)    # 8:00 PM ET
    
    # Scheduling Parameters (seconds)
    POSITION_MONITOR_INTERVAL = 15      # Check positions every 15 seconds
    HEALTH_CHECK_INTERVAL = 3600        # System checks every hour
    DAILY_TICKER_REFRESH_TIME = dt_time(4, 0)  # 4:00 AM ET daily
    REPORT_GENERATION_TIME = dt_time(17, 0)    # 5:00 PM ET daily
    
    # Scan Intervals by Market Phase (seconds)
    SCAN_INTERVALS = {
        "pre_market": 1800,    # 30 minutes
        "market_hours": 300,   # 5 minutes
        "post_market": 3600,   # 60 minutes
        "overnight": 7200      # 120 minutes
    }
    
    # Logging Configuration
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Initialize configuration
config = Config()

# ======================== MARKET CALENDAR ======================== #
class TradingCalendar:
    def __init__(self):
        self.ny_tz = pytz.timezone('America/New_York')
        self.trading_days = [0, 1, 2, 3, 4]  # Mon-Fri
        self.holidays = self.load_holidays()
        
    def load_holidays(self):
        # In production, fetch from reliable source like NYSE calendar
        # For now, hardcode major US holidays 2023
        return [
            datetime(2023, 1, 2).date(),   # New Year's (observed)
            datetime(2023, 1, 16).date(),   # MLK Day
            datetime(2023, 2, 20).date(),   # Presidents' Day
            datetime(2023, 4, 7).date(),    # Good Friday
            datetime(2023, 5, 29).date(),   # Memorial Day
            datetime(2023, 6, 19).date(),   # Juneteenth
            datetime(2023, 7, 4).date(),    # Independence Day
            datetime(2023, 9, 4).date(),    # Labor Day
            datetime(2023, 11, 23).date(),  # Thanksgiving
            datetime(2023, 12, 25).date()   # Christmas
        ]
    
    def is_trading_day(self, dt=None):
        dt = dt or datetime.now(self.ny_tz)
        date_val = dt.date()
        return dt.weekday() in self.trading_days and date_val not in self.holidays
    
    def get_market_phase(self, dt=None):
        dt = dt or datetime.now(self.ny_tz)
        if not self.is_trading_day(dt):
            return "closed"
        
        current_time = dt.time()
        if current_time < config.PRE_MARKET_START:
            return "overnight"
        elif current_time < config.MARKET_OPEN:
            return "pre_market"
        elif current_time <= config.MARKET_CLOSE:
            return "market_hours"
        elif current_time <= config.POST_MARKET_END:
            return "post_market"
        else:
            return "overnight"
    
    def seconds_until_next_event(self, event_time):
        now = datetime.now(self.ny_tz)
        event_dt = datetime.combine(now.date(), event_time)
        
        # If event already passed today, schedule for tomorrow
        if now > event_dt:
            event_dt += timedelta(days=1)
            
        # Skip weekends/holidays
        while not self.is_trading_day(event_dt):
            event_dt += timedelta(days=1)
            
        return (event_dt - now).total_seconds()
    
    def is_time_for_daily_refresh(self, dt=None):
        """Check if it's time for daily refresh and we haven't refreshed today"""
        dt = dt or datetime.now(self.ny_tz)
        return dt.time() >= config.DAILY_TICKER_REFRESH_TIME

# ======================== LOGGING SETUP ======================== #
def setup_logging():
    """Configure logging with file and console handlers"""
    # Create logs directory if not exists
    os.makedirs("logs", exist_ok=True)
    
    # Create main logger
    logger = logging.getLogger("QuantTradingSystem")
    logger.setLevel(config.LOG_LEVEL)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(config.LOG_FORMAT)
    
    # File handler - all logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/trading_system_{timestamp}.log"
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

    async def _fetch_exchange_tickers(self, session, exchange):
        logger.info(f"Fetching tickers for {exchange}")
        all_results = []
        next_url = None
        params = {
            "market": "stocks",
            "exchange": exchange,
            "active": "true",
            "limit": 1000,
            "apiKey": self.api_key
        }
        
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
        return True

    async def refresh_all_tickers(self):
        """Public async method to refresh tickers"""
        with self.refresh_lock:
            return await self._refresh_all_tickers_async()

    def start(self):
        if not self.active:
            self.active = True
            # Initialize cache only when started
            self._init_cache()
            self.initial_refresh_complete.set()
            logger.info("Ticker scanner started")

    def stop(self):
        self.active = False
        logger.info("Ticker scanner stopped")

    def get_current_tickers_list(self):
        with self.cache_lock:
            return self.ticker_cache['ticker'].tolist()

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

# ======================== TRADING SYSTEM CORE ======================== #
class QuantTradingSystem:
    def __init__(self, ticker_scanner):
        self.ticker_scanner = ticker_scanner
        self.session = None
        self.spy_data = None
        self.spy_3mo_return = 0
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        self.shutting_down = False
        self.failure_logger = None
        self.spy_sma_50 = None
        self.spy_sma_200 = None
        self.calendar = TradingCalendar()
        self.open_positions = []  # Format: [{'symbol': 'AAPL', 'entry_price': 150, 'quantity': 10, 'stop_loss': 145}]
        self.account_value = config.ACCOUNT_VALUE
        self.consecutive_errors = 0
        self.last_health_check = 0
        self.last_state_save = 0
        self.failure_summary = {}
        self.last_data_refresh = 0
        self.state_loaded = False  # Track if state has been loaded
        self.refresh_in_progress = False  # Add refresh lock flag
        self.last_refresh_date = None  # Track last successful refresh date
        self.daily_refresh_done = False  # Track if daily refresh completed
        
    async def async_init(self):
        self.session = aiohttp.ClientSession()
        await self.refresh_market_data()
        self.ticker_scanner.initial_refresh_complete.wait()
        # Only load state once
        if not self.state_loaded:
            self.load_system_state()
            self.state_loaded = True
        
    async def async_close(self):
        if self.session:
            self.shutting_down = True
            await self.session.close()
            logger.debug("HTTP session closed")
        
        if self.failure_logger:
            for handler in self.failure_logger.handlers[:]:
                handler.close()
                self.failure_logger.removeHandler(handler)

    def should_refresh_market_data(self):
        """Check if market data needs refreshing"""
        current_time = time.time()
        # Don't refresh if we just did one
        if current_time - self.last_data_refresh < 5:  # 5-second cooldown
            return False
            
        # Check if we're already refreshing
        if self.refresh_in_progress:
            return False
            
        # Get current date and time
        now = datetime.now(self.calendar.ny_tz)
        current_date = now.date()
        
        # If we've already done a daily refresh today, skip
        if self.daily_refresh_done and self.last_refresh_date == current_date:
            return False
            
        # Check if it's past refresh time (4:00 AM ET)
        return now.time() >= config.DAILY_TICKER_REFRESH_TIME

    async def refresh_market_data(self):
        """Refresh all market data including tickers and SPY"""
        # Only refresh if needed and not already refreshing
        if not self.should_refresh_market_data() or self.refresh_in_progress:
            return False
            
        self.refresh_in_progress = True
        logger.info("Refreshing market data")
        start_time = time.time()
        
        try:
            # Refresh ticker data
            if await self.ticker_scanner.refresh_all_tickers():
                logger.info("Ticker data refreshed")
            
            # Refresh SPY data
            self.spy_data = await self.async_get_polygon_data("SPY")
            if self.spy_data is not None:
                self.spy_3mo_return = self.calculate_3mo_return(self.spy_data)
                if len(self.spy_data) >= 200:
                    self.spy_sma_50 = self.spy_data['Close'].rolling(50).mean().iloc[-1]
                    self.spy_sma_200 = self.spy_data['Close'].rolling(200).mean().iloc[-1]
                logger.info("SPY data refreshed")
            
            # Update refresh state
            self.last_refresh_date = datetime.now(self.calendar.ny_tz).date()
            self.daily_refresh_done = True
            elapsed = time.time() - start_time
            logger.info(f"Market data refresh completed in {elapsed:.2f}s")
            return True
        except Exception as e:
            logger.error(f"Market data refresh failed: {e}")
            return False
        finally:
            self.refresh_in_progress = False
            self.last_data_refresh = time.time()

    async def async_get_polygon_data(self, ticker):
        """Fixed data fetching method with proper URL encoding and error handling"""
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
                if response.status == 200:
                    data = await response.json()
                    
                    # Handle empty results
                    if data.get('resultsCount', 0) == 0:
                        logger.warning(f"No results found for {ticker}")
                        return None
                        
                    df = pd.DataFrame(data['results'])
                    df.rename(columns={
                        'v': 'Volume', 'o': 'Open', 'c': 'Close',
                        'h': 'High', 'l': 'Low', 't': 'Timestamp'
                    }, inplace=True)
                    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
                    return df
                elif response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', 1))
                    logger.warning(f"Rate limit hit for {ticker}, retrying after {retry_after} seconds")
                    await asyncio.sleep(retry_after)
                    return await self.async_get_polygon_data(ticker)
                else:
                    logger.error(f"API request for {ticker} failed: {response.status}")
                    return None
        except asyncio.TimeoutError:
            logger.error(f"Request for {ticker} timed out")
            return None
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None

    def calculate_3mo_return(self, df):
        if len(df) < 63: 
            return 0
        return ((df['Close'].iloc[-1] / df['Close'].iloc[-63]) - 1) * 100

    def calculate_rsi(self, prices):
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
            
            consolidation_high = df['High'].rolling(20).max().iloc[-2]
            breakout_confirmed = close > consolidation_high * config.CONSOLIDATION_BREAKOUT_FACTOR
            
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
        except Exception:
            return None

    def calculate_upside_score(self, indicators):
        breakout_score = 30 if indicators['Breakout_Confirmed'] else 0
        rs_score = min(25, max(0, indicators['Relative_Strength'] * 0.5))
        vol_ratio = indicators['Volume_Ratio']
        volume_score = min(20, max(0, (vol_ratio - 1.0) * 10)) if vol_ratio > 1.0 else 0
        vol_contraction = indicators['Volatility_Ratio']
        volatility_score = 15 * max(0, 1 - min(1, vol_contraction/0.7))
        rsi = indicators['RSI']
        rsi_score = 10 * (1 - abs(rsi - 60)/20) if config.RSI_MIN <= rsi <= config.RSI_MAX else 0
        return min(100, breakout_score + rs_score + volume_score + volatility_score + rsi_score)

    async def process_ticker(self, ticker):
        if self.shutting_down:
            return None, ["Shutdown in progress"]
        
        data = await self.async_get_polygon_data(ticker)
        # Fix: Check if data is None or empty DataFrame
        if data is None or data.empty:
            return None, ["Failed to fetch data"]
        
        # Fix: Add explicit check for DataFrame length
        if len(data) < 200:
            return None, ["Insufficient historical data"]
        
        indicators = self.calculate_indicators(data)
        if indicators is None: 
            return None, ["Indicator calculation failed"]
        
        try:
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
            position_size = (config.ACCOUNT_VALUE * config.MAX_RISK_PERCENT) / risk_per_share if risk_per_share > 0 else 0
            
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
            return None, [f"Evaluation error: {str(e)}"]

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

    async def scan_tickers_async(self):
        scan_start_time = datetime.now()
        self.failure_logger = self.setup_failure_logger(scan_start_time)
        self.failure_summary = {}
        
        # Market regime filter - skip bear markets
        if self.spy_sma_50 and self.spy_sma_200 and self.spy_sma_50 < self.spy_sma_200:
            logger.warning("Bear market detected (SPY 50d < 200d), skipping scan")
            self.log_failure_details("MARKET", ["Bear market detected"])
            return pd.DataFrame(), self.failure_summary
        
        results = []
        failure_summary = {}
        tickers = self.ticker_scanner.get_current_tickers_list()
        if not tickers:
            logger.warning("No tickers available for scanning")
            return pd.DataFrame(), self.failure_summary
        
        scan_tickers = tickers[:config.MAX_TICKERS]
        total_tickers = len(scan_tickers)
        logger.info(f"Starting scan of {total_tickers} stocks")
        
        semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
        
        async def process_with_semaphore(ticker):
            async with semaphore:
                if self.shutting_down:
                    return None, ["Shutdown"]
                result, reasons = await self.process_ticker(ticker)
                if self.shutting_down:
                    return None, ["Shutdown"]
                await asyncio.sleep(config.RATE_LIMIT_DELAY)
                return ticker, result, reasons
        
        # Create tasks properly
        tasks = [asyncio.create_task(process_with_semaphore(t)) for t in scan_tickers]
        
        # Process with progress bar
        pbar = tqdm(total=len(tasks), desc="Scanning stocks", disable=True)
        
        try:
            for future in asyncio.as_completed(tasks):
                if self.shutting_down:
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
                    if not self.shutting_down:
                        logger.error(f"Error in scan task: {e}")
                
                # Update progress
                pbar.update(1)
        
        finally:
            pbar.close()
        
        # Store failure summary for UI
        self.failure_summary = failure_summary
        
        if results:
            df = pd.DataFrame(results)
            df['Rank'] = df['Upside_Score'].rank(ascending=False).astype(int)
            return df.sort_values('Upside_Score', ascending=False), self.failure_summary
        return pd.DataFrame(), self.failure_summary
    
    def save_results(self, df):
        filename = f"{config.RESULTS_FILE_PREFIX}{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"Results saved to {filename}")
        return filename

    def get_scan_interval(self):
        market_phase = self.calendar.get_market_phase()
        return config.SCAN_INTERVALS.get(market_phase, 7200)
    
    def is_time_for_daily_refresh(self):
        return self.calendar.is_time_for_daily_refresh()
    
    async def monitor_positions(self):
        """Monitor open positions and execute stop-loss/take-profit"""
        if not self.open_positions:
            return
            
        # In production, replace with real-time price feed
        logger.info(f"Monitoring {len(self.open_positions)} open positions")
        
        # Placeholder - in reality you'd fetch real-time prices
        for position in self.open_positions:
            try:
                # Fetch current price - in reality from real-time API
                current_price = await self.get_realtime_price(position['symbol'])
                
                # Check stop loss
                if current_price <= position['stop_loss']:
                    logger.info(f"Stop loss triggered for {position['symbol']} at {current_price}")
                    await self.execute_trade(position['symbol'], "SELL", position['quantity'])
                    self.open_positions.remove(position)
                    
                # Check profit target
                elif current_price >= position['profit_target']:
                    logger.info(f"Profit target hit for {position['symbol']} at {current_price}")
                    await self.execute_trade(position['symbol'], "SELL", position['quantity'])
                    self.open_positions.remove(position)
                    
            except Exception as e:
                logger.error(f"Position monitoring error for {position['symbol']}: {e}")
    
    async def get_realtime_price(self, symbol):
        """Placeholder for real-time price feed"""
        # In production, connect to Polygon websocket or broker API
        return 150.0  # Dummy value
    
    async def execute_trade(self, symbol, side, quantity):
        """Placeholder for trade execution"""
        # In production, integrate with broker API
        logger.info(f"Executing {side} order for {quantity} shares of {symbol}")
        # Placeholder: Implement actual order placement
        return True
    
    def run_health_check(self):
        """Perform system health diagnostics"""
        now = time.time()
        if now - self.last_health_check < config.HEALTH_CHECK_INTERVAL:
            return
            
        self.last_health_check = now
        logger.info("Running system health check")
        
        # API connectivity
        api_ok = self.check_polygon_api()
        
        # System resources
        cpu_percent = psutil.cpu_percent()
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Error tracking
        error_status = "OK" if self.consecutive_errors == 0 else f"WARNING: {self.consecutive_errors} consecutive errors"
        
        # Consolidated health report
        health_report = (
            f"System Health: API: {'OK' if api_ok else 'FAILED'}, "
            f"CPU: {cpu_percent}%, Mem: {mem.percent}% ({mem.used/1e6:.1f}MB/{mem.total/1e6:.1f}MB), "
            f"Disk: {disk.percent}% ({disk.used/1e9:.1f}GB/{disk.total/1e9:.1f}GB), "
            f"Errors: {error_status}"
        )
        logger.info(health_report)
        
        # Circuit breaker
        if self.consecutive_errors > 5:
            logger.critical("Circuit breaker triggered - too many consecutive errors")
            # Implement pause or shutdown logic
        
        return api_ok and cpu_percent < 90 and mem.percent < 90 and disk.percent < 90
    
    def check_polygon_api(self):
        """Check Polygon API connectivity"""
        try:
            response = requests.get(
                f"https://api.polygon.io/v1/marketstatus?apiKey={config.POLYGON_API_KEY}",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def save_system_state(self):
        """Persist critical system state"""
        now = time.time()
        if now - self.last_state_save < 300:  # Save every 5 minutes
            return
            
        self.last_state_save = now
        state = {
            'open_positions': self.open_positions,
            'account_value': self.account_value,
            'last_scan_time': self.last_scan_time if hasattr(self, 'last_scan_time') else 0,
            'consecutive_errors': self.consecutive_errors,
            'last_refresh_date': self.last_refresh_date,
            'daily_refresh_done': self.daily_refresh_done
        }
        
        try:
            with open(config.STATE_FILE, 'wb') as f:
                pickle.dump(state, f)
            logger.debug("System state saved")
        except Exception as e:
            logger.error(f"State save failed: {e}")
    
    def load_system_state(self):
        """Load persisted system state"""
        if self.state_loaded:  # Prevent duplicate loading
            return
            
        if not os.path.exists(config.STATE_FILE):
            return
            
        try:
            with open(config.STATE_FILE, 'rb') as f:
                state = pickle.load(f)
                self.open_positions = state.get('open_positions', [])
                self.account_value = state.get('account_value', config.ACCOUNT_VALUE)
                if hasattr(self, 'last_scan_time'):
                    self.last_scan_time = state.get('last_scan_time', 0)
                self.consecutive_errors = state.get('consecutive_errors', 0)
                self.last_refresh_date = state.get('last_refresh_date', None)
                self.daily_refresh_done = state.get('daily_refresh_done', False)
            logger.info("System state loaded")
            self.state_loaded = True
            
            # Reset daily refresh if it's a new day
            current_date = datetime.now(self.calendar.ny_tz).date()
            if self.last_refresh_date != current_date:
                self.daily_refresh_done = False
                logger.info("New trading day detected, resetting refresh flag")
        except Exception as e:
            logger.error(f"State load failed: {e}")

# ======================== PYQT UI COMPONENTS ======================== #
class ScannerThread(QThread):
    scan_completed = pyqtSignal(pd.DataFrame)
    status_update = pyqtSignal(str)
    scan_finished = pyqtSignal()
    results_saved = pyqtSignal(str)
    log_message = pyqtSignal(str)
    thread_finished = pyqtSignal()
    position_update = pyqtSignal(list)
    failure_summary = pyqtSignal(dict)
    data_refreshed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.shutdown_event = asyncio.Event()
        self.is_running = False
        self.trading_system = None
        self.ticker_scanner = PolygonTickerScanner()
        self.log_queue = queue.Queue()
        self.log_timer = QTimer()
        self.log_timer.setSingleShot(False)
        self.log_timer.timeout.connect(self.process_log_queue)
        self.last_scan_time = 0
        self.last_position_check = 0
        self.last_health_check = 0
        self.last_state_save = 0
        self.calendar = TradingCalendar()

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.is_running = True
        self.status_update.emit("Starting trading system...")
        self.log_message.emit("Initializing ticker database...")
        
        try:
            # Start log timer in main thread context
            self.log_timer.start(100)
            
            # Initialize ticker scanner
            self.ticker_scanner.start()
            self.ticker_scanner.initial_refresh_complete.wait()
            num_tickers = len(self.ticker_scanner.get_current_tickers_list())
            self.log_message.emit(f"Ticker refresh complete: {num_tickers} symbols")
            
            # Initialize trading system
            self.trading_system = QuantTradingSystem(self.ticker_scanner)
            loop.run_until_complete(self.trading_system.async_init())
            self.log_message.emit("System initialized")
            
            # Main trading loop
            while self.is_running:
                current_time = time.time()
                
                # 1. Position monitoring (highest priority)
                if current_time - self.last_position_check > config.POSITION_MONITOR_INTERVAL:
                    loop.run_until_complete(self.trading_system.monitor_positions())
                    self.last_position_check = current_time
                    self.position_update.emit(self.trading_system.open_positions)
                
                # 2. Data refresh BEFORE scanning - only if needed
                if self.trading_system.should_refresh_market_data():
                    self.log_message.emit("Starting pre-scan data refresh")
                    loop.run_until_complete(self.trading_system.refresh_market_data())
                    self.data_refreshed.emit()
                
                # 3. Scheduled scanning
                scan_interval = self.trading_system.get_scan_interval()
                if current_time - self.last_scan_time > scan_interval:
                    self.status_update.emit("Starting scan...")
                    self.log_message.emit(f"Scanning for opportunities (interval: {scan_interval}s)...")
                    results, failure_summary = loop.run_until_complete(self.trading_system.scan_tickers_async())
                    
                    # Emit failure summary for UI
                    self.failure_summary.emit(failure_summary)
                    
                    if results is not None and not results.empty:
                        self.scan_completed.emit(results)
                        self.status_update.emit(f"Found {len(results)} opportunities")
                        filename = self.trading_system.save_results(results)
                        if filename:
                            self.results_saved.emit(filename)
                    else:
                        self.status_update.emit("No opportunities found")
                    
                    self.last_scan_time = current_time
                    next_scan = datetime.fromtimestamp(
                        self.last_scan_time + scan_interval).strftime('%H:%M:%S')
                    self.status_update.emit(f"Next scan at {next_scan}")
                    self.scan_finished.emit()
                
                # 4. Health checks
                if current_time - self.last_health_check > config.HEALTH_CHECK_INTERVAL:
                    self.trading_system.run_health_check()
                    self.last_health_check = current_time
                
                # 5. State persistence
                self.trading_system.save_system_state()
                
                # Adaptive sleep
                sleep_time = min(
                    config.POSITION_MONITOR_INTERVAL,
                    scan_interval - (time.time() - self.last_scan_time)
                )
                time.sleep(max(0.1, sleep_time))
            
        except Exception as e:
            self.log_message.emit(f"CRITICAL ERROR: {str(e)}\n{traceback.format_exc()}")
            self.trading_system.consecutive_errors += 1
        finally:
            self.status_update.emit("Shutting down...")
            self.log_message.emit("Cleaning up resources...")
            
            # Clean up asyncio tasks properly
            if self.trading_system:
                try:
                    loop.run_until_complete(self.trading_system.async_close())
                except Exception as e:
                    logger.error(f"Error closing trading system: {e}")
            
            self.ticker_scanner.stop()
            
            # Save final state
            self.trading_system.save_system_state()
            
            # Cancel all pending tasks
            tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
            if tasks:
                for task in tasks:
                    task.cancel()
                # Wait for tasks to handle cancellation
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            
            loop.close()
            self.is_running = False
            self.status_update.emit("Trading system stopped")
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

class PositionModel(QAbstractTableModel):
    def __init__(self, positions):
        super().__init__()
        self.positions = positions or []
        self.columns = ["Symbol", "Entry Price", "Current Price", "Stop Loss", "Profit Target", "Quantity", "P&L"]
        
    def rowCount(self, parent=None):
        return len(self.positions)
    
    def columnCount(self, parent=None):
        return len(self.columns)
    
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
            
        row = index.row()
        col = index.column()
        position = self.positions[row]
        
        if role == Qt.DisplayRole:
            if col == 0:  # Symbol
                return position['symbol']
            elif col == 1:  # Entry Price
                return f"{position['entry_price']:.2f}"
            elif col == 2:  # Current Price (placeholder)
                return "150.00"  # In reality, fetch from real-time feed
            elif col == 3:  # Stop Loss
                return f"{position['stop_loss']:.2f}"
            elif col == 4:  # Profit Target
                return f"{position['profit_target']:.2f}"
            elif col == 5:  # Quantity
                return str(position['quantity'])
            elif col == 6:  # P&L
                # Placeholder calculation
                current_price = 150.00
                return f"{(current_price - position['entry_price']) * position['quantity']:.2f}"
        
        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter
            
        if role == Qt.BackgroundRole:
            # Highlight positions near stop loss
            current_price = 150.00  # Placeholder
            if current_price < position['entry_price'] and current_price < position['stop_loss'] * 1.05:
                return QColor(255, 200, 200)  # Light red
            
        return None
    
    def headerData(self, section, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.columns[section]
        return None

class ScannerUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quant Trading System")
        self.setGeometry(100, 100, 1400, 900)
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
        
        self.start_btn = QPushButton("Start Trading")
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.start_btn.clicked.connect(self.start_scanner)
        self.start_btn.setFixedHeight(40)
        
        self.stop_btn = QPushButton("Stop Trading")
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self.stop_btn.clicked.connect(self.stop_scanner)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setFixedHeight(40)
        
        self.refresh_btn = QPushButton("Force Refresh")
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
        
        # Stats panel
        stats_layout = QHBoxLayout()
        
        self.account_label = QLabel(f"Account Value: ${config.ACCOUNT_VALUE:,.2f}")
        self.account_label.setStyleSheet("font-weight: bold; color: #2E7D32;")
        
        self.position_label = QLabel("Open Positions: 0")
        self.position_label.setStyleSheet("font-weight: bold; color: #1565C0;")
        
        self.market_label = QLabel("Market: Closed")
        self.market_label.setStyleSheet("font-weight: bold;")
        
        stats_layout.addWidget(self.account_label)
        stats_layout.addWidget(self.position_label)
        stats_layout.addWidget(self.market_label)
        stats_layout.addStretch()
        
        # Tab widget
        self.tab_widget = QTabWidget()
        
        # Dashboard tab
        dashboard_tab = QWidget()
        dashboard_layout = QVBoxLayout(dashboard_tab)
        
        # Create a splitter for dashboard sections
        splitter = QSplitter(Qt.Vertical)
        
        # Positions panel
        positions_widget = QWidget()
        positions_layout = QVBoxLayout(positions_widget)
        positions_layout.addWidget(QLabel("Current Positions:"))
        
        self.positions_table = QTableView()
        self.positions_table.setSortingEnabled(True)
        self.positions_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.positions_table.setAlternatingRowColors(True)
        positions_layout.addWidget(self.positions_table)
        splitter.addWidget(positions_widget)
        
        # Results panel
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.addWidget(QLabel("Latest Opportunities:"))
        
        self.results_table = QTableView()
        self.results_table.setSortingEnabled(True)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QTableView.SelectRows)
        self.results_table.setSelectionMode(QTableView.SingleSelection)
        results_layout.addWidget(self.results_table)
        splitter.addWidget(results_widget)
        
        # Failure summary panel
        failure_widget = QWidget()
        failure_layout = QVBoxLayout(failure_widget)
        failure_layout.addWidget(QLabel("Breakout Failure Summary:"))
        
        self.failure_table = QTableWidget()
        self.failure_table.setColumnCount(2)
        self.failure_table.setHorizontalHeaderLabels(["Failure Reason", "Count"])
        self.failure_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.failure_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.failure_table.setSelectionMode(QAbstractItemView.NoSelection)
        failure_layout.addWidget(self.failure_table)
        splitter.addWidget(failure_widget)
        
        # Set initial sizes
        splitter.setSizes([300, 400, 200])
        dashboard_layout.addWidget(splitter)
        
        # Logs tab
        logs_tab = QWidget()
        logs_layout = QVBoxLayout(logs_tab)
        
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Courier New", 10))
        self.log_view.setStyleSheet("background-color: #f0f0f0;")
        
        logs_layout.addWidget(QLabel("Trading Logs:"))
        logs_layout.addWidget(self.log_view)
        
        # Add tabs
        self.tab_widget.addTab(dashboard_tab, "Dashboard")
        self.tab_widget.addTab(logs_tab, "Logs")
        
        # Add widgets to main layout
        main_layout.addLayout(control_layout)
        main_layout.addLayout(stats_layout)
        main_layout.addWidget(self.tab_widget)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Timer for stats updates
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(5000)  # Update every 5 seconds
        
        # Market status timer
        self.market_timer = QTimer()
        self.market_timer.timeout.connect(self.update_market_status)
        self.market_timer.start(60000)  # Update every minute
        
        # Initial log message
        self.update_log("Trading system started. Click 'Start Trading' to begin.")
        self.update_log(f"Using Polygon API key: {config.POLYGON_API_KEY[:6]}...")
        self.update_market_status()

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
        
        # Create scanner thread only when starting
        self.scanner_thread = ScannerThread()
        self.scanner_thread.scan_completed.connect(self.update_results)
        self.scanner_thread.status_update.connect(self.update_status)
        self.scanner_thread.scan_finished.connect(self.scan_finished)
        self.scanner_thread.results_saved.connect(self.results_saved)
        self.scanner_thread.log_message.connect(self.update_log)
        self.scanner_thread.thread_finished.connect(self.thread_finished)
        self.scanner_thread.position_update.connect(self.update_positions)
        self.scanner_thread.failure_summary.connect(self.update_failure_summary)
        self.scanner_thread.data_refreshed.connect(self.data_refreshed)
        
        self.scanner_thread.start()
        self.status_bar.showMessage("Trading system started")

    def stop_scanner(self):
        if self.scanner_thread and self.scanner_thread.isRunning():
            self.stop_btn.setEnabled(False)
            self.scanner_thread.stop()
            self.status_bar.showMessage("Stopping trading system...")

    def force_refresh(self):
        if self.scanner_thread and self.scanner_thread.isRunning():
            # Access the ticker scanner through the thread
            self.scanner_thread.ticker_scanner._refresh_all_tickers()
            self.update_log("Forced ticker refresh initiated")
            self.status_bar.showMessage("Refreshing ticker data...")
        else:
            self.update_log("System not running - cannot refresh")
        
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
        self.update_log(f"Displaying {len(results_df)} opportunities")

    def update_positions(self, positions):
        self.position_label.setText(f"Open Positions: {len(positions)}")
        model = PositionModel(positions)
        self.positions_table.setModel(model)

    def update_failure_summary(self, failure_data):
        """Update the failure summary table with scan statistics"""
        self.failure_table.setRowCount(0)  # Clear existing rows
        
        if not failure_data:
            return
            
        # Sort by count descending
        sorted_failures = sorted(failure_data.items(), key=lambda x: x[1], reverse=True)
        
        for reason, count in sorted_failures:
            row = self.failure_table.rowCount()
            self.failure_table.insertRow(row)
            
            reason_item = QTableWidgetItem(reason)
            count_item = QTableWidgetItem(str(count))
            
            # Apply formatting
            reason_item.setFlags(reason_item.flags() ^ Qt.ItemIsEditable)
            count_item.setFlags(count_item.flags() ^ Qt.ItemIsEditable)
            count_item.setTextAlignment(Qt.AlignCenter)
            
            # Add to table
            self.failure_table.setItem(row, 0, reason_item)
            self.failure_table.setItem(row, 1, count_item)

    def data_refreshed(self):
        """Handle data refresh completion"""
        self.status_bar.showMessage("Market data refreshed")
        self.update_log("Market data refresh completed")

    def update_status(self, message):
        self.status_label.setText(f"Status: {message}")

    def scan_finished(self):
        self.refresh_btn.setEnabled(True)
        self.status_bar.showMessage("Scan completed")

    def results_saved(self, filename):
        self.status_bar.showMessage(f"Results saved to {filename}")

    def thread_finished(self):
        """Handle thread completion"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_bar.showMessage("Trading system stopped")

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
            "<b>Quantitative Trading System</b><br><br>"
            "Professional algorithmic trading system for 24/7 market operation<br><br>"
            "Version: 3.0 (Enterprise Edition)<br>"
            "Data Provider: Polygon.io<br><br>"
            "Features:<br>"
            "- Market-aware scheduling (pre/market/post/overnight)<br>"
            "- Real-time position monitoring with dynamic stop loss<br>"
            "- Adaptive scanning based on market conditions<br>"
            "- Robust error handling and circuit breakers<br>"
            "- Daily performance reports<br>"
            "- Position sizing with risk management<br><br>"
            "© 2023 Quantitative Trading System"
        )
        QMessageBox.about(self, "About Trading System", about_text)

    def update_stats(self):
        """Update statistics periodically"""
        if self.scanner_thread and self.scanner_thread.isRunning() and hasattr(self.scanner_thread, 'ticker_scanner'):
            tickers = self.scanner_thread.ticker_scanner.get_current_tickers_list()
            last_refresh = datetime.fromtimestamp(
                self.scanner_thread.ticker_scanner.last_refresh_time
            ).strftime('%Y-%m-%d %H:%M') if self.scanner_thread.ticker_scanner.last_refresh_time > 0 else "Never"
            
            # In a real system, you'd update account value from the trading system
            # self.account_label.setText(f"Account Value: ${self.scanner_thread.trading_system.account_value:,.2f}")

    def update_market_status(self):
        """Update market status display"""
        calendar = TradingCalendar()
        phase = calendar.get_market_phase()
        
        if phase == "market_hours":
            self.market_label.setText("Market: OPEN")
            self.market_label.setStyleSheet("font-weight: bold; color: #2E7D32;")
        elif phase == "closed":
            self.market_label.setText("Market: CLOSED")
            self.market_label.setStyleSheet("font-weight: bold; color: #B71C1C;")
        else:
            self.market_label.setText(f"Market: {phase.upper()}")
            self.market_label.setStyleSheet("font-weight: bold; color: #FF8F00;")

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