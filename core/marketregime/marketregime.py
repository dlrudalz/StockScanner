import numpy as np
import pandas as pd
import asyncio
import aiohttp
import time
import os
import logging
import json
from datetime import datetime, timedelta
from urllib.parse import urlencode
from threading import Lock, Event, RLock
import sys
import signal
from tzlocal import get_localzone
import sqlite3
import contextlib
from typing import List, Dict, Optional, Any, Tuple
import argparse
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ======================== CONFIGURATION ======================== #
class Config:
    # API Configuration
    POLYGON_API_KEY = "ld1Poa63U6t4Y2MwOCA2JeKQyHVrmyg8"
    
    # Scanner Configuration - Using ONLY Nasdaq Composite
    COMPOSITE_INDICES = ["^IXIC"]  # NASDAQ Composite only
    MAX_CONCURRENT_REQUESTS = 100
    RATE_LIMIT_DELAY = 0.02
    SCAN_TIME = "08:30"
    
    # Database Configuration
    DATABASE_PATH_TICKER = r"C:\Users\kyung\StockScanner\core\ticker_data.db"
    DATABASE_PATH_REGIME = r"C:\Users\kyung\StockScanner\core\market_regime_data.db"
    
    # Market Regime Configuration
    REGIME_SCAN_INTERVAL = 3600  # 1 hour in seconds
    HISTORICAL_DATA_DAYS = 365  # 1 year of historical data
    HMM_N_COMPONENTS = 3  # Number of market regimes (bull, bear, sideways)
    HMM_N_ITER = 100  # Number of HMM iterations
    HMM_COVARIANCE_TYPE = "diag"  # Covariance type
    
    # Enhanced Model Configuration
    USE_ENSEMBLE_MODEL = True  # Use ensemble of HMM and GMM
    MODEL_VERSION = "enhanced_v3.0"  # Model version identifier
    USE_ANOMALY_DETECTION = True  # Detect anomalous market conditions
    N_CLUSTERS = 4  # Allow for more nuanced regime detection
    PCA_COMPONENTS = 10  # Reduce dimensionality for better clustering
    ROLLING_TRAINING_WINDOW = 252  # Use 1 year of data for training (approx 252 trading days)
    
    # Logging Configuration
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Initialize configuration
config = Config()

# ======================== LOGGING SETUP ======================== #
def setup_logging():
    """Configure logging with file and console handlers"""
    os.makedirs("logs", exist_ok=True)
    
    logger = logging.getLogger("MarketRegimeScanner")
    logger.setLevel(config.LOG_LEVEL)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter(config.LOG_FORMAT)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/market_regime_scanner_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(config.LOG_LEVEL)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(config.LOG_LEVEL)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# ======================== DATABASE MANAGER ======================== #
class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        # Ensure the directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
        
    def _init_database(self):
        """Initialize database with required tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create tickers table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tickers (
                    ticker TEXT PRIMARY KEY,
                    name TEXT,
                    primary_exchange TEXT,
                    last_updated_utc TEXT,
                    type TEXT,
                    market TEXT,
                    locale TEXT,
                    currency_name TEXT,
                    active INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create historical_tickers table to track changes over time
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS historical_tickers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    name TEXT,
                    primary_exchange TEXT,
                    last_updated_utc TEXT,
                    type TEXT,
                    market TEXT,
                    locale TEXT,
                    currency_name TEXT,
                    active INTEGER,
                    change_type TEXT,  -- 'added', 'removed', 'updated'
                    change_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tickers_exchange ON tickers(primary_exchange)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tickers_active ON tickers(active)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_historical_tickers_date ON historical_tickers(change_date)')
            
            conn.commit()
            
    @contextlib.contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable row factory for dict-like access
        try:
            yield conn
        finally:
            conn.close()
            
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute a SELECT query and return results as dictionaries"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
            
    def execute_write(self, query: str, params: tuple = ()) -> int:
        """Execute a write query and return number of affected rows"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount
            
    def upsert_tickers(self, tickers: List[Dict]) -> Tuple[int, int]:
        """Insert or update tickers in the database"""
        inserted = 0
        updated = 0
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            for ticker_data in tickers:
                # Check if ticker exists
                cursor.execute(
                    "SELECT ticker FROM tickers WHERE ticker = ?", 
                    (ticker_data['ticker'],)
                )
                exists = cursor.fetchone()
                
                if exists:
                    # Update existing ticker
                    cursor.execute('''
                        UPDATE tickers 
                        SET name = ?, primary_exchange = ?, last_updated_utc = ?, 
                            type = ?, market = ?, locale = ?, currency_name = ?, 
                            active = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE ticker = ?
                    ''', (
                        ticker_data.get('name'),
                        ticker_data.get('primary_exchange'),
                        ticker_data.get('last_updated_utc'),
                        ticker_data.get('type'),
                        ticker_data.get('market'),
                        ticker_data.get('locale'),
                        ticker_data.get('currency_name'),
                        1,  # active
                        ticker_data['ticker']
                    ))
                    
                    if cursor.rowcount > 0:
                        updated += 1
                        
                        # Record in historical table
                        cursor.execute('''
                            INSERT INTO historical_tickers 
                            (ticker, name, primary_exchange, last_updated_utc, 
                             type, market, locale, currency_name, active, change_type)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            ticker_data['ticker'],
                            ticker_data.get('name'),
                            ticker_data.get('primary_exchange'),
                            ticker_data.get('last_updated_utc'),
                            ticker_data.get('type'),
                            ticker_data.get('market'),
                            ticker_data.get('locale'),
                            ticker_data.get('currency_name'),
                            1,
                            'updated'
                        ))
                else:
                    # Insert new ticker
                    cursor.execute('''
                        INSERT INTO tickers 
                        (ticker, name, primary_exchange, last_updated_utc, 
                         type, market, locale, currency_name, active)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        ticker_data['ticker'],
                        ticker_data.get('name'),
                        ticker_data.get('primary_exchange'),
                        ticker_data.get('last_updated_utc'),
                        ticker_data.get('type'),
                        ticker_data.get('market'),
                        ticker_data.get('locale'),
                        ticker_data.get('currency_name'),
                        1  # active
                    ))
                    
                    if cursor.rowcount > 0:
                        inserted += 1
                        
                        # Record in historical table
                        cursor.execute('''
                            INSERT INTO historical_tickers 
                            (ticker, name, primary_exchange, last_updated_utc, 
                             type, market, locale, currency_name, active, change_type)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            ticker_data['ticker'],
                            ticker_data.get('name'),
                            ticker_data.get('primary_exchange'),
                            ticker_data.get('last_updated_utc'),
                            ticker_data.get('type'),
                            ticker_data.get('market'),
                            ticker_data.get('locale'),
                            ticker_data.get('currency_name'),
                            1,
                            'added'
                        ))
            
            conn.commit()
            
        return inserted, updated
        
    def mark_tickers_inactive(self, tickers: List[str]) -> int:
        """Mark tickers as inactive (removed from exchange)"""
        marked = 0
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            for ticker in tickers:
                # Get current data before marking inactive
                cursor.execute(
                    "SELECT * FROM tickers WHERE ticker = ?", 
                    (ticker,)
                )
                row = cursor.fetchone()
                if row:
                    # Convert sqlite3.Row to dictionary
                    current_data = dict(row)
                    # Mark as inactive
                    cursor.execute(
                        "UPDATE tickers SET active = 0, updated_at = CURRENT_TIMESTAMP WHERE ticker = ?",
                        (ticker,)
                    )
                    
                    if cursor.rowcount > 0:
                        marked += 1
                        
                        # Record in historical table
                        cursor.execute('''
                            INSERT INTO historical_tickers 
                            (ticker, name, primary_exchange, last_updated_utc, 
                             type, market, locale, currency_name, active, change_type)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            current_data['ticker'],
                            current_data['name'],
                            current_data['primary_exchange'],
                            current_data['last_updated_utc'],
                            current_data['type'],
                            current_data['market'],
                            current_data['locale'],
                            current_data.get('currency_name', ''),
                            0,
                            'removed'
                        ))
            
            conn.commit()
            
        return marked
        
    def get_all_active_tickers(self) -> List[Dict]:
        """Get all active tickers from the database"""
        return self.execute_query(
            "SELECT * FROM tickers WHERE active = 1 ORDER BY ticker"
        )
        
    def get_ticker_details(self, ticker: str) -> Optional[Dict]:
        """Get details for a specific ticker"""
        result = self.execute_query(
            "SELECT * FROM tickers WHERE ticker = ?", 
            (ticker,)
        )
        return result[0] if result else None
        
    def search_tickers(self, search_term: str, limit: int = 50) -> List[Dict]:
        """Search tickers by name or symbol"""
        return self.execute_query(
            "SELECT * FROM tickers WHERE (ticker LIKE ? OR name LIKE ?) AND active = 1 ORDER BY ticker LIMIT ?",
            (f"%{search_term}%", f"%{search_term}%", limit)
        )
        
    def get_ticker_history(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Get historical changes for a ticker"""
        return self.execute_query(
            "SELECT * FROM historical_tickers WHERE ticker = ? ORDER by change_date DESC LIMIT ?",
            (ticker, limit)
        )
        
    def update_metadata(self, key: str, value: Any) -> None:
        """Update metadata key-value pair"""
        self.execute_write(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, json.dumps(value) if isinstance(value, (list, dict)) else str(value))
        )
        
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key"""
        result = self.execute_query(
            "SELECT value FROM metadata WHERE key = ?",
            (key,)
        )
        
        if result:
            value = result[0]['value']
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        return default

# ======================== MARKET REGIME DATABASE ======================== #
class MarketRegimeDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        # Ensure the directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
        
    def _init_database(self):
        """Initialize database with required tables for market regime data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Drop existing tables if they exist (for development)
            cursor.execute('DROP TABLE IF EXISTS market_regimes')
            cursor.execute('DROP TABLE IF EXISTS regime_statistics')
            
            # Create market_regimes table with confidence column
            cursor.execute('''
                CREATE TABLE market_regimes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    regime INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    features TEXT,
                    model_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create regime_statistics table
            cursor.execute('''
                CREATE TABLE regime_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    regime INTEGER NOT NULL,
                    start_date DATETIME NOT NULL,
                    end_date DATETIME,
                    duration_days INTEGER,
                    return_pct REAL,
                    volatility REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX idx_regimes_timestamp ON market_regimes(timestamp)')
            cursor.execute('CREATE INDEX idx_regimes_regime ON market_regimes(regime)')
            cursor.execute('CREATE INDEX idx_statistics_regime ON regime_statistics(regime)')
            cursor.execute('CREATE INDEX idx_statistics_date ON regime_statistics(start_date)')
            
            conn.commit()
            
    @contextlib.contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable row factory for dict-like access
        try:
            yield conn
        finally:
            conn.close()
            
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute a SELECT query and return results as dictionaries"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
            
    def execute_write(self, query: str, params: tuple = ()) -> int:
        """Execute a write query and return number of affected rows"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount
            
    def save_market_regime(self, timestamp: datetime, regime: int, confidence: float, 
                          features: Dict, model_version: str) -> int:
        """Save market regime prediction to database"""
        return self.execute_write(
            '''INSERT INTO market_regimes 
               (timestamp, regime, confidence, features, model_version) 
               VALUES (?, ?, ?, ?, ?)''',
            (timestamp, regime, confidence, json.dumps(features), model_version)
        )
        
    def get_latest_regime(self) -> Optional[Dict]:
        """Get the latest market regime from database"""
        result = self.execute_query(
            "SELECT * FROM market_regimes ORDER BY timestamp DESC LIMIT 1"
        )
        return result[0] if result else None
        
    def get_regimes_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get market regimes within a date range"""
        return self.execute_query(
            "SELECT * FROM market_regimes WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
            (start_date, end_date)
        )
        
    def save_regime_statistics(self, regime: int, start_date: datetime, end_date: datetime, 
                              duration_days: int, return_pct: float, volatility: float) -> int:
        """Save regime statistics to database"""
        return self.execute_write(
            '''INSERT INTO regime_statistics 
               (regime, start_date, end_date, duration_days, return_pct, volatility) 
               VALUES (?, ?, ?, ?, ?, ?)''',
            (regime, start_date, end_date, duration_days, return_pct, volatility)
        )
        
    def get_regime_statistics(self, regime: Optional[int] = None) -> List[Dict]:
        """Get regime statistics, optionally filtered by regime"""
        if regime is not None:
            return self.execute_query(
                "SELECT * FROM regime_statistics WHERE regime = ? ORDER BY start_date",
                (regime,)
            )
        else:
            return self.execute_query(
                "SELECT * FROM regime_statistics ORDER BY start_date"
            )

# ======================== TICKER SCANNER ======================== #
class PolygonTickerScanner:
    def __init__(self):
        self.api_key = config.POLYGON_API_KEY
        self.base_url = "https://api.polygon.io/v3/reference/tickers"
        # Use ONLY Nasdaq Composite
        self.composite_indices = config.COMPOSITE_INDICES
        self.active = False
        self.cache_lock = RLock()
        self.refresh_lock = Lock()
        self.known_missing_tickers = set()
        self.initial_refresh_complete = Event()
        self.last_refresh_time = 0
        self.ticker_cache = pd.DataFrame(columns=[
            "ticker", "name", "primary_exchange", "last_updated_utc", "type", "market", "locale"
        ])
        self.current_tickers_set = set()
        self.local_tz = get_localzone()
        self.semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
        self.db = DatabaseManager(config.DATABASE_PATH_TICKER)
        self.shutdown_requested = False
        logger.info(f"Using local timezone: {self.local_tz}")
        logger.info(f"Database path: {config.DATABASE_PATH_TICKER}")
        logger.info(f"Using composite indices: {', '.join(self.composite_indices)}")
        
    def _init_cache(self):
        """Initialize or load ticker cache from database"""
        self.last_refresh_time = self.db.get_metadata('last_refresh_time', 0)
        
        # Load active tickers from database
        db_tickers = self.db.get_all_active_tickers()
        
        if db_tickers:
            self.ticker_cache = pd.DataFrame(db_tickers)
            logger.info(f"Loaded {len(self.ticker_cache)} tickers from database")
        else:
            self.ticker_cache = pd.DataFrame(columns=[
                "ticker", "name", "primary_exchange", "last_updated_utc", "type", "market", "locale"
            ])
            logger.info("No tickers found in database")
        
        self.current_tickers_set = set(self.ticker_cache['ticker'].tolist()) if not self.ticker_cache.empty else set()
        
        # Load known missing tickers from database
        self.known_missing_tickers = set(self.db.get_metadata('known_missing_tickers', []))
        
        self.initial_refresh_complete.set()

    async def _call_polygon_api(self, session, url):
        """Make API call with retry logic and rate limiting"""
        # Check for shutdown before making the request
        if self.shutdown_requested:
            return None
            
        async with self.semaphore:
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
            except asyncio.TimeoutError:
                logger.warning(f"Timeout for URL: {url}")
                return None
            except asyncio.CancelledError:
                logger.info("API request cancelled")
                return None
            except Exception as e:
                logger.error(f"API request exception: {e}")
                return None

    async def _fetch_composite_tickers(self, session, composite_index):
        """Fetch all tickers for a specific composite index"""
        logger.info(f"Fetching tickers for composite index {composite_index}")
        all_results = []
        next_url = None
        
        # Use current date
        date_param = datetime.now().strftime("%Y-%m-%d")
            
        # Different API endpoint for composite indices
        if composite_index == "^IXIC":  # NASDAQ Composite
            exchange = "XNAS"
        else:
            logger.error(f"Unknown composite index: {composite_index}")
            return []
            
        params = {
            "market": "stocks",
            "exchange": exchange,
            "active": "true",
            "limit": 1000,  # Maximum allowed by Polygon
            "apiKey": self.api_key,
            "date": date_param
        }
        
        # Initial URL construction
        url = f"{self.base_url}?{urlencode(params)}"
        page_count = 0
        
        while url and not self.shutdown_requested:
            data = await self._call_polygon_api(session, url)
            if not data or self.shutdown_requested:
                break
                
            results = data.get("results", [])
            # Filter for common stocks only and add composite index info
            stock_results = [
                {**r, "composite_index": composite_index} 
                for r in results 
                if r.get('type', '').upper() == 'CS'
            ]
            all_results.extend(stock_results)
            
            next_url = data.get("next_url", None)
            url = f"{next_url}&apiKey={self.api_key}" if next_url else None
            page_count += 1
            
            # Minimal delay for premium API access
            await asyncio.sleep(config.RATE_LIMIT_DELAY)
        
        if self.shutdown_requested:
            logger.info(f"Shutdown requested, aborting {composite_index} fetch")
            return []
            
        logger.info(f"Completed {composite_index}: {len(all_results)} stocks across {page_count} pages")
        return all_results

    async def _refresh_all_tickers_async(self):
        """Refresh all tickers with parallel composite index processing"""
        start_time = time.time()
        
        logger.info("Starting full ticker refresh")
        
        # Check for shutdown before starting
        if self.shutdown_requested:
            logger.info("Shutdown requested, aborting refresh")
            return False
            
        async with aiohttp.ClientSession() as session:
            # Fetch all composite indices in parallel
            tasks = [self._fetch_composite_tickers(session, idx) for idx in self.composite_indices]
            composite_results = await asyncio.gather(*tasks)
            
            # Check for shutdown after fetching
            if self.shutdown_requested:
                logger.info("Shutdown requested during data processing")
                return False
                
            # Flatten results
            all_results = []
            for results in composite_results:
                if results:
                    all_results.extend(results)
        
        if not all_results:
            logger.warning("Refresh fetched no results")
            return False
            
        # Create DataFrame with only necessary columns
        new_df = pd.DataFrame(all_results)[["ticker", "name", "primary_exchange", "last_updated_utc", "type", "market", "locale", "currency_name"]]
        new_tickers = set(new_df['ticker'].tolist())
        
        with self.cache_lock:
            # Original logic for live mode
            old_tickers = set(self.current_tickers_set)
            added = new_tickers - old_tickers
            removed = old_tickers - new_tickers
            
            # Convert DataFrame to list of dictionaries for database storage
            tickers_data = new_df.to_dict('records')
            
            # Update database
            inserted, updated = self.db.upsert_tickers(tickers_data)
            
            # Mark removed tickers as inactive
            if removed:
                marked_inactive = self.db.mark_tickers_inactive(list(removed))
                logger.info(f"Marked {marked_inactive} tickers as inactive")
            
            # Update in-memory cache
            self.ticker_cache = new_df
            self.current_tickers_set = new_tickers
            
            # Update known missing tickers
            rediscovered = added & self.known_missing_tickers
            if rediscovered:
                self.known_missing_tickers -= rediscovered
                self.db.update_metadata('known_missing_tickers', list(self.known_missing_tickers))
            
        self.last_refresh_time = time.time()
        
        self.db.update_metadata('last_refresh_time', self.last_refresh_time)
        
        elapsed = time.time() - start_time
        logger.info(f"Ticker refresh completed in {elapsed:.2f}s")
        
        logger.info(f"Total: {len(new_df)} | Added: {len(added)} | Removed: {len(removed)}")
        logger.info(f"Database: {inserted} inserted, {updated} updated")
            
        return True

    async def refresh_all_tickers(self):
        """Public async method to refresh tickers"""
        with self.refresh_lock:
            return await self._refresh_all_tickers_async()

    def start(self):
        if not self.active:
            self.active = True
            self.shutdown_requested = False
            self._init_cache()
            self.initial_refresh_complete.set()
            logger.info("Ticker scanner started")

    def stop(self):
        self.active = False
        self.shutdown_requested = True
        logger.info("Ticker scanner stopped")
        
    async def shutdown(self):
        """Cleanup resources"""
        self.stop()
        logger.info("Ticker scanner shutdown complete")

    def get_current_tickers_list(self):
        with self.cache_lock:
            return self.ticker_cache['ticker'].tolist()

    def get_ticker_details(self, ticker):
        """Get details for a specific ticker from cache"""
        with self.cache_lock:
            result = self.ticker_cache[self.ticker_cache['ticker'] == ticker]
            return result.to_dict('records')[0] if not result.empty else None
            
    def search_tickers_db(self, search_term: str, limit: int = 50) -> List[Dict]:
        """Search tickers in database by name or symbol"""
        return self.db.search_tickers(search_term, limit)
        
    def get_ticker_history_db(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Get historical changes for a ticker from database"""
        return self.db.get_ticker_history(ticker, limit)

# ======================== MARKET REGIME SCANNER ======================== #
class MarketRegimeScanner:
    def __init__(self, ticker_scanner: PolygonTickerScanner):
        self.ticker_scanner = ticker_scanner
        self.api_key = config.POLYGON_API_KEY
        self.base_url = "https://api.polygon.io/v2/aggs/ticker"
        self.active = False
        self.shutdown_requested = False
        self.local_tz = get_localzone()
        self.semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
        self.regime_db = MarketRegimeDatabase(config.DATABASE_PATH_REGIME)
        self.hmm_model = None
        self.gmm_model = None
        self.rf_model = None
        self.anomaly_detector = None
        self.pca = None
        self.kmeans = None
        self.training_data = None
        self.scaler = StandardScaler()
        self.model_version = config.MODEL_VERSION
        
        # Focus on Nasdaq Composite for regime analysis
        self.market_indices = {
            "^IXIC": "COMP",    # NASDAQ Composite (primary focus)
        }
        
        logger.info(f"Market regime database path: {config.DATABASE_PATH_REGIME}")
        
    async def _fetch_historical_data(self, session, ticker: str, days: int, end_date: datetime = None) -> Optional[pd.DataFrame]:
        """Fetch historical data for a ticker with optional end date for backtesting"""
        if end_date is None:
            end_date = datetime.now()
        
        start_date = end_date - timedelta(days=days)
        
        # Format dates for Polygon API
        end_date_str = end_date.strftime("%Y-%m-%d")
        start_date_str = start_date.strftime("%Y-%m-%d")
        
        # Use the correct symbol for Polygon API with I: prefix for indices
        polygon_symbol = self.market_indices.get(ticker, ticker)
        
        # Construct the API URL
        url = f"{self.base_url}/{polygon_symbol}/range/1/day/{start_date_str}/{end_date_str}?apiKey={self.api_key}"
        
        logger.debug(f"Fetching historical data from: {url}")
        
        async with self.semaphore:
            try:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('resultsCount', 0) > 0:
                            df = pd.DataFrame(data['results'])
                            df['t'] = pd.to_datetime(df['t'], unit='ms')
                            df.set_index('t', inplace=True)
                            df['ticker'] = ticker
                            return df
                        else:
                            logger.warning(f"No results for {ticker} (Polygon: {polygon_symbol}): {data}")
                            return None
                    elif response.status == 429:
                        retry_after = int(response.headers.get('Retry-After', 1))
                        logger.warning(f"Rate limit hit for {ticker}, retrying after {retry_after} seconds")
                        await asyncio.sleep(retry_after)
                        return await self._fetch_historical_data(session, ticker, days, end_date)
                    else:
                        error_text = await response.text()
                        logger.warning(f"Failed to fetch data for {ticker} (Polygon: {polygon_symbol}): {response.status} - {error_text}")
                        return None
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching data for {ticker}")
                return None
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
                return None
                
    async def fetch_market_data(self, days: int = config.HISTORICAL_DATA_DAYS) -> pd.DataFrame:
        """Fetch historical market data for all market indices"""
        logger.info(f"Fetching {days} days of market data for regime analysis")
        
        # Get the market indices
        tickers = list(self.market_indices.keys())
        
        all_data = []
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_historical_data(session, ticker, days) for ticker in tickers]
            results = await asyncio.gather(*tasks)
            
            for result in results:
                if result is not None and not result.empty:
                    all_data.append(result)
        
        if not all_data:
            logger.error("No market data fetched")
            return pd.DataFrame()
            
        # Combine all data
        combined_data = pd.concat(all_data)
        
        # Pivot to get close prices for each ticker
        close_prices = combined_data.pivot(columns='ticker', values='c')
        
        logger.info(f"Fetched market data with shape: {close_prices.shape}")
        return close_prices
        
    def calculate_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate more sophisticated features for regime detection"""
        features = pd.DataFrame(index=data.index)
        
        # Calculate returns
        for ticker in data.columns:
            returns = data[ticker].pct_change().dropna()
            features[f'{ticker}_return'] = returns
            
            # Advanced volatility measures
            volatility_20 = returns.rolling(window=20).std()
            volatility_50 = returns.rolling(window=50).std()
            features[f'{ticker}_volatility_ratio'] = volatility_20 / volatility_50
            
            # Advanced momentum indicators
            moving_avg_20 = data[ticker].rolling(window=20).mean()
            moving_avg_50 = data[ticker].rolling(window=50).mean()
            features[f'{ticker}_momentum_ratio'] = moving_avg_20 / moving_avg_50
            
            # Volatility clustering feature
            features[f'{ticker}_volatility_clustering'] = returns.rolling(window=20).apply(
                lambda x: x.autocorr(lag=1), raw=False
            )
            
            # Jump detection
            features[f'{ticker}_jumps'] = returns.rolling(window=20).apply(
                lambda x: np.sum(np.abs(x) > 2 * np.std(x)), raw=False
            )
            
            # Add RSI (Relative Strength Index)
            delta = data[ticker].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features[f'{ticker}_rsi'] = 100 - (100 / (1 + rs))
            
            # Add MACD
            exp12 = data[ticker].ewm(span=12, adjust=False).mean()
            exp26 = data[ticker].ewm(span=26, adjust=False).mean()
            features[f'{ticker}_macd'] = exp12 - exp26
            features[f'{ticker}_macd_signal'] = features[f'{ticker}_macd'].ewm(span=9, adjust=False).mean()
            
            # Add Bollinger Bands
            rolling_mean = data[ticker].rolling(window=20).mean()
            rolling_std = data[ticker].rolling(window=20).std()
            features[f'{ticker}_bollinger_upper'] = rolling_mean + (rolling_std * 2)
            features[f'{ticker}_bollinger_lower'] = rolling_mean - (rolling_std * 2)
            features[f'{ticker}_bollinger_pct'] = (data[ticker] - features[f'{ticker}_bollinger_lower']) / (
                features[f'{ticker}_bollinger_upper'] - features[f'{ticker}_bollinger_lower'])
        
        # Cross-asset correlations (rolling)
        returns_matrix = data.pct_change().dropna()
        rolling_corr = returns_matrix.rolling(window=20).corr()
        
        # Add correlation features
        for i, ticker1 in enumerate(data.columns):
            for j, ticker2 in enumerate(data.columns):
                if i < j:  # Avoid duplicates
                    corr_series = rolling_corr[ticker1].xs(ticker2, level=1)
                    features[f'corr_{ticker1}_{ticker2}'] = corr_series
        
        # Market breadth indicators
        advancers = (returns_matrix > 0).rolling(window=5).mean()
        features['market_breadth'] = advancers.mean(axis=1)
        
        # Economic regime indicators
        features['term_structure'] = data.get('^TNX', data.get('^IRX', pd.Series(0, index=data.index))) / 100  # Default to 0 if not available
        
        # Drop rows with NaN values
        features.dropna(inplace=True)
        
        logger.info(f"Calculated advanced features with shape: {features.shape}")
        return features
        
    def train_enhanced_ensemble(self, features: pd.DataFrame):
        """Train an enhanced ensemble of models for regime detection"""
        scaled_features = self.scaler.fit_transform(features)
        
        # Store training data for rolling updates
        self.training_data = scaled_features
        
        # Dimensionality reduction
        self.pca = PCA(n_components=config.PCA_COMPONENTS)
        pca_features = self.pca.fit_transform(scaled_features)
        
        # HMM Model
        self.hmm_model = hmm.GaussianHMM(
            n_components=config.HMM_N_COMPONENTS,
            covariance_type=config.HMM_COVARIANCE_TYPE,
            n_iter=config.HMM_N_ITER,
            random_state=42
        )
        self.hmm_model.fit(pca_features)
        
        # GMM Model
        self.gmm_model = GaussianMixture(
            n_components=config.HMM_N_COMPONENTS,
            covariance_type=config.HMM_COVARIANCE_TYPE,
            max_iter=config.HMM_N_ITER,
            random_state=42
        )
        self.gmm_model.fit(pca_features)
        
        # K-means clustering for additional perspective
        self.kmeans = KMeans(n_clusters=config.N_CLUSTERS, random_state=42)
        self.kmeans.fit(pca_features)
        
        # Random Forest classifier for feature importance
        # Create labels from consensus of models
        hmm_labels = self.hmm_model.predict(pca_features)
        gmm_labels = self.gmm_model.predict(pca_features)
        
        # Create consensus labels (mode of all models)
        consensus_labels = []
        for i in range(len(hmm_labels)):
            votes = [hmm_labels[i], gmm_labels[i], self.kmeans.labels_[i]]
            consensus_labels.append(max(set(votes), key=votes.count))
        
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(scaled_features, consensus_labels)
        
        # Anomaly detection for unusual market conditions
        self.anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
        self.anomaly_detector.fit(scaled_features)
        
        logger.info("Enhanced ensemble model training completed")
        logger.info(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
    def predict_enhanced_regime(self, features: pd.DataFrame) -> Tuple[int, float, Dict]:
        """Predict using enhanced ensemble approach"""
        scaled_features = self.scaler.transform(features)
        pca_features = self.pca.transform(scaled_features)
        
        # Get predictions from all models
        hmm_pred = self.hmm_model.predict(pca_features[-5:])  # Use last 5 days
        gmm_pred = self.gmm_model.predict(pca_features[-5:])
        kmeans_pred = self.kmeans.predict(pca_features[-5:])
        rf_pred = self.rf_model.predict(scaled_features[-5:])
        anomaly_score = self.anomaly_detector.score_samples(scaled_features[-5:])
        
        # Get probabilities for confidence calculation
        hmm_probs = self.hmm_model.predict_proba(pca_features[-5:])
        gmm_probs = self.gmm_model.predict_proba(pca_features[-5:])
        rf_probs = self.rf_model.predict_proba(scaled_features[-5:])
        
        # Use the most recent predictions
        recent_hmm = hmm_pred[-1]
        recent_gmm = gmm_pred[-1]
        recent_kmeans = kmeans_pred[-1]
        recent_rf = rf_pred[-1]
        
        # Calculate confidence based on model agreement and probabilities
        model_agreement = np.mean([
            recent_hmm == recent_gmm,
            recent_hmm == recent_kmeans,
            recent_hmm == recent_rf,
            recent_gmm == recent_kmeans,
            recent_gmm == recent_rf,
            recent_kmeans == recent_rf
        ])
        
        # Average the probabilities from models that support the final decision
        final_regime = recent_hmm  # Start with HMM as base
        regime_votes = [recent_hmm, recent_gmm, recent_kmeans, recent_rf]
        final_regime = max(set(regime_votes), key=regime_votes.count)
        
        supporting_probs = []
        if recent_hmm == final_regime:
            supporting_probs.append(np.max(hmm_probs, axis=1)[-1])
        if recent_gmm == final_regime:
            supporting_probs.append(np.max(gmm_probs, axis=1)[-1])
        if recent_rf == final_regime:
            supporting_probs.append(np.max(rf_probs, axis=1)[-1])
        
        confidence = np.mean(supporting_probs) if supporting_probs else 0.5
        confidence = confidence * (0.7 + 0.3 * model_agreement)  # Scale by agreement
        
        # Adjust confidence for anomalies
        anomaly_factor = 1.0 - min(1.0, max(0.0, (1.0 - np.mean(anomaly_score)) * 2))  # Map [-1, 1] to [0, 1]
        confidence = confidence * (0.8 + 0.2 * anomaly_factor)  # Reduce confidence in anomalous conditions
        
        # Prepare feature values for storage
        feature_values = {
            col: features[col].iloc[-1] for col in features.columns
        }
        feature_values['anomaly_score'] = np.mean(anomaly_score)
        feature_values['model_agreement'] = model_agreement
        
        return final_regime, confidence, feature_values
        
    def interpret_regime(self, regime: int, features: Dict) -> Dict:
        """Enhanced regime interpretation with contextual information"""
        base_interpretation = {
            0: {"label": "Bear Market", "color": "red", "description": "Declining prices, high volatility"},
            1: {"label": "Sideways Market", "color": "gray", "description": "Range-bound prices, moderate volatility"},
            2: {"label": "Bull Market", "color": "green", "description": "Rising prices, low to moderate volatility"},
            3: {"label": "High Volatility", "color": "orange", "description": "Extreme price movements in both directions"}
        }
        
        # Default to unknown if regime not in interpretation
        interpretation = base_interpretation.get(regime, {
            "label": f"Unknown Regime {regime}",
            "color": "purple",
            "description": "Unclassified market regime"
        })
        
        # Add contextual details based on features
        volatility = features.get('^IXIC_volatility_ratio', 1.0)
        momentum = features.get('^IXIC_momentum_ratio', 1.0)
        rsi = features.get('^IXIC_rsi', 50)
        anomaly_score = features.get('anomaly_score', 0)
        
        # Enhance description based on market conditions
        details = []
        
        if volatility > 1.5:
            details.append("High volatility environment")
        elif volatility < 0.8:
            details.append("Low volatility environment")
            
        if momentum > 1.05:
            details.append("Strong upward momentum")
        elif momentum < 0.95:
            details.append("Strong downward momentum")
            
        if rsi > 70:
            details.append("Overbought conditions")
        elif rsi < 30:
            details.append("Oversold conditions")
            
        if anomaly_score < -0.5:
            details.append("Anomalous market behavior detected")
            
        if details:
            interpretation["details"] = ", ".join(details)
        
        return interpretation
            
    async def scan_market_regime(self):
        """Perform a complete market regime scan"""
        if self.shutdown_requested:
            return False
            
        logger.info("Starting market regime scan")
        start_time = time.time()
        
        try:
            # Fetch market data
            market_data = await self.fetch_market_data()
            if market_data.empty:
                logger.error("No market data available for regime analysis")
                return False
                
            # Calculate features
            features = self.calculate_advanced_features(market_data)
            if features.empty:
                logger.error("No features calculated for regime analysis")
                return False
                
            # Train models if not already trained
            if self.hmm_model is None:
                self.train_enhanced_ensemble(features)
                
            # Predict current regime
            regime, confidence, feature_values = self.predict_enhanced_regime(features)
                
            interpretation = self.interpret_regime(regime, feature_values)
            regime_label = interpretation["label"]
            
            # Save to database
            timestamp = datetime.now()
            self.regime_db.save_market_regime(
                timestamp, regime, confidence, feature_values, self.model_version
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Market regime scan completed in {elapsed:.2f}s")
            logger.info(f"Current market regime: {regime_label} (confidence: {confidence:.2f})")
            logger.info(f"Details: {interpretation.get('details', 'No additional details')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during market regime scan: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    def start(self):
        if not self.active:
            self.active = True
            self.shutdown_requested = False
            logger.info("Market regime scanner started")
            
    def stop(self):
        self.active = False
        self.shutdown_requested = True
        logger.info("Market regime scanner stopped")
        
    async def shutdown(self):
        """Cleanup resources"""
        self.stop()
        logger.info("Market regime scanner shutdown complete")

# ======================== SCHEDULER ======================== #
async def run_scheduled_ticker_refresh(scanner):
    """Run immediate scan on startup and then daily at scheduled time"""
    # Run immediate scan on startup
    logger.info("Starting immediate ticker scan on startup")
    try:
        success = await scanner.refresh_all_tickers()
        if success:
            logger.info("Initial ticker scan completed successfully")
        else:
            logger.warning("Initial ticker scan encountered errors")
    except asyncio.CancelledError:
        logger.info("Initial ticker scan cancelled")
        return
    except Exception as e:
        logger.error(f"Error during initial ticker scan: {e}")
    
    # Continue with daily scans
    while scanner.active and not scanner.shutdown_requested:
        now = datetime.now(scanner.local_tz)
        
        # Calculate next run time (today at 8:30 AM)
        target_time = datetime.strptime(config.SCAN_TIME, "%H:%M").time()
        target_datetime = now.replace(
            hour=target_time.hour,
            minute=target_time.minute,
            second=0,
            microsecond=0
        )
        
        # If we already passed today's scheduled time, set for tomorrow
        if now > target_datetime:
            target_datetime += timedelta(days=1)
        
        sleep_seconds = (target_datetime - now).total_seconds()
        hours = sleep_seconds // 3600
        minutes = (sleep_seconds % 3600) // 60

        logger.info(f"Next ticker refresh scheduled at {target_datetime} ({hours} hours and {minutes} minutes from now)")
        
        # Wait until scheduled time, but check every second if we should stop
        while sleep_seconds > 0 and scanner.active and not scanner.shutdown_requested:
            try:
                # Sleep in small increments to be responsive to shutdown requests
                await asyncio.sleep(min(1, sleep_seconds))
                sleep_seconds -= 1
            except asyncio.CancelledError:
                logger.info("Sleep interrupted by shutdown")
                return
            
        if not scanner.active or scanner.shutdown_requested:
            break
            
        # Run the refresh
        logger.info("Starting scheduled ticker refresh")
        try:
            success = await scanner.refresh_all_tickers()
            if success:
                logger.info("Scheduled ticker refresh completed successfully")
            else:
                logger.warning("Scheduled ticker refresh encountered errors")
        except asyncio.CancelledError:
            logger.info("Ticker refresh cancelled")
            return
        except Exception as e:
            logger.error(f"Error during scheduled ticker refresh: {e}")

async def run_scheduled_regime_scan(regime_scanner, wait_for_ticker=True):
    """Run immediate regime scan on startup and then at scheduled intervals aligned with local clock"""
    # Wait for ticker scan to complete if requested
    if wait_for_ticker:
        logger.info("Waiting for initial ticker scan to complete before starting market regime scan")
        await asyncio.sleep(5)  # Give ticker scan a head start
    
    # Run immediate scan on startup
    logger.info("Starting immediate market regime scan")
    try:
        success = await regime_scanner.scan_market_regime()
        if success:
            logger.info("Initial market regime scan completed successfully")
        else:
            logger.warning("Initial market regime scan encountered errors")
    except asyncio.CancelledError:
        logger.info("Initial market regime scan cancelled")
        return
    except Exception as e:
        logger.error(f"Error during initial market regime scan: {e}")
    
    # Continue with scheduled scans aligned to local clock
    while regime_scanner.active and not regime_scanner.shutdown_requested:
        now = datetime.now(regime_scanner.local_tz)
        
        # Calculate next full hour
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        sleep_seconds = (next_hour - now).total_seconds()
        
        hours = sleep_seconds // 3600
        minutes = (sleep_seconds % 3600) // 60

        logger.info(f"Next market regime scan at {next_hour.strftime('%H:%M:%S')} ({hours} hours and {minutes} minutes from now)")
        
        # Wait until scheduled time, but check every second if we should stop
        while sleep_seconds > 0 and regime_scanner.active and not regime_scanner.shutdown_requested:
            try:
                # Sleep in small increments to be responsive to shutdown requests
                await asyncio.sleep(min(1, sleep_seconds))
                sleep_seconds -= 1
            except asyncio.CancelledError:
                logger.info("Sleep interrupted by shutdown")
                return
            
        if not regime_scanner.active or regime_scanner.shutdown_requested:
            break
            
        # Run the scan
        logger.info("Starting scheduled market regime scan")
        try:
            success = await regime_scanner.scan_market_regime()
            if success:
                logger.info("Scheduled market regime scan completed successfully")
            else:
                logger.warning("Scheduled market regime scan encountered errors")
        except asyncio.CancelledError:
            logger.info("Market regime scan cancelled")
            return
        except Exception as e:
            logger.error(f"Error during scheduled market regime scan: {e}")

# ======================== MAIN EXECUTION ======================== #
async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stock Ticker Fetcher and Market Regime Scanner')
    parser.add_argument('--search', type=str, help='Search for a ticker by name or symbol')
    parser.add_argument('--history', type=str, help='Get history for a specific ticker')
    parser.add_argument('--list', action='store_true', help='List all active tickers')
    parser.add_argument('--regime', action='store_true', help='Get current market regime')
    parser.add_argument('--regime-history', type=int, nargs='?', const=7, help='Get market regime history for past N days (default: 7)')
    args = parser.parse_args()
    
    ticker_scanner = PolygonTickerScanner()
    regime_scanner = MarketRegimeScanner(ticker_scanner)
    
    # Handle command line arguments
    if args.search:
        results = ticker_scanner.search_tickers_db(args.search)
        if results:
            print(f"Found {len(results)} matching tickers:")
            for result in results:
                print(f"{result['ticker']}: {result['name']} ({result['primary_exchange']})")
        else:
            print("No matching tickers found")
        return
    
    if args.history:
        results = ticker_scanner.get_ticker_history_db(args.history)
        if results:
            print(f"History for {args.history}:")
            for result in results:
                print(f"{result['change_date']}: {result['change_type']}")
        else:
            print(f"No history found for {args.history}")
        return
    
    if args.list:
        results = ticker_scanner.db.get_all_active_tickers()
        if results:
            print(f"Found {len(results)} active tickers:")
            for result in results:
                print(f"{result['ticker']}: {result['name']} ({result['primary_exchange']})")
        else:
            print("No active tickers found")
        return
        
    if args.regime:
        regime_scanner.start()
        await regime_scanner.scan_market_regime()
        latest_regime = regime_scanner.regime_db.get_latest_regime()
        if latest_regime:
            interpretation = regime_scanner.interpret_regime(latest_regime['regime'], json.loads(latest_regime['features']))
            print(f"Current market regime: {interpretation['label']}")
            print(f"Confidence: {latest_regime['confidence']:.2f}")
            print(f"Description: {interpretation['description']}")
            if 'details' in interpretation:
                print(f"Details: {interpretation['details']}")
            print(f"Timestamp: {latest_regime['timestamp']}")
        else:
            print("No market regime data available")
        await regime_scanner.shutdown()
        return
        
    if args.regime_history is not None:
        days = args.regime_history if args.regime_history > 0 else 7
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        results = regime_scanner.regime_db.get_regimes_by_date_range(start_date, end_date)
        if results:
            print(f"Market regime history for the past {days} days:")
            for result in results:
                interpretation = regime_scanner.interpret_regime(result['regime'], json.loads(result['features']))
                print(f"{result['timestamp']}: {interpretation['label']} (confidence: {result['confidence']:.2f})")
        else:
            print("No market regime history available")
        return
    
    # Normal operation - run both scanners
    ticker_scanner.start()
    regime_scanner.start()
    
    # Wait for initial cache load
    await asyncio.get_event_loop().run_in_executor(None, ticker_scanner.initial_refresh_complete.wait)
    
    # Create tasks for the schedulers - ticker first, then regime with a delay
    ticker_scheduler_task = asyncio.create_task(run_scheduled_ticker_refresh(ticker_scanner))
    
    # Wait a moment for the ticker scan to start
    await asyncio.sleep(2)
    
    regime_scheduler_task = asyncio.create_task(run_scheduled_regime_scan(regime_scanner))
    
    # Set up signal handlers
    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()
    
    def signal_handler():
        """Handle shutdown signals immediately"""
        print("\nReceived interrupt signal, shutting down...")
        ticker_scanner.stop()
        regime_scanner.stop()
        stop_event.set()
        # Cancel all tasks
        for task in asyncio.all_tasks(loop):
            if task is not asyncio.current_task():
                task.cancel()
    
    # Register signal handlers
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)
    else:
        # Windows signal handling
        signal.signal(signal.SIGINT, lambda s, f: signal_handler())
    
    try:
        # Create a task for the stop_event.wait() coroutine
        stop_task = asyncio.create_task(stop_event.wait())
        
        # Wait for either shutdown event or task completion
        done, pending = await asyncio.wait(
            [ticker_scheduler_task, regime_scheduler_task, stop_task], 
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel the scheduler tasks if they're still running
        if not ticker_scheduler_task.done():
            ticker_scheduler_task.cancel()
            try:
                await ticker_scheduler_task
            except asyncio.CancelledError:
                pass
                
        if not regime_scheduler_task.done():
            regime_scheduler_task.cancel()
            try:
                await regime_scheduler_task
            except asyncio.CancelledError:
                pass
                
        # Cancel the stop task if it's still running
        if not stop_task.done():
            stop_task.cancel()
            try:
                await stop_task
            except asyncio.CancelledError:
                pass
                
    except asyncio.CancelledError:
        logger.info("Main task cancelled")
    finally:
        # Shutdown the scanners
        await ticker_scanner.shutdown()
        await regime_scanner.shutdown()

if __name__ == "__main__":
    # Windows event loop policy
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")  