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
import pytz
from threading import Lock, Event, RLock
import sys
import signal
import requests
from tzlocal import get_localzone
from collections import defaultdict
import sqlite3
import contextlib
from typing import List, Dict, Set, Optional, Any, Tuple
import argparse
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings("ignore")

# ======================== CONFIGURATION ======================== #
class Config:
    # API Configuration
    POLYGON_API_KEY = "ld1Poa63U6t4Y2MwOCA2JeKQyHVrmyg8"
    
    # Scanner Configuration - Using composite indices instead of exchanges
    COMPOSITE_INDICES = ["^IXIC", "^NYA", "^XAX"]  # NASDAQ Composite, NYSE Composite, NYSE AMEX Composite
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
    MODEL_VERSION = "enhanced_v2.0"  # Model version identifier
    
    # Backtesting Configuration
    BACKTEST_HISTORICAL_DAYS = 30  # Default days to look back for backtesting
    
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

# ======================== DATABASE MIGRATION SYSTEM ======================== #
class DatabaseMigration:
    """Database migration system to handle schema changes"""
    
    def __init__(self, db_path: str, migrations: List[Tuple[int, str]]):
        self.db_path = db_path
        self.migrations = sorted(migrations, key=lambda x: x[0])
        self.current_version = 0
        
    @contextlib.contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
            
    def get_current_version(self) -> int:
        """Get current database version"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                # Check if migrations table exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='migrations'")
                if cursor.fetchone():
                    cursor.execute("SELECT MAX(version) as version FROM migrations")
                    result = cursor.fetchone()
                    return result['version'] if result and result['version'] is not None else 0
                return 0
        except sqlite3.Error:
            return 0
            
    def apply_migrations(self):
        """Apply all pending migrations"""
        current_version = self.get_current_version()
        logger.info(f"Current database version: {current_version}")
        
        pending_migrations = [m for m in self.migrations if m[0] > current_version]
        
        if not pending_migrations:
            logger.info("Database is up to date")
            return
            
        logger.info(f"Applying {len(pending_migrations)} migrations")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create migrations table if it doesn't exist
            if current_version == 0:
                cursor.execute('''
                    CREATE TABLE migrations (
                        version INTEGER PRIMARY KEY,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        description TEXT
                    )
                ''')
            
            for version, sql in pending_migrations:
                try:
                    # Execute migration SQL
                    if isinstance(sql, str):
                        cursor.executescript(sql)
                    elif isinstance(sql, list):
                        for statement in sql:
                            cursor.execute(statement)
                    
                    # Record migration
                    cursor.execute(
                        "INSERT INTO migrations (version, description) VALUES (?, ?)",
                        (version, f"Migration to version {version}")
                    )
                    
                    conn.commit()
                    logger.info(f"Applied migration to version {version}")
                    
                except sqlite3.Error as e:
                    conn.rollback()
                    logger.error(f"Failed to apply migration {version}: {e}")
                    raise

# ======================== DATABASE MANAGER ======================== #
class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        # Ensure the directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Define database migrations
        self.migrations = [
            (1, [
                '''CREATE TABLE IF NOT EXISTS tickers (
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
                )''',
                '''CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''',
                '''CREATE TABLE IF NOT EXISTS historical_tickers (
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
                    change_type TEXT,
                    change_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''',
                '''CREATE TABLE IF NOT EXISTS backtest_tickers (
                    date TEXT,
                    year INTEGER,
                    ticker TEXT,
                    name TEXT,
                    primary_exchange TEXT,
                    last_updated_utc TEXT,
                    type TEXT,
                    market TEXT,
                    locale TEXT,
                    currency_name TEXT,
                    PRIMARY KEY (date, ticker)
                )''',
                '''CREATE INDEX IF NOT EXISTS idx_tickers_exchange ON tickers(primary_exchange)''',
                '''CREATE INDEX IF NOT EXISTS idx_tickers_active ON tickers(active)''',
                '''CREATE INDEX IF NOT EXISTS idx_historical_tickers_date ON historical_tickers(change_date)''',
                '''CREATE INDEX IF NOT EXISTS idx_backtest_tickers_date ON backtest_tickers(date)''',
                '''CREATE INDEX IF NOT EXISTS idx_backtest_tickers_year ON backtest_tickers(year)'''
            ])
        ]
        
        # Apply migrations
        migrator = DatabaseMigration(db_path, self.migrations)
        migrator.apply_migrations()
            
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
        
    def upsert_backtest_tickers(self, tickers: List[Dict], date_str: str) -> int:
        """Insert or update tickers in the backtest table with year"""
        inserted = 0
        year = int(date_str.split('-')[0])  # Extract year from date
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            for ticker_data in tickers:
                cursor.execute('''
                    INSERT OR REPLACE INTO backtest_tickers 
                    (date, year, ticker, name, primary_exchange, last_updated_utc, 
                     type, market, locale, currency_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    date_str,
                    year,
                    ticker_data['ticker'],
                    ticker_data.get('name'),
                    ticker_data.get('primary_exchange'),
                    ticker_data.get('last_updated_utc'),
                    ticker_data.get('type'),
                    ticker_data.get('market'),
                    ticker_data.get('locale'),
                    ticker_data.get('currency_name')
                ))
                
                if cursor.rowcount > 0:
                    inserted += 1
            
            conn.commit()
            
        return inserted
        
    def get_backtest_tickers(self, date_str: str) -> List[Dict]:
        """Get tickers for a specific backtest date"""
        return self.execute_query(
            "SELECT * FROM backtest_tickers WHERE date = ? ORDER BY ticker",
            (date_str,)
        )
        
    def get_backtest_tickers_by_year(self, year: int) -> List[Dict]:
        """Get tickers for a specific backtest year"""
        return self.execute_query(
            "SELECT * FROM backtest_tickers WHERE year = ? ORDER BY date, ticker",
            (year,)
        )
        
    def get_backtest_dates(self) -> List[str]:
        """Get all available backtest dates"""
        result = self.execute_query(
            "SELECT DISTINCT date FROM backtest_tickers ORDER BY date"
        )
        return [row['date'] for row in result]
        
    def get_backtest_years(self) -> List[int]:
        """Get all available backtest years"""
        result = self.execute_query(
            "SELECT DISTINCT year FROM backtest_tickers ORDER BY year"
        )
        return [row['year'] for row in result]

# ======================== MARKET REGIME DATABASE ======================== #
class MarketRegimeDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        # Ensure the directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Define database migrations
        self.migrations = [
            (1, [
                '''CREATE TABLE IF NOT EXISTS market_regimes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    regime INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    features TEXT,
                    model_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''',
                '''CREATE TABLE IF NOT EXISTS regime_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    regime INTEGER NOT NULL,
                    start_date DATETIME NOT NULL,
                    end_date DATETIME,
                    duration_days INTEGER,
                    return_pct REAL,
                    volatility REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''',
                '''CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    backtest_date TEXT NOT NULL,
                    backtest_year INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL,
                    regime INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    features TEXT,
                    model_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''',
                '''CREATE TABLE IF NOT EXISTS backtest_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    backtest_date TEXT NOT NULL,
                    backtest_year INTEGER NOT NULL,
                    analysis_type TEXT NOT NULL,
                    analysis_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''',
                '''CREATE INDEX IF NOT EXISTS idx_regimes_timestamp ON market_regimes(timestamp)''',
                '''CREATE INDEX IF NOT EXISTS idx_regimes_regime ON market_regimes(regime)''',
                '''CREATE INDEX IF NOT EXISTS idx_statistics_regime ON regime_statistics(regime)''',
                '''CREATE INDEX IF NOT EXISTS idx_statistics_date ON regime_statistics(start_date)''',
                '''CREATE INDEX IF NOT EXISTS idx_backtest_date ON backtest_results(backtest_date)''',
                '''CREATE INDEX IF NOT EXISTS idx_backtest_year ON backtest_results(backtest_year)''',
                '''CREATE INDEX IF NOT EXISTS idx_backtest_timestamp ON backtest_results(timestamp)''',
                '''CREATE INDEX IF NOT EXISTS idx_analysis_date ON backtest_analysis(backtest_date)''',
                '''CREATE INDEX IF NOT EXISTS idx_analysis_year ON backtest_analysis(backtest_year)''',
                '''CREATE INDEX IF NOT EXISTS idx_analysis_type ON backtest_analysis(analysis_type)'''
            ])
        ]
        
        # Apply migrations
        migrator = DatabaseMigration(db_path, self.migrations)
        migrator.apply_migrations()
            
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
            
    def save_backtest_result(self, backtest_date: str, timestamp: datetime, regime: int, 
                           confidence: float, features: Dict, model_version: str) -> int:
        """Save backtest result to database"""
        backtest_year = int(backtest_date.split('-')[0])
        return self.execute_write(
            '''INSERT INTO backtest_results 
               (backtest_date, backtest_year, timestamp, regime, confidence, features, model_version) 
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (backtest_date, backtest_year, timestamp, regime, confidence, json.dumps(features), model_version)
        )
        
    def get_backtest_results(self, backtest_date: str) -> List[Dict]:
        """Get backtest results for a specific date"""
        return self.execute_query(
            "SELECT * FROM backtest_results WHERE backtest_date = ? ORDER BY timestamp",
            (backtest_date,)
        )
        
    def get_backtest_results_by_year(self, year: int) -> List[Dict]:
        """Get backtest results for a specific year"""
        return self.execute_query(
            "SELECT * FROM backtest_results WHERE backtest_year = ? ORDER BY backtest_date, timestamp",
            (year,)
        )
        
    def get_backtest_dates(self) -> List[str]:
        """Get all available backtest dates"""
        result = self.execute_query(
            "SELECT DISTINCT backtest_date FROM backtest_results ORDER BY backtest_date"
        )
        return [row['backtest_date'] for row in result]
        
    def get_backtest_years(self) -> List[int]:
        """Get all available backtest years"""
        result = self.execute_query(
            "SELECT DISTINCT backtest_year FROM backtest_results ORDER BY backtest_year"
        )
        return [row['backtest_year'] for row in result]
        
    def save_backtest_analysis(self, backtest_date: str, analysis_type: str, analysis_data: Dict) -> int:
        """Save backtest analysis to database"""
        backtest_year = int(backtest_date.split('-')[0])
        return self.execute_write(
            '''INSERT INTO backtest_analysis 
               (backtest_date, backtest_year, analysis_type, analysis_data) 
               VALUES (?, ?, ?, ?)''',
            (backtest_date, backtest_year, analysis_type, json.dumps(analysis_data))
        )
        
    def get_backtest_analysis(self, backtest_date: str, analysis_type: str) -> Optional[Dict]:
        """Get backtest analysis for a specific date and type"""
        result = self.execute_query(
            "SELECT * FROM backtest_analysis WHERE backtest_date = ? AND analysis_type = ? ORDER BY created_at DESC LIMIT 1",
            (backtest_date, analysis_type)
        )
        if result:
            analysis_data = result[0]['analysis_data']
            return json.loads(analysis_data) if analysis_data else {}
        return None

# ======================== TICKER SCANNER ======================== #
class PolygonTickerScanner:
    def __init__(self):
        self.api_key = config.POLYGON_API_KEY
        self.base_url = "https://api.polygon.io/v3/reference/tickers"
        # Use composite indices instead of exchanges
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
        # Backtesting attributes
        self.backtest_mode = False
        self.backtest_date = None
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
        
        # Use current date or backtest date
        if self.backtest_mode and self.backtest_date:
            date_param = self.backtest_date
            logger.info(f"Using historical date: {date_param}")
        else:
            date_param = datetime.now().strftime("%Y-%m-%d")
            
        # Different API endpoint for composite indices
        if composite_index == "^IXIC":  # NASDAQ Composite
            exchange = "XNAS"
        elif composite_index == "^NYA":  # NYSE Composite
            exchange = "XNYS"
        elif composite_index == "^XAX":  # NYSE AMEX Composite
            exchange = "XASE"
        else:
            logger.error(f"Unknown composite index: {composite_index}")
            return []
            
        params = {
            "market": "stocks",
            "exchange": exchange,
            "active": "true",
            "limit": 1000,  # Maximum allowed by Polygon
            "apiKey": self.api_key,
            "date": date_param  # Add date parameter for historical data
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
        
        if self.backtest_mode and self.backtest_date:
            logger.info(f"Starting historical ticker refresh for {self.backtest_date}")
        else:
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
            # For backtest mode, we don't update the main database
            if self.backtest_mode and self.backtest_date:
                # Store backtest results
                tickers_data = new_df.to_dict('records')
                inserted = self.db.upsert_backtest_tickers(tickers_data, self.backtest_date)
                logger.info(f"Stored {inserted} tickers in backtest database for {self.backtest_date}")
            else:
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
        
        if not self.backtest_mode:
            self.db.update_metadata('last_refresh_time', self.last_refresh_time)
        
        elapsed = time.time() - start_time
        logger.info(f"Ticker refresh completed in {elapsed:.2f}s")
        
        if not self.backtest_mode:
            logger.info(f"Total: {len(new_df)} | Added: {len(added)} | Removed: {len(removed)}")
            logger.info(f"Database: {inserted} inserted, {updated} updated")
        else:
            logger.info(f"Historical data: {len(new_df)} tickers for {self.backtest_date}")
            
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
        
    def get_backtest_tickers(self, date_str: str) -> List[Dict]:
        """Get tickers for a specific backtest date"""
        return self.db.get_backtest_tickers(date_str)
        
    def get_backtest_tickers_by_year(self, year: int) -> List[Dict]:
        """Get tickers for a specific backtest year"""
        return self.db.get_backtest_tickers_by_year(year)
        
    def get_backtest_dates(self) -> List[str]:
        """Get all available backtest dates"""
        return self.db.get_backtest_dates()
        
    def get_backtest_years(self) -> List[int]:
        """Get all available backtest years"""
        return self.db.get_backtest_years()

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
        self.scaler = StandardScaler()
        self.model_version = config.MODEL_VERSION
        
        # Backtesting attributes
        self.backtest_mode = False
        self.backtest_date = None
        
        # Use the composite indices for market regime detection
        self.market_indices = {
            "^IXIC": "COMP",    # NASDAQ Composite
            "^NYA": "NYA",      # NYSE Composite
            "^XAX": "XAX"       # NYSE AMEX Composite
        }
        
        logger.info(f"Market regime database path: {config.DATABASE_PATH_REGIME}")
        logger.info(f"Using composite indices for regime detection: {', '.join(self.market_indices.keys())}")
        
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
                
    async def fetch_market_data(self, days: int = config.HISTORICAL_DATA_DAYS, end_date: datetime = None) -> pd.DataFrame:
        """Fetch historical market data for all market indices with optional end date for backtesting"""
        logger.info(f"Fetching {days} days of market data for regime analysis")
        
        # Get the market indices
        tickers = list(self.market_indices.keys())
        
        all_data = []
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_historical_data(session, ticker, days, end_date) for ticker in tickers]
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
        
        # Drop rows with NaN values
        features.dropna(inplace=True)
        
        logger.info(f"Calculated advanced features with shape: {features.shape}")
        return features
        
    def train_ensemble(self, features: pd.DataFrame):
        """Train an ensemble of models for better regime detection"""
        scaled_features = self.scaler.fit_transform(features)
        
        # HMM Model
        self.hmm_model = hmm.GaussianHMM(
            n_components=config.HMM_N_COMPONENTS,
            covariance_type=config.HMM_COVARIANCE_TYPE,
            n_iter=config.HMM_N_ITER,
            random_state=42
        )
        self.hmm_model.fit(scaled_features)
        
        # GMM Model for comparison
        self.gmm_model = GaussianMixture(
            n_components=config.HMM_N_COMPONENTS,
            covariance_type=config.HMM_COVARIANCE_TYPE,
            max_iter=config.HMM_N_ITER,
            random_state=42
        )
        self.gmm_model.fit(scaled_features)
        
        logger.info("Ensemble model training completed")
        
    def predict_regime_ensemble(self, features: pd.DataFrame) -> Tuple[int, float, Dict]:
        """Predict using ensemble approach with model confidence"""
        scaled_features = self.scaler.transform(features)
        
        # Get predictions from both models
        hmm_predictions = self.hmm_model.predict(scaled_features)
        gmm_predictions = self.gmm_model.predict(scaled_features)
        
        # Use the most recent predictions
        hmm_regime = hmm_predictions[-1]
        gmm_regime = gmm_predictions[-1]
        
        # Calculate model agreement
        agreement = np.mean(hmm_predictions == gmm_predictions)
        
        # If models agree, use that regime
        if hmm_regime == gmm_regime:
            regime = hmm_regime
            confidence = agreement * 0.8 + 0.2  # Boost confidence when models agree
        else:
            # Use the model with higher confidence in its prediction
            hmm_probs = self.hmm_model.predict_proba(scaled_features)
            gmm_probs = self.gmm_model.predict_proba(scaled_features)
            
            hmm_confidence = np.max(hmm_probs, axis=1)[-1]
            gmm_confidence = np.max(gmm_probs, axis=1)[-1]
            
            if hmm_confidence > gmm_confidence:
                regime = hmm_regime
                confidence = hmm_confidence * agreement
            else:
                regime = gmm_regime
                confidence = gmm_confidence * agreement
        
        # Prepare feature values for storage
        feature_values = {
            col: features[col].iloc[-1] for col in features.columns
        }
        
        return regime, confidence, feature_values
        
    def predict_regime_single(self, features: pd.DataFrame) -> Tuple[int, float, Dict]:
        """Predict current market regime using trained HMM only"""
        if self.hmm_model is None:
            logger.error("HMM model not trained")
            return -1, 0.0, {}
            
        # Use the most recent features
        recent_features = features.iloc[-5:].copy()  # Use last 5 days for more robust prediction
        
        # Scale the features
        scaled_features = self.scaler.transform(recent_features)
        
        # Predict regime for the last 5 days
        regimes = self.hmm_model.predict(scaled_features)
        
        # Use the most recent regime
        regime = regimes[-1]
        
        # Calculate confidence based on consistency of recent predictions
        confidence = np.mean(regimes == regime)  # Percentage of recent predictions that match
        
        # Apply a confidence cap to avoid 100% certainty
        confidence = min(confidence, 0.95)
        
        # Add a small uncertainty factor
        confidence = confidence * 0.9 + 0.05  # Ensures confidence is between 0.05 and 0.90
        
        # Prepare feature values for storage
        feature_values = {
            col: recent_features[col].iloc[-1] for col in recent_features.columns
        }
        
        # Log the confidence calculation details
        logger.debug(f"Recent regime predictions: {regimes}")
        logger.debug(f"Confidence calculation: {np.mean(regimes == regime):.2f} -> {confidence:.2f}")
        
        return regime, confidence, feature_values
        
    def interpret_regime(self, regime: int) -> str:
        """Interpret the numerical regime value"""
        # This is a simple interpretation - you might want to enhance this
        # based on your specific market analysis
        if regime == 0:
            return "Bear Market"
        elif regime == 1:
            return "Sideways Market"
        elif regime == 2:
            return "Bull Market"
        else:
            return f"Unknown Regime {regime}"
            
    async def scan_market_regime(self, backtest_date: str = None):
        """Perform a complete market regime scan with optional backtesting"""
        if self.shutdown_requested:
            return False
            
        logger.info("Starting market regime scan")
        start_time = time.time()
        
        try:
            # Set backtest mode if date is provided
            if backtest_date:
                self.backtest_mode = True
                self.backtest_date = backtest_date
                end_date = datetime.strptime(backtest_date, "%Y-%m-%d")
                logger.info(f"Running backtest for date: {backtest_date}")
            else:
                end_date = None
            
            # Fetch market data
            market_data = await self.fetch_market_data(end_date=end_date)
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
                if config.USE_ENSEMBLE_MODEL:
                    self.train_ensemble(features)
                else:
                    scaled_features = self.scaler.fit_transform(features)
                    self.hmm_model = hmm.GaussianHMM(
                        n_components=config.HMM_N_COMPONENTS,
                        covariance_type=config.HMM_COVARIANCE_TYPE,
                        n_iter=config.HMM_N_ITER,
                        random_state=42
                    )
                    self.hmm_model.fit(scaled_features)
                
            # Predict current regime
            if config.USE_ENSEMBLE_MODEL:
                regime, confidence, feature_values = self.predict_regime_ensemble(features)
            else:
                regime, confidence, feature_values = self.predict_regime_single(features)
                
            regime_label = self.interpret_regime(regime)
            
            # Save to appropriate database table
            timestamp = datetime.now() if not backtest_date else datetime.strptime(backtest_date, "%Y-%m-%d")
            
            if backtest_date:
                # Save to backtest results
                self.regime_db.save_backtest_result(
                    backtest_date, timestamp, regime, confidence, feature_values, self.model_version
                )
                logger.info(f"Backtest result saved for {backtest_date}: {regime_label} (confidence: {confidence:.2f})")
                
                # Perform backtest analysis
                self.perform_backtest_analysis(backtest_date, regime, confidence, feature_values)
            else:
                # Save to regular market regimes table
                self.regime_db.save_market_regime(
                    timestamp, regime, confidence, feature_values, self.model_version
                )
                logger.info(f"Market regime saved: {regime_label} (confidence: {confidence:.2f})")
            
            elapsed = time.time() - start_time
            logger.info(f"Market regime scan completed in {elapsed:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during market regime scan: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        finally:
            # Reset backtest mode
            self.backtest_mode = False
            self.backtest_date = None
            
    def perform_backtest_analysis(self, backtest_date: str, regime: int, confidence: float, features: Dict):
        """Perform analysis on backtest results"""
        logger.info(f"Performing backtest analysis for {backtest_date}")
        
        # Get historical regimes for comparison
        start_date = (datetime.strptime(backtest_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
        historical_regimes = self.regime_db.get_regimes_by_date_range(
            datetime.strptime(start_date, "%Y-%m-%d"),
            datetime.strptime(backtest_date, "%Y-%m-%d")
        )
        
        # Calculate regime persistence
        if historical_regimes:
            recent_regimes = [r['regime'] for r in historical_regimes[-5:]]  # Last 5 regimes
            regime_persistence = sum(1 for r in recent_regimes if r == regime) / len(recent_regimes)
        else:
            regime_persistence = 0
        
        # Calculate feature trends
        feature_trends = {}
        for feature_name, feature_value in features.items():
            # Simple trend calculation (could be enhanced)
            if 'return' in feature_name:
                feature_trends[f"{feature_name}_trend"] = "positive" if feature_value > 0 else "negative"
            elif 'volatility' in feature_name:
                feature_trends[f"{feature_name}_trend"] = "high" if feature_value > 1 else "low"
        
        # Prepare analysis data
        analysis_data = {
            "regime": regime,
            "regime_label": self.interpret_regime(regime),
            "confidence": confidence,
            "regime_persistence": regime_persistence,
            "feature_trends": feature_trends,
            "historical_regime_count": len(historical_regimes),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Save analysis to database
        self.regime_db.save_backtest_analysis(backtest_date, "basic_analysis", analysis_data)
        logger.info(f"Backtest analysis completed for {backtest_date}")
        
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

# ======================== BACKTESTER ======================== #
class Backtester:
    def __init__(self, ticker_scanner: PolygonTickerScanner, regime_scanner: MarketRegimeScanner):
        self.ticker_scanner = ticker_scanner
        self.regime_scanner = regime_scanner
        
    async def run_backtest(self, start_date, end_date=None):
        """
        Run backtest for a specific date range
        Args:
            start_date: datetime object or string in YYYY-MM-DD format
            end_date: datetime object or string in YYYY-MM-DD format (optional)
        """
        if end_date is None:
            end_date = start_date
            
        # Convert to datetime if strings are provided
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
        logger.info(f"Starting backtest from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Only weekdays
                date_str = current_date.strftime("%Y-%m-%d")
                logger.info(f"Running backtest for {date_str}")
                
                # First, ensure we have ticker data for this date
                existing_tickers = self.ticker_scanner.db.get_backtest_tickers(date_str)
                if not existing_tickers:
                    logger.info(f"No ticker data found for {date_str}, fetching...")
                    self.ticker_scanner.backtest_mode = True
                    self.ticker_scanner.backtest_date = date_str
                    success = await self.ticker_scanner.refresh_all_tickers()
                    if not success:
                        logger.warning(f"Failed to fetch ticker data for {date_str}")
                        continue
                
                # Now run the regime scan for this date
                success = await self.regime_scanner.scan_market_regime(date_str)
                if success:
                    logger.info(f"Successfully completed regime backtest for {date_str}")
                else:
                    logger.warning(f"Failed to complete regime backtest for {date_str}")
            
            current_date += timedelta(days=1)
            
        logger.info("Backtest completed")
        
    def get_backtest_results(self, date_str: str) -> List[Dict]:
        """Get backtest results for a specific date"""
        return self.regime_scanner.regime_db.get_backtest_results(date_str)
        
    def get_backtest_analysis(self, date_str: str) -> Optional[Dict]:
        """Get backtest analysis for a specific date"""
        return self.regime_scanner.regime_db.get_backtest_analysis(date_str, "basic_analysis")
        
    def get_backtest_dates(self) -> List[str]:
        """Get all available backtest dates"""
        return self.regime_scanner.regime_db.get_backtest_dates()
        
    def get_backtest_years(self) -> List[int]:
        """Get all available backtest years"""
        return self.regime_scanner.regime_db.get_backtest_years()

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
    parser = argparse.ArgumentParser(description='Stock Ticker Fetcher and Market Regime Scanner with Backtesting')
    parser.add_argument('--search', type=str, help='Search for a ticker by name or symbol')
    parser.add_argument('--history', type=str, help='Get history for a specific ticker')
    parser.add_argument('--list', action='store_true', help='List all active tickers')
    parser.add_argument('--regime', action='store_true', help='Get current market regime')
    parser.add_argument('--regime-history', type=int, nargs='?', const=7, help='Get market regime history for past N days (default: 7)')
    
    # Backtesting arguments
    parser.add_argument('--backtest', type=str, help='Run backtest for a specific date (YYYY-MM-DD)')
    parser.add_argument('--backtest-range', type=str, help='Run backtest for a date range (YYYY-MM-DD:YYYY-MM-DD)')
    parser.add_argument('--backtest-year', type=int, help='Run backtest for a specific year')
    parser.add_argument('--list-backtests', action='store_true', help='List available backtest dates')
    parser.add_argument('--list-backtest-years', action='store_true', help='List available backtest years')
    parser.add_argument('--show-backtest-results', type=str, help='Show results for a specific backtest date (YYYY-MM-DD)')
    parser.add_argument('--show-backtest-analysis', type=str, help='Show analysis for a specific backtest date (YYYY-MM-DD)')
    
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
            regime_label = regime_scanner.interpret_regime(latest_regime['regime'])
            print(f"Current market regime: {regime_label}")
            print(f"Confidence: {latest_regime['confidence']:.2f}")
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
                regime_label = regime_scanner.interpret_regime(result['regime'])
                print(f"{result['timestamp']}: {regime_label} (confidence: {result['confidence']:.2f})")
        else:
            print("No market regime history available")
        return
    
    # Backtesting commands
    if (args.backtest or args.backtest_range or args.backtest_year or 
        args.list_backtests or args.list_backtest_years or 
        args.show_backtest_results or args.show_backtest_analysis):
        
        backtester = Backtester(ticker_scanner, regime_scanner)
        
        if args.list_backtests:
            # List available backtest dates
            dates = backtester.get_backtest_dates()
            if dates:
                print("Available backtest dates:")
                for date in dates:
                    print(f"  {date}")
            else:
                print("No backtest data available")
            return
            
        if args.list_backtest_years:
            # List available backtest years
            years = backtester.get_backtest_years()
            if years:
                print("Available backtest years:")
                for year in years:
                    print(f"  {year}")
            else:
                print("No backtest data available")
            return
            
        if args.show_backtest_results:
            # Show results for a specific backtest date
            results = backtester.get_backtest_results(args.show_backtest_results)
            if results:
                print(f"Backtest results for {args.show_backtest_results}:")
                for result in results:
                    regime_label = regime_scanner.interpret_regime(result['regime'])
                    print(f"  {result['timestamp']}: {regime_label} (confidence: {result['confidence']:.2f})")
            else:
                print(f"No backtest results found for {args.show_backtest_results}")
            return
            
        if args.show_backtest_analysis:
            # Show analysis for a specific backtest date
            analysis = backtester.get_backtest_analysis(args.show_backtest_analysis)
            if analysis:
                print(f"Backtest analysis for {args.show_backtest_analysis}:")
                print(f"  Regime: {analysis.get('regime_label', 'N/A')}")
                print(f"  Confidence: {analysis.get('confidence', 0):.2f}")
                print(f"  Regime Persistence: {analysis.get('regime_persistence', 0):.2f}")
                print(f"  Historical Regime Count: {analysis.get('historical_regime_count', 0)}")
                print("  Feature Trends:")
                for feature, trend in analysis.get('feature_trends', {}).items():
                    print(f"    {feature}: {trend}")
            else:
                print(f"No backtest analysis found for {args.show_backtest_analysis}")
            return
        
        if args.backtest:
            # Single date backtest
            await backtester.run_backtest(args.backtest)
        elif args.backtest_range:
            # Date range backtest
            start_str, end_str = args.backtest_range.split(':')
            await backtester.run_backtest(start_str, end_str)
        elif args.backtest_year:
            # Year backtest
            start_date = datetime(args.backtest_year, 1, 1)
            end_date = datetime(args.backtest_year, 12, 31)
            await backtester.run_backtest(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    else:
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