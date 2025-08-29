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
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
    
    # Database Configuration - Use the specific path you requested
    DATABASE_PATH_TICKER = r"C:\Users\kyung\StockScanner\core\ticker_data.db"
    DATABASE_PATH_MARKET_REGIME = r"C:\Users\kyung\StockScanner\core\market_regime_data.db"
    
    # Market Regime Configuration
    BENCHMARK_SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]  # Broad market ETFs for regime detection
    HISTORICAL_DAYS = 365 * 3  # 3 years of historical data
    HMM_N_COMPONENTS = 4  # Number of market regimes to detect (Bull, Bear, Correction, Sideways)
    HMM_N_ITER = 1000  # Maximum iterations for HMM training
    REGIME_UPDATE_INTERVAL = 3600  # Update regime every hour (in seconds)
    
    # Logging Configuration
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Initialize configuration
config = Config()

# ======================== LOGGING SETUP ======================== #
def setup_logging():
    """Configure logging with file and console handlers"""
    os.makedirs("logs", exist_ok=True)
    
    logger = logging.getLogger("TickerFetcher")
    logger.setLevel(config.LOG_LEVEL)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter(config.LOG_FORMAT)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/ticker_fetcher_{timestamp}.log"
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
class MarketRegimeDatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        # Ensure the directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
        
    def _init_database(self):
        """Initialize market regime database with required tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create market_regimes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_regimes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    regime INTEGER,
                    regime_probability REAL,
                    volatility REAL,
                    trend_strength REAL,
                    market_health REAL,
                    features TEXT,
                    model_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create market_regime_history table for historical predictions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_regime_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE,
                    regime INTEGER,
                    regime_probability REAL,
                    volatility REAL,
                    trend_strength REAL,
                    market_health REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_regimes_timestamp ON market_regimes(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_regimes_history_date ON market_regime_history(date)')
            
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
            
    def save_market_regime(self, regime_data: Dict) -> int:
        """Save market regime data to database"""
        query = '''
            INSERT INTO market_regimes 
            (regime, regime_probability, volatility, trend_strength, market_health, features, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        '''
        return self.execute_write(
            query,
            (
                regime_data['regime'],
                regime_data['regime_probability'],
                regime_data['volatility'],
                regime_data['trend_strength'],
                regime_data['market_health'],
                json.dumps(regime_data['features']),
                regime_data.get('model_version', 'v1.0')
            )
        )
        
    def save_daily_regime(self, regime_data: Dict) -> int:
        """Save daily market regime data to history table"""
        query = '''
            INSERT OR REPLACE INTO market_regime_history 
            (date, regime, regime_probability, volatility, trend_strength, market_health)
            VALUES (?, ?, ?, ?, ?, ?)
        '''
        return self.execute_write(
            query,
            (
                regime_data['date'],
                regime_data['regime'],
                regime_data['regime_probability'],
                regime_data['volatility'],
                regime_data['trend_strength'],
                regime_data['market_health']
            )
        )
        
    def get_latest_regime(self) -> Optional[Dict]:
        """Get the latest market regime data"""
        result = self.execute_query(
            "SELECT * FROM market_regimes ORDER BY timestamp DESC LIMIT 1"
        )
        return result[0] if result else None
        
    def get_regime_history(self, days: int = 30) -> List[Dict]:
        """Get market regime history for the specified number of days"""
        return self.execute_query(
            "SELECT * FROM market_regime_history ORDER BY date DESC LIMIT ?",
            (days,)
        )

# ======================== HMM MARKET REGIME DETECTION ======================== #
class HMMMarketRegimeDetector:
    def __init__(self, n_components=4, n_iter=1000):
        self.n_components = n_components
        self.n_iter = n_iter
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = ['returns', 'volatility', 'momentum', 'atr']
        self.regime_names = {
            0: "High Volatility Bear",
            1: "Low Volatility Bull",
            2: "Correction",
            3: "Sideways"
        }
        
    def calculate_features(self, df):
        """Calculate technical features for regime detection"""
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Calculate volatility (rolling standard deviation of returns)
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Calculate momentum (12-day rate of change)
        df['momentum'] = df['close'].pct_change(periods=12)
        
        # Calculate Average True Range (ATR)
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = np.abs(df['high'] - df['close'].shift())
        df['low_close'] = np.abs(df['low'] - df['close'].shift())
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        # Drop NaN values
        df = df.dropna()
        
        return df
        
    def prepare_features(self, df):
        """Prepare features for HMM training"""
        feature_df = self.calculate_features(df.copy())
        features = feature_df[self.feature_columns].values
        
        # Scale features
        if len(features) > 0:
            features_scaled = self.scaler.fit_transform(features)
            return features_scaled, feature_df
        return np.array([]), feature_df
        
    def fit(self, features):
        """Train the HMM model"""
        if len(features) == 0:
            raise ValueError("No features available for training")
            
        self.model = hmm.GaussianHMM(
            n_components=self.n_components,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=42
        )
        self.model.fit(features)
        return self.model
        
    def predict(self, features):
        """Predict market regime using trained HMM"""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
            
        if len(features) == 0:
            raise ValueError("No features available for prediction")
            
        # Predict regime
        regime = self.model.predict(features[-1].reshape(1, -1))[0]
        
        # Calculate regime probability
        log_prob = self.model.score(features[-1].reshape(1, -1))
        regime_probability = np.exp(log_prob) if log_prob < 0 else 1 / (1 + np.exp(-log_prob))
        
        return regime, regime_probability
        
    def get_regime_characteristics(self, features_df, regime):
        """Calculate characteristics of the detected regime"""
        recent_data = features_df.tail(20)
        
        # Calculate volatility (recent average)
        volatility = recent_data['volatility'].mean() if 'volatility' in recent_data.columns else 0
        
        # Calculate trend strength (absolute momentum)
        trend_strength = abs(recent_data['returns'].mean()) if 'returns' in recent_data.columns else 0
        
        # Calculate market health (combination of factors)
        market_health = 0.5  # Default neutral
        
        if not recent_data.empty:
            positive_returns = (recent_data['returns'] > 0).mean()
            volatility_factor = 1 - min(volatility * 10, 1)  # Normalize volatility
            market_health = (positive_returns * 0.7 + volatility_factor * 0.3)
        
        return {
            'regime': int(regime),
            'regime_name': self.regime_names.get(regime, "Unknown"),
            'volatility': float(volatility),
            'trend_strength': float(trend_strength),
            'market_health': float(market_health)
        }

# ======================== MARKET REGIME SCANNER ======================== #
class MarketRegimeScanner:
    def __init__(self):
        self.api_key = config.POLYGON_API_KEY
        self.base_url = "https://api.polygon.io/v2/aggs/ticker"
        self.benchmark_symbols = config.BENCHMARK_SYMBOLS
        self.historical_days = config.HISTORICAL_DAYS
        self.regime_db = MarketRegimeDatabaseManager(config.DATABASE_PATH_MARKET_REGIME)
        self.detector = HMMMarketRegimeDetector(
            n_components=config.HMM_N_COMPONENTS,
            n_iter=config.HMM_N_ITER
        )
        self.active = False
        self.shutdown_requested = False
        self.last_scan_time = 0
        self.local_tz = get_localzone()
        self.semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
        
    async def _fetch_historical_data(self, session, symbol):
        """Fetch historical data for a symbol"""
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.historical_days)
        
        # Format dates for API
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        url = f"{self.base_url}/{symbol}/range/1/day/{start_str}/{end_str}?apiKey={self.api_key}&adjusted=true&sort=asc"
        
        async with self.semaphore:
            try:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('status') == 'OK' and data.get('resultsCount', 0) > 0:
                            df = pd.DataFrame(data['results'])
                            df['date'] = pd.to_datetime(df['t'], unit='ms')
                            df.set_index('date', inplace=True)
                            df.rename(columns={
                                'o': 'open',
                                'h': 'high',
                                'l': 'low',
                                'c': 'close',
                                'v': 'volume'
                            }, inplace=True)
                            return df[['open', 'high', 'low', 'close', 'volume']]
                    return None
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                return None
                
    async def fetch_all_historical_data(self):
        """Fetch historical data for all benchmark symbols"""
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_historical_data(session, symbol) for symbol in self.benchmark_symbols]
            results = await asyncio.gather(*tasks)
            
            # Combine data from all symbols
            combined_data = {}
            for i, symbol in enumerate(self.benchmark_symbols):
                if results[i] is not None:
                    combined_data[symbol] = results[i]
            
            return combined_data
            
    def prepare_training_data(self, historical_data):
        """Prepare training data from historical prices"""
        if not historical_data:
            return None
            
        # Use SPY as primary benchmark, fallback to others
        primary_symbol = "SPY"
        if primary_symbol not in historical_data:
            primary_symbol = list(historical_data.keys())[0]
            
        df = historical_data[primary_symbol].copy()
        features, feature_df = self.detector.prepare_features(df)
        
        return features, feature_df
        
    def detect_regime(self, historical_data):
        """Detect current market regime using HMM"""
        try:
            # Prepare training data
            features, feature_df = self.prepare_training_data(historical_data)
            
            if features is None or len(features) == 0:
                logger.error("No features available for regime detection")
                return None
                
            # Train HMM model
            self.detector.fit(features)
            
            # Predict current regime
            current_regime, regime_probability = self.detector.predict(features)
            
            # Get regime characteristics
            regime_info = self.detector.get_regime_characteristics(feature_df, current_regime)
            regime_info['regime_probability'] = float(regime_probability)
            
            # Add feature information
            latest_features = feature_df[self.detector.feature_columns].iloc[-1].to_dict()
            regime_info['features'] = latest_features
            
            logger.info(f"Detected market regime: {regime_info['regime_name']} "
                       f"(Probability: {regime_probability:.2f}, "
                       f"Volatility: {regime_info['volatility']:.4f}, "
                       f"Health: {regime_info['market_health']:.2f})")
            
            return regime_info
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return None
            
    async def scan_market_regime(self):
        """Perform a complete market regime scan"""
        if self.shutdown_requested:
            return None
            
        logger.info("Starting market regime scan")
        start_time = time.time()
        
        try:
            # Fetch historical data
            historical_data = await self.fetch_all_historical_data()
            
            if not historical_data:
                logger.error("No historical data fetched for regime detection")
                return None
                
            # Detect market regime
            regime_info = self.detect_regime(historical_data)
            
            if regime_info:
                # Save to database
                self.regime_db.save_market_regime(regime_info)
                
                # Also save daily record
                today = datetime.now().date().isoformat()
                self.regime_db.save_daily_regime({
                    'date': today,
                    'regime': regime_info['regime'],
                    'regime_probability': regime_info['regime_probability'],
                    'volatility': regime_info['volatility'],
                    'trend_strength': regime_info['trend_strength'],
                    'market_health': regime_info['market_health']
                })
                
                elapsed = time.time() - start_time
                logger.info(f"Market regime scan completed in {elapsed:.2f}s")
                
            return regime_info
            
        except Exception as e:
            logger.error(f"Error in market regime scan: {e}")
            return None
            
    def start(self):
        """Start the market regime scanner"""
        self.active = True
        self.shutdown_requested = False
        logger.info("Market regime scanner started")
        
    def stop(self):
        """Stop the market regime scanner"""
        self.active = False
        self.shutdown_requested = True
        logger.info("Market regime scanner stopped")
        
    def get_latest_regime(self):
        """Get the latest market regime from database"""
        return self.regime_db.get_latest_regime()
        
    def get_regime_history(self, days=30):
        """Get market regime history"""
        return self.regime_db.get_regime_history(days)

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
        self.regime_scanner = MarketRegimeScanner()  # Add regime scanner
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
            self.regime_scanner.start()  # Start regime scanner
            logger.info("Ticker scanner started")

    def stop(self):
        self.active = False
        self.shutdown_requested = True
        self.regime_scanner.stop()  # Stop regime scanner
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
        
    def get_market_regime(self):
        """Get the latest market regime"""
        return self.regime_scanner.get_latest_regime()
        
    def get_regime_history(self, days=30):
        """Get market regime history"""
        return self.regime_scanner.get_regime_history(days)

# ======================== SCHEDULER ======================== #
async def run_scheduled_refresh(scanner):
    """Run immediate scan on startup and then daily at scheduled time"""
    # Run immediate scan on startup
    logger.info("Starting immediate scan on startup")
    try:
        success = await scanner.refresh_all_tickers()
        if success:
            logger.info("Initial scan completed successfully")
        else:
            logger.warning("Initial scan encountered errors")
    except asyncio.CancelledError:
        logger.info("Initial scan cancelled")
        return
    except Exception as e:
        logger.error(f"Error during initial scan: {e}")
    
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

        logger.info(f"Next refresh scheduled at {target_datetime} ({hours} hours and {minutes} minutes from now)")
        
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
        logger.info("Starting scheduled refresh")
        try:
            success = await scanner.refresh_all_tickers()
            if success:
                logger.info("Scheduled refresh completed successfully")
            else:
                logger.warning("Scheduled refresh encountered errors")
        except asyncio.CancelledError:
            logger.info("Refresh cancelled")
            return
        except Exception as e:
            logger.error(f"Error during scheduled refresh: {e}")

async def run_market_regime_scheduler(scanner):
    """Run market regime scans on a schedule with immediate execution"""
    # Run immediate scan on startup
    logger.info("Starting immediate market regime scan")
    try:
        regime_info = await scanner.regime_scanner.scan_market_regime()
        if regime_info:
            logger.info(f"Initial market regime scan completed: {regime_info['regime_name']}")
        else:
            logger.warning("Initial market regime scan encountered errors")
    except asyncio.CancelledError:
        logger.info("Initial market regime scan cancelled")
        return
    except Exception as e:
        logger.error(f"Error during initial market regime scan: {e}")
    
    # Continue with scheduled scans
    while scanner.active and not scanner.shutdown_requested:
        # Calculate next run time (current time + interval)
        next_run = time.time() + config.REGIME_UPDATE_INTERVAL
        sleep_seconds = config.REGIME_UPDATE_INTERVAL
        
        logger.info(f"Next market regime scan in {sleep_seconds/3600:.1f} hours")
        
        # Wait until scheduled time, but check every second if we should stop
        while sleep_seconds > 0 and scanner.active and not scanner.shutdown_requested:
            try:
                # Sleep in small increments to be responsive to shutdown requests
                await asyncio.sleep(min(60, sleep_seconds))  # Check every minute
                sleep_seconds = max(0, next_run - time.time())
            except asyncio.CancelledError:
                logger.info("Market regime sleep interrupted by shutdown")
                return
            
        if not scanner.active or scanner.shutdown_requested:
            break
            
        # Run the regime scan
        logger.info("Starting scheduled market regime scan")
        try:
            regime_info = await scanner.regime_scanner.scan_market_regime()
            if regime_info:
                logger.info(f"Scheduled market regime scan completed: {regime_info['regime_name']}")
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
    parser = argparse.ArgumentParser(description='Stock Ticker Fetcher')
    parser.add_argument('--search', type=str, help='Search for a ticker by name or symbol')
    parser.add_argument('--history', type=str, help='Get history for a specific ticker')
    parser.add_argument('--list', action='store_true', help='List all active tickers')
    parser.add_argument('--regime', action='store_true', help='Get current market regime')
    parser.add_argument('--regime-history', type=int, default=30, help='Get market regime history for specified days')
    args = parser.parse_args()
    
    # Handle market regime queries first (they don't need the full scanner)
    if args.regime or args.regime_history:
        regime_scanner = MarketRegimeScanner()
        regime_scanner.start()
        
        if args.regime:
            regime = regime_scanner.get_latest_regime()
            if regime:
                print(f"Current Market Regime: {regime.get('regime_name', 'Unknown')}")
                print(f"Probability: {regime.get('regime_probability', 0):.2f}")
                print(f"Volatility: {regime.get('volatility', 0):.4f}")
                print(f"Trend Strength: {regime.get('trend_strength', 0):.2f}")
                print(f"Market Health: {regime.get('market_health', 0):.2f}")
            else:
                print("No market regime data available. Running a scan...")
                regime_info = await regime_scanner.scan_market_regime()
                if regime_info:
                    print(f"Current Market Regime: {regime_info['regime_name']}")
                    print(f"Probability: {regime_info['regime_probability']:.2f}")
                    print(f"Volatility: {regime_info['volatility']:.4f}")
                    print(f"Trend Strength: {regime_info['trend_strength']:.2f}")
                    print(f"Market Health: {regime_info['market_health']:.2f}")
                else:
                    print("Failed to get market regime data.")
        
        if args.regime_history:
            history = regime_scanner.get_regime_history(args.regime_history)
            if history:
                print(f"Market Regime History (last {args.regime_history} days):")
                for day in history:
                    date_str = day['date'] if isinstance(day['date'], str) else day['date'].split(' ')[0]
                    print(f"{date_str}: Regime {day['regime']} "
                          f"(Prob: {day.get('regime_probability', 0):.2f}, "
                          f"Vol: {day.get('volatility', 0):.4f})")
            else:
                print("No market regime history available.")
        
        regime_scanner.stop()
        return
    
    # Handle other ticker-related queries
    scanner = PolygonTickerScanner()
    
    if args.search:
        results = scanner.search_tickers_db(args.search)
        if results:
            print(f"Found {len(results)} matching tickers:")
            for result in results:
                print(f"{result['ticker']}: {result['name']} ({result['primary_exchange']})")
        else:
            print("No matching tickers found")
        return
    
    if args.history:
        results = scanner.get_ticker_history_db(args.history)
        if results:
            print(f"History for {args.history}:")
            for result in results:
                print(f"{result['change_date']}: {result['change_type']}")
        else:
            print(f"No history found for {args.history}")
        return
    
    if args.list:
        results = scanner.db.get_all_active_tickers()
        if results:
            print(f"Found {len(results)} active tickers:")
            for result in results:
                print(f"{result['ticker']}: {result['name']} ({result['primary_exchange']})")
        else:
            print("No active tickers found")
        return
    
    # Normal operation
    scanner.start()
    
    # Wait for initial cache load
    await asyncio.get_event_loop().run_in_executor(None, scanner.initial_refresh_complete.wait)
    
    # Create tasks for both schedulers
    ticker_scheduler_task = asyncio.create_task(run_scheduled_refresh(scanner))
    regime_scheduler_task = asyncio.create_task(run_market_regime_scheduler(scanner))
    
    # Set up signal handlers
    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()
    
    def signal_handler():
        """Handle shutdown signals immediately"""
        print("\nReceived interrupt signal, shutting down...")
        scanner.stop()
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
        
        # Cancel all tasks if they're still running
        for task in [ticker_scheduler_task, regime_scheduler_task, stop_task]:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                
    except asyncio.CancelledError:
        logger.info("Main task cancelled")
    finally:
        # Shutdown the scanner
        await scanner.shutdown()

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