import numpy as np
import pandas as pd
import asyncio
import aiohttp
import time
import os
import logging
import json
import threading
from datetime import datetime, timedelta, date
from urllib.parse import urlencode
from threading import Lock, Event, RLock
from collections import defaultdict
import sys
import signal
from tzlocal import get_localzone
import contextlib
from typing import List, Dict, Optional, Any, Tuple
import argparse
import asyncpg
from asyncpg.pool import Pool
import pandas_market_calendars as mcal
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
import uuid

# ======================== HMM IMPORTS ======================== #
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ======================== CUSTOM EXCEPTIONS ======================== #
class TickerScannerError(Exception):
    """Base exception for Ticker Scanner"""
    pass

class DatabaseError(TickerScannerError):
    """Database related errors"""
    pass

class APIError(TickerScannerError):
    """API related errors"""
    pass

class ConfigurationError(TickerScannerError):
    """Configuration related errors"""
    pass

class CircuitBreakerError(TickerScannerError):
    """Circuit breaker related errors"""
    pass

# ======================== ERROR HANDLING DECORATOR ======================== #
def handle_errors(max_retries=3, retry_delay=1):
    """Unified error handling decorator for both sync and async methods"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            retries = 0
            while retries <= max_retries:
                try:
                    return await func(*args, **kwargs)
                except (APIError, DatabaseError) as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Max retries exceeded for {func.__name__}: {e}")
                        raise
                    logger.warning(f"Retry {retries}/{max_retries} for {func.__name__} after error: {e}")
                    await asyncio.sleep(retry_delay * retries)
                except Exception as e:
                    logger.error(f"Unexpected error in {func.__name__}: {e}")
                    raise
            return None

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            retries = 0
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except (APIError, DatabaseError) as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Max retries exceeded for {func.__name__}: {e}")
                        raise
                    logger.warning(f"Retry {retries}/{max_retries} for {func.__name__} after error: {e}")
                    time.sleep(retry_delay * retries)
                except Exception as e:
                    logger.error(f"Unexpected error in {func.__name__}: {e}")
                    raise
            return None

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# ======================== CONFIGURATION ======================== #
class Config:
    # API Configuration - Use environment variables with fallbacks
    POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "ld1Poa63U6t4Y2MwOCA2JeKQyHVrmyg8")
    
    # Scanner Configuration - Using ONLY NASDAQ Exchange (XNAS)
    EXCHANGES = json.loads(os.getenv("EXCHANGES", '["XNAS"]'))  # NASDAQ Exchange only
    MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "100"))
    RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "0.02"))
    SCAN_TIME = os.getenv("SCAN_TIME", "08:30")
    
    # Error Handling Configuration
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY = int(os.getenv("RETRY_DELAY", "5"))
    CIRCUIT_THRESHOLD = int(os.getenv("CIRCUIT_THRESHOLD", "10"))
    CIRCUIT_TIMEOUT = int(os.getenv("CIRCUIT_TIMEOUT", "300"))  # 5 minutes
    
    # Database Configuration - PostgreSQL
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB = os.getenv("POSTGRES_DB", "stock_scanner")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "hodumaru")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "Leetkd214")
    
    # Market Calendar Configuration
    MARKET_CALENDAR = os.getenv("MARKET_CALENDAR", "NASDAQ")  # Use NASDAQ calendar
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv("LOG_FORMAT", '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Thread Pool Configuration
    THREAD_POOL_SIZE = int(os.getenv("THREAD_POOL_SIZE", "10"))
    
    # HMM Configuration
    HMM_N_REGIMES_RANGE = list(range(2, 6))  # Test 2 to 5 regimes
    HMM_DAYS_OF_DATA = int(os.getenv("HMM_DAYS_OF_DATA", "365"))

# Initialize configuration
config = Config()

# ======================== LOGGING SETUP ======================== #
def setup_logging():
    """Configure logging with file and console handlers"""
    os.makedirs("logs", exist_ok=True)
    
    logger = logging.getLogger("TickerScanner")
    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter(config.LOG_FORMAT)
    
    # File handler with rotation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/ticker_scanner_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Error handler
    error_handler = logging.FileHandler("logs/errors.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    return logger

logger = setup_logging()

# ======================== PERFORMANCE MONITORING ======================== #
def monitor_performance(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    async def async_wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = await func(self, *args, **kwargs)
        end_time = time.time()
        
        duration = end_time - start_time
        logger.info(f"{func.__name__} executed in {duration:.2f}s")
        
        # Track metrics
        if hasattr(self, 'performance_metrics'):
            if func.__name__ not in self.performance_metrics:
                self.performance_metrics[func.__name__] = {
                    'total_duration': 0,
                    'count': 0,
                    'last_execution': 0
                }
            
            self.performance_metrics[func.__name__]['total_duration'] += duration
            self.performance_metrics[func.__name__]['count'] += 1
            self.performance_metrics[func.__name__]['last_execution'] = end_time
        
        return result
    
    @wraps(func)
    def sync_wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        
        duration = end_time - start_time
        logger.info(f"{func.__name__} executed in {duration:.2f}s")
        
        # Track metrics
        if hasattr(self, 'performance_metrics'):
            if func.__name__ not in self.performance_metrics:
                self.performance_metrics[func.__name__] = {
                    'total_duration': 0,
                    'count': 0,
                    'last_execution': 0
                }
            
            self.performance_metrics[func.__name__]['total_duration'] += duration
            self.performance_metrics[func.__name__]['count'] += 1
            self.performance_metrics[func.__name__]['last_execution'] = end_time
        
        return result
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

# ======================== BASE DATABASE MANAGER ======================== #
class BaseDatabaseManager:
    """Base class for database managers with common functionality - Async version"""
    
    def __init__(self, minconn, maxconn):
        try:
            self.pool = None
            self.minconn = minconn
            self.maxconn = maxconn
            self._init_lock = asyncio.Lock()
            self.performance_metrics = {}
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")

    async def initialize(self):
        """Initialize the connection pool asynchronously"""
        async with self._init_lock:
            if self.pool is None:
                try:
                    self.pool = await asyncpg.create_pool(
                        min_size=self.minconn,
                        max_size=self.maxconn,
                        host=config.POSTGRES_HOST,
                        port=config.POSTGRES_PORT,
                        database=config.POSTGRES_DB,
                        user=config.POSTGRES_USER,
                        password=config.POSTGRES_PASSWORD
                    )
                    await self._init_database()
                except Exception as e:
                    logger.error(f"Database connection failed: {e}")
                    raise DatabaseError(f"Database connection failed: {e}")

    @contextlib.asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool with proper error handling"""
        if self.pool is None:
            await self.initialize()
            
        conn = None
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                conn = await self.pool.acquire()
                # Validate connection is still open
                await conn.execute("SELECT 1")
                yield conn
                break
            except (asyncpg.PostgresConnectionError, asyncpg.InterfaceError) as e:
                logger.warning(f"Database connection error (attempt {retry_count+1}): {e}")
                if conn:
                    try:
                        await self.pool.release(conn)
                    except:
                        pass
                retry_count += 1
                if retry_count >= max_retries:
                    raise DatabaseError(f"Failed to get valid connection after {max_retries} attempts")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Database error: {e}")
                if conn:
                    try:
                        await self.pool.release(conn)
                    except:
                        pass
                raise DatabaseError(f"Database error: {e}")
            finally:
                if conn and not conn.is_closed() and retry_count < max_retries:
                    try:
                        await self.pool.release(conn)
                    except Exception as e:
                        logger.error(f"Error returning connection to pool: {e}")

    async def close_all_connections(self):
        """Close all connections in the pool"""
        if self.pool:
            await self.pool.close()

    async def _init_database(self):
        """Initialize database tables - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _init_database")

    def _convert_numpy_types(self, params):
        """Convert numpy data types to native Python types for database compatibility"""
        converted_params = []
        for param in params:
            if isinstance(param, np.integer):
                converted_params.append(int(param))
            elif isinstance(param, np.floating):
                converted_params.append(float(param))
            else:
                converted_params.append(param)
        return tuple(converted_params)
    
    @monitor_performance
    @handle_errors()
    async def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        try:
            async with self.get_connection() as conn:
                records = await conn.fetch(query, *params)
                # Convert asyncpg Records to dictionaries with column names
                return [dict(record) for record in records]
        except Exception as e:
            logger.error(f"Database query error: {e}, Query: {query}, Params: {params}")
            raise DatabaseError(f"Query execution failed: {e}")
            
    @monitor_performance
    @handle_errors()
    async def execute_write(self, query: str, params: tuple = ()) -> int:
        try:
            converted_params = self._convert_numpy_types(params)
            async with self.get_connection() as conn:
                result = await conn.execute(query, *converted_params)
                # For INSERT/UPDATE/DELETE, result is a string like "INSERT 0 1"
                # We need to parse this to get the row count
                if "INSERT" in result:
                    return int(result.split()[-1])
                elif "UPDATE" in result or "DELETE" in result:
                    return int(result.split()[-1])
                return 0
        except Exception as e:
            logger.error(f"Database write error: {e}, Query: {query}, Params: {converted_params}")
            raise DatabaseError(f"Write execution failed: {e}")
    
    async def get_pool_status(self):
        """Get connection pool status"""
        if self.pool:
            return {
                'min_size': self.pool.get_min_size(),
                'max_size': self.pool.get_max_size(),
                'size': self.pool.get_size(),
                'free': self.pool.get_free_size(),
            }
        return {}

# ======================== MAIN DATABASE MANAGER ======================== #
class DatabaseManager(BaseDatabaseManager):
    """Main database manager for ticker operations - Async version"""
    
    def __init__(self):
        super().__init__(minconn=3, maxconn=20)
    
    async def _init_database(self):
        async with self.get_connection() as conn:
            # Create tickers table
            await conn.execute('''
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
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Drop the existing historical_tickers table if it exists
            await conn.execute('DROP TABLE IF EXISTS historical_tickers CASCADE')
            
            # Create historical_tickers table with proper partitioning
            await conn.execute('''
                CREATE TABLE historical_tickers (
                    id SERIAL,
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
                    change_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (id, change_date)
                ) PARTITION BY RANGE (change_date)
            ''')
            
            # Create default partition
            await conn.execute('''
                CREATE TABLE historical_tickers_default 
                PARTITION OF historical_tickers DEFAULT
            ''')
            
            # Create market_regimes table for HMM results
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS market_regimes (
                    id SERIAL PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    n_regimes INTEGER NOT NULL,
                    current_regime INTEGER NOT NULL,
                    regime_stats JSONB,
                    regime_history JSONB,
                    features_date_range JSONB,
                    model_fit_date TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create market_analysis table for overall market analysis
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS market_analysis (
                    id SERIAL PRIMARY KEY,
                    analysis_date TIMESTAMP,
                    total_tickers_analyzed INTEGER,
                    regime_distribution JSONB,
                    most_common_regime INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes with partial indexes
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_tickers_exchange ON tickers(primary_exchange)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_tickers_active ON tickers(active) WHERE active = 1')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_tickers_updated_at ON tickers(updated_at)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_historical_tickers_date ON historical_tickers(change_date)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_historical_tickers_ticker_date ON historical_tickers(ticker, change_date)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_historical_tickers_change_type ON historical_tickers(change_type) WHERE change_type IS NOT NULL')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_market_regimes_ticker ON market_regimes(ticker)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_market_regimes_date ON market_regimes(created_at)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_market_analysis_date ON market_analysis(analysis_date)')
            
            logger.info("Database tables initialized successfully")

    @monitor_performance
    @handle_errors()
    async def upsert_tickers(self, tickers: List[Dict]) -> Tuple[int, int]:
        if not tickers:
            return 0, 0
            
        # Prepare arrays for bulk operation
        tickers_arr = []
        names_arr = []
        primary_exchange_arr = []
        last_updated_utc_arr = []
        type_arr = []
        market_arr = []
        locale_arr = []
        currency_name_arr = []
        active_arr = []
        
        for t in tickers:
            tickers_arr.append(t['ticker'])
            names_arr.append(t.get('name'))
            primary_exchange_arr.append(t.get('primary_exchange'))
            last_updated_utc_arr.append(t.get('last_updated_utc'))
            type_arr.append(t.get('type'))
            market_arr.append(t.get('market'))
            locale_arr.append(t.get('locale'))
            currency_name_arr.append(t.get('currency_name'))
            active_arr.append(1)
        
        inserted = 0
        updated = 0
        
        async with self.get_connection() as conn:
            # Use transaction for better performance
            async with conn.transaction():
                try:
                    # Get existing tickers
                    placeholders = ','.join(['$' + str(i+1) for i in range(len(tickers_arr))])
                    existing = await conn.fetch(
                        f"SELECT ticker FROM tickers WHERE ticker IN ({placeholders})", 
                        *tickers_arr
                    )
                    existing_tickers = {row['ticker'] for row in existing}
                    
                    # Bulk upsert using UNNEST
                    result = await conn.fetchrow('''
                        WITH input_data AS (
                            SELECT 
                                unnest($1::text[]) AS ticker,
                                unnest($2::text[]) AS name,
                                unnest($3::text[]) AS primary_exchange,
                                unnest($4::text[]) AS last_updated_utc,
                                unnest($5::text[]) AS type,
                                unnest($6::text[]) AS market,
                                unnest($7::text[]) AS locale,
                                unnest($8::text[]) AS currency_name,
                                unnest($9::int[]) AS active
                        ),
                        updated AS (
                            UPDATE tickers t
                            SET 
                                name = i.name,
                                primary_exchange = i.primary_exchange,
                                last_updated_utc = i.last_updated_utc,
                                type = i.type,
                                market = i.market,
                                locale = i.locale,
                                currency_name = i.currency_name,
                                active = i.active,
                                updated_at = CURRENT_TIMESTAMP
                            FROM input_data i
                            WHERE t.ticker = i.ticker
                            RETURNING t.ticker
                        ),
                        inserted AS (
                            INSERT INTO tickers 
                                (ticker, name, primary_exchange, last_updated_utc, 
                                 type, market, locale, currency_name, active)
                            SELECT 
                                i.ticker, i.name, i.primary_exchange, i.last_updated_utc,
                                i.type, i.market, i.locale, i.currency_name, i.active
                            FROM input_data i
                            WHERE i.ticker NOT IN (SELECT ticker FROM updated)
                            RETURNING ticker
                        )
                        SELECT 
                            (SELECT COUNT(*) FROM inserted) AS inserted_count,
                            (SELECT COUNT(*) FROM updated) AS updated_count
                    ''', tickers_arr, names_arr, primary_exchange_arr, last_updated_utc_arr,
                    type_arr, market_arr, locale_arr, currency_name_arr, active_arr)
                    
                    inserted = result['inserted_count'] if result else 0
                    updated = result['updated_count'] if result else 0
                    
                    # Bulk insert historical records
                    historical_data = []
                    for t in tickers:
                        change_type = 'added' if t['ticker'] not in existing_tickers else 'updated'
                        historical_data.append((
                            t['ticker'], t.get('name'), t.get('primary_exchange'), 
                            t.get('last_updated_utc'), t.get('type'), t.get('market'),
                            t.get('locale'), t.get('currency_name'), 1, change_type
                        ))
                    
                    # Use execute many for bulk insert
                    await conn.executemany(
                        '''
                        INSERT INTO historical_tickers 
                        (ticker, name, primary_exchange, last_updated_utc, 
                         type, market, locale, currency_name, active, change_type)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        ''',
                        historical_data
                    )
                    
                except Exception as e:
                    logger.error(f"Transaction failed during ticker upsert: {e}")
                    raise DatabaseError(f"Transaction failed during ticker upsert: {e}")
            
        return inserted, updated

    @handle_errors()
    async def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key"""
        result = await self.execute_query(
            "SELECT value FROM metadata WHERE key = $1",
            (key,)
        )
        
        if result:
            value = result[0]['value']
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        return default
    
    @handle_errors()
    async def update_metadata(self, key: str, value: Any) -> None:
        """Update metadata key-value pair"""
        await self.execute_write(
            "INSERT INTO metadata (key, value) VALUES ($1, $2) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = CURRENT_TIMESTAMP",
            (key, json.dumps(value) if isinstance(value, (list, dict)) else str(value))
        )

    @handle_errors()
    async def get_all_active_tickers(self) -> List[Dict]:
        """Get all active tickers from the database"""
        return await self.execute_query(
            "SELECT * FROM tickers WHERE active = 1 ORDER BY ticker"
        )

    @handle_errors()
    async def search_tickers(self, search_term: str, limit: int = 50) -> List[Dict]:
        """Search tickers by name or symbol"""
        return await self.execute_query(
            "SELECT * FROM tickers WHERE (ticker ILIKE $1 OR name ILIKE $2) AND active = 1 ORDER BY ticker LIMIT $3",
            (f"%{search_term}%", f"%{search_term}%", limit)
        )

    @handle_errors()
    async def get_ticker_history(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Get historical changes for a ticker"""
        return await self.execute_query(
            "SELECT * FROM historical_tickers WHERE ticker = $1 ORDER by change_date DESC LIMIT $2",
            (ticker, limit)
        )

    @monitor_performance
    @handle_errors()
    async def mark_tickers_inactive(self, tickers: List[str]) -> int:
        """Mark tickers as inactive using bulk operations"""
        if not tickers:
            return 0
            
        marked = 0
        
        async with self.get_connection() as conn:
            # Get current data for tickers to be marked inactive
            rows = await conn.fetch(
                "SELECT * FROM tickers WHERE ticker = ANY($1) AND active = 1", 
                tickers
            )
            
            if not rows:
                return 0
                
            # Use transaction for better performance
            async with conn.transaction():
                try:
                    # Bulk update to mark as inactive
                    update_result = await conn.execute(
                        "UPDATE tickers SET active = 0, updated_at = CURRENT_TIMESTAMP WHERE ticker = ANY($1)",
                        tickers
                    )
                    
                    marked = int(update_result.split()[-1]) if "UPDATE" in update_result else 0
                    
                    # Bulk insert historical records
                    historical_data = []
                    for row in rows:
                        # Convert asyncpg Record to dict
                        row_dict = dict(row)
                        historical_data.append((
                            row_dict['ticker'], row_dict.get('name'), row_dict.get('primary_exchange'), 
                            row_dict.get('last_updated_utc'), row_dict.get('type'), row_dict.get('market'),
                            row_dict.get('locale'), row_dict.get('currency_name'), 0, 'removed'
                        ))
                    
                    await conn.executemany(
                        '''
                        INSERT INTO historical_tickers 
                        (ticker, name, primary_exchange, last_updated_utc, 
                        type, market, locale, currency_name, active, change_type)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        ''',
                        historical_data
                    )
                    
                except Exception as e:
                    logger.error(f"Transaction failed during mark inactive: {e}")
                    raise DatabaseError(f"Transaction failed during mark inactive: {e}")
                
        return marked

    @handle_errors()
    async def get_ticker_details(self, ticker: str) -> Optional[Dict]:
        """Get details for a specific ticker"""
        result = await self.execute_query(
            "SELECT * FROM tickers WHERE ticker = $1", 
            (ticker,)
        )
        return result[0] if result else None

# ======================== MARKET REGIME DETECTOR ======================== #
class MarketRegimeDetector:
    """
    Market Regime Detection using Hidden Markov Models with BIC model selection
    """
    
    def __init__(self, db_manager: DatabaseManager, n_regimes_range: range = None):
        self.db = db_manager
        self.n_regimes_range = n_regimes_range or config.HMM_N_REGIMES_RANGE
        self.model = None
        self.scaler = StandardScaler()
        self.regime_labels = None
        self.best_n_regimes = None
        self.feature_columns = [
            'returns', 'volatility', 'volume_change', 'high_low_ratio',
            'price_range', 'momentum', 'rsi'
        ]
        self.performance_metrics = {}
        
    async def fetch_price_data(self, ticker: str, days: int = None) -> pd.DataFrame:
        """Fetch historical price data for a ticker"""
        if days is None:
            days = config.HMM_DAYS_OF_DATA
            
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            async with aiohttp.ClientSession() as session:
                url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}?adjusted=true&sort=asc&limit={days}&apiKey={config.POLYGON_API_KEY}"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('results'):
                            df = pd.DataFrame(data['results'])
                            # Polygon.io response format
                            df.rename(columns={
                                'o': 'open',
                                'h': 'high', 
                                'l': 'low',
                                'c': 'close',
                                'v': 'volume',
                                't': 'timestamp'
                            }, inplace=True)
                            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                            df.set_index('timestamp', inplace=True)
                            return df
                    else:
                        logger.warning(f"API returned status {response.status} for {ticker}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching price data for {ticker}: {e}")
            return pd.DataFrame()
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical features for regime detection"""
        if df.empty:
            return pd.DataFrame()
            
        features = pd.DataFrame(index=df.index)
        
        # Price returns
        features['returns'] = df['close'].pct_change()
        
        # Volatility (rolling standard deviation of returns)
        features['volatility'] = features['returns'].rolling(window=20).std()
        
        # Volume changes
        features['volume_change'] = df['volume'].pct_change()
        
        # High-Low ratio
        features['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        
        # Price range (normalized)
        features['price_range'] = (df['high'] - df['low']) / df['low']
        
        # Momentum
        features['momentum'] = df['close'] / df['close'].shift(5) - 1
        
        # RSI (simplified)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Drop NaN values
        features = features.dropna()
        
        return features
    
    def find_optimal_regimes(self, features: pd.DataFrame) -> int:
        """Find optimal number of regimes using BIC"""
        if features.empty or len(features) < 50:
            return min(self.n_regimes_range)
            
        X = self.scaler.fit_transform(features[self.feature_columns])
        bic_scores = []
        models = []
        
        for n_components in self.n_regimes_range:
            try:
                # Create and fit HMM
                model = hmm.GaussianHMM(
                    n_components=n_components,
                    covariance_type="full",
                    n_iter=1000,
                    random_state=42
                )
                model.fit(X)
                
                # Calculate BIC
                log_likelihood = model.score(X)
                n_params = n_components * (n_components - 1) + 2 * n_components * features.shape[1]
                bic = -2 * log_likelihood + n_params * np.log(len(X))
                
                bic_scores.append(bic)
                models.append(model)
                
                logger.info(f"n_components={n_components}, BIC={bic:.2f}, log_likelihood={log_likelihood:.2f}")
                
            except Exception as e:
                logger.warning(f"Failed to fit HMM with {n_components} components: {e}")
                bic_scores.append(np.inf)
                models.append(None)
        
        # Find model with minimum BIC
        if all(np.isinf(bic_scores)):
            best_n = min(self.n_regimes_range)
        else:
            best_n = self.n_regimes_range[np.argmin(bic_scores)]
        
        self.best_n_regimes = best_n
        logger.info(f"Optimal number of regimes: {best_n}")
        
        return best_n
    
    def fit_hmm(self, features: pd.DataFrame, n_regimes: int = None) -> hmm.GaussianHMM:
        """Fit HMM with specified number of regimes"""
        if features.empty:
            return None
            
        if n_regimes is None:
            n_regimes = self.best_n_regimes or self.find_optimal_regimes(features)
        
        X = self.scaler.fit_transform(features[self.feature_columns])
        
        try:
            self.model = hmm.GaussianHMM(
                n_components=n_regimes,
                covariance_type="full",
                n_iter=1000,
                random_state=42
            )
            self.model.fit(X)
            
            # Get regime labels
            self.regime_labels = self.model.predict(X)
            
            # Analyze regime characteristics
            self._analyze_regimes(features, self.regime_labels)
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error fitting HMM: {e}")
            return None
    
    def _analyze_regimes(self, features: pd.DataFrame, regimes: np.ndarray):
        """Analyze characteristics of each regime"""
        regime_stats = {}
        
        for regime in np.unique(regimes):
            regime_mask = regimes == regime
            regime_data = features[regime_mask]
            
            stats = {
                'count': len(regime_data),
                'return_mean': regime_data['returns'].mean(),
                'return_std': regime_data['returns'].std(),
                'volatility_mean': regime_data['volatility'].mean(),
                'volume_change_mean': regime_data['volume_change'].mean(),
                'rsi_mean': regime_data['rsi'].mean()
            }
            
            # Classify regime type based on characteristics
            if stats['volatility_mean'] > features['volatility'].median():
                if stats['return_mean'] > 0:
                    regime_type = "High Volatility Bull"
                else:
                    regime_type = "High Volatility Bear"
            else:
                if stats['return_mean'] > 0:
                    regime_type = "Low Volatility Bull"
                else:
                    regime_type = "Low Volatility Bear"
            
            stats['regime_type'] = regime_type
            regime_stats[regime] = stats
            
            logger.info(f"Regime {regime} ({regime_type}): {stats}")
        
        self.regime_stats = regime_stats
    
    def predict_regime(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regimes for new data"""
        if self.model is None or features.empty:
            return np.array([])
            
        try:
            X = self.scaler.transform(features[self.feature_columns])
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error predicting regimes: {e}")
            return np.array([])
    
    @monitor_performance
    @handle_errors(max_retries=3, retry_delay=2)
    async def detect_market_regime(self, ticker: str, days: int = None) -> Dict[str, Any]:
        """Main method to detect market regime for a ticker"""
        logger.info(f"Detecting market regime for {ticker}")
        
        try:
            # Fetch price data
            price_data = await self.fetch_price_data(ticker, days)
            if price_data.empty:
                logger.warning(f"No price data available for {ticker}")
                return {}
            
            # Calculate features
            features = self.calculate_features(price_data)
            if features.empty:
                logger.warning(f"No features calculated for {ticker}")
                return {}
            
            # Find optimal number of regimes
            n_regimes = self.find_optimal_regimes(features)
            
            # Fit HMM
            model = self.fit_hmm(features, n_regimes)
            if model is None:
                logger.warning(f"Failed to fit HMM for {ticker}")
                return {}
            
            # Get current regime
            current_features = features.iloc[-1:][self.feature_columns]
            current_regime = self.predict_regime(current_features)
            
            # Prepare results
            result = {
                'ticker': ticker,
                'n_regimes': n_regimes,
                'current_regime': int(current_regime[0]) if len(current_regime) > 0 else -1,
                'regime_stats': self.regime_stats,
                'regime_history': self.regime_labels.tolist(),
                'features_date_range': {
                    'start': features.index[0].strftime('%Y-%m-%d'),
                    'end': features.index[-1].strftime('%Y-%m-%d')
                },
                'model_fit_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info(f"Market regime detection completed for {ticker}: Regime {result['current_regime']}")
            
            # Store results in database
            await self.store_regime_results(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in market regime detection for {ticker}: {e}")
            return {}
    
    async def store_regime_results(self, result: Dict[str, Any]):
        """Store regime detection results in database"""
        try:
            await self.db.execute_write('''
                INSERT INTO market_regimes 
                (ticker, n_regimes, current_regime, regime_stats, regime_history, features_date_range, model_fit_date)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            ''', (
                result['ticker'],
                result['n_regimes'],
                result['current_regime'],
                json.dumps(result['regime_stats']),
                json.dumps(result['regime_history']),
                json.dumps(result['features_date_range']),
                result['model_fit_date']
            ))
            
            logger.info(f"Stored regime results for {result['ticker']} in database")
            
        except Exception as e:
            logger.error(f"Error storing regime results: {e}")
    
    async def get_regime_history(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Get regime history for a ticker"""
        try:
            return await self.db.execute_query(
                "SELECT * FROM market_regimes WHERE ticker = $1 ORDER BY created_at DESC LIMIT $2",
                (ticker, limit)
            )
        except Exception as e:
            logger.error(f"Error fetching regime history: {e}")
            return []

# ======================== TICKER SCANNER ======================== #
class PolygonTickerScanner:
    def __init__(self):
        self.api_key = config.POLYGON_API_KEY
        self.base_url = "https://api.polygon.io/v3/reference/tickers"
        # Use ONLY NASDAQ Exchange (XNAS)
        self.exchanges = config.EXCHANGES
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
        self.db = DatabaseManager()
        self.shutdown_requested = False
        # Market calendar
        self.market_calendar = mcal.get_calendar(config.MARKET_CALENDAR)
        # Circuit breaker attributes
        self.api_error_count = 0
        self.circuit_open = False
        self.circuit_open_time = 0
        self.CIRCUIT_THRESHOLD = config.CIRCUIT_THRESHOLD
        self.CIRCUIT_TIMEOUT = config.CIRCUIT_TIMEOUT
        # Performance metrics
        self.performance_metrics = {}
        # Async lock for thread-safe operations
        self._async_lock = asyncio.Lock()
        
        logger.info(f"Using local timezone: {self.local_tz}")
        logger.info(f"Using PostgreSQL database: {config.POSTGRES_DB}")
        logger.info(f"Using exchanges: {', '.join(self.exchanges)}")
        logger.info(f"Using market calendar: {config.MARKET_CALENDAR}")
        
    async def _init_cache(self):
        """Initialize or load ticker cache from database"""
        try:
            self.last_refresh_time = await self.db.get_metadata('last_refresh_time', 0)
            
            # Load active tickers from database
            db_tickers = await self.db.get_all_active_tickers()
            
            if db_tickers:
                # Convert to DataFrame with proper column names
                self.ticker_cache = pd.DataFrame(db_tickers)
                logger.info(f"Loaded {len(self.ticker_cache)} tickers from database")
                
                # Find the ticker column (case insensitive)
                column_names_lower = [str(col).lower() for col in self.ticker_cache.columns]
                if 'ticker' in column_names_lower:
                    ticker_col_idx = column_names_lower.index('ticker')
                    ticker_col = self.ticker_cache.columns[ticker_col_idx]
                    self.current_tickers_set = set(self.ticker_cache[ticker_col].tolist())
                else:
                    # Try to find a column that might contain ticker symbols
                    for col in self.ticker_cache.columns:
                        if any(keyword in str(col).lower() for keyword in ['symbol', 'ticker', 'code', 'id']):
                            self.current_tickers_set = set(self.ticker_cache[col].tolist())
                            logger.warning(f"Using '{col}' as ticker column")
                            break
                    else:
                        # Use the first column as fallback
                        ticker_col = self.ticker_cache.columns[0]
                        self.current_tickers_set = set(self.ticker_cache[ticker_col].tolist())
                        logger.warning(f"Using first column '{ticker_col}' as ticker column")
            else:
                self.ticker_cache = pd.DataFrame(columns=[
                    "ticker", "name", "primary_exchange", "last_updated_utc", "type", "market", "locale"
                ])
                logger.info("No tickers found in database")
                self.current_tickers_set = set()
            
            # Load known missing tickers from database
            missing_tickers = await self.db.get_metadata('known_missing_tickers', [])
            self.known_missing_tickers = set(missing_tickers) if missing_tickers else set()
            
            self.initial_refresh_complete.set()
            logger.info(f"Cache initialized with {len(self.current_tickers_set)} tickers")
        except Exception as e:
            logger.error(f"Failed to initialize cache: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise DatabaseError(f"Cache initialization failed: {e}")

    @handle_errors()
    def is_trading_day(self, date):
        """Check if a date is a trading day using market calendar"""
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d").date()
        elif isinstance(date, datetime):
            date = date.date()
            
        schedule = self.market_calendar.schedule(start_date=date, end_date=date)
        return not schedule.empty

    @handle_errors(max_retries=config.MAX_RETRIES, retry_delay=config.RETRY_DELAY)
    async def _call_polygon_api(self, session, url, retry_count=0):
        """Make API call with retry logic and rate limiting"""
        # Check for shutdown before making the request
        if self.shutdown_requested:
            return None
            
        # Check circuit breaker
        if self.circuit_open:
            if time.time() - self.circuit_open_time < self.CIRCUIT_TIMEOUT:
                logger.warning("Circuit breaker is open, skipping API call")
                raise CircuitBreakerError("Circuit breaker is open")
            else:
                logger.info("Circuit breaker timeout elapsed, trying again")
                self.circuit_open = False
                self.api_error_count = 0
                
        if retry_count >= config.MAX_RETRIES:
            logger.error(f"Max retries ({config.MAX_RETRIES}) exceeded for URL: {url}")
            raise APIError(f"Max retries exceeded for URL: {url}")
            
        async with self.semaphore:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        # Reset error count on success
                        self.api_error_count = 0
                        return await response.json()
                    elif response.status == 429:
                        retry_after = int(response.headers.get('Retry-After', config.RETRY_DELAY))
                        logger.warning(f"Rate limit hit, retrying after {retry_after} seconds (attempt {retry_count+1})")
                        await asyncio.sleep(retry_after)
                        return await self._call_polygon_api(session, url, retry_count+1)
                    elif response.status >= 500:
                        self.api_error_count += 1
                        logger.warning(f"Server error {response.status}, retrying in {config.RETRY_DELAY}s (attempt {retry_count+1})")
                        await asyncio.sleep(config.RETRY_DELAY)
                        return await self._call_polygon_api(session, url, retry_count+1)
                    else:
                        self.api_error_count += 1
                        logger.error(f"API request failed: {response.status} for URL: {url}")
                        raise APIError(f"API request failed: {response.status} for URL: {url}")
            except asyncio.TimeoutError:
                self.api_error_count += 1
                logger.warning(f"Timeout for URL: {url}, retrying (attempt {retry_count+1})")
                await asyncio.sleep(config.RETRY_DELAY)
                return await self._call_polygon_api(session, url, retry_count+1)
            except aiohttp.ClientError as e:
                self.api_error_count += 1
                logger.error(f"Client error for URL {url}: {e}, retrying (attempt {retry_count+1})")
                await asyncio.sleep(config.RETRY_DELAY)
                return await self._call_polygon_api(session, url, retry_count+1)
            except Exception as e:
                self.api_error_count += 1
                logger.error(f"Unexpected error for URL {url}: {e}")
                raise APIError(f"Unexpected error for URL {url}: {e}")
            finally:
                # Check if we need to open the circuit breaker
                if self.api_error_count >= self.CIRCUIT_THRESHOLD and not self.circuit_open:
                    self.circuit_open = True
                    self.circuit_open_time = time.time()
                    logger.error(f"Circuit breaker opened due to excessive errors ({self.api_error_count})")
                    raise CircuitBreakerError(f"Circuit breaker opened due to excessive errors ({self.api_error_count})")

    @handle_errors()
    def _validate_ticker_data(self, ticker_data):
        """Validate ticker data before processing"""
        required_fields = ['ticker']
        validated_data = []
        
        for ticker in ticker_data:
            # Check required fields
            if not all(field in ticker for field in required_fields):
                logger.warning(f"Skipping invalid ticker data: {ticker}")
                continue
                
            # Sanitize fields
            sanitized = {
                'ticker': str(ticker.get('ticker', '')).strip(),
                'name': str(ticker.get('name', '')).strip() if ticker.get('name') else None,
                'primary_exchange': str(ticker.get('primary_exchange', '')).strip() if ticker.get('primary_exchange') else None,
                'last_updated_utc': str(ticker.get('last_updated_utc', '')).strip() if ticker.get('last_updated_utc') else None,
                'type': str(ticker.get('type', '')).strip() if ticker.get('type') else None,
                'market': str(ticker.get('market', '')).strip() if ticker.get('market') else None,
                'locale': str(ticker.get('locale', '')).strip() if ticker.get('locale') else None,
                'currency_name': str(ticker.get('currency_name', '')).strip() if ticker.get('currency_name') else None,
            }
            
            # Skip empty tickers
            if not sanitized['ticker']:
                continue
                
            validated_data.append(sanitized)
        
        return validated_data

    @monitor_performance
    @handle_errors(max_retries=config.MAX_RETRIES, retry_delay=config.RETRY_DELAY)
    async def _fetch_exchange_tickers(self, session, exchange):
        """Fetch all tickers for a specific exchange"""
        logger.info(f"Fetching tickers for exchange {exchange}")
        all_results = []
        next_url = None
        page_count = 0
        max_pages = 50  # Safety limit
        
        # Use current date
        date_param = datetime.now().strftime("%Y-%m-%d")
            
        # API parameters for exchange
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
        
        while url and page_count < max_pages and not self.shutdown_requested:
            data = await self._call_polygon_api(session, url)
            if not data or self.shutdown_requested:
                break
                
            results = data.get("results", [])
            if not results:
                logger.warning(f"No results in page {page_count + 1} for {exchange}")
                break
                
            # Filter for common stocks only and add exchange info
            stock_results = [
                {**r, "exchange": exchange} 
                for r in results 
                if r.get('type', '').upper() == 'CS'
            ]
            all_results.extend(stock_results)
            
            next_url = data.get("next_url", None)
            url = f"{next_url}&apiKey={self.api_key}" if next_url else None
            page_count += 1
            
            # Add progressive delay to avoid rate limiting
            delay = config.RATE_LIMIT_DELAY * (1 + page_count / 10)
            await asyncio.sleep(min(delay, 5.0))  # Cap at 5 seconds
        
        if page_count >= max_pages:
            logger.warning(f"Reached maximum page limit ({max_pages}) for {exchange}")
        
        if self.shutdown_requested:
            logger.info(f"Shutdown requested, aborting {exchange} fetch")
            return []
            
        logger.info(f"Completed {exchange}: {len(all_results)} stocks across {page_count} pages")
        return all_results

    @monitor_performance
    @handle_errors(max_retries=config.MAX_RETRIES, retry_delay=config.RETRY_DELAY)
    async def _refresh_all_tickers_async(self):
        """Refresh all tickers with parallel exchange processing"""
        start_time = time.time()
        
        logger.info("Starting full ticker refresh")
        
        # Check for shutdown before starting
        if self.shutdown_requested:
            logger.info("Shutdown requested, aborting refresh")
            return False
            
        try:
            async with aiohttp.ClientSession() as session:
                # Fetch all exchanges in parallel
                tasks = [self._fetch_exchange_tickers(session, exchange) for exchange in self.exchanges]
                exchange_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                all_results = []
                for i, results in enumerate(exchange_results):
                    if isinstance(results, Exception):
                        logger.error(f"Error fetching {self.exchanges[i]}: {results}")
                        continue
                    if results:
                        all_results.extend(results)
                
                # Check for shutdown after fetching
                if self.shutdown_requested:
                    logger.info("Shutdown requested during data processing")
                    return False
        except Exception as e:
            logger.error(f"Error during API fetch: {e}")
            # Fall back to database if API fails
            return await self._fallback_to_database()
        
        if not all_results:
            logger.warning("Refresh fetched no results")
            return await self._fallback_to_database()
            
        # Validate and sanitize data
        validated_results = self._validate_ticker_data(all_results)
        if not validated_results:
            logger.warning("No valid ticker data after validation")
            return await self._fallback_to_database()
            
        # Create DataFrame with only necessary columns
        new_df = pd.DataFrame(validated_results)[["ticker", "name", "primary_exchange", "last_updated_utc", "type", "market", "locale", "currency_name"]]
        new_tickers = set(new_df['ticker'].tolist())
        
        with self.cache_lock:
            # Original logic for live mode
            old_tickers = set(self.current_tickers_set)
            added = new_tickers - old_tickers
            removed = old_tickers - new_tickers
            
            # Convert DataFrame to list of dictionaries for database storage
            tickers_data = new_df.to_dict('records')
            
            # Update database using main db
            inserted, updated = await self.db.upsert_tickers(tickers_data)
            
            # Mark removed tickers as inactive
            if removed:
                marked_inactive = await self.db.mark_tickers_inactive(list(removed))
                logger.info(f"Marked {marked_inactive} tickers as inactive")
            
            # Update in-memory cache
            self.ticker_cache = new_df
            self.current_tickers_set = new_tickers
            
            # Update known missing tickers
            rediscovered = added & self.known_missing_tickers
            if rediscovered:
                self.known_missing_tickers -= rediscovered
                await self.db.update_metadata('known_missing_tickers', list(self.known_missing_tickers))
            
        self.last_refresh_time = time.time()
        
        await self.db.update_metadata('last_refresh_time', self.last_refresh_time)
        
        elapsed = time.time() - start_time
        logger.info(f"Ticker refresh completed in {elapsed:.2f}s")
        
        logger.info(f"Total: {len(new_df)} | Added: {len(added)} | Removed: {len(removed)}")
        logger.info(f"Database: {inserted} inserted, {updated} updated")
            
        return True

    @handle_errors()
    async def _fallback_to_database(self):
        """Fallback to database if API fails"""
        logger.info("Attempting database fallback")
        
        with self.cache_lock:
            db_tickers = await self.db.get_all_active_tickers()
            if db_tickers:
                self.ticker_cache = pd.DataFrame(db_tickers)
                self.current_tickers_set = set(self.ticker_cache['ticker'].tolist())
                logger.info(f"Fallback to database: loaded {len(self.ticker_cache)} tickers")
                return True
            else:
                logger.error("API failed and no fallback data available")
                return False

    @monitor_performance
    @handle_errors(max_retries=config.MAX_RETRIES, retry_delay=config.RETRY_DELAY)
    async def refresh_all_tickers(self):
        """Public async method to refresh tickers"""
        with self.refresh_lock:
            return await self._refresh_all_tickers_async()

    async def start(self):
        if not self.active:
            self.active = True
            self.shutdown_requested = False
            await self._init_cache()
            self.initial_refresh_complete.set()
            logger.info("Ticker scanner started")

    def stop(self):
        self.active = False
        self.shutdown_requested = True
        logger.info("Ticker scanner stopped")
        
    async def shutdown(self):
        """Cleanup resources"""
        self.stop()
        
        # Cancel all running tasks
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        
        # Wait for tasks to finish with timeout
        if tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for tasks to complete during shutdown")
        
        # Close database connections asynchronously
        await self.db.close_all_connections()
        
        logger.info("Ticker scanner shutdown complete")

    @handle_errors()
    def get_current_tickers_list(self):
        with self.cache_lock:
            return self.ticker_cache['ticker'].tolist()

    @handle_errors()
    def get_ticker_details(self, ticker):
        """Get details for a specific ticker from cache"""
        with self.cache_lock:
            result = self.ticker_cache[self.ticker_cache['ticker'] == ticker]
            return result.to_dict('records')[0] if not result.empty else None
            
    @handle_errors()
    async def search_tickers_db(self, search_term: str, limit: int = 50) -> List[Dict]:
        """Search tickers in database by name or symbol"""
        return await self.db.search_tickers(search_term, limit)
        
    @handle_errors()
    async def get_ticker_history_db(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Get historical changes for a ticker from database"""
        return await self.db.get_ticker_history(ticker, limit)

# ======================== ENHANCED SCANNER WITH MARKET REGIME DETECTION ======================== #
class EnhancedPolygonTickerScanner(PolygonTickerScanner):
    """Enhanced scanner with market regime detection"""
    
    def __init__(self):
        super().__init__()
        self.regime_detector = MarketRegimeDetector(self.db)
    
    async def analyze_market_regimes(self, tickers: List[str] = None, 
                                   sample_size: int = 50) -> Dict[str, Any]:
        """Analyze market regimes for multiple tickers"""
        if tickers is None:
            # Use a sample of tickers for analysis
            all_tickers = self.get_current_tickers_list()
            if len(all_tickers) > sample_size:
                tickers = np.random.choice(all_tickers, sample_size, replace=False).tolist()
            else:
                tickers = all_tickers
        
        logger.info(f"Analyzing market regimes for {len(tickers)} tickers")
        
        results = {}
        regime_summary = defaultdict(int)
        
        for ticker in tickers:
            try:
                regime_result = await self.regime_detector.detect_market_regime(ticker)
                if regime_result:
                    results[ticker] = regime_result
                    regime_summary[regime_result['current_regime']] += 1
                    
                    # Small delay to avoid rate limiting
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error analyzing regime for {ticker}: {e}")
                continue
        
        # Overall market analysis
        market_analysis = {
            'total_tickers_analyzed': len(results),
            'regime_distribution': dict(regime_summary),
            'most_common_regime': max(regime_summary, key=regime_summary.get) if regime_summary else -1,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info(f"Market regime analysis completed: {market_analysis}")
        
        # Store overall analysis
        await self.regime_detector.db.execute_write('''
            INSERT INTO market_analysis 
            (analysis_date, total_tickers_analyzed, regime_distribution, most_common_regime)
            VALUES ($1, $2, $3, $4)
        ''', (
            market_analysis['analysis_date'],
            market_analysis['total_tickers_analyzed'],
            json.dumps(market_analysis['regime_distribution']),
            market_analysis['most_common_regime']
        ))
        
        return {
            'market_analysis': market_analysis,
            'detailed_results': results
        }

# ======================== SCHEDULER ======================== #
@monitor_performance
@handle_errors()
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
            
        # Skip non-trading days
        if not scanner.is_trading_day(datetime.now()):
            logger.info("Skipping refresh on non-trading day")
            continue
            
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

# ======================== MAIN EXECUTION ======================== #
async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stock Ticker Fetcher with Market Regime Detection')
    parser.add_argument('--search', type=str, help='Search for a ticker by name or symbol')
    parser.add_argument('--history', type=str, help='Get history for a specific ticker')
    parser.add_argument('--list', action='store_true', help='List all active tickers')
    parser.add_argument('--refresh', action='store_true', help='Force immediate ticker refresh')
    parser.add_argument('--detect-regime', type=str, help='Detect market regime for a specific ticker')
    parser.add_argument('--analyze-market', action='store_true', help='Analyze market regimes for multiple tickers')
    parser.add_argument('--regime-history', type=str, help='Get regime history for a ticker')
    
    args = parser.parse_args()
    
    # Use enhanced scanner
    ticker_scanner = EnhancedPolygonTickerScanner()
    
    # Handle new regime detection arguments
    if args.detect_regime:
        await ticker_scanner.start()
        result = await ticker_scanner.regime_detector.detect_market_regime(args.detect_regime)
        if result:
            print(f"Market regime for {args.detect_regime}:")
            print(f"Current Regime: {result['current_regime']}")
            print(f"Number of Regimes: {result['n_regimes']}")
            print(f"Regime Statistics: {json.dumps(result['regime_stats'], indent=2)}")
        else:
            print(f"Failed to detect regime for {args.detect_regime}")
        await ticker_scanner.shutdown()
        return
    
    if args.analyze_market:
        await ticker_scanner.start()
        result = await ticker_scanner.analyze_market_regimes()
        print("Market Regime Analysis:")
        print(json.dumps(result['market_analysis'], indent=2))
        await ticker_scanner.shutdown()
        return
    
    if args.regime_history:
        await ticker_scanner.start()
        history = await ticker_scanner.regime_detector.get_regime_history(args.regime_history)
        if history:
            print(f"Regime history for {args.regime_history}:")
            for entry in history:
                print(f"Date: {entry['created_at']}, Regime: {entry['current_regime']}, N_Regimes: {entry['n_regimes']}")
        else:
            print(f"No regime history found for {args.regime_history}")
        await ticker_scanner.shutdown()
        return
    
    # Original argument handling
    if args.search:
        await ticker_scanner.start()
        results = await ticker_scanner.search_tickers_db(args.search)
        if results:
            print(f"Found {len(results)} matching tickers:")
            for result in results:
                print(f"{result['ticker']}: {result['name']} ({result['primary_exchange']})")
        else:
            print("No matching tickers found")
        await ticker_scanner.shutdown()
        return
    
    if args.history:
        await ticker_scanner.start()
        results = await ticker_scanner.get_ticker_history_db(args.history)
        if results:
            print(f"History for {args.history}:")
            for result in results:
                print(f"{result['change_date']}: {result['change_type']}")
        else:
            print(f"No history found for {args.history}")
        await ticker_scanner.shutdown()
        return
    
    if args.list:
        await ticker_scanner.start()
        results = await ticker_scanner.db.get_all_active_tickers()
        if results:
            print(f"Found {len(results)} active tickers:")
            for result in results:
                print(f"{result['ticker']}: {result['name']} ({result['primary_exchange']})")
        else:
            print("No active tickers found")
        await ticker_scanner.shutdown()
        return

    if args.refresh:
        # Just do a refresh and exit
        await ticker_scanner.start()
        await ticker_scanner.refresh_all_tickers()
        await ticker_scanner.shutdown()
        return
        
    # Normal operation - run scanner continuously
    await ticker_scanner.start()
    
    # Wait for initial cache load
    await asyncio.get_event_loop().run_in_executor(None, ticker_scanner.initial_refresh_complete.wait)
    
    # Create tasks for the scheduler
    ticker_scheduler_task = asyncio.create_task(run_scheduled_ticker_refresh(ticker_scanner))
    
    # Set up signal handlers
    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()
    
    def signal_handler():
        """Handle shutdown signals immediately"""
        print("\nReceived interrupt signal, shutting down...")
        ticker_scanner.stop()
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
            [ticker_scheduler_task, stop_task], 
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel the scheduler tasks if they're still running
        if not ticker_scheduler_task.done():
            ticker_scheduler_task.cancel()
            try:
                await ticker_scheduler_task
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
        # Shutdown the scanner
        await ticker_scanner.shutdown()

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