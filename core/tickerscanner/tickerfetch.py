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
    
    # Scanner Configuration - Using ONLY Nasdaq Composite
    COMPOSITE_INDICES = json.loads(os.getenv("COMPOSITE_INDICES", '["^IXIC"]'))  # NASDAQ Composite only
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
    
    # Backtesting Configuration
    BACKTEST_HISTORICAL_DAYS = int(os.getenv("BACKTEST_HISTORICAL_DAYS", "30"))  # Days of historical data
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv("LOG_FORMAT", '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Thread Pool Configuration
    THREAD_POOL_SIZE = int(os.getenv("THREAD_POOL_SIZE", "10"))

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
            
            # Create indexes with partial indexes
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_tickers_exchange ON tickers(primary_exchange)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_tickers_active ON tickers(active) WHERE active = 1')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_tickers_updated_at ON tickers(updated_at)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_historical_tickers_date ON historical_tickers(change_date)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_historical_tickers_ticker_date ON historical_tickers(ticker, change_date)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_historical_tickers_change_type ON historical_tickers(change_type) WHERE change_type IS NOT NULL')
            
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
    
    @handle_errors()
    async def get_slow_queries(self, threshold_ms=1000, limit=10):
        """Get slow queries from pg_stat_statements"""
        try:
            return await self.execute_query('''
                SELECT 
                    query, 
                    calls, 
                    total_exec_time, 
                    mean_exec_time,
                    rows
                FROM pg_stat_statements 
                WHERE mean_exec_time > $1
                ORDER BY mean_exec_time DESC
                LIMIT $2
            ''', (threshold_ms, limit))
        except Exception as e:
            logger.error(f"Failed to get slow queries: {e}")
            return []

# ======================== BACKTEST DATABASE MANAGER ======================== #
class BacktestDatabaseManager(BaseDatabaseManager):
    """Database manager for backtesting operations - Async version"""
    
    def __init__(self):
        # Call the parent class constructor with the correct parameters
        super().__init__(minconn=5, maxconn=30)
    
    async def _init_database(self):
        async with self.get_connection() as conn:
            # Drop and recreate the backtest_tickers table (temporary data)
            await conn.execute('DROP TABLE IF EXISTS backtest_tickers')
            
            # Create backtest_tickers table for historical data with proper primary key
            await conn.execute('''
                CREATE TABLE backtest_tickers (
                    date TEXT NOT NULL,
                    year INTEGER NOT NULL,
                    ticker TEXT NOT NULL,
                    name TEXT,
                    primary_exchange TEXT,
                    last_updated_utc TEXT,
                    type TEXT,
                    market TEXT,
                    locale TEXT,
                    currency_name TEXT,
                    PRIMARY KEY (date, ticker)
                )
            ''')
            
            # Create backtest_final_results table for storing final backtest results
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS backtest_final_results (
                    id SERIAL PRIMARY KEY,
                    run_id TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    start_year INTEGER,
                    end_year INTEGER,
                    ticker TEXT,
                    name TEXT,
                    primary_exchange TEXT,
                    last_updated_utc TEXT,
                    type TEXT,
                    market TEXT,
                    locale TEXT,
                    currency_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create composite_availability table if it doesn't exist
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS composite_availability (
                    id SERIAL PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    name TEXT,
                    primary_exchange TEXT,
                    last_updated_utc TEXT,
                    type TEXT,
                    market TEXT,
                    locale TEXT,
                    currency_name TEXT,
                    available_from DATE NOT NULL,
                    available_until DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, available_from)
                )
            ''')
            
            # Create indexes for better performance with partial indexes
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_backtest_tickers_date ON backtest_tickers(date)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_backtest_tickers_year ON backtest_tickers(year)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_backtest_tickers_ticker_year ON backtest_tickers(ticker, year)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_backtest_final_dates ON backtest_final_results(start_date, end_date)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_backtest_final_years ON backtest_final_results(start_year, end_year)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_backtest_final_ticker ON backtest_final_results(ticker)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_backtest_final_run_id ON backtest_final_results(run_id)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_composite_availability_ticker ON composite_availability(ticker)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_composite_availability_dates ON composite_availability(available_from, available_until) WHERE available_until IS NOT NULL')
            
            logger.info("Backtest database tables initialized successfully")
            
    @monitor_performance
    @handle_errors()
    async def upsert_backtest_tickers(self, tickers: List[Dict], date_str: str) -> int:
        if not tickers:
            return 0
            
        year = int(date_str.split('-')[0])  # Extract year from date
        
        # Prepare arrays for bulk operation
        dates = [date_str] * len(tickers)
        years = [year] * len(tickers)
        ticker_symbols = [t['ticker'] for t in tickers]
        names = [t.get('name') for t in tickers]
        primary_exchanges = [t.get('primary_exchange') for t in tickers]
        last_updated_utcs = [t.get('last_updated_utc') for t in tickers]
        types = [t.get('type') for t in tickers]
        markets = [t.get('market') for t in tickers]
        locales = [t.get('locale') for t in tickers]
        currency_names = [t.get('currency_name') for t in tickers]
        
        async with self.get_connection() as conn:
            result = await conn.execute('''
                INSERT INTO backtest_tickers 
                (date, year, ticker, name, primary_exchange, last_updated_utc, 
                 type, market, locale, currency_name)
                SELECT 
                    unnest($1::text[]), unnest($2::int[]), unnest($3::text[]), unnest($4::text[]), unnest($5::text[]),
                    unnest($6::text[]), unnest($7::text[]), unnest($8::text[]), unnest($9::text[]), unnest($10::text[])
                ON CONFLICT (date, ticker) DO UPDATE SET
                    name = EXCLUDED.name,
                    primary_exchange = EXCLUDED.primary_exchange,
                    last_updated_utc = EXCLUDED.last_updated_utc,
                    type = EXCLUDED.type,
                    market = EXCLUDED.market,
                    locale = EXCLUDED.locale,
                    currency_name = EXCLUDED.currency_name
            ''', (
                dates, years, ticker_symbols, names, primary_exchanges,
                last_updated_utcs, types, markets, locales, currency_names
            ))
            
            return len(tickers)
        
    @handle_errors()
    async def get_backtest_tickers(self, date_str: str) -> List[Dict]:
        return await self.execute_query(
            "SELECT * FROM backtest_tickers WHERE date = $1 ORDER BY ticker",
            (date_str,)
        )
        
    @handle_errors()
    async def get_backtest_tickers_by_year(self, year: int) -> List[Dict]:
        return await self.execute_query(
            "SELECT * FROM backtest_tickers WHERE year = $1 ORDER BY date, ticker",
            (year,)
        )
        
    @handle_errors()
    async def get_backtest_dates(self) -> List[str]:
        result = await self.execute_query(
            "SELECT DISTINCT date FROM backtest_tickers ORDER BY date"
        )
        return [row['date'] for row in result]
        
    @handle_errors()
    async def get_backtest_years(self) -> List[int]:
        result = await self.execute_query(
            "SELECT DISTINCT year FROM backtest_tickers ORDER BY year"
        )
        return [row['year'] for row in result]
        
    @monitor_performance
    @handle_errors()
    async def upsert_backtest_final_results(self, tickers_data: List[Dict], start_date: str, end_date: str, run_id: str = "default") -> int:
        if not tickers_data:
            return 0
            
        inserted = 0
        start_year = int(start_date.split('-')[0])
        end_year = int(end_date.split('-')[0])
        
        async with self.get_connection() as conn:
            # Delete existing results for this specific run_id and date range
            await conn.execute(
                "DELETE FROM backtest_final_results WHERE start_date = $1 AND end_date = $2 AND run_id = $3",
                start_date, end_date, run_id
            )
            
            # Prepare arrays for bulk operation
            run_ids = [run_id] * len(tickers_data)
            start_dates = [start_date] * len(tickers_data)
            end_dates = [end_date] * len(tickers_data)
            start_years = [start_year] * len(tickers_data)
            end_years = [end_year] * len(tickers_data)
            ticker_symbols = [t['ticker'] for t in tickers_data]
            names = [t.get('name') for t in tickers_data]
            primary_exchanges = [t.get('primary_exchange') for t in tickers_data]
            last_updated_utcs = [t.get('last_updated_utc') for t in tickers_data]
            types = [t.get('type') for t in tickers_data]
            markets = [t.get('market') for t in tickers_data]
            locales = [t.get('locale') for t in tickers_data]
            currency_names = [t.get('currency_name') for t in tickers_data]
            
            # Use execute many for bulk insert
            data_tuples = list(zip(
                run_ids, start_dates, end_dates, start_years, end_years,
                ticker_symbols, names, primary_exchanges, last_updated_utcs,
                types, markets, locales, currency_names
            ))
            
            await conn.executemany(
                '''
                INSERT INTO backtest_final_results 
                (run_id, start_date, end_date, start_year, end_year, ticker, name, primary_exchange, last_updated_utc, 
                 type, market, locale, currency_name)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ''',
                data_tuples
            )
            
            # We don't have a direct row count from executemany, so we'll use the input length
            inserted = len(tickers_data)
            
        return inserted
        
    @handle_errors()
    async def get_backtest_final_results(self, start_date: str, end_date: str, run_id: str = "default") -> List[Dict]:
        return await self.execute_query(
            "SELECT * FROM backtest_final_results WHERE start_date = $1 AND end_date = $2 AND run_id = $3 ORDER BY ticker",
            start_date, end_date, run_id
        )
        
    @handle_errors()
    async def get_backtest_final_results_by_year(self, year: int, run_id: str = "default") -> List[Dict]:
        return await self.execute_query(
            "SELECT * FROM backtest_final_results WHERE start_year <= $1 AND end_year >= $2 AND run_id = $3 ORDER BY ticker",
            year, year, run_id
        )
        
    @handle_errors()
    async def get_all_backtest_runs(self) -> List[Dict]:
        return await self.execute_query(
            "SELECT DISTINCT run_id, start_date, end_date, start_year, end_year FROM backtest_final_results ORDER BY start_date, end_date"
        )
        
    @handle_errors()
    async def get_backtest_runs_by_date_range(self, start_date: str, end_date: str) -> List[Dict]:
        return await self.execute_query(
            "SELECT DISTINCT run_id, start_date, end_date, start_year, end_year FROM backtest_final_results WHERE start_date = $1 AND end_date = $2 ORDER BY run_id",
            start_date, end_date
        )
        
    @handle_errors()
    async def upsert_composite_availability(self, ticker_data: Dict, available_from: date, available_until: date = None) -> int:
        """Upsert a ticker's availability in the composite index"""
        return await self.execute_write('''
            INSERT INTO composite_availability 
            (ticker, name, primary_exchange, last_updated_utc, type, market, locale, currency_name, available_from, available_until)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (ticker, available_from) 
            DO UPDATE SET
                name = EXCLUDED.name,
                primary_exchange = EXCLUDED.primary_exchange,
                last_updated_utc = EXCLUDED.last_updated_utc,
                type = EXCLUDED.type,
                market = EXCLUDED.market,
                locale = EXCLUDED.locale,
                currency_name = EXCLUDED.currency_name,
                available_until = EXCLUDED.available_until,
                updated_at = CURRENT_TIMESTAMP
        ''', (
            ticker_data['ticker'],
            ticker_data.get('name'),
            ticker_data.get('primary_exchange'),
            ticker_data.get('last_updated_utc'),
            ticker_data.get('type'),
            ticker_data.get('market'),
            ticker_data.get('locale'),
            ticker_data.get('currency_name'),
            available_from,
            available_until
        ))
    
    @handle_errors()
    async def get_composite_availability(self, ticker: str = None, start_date: str = None, end_date: str = None) -> List[Dict]:
        """Get composite availability for tickers, optionally filtered by ticker and date range"""
        query = "SELECT * FROM composite_availability WHERE 1=1"
        params = []
        
        if ticker:
            query += " AND ticker = $1"
            params.append(ticker)
        
        if start_date:
            query += " AND (available_until IS NULL OR available_until >= $2)"
            params.append(start_date)
        
        if end_date:
            query += " AND available_from <= $3"
            params.append(end_date)
        
        query += " ORDER BY ticker, available_from"
        
        return await self.execute_query(query, *params)
    
    @handle_errors()
    async def get_tickers_available_in_range(self, start_date: str, end_date: str) -> List[Dict]:
        """Get all tickers that were available in the composite during the entire date range"""
        return await self.execute_query('''
            SELECT DISTINCT ON (ticker) *
            FROM composite_availability 
            WHERE available_from <= $1 AND (available_until IS NULL OR available_until >= $2)
            ORDER BY ticker, available_from
        ''', start_date, end_date)
    
    @monitor_performance
    @handle_errors()
    async def update_availability_period(self, start_date: str, end_date: str, run_id: str = "default") -> int:
        """
        Update the composite_availability table based on backtest results
        This will set available_from and available_until based on the backtest period
        """
        # First, get all tickers from the backtest_final_results for this run
        final_tickers = await self.get_backtest_final_results(start_date, end_date, run_id)
        
        if not final_tickers:
            return 0
        
        # Convert string dates to date objects
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # Prepare arrays for bulk operation
        tickers = []
        names = []
        primary_exchanges = []
        last_updated_utcs = []
        types = []
        markets = []
        locales = []
        currency_names = []
        available_froms = []
        available_untils = []
        
        for ticker_data in final_tickers:
            tickers.append(ticker_data['ticker'])
            names.append(ticker_data.get('name'))
            primary_exchanges.append(ticker_data.get('primary_exchange'))
            last_updated_utcs.append(ticker_data.get('last_updated_utc'))
            types.append(ticker_data.get('type'))
            markets.append(ticker_data.get('market'))
            locales.append(ticker_data.get('locale'))
            currency_names.append(ticker_data.get('currency_name'))
            available_froms.append(start_date_obj)
            available_untils.append(end_date_obj)
        
        updated_count = 0
        
        # Use execute many for bulk operation
        try:
            async with self.get_connection() as conn:
                data_tuples = list(zip(
                    tickers, names, primary_exchanges, last_updated_utcs, types,
                    markets, locales, currency_names, available_froms, available_untils
                ))
                
                await conn.executemany(
                    '''
                    INSERT INTO composite_availability 
                    (ticker, name, primary_exchange, last_updated_utc, type, market, locale, currency_name, available_from, available_until)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (ticker, available_from) 
                    DO UPDATE SET
                        name = EXCLUDED.name,
                        primary_exchange = EXCLUDED.primary_exchange,
                        last_updated_utc = EXCLUDED.last_updated_utc,
                        type = EXCLUDED.type,
                        market = EXCLUDED.market,
                        locale = EXCLUDED.locale,
                        currency_name = EXCLUDED.currency_name,
                        available_until = EXCLUDED.available_until,
                        updated_at = CURRENT_TIMESTAMP
                    ''',
                    data_tuples
                )
                
                updated_count = len(data_tuples)
        except Exception as e:
            logger.error(f"Database error during bulk availability update: {e}")
            # Fall back to individual inserts if bulk operation fails
            updated_count = await self._update_availability_individual(final_tickers, start_date_obj, end_date_obj)
        
        return updated_count

    async def _update_availability_individual(self, tickers_data: List[Dict], start_date: date, end_date: date) -> int:
        """Fallback method to update availability one by one"""
        updated_count = 0
        for ticker_data in tickers_data:
            try:
                result = await self.upsert_composite_availability(ticker_data, start_date, end_date)
                if result:
                    updated_count += 1
            except Exception as e:
                logger.error(f"Failed to update availability for {ticker_data.get('ticker', 'unknown')}: {e}")
        
        return updated_count

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
        self.db = DatabaseManager()  # Updated to use async PostgreSQL
        self.backtest_db = BacktestDatabaseManager() 
        self.shutdown_requested = False
        # Backtesting attributes
        self.backtest_mode = False
        self.backtest_date = None
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
        logger.info(f"Using composite indices: {', '.join(self.composite_indices)}")
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
                
                # Debug: Check column names
                logger.info(f"DataFrame columns: {self.ticker_cache.columns.tolist()}")
                
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
    async def _fetch_composite_tickers(self, session, composite_index):
        """Fetch all tickers for a specific composite index"""
        logger.info(f"Fetching tickers for composite index {composite_index}")
        all_results = []
        next_url = None
        page_count = 0
        max_pages = 50  # Safety limit
        
        # Use current date or backtest date
        if self.backtest_mode and self.backtest_date:
            date_param = self.backtest_date
            logger.info(f"Using historical date: {date_param}")
            
            # Check if it's a trading day for historical data
            if not self.is_trading_day(date_param):
                logger.info(f"Skipping non-trading day: {date_param}")
                return []
        else:
            date_param = datetime.now().strftime("%Y-%m-%d")
            
        # Different API endpoint for composite indices
        if composite_index == "^IXIC":  # NASDAQ Composite
            exchange = "XNAS"
        else:
            logger.error(f"Unknown composite index: {composite_index}")
            raise ConfigurationError(f"Unknown composite index: {composite_index}")
            
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
        
        while url and page_count < max_pages and not self.shutdown_requested:
            data = await self._call_polygon_api(session, url)
            if not data or self.shutdown_requested:
                break
                
            results = data.get("results", [])
            if not results:
                logger.warning(f"No results in page {page_count + 1} for {composite_index}")
                break
                
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
            
            # Add progressive delay to avoid rate limiting
            delay = config.RATE_LIMIT_DELAY * (1 + page_count / 10)
            await asyncio.sleep(min(delay, 5.0))  # Cap at 5 seconds
        
        if page_count >= max_pages:
            logger.warning(f"Reached maximum page limit ({max_pages}) for {composite_index}")
        
        if self.shutdown_requested:
            logger.info(f"Shutdown requested, aborting {composite_index} fetch")
            return []
            
        logger.info(f"Completed {composite_index}: {len(all_results)} stocks across {page_count} pages")
        return all_results

    @monitor_performance
    @handle_errors(max_retries=config.MAX_RETRIES, retry_delay=config.RETRY_DELAY)
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
            
        try:
            async with aiohttp.ClientSession() as session:
                # Fetch all composite indices in parallel
                tasks = [self._fetch_composite_tickers(session, idx) for idx in self.composite_indices]
                composite_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                all_results = []
                for i, results in enumerate(composite_results):
                    if isinstance(results, Exception):
                        logger.error(f"Error fetching {self.composite_indices[i]}: {results}")
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
            # For backtest mode, we don't update the main database
            if self.backtest_mode and self.backtest_date:
                # Store backtest results
                tickers_data = new_df.to_dict('records')
                # Use backtest_db instead of db
                inserted = await self.backtest_db.upsert_backtest_tickers(tickers_data, self.backtest_date)
                logger.info(f"Stored {inserted} tickers in backtest database for {self.backtest_date}")
            else:
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
        
        if not self.backtest_mode:
            await self.db.update_metadata('last_refresh_time', self.last_refresh_time)
        
        elapsed = time.time() - start_time
        logger.info(f"Ticker refresh completed in {elapsed:.2f}s")
        
        if not self.backtest_mode:
            logger.info(f"Total: {len(new_df)} | Added: {len(added)} | Removed: {len(removed)}")
            logger.info(f"Database: {inserted} inserted, {updated} updated")
        else:
            logger.info(f"Historical data: {len(new_df)} tickers for {self.backtest_date}")
            
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
        await self.backtest_db.close_all_connections()
        
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
        
    @handle_errors()
    async def get_backtest_tickers(self, date_str: str) -> List[Dict]:
        """Get tickers for a specific backtest date"""
        return await self.backtest_db.get_backtest_tickers(date_str)
        
    @handle_errors()
    async def get_backtest_tickers_by_year(self, year: int) -> List[Dict]:
        """Get tickers for a specific backtest year"""
        return await self.backtest_db.get_backtest_tickers_by_year(year)
        
    @handle_errors()
    async def get_backtest_dates(self) -> List[str]:
        """Get all available backtest dates"""
        return await self.backtest_db.get_backtest_dates()
        
    @handle_errors()
    async def get_backtest_years(self) -> List[int]:
        """Get all available backtest years"""
        return await self.backtest_db.get_backtest_years()
        
    @handle_errors()
    async def get_backtest_final_results(self, start_date: str, end_date: str, run_id: str = "default") -> List[Dict]:
        """Get final backtest results for a specific date range"""
        return await self.backtest_db.get_backtest_final_results(start_date, end_date, run_id)
        
    @handle_errors()
    async def get_backtest_final_results_by_year(self, year: int, run_id: str = "default") -> List[Dict]:
        """Get final backtest results for a specific year"""
        return await self.backtest_db.get_backtest_final_results_by_year(year, run_id)
        
    @handle_errors()
    async def get_all_backtest_runs(self) -> List[Dict]:
        """Get all backtest runs with their date ranges"""
        return await self.backtest_db.get_all_backtest_runs()
        
    @handle_errors()
    async def get_backtest_runs_by_date_range(self, start_date: str, end_date: str) -> List[Dict]:
        """Get all backtest runs for a specific date range"""
        return await self.backtest_db.get_backtest_runs_by_date_range(start_date, end_date)
        
    @handle_errors()
    async def get_composite_availability(self, ticker: str = None, start_date: str = None, end_date: str = None) -> List[Dict]:
        """Get composite availability for tickers"""
        return await self.backtest_db.get_composite_availability(ticker, start_date, end_date)
    
    @handle_errors()
    async def get_tickers_available_in_range(self, start_date: str, end_date: str) -> List[Dict]:
        """Get all tickers that were available in the composite during the entire date range"""
        return await self.backtest_db.get_tickers_available_in_range(start_date, end_date)
        
    @handle_errors()
    async def health_check(self):
        """Comprehensive health check"""
        status = {
            'api_accessible': False,
            'database_connected': False,
            'cache_updated': False,
            'last_refresh': self.last_refresh_time,
            'ticker_count': len(self.current_tickers_set),
            'shutdown_requested': self.shutdown_requested,
            'circuit_open': self.circuit_open,
            'api_error_count': self.api_error_count
        }
        
        # Check API accessibility
        try:
            async with aiohttp.ClientSession() as session:
                test_url = f"https://api.polygon.io/v3/reference/tickers?ticker=AAPL&apiKey={self.api_key}"
                async with session.get(test_url, timeout=5) as response:
                    status['api_accessible'] = response.status == 200
        except:
            status['api_accessible'] = False
        
        # Check database connection
        try:
            async with self.db.get_connection() as conn:
                await conn.execute("SELECT 1")
                status['database_connected'] = True
        except:
            status['database_connected'] = False
        
        # Check cache freshness
        status['cache_updated'] = time.time() - self.last_refresh_time < 86400  # 24 hours
        
        return status

# ======================== BACKTESTER ======================== #
class Backtester:
    def __init__(self, ticker_scanner):
        self.ticker_scanner = ticker_scanner
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    @monitor_performance
    @handle_errors()
    async def run_backtest(self, start_date, end_date=None, test_tickers=True):
        """
        Run backtest for a specific date range
        Args:
            start_date: datetime object or string in YYYY-MM-DD format
            end_date: datetime object or string in YYYY-MM-DD format (optional)
            test_tickers: Whether to test ticker fetching
        """
        if end_date is None:
            end_date = start_date
            
        # Convert to datetime if strings are provided
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
        logger.info(f"Starting backtest from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"Testing: Tickers={test_tickers}")
        logger.info(f"Run ID: {self.run_id}")
        
        # Dictionary to track ticker availability across all dates
        ticker_availability = defaultdict(list)
        
        current_date = start_date
        while current_date <= end_date:
            # Only process trading days
            if self.ticker_scanner.is_trading_day(current_date):
                date_str = current_date.strftime("%Y-%m-%d")
                
                if test_tickers:
                    # Use the ticker_scanner's method, not the db directly
                    existing_ticker_data = await self.ticker_scanner.get_backtest_tickers(date_str)
                    if not existing_ticker_data:
                        await self.run_single_day_ticker_backtest(current_date)
            
            current_date += timedelta(days=1)
        
        # If testing tickers, filter tickers to only those available for the entire period
        if test_tickers:
            # Get all dates in the range
            all_dates = [d.strftime("%Y-%m-%d") for d in self._date_range(start_date, end_date) 
                        if self.ticker_scanner.is_trading_day(d)]
            
            # Track which tickers were available on each date
            for date_str in all_dates:
                # Use the ticker_scanner's method, not the db directly
                ticker_data = await self.ticker_scanner.get_backtest_tickers(date_str)
                if ticker_data:
                    for ticker in ticker_data:
                        ticker_availability[ticker['ticker']].append(date_str)
            
            # Filter tickers to only those available for the entire period
            full_period_tickers = []
            for ticker, available_dates in ticker_availability.items():
                # Check if ticker was available for all dates in the range
                if len(available_dates) == len(all_dates):
                    # Get the most recent data for this ticker
                    latest_date = max(available_dates)
                    # Use the backtest_db directly for this query
                    ticker_data = await self.ticker_scanner.backtest_db.execute_query(
                        "SELECT * FROM backtest_tickers WHERE ticker = $1 AND date = $2",
                        ticker, latest_date
                    )
                    if ticker_data:
                        full_period_tickers.append(ticker_data[0])
            
            # Save the filtered results to the database
            if full_period_tickers:
                start_str = start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")
                # Use the ticker_scanner's method, not the db directly
                inserted = await self.ticker_scanner.backtest_db.upsert_backtest_final_results(
                    full_period_tickers, start_str, end_str, self.run_id
                )
                
                # Update the composite availability table
                updated = await self.ticker_scanner.backtest_db.update_availability_period(
                    start_str, end_str, self.run_id
                )
                logger.info(f"Saved {inserted} tickers to database and updated availability for {updated} tickers")
            else:
                logger.warning("No tickers were active throughout the entire period")
            
        logger.info("Backtest completed")
        
    def _date_range(self, start_date, end_date):
        """Generate a range of dates between start_date and end_date"""
        for n in range(int((end_date - start_date).days) + 1):
            yield start_date + timedelta(n)
        
    @monitor_performance
    @handle_errors()
    async def run_single_day_ticker_backtest(self, target_date):
        """
        Run ticker backtest for a single day
        """
        # Skip non-trading days
        if not self.ticker_scanner.is_trading_day(target_date):
            logger.info(f"Skipping non-trading day: {target_date.strftime('%Y-%m-%d')}")
            return False
            
        logger.info(f"Running ticker backtest for {target_date.strftime('%Y-%m-%d')}")
        
        # Format date for API
        date_str = target_date.strftime("%Y-%m-%d")
        
        # Use the ticker_scanner's method, not the db directly
        existing_data = await self.ticker_scanner.get_backtest_tickers(date_str)
        if existing_data:
            logger.info(f"Using cached ticker data for {date_str} ({len(existing_data)} tickers)")
            return True
        
        # Temporarily set scanner to backtest mode
        original_mode = self.ticker_scanner.backtest_mode
        self.ticker_scanner.backtest_mode = True
        self.ticker_scanner.backtest_date = date_str
        
        try:
            # Use the existing refresh method but with historical date
            success = await self.ticker_scanner.refresh_all_tickers()
            if success:
                logger.info(f"Successfully fetched ticker data for {date_str}")
                return True
            else:
                logger.warning(f"Failed to fetch ticker data for {date_str}")
                return False
        except Exception as e:
            logger.error(f"Error during ticker backtest for {date_str}: {e}")
            return False
        finally:
            # Restore original mode
            self.ticker_scanner.backtest_mode = original_mode
            self.ticker_scanner.backtest_date = None

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
    parser = argparse.ArgumentParser(description='Stock Ticker Fetcher with Backtesting')
    parser.add_argument('--search', type=str, help='Search for a ticker by name or symbol')
    parser.add_argument('--history', type=str, help='Get history for a specific ticker')
    parser.add_argument('--list', action='store_true', help='List all active tickers')
    
    # Backtesting arguments
    parser.add_argument('--backtest', type=str, help='Run backtest for a specific date (YYYY-MM-DD)')
    parser.add_argument('--backtest-range', type=str, help='Run backtest for a date range (YYYY-MM-DD:YYYY-MM-DD)')
    parser.add_argument('--backtest-year', type=int, help='Run backtest for a specific year')
    parser.add_argument('--list-backtests', action='store_true', help='List available backtest dates')
    parser.add_argument('--list-backtest-years', action='store_true', help='List available backtest years')
    parser.add_argument('--list-backtest-runs', action='store_true', help='List available backtest runs')
    parser.add_argument('--show-backtest-results', type=str, help='Show results for a specific backtest run (format: YYYY-MM-DD:YYYY-MM-DD)')
    parser.add_argument('--show-year-results', type=int, help='Show results for a specific year')
    parser.add_argument('--run-id', type=str, default="default", help='Specify a run ID for backtest results')
    
    # Composite availability arguments
    parser.add_argument('--composite-availability', type=str, help='Get composite availability for a ticker (optional: specify ticker)')
    parser.add_argument('--available-in-range', type=str, help='Get tickers available in a date range (format: YYYY-MM-DD:YYYY-MM-DD)')
    
    # Health check argument
    parser.add_argument('--health', action='store_true', help='Run health check')
    
    args = parser.parse_args()
    
    ticker_scanner = PolygonTickerScanner()
    
    # Health check
    if args.health:
        health_status = await ticker_scanner.health_check()
        print("Health Check Results:")
        for key, value in health_status.items():
            print(f"  {key}: {value}")
        return
    
    # Check if we're running in backtest mode
    if (args.backtest or args.backtest_range or args.backtest_year or 
        args.list_backtests or args.list_backtest_years or args.list_backtest_runs or 
        args.show_backtest_results or args.show_year_results or
        args.composite_availability or args.available_in_range):
        
        if args.composite_availability:
            # Get composite availability for a ticker
            results = await ticker_scanner.get_composite_availability(args.composite_availability)
            if results:
                print(f"Composite availability for {args.composite_availability}:")
                for result in results:
                    print(f"  {result['available_from']} to {result['available_until'] or 'Present'}")
            else:
                print(f"No composite availability data found for {args.composite_availability}")
            return
            
        if args.available_in_range:
            # Get tickers available in a date range
            start_date, end_date = args.available_in_range.split(':')
            results = await ticker_scanner.get_tickers_available_in_range(start_date, end_date)
            if results:
                print(f"Tickers available from {start_date} to {end_date}:")
                for result in results:
                    print(f"  {result['ticker']}: {result['name']} (From {result['available_from']} to {result['available_until'] or 'Present'})")
            else:
                print(f"No tickers found available from {start_date} to {end_date}")
            return
        
        if args.list_backtests:
            # List available backtest dates
            dates = await ticker_scanner.get_backtest_dates()
            if dates:
                print("Available backtest dates:")
                for date in dates:
                    print(f"  {date}")
            else:
                print("No backtest data available")
            return
        
        if args.list_backtest_years:
            # List available backtest years
            years = await ticker_scanner.get_backtest_years()
            if years:
                print("Available backtest years:")
                for year in years:
                    print(f"  {year}")
            else:
                print("No backtest data available")
            return
            
        if args.list_backtest_runs:
            # List available backtest runs
            runs = await ticker_scanner.get_all_backtest_runs()
            if runs:
                print("Available backtest runs:")
                for run in runs:
                    print(f"  {run['run_id']}: {run['start_date']} to {run['end_date']} (Years: {run['start_year']}-{run['end_year']})")
            else:
                print("No backtest runs available")
            return
            
        if args.show_backtest_results:
            # Show results for a specific backtest run
            start_date, end_date = args.show_backtest_results.split(':')
            results = await ticker_scanner.get_backtest_final_results(start_date, end_date, args.run_id)
            if results:
                print(f"Backtest results for {start_date} to {end_date} (Run ID: {args.run_id}):")
                print(f"Found {len(results)} tickers that were active throughout the period")
                for result in results[:10]:  # Show first 10 results
                    print(f"  {result['ticker']}: {result['name']}")
                if len(results) > 10:
                    print(f"  ... and {len(results) - 10} more")
            else:
                print(f"No results found for {start_date} to {end_date} with run ID {args.run_id}")
            return
            
        if args.show_year_results:
            # Show results for a specific year
            year = args.show_year_results
            results = await ticker_scanner.get_backtest_final_results_by_year(year, args.run_id)
            if results:
                print(f"Backtest results for year {year} (Run ID: {args.run_id}):")
                print(f"Found {len(results)} tickers that were active in this year")
                for result in results[:10]:  # Show first 10 results
                    print(f"  {result['ticker']}: {result['name']} (From {result['start_date']} to {result['end_date']})")
                if len(results) > 10:
                    print(f"  ... and {len(results) - 10} more")
            else:
                print(f"No results found for year {year} with run ID {args.run_id}")
            return
                
        backtester = Backtester(ticker_scanner)
        backtester.run_id = args.run_id  # Use the provided run ID
        
        # Determine what to test
        test_tickers = True
        
        if args.backtest:
            # Single date backtest
            if test_tickers:
                await backtester.run_single_day_ticker_backtest(datetime.strptime(args.backtest, "%Y-%m-%d"))
        elif args.backtest_range:
            # Date range backtest
            start_str, end_str = args.backtest_range.split(':')
            start_date = datetime.strptime(start_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_str, "%Y-%m-%d")
            await backtester.run_backtest(start_date, end_date, test_tickers)
        elif args.backtest_year:
            # Year backtest
            start_date = datetime(args.backtest_year, 1, 1)
            end_date = datetime(args.backtest_year, 12, 31)
            await backtester.run_backtest(start_date, end_date, test_tickers)
    else:
        # Handle other command line arguments
        if args.search:
            results = await ticker_scanner.search_tickers_db(args.search)
            if results:
                print(f"Found {len(results)} matching tickers:")
                for result in results:
                    print(f"{result['ticker']}: {result['name']} ({result['primary_exchange']})")
            else:
                print("No matching tickers found")
            return
        
        if args.history:
            results = await ticker_scanner.get_ticker_history_db(args.history)
            if results:
                print(f"History for {args.history}:")
                for result in results:
                    print(f"{result['change_date']}: {result['change_type']}")
            else:
                print(f"No history found for {args.history}")
            return
        
        if args.list:
            results = await ticker_scanner.db.get_all_active_tickers()
            if results:
                print(f"Found {len(results)} active tickers:")
                for result in results:
                    print(f"{result['ticker']}: {result['name']} ({result['primary_exchange']})")
            else:
                print("No active tickers found")
            return
        
        # Normal operation - run scanner
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