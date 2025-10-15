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
    
    # Market Regime Configuration
    REGIME_LOOKBACK_PERIOD = int(os.getenv("REGIME_LOOKBACK_PERIOD", "63"))
    REGIME_VOLATILITY_LOOKBACK = int(os.getenv("REGIME_VOLATILITY_LOOKBACK", "21"))

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

# ======================== XNAS MARKET REGIME CLASSIFIER ======================== #
class XNASMarketRegime:
    """
    Efficient Market Regime Classifier for XNAS Exchange
    Optimized for Trend Following with Momentum Confirmation
    """
    
    def __init__(self, lookback_period: int = None, volatility_lookback: int = None):
        self.lookback_period = lookback_period or config.REGIME_LOOKBACK_PERIOD
        self.volatility_lookback = volatility_lookback or config.REGIME_VOLATILITY_LOOKBACK
        self.regime_names = {
            0: "High Volatility Bearish",
            1: "Bearish", 
            2: "Neutral/Congestion",
            3: "Bullish",
            4: "High Volatility Bullish"
        }
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators needed for regime classification"""
        df = data.copy()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Trend indicators
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # Momentum indicators
        df['rsi_14'] = self.calculate_rsi(df['close'], 14)
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        # Volatility indicators
        df['volatility_21'] = df['returns'].rolling(window=21).std() * np.sqrt(252)
        df['atr_14'] = self.calculate_atr(df, 14)
        df['bb_upper'], df['bb_lower'] = self.calculate_bollinger_bands(df['close'], 20)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
        
        # Volume indicators
        if 'volume' in df.columns:
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            df['obv'] = self.calculate_obv(df)
        
        # Advanced features for XNAS
        df['trend_strength'] = self.calculate_trend_strength(df)
        df['market_efficiency'] = self.calculate_market_efficiency(df)
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(np.maximum(high_low, high_close), low_close)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return upper_band, lower_band
    
    def calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On Balance Volume"""
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv
    
    def calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate overall trend strength using multiple timeframes"""
        # Short-term trend (20-day)
        short_trend = (df['close'] - df['sma_20']) / df['sma_20']
        
        # Medium-term trend (50-day)
        medium_trend = (df['close'] - df['sma_50']) / df['sma_50']
        
        # Long-term trend (200-day)
        long_trend = (df['close'] - df['sma_200']) / df['sma_200']
        
        # MACD trend
        macd_trend = df['macd'] - df['macd_signal']
        
        # Combined trend strength (weighted average)
        trend_strength = (
            0.3 * short_trend + 
            0.4 * medium_trend + 
            0.2 * long_trend + 
            0.1 * macd_trend
        )
        
        return trend_strength
    
    def calculate_market_efficiency(self, df: pd.DataFrame) -> pd.Series:
        """Calculate market efficiency ratio (trending vs mean-reverting)"""
        price_changes = df['close'].diff()
        net_movement = price_changes.rolling(window=20).apply(
            lambda x: np.abs(x.iloc[-1] - x.iloc[0]) if len(x) == 20 else np.nan
        )
        total_movement = price_changes.rolling(window=20).apply(
            lambda x: np.sum(np.abs(x)) if len(x) == 20 else np.nan
        )
        efficiency = net_movement / total_movement
        return efficiency
    
    def classify_regime(self, data: pd.DataFrame) -> pd.Series:
        """
        Classify market regime into 5 categories optimized for trend following
        Returns: Series with regime codes (0-4)
        """
        df = self.calculate_technical_indicators(data)
        
        # Initialize regime array
        regimes = np.zeros(len(df))
        
        for i in range(max(self.lookback_period, 200), len(df)):
            current_data = df.iloc[:i+1]
            current_row = df.iloc[i]
            
            # Skip if insufficient data
            if current_data.isna().any().any():
                continue
            
            # 1. Trend Direction
            trend_score = 0
            if current_row['sma_20'] > current_row['sma_50'] > current_row['sma_200']:
                trend_score += 2  # Strong uptrend
            elif current_row['sma_20'] > current_row['sma_50']:
                trend_score += 1  # Mild uptrend
            elif current_row['sma_20'] < current_row['sma_50'] < current_row['sma_200']:
                trend_score -= 2  # Strong downtrend
            elif current_row['sma_20'] < current_row['sma_50']:
                trend_score -= 1  # Mild downtrend
            
            # 2. Momentum
            momentum_score = 0
            if current_row['rsi_14'] > 60 and current_row['macd'] > current_row['macd_signal']:
                momentum_score += 2
            elif current_row['rsi_14'] > 50 and current_row['macd'] > current_row['macd_signal']:
                momentum_score += 1
            elif current_row['rsi_14'] < 40 and current_row['macd'] < current_row['macd_signal']:
                momentum_score -= 2
            elif current_row['rsi_14'] < 50 and current_row['macd'] < current_row['macd_signal']:
                momentum_score -= 1
            
            # 3. Volatility Assessment
            vol_percentile = (current_row['volatility_21'] > 
                            current_data['volatility_21'].rolling(63).mean().iloc[i])
            
            # 4. Combined Score
            total_score = trend_score + momentum_score
            
            # 5. Regime Classification
            if total_score >= 3 and vol_percentile:
                regimes[i] = 4  # High Volatility Bullish
            elif total_score >= 2:
                regimes[i] = 3  # Bullish
            elif total_score <= -3 and vol_percentile:
                regimes[i] = 0  # High Volatility Bearish
            elif total_score <= -2:
                regimes[i] = 1  # Bearish
            else:
                regimes[i] = 2  # Neutral/Congestion
        
        return pd.Series(regimes, index=df.index, name='market_regime')
    
    def get_regime_statistics(self, data: pd.DataFrame, regimes: pd.Series) -> Dict:
        """Calculate statistics for each market regime"""
        df = data.copy()
        df['regime'] = regimes
        df['returns'] = df['close'].pct_change()
        
        stats = {}
        for regime_code, regime_name in self.regime_names.items():
            regime_data = df[df['regime'] == regime_code]
            if len(regime_data) > 0:
                stats[regime_name] = {
                    'count': len(regime_data),
                    'frequency': len(regime_data) / len(df),
                    'avg_return': regime_data['returns'].mean(),
                    'return_std': regime_data['returns'].std(),
                    'sharpe_ratio': (regime_data['returns'].mean() * 252) / 
                                   (regime_data['returns'].std() * np.sqrt(252)) 
                                   if regime_data['returns'].std() > 0 else 0,
                    'avg_volatility': regime_data['volatility_21'].mean() 
                                    if 'volatility_21' in regime_data.columns else 0
                }
        
        return stats
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on market regime
        Returns DataFrame with signals and regime information
        """
        regimes = self.classify_regime(data)
        df = data.copy()
        df['market_regime'] = regimes
        df['regime_name'] = df['market_regime'].map(self.regime_names)
        
        # Generate signals for trend following strategy
        df['signal'] = 0
        
        bullish_mask = df['market_regime'].isin([3, 4])  # Bullish regimes
        bearish_mask = df['market_regime'].isin([0, 1])  # Bearish regimes
        
        # Long signals in bullish regimes with momentum confirmation
        df.loc[bullish_mask & (df['rsi_14'] > 50) & (df['macd'] > df['macd_signal']), 'signal'] = 1
        
        # Short signals in bearish regimes with momentum confirmation  
        df.loc[bearish_mask & (df['rsi_14'] < 50) & (df['macd'] < df['macd_signal']), 'signal'] = -1
        
        # Exit signals when regime changes or momentum weakens
        regime_changes = df['market_regime'].diff().fillna(0) != 0
        momentum_weakens = ((df['signal'] == 1) & (df['rsi_14'] < 45)) | \
                          ((df['signal'] == -1) & (df['rsi_14'] > 55))
        
        df.loc[regime_changes | momentum_weakens, 'signal'] = 0
        
        return df

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
            
            # Create market_regime table for storing regime analysis
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS market_regime (
                    id SERIAL PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    date DATE NOT NULL,
                    regime INTEGER NOT NULL,
                    regime_name TEXT NOT NULL,
                    signal INTEGER NOT NULL,
                    close_price DECIMAL(15,4),
                    volume BIGINT,
                    sma_20 DECIMAL(15,4),
                    sma_50 DECIMAL(15,4),
                    sma_200 DECIMAL(15,4),
                    rsi_14 DECIMAL(10,4),
                    macd DECIMAL(10,4),
                    macd_signal DECIMAL(10,4),
                    volatility_21 DECIMAL(10,6),
                    trend_strength DECIMAL(10,6),
                    market_efficiency DECIMAL(10,6),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, date)
                )
            ''')
            
            # Create indexes with partial indexes
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_tickers_exchange ON tickers(primary_exchange)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_tickers_active ON tickers(active) WHERE active = 1')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_tickers_updated_at ON tickers(updated_at)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_historical_tickers_date ON historical_tickers(change_date)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_historical_tickers_ticker_date ON historical_tickers(ticker, change_date)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_historical_tickers_change_type ON historical_tickers(change_type) WHERE change_type IS NOT NULL')
            
            # Indexes for market_regime table
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_market_regime_ticker_date ON market_regime(ticker, date)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_market_regime_regime ON market_regime(regime)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_market_regime_signal ON market_regime(signal)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_market_regime_date ON market_regime(date)')
            
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

    @monitor_performance
    @handle_errors()
    async def store_market_regime_data(self, regime_data: List[Dict]) -> int:
        """Store market regime analysis results in database"""
        if not regime_data:
            return 0
            
        stored_count = 0
        
        async with self.get_connection() as conn:
            # Use transaction for better performance
            async with conn.transaction():
                try:
                    for data in regime_data:
                        result = await conn.execute('''
                            INSERT INTO market_regime 
                            (ticker, date, regime, regime_name, signal, close_price, volume,
                             sma_20, sma_50, sma_200, rsi_14, macd, macd_signal, 
                             volatility_21, trend_strength, market_efficiency)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                            ON CONFLICT (ticker, date) DO UPDATE SET
                                regime = EXCLUDED.regime,
                                regime_name = EXCLUDED.regime_name,
                                signal = EXCLUDED.signal,
                                close_price = EXCLUDED.close_price,
                                volume = EXCLUDED.volume,
                                sma_20 = EXCLUDED.sma_20,
                                sma_50 = EXCLUDED.sma_50,
                                sma_200 = EXCLUDED.sma_200,
                                rsi_14 = EXCLUDED.rsi_14,
                                macd = EXCLUDED.macd,
                                macd_signal = EXCLUDED.macd_signal,
                                volatility_21 = EXCLUDED.volatility_21,
                                trend_strength = EXCLUDED.trend_strength,
                                market_efficiency = EXCLUDED.market_efficiency,
                                created_at = CURRENT_TIMESTAMP
                        ''', 
                        data['ticker'], data['date'], data['regime'], data['regime_name'],
                        data['signal'], data.get('close_price'), data.get('volume'),
                        data.get('sma_20'), data.get('sma_50'), data.get('sma_200'),
                        data.get('rsi_14'), data.get('macd'), data.get('macd_signal'),
                        data.get('volatility_21'), data.get('trend_strength'), 
                        data.get('market_efficiency'))
                        
                        if "INSERT" in result or "UPDATE" in result:
                            stored_count += 1
                            
                except Exception as e:
                    logger.error(f"Transaction failed during market regime storage: {e}")
                    raise DatabaseError(f"Transaction failed during market regime storage: {e}")
                
        return stored_count

    @handle_errors()
    async def get_market_regime_history(self, ticker: str, limit: int = 100) -> List[Dict]:
        """Get market regime history for a specific ticker"""
        return await self.execute_query(
            "SELECT * FROM market_regime WHERE ticker = $1 ORDER BY date DESC LIMIT $2",
            (ticker, limit)
        )

    @handle_errors()
    async def get_current_market_regime(self, ticker: str) -> Optional[Dict]:
        """Get the most recent market regime for a ticker"""
        result = await self.execute_query(
            "SELECT * FROM market_regime WHERE ticker = $1 ORDER BY date DESC LIMIT 1",
            (ticker,)
        )
        return result[0] if result else None

    @handle_errors()
    async def get_regime_distribution(self, date: str = None) -> Dict:
        """Get distribution of market regimes for a specific date"""
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
            
        result = await self.execute_query('''
            SELECT regime, regime_name, COUNT(*) as count
            FROM market_regime 
            WHERE date = $1
            GROUP BY regime, regime_name
            ORDER BY regime
        ''', (date,))
        
        return {row['regime_name']: row['count'] for row in result}

# ======================== TICKER SCANNER WITH MARKET REGIME ======================== #
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
        # Market Regime Classifier
        self.regime_classifier = XNASMarketRegime()
        
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

    # ======================== MARKET REGIME METHODS ======================== #
    
    @monitor_performance
    @handle_errors(max_retries=config.MAX_RETRIES, retry_delay=config.RETRY_DELAY)
    async def fetch_price_data(self, ticker: str, days: int = 252) -> Optional[pd.DataFrame]:
        """Fetch historical price data for market regime analysis"""
        try:
            # Use Polygon API to fetch daily bars
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{datetime.now().date() - timedelta(days=days)}/{datetime.now().date()}?apiKey={self.api_key}&limit={days}"
            
            async with aiohttp.ClientSession() as session:
                data = await self._call_polygon_api(session, url)
                
                if not data or 'results' not in data:
                    logger.warning(f"No price data available for {ticker}")
                    return None
                
                # Convert to DataFrame
                results = data['results']
                df = pd.DataFrame(results)
                
                # Rename columns to match our expected format
                df = df.rename(columns={
                    't': 'timestamp',
                    'o': 'open',
                    'h': 'high', 
                    'l': 'low',
                    'c': 'close',
                    'v': 'volume'
                })
                
                # Convert timestamp to datetime
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('date', inplace=True)
                
                # Keep only necessary columns
                df = df[['open', 'high', 'low', 'close', 'volume']]
                
                return df
                
        except Exception as e:
            logger.error(f"Error fetching price data for {ticker}: {e}")
            return None

    @monitor_performance
    @handle_errors()
    async def analyze_market_regime(self, ticker: str) -> Optional[Dict]:
        """Analyze market regime for a specific ticker"""
        try:
            # Fetch price data
            price_data = await self.fetch_price_data(ticker)
            if price_data is None or len(price_data) < 200:  # Need at least 200 days for proper analysis
                logger.warning(f"Insufficient data for market regime analysis of {ticker}")
                return None
            
            # Generate market regime signals
            regime_data = self.regime_classifier.generate_signals(price_data)
            
            # Get the latest regime data
            latest_data = regime_data.iloc[-1]
            
            # Prepare data for storage
            regime_result = {
                'ticker': ticker,
                'date': latest_data.name.date() if hasattr(latest_data.name, 'date') else datetime.now().date(),
                'regime': int(latest_data['market_regime']),
                'regime_name': latest_data['regime_name'],
                'signal': int(latest_data['signal']),
                'close_price': float(latest_data['close']),
                'volume': int(latest_data.get('volume', 0)),
                'sma_20': float(latest_data.get('sma_20', 0)),
                'sma_50': float(latest_data.get('sma_50', 0)),
                'sma_200': float(latest_data.get('sma_200', 0)),
                'rsi_14': float(latest_data.get('rsi_14', 0)),
                'macd': float(latest_data.get('macd', 0)),
                'macd_signal': float(latest_data.get('macd_signal', 0)),
                'volatility_21': float(latest_data.get('volatility_21', 0)),
                'trend_strength': float(latest_data.get('trend_strength', 0)),
                'market_efficiency': float(latest_data.get('market_efficiency', 0))
            }
            
            return regime_result
            
        except Exception as e:
            logger.error(f"Error analyzing market regime for {ticker}: {e}")
            return None

    @monitor_performance
    @handle_errors()
    async def batch_analyze_market_regime(self, tickers: List[str], batch_size: int = 10) -> Dict:
        """Analyze market regime for multiple tickers in batches"""
        results = {}
        total_tickers = len(tickers)
        
        logger.info(f"Starting market regime analysis for {total_tickers} tickers in batches of {batch_size}")
        
        for i in range(0, total_tickers, batch_size):
            batch = tickers[i:i + batch_size]
            batch_tasks = [self.analyze_market_regime(ticker) for ticker in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process batch results
            for j, result in enumerate(batch_results):
                ticker = batch[j]
                if isinstance(result, Exception):
                    logger.error(f"Error processing {ticker}: {result}")
                    results[ticker] = None
                else:
                    results[ticker] = result
            
            # Store results in database
            valid_results = [r for r in batch_results if r is not None and not isinstance(r, Exception)]
            if valid_results:
                stored_count = await self.db.store_market_regime_data(valid_results)
                logger.info(f"Stored {stored_count} market regime results from batch {i//batch_size + 1}")
            
            # Rate limiting between batches
            if i + batch_size < total_tickers:
                await asyncio.sleep(1)  # 1 second delay between batches
        
        # Calculate statistics
        successful = len([r for r in results.values() if r is not None])
        logger.info(f"Market regime analysis completed: {successful}/{total_tickers} successful")
        
        return results

    @handle_errors()
    async def get_market_regime_stats(self, date: str = None) -> Dict:
        """Get market regime statistics for all analyzed tickers"""
        return await self.db.get_regime_distribution(date)

    @handle_errors()
    async def get_ticker_regime_history(self, ticker: str, limit: int = 100) -> List[Dict]:
        """Get market regime history for a specific ticker"""
        return await self.db.get_market_regime_history(ticker, limit)

    @handle_errors()
    async def get_current_ticker_regime(self, ticker: str) -> Optional[Dict]:
        """Get current market regime for a specific ticker"""
        return await self.db.get_current_market_regime(ticker)

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
            
            # After ticker refresh, run market regime analysis on a sample of tickers
            active_tickers = scanner.get_current_tickers_list()
            if active_tickers:
                # Analyze first 50 tickers as a sample (adjust as needed)
                sample_tickers = active_tickers[:50]
                logger.info(f"Starting market regime analysis for {len(sample_tickers)} sample tickers")
                await scanner.batch_analyze_market_regime(sample_tickers)
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
                
                # Run market regime analysis after ticker refresh
                active_tickers = scanner.get_current_tickers_list()
                if active_tickers:
                    # Analyze a sample of tickers (adjust batch size based on your needs)
                    sample_tickers = active_tickers[:100]  # Analyze first 100 tickers
                    logger.info(f"Starting market regime analysis for {len(sample_tickers)} tickers")
                    await scanner.batch_analyze_market_regime(sample_tickers, batch_size=20)
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
    parser = argparse.ArgumentParser(description='Stock Ticker Fetcher with Market Regime Analysis')
    parser.add_argument('--search', type=str, help='Search for a ticker by name or symbol')
    parser.add_argument('--history', type=str, help='Get history for a specific ticker')
    parser.add_argument('--list', action='store_true', help='List all active tickers')
    parser.add_argument('--refresh', action='store_true', help='Force immediate ticker refresh')
    parser.add_argument('--regime', type=str, help='Analyze market regime for specific ticker')
    parser.add_argument('--regime-stats', action='store_true', help='Show market regime statistics')
    parser.add_argument('--regime-history', type=str, help='Get regime history for specific ticker')
    parser.add_argument('--batch-regime', type=int, help='Analyze market regime for first N tickers')
    
    args = parser.parse_args()
    
    ticker_scanner = PolygonTickerScanner()
    
    # Handle command line arguments
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

    if args.regime:
        # Analyze market regime for specific ticker
        await ticker_scanner.start()
        regime_data = await ticker_scanner.analyze_market_regime(args.regime)
        if regime_data:
            print(f"Market Regime Analysis for {args.regime}:")
            print(f"  Regime: {regime_data['regime_name']} ({regime_data['regime']})")
            print(f"  Signal: {regime_data['signal']} (1=Long, -1=Short, 0=Neutral)")
            print(f"  Close Price: ${regime_data['close_price']:.2f}")
            print(f"  RSI: {regime_data['rsi_14']:.2f}")
            print(f"  Trend Strength: {regime_data['trend_strength']:.4f}")
        else:
            print(f"Could not analyze market regime for {args.regime}")
        await ticker_scanner.shutdown()
        return

    if args.regime_stats:
        # Show market regime statistics
        await ticker_scanner.start()
        stats = await ticker_scanner.get_market_regime_stats()
        if stats:
            print("Market Regime Distribution:")
            for regime, count in stats.items():
                print(f"  {regime}: {count} tickers")
        else:
            print("No market regime data available")
        await ticker_scanner.shutdown()
        return

    if args.regime_history:
        # Get regime history for specific ticker
        await ticker_scanner.start()
        history = await ticker_scanner.get_ticker_regime_history(args.regime_history, 10)
        if history:
            print(f"Market Regime History for {args.regime_history}:")
            for entry in history:
                print(f"  {entry['date']}: {entry['regime_name']} (Signal: {entry['signal']})")
        else:
            print(f"No regime history found for {args.regime_history}")
        await ticker_scanner.shutdown()
        return

    if args.batch_regime:
        # Batch analyze market regime
        await ticker_scanner.start()
        active_tickers = ticker_scanner.get_current_tickers_list()
        if active_tickers:
            sample_size = min(args.batch_regime, len(active_tickers))
            sample_tickers = active_tickers[:sample_size]
            print(f"Analyzing market regime for {sample_size} tickers...")
            results = await ticker_scanner.batch_analyze_market_regime(sample_tickers)
            
            # Show summary
            regime_counts = {}
            for ticker, data in results.items():
                if data:
                    regime_name = data['regime_name']
                    regime_counts[regime_name] = regime_counts.get(regime_name, 0) + 1
            
            print("Market Regime Summary:")
            for regime, count in regime_counts.items():
                print(f"  {regime}: {count} tickers")
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