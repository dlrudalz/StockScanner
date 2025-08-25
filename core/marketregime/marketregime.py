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
from threading import Lock, Event
import sys
import requests
from tzlocal import get_localzone
import pandas_ta as ta
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import pickle
import hashlib

# ======================== CONFIGURATION ======================== #
class Config:
    # API Configuration
    POLYGON_API_KEY = "ld1Poa63U6t4Y2MwOCA2JeKQyHVrmyg8"
    
    # Scanner Configuration
    EXCHANGES = ["XNAS", "XNYS", "XASE"]
    MAX_CONCURRENT_REQUESTS = 50
    RATE_LIMIT_DELAY = 0.05  # 50ms between requests
    SCAN_TIME = "08:30"  # Daily scan time in local time
    STRATEGY_TIME = "10:00"  # Strategy execution time
    
    # File Management
    TICKER_CACHE_FILE = "ticker_cache.parquet"
    MISSING_TICKERS_FILE = "missing_tickers.json"
    METADATA_FILE = "scanner_metadata.json"
    STRATEGY_LOG_FILE = "strategy_decisions.json"
    REGIME_MODEL_FILE = "regime_model.pkl"
    LIQUIDITY_CACHE_FILE = "liquidity_cache.pkl"
    DATA_VALIDATION_LOG = "data_validation.log"
    DATA_QUALITY_FILE = "data_quality_report.json"
    
    # Strategy Parameters
    MARKET_INDEX_COMPONENTS = 100  # Top N stocks for market index
    SECTOR_STOCKS_PER_GROUP = 30   # Stocks per sector group
    REGIME_LOOKBACK = 365           # 1 year + buffer for trading days
    MIN_STOCKS_FOR_SECTOR = 10      # Minimum stocks to form a sector group
    REGIME_UPDATE_DAYS = 7          # Update regime weekly
    SECTOR_ANALYSIS_DAYS = 90       # Lookback for sector analysis
    MIN_MARKET_DATA_DAYS = 60       # Minimum days required for regime detection
    MIN_ADV = 1000000               # $1M minimum average daily dollar volume
    LIQUIDITY_CACHE_DAYS = 7        # Cache liquidity data for 7 days
    
    # Recovery Strategy Parameters
    RECOVERY_MIN_RSI = 35            # Maximum RSI for recovery candidates
    RECOVERY_MIN_PCT_BELOW_200DMA = 0.20  # Min % below 200DMA
    RECOVERY_VOLUME_SURGE = 1.5      # Min volume surge multiplier
    
    # Data Validation Parameters
    MAX_DATA_GAP_DAYS = 1            # Max allowed gap in days in price data
    MIN_DATA_COMPLETENESS = 0.98     # Min % of expected data points
    MIN_CORRELATION_THRESHOLD = 0.7  # Min correlation for sector validation
    MAX_SECTOR_OVERLAP = 0.05        # Max % of stocks overlapping between sectors
    DATA_HASH_FILE = "data_hashes.json"
    MAX_FETCH_RETRIES = 3            # Max retries for data fetching
    
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
    logger = logging.getLogger("QuantSystem")
    logger.setLevel(config.LOG_LEVEL)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(config.LOG_FORMAT)
    
    # File handler - all logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/quant_system_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(config.LOG_LEVEL)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Validation log handler
    validation_handler = logging.FileHandler(config.DATA_VALIDATION_LOG)
    validation_handler.setLevel(logging.WARNING)
    validation_handler.setFormatter(formatter)
    logger.addHandler(validation_handler)
    
    # Console handler - info and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(config.LOG_LEVEL)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# ======================== DATA QUALITY MONITOR ======================== #
class DataQualityMonitor:
    def __init__(self):
        self.quality_metrics = {}
        
    def track_symbol_quality(self, symbol, df):
        """Track data quality metrics for each symbol"""
        if df.empty:
            self.quality_metrics[symbol] = {
                'status': 'FAILED',
                'completeness': 0,
                'last_update': datetime.now().isoformat()
            }
            return
            
        # Calculate completeness
        days_span = (df.index.max() - df.index.min()).days
        expected_days = int(days_span * 0.7)  # 70% of calendar days are trading days
        actual_days = len(df.dropna())
        completeness = actual_days / expected_days if expected_days > 0 else 0
        
        self.quality_metrics[symbol] = {
            'status': 'SUCCESS' if completeness >= config.MIN_DATA_COMPLETENESS else 'WARNING',
            'completeness': completeness,
            'data_points': actual_days,
            'expected_points': expected_days,
            'last_update': datetime.now().isoformat()
        }
        
    def get_quality_report(self):
        """Generate a data quality report"""
        successful = [s for s, m in self.quality_metrics.items() if m['status'] == 'SUCCESS']
        warnings = [s for s, m in self.quality_metrics.items() if m['status'] == 'WARNING']
        failed = [s for s, m in self.quality_metrics.items() if m['status'] == 'FAILED']
        
        successful_completeness = [m['completeness'] for s, m in self.quality_metrics.items() 
                                  if m['status'] == 'SUCCESS']
        avg_completeness = np.mean(successful_completeness) if successful_completeness else 0
        
        return {
            'total_symbols': len(self.quality_metrics),
            'successful': len(successful),
            'warnings': len(warnings),
            'failed': len(failed),
            'avg_completeness': avg_completeness,
            'details': self.quality_metrics
        }
    
    def save_quality_report(self):
        """Save quality report to file"""
        try:
            report = self.get_quality_report()
            with open(config.DATA_QUALITY_FILE, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Data quality report saved: {report['successful']}/{report['total_symbols']} successful")
        except Exception as e:
            logger.error(f"Failed to save quality report: {e}")

# ======================== DATA VALIDATION UTILITIES ======================== #
class DataValidator:
    @staticmethod
    def calculate_data_hash(data):
        """Calculate SHA256 hash of data for integrity checking"""
        if isinstance(data, pd.DataFrame):
            return hashlib.sha256(pd.util.hash_pandas_object(data).values).hexdigest()
        elif isinstance(data, dict):
            return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        return ""
    
    @staticmethod
    def validate_price_data(df, symbol):
        """Validate price data for completeness and quality"""
        issues = []
        
        # 1. Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            issues.append(f"Missing values: {missing_values.to_dict()}")
        
        # 2. Check for duplicates
        duplicate_dates = df.index.duplicated().sum()
        if duplicate_dates > 0:
            issues.append(f"{duplicate_dates} duplicate dates found")
        
        # 3. Check for gaps in data
        date_diff = df.index.to_series().diff().dt.days
        gap_days = date_diff[date_diff > 1]
        if not gap_days.empty:
            max_gap = gap_days.max()
            if max_gap > config.MAX_DATA_GAP_DAYS:
                issues.append(f"Data gaps up to {max_gap} days")
        
        # 4. Check data completeness
        expected_days = (df.index.max() - df.index.min()).days + 1
        actual_days = len(df)
        completeness = actual_days / expected_days
        if completeness < config.MIN_DATA_COMPLETENESS:
            issues.append(f"Low completeness: {completeness:.2%} ({actual_days}/{expected_days} days)")
        
        # 5. Check for outliers
        price_changes = df['close'].pct_change().abs()
        outlier_threshold = price_changes.quantile(0.99) * 3
        outliers = price_changes[price_changes > outlier_threshold]
        if not outliers.empty:
            issues.append(f"{len(outliers)} price outliers detected")
        
        # 6. Volume validation
        zero_volume = (df['volume'] <= 0).sum()
        if zero_volume > 0:
            issues.append(f"{zero_volume} days with zero volume")
        
        if issues:
            logger.warning(f"Data issues for {symbol}: {' | '.join(issues)}")
            return False
        return True
    
    @staticmethod
    def validate_sector_groups(sector_groups):
        """Ensure sector groups don't have excessive overlap"""
        all_stocks = []
        for sector, stocks in sector_groups.items():
            all_stocks.extend(stocks)
        
        total_stocks = len(all_stocks)
        unique_stocks = len(set(all_stocks))
        overlap = total_stocks - unique_stocks
        
        if overlap > 0:
            overlap_pct = overlap / total_stocks
            if overlap_pct > config.MAX_SECTOR_OVERLAP:
                logger.warning(f"High sector overlap: {overlap_pct:.2%} ({overlap} duplicates)")
                return False
        return True
    
    @staticmethod
    def validate_index_components(components, sector_groups):
        """Ensure market index components don't overlap with sector groups"""
        sector_stocks = set()
        for stocks in sector_groups.values():
            sector_stocks.update(stocks)
        
        overlap = set(components) & sector_stocks
        if overlap:
            logger.warning(f"{len(overlap)} stocks overlap between market index and sector groups")
            return False
        return True
    
    @staticmethod
    def validate_sector_returns(returns_dict):
        """Validate sector returns for correlation consistency"""
        sectors = list(returns_dict.keys())
        corr_matrix = pd.DataFrame()
        
        for sector, returns in returns_dict.items():
            corr_matrix[sector] = returns
        
        corr_matrix = corr_matrix.corr()
        avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].mean()
        
        if avg_correlation < config.MIN_CORRELATION_THRESHOLD:
            logger.warning(f"Low sector correlation: {avg_correlation:.4f}")
            return False
        return True
    
    @staticmethod
    def save_data_hash(data_type, data_hash):
        """Save data hash for future validation"""
        try:
            if os.path.exists(config.DATA_HASH_FILE):
                with open(config.DATA_HASH_FILE, 'r') as f:
                    hashes = json.load(f)
            else:
                hashes = {}
                
            hashes[data_type] = data_hash
            
            with open(config.DATA_HASH_FILE, 'w') as f:
                json.dump(hashes, f)
        except Exception as e:
            logger.error(f"Failed to save data hash: {e}")
    
    @staticmethod
    def verify_data_hash(data_type, current_hash):
        """Verify data hasn't changed unexpectedly"""
        try:
            if not os.path.exists(config.DATA_HASH_FILE):
                return True
                
            with open(config.DATA_HASH_FILE, 'r') as f:
                hashes = json.load(f)
                
            previous_hash = hashes.get(data_type, "")
            if previous_hash and previous_hash != current_hash:
                logger.warning(f"Data integrity check failed for {data_type}")
                return False
        except Exception as e:
            logger.error(f"Hash verification failed: {e}")
        return True

# Initialize validator
data_validator = DataValidator()

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
            "ticker", "name", "primary_exchange", "last_updated_utc", "type", "sector"
        ])
        self.current_tickers_set = set()
        self.local_tz = get_localzone()
        logger.info(f"Using local timezone: {self.local_tz}")
        
    def _init_cache(self):
        """Initialize or load ticker cache from disk with validation"""
        metadata = self._load_metadata()
        self.last_refresh_time = metadata.get('last_refresh_time', 0)
        
        if os.path.exists(config.TICKER_CACHE_FILE):
            try:
                # Load with validation
                self.ticker_cache = self._load_and_validate_cache()
                logger.info(f"Loaded ticker cache with {len(self.ticker_cache)} symbols")
                
                # Verify data integrity
                cache_hash = data_validator.calculate_data_hash(self.ticker_cache)
                if not data_validator.verify_data_hash("ticker_cache", cache_hash):
                    logger.warning("Ticker cache integrity check failed, forcing refresh")
                    self.ticker_cache = pd.DataFrame(columns=[
                        "ticker", "name", "primary_exchange", "last_updated_utc", "type", "sector"
                    ])
            except Exception as e:
                logger.error(f"Cache load error: {e}")
                self.ticker_cache = pd.DataFrame(columns=[
                    "ticker", "name", "primary_exchange", "last_updated_utc", "type", "sector"
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

    def _load_and_validate_cache(self):
        """Load cache with data validation checks"""
        df = pd.read_parquet(config.TICKER_CACHE_FILE)
        
        # Basic validation checks
        if df.empty:
            logger.warning("Loaded empty ticker cache")
            return df
            
        # Check for duplicate tickers
        duplicates = df['ticker'].duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Removed {duplicates} duplicate tickers from cache")
            df = df.drop_duplicates(subset='ticker', keep='last')
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values in cache: {missing_values.to_dict()}")
            
            # Handle missing sectors
            if 'sector' in missing_values and missing_values['sector'] > 0:
                df['sector'] = df['sector'].fillna('Unknown')
        
        return df

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
            stock_results = [
                r for r in results 
                if r.get('type', '').upper() == 'CS' and 
                r.get('currency_name', '').upper() == 'USD'
            ]
            all_results.extend(stock_results)
            
            next_url = data.get("next_url", None)
            if not next_url:
                break
                
            # Minimal delay for premium API access
            await asyncio.sleep(config.RATE_LIMIT_DELAY)
        
        # Process results to extract sector information
        processed = []
        for r in all_results:
            processed.append({
                "ticker": r.get('ticker', ''),
                "name": r.get('name', ''),
                "primary_exchange": r.get('primary_exchange', ''),
                "last_updated_utc": r.get('last_updated_utc', ''),
                "type": r.get('type', ''),
                "sector": r.get('sic_description', '')  # Use SIC description as sector
            })
        
        logger.info(f"Completed {exchange}: {len(processed)} stocks")
        return processed

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
            return False
            
        new_df = pd.DataFrame(all_results)
        
        # Data validation: Check for duplicates
        duplicates = new_df['ticker'].duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Removed {duplicates} duplicate tickers from refresh results")
            new_df = new_df.drop_duplicates(subset='ticker', keep='last')
        
        new_tickers = set(new_df['ticker'].tolist())
        
        with self.cache_lock:
            old_tickers = set(self.current_tickers_set)
            added = new_tickers - old_tickers
            removed = old_tickers - new_tickers
            
            self.ticker_cache = new_df
            self.ticker_cache.to_parquet(config.TICKER_CACHE_FILE)
            self.current_tickers_set = new_tickers
            
            # Save data hash for integrity checking
            cache_hash = data_validator.calculate_data_hash(new_df)
            data_validator.save_data_hash("ticker_cache", cache_hash)
            
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

    def get_tickers_with_metadata(self):
        """Return tickers with sector metadata"""
        with self.cache_lock:
            return self.ticker_cache.copy()

# ======================== HISTORICAL DATA FETCHER ======================== #
class HistoricalDataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v2/aggs/ticker"
        
    async def fetch_data(self, session, symbol, timespan='day', multiplier=1):
        """Fetch historical price data with multiple fallback strategies"""
        # Try multiple date ranges to ensure completeness
        date_ranges = [
            (datetime.now() - timedelta(days=int(config.REGIME_LOOKBACK * 1.8)), datetime.now()),  # Primary with buffer
            (datetime.now() - timedelta(days=int(config.REGIME_LOOKBACK * 2.5)), datetime.now()),  # Extended fallback
        ]
        
        for start_date, end_date in date_ranges:
            params = {
                "adjusted": "true",
                "sort": "asc",
                "limit": 50000,
                "apiKey": self.api_key
            }
            url = f"{self.base_url}/{symbol}/range/{multiplier}/{timespan}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}?{urlencode(params)}"
            
            try:
                async with session.get(url, timeout=20) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("resultsCount", 0) > 0:
                            df = pd.DataFrame(data['results'])
                            df['date'] = pd.to_datetime(df['t'], unit='ms')
                            df.set_index('date', inplace=True)
                            df = df[['c', 'h', 'l', 'v', 'vw', 'n']]
                            df.columns = ['close', 'high', 'low', 'volume', 'vwap', 'transactions']
                            
                            # Validate we have enough data
                            if self.validate_data_completeness(df, symbol):
                                return self.clean_data(df, symbol)
            except Exception as e:
                logger.warning(f"Attempt failed for {symbol}: {e}")
                continue
        
        logger.error(f"All attempts failed for {symbol}")
        return pd.DataFrame()
    
    def validate_data_completeness(self, df, symbol):
        """Validate if we have sufficient trading days"""
        if df.empty:
            return False
            
        # Calculate expected trading days (approx 252 per year)
        days_span = (df.index.max() - df.index.min()).days
        expected_days = int(days_span * 0.7)  # 70% of calendar days are trading days
        
        # Count actual trading days with data
        actual_days = len(df.dropna())
        
        completeness = actual_days / expected_days
        if completeness < config.MIN_DATA_COMPLETENESS:
            logger.warning(f"Incomplete data for {symbol}: {completeness:.2%}")
            return False
            
        return True

    def clean_data(self, df, symbol):
        """Clean and normalize price data with enhanced gap handling"""
        if df.empty:
            return df
            
        # 1. Remove duplicate dates (keep last)
        df = df[~df.index.duplicated(keep='last')]
        
        # 2. Create complete date index for trading days
        full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        df = df.reindex(full_index)
        
        # 3. Handle missing values - forward fill then backfill
        df = df.ffill().bfill()
        
        # 4. Validate data quality
        data_validator.validate_price_data(df, symbol)
        
        return df

    async def fetch_with_retry(self, session, symbol, max_retries=3):
        """Fetch data with retry mechanism"""
        for attempt in range(max_retries):
            try:
                df = await self.fetch_data(session, symbol)
                if not df.empty and self.validate_data_completeness(df, symbol):
                    return df
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed for {symbol}: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
        logger.error(f"All {max_retries} attempts failed for {symbol}")
        return pd.DataFrame()

    async def fetch_bulk_data(self, session, symbols):
        """Fetch historical data for multiple symbols efficiently with retries"""
        tasks = []
        for symbol in symbols:
            tasks.append(self.fetch_with_retry(session, symbol, config.MAX_FETCH_RETRIES))
        return await asyncio.gather(*tasks)

# ======================== QUANT STRATEGY ENGINE ======================== #
class MarketRegimeEngine:
    def __init__(self, scanner):
        self.scanner = scanner
        self.data_fetcher = HistoricalDataFetcher(config.POLYGON_API_KEY)
        self.quality_monitor = DataQualityMonitor()
        self.market_regime = "UNKNOWN"
        self.sector_regimes = {}
        self.regime_model = None
        self.last_regime_update = 0
        self.sector_mapping = {}
        self.market_index_components = []
        self.liquidity_cache = {}
        self.last_liquidity_update = 0
        self.valid_price_data = {}  # Cache for individual stock price data
        
        # Load liquidity cache if exists
        self._load_liquidity_cache()
        
    def _load_liquidity_cache(self):
        """Load liquidity cache from disk with validation"""
        if os.path.exists(config.LIQUIDITY_CACHE_FILE):
            try:
                with open(config.LIQUIDITY_CACHE_FILE, 'rb') as f:
                    self.liquidity_cache = pickle.load(f)
                
                # Validate cache structure
                if not isinstance(self.liquidity_cache, dict):
                    raise ValueError("Invalid liquidity cache format")
                    
                # Verify data integrity
                cache_hash = data_validator.calculate_data_hash(self.liquidity_cache)
                if not data_validator.verify_data_hash("liquidity_cache", cache_hash):
                    logger.warning("Liquidity cache integrity check failed, forcing refresh")
                    self.liquidity_cache = {}
                else:
                    logger.info(f"Loaded liquidity cache with {len(self.liquidity_cache)} entries")
            except Exception as e:
                logger.error(f"Failed to load liquidity cache: {e}")
                self.liquidity_cache = {}
        else:
            self.liquidity_cache = {}
            
    def _save_liquidity_cache(self):
        """Save liquidity cache to disk with integrity hash"""
        try:
            with open(config.LIQUIDITY_CACHE_FILE, 'wb') as f:
                pickle.dump(self.liquidity_cache, f)
            
            # Save data hash for future validation
            cache_hash = data_validator.calculate_data_hash(self.liquidity_cache)
            data_validator.save_data_hash("liquidity_cache", cache_hash)
        except Exception as e:
            logger.error(f"Failed to save liquidity cache: {e}")
            
    async def fetch_liquidity_data(self, symbols):
        """Fetch liquidity data (ADV) for symbols with caching"""
        logger.info(f"Fetching liquidity data for {len(symbols)} symbols")
        
        # Check cache validity
        cache_valid = time.time() - self.last_liquidity_update < (config.LIQUIDITY_CACHE_DAYS * 86400)
        cached_results = {}
        new_symbols = []
        
        # Check cache for existing data
        if cache_valid:
            for symbol in symbols:
                if symbol in self.liquidity_cache:
                    cached_results[symbol] = self.liquidity_cache[symbol]
                else:
                    new_symbols.append(symbol)
        else:
            new_symbols = symbols
            self.liquidity_cache = {}
        
        # Fetch new data if needed
        if new_symbols:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
            
            async with aiohttp.ClientSession() as session:
                all_data = await self.data_fetcher.fetch_bulk_data(session, new_symbols)
            
            # Process results
            new_results = {}
            for symbol, df in zip(new_symbols, all_data):
                if not df.empty:
                    # Calculate average dollar volume (ADV)
                    df['dollar_volume'] = df['close'] * df['volume']
                    adv = df['dollar_volume'].mean()
                    new_results[symbol] = adv
                else:
                    new_results[symbol] = 0
            
            # Update cache
            self.liquidity_cache.update(new_results)
            self.last_liquidity_update = time.time()
            self._save_liquidity_cache()
            cached_results.update(new_results)
            
            logger.info(f"Fetched liquidity data for {len(new_symbols)} symbols")
        
        return cached_results
        
    async def build_market_index(self):
        """Create market index from most liquid scanner stocks with quality filtering"""
        logger.info("Building market index from most liquid stocks")
        
        # Get current tickers with metadata
        tickers_df = self.scanner.get_tickers_with_metadata()
        if tickers_df.empty:
            logger.error("No tickers available for market index")
            return []
            
        # Step 1: Pre-filter by exchange and stock type
        filtered_df = tickers_df[
            (tickers_df['primary_exchange'].isin(config.EXCHANGES)) &
            (tickers_df['type'] == 'CS')
        ]
        
        if len(filtered_df) == 0:
            logger.error("No valid stocks found after filtering")
            return []
            
        # Step 2: Take top 500 candidates for liquidity check
        candidate_symbols = filtered_df['ticker'].tolist()[:500]
        
        # Step 3: Fetch liquidity data
        liquidity_data = await self.fetch_liquidity_data(candidate_symbols)
        
        # Step 4: Create liquidity dataframe
        liquidity_df = pd.DataFrame({
            'ticker': list(liquidity_data.keys()),
            'adv': list(liquidity_data.values())
        })
        
        # Merge with ticker data
        merged_df = filtered_df.merge(liquidity_df, on='ticker', how='left')
        
        # Step 5: Filter by minimum liquidity
        merged_df = merged_df[merged_df['adv'] > config.MIN_ADV]
        
        if len(merged_df) == 0:
            logger.error("No liquid stocks found for market index")
            return []
            
        # Step 6: Check data quality for all candidates
        async with aiohttp.ClientSession() as session:
            all_data = await self.data_fetcher.fetch_bulk_data(session, merged_df['ticker'].tolist())
        
        # Track quality and filter
        high_quality_symbols = []
        for symbol, data in zip(merged_df['ticker'].tolist(), all_data):
            self.quality_monitor.track_symbol_quality(symbol, data)
            
            if not data.empty and self.quality_monitor.quality_metrics[symbol]['status'] == 'SUCCESS':
                high_quality_symbols.append(symbol)
                self.valid_price_data[symbol] = data  # Cache valid data
        
        # Prioritize high-quality symbols
        merged_df = merged_df[merged_df['ticker'].isin(high_quality_symbols)]
        
        # Generate quality report
        self.quality_monitor.save_quality_report()
        
        if len(merged_df) == 0:
            logger.error("No high-quality data stocks found for market index")
            return []
            
        # Step 7: Select top N by liquidity
        merged_df = merged_df.sort_values('adv', ascending=False)
        components = merged_df.head(config.MARKET_INDEX_COMPONENTS)['ticker'].tolist()
        
        self.market_index_components = components
        logger.info(f"Selected {len(components)} most liquid stocks for market index")
        return components
        
    async def create_sector_groups(self):
        """Create sector groups from scanner stocks with validation"""
        logger.info("Creating sector groups from scanner data")
        
        # Get current tickers with metadata
        tickers_df = self.scanner.get_tickers_with_metadata()
        if tickers_df.empty:
            logger.error("No tickers available for sector groups")
            return {}
            
        # Clean and categorize sectors
        tickers_df['sector'] = tickers_df['sector'].fillna('Unknown')
        tickers_df['sector'] = tickers_df['sector'].str.strip().str.title()
        
        # Group by sector and assign stocks
        sector_groups = {}
        assigned_stocks = set()
        
        # Process sectors with most stocks first
        sector_counts = tickers_df['sector'].value_counts()
        for sector in sector_counts.index:
            if sector_counts[sector] < config.MIN_STOCKS_FOR_SECTOR:
                continue
                
            # Get stocks not yet assigned
            sector_stocks = tickers_df[tickers_df['sector'] == sector]
            available_stocks = sector_stocks[~sector_stocks['ticker'].isin(assigned_stocks)]
            
            if len(available_stocks) < config.MIN_STOCKS_FOR_SECTOR:
                continue
                
            # Take a sample of stocks
            sample_size = min(config.SECTOR_STOCKS_PER_GROUP, len(available_stocks))
            stocks = available_stocks['ticker'].sample(sample_size, replace=False).tolist()
            sector_groups[sector] = stocks
            assigned_stocks.update(stocks)
        
        # Validate sector groups
        if not data_validator.validate_sector_groups(sector_groups):
            logger.warning("Sector group validation failed, attempting repair")
            # Simple repair: remove overlapping stocks
            all_stocks = set()
            for sector, stocks in sector_groups.items():
                unique_stocks = set(stocks) - all_stocks
                sector_groups[sector] = list(unique_stocks)
                all_stocks.update(unique_stocks)
        
        self.sector_mapping = sector_groups
        logger.info(f"Created {len(sector_groups)} sector groups")
        return sector_groups
        
    def calculate_index_returns(self, data_dict):
        """Calculate index returns from constituent data with validation"""
        # Find the common date range
        all_dates = set()
        for df in data_dict.values():
            if not df.empty:
                all_dates.update(df.index)
        
        if not all_dates:
            return pd.Series(dtype=float)
            
        all_dates = sorted(all_dates)
        min_date = min(all_dates)
        max_date = max(all_dates)
        
        # Create a date range that covers all available data
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        
        # Create index dataframe
        index_df = pd.DataFrame(index=date_range)
        
        for symbol, df in data_dict.items():
            if not df.empty:
                # Reindex to common date range
                aligned = df.reindex(date_range)
                
                # Fix chained assignment warning
                aligned = aligned.assign(
                    close=aligned['close'].ffill().bfill()
                )
                
                index_df[symbol] = aligned['close']
        
        # Drop columns with all NaNs
        index_df.dropna(axis=1, how='all', inplace=True)
        
        if index_df.empty:
            return pd.Series(dtype=float)
            
        # Calculate equal-weighted index returns
        index_df['index_value'] = index_df.mean(axis=1)
        index_df['returns'] = index_df['index_value'].pct_change()
        returns = index_df['returns'].dropna()
        
        # Validate returns
        if returns.isnull().any() or returns.isin([np.inf, -np.inf]).any():
            logger.warning("Invalid returns detected in index calculation")
            returns = returns.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        
        return returns
        
    async def analyze_market_regime(self):
        """Detect market regime using scanner-based index with validation"""
        logger.info("Analyzing market regime from exchange stocks")
        
        # Ensure we have market index components
        if not self.market_index_components:
            await self.build_market_index()
            
        # Fetch data for all components
        async with aiohttp.ClientSession() as session:
            all_data = await self.data_fetcher.fetch_bulk_data(session, self.market_index_components)
        
        # Create data dictionary and cache valid price data
        data_dict = dict(zip(self.market_index_components, all_data))
        self.valid_price_data = {k: v for k, v in data_dict.items() if not v.empty}
        
        if len(self.valid_price_data) < 10:
            logger.error("Insufficient data for market index construction")
            return
            
        # Calculate index returns
        returns = self.calculate_index_returns(self.valid_price_data)
        if returns.empty or len(returns) < config.MIN_MARKET_DATA_DAYS:
            logger.error("Insufficient returns data for regime detection")
            return
            
        # Calculate features for regime detection
        features = pd.DataFrame()
        features['returns'] = returns
        features['volatility'] = returns.rolling(14).std()
        
        # Add new features for recovery detection
        features['recovery_strength'] = returns.rolling(5).mean() / returns.rolling(20).std()
        features['high_low_ratio'] = (returns > 0).rolling(10).sum() / 10.0
        features.dropna(inplace=True)
        
        if len(features) < 30:
            logger.error("Insufficient feature data for regime detection")
            return
            
        # Prepare data for HMM
        X = features.values
        scaler = StandardScaler()
        scaled_X = scaler.fit_transform(X)
        
        # Train or update HMM model with 4 states (now including RECOVERY)
        if self.regime_model is None:
            logger.info("Initializing new HMM model with 4 states")
            self.regime_model = hmm.GaussianHMM(
                n_components=4,  # Now 4 states: BEAR, BULL, SIDEWAYS, RECOVERY
                covariance_type="diag", 
                n_iter=100
            )
            
        self.regime_model.fit(scaled_X)
        regimes = self.regime_model.predict(scaled_X)
        current_regime = regimes[-1]
        
        # Get the means of the features for each state
        means = self.regime_model.means_
        
        # Create a scoring system for each state
        state_scores = []
        for i in range(4):
            ret_mean = means[i, 0]   # returns
            vol_mean = means[i, 1]   # volatility
            
            # Calculate scores for each regime type
            recovery_score = ret_mean + vol_mean    # High returns + high volatility
            bull_score = ret_mean - vol_mean        # High returns - low volatility
            bear_score = -ret_mean + vol_mean       # Negative returns + high volatility
            sideways_score = -abs(ret_mean) - vol_mean  # Low absolute returns and volatility
            
            state_scores.append({
                'state': i,
                'recovery': recovery_score,
                'bull': bull_score,
                'bear': bear_score,
                'sideways': sideways_score
            })
        
        # Assign each state to the regime with the highest score
        regime_assignment = {}
        for state in state_scores:
            regime = max(['recovery', 'bull', 'bear', 'sideways'], key=lambda r: state[r])
            regime_assignment[state['state']] = regime.upper()
        
        # Set current market regime
        self.market_regime = regime_assignment.get(current_regime, "UNKNOWN")
        self.last_regime_update = time.time()
        logger.info(f"Detected market regime: {self.market_regime}")
        return self.market_regime
        
    async def analyze_sector_regimes(self):
        """Analyze sector regimes using scanner stocks with validation"""
        logger.info("Analyzing sector regimes from exchange stocks")
        
        # Ensure we have sector groups
        if not self.sector_mapping:
            await self.create_sector_groups()
            
        sector_regimes = {}
        returns_dict = {}  # For sector correlation validation
        
        for sector, symbols in self.sector_mapping.items():
            logger.info(f"Analyzing {sector} sector with {len(symbols)} stocks")
            
            # Fetch data for all symbols in sector
            async with aiohttp.ClientSession() as session:
                all_data = await self.data_fetcher.fetch_bulk_data(session, symbols)
            
            # Create data dictionary
            data_dict = dict(zip(symbols, all_data))
            
            # Filter out empty dataframes
            valid_data = {k: v for k, v in data_dict.items() if not v.empty}
            if len(valid_data) < config.MIN_STOCKS_FOR_SECTOR // 2:
                logger.warning(f"Insufficient data for {sector} sector analysis")
                continue
                
            # Calculate sector returns
            returns = self.calculate_index_returns(valid_data)
            if returns.empty:
                logger.warning(f"Failed to calculate returns for {sector} sector")
                continue
            
            # Store for correlation validation
            returns_dict[sector] = returns
                
            # Calculate momentum metrics
            momentum_1m = returns.tail(21).mean() * 21  # Approximate 1-month momentum
            momentum_5d = returns.tail(5).mean() * 5     # 5-day momentum
            volatility = returns.std()
            
            # Calculate RSI if enough data
            rsi = np.nan
            if len(returns) > 14:
                try:
                    rsi = ta.rsi(pd.Series(returns, index=returns.index), length=14).iloc[-1]
                except Exception as e:
                    logger.warning(f"RSI calculation failed for {sector}: {e}")
            
            # Determine sector regime
            if momentum_1m > 0.05 and volatility < 0.015:
                regime = "STRONG"
            elif momentum_1m > 0.02:
                regime = "BULLISH"
            elif momentum_1m < -0.05 and volatility > 0.025:
                regime = "WEAK"
            elif momentum_1m < -0.02:
                regime = "BEARISH"
            else:
                regime = "NEUTRAL"
                
            # Add RSI-based qualification if available
            if not np.isnan(rsi):
                if regime in ["STRONG", "BULLISH"] and rsi > 70:
                    regime = "OVERBOUGHT"
                elif regime in ["WEAK", "BEARISH"] and rsi < 30:
                    regime = "OVERSOLD"
                    
            sector_regimes[sector] = {
                "regime": regime,
                "momentum_1m": momentum_1m,
                "momentum_5d": momentum_5d,
                "volatility": volatility,
                "rsi": rsi,
                "stocks": symbols
            }
        
        # Validate sector correlations
        if not data_validator.validate_sector_returns(returns_dict):
            logger.warning("Sector returns validation failed")
        
        self.sector_regimes = sector_regimes
        logger.info("Sector regime analysis completed")
        return sector_regimes

    def generate_trading_signals(self):
        """Generate trading signals based on exchange-specific regimes"""
        logger.info("Generating trading signals")
        signals = {}
        
        # Market regime-based strategy adjustments
        if self.market_regime == "BULL":
            # Bull market: Focus on strong sectors
            strong_sectors = [s for s, data in self.sector_regimes.items() 
                             if data["regime"] in ["STRONG", "BULLISH"]]
            
            # Select top 3 strongest sectors by momentum
            strong_sectors.sort(key=lambda s: self.sector_regimes[s]["momentum_1m"], reverse=True)
            selected_sectors = strong_sectors[:3]
            
            # Allocate to selected sectors
            weight_per_sector = 0.75 / len(selected_sectors) if selected_sectors else 0
            for sector in selected_sectors:
                # Select top 2 stocks in sector
                stocks = self.sector_regimes[sector]["stocks"][:2]
                weight_per_stock = weight_per_sector / len(stocks)
                for stock in stocks:
                    signals[stock] = {
                        "action": "BUY",
                        "weight": weight_per_stock,
                        "reason": f"Top stock in strong {sector} sector (Bull market)"
                    }
            
            signals["CASH"] = {
                "action": "HOLD",
                "weight": 0.25,
                "reason": "Cash reserve in bull market"
            }
            
        elif self.market_regime == "BEAR":
            # Bear market: Focus on defensive sectors
            defensive_sectors = [s for s, data in self.sector_regimes.items() 
                                if data["regime"] in ["NEUTRAL", "OVERSOLD"] and
                                data["volatility"] < 0.02]
            
            # Allocate to defensive sectors
            weight_per_sector = 0.6 / len(defensive_sectors) if defensive_sectors else 0
            for sector in defensive_sectors:
                # Select most stable stock in sector
                stocks = self.sector_regimes[sector]["stocks"][:1]  # Just top stock
                for stock in stocks:
                    signals[stock] = {
                        "action": "BUY",
                        "weight": weight_per_sector,
                        "reason": f"Defensive stock in {sector} sector (Bear market)"
                    }
            
            # Short weakest sector
            weak_sectors = [s for s, data in self.sector_regimes.items() 
                           if data["regime"] in ["WEAK", "BEARISH"]]
            if weak_sectors:
                weakest_sector = min(weak_sectors, 
                                    key=lambda s: self.sector_regimes[s]["momentum_1m"])
                # Short weakest stock in weakest sector
                stock = self.sector_regimes[weakest_sector]["stocks"][0]
                signals[stock] = {
                    "action": "SHORT",
                    "weight": 0.2,
                    "reason": f"Weakest stock in weakest {weakest_sector} sector"
                }
            
            signals["CASH"] = {
                "action": "HOLD",
                "weight": 0.2,
                "reason": "Cash reserve in bear market"
            }
            
        elif self.market_regime == "RECOVERY":
            # Recovery market: Focus on deeply oversold sectors with improving momentum
            recovery_sectors = [s for s, data in self.sector_regimes.items() 
                              if not np.isnan(data["rsi"]) and 
                                 data["rsi"] < config.RECOVERY_MIN_RSI and 
                                 data["momentum_5d"] > 0]  # Positive short-term momentum
            
            # Select strongest 3 recovery candidates by RSI (lowest RSI first)
            recovery_sectors.sort(key=lambda s: self.sector_regimes[s]["rsi"])
            selected_sectors = recovery_sectors[:3]
            
            # Allocate 60% of portfolio to recovery candidates
            weight_per_sector = 0.6 / len(selected_sectors) if selected_sectors else 0
            
            for sector in selected_sectors:
                # Get stocks in this sector
                stocks = self.sector_regimes[sector]["stocks"]
                
                # Filter stocks by liquidity and technical criteria
                valid_stocks = []
                for stock in stocks:
                    # Skip if we don't have price data
                    if stock not in self.valid_price_data:
                        continue

                    df = self.valid_price_data[stock]
                    
                    # Check if we have enough data for 200DMA (at least 200 days)
                    if len(df) < 200:
                        continue
                        
                    # Calculate technical metrics
                    current_price = df['close'].iloc[-1]
                    ma_200 = df['close'].rolling(200).mean().iloc[-1]
                    ma_20 = df['close'].rolling(20).mean().iloc[-1]
                    
                    # Calculate percentage below 200DMA
                    pct_below_200dma = (ma_200 - current_price) / ma_200
                    
                    # Check if stock meets recovery criteria
                    if (pct_below_200dma >= config.RECOVERY_MIN_PCT_BELOW_200DMA and
                        current_price > ma_20 and
                        self.liquidity_cache.get(stock, 0) > config.MIN_ADV * config.RECOVERY_VOLUME_SURGE):
                        valid_stocks.append(stock)
                
                # Take top 2 valid stocks by liquidity
                valid_stocks.sort(key=lambda s: self.liquidity_cache.get(s, 0), reverse=True)
                for stock in valid_stocks[:2]:
                    signals[stock] = {
                        "action": "BUY",
                        "weight": weight_per_stock / 2,
                        "reason": f"Recovery candidate in {sector} sector",
                        "holding_period": "short"  # 5-15 day horizon
                    }
            
            # 40% cash buffer for volatility management
            signals["CASH"] = {
                "action": "HOLD",
                "weight": 0.4,
                "reason": "Cash buffer for recovery volatility"
            }
            
        else:  # Sideways or unknown
            # Sideways market: Mean-reversion strategy
            mean_reversion_candidates = []
            for sector, data in self.sector_regimes.items():
                if data["regime"] == "OVERSOLD":
                    # Long candidates
                    for stock in data["stocks"][:2]:  # Top 2 candidates
                        mean_reversion_candidates.append((stock, "BUY", sector))
                elif data["regime"] == "OVERBOUGHT":
                    # Short candidates
                    for stock in data["stocks"][:2]:  # Top 2 candidates
                        mean_reversion_candidates.append((stock, "SHORT", sector))
            
            # Allocate to candidates
            if mean_reversion_candidates:
                weight_per_position = 0.8 / len(mean_reversion_candidates)
                for stock, action, sector in mean_reversion_candidates:
                    signals[stock] = {
                        "action": action,
                        "weight": weight_per_position,
                        "reason": f"Mean-reversion in {sector} sector (Sideways market)"
                    }
            
            signals["CASH"] = {
                "action": "HOLD",
                "weight": 0.2,
                "reason": "Cash reserve in sideways market"
            }
        
        return signals
        
    def save_strategy_decisions(self, signals):
        """Save strategy decisions to JSON file"""
        decision = {
            "timestamp": datetime.now().isoformat(),
            "market_regime": self.market_regime,
            "sector_regimes": self.sector_regimes,
            "signals": signals
        }
        
        try:
            # Read existing data
            if os.path.exists(config.STRATEGY_LOG_FILE):
                with open(config.STRATEGY_LOG_FILE, 'r') as f:
                    existing = json.load(f)
            else:
                existing = []
                
            # Append new decision
            existing.append(decision)
            
            # Save back to file
            with open(config.STRATEGY_LOG_FILE, 'w') as f:
                json.dump(existing, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save strategy decisions: {e}")

    async def execute_strategy(self):
        """Run full strategy pipeline"""
        logger.info("Executing quant strategy")
        
        # Update market regime (weekly)
        if time.time() - self.last_regime_update > (config.REGIME_UPDATE_DAYS * 86400):
            logger.info("Updating market regime (weekly)")
            await self.analyze_market_regime()
        
        # Update sector regimes (daily)
        logger.info("Updating sector regimes (daily)")
        await self.analyze_sector_regimes()
        
        # Generate trading signals
        signals = self.generate_trading_signals()
        
        # Save decisions
        self.save_strategy_decisions(signals)
        
        # Log signals
        logger.info(f"Market Regime: {self.market_regime}")
        logger.info("Sector Regimes:")
        for sector, data in self.sector_regimes.items():
            logger.info(f"- {sector}: {data['regime']} (Momentum: {data['momentum_1m']:.2%}, Vol: {data['volatility']:.4f})")
        
        logger.info("Generated Signals:")
        for asset, signal in signals.items():
            logger.info(f"{asset}: {signal['action']} ({signal['weight']*100:.1f}%) - {signal['reason']}")
        
        return signals

# ======================== SCHEDULERS ======================== #
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
    except Exception as e:
        logger.error(f"Error during initial scan: {e}")
    
    # Continue with daily scans
    while scanner.active:
        now = datetime.now(scanner.local_tz)
        
        # Calculate next run time
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
        logger.info(f"Next refresh scheduled at {target_datetime} ({sleep_seconds:.0f} seconds)")
        
        # Wait until scheduled time
        await asyncio.sleep(sleep_seconds)
        
        # Run the refresh
        logger.info("Starting scheduled refresh")
        try:
            success = await scanner.refresh_all_tickers()
            if success:
                logger.info("Scheduled refresh completed successfully")
            else:
                logger.warning("Scheduled refresh encountered errors")
        except Exception as e:
            logger.error(f"Error during scheduled refresh: {e}")

async def run_quant_strategy(strategy_engine):
    """Run quant strategy on a scheduled basis"""
    # Wait for scanner to initialize
    await asyncio.sleep(10)
    
    # Initialize market regime
    try:
        await strategy_engine.analyze_market_regime()
    except Exception as e:
        logger.error(f"Initial market regime analysis failed: {e}")
    
    # Continue with daily execution
    while True:
        now = datetime.now(strategy_engine.scanner.local_tz)
        
        # Calculate next run time
        target_time = datetime.strptime(config.STRATEGY_TIME, "%H:%M").time()
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
        logger.info(f"Next strategy run at {target_datetime} ({sleep_seconds:.0f} seconds)")
        
        await asyncio.sleep(sleep_seconds)
        
        # Execute strategy
        try:
            await strategy_engine.execute_strategy()
        except Exception as e:
            logger.error(f"Strategy execution failed: {str(e)}")

# ======================== MAIN EXECUTION ======================== #
async def main():
    scanner = PolygonTickerScanner()
    scanner.start()
    
    # Wait for initial cache load
    scanner.initial_refresh_complete.wait()
    
    # Create strategy engine
    strategy_engine = MarketRegimeEngine(scanner)
    
    # Run both scanner and strategy in parallel
    await asyncio.gather(
        run_scheduled_refresh(scanner),
        run_quant_strategy(strategy_engine)
    )

if __name__ == "__main__":
    # Windows event loop policy
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("System stopped by user")