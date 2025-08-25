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
import signal
import requests
from tzlocal import get_localzone
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary
import threading
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import backoff

# ======================== CONFIGURATION ======================== #
@dataclass
class Config:
    # API Configuration
    POLYGON_API_KEY: str = os.getenv("OZzn0oK0H2yG6rpIvVhGfgXgnUTrL31z")
    
    # Scanner Configuration
    EXCHANGES: List[str] = None  # Will be initialized in __post_init__
    SCAN_UNIVERSE: List[str] = None  # Specific tickers to scan, None for all
    MAX_CONCURRENT_REQUESTS: int = 50
    RATE_LIMIT_DELAY: float = 0.05  # 50ms between requests
    SCAN_INTERVAL_MINUTES: int = 5  # Interval between scans during market hours
    PRE_MARKET_SCAN_TIME: str = "08:30"  # Pre-market scan time in local time
    MARKET_OPEN_TIME: str = "09:30"
    MARKET_CLOSE_TIME: str = "16:00"
    
    # Health Monitoring
    METRICS_PORT: int = 8000
    HEALTH_CHECK_INTERVAL: int = 30  # seconds
    
    # File Management
    DATA_DIR: str = "data"
    TICKER_CACHE_FILE: str = "ticker_cache.parquet"
    SIGNALS_FILE: str = "signals.parquet"
    MISSING_TICKERS_FILE: str = "missing_tickers.json"
    METADATA_FILE: str = "scanner_metadata.json"
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    def __post_init__(self):
        self.EXCHANGES = self.EXCHANGES or ["XNAS", "XNYS", "XASE"]
        os.makedirs(self.DATA_DIR, exist_ok=True)
        self.TICKER_CACHE_FILE = os.path.join(self.DATA_DIR, self.TICKER_CACHE_FILE)
        self.SIGNALS_FILE = os.path.join(self.DATA_DIR, self.SIGNALS_FILE)
        self.MISSING_TICKERS_FILE = os.path.join(self.DATA_DIR, self.MISSING_TICKERS_FILE)
        self.METADATA_FILE = os.path.join(self.DATA_DIR, self.METADATA_FILE)

# Initialize configuration
config = Config()

# ======================== METRICS SETUP ======================== #
# Metrics for monitoring
class ScannerMetrics:
    def __init__(self):
        # API Metrics
        self.api_requests_total = Counter('api_requests_total', 'Total API requests', ['endpoint', 'status'])
        self.api_request_duration = Histogram('api_request_duration_seconds', 'API request duration')
        self.api_rate_limit_events = Counter('api_rate_limit_events_total', 'API rate limit events')
        
        # Data Quality Metrics
        self.tickers_total = Gauge('tickers_total', 'Total number of tickers in universe')
        self.data_points_processed = Counter('data_points_processed_total', 'Total data points processed')
        self.missing_data_points = Counter('missing_data_points_total', 'Missing data points')
        self.data_quality_errors = Counter('data_quality_errors_total', 'Data quality errors')
        
        # Scanner Metrics
        self.scans_total = Counter('scans_total', 'Total scans performed')
        self.scan_duration = Summary('scan_duration_seconds', 'Time spent processing scans')
        self.signals_generated = Counter('signals_generated_total', 'Signals generated', ['signal_type'])
        self.scanner_errors = Counter('scanner_errors_total', 'Scanner errors')
        
        # System Metrics
        self.memory_usage = Gauge('memory_usage_bytes', 'Memory usage in bytes')
        self.cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percent')
        self.disk_usage = Gauge('disk_usage_bytes', 'Disk usage in bytes')
        self.uptime = Gauge('uptime_seconds', 'Uptime in seconds')
        
        # Last update timestamps
        self.last_scan_time = Gauge('last_scan_time_seconds', 'Last scan time in unixtime')
        self.last_successful_scan_time = Gauge('last_successful_scan_time_seconds', 'Last successful scan time in unixtime')
        self.last_data_refresh_time = Gauge('last_data_refresh_time_seconds', 'Last data refresh time in unixtime')
        
        self.start_time = time.time()

    def update_system_metrics(self):
        """Update system resource metrics"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            self.memory_usage.set(process.memory_info().rss)
            self.cpu_usage.set(process.cpu_percent())
            
            disk_usage = psutil.disk_usage('.')
            self.disk_usage.set(disk_usage.used)
            
            self.uptime.set(time.time() - self.start_time)
        except ImportError:
            # psutil not installed, skip these metrics
            pass

# Initialize metrics
metrics = ScannerMetrics()

# ======================== LOGGING SETUP ======================== #
def setup_logging():
    """Configure logging with file and console handlers"""
    # Create logs directory if not exists
    os.makedirs("logs", exist_ok=True)
    
    # Convert string level to logging level
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    
    # Create main logger
    logger = logging.getLogger("StockScanner")
    logger.setLevel(log_level)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(config.LOG_FORMAT)
    
    # File handler - all logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/scanner_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler - info and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# ======================== MARKET STATUS ======================== #
class MarketStatus(Enum):
    PRE_MARKET = "pre_market"
    OPEN = "open"
    POST_MARKET = "post_market"
    CLOSED = "closed"

def get_market_status(now: datetime = None) -> MarketStatus:
    """Determine the current market status"""
    if now is None:
        now = datetime.now(pytz.timezone('US/Eastern'))
    
    # Convert to Eastern time if needed
    if now.tzinfo is None:
        now = pytz.timezone('US/Eastern').localize(now)
    elif str(now.tzinfo) != 'US/Eastern':
        now = now.astimezone(pytz.timezone('US/Eastern'))
    
    time_str = now.strftime("%H:%M")
    weekday = now.weekday()
    
    # Market is closed on weekends
    if weekday >= 5:  # 5=Saturday, 6=Sunday
        return MarketStatus.CLOSED
    
    # Check market hours
    if time_str < config.PRE_MARKET_SCAN_TIME:
        return MarketStatus.CLOSED
    elif time_str < config.MARKET_OPEN_TIME:
        return MarketStatus.PRE_MARKET
    elif time_str < config.MARKET_CLOSE_TIME:
        return MarketStatus.OPEN
    else:
        return MarketStatus.POST_MARKET

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
            "ticker", "name", "primary_exchange", "last_updated_utc", "type", "market_cap"
        ])
        self.current_tickers_set = set()
        self.local_tz = get_localzone()
        logger.info(f"Using local timezone: {self.local_tz}")
        
    def _init_cache(self):
        """Initialize or load ticker cache from disk"""
        metadata = self._load_metadata()
        self.last_refresh_time = metadata.get('last_refresh_time', 0)
        
        if os.path.exists(config.TICKER_CACHE_FILE):
            try:
                self.ticker_cache = pd.read_parquet(config.TICKER_CACHE_FILE)
                logger.info(f"Loaded ticker cache with {len(self.ticker_cache)} symbols")
                metrics.tickers_total.set(len(self.ticker_cache))
            except Exception as e:
                logger.error(f"Cache load error: {e}")
                self.ticker_cache = pd.DataFrame(columns=[
                    "ticker", "name", "primary_exchange", "last_updated_utc", "type", "market_cap"
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

    @backoff.on_exception(backoff.expo, aiohttp.ClientError, max_tries=3)
    async def _call_polygon_api(self, session, url):
        start_time = time.time()
        try:
            async with session.get(url, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    metrics.api_requests_total.labels(endpoint=url.split('?')[0], status='success').inc()
                    metrics.api_request_duration.observe(time.time() - start_time)
                    return data
                elif response.status == 429:
                    metrics.api_requests_total.labels(endpoint=url.split('?')[0], status='rate_limited').inc()
                    metrics.api_rate_limit_events.inc()
                    retry_after = int(response.headers.get('Retry-After', 5))
                    logger.warning(f"Rate limit hit, retrying after {retry_after} seconds")
                    await asyncio.sleep(retry_after)
                    raise aiohttp.ClientResponseError(
                        status=429,
                        message="Rate limit exceeded",
                        request_info=response.request_info
                    )
                else:
                    metrics.api_requests_total.labels(endpoint=url.split('?')[0], status='error').inc()
                    logger.error(f"API request failed: {response.status}")
                    return None
        except asyncio.CancelledError:
            logger.debug("API request cancelled")
            raise
        except Exception as e:
            metrics.api_requests_total.labels(endpoint=url.split('?')[0], status='exception').inc()
            logger.error(f"API request exception: {e}")
            raise

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
        
        while url and self.active:
            data = await self._call_polygon_api(session, url)
            if not data:
                break
                
            results = data.get("results", [])
            # Filter for common stocks only
            stock_results = [r for r in results if r.get('type', '').upper() == 'CS']
            all_results.extend(stock_results)
            
            next_url = data.get("next_url", None)
            url = f"{next_url}&apiKey={self.api_key}" if next_url else None
            
            # Minimal delay for premium API access
            await asyncio.sleep(config.RATE_LIMIT_DELAY)
        
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
                except Exception as e:
                    logger.error(f"Error fetching exchange tickers: {e}")
                    continue
        
        if not all_results:
            logger.warning("Refresh fetched no results")
            return False
            
        new_df = pd.DataFrame(all_results)
        # Select and rename columns
        if not new_df.empty:
            new_df = new_df[["ticker", "name", "primary_exchange", "last_updated_utc", "type", "market_cap"]]
        
        new_tickers = set(new_df['ticker'].tolist()) if not new_df.empty else set()
        
        with self.cache_lock:
            old_tickers = set(self.current_tickers_set)
            added = new_tickers - old_tickers
            removed = old_tickers - new_tickers
            
            self.ticker_cache = new_df
            try:
                self.ticker_cache.to_parquet(config.TICKER_CACHE_FILE)
            except Exception as e:
                logger.error(f"Failed to save ticker cache: {e}")
            
            self.current_tickers_set = new_tickers
            metrics.tickers_total.set(len(new_df))
            
            rediscovered = added & self.known_missing_tickers
            if rediscovered:
                self.known_missing_tickers -= rediscovered
                self._save_missing_tickers()
        
        self.last_refresh_time = time.time()
        metrics.last_data_refresh_time.set(self.last_refresh_time)
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
        
    async def shutdown(self):
        """Cleanup resources"""
        self.stop()
        logger.info("Ticker scanner shutdown complete")

    def get_current_tickers_list(self):
        with self.cache_lock:
            return self.ticker_cache['ticker'].tolist() if not self.ticker_cache.empty else []

# ======================== STOCK SCANNER ======================== #
class StockScanner:
    def __init__(self, ticker_fetcher):
        self.ticker_fetcher = ticker_fetcher
        self.active = False
        self.scan_lock = Lock()
        self.last_scan_result = pd.DataFrame()
        self.signals = pd.DataFrame()
        
    async def fetch_stock_data(self, session, ticker):
        """Fetch data for a single ticker"""
        # This is a placeholder - implement your actual data fetching logic
        # For example, fetch latest quotes, aggregates, etc.
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev?adjusted=true&apiKey={config.POLYGON_API_KEY}"
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('resultsCount', 0) > 0:
                        result = data['results'][0]
                        return {
                            'ticker': ticker,
                            'open': result.get('o'),
                            'high': result.get('h'),
                            'low': result.get('l'),
                            'close': result.get('c'),
                            'volume': result.get('v'),
                            'timestamp': result.get('t')
                        }
            return None
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None

    async def scan_tickers(self, tickers):
        """Scan a list of tickers for signals"""
        start_time = time.time()
        logger.info(f"Starting scan of {len(tickers)} tickers")
        
        results = []
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
            
            async def fetch_with_semaphore(ticker):
                async with semaphore:
                    await asyncio.sleep(config.RATE_LIMIT_DELAY)
                    return await self.fetch_stock_data(session, ticker)
            
            tasks = [fetch_with_semaphore(ticker) for ticker in tickers]
            for future in asyncio.as_completed(tasks):
                try:
                    result = await future
                    if result:
                        results.append(result)
                        metrics.data_points_processed.inc()
                except Exception as e:
                    logger.error(f"Error in scan task: {e}")
                    metrics.scanner_errors.inc()
        
        # Convert to DataFrame
        if results:
            scan_df = pd.DataFrame(results)
            
            # Apply your signal detection logic here
            # This is a placeholder - implement your actual signal logic
            scan_df['signal'] = self.generate_signals(scan_df)
            
            # Count signals
            signal_counts = scan_df['signal'].value_counts()
            for signal_type, count in signal_counts.items():
                metrics.signals_generated.labels(signal_type=signal_type).inc(count)
            
            elapsed = time.time() - start_time
            metrics.scan_duration.observe(elapsed)
            metrics.last_scan_time.set(time.time())
            metrics.last_successful_scan_time.set(time.time())
            metrics.scans_total.inc()
            
            logger.info(f"Scan completed in {elapsed:.2f}s. Found {len(scan_df[scan_df['signal'] != 'none'])} signals.")
            
            return scan_df
        else:
            logger.warning("Scan completed with no results")
            return pd.DataFrame()

    def generate_signals(self, data_df):
        """Generate trading signals based on data"""
        # This is a placeholder - implement your actual signal logic
        # Example: Simple momentum signal
        signals = []
        for _, row in data_df.iterrows():
            if row['volume'] > 1000000 and row['close'] > row['open'] * 1.02:
                signals.append('bullish')
            elif row['volume'] > 1000000 and row['close'] < row['open'] * 0.98:
                signals.append('bearish')
            else:
                signals.append('none')
        return signals

    async def run_scan(self):
        """Run a complete scan"""
        if not self.ticker_fetcher.initial_refresh_complete.is_set():
            logger.warning("Ticker refresh not complete, skipping scan")
            return
        
        tickers = self.ticker_fetcher.get_current_tickers_list()
        if not tickers:
            logger.warning("No tickers available for scanning")
            return
        
        # If we have a specific scan universe, use it
        if config.SCAN_UNIVERSE:
            tickers = [t for t in tickers if t in config.SCAN_UNIVERSE]
        
        with self.scan_lock:
            try:
                scan_result = await self.scan_tickers(tickers)
                if not scan_result.empty:
                    self.last_scan_result = scan_result
                    
                    # Save signals
                    signals = scan_result[scan_result['signal'] != 'none']
                    if not signals.empty:
                        if os.path.exists(config.SIGNALS_FILE):
                            try:
                                existing_signals = pd.read_parquet(config.SIGNALS_FILE)
                                all_signals = pd.concat([existing_signals, signals], ignore_index=True)
                                all_signals.to_parquet(config.SIGNALS_FILE)
                            except Exception as e:
                                logger.error(f"Error saving signals: {e}")
                        else:
                            signals.to_parquet(config.SIGNALS_FILE)
                        
                        self.signals = signals
            except Exception as e:
                logger.error(f"Error during scan: {e}")
                metrics.scanner_errors.inc()

    def start(self):
        self.active = True
        logger.info("Stock scanner started")

    def stop(self):
        self.active = False
        logger.info("Stock scanner stopped")

# ======================== SCHEDULER ======================== #
async def run_scheduled_operations(ticker_scanner, stock_scanner):
    """Run scheduled operations based on market status"""
    # Run immediate scan on startup
    logger.info("Starting immediate ticker refresh on startup")
    try:
        success = await ticker_scanner.refresh_all_tickers()
        if success:
            logger.info("Initial ticker refresh completed successfully")
        else:
            logger.warning("Initial ticker refresh encountered errors")
    except Exception as e:
        logger.error(f"Error during initial ticker refresh: {e}")
    
    # Start stock scanner after initial refresh
    stock_scanner.start()
    
    # Continue with scheduled operations
    while ticker_scanner.active and stock_scanner.active:
        now = datetime.now(ticker_scanner.local_tz)
        market_status = get_market_status(now)
        
        # Daily ticker refresh before market open
        target_refresh_time = datetime.strptime(config.PRE_MARKET_SCAN_TIME, "%H:%M").time()
        target_refresh_datetime = now.replace(
            hour=target_refresh_time.hour,
            minute=target_refresh_time.minute,
            second=0,
            microsecond=0
        )
        
        # If we already passed today's refresh time, set for tomorrow
        if now > target_refresh_datetime:
            target_refresh_datetime += timedelta(days=1)
        
        # Determine next scan time based on market status
        if market_status in [MarketStatus.PRE_MARKET, MarketStatus.OPEN, MarketStatus.POST_MARKET]:
            # During market hours, scan at regular intervals
            next_scan_time = now + timedelta(minutes=config.SCAN_INTERVAL_MINUTES)
        else:
            # Outside market hours, scan at next market open
            next_scan_time = target_refresh_datetime
        
        # Calculate sleep time until next operation
        sleep_seconds = min(
            (target_refresh_datetime - now).total_seconds(),
            (next_scan_time - now).total_seconds()
        )
        
        if sleep_seconds > 0:
            logger.info(f"Next operation in {sleep_seconds:.0f} seconds (Market: {market_status.value})")
            
            # Wait until next operation, but check every second if we should stop
            while sleep_seconds > 0 and ticker_scanner.active and stock_scanner.active:
                await asyncio.sleep(min(1, sleep_seconds))
                sleep_seconds -= 1
                
        if not (ticker_scanner.active and stock_scanner.active):
            break
            
        # Check what operation to perform
        now = datetime.now(ticker_scanner.local_tz)
        if now.time() >= target_refresh_time and now.date() != datetime.fromtimestamp(ticker_scanner.last_refresh_time).date():
            # Run daily ticker refresh
            logger.info("Starting daily ticker refresh")
            try:
                success = await ticker_scanner.refresh_all_tickers()
                if success:
                    logger.info("Daily ticker refresh completed successfully")
                else:
                    logger.warning("Daily ticker refresh encountered errors")
            except Exception as e:
                logger.error(f"Error during daily ticker refresh: {e}")
        else:
            # Run stock scan
            logger.info(f"Starting stock scan (Market: {market_status.value})")
            try:
                await stock_scanner.run_scan()
            except Exception as e:
                logger.error(f"Error during stock scan: {e}")
        
        # Update system metrics
        metrics.update_system_metrics()

# ======================== HEALTH CHECK SERVER ======================== #
def start_health_server():
    """Start Prometheus metrics server"""
    try:
        start_http_server(config.METRICS_PORT)
        logger.info(f"Metrics server started on port {config.METRICS_PORT}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")

# ======================== SHUTDOWN HANDLER ======================== #
async def shutdown(signal, ticker_scanner, stock_scanner, scheduler_task):
    """Cleanup tasks tied to the service's shutdown."""
    logger.info(f"Received exit signal {signal.name}...")
    
    # Cancel the scheduler task
    scheduler_task.cancel()
    
    # Stop the scanners
    stock_scanner.stop()
    await ticker_scanner.shutdown()
    
    # Wait for the scheduler task to be cancelled
    try:
        await scheduler_task
    except asyncio.CancelledError:
        logger.info("Scheduler task cancelled successfully")
    
    # Gather all running tasks (except the current one)
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    
    logger.info("Cancelling all outstanding tasks")
    await asyncio.gather(*tasks, return_exceptions=True)
    
    logger.info("Shutdown complete")

# ======================== MAIN EXECUTION ======================== #
async def main():
    # Start health metrics server
    health_thread = threading.Thread(target=start_health_server, daemon=True)
    health_thread.start()
    
    # Initialize scanners
    ticker_scanner = PolygonTickerScanner()
    stock_scanner = StockScanner(ticker_scanner)
    
    # Start ticker scanner
    ticker_scanner.start()
    
    # Wait for initial cache load
    ticker_scanner.initial_refresh_complete.wait()
    
    # Create a task for the scheduler
    scheduler_task = asyncio.create_task(run_scheduled_operations(ticker_scanner, stock_scanner))
    
    try:
        # Set up signal handlers for proper shutdown (only on Unix systems)
        if sys.platform != "win32":
            loop = asyncio.get_running_loop()
            for sig in [signal.SIGINT, signal.SIGTERM]:
                loop.add_signal_handler(
                    sig, 
                    lambda: asyncio.create_task(shutdown(sig, ticker_scanner, stock_scanner, scheduler_task))
                )
        
        # Wait for the scheduler task to complete (which it won't unless stopped)
        await scheduler_task
    except asyncio.CancelledError:
        logger.info("Main task cancelled")
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down")
        await shutdown(signal.SIGINT, ticker_scanner, stock_scanner, scheduler_task)
    finally:
        stock_scanner.stop()
        await ticker_scanner.shutdown()

if __name__ == "__main__":
    # Windows event loop policy
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Scanner stopped by user")