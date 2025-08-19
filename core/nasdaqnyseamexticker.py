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

# ======================== CONFIGURATION ======================== #
class Config:
    # API Configuration
    POLYGON_API_KEY = "ld1Poa63U6t4Y2MwOCA2JeKQyHVrmyg8"
    
    # Scanner Configuration
    EXCHANGES = ["XNAS", "XNYS", "XASE"]
    MAX_CONCURRENT_REQUESTS = 50
    RATE_LIMIT_DELAY = 0.05  # 50ms between requests
    SCAN_TIME = "08:30"  # Daily scan time in local time
    
    # File Management
    TICKER_CACHE_FILE = "ticker_cache.parquet"
    MISSING_TICKERS_FILE = "missing_tickers.json"
    METADATA_FILE = "scanner_metadata.json"
    
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
    logger = logging.getLogger("TickerFetcher")
    logger.setLevel(config.LOG_LEVEL)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(config.LOG_FORMAT)
    
    # File handler - all logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/ticker_fetcher_{timestamp}.log"
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
        
        if not all_results:
            logger.warning("Refresh fetched no results")
            return False
            
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
    except Exception as e:
        logger.error(f"Error during initial scan: {e}")
    
    # Continue with daily scans
    while scanner.active:
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
        logger.info(f"Next refresh scheduled at {target_datetime} ({sleep_seconds:.0f} seconds from now)")
        
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

# ======================== MAIN EXECUTION ======================== #
async def main():
    scanner = PolygonTickerScanner()
    scanner.start()
    
    # Wait for initial cache load
    scanner.initial_refresh_complete.wait()
    
    # Run the scheduler with immediate first scan
    await run_scheduled_refresh(scanner)

if __name__ == "__main__":
    # Windows event loop policy
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Scanner stopped by user")