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
            
            # Create backtest_tickers table for historical data with year column
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtest_tickers (
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
                )
            ''')
            
            # Create backtest_final_results table for storing final backtest results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtest_final_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tickers_exchange ON tickers(primary_exchange)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tickers_active ON tickers(active)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_historical_tickers_date ON historical_tickers(change_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_backtest_tickers_date ON backtest_tickers(date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_backtest_tickers_year ON backtest_tickers(year)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_backtest_final_dates ON backtest_final_results(start_date, end_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_backtest_final_years ON backtest_final_results(start_year, end_year)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_backtest_final_ticker ON backtest_final_results(ticker)')
            
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
        
    def upsert_backtest_final_results(self, tickers_data: List[Dict], start_date: str, end_date: str) -> int:
        """Insert final backtest results into the database with year information"""
        inserted = 0
        start_year = int(start_date.split('-')[0])
        end_year = int(end_date.split('-')[0])
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # First, delete any existing results for this date range
            cursor.execute(
                "DELETE FROM backtest_final_results WHERE start_date = ? AND end_date = ?",
                (start_date, end_date)
            )
            
            for ticker_data in tickers_data:
                cursor.execute('''
                    INSERT INTO backtest_final_results 
                    (start_date, end_date, start_year, end_year, ticker, name, primary_exchange, last_updated_utc, 
                     type, market, locale, currency_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    start_date,
                    end_date,
                    start_year,
                    end_year,
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
        
    def get_backtest_final_results(self, start_date: str, end_date: str) -> List[Dict]:
        """Get final backtest results for a specific date range"""
        return self.execute_query(
            "SELECT * FROM backtest_final_results WHERE start_date = ? AND end_date = ? ORDER BY ticker",
            (start_date, end_date)
        )
        
    def get_backtest_final_results_by_year(self, year: int) -> List[Dict]:
        """Get final backtest results for a specific year"""
        return self.execute_query(
            "SELECT * FROM backtest_final_results WHERE start_year <= ? AND end_year >= ? ORDER BY ticker",
            (year, year)
        )
        
    def get_all_backtest_runs(self) -> List[Dict]:
        """Get all backtest runs with their date ranges"""
        return self.execute_query(
            "SELECT DISTINCT start_date, end_date, start_year, end_year FROM backtest_final_results ORDER BY start_date, end_date"
        )

# ======================== BACKTESTER ======================== #
class Backtester:
    def __init__(self, scanner):
        self.scanner = scanner
        
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
        
        # Dictionary to track ticker availability across all dates
        ticker_availability = defaultdict(list)
        
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Only weekdays
                date_str = current_date.strftime("%Y-%m-%d")
                # Check if we already have data for this date
                existing_data = self.scanner.db.get_backtest_tickers(date_str)
                if not existing_data:
                    await self.run_single_day_backtest(current_date)
                    existing_data = self.scanner.db.get_backtest_tickers(date_str)
                
                # Track which tickers were available on this date
                if existing_data:
                    for ticker_data in existing_data:
                        ticker_availability[ticker_data['ticker']].append(date_str)
            
            current_date += timedelta(days=1)
        
        # Filter tickers to only those available for the entire period
        full_period_tickers = []
        for ticker, available_dates in ticker_availability.items():
            # Check if ticker was available for all dates in the range
            if len(available_dates) == len([d for d in self._date_range(start_date, end_date) if d.weekday() < 5]):
                # Get the most recent data for this ticker
                latest_date = max(available_dates)
                ticker_data = self.scanner.db.execute_query(
                    "SELECT * FROM backtest_tickers WHERE ticker = ? AND date = ?",
                    (ticker, latest_date)
                )
                if ticker_data:
                    full_period_tickers.append(ticker_data[0])
        
        # Save the filtered results to the database
        if full_period_tickers:
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            inserted = self.scanner.db.upsert_backtest_final_results(full_period_tickers, start_str, end_str)
            logger.info(f"Saved {inserted} tickers to database that were active throughout the entire period")
        else:
            logger.warning("No tickers were active throughout the entire period")
            
        logger.info("Backtest completed")
        
    def _date_range(self, start_date, end_date):
        """Generate a range of dates between start_date and end_date"""
        for n in range(int((end_date - start_date).days) + 1):
            yield start_date + timedelta(n)
        
    async def run_single_day_backtest(self, target_date):
        """
        Run backtest for a single day
        """
        logger.info(f"Running backtest for {target_date.strftime('%Y-%m-%d')}")
        
        # Format date for API
        date_str = target_date.strftime("%Y-%m-%d")
        
        # Check if we already have data for this date
        existing_data = self.scanner.db.get_backtest_tickers(date_str)
        if existing_data:
            logger.info(f"Using cached data for {date_str} ({len(existing_data)} tickers)")
            return True
        
        # Temporarily set scanner to backtest mode
        original_mode = self.scanner.backtest_mode
        self.scanner.backtest_mode = True
        self.scanner.backtest_date = date_str
        
        try:
            # Use the existing refresh method but with historical date
            success = await self.scanner.refresh_all_tickers()
            if success:
                logger.info(f"Successfully fetched data for {date_str}")
                return True
            else:
                logger.warning(f"Failed to fetch data for {date_str}")
                return False
        except Exception as e:
            logger.error(f"Error during backtest for {date_str}: {e}")
            return False
        finally:
            # Restore original mode
            self.scanner.backtest_mode = original_mode
            self.scanner.backtest_date = None

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
        
    def get_backtest_final_results(self, start_date: str, end_date: str) -> List[Dict]:
        """Get final backtest results for a specific date range"""
        return self.db.get_backtest_final_results(start_date, end_date)
        
    def get_backtest_final_results_by_year(self, year: int) -> List[Dict]:
        """Get final backtest results for a specific year"""
        return self.db.get_backtest_final_results_by_year(year)
        
    def get_all_backtest_runs(self) -> List[Dict]:
        """Get all backtest runs with their date ranges"""
        return self.db.get_all_backtest_runs()

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

# ======================== MAIN EXECUTION ======================== #
async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stock Ticker Fetcher with Backtesting')
    parser.add_argument('--backtest', type=str, help='Run backtest for a specific date (YYYY-MM-DD)')
    parser.add_argument('--backtest-range', type=str, help='Run backtest for a date range (YYYY-MM-DD:YYYY-MM-DD)')
    parser.add_argument('--backtest-year', type=int, help='Run backtest for a specific year')
    parser.add_argument('--list-backtests', action='store_true', help='List available backtest dates')
    parser.add_argument('--list-backtest-years', action='store_true', help='List available backtest years')
    parser.add_argument('--list-backtest-runs', action='store_true', help='List available backtest runs')
    parser.add_argument('--show-backtest-results', type=str, help='Show results for a specific backtest run (format: YYYY-MM-DD:YYYY-MM-DD)')
    parser.add_argument('--show-year-results', type=int, help='Show results for a specific year')
    args = parser.parse_args()
    
    scanner = PolygonTickerScanner()
    
    # Check if we're running in backtest mode
    if (args.backtest or args.backtest_range or args.backtest_year or 
        args.list_backtests or args.list_backtest_years or args.list_backtest_runs or 
        args.show_backtest_results or args.show_year_results):
        
        if args.list_backtests:
            # List available backtest dates
            dates = scanner.get_backtest_dates()
            if dates:
                print("Available backtest dates:")
                for date in dates:
                    print(f"  {date}")
            else:
                print("No backtest data available")
            return
        
        if args.list_backtest_years:
            # List available backtest years
            years = scanner.get_backtest_years()
            if years:
                print("Available backtest years:")
                for year in years:
                    print(f"  {year}")
            else:
                print("No backtest data available")
            return
            
        if args.list_backtest_runs:
            # List available backtest runs
            runs = scanner.get_all_backtest_runs()
            if runs:
                print("Available backtest runs:")
                for run in runs:
                    print(f"  {run['start_date']} to {run['end_date']} (Years: {run['start_year']}-{run['end_year']})")
            else:
                print("No backtest runs available")
            return
            
        if args.show_backtest_results:
            # Show results for a specific backtest run
            start_date, end_date = args.show_backtest_results.split(':')
            results = scanner.get_backtest_final_results(start_date, end_date)
            if results:
                print(f"Backtest results for {start_date} to {end_date}:")
                print(f"Found {len(results)} tickers that were active throughout the period")
                for result in results[:10]:  # Show first 10 results
                    print(f"  {result['ticker']}: {result['name']}")
                if len(results) > 10:
                    print(f"  ... and {len(results) - 10} more")
            else:
                print(f"No results found for {start_date} to {end_date}")
            return
            
        if args.show_year_results:
            # Show results for a specific year
            year = args.show_year_results
            results = scanner.get_backtest_final_results_by_year(year)
            if results:
                print(f"Backtest results for year {year}:")
                print(f"Found {len(results)} tickers that were active in this year")
                for result in results[:10]:  # Show first 10 results
                    print(f"  {result['ticker']}: {result['name']} (From {result['start_date']} to {result['end_date']})")
                if len(results) > 10:
                    print(f"  ... and {len(results) - 10} more")
            else:
                print(f"No results found for year {year}")
            return
        
        backtester = Backtester(scanner)
        
        if args.backtest:
            # Single date backtest
            await backtester.run_single_day_backtest(datetime.strptime(args.backtest, "%Y-%m-%d"))
        elif args.backtest_range:
            # Date range backtest
            start_str, end_str = args.backtest_range.split(':')
            start_date = datetime.strptime(start_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_str, "%Y-%m-%d")
            await backtester.run_backtest(start_date, end_date)
        elif args.backtest_year:
            # Year backtest
            start_date = datetime(args.backtest_year, 1, 1)
            end_date = datetime(args.backtest_year, 12, 31)
            await backtester.run_backtest(start_date, end_date)
    else:
        # Normal operation
        scanner.start()
        
        # Wait for initial cache load
        await asyncio.get_event_loop().run_in_executor(None, scanner.initial_refresh_complete.wait)
        
        # Create a task for the scheduler
        scheduler_task = asyncio.create_task(run_scheduled_refresh(scanner))
        
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
                [scheduler_task, stop_task], 
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel the scheduler task if it's still running
            if not scheduler_task.done():
                scheduler_task.cancel()
                try:
                    await scheduler_task
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