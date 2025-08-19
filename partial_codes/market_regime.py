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
import sqlite3
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

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
    
    # Database Configuration
    REGIME_DB_FILE = "regime_data.db"
    
    # Backtesting Configuration
    BACKTEST_START_DATE = "2010-01-01"

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

# ======================== DATABASE INTEGRATION ======================== #
class RegimeDatabase:
    def __init__(self, db_path=config.REGIME_DB_FILE):
        self.db_path = db_path
        self._initialize_db()
        
    def _initialize_db(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            # Create market regimes table
            c.execute('''CREATE TABLE IF NOT EXISTS market_regimes (
                         date TEXT PRIMARY KEY,
                         regime TEXT,
                         return_21d REAL,
                         return_63d REAL,
                         return_126d REAL,
                         return_252d REAL,
                         volatility REAL,
                         adx REAL)''')
            
            # Create sector regimes table
            c.execute('''CREATE TABLE IF NOT EXISTS sector_regimes (
                         date TEXT,
                         sector TEXT,
                         regime TEXT,
                         return_21d REAL,
                         return_63d REAL,
                         return_126d REAL,
                         return_252d REAL,
                         volatility REAL,
                         adx REAL,
                         PRIMARY KEY (date, sector))''')
            conn.commit()
    
    def save_market_regime(self, date, regime_data):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            features = regime_data.get('features', {})
            c.execute('''INSERT OR REPLACE INTO market_regimes 
                         (date, regime, return_21d, return_63d, return_126d, return_252d, volatility, adx)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                     (date, regime_data['regime'],
                      features.get('return_21d'), features.get('return_63d'),
                      features.get('return_126d'), features.get('return_252d'),
                      features.get('volatility'), features.get('adx')))
            conn.commit()
    
    def save_sector_regime(self, date, sector, sector_data):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            features = sector_data.get('features', {})
            c.execute('''INSERT OR REPLACE INTO sector_regimes 
                         (date, sector, regime, return_21d, return_63d, return_126d, return_252d, volatility, adx)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (date, sector, sector_data['regime'],
                      features.get('return_21d'), features.get('return_63d'),
                      features.get('return_126d'), features.get('return_252d'),
                      features.get('volatility'), features.get('adx')))
            conn.commit()
    
    def get_historical_regimes(self, start_date=None, end_date=None):
        """Retrieve historical regime data for analysis"""
        with sqlite3.connect(self.db_path) as conn:
            # Market regimes
            market_query = "SELECT * FROM market_regimes"
            params = []
            
            if start_date:
                market_query += " WHERE date >= ?"
                params.append(start_date)
            if end_date:
                market_query += " AND date <= ?" if start_date else " WHERE date <= ?"
                params.append(end_date)
                
            market_df = pd.read_sql(market_query, conn, parse_dates=['date'], params=params)
            
            # Sector regimes
            sector_query = "SELECT * FROM sector_regimes"
            if start_date or end_date:
                sector_query += " WHERE"
                conditions = []
                if start_date:
                    conditions.append("date >= ?")
                if end_date:
                    conditions.append("date <= ?")
                sector_query += " AND ".join(conditions)
                
            sector_df = pd.read_sql(sector_query, conn, parse_dates=['date'], params=params)
        
        return {
            "market": market_df.set_index('date'),
            "sectors": sector_df.pivot(index='date', columns='sector', values='regime')
        }

# ======================== BACKTESTING INTERFACE ======================== #
class RegimeBacktester:
    def __init__(self, db):
        self.db = db
        self.sector_etf_map = {
            "Materials": "XLB",
            "Communications": "XLC",
            "Energy": "XLE",
            "Financials": "XLF",
            "Industrials": "XLI",
            "Technology": "XLK",
            "Consumer Staples": "XLP",
            "Real Estate": "XLRE",
            "Utilities": "XLU",
            "Healthcare": "XLV",
            "Consumer Discretionary": "XLY"
        }
    
    def backtest_regime_strategy(self, strategy_func, start_date=config.BACKTEST_START_DATE, end_date=None):
        """
        Backtest a trading strategy based on regime signals
        
        Args:
            strategy_func: Function with signature (date, regime_data, portfolio) -> trades
            start_date: Start of backtest period
            end_date: End of backtest period (default: today)
        """
        # Load historical regime data
        regime_data = self.db.get_historical_regimes(start_date, end_date)
        market_data = regime_data['market']
        sector_data = regime_data['sectors']
        
        # Initialize portfolio
        portfolio = {
            'cash': 1000000,
            'positions': {},
            'value_history': [],
            'regime_history': []
        }
        
        # Get trading calendar (simplified)
        dates = market_data.index.sort_values()
        
        # Run backtest day by day
        for date in dates:
            # Get current regimes
            try:
                current_regimes = {
                    "market": market_data.loc[date, 'regime'],
                    "sectors": sector_data.loc[date].to_dict()
                }
            except KeyError:
                continue  # Skip dates with missing data
            
            # Execute strategy
            trades = strategy_func(date, current_regimes, portfolio)
            
            # Execute trades (simplified)
            self._execute_trades(trades, date, portfolio)
            
            # Update portfolio value
            portfolio_value = portfolio['cash'] + sum(
                self._get_asset_price(ticker, date) * qty 
                for ticker, qty in portfolio['positions'].items()
            )
            
            # Record daily stats
            portfolio['value_history'].append((date, portfolio_value))
            portfolio['regime_history'].append(current_regimes['market'])
        
        # Generate performance report
        return self._generate_report(portfolio, start_date, end_date)
    
    def _execute_trades(self, trades, date, portfolio):
        """Simplified trade execution"""
        for trade in trades:
            ticker = trade['ticker']
            shares = trade['shares']
            action = trade['action']
            
            if action == 'BUY':
                price = self._get_asset_price(ticker, date)
                cost = shares * price
                if cost <= portfolio['cash']:
                    portfolio['cash'] -= cost
                    portfolio['positions'][ticker] = portfolio['positions'].get(ticker, 0) + shares
            elif action == 'SELL':
                current_shares = portfolio['positions'].get(ticker, 0)
                if shares <= current_shares:
                    price = self._get_asset_price(ticker, date)
                    portfolio['cash'] += shares * price
                    portfolio['positions'][ticker] = current_shares - shares
                    if portfolio['positions'][ticker] == 0:
                        del portfolio['positions'][ticker]
    
    def _get_asset_price(self, ticker, date):
        """Mock price lookup - in real implementation use historical data"""
        # Placeholder implementation - in production, connect to price database
        return 100  # Fixed placeholder price
    
    def _generate_report(self, portfolio, start_date, end_date):
        """Generate performance analysis report"""
        # Create equity curve
        dates, values = zip(*portfolio['value_history'])
        returns = pd.Series(values, index=dates).pct_change().dropna()
        
        # Calculate performance metrics
        total_return = values[-1] / values[0] - 1
        annualized_return = (1 + total_return) ** (252 / len(dates)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = annualized_return / volatility if volatility > 0 else 0
        max_drawdown = (pd.Series(values) / pd.Series(values).cummax() - 1).min()
        
        # Create plot
        plt.figure(figsize=(14, 10))
        
        # Equity curve
        plt.subplot(2, 1, 1)
        plt.plot(dates, values)
        plt.title('Portfolio Equity Curve')
        plt.ylabel('Value')
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        plt.grid(True)
        
        # Regime visualization
        plt.subplot(2, 1, 2)
        regime_map = {'Bull': 0, 'Bear': 1, 'High Volatility': 2, 'Sideways': 3, 'Transitional': 4}
        regime_colors = ['green', 'red', 'orange', 'blue', 'gray']
        regime_values = [regime_map.get(r, 4) for r in portfolio['regime_history']]
        
        plt.scatter(dates, regime_values, c=regime_values, 
                   cmap=plt.cm.colors.ListedColormap(regime_colors), s=15)
        plt.yticks([0, 1, 2, 3, 4], ['Bull', 'Bear', 'High Vol', 'Sideways', 'Transitional'])
        plt.title('Market Regimes During Backtest')
        plt.xlabel('Date')
        plt.grid(True)
        
        # Add performance metrics to plot
        plt.figtext(0.15, 0.01, 
                   f"Ann. Return: {annualized_return:.2%} | Volatility: {volatility:.2%} | "
                   f"Sharpe: {sharpe:.2f} | Max DD: {max_drawdown:.2%}",
                   fontsize=10)
        
        # Save plot
        os.makedirs("backtests", exist_ok=True)
        plot_file = f"backtests/backtest_{start_date}_{end_date or 'present'}.png"
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(plot_file)
        plt.close()
        
        return {
            "start_value": values[0],
            "end_value": values[-1],
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "plot_file": plot_file
        }

# ======================== MARKET REGIME CLASSIFIER ======================== #
class MarketRegimeClassifier:
    VOLATILITY_LOOKBACK = 63  # 3 months (trading days)
    RETURN_LOOKBACKS = [21, 63, 126, 252]  # 1m, 3m, 6m, 1y
    VOL_THRESHOLD_HIGH = 0.20  # 20% annualized volatility
    VOL_THRESHOLD_LOW = 0.15   # 15% annualized volatility
    TREND_THRESHOLD = 0.05     # 5% return threshold
    ADX_THRESHOLD = 25         # ADX threshold for strong trends
    
    def __init__(self, api_key, db):
        self.api_key = api_key
        self.db = db
        self.market_etf = "SPY"
        self.sector_etfs = {
            "Materials": "XLB",
            "Communications": "XLC",
            "Energy": "XLE",
            "Financials": "XLF",
            "Industrials": "XLI",
            "Technology": "XLK",
            "Consumer Staples": "XLP",
            "Real Estate": "XLRE",
            "Utilities": "XLU",
            "Healthcare": "XLV",
            "Consumer Discretionary": "XLY"
        }
        self.regime_data = {}
        self.last_updated = None
        self.data_lock = Lock()
        self.backtester = RegimeBacktester(db)
        logger.info("Market regime classifier initialized")

    async def fetch_historical_data(self, session, ticker):
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=365 * 2)  # 2 years history
        url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}"
               f"/range/1/day/{start_date.isoformat()}/{end_date.isoformat()}?"
               f"adjusted=true&sort=asc&limit=50000&apiKey={self.api_key}")
        
        for attempt in range(3):
            try:
                async with session.get(url, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('status') == "OK" and data.get('count', 0) > 0:
                            return pd.DataFrame(data['results'])
                    elif response.status == 429:
                        retry_after = int(response.headers.get('Retry-After', 5))
                        logger.warning(f"Rate limited for {ticker}. Retrying after {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                    else:
                        logger.warning(f"API error for {ticker}: {response.status}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Attempt {attempt+1} for {ticker}: {e}")
                await asyncio.sleep(2 ** attempt)
        
        logger.error(f"Failed to fetch data for {ticker} after 3 attempts")
        return None

    def calculate_features(self, df, ticker):
        if df is None or len(df) < max(self.RETURN_LOOKBACKS):
            logger.warning(f"Insufficient data for {ticker}")
            return None
            
        try:
            df['date'] = pd.to_datetime(df['t'], unit='ms')
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            df['returns'] = df['c'].pct_change()
            
            # Calculate returns for different lookback periods
            features = {}
            for lookback in self.RETURN_LOOKBACKS:
                if len(df) > lookback:
                    features[f'return_{lookback}d'] = df['c'].pct_change(lookback).iloc[-1]
            
            # Calculate volatility (annualized)
            if len(df) > self.VOLATILITY_LOOKBACK:
                features['volatility'] = df['returns'].rolling(self.VOLATILITY_LOOKBACK).std().iloc[-1] * np.sqrt(252)
            
            # Calculate trend strength (ADX approximation)
            if len(df) > 14:
                high = df['h']
                low = df['l']
                close = df['c']
                
                # Calculate True Range
                tr = pd.concat([
                    high - low,
                    abs(high - close.shift()),
                    abs(low - close.shift())
                ], axis=1).max(axis=1)
                
                # Calculate ATR
                atr = tr.rolling(14).mean()
                
                # Calculate Directional Movements
                up_move = high.diff()
                down_move = low.diff().abs()
                plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
                minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
                
                # Calculate Directional Indicators
                plus_di = 100 * (plus_dm.rolling(14).sum() / atr)
                minus_di = 100 * (minus_dm.rolling(14).sum() / atr)
                
                # Calculate DX and ADX
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
                features['adx'] = dx.rolling(14).mean().iloc[-1]
            
            return features
        except Exception as e:
            logger.error(f"Feature calculation error for {ticker}: {e}")
            return None

    def classify_regime(self, features):
        if not features:
            return "Unknown"
        
        # Check for strong trends using ADX
        if features.get('adx', 0) > self.ADX_THRESHOLD:  # Strong trend
            if features.get('return_21d', 0) > self.TREND_THRESHOLD:
                return "Bull"
            elif features.get('return_21d', 0) < -self.TREND_THRESHOLD:
                return "Bear"
        
        # Check volatility conditions
        volatility = features.get('volatility', 0)
        if volatility > self.VOL_THRESHOLD_HIGH:
            return "High Volatility"
        elif volatility < self.VOL_THRESHOLD_LOW:
            # Check if we're in a sideways market
            return_63d = abs(features.get('return_63d', 0))
            return_126d = abs(features.get('return_126d', 0))
            if return_63d < self.TREND_THRESHOLD and return_126d < self.TREND_THRESHOLD:
                return "Sideways"
        
        return "Transitional"

    async def update_regimes(self):
        start_time = time.time()
        logger.info("Updating market and sector regimes")
        
        async with aiohttp.ClientSession() as session:
            # Process market ETF
            market_data = await self.fetch_historical_data(session, self.market_etf)
            market_features = self.calculate_features(market_data, self.market_etf)
            market_regime = self.classify_regime(market_features) if market_features else "Unknown"
            
            # Process sector ETFs
            sector_tasks = []
            for sector, etf in self.sector_etfs.items():
                sector_tasks.append(self.process_sector(session, sector, etf))
            
            sector_results = await asyncio.gather(*sector_tasks)
            sector_regimes = dict(sector_results)
        
        with self.data_lock:
            self.regime_data = {
                "market": {
                    "ticker": self.market_etf,
                    "regime": market_regime,
                    "features": market_features
                },
                "sectors": sector_regimes,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.last_updated = time.time()
            
            # Save to database
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
            self.db.save_market_regime(date_str, self.regime_data['market'])
            for sector, data in self.regime_data['sectors'].items():
                self.db.save_sector_regime(date_str, sector, data)
            logger.info("Regime data saved to database")
        
        elapsed = time.time() - start_time
        logger.info(f"Regime update completed in {elapsed:.2f}s")
        return self.regime_data

    async def process_sector(self, session, sector, etf):
        data = await self.fetch_historical_data(session, etf)
        features = self.calculate_features(data, etf)
        regime = self.classify_regime(features) if features else "Unknown"
        return sector, {
            "ticker": etf,
            "regime": regime,
            "features": features
        }

    def get_current_regimes(self):
        with self.data_lock:
            return self.regime_data.copy()
    
    def run_backtest(self, strategy_func, start_date=config.BACKTEST_START_DATE, end_date=None):
        """Run backtest using the specified strategy function"""
        logger.info(f"Starting backtest from {start_date} to {end_date or 'present'}")
        return self.backtester.backtest_regime_strategy(strategy_func, start_date, end_date)

# ======================== REGIME MONITOR ======================== #
class RegimeMonitor:
    def __init__(self, classifier):
        self.classifier = classifier
        self.active = False
        self.market_close_time = "16:30"  # 4:30 PM local time
        self.local_tz = get_localzone()
        logger.info(f"Using local timezone: {self.local_tz}")

    async def run_scheduled_updates(self):
        # Initial update
        await self.classifier.update_regimes()
        
        while self.active:
            now = datetime.now(self.local_tz)
            
            # Calculate next run time (today at market close)
            target_time = datetime.strptime(self.market_close_time, "%H:%M").time()
            target_datetime = now.replace(
                hour=target_time.hour,
                minute=target_time.minute,
                second=0,
                microsecond=0
            )
            
            # If we already passed today's market close, set for tomorrow
            if now > target_datetime:
                target_datetime += timedelta(days=1)
            
            # Skip weekends
            while target_datetime.weekday() >= 5:  # Saturday=5, Sunday=6
                target_datetime += timedelta(days=1)
            
            sleep_seconds = (target_datetime - now).total_seconds()
            logger.info(f"Next regime update at {target_datetime} ({sleep_seconds:.0f} seconds from now)")
            
            # Wait until scheduled time
            await asyncio.sleep(sleep_seconds)
            
            # Run the refresh
            logger.info("Starting scheduled regime update")
            try:
                await self.classifier.update_regimes()
                logger.info("Scheduled regime update completed")
            except Exception as e:
                logger.error(f"Error during regime update: {e}")
                # Retry in 1 hour
                await asyncio.sleep(3600)

    def start(self):
        if not self.active:
            self.active = True
            asyncio.create_task(self.run_scheduled_updates())
            logger.info("Regime monitor started")

    def stop(self):
        self.active = False
        logger.info("Regime monitor stopped")

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

# ======================== EXAMPLE STRATEGY ======================== #
def example_regime_strategy(date, regime_data, portfolio):
    """Example strategy based on market regimes"""
    trades = []
    market_regime = regime_data['market']
    cash_percentage = portfolio['cash'] / (portfolio['cash'] + sum(
        portfolio['positions'].values()
    )) if portfolio['positions'] else 1.0
    
    # Bull market strategy - buy growth sectors
    if market_regime == "Bull":
        for sector, sector_regime in regime_data['sectors'].items():
            if sector in ["Technology", "Consumer Discretionary"] and sector_regime == "Bull":
                trades.append({
                    'ticker': "XLK" if sector == "Technology" else "XLY",
                    'shares': 100,
                    'action': 'BUY'
                })
    
    # Bear market strategy - move to defensive sectors
    elif market_regime == "Bear":
        # First reduce exposure to 50% cash
        if cash_percentage < 0.5:
            for ticker, shares in portfolio['positions'].items():
                if shares > 0:
                    sell_qty = max(1, int(shares * (0.5 - cash_percentage)))
                    trades.append({
                        'ticker': ticker,
                        'shares': sell_qty,
                        'action': 'SELL'
                    })
    
    # High volatility strategy - reduce exposure
    elif market_regime == "High Volatility":
        # Maintain 70% cash position
        if cash_percentage < 0.7:
            for ticker, shares in portfolio['positions'].items():
                if shares > 0:
                    sell_qty = max(1, int(shares * (0.7 - cash_percentage)))
                    trades.append({
                        'ticker': ticker,
                        'shares': sell_qty,
                        'action': 'SELL'
                    })
    
    return trades

# ======================== MAIN EXECUTION ======================== #
async def main():
    # Initialize database
    db = RegimeDatabase()
    
    # Start ticker scanner
    scanner = PolygonTickerScanner()
    scanner.start()
    scanner.initial_refresh_complete.wait()
    
    # Initialize regime classifier and monitor
    regime_classifier = MarketRegimeClassifier(config.POLYGON_API_KEY, db)
    regime_monitor = RegimeMonitor(regime_classifier)
    regime_monitor.start()
    
    # Run scanners in parallel
    await asyncio.gather(
        run_scheduled_refresh(scanner),
    )

if __name__ == "__main__":
    # Windows event loop policy
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application stopped by user")