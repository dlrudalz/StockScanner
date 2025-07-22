from __future__ import annotations
import asyncio
import aiohttp
import async_timeout
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import logging
import talib
from typing import Optional, List, Dict, Tuple, Literal, DefaultDict, Set
import random
import os
import ftplib
import io
import re
import time
from collections import defaultdict
from tqdm import tqdm
import sys
from scipy.stats import linregress
import json
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, TrailingStrategy
from sklearn.model_selection import ParameterGrid

# Define market regime types
TrendRegime = Literal["strong_bull", "weak_bull", "neutral", "weak_bear", "strong_bear"]
VolatilityRegime = Literal["low_vol", "medium_vol", "high_vol"]


class Debugger:
    """Enhanced centralized debugging functionality with better control"""
    
    def __init__(self, enabled: bool = False, level: str = "INFO", log_file: Optional[str] = None):
        """
        Initialize debugger with more robust controls
        
        Args:
            enabled: Whether logging is enabled at all
            level: Minimum log level to display ("DEBUG", "INFO", "WARNING", "ERROR")
            log_file: Optional file path to write logs to
        """
        self.enabled = enabled
        self.level = self._validate_level(level.upper())
        self.log_file = log_file
        self._setup_logging()
        
    def _validate_level(self, level: str) -> str:
        """Ensure the provided level is valid"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if level not in valid_levels:
            raise ValueError(f"Invalid log level '{level}'. Must be one of: {', '.join(valid_levels)}")
        return level
        
    def _setup_logging(self):
        """Configure logging based on debug settings"""
        if not self.enabled:
            logging.getLogger().addHandler(logging.NullHandler())
            self.logger = logging.getLogger(__name__)
            self.logger.disabled = True
            return
            
        log_level = getattr(logging, self.level, logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Clear any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            
        # Set up new handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handlers = [logging.StreamHandler()]
        
        if self.log_file:
            handlers.append(logging.FileHandler(self.log_file))
            
        for handler in handlers:
            handler.setLevel(log_level)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        logger.setLevel(log_level)
        self.logger = logger
        
    def _should_log(self, level: str) -> bool:
        """Check if a message of given level should be logged"""
        if not self.enabled:
            return False
            
        message_level = getattr(logging, level.upper(), logging.INFO)
        min_level = getattr(logging, self.level, logging.INFO)
        return message_level >= min_level
        
    def log(self, message: str, level: str = "INFO"):
        """Centralized logging method with better level handling"""
        if not self._should_log(level):
            return
            
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        try:
            log_method(message)
        except Exception as e:
            print(f"Logging failed ({level}): {message} | Error: {str(e)}")
            
    def debug(self, message: str):
        self.log(message, "DEBUG")
        
    def info(self, message: str):
        self.log(message, "INFO")
        
    def warning(self, message: str):
        self.log(message, "WARNING")
        
    def error(self, message: str):
        self.log(message, "ERROR")
        
    def critical(self, message: str):
        self.log(message, "CRITICAL")
        
    def exception(self, message: str):
        """Exception logging with stack trace"""
        if self._should_log("ERROR"):
            self.logger.exception(message)
            
    def log_dataframe(self, df: pd.DataFrame, message: str = "", level: str = "DEBUG"):
        """Log a pandas DataFrame in readable format"""
        if not self._should_log(level):
            return
            
        if message:
            self.log(message, level)
            
        buf = io.StringIO()
        df.info(buf=buf)
        self.log("\n" + buf.getvalue(), level)
        
        if len(df) <= 10:
            self.log("\n" + str(df), level)
        else:
            self.log(f"\nFirst 5 rows:\n{df.head()}\n\nLast 5 rows:\n{df.tail()}", level)
            
    def log_dict(self, data: dict, message: str = "", level: str = "DEBUG"):
        """Log a dictionary in readable format"""
        if not self._should_log(level):
            return
            
        if message:
            self.log(message, level)
            
        formatted = json.dumps(data, indent=2, default=str)
        self.log(f"\n{formatted}", level)


class NASDAQTraderFTP:
    """Class to handle NASDAQ Trader FTP operations for ticker data"""
    FTP_SERVER = 'ftp.nasdaqtrader.com'
    FTP_DIR = 'SymbolDirectory'
    
    def __init__(self, debugger: Debugger):
        self.debugger = debugger
        self.cache_dir = 'nasdaq_ftp_cache'
        os.makedirs(self.cache_dir, exist_ok=True)
        self.scanner = None

    def _connect_ftp(self) -> ftplib.FTP:
        """Connect to NASDAQ Trader FTP server"""
        try:
            ftp = ftplib.FTP(self.FTP_SERVER)
            ftp.login()  # Anonymous login
            ftp.cwd(self.FTP_DIR)
            return ftp
        except Exception as e:
            self.debugger.error(f"FTP connection error: {e}")
            raise
    
    def get_tickers_from_ftp(self) -> List[str]:
        """Get all active tickers from NASDAQ FTP"""
        try:
            ftp = self._connect_ftp()
            
            nasdaq_file = "nasdaqlisted.txt"
            with io.BytesIO() as buffer:
                ftp.retrbinary(f"RETR {nasdaq_file}", buffer.write)
                buffer.seek(0)
                nasdaq_data = buffer.getvalue().decode('utf-8').splitlines()
            
            traded_file = "nasdaqtraded.txt"
            with io.BytesIO() as buffer:
                ftp.retrbinary(f"RETR {traded_file}", buffer.write)
                buffer.seek(0)
                traded_data = buffer.getvalue().decode('utf-8').splitlines()
            
            ftp.quit()
            
            tickers = set()
            
            # Parse NASDAQ listed file
            for line in nasdaq_data[1:-1]:  # Skip header and footer
                parts = line.split('|')
                if len(parts) > 0 and parts[0]:
                    ticker = parts[0].replace('$', '').replace('.', '')
                    if self._is_valid_ticker(ticker):
                        tickers.add(ticker)
            
            # Parse NASDAQ traded file
            for line in traded_data[1:-1]:  # Skip header and footer
                parts = line.split('|')
                if len(parts) > 0 and parts[0]:
                    ticker = parts[0].replace('$', '').replace('.', '')
                    if self._is_valid_ticker(ticker):
                        tickers.add(ticker)
            
            return sorted(tickers)
            
        except Exception as e:
            self.debugger.error(f"Error getting tickers from FTP: {e}")
            return []
    
    def _is_valid_ticker(self, ticker: str, scanner: Optional['StockScanner'] = None) -> bool:
        """Validate ticker format using either basic checks or scanner's comprehensive checks"""
        if scanner or self.scanner:
            return (scanner or self.scanner)._is_valid_ticker(ticker)
        
        # Basic validation if no scanner is provided
        if not ticker or len(ticker) > 6:
            return False
            
        if not re.match(r'^[A-Z-]+$', ticker):
            return False
            
        return True


class AsyncPolygonIOClient:
    """Enhanced Polygon.io API client with caching and multi-timeframe support"""
    
    def __init__(self, api_key: str, debugger: Debugger):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.rate_limit_delay = 1.2
        self.last_request_time = 0
        self.session = None
        self.semaphore = asyncio.Semaphore(5)
        self.cache = {}
        self.debugger = debugger

    async def __aenter__(self):
        """Initialize async session"""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return self
        
    async def __aexit__(self, exc_type, exc, tb):
        """Cleanup async session"""
        await self.session.close()

    async def get_aggregates(self, ticker: str, days: int = 400, 
                        timespan: str = "day") -> Optional[pd.DataFrame]:
        """Get extended historical price data with caching"""
        cache_key = f"{ticker}_{days}_{timespan}"
        if cache_key in self.cache:
            return self.cache[cache_key].copy()

        try:
            end_date = date.today() - timedelta(days=1)
            start_date = end_date - timedelta(days=int(days * 1.2))  # 20% buffer

            self.debugger.debug(f"Fetching {ticker} {timespan} data from {start_date} to {end_date}")

            endpoint = f"/v2/aggs/ticker/{ticker}/range/1/{timespan}/{start_date}/{end_date}"
            params = {
                "adjusted": "true",
                "apiKey": self.api_key,
                "sort": "asc",
                "limit": 50000  # Max allowed by Polygon
            }

            async with self.semaphore:
                await self._throttle()
                async with async_timeout.timeout(45):  # Increased timeout
                    async with self.session.get(f"{self.base_url}{endpoint}", params=params) as response:
                        if response.status == 429:
                            await asyncio.sleep(15)  # Longer wait for rate limits
                            return await self.get_aggregates(ticker, days, timespan)
                        response.raise_for_status()
                        data = await response.json()

            if not data.get("results"):
                self.debugger.debug(f"No results for {ticker}")
                return None
                
            df = pd.DataFrame(data["results"])
            df["date"] = pd.to_datetime(df["t"], unit="ms")
            df = df.set_index("date")
            
            # Clean and validate data
            df = df[~df.index.duplicated(keep='first')]
            df = df.asfreq('D').ffill()  # Ensure daily frequency
            
            self.cache[cache_key] = df.copy()
            return df

        except Exception as e:
            self.debugger.error(f"Failed to fetch {ticker}: {str(e)}")
            return None

    async def get_all_tickers(self, market: str = "stocks") -> List[Dict]:
        """Get all active tickers from Polygon with better filtering"""
        try:
            tickers = []
            endpoint = f"/v3/reference/tickers"
            params = {
                "market": market,
                "active": "true",
                "apiKey": self.api_key,
                "limit": 1000
            }

            while True:
                async with self.semaphore:
                    await self._throttle()
                    async with async_timeout.timeout(30):
                        async with self.session.get(f"{self.base_url}{endpoint}", params=params) as response:
                            if response.status == 429:
                                await asyncio.sleep(10)
                                continue
                            response.raise_for_status()
                            data = await response.json()

                            if "results" not in data:
                                self.debugger.warning("No results in ticker response")
                                break

                            filtered = [
                                t for t in data["results"] 
                                if t.get("type") in ("CS", "ETF") and
                                not t["ticker"].endswith(('W', 'R', 'P', 'Q')) and
                                not any(c.isdigit() for c in t["ticker"]) and
                                not re.match(r'^[A-Z]+[23][XL]$', t["ticker"]) and
                                t.get("primary_exchange") in ["NASDAQ", "NYSE", "NYSEARCA"]
                            ]
                            
                            tickers.extend(filtered)
                            
                            if "next_url" in data:
                                endpoint = data["next_url"].replace(self.base_url, "")
                                params = {"apiKey": self.api_key}
                            else:
                                break

            return tickers

        except Exception as e:
            self.debugger.error(f"Failed to fetch tickers: {str(e)}")
            return []

    async def get_ticker_details(self, ticker: str) -> Optional[Dict]:
        """Get details for a specific ticker"""
        try:
            endpoint = f"/v3/reference/tickers/{ticker}"
            params = {"apiKey": self.api_key}

            async with self.semaphore:
                await self._throttle()
                async with async_timeout.timeout(30):
                    async with self.session.get(f"{self.base_url}{endpoint}", params=params) as response:
                        if response.status == 429:
                            await asyncio.sleep(10)
                            return await self.get_ticker_details(ticker)
                        response.raise_for_status()
                        return await response.json()

        except Exception as e:
            self.debugger.error(f"Failed to fetch details for {ticker}: {str(e)}")
            return None

    async def _throttle(self):
        """Enforce rate limiting with jitter"""
        now = asyncio.get_event_loop().time()
        elapsed = now - self.last_request_time
        if elapsed < self.rate_limit_delay:
            delay = self.rate_limit_delay - elapsed + random.uniform(0.1, 0.3)
            await asyncio.sleep(delay)
        self.last_request_time = now


class MarketRegimeDetector:
    """Advanced market regime detector with multi-timeframe analysis"""
    
    def __init__(self, polygon_client: AsyncPolygonIOClient, debugger: Debugger):
        self.polygon = polygon_client
        self.daily_data = None
        self.weekly_data = None
        self.volatility_regimes = ["low_vol", "medium_vol", "high_vol"]
        self.trend_regimes = ["strong_bull", "weak_bull", "neutral", "weak_bear", "strong_bear"]
        self.debugger = debugger
        
    async def initialize(self) -> bool:
        """Initialize with daily and weekly data"""
        try:
            self.debugger.debug("Initializing market regime detector...")
            
            self.daily_data = await self.polygon.get_aggregates("QQQ", days=252, timespan="day")
            self.weekly_data = await self.polygon.get_aggregates("QQQ", days=252, timespan="week")
            
            if self.daily_data is None or self.weekly_data is None:
                self.debugger.debug("Failed to fetch QQQ data - API returned None")
                return False
                
            if len(self.daily_data) < 100 or len(self.weekly_data) < 20:
                self.debugger.debug(f"Insufficient data points: Daily={len(self.daily_data)}, Weekly={len(self.weekly_data)}")
                return False
                
            self.debugger.debug(f"Loaded {len(self.daily_data)} daily and {len(self.weekly_data)} weekly data points")
            self._calculate_technical_indicators()
            return True
            
        except Exception as e:
            self.debugger.error(f"Error initializing detector: {str(e)}")
            return False
        
    def _calculate_technical_indicators(self):
        """Calculate advanced technical indicators"""
        daily_closes = self.daily_data['c'].values
        daily_highs = self.daily_data['h'].values
        daily_lows = self.daily_data['l'].values
        daily_volumes = self.daily_data['v'].values
        
        self.daily_data['sma_50'] = talib.SMA(daily_closes, timeperiod=50)
        self.daily_data['sma_200'] = talib.SMA(daily_closes, timeperiod=200)
        self.daily_data['rsi_14'] = talib.RSI(daily_closes, timeperiod=14)
        self.daily_data['macd'], self.daily_data['macd_signal'], _ = talib.MACD(daily_closes)
        self.daily_data['atr_14'] = talib.ATR(daily_highs, daily_lows, daily_closes, timeperiod=14)
        self.daily_data['adx'] = talib.ADX(daily_highs, daily_lows, daily_closes, timeperiod=14)
        
        log_returns = np.log(daily_closes[1:]/daily_closes[:-1])
        self.daily_data['hist_vol_30'] = pd.Series(log_returns).rolling(30).std() * np.sqrt(252)
        
        hl_ratio = np.log(self.daily_data['h']/self.daily_data['l'])
        self.daily_data['parkinson_vol'] = hl_ratio.rolling(14).std() * np.sqrt(252)
        
        self.daily_data['volume_sma_20'] = talib.SMA(daily_volumes, timeperiod=20)
        self.daily_data['volume_ratio'] = daily_volumes / self.daily_data['volume_sma_20']
        
        weekly_closes = self.weekly_data['c'].values
        self.weekly_data['sma_10'] = talib.SMA(weekly_closes, timeperiod=10)
        self.weekly_data['sma_40'] = talib.SMA(weekly_closes, timeperiod=40)
        
        trend_components = []
        trend_components.append(0.3 * (self.daily_data['sma_50'] > self.daily_data['sma_200']))
        trend_components.append(0.2 * (self.weekly_data['sma_10'] > self.weekly_data['sma_40']).resample('D').ffill())
        trend_components.append(0.2 * (self.daily_data['adx'] / 100))
        trend_components.append(0.1 * (self.daily_data['macd'] > self.daily_data['macd_signal']))
        trend_components.append(0.1 * (self.daily_data['c'] > self.daily_data['sma_50']))
        trend_components.append(0.1 * (self.daily_data['rsi_14'] / 100))
        
        self.daily_data['trend_strength'] = sum(tc[:len(self.daily_data)] for tc in trend_components)

    async def get_scan_criteria(self) -> Dict:
        """More inclusive scanning criteria with regime awareness"""
        base_criteria = {
            "min_volume": 500_000,
            "min_price": 5.00,
            "max_price": 500.00,
            "min_market_cap": 300_000_000,
            "momentum_days": 30,
            "min_momentum": 0.05,
            "min_relative_strength": 1.0,
            "volatility_multiplier": 1.0,
            "pattern_score_threshold": 1,
            "days_to_scan": 90
        }
        
        trend, vol = await self.get_current_regime()
        
        if trend == "strong_bull":
            base_criteria.update({
                "min_momentum": 0.10,
                "min_relative_strength": 1.1,
                "pattern_priority": ["breakout", "golden_cross", "bullish_ma_stack"]
            })
        elif trend == "weak_bull":
            base_criteria.update({
                "min_momentum": 0.07,
                "min_relative_strength": 1.05,
                "pattern_priority": ["support_bounce", "above_200ma"]
            })
        elif trend == "neutral":
            base_criteria.update({
                "min_momentum": 0.03,
                "min_relative_strength": 0.95,
                "pattern_priority": ["mean_reversion", "hammer", "doji"]
            })
        elif trend == "weak_bear":
            base_criteria.update({
                "min_momentum": -0.03,
                "min_relative_strength": 0.90,
                "pattern_priority": ["oversold", "falling_wedge"]
            })
        elif trend == "strong_bear":
            base_criteria.update({
                "min_momentum": -0.05,
                "min_relative_strength": 0.85,
                "pattern_priority": ["short_squeeze", "double_bottom"]
            })
        
        if vol == "low_vol":
            base_criteria.update({
                "min_volatility": 0.01,
                "max_volatility": 0.03
            })
        elif vol == "medium_vol":
            base_criteria.update({
                "min_volatility": 0.03,
                "max_volatility": 0.06
            })
        elif vol == "high_vol":
            base_criteria.update({
                "min_volatility": 0.06,
                "max_volatility": 0.15
            })
        
        return base_criteria

    async def get_current_regime(self) -> Tuple[str, str]:
        """Get the most likely current market regime"""
        regimes = await self.detect_regime()
        
        trend_regime = max(
            ((r, regimes[r]) for r in self.trend_regimes),
            key=lambda x: x[1],
            default=("neutral", 0)
        )[0]
        
        vol_regime = max(
            ((r, regimes[r]) for r in self.volatility_regimes),
            key=lambda x: x[1],
            default=("medium_vol", 0)
        )[0]
        
        return trend_regime, vol_regime
    
    async def detect_regime(self) -> Dict[str, float]:
        """Enhanced regime detection with multi-timeframe analysis"""
        if self.daily_data is None:
            return self._default_regime_probabilities()
        
        recent_daily = self.daily_data.iloc[-1]
        recent_weekly = self.weekly_data.iloc[-1]
        
        price_above_200sma = recent_daily['c'] > recent_daily['sma_200']
        sma_50_above_200 = recent_daily['sma_50'] > recent_daily['sma_200']
        weekly_sma_10_above_40 = recent_weekly['sma_10'] > recent_weekly['sma_40']
        trend_strength = recent_daily['trend_strength']
        rsi_14 = recent_daily['rsi_14']
        macd_above_signal = recent_daily['macd'] > recent_daily['macd_signal']
        
        atr_14 = recent_daily['atr_14']
        atr_30 = self.daily_data['atr_14'].iloc[-30:].mean()
        vol_ratio = atr_14 / atr_30 if atr_30 > 0 else 1.0
        hist_vol = recent_daily['hist_vol_30']
        parkinson_vol = recent_daily['parkinson_vol']
        
        momentum_5d = (recent_daily['c'] / self.daily_data['c'].iloc[-5] - 1) * 100
        momentum_30d = (recent_daily['c'] / self.daily_data['c'].iloc[-30] - 1) * 100
        weekly_momentum = (recent_weekly['c'] / self.weekly_data['c'].iloc[-4] - 1) * 100
        
        volume_spike = recent_daily['volume_ratio'] > 1.5
        
        regimes = {
            "strong_bull": self._strong_bull_confidence(
                price_above_200sma, sma_50_above_200, weekly_sma_10_above_40,
                trend_strength, rsi_14, macd_above_signal,
                momentum_5d, momentum_30d, weekly_momentum
            ),
            "weak_bull": self._weak_bull_confidence(
                price_above_200sma, sma_50_above_200, weekly_sma_10_above_40,
                trend_strength, rsi_14, macd_above_signal,
                momentum_5d, momentum_30d, weekly_momentum
            ),
            "neutral": self._neutral_confidence(
                rsi_14, vol_ratio, atr_14, atr_30,
                trend_strength, hist_vol, parkinson_vol
            ),
            "weak_bear": self._weak_bear_confidence(
                price_above_200sma, sma_50_above_200, weekly_sma_10_above_40,
                trend_strength, rsi_14, macd_above_signal,
                momentum_5d, momentum_30d, weekly_momentum
            ),
            "strong_bear": self._strong_bear_confidence(
                price_above_200sma, sma_50_above_200, weekly_sma_10_above_40,
                trend_strength, rsi_14, macd_above_signal,
                momentum_5d, momentum_30d, weekly_momentum
            ),
            "low_vol": self._low_vol_confidence(
                vol_ratio, atr_14, atr_30, hist_vol, parkinson_vol
            ),
            "medium_vol": self._medium_vol_confidence(
                vol_ratio, atr_14, atr_30, hist_vol, parkinson_vol
            ),
            "high_vol": self._high_vol_confidence(
                vol_ratio, atr_14, atr_30, hist_vol, parkinson_vol, volume_spike
            )
        }
        
        return self._normalize_regime_probabilities(regimes)

    async def get_regime_description(self) -> Dict:
        """Enhanced regime description with transition info"""
        regimes = await self.detect_regime()
        trend_regime, vol_regime = await self.get_current_regime()
        transition_info = self._analyze_regime_transitions()
        
        return {
            "primary_trend": str(trend_regime),
            "primary_volatility": str(vol_regime),
            "trend_probabilities": {str(r): float(regimes[r]) for r in self.trend_regimes},
            "volatility_probabilities": {str(r): float(regimes[r]) for r in self.volatility_regimes},
            "transition_analysis": {
                'potential_transition': bool(transition_info['potential_transition']),
                'trend_weakening': bool(transition_info['trend_weakening']),
                'vol_increasing': bool(transition_info['vol_increasing'])
            },
            "timestamp": datetime.now().isoformat()
        }

    def _analyze_regime_transitions(self) -> Dict[str, bool]:
        """Detect potential regime transitions"""
        if self.daily_data is None or len(self.daily_data) < 5:
            return {
                'potential_transition': False,
                'trend_weakening': False,
                'vol_increasing': False
            }
        
        recent_daily = self.daily_data.iloc[-5:]  # Last 5 days
        
        # Trend transition signals
        trend_weakening = (
            (recent_daily['trend_strength'].pct_change(fill_method=None).mean() < -0.05) or
            (recent_daily['rsi_14'].iloc[-1] < 30 and recent_daily['rsi_14'].iloc[-1] < recent_daily['rsi_14'].iloc[-5])
        )
        
        # Volatility transition signals
        vol_increasing = (
            (recent_daily['atr_14'].pct_change().mean() > 0.1) or
            (recent_daily['parkinson_vol'].iloc[-1] > recent_daily['parkinson_vol'].iloc[-5] * 1.2)
        )
        
        return {
            'potential_transition': trend_weakening or vol_increasing,
            'trend_weakening': trend_weakening,
            'vol_increasing': vol_increasing
        }

    def _default_regime_probabilities(self) -> Dict[str, float]:
        return {
            "strong_bull": 0.2, "weak_bull": 0.2, "neutral": 0.2, 
            "weak_bear": 0.2, "strong_bear": 0.2,
            "low_vol": 0.33, "medium_vol": 0.34, "high_vol": 0.33
        }
    
    def _normalize_regime_probabilities(self, regimes: Dict[str, float]) -> Dict[str, float]:
        trend_total = sum(regimes[r] for r in self.trend_regimes)
        vol_total = sum(regimes[r] for r in self.volatility_regimes)
        
        normalized = {}
        for r in regimes:
            if r in self.trend_regimes:
                normalized[r] = regimes[r] / trend_total if trend_total > 0 else 0
            else:
                normalized[r] = regimes[r] / vol_total if vol_total > 0 else 0
        return normalized
    
    def _strong_bull_confidence(self, price_above_200sma: bool, sma_50_above_200: bool,
                              weekly_sma_10_above_40: bool, trend_strength: float, 
                              rsi_14: float, macd_above_signal: bool,
                              momentum_5d: float, momentum_30d: float,
                              weekly_momentum: float) -> float:
        score = 0
        if price_above_200sma: score += 0.15
        if sma_50_above_200: score += 0.15
        if weekly_sma_10_above_40: score += 0.1
        if trend_strength > 0.75: score += 0.15
        if 60 < rsi_14 <= 80: score += 0.1
        if macd_above_signal: score += 0.1
        if momentum_5d > 1.5: score += 0.1
        if momentum_30d > 5.0: score += 0.1
        if weekly_momentum > 3.0: score += 0.05
        return score
    
    def _weak_bull_confidence(self, price_above_200sma: bool, sma_50_above_200: bool,
                            weekly_sma_10_above_40: bool, trend_strength: float,
                            rsi_14: float, macd_above_signal: bool,
                            momentum_5d: float, momentum_30d: float,
                            weekly_momentum: float) -> float:
        score = 0
        if price_above_200sma: score += 0.15
        if sma_50_above_200: score += 0.1
        if weekly_sma_10_above_40: score += 0.05
        if trend_strength > 0.55: score += 0.15
        if 50 < rsi_14 <= 60: score += 0.15
        if macd_above_signal: score += 0.1
        if momentum_5d > 0: score += 0.1
        if momentum_30d > 2.0: score += 0.1
        if weekly_momentum > 1.0: score += 0.1
        return score
    
    def _neutral_confidence(self, rsi_14: float, vol_ratio: float, atr_14: float,
                          atr_30: float, trend_strength: float,
                          hist_vol: float, parkinson_vol: float) -> float:
        score = 0
        if 40 <= rsi_14 <= 60: score += 0.25
        if 0.9 <= vol_ratio <= 1.1: score += 0.2
        if 0.3 <= trend_strength <= 0.7: score += 0.2
        if 0.8 <= (atr_14 / atr_30) <= 1.2 if atr_30 > 0 else False: score += 0.15
        if 0.9 <= (hist_vol / parkinson_vol) <= 1.1 if parkinson_vol > 0 else False: score += 0.2
        return score
    
    def _weak_bear_confidence(self, price_above_200sma: bool, sma_50_above_200: bool,
                             weekly_sma_10_above_40: bool, trend_strength: float,
                             rsi_14: float, macd_above_signal: bool,
                             momentum_5d: float, momentum_30d: float,
                             weekly_momentum: float) -> float:
        score = 0
        if not price_above_200sma: score += 0.15
        if not sma_50_above_200: score += 0.1
        if not weekly_sma_10_above_40: score += 0.05
        if trend_strength < 0.45: score += 0.15
        if 30 <= rsi_14 < 50: score += 0.15
        if not macd_above_signal: score += 0.1
        if momentum_5d < 0: score += 0.1
        if momentum_30d < -2.0: score += 0.1
        if weekly_momentum < -1.0: score += 0.1
        return score
    
    def _strong_bear_confidence(self, price_above_200sma: bool, sma_50_above_200: bool,
                               weekly_sma_10_above_40: bool, trend_strength: float,
                               rsi_14: float, macd_above_signal: bool,
                               momentum_5d: float, momentum_30d: float,
                               weekly_momentum: float) -> float:
        score = 0
        if not price_above_200sma: score += 0.15
        if not sma_50_above_200: score += 0.15
        if not weekly_sma_10_above_40: score += 0.1
        if trend_strength < 0.25: score += 0.15
        if rsi_14 < 30: score += 0.1
        if not macd_above_signal: score += 0.1
        if momentum_5d < -1.5: score += 0.1
        if momentum_30d < -5.0: score += 0.1
        if weekly_momentum < -3.0: score += 0.05
        return score
    
    def _low_vol_confidence(self, vol_ratio: float, atr_14: float, atr_30: float,
                           hist_vol: float, parkinson_vol: float) -> float:
        if (vol_ratio < 0.7 and 
            (atr_14 / atr_30) < 0.7 if atr_30 > 0 else False and
            hist_vol < 0.15 and 
            parkinson_vol < 0.15):
            return 0.9
        return 0
    
    def _medium_vol_confidence(self, vol_ratio: float, atr_14: float, atr_30: float,
                              hist_vol: float, parkinson_vol: float) -> float:
        if (0.7 <= vol_ratio <= 1.3 and 
            0.7 <= (atr_14 / atr_30) <= 1.3 if atr_30 > 0 else False and
            0.15 <= hist_vol <= 0.30 and 
            0.15 <= parkinson_vol <= 0.30):
            return 0.9
        return 0
    
    def _high_vol_confidence(self, vol_ratio: float, atr_14: float, atr_30: float,
                            hist_vol: float, parkinson_vol: float,
                            volume_spike: bool) -> float:
        if ((vol_ratio > 1.3 or 
             (atr_14 / atr_30) > 1.3 if atr_30 > 0 else False or
             hist_vol > 0.30 or 
             parkinson_vol > 0.30) and
            volume_spike):
            return 0.9
        return 0


class StockScanner:
    """Complete stock scanner with multi-strategy support"""
    
    def __init__(
        self, 
        polygon_client: AsyncPolygonIOClient, 
        debugger: Debugger,
        max_tickers_to_scan: Optional[int] = 100
    ):
        self.polygon = polygon_client
        self.regime_detector = MarketRegimeDetector(polygon_client, debugger)
        self.ftp_client = NASDAQTraderFTP(debugger)
        self.ftp_client.scanner = self
        self.debugger = debugger
        self.yahoo_fallback = YahooFinanceFallback(debugger)
        
        # Initialize all instance attributes
        self.scan_results = []
        self.rejection_stats = defaultdict(int)
        self.tickers_to_scan = []
        self.max_tickers_to_scan = max_tickers_to_scan
        
        self.strategy_config = {
            "hypergrowth": {
                "allocation": (0.4, 0.6),
                "holding_period": (14, 42),
                "stop_loss": 0.15,
                "profit_targets": [(0.25, 0.5), (0.5, 0.3), (1.0, 0.2)],
                "filters": {
                    "min_revenue_growth": 0.25,
                    "min_volume_ratio": 1.5,
                    "min_price_relative": 1.1,
                    "min_momentum": 0.10
                },
                "backtest_class": FixedHypergrowthStrategy
            },
            "momentum": {
                "allocation": (0.3, 0.4),
                "holding_period": (90, 365),
                "stop_loss": 0.25,
                "profit_targets": [(0.5, 0.5), (1.0, 0.5)],
                "filters": {
                    "ma_condition": "50ma > 200ma",
                    "min_adx": 20,
                    "min_trend_duration": 20
                },
                "backtest_class": FixedMomentumStrategy
            },
            "breakout": {
                "allocation": (0.1, 0.2),
                "holding_period": (7, 21),
                "stop_loss": 0.08,
                "profit_targets": [(0.15, 0.5), (0.25, 0.5)],
                "filters": {
                    "min_volume_ratio": 2.0,
                    "min_volatility": 0.04,
                    "consolidation_days": 10,
                    "max_price": 300
                },
                "backtest_class": FixedBreakoutStrategy
            }
        }

    async def initialize(self):
        """Initialize scanner with market regime and tickers"""
        if not await self.regime_detector.initialize():
            self.debugger.error("Failed to initialize market regime detector")
            return False
            
        self.tickers_to_scan = await self._load_tickers_to_scan()
        if not self.tickers_to_scan:
            self.debugger.error("No tickers to scan")
            return False
            
        self.debugger.info(f"Loaded {len(self.tickers_to_scan)} tickers for scanning")
        return True

    def _is_valid_ticker(self, ticker: str) -> bool:
        """Comprehensive ticker validation"""
        if not ticker or len(ticker) > 6:
            return False
            
        if not re.match(r'^[A-Z]+$', ticker):
            return False
            
        # Filter out leveraged ETFs and non-standard symbols
        if any(x in ticker for x in ['2X', '3X', '1X', '2', '3']):
            return False
            
        # Filter out warrants, rights, preferred shares, etc.
        non_stock_suffixes = ('W', 'R', 'P', 'Q', 'Y', 'F', 'V', 'J', 'M', 'Z')
        if ticker.endswith(non_stock_suffixes):
            return False
            
        # Filter out leveraged/inverse ETFs
        if re.match(r'^[A-Z]+[23][XL]$', ticker):
            return False
            
        return True

    async def _load_tickers_to_scan(self) -> List[str]:
        """Load tickers to scan, respecting max_tickers_to_scan"""
        try:
            # Debug: Confirm the limit
            limit_msg = "all tickers" if self.max_tickers_to_scan is None else f"max {self.max_tickers_to_scan} tickers"
            self.debugger.debug(f"Loading tickers ({limit_msg})")

            # Get tickers from Polygon (apply limit early if specified)
            polygon_tickers = await self.polygon.get_all_tickers()
            polygon_symbols = [t["ticker"] for t in polygon_tickers]
            if self.max_tickers_to_scan is not None:
                polygon_symbols = polygon_symbols[:self.max_tickers_to_scan]

            # Get tickers from FTP (apply limit early if specified)
            ftp_tickers = self.ftp_client.get_tickers_from_ftp()
            if self.max_tickers_to_scan is not None:
                ftp_tickers = ftp_tickers[:self.max_tickers_to_scan]

            # Combine, deduplicate, and enforce limit again if specified
            all_tickers = list(set(polygon_symbols + ftp_tickers))
            if self.max_tickers_to_scan is not None:
                all_tickers = all_tickers[:self.max_tickers_to_scan]

            # Filter invalid tickers and shuffle
            valid_tickers = [t for t in all_tickers if self._is_valid_ticker(t)]
            random.shuffle(valid_tickers)

            # Final limit (in case filtering removed too many)
            if self.max_tickers_to_scan is not None:
                valid_tickers = valid_tickers[:self.max_tickers_to_scan]

            return valid_tickers

        except Exception as e:
            self.debugger.error(f"Error loading tickers: {str(e)}")
            return []

    async def scan_tickers(self):
        """Main scanning procedure with automatic backtesting"""
        if not await self.initialize():
            return
            
        criteria = await self.regime_detector.get_scan_criteria()
        self.debugger.info(f"Starting scan with criteria: {criteria}")
        
        # Process in batches with progress bar
        batch_size = 50
        with tqdm(total=len(self.tickers_to_scan), desc="Scanning tickers") as pbar:
            for i in range(0, len(self.tickers_to_scan), batch_size):
                batch = self.tickers_to_scan[i:i + batch_size]
                await self._process_batch(batch, criteria)
                pbar.update(len(batch))
                await asyncio.sleep(1)  # Rate limiting
                
        self._classify_strategies()
        
        if self.scan_results:
            self.print_summary()
            await self._backtest_top_candidates()
        else:
            self.debugger.warning("No qualifying stocks found")

    async def _process_batch(self, tickers: List[str], criteria: Dict):
        """Process a batch of tickers with parallel requests"""
        tasks = [self._scan_single_ticker(ticker, criteria) for ticker in tickers]
        await asyncio.gather(*tasks)

    async def _scan_single_ticker(self, ticker: str, criteria: Dict) -> Optional[Dict]:
        """Complete analysis for a single ticker"""
        try:
            # Get price data and details
            price_data = await self.polygon.get_aggregates(ticker, days=criteria["days_to_scan"])
            if price_data is None or len(price_data) < 20:
                self._log_rejection(ticker, "insufficient data")
                return None
                
            details = await self.polygon.get_ticker_details(ticker)
            
            # Calculate indicators
            indicators = self._calculate_indicators(price_data)
            patterns = self._detect_patterns(price_data)
            
            # Validate against basic criteria
            if not self._validate_ticker(price_data, indicators, criteria):
                return None
                
            # Score and classify
            score = self._calculate_composite_score(price_data, indicators, patterns, criteria)
            strategy = self._determine_primary_strategy(price_data, indicators, details)
            
            # Store results
            result = {
                "ticker": ticker,
                "data": price_data,
                "indicators": indicators,
                "patterns": patterns,
                "details": details,
                "score": score,
                "strategy": strategy
            }
            
            self.scan_results.append(result)
            return result
            
        except Exception as e:
            self._log_rejection(ticker, f"scan error: {str(e)}")
            return None

    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate all technical indicators"""
        closes = data['c'].values
        highs = data['h'].values
        lows = data['l'].values
        volumes = data['v'].values
        
        return {
            'sma_50': talib.SMA(closes, 50)[-1],
            'sma_200': talib.SMA(closes, 200)[-1],
            'atr': talib.ATR(highs, lows, closes, 14)[-1],
            'rsi': talib.RSI(closes, 14)[-1],
            'adx': talib.ADX(highs, lows, closes, 14)[-1],
            'volume_ma': np.mean(volumes[-20:]),
            'momentum': (closes[-1] / closes[-30] - 1) * 100 if len(closes) >= 30 else 0,
            'volatility': np.std(np.log(closes[-30:]/closes[-31:-1])) * np.sqrt(252) * 100 if len(closes) >= 31 else 0
        }

    def _detect_patterns(self, data: pd.DataFrame) -> List[str]:
        """Identify technical patterns"""
        patterns = []
        o, h, l, c = data['o'].values, data['h'].values, data['l'].values, data['c'].values
        
        # Candlestick patterns
        if talib.CDLHAMMER(o, h, l, c)[-1] > 0: patterns.append("hammer")
        if talib.CDLENGULFING(o, h, l, c)[-1] > 0: patterns.append("bullish_engulfing")
        
        # Support/resistance
        resistance = np.max(h[-20:-1])
        support = np.min(l[-20:-1])
        if c[-1] > resistance: patterns.append("breakout")
        if c[-1] < support: patterns.append("breakdown")
        
        # Moving average cross
        sma50 = talib.SMA(c, 50)
        sma200 = talib.SMA(c, 200)
        if sma50[-1] > sma200[-1] and sma50[-2] <= sma200[-2]: patterns.append("golden_cross")
        
        return patterns

    def _validate_ticker(self, data: pd.DataFrame, indicators: Dict, criteria: Dict) -> bool:
        """Check if ticker meets basic criteria"""
        price = data['c'].iloc[-1]
        volume = data['v'].iloc[-1]
        
        if not (criteria["min_price"] <= price <= criteria["max_price"]):
            return False
        if volume < criteria["min_volume"]:
            return False
        if indicators['momentum'] < criteria["min_momentum"]:
            return False
            
        return True

    def _calculate_composite_score(self, data: pd.DataFrame, indicators: Dict, 
                                patterns: List[str], criteria: Dict) -> float:
        """Calculate normalized quality score (0-100)"""
        weights = {
            'momentum': 0.35,
            'volume': 0.25,
            'volatility': 0.15,
            'patterns': 0.25
        }
        
        # Normalize components to 0-1 range
        norm_momentum = min(max(indicators['momentum'] / 30, 0), 1)  # Cap at ±30% momentum
        norm_volume = min(max(indicators['volume_ma'] / criteria["min_volume"], 0), 2)  # Cap at 2x min volume
        norm_volatility = min(max((indicators['volatility'] - 0.2)/0.3, 0), 1)  # 20-50% vol range
        norm_patterns = min(len(patterns) / 3, 1)  # Max 3 patterns
        
        return 100 * (
            weights['momentum'] * norm_momentum +
            weights['volume'] * norm_volume +
            weights['volatility'] * norm_volatility +
            weights['patterns'] * norm_patterns
        )

    def _determine_primary_strategy(self, data: pd.DataFrame, indicators: Dict, 
                                  details: Dict) -> str:
        """Classify stock into best-fitting strategy"""
        scores = {
            'hypergrowth': self._score_hypergrowth(data, indicators, details),
            'momentum': self._score_momentum(data, indicators),
            'breakout': self._score_breakout(data, indicators)
        }
        return max(scores.items(), key=lambda x: x[1])[0]

    def _score_hypergrowth(self, data: pd.DataFrame, indicators: Dict, details: Dict) -> float:
        """More realistic hypergrowth scoring"""
        score = 0
        cfg = self.strategy_config['hypergrowth']['filters']
        
        # Fundamental factors (if available)
        if details:
            if details.get('revenue_growth_rate', 0) > cfg['min_revenue_growth']:
                score += 0.4
            elif 'revenue_growth_rate' not in details:
                score += 0.2  # Partial credit if data unavailable
        
        # Technical factors
        price_growth_30d = indicators['momentum']
        if price_growth_30d > cfg['min_momentum']:
            score += 0.3 * min(price_growth_30d/50, 1)  # Scale with growth rate
            
        volume_ratio = data['v'].iloc[-1] / indicators['volume_ma']
        if volume_ratio > cfg['min_volume_ratio']:
            score += 0.2 * min(volume_ratio/3, 1)  # Cap at 3x volume
            
        if data['c'].iloc[-1] > data['c'].iloc[-20] * cfg['min_price_relative']:
            score += 0.1
            
        return min(score, 1.0)  # Cap at 1.0

    def _score_momentum(self, data: pd.DataFrame, indicators: Dict) -> float:
        """Score for momentum strategy"""
        score = 0
        cfg = self.strategy_config['momentum']['filters']
        
        if indicators['sma_50'] > indicators['sma_200']:
            score += 0.4
        if indicators['adx'] > cfg['min_adx']:
            score += 0.3
        if indicators['momentum'] > 0:
            score += 0.2
        if len(data) > cfg['min_trend_duration']:
            score += 0.1
            
        return score

    def _score_breakout(self, data: pd.DataFrame, indicators: Dict) -> float:
        """Score for breakout strategy"""
        score = 0
        cfg = self.strategy_config['breakout']['filters']
        
        if "breakout" in self._detect_patterns(data):
            score += 0.5
        if data['v'].iloc[-1] > indicators['volume_ma'] * cfg['min_volume_ratio']:
            score += 0.3
        if indicators['volatility'] > cfg['min_volatility']:
            score += 0.2
            
        return score

    def _score_strategy(self, stock: Dict, strategy: str) -> float:
        """Score how well a stock fits a particular strategy"""
        if strategy == 'hypergrowth':
            return self._score_hypergrowth(stock['data'], stock['indicators'], stock['details'])
        elif strategy == 'momentum':
            return self._score_momentum(stock['data'], stock['indicators'])
        elif strategy == 'breakout':
            return self._score_breakout(stock['data'], stock['indicators'])
        return 0.0

    def _classify_strategies(self):
        """Enhanced strategy classification with forced diversification"""
        if not self.scan_results:
            return
            
        # First pass - classify all stocks normally
        for stock in self.scan_results:
            stock['strategy'] = self._determine_primary_strategy(
                stock['data'], stock['indicators'], stock['details'])
        
        # Calculate current allocations
        strategy_counts = defaultdict(int)
        for stock in self.scan_results:
            strategy_counts[stock['strategy']] += 1
        
        total_stocks = len(self.scan_results)
        
        # Force minimum allocations for underrepresented strategies
        for strategy, config in self.strategy_config.items():
            min_target = max(1, int(config['allocation'][0] * total_stocks))
            current = strategy_counts.get(strategy, 0)
            
            if current < min_target:
                # Find borderline candidates that could fit this strategy
                candidates = sorted(
                    [s for s in self.scan_results 
                    if s['strategy'] != strategy and
                    self._score_strategy(s, strategy) > 0.4],  # At least 40% fit
                    key=lambda x: abs(x['score'] - self._score_strategy(x, strategy)),
                    reverse=True
                )
                
                # Reclassify top candidates
                for stock in candidates[:min_target - current]:
                    stock['strategy'] = strategy
                    strategy_counts[strategy] += 1
                    strategy_counts[stock['strategy']] -= 1
        
        # Relax criteria for strategies that are still underrepresented
        for strategy, config in self.strategy_config.items():
            min_target = max(1, int(config['allocation'][0] * total_stocks))
            current = strategy_counts.get(strategy, 0)
            
            if current < min_target:
                self._adjust_strategy_criteria(strategy, relaxation_factor=0.8)
                self.debugger.info(f"Relaxed {strategy} criteria due to low allocation")
                
                # Reclassify with relaxed criteria
                for stock in self.scan_results:
                    if stock['strategy'] != strategy and self._score_strategy(stock, strategy) > 0.5:
                        stock['strategy'] = strategy
                        strategy_counts[strategy] += 1
                        strategy_counts[stock['strategy']] -= 1
                        if strategy_counts[strategy] >= min_target:
                            break

    def _adjust_strategy_criteria(self, strategy: str, relaxation_factor: float = 0.8):
        """Temporarily relax strategy criteria to get more candidates"""
        if strategy == "hypergrowth":
            self.strategy_config["hypergrowth"]["filters"]["min_revenue_growth"] *= relaxation_factor
            self.strategy_config["hypergrowth"]["filters"]["min_momentum"] *= relaxation_factor
        elif strategy == "momentum":
            self.strategy_config["momentum"]["filters"]["min_adx"] *= relaxation_factor
            self.strategy_config["momentum"]["filters"]["min_trend_duration"] *= relaxation_factor

    async def _backtest_top_candidates(self, n_per_strategy: int = 3):
        """Backtest top candidates from each strategy"""
        if not self.scan_results:
            return
            
        by_strategy = defaultdict(list)
        for stock in self.scan_results:
            by_strategy[stock['strategy']].append(stock)
            
        for strategy, stocks in by_strategy.items():
            top_stocks = sorted(stocks, key=lambda x: x['score'], reverse=True)[:n_per_strategy]
            for stock in top_stocks:
                self.debugger.info(f"\nBacktesting {stock['ticker']} ({strategy})...")
                await run_backtest(stock, self.debugger)

    def generate_trade_plan(self, ticker: str) -> Optional[Dict]:
        """Generate complete trade plan for a ticker"""
        stock = next((s for s in self.scan_results if s['ticker'] == ticker), None)
        if not stock:
            return None
            
        config = self.strategy_config[stock['strategy']]
        price = stock['data']['c'].iloc[-1]
        atr = stock['indicators']['atr']
        
        return {
            "ticker": ticker,
            "strategy": stock['strategy'],
            "entry_price": price,
            "position_size": self._calculate_position_size(stock, config),
            "stop_loss": price * (1 - config['stop_loss']),
            "take_profit": self._calculate_profit_targets(price, config),
            "holding_days": random.randint(*config['holding_period']),
            "risk_reward": self._calculate_risk_reward(price, config),
            "monitoring_rules": self._get_monitoring_rules(stock['strategy'])
        }

    def _calculate_position_size(self, stock: Dict, config: Dict) -> float:
        """Calculate position size based on risk parameters"""
        account_size = 100000  # Example account size
        risk_per_trade = 0.01  # 1% of account
        risk_amount = account_size * risk_per_trade
        risk_per_share = stock['data']['c'].iloc[-1] * config['stop_loss']
        shares = risk_amount / risk_per_share
        
        # Respect allocation limits
        max_allocation = config['allocation'][1] * account_size
        position_value = shares * stock['data']['c'].iloc[-1]
        
        if position_value > max_allocation:
            shares = max_allocation / stock['data']['c'].iloc[-1]
            
        return round(shares)

    def _calculate_profit_targets(self, entry_price: float, config: Dict) -> List[Dict]:
        """Generate strategy-appropriate profit targets"""
        return [
            {
                "price": round(entry_price * (1 + target[0]), 2),
                "size": target[1]
            } 
            for target in config['profit_targets']
        ]

    def _calculate_risk_reward(self, entry_price: float, config: Dict) -> float:
        """Calculate average risk-reward ratio"""
        avg_return = sum(t[0] * t[1] for t in config['profit_targets'])
        return round(avg_return / config['stop_loss'], 2)

    def _get_monitoring_rules(self, strategy: str) -> List[str]:
        """Get strategy-specific monitoring guidelines"""
        rules = {
            "hypergrowth": [
                "Monitor earnings calendar",
                "Watch for volume drying up",
                "Consider partial profits at 25% gain"
            ],
            "momentum": [
                "Track 50/200 MA relationship",
                "Watch for ADX < 25",
                "Trail stop after 10% gain"
            ],
            "breakout": [
                "Confirm volume on breakout",
                "Watch for failed breaks",
                "Close if returns to breakout point"
            ]
        }
        return rules.get(strategy, [])

    def print_summary(self):
        """Print comprehensive scan results"""
        if not self.scan_results:
            self.debugger.info("No qualifying stocks found")
            return
            
        # Strategy allocation
        self.debugger.info("\nStrategy Allocation:")
        self.debugger.info("-------------------")
        counts = defaultdict(int)
        for stock in self.scan_results:
            counts[stock['strategy']] += 1
            
        for strategy, config in self.strategy_config.items():
            count = counts.get(strategy, 0)
            pct = (count / len(self.scan_results)) * 100
            self.debugger.info(f"{strategy.upper():<12}: {count:>3} stocks ({pct:.1f}%)")
        
        # Top candidates table
        self.debugger.info("\nTop Candidates:")
        self.debugger.info("--------------")
        headers = ["Ticker", "Price", "Strategy", "Momentum", "Volatility", "Score", "Patterns"]
        self.debugger.info("{:<6} {:<8} {:<12} {:<9} {:<10} {:<6} {}".format(*headers))
        
        for stock in sorted(self.scan_results, key=lambda x: x['score'], reverse=True)[:20]:
            self.debugger.info(
                f"{stock['ticker']:<6} "
                f"${stock['data']['c'].iloc[-1]:>7.2f} "
                f"{stock['strategy'].upper():<12} "
                f"{stock['indicators']['momentum']:>8.1f}% "
                f"{stock['indicators']['volatility']:>9.2f}% "
                f"{stock['score']:>5.1f} "
                f"{', '.join(stock['patterns'])}"
            )

    def _log_rejection(self, ticker: str, reason: str):
        """Track why stocks were rejected"""
        self.rejection_stats[reason] += 1
        self.debugger.debug(f"Rejected {ticker}: {reason}")

    def print_rejection_summary(self):
        """Show rejection reasons statistics"""
        if not self.rejection_stats:
            return
            
        self.debugger.info("\nRejection Reasons:")
        self.debugger.info("-----------------")
        for reason, count in sorted(self.rejection_stats.items(), key=lambda x: -x[1]):
            self.debugger.info(f"{count:>5} | {reason}")


class FixedHypergrowthStrategy(Strategy):
    """Hypergrowth strategy that assumes stock already qualified"""
    def init(self):
        self.sma20 = self.I(talib.SMA, self.data.Close, 20)
        self.sma50 = self.I(talib.SMA, self.data.Close, 50)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.rsi = self.I(talib.RSI, self.data.Close, 14)
        
    def next(self):
        price = self.data.Close[-1]
        atr = self.atr[-1]
        
        # Entry - we already know this is a hypergrowth stock
        if not self.position:
            self.buy(sl=price - 2*atr)
        
        # Exit conditions only
        if self.position and any([
            price < self.sma20[-1],
            self.rsi[-1] < 45,
            self.position.pl > 0.20,  # 20% profit
        ]):
            self.position.close()


class FixedMomentumStrategy(Strategy):
    """Momentum strategy that assumes stock already qualified"""
    def init(self):
        self.sma50 = self.I(talib.SMA, self.data.Close, 50)
        self.sma200 = self.I(talib.SMA, self.data.Close, 200)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.rsi = self.I(talib.RSI, self.data.Close, 14)
        
    def next(self):
        price = self.data.Close[-1]
        atr = self.atr[-1]
        
        # Entry - we already know this is a momentum stock
        if not self.position:
            self.buy(sl=price - 2.5*atr)
        
        # Exit conditions only
        if self.position and any([
            price < self.sma50[-1],
            self.rsi[-1] > 70,
            self.position.pl < -0.05  # 5% loss
        ]):
            self.position.close()


class FixedBreakoutStrategy(Strategy):
    """Breakout strategy that assumes stock already qualified"""
    def init(self):
        self.support = self.I(lambda x: x.Low.rolling(20).min(), self.data)
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.rsi = self.I(talib.RSI, self.data.Close, 14)

    def next(self):
        current_close = self.data.Close[-1]
        
        # Entry - we already know this is a breakout stock
        if not self.position:
            stop_loss = min(
                self.support[-1],
                current_close - 2 * self.atr[-1]
            )
            self.buy(sl=stop_loss)
        
        # Exit conditions only
        if self.position and (current_close < self.support[-1] or self.rsi[-1] > 70):
            self.position.close()
            
        if self.position and current_close > self.position.entry_price * 1.05:
            self.position.sl = max(
                self.position.sl or 0,
                current_close - 1.5 * self.atr[-1]
            )


class YahooFinanceFallback:
    """Fallback to Yahoo Finance for backtesting when Polygon data is insufficient"""
    
    def __init__(self, debugger: Debugger):
        self.debugger = debugger
        self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"
        
    async def get_historical_data(self, ticker: str, days: int = 400) -> Optional[pd.DataFrame]:
        """Get historical data from Yahoo Finance"""
        try:
            end_date = int(datetime.now().timestamp())
            start_date = int((datetime.now() - timedelta(days=days)).timestamp())
            
            url = f"{self.base_url}{ticker}"
            params = {
                "period1": start_date,
                "period2": end_date,
                "interval": "1d",
                "events": "history"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 404:
                        self.debugger.debug(f"Ticker {ticker} not found on Yahoo Finance")
                        return None
                    response.raise_for_status()
                    data = await response.json()
                    
                if not data.get('chart', {}).get('result'):
                    return None
                    
                quotes = data['chart']['result'][0]['indicators']['quote'][0]
                timestamps = data['chart']['result'][0]['timestamp']
                
                df = pd.DataFrame({
                    'Open': quotes['open'],
                    'High': quotes['high'],
                    'Low': quotes['low'],
                    'Close': quotes['close'],
                    'Volume': quotes['volume']
                }, index=pd.to_datetime(timestamps, unit='s'))
                
                # Clean and validate data
                df = df[~df.index.duplicated(keep='first')]
                df = df.dropna()
                df = df.asfreq('D').ffill()
                
                return df
                
        except Exception as e:
            self.debugger.error(f"Yahoo Finance error for {ticker}: {str(e)}")
            return None


async def run_backtest(stock_data: Dict, debugger: Debugger) -> Optional[Dict]:
    """Enhanced backtest function with Yahoo Finance fallback and robust error handling"""
    try:
        # 1. Validate input structure
        if not stock_data or 'ticker' not in stock_data:
            debugger.error("Invalid stock_data - missing ticker")
            return None

        ticker = stock_data['ticker']
        strategy_name = stock_data.get('strategy', 'breakout').lower()
        
        # 2. Try to get data from Polygon first
        polygon_data = stock_data.get('data')
        
        # 3. If Polygon data is insufficient, try Yahoo Finance fallback
        if polygon_data is None or len(polygon_data) < 20:
            debugger.info(f"Insufficient Polygon data for {ticker}, trying Yahoo Finance fallback...")
            async with YahooFinanceFallback(debugger) as yahoo:
                yahoo_data = await yahoo.get_historical_data(ticker)
                if yahoo_data is not None and len(yahoo_data) >= 20:
                    polygon_data = yahoo_data
                    data_source = "Yahoo Finance"
                else:
                    debugger.error(f"Insufficient data from both Polygon and Yahoo for {ticker}")
                    return None
        else:
            data_source = "Polygon"

        # 4. Prepare the DataFrame
        try:
            df = polygon_data.copy()
            
            # Convert all numeric columns
            numeric_cols = ['o', 'h', 'l', 'c', 'v']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Rename columns to match Backtesting.py expectations
            column_map = {
                'o': 'Open',
                'h': 'High',
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume'
            }
            df = df.rename(columns=column_map)
            
            # Drop any rows with missing essential values
            df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
            
            # Ensure Volume has reasonable values
            if 'Volume' in df.columns:
                df['Volume'] = df['Volume'].fillna(0).clip(lower=0)
            
            # Convert index to datetime if it isn't already
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors='coerce')
                df = df[df.index.notna()]  # Remove rows with invalid dates
            
            # Sort by date ascending
            df = df.sort_index()
            
            # Verify we have sufficient data
            if len(df) < 20:
                debugger.warning(f"Insufficient data points ({len(df)}) for {ticker}")
                return None
                
        except Exception as e:
            debugger.error(f"Data preparation failed for {ticker}: {str(e)}")
            return None

        # 5. Select the appropriate strategy class
        strategy_classes = {
            'hypergrowth': FixedHypergrowthStrategy,
            'momentum': FixedMomentumStrategy,
            'breakout': FixedBreakoutStrategy
        }
        
        if strategy_name not in strategy_classes:
            debugger.error(f"Unknown strategy: {strategy_name}")
            return None
            
        strategy_class = strategy_classes[strategy_name]

        # 6. Run the backtest with proper error handling
        try:
            bt = Backtest(
                df,
                strategy_class,
                commission=.001,
                margin=1.0,
                trade_on_close=True,
                exclusive_orders=True,
                cash=100000
            )
            
            # Run with default parameters
            results = bt.run()
            
            # Convert results to dict if needed
            if hasattr(results, '_asdict'):
                results = results._asdict()
            elif isinstance(results, pd.Series):
                results = results.to_dict()
            
            # Handle no-trades case
            if results.get('# Trades', 0) == 0:
                debugger.warning(f"No trades executed for {ticker} - strategy conditions not met")
                return None
                
            # Calculate additional metrics
            sharpe = results.get('Sharpe Ratio', 0)
            max_drawdown = results.get('Max. Drawdown [%]', 0)
            
            # Log comprehensive results
            debugger.info(
                f"Backtest results for {ticker} ({strategy_name}) using {data_source}:\n"
                f"Period: {df.index[0].date()} to {df.index[-1].date()}\n"
                f"Trades: {results.get('# Trades', 0)}\n"
                f"Return: {results.get('Return [%]', 0):.1f}%\n"
                f"Win Rate: {results.get('Win Rate [%]', 0):.1f}%\n"
                f"Sharpe Ratio: {sharpe:.2f}\n"
                f"Max Drawdown: {max_drawdown:.1f}%"
            )
            
            # Add additional metrics to results
            results.update({
                'data_source': data_source,
                'start_date': df.index[0].date(),
                'end_date': df.index[-1].date(),
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown
            })
            
            return results
            
        except Exception as e:
            debugger.error(f"Backtest execution failed for {ticker}: {str(e)}")
            return None
            
    except Exception as e:
        debugger.error(f"Backtest setup failed for {ticker}: {str(e)}")
        return None


async def optimize_hypergrowth_params(df: pd.DataFrame):
    """Grid search for optimal hypergrowth parameters"""
    best_score = -np.inf
    best_params = None
    
    param_grid = {
        'price_growth_30d': [0.10, 0.12, 0.15],
        'volume_spike': [1.5, 2.0, 2.5],
        'rsi_range': [(60,80), (65,85)]
    }
    
    for params in ParameterGrid(param_grid):
        temp_df = df.copy()
        temp_df = temp_df[
            (temp_df['30d_return'] > params['price_growth_30d']) &
            (temp_df['volume'] > params['volume_spike'] * temp_df['avg_volume_10d']) &
            (temp_df['rsi_5'].between(*params['rsi_range']))
        ]
        
        score = (
            0.5 * temp_df['30d_return'].mean() +
            0.3 * (temp_df['volume'] / temp_df['avg_volume_10d']).mean() -
            0.2 * temp_df['atr_14'].std()
        )
        
        if score > best_score:
            best_score = score
            best_params = params
    
    return best_params, best_score


async def main():
    POLYGON_API_KEY = "OZzn0oK0H2yG6rpIvVhGfgXgnUTrL31z"
    
    # Initialize debugger with logging disabled
    debugger = Debugger(enabled=True)
    # Or alternatively, only show warnings and errors:
    # debugger = Debugger(enabled=True, level="WARNING")
    
    async with AsyncPolygonIOClient(POLYGON_API_KEY, debugger) as client:
        scanner = StockScanner(client, debugger, max_tickers_to_scan=100)
        
        if not await scanner.initialize():
            debugger.error("Scanner initialization failed")
            return
            
        regime = await scanner.regime_detector.get_regime_description()
        debugger.info(f"\nCurrent Market Regime:\n{json.dumps(regime, indent=2)}")
        
        await scanner.scan_tickers()
        
        if scanner.scan_results:
            with open('scan_results.json', 'w') as f:
                json.dump([{
                    'ticker': r['ticker'],
                    'strategy': r['strategy'],
                    'score': r['score']
                } for r in scanner.scan_results], f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())