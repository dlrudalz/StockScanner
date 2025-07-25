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
        self.enabled = enabled
        self.level = self._validate_level(level.upper())
        self.log_file = log_file
        self._setup_logging()
        
    def _validate_level(self, level: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if level not in valid_levels:
            raise ValueError(f"Invalid log level '{level}'. Must be one of: {', '.join(valid_levels)}")
        return level
        
    def _setup_logging(self):
        if not self.enabled:
            logging.getLogger().addHandler(logging.NullHandler())
            self.logger = logging.getLogger(__name__)
            self.logger.disabled = True
            return
            
        log_level = getattr(logging, self.level, logging.INFO)
        logger = logging.getLogger(__name__)
        
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            
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
        if not self.enabled:
            return False
            
        message_level = getattr(logging, level.upper(), logging.INFO)
        min_level = getattr(logging, self.level, logging.INFO)
        return message_level >= min_level
        
    def log(self, message: str, level: str = "INFO"):
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
        if self._should_log("ERROR"):
            self.logger.exception(message)
            
    def log_dataframe(self, df: pd.DataFrame, message: str = "", level: str = "DEBUG"):
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
        try:
            ftp = ftplib.FTP(self.FTP_SERVER)
            ftp.login()
            ftp.cwd(self.FTP_DIR)
            return ftp
        except Exception as e:
            self.debugger.error(f"FTP connection error: {e}")
            raise
    
    def get_tickers_from_ftp(self) -> List[str]:
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
            
            for line in nasdaq_data[1:-1]:
                parts = line.split('|')
                if len(parts) > 0 and parts[0]:
                    ticker = parts[0].replace('$', '').replace('.', '')
                    if self._is_valid_ticker(ticker):
                        tickers.add(ticker)
            
            for line in traded_data[1:-1]:
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
        if scanner or self.scanner:
            return (scanner or self.scanner)._is_valid_ticker(ticker)
        
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
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return self
        
    async def __aexit__(self, exc_type, exc, tb):
        await self.session.close()

    async def get_aggregates(self, ticker: str, days: int = 400, 
                        timespan: str = "day") -> Optional[pd.DataFrame]:
        cache_key = f"{ticker}_{days}_{timespan}"
        if cache_key in self.cache:
            return self.cache[cache_key].copy()

        try:
            end_date = date.today() - timedelta(days=1)
            start_date = end_date - timedelta(days=int(days * 1.2))

            self.debugger.debug(f"Fetching {ticker} {timespan} data from {start_date} to {end_date}")

            endpoint = f"/v2/aggs/ticker/{ticker}/range/1/{timespan}/{start_date}/{end_date}"
            params = {
                "adjusted": "true",
                "apiKey": self.api_key,
                "sort": "asc",
                "limit": 50000
            }

            async with self.semaphore:
                await self._throttle()
                async with async_timeout.timeout(45):
                    async with self.session.get(f"{self.base_url}{endpoint}", params=params) as response:
                        if response.status == 429:
                            await asyncio.sleep(15)
                            return await self.get_aggregates(ticker, days, timespan)
                        response.raise_for_status()
                        data = await response.json()

            if not data.get("results"):
                self.debugger.debug(f"No results for {ticker}")
                return None
                
            df = pd.DataFrame(data["results"])
            df["date"] = pd.to_datetime(df["t"], unit="ms")
            df = df.set_index("date")
            
            df = df[~df.index.duplicated(keep='first')]
            df = df.asfreq('D').ffill()
            
            self.cache[cache_key] = df.copy()
            return df

        except Exception as e:
            self.debugger.error(f"Failed to fetch {ticker}: {str(e)}")
            return None

    async def get_all_tickers(self, market: str = "stocks") -> List[Dict]:
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
        if self.daily_data is None or len(self.daily_data) < 5:
            return {
                'potential_transition': False,
                'trend_weakening': False,
                'vol_increasing': False
            }
        
        recent_daily = self.daily_data.iloc[-5:]
        
        trend_weakening = (
            (recent_daily['trend_strength'].pct_change(fill_method=None).mean() < -0.05) or
            (recent_daily['rsi_14'].iloc[-1] < 30 and recent_daily['rsi_14'].iloc[-1] < recent_daily['rsi_14'].iloc[-5])
        )
        
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
    """Complete stock scanner with multi-strategy support and strategy-specific scanning"""
    
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
                    "min_volume": 1_000_000,
                    "min_price": 10.00,
                    "max_price": 500.00,
                    "min_market_cap": 300_000_000,
                    "min_revenue_growth": 0.25,
                    "min_volume_ratio": 1.5,
                    "min_price_relative": 1.1,
                    "min_momentum": 0.10,
                    "max_volatility": 0.40,
                    "momentum_days": 30,
                    "min_relative_strength": 1.1,
                    "days_to_scan": 90,
                    "pattern_priority": ["breakout", "golden_cross", "bullish_ma_stack"]
                },
                "backtest_class": FixedHypergrowthStrategy
            },
            "momentum": {
                "allocation": (0.3, 0.4),
                "holding_period": (90, 365),
                "stop_loss": 0.25,
                "profit_targets": [(0.5, 0.5), (1.0, 0.5)],
                "filters": {
                    "min_volume": 750_000,
                    "min_price": 5.00,
                    "max_price": 500.00,
                    "min_market_cap": 300_000_000,
                    "ma_condition": "50ma > 200ma",
                    "min_adx": 25,
                    "min_trend_duration": 60,
                    "max_volatility": 0.30,
                    "momentum_days": 30,
                    "min_relative_strength": 1.1,
                    "days_to_scan": 90,
                    "pattern_priority": ["breakout", "golden_cross", "bullish_ma_stack"]
                },
                "backtest_class": FixedMomentumStrategy
            },
            "breakout": {
                "allocation": (0.2, 0.3),
                "holding_period": (7, 21),
                "stop_loss": 0.08,
                "profit_targets": [(0.15, 0.5), (0.25, 0.5)],
                "filters": {
                    "min_volume": 1_500_000,
                    "min_price": 5.00,
                    "max_price": 300.00,
                    "min_market_cap": 300_000_000,
                    "min_volume_ratio": 2.5,
                    "min_volatility": 0.05,
                    "consolidation_days": 10,
                    "max_consolidation": 0.10,
                    "momentum_days": 30,
                    "min_relative_strength": 1.1,
                    "days_to_scan": 90,
                    "pattern_priority": ["breakout", "golden_cross", "bullish_ma_stack"]
                },
                "backtest_class": FixedBreakoutStrategy
            }
        }

    async def initialize(self):
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
        if not ticker or len(ticker) > 6:
            return False
            
        if not re.match(r'^[A-Z]+$', ticker):
            return False
            
        if any(x in ticker for x in ['2X', '3X', '1X', '2', '3']):
            return False
            
        non_stock_suffixes = ('W', 'R', 'P', 'Q', 'Y', 'F', 'V', 'J', 'M', 'Z')
        if ticker.endswith(non_stock_suffixes):
            return False
            
        if re.match(r'^[A-Z]+[23][XL]$', ticker):
            return False
            
        return True

    async def _load_tickers_to_scan(self) -> List[str]:
        try:
            limit_msg = "all tickers" if self.max_tickers_to_scan is None else f"max {self.max_tickers_to_scan} tickers"
            self.debugger.debug(f"Loading tickers ({limit_msg})")

            polygon_tickers = await self.polygon.get_all_tickers()
            polygon_symbols = [t["ticker"] for t in polygon_tickers]
            if self.max_tickers_to_scan is not None:
                polygon_symbols = polygon_symbols[:self.max_tickers_to_scan]

            ftp_tickers = self.ftp_client.get_tickers_from_ftp()
            if self.max_tickers_to_scan is not None:
                ftp_tickers = ftp_tickers[:self.max_tickers_to_scan]

            all_tickers = list(set(polygon_symbols + ftp_tickers))
            if self.max_tickers_to_scan is not None:
                all_tickers = all_tickers[:self.max_tickers_to_scan]

            valid_tickers = [t for t in all_tickers if self._is_valid_ticker(t)]
            random.shuffle(valid_tickers)

            if self.max_tickers_to_scan is not None:
                valid_tickers = valid_tickers[:self.max_tickers_to_scan]

            return valid_tickers

        except Exception as e:
            self.debugger.error(f"Error loading tickers: {str(e)}")
            return []
        
    async def scan_tickers(self):
        if not await self.initialize():
            return

        # Step 1: Lightweight prefilter (no scoring)
        self.debugger.info("Running prefilter scan...")
        self.prefiltered_tickers = await self._run_prefilter_scan()  # Store as instance variable
        
        if not self.prefiltered_tickers:
            self.debugger.warning("No stocks passed prefilter")
            return

        # Step 2: Run all strategies on prefiltered stocks
        scan_tasks = [
            self._scan_for_strategy("hypergrowth", self.prefiltered_tickers),
            self._scan_for_strategy("momentum", self.prefiltered_tickers),
            self._scan_for_strategy("breakout", self.prefiltered_tickers)
        ]
        await asyncio.gather(*scan_tasks)
        
        self._balance_strategy_allocations()
        
        if self.scan_results:
            self.print_summary()
            await self._backtest_top_candidates()

    async def _run_prefilter_scan(self) -> List[str]:
        """Ultra-light prefilter that only removes completely untradeable stocks"""
        prefilter_criteria = {
            "min_days_data": 5,        # Need at least some history
            "min_avg_volume": 50_000,  # Absolute minimum liquidity
            "max_price": 10_000,       # Only filter out extremely high-priced stocks
            "days_to_scan": 30         # Short lookback period
        }
        
        prefiltered = []
        batch_size = 50
        
        with tqdm(total=len(self.tickers_to_scan), desc="Prefiltering") as pbar:
            for i in range(0, len(self.tickers_to_scan), batch_size):
                batch = self.tickers_to_scan[i:i + batch_size]
                tasks = [self._check_prefilter_criteria(ticker, prefilter_criteria) 
                        for ticker in batch]
                results = await asyncio.gather(*tasks)
                prefiltered.extend([ticker for ticker, passes in zip(batch, results) if passes])
                pbar.update(len(batch))
        
        return prefiltered

    async def _check_prefilter_criteria(self, ticker: str, criteria: Dict) -> bool:
        """Basic yes/no check - only filters out completely invalid stocks"""
        try:
            price_data = await self.polygon.get_aggregates(ticker, days=criteria["days_to_scan"])
            
            # Absolute minimum checks
            if (price_data is None or 
                len(price_data) < criteria["min_days_data"] or 
                price_data['v'].mean() < criteria["min_avg_volume"] or
                price_data['c'].iloc[-1] > criteria["max_price"]):
                return False
                
            return True
        except Exception:
            return False

    async def _scan_for_strategy(self, strategy_name, prefiltered_tickers):
        criteria = self.strategy_config[strategy_name]["filters"]
        self.debugger.info(f"Scanning for {strategy_name} with criteria: {criteria}")
        
        strategy_results = []
        batch_size = 50
        
        with tqdm(total=len(prefiltered_tickers), desc=f"Scanning {strategy_name}") as pbar:
            for i in range(0, len(prefiltered_tickers), batch_size):
                batch = prefiltered_tickers[i:i + batch_size]
                tasks = [self._scan_single_ticker(ticker, criteria, strategy_name) 
                        for ticker in batch]
                batch_results = await asyncio.gather(*tasks)
                
                for result in batch_results:
                    if result is not None:
                        result['strategy'] = strategy_name
                        result['strategy_score'] = self._score_strategy(result, strategy_name)
                        strategy_results.append(result)
                
                pbar.update(len(batch))
                await asyncio.sleep(1)
        
        self.scan_results.extend(strategy_results)

    def _get_strategy_criteria(self, strategy: str, base_criteria: Dict) -> Dict:
        strategy_criteria = base_criteria.copy()
        strategy_criteria.update(self.strategy_config[strategy]["filters"])
        
        if strategy == "hypergrowth":
            strategy_criteria["min_momentum"] = max(
                strategy_criteria["min_momentum"],
                self.strategy_config["hypergrowth"]["filters"]["min_momentum"]
            )
        elif strategy == "momentum":
            strategy_criteria["min_trend_duration"] = max(
                strategy_criteria.get("min_trend_duration", 0),
                self.strategy_config["momentum"]["filters"]["min_trend_duration"]
            )
        elif strategy == "breakout":
            strategy_criteria["min_volume_ratio"] = max(
                strategy_criteria.get("min_volume_ratio", 0),
                self.strategy_config["breakout"]["filters"]["min_volume_ratio"]
            )
        
        return strategy_criteria

    async def _scan_single_ticker(self, ticker: str, criteria: Dict, strategy: str) -> Optional[Dict]:
        try:
            price_data = await self.polygon.get_aggregates(ticker, days=criteria["days_to_scan"])
            if price_data is None or len(price_data) < 20:
                self._log_rejection(ticker, "insufficient data")
                return None
                
            details = await self.polygon.get_ticker_details(ticker)
            
            indicators = self._calculate_indicators(price_data)
            patterns = self._detect_patterns(price_data)
            
            if not self._validate_ticker_for_strategy(price_data, indicators, criteria, strategy):
                return None
                
            score = self._calculate_strategy_score(price_data, indicators, patterns, details, strategy)
            
            return {
                "ticker": ticker,
                "data": price_data,
                "indicators": indicators,
                "patterns": patterns,
                "details": details,
                "score": score
            }
            
        except Exception as e:
            self._log_rejection(ticker, f"scan error: {str(e)}")
            return None

    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
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
        patterns = []
        o, h, l, c = data['o'].values, data['h'].values, data['l'].values, data['c'].values
        
        if talib.CDLHAMMER(o, h, l, c)[-1] > 0: patterns.append("hammer")
        if talib.CDLENGULFING(o, h, l, c)[-1] > 0: patterns.append("bullish_engulfing")
        
        resistance = np.max(h[-20:-1])
        support = np.min(l[-20:-1])
        if c[-1] > resistance: patterns.append("breakout")
        if c[-1] < support: patterns.append("breakdown")
        
        sma50 = talib.SMA(c, 50)
        sma200 = talib.SMA(c, 200)
        if sma50[-1] > sma200[-1] and sma50[-2] <= sma200[-2]: patterns.append("golden_cross")
        
        return patterns

    def _validate_ticker_for_strategy(self, data: pd.DataFrame, indicators: Dict, 
                                    criteria: Dict, strategy: str) -> bool:
        price = data['c'].iloc[-1]
        volume = data['v'].iloc[-1]
        
        if not (criteria["min_price"] <= price <= criteria["max_price"]):
            return False
        if volume < criteria["min_volume"]:
            return False
            
        if strategy == "hypergrowth":
            if indicators['momentum'] < criteria.get("min_momentum", 0):
                return False
            if indicators['volatility'] > criteria.get("max_volatility", float('inf')):
                return False
                
        elif strategy == "momentum":
            if indicators['adx'] < criteria.get("min_adx", 0):
                return False
            if len(data) < criteria.get("min_trend_duration", 0):
                return False
                
        elif strategy == "breakout":
            volume_ratio = volume / indicators['volume_ma']
            if volume_ratio < criteria.get("min_volume_ratio", 0):
                return False
            if indicators['volatility'] < criteria.get("min_volatility", 0):
                return False
                
        return True

    def _calculate_strategy_score(self, data: pd.DataFrame, indicators: Dict, 
                                patterns: List[str], details: Dict, strategy: str) -> float:
        if strategy == "hypergrowth":
            return self._calculate_hypergrowth_score(data, indicators, details)
        elif strategy == "momentum":
            return self._calculate_momentum_score(data, indicators)
        elif strategy == "breakout":
            return self._calculate_breakout_score(data, indicators, patterns)
        return 0.0

    def _calculate_hypergrowth_score(self, data: pd.DataFrame, indicators: Dict, details: Dict) -> float:
        weights = {
            'fundamentals': 0.4,
            'momentum': 0.3,
            'volume': 0.2,
            'volatility': 0.1
        }
        
        score = 0
        
        if details:
            revenue_growth = details.get('revenue_growth_rate', 0)
            if revenue_growth > self.strategy_config["hypergrowth"]["filters"]["min_revenue_growth"]:
                score += weights['fundamentals'] * min(revenue_growth / 0.5, 1)
            elif 'revenue_growth_rate' not in details:
                score += weights['fundamentals'] * 0.5
        
        price_growth = indicators['momentum']
        min_momentum = self.strategy_config["hypergrowth"]["filters"]["min_momentum"]
        if price_growth > min_momentum:
            score += weights['momentum'] * min(price_growth/(min_momentum*3), 1)
            
        volume_ratio = data['v'].iloc[-1] / indicators['volume_ma']
        min_volume_ratio = self.strategy_config["hypergrowth"]["filters"]["min_volume_ratio"]
        if volume_ratio > min_volume_ratio:
            score += weights['volume'] * min(volume_ratio/(min_volume_ratio*2), 1)
            
        max_vol = self.strategy_config["hypergrowth"]["filters"]["max_volatility"]
        vol_penalty = max(0, (indicators['volatility'] - max_vol/2) / (max_vol/2))
        score += weights['volatility'] * (1 - min(vol_penalty, 1))
        
        return 100 * min(score, 1.0)

    def _calculate_momentum_score(self, data: pd.DataFrame, indicators: Dict) -> float:
        weights = {
            'trend_strength': 0.5,
            'duration': 0.2,
            'volatility': 0.15,
            'volume': 0.15
        }
        
        score = 0
        
        min_adx = self.strategy_config["momentum"]["filters"]["min_adx"]
        adx_score = min(indicators['adx'] / min_adx, 1.5)
        score += weights['trend_strength'] * min(adx_score, 1)
        
        min_duration = self.strategy_config["momentum"]["filters"]["min_trend_duration"]
        duration_score = min(len(data) / min_duration, 2)
        score += weights['duration'] * min(duration_score, 1)
        
        if indicators['sma_50'] > indicators['sma_200']:
            score += 0.1
            
        volume_ratio = data['v'].iloc[-1] / indicators['volume_ma']
        score += weights['volume'] * min(volume_ratio / 2, 1)
        
        max_vol = self.strategy_config["momentum"]["filters"]["max_volatility"]
        vol_score = 1 - min(indicators['volatility'] / max_vol, 1)
        score += weights['volatility'] * vol_score
        
        return 100 * min(score, 1.0)

    def _calculate_breakout_score(self, data: pd.DataFrame, indicators: Dict, patterns: List[str]) -> float:
        weights = {
            'volume_ratio': 0.4,
            'pattern_score': 0.3,
            'volatility': 0.2,
            'consolidation': 0.1
        }
        
        score = 0
        
        volume_ratio = data['v'].iloc[-1] / indicators['volume_ma']
        min_ratio = self.strategy_config["breakout"]["filters"]["min_volume_ratio"]
        score += weights['volume_ratio'] * min(volume_ratio / min_ratio, 2)
        
        pattern_bonus = 0
        if "breakout" in patterns:
            pattern_bonus += 0.5
        if any(p in patterns for p in ["bullish_engulfing", "hammer"]):
            pattern_bonus += 0.3
        score += weights['pattern_score'] * min(pattern_bonus, 1)
        
        min_vol = self.strategy_config["breakout"]["filters"]["min_volatility"]
        vol_score = min(indicators['volatility'] / (min_vol * 2), 1)
        score += weights['volatility'] * vol_score
        
        recent_high = data['h'].iloc[-20:].max()
        recent_low = data['l'].iloc[-20:].min()
        consolidation_range = (recent_high - recent_low) / recent_low
        max_consolidation = self.strategy_config["breakout"]["filters"]["max_consolidation"]
        consolidation_score = 1 - min(consolidation_range / max_consolidation, 1)
        score += weights['consolidation'] * consolidation_score
        
        return 100 * min(score, 1.0)

    def _score_strategy(self, stock: Dict, strategy: str) -> float:
        if strategy == 'hypergrowth':
            return self._calculate_hypergrowth_score(stock['data'], stock['indicators'], stock['details'])
        elif strategy == 'momentum':
            return self._calculate_momentum_score(stock['data'], stock['indicators'])
        elif strategy == 'breakout':
            return self._calculate_breakout_score(stock['data'], stock['indicators'], stock['patterns'])
        return 0.0
    
    def _balance_strategy_allocations(self):
        if not self.scan_results:
            return
            
        target_allocations = {
            'hypergrowth': 0.45,
            'momentum': 0.45,
            'breakout': 0.10
        }
        
        total_needed = min(len(self.scan_results), 50)
        
        grouped = defaultdict(list)
        for stock in self.scan_results:
            grouped[stock['strategy']].append(stock)
            
        for strategy in grouped:
            grouped[strategy].sort(key=lambda x: x['strategy_score'], reverse=True)
        
        targets = {s: int(total_needed * target_allocations[s]) for s in target_allocations}
        
        final_results = []
        for strategy in targets:
            final_results.extend(grouped[strategy][:targets[strategy]])
        
        self.scan_results = final_results

    async def _backtest_top_candidates(self, n_per_strategy: int = 3):
        if not self.scan_results:
            return
            
        by_strategy = defaultdict(list)
        for stock in self.scan_results:
            by_strategy[stock['strategy']].append(stock)
            
        for strategy, stocks in by_strategy.items():
            top_stocks = sorted(stocks, key=lambda x: x['strategy_score'], reverse=True)[:n_per_strategy]
            for stock in top_stocks:
                self.debugger.info(f"\nBacktesting {stock['ticker']} ({strategy})...")
                await run_backtest(stock, self.debugger)

    def generate_trade_plan(self, ticker: str) -> Optional[Dict]:
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
        account_size = 100000
        risk_per_trade = 0.01
        risk_amount = account_size * risk_per_trade
        risk_per_share = stock['data']['c'].iloc[-1] * config['stop_loss']
        shares = risk_amount / risk_per_share
        
        max_allocation = config['allocation'][1] * account_size
        position_value = shares * stock['data']['c'].iloc[-1]
        
        if position_value > max_allocation:
            shares = max_allocation / stock['data']['c'].iloc[-1]
            
        return round(shares)

    def _calculate_profit_targets(self, entry_price: float, config: Dict) -> List[Dict]:
        return [
            {
                "price": round(entry_price * (1 + target[0]), 2),
                "size": target[1]
            } 
            for target in config['profit_targets']
        ]

    def _calculate_risk_reward(self, entry_price: float, config: Dict) -> float:
        avg_return = sum(t[0] * t[1] for t in config['profit_targets'])
        return round(avg_return / config['stop_loss'], 2)

    def _get_monitoring_rules(self, strategy: str) -> List[str]:
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
        if not self.scan_results:
            self.debugger.info("No qualifying stocks found")
            return
            
        self.debugger.info("\nStrategy Allocation:")
        self.debugger.info("-------------------")
        counts = defaultdict(int)
        for stock in self.scan_results:
            counts[stock['strategy']] += 1
            
        for strategy, config in self.strategy_config.items():
            count = counts.get(strategy, 0)
            pct = (count / len(self.scan_results)) * 100
            self.debugger.info(f"{strategy.upper():<12}: {count:>3} stocks ({pct:.1f}%)")
        
        for strategy in self.strategy_config.keys():
            strategy_stocks = [s for s in self.scan_results if s['strategy'] == strategy]
            if not strategy_stocks:
                continue
                
            self.debugger.info(f"\nTop {strategy.upper()} Candidates:")
            self.debugger.info("----------------------")
            headers = ["Ticker", "Price", "Score", "Momentum", "Volatility", "Patterns"]
            self.debugger.info("{:<6} {:<8} {:<6} {:<9} {:<10} {}".format(*headers))
            
            for stock in sorted(strategy_stocks, key=lambda x: x['strategy_score'], reverse=True)[:10]:
                self.debugger.info(
                    f"{stock['ticker']:<6} "
                    f"${stock['data']['c'].iloc[-1]:>7.2f} "
                    f"{stock['strategy_score']:>5.1f} "
                    f"{stock['indicators']['momentum']:>8.1f}% "
                    f"{stock['indicators']['volatility']:>9.2f}% "
                    f"{', '.join(stock['patterns'])}"
                )

    def _log_rejection(self, ticker: str, reason: str):
        self.rejection_stats[reason] += 1
        self.debugger.debug(f"Rejected {ticker}: {reason}")

    def print_rejection_summary(self):
        if not self.rejection_stats:
            return
            
        self.debugger.info("\nRejection Reasons:")
        self.debugger.info("-----------------")
        for reason, count in sorted(self.rejection_stats.items(), key=lambda x: -x[1]):
            self.debugger.info(f"{count:>5} | {reason}")

class FixedHypergrowthStrategy(Strategy):
    def init(self):
        self.sma20 = self.I(talib.SMA, self.data.Close, 20)
        self.sma50 = self.I(talib.SMA, self.data.Close, 50)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.rsi = self.I(talib.RSI, self.data.Close, 14)
        
    def next(self):
        price = self.data.Close[-1]
        atr = self.atr[-1]
        
        if not self.position:
            self.buy(sl=price - 2*atr)
        
        if self.position and any([
            price < self.sma20[-1],
            self.rsi[-1] < 45,
            self.position.pl > 0.20,
        ]):
            self.position.close()

class FixedMomentumStrategy(Strategy):
    def init(self):
        self.sma50 = self.I(talib.SMA, self.data.Close, 50)
        self.sma200 = self.I(talib.SMA, self.data.Close, 200)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.rsi = self.I(talib.RSI, self.data.Close, 14)
        
    def next(self):
        price = self.data.Close[-1]
        atr = self.atr[-1]
        
        if not self.position:
            self.buy(sl=price - 2.5*atr)
        
        if self.position and any([
            price < self.sma50[-1],
            self.rsi[-1] > 70,
            self.position.pl < -0.05
        ]):
            self.position.close()

class FixedBreakoutStrategy(Strategy):
    def init(self):
        # Initialize variables for manual support calculation
        self.low_window = []
        self.window_size = 20
        
        # Standard indicators
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.rsi = self.I(talib.RSI, self.data.Close, 14)
        
        # Track entry price and stop loss manually
        self.entry_price = None
        self.current_sl = None

    def next(self):
        # Manual rolling minimum calculation
        self.low_window.append(self.data.Low[-1])
        if len(self.low_window) > self.window_size:
            self.low_window.pop(0)
        
        current_support = min(self.low_window) if len(self.low_window) == self.window_size else None
        
        # Wait until we have enough data
        if current_support is None:
            return
            
        current_close = self.data.Close[-1]
        current_low = self.data.Low[-1]
        
        if not self.position:
            # Calculate initial stop loss
            stop_loss = min(
                max(current_support, current_low * 0.95),
                current_close - 2 * self.atr[-1]
            )
            
            # Entry condition
            if current_close > current_support * 1.05:
                self.entry_price = current_close
                self.current_sl = stop_loss
                self.buy(sl=stop_loss)
        else:
            # Exit conditions
            if (current_close < current_support or 
                self.rsi[-1] > 70 or
                current_close < self.entry_price * 0.95 or
                current_close <= self.current_sl):
                
                self.position.close()
                self.entry_price = None
                self.current_sl = None
            
            # Trailing stop logic
            elif current_close > self.entry_price * 1.05:
                new_sl = current_close - 1.5 * self.atr[-1]
                if new_sl > self.current_sl:
                    self.current_sl = new_sl
                    self.position.sl = new_sl

class YahooFinanceFallback:
    def __init__(self, debugger: Debugger):
        self.debugger = debugger
        self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"
        
    async def get_historical_data(self, ticker: str, days: int = 400) -> Optional[pd.DataFrame]:
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
                
                df = df[~df.index.duplicated(keep='first')]
                df = df.dropna()
                df = df.asfreq('D').ffill()
                
                return df
                
        except Exception as e:
            self.debugger.error(f"Yahoo Finance error for {ticker}: {str(e)}")
            return None

async def run_backtest(stock_data: Dict, debugger: Debugger) -> Optional[Dict]:
    try:
        if not stock_data or 'ticker' not in stock_data:
            debugger.error("Invalid stock_data - missing ticker")
            return None

        ticker = stock_data['ticker']
        strategy_name = stock_data.get('strategy', 'breakout').lower()
        
        polygon_data = stock_data.get('data')
        
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

        try:
            df = polygon_data.copy()
            
            numeric_cols = ['o', 'h', 'l', 'c', 'v']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            column_map = {
                'o': 'Open',
                'h': 'High',
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume'
            }
            df = df.rename(columns=column_map)
            
            df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
            
            if 'Volume' in df.columns:
                df['Volume'] = df['Volume'].fillna(0).clip(lower=0)
            
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors='coerce')
                df = df[df.index.notna()]
            
            df = df.sort_index()
            
            if len(df) < 20:
                debugger.warning(f"Insufficient data points ({len(df)}) for {ticker}")
                return None
                
        except Exception as e:
            debugger.error(f"Data preparation failed for {ticker}: {str(e)}")
            return None

        strategy_classes = {
            'hypergrowth': FixedHypergrowthStrategy,
            'momentum': FixedMomentumStrategy,
            'breakout': FixedBreakoutStrategy
        }
        
        if strategy_name not in strategy_classes:
            debugger.error(f"Unknown strategy: {strategy_name}")
            return None
            
        strategy_class = strategy_classes[strategy_name]

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
            
            results = bt.run()
            
            if hasattr(results, '_asdict'):
                results = results._asdict()
            elif isinstance(results, pd.Series):
                results = results.to_dict()
            
            if results.get('# Trades', 0) == 0:
                debugger.warning(f"No trades executed for {ticker} - strategy conditions not met")
                return None
                
            sharpe = results.get('Sharpe Ratio', 0)
            max_drawdown = results.get('Max. Drawdown [%]', 0)
            
            debugger.info(
                f"Backtest results for {ticker} ({strategy_name}) using {data_source}:\n"
                f"Period: {df.index[0].date()} to {df.index[-1].date()}\n"
                f"Trades: {results.get('# Trades', 0)}\n"
                f"Return: {results.get('Return [%]', 0):.1f}%\n"
                f"Win Rate: {results.get('Win Rate [%]', 0):.1f}%\n"
                f"Sharpe Ratio: {sharpe:.2f}\n"
                f"Max Drawdown: {max_drawdown:.1f}%"
            )
            
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
        scanner = StockScanner(client, debugger, max_tickers_to_scan=70)
        
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