import asyncio
import aiohttp
import async_timeout
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import logging
import talib
from typing import Literal, Tuple, Dict, Optional, List
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Define market regime types
TrendRegime = Literal["strong_bull", "weak_bull", "neutral", "weak_bear", "strong_bear"]
VolatilityRegime = Literal["low_vol", "medium_vol", "high_vol"]

class NASDAQTraderFTP:
    """Class to handle NASDAQ Trader FTP operations for ticker data"""
    FTP_SERVER = 'ftp.nasdaqtrader.com'
    FTP_DIR = 'SymbolDirectory'
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.cache_dir = 'nasdaq_ftp_cache'
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _connect_ftp(self) -> ftplib.FTP:
        """Connect to NASDAQ Trader FTP server"""
        try:
            ftp = ftplib.FTP(self.FTP_SERVER)
            ftp.login()  # Anonymous login
            ftp.cwd(self.FTP_DIR)
            return ftp
        except Exception as e:
            if self.debug:
                print(f"FTP connection error: {e}")
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
            print(f"Error getting tickers from FTP: {e}")
            return []
    
    def _is_valid_ticker(self, ticker: str) -> bool:
        """Validate ticker format and filter non-stock securities"""
        if not ticker or len(ticker) > 6:  # Increased to 6 to catch some edge cases
            return False
            
        # Skip tickers with numbers or special characters (except - for some ETFs)
        if not re.match(r'^[A-Z-]+$', ticker):
            return False
            
        # Skip leveraged/inverse ETFs (common patterns)
        if re.match(r'^[A-Z]+[23][XL]$', ticker):
            return False
            
        # Skip leveraged/inverse ETFs with - in name
        if '-' in ticker and (ticker.endswith(('X', 'L')) and any(c.isdigit() for c in ticker)):
            return False
            
        # Skip known non-stock suffixes
        non_stock_suffixes = ('W', 'R', 'P', 'Q', 'Y', 'F', 'V', 'J', 'M', 'Z')
        if ticker.endswith(non_stock_suffixes):
            return False
            
        # Skip when ticker starts with these (often special securities)
        non_stock_prefixes = ('X', 'Y', 'Z')
        if ticker.startswith(non_stock_prefixes) and len(ticker) > 1:
            return False
            
        return True

class AsyncPolygonIOClient:
    """Enhanced Polygon.io API client with caching and multi-timeframe support"""
    
    def __init__(self, api_key: str, verbose: bool = False):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.rate_limit_delay = 1.2
        self.last_request_time = 0
        self.session = None
        self.semaphore = asyncio.Semaphore(5)
        self.cache = {}
        self.verbose = verbose

    async def __aenter__(self):
        """Initialize async session"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
        
    async def __aexit__(self, exc_type, exc, tb):
        """Cleanup async session"""
        await self.session.close()

    async def get_aggregates(self, ticker: str, days: int = 300, 
                           timespan: str = "day") -> Optional[pd.DataFrame]:
        """Get historical price data with caching and multi-timeframe support"""
        cache_key = f"{ticker}_{days}_{timespan}"
        if cache_key in self.cache:
            return self.cache[cache_key].copy()

        try:
            # Request more days to account for weekends/holidays
            end_date = date.today() - timedelta(days=1)  # Yesterday
            start_date = end_date - timedelta(days=int(days * 1.5))

            if self.verbose:
                print(f"Fetching {ticker} {timespan} data from {start_date} to {end_date}")

            endpoint = f"/v2/aggs/ticker/{ticker}/range/1/{timespan}/{start_date}/{end_date}"
            params = {
                "adjusted": "true",
                "apiKey": self.api_key,
                "sort": "asc"  # Ensure chronological order
            }

            async with self.semaphore:
                await self._throttle()
                async with async_timeout.timeout(30):
                    async with self.session.get(f"{self.base_url}{endpoint}", params=params) as response:
                        if response.status == 429:
                            await asyncio.sleep(10)
                            return await self.get_aggregates(ticker, days, timespan)
                        response.raise_for_status()
                        data = await response.json()

            if not data.get("results"):
                if self.verbose:
                    print(f"No results for {ticker}")
                return None
                
            df = pd.DataFrame(data["results"])
            df["date"] = pd.to_datetime(df["t"], unit="ms")
            df = df.set_index("date")
            self.cache[cache_key] = df.copy()
            return df

        except Exception as e:
            if self.verbose:
                print(f"Failed to fetch {ticker}: {str(e)}")
            return None

    async def get_all_tickers(self, market: str = "stocks") -> List[Dict]:
        """Get all active tickers from Polygon with better filtering"""
        try:
            tickers = []
            endpoint = f"/v3/reference/tickers"
            params = {
                "market": market,
                "active": "false",  # Include inactive tickers
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

                # Filter for common stocks and ETFs, excluding special securities
                filtered = [
                    t for t in data["results"] 
                    if (t["type"] == "CS" or t["type"] == "ETF") and  # Common stock or ETF
                    not t["ticker"].endswith(('W', 'R', 'P', 'Q')) and  # Exclude warrants, rights, etc.
                    not any(c.isdigit() for c in t["ticker"]) and  # Exclude tickers with numbers
                    not re.match(r'^[A-Z]+[23][XL]$', t["ticker"]) and  # Exclude leveraged ETFs
                    t["primary_exchange"] in ["NASDAQ", "NYSE", "NYSEARCA"]  # Major US exchanges
                ]
                
                tickers.extend(filtered)
                
                if "next_url" in data:
                    endpoint = data["next_url"].replace(self.base_url, "")
                    params = {"apiKey": self.api_key}
                else:
                    break

            return tickers

        except Exception as e:
            print(f"Failed to fetch tickers: {str(e)}")
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
            if self.verbose:
                print(f"Failed to fetch details for {ticker}: {str(e)}")
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
    
    def __init__(self, polygon_client: AsyncPolygonIOClient, verbose: bool = False):
        self.polygon = polygon_client
        self.daily_data = None
        self.weekly_data = None
        self.volatility_regimes = ["low_vol", "medium_vol", "high_vol"]
        self.trend_regimes = ["strong_bull", "weak_bull", "neutral", "weak_bear", "strong_bear"]
        self.verbose = verbose
        
    async def initialize(self) -> bool:
        """Initialize with daily and weekly data"""
        try:
            if self.verbose:
                print("Initializing market regime detector...")
            
            # Get daily and weekly data
            self.daily_data = await self.polygon.get_aggregates("QQQ", days=252, timespan="day")
            self.weekly_data = await self.polygon.get_aggregates("QQQ", days=252, timespan="week")
            
            if self.daily_data is None or self.weekly_data is None:
                if self.verbose:
                    print("Failed to fetch QQQ data - API returned None")
                return False
                
            if len(self.daily_data) < 100 or len(self.weekly_data) < 20:
                if self.verbose:
                    print(f"Insufficient data points: Daily={len(self.daily_data)}, Weekly={len(self.weekly_data)}")
                return False
                
            if self.verbose:
                print(f"Loaded {len(self.daily_data)} daily and {len(self.weekly_data)} weekly data points")
            self._calculate_technical_indicators()
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"Error initializing detector: {str(e)}")
            return False
        
    def _calculate_technical_indicators(self):
        """Calculate advanced technical indicators"""
        # Daily indicators
        daily_closes = self.daily_data['c'].values
        daily_highs = self.daily_data['h'].values
        daily_lows = self.daily_data['l'].values
        daily_volumes = self.daily_data['v'].values
        
        # Basic indicators
        self.daily_data['sma_50'] = talib.SMA(daily_closes, timeperiod=50)
        self.daily_data['sma_200'] = talib.SMA(daily_closes, timeperiod=200)
        self.daily_data['rsi_14'] = talib.RSI(daily_closes, timeperiod=14)
        self.daily_data['macd'], self.daily_data['macd_signal'], _ = talib.MACD(daily_closes)
        self.daily_data['atr_14'] = talib.ATR(daily_highs, daily_lows, daily_closes, timeperiod=14)
        self.daily_data['adx'] = talib.ADX(daily_highs, daily_lows, daily_closes, timeperiod=14)
        
        # Advanced volatility metrics
        log_returns = np.log(daily_closes[1:]/daily_closes[:-1])
        self.daily_data['hist_vol_30'] = pd.Series(log_returns).rolling(30).std() * np.sqrt(252)
        
        hl_ratio = np.log(self.daily_data['h']/self.daily_data['l'])
        self.daily_data['parkinson_vol'] = hl_ratio.rolling(14).std() * np.sqrt(252)
        
        # Volume analysis
        self.daily_data['volume_sma_20'] = talib.SMA(daily_volumes, timeperiod=20)
        self.daily_data['volume_ratio'] = daily_volumes / self.daily_data['volume_sma_20']
        
        # Weekly indicators
        weekly_closes = self.weekly_data['c'].values
        self.weekly_data['sma_10'] = talib.SMA(weekly_closes, timeperiod=10)  # 10 weeks ~ 50 days
        self.weekly_data['sma_40'] = talib.SMA(weekly_closes, timeperiod=40)  # 40 weeks ~ 200 days
        
        # Composite trend strength (daily + weekly)
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
            "min_volume": 500_000,  # Reduced from 1M
            "min_price": 5.00,      # Avoid penny stocks but include mid-caps
            "max_price": 500.00,    # Reduced from 10,000
            "min_market_cap": 300_000_000,  # $300M instead of $50M
            "momentum_days": 30,
            "min_momentum": 0.05,   # 5% instead of 15% 
            "min_relative_strength": 1.0,  # Match QQQ instead of outperform
            "volatility_multiplier": 1.0,
            "pattern_score_threshold": 1,  # Accept even 1 basic pattern
            "days_to_scan": 90
        }
        
        # Get current regime
        trend, vol = await self.get_current_regime()
        
        # Adjust based on trend regime
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
                "min_momentum": -0.03,  # Can be slightly negative
                "min_relative_strength": 0.90,
                "pattern_priority": ["oversold", "falling_wedge"]
            })
        elif trend == "strong_bear":
            base_criteria.update({
                "min_momentum": -0.05,
                "min_relative_strength": 0.85,
                "pattern_priority": ["short_squeeze", "double_bottom"]
            })
        
        # Adjust based on volatility regime
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

    def _analyze_regime_transitions(self) -> Dict[str, bool]:
        """Detect potential regime transitions"""
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

    async def detect_regime(self) -> Dict[str, float]:
        """Enhanced regime detection with multi-timeframe analysis"""
        if self.daily_data is None:
            return self._default_regime_probabilities()
        
        recent_daily = self.daily_data.iloc[-1]
        recent_weekly = self.weekly_data.iloc[-1]
        transition_info = self._analyze_regime_transitions()
        
        # Calculate key metrics
        price_above_200sma = recent_daily['c'] > recent_daily['sma_200']
        sma_50_above_200 = recent_daily['sma_50'] > recent_daily['sma_200']
        weekly_sma_10_above_40 = recent_weekly['sma_10'] > recent_weekly['sma_40']
        trend_strength = recent_daily['trend_strength']
        rsi_14 = recent_daily['rsi_14']
        macd_above_signal = recent_daily['macd'] > recent_daily['macd_signal']
        
        # Volatility metrics
        atr_14 = recent_daily['atr_14']
        atr_30 = self.daily_data['atr_14'].iloc[-30:].mean()
        vol_ratio = atr_14 / atr_30 if atr_30 > 0 else 1.0
        hist_vol = recent_daily['hist_vol_30']
        parkinson_vol = recent_daily['parkinson_vol']
        
        # Momentum calculations
        momentum_5d = (recent_daily['c'] / self.daily_data['c'].iloc[-5] - 1) * 100
        momentum_30d = (recent_daily['c'] / self.daily_data['c'].iloc[-30] - 1) * 100
        weekly_momentum = (recent_weekly['c'] / self.weekly_data['c'].iloc[-4] - 1) * 100  # 4 weeks ~ 1 month
        
        # Volume analysis
        volume_spike = recent_daily['volume_ratio'] > 1.5
        
        # Calculate regime probabilities with transition adjustments
        regimes = {
            "strong_bull": self._strong_bull_confidence(
                price_above_200sma, sma_50_above_200, weekly_sma_10_above_40,
                trend_strength, rsi_14, macd_above_signal,
                momentum_5d, momentum_30d, weekly_momentum,
                transition_info
            ),
            "weak_bull": self._weak_bull_confidence(
                price_above_200sma, sma_50_above_200, weekly_sma_10_above_40,
                trend_strength, rsi_14, macd_above_signal,
                momentum_5d, momentum_30d, weekly_momentum,
                transition_info
            ),
            "neutral": self._neutral_confidence(
                rsi_14, vol_ratio, atr_14, atr_30,
                trend_strength, hist_vol, parkinson_vol,
                transition_info
            ),
            "weak_bear": self._weak_bear_confidence(
                price_above_200sma, sma_50_above_200, weekly_sma_10_above_40,
                trend_strength, rsi_14, macd_above_signal,
                momentum_5d, momentum_30d, weekly_momentum,
                transition_info
            ),
            "strong_bear": self._strong_bear_confidence(
                price_above_200sma, sma_50_above_200, weekly_sma_10_above_40,
                trend_strength, rsi_14, macd_above_signal,
                momentum_5d, momentum_30d, weekly_momentum,
                transition_info
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
                              weekly_momentum: float, transition_info: Dict) -> float:
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
        
        # Reduce confidence if potential transition
        if transition_info['potential_transition']:
            score *= 0.7
            
        return score
    
    def _weak_bull_confidence(self, price_above_200sma: bool, sma_50_above_200: bool,
                            weekly_sma_10_above_40: bool, trend_strength: float,
                            rsi_14: float, macd_above_signal: bool,
                            momentum_5d: float, momentum_30d: float,
                            weekly_momentum: float, transition_info: Dict) -> float:
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
        
        if transition_info['potential_transition']:
            score *= 0.8
            
        return score
    
    def _neutral_confidence(self, rsi_14: float, vol_ratio: float, atr_14: float,
                          atr_30: float, trend_strength: float,
                          hist_vol: float, parkinson_vol: float,
                          transition_info: Dict) -> float:
        score = 0
        if 40 <= rsi_14 <= 60: score += 0.25
        if 0.9 <= vol_ratio <= 1.1: score += 0.2
        if 0.3 <= trend_strength <= 0.7: score += 0.2
        if 0.8 <= (atr_14 / atr_30) <= 1.2 if atr_30 > 0 else False: score += 0.15
        if 0.9 <= (hist_vol / parkinson_vol) <= 1.1 if parkinson_vol > 0 else False: score += 0.2
        
        if transition_info['potential_transition']:
            score *= 0.9
            
        return score
    
    def _weak_bear_confidence(self, price_above_200sma: bool, sma_50_above_200: bool,
                             weekly_sma_10_above_40: bool, trend_strength: float,
                             rsi_14: float, macd_above_signal: bool,
                             momentum_5d: float, momentum_30d: float,
                             weekly_momentum: float, transition_info: Dict) -> float:
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
        
        if transition_info['potential_transition']:
            score *= 0.8
            
        return score
    
    def _strong_bear_confidence(self, price_above_200sma: bool, sma_50_above_200: bool,
                               weekly_sma_10_above_40: bool, trend_strength: float,
                               rsi_14: float, macd_above_signal: bool,
                               momentum_5d: float, momentum_30d: float,
                               weekly_momentum: float, transition_info: Dict) -> float:
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
        
        if transition_info['potential_transition']:
            score *= 0.7
            
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
    
    async def get_regime_description(self) -> Dict:
        """Enhanced regime description with transition info"""
        regimes = await self.detect_regime()
        trend_regime, vol_regime = await self.get_current_regime()
        transition_info = self._analyze_regime_transitions()
        
        return {
            "primary_trend": trend_regime,
            "primary_volatility": vol_regime,
            "trend_probabilities": {r: regimes[r] for r in self.trend_regimes},
            "volatility_probabilities": {r: regimes[r] for r in self.volatility_regimes},
            "transition_analysis": transition_info,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_scan_criteria(self) -> Dict:
        """More inclusive scanning criteria with regime awareness"""
        base_criteria = {
            "min_volume": 500_000,  # Reduced from 1M
            "min_price": 5.00,      # Avoid penny stocks but include mid-caps
            "max_price": 500.00,    # Reduced from 10,000
            "min_market_cap": 300_000_000,  # $300M instead of $50M
            "momentum_days": 30,
            "min_momentum": 0.05,   # 5% instead of 15% 
            "min_relative_strength": 1.0,  # Match QQQ instead of outperform
            "volatility_multiplier": 1.0,
            "pattern_score_threshold": 1,  # Accept even 1 basic pattern
            "days_to_scan": 90
        }
        
        # Get current regime
        trend, vol = await self.get_current_regime()
        
        # Adjust based on trend regime
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
                "min_momentum": -0.03,  # Can be slightly negative
                "min_relative_strength": 0.90,
                "pattern_priority": ["oversold", "falling_wedge"]
            })
        elif trend == "strong_bear":
            base_criteria.update({
                "min_momentum": -0.05,
                "min_relative_strength": 0.85,
                "pattern_priority": ["short_squeeze", "double_bottom"]
            })
        
        # Adjust based on volatility regime
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

class StockScanner:
    """Dynamic stock scanner that adapts to market conditions"""
    
    def __init__(self, polygon_client: AsyncPolygonIOClient, verbose: bool = False):
        self.polygon = polygon_client
        self.regime_detector = MarketRegimeDetector(polygon_client, verbose=verbose)
        self.scan_results = []
        self.tickers_to_scan = []
        self.max_tickers_to_scan = None
        self.ftp_client = NASDAQTraderFTP()
        self.rejection_reasons = defaultdict(list)  # Track why tickers were rejected
        self.verbose = verbose
        self.rejection_stats = defaultdict(int)
        
    async def initialize(self):
        """Initialize the scanner"""
        if not await self.regime_detector.initialize():
            print("Failed to initialize market regime detector")
            return False
        
        # Load tickers to scan
        self.tickers_to_scan = await self._load_tickers_to_scan()
        if not self.tickers_to_scan:
            print("No tickers to scan")
            return False
            
        if self.verbose:
            print(f"Loaded {len(self.tickers_to_scan)} tickers to scan")
        return True
    
    async def _load_tickers_to_scan(self) -> List[str]:
        """Load tickers to scan with enhanced filtering"""
        try:
            if self.verbose:
                print("Loading tickers from NASDAQ FTP...")
            ftp_tickers = self.ftp_client.get_tickers_from_ftp()
            
            if ftp_tickers and len(ftp_tickers) > 1000:
                if self.verbose:
                    print(f"Found {len(ftp_tickers)} tickers via FTP")
                return ftp_tickers[:self.max_tickers_to_scan]
            
            # Fallback to Polygon if FTP fails
            if self.verbose:
                print("Falling back to Polygon API for tickers...")
            all_tickers = await self.polygon.get_all_tickers()
            
            if not all_tickers:
                print("No tickers returned from Polygon API")
                return []
                
            # Additional filtering
            filtered_tickers = []
            for t in all_tickers:
                ticker = t["ticker"]
                
                # Skip leveraged/inverse ETFs
                if re.match(r'^[A-Z]+[23][XL]$', ticker):
                    continue
                    
                # Skip tickers with numbers (like bonds)
                if any(c.isdigit() for c in ticker):
                    continue
                    
                # Skip preferred shares and other special securities
                if t.get("currency_name") != "usd":
                    continue
                    
                filtered_tickers.append(ticker)
            
            return filtered_tickers[:self.max_tickers_to_scan]
            
        except Exception as e:
            print(f"Error loading tickers: {str(e)}")
            return []
        
    async def scan_tickers(self):
        """Scan all tickers with clean progress display only"""
        if not self.tickers_to_scan:
            print("No tickers to scan")
            return
            
        criteria = await self.regime_detector.get_scan_criteria()
        print(f"\nScanning {len(self.tickers_to_scan)} tickers...")
        print(f"Current Criteria: {criteria}")
        
        # Initialize progress bar
        pbar = tqdm(
            total=len(self.tickers_to_scan),
            desc="Progress",
            unit="ticker",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            file=sys.stdout,
            dynamic_ncols=True
        )
        
        # Process in batches
        batch_size = 50
        for i in range(0, len(self.tickers_to_scan), batch_size):
            batch = self.tickers_to_scan[i:i + batch_size]
            await self._scan_batch(batch, criteria)
            pbar.update(len(batch))
            await asyncio.sleep(1)  # Rate limiting
            
        pbar.close()
        
        # If no results, relax criteria once
        if len(self.scan_results) == 0 and criteria["min_momentum"] > 0:
            print("\nNo results found - relaxing criteria...")
            relaxed_criteria = criteria.copy()
            relaxed_criteria["min_momentum"] = max(0, criteria["min_momentum"] * 0.7)
            relaxed_criteria["min_volume"] = int(criteria["min_volume"] * 0.7)
            await self._scan_batch(self.tickers_to_scan, relaxed_criteria)
        
        print(f"\nScan complete. Found {len(self.scan_results)} qualifying stocks")
        self.print_rejection_summary()

    def print_rejection_summary(self):
        """Show why stocks failed to qualify"""
        if not self.rejection_stats:
            print("No rejection data available")
            return
        
        print("\nRejection Reasons Summary:")
        for reason, count in sorted(self.rejection_stats.items(), key=lambda x: -x[1]):
            print(f"{count:>5} | {reason}")

    def print_top_stocks(self, n: int = 20) -> None:
        """Print top n candidates in clean table format"""
        top_candidates = self.get_top_candidates(n)
        
        if not top_candidates:
            print("No qualifying stocks found")
            return
            
        # Prepare table header
        header = (
            f"{'Ticker':<6} | {'Price':>8} | {'Momentum':>8} | "
            f"{'Rel Str':>7} | {'Volatility':>9} | {'Score':>5} | {'Patterns'}"
        )
        separator = "-" * len(header)
        
        # Print table
        print("\nTOP QUALIFYING STOCKS:")
        print(separator)
        print(header)
        print(separator)
        
        for stock in top_candidates:
            print(
                f"{stock['ticker']:<6} | "
                f"${stock['price']:>7.2f} | "
                f"{stock['momentum']:>7.1f}% | "
                f"{stock['relative_strength']:>6.2f} | "
                f"{stock['volatility']:>8.2f}% | "
                f"{stock['composite_score']:>5.1f} | "
                f"{', '.join(stock['patterns_found'])}"
            )
        
        print(separator)

    async def _scan_batch(self, tickers: List[str], criteria: Dict):
        """Scan a batch of tickers"""
        tasks = [self._scan_ticker(ticker, criteria) for ticker in tickers]
        await asyncio.gather(*tasks)
    
    async def _scan_ticker(self, ticker: str, criteria: Dict) -> Optional[Dict]:
        """Scan a single ticker against the criteria"""
        rejection_reasons = []
        
        try:
            # Get price data with adjusted prices
            price_data = await self.polygon.get_aggregates(
                ticker, 
                days=criteria["days_to_scan"]
            )
            
            if price_data is None:
                self._log_rejection(ticker, ["no price data available"])
                return None
                
            if len(price_data) < 20:
                self._log_rejection(ticker, [f"insufficient data ({len(price_data)} days)"])
                return None
                
            # Calculate basic metrics
            closes = price_data["c"].values
            latest_close = closes[-1]
            volumes = price_data["v"].values
            avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0
            
            # Check basic criteria
            if latest_close < criteria["min_price"]:
                rejection_reasons.append(f"price ${latest_close:.2f} < ${criteria['min_price']}")
            elif latest_close > criteria["max_price"]:
                rejection_reasons.append(f"price ${latest_close:.2f} > ${criteria['max_price']}")
                
            if avg_volume < criteria["min_volume"]:
                rejection_reasons.append(f"volume {avg_volume:,.0f} < {criteria['min_volume']:,.0f}")
                
            if rejection_reasons:
                self._log_rejection(ticker, rejection_reasons)
                return None
                
            # Calculate momentum
            lookback = min(criteria["momentum_days"], len(closes))
            momentum = (closes[-1] / closes[-lookback] - 1) * 100
            
            if momentum < criteria["min_momentum"]:
                self._log_rejection(ticker, [f"momentum {momentum:.1f}% < {criteria['min_momentum']:.1f}%"])
                return None
                
            # Calculate relative strength vs QQQ
            qqq_data = self.regime_detector.daily_data
            if qqq_data is not None and len(qqq_data) >= lookback:
                qqq_momentum = (qqq_data["c"].iloc[-1] / qqq_data["c"].iloc[-lookback] - 1) * 100
                relative_strength = momentum / qqq_momentum if qqq_momentum != 0 else 1.0
                if relative_strength < criteria["min_relative_strength"]:
                    self._log_rejection(ticker, [f"relative strength {relative_strength:.2f} < {criteria['min_relative_strength']:.2f}"])
                    return None
            else:
                relative_strength = 1.0
                
            # Calculate volatility
            highs = price_data["h"].values
            lows = price_data["l"].values
            atr = talib.ATR(highs, lows, closes, timeperiod=14)
            current_atr = atr[-1] if len(atr) > 0 else 0
            atr_percent = (current_atr / latest_close) * 100 if latest_close > 0 else 0
            
            if criteria.get("min_volatility") and atr_percent < criteria["min_volatility"]:
                self._log_rejection(ticker, [f"volatility {atr_percent:.2f}% < {criteria['min_volatility']:.2f}%"])
                return None
            elif criteria.get("max_volatility") and atr_percent > criteria["max_volatility"]:
                self._log_rejection(ticker, [f"volatility {atr_percent:.2f}% > {criteria['max_volatility']:.2f}%"])
                return None
                
            # Check technical patterns
            pattern_score = self._check_technical_patterns(price_data)
            patterns_found = self._get_patterns_found(price_data)
            
            if pattern_score < criteria["pattern_score_threshold"]:
                self._log_rejection(ticker, ["insufficient pattern score"])
                return None
                
            # Get additional details
            details = await self.polygon.get_ticker_details(ticker)
            market_cap = details.get("market_cap", 0) if details else 0
            
            if market_cap < criteria["min_market_cap"]:
                self._log_rejection(ticker, [f"market cap ${market_cap:,.0f} < ${criteria['min_market_cap']:,.0f}"])
                return None
                
            if criteria.get("max_market_cap") and market_cap > criteria["max_market_cap"]:
                self._log_rejection(ticker, [f"market cap ${market_cap:,.0f} > ${criteria['max_market_cap']:,.0f}"])
                return None
                
            # Calculate composite score
            score = self._calculate_composite_score(
                momentum=momentum,
                relative_strength=relative_strength,
                volume=avg_volume,
                volatility=atr_percent,
                pattern_score=pattern_score,
                criteria=criteria
            )
            
            # Add to results
            result = {
                "ticker": ticker,
                "price": latest_close,
                "momentum": momentum,
                "relative_strength": relative_strength,
                "volume": avg_volume,
                "volatility": atr_percent,
                "market_cap": market_cap,
                "pattern_score": pattern_score,
                "composite_score": score,
                "patterns_found": patterns_found
            }
            
            self.scan_results.append(result)
            return result
            
        except Exception as e:
            self._log_rejection(ticker, [f"scan error: {str(e)}"])
            return None

    def _log_rejection(self, ticker: str, reasons: List[str]):
        """Track why tickers were rejected"""
        for reason in reasons:
            self.rejection_stats[reason] += 1
        
        if self.verbose and reasons:
            print(f"Rejected {ticker}: {', '.join(reasons)}")

    def _check_technical_patterns(self, price_data: pd.DataFrame) -> int:
        """Enhanced technical pattern detection with bias fixes"""
        patterns_found = []
        
        # Get price data arrays - using iloc[-2] for signal generation
        closes = price_data["c"].values
        opens = price_data["o"].values
        highs = price_data["h"].values
        lows = price_data["l"].values
        volumes = price_data["v"].values
        
        # Need at least 60 periods of history
        if len(closes) < 60:
            return 0
            
        # 1. Breakout/Support/Resistance Detection
        breakout_patterns = self._detect_breakout_support_resistance(
            closes[:-1],  # Exclude current candle
            highs[:-1],
            lows[:-1],
            volumes[:-1]
        )
        
        # 2. Moving Average Crossovers
        ma_patterns = self._detect_moving_average_crossovers(closes[:-1])
        
        # 3. Candlestick Patterns
        candle_patterns = self._detect_candlestick_patterns(
            opens[-3:],  # Last 3 complete candles
            highs[-3:],
            lows[-3:],
            closes[-3:]
        )
        
        patterns_found.extend(breakout_patterns)
        patterns_found.extend(ma_patterns)
        patterns_found.extend(candle_patterns)
        
        return len(patterns_found)

    def _detect_breakout_support_resistance(self, closes: np.ndarray, 
                                          highs: np.ndarray, lows: np.ndarray, 
                                          volumes: np.ndarray) -> List[str]:
        """Improved breakout detection without look-ahead"""
        patterns = []
        lookback = 60
        recent = 5
        
        if len(closes) < lookback:
            return patterns
            
        resistance = np.max(highs[-lookback:-recent])
        support = np.min(lows[-lookback:-recent])
        current_close = closes[-1]
        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0
        
        # Breakout/breakdown with volume confirmation
        if current_close > resistance and volumes[-1] > 1.5 * avg_volume:
            patterns.append("breakout")
        elif current_close < support and volumes[-1] > 1.5 * avg_volume:
            patterns.append("breakdown")
            
        # Support/resistance tests
        if current_close > support and np.min(lows[-recent:]) >= support * 0.995:
            patterns.append("support_bounce")
        if current_close < resistance and np.max(highs[-recent:]) >= resistance * 0.995:
            patterns.append("resistance_test")
            
        return patterns

    def _detect_moving_average_crossovers(self, closes: np.ndarray) -> List[str]:
        """MA crossovers with no look-ahead"""
        patterns = []
        
        if len(closes) < 200:
            return patterns
            
        sma_50 = talib.SMA(closes, 50)
        sma_200 = talib.SMA(closes, 200)
        
        # Golden/Death Cross
        if sma_50[-2] <= sma_200[-2] and sma_50[-1] > sma_200[-1]:
            patterns.append("golden_cross")
        elif sma_50[-2] >= sma_200[-2] and sma_50[-1] < sma_200[-1]:
            patterns.append("death_cross")
            
        # Price crossing 200 SMA
        if closes[-1] > sma_200[-1] and closes[-2] <= sma_200[-2]:
            patterns.append("above_200ma")
        elif closes[-1] < sma_200[-1] and closes[-2] >= sma_200[-2]:
            patterns.append("below_200ma")
            
        return patterns

    def _detect_candlestick_patterns(self, opens: np.ndarray, highs: np.ndarray,
                                   lows: np.ndarray, closes: np.ndarray) -> List[str]:
        """Reliable candlestick patterns"""
        patterns = []
        
        if len(closes) < 3:
            return patterns
            
        # Single candle patterns
        if talib.CDLHAMMER(opens, highs, lows, closes)[-1] > 0:
            patterns.append("hammer")
        if talib.CDLDOJI(opens, highs, lows, closes)[-1] > 0:
            patterns.append("doji")
            
        # Multi-candle patterns
        if len(closes) >= 3:
            if talib.CDLMORNINGSTAR(opens, highs, lows, closes)[-1] > 0:
                patterns.append("morning_star")
            if talib.CDLEVENINGSTAR(opens, highs, lows, closes)[-1] > 0:
                patterns.append("evening_star")
                
        return patterns

    def _get_patterns_found(self, price_data: pd.DataFrame) -> List[str]:
        """Get all detected patterns"""
        patterns = []
        closes = price_data["c"].values
        opens = price_data["o"].values
        highs = price_data["h"].values
        lows = price_data["l"].values
        volumes = price_data["v"].values
        
        patterns.extend(self._detect_breakout_support_resistance(closes, highs, lows, volumes))
        patterns.extend(self._detect_moving_average_crossovers(closes))
        patterns.extend(self._detect_candlestick_patterns(opens[-3:], highs[-3:], lows[-3:], closes[-3:]))
        
        return sorted(list(set(patterns)))

    def _calculate_composite_score(self, momentum: float, relative_strength: float,
                                 volume: float, volatility: float, 
                                 pattern_score: float, criteria: Dict) -> float:
        """Balanced scoring system"""
        weights = {
            "momentum": 0.3,
            "relative_strength": 0.2,
            "volume": 0.2,
            "volatility": 0.1,
            "pattern": 0.2
        }
        
        # Normalized factors
        norm_momentum = min(max(momentum / 20, 0), 1)  # Cap at 20% momentum
        norm_rs = min(max(relative_strength, 0.5), 2)  # 0.5x-2x benchmark
        norm_volume = 1 if volume >= criteria["min_volume"] else 0
        norm_pattern = min(pattern_score / 3, 1)  # Max 3 patterns
        
        score = (
            weights["momentum"] * norm_momentum +
            weights["relative_strength"] * norm_rs +
            weights["volume"] * norm_volume +
            weights["pattern"] * norm_pattern
        )
        
        return score * 100

    def get_top_candidates(self, n: int = 20) -> List[Dict]:
        """Get top n candidates sorted by composite score"""
        return sorted(
            self.scan_results, 
            key=lambda x: x["composite_score"], 
            reverse=True
        )[:n]

    def _log_rejection(self, ticker: str, reasons: List[str]):
        """Log why a ticker was rejected"""
        if not hasattr(self, 'rejection_stats'):
            self.rejection_stats = defaultdict(int)
        
        for reason in reasons:
            self.rejection_stats[reason] += 1
            
        if self.verbose and reasons:
            print(f"Rejected {ticker}: {', '.join(reasons)}")

    def print_rejection_summary(self):
        """Show why stocks failed to qualify"""
        if not hasattr(self, 'rejection_stats'):
            print("No rejection data available")
            return
        
        print("\nRejection Reasons Summary:")
        for reason, count in sorted(self.rejection_stats.items(), key=lambda x: -x[1]):
            print(f"{count:>5} | {reason}")

async def main():
    POLYGON_API_KEY = "OZzn0oK0H2yG6rpIvVhGfgXgnUTrL31z"
    
    async with AsyncPolygonIOClient(POLYGON_API_KEY, verbose=True) as client:
        scanner = StockScanner(client, verbose=True)
        scanner.max_tickers_to_scan = 500  # Limit for testing
        
        if not await scanner.initialize():
            print("Failed to initialize scanner")
            return
            
        # Get current market regime
        regime_info = await scanner.regime_detector.get_regime_description()
        print("\nCurrent Market Regime Analysis:")
        print(f"Primary Trend: {regime_info['primary_trend']}")
        print(f"Primary Volatility: {regime_info['primary_volatility']}")
        
        # Run the scanner
        await scanner.scan_tickers()
        scanner.print_top_stocks(20)

if __name__ == "__main__":
    asyncio.run(main())