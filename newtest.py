import os
import ftplib
import pandas as pd
import numpy as np
import aiohttp
import asyncio
import talib
from datetime import datetime, timedelta
from typing import Literal, Dict, List, Tuple, Optional, DefaultDict, Any
from collections import defaultdict
from dataclasses import dataclass
import logging
import zipfile
import io
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Try to get API key from config.py first, then environment variables

try:
    from config import POLYGON_API_KEY
except ImportError:
    POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
    if not POLYGON_API_KEY:
        raise ValueError("Missing Polygon API key. Please set in config.py or environment variables")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    NASDAQ_FTP_SERVER = "ftp.nasdaqtrader.com"
    NASDAQ_FTP_DIR = "SymbolDirectory"
    TICKER_FILES = {
        "nasdaq": "nasdaqlisted.txt",
        "nyse": "nyse-listed.txt", 
        "amex": "amex-listed.txt"
    }
    POLYGON_API_KEY = POLYGON_API_KEY
    MARKET_REGIME_SYMBOL = "SPY"
    MAX_CONCURRENT_REQUESTS = 10
    REQUEST_DELAY = 0.1

@dataclass
class TickerInfo:
    symbol: str
    name: str
    exchange: str
    ipo_date: Optional[datetime] = None
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    is_etf: bool = False

class PolygonIOClient:
    """Async client for Polygon.io API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.session = None
        self.last_request_time = 0
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()
            
    async def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < Config.REQUEST_DELAY:
            await asyncio.sleep(Config.REQUEST_DELAY - elapsed)
        self.last_request_time = time.time()
            
    async def get_aggregates(self, ticker: str, days: int, timespan: str = "day") -> Optional[pd.DataFrame]:
        """Get historical aggregates"""
        try:
            await self._rate_limit()
            
            multiplier = 1
            if timespan == "week":
                multiplier = 7
            elif timespan == "month":
                multiplier = 30
                
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
            
            async with self.session.get(url, params={"apiKey": self.api_key}) as response:
                if response.status != 200:
                    logger.error(f"Polygon API error for {ticker}: {response.status}")
                    return None
                    
                data = await response.json()
                if data.get("resultsCount", 0) == 0:
                    return None
                    
                df = pd.DataFrame(data["results"])
                df["t"] = pd.to_datetime(df["t"], unit="ms")
                df.set_index("t", inplace=True)
                df.rename(columns={
                    "o": "Open",
                    "h": "High",
                    "l": "Low",
                    "c": "Close",
                    "v": "Volume",
                    "vw": "VWAP"
                }, inplace=True)
                return df
        except Exception as e:
            logger.error(f"Error getting aggregates for {ticker}: {str(e)}")
            return None
            
    async def get_ticker_details(self, ticker: str) -> Optional[Dict]:
        """Get fundamental ticker details"""
        try:
            await self._rate_limit()
            
            url = f"{self.base_url}/v3/reference/tickers/{ticker}"
            async with self.session.get(url, params={"apiKey": self.api_key}) as response:
                if response.status != 200:
                    return None
                return await response.json()
        except Exception as e:
            logger.error(f"Error getting ticker details for {ticker}: {str(e)}")
            return None
            
    async def get_ipo_calendar(self, date_from: str, date_to: str) -> Optional[List[Dict]]:
        """Get upcoming IPOs"""
        try:
            await self._rate_limit()
            
            url = f"{self.base_url}/v1/meta/calendars/ipo"
            params = {
                "apiKey": self.api_key,
                "from": date_from,
                "to": date_to
            }
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    return None
                data = await response.json()
                return data.get("results", [])
        except Exception as e:
            logger.error(f"Error getting IPO calendar: {str(e)}")
            return None
            
    async def get_daily_open_close(self, ticker: str, date: str) -> Optional[Dict]:
        """Get daily open/close data"""
        try:
            await self._rate_limit()
            
            url = f"{self.base_url}/v1/open-close/{ticker}/{date}"
            async with self.session.get(url, params={"apiKey": self.api_key}) as response:
                if response.status != 200:
                    return None
                return await response.json()
        except Exception as e:
            logger.error(f"Error getting daily open/close for {ticker}: {str(e)}")
            return None

class TickerFetcher:
    """Fetches tickers from NASDAQ FTP and enriches with Polygon.io data"""
    
    def __init__(self):
        self.ftp = ftplib.FTP(Config.NASDAQ_FTP_SERVER)
        self.ftp.login()
        self.ftp.cwd(Config.NASDAQ_FTP_DIR)
        
    def fetch_tickers(self) -> Dict[str, List[TickerInfo]]:
        """Fetch all tickers from NASDAQ, NYSE, and AMEX"""
        tickers = defaultdict(list)
        
        for exchange, filename in Config.TICKER_FILES.items():
            try:
                with io.BytesIO() as buffer:
                    self.ftp.retrbinary(f"RETR {filename}", buffer.write)
                    buffer.seek(0)
                    
                    if filename.endswith(".txt"):
                        df = pd.read_csv(buffer, sep="|")
                    elif filename.endswith(".zip"):
                        with zipfile.ZipFile(buffer) as z:
                            with z.open(z.namelist()[0]) as f:
                                df = pd.read_csv(f, sep="|")
                    
                    # Clean up dataframe
                    df = df[~df["Symbol"].str.contains("test", case=False)]
                    df = df[~df["Symbol"].str.contains("\$")]
                    df = df[~df["Symbol"].str.contains("\^")]
                    df = df[df["Symbol"].str.len() <= 5]
                    
                    for _, row in df.iterrows():
                        ticker = TickerInfo(
                            symbol=row["Symbol"],
                            name=row.get("Security Name", ""),
                            exchange=exchange.upper(),
                            is_etf="ETF" in row.get("Security Name", "")
                        )
                        tickers[exchange].append(ticker)
            except Exception as e:
                logger.error(f"Error fetching {exchange} tickers: {str(e)}")
                continue
                
        self.ftp.quit()
        return tickers
        
    async def enrich_ticker_info(self, ticker_info: TickerInfo, polygon_client: PolygonIOClient) -> TickerInfo:
        """Enrich ticker with IPO date, market cap, etc."""
        if ticker_info.is_etf:
            return ticker_info
            
        details = await polygon_client.get_ticker_details(ticker_info.symbol)
        if details:
            ticker_info.market_cap = details.get("market_cap")
            ticker_info.sector = details.get("sector")
            list_date = details.get("list_date")
            if list_date:
                try:
                    ticker_info.ipo_date = pd.to_datetime(list_date)
                except:
                    ticker_info.ipo_date = None
        return ticker_info

class EnhancedMarketRegimeDetector:
    """Advanced market regime detector with additional indicators and performance metrics"""
    
    def __init__(self, polygon_client: PolygonIOClient, debugger: logging.Logger):
        self.polygon = polygon_client
        self.daily_data = None
        self.weekly_data = None
        self.monthly_data = None
        self.volatility_regimes = ["low_vol", "medium_vol", "high_vol"]
        self.trend_regimes = ["strong_bull", "weak_bull", "neutral", "weak_bear", "strong_bear"]
        self.debugger = debugger
        self.performance_metrics = {
            'historical_accuracy': [],
            'regime_duration': defaultdict(list),
            'transition_stats': defaultdict(int)
        }
        self.current_regime = None
        self.previous_regime = None
        self.regime_start_date = None
        
    async def initialize(self) -> bool:
        try:
            self.debugger.debug("Initializing enhanced market regime detector...")
            
            # Load multiple timeframes
            self.daily_data = await self._get_validated_data(Config.MARKET_REGIME_SYMBOL, days=252, timespan="day")
            self.weekly_data = await self._get_validated_data(Config.MARKET_REGIME_SYMBOL, days=252*3, timespan="week")
            self.monthly_data = await self._get_validated_data(Config.MARKET_REGIME_SYMBOL, days=252*5, timespan="month")
            
            # Check if any of the dataframes are None or empty
            if (self.daily_data is None or len(self.daily_data) == 0 or
                self.weekly_data is None or len(self.weekly_data) == 0 or
                self.monthly_data is None or len(self.monthly_data) == 0):
                self.debugger.debug("One or more dataframes failed validation")
                return False
                
            self._calculate_technical_indicators()
            self._initialize_performance_tracking()
            return True
            
        except Exception as e:
            self.debugger.error(f"Error initializing detector: {str(e)}")
            return False
    
    async def _get_validated_data(self, ticker: str, days: int, timespan: str) -> Optional[pd.DataFrame]:
        """Get and validate market data"""
        try:
            data = await self.polygon.get_aggregates(ticker, days=days, timespan=timespan)
            if data is None or len(data) < 20:
                self.debugger.debug(f"Insufficient {timespan} data for {ticker}")
                return None
            
            # Validate data quality
            if data['Close'].isnull().sum() > 0.1 * len(data):
                self.debugger.debug(f"Too many nulls in {ticker} {timespan} data")
                return None
                
            return data
        except Exception as e:
            self.debugger.error(f"Error getting data for {ticker}: {str(e)}")
            return None
    
    def _calculate_technical_indicators(self):
        """Calculate all technical indicators with validation"""
        try:
            # Daily indicators
            daily_closes = self.daily_data['Close'].values
            daily_highs = self.daily_data['High'].values
            daily_lows = self.daily_data['Low'].values
            daily_volumes = self.daily_data['Volume'].values
            
            # Core trend indicators
            self.daily_data['sma_50'] = talib.SMA(daily_closes, timeperiod=50)
            self.daily_data['sma_200'] = talib.SMA(daily_closes, timeperiod=200)
            self.daily_data['ema_20'] = talib.EMA(daily_closes, timeperiod=20)
            
            # Momentum indicators
            self.daily_data['rsi_14'] = talib.RSI(daily_closes, timeperiod=14)
            self.daily_data['macd'], self.daily_data['macd_signal'], _ = talib.MACD(daily_closes)
            self.daily_data['adx'] = talib.ADX(daily_highs, daily_lows, daily_closes, timeperiod=14)
            self.daily_data['cci'] = talib.CCI(daily_highs, daily_lows, daily_closes, timeperiod=20)
            
            # Volatility indicators
            self.daily_data['atr_14'] = talib.ATR(daily_highs, daily_lows, daily_closes, timeperiod=14)
            self.daily_data['natr_14'] = talib.NATR(daily_highs, daily_lows, daily_closes, timeperiod=14)
            self.daily_data['bollinger_upper'], _, self.daily_data['bollinger_lower'] = talib.BBANDS(
                daily_closes, timeperiod=20, nbdevup=2, nbdevdn=2)
            
            log_returns = np.log(daily_closes[1:]/daily_closes[:-1])
            self.daily_data['hist_vol_30'] = pd.Series(log_returns).rolling(30).std() * np.sqrt(252)
            
            hl_ratio = np.log(self.daily_data['High']/self.daily_data['Low'])
            self.daily_data['parkinson_vol'] = hl_ratio.rolling(14).std() * np.sqrt(252)
            
            # Volume indicators
            self.daily_data['volume_sma_20'] = talib.SMA(daily_volumes, timeperiod=20)
            self.daily_data['volume_ratio'] = daily_volumes / self.daily_data['volume_sma_20']
            self.daily_data['obv'] = talib.OBV(daily_closes, daily_volumes)
            
            # Weekly indicators
            weekly_closes = self.weekly_data['Close'].values
            self.weekly_data['sma_10'] = talib.SMA(weekly_closes, timeperiod=10)
            self.weekly_data['sma_40'] = talib.SMA(weekly_closes, timeperiod=40)
            self.weekly_data['rsi_8'] = talib.RSI(weekly_closes, timeperiod=8)
            
            # Monthly indicators
            monthly_closes = self.monthly_data['Close'].values
            self.monthly_data['sma_6'] = talib.SMA(monthly_closes, timeperiod=6)
            self.monthly_data['sma_12'] = talib.SMA(monthly_closes, timeperiod=12)
            
            # Composite trend strength score (0-1 scale)
            trend_components = []
            trend_components.append(0.25 * (self.daily_data['sma_50'] > self.daily_data['sma_200']))
            trend_components.append(0.20 * (self.weekly_data['sma_10'] > self.weekly_data['sma_40']).resample('D').ffill())
            trend_components.append(0.15 * (self.monthly_data['sma_6'] > self.monthly_data['sma_12']).resample('D').ffill())
            trend_components.append(0.15 * (self.daily_data['adx'] / 100))
            trend_components.append(0.10 * (self.daily_data['macd'] > self.daily_data['macd_signal']))
            trend_components.append(0.10 * (self.daily_data['Close'] > self.daily_data['sma_50']))
            trend_components.append(0.05 * (self.daily_data['rsi_14'] / 100))
            
            self.daily_data['trend_strength'] = sum(tc[:len(self.daily_data)] for tc in trend_components)
            
            # Market breadth indicator (simplified)
            self.daily_data['adv_vol'] = np.where(self.daily_data['Close'] > self.daily_data['Close'].shift(1), 
                                                self.daily_data['Volume'], 0)
            self.daily_data['dec_vol'] = np.where(self.daily_data['Close'] < self.daily_data['Close'].shift(1), 
                                                self.daily_data['Volume'], 0)
            self.daily_data['adv_dec_ratio'] = (self.daily_data['adv_vol'].rolling(5).sum() / 
                                              (self.daily_data['dec_vol'].rolling(5).sum() + 1e-6))
            
        except Exception as e:
            self.debugger.error(f"Indicator calculation error: {str(e)}")
            raise
    
    def _initialize_performance_tracking(self):
        """Initialize performance tracking metrics"""
        if len(self.daily_data) < 100:
            return
            
        # Backfill historical regime classification
        for i in range(100, len(self.daily_data)):
            historic_data = self.daily_data.iloc[:i]
            regimes = self._calculate_historic_regime(historic_data)
            self._update_performance_metrics(regimes, historic_data.index[-1])
    
    def _calculate_historic_regime(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate regime probabilities for historical data"""
        recent_daily = data.iloc[-1]
        
        # Calculate all components needed for regime detection
        price_above_200sma = recent_daily['Close'] > recent_daily['sma_200']
        sma_50_above_200 = recent_daily['sma_50'] > recent_daily['sma_200']
        trend_strength = recent_daily['trend_strength']
        rsi_14 = recent_daily['rsi_14']
        macd_above_signal = recent_daily['macd'] > recent_daily['macd_signal']
        
        atr_14 = recent_daily['atr_14']
        atr_30 = data['atr_14'].iloc[-30:].mean()
        vol_ratio = atr_14 / atr_30 if atr_30 > 0 else 1.0
        hist_vol = recent_daily['hist_vol_30']
        parkinson_vol = recent_daily['parkinson_vol']
        
        momentum_5d = (recent_daily['Close'] / data['Close'].iloc[-5] - 1) * 100
        momentum_30d = (recent_daily['Close'] / data['Close'].iloc[-30] - 1) * 100
        
        volume_spike = recent_daily['volume_ratio'] > 1.5
        
        return self._calculate_regime_probabilities(
            price_above_200sma, sma_50_above_200, trend_strength,
            rsi_14, macd_above_signal, momentum_5d, momentum_30d,
            vol_ratio, atr_14, atr_30, hist_vol, parkinson_vol, volume_spike
        )
    
    async def detect_regime(self) -> Dict[str, float]:
        """Detect current market regime with performance tracking"""
        if self.daily_data is None:
            return self._default_regime_probabilities()
        
        recent_daily = self.daily_data.iloc[-1]
        
        # Calculate all components
        price_above_200sma = recent_daily['Close'] > recent_daily['sma_200']
        sma_50_above_200 = recent_daily['sma_50'] > recent_daily['sma_200']
        trend_strength = recent_daily['trend_strength']
        rsi_14 = recent_daily['rsi_14']
        macd_above_signal = recent_daily['macd'] > recent_daily['macd_signal']
        
        atr_14 = recent_daily['atr_14']
        atr_30 = self.daily_data['atr_14'].iloc[-30:].mean()
        vol_ratio = atr_14 / atr_30 if atr_30 > 0 else 1.0
        hist_vol = recent_daily['hist_vol_30']
        parkinson_vol = recent_daily['parkinson_vol']
        
        momentum_5d = (recent_daily['Close'] / self.daily_data['Close'].iloc[-5] - 1) * 100
        momentum_30d = (recent_daily['Close'] / self.daily_data['Close'].iloc[-30] - 1) * 100
        
        volume_spike = recent_daily['volume_ratio'] > 1.5
        
        regimes = self._calculate_regime_probabilities(
            price_above_200sma, sma_50_above_200, trend_strength,
            rsi_14, macd_above_signal, momentum_5d, momentum_30d,
            vol_ratio, atr_14, atr_30, hist_vol, parkinson_vol, volume_spike
        )
        
        # Update performance tracking
        self._update_performance_metrics(regimes, self.daily_data.index[-1])
        
        return regimes
    
    def _calculate_regime_probabilities(self, price_above_200sma: bool, sma_50_above_200: bool, trend_strength: float,
                                     rsi_14: float, macd_above_signal: bool, momentum_5d: float, momentum_30d: float,
                                     vol_ratio: float, atr_14: float, atr_30: float, hist_vol: float, parkinson_vol: float, volume_spike: bool) -> Dict[str, float]:
        """Core regime probability calculation"""
        regimes = {
            "strong_bull": self._strong_bull_confidence(
                price_above_200sma, sma_50_above_200, trend_strength,
                rsi_14, macd_above_signal, momentum_5d, momentum_30d
            ),
            "weak_bull": self._weak_bull_confidence(
                price_above_200sma, sma_50_above_200, trend_strength,
                rsi_14, macd_above_signal, momentum_5d, momentum_30d
            ),
            "neutral": self._neutral_confidence(
                rsi_14, vol_ratio, atr_14, atr_30, trend_strength, hist_vol, parkinson_vol
            ),
            "weak_bear": self._weak_bear_confidence(
                price_above_200sma, sma_50_above_200, trend_strength,
                rsi_14, macd_above_signal, momentum_5d, momentum_30d
            ),
            "strong_bear": self._strong_bear_confidence(
                price_above_200sma, sma_50_above_200, trend_strength,
                rsi_14, macd_above_signal, momentum_5d, momentum_30d
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
    
    def _update_performance_metrics(self, regimes: Dict[str, float], date: datetime):
        """Update performance tracking metrics"""
        current_trend = max(
            ((r, regimes[r]) for r in self.trend_regimes),
            key=lambda x: x[1],
            default=("neutral", 0)
        )[0]
        
        current_vol = max(
            ((r, regimes[r]) for r in self.volatility_regimes),
            key=lambda x: x[1],
            default=("medium_vol", 0)
        )[0]
        
        current_regime = (current_trend, current_vol)
        
        # Track regime duration
        if self.current_regime != current_regime:
            if self.current_regime is not None:
                duration = (date - self.regime_start_date).days
                self.performance_metrics['regime_duration'][self.current_regime].append(duration)
                self.performance_metrics['transition_stats'][(self.current_regime, current_regime)] += 1
            
            self.previous_regime = self.current_regime
            self.current_regime = current_regime
            self.regime_start_date = date
        
        # Track accuracy (simplified - would need actual future returns to measure real accuracy)
        if len(self.daily_data) > 30:
            # Use iloc for positional indexing
            future_return = (self.daily_data['Close'].shift(-30).iloc[-1] / self.daily_data['Close'].iloc[-1] - 1) * 100
            correct_prediction = self._validate_regime_prediction(current_trend, future_return)
            self.performance_metrics['historical_accuracy'].append(correct_prediction)
    
    def _validate_regime_prediction(self, predicted_trend: str, future_return: float) -> bool:
        """Validate if the regime prediction was correct"""
        if predicted_trend in ["strong_bull", "weak_bull"] and future_return > 2:
            return True
        elif predicted_trend == "neutral" and -2 <= future_return <= 2:
            return True
        elif predicted_trend in ["weak_bear", "strong_bear"] and future_return < -2:
            return True
        return False
    
    def get_performance_metrics(self) -> Dict:
        """Get calculated performance metrics"""
        metrics = {
            'accuracy_30d': np.mean(self.performance_metrics['historical_accuracy']) if self.performance_metrics['historical_accuracy'] else 0,
            'avg_regime_duration': {str(k): np.mean(v) if v else 0 
                                  for k, v in self.performance_metrics['regime_duration'].items()},
            'common_transitions': sorted(self.performance_metrics['transition_stats'].items(), 
                                       key=lambda x: -x[1]),
            'current_regime_duration': (datetime.now() - self.regime_start_date).days if self.regime_start_date else 0,
            'regime_stability': self._calculate_regime_stability()
        }
        return metrics
    
    def _calculate_regime_stability(self) -> float:
        """Calculate a stability score (0-1) for the current regime"""
        if not self.performance_metrics['regime_duration']:
            return 0.5
        
        current_duration = (datetime.now() - self.regime_start_date).days if self.regime_start_date else 0
        avg_duration = np.mean(self.performance_metrics['regime_duration'].get(self.current_regime, [current_duration]))
        
        if current_duration < avg_duration * 0.5:
            return 0.2  # Early in regime
        elif current_duration > avg_duration * 1.5:
            return 0.8  # Extended regime
        return 0.5  # Normal duration
    
    def _strong_bull_confidence(self, price_above_200sma: bool, sma_50_above_200: bool,
                              trend_strength: float, rsi_14: float, macd_above_signal: bool,
                              momentum_5d: float, momentum_30d: float) -> float:
        score = 0
        if price_above_200sma: score += 0.15
        if sma_50_above_200: score += 0.15
        if trend_strength > 0.75: score += 0.15
        if 60 < rsi_14 <= 80: score += 0.1
        if macd_above_signal: score += 0.1
        if momentum_5d > 1.5: score += 0.1
        if momentum_30d > 5.0: score += 0.1
        return score
    
    def _weak_bull_confidence(self, price_above_200sma: bool, sma_50_above_200: bool,
                            trend_strength: float, rsi_14: float, macd_above_signal: bool,
                            momentum_5d: float, momentum_30d: float) -> float:
        score = 0
        if price_above_200sma: score += 0.15
        if sma_50_above_200: score += 0.1
        if trend_strength > 0.55: score += 0.15
        if 50 < rsi_14 <= 60: score += 0.15
        if macd_above_signal: score += 0.1
        if momentum_5d > 0: score += 0.1
        if momentum_30d > 2.0: score += 0.1
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
                             trend_strength: float, rsi_14: float, macd_above_signal: bool,
                             momentum_5d: float, momentum_30d: float) -> float:
        score = 0
        if not price_above_200sma: score += 0.15
        if not sma_50_above_200: score += 0.1
        if trend_strength < 0.45: score += 0.15
        if 30 <= rsi_14 < 50: score += 0.15
        if not macd_above_signal: score += 0.1
        if momentum_5d < 0: score += 0.1
        if momentum_30d < -2.0: score += 0.1
        return score
    
    def _strong_bear_confidence(self, price_above_200sma: bool, sma_50_above_200: bool,
                               trend_strength: float, rsi_14: float, macd_above_signal: bool,
                               momentum_5d: float, momentum_30d: float) -> float:
        score = 0
        if not price_above_200sma: score += 0.15
        if not sma_50_above_200: score += 0.15
        if trend_strength < 0.25: score += 0.15
        if rsi_14 < 30: score += 0.1
        if not macd_above_signal: score += 0.1
        if momentum_5d < -1.5: score += 0.1
        if momentum_30d < -5.0: score += 0.1
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
    
    def _default_regime_probabilities(self) -> Dict[str, float]:
        return {
            "strong_bull": 0.2, "weak_bull": 0.2, "neutral": 0.2, 
            "weak_bear": 0.2, "strong_bear": 0.2,
            "low_vol": 0.33, "medium_vol": 0.34, "high_vol": 0.33
        }
    
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
    
    async def get_regime_description(self) -> Dict:
        """Enhanced regime description with performance metrics"""
        regimes = await self.detect_regime()
        trend_regime, vol_regime = await self.get_current_regime()
        transition_info = self._analyze_regime_transitions()
        performance = self.get_performance_metrics()
        
        # Safely handle missing transition stats
        top_transitions = []
        if 'common_transitions' in performance and performance['common_transitions']:
            top_transitions = performance['common_transitions'][:3]  # Get top 3 transitions
        
        return {
            "primary_trend": str(trend_regime),
            "primary_volatility": str(vol_regime),
            "trend_probabilities": {str(r): float(regimes[r]) for r in self.trend_regimes},
            "volatility_probabilities": {str(r): float(regimes[r]) for r in self.volatility_regimes},
            "transition_analysis": {
                'potential_transition': bool(transition_info.get('potential_transition', False)),
                'trend_weakening': bool(transition_info.get('trend_weakening', False)),
                'vol_increasing': bool(transition_info.get('vol_increasing', False))
            },
            "performance_metrics": {
                'accuracy_30d': performance.get('accuracy_30d', 0),
                'top_transitions': top_transitions,
                'avg_durations': performance.get('avg_regime_duration', {}),
                'current_duration': performance.get('current_regime_duration', 0),
                'stability': performance.get('regime_stability', 0.5)
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def format_regime_report(self, regime_data: Dict) -> str:
        """Format the regime data into a clean text report"""
        # Safely extract all values with defaults
        primary_trend = regime_data.get('primary_trend', 'neutral')
        primary_vol = regime_data.get('primary_volatility', 'medium_vol')
        trend_probs = regime_data.get('trend_probabilities', {})
        vol_probs = regime_data.get('volatility_probabilities', {})
        performance = regime_data.get('performance_metrics', {})
        
        report = [
            f"Market Analysis Report - {regime_data.get('timestamp', '')}",
            "\n=== MARKET REGIME ===",
            f"Primary Trend: {primary_trend.upper()} "
            f"({round(trend_probs.get(primary_trend, 0)*100, 1)}% probability)",
            f"Volatility: {primary_vol.upper()} "
            f"({round(vol_probs.get(primary_vol, 0)*100, 1)}% probability)",
            "\nTrend Probabilities:",
            *[f"- {k.replace('_', ' ').title()}: {round(v*100, 1)}%" 
              for k, v in trend_probs.items()],
            "\nVolatility Probabilities:",
            *[f"- {k.replace('_', ' ').title()}: {round(v*100, 1)}%" 
              for k, v in vol_probs.items()],
            "\n=== REGIME TRANSITIONS ===",
            f"Current Regime Duration: {performance.get('current_duration', 0)} days",
            f"Stability: {self._get_stability_description(performance.get('stability', 0.5))}",
        ]
        
        # Only add transitions if they exist
        if performance.get('top_transitions'):
            report.extend([
                "\nMost Common Transitions:",
                *[f"{i+1}. {k[0][0]} → {k[1][0]} ({v} occurrences)" 
                  for i, (k, v) in enumerate(performance['top_transitions'])]
            ])
        
        # Only add durations if they exist
        if performance.get('avg_durations'):
            report.extend([
                "\nAverage Regime Durations:",
                *[f"- {k}: {v:.1f} days" for k, v in performance['avg_durations'].items()][:3]
            ])
        
        return "\n".join(report)

    def _get_stability_description(self, score: float) -> str:
        if score < 0.3:
            return "Low"
        elif score < 0.7:
            return "Medium"
        return "High"

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

# Strategy Configuration and Classes
STRATEGY_CONFIG = {
    "trend_following": {
        "allocation": (0.20, 0.30),
        "holding_period": (30, 90),
        "stop_loss": 0.08,
        "profit_targets": [(0.25, 0.4), (0.50, 0.4), (1.0, 0.2)],
        "filters": {
            "min_volume": 750_000,
            "min_price": 5.00,
            "max_price": 500.00,
            "min_market_cap": 300_000_000,
            "min_trend_duration": 60,
            "min_adx": 25,
            "sma_condition": "50>200",
            "price_above_sma": 50,
            "min_volume_ratio": 1.2,
            "max_volatility": 0.30,
            "min_volatility": 0.05,
            "days_to_scan": 120,
            "pattern_priority": ["golden_cross", "bullish_ma_stack"]
        }
    },
    "mean_reversion": {
        "allocation": (0.15, 0.25),
        "holding_period": (5, 21),
        "stop_loss": 0.12,
        "profit_targets": [(0.15, 0.6), (0.25, 0.4)],
        "filters": {
            "min_volume": 500_000,
            "min_price": 2.00,
            "max_price": 200.00,
            "min_market_cap": 100_000_000,
            "max_rsi": 30,
            "bollinger_band_position": "lower",
            "min_volatility": 0.10,
            "max_volatility": 0.40,
            "days_to_scan": 90,
            "pattern_priority": ["hammer", "doji", "support_bounce"]
        }
    },
    "breakout": {
        "allocation": (0.15, 0.25),
        "holding_period": (7, 30),
        "stop_loss": 0.07,
        "profit_targets": [(0.15, 0.5), (0.25, 0.3), (0.50, 0.2)],
        "filters": {
            "min_volume": 1_500_000,
            "min_price": 5.00,
            "max_price": 300.00,
            "min_market_cap": 300_000_000,
            "min_volume_ratio": 1.5,
            "consolidation_days": 10,
            "max_consolidation": 0.15,
            "min_volatility": 0.10,
            "max_volatility": 0.50,
            "days_to_scan": 60,
            "pattern_priority": ["breakout", "bullish_engulfing"]
        }
    },
    "ipo_fade": {
        "allocation": (0.10, 0.15),
        "holding_period": (1, 5),
        "stop_loss": 0.05,
        "profit_targets": [(0.08, 1.0)],
        "filters": {
            "max_days_since_ipo": 5,
            "min_opening_pop": 0.50,
            "min_premarket_volume": 1_000_000,
            "vwap_divergence": 0.03,
            "days_to_scan": 5
        }
    },
    "lockup_expiry": {
        "allocation": (0.10, 0.20),
        "holding_period": (30, 90),
        "stop_loss": 0.25,
        "profit_targets": [(0.30, 0.7), (0.50, 0.3)],
        "filters": {
            "days_to_lockup": 10,
            "max_price_vs_ipo": 0.85,
            "min_volume_decline": 0.30,
            "min_put_oi_ratio": 1.5,
            "days_to_scan": 180
        }
    },
    "catalyst_scalping": {
        "allocation": (0.05, 0.10),
        "holding_period": (0, 1),  # Intraday only
        "stop_loss": 0.03,
        "profit_targets": [(0.02, 1.0)],
        "filters": {
            "catalyst_window": 1,  # Hours until catalyst
            "min_premarket_gap": 0.05,
            "min_premarket_volume": 500_000,
            "time_windows": ["09:45-10:15", "15:30-16:00"],
            "min_vwap_deviation": 0.01,
            "days_to_scan": 5
        }
    }
}

class MarketScanner:
    """Main scanning engine with market regime awareness"""
    
    def __init__(self, polygon_client: PolygonIOClient):
        self.polygon = polygon_client
        self.regime_detector = EnhancedMarketRegimeDetector(polygon_client, logger)
        self.strategy_config = STRATEGY_CONFIG
        
    async def initialize(self):
        """Initialize scanner components"""
        if not await self.regime_detector.initialize():
            raise RuntimeError("Failed to initialize market regime detector")
            
    async def scan_market(self) -> Dict[str, List[Dict]]:
        """Run full market scan with regime-aware strategy selection"""
        # Get current market regime
        regime_report = await self.regime_detector.get_regime_description()
        logger.info(f"Current Market Regime:\n{self.regime_detector.format_regime_report(regime_report)}")
        
        # Adjust strategies based on regime
        active_strategies = self._select_strategies_by_regime(regime_report)
        
        # Fetch all tickers
        ticker_fetcher = TickerFetcher()
        all_tickers = ticker_fetcher.fetch_tickers()
        
        # Flatten tickers across exchanges
        tickers = []
        for exchange in all_tickers.values():
            tickers.extend(exchange)
            
        # Enrich ticker info (IPO dates, market caps, etc.)
        enriched_tickers = []
        for ticker in tickers[:1000]:  # Limit for demo purposes
            enriched = await ticker_fetcher.enrich_ticker_info(ticker, self.polygon)
            enriched_tickers.append(enriched)
            
        # Run strategy-specific scans
        results = {}
        for strategy_name in active_strategies:
            strategy = self.strategy_config[strategy_name]
            results[strategy_name] = await self._scan_for_strategy(strategy_name, strategy, enriched_tickers)
            
        return results
        
    def _select_strategies_by_regime(self, regime_report: Dict) -> List[str]:
        """Select strategies based on current market regime"""
        trend = regime_report["primary_trend"]
        volatility = regime_report["primary_volatility"]
        
        # Strategy selection rules
        if trend in ["strong_bull", "weak_bull"]:
            preferred = ["trend_following", "breakout", "catalyst_scalping"]
            if volatility == "high_vol":
                preferred.append("mean_reversion")
        elif trend == "neutral":
            preferred = ["mean_reversion", "breakout"]
        else:  # Bear market
            preferred = ["mean_reversion", "ipo_fade"]
            if volatility == "high_vol":
                preferred.append("lockup_expiry")
                
        # Always include some strategies
        preferred.extend(["ipo_fade", "lockup_expiry"])
        
        return list(set(preferred))  # Remove duplicates
    
    async def _scan_for_strategy(self, strategy_name: str, strategy_config: Dict, tickers: List[TickerInfo]) -> List[Dict]:
        """Async version of strategy scanning"""
        results = []
        
        # Filter tickers by basic criteria first
        filtered = [
            t for t in tickers 
            if await self._passes_basic_filters(t, strategy_config["filters"])
        ]
        
        # Process filtered tickers
        for ticker in filtered[:100]:  # Limit for demo
            data = await self.polygon.get_aggregates(
                ticker.symbol,
                days=strategy_config["filters"].get("days_to_scan", 90),
                timespan="day"
            )
            
            if data is None or len(data) < 10:
                continue
                
            if await self._passes_strategy_filters(ticker, data, strategy_config["filters"], strategy_name):
                stop_loss = await self._calculate_stop_loss(data, strategy_config)
                results.append({
                    "symbol": ticker.symbol,
                    "name": ticker.name,
                    "exchange": ticker.exchange,
                    "strategy": strategy_name,
                    "entry_price": data["Close"].iloc[-1],
                    "stop_loss": stop_loss,
                    "targets": strategy_config["profit_targets"],
                    "holding_period": strategy_config["holding_period"]
                })
                
        return results

    async def _passes_basic_filters(self, ticker: TickerInfo, filters: Dict) -> bool:
        """Async version of basic filter checks"""
        if ticker.market_cap is None:
            return False
            
        # Market cap filter
        min_cap = filters.get("min_market_cap", 0)
        if ticker.market_cap < min_cap:
            return False
            
        # IPO age filter
        if "max_days_since_ipo" in filters:
            if ticker.ipo_date is None:
                return False
            days_since_ipo = (datetime.now() - ticker.ipo_date).days
            if days_since_ipo > filters["max_days_since_ipo"]:
                return False
                
        return True

    async def _passes_strategy_filters(self, ticker: TickerInfo, data: pd.DataFrame, filters: Dict, strategy_name: str) -> bool:
        """Async version of strategy-specific filters"""
        closes = data["Close"].values
        highs = data["High"].values
        lows = data["Low"].values
        volumes = data["Volume"].values
        last_close = closes[-1]
        
        # Price filter
        if not (filters.get("min_price", 0) <= last_close <= filters.get("max_price", float("inf"))):
            return False
            
        # Volume filter
        min_volume = filters.get("min_volume", 0)
        if volumes[-1] < min_volume:
            return False
            
        # Volume ratio filter
        if "min_volume_ratio" in filters:
            sma_volume = talib.SMA(volumes, timeperiod=20)[-1]
            if volumes[-1] / sma_volume < filters["min_volume_ratio"]:
                return False
                
        # Strategy-specific filters
        if strategy_name == "trend_following":
            sma50 = talib.SMA(closes, timeperiod=50)[-1]
            sma200 = talib.SMA(closes, timeperiod=200)[-1]
            adx = talib.ADX(highs, lows, closes, timeperiod=14)[-1]
            
            if not (sma50 > sma200 and adx > filters.get("min_adx", 25)):
                return False
                
        elif strategy_name == "mean_reversion":
            rsi = talib.RSI(closes, timeperiod=14)[-1]
            lower_band = talib.BBANDS(closes, timeperiod=20, nbdevdn=2)[2][-1]
            
            if not (rsi < filters.get("max_rsi", 30) and last_close <= lower_band):
                return False
                
        elif strategy_name == "breakout":
            recent_max = np.max(closes[-filters.get("consolidation_days", 10):])
            recent_min = np.min(closes[-filters.get("consolidation_days", 10):])
            consolidation_range = (recent_max - recent_min) / recent_min
            
            if consolidation_range > filters.get("max_consolidation", 0.15):
                return False
                
            if last_close < recent_max:
                return False
                
        elif strategy_name == "ipo_fade":
            ipo_data = await self.polygon.get_daily_open_close(ticker.symbol, ticker.ipo_date.strftime("%Y-%m-%d"))
            if not ipo_data:
                return False
                
            ipo_price = ipo_data.get("open")
            if not ipo_price:
                return False
                
            if not (last_close / ipo_price - 1) >= filters.get("min_opening_pop", 0.5):
                return False
                
        elif strategy_name == "lockup_expiry":
            if ticker.ipo_date is None:
                return False
                
            days_since_ipo = (datetime.now() - ticker.ipo_date).days
            lockup_period = filters.get("days_to_lockup", 90)
            
            if not (lockup_period - 10 <= days_since_ipo <= lockup_period + 10):
                return False
                
            ipo_data = await self.polygon.get_daily_open_close(ticker.symbol, ticker.ipo_date.strftime("%Y-%m-%d"))
            if not ipo_data:
                return False
                
            ipo_price = ipo_data.get("open")
            if not ipo_price:
                return False
                
            if not last_close / ipo_price <= filters.get("max_price_vs_ipo", 0.85):
                return False
                
        return True

    async def _calculate_stop_loss(self, data: pd.DataFrame, strategy_config: Dict) -> float:
        """Async version of stop loss calculation"""
        closes = data["Close"].values
        last_close = closes[-1]
        
        if strategy_config["strategy"] == "trend_following":
            atr = talib.ATR(data["High"], data["Low"], closes, timeperiod=14)[-1]
            return last_close - 1.5 * atr
        elif strategy_config["strategy"] == "mean_reversion":
            atr = talib.ATR(data["High"], data["Low"], closes, timeperiod=14)[-1]
            return last_close - 2 * atr
        else:
            return last_close * (1 - strategy_config["stop_loss"])

    async def _check_strategy_filters(self, ticker: TickerInfo, data: pd.DataFrame, filters: Dict, strategy_name: str) -> bool:
        """Async version of strategy filter checks"""
        closes = data["Close"].values
        highs = data["High"].values
        lows = data["Low"].values
        volumes = data["Volume"].values
        last_close = closes[-1]
        
        # Price filter
        if not (filters.get("min_price", 0) <= last_close <= filters.get("max_price", float("inf"))):
            return False
            
        # Volume filter
        min_volume = filters.get("min_volume", 0)
        if volumes[-1] < min_volume:
            return False
            
        # Volume ratio filter
        if "min_volume_ratio" in filters:
            sma_volume = talib.SMA(volumes, timeperiod=20)[-1]
            if volumes[-1] / sma_volume < filters["min_volume_ratio"]:
                return False
                
        # Strategy-specific filters
        if strategy_name == "trend_following":
            sma50 = talib.SMA(closes, timeperiod=50)[-1]
            sma200 = talib.SMA(closes, timeperiod=200)[-1]
            adx = talib.ADX(highs, lows, closes, timeperiod=14)[-1]
            
            if not (sma50 > sma200 and adx > filters.get("min_adx", 25)):
                return False
                
        elif strategy_name == "mean_reversion":
            rsi = talib.RSI(closes, timeperiod=14)[-1]
            lower_band = talib.BBANDS(closes, timeperiod=20, nbdevdn=2)[2][-1]
            
            if not (rsi < filters.get("max_rsi", 30) and last_close <= lower_band):
                return False
                
        elif strategy_name == "breakout":
            # Check consolidation
            recent_max = np.max(closes[-filters.get("consolidation_days", 10):])
            recent_min = np.min(closes[-filters.get("consolidation_days", 10):])
            consolidation_range = (recent_max - recent_min) / recent_min
            
            if consolidation_range > filters.get("max_consolidation", 0.15):
                return False
                
            # Check breakout
            if last_close < recent_max:
                return False
                
        elif strategy_name == "ipo_fade":
            # Get IPO price and first day data - must use await here
            ipo_data = await self.polygon.get_daily_open_close(ticker.symbol, ticker.ipo_date.strftime("%Y-%m-%d"))
            if not ipo_data:
                return False
                
            ipo_price = ipo_data.get("open")
            if not ipo_price:
                return False
                
            if not (last_close / ipo_price - 1) >= filters.get("min_opening_pop", 0.5):
                return False
                
        elif strategy_name == "lockup_expiry":
            if ticker.ipo_date is None:
                return False
                
            days_since_ipo = (datetime.now() - ticker.ipo_date).days
            lockup_period = filters.get("days_to_lockup", 90)
            
            if not (lockup_period - 10 <= days_since_ipo <= lockup_period + 10):
                return False
                
            # Need to get IPO price here too
            ipo_data = await self.polygon.get_daily_open_close(ticker.symbol, ticker.ipo_date.strftime("%Y-%m-%d"))
            if not ipo_data:
                return False
                
            ipo_price = ipo_data.get("open")
            if not ipo_price:
                return False
                
            if not last_close / ipo_price <= filters.get("max_price_vs_ipo", 0.85):
                return False
                
        return True
            
    async def generate_trade_signals(self, scan_results: Dict[str, List[Dict]]) -> List[Dict]:
        """Generate final trade signals with position sizing"""
        regime = await self.regime_detector.get_current_regime()
        signals = []
        
        for strategy_name, candidates in scan_results.items():
            allocation_range = self.strategy_config[strategy_name]["allocation"]
            
            # Adjust allocation based on regime
            if regime[0] in ["strong_bull", "weak_bull"] and strategy_name in ["trend_following", "breakout"]:
                allocation = allocation_range[1]  # Use upper bound
            elif regime[0] in ["weak_bear", "strong_bear"] and strategy_name == "mean_reversion":
                allocation = allocation_range[1]
            else:
                allocation = np.mean(allocation_range)
                
            # Limit number of positions
            max_positions = {
                "trend_following": 10,
                "mean_reversion": 8,
                "breakout": 6,
                "ipo_fade": 4,
                "lockup_expiry": 5,
                "catalyst_scalping": 3
            }.get(strategy_name, 5)
            
            for candidate in candidates[:max_positions]:
                signals.append({
                    **candidate,
                    "allocation": allocation / max_positions,
                    "regime": regime[0],
                    "volatility_regime": regime[1],
                    "timestamp": datetime.now().isoformat()
                })
                
        return signals

async def main():
    """Run the complete scanning process"""
    if not Config.POLYGON_API_KEY:
        logger.error("POLYGON_API_KEY environment variable not set")
        return
        
    async with PolygonIOClient(Config.POLYGON_API_KEY) as polygon_client:
        scanner = MarketScanner(polygon_client)
        try:
            await scanner.initialize()
            
            # Run market scan
            logger.info("Starting market scan...")
            scan_results = await scanner.scan_market()
            
            # Generate trade signals
            signals = await scanner.generate_trade_signals(scan_results)
            
            # Output results
            logger.info(f"Generated {len(signals)} trade signals:")
            for signal in signals[:5]:  # Print first 5 for demo
                print(signal)
                
        except Exception as e:
            logger.error(f"Error in main scanning process: {str(e)}")
            raise

if __name__ == "__main__":
    asyncio.run(main())