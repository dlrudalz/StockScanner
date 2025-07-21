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

class AsyncPolygonIOClient:
    """Enhanced Polygon.io API client with caching and multi-timeframe support"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.rate_limit_delay = 1.2
        self.last_request_time = 0
        self.session = None
        self.semaphore = asyncio.Semaphore(5)
        self.cache = {}

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

            logger.info(f"Fetching {ticker} {timespan} data from {start_date} to {end_date}")

            endpoint = f"/v2/aggs/ticker/{ticker}/range/1/{timespan}/{start_date}/{end_date}"
            params = {"adjusted": "true", "apiKey": self.api_key}

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
                logger.error(f"No results for {ticker}")
                return None
                
            df = pd.DataFrame(data["results"])
            df["date"] = pd.to_datetime(df["t"], unit="ms")
            df = df.set_index("date")
            self.cache[cache_key] = df.copy()
            return df

        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {str(e)}")
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
    
    def __init__(self, polygon_client: AsyncPolygonIOClient):
        self.polygon = polygon_client
        self.daily_data = None
        self.weekly_data = None
        self.volatility_regimes = ["low_vol", "medium_vol", "high_vol"]
        self.trend_regimes = ["strong_bull", "weak_bull", "neutral", "weak_bear", "strong_bear"]
        
    async def initialize(self) -> bool:
        """Initialize with daily and weekly data"""
        try:
            logger.info("Initializing market regime detector...")
            
            # Get daily and weekly data
            self.daily_data = await self.polygon.get_aggregates("QQQ", days=252, timespan="day")
            self.weekly_data = await self.polygon.get_aggregates("QQQ", days=252, timespan="week")
            
            if self.daily_data is None or self.weekly_data is None:
                logger.error("Failed to fetch QQQ data - API returned None")
                return False
                
            if len(self.daily_data) < 100 or len(self.weekly_data) < 20:
                logger.error(f"Insufficient data points: Daily={len(self.daily_data)}, Weekly={len(self.weekly_data)}")
                return False
                
            logger.info(f"Loaded {len(self.daily_data)} daily and {len(self.weekly_data)} weekly data points")
            self._calculate_technical_indicators()
            return True
            
        except Exception as e:
            logger.error(f"Error initializing detector: {str(e)}", exc_info=True)
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

async def main():
    POLYGON_API_KEY = "OZzn0oK0H2yG6rpIvVhGfgXgnUTrL31z"
    
    async with AsyncPolygonIOClient(POLYGON_API_KEY) as client:
        # Initialize detector
        detector = MarketRegimeDetector(client)
        if not await detector.initialize():
            print("\nWarning: Limited data available. Results may be less accurate.")
            
        # Get current regime
        trend, vol = await detector.get_current_regime()
        print(f"\nCurrent Market Regime: {trend} trend, {vol} volatility")
        
        # Get detailed regime analysis
        regime_info = await detector.get_regime_description()
        print("\nDetailed Analysis:")
        print(f"Primary Trend: {regime_info['primary_trend']}")
        print(f"Primary Volatility: {regime_info['primary_volatility']}")
        
        print("\nTrend Probabilities:")
        for regime, prob in regime_info['trend_probabilities'].items():
            print(f"  {regime:12}: {prob:.1%}")
            
        print("\nVolatility Probabilities:")
        for regime, prob in regime_info['volatility_probabilities'].items():
            print(f"  {regime:12}: {prob:.1%}")
            
        print("\nTransition Analysis:")
        for k, v in regime_info['transition_analysis'].items():
            print(f"  {k:20}: {v}")

if __name__ == "__main__":
    asyncio.run(main())