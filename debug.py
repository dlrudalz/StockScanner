import asyncio
import aiohttp
import async_timeout
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import logging
import talib
from typing import Dict, List, Optional
from collections import defaultdict
from tqdm import tqdm
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class AsyncPolygonIOClient:
    """Enhanced Polygon.io API client with robust error handling"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.session = None
        self.semaphore = asyncio.Semaphore(5)  # Concurrency limit
        self.rate_limit_delay = 0.5  # Seconds between requests
        self.last_request_time = 0

    async def __aenter__(self):
        """Initialize async session"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
        
    async def __aexit__(self, exc_type, exc, tb):
        """Cleanup async session"""
        await self.session.close()

    async def get_aggregates(self, ticker: str, days: int = 90) -> Optional[pd.DataFrame]:
        """Get historical price data with comprehensive error handling"""
        try:
            # Calculate date range with buffer for holidays
            end_date = date.today() - timedelta(days=1)  # Yesterday
            start_date = end_date - timedelta(days=int(days * 1.5))
            
            logger.info(f"\nFetching {ticker} from {start_date} to {end_date}")
            
            params = {
                "adjusted": "true",
                "apiKey": self.api_key,
                "limit": days + 20  # Extra buffer
            }
            
            async with self.semaphore:
                await self._throttle()
                async with async_timeout.timeout(30):
                    url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
                    
                    async with self.session.get(url, params=params) as response:
                        logger.info(f"HTTP Status: {response.status}")
                        
                        if response.status == 429:
                            logger.warning("Rate limited, waiting 10 seconds...")
                            await asyncio.sleep(10)
                            return await self.get_aggregates(ticker, days)
                            
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"API Error: {error_text}")
                            return None
                            
                        data = await response.json()
                        
                        if not data.get("results"):
                            logger.warning(f"No results for {ticker}")
                            return None
                            
                        df = pd.DataFrame(data["results"])
                        
                        # Validate data structure
                        required_columns = {'c', 'v', 't'}
                        if not required_columns.issubset(df.columns):
                            logger.error(f"Missing columns in response for {ticker}")
                            return None
                            
                        df["date"] = pd.to_datetime(df["t"], unit="ms")
                        df = df.set_index("date").sort_index()
                        
                        # Data quality checks
                        if len(df) < days * 0.7:  # Allow 30% missing days
                            logger.warning(f"Only {len(df)} trading days for {ticker}")
                            
                        if df['c'].isnull().any():
                            logger.error(f"NaN values in close prices for {ticker}")
                            return None
                            
                        logger.info(f"Successfully retrieved {len(df)} days for {ticker}")
                        return df
                        
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching data for {ticker}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {ticker}: {str(e)}")
            return None

    async def _throttle(self):
        """Enforce rate limiting"""
        now = asyncio.get_event_loop().time()
        elapsed = now - self.last_request_time
        if elapsed < self.rate_limit_delay:
            wait_time = self.rate_limit_delay - elapsed
            await asyncio.sleep(wait_time)
        self.last_request_time = now

class StockScannerDebugger:
    """Comprehensive scanner debugger with rejection tracking"""
    
    def __init__(self, polygon_client):
        self.polygon = polygon_client
        self.rejection_reasons = defaultdict(list)
        self.price_data_cache = {}
        
    async def debug_scan(self, test_tickers: List[str]):
        """Main debug method to analyze multiple tickers"""
        logger.info("\n=== STOCK SCANNER DEBUGGER ===")
        logger.info(f"Testing {len(test_tickers)} tickers\n")
        
        for ticker in test_tickers:
            await self._analyze_ticker(ticker)
        
        self._print_summary_report()
        
    async def _analyze_ticker(self, ticker: str):
        """Full analysis pipeline for a single ticker"""
        logger.info(f"\n🔍 Analyzing {ticker}")
        
        # 1. Data Acquisition
        data = await self._get_price_data(ticker)
        if data is None:
            return
            
        # 2. Basic Requirements Check
        if not await self._check_basic_requirements(ticker, data):
            return
            
        # 3. Technical Analysis
        await self._check_technical_conditions(ticker, data)
        
        # 4. Full Scan Simulation
        await self._simulate_full_scan(ticker, data)
    
    async def _get_price_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch price data with debug logging"""
        logger.info("  📊 Fetching price data...")
        data = await self.polygon.get_aggregates(ticker, 90)
        
        if data is None:
            logger.error("  ❌ Failed to get price data")
            self.rejection_reasons[ticker].append("no_price_data")
            return None
            
        if len(data) < 60:
            logger.warning(f"  ⚠️ Only {len(data)} trading days (expected ~90)")
            self.rejection_reasons[ticker].append("insufficient_data")
            
        logger.info(f"  ✅ Retrieved {len(data)} days ending {data.index[-1].date()}")
        logger.info(f"  Last Close: ${data['c'].iloc[-1]:.2f}, Volume: {data['v'].iloc[-1]:,.0f}")
        
        self.price_data_cache[ticker] = data
        return data
    
    async def _check_basic_requirements(self, ticker: str, data: pd.DataFrame) -> bool:
        """Check minimum price and volume requirements"""
        logger.info("  🔎 Checking basic requirements...")
        passes = True
        latest = data.iloc[-1]
        
        # Price Validation
        price = latest['c']
        if not (5 < price < 10000):  # Adjusted reasonable range
            logger.error(f"  ❌ Price ${price:.2f} outside valid range")
            self.rejection_reasons[ticker].append(f"price_out_of_range")
            passes = False
            
        # Volume Validation
        avg_volume = data['v'].mean()
        if avg_volume < 100000:  # 100k average volume
            logger.error(f"  ❌ Low avg volume: {avg_volume:,.0f}")
            self.rejection_reasons[ticker].append("low_volume")
            passes = False
            
        if passes:
            logger.info("  ✅ Passes basic requirements")
        return passes
    
    async def _check_technical_conditions(self, ticker: str, data: pd.DataFrame):
        """Analyze technical indicators"""
        logger.info("  📈 Checking technical conditions...")
        closes = data['c'].values
        highs = data['h'].values
        lows = data['l'].values
        
        # Calculate indicators
        momentum_90 = (closes[-1] / closes[0] - 1) * 100
        sma_50 = talib.SMA(closes, timeperiod=50)[-1]
        rsi_14 = talib.RSI(closes, timeperiod=14)[-1]
        atr_14 = talib.ATR(highs, lows, closes, timeperiod=14)[-1]
        
        logger.info(f"  90-Day Momentum: {momentum_90:.1f}%")
        logger.info(f"  SMA 50: {sma_50:.2f}")
        logger.info(f"  RSI 14: {rsi_14:.1f}")
        logger.info(f"  ATR 14: {atr_14:.2f}")
        
        # Momentum Check
        if momentum_90 < 5:  # Minimum 5% momentum
            logger.error("  ❌ Insufficient momentum")
            self.rejection_reasons[ticker].append("low_momentum")
            
        # Trend Check
        if closes[-1] < sma_50:
            logger.error("  ❌ Below 50-day SMA")
            self.rejection_reasons[ticker].append("below_sma_50")
            
        # Overbought/Oversold
        if rsi_14 > 70:
            logger.error("  ❌ Overbought (RSI > 70)")
            self.rejection_reasons[ticker].append("overbought")
        elif rsi_14 < 30:
            logger.error("  ❌ Oversold (RSI < 30)")
            self.rejection_reasons[ticker].append("oversold")
    
    async def _simulate_full_scan(self, ticker: str, data: pd.DataFrame):
        """Simulate complete scan with scoring"""
        logger.info("  🧪 Simulating full scan...")
        
        # Mock criteria - adjust these to match your actual criteria
        criteria = {
            "min_price": 10,
            "max_price": 1000,
            "min_volume": 100000,
            "min_momentum": 5,
            "min_relative_strength": 0.9,
            "days_to_scan": 90,
            "pattern_priority": ["breakout", "new_high"]
        }
        
        # Calculate metrics
        closes = data['c'].values
        latest_close = closes[-1]
        momentum = (closes[-1] / closes[0] - 1) * 100
        avg_volume = data['v'].mean()
        
        # Check all criteria
        passes = True
        
        if not (criteria["min_price"] <= latest_close <= criteria["max_price"]):
            logger.error(f"  ❌ Price ${latest_close:.2f} outside range")
            self.rejection_reasons[ticker].append("price_range")
            passes = False
            
        if avg_volume < criteria["min_volume"]:
            logger.error(f"  ❌ Volume {avg_volume:,.0f} below minimum")
            self.rejection_reasons[ticker].append("volume")
            passes = False
            
        if momentum < criteria["min_momentum"]:
            logger.error(f"  ❌ Momentum {momentum:.1f}% too low")
            self.rejection_reasons[ticker].append("momentum")
            passes = False
            
        if passes:
            logger.info("  ✅ Would qualify in full scan")
            # Calculate mock composite score
            score = self._calculate_mock_score(data)
            logger.info(f"  Composite Score: {score:.1f}/100")
        else:
            logger.info("  ❌ Would reject in full scan")
    
    def _calculate_mock_score(self, data: pd.DataFrame) -> float:
        """Calculate mock composite score (0-100)"""
        closes = data['c'].values
        volumes = data['v'].values
        
        # Score components (0-1 scale)
        momentum_score = min((closes[-1] / closes[0] - 1) * 10, 1)  # Max 10% = 1.0
        volume_score = min(np.log10(volumes.mean() / 100000) / 3, 1)  # Log scale
        consistency_score = 0.7 if (closes[-1] > closes[-20]) else 0.3
        
        # Weighted composite
        return 100 * (0.4 * momentum_score + 0.3 * volume_score + 0.3 * consistency_score)
    
    def _print_summary_report(self):
        """Print comprehensive rejection analysis"""
        logger.info("\n=== SCAN DIAGNOSTIC REPORT ===")
        
        # Rejection reason counts
        reason_counts = defaultdict(int)
        qualified = 0
        
        for ticker, reasons in self.rejection_reasons.items():
            if not reasons:
                qualified += 1
            for reason in reasons:
                reason_counts[reason] += 1
                
        # Print rejection breakdown
        logger.info("\nRejection Reasons:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"- {reason}: {count} tickers")
            
        # Print qualification stats
        logger.info(f"\n✅ {qualified} tickers passed all checks")
        logger.info(f"❌ {len(self.rejection_reasons) - qualified} tickers rejected")
        
        # Print sample qualified tickers
        qualified_tickers = [t for t in self.rejection_reasons if not self.rejection_reasons[t]]
        if qualified_tickers:
            logger.info("\nSample Qualified Tickers:")
            for ticker in qualified_tickers[:5]:  # Show max 5
                logger.info(f"- {ticker}")

async def main():
    POLYGON_API_KEY = "OZzn0oK0H2yG6rpIvVhGfgXgnUTrL31z"  # <-- Critical!
    
    # Configure test tickers
    test_tickers = [
        # Large Caps
        'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL',
        # ETFs
        'SPY', 'QQQ', 'IWM',
        # Potential Rejects
        'ACB', 'NKLA', 'BB',
        # Special Cases
        'A', 'BRK.A'
    ]
    
    async with AsyncPolygonIOClient(POLYGON_API_KEY) as client:
        # Initial connectivity test
        logger.info("=== CONNECTIVITY TEST ===")
        try:
            test_url = f"{client.base_url}/v1/marketstatus/now"
            async with client.session.get(test_url) as response:
                logger.info(f"API Connectivity: HTTP {response.status}")
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            return
            
        # Run debug scan
        debugger = StockScannerDebugger(client)
        await debugger.debug_scan(test_tickers)

if __name__ == "__main__":
    # Configure asyncio to handle nested async calls
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())