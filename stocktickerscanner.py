from polygon import RESTClient 
import ftplib
import pandas as pd
from io import StringIO
import time
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Literal, Tuple, Union
import logging
from functools import reduce
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_scanner.log'),
        logging.StreamHandler() 
    ]
)
logger = logging.getLogger(__name__)

# Define market regime types
MarketRegime = Literal["bull", "bear", "recovery", "consolidation", "high_volatility"]

@dataclass
class StockData:
    ticker: str
    price_data: pd.DataFrame
    fundamentals: Dict
    details: Dict
    short_interest: Optional[float]
    short_interest_date: Optional[datetime]
    short_interest_ratio: Optional[float]

class NASDAQTraderFTP:
    """Handles NASDAQ Trader FTP operations with common stock filtering"""
    
    FTP_SERVER = 'ftp.nasdaqtrader.com'
    FTP_DIR = 'SymbolDirectory'
    NASDAQ_LISTED_FILE = 'nasdaqlisted.txt'
    OTHER_LISTED_FILE = 'otherlisted.txt'
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 5.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.ftp = None
        
    def connect(self) -> bool:
        """Establish FTP connection with retries"""
        for attempt in range(self.max_retries):
            try:
                self.ftp = ftplib.FTP(self.FTP_SERVER, timeout=30)
                self.ftp.login()  # Anonymous login
                self.ftp.cwd(self.FTP_DIR)
                return True
            except (ftplib.all_errors, ConnectionError) as e:
                logger.warning(f"FTP connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                continue
        return False
    
    def disconnect(self):
        """Close FTP connection if active"""
        if self.ftp:
            try:
                self.ftp.quit()
            except:
                self.ftp.close()
            finally:
                self.ftp = None
    
    def download_file(self, filename: str) -> Optional[str]:
        """Download a file from the FTP server"""
        if not self.connect():
            return None
            
        try:
            data = []
            self.ftp.retrlines(f'RETR {filename}', data.append)
            return '\n'.join(data)
        except ftplib.all_errors as e:
            logger.error(f"Error downloading {filename}: {e}")
            return None
        finally:
            self.disconnect()
    
    def _filter_common_stocks(self, df: pd.DataFrame, symbol_col: str) -> pd.DataFrame:
        """Apply common stock filters to a dataframe"""
        return df[
            (df['ETF'] == 'N') &  # Not an ETF
            (df['Test Issue'] == 'N') &  # Not a test issue
            (~df['Security Name'].str.contains(
                'Preferred|Debenture|Bond|Note|Warrant|Right|Unit|Depositary|Trust|Fund',
                case=False, na=False, regex=True)) &
            (~df[symbol_col].str.contains(
                r'\.|\^|\$|\+|=|-|\\|/|_|~',
                na=False, regex=True)) &  # No special characters
            (df[symbol_col].str.len() <= 5) &  # Reasonable ticker length
            (df[symbol_col].str.match('^[A-Za-z]+$'))  # Letters only
        ]
    
    def get_common_stocks(self) -> List[str]:
        """Get only common stock tickers from all exchanges"""
        nasdaq_content = self.download_file(self.NASDAQ_LISTED_FILE)
        other_content = self.download_file(self.OTHER_LISTED_FILE)
        
        tickers = []
        
        if nasdaq_content:
            try:
                df = pd.read_csv(StringIO(nasdaq_content), sep='|')
                df = df[~df['Symbol'].str.contains('File Creation Time', na=False)]
                df = self._filter_common_stocks(df, 'Symbol')
                tickers.extend(df['Symbol'].astype(str).tolist())
            except Exception as e:
                logger.error(f"Error parsing NASDAQ tickers: {e}")
        
        if other_content:
            try:
                df = pd.read_csv(StringIO(other_content), sep='|')
                df = df[~df['ACT Symbol'].str.contains('File Creation Time', na=False)]
                df = self._filter_common_stocks(df, 'ACT Symbol')
                tickers.extend(df['ACT Symbol'].astype(str).tolist())
            except Exception as e:
                logger.error(f"Error parsing other exchange tickers: {e}")
        
        # Final cleanup and deduplication
        tickers = [t.upper().strip() for t in tickers if t and t != 'nan']
        return sorted(list(set(tickers)))

class PolygonIOClient:
    """Complete Polygon.io API client with all endpoints properly implemented"""
    
    def __init__(self, api_key: str):
        self.client = RESTClient(api_key=api_key)
        self.last_request_time = None
        self.rate_limit_delay = 12.0  # 5 requests/minute for free tier
        
    def _throttle(self):
        """Enforce rate limiting between API calls"""
        if self.last_request_time:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def get_ticker_details(self, ticker: str) -> Dict:
        """Get ticker details with robust error handling for missing fields"""
        self._throttle()
        try:
            details = self.client.get_ticker_details(ticker)
            return {
                'symbol': getattr(details, 'ticker', ticker),
                'name': getattr(details, 'name', None),
                'type': getattr(details, 'type', None),
                'market_cap': getattr(details, 'market_cap', None),
                'pe_ratio': getattr(details, 'pe_ratio', None),
                'dividend_yield': getattr(details, 'dividend_yield', None),
                'sector': getattr(details, 'sector', None),
                'industry': getattr(details, 'industry', None),
                'shares_outstanding': getattr(details, 'share_class_shares_outstanding', None)
            }
        except Exception as e:
            logger.warning(f"Failed to get details for {ticker}: {str(e)}")
            return {
                'symbol': ticker,
                'name': None,
                'type': None,
                'market_cap': None,
                'pe_ratio': None,
                'dividend_yield': None,
                'sector': None,
                'industry': None,
                'shares_outstanding': None
            }
    
    def get_aggregates(self, ticker: str, days: int = 90) -> Optional[pd.DataFrame]:
        """Get historical price data"""
        self._throttle()
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            aggs = []
            for a in self.client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d"),
                limit=days,
                adjusted=True
            ):
                aggs.append(a)
                
            if len(aggs) < 30:
                logger.debug(f"Insufficient data points ({len(aggs)}) for {ticker}")
                return None
                
            df = pd.DataFrame([{
                'date': pd.to_datetime(agg.timestamp, unit='ms'),
                'open': agg.open,
                'high': agg.high,
                'low': agg.low,
                'close': agg.close,
                'volume': agg.volume,
                'vwap': agg.vwap
            } for agg in aggs])
            
            return df.set_index('date')
            
        except Exception as e:
            logger.warning(f"Failed to get aggregates for {ticker}: {str(e)}")
            return None
    
    def get_financials(self, ticker: str, price_data: Optional[pd.DataFrame] = None) -> Dict:
        """Get and calculate financial data with fallbacks"""
        self._throttle()
        fundamentals = {
            'ticker': ticker,
            'pe_ratio': None,
            'market_cap': None,
            'dividend_yield': None,
            'revenue': None,
            'earnings': None,
            'profit_margin': None,
            'revenue_growth': None,
            'earnings_growth': None,
            'operating_cash_flow': None,
            'free_cash_flow': None,
            'total_debt': None,
            'total_equity': None,
            'debt_to_equity': None,
            'shares_outstanding': None,
            'eps': None
        }
        
        try:
            # Get basic details first
            details = self.get_ticker_details(ticker)
            if details:
                fundamentals.update({
                    'market_cap': details.get('market_cap'),
                    'dividend_yield': details.get('dividend_yield'),
                    'shares_outstanding': details.get('shares_outstanding')
                })
            
            # Get financial statements if available
            try:
                financials = list(self.client.vx.list_stock_financials(
                    ticker=ticker,
                    limit=2,
                    timeframe="annual",
                    include_sources=False
                ))
                
                if financials:
                    current = financials[0].financials
                    prev = financials[1].financials if len(financials) > 1 else None
                    
                    fundamentals.update({
                        'revenue': current.get('revenue'),
                        'earnings': current.get('net_income'),
                        'operating_cash_flow': current.get('operating_cash_flow'),
                        'free_cash_flow': current.get('free_cash_flow'),
                        'total_debt': current.get('total_debt'),
                        'total_equity': current.get('total_equity'),
                        'shares_outstanding': current.get('weighted_shares_outstanding') or fundamentals['shares_outstanding']
                    })
                    
                    # Calculate EPS if possible
                    if fundamentals['earnings'] and fundamentals['shares_outstanding']:
                        fundamentals['eps'] = fundamentals['earnings'] / fundamentals['shares_outstanding']
                    
                    # Calculate P/E ratio if not provided
                    if not fundamentals['pe_ratio']:
                        if fundamentals['market_cap'] and fundamentals['earnings'] and fundamentals['earnings'] > 0:
                            fundamentals['pe_ratio'] = fundamentals['market_cap'] / fundamentals['earnings']
                        elif fundamentals['eps'] and fundamentals['eps'] > 0 and price_data is not None and not price_data.empty:
                            current_price = price_data['close'].iloc[-1]
                            fundamentals['pe_ratio'] = current_price / fundamentals['eps']
                    
                    # Calculate other ratios
                    if fundamentals['revenue'] and fundamentals['earnings']:
                        fundamentals['profit_margin'] = fundamentals['earnings'] / fundamentals['revenue']
                    
                    if fundamentals['total_debt'] and fundamentals['total_equity']:
                        fundamentals['debt_to_equity'] = fundamentals['total_debt'] / fundamentals['total_equity']
                    
                    # Calculate growth rates
                    if prev:
                        if current.get('revenue') and prev.get('revenue') and prev['revenue'] != 0:
                            fundamentals['revenue_growth'] = (current['revenue'] - prev['revenue']) / prev['revenue']
                        if current.get('net_income') and prev.get('net_income') and prev['net_income'] != 0:
                            fundamentals['earnings_growth'] = (current['net_income'] - prev['net_income']) / prev['net_income']
            
            except AttributeError:
                logger.debug(f"No financials endpoint available for {ticker}")
        
        except Exception as e:
            logger.warning(f"Error processing financials for {ticker}: {str(e)}")
        
        return fundamentals
    
    def get_short_interest(self, ticker: str) -> Tuple[Optional[float], Optional[datetime], Optional[float]]:
        """Get short interest data with fallback calculation"""
        self._throttle()
        try:
            # Try to get from API
            items = list(self.client.list_short_interest(
                ticker=ticker,
                limit=1
            ))
            
            if items:
                item = items[0]
                return (
                    item.short_interest,
                    pd.to_datetime(item.settlement_date),
                    item.days_to_cover
                )
            
            # Fallback if no data from API
            return None, None, None
            
        except Exception as e:
            logger.warning(f"Failed to get short interest for {ticker}: {str(e)}")
            return None, None, None
    
    def calculate_short_interest_ratio(self, short_interest: float, avg_volume: float) -> Optional[float]:
        """Calculate short interest ratio (days to cover)"""
        if not short_interest or not avg_volume or avg_volume <= 0:
            return None
        return short_interest / avg_volume
    
    def get_stock_data(self, ticker: str, days: int = 90) -> StockData:
        """Get complete stock data with calculated fields"""
        try:
            price_data = self.get_aggregates(ticker, days)
            avg_volume = price_data['volume'].mean() if price_data is not None else None
            
            details = self.get_ticker_details(ticker)
            fundamentals = self.get_financials(ticker, price_data)
            
            short_interest, settlement_date, short_interest_ratio = self.get_short_interest(ticker)
            
            # Calculate ratio if not provided
            if short_interest_ratio is None and short_interest is not None and avg_volume is not None:
                short_interest_ratio = self.calculate_short_interest_ratio(short_interest, avg_volume)
            
            return StockData(
                ticker=ticker,
                price_data=price_data,
                fundamentals=fundamentals,
                details=details,
                short_interest=short_interest,
                short_interest_date=settlement_date,
                short_interest_ratio=short_interest_ratio
            )
            
        except Exception as e:
            logger.error(f"Error getting data for {ticker}: {str(e)}")
            return StockData(
                ticker=ticker,
                price_data=None,
                fundamentals=None,
                details=None,
                short_interest=None,
                short_interest_date=None,
                short_interest_ratio=None
            )

class MarketRegimeDetector:
    """Detects current market regime using SPY technicals"""
    
    def __init__(self, polygon_client: PolygonIOClient):
        self.polygon = polygon_client
        self.spy_data = None
        
    def initialize(self) -> bool:
        """Load SPY data from Polygon.io with proper error handling"""
        logger.info("Initializing market regime detector...")
        spy_data = self.polygon.get_aggregates("SPY", days=200)
        if spy_data is None or spy_data.empty:
            logger.error("Failed to get SPY data for regime detection")
            return False
            
        self.spy_data = spy_data
        self._calculate_indicators()
        return True
    
    def _calculate_indicators(self):
        """Calculate technical indicators for regime detection"""
        # Ensure we have the required columns
        if not all(col in self.spy_data.columns for col in ['close', 'high', 'low', 'volume']):
            logger.error("Missing required columns in SPY data")
            return
            
        closes = self.spy_data['close'].values
        
        # Momentum Indicators
        self.spy_data['sma_50'] = self.spy_data['close'].rolling(50).mean()
        self.spy_data['sma_200'] = self.spy_data['close'].rolling(200).mean()
        
        # Calculate 14-day momentum only if we have enough data
        if len(closes) >= 14:
            self.spy_data['momentum_14'] = closes[-1] / closes[-14] - 1
        else:
            self.spy_data['momentum_14'] = 0
            
        # Volatility Indicators
        self.spy_data['atr_14'] = self._calculate_atr(14)
        self.spy_data['volatility_30'] = self.spy_data['close'].pct_change().rolling(30).std()
        
        # Volume Indicators
        self.spy_data['volume_ma_21'] = self.spy_data['volume'].rolling(21).mean()
        self.spy_data['volume_ratio'] = self.spy_data['volume'] / self.spy_data['volume_ma_21']
    
    def _calculate_atr(self, window: int) -> pd.Series:
        """Calculate Average True Range with proper column references"""
        high_low = self.spy_data['high'] - self.spy_data['low']
        high_close = np.abs(self.spy_data['high'] - self.spy_data['close'].shift())
        low_close = np.abs(self.spy_data['low'] - self.spy_data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window).mean()
    
    def detect_regime(self) -> Dict[MarketRegime, float]:
        """Detect current market regime with confidence scores"""
        if self.spy_data is None:
            return {"bull": 0.5, "bear": 0.3, "recovery": 0.1, "consolidation": 0.1, "high_volatility": 0.0}
        
        recent = self.spy_data.iloc[-1]
        momentum = recent['momentum_14']
        sma_ratio = recent['sma_50'] / recent['sma_200']
        atr = recent['atr_14']
        vol_ratio = recent['volume_ratio']
        
        # Calculate regime probabilities
        regimes = {
            "bull": self._bull_market_confidence(momentum, sma_ratio, vol_ratio),
            "bear": self._bear_market_confidence(momentum, sma_ratio, vol_ratio),
            "recovery": self._recovery_confidence(momentum, sma_ratio),
            "consolidation": self._consolidation_confidence(atr, momentum),
            "high_volatility": self._high_vol_confidence(atr)
        }
        
        # Normalize to sum to 1
        total = sum(regimes.values())
        return {k: v/total for k, v in regimes.items()}
    
    def _bull_market_confidence(self, momentum: float, sma_ratio: float, vol_ratio: float) -> float:
        """Confidence score for bull market conditions"""
        score = 0
        if sma_ratio > 1.0:
            score += 0.4
        if momentum > 0.02:
            score += 0.3 * min(momentum / 0.05, 1)
        if vol_ratio > 1.1:
            score += 0.3
        return score
    
    def _bear_market_confidence(self, momentum: float, sma_ratio: float, vol_ratio: float) -> float:
        """Confidence score for bear market conditions"""
        score = 0
        if sma_ratio < 1.0:
            score += 0.4
        if momentum < -0.02:
            score += 0.3 * min(abs(momentum) / 0.05, 1)
        if vol_ratio > 1.1:
            score += 0.3
        return score
    
    def _recovery_confidence(self, momentum: float, sma_ratio: float) -> float:
        """Confidence score for recovery after downturn"""
        current_above = sma_ratio > 1.0
        prev_above = self.spy_data.iloc[-2]['sma_50'] / self.spy_data.iloc[-2]['sma_200'] > 1.0
        if not prev_above and current_above and momentum > 0:
            return 0.8
        return 0
    
    def _consolidation_confidence(self, atr: float, momentum: float) -> float:
        """Confidence score for consolidation/low volatility"""
        median_atr = self.spy_data['atr_14'].rolling(126).median().iloc[-1]
        if atr < median_atr * 0.7 and abs(momentum) < 0.01:
            return 0.9
        return 0
    
    def _high_vol_confidence(self, atr: float) -> float:
        """Confidence score for high volatility regime"""
        median_atr = self.spy_data['atr_14'].rolling(126).median().iloc[-1]
        if atr > median_atr * 1.5:
            return 0.9
        return 0

    def get_current_regime(self) -> MarketRegime:
        """Get the most likely current market regime"""
        regimes = self.detect_regime()
        return max(regimes.items(), key=lambda x: x[1])[0]

class DynamicGrowthScorer:
    """Scores stocks based on growth potential with regime awareness"""
    
    def __init__(self, market_detector: MarketRegimeDetector):
        self.market_detector = market_detector
        self.base_weights = {
            'momentum': 0.30,
            'volume': 0.15,
            'fundamentals': 0.40, 
            'relative_strength': 0.15
        }
        self.quality_metrics = {
            'positive_cashflow': lambda f: f.get('operatingCashFlow', 0) > 0,
            'reasonable_debt': lambda f: f.get('debtToEquity', 1) < 1.0,
            'positive_fcf': lambda f: f.get('freeCashFlow', 0) > 0,
            'profitable': lambda f: f.get('profitMargin', 0) > 0,
            'growing_revenue': lambda f: f.get('revenueGrowth', 0) > 0
        }

    def calculate_growth_score(self, stock_data: StockData) -> float:
        """Calculate composite growth score with robust error handling"""
        try:
            # Data validation
            if stock_data.price_data is None or len(stock_data.price_data) < 30:
                return 0.0
                
            # Parse fundamentals with safe defaults
            ticker = stock_data.ticker
            rev_growth = self._parse_growth_rate(stock_data.fundamentals.get('revenueGrowth'))
            earn_growth = self._parse_growth_rate(stock_data.fundamentals.get('earningsGrowth'))
            profit_margin = self._parse_percentage(stock_data.fundamentals.get('profitMargin'))
            fcf_growth = self._parse_growth_rate(stock_data.fundamentals.get('freeCashFlowGrowth'))

            # Calculate components
            momentum_score = self._calculate_momentum(stock_data.price_data['close'].values)
            volume_score = self._calculate_volume(stock_data.price_data['volume'].values)
            fundamental_score = self._calculate_fundamental(
                rev_growth, earn_growth, profit_margin, fcf_growth
            )
            rsi_score = self._calculate_rsi(stock_data.price_data['close'].values)
            quality_score = self._calculate_quality_score(stock_data.fundamentals)

            # Apply quality gate
            if quality_score < 0.4:
                logger.debug(f"Rejected {ticker}: Low quality score {quality_score:.2f}")
                return 0.0

            # Get regime-adjusted weights
            weights = self._get_regime_adjusted_weights()
            
            # Composite score calculation
            total_score = (
                weights['momentum'] * momentum_score +
                weights['volume'] * volume_score +
                weights['fundamentals'] * fundamental_score +
                weights['relative_strength'] * rsi_score
            )

            # Debug logging
            logger.debug(
                f"\n{ticker} Score Components:\n"
                f"Momentum: {momentum_score:.1f} (Weight: {weights['momentum']:.2f})\n"
                f"Volume: {volume_score:.1f} (Weight: {weights['volume']:.2f})\n"
                f"Fundamentals: {fundamental_score:.1f} (Weight: {weights['fundamentals']:.2f})\n"
                f"RSI: {rsi_score:.1f} (Weight: {weights['relative_strength']:.2f})\n"
                f"Quality: {quality_score:.1%}\n"
                f"TOTAL: {total_score:.1f}\n"
                f"Regime: {self.market_detector.get_current_regime()}"
            )

            return max(1.0, total_score)  # Ensure minimum score of 1

        except Exception as e:
            logger.error(f"Error scoring {stock_data.ticker}: {str(e)}")
            return 0.0

    def _calculate_momentum(self, closes: np.ndarray) -> float:
        """Calculate weighted momentum score with downside protection"""
        mom_5d = self._safe_price_change(closes, 5)
        mom_20d = self._safe_price_change(closes, 20)
        mom_60d = self._safe_price_change(closes, 60)
        return (0.3 * mom_5d + 0.4 * mom_20d + 0.3 * mom_60d) * 100

    def _safe_price_change(self, prices: np.ndarray, days: int) -> float:
        """Calculate price change with safe lookback"""
        if len(prices) < days:
            return 0.0
        return max(-0.2, (prices[-1] / prices[-days] - 1))  # Cap drawdown at -20%

    def _calculate_volume(self, volumes: np.ndarray) -> float:
        """Calculate volume trend score"""
        lookback = min(10, len(volumes))
        avg_volume = np.mean(volumes[-lookback:]) if lookback > 0 else volumes[-1]
        return np.log1p(volumes[-1] / avg_volume) * 20  # Scaled log transform

    def _calculate_fundamental(self, rev_growth: float, earn_growth: float,
                             profit_margin: float, fcf_growth: float) -> float:
        """Calculate fundamental score with normalization"""
        return (
            0.4 * min(rev_growth, 100) +  # Cap at 100% growth
            0.3 * min(earn_growth, 200) +  # Allow higher EPS growth
            0.2 * min(profit_margin, 30) +  # Cap margin at 30%
            0.1 * min(fcf_growth, 150)      # Cap FCF growth at 150%
        )

    def _calculate_rsi(self, prices: np.ndarray, window: int = 14) -> float:
        """Calculate RSI score component (50-100 scale)"""
        if len(prices) < window + 1:
            return 50.0
            
        delta = pd.Series(prices).diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window).mean().iloc[-1]
        avg_loss = loss.rolling(window).mean().iloc[-1]
        
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return min(100, max(0, (60 - abs(rsi - 50)) * 2))  # Convert to 0-100 scale

    def _calculate_quality_score(self, fundamentals: Dict) -> float:
        """Calculate quality score (0-1) using configured metrics"""
        passed = 0
        for name, metric in self.quality_metrics.items():
            try:
                if metric(fundamentals):
                    passed += 1
            except:
                continue
        return passed / len(self.quality_metrics)

    def _get_regime_adjusted_weights(self) -> Dict[str, float]:
        """Dynamic weight adjustments based on market regime"""
        regime = self.market_detector.get_current_regime()
        weights = self.base_weights.copy()
        
        regime_adjustments = {
            "bull": {'momentum': 1.3, 'relative_strength': 0.9},
            "bear": {'fundamentals': 1.4, 'momentum': 0.7},
            "recovery": {'volume': 1.5, 'momentum': 1.2},
            "high_volatility": {'relative_strength': 1.6, 'momentum': 0.6},
            "consolidation": {'relative_strength': 1.3, 'momentum': 1.1}
        }.get(regime, {})

        for factor, multiplier in regime_adjustments.items():
            weights[factor] *= multiplier
        
        # Normalize weights
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}

    def _parse_growth_rate(self, value) -> float:
        """Parse growth rate values from API response"""
        if value is None:
            return 0.0
        if isinstance(value, str):
            value = float(value.strip('%'))
        return value * 100 if abs(value) < 1 else value

    def _parse_percentage(self, value) -> float:
        """Parse percentage values from API response"""
        if value is None:
            return 0.0
        if isinstance(value, str):
            value = float(value.strip('%'))
        return value if value <= 1 else value / 100

def generate_recommendations(regime: MarketRegime):
    """Generate trading recommendations based on market regime"""
    print("\nRecommended Actions Based on Market Regime:")
    
    if regime == "bull":
        print("- Focus on high momentum growth stocks")
        print("- Consider trailing stop losses at 8-10% below current price")
        print("- Look for breakouts to new 52-week highs")
    elif regime == "bear":
        print("- Prioritize stocks with strong fundamentals (high revenue/earnings growth)")
        print("- Use smaller position sizes (25-50% of normal)")
        print("- Consider defensive sectors (utilities, consumer staples)")
    elif regime == "recovery":
        print("- Look for stocks showing early strength with improving volume")
        print("- Focus on stocks trading above their 50-day moving average")
        print("- Consider partial positions to start")
    elif regime == "consolidation":
        print("- Watch for breakout candidates with tightening ranges")
        print("- Consider buying support levels and selling resistance")
        print("- Focus on stocks with relative strength")
    elif regime == "high_volatility":
        print("- Focus on stocks with stable earnings and low debt")
        print("- Use smaller position sizes and wider stops")
        print("- Consider buying volatility dips rather than chasing strength")
    
    print("- Always use proper risk management regardless of market regime")

class StockScanner:
    """Main stock scanning application"""
    
    def __init__(self, polygon_api_key: str):
        self.polygon_client = PolygonIOClient(polygon_api_key)
        self.ftp_client = NASDAQTraderFTP()
        self.regime_detector = MarketRegimeDetector(self.polygon_client)
        self.growth_scorer = None
        
    def initialize(self) -> bool:
        """Initialize all components"""
        if not self.regime_detector.initialize():
            return False
        self.growth_scorer = DynamicGrowthScorer(self.regime_detector)
        return True
    
    def run_scan(self, max_tickers: int = 100, min_volume: int = 50000) -> List[Dict]:
        """Run the full stock scanning process"""
        # Get current market regime
        current_regime = self.regime_detector.get_current_regime()
        regime_scores = self.regime_detector.detect_regime()
        
        print("\n📈 Current Market Conditions:")
        print("=========================")
        print(f"Primary Regime: {current_regime}")
        for regime, score in regime_scores.items():
            print(f"  {regime}: {score:.1%} confidence")
        print("")
        
        # Get common stocks
        print("🔍 Fetching common stock tickers from NASDAQ FTP...")
        try:
            common_stocks = self.ftp_client.get_common_stocks()
            if not common_stocks:
                print("❌ Failed to fetch common stocks - FTP connection or parsing error")
                return []
        except Exception as e:
            print(f"❌ Fatal FTP error: {str(e)}")
            return []
        
        print(f"✅ Found {len(common_stocks)} potential common stocks")
        
        # Apply limit if set
        if max_tickers:
            common_stocks = common_stocks[:max_tickers]
            print(f"⏳ Processing limited to {max_tickers} tickers")
        
        # Process stocks
        qualified_stocks = []
        
        for i, ticker in enumerate(common_stocks, 1):
            print(f"\n🔎 Analyzing {ticker} ({i}/{len(common_stocks)})")
            
            try:
                # Get all stock data in one call
                stock_data = self.polygon_client.get_stock_data(ticker)
                if stock_data.price_data is None:
                    print(f"❌ Rejected {ticker}: Missing price data")
                    continue
                    
                # Check volume filter
                avg_volume = stock_data.price_data['volume'].mean()
                if avg_volume < min_volume:
                    print(f"❌ Rejected {ticker}: Volume {avg_volume:,.0f} below minimum {min_volume:,}")
                    continue
                
                # Calculate growth score
                score = self.growth_scorer.calculate_growth_score(stock_data)
                if score <= 0:
                    print(f"❌ Rejected {ticker}: Low growth score ({score:.1f})")
                    continue
                
                # Stock qualified!
                print(f"✅ QUALIFIED: {ticker} with score {score:.1f}")
                qualified_stocks.append({
                    'ticker': ticker,
                    'score': score,
                    'price': stock_data.price_data['close'].iloc[-1],
                    'volume': avg_volume,
                    'mom_20d': (stock_data.price_data['close'].iloc[-1] / stock_data.price_data['close'].iloc[-20] - 1) * 100 
                               if len(stock_data.price_data) >= 20 else 0,
                    'rev_growth': stock_data.fundamentals.get('revenue_growth', 0),
                    'eps_growth': stock_data.fundamentals.get('earnings_growth', 0),
                    'pe_ratio': stock_data.fundamentals.get('pe_ratio', 0),
                    'short_interest': stock_data.short_interest,
                    'short_interest_date': stock_data.short_interest_date.strftime('%Y-%m-%d') if stock_data.short_interest_date else None,
                    'short_interest_ratio': stock_data.short_interest_ratio,
                    'fcf_growth': stock_data.fundamentals.get('free_cash_flow_growth', 0),
                    'quality_score': self.growth_scorer._calculate_quality_score(stock_data.fundamentals)
                })
                
            except Exception as e:
                print(f"❌ Rejected {ticker}: Unexpected error - {str(e)}")
                continue
        
        return qualified_stocks
    
    def display_results(self, qualified_stocks: List[Dict], top_n: int = 10):
        """Display the scan results"""
        print("\n📊 Final Results:")
        print("================")
        print(f"Qualified stocks: {len(qualified_stocks)}")
        
        if qualified_stocks:
            # Sort qualified stocks
            qualified_stocks.sort(key=lambda x: x['score'], reverse=True)
            
            # Display top stocks
            print("\n🏆 Top Growth Stocks:")
            print("==================")
            print(f"Market Regime: {self.regime_detector.get_current_regime()}")
            print("")
            
            for idx, stock in enumerate(qualified_stocks[:top_n], 1):
                print(
                    f"{idx}. {stock['ticker']}: \n"
                    f"   Score: {stock['score']:.1f} \n"
                    f"   Price: ${stock['price']:.2f} \n"
                    f"   Volume: {stock['volume']:,.0f} \n"
                    f"   20D Momentum: {stock['mom_20d']:.1f}% \n"
                    f"   Revenue Growth: {stock['rev_growth']:.1%} \n"
                    f"   EPS Growth: {stock['eps_growth']:.1%} \n"
                    f"   FCF Growth: {stock['fcf_growth']:.1%} \n"
                    f"   P/E Ratio: {stock['pe_ratio']:.1f} \n"
                    f"   Short Interest: {stock['short_interest']:,.0f} shares (as of {stock['short_interest_date']}) \n"
                    f"   Short Interest Ratio: {stock['short_interest_ratio']:.2f} days to cover \n"
                    f"   Quality Score: {stock['quality_score']:.0%}\n"
                )
            
            # Generate regime-specific recommendations
            generate_recommendations(self.regime_detector.get_current_regime())

def main():
    # Configuration
    POLYGON_API_KEY = "OZzn0oK0H2yG6rpIvVhGfgXgnUTrL31z"
    MAX_TICKERS_TO_PROCESS = 100
    TOP_N_STOCKS = 10
    MIN_VOLUME = 50000
    
    # Initialize scanner
    scanner = StockScanner(POLYGON_API_KEY)
    if not scanner.initialize():
        print("❌ Failed to initialize scanner")
        return
    
    # Run scan
    qualified_stocks = scanner.run_scan(
        max_tickers=MAX_TICKERS_TO_PROCESS,
        min_volume=MIN_VOLUME
    )
    
    # Display results
    scanner.display_results(qualified_stocks, top_n=TOP_N_STOCKS)
    
    print(f"\n🎉 Processing complete. Analyzed {MAX_TICKERS_TO_PROCESS} stocks.")

if __name__ == "__main__":
    main()