import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
from newsapi import NewsApiClient
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import time
import warnings
from typing import List, Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
from collections import defaultdict
import concurrent.futures
from tqdm import tqdm
import duckdb
import os
import pytz
import requests
import random
import ftplib
import io
import re
import json
import pickle
from tenacity import retry, stop_after_attempt, wait_exponential

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
nltk.download('vader_lexicon', quiet=True)

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
    
    def _get_ftp_file(self, filename: str) -> pd.DataFrame:
        """Download and parse a file from NASDAQ FTP"""
        cache_file = os.path.join(self.cache_dir, f"{filename}.parquet")
        
        # Check cache first
        if os.path.exists(cache_file):
            cache_time = os.path.getmtime(cache_file)
            if (time.time() - cache_time) < 86400:  # 1 day cache
                return pd.read_parquet(cache_file)
        
        ftp = self._connect_ftp()
        try:
            # Download file to memory
            with io.BytesIO() as buffer:
                ftp.retrbinary(f"RETR {filename}", buffer.write)
                buffer.seek(0)
                
                # Parse different file types
                if filename == 'nasdaqlisted.txt':
                    df = self._parse_nasdaq_listed(buffer)
                elif filename == 'otherlisted.txt':
                    df = self._parse_other_listed(buffer)
                elif filename == 'nasdaqtraded.txt':
                    df = self._parse_nasdaq_traded(buffer)
                else:
                    raise ValueError(f"Unknown file type: {filename}")
                
                # Save to cache
                df.to_parquet(cache_file)
                return df
                
        finally:
            ftp.quit()
    
    def _parse_nasdaq_listed(self, buffer) -> pd.DataFrame:
        """Parse nasdaqlisted.txt file"""
        data = buffer.getvalue().decode('utf-8').splitlines()
        headers = data[0].split('|')
        records = []
        
        for line in data[1:-1]:  # Skip header and footer
            parts = line.split('|')
            if len(parts) == len(headers):
                records.append(parts)
        
        df = pd.DataFrame(records, columns=headers)
        df['Exchange'] = 'NASDAQ'
        return df
    
    def _parse_other_listed(self, buffer) -> pd.DataFrame:
        """Parse otherlisted.txt file"""
        data = buffer.getvalue().decode('utf-8').splitlines()
        headers = data[0].split('|')
        records = []
        
        for line in data[1:-1]:  # Skip header and footer
            parts = line.split('|')
            if len(parts) == len(headers):
                records.append(parts)
        
        df = pd.DataFrame(records, columns=headers)
        if 'ACT Symbol' in df.columns:
            df['Symbol'] = df['ACT Symbol']
        return df
    
    def _parse_nasdaq_traded(self, buffer) -> pd.DataFrame:
        """Parse nasdaqtraded.txt file"""
        data = buffer.getvalue().decode('utf-8').splitlines()
        headers = data[0].split('|')
        records = []
        
        for line in data[1:-1]:  # Skip header and footer
            parts = line.split('|')
            if len(parts) == len(headers):
                records.append(parts)
        
        df = pd.DataFrame(records, columns=headers)
        return df
    
    def get_all_symbols(self) -> pd.DataFrame:
        """Get all symbols from NASDAQ FTP"""
        try:
            nasdaq = self._get_ftp_file('nasdaqlisted.txt')
            other = self._get_ftp_file('otherlisted.txt')
            traded = self._get_ftp_file('nasdaqtraded.txt')
            
            all_symbols = pd.concat([
                nasdaq[['Symbol', 'Security Name', 'Exchange']],
                other[['Symbol', 'Security Name', 'Exchange']]
            ]).drop_duplicates('Symbol')
            
            etfs = traded[traded['ETF'] == 'Y'][['Symbol', 'Security Name']]
            etfs['Exchange'] = 'ETF'
            all_symbols = pd.concat([all_symbols, etfs]).drop_duplicates('Symbol')
            all_symbols['Symbol'] = all_symbols['Symbol'].str.replace(r'[\$\.\^]', '', regex=True)
            
            return all_symbols.sort_values('Symbol')
            
        except Exception as e:
            if self.debug:
                print(f"Error getting symbols from FTP: {e}")
            return pd.DataFrame(columns=['Symbol', 'Security Name', 'Exchange'])
    
    def get_active_symbols(self) -> List[str]:
        """Get only active trading symbols"""
        try:
            nasdaq = self._get_ftp_file('nasdaqlisted.txt')
            other = self._get_ftp_file('otherlisted.txt')
            
            nasdaq_active = nasdaq[nasdaq['Financial Status'] == 'N']['Symbol']
            other_active = other[other['Test Issue'] == 'N']['Symbol']
            
            active_symbols = pd.concat([nasdaq_active, other_active]).unique()
            active_symbols = [s.replace('$', '').replace('.', '') for s in active_symbols]
            
            return sorted(list(set(active_symbols)))
            
        except Exception as e:
            if self.debug:
                print(f"Error getting active symbols: {e}")
            return []
    
    def get_tickers_from_ftp(self) -> List[str]:
        """Direct method to get all tickers from NASDAQ FTP without analysis"""
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
            
            for line in nasdaq_data[1:]:
                parts = line.split('|')
                if len(parts) > 0 and parts[0]:
                    tickers.add(parts[0].replace('$', '').replace('.', ''))
            
            for line in traded_data[1:]:
                parts = line.split('|')
                if len(parts) > 0 and parts[0]:
                    tickers.add(parts[0].replace('$', '').replace('.', ''))
            
            return sorted(tickers)
            
        except Exception as e:
            if self.debug:
                print(f"Error in get_tickers_from_ftp: {e}")
            return []

class HypergrowthScanner:
    def __init__(self, scanner):
        """Initialize the hypergrowth scanner with reference to main scanner"""
        self.scanner = scanner
        self.criteria = {
            'min_revenue_growth': 0.40,  # 40% YoY
            'min_earnings_growth': 0.25,  # 25% YoY
            'min_qoq_growth': 0.15,      # 15% QoQ
            'max_market_cap': 10e9,      # $10B
            'min_institutional_change': 0.30,
            'max_age_years': 5,          # Company age
            'sectors': ['Technology', 'Healthcare', 'Consumer Cyclical'],
            'min_score': 70              # Minimum to qualify as hypergrowth
        }
        
        if self.scanner.debug:
            print("✅ HypergrowthScanner initialized with criteria:")
            print(f" - Min Revenue Growth: {self.criteria['min_revenue_growth']*100}%")
            print(f" - Max Market Cap: ${self.criteria['max_market_cap']/1e9}B")
    
    def _validate_ticker_format(self, ticker: str) -> bool:
        """More strict ticker validation"""
        if not isinstance(ticker, str) or len(ticker) > 5 or len(ticker) < 1:
            return False
        if not ticker.isalpha():  # Only letters allowed
            return False
        if ticker.endswith(('W', 'R', 'U', 'P')):  # Skip warrants, rights, units
            return False
        return True
            
    def detect_hypergrowth(self, ticker: str) -> Optional[Dict]:
        """
        Detect hypergrowth stocks with robust error handling and caching
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
        Returns:
            Dictionary with hypergrowth analysis or None if invalid/error
        """
        # Validate ticker format first
        if not self._validate_ticker_format(ticker):
            return None

        # Cache key and retrieval with error handling
        cache_key = f"hypergrowth_{ticker}"
        try:
            cached_data = self.scanner._get_cached_data(cache_key)
            if cached_data and isinstance(cached_data, dict):
                if self.scanner.debug:
                    print(f"📦 Using cached data for {ticker}")
                return cached_data
        except Exception as e:
            if self.scanner.debug:
                print(f"⚠️ Cache read error for {ticker}: {str(e)}")

        # Initialize stock object with rate limiting
        try:
            self.scanner._rate_limit()
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Basic validation checks
            if not info or not isinstance(info, dict):
                return None
            if info.get('quoteType', '').lower() not in ['equity', 'etf']:
                return None
        except Exception as e:
            if self.scanner.debug:
                print(f"⚠️ Failed to initialize {ticker}: {str(e)}")
            return None

        # Get growth metrics with comprehensive error handling
        try:
            metrics = self._get_growth_metrics(stock, info)
            if not metrics:
                return None

            # Calculate composite growth score (0-100)
            score = min(100, max(0, self._calculate_growth_score(metrics)))
            
            result = {
                'ticker': ticker,
                'is_hypergrowth': score >= self.criteria['min_score'],
                'hypergrowth_score': score,
                'revenue_growth_yoy': metrics.get('revenue_growth_yoy', 0),
                'earnings_growth_yoy': metrics.get('earnings_growth_yoy', 0),
                'revenue_growth_qoq': metrics.get('revenue_growth_qoq', 0),
                'institutional_change': metrics.get('institutional_change', 0),
                'volume_growth': metrics.get('volume_growth', 0),
                'price_growth': metrics.get('price_growth', 0),
                'sector': info.get('sector', 'Unknown'),
                'market_cap': info.get('marketCap'),
                'last_updated': datetime.now().isoformat()
            }

            # Cache results with error handling
            try:
                self.scanner._cache_ticker_data(cache_key, result)
            except Exception as e:
                if self.scanner.debug:
                    print(f"⚠️ Cache write failed for {ticker}: {str(e)}")

            return result

        except Exception as e:
            if self.scanner.debug:
                print(f"⚠️ Hypergrowth analysis failed for {ticker}: {str(e)}")
            return None
    
    def _is_eligible(self, info: Dict) -> bool:
        """Check if stock is eligible for hypergrowth consideration"""
        try:
            # Market cap filter
            market_cap = info.get('marketCap', float('inf'))
            if market_cap > self.criteria['max_market_cap']:
                return False
                
            # Sector filter
            sector = info.get('sector', '')
            if sector not in self.criteria['sectors']:
                return False
                
            # Age filter
            ipo_year = info.get('ipoYear')
            if ipo_year and (datetime.now().year - ipo_year) > self.criteria['max_age_years']:
                return False
                
            return True
        except:
            return False
        
    def _safe_growth_rate(self, series: pd.Series) -> float:
        """
        Safely calculate growth rate from time series data
        Args:
            series: Pandas Series of values
        Returns:
            Growth rate (slope/mean) or 0 if calculation fails
        """
        try:
            if len(series) < 10:
                return 0.0
            
            # Remove outliers using IQR
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            filtered = series[(series >= q1 - 1.5*iqr) & (series <= q3 + 1.5*iqr)]
            
            if len(filtered) < 5:
                return 0.0
                
            x = np.arange(len(filtered))
            y = filtered.values.astype(float)
            slope = np.polyfit(x, y, 1)[0]
            mean = np.nanmean(y)
            
            return slope / mean if mean != 0 else 0.0
        except:
            return 0.0
    
    def _get_growth_metrics(self, stock, info: Dict) -> Dict:
        """
        Calculate growth metrics with robust error handling and data validation
        Args:
            stock: yfinance Ticker object
            info: Stock info dictionary
        Returns:
            Dictionary of growth metrics with default 0 values on error
        """
        metrics = {
            'revenue_growth_yoy': 0,
            'earnings_growth_yoy': 0,
            'revenue_growth_qoq': 0,
            'institutional_change': 0,
            'volume_growth': 0,
            'price_growth': 0
        }

        # 1. Revenue Growth Calculations
        try:
            financials = getattr(stock, 'quarterly_financials', None)
            if financials is not None and 'Total Revenue' in financials.index:
                revenue = financials.loc['Total Revenue']
                
                # Year-over-year growth
                if len(revenue) >= 4:
                    try:
                        current = revenue.iloc[0]
                        previous = revenue.iloc[4]
                        if previous != 0:
                            metrics['revenue_growth_yoy'] = (current - previous) / abs(previous)
                    except:
                        pass
                
                # Quarter-over-quarter growth
                if len(revenue) >= 2:
                    try:
                        current = revenue.iloc[0]
                        previous = revenue.iloc[1]
                        if previous != 0:
                            metrics['revenue_growth_qoq'] = (current - previous) / abs(previous)
                    except:
                        pass
        except Exception as e:
            if self.scanner.debug:
                print(f"⚠️ Revenue growth calc error for {getattr(stock, 'ticker', '?')}: {str(e)}")

        # 2. Earnings Growth Calculations
        try:
            earnings = getattr(stock, 'quarterly_earnings', None)
            if earnings is not None and len(earnings) >= 4:
                try:
                    current = earnings.iloc[0]['Earnings']
                    previous = earnings.iloc[4]['Earnings']
                    if previous != 0:
                        metrics['earnings_growth_yoy'] = (current - previous) / abs(previous)
                except:
                    pass
        except Exception as e:
            if self.scanner.debug:
                print(f"⚠️ Earnings growth calc error for {getattr(stock, 'ticker', '?')}: {str(e)}")

        # 3. Institutional Activity
        try:
            holders = getattr(stock, 'institutional_holders', None)
            if holders is not None and len(holders) >= 2:
                try:
                    current = holders.iloc[0]['Shares']
                    previous = holders.iloc[1]['Shares']
                    if previous != 0:
                        metrics['institutional_change'] = (current - previous) / previous
                except:
                    pass
        except Exception as e:
            if self.scanner.debug:
                print(f"⚠️ Institutional data error for {getattr(stock, 'ticker', '?')}: {str(e)}")

        # 4. Price and Volume Trends
        try:
            hist = stock.history(period='6mo', interval='1d', timeout=10)
            if hist is not None and len(hist) >= 30:
                # Clean data by removing zeros and outliers
                clean_volume = hist['Volume'].replace(0, np.nan).dropna()
                if len(clean_volume) >= 10:
                    metrics['volume_growth'] = self._safe_growth_rate(clean_volume)
                
                clean_price = hist['Close'].replace(0, np.nan).dropna()
                if len(clean_price) >= 10:
                    metrics['price_growth'] = self._safe_growth_rate(clean_price)
        except Exception as e:
            if self.scanner.debug:
                print(f"⚠️ Price/volume calc error for {getattr(stock, 'ticker', '?')}: {str(e)}")

        return metrics

    def _calculate_growth_score(self, metrics: Dict) -> float:
        """Calculate composite growth score (0-100) with weighted components"""
        score = 0
        
        # Revenue growth (max 30 points)
        score += min(30, metrics['revenue_growth_yoy'] * 75)  # 40% growth = 30 points
        score += min(15, metrics['revenue_growth_qoq'] * 100) # 15% growth = 15 points
        
        # Earnings growth (max 20 points)
        score += min(20, metrics['earnings_growth_yoy'] * 80) # 25% growth = 20 points
        
        # Institutional activity (max 15 points)
        score += min(15, metrics['institutional_change'] * 50) # 30% change = 15 points
        
        # Momentum (max 20 points)
        score += min(10, metrics['volume_growth'] * 100)
        score += min(10, metrics['price_growth'] * 100)
        
        return min(100, score)  # Cap at 100
    
    def _calculate_growth_rate(self, series: pd.Series) -> float:
        """Calculate growth rate with robust data validation"""
        try:
            if len(series) < 10:
                return 0
                
            # Remove zeros and outliers
            clean_series = series.replace(0, np.nan).dropna()
            if len(clean_series) < 5:
                return 0
                
            # Use robust linear regression
            x = np.arange(len(clean_series))
            y = clean_series.values
            slope = np.polyfit(x, y, 1)[0]
            avg = np.nanmean(y)
            return slope / avg if avg != 0 else 0
        except:
            return 0

class NASDAQStockScanner:
    def __init__(self, newsapi_key: str = None, debug: bool = True, max_workers: int = 4):
        """Initialize the NASDAQ stock scanner with all components"""
        # Initialize core parameters first
        self.debug = debug
        self.max_workers = max_workers
        self.min_request_interval = 1.5  # seconds between requests
        self.last_request_time = 0
        self.blacklisted_tickers = ['BRK.A', 'BF.A', 'BF.B']  # Known problematic tickers
        
        # Initialize cache system
        self.cache_dir = 'scanner_cache'
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_db = duckdb.connect(database=':memory:')
        self.cache_db.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key VARCHAR PRIMARY KEY,
                value BLOB,
                timestamp TIMESTAMP
            )
        """)
        
        # Initialize components
        self.ftp_client = NASDAQTraderFTP(debug=debug)
        self.hypergrowth_scanner = HypergrowthScanner(self)
        self.newsapi = NewsApiClient(api_key=newsapi_key) if newsapi_key else None
        self.sia = SentimentIntensityAnalyzer()
        self.current_regime = None
        
        # Initialize criteria and parameters
        self._initialize_criteria()
        self._initialize_technical_params()
        self._initialize_classification_params()
        
        # Clean old cache entries
        self._clean_cache()

        if self.debug:
            print("✅ NASDAQStockScanner initialized successfully")
            print(f"📊 Cache system ready | Max workers: {max_workers}")

    def _initialize_criteria(self):
        """Initialize screening criteria with hypergrowth considerations"""
        self.criteria = {
            'price_ranges': {
                'value': (0.50, 20),  # Expanded lower range for hypergrowth
                'growth': (20, 100),
                'premium': (100, 1000)
            },
            'avg_volume': 100e3,
            'rsi_range': (30, 75),
            'roe': 0.10,
            'debt_equity': 2.0,
            'profit_margin': 0.05,
            'operating_margin': 0.08,
            'pe_ratio': 40,
            'peg_ratio': 3.0,
            'ps_ratio': 15,
            'ev_ebitda': 20,
            'fcf_yield': 0.03,
            'current_ratio': 0.8,
            'sales_growth': 0.08,
            'earnings_growth': 0.05,
            'institutional_ownership': 0.10,
            'beta_range': (0.7, 1.5),
            'must_beat_eps': False,
            'hypergrowth': {
                'min_score': 70,
                'min_revenue_growth': 0.40,
                'min_earnings_growth': 0.25,
                'max_market_cap': 10e9
            },
            'sector_weights': {
                'Technology': {
                    'pe_ratio': 50,
                    'peg_ratio': 3.5,
                    'ps_ratio': 20,
                    'profit_margin': 0.03,
                    'sales_growth': 0.15
                },
                'Healthcare': {
                    'pe_ratio': 40,
                    'peg_ratio': 3.0,
                    'ps_ratio': 15,
                    'profit_margin': 0.10,
                    'sales_growth': 0.12
                },
                'Consumer Cyclical': {
                    'pe_ratio': 35,
                    'peg_ratio': 2.5,
                    'ps_ratio': 12,
                    'profit_margin': 0.08,
                    'sales_growth': 0.10
                }
            }
        }

    def _initialize_technical_params(self):
        """Initialize technical analysis parameters"""
        self.technical_params = {
            'min_macd_signal_diff': -0.1,
            'min_obv_trend': 0.02,
            'min_adl_trend': 0.02,
            'max_bollinger_percent': 0.9,
            'min_vwap_support': 0.85,
            'above_20ma': True,
            'above_50ma': True,
            'above_200ma': False,
            'momentum_rsi_buffer': 5,
            'hypergrowth_technical': {
                'min_volume_trend': 0.05,
                'min_price_trend': 0.10,
                'rsi_range': (40, 80)
            }
        }

    def _initialize_classification_params(self):
        """Initialize stock classification parameters"""
        self.classification_params = {
            'fundamental_classes': {
                'elite_quality': {
                    'min_roe': 0.20,
                    'max_debt_equity': 0.5,
                    'min_profit_margin': 0.15,
                    'min_fcf_yield': 0.05
                },
                'growth_at_value': {
                    'min_sales_growth': 0.15,
                    'max_ps_ratio': 5,
                    'max_peg_ratio': 2
                },
                'deep_value': {
                    'max_pe_ratio': 15,
                    'max_pb_ratio': 2,
                    'min_fcf_yield': 0.08
                },
                'hypergrowth': {
                    'min_sales_growth': 0.30,
                    'max_ps_ratio': 10,
                    'min_earnings_growth': 0.20
                }
            },
            'technical_signals': {
                'bull_confirm': {
                    'min_rsi': 40,
                    'max_rsi': 70,
                    'min_price_trend': 0.02,
                    'above_ma': '50ma'
                },
                'bear_reversal': {
                    'max_rsi': 40,
                    'min_volume_trend': 0.1,
                    'above_ma': '200ma'
                },
                'hypergrowth_breakout': {
                    'min_rsi': 30,
                    'max_rsi': 60,
                    'min_volume_trend': 0.15,
                    'above_ma': '20ma'
                }
            },
            'base_success_rates': {
                'elite_quality': 0.65,
                'growth_at_value': 0.55,
                'deep_value': 0.45,
                'hypergrowth': 0.60,
                'bull_confirm': 0.6,
                'bear_reversal': 0.4,
                'hypergrowth_breakout': 0.7
            }
        }

    def _init_cache_system(self):
        """Initialize the caching system with robust error handling"""
        self.cache_dir = 'scanner_cache'
        os.makedirs(self.cache_dir, exist_ok=True)
        try:
            self.cache_db = duckdb.connect(database=os.path.join(self.cache_dir, 'cache.db'))
            self.cache_db.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key VARCHAR PRIMARY KEY,
                    value BLOB,
                    timestamp TIMESTAMP
                )
            """)
        except Exception as e:
            if self.debug:
                print(f"Cache initialization error: {str(e)}")
            # Fallback to in-memory database
            self.cache_db = duckdb.connect(database=':memory:')
            self.cache_db.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key VARCHAR PRIMARY KEY,
                    value BLOB,
                    timestamp TIMESTAMP
                )
            """)

    def _get_cached_data(self, key: str) -> Optional[Any]:
        """Safe cache retrieval with connection handling"""
        try:
            result = None
            try:
                result = self.cache_db.execute("""
                    SELECT value FROM cache 
                    WHERE key = ? AND timestamp > CURRENT_TIMESTAMP - INTERVAL '1 day'
                """, [key]).fetchone()
            except:
                # Reconnect if connection failed
                self._init_cache_system()
                result = self.cache_db.execute("""
                    SELECT value FROM cache 
                    WHERE key = ? AND timestamp > CURRENT_TIMESTAMP - INTERVAL '1 day'
                """, [key]).fetchone()
            
            if result:
                return pickle.loads(result[0])
            return None
        except Exception as e:
            if self.debug:
                print(f"Cache read error for {key}: {str(e)}")
            return None

    def _cache_ticker_data(self, key: str, value: Any) -> None:
        """Safe cache storage with connection handling"""
        try:
            serialized = pickle.dumps(value)
            try:
                self.cache_db.execute("""
                    INSERT OR REPLACE INTO cache VALUES (?, ?, CURRENT_TIMESTAMP)
                """, [key, serialized])
            except:
                # Reconnect if connection failed
                self._init_cache_system()
                self.cache_db.execute("""
                    INSERT OR REPLACE INTO cache VALUES (?, ?, CURRENT_TIMESTAMP)
                """, [key, serialized])
        except Exception as e:
            if self.debug:
                print(f"Cache write error for {key}: {str(e)}")

    def _clean_cache(self) -> None:
        """Safe cache cleanup"""
        try:
            self.cache_db.execute("""
                DELETE FROM cache WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '2 days'
            """)
            self.cache_db.execute("VACUUM")
        except Exception as e:
            if self.debug:
                print(f"Cache cleanup error: {str(e)}")
            self._init_cache_system()

    def _rate_limit(self, min_interval: float = None):
        """Enhanced rate limiting with jitter"""
        if min_interval is None:
            min_interval = self.min_request_interval
            
        elapsed = time.time() - self.last_request_time
        if elapsed < min_interval:
            # Add random jitter to avoid being blocked
            sleep_time = min_interval - elapsed + random.uniform(0, 0.5)
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _prescan_ticker(self, ticker: str) -> Optional[Dict]:
        """Enhanced pre-scan with better error handling"""
        try:
            # Skip if ticker format is invalid
            if not self.hypergrowth_scanner._validate_ticker_format(ticker):
                if self.debug:
                    print(f"Skipping invalid ticker: {ticker}")
                return None
                
            # Get basic info with retry logic
            for attempt in range(3):
                try:
                    self._rate_limit()
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    # Skip if no valid info or wrong security type
                    if not info or info.get('quoteType', '').lower() not in ['equity', 'etf']:
                        return None
                        
                    # Check liquidity requirements
                    avg_volume = info.get('averageVolume', 0)
                    if avg_volume < self.criteria['avg_volume'] * 0.3:  # 30% of threshold
                        return None
                        
                    return self._create_normal_result(ticker, info)
                    
                except Exception as e:
                    if "404" in str(e):
                        return None  # Skip 404 errors
                    if attempt == 2:  # Final attempt
                        if self.debug:
                            print(f"Failed to scan {ticker} after 3 attempts")
                        return None
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
                    
        except Exception as e:
            if self.debug:
                print(f"Error in pre-scan for {ticker}: {str(e)}")
            return None
    
    def _create_hypergrowth_result(self, ticker: str, basic_info: Dict, hypergrowth: Dict) -> Dict:
        """Create result for hypergrowth stocks with relaxed criteria"""
        return {
            'ticker': ticker,
            'company': basic_info.get('longName', ticker),
            'sector': basic_info.get('sector', 'Unknown'),
            'industry': basic_info.get('industry', 'Unknown'),
            'price': basic_info.get('currentPrice') or basic_info.get('regularMarketPrice'),
            'volume': basic_info.get('averageVolume', 0),
            'market_cap': basic_info.get('marketCap'),
            'pe_ratio': basic_info.get('trailingPE'),
            'exchange': basic_info.get('exchange', 'Unknown'),
            'is_hypergrowth': True,
            'hypergrowth_score': hypergrowth['hypergrowth_score'],
            'revenue_growth': hypergrowth['revenue_growth_yoy'],
            'earnings_growth': hypergrowth['earnings_growth_yoy'],
            'relaxed_filters': True
        }

    def _safe_compare(self, value, threshold, comp_type='min', name="", rejection_reasons=None):
        """Safe comparison helper with error handling"""
        if rejection_reasons is None:
            rejection_reasons = []
        
        try:
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                rejection_reasons.append(f"Missing {name}")
                return False
                
            value = float(value)
            if isinstance(threshold, tuple):
                threshold = tuple(float(t) for t in threshold)
            else:
                threshold = float(threshold)
                
            if comp_type == 'min':
                result = value >= threshold
                msg = f"{name} {value:.2f} < {threshold:.2f}" if not result else ""
            elif comp_type == 'max':
                result = value <= threshold
                msg = f"{name} {value:.2f} > {threshold:.2f}" if not result else ""
            elif comp_type == 'range':
                result = threshold[0] <= value <= threshold[1]
                msg = f"{name} {value:.2f} not in {threshold}" if not result else ""
            elif comp_type == 'bool':
                result = bool(value) == bool(threshold)
                msg = f"{name} condition not met" if not result else ""
            else:
                return False
                
            if not result and msg:
                rejection_reasons.append(msg)
            return result
            
        except Exception as e:
            rejection_reasons.append(f"Comparison error for {name}: {str(e)}")
            return False

    def get_nasdaq_listings(self, exchange='NASDAQ'):
        """More robust NASDAQ API fetcher with retries"""
        self._rate_limit()
        url = f"https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&exchange={exchange}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nasdaq.com/',
            'Origin': 'https://www.nasdaq.com'
        }
        
        for attempt in range(3):
            try:
                response = requests.get(url, headers=headers, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data:
                        if 'rows' in data['data']:
                            return [stock['symbol'] for stock in data['data']['rows'] if 'symbol' in stock]
                        elif 'table' in data['data'] and 'rows' in data['data']['table']:
                            return [stock['symbol'] for stock in data['data']['table']['rows'] if 'symbol' in stock]
                elif response.status_code == 403:
                    if self.debug:
                        print(f"Access denied for {exchange}, trying different headers...")
                    headers['User-Agent'] = random.choice([
                        'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
                        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
                        'Mozilla/5.0 (X11; Linux x86_64)'
                    ])
            except Exception as e:
                if attempt == 2:
                    if self.debug:
                        print(f"Final attempt failed for {exchange}: {e}")
                    return []
                time.sleep(2)
        return []

    def get_all_tickers(self) -> List[str]:
        """Get tickers from NASDAQ FTP as primary source with API fallback"""
        try:
            symbols = self.ftp_client.get_tickers_from_ftp()
            if symbols and len(symbols) > 1000:
                if self.debug:
                    print(f"Retrieved {len(symbols)} active symbols from NASDAQ FTP")
                return symbols
            
            symbols = self.ftp_client.get_active_symbols()
            if symbols and len(symbols) > 1000:
                if self.debug:
                    print(f"Retrieved {len(symbols)} active symbols via get_active_symbols")
                return symbols
            
            if self.debug:
                print("Falling back to API for ticker list")
            return self.get_all_tickers_api_fallback()
            
        except Exception as e:
            if self.debug:
                print(f"Error in get_all_tickers: {e}")
            return self.get_all_tickers_api_fallback()
    
    def get_all_tickers_api_fallback(self) -> List[str]:
        """Fallback method to get tickers from API when FTP fails"""
        cache_file = f"{self.cache_dir}/all_tickers.parquet"
        
        if os.path.exists(cache_file):
            try:
                cached = pd.read_parquet(cache_file)
                cache_time = pd.to_datetime(cached['timestamp'].iloc[0])
                if (datetime.now() - cache_time).days < 1:
                    return cached['ticker'].tolist()
            except:
                pass
        
        if self.debug:
            print("Fetching listings from all major exchanges...")
        
        exchanges = ['NASDAQ', 'NYSE', 'AMEX']
        all_tickers = []
        
        for exchange in exchanges:
            try:
                tickers = self.get_nasdaq_listings(exchange)
                if tickers:
                    all_tickers.extend(tickers)
                    if self.debug:
                        print(f"Found {len(tickers)} tickers on {exchange}")
            except Exception as e:
                if self.debug:
                    print(f"Error fetching {exchange} listings: {e}")
        
        if not all_tickers:
            if self.debug:
                print("Using comprehensive fallback ticker list")
            all_tickers = [
                'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA',
                'JPM', 'V', 'WMT', 'PG', 'JNJ', 'XOM', 'CVX', 'HD', 'MA',
                'BAC', 'DIS', 'NFLX', 'INTC', 'CSCO', 'PEP', 'KO', 'ABT',
                'TMO', 'AVGO', 'QCOM', 'AMD', 'ADBE', 'CRM', 'NKE', 'COST',
                'PYPL', 'SBUX', 'MDT', 'BMY', 'TXN', 'AMGN', 'GILD', 'BKNG',
                'INTU', 'ISRG', 'ADI', 'MU', 'NOW', 'LRCX', 'REGN', 'CHTR'
            ]
        else:
            all_tickers = list(set(all_tickers))
        
        pd.DataFrame({
            'ticker': all_tickers,
            'timestamp': datetime.now()
        }).to_parquet(cache_file)
        
        if self.debug:
            print(f"Total tickers to scan: {len(all_tickers)}")
        
        return all_tickers

    def get_new_listings(self, days=60) -> List[str]:
        """Get recent IPOs from NASDAQ with improved error handling"""
        self._rate_limit()
        url = "https://api.nasdaq.com/api/ipo/calendar"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data:
                    if 'pricerange' in data['data'] and 'rows' in data['data']['pricerange']:
                        ipos = data['data']['pricerange']['rows']
                    elif 'rows' in data['data']:
                        ipos = data['data']['rows']
                    else:
                        return []
                        
                    cutoff_date = datetime.now() - timedelta(days=days)
                    new_tickers = []
                    for ipo in ipos:
                        try:
                            if 'symbol' in ipo and ipo['symbol']:
                                if ('expectedDate' in ipo and ipo['expectedDate'] and 
                                    pd.to_datetime(ipo['expectedDate']) >= cutoff_date):
                                    new_tickers.append(ipo['symbol'])
                                elif ('filedDate' in ipo and ipo['filedDate'] and 
                                      pd.to_datetime(ipo['filedDate']) >= cutoff_date):
                                    new_tickers.append(ipo['symbol'])
                        except:
                            continue
                    return new_tickers
        except Exception as e:
            if self.debug:
                print(f"Error fetching new listings: {e}")
        return []

    def detect_market_regime(self, benchmark='SPY', lookback=200) -> str:
        """Enhanced market regime detection with rate limiting"""
        try:
            # Rate limiting
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
            self.last_request_time = time.time()
            
            stock = yf.Ticker(benchmark)
            hist = stock.history(period=f'{lookback}d', timeout=10)
            
            if len(hist) < lookback:
                return "neutral"  # Fallback to neutral if insufficient data
                
            # Calculate indicators
            sma_50 = hist['Close'].rolling(50).mean()
            sma_200 = hist['Close'].rolling(200).mean()
            current_close = hist['Close'].iloc[-1]
            
            price_ratio = current_close / sma_200.iloc[-1]
            ma_ratio = sma_50.iloc[-1] / sma_200.iloc[-1]
            volatility = hist['Close'].pct_change().std() * np.sqrt(252)
            
            # Regime determination
            if price_ratio > 1.10 and ma_ratio > 1.05 and volatility < 0.20:
                return "bull"
            elif price_ratio < 0.90 and ma_ratio < 0.95 and volatility > 0.25:
                return "bear"
            elif 0.95 < price_ratio < 1.10 and 0.97 < ma_ratio < 1.03:
                return "neutral"
            else:
                return "transitional"
                
        except Exception as e:
            if "Too Many Requests" in str(e):
                if self.debug:
                    print("Yahoo Finance rate limited - using cached regime")
                return self.current_regime or "neutral"  # Fallback to last known regime
            if self.debug:
                print(f"Error detecting market regime: {e}")
            return "neutral"  # Default to neutral on error

    def _adjust_criteria_for_regime(self, regime: str):
        """Dynamic criteria adjustment based on market regime"""
        if regime == "bull":
            self.criteria['pe_ratio'] = 40
            self.criteria['rsi_range'] = (35, 80)
            self.technical_params['above_200ma'] = False
            self.criteria['institutional_ownership'] = 0.10
            self.criteria['sales_growth'] = 0.15
            self.criteria['hyper_growth_threshold'] = 0.20
            self.criteria['growth_stock_multipliers']['sales_growth'] = 3.0
            
            # Adjust technical signals for bull market
            self.classification_params['technical_signals']['bull_confirm']['min_rsi'] = 40
            self.classification_params['technical_signals']['bull_confirm']['max_rsi'] = 70
            
        elif regime == "bear":
            self.criteria['pe_ratio'] = 25
            self.criteria['debt_equity'] = 1.0
            self.technical_params['above_200ma'] = True
            self.criteria['institutional_ownership'] = 0.25
            self.criteria['profit_margin'] = 0.05
            self.criteria['hyper_growth_threshold'] = 0.30
            self.criteria['growth_stock_multipliers']['sales_growth'] = 2.0
            
            # Adjust technical signals for bear market
            self.classification_params['technical_signals']['bear_reversal']['max_rsi'] = 40
            self.classification_params['technical_signals']['bear_reversal']['min_volume_trend'] = 0.1
            
        elif regime == "neutral":
            self.criteria['pe_ratio'] = 35
            self.criteria['institutional_ownership'] = 0.15
            self.technical_params['above_200ma'] = False
            self.criteria['sales_growth'] = 0.08
            
        elif regime == "transitional":
            self.criteria['pe_ratio'] = 30
            self.criteria['institutional_ownership'] = 0.20
            self.technical_params['above_200ma'] = False
            self.criteria['profit_margin'] = 0.04

    def _get_institutional_ownership(self, ticker: str) -> Dict:
        """Enhanced institutional ownership data collection"""
        try:
            stock = yf.Ticker(ticker)
            inst_holders = stock.institutional_holders
            if inst_holders is not None and not inst_holders.empty:
                total = inst_holders['Shares'].sum()
                outstanding = stock.info.get('sharesOutstanding')
                if outstanding:
                    pct_owned = total / outstanding
                    if len(inst_holders) >= 2:
                        prev_total = inst_holders.iloc[1]['Shares']
                        ownership_change = (total - prev_total) / prev_total if prev_total != 0 else 0
                    else:
                        ownership_change = 0
                        
                    return {
                        'institutional_holders': len(inst_holders),
                        'institutional_shares': total,
                        'institutional_pct': pct_owned,
                        'ownership_change': ownership_change,
                        'top_holders': inst_holders.head(5).to_dict('records')
                    }
        except Exception as e:
            if self.debug:
                print(f"Error getting institutional data for {ticker}: {e}")
        return None
    
    def scan_all_stocks(self) -> Dict:
        """Complete stock scanning process with enhanced error handling and performance"""
        try:
            # Detect market regime and adjust criteria
            self.current_regime = self.detect_market_regime()
            self._adjust_criteria_for_regime(self.current_regime)
            
            # Get all tickers and new listings
            tickers = self.get_all_tickers()
            new_listings = self.get_new_listings()
            
            if self.debug:
                print(f"\n🔍 Starting comprehensive scan (Regime: {self.current_regime.upper()})")
                print(f"📊 Total tickers: {len(tickers)} | New listings: {len(new_listings)}")

            # Stage 1: Robust pre-scan with hypergrowth detection
            pre_scan_results = []
            hypergrowth_candidates = []
            
            if self.debug:
                print("🚦 Running stage 1: Pre-scan with validation...")
            
            # Enhanced batch processing parameters
            batch_size = 200  # Increased batch size for better throughput
            sleep_between_batches = 10  # Longer sleep to avoid rate limiting
            total_batches = (len(tickers) + batch_size - 1) // batch_size
            
            for batch_num in tqdm(range(0, len(tickers), batch_size),
                            total=total_batches,
                            desc="Processing batches"):
                batch = tickers[batch_num:batch_num+batch_size]
                
                # Skip known invalid tickers
                filtered_batch = [
                    t for t in batch 
                    if self.hypergrowth_scanner._validate_ticker_format(t) 
                    and t not in self.blacklisted_tickers
                ]
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {executor.submit(self._prescan_ticker, t): t for t in filtered_batch}
                    
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()
                            if result:
                                pre_scan_results.append(result)
                                if result.get('is_hypergrowth'):
                                    hypergrowth_candidates.append(result)
                        except Exception as e:
                            if self.debug:
                                print(f"Error processing ticker {futures[future]}: {str(e)}")
                
                # Progress reporting
                if self.debug and batch_num % 500 == 0:
                    print(f"Processed {min(batch_num+batch_size, len(tickers))}/{len(tickers)} tickers")
                
                time.sleep(sleep_between_batches)  # Rate limiting between batches

            qualified_for_full_scan = [r['ticker'] for r in pre_scan_results if r is not None]
            
            if self.debug:
                print(f"✅ Pre-scan complete. {len(qualified_for_full_scan)} tickers qualified")
                print(f"🌟 {len(hypergrowth_candidates)} hypergrowth candidates found")

            # Stage 2: Full analysis on pre-qualified tickers
            qualified_stocks = []
            sector_distribution = defaultdict(int)
            price_tier_counts = defaultdict(int)
            data_quality_stats = {'insufficient_data': 0, 'full_data': 0}
            
            if qualified_for_full_scan:
                full_scan_batch_size = min(100, len(qualified_for_full_scan))
                num_full_batches = max(1, len(qualified_for_full_scan) // full_scan_batch_size)
                
                for i in tqdm(range(0, len(qualified_for_full_scan), full_scan_batch_size),
                            total=num_full_batches,
                            desc="Full analysis",
                            unit="batch"):
                    batch = qualified_for_full_scan[i:i+full_scan_batch_size]
                    
                    try:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                            batch_futures = {executor.submit(self._parallel_scan_ticker, t): t for t in batch}
                            
                            for future in concurrent.futures.as_completed(batch_futures):
                                try:
                                    result = future.result()
                                    if result:
                                        qualified_stocks.append(result)
                                        sector_distribution[result.get('sector', 'Unknown')] += 1
                                        
                                        # Categorize by price tier
                                        price = result.get('price', 0)
                                        if price < 20:
                                            price_tier_counts['value'] += 1
                                        elif price < 100:
                                            price_tier_counts['growth'] += 1
                                        else:
                                            price_tier_counts['premium'] += 1
                                        
                                        # Track data quality
                                        if result.get('has_sufficient_data', True):
                                            data_quality_stats['full_data'] += 1
                                        else:
                                            data_quality_stats['insufficient_data'] += 1
                                except Exception as e:
                                    if self.debug:
                                        print(f"Error processing scan result: {str(e)}")
                        
                        time.sleep(sleep_between_batches)  # Rate limiting between batches
                    except Exception as e:
                        if self.debug:
                            print(f"Error processing batch: {str(e)}")
                        continue
            else:
                if self.debug:
                    print("⚠️ No tickers qualified for full scan")

            # Categorize results
            categorized = {
                'new_listings': [],
                'value_stocks': [],
                'growth_stocks': [],
                'premium_stocks': []
            }
            
            for stock in qualified_stocks:
                try:
                    price = stock.get('price', 0)
                    if stock.get('ticker') in new_listings:
                        categorized['new_listings'].append(stock)
                    elif price < 20:
                        categorized['value_stocks'].append(stock)
                    elif price < 100:
                        categorized['growth_stocks'].append(stock)
                    else:
                        categorized['premium_stocks'].append(stock)
                except Exception as e:
                    if self.debug:
                        print(f"Error categorizing stock: {str(e)}")
                    continue

            # Identify special opportunities
            special_opportunities = self._identify_special_opportunities(qualified_stocks)
            
            # Generate sector distribution plot
            if sector_distribution and self.debug:
                self._plot_sector_distribution(sector_distribution)

            return {
                'qualified_stocks': qualified_stocks,
                'balanced_stocks': self._create_balanced_results(categorized),
                'categorized_stocks': categorized,
                'hypergrowth_candidates': hypergrowth_candidates,
                'special_opportunities': special_opportunities,
                'sector_distribution': dict(sector_distribution),
                'price_tier_distribution': dict(price_tier_counts),
                'total_scanned': len(tickers),
                'pre_scan_passed': len(qualified_for_full_scan),
                'total_qualified': len(qualified_stocks),
                'market_regime': self.current_regime,
                'new_listings_scanned': len(new_listings),
                'new_listings_qualified': len(categorized['new_listings']),
                'data_quality': data_quality_stats,
                'scan_status': 'completed'
            }

        except Exception as e:
            if self.debug:
                print(f"🚨 Critical error in scan_all_stocks: {str(e)}")
            return {
                'qualified_stocks': [],
                'error': str(e),
                'scan_status': 'failed'
            }
        
    def _create_balanced_results(self, categorized: Dict) -> List[Dict]:
        """Create balanced results from each category"""
        balanced = []
        min_per_category = max(5, sum(len(v) for v in categorized.values()) // 10)
        
        for category in ['value_stocks', 'growth_stocks', 'premium_stocks']:
            stocks = sorted(categorized[category], 
                        key=lambda x: x['success_probability'], 
                        reverse=True)
            balanced.extend(stocks[:min(min_per_category, len(stocks))])
        
        # Add all new listings
        balanced.extend(categorized['new_listings'])
        return balanced

    def _identify_special_opportunities(self, stocks: List[Dict]) -> List[Dict]:
        """Identify potential high-reward small cap opportunities"""
        special_criteria = {
            'market_cap': (10e6, 500e6),  # $10M-$500M market cap
            'sales_growth': 0.25,  # 25%+ sales growth
            'pe_ratio': 15,  # Below 15 P/E
            'debt_equity': 1.0,  # Conservative debt
            'volume': 50e3  # Lower volume threshold for small caps
        }
        
        special_stocks = []
        for stock in stocks:
            try:
                meets = all([
                    special_criteria['market_cap'][0] <= stock['market_cap'] <= special_criteria['market_cap'][1],
                    stock.get('sales_growth', 0) >= special_criteria['sales_growth'],
                    stock.get('pe_ratio', 999) <= special_criteria['pe_ratio'],
                    stock.get('debt_equity', 999) <= special_criteria['debt_equity'],
                    stock.get('volume', 0) >= special_criteria['volume']
                ])
                if meets:
                    # Calculate special score
                    growth_score = min(50, stock['sales_growth'] * 200)  # 25% growth = 50 points
                    value_score = min(30, (1/stock['pe_ratio']) * 450 if stock['pe_ratio'] > 0 else 0)  # PE of 15 = 30 points
                    momentum_score = min(20, stock.get('price_trend', 0) * 100)
                    stock['special_score'] = growth_score + value_score + momentum_score
                    special_stocks.append(stock)
            except:
                continue
        
        return sorted(special_stocks, key=lambda x: x['special_score'], reverse=True)

    def _get_sales_growth(self, stock) -> float:
        """Enhanced sales growth calculation with small-cap adjustments"""
        try:
            financials = stock.quarterly_financials
            if financials is None or len(financials) < 4:
                return None
                
            market_cap = stock.info.get('marketCap', 0)
            is_small_cap = market_cap < 2e9  # Under $2B
            
            revenues = financials.loc['Total Revenue'].sort_index(ascending=False)
            if len(revenues) < 2:
                return None
                
            # For small caps, look at sequential growth and acceleration
            if is_small_cap and len(revenues) >= 3:
                q2q_growth = []
                for i in range(min(3, len(revenues)-1)):  # Fixed line
                    if revenues.iloc[i+1] != 0:
                        growth = (revenues.iloc[i] - revenues.iloc[i+1]) / abs(revenues.iloc[i+1])
                        q2q_growth.append(growth)
                
                if len(q2q_growth) >= 2:
                    latest_growth = q2q_growth[0]
                    momentum = q2q_growth[0] - q2q_growth[1]
                    return latest_growth + (momentum * 0.3)  # Reward acceleration
            
            # Standard YoY calculation
            latest = revenues.iloc[0]
            previous = revenues.iloc[4] if len(revenues) > 4 else revenues.iloc[1]
            
            if previous == 0:
                return None
                
            return (latest - previous) / abs(previous)
            
        except Exception as e:
            if self.debug:
                print(f"Error calculating sales growth: {e}")
            return None
        
    def _safe_calculate_trend(self, series: pd.Series, window=20) -> float:
        """Enhanced trend calculation with outlier removal"""
        try:
            if len(series) < window or series.isnull().all():
                return 0
                
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            filtered = series[(series >= q1 - 1.5*iqr) & (series <= q3 + 1.5*iqr)]
            
            if len(filtered) < 5:
                return 0
                
            x = np.arange(len(filtered))
            y = filtered.values
            slope = np.polyfit(x, y, 1)[0]
            return slope / np.nanmean(y) if np.nanmean(y) != 0 else 0
            
        except Exception as e:
            if self.debug:
                print(f"Error calculating trend: {e}")
            return 0

    def _check_eps_beats(self, stock) -> bool:
        """Enhanced EPS beats check with trend analysis"""
        try:
            earnings = stock.quarterly_earnings
            if earnings is None or len(earnings) < 4:
                return False
                
            beats = 0
            surprises = []
            for i in range(min(4, len(earnings))):
                if earnings.iloc[i]['Actual'] > earnings.iloc[i]['Estimate']:
                    beats += 1
                surprises.append((earnings.iloc[i]['Actual'] - earnings.iloc[i]['Estimate']) / 
                              abs(earnings.iloc[i]['Estimate']) if earnings.iloc[i]['Estimate'] != 0 else 0)
            
            if len(surprises) >= 3:
                trend = self._safe_calculate_trend(pd.Series(surprises))
                if trend > 0.05:
                    return True
                    
            return beats >= 2
            
        except Exception as e:
            if self.debug:
                print(f"Error checking EPS beats: {e}")
            return False

    def _format_value(self, value, suffix: str = '') -> str:
        """Enhanced value formatting"""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "N/A"
        if isinstance(value, float):
            if abs(value) >= 1e9:
                return f"{value/1e9:.2f}B{suffix}"
            elif abs(value) >= 1e6:
                return f"{value/1e6:.2f}M{suffix}"
            elif abs(value) >= 1e3:
                return f"{value/1e3:.1f}K{suffix}"
            elif abs(value) < 0.01:
                return f"{value:.4f}{suffix}"
            else:
                return f"{value:.2f}{suffix}"
        return f"{value}{suffix}"

    def _get_financial_metrics(self, stock):
        """Enhanced financial data fetcher with EV/EBITDA and free cash flow"""
        metrics = {
            'ticker': stock.ticker,
            'company': 'N/A',
            'sector': 'Unknown',
            'industry': 'Unknown',
            'price': 0.0,
            'pe_ratio': 35.0,
            'peg_ratio': 2.0,
            'ps_ratio': 5.0,
            'pb_ratio': 3.0,
            'ev_ebitda': 15.0,  # Enterprise Value / EBITDA
            'fcf_yield': 0.0,   # Free Cash Flow Yield
            'debt_equity': 1.5,
            'current_ratio': 1.5,
            'roe': 0.08,
            'volume': 0.0,
            'institutional_pct': 0.15,
            'profit_margins': 0.05,
            'operating_margins': 0.08,
            'eps_growth': 0.0,
            'eps_beats': False,
            'sales_growth': 0.0,
            'earnings_growth': 0.0
        }

        try:
            info = None
            for attempt in range(3):
                try:
                    self._rate_limit()
                    info = stock.info
                    if info and isinstance(info, dict):
                        break
                    time.sleep(1 + attempt)
                except Exception as e:
                    if self.debug and attempt == 2:
                        print(f"Info fetch failed for {stock.ticker}: {str(e)}")

            if not info:
                return metrics

            # Basic info
            for field in ['longName', 'shortName', 'sector', 'industry']:
                if field in info and info[field]:
                    key = 'company' if field == 'longName' else field.lower()
                    metrics[key] = info[field]

            # Numeric metrics
            numeric_map = {
                'currentPrice': 'price',
                'regularMarketPrice': 'price',
                'trailingPE': 'pe_ratio',
                'pegRatio': 'peg_ratio',
                'priceToSalesTrailing12Months': 'ps_ratio',
                'priceToBook': 'pb_ratio',
                'enterpriseToEbitda': 'ev_ebitda',
                'freeCashflow': 'free_cash_flow',
                'debtToEquity': 'debt_equity',
                'currentRatio': 'current_ratio',
                'returnOnEquity': 'roe',
                'averageVolume': 'volume',
                'heldPercentInstitutions': 'institutional_pct',
                'profitMargins': 'profit_margins',
                'operatingMargins': 'operating_margins'
            }

            for yf_field, our_field in numeric_map.items():
                try:
                    val = info.get(yf_field)
                    if val is not None:
                        metrics[our_field] = float(val)
                except (TypeError, ValueError):
                    continue

            # Calculate free cash flow yield if we have market cap and FCF
            if 'free_cash_flow' in metrics and 'marketCap' in info and info['marketCap']:
                metrics['fcf_yield'] = metrics['free_cash_flow'] / info['marketCap']

            # Earnings data
            try:
                earnings = stock.quarterly_earnings
                if earnings is not None and len(earnings) >= 4:
                    beats = sum(1 for i in range(min(4, len(earnings))) 
                            if earnings.iloc[i]['Actual'] > earnings.iloc[i]['Estimate'])
                    metrics['eps_beats'] = beats >= 2
                    
                    if len(earnings) >= 2:
                        metrics['eps_growth'] = (earnings.iloc[0]['Earnings'] - earnings.iloc[1]['Earnings']) / abs(earnings.iloc[1]['Earnings']) if earnings.iloc[1]['Earnings'] != 0 else 0
            except Exception:
                pass

            # Revenue growth
            try:
                financials = stock.quarterly_financials
                if financials is not None and 'Total Revenue' in financials.index:
                    revenues = financials.loc['Total Revenue'].sort_index(ascending=False)
                    if len(revenues) >= 4:
                        metrics['sales_growth'] = (revenues.iloc[0] - revenues.iloc[4]) / abs(revenues.iloc[4]) if revenues.iloc[4] != 0 else 0
            except Exception:
                pass

            # Sector-specific adjustments
            sector = metrics['sector']
            if sector in self.criteria['sector_weights']:
                for k, v in self.criteria['sector_weights'][sector].items():
                    if k in metrics:
                        metrics[k] = v * 1.1  # Slightly more lenient than sector max

        except Exception as e:
            if self.debug:
                print(f"Critical error in metrics for {stock.ticker}: {str(e)}")

        return metrics
    
    def _get_technical_indicators(self, hist: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators with complete error handling and null safety
        """
        indicators = {
            'rsi': 50.0,
            'macd_diff': 0.0,
            'bollinger_percent': 0.5,
            'above_20ma': False,
            'above_50ma': False,
            'above_200ma': False,
            'obv_trend': 0.0,
            'adl_trend': 0.0,
            'volume_trend': 0.0,
            'price_trend': 0.0,
            'vwap_position': 0.0,
            'change_pct': 0.0
        }

        if hist is None or len(hist) < 20:
            return indicators

        try:
            required_cols = {
                'Open': lambda: hist['Close'].copy(),
                'High': lambda: hist['Close'].copy(),
                'Low': lambda: hist['Close'].copy(),
                'Close': lambda: pd.Series(0, index=hist.index),
                'Volume': lambda: pd.Series(0, index=hist.index)
            }
            
            for col, fallback in required_cols.items():
                if col not in hist.columns or hist[col].isnull().all():
                    hist[col] = fallback()

            hist = hist.ffill().bfill()
            if len(hist) < 20:
                return indicators

            for window in [20, 50, 200]:
                try:
                    hist[f'MA_{window}'] = hist['Close'].rolling(window).mean()
                except:
                    hist[f'MA_{window}'] = 0.0

            try:
                hist['RSI'] = ta.momentum.rsi(hist['Close'], window=14).clip(0, 100)
            except:
                hist['RSI'] = 50.0

            try:
                macd = ta.trend.MACD(hist['Close'])
                hist['MACD_diff'] = macd.macd_diff()
            except:
                hist['MACD_diff'] = 0.0

            try:
                bb = ta.volatility.BollingerBands(hist['Close'])
                hist['BB_percent'] = ((hist['Close'] - bb.bollinger_lband()) / 
                                    (bb.bollinger_hband() - bb.bollinger_lband()))
                hist['BB_percent'] = hist['BB_percent'].clip(0, 1)
            except:
                hist['BB_percent'] = 0.5

            try:
                hist['OBV'] = ta.volume.OnBalanceVolumeIndicator(
                    close=hist['Close'], 
                    volume=hist['Volume']
                ).on_balance_volume()
            except:
                hist['OBV'] = 0

            try:
                hist['ADL'] = ta.volume.AccDistIndexIndicator(
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'],
                    volume=hist['Volume']
                ).acc_dist_index()
            except:
                hist['ADL'] = 0

            try:
                hist['VWAP'] = (hist['Volume'] * (hist['High'] + hist['Low'] + hist['Close']) / 3).cumsum() / hist['Volume'].cumsum()
            except:
                hist['VWAP'] = hist['Close']

            def safe_trend(series, window=20):
                try:
                    if len(series) < window:
                        return 0.0
                    x = np.arange(len(series))
                    y = series.values
                    slope = np.polyfit(x, y, 1)[0]
                    return slope / np.nanmean(y) if np.nanmean(y) != 0 else 0.0
                except:
                    return 0.0

            last = hist.iloc[-1]
            vwap_pos = ((last['Close'] - last.get('VWAP', last['Close'])) / 
                    last.get('VWAP', 1.0)) * 100 if last.get('VWAP', 0) != 0 else 0.0
            
            change_pct = ((last['Close'] - hist.iloc[-2]['Close']) / 
                        hist.iloc[-2]['Close'] * 100) if len(hist) > 1 else 0.0

            indicators.update({
                'rsi': float(last.get('RSI', 50.0)),
                'macd_diff': float(last.get('MACD_diff', 0.0)),
                'bollinger_percent': float(last.get('BB_percent', 0.5)),
                'above_20ma': bool(last['Close'] > last.get('MA_20', 0)),
                'above_50ma': bool(last['Close'] > last.get('MA_50', 0)),
                'above_200ma': bool(last['Close'] > last.get('MA_200', 0)),
                'obv_trend': safe_trend(hist['OBV'].tail(20)),
                'adl_trend': safe_trend(hist['ADL'].tail(20)),
                'volume_trend': safe_trend(hist['Volume'].tail(20)),
                'price_trend': safe_trend(hist['Close'].tail(20)),
                'vwap_position': float(vwap_pos),
                'change_pct': float(change_pct)
            })

        except Exception as e:
            if self.debug:
                print(f"Error calculating tech indicators: {str(e)}")

        return indicators
    
    def _calculate_volume_change(self, hist: pd.DataFrame) -> float:
        """Calculate volume change safely"""
        try:
            if len(hist) < 20:
                return 0
            recent_volume = hist['Volume'].tail(10).mean()
            older_volume = hist['Volume'].iloc[-20:-10].mean()
            return (recent_volume - older_volume) / older_volume if older_volume != 0 else 0
        except:
            return 0

    def _calculate_price_position(self, hist: pd.DataFrame) -> float:
        """Calculate price position safely"""
        try:
            if len(hist) < 5:
                return 0.5
            current_close = hist['Close'].iloc[-1]
            return (current_close - hist['Low'].min()) / (hist['High'].max() - hist['Low'].min())
        except:
            return 0.5

    def _calculate_price_change(self, hist: pd.DataFrame) -> float:
        """Calculate price change safely"""
        try:
            if len(hist) < 2:
                return 0
            return (hist.iloc[-1]['Close'] - hist.iloc[-2]['Close']) / hist.iloc[-2]['Close'] * 100
        except:
            return 0

    def _calculate_vwap_position(self, hist: pd.DataFrame) -> float:
        """Calculate VWAP position safely"""
        try:
            if len(hist) < 1 or 'Close' not in hist:
                return 0
            current_close = hist['Close'].iloc[-1]
            vwap = (hist['Volume'] * (hist['High'] + hist['Low'] + hist['Close']) / 3).cumsum() / hist['Volume'].cumsum()
            return (current_close - vwap.iloc[-1]) / vwap.iloc[-1] * 100
        except:
            return 0

    def _classify_fundamentals(self, metrics: Dict) -> str:
        """Classify stocks into fundamental categories"""
        # Elite Quality: High profitability, low debt
        if (metrics.get('roe', 0) >= self.classification_params['fundamental_classes']['elite_quality']['min_roe'] and 
            metrics.get('debt_equity', 999) <= self.classification_params['fundamental_classes']['elite_quality']['max_debt_equity'] and 
            metrics.get('profit_margins', 0) >= self.classification_params['fundamental_classes']['elite_quality']['min_profit_margin'] and 
            metrics.get('fcf_yield', 0) >= self.classification_params['fundamental_classes']['elite_quality']['min_fcf_yield']):
            return 'elite_quality'
        
        # Growth at Reasonable Value
        elif (metrics.get('sales_growth', 0) >= self.classification_params['fundamental_classes']['growth_at_value']['min_sales_growth'] and 
              metrics.get('ps_ratio', 999) <= self.classification_params['fundamental_classes']['growth_at_value']['max_ps_ratio'] and 
              metrics.get('peg_ratio', 999) <= self.classification_params['fundamental_classes']['growth_at_value']['max_peg_ratio']):
            return 'growth_at_value'
        
        # Deep Value
        elif (metrics.get('pe_ratio', 999) <= self.classification_params['fundamental_classes']['deep_value']['max_pe_ratio'] and 
              metrics.get('pb_ratio', 999) <= self.classification_params['fundamental_classes']['deep_value']['max_pb_ratio'] and 
              metrics.get('fcf_yield', 0) >= self.classification_params['fundamental_classes']['deep_value']['min_fcf_yield']):
            return 'deep_value'
        
        return 'neutral'

    def _classify_technicals(self, technicals: Dict, regime: str) -> str:
        """Classify technical patterns based on market regime"""
        if regime == 'bull':
            # Look for bullish confirmation signals
            if (technicals.get('rsi', 50) >= self.classification_params['technical_signals']['bull_confirm']['min_rsi'] and 
                technicals.get('rsi', 50) <= self.classification_params['technical_signals']['bull_confirm']['max_rsi'] and 
                technicals.get('price_trend', 0) >= self.classification_params['technical_signals']['bull_confirm']['min_price_trend']):
                
                if self.classification_params['technical_signals']['bull_confirm']['above_ma'] == '50ma' and technicals.get('above_50ma', False):
                    return 'bull_confirm'
                elif self.classification_params['technical_signals']['bull_confirm']['above_ma'] == '200ma' and technicals.get('above_200ma', False):
                    return 'bull_confirm'
        
        elif regime == 'bear':
            # Look for bear market reversal signals
            if (technicals.get('rsi', 50) <= self.classification_params['technical_signals']['bear_reversal']['max_rsi'] and 
                technicals.get('volume_trend', 0) >= self.classification_params['technical_signals']['bear_reversal']['min_volume_trend']):
                
                if self.classification_params['technical_signals']['bear_reversal']['above_ma'] == '50ma' and technicals.get('above_50ma', False):
                    return 'bear_reversal'
                elif self.classification_params['technical_signals']['bear_reversal']['above_ma'] == '200ma' and technicals.get('above_200ma', False):
                    return 'bear_reversal'
        
        return 'neutral'

    def _calculate_probability(self, fundamental_class: str, technical_signal: str) -> float:
        """Calculate success probability based on historical base rates"""
        base_rate = self.classification_params['base_success_rates'].get(fundamental_class, 0.5)
        signal_boost = self.classification_params['base_success_rates'].get(technical_signal, 0.5)
        
        # Weighted average favoring fundamentals
        return (base_rate * 0.7) + (signal_boost * 0.3)

    def _parallel_scan_ticker(self, ticker: str) -> Optional[Dict]:
        """Complete ticker analysis with hypergrowth detection and classification"""
        try:
            # Skip invalid tickers
            if any(x in ticker for x in ['^', '$', '.', '/']):
                return None
            if re.search(r'[\.-](W|WS|U|R|PR|P)[\d]*$', ticker) or ticker.endswith(('W', 'WS', 'U', 'R')) and len(ticker) > 4:
                return None

            # Get stock data with rate limiting
            self._rate_limit()
            stock = yf.Ticker(ticker)
            
            # Get basic info and check security type
            info = stock.info
            quote_type = info.get('quoteType', '')
            if quote_type not in ['EQUITY', 'ETF']:
                return None

            # Get historical data
            hist = None
            try:
                self._rate_limit()
                hist = stock.history(period='6mo', interval='1d', timeout=10)
                if hist is None or hist.empty or len(hist) < 30:
                    hist = None
            except Exception as e:
                if self.debug:
                    print(f"History failed for {ticker}: {str(e)}")
                hist = None

            # Get financial metrics
            metrics = self._get_financial_metrics(stock)
            
            # Fallback volume from history if missing
            if (metrics.get('volume') is None or metrics['volume'] < 1) and hist is not None:
                try:
                    metrics['volume'] = hist['Volume'].mean()
                except Exception:
                    metrics['volume'] = 0.0

            # Get technical indicators
            technicals = {}
            if hist is not None:
                try:
                    technicals = self._get_technical_indicators(hist)
                except Exception as e:
                    if self.debug:
                        print(f"Technical indicators failed for {ticker}: {str(e)}")
                    technicals = {
                        'rsi': 50,
                        'macd_diff': 0,
                        'bollinger_percent': 0.5,
                        'above_20ma': False,
                        'above_50ma': False,
                        'above_200ma': False
                    }

            # Check for hypergrowth status
            hypergrowth = self.hypergrowth_scanner.detect_hypergrowth(ticker)
            is_hypergrowth = hypergrowth['is_hypergrowth'] if hypergrowth else False
            
            # Classify the stock
            fundamental_class = self._classify_fundamentals(metrics)
            technical_signal = self._classify_technicals(technicals, self.current_regime)
            
            # Calculate success probability with hypergrowth boost
            base_probability = self._calculate_probability(fundamental_class, technical_signal)
            if is_hypergrowth:
                base_probability = min(0.95, base_probability * 1.2)  # 20% boost for hypergrowth
                
            # Calculate stop loss
            stop_loss = self._calculate_stop_loss(hist) if hist is not None else None
            
            # Build analysis dictionary
            analysis = {
                'ticker': ticker,
                'company': metrics.get('company', 'N/A'),
                'sector': metrics.get('sector', 'Unknown'),
                'industry': metrics.get('industry', 'Unknown'),
                'is_hypergrowth': is_hypergrowth,
                'hypergrowth_score': hypergrowth['hypergrowth_score'] if hypergrowth else 0,
                'fundamental_class': fundamental_class,
                'technical_signal': technical_signal,
                'success_probability': base_probability,
                'stop_loss': stop_loss,
                'has_sufficient_data': hist is not None,
                'price': metrics.get('price', 0),
                'volume': metrics.get('volume', 0),
                'market_cap': metrics.get('market_cap'),
                'pe_ratio': metrics.get('pe_ratio'),
                'peg_ratio': metrics.get('peg_ratio'),
                'ps_ratio': metrics.get('ps_ratio'),
                'pb_ratio': metrics.get('pb_ratio'),
                'ev_ebitda': metrics.get('ev_ebitda'),
                'fcf_yield': metrics.get('fcf_yield'),
                'debt_equity': metrics.get('debt_equity'),
                'current_ratio': metrics.get('current_ratio'),
                'roe': metrics.get('roe'),
                'institutional_pct': metrics.get('institutional_pct'),
                'profit_margins': metrics.get('profit_margins'),
                'operating_margins': metrics.get('operating_margins'),
                'eps_growth': metrics.get('eps_growth'),
                'eps_beats': metrics.get('eps_beats'),
                'sales_growth': metrics.get('sales_growth'),
                'earnings_growth': metrics.get('earnings_growth'),
                'rsi': technicals.get('rsi', 50),
                'macd_diff': technicals.get('macd_diff', 0),
                'bollinger_percent': technicals.get('bollinger_percent', 0.5),
                'above_20ma': technicals.get('above_20ma', False),
                'above_50ma': technicals.get('above_50ma', False),
                'above_200ma': technicals.get('above_200ma', False),
                'obv_trend': technicals.get('obv_trend', 0),
                'adl_trend': technicals.get('adl_trend', 0),
                'volume_trend': technicals.get('volume_trend', 0),
                'price_trend': technicals.get('price_trend', 0),
                'vwap_position': technicals.get('vwap_position', 0),
                'change_pct': technicals.get('change_pct', 0)
            }

            # Determine if stock meets criteria (relaxed for hypergrowth)
            meets_criteria = self._check_classification_criteria(analysis)
            
            if self.debug:
                status = "✅" if meets_criteria else "❌"
                print(f"{status} {ticker}: {fundamental_class.replace('_',' ').title()} "
                    f"({technical_signal.replace('_',' ').title()}, "
                    f"{base_probability:.0%} confidence)")
                if is_hypergrowth:
                    print(f"   🌟 Hypergrowth (Score: {analysis['hypergrowth_score']:.1f}/100)")

            return analysis if meets_criteria else None

        except Exception as e:
            if self.debug:
                print(f"Unexpected error processing {ticker}: {str(e)}")
            return None

    def _calculate_stop_loss(self, hist: pd.DataFrame) -> float:
        """Calculate stop loss price using ATR"""
        if hist is None or len(hist) < 14:
            return None
            
        try:
            atr = ta.volatility.average_true_range(
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                window=14
            ).iloc[-1]
            highest_high = hist['High'].max()
            return highest_high - (atr * 2)
        except:
            return None
            
    def _passes_liquidity_checks(self, info: Dict) -> bool:
        """Conservative liquidity requirements"""
        min_volume = self.criteria['avg_volume'] * 0.3  # 30% of main threshold
        volume = info.get('averageVolume', 0)
        
        # Special handling for small caps with growth characteristics
        market_cap = info.get('marketCap', float('inf'))
        if market_cap < 500e6:  # Below $500M
            min_volume *= 0.7  # Be more lenient for small caps
            
        return volume >= min_volume

    def _passes_valuation_checks(self, info: Dict) -> bool:
        """Sector-aware valuation guardrails"""
        sector = info.get('sector', 'Unknown')
        price = info.get('currentPrice') or info.get('regularMarketPrice')
        
        if not price:
            return False
            
        # Absolute price range check (very wide)
        if not (0.50 <= price <= 1000):  # $0.50 to $1000
            return False
            
        # Sector-specific checks
        sector_params = self.criteria['sector_weights'].get(sector, {})
        
        # PE ratio check (if available)
        pe = info.get('trailingPE')
        if pe and sector_params.get('max_pe_ratio'):
            max_pe = sector_params['max_pe_ratio'] * 1.5  # 50% buffer
            if pe > max_pe:
                return False
                
        return True

    def _create_normal_result(self, ticker: str, info: dict) -> dict:
        """Create result dict for normal stocks"""
        return {
            'ticker': ticker,
            'company': info.get('longName', ticker),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'price': info.get('currentPrice') or info.get('regularMarketPrice'),
            'volume': info.get('averageVolume', 0),
            'market_cap': info.get('marketCap'),
            'pe_ratio': info.get('trailingPE'),
            'exchange': info.get('exchange', 'Unknown'),
            'is_hypergrowth': False,
            'relaxed_filters': False
        }

    def _check_classification_criteria(self, analysis: Dict) -> bool:
        """Check if stock meets our classification criteria"""
        # Must have sufficient data
        if not analysis.get('has_sufficient_data', False):
            return False
            
        # Must have valid fundamental classification
        if analysis.get('fundamental_class', 'neutral') == 'neutral':
            return False
            
        # Must meet minimum probability threshold
        if analysis.get('success_probability', 0) < 0.5:  # 50% minimum
            return False
            
        # Volume and liquidity checks
        if analysis.get('volume', 0) < self.criteria['avg_volume'] * 0.5:  # 50% of normal threshold
            return False
            
        return True

    def _plot_sector_distribution(self, sector_data: Dict):
        """Enhanced sector distribution visualization"""
        sectors = list(sector_data.keys())
        counts = list(sector_data.values())
        
        plt.figure(figsize=(12, 7))
        bars = plt.bar(sectors, counts, color=plt.cm.tab20.colors[:len(sectors)])
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.title(f'Sector Distribution of Qualified Stocks ({self.current_regime.upper()} Market)\n'
                 f'Total Qualified: {sum(counts)} Stocks', pad=20)
        plt.xlabel('Sector', labelpad=10)
        plt.ylabel('Number of Stocks', labelpad=10)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if self.debug:
            plt.show()
        else:
            os.makedirs('output', exist_ok=True)
            plt.savefig('output/sector_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()

    def generate_report(self, scan_results: Dict) -> str:
        """Generate comprehensive report with hypergrowth highlights"""
        report = []
        report.append("📈 NASDAQ STOCK SCREENER REPORT")
        report.append("=" * 90)
        report.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append(f"🏛 Market Regime: {scan_results.get('market_regime', 'unknown').upper()}")
        report.append(f"🔢 Total Scanned: {scan_results['total_scanned']} | "
                    f"Qualified: {scan_results['total_qualified']}")
        
        # Hypergrowth Candidates Section
        if scan_results.get('hypergrowth_candidates'):
            report.append("\n🚀 HYPERGROWTH STARS (Future AMZN/NVDA Potential)")
            report.append("-" * 50)
            for stock in sorted(scan_results['hypergrowth_candidates'],
                            key=lambda x: x['hypergrowth_score'],
                            reverse=True)[:10]:
                report.append(
                    f"⭐ {stock['ticker']}: {stock.get('company', 'N/A')} "
                    f"(Score: {stock['hypergrowth_score']:.1f}/100)"
                )
                report.append(
                    f"   📊 Sector: {stock.get('sector', 'Unknown')} | "
                    f"Price: ${stock.get('price',0):.2f} | "
                    f"MCap: {self._format_value(stock.get('market_cap'))}"
                )
                report.append(
                    f"   📈 Growth: Rev {stock.get('sales_growth',0)*100:.1f}% | "
                    f"Earnings {stock.get('earnings_growth',0)*100:.1f}%"
                )
                report.append(
                    f"   💰 Valuation: P/E {self._format_value(stock.get('pe_ratio'))} | "
                    f"P/S {self._format_value(stock.get('ps_ratio'))}"
                )
                report.append(
                    f"   🔍 Technicals: RSI {stock.get('rsi',0):.1f} | "
                    f"Above MAs: 20{'✓' if stock.get('above_20ma') else '✗'}/"
                    f"50{'✓' if stock.get('above_50ma') else '✗'}/"
                    f"200{'✓' if stock.get('above_200ma') else '✗'}"
                )
                report.append("-" * 50)

        # Top Stocks by Classification
        report.append("\n🏆 TOP STOCKS BY CLASSIFICATION")
        
        # Elite Quality Stocks
        elite = [s for s in scan_results['qualified_stocks'] 
                if s.get('fundamental_class') == 'elite_quality']
        if elite:
            report.append("\n🌟 ELITE QUALITY (High Profitability, Low Debt)")
            for stock in sorted(elite, key=lambda x: x['success_probability'], reverse=True)[:5]:
                report.append(
                    f"- {stock['ticker']}: {stock.get('company', 'N/A')} "
                    f"(Confidence: {stock['success_probability']:.0%})"
                )
                report.append(
                    f"  ROE: {self._format_value(stock.get('roe',0), '%')} | "
                    f"Debt/Equity: {self._format_value(stock.get('debt_equity',0))} | "
                    f"FCF Yield: {self._format_value(stock.get('fcf_yield',0), '%')}"
                )
                report.append(
                    f"  Technicals: {stock['technical_signal'].replace('_',' ').title()} | "
                    f"RSI: {stock.get('rsi',0):.1f} | "
                    f"Stop Loss: {self._format_value(stock.get('stop_loss')) if stock.get('stop_loss') else 'N/A'}"
                )

        # Growth at Value Stocks
        growth_value = [s for s in scan_results['qualified_stocks'] 
                    if s.get('fundamental_class') == 'growth_at_value']
        if growth_value:
            report.append("\n📈 GROWTH AT REASONABLE VALUE")
            for stock in sorted(growth_value, key=lambda x: x['success_probability'], reverse=True)[:5]:
                report.append(
                    f"- {stock['ticker']}: {stock.get('company', 'N/A')} "
                    f"(Confidence: {stock['success_probability']:.0%})"
                )
                report.append(
                    f"  P/S: {self._format_value(stock.get('ps_ratio',0))} | "
                    f"PEG: {self._format_value(stock.get('peg_ratio',0))} | "
                    f"Growth: {self._format_value(stock.get('sales_growth',0)*100, '%')}"
                )
                report.append(
                    f"  Technicals: {stock['technical_signal'].replace('_',' ').title()} | "
                    f"MACD Diff: {stock.get('macd_diff',0):.2f} | "
                    f"Volume Trend: {stock.get('volume_trend',0):.2%}"
                )

        # Special Opportunities
        if scan_results.get('special_opportunities'):
            report.append("\n💎 SPECIAL OPPORTUNITIES (High-Risk/High-Reward)")
            for stock in sorted(scan_results['special_opportunities'],
                            key=lambda x: x.get('special_score',0),
                            reverse=True)[:3]:
                report.append(
                    f"- {stock['ticker']}: {stock.get('company', 'N/A')} "
                    f"(Special Score: {stock['special_score']:.1f})"
                )
                report.append(
                    f"  MCap: {self._format_value(stock['market_cap'])} | "
                    f"Growth: {self._format_value(stock.get('sales_growth',0)*100, '%')} | "
                    f"Short %: {self._format_value(stock.get('shortPercentOfFloat',0)*100, '%')}"
                )

        # Market Regime Recommendations
        report.append("\n💡 MARKET REGIME STRATEGY")
        regime = scan_results.get('market_regime', 'neutral')
        if regime == 'bull':
            report.append("- Focus on growth stocks and hypergrowth candidates")
            report.append("- Consider momentum strategies with tight stop losses")
            report.append("- Trail stops to lock in profits during uptrends")
        elif regime == 'bear':
            report.append("- Focus on elite quality stocks with strong balance sheets")
            report.append("- Consider defensive sectors and short opportunities")
            report.append("- Use smaller position sizes and wider stops")
        else:
            report.append("- Balanced approach between growth and value")
            report.append("- Consider dollar-cost averaging into positions")
            report.append("- Maintain diversified portfolio across sectors")

        report.append("\n🔍 NEXT STEPS:")
        report.append("- Review technical setups for top picks in each category")
        report.append("- Verify fundamentals match investment thesis")
        report.append("- Consider position sizing based on confidence levels")
        if scan_results.get('hypergrowth_candidates'):
            report.append("- Monitor hypergrowth candidates for breakout opportunities")

        return "\n".join(report)


if __name__ == "__main__":
    scanner = NASDAQStockScanner(
        debug=True,
        max_workers=4,
        newsapi_key=None  # Set your NewsAPI key if needed
    )

    print("🚀 Starting NASDAQ Stock Scanner with Hypergrowth Detection...")
    try:
        results = scanner.scan_all_stocks()
        
        if results['qualified_stocks']:
            report = scanner.generate_report(results)
            print("\n" + report)
            
            # Save results
            pd.DataFrame(results['qualified_stocks']).to_parquet("scan_results.parquet")
            print("\n✅ Scan completed successfully!")
        else:
            print("\n❌ No qualifying stocks found")
            
    except Exception as e:
        print(f"\n❌ Error during scanning: {str(e)}")