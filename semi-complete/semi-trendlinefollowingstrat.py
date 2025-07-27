import requests
import pandas as pd
import numpy as np
import time
import math
import os
import json
import warnings
from datetime import datetime, timedelta
from threading import Thread
from queue import Queue
from hmmlearn import hmm
from websocket import create_connection, WebSocketConnectionClosedException
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from config import POLYGON_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# ======================
# Combined Configuration
# ======================
EXCHANGES = ["XNYS", "XNAS", "XASE"]  # NYSE, NASDAQ, AMEX
MAX_TICKERS_PER_EXCHANGE = 200
RATE_LIMIT = 0.001  # seconds between requests
MIN_DAYS_DATA = 200  # Minimum days of data required for analysis
N_STATES = 3  # Bull/Neutral/Bear regimes
SECTOR_SAMPLE_SIZE = 50  # Stocks per sector for composite
TRANSITION_WINDOW = 30  # Days to analyze around regime transitions
ALLOCATION_PER_TICKER = 10000  # $10,000 per position
MAX_TICKERS_TO_SCAN = 300  # Limit for weekly scanner

# Global Cache
DATA_CACHE = {
    'tickers': None,
    'sector_mappings': None,
    'last_updated': None,
    'stock_data': {},
    'last_regime': None
}

# Suppress warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =================
# Alpaca Setup
# =================
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

# =====================
# Core Classes
# =====================
class MarketRegimeAnalyzer:
    def __init__(self, n_states=3):
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=2000,
            tol=1e-4,
            init_params='stmc',
            params='stmc',
            random_state=42
        )
        self.state_labels = {}
        self.feature_scaler = StandardScaler()
        
    def prepare_market_data(self, tickers, sample_size=100):
        prices_data = []
        valid_tickers = []
        
        print("\nBuilding market composite from multiple exchanges...")
        for symbol in tqdm(tickers[:sample_size]):
            prices = fetch_stock_data(symbol)
            if prices is not None and len(prices) >= MIN_DAYS_DATA:
                prices_data.append(prices)
                valid_tickers.append(symbol)
        
        if not prices_data:
            raise ValueError("Insufficient data to create market composite")
        
        composite = pd.concat(prices_data, axis=1)
        composite.columns = valid_tickers
        return composite.mean(axis=1).dropna()

    def analyze_regime(self, index_data, n_states=None):
        if n_states is None:
            n_states = self.model.n_components
        
        # Calculate features
        log_returns = np.log(index_data).diff().dropna()
        features = pd.DataFrame({
            'returns': log_returns,
            'volatility': log_returns.rolling(21).std(),
            'momentum': log_returns.rolling(14).mean()
        }).dropna()
        
        if len(features) < 60:
            raise ValueError(f"Only {len(features)} days of feature data")
        
        # Scale features
        scaled_features = self.feature_scaler.fit_transform(features)
        
        # Create model with potentially different state count
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=2000,
            tol=1e-4,
            random_state=42
        )
        
        # Fit model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(scaled_features)
        
        # Label states
        state_means = sorted([
            (i, np.mean(model.means_[i])) 
            for i in range(model.n_components)
        ], key=lambda x: x[1])
        
        state_labels = {}
        if n_states == 3:
            state_labels = {
                state_means[0][0]: 'Bear',
                state_means[1][0]: 'Neutral',
                state_means[2][0]: 'Bull'
            }
        elif n_states == 4:
            state_labels = {
                state_means[0][0]: 'Severe Bear',
                state_means[1][0]: 'Mild Bear',
                state_means[2][0]: 'Mild Bull',
                state_means[3][0]: 'Strong Bull'
            }
        else:
            for i in range(n_states):
                state_labels[i] = f'State {i+1}'
        
        # Predict regimes
        states = model.predict(scaled_features)
        state_probs = model.predict_proba(scaled_features)
        
        return {
            'model': model,
            'regimes': [state_labels[s] for s in states],
            'probabilities': state_probs,
            'features': features,
            'index_data': index_data[features.index[0]:],
            'state_labels': state_labels
        }

class SectorRegimeSystem:
    def __init__(self):
        self.sector_mappings = {}
        self.sector_composites = {}
        self.sector_analyzers = {}
        self.overall_analyzer = MarketRegimeAnalyzer()
        self.transition_performance = {}
        self.sector_weights = {}
        self.sector_scores = {}
        
    def map_tickers_to_sectors(self, tickers):
        self.sector_mappings = {}
        
        for symbol in tqdm(tickers):
            url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
            params = {"apiKey": POLYGON_API_KEY}
            time.sleep(RATE_LIMIT)
            
            try:
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json().get('results', {})
                    sector = data.get('sic_description', 'Unknown')
                    if sector == 'Unknown':
                        sector = data.get('primary_exchange', 'Unknown')
                    self.sector_mappings.setdefault(sector, []).append(symbol)
            except Exception as e:
                print(f"Sector mapping failed for {symbol}: {str(e)}")
        
        # Remove unknown sectors and small sectors
        self.sector_mappings = {k: v for k, v in self.sector_mappings.items() 
                              if k != 'Unknown' and len(v) > 10}
        return self.sector_mappings
    
    def calculate_sector_weights(self):
        total_mcap = 0
        sector_mcaps = {}
        
        for sector, tickers in self.sector_mappings.items():
            sector_mcap = 0
            for symbol in tickers[:100]:
                try:
                    mcap = get_market_cap(symbol)
                    sector_mcap += mcap if mcap else 0
                except Exception as e:
                    print(f"Error getting market cap for {symbol}: {str(e)}")
            
            sector_mcaps[sector] = sector_mcap
            total_mcap += sector_mcap
        
        self.sector_weights = {sector: mcap/total_mcap if total_mcap > 0 else 1/len(sector_mcaps)
                             for sector, mcap in sector_mcaps.items()}
        return self.sector_weights
    
    def build_sector_composites(self, sample_size=50):
        print("\nBuilding sector composites...")
        self.sector_composites = {}
        
        for sector, tickers in tqdm(self.sector_mappings.items()):
            prices_data = []
            valid_tickers = []
            
            for symbol in tickers[:sample_size]:
                try:
                    prices = fetch_stock_data(symbol)
                    if prices is not None and len(prices) >= MIN_DAYS_DATA:
                        prices_data.append(prices)
                        valid_tickers.append(symbol)
                except Exception as e:
                    print(f"Error processing {symbol}: {str(e)}")
            
            if prices_data:
                composite = pd.concat(prices_data, axis=1)
                composite.columns = valid_tickers
                self.sector_composites[sector] = composite.mean(axis=1).dropna()
    
    def analyze_sector_regimes(self, n_states=3):
        print("\nAnalyzing sector regimes...")
        self.sector_analyzers = {}
        
        for sector, composite in tqdm(self.sector_composites.items()):
            try:
                analyzer = MarketRegimeAnalyzer()
                results = analyzer.analyze_regime(composite, n_states=n_states)
                
                self.sector_analyzers[sector] = {
                    'results': results,
                    'composite': composite,
                    'volatility': composite.pct_change().std()
                }
            except Exception as e:
                print(f"Error analyzing {sector}: {str(e)}")
                continue
    
    def calculate_sector_scores(self, market_regime):
        self.sector_scores = {}
        
        if not self.sector_analyzers:
            return pd.Series()
            
        for sector, data in self.sector_analyzers.items():
            try:
                if 'results' not in data or 'probabilities' not in data['results']:
                    continue
                    
                current_probs = data['results']['probabilities'][-1]
                state_labels = data['results'].get('state_labels', {})
                
                # Calculate bull/bear probabilities
                bull_prob = sum(current_probs[i] for i, label in state_labels.items() 
                              if 'Bull' in label) if state_labels else 0
                bear_prob = sum(current_probs[i] for i, label in state_labels.items() 
                              if 'Bear' in label) if state_labels else 0
                
                # Calculate momentum
                momentum = data['composite'].pct_change(21).iloc[-1] if len(data['composite']) > 21 else 0
                
                # Base score
                base_score = bull_prob - bear_prob + (momentum * 10)
                
                # Apply regime-specific adjustments
                if market_regime == "Bull":
                    beta_factor = 1 + (data['volatility'] / 0.02)
                    base_score *= beta_factor
                elif market_regime == "Bear":
                    volatility_factor = 1 + (0.04 - min(data['volatility'], 0.04)) / 0.02
                    base_score *= volatility_factor
                
                # Apply market cap weighting
                weight = self.sector_weights.get(sector, 0)
                self.sector_scores[sector] = base_score * (1 + weight * 0.5)
                
            except Exception as e:
                self.sector_scores[sector] = 0
        
        return pd.Series(self.sector_scores).sort_values(ascending=False)

class SmartStopLoss:
    def __init__(self, entry_price, atr, adx, activation_percent=0.05, base_multiplier=1.5):
        self.entry = entry_price
        self.initial_atr = atr
        self.base_adx = adx
        self.activation_percent = activation_percent
        self.base_multiplier = base_multiplier
        self.activated = False
        self.highest_high = entry_price
        self.current_stop = entry_price - (base_multiplier * atr)
        self.growth_potential = 1.0
        self.consecutive_confirmations = 0
        self.last_direction = "up"
        self.previous_close = entry_price
        
    def update(self, current_bar):
        current_high = current_bar['high']
        current_low = current_bar['low']
        current_close = current_bar['close']
        current_adx = current_bar.get('adx', self.base_adx)
        
        # Calculate volume ratio
        volume_ratio = current_bar['volume'] / current_bar.get('avg_volume', 1e6)
        
        # Update highest high
        if current_high > self.highest_high:
            self.highest_high = current_high
            self.consecutive_confirmations = 0

        # Calculate growth potential
        adx_strength = min(1.0, current_adx / 50)
        volume_boost = min(1.5, max(0.5, volume_ratio / 1.2))
        self.growth_potential = max(0.5, min(2.0, adx_strength * volume_boost))
        
        # Calculate momentum direction
        current_direction = "up" if current_close > self.previous_close else "down"
        if current_direction != self.last_direction:
            self.consecutive_confirmations = 0
        self.last_direction = current_direction
        self.previous_close = current_close
        
        # Check activation condition
        if not self.activated and self.highest_high >= self.entry * (1 + self.activation_percent):
            self.activated = True
            
        # Calculate dynamic multiplier
        if self.activated:
            # ADX-based adjustment
            adx_factor = 1.0 + (min(current_adx, 60) / 100)
            
            # Combine with growth potential
            dynamic_multiplier = self.base_multiplier * adx_factor * self.growth_potential
            
            # Set bounds for multiplier
            dynamic_multiplier = max(0.5, min(3.0, dynamic_multiplier))
            
            # Calculate new stop
            new_stop = self.highest_high - (dynamic_multiplier * self.initial_atr)
            
            # Only move stop up, never down
            if new_stop > self.current_stop:
                self.current_stop = new_stop
                
        return self.current_stop

    def should_hold(self, current_bar):
        current_low = current_bar['low']
        current_close = current_bar['close']
        rsi = current_bar.get('rsi', 50)
        volatility_ratio = current_bar.get('volatility_ratio', 1.0)
        
        # 1. Strong momentum override
        price_change = (current_close / self.previous_close - 1) * 100
        if price_change > 3:
            return True
            
        # 2. Volatility contraction protection
        if volatility_ratio < 0.7:
            return True
            
        # 3. ADX strengthening override
        if self.growth_potential > 1.5:
            return True
            
        # 4. Oversold bounce prevention
        if rsi < 40 and (current_close > current_low * 1.02):
            return True
            
        # 5. Confirmation sequence requirement
        if self.consecutive_confirmations < 2:
            self.consecutive_confirmations += 1
            return True
            
        return False

    def should_exit(self, current_bar):
        current_low = current_bar['low']
        current_close = current_bar['close']
        rsi = current_bar.get('rsi', 50)
        volatility_ratio = current_bar.get('volatility_ratio', 1.0)
        
        # Check if price is near stop level
        near_stop = current_close <= self.current_stop * 1.02
        
        # Check if stop is breached
        stop_breached = current_low <= self.current_stop
        
        if not (near_stop or stop_breached):
            return False
            
        # Check hold conditions first
        if self.should_hold(current_bar):
            return False
            
        # Confirm exit with additional criteria
        if stop_breached:
            # 1. Closing price confirmation
            if current_close <= self.current_stop:
                return True
                
            # 2. Volume confirmation (if available)
            if 'volume' in current_bar and 'avg_volume' in current_bar:
                volume_ratio = current_bar['volume'] / current_bar['avg_volume']
                if volume_ratio > 1.2:
                    return True
                    
        return False

    def get_status(self):
        return {
            'current_stop': self.current_stop,
            'growth_potential': self.growth_potential,
            'activated': self.activated,
            'consecutive_confirmations': self.consecutive_confirmations
        }

class PolygonTrendScanner:
    def __init__(self, max_tickers=None):
        self.api_key = POLYGON_API_KEY
        self.base_url = "https://api.polygon.io/v2"
        self.tickers = self.load_tickers(max_tickers)
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        self.base_multiplier = 1.5  # Default stop loss multiplier

    def load_tickers(self, max_tickers=None):
        if DATA_CACHE.get('tickers'):
            tickers = DATA_CACHE['tickers']
        else:
            tickers = get_all_tickers()
            DATA_CACHE['tickers'] = tickers
            
        if max_tickers is not None and max_tickers > 0:
            return tickers[:max_tickers]
        return tickers

    def get_polygon_data(self, ticker):
        url = f"{self.base_url}/aggs/ticker/{ticker}/range/1/day/{self.start_date}/{self.end_date}"
        params = {'adjusted': 'true', 'apiKey': self.api_key}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'results' not in data or len(data['results']) < 200:
                return None
                
            df = pd.DataFrame(data['results'])
            df['date'] = pd.to_datetime(df['t'], unit='ms')
            df.set_index('date', inplace=True)
            df.rename(columns={
                'o': 'Open',
                'h': 'High',
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume'
            }, inplace=True)
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            return None

    def calculate_indicators(self, df):
        if df is None or len(df) < 200:
            return None
            
        try:
            latest = df.iloc[-1].copy()
            sma_50 = df['Close'].rolling(50).mean().iloc[-1]
            sma_200 = df['Close'].rolling(200).mean().iloc[-1]
            
            distance_sma50 = ((latest['Close'] - sma_50) / sma_50) * 100
            distance_sma200 = ((latest['Close'] - sma_200) / sma_200) * 100
            
            # Calculate True Range and ATR
            df['prev_close'] = df['Close'].shift(1)
            df['H-L'] = df['High'] - df['Low']
            df['H-PC'] = abs(df['High'] - df['prev_close'])
            df['L-PC'] = abs(df['Low'] - df['prev_close'])
            df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            atr = df['TR'].rolling(14).mean().iloc[-1]
            
            # Calculate ADX
            plus_dm = df['High'].diff()
            minus_dm = -df['Low'].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            atr_14 = df['TR'].rolling(14).mean()
            plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
            minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(14).mean().iloc[-1]
            
            # Calculate 10-day change
            if len(df) >= 10:
                ten_day_change = ((latest['Close'] / df['Close'].iloc[-10]) - 1) * 100
            else:
                ten_day_change = 0
                
            # Calculate average volume
            avg_volume = df['Volume'].rolling(30).mean().iloc[-1]
                
            return {
                'Close': float(latest['Close']),
                'SMA_50': float(sma_50),
                'SMA_200': float(sma_200),
                'Distance_from_SMA50': float(distance_sma50),
                'Distance_from_SMA200': float(distance_sma200),
                'Volume': float(latest['Volume']),
                'ATR': float(atr),
                'ADX': float(adx),
                '10D_Change': float(ten_day_change),
                'AvgVolume': float(avg_volume)
            }
            
        except Exception as e:
            return None
        finally:
            cols_to_drop = ['prev_close', 'H-L', 'H-PC', 'L-PC', 'TR']
            for col in cols_to_drop:
                if col in df.columns:
                    df.drop(col, axis=1, inplace=True)

    def scan_tickers(self):
        results = []
        
        for i, ticker in enumerate(self.tickers):
            if i > 0 and i % 5 == 0:
                time.sleep(1)
                
            data = self.get_polygon_data(ticker)
            if data is None:
                continue
                
            indicators = self.calculate_indicators(data)
            
            if indicators is None:
                continue
                
            try:
                # Trend conditions
                above_sma50 = indicators['Close'] > indicators['SMA_50']
                above_sma200 = indicators['Close'] > indicators['SMA_200']
                strong_adx = indicators['ADX'] > 25
                
                if above_sma50 and above_sma200 and strong_adx:
                    # Calculate composite score (0-100)
                    adx_component = min(40, (indicators['ADX'] / 50) * 40)
                    sma50_component = min(30, max(0, indicators['Distance_from_SMA50']) * 0.3)
                    volume_component = min(20, math.log10(max(1, indicators['Volume']/10000)))
                    momentum_component = min(10, max(0, indicators['10D_Change']))
                    
                    score = min(100, adx_component + sma50_component + volume_component + momentum_component)
                    
                    # Initialize stop loss system
                    stop_system = SmartStopLoss(
                        entry_price=indicators['Close'],
                        atr=indicators['ATR'],
                        adx=indicators['ADX'],
                        activation_percent=0.05,
                        base_multiplier=self.base_multiplier
                    )
                    
                    # Calculate risk metrics
                    risk_per_share = indicators['Close'] - stop_system.current_stop
                    risk_percent = (risk_per_share / indicators['Close']) * 100
                    
                    results.append({
                        'Ticker': ticker,
                        'Score': round(score, 1),
                        'Price': round(indicators['Close'], 2),
                        'ADX': round(indicators['ADX'], 1),
                        'SMA50_Distance%': round(indicators['Distance_from_SMA50'], 1),
                        'SMA200_Distance%': round(indicators['Distance_from_SMA200'], 1),
                        'Volume': int(indicators['Volume']),
                        'ATR': round(indicators['ATR'], 2),
                        'ATR_Ratio': round(
                            (indicators['Close'] - indicators['SMA_50'])/indicators['ATR'], 1),
                        '10D_Change%': round(indicators['10D_Change'], 1),
                        'Initial_Stop': round(stop_system.current_stop, 2),
                        'Risk_per_Share': round(risk_per_share, 2),
                        'Risk_Percent': round(risk_percent, 2),
                        'ATR_Multiplier': self.base_multiplier,
                        'Activation_Percent': 5.0
                    })
                    
            except Exception as e:
                continue
        
        # Create DataFrame and sort
        if results:
            df_results = pd.DataFrame(results)
            df_results['Rank'] = df_results['Score'].rank(ascending=False, method='min').astype(int)
            return df_results.sort_values('Score', ascending=False)\
                            .reset_index(drop=True)\
                            [['Rank', 'Ticker', 'Score', 'Price', 
                              'ADX', 'SMA50_Distance%', 'SMA200_Distance%',
                              '10D_Change%', 'Volume', 'ATR', 'ATR_Ratio',
                              'Initial_Stop', 'Risk_per_Share', 'Risk_Percent',
                              'ATR_Multiplier', 'Activation_Percent']]
        return pd.DataFrame()

# ========================
# Utility Functions
# ========================
def get_all_tickers():
    all_tickers = []
    
    for exchange in EXCHANGES:
        url = "https://api.polygon.io/v3/reference/tickers"
        params = {
            "exchange": exchange,
            "market": "stocks",
            "active": "true",
            "limit": MAX_TICKERS_PER_EXCHANGE,
            "apiKey": POLYGON_API_KEY
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'results' in data and data['results']:
                tickers = sorted(
                    [(t['ticker'], t.get('market_cap', 0)) for t in data['results']],
                    key=lambda x: x[1],
                    reverse=True
                )
                exchange_tickers = [t[0] for t in tickers]
                all_tickers.extend(exchange_tickers)
        except Exception as e:
            # Fallback to index components if primary method fails
            if exchange == "XNYS":
                tables = pd.read_html("https://en.wikipedia.org/wiki/NYSE_Composite")
                all_tickers.extend(tables[2]['Symbol'].tolist()[:MAX_TICKERS_PER_EXCHANGE])
            elif exchange == "XNAS":
                tables = pd.read_html("https://en.wikipedia.org/wiki/NASDAQ-100")
                all_tickers.extend(tables[4]['Ticker'].tolist()[:MAX_TICKERS_PER_EXCHANGE])
            elif exchange == "XASE":
                tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_American_Stock_Exchange_companies")
                all_tickers.extend(tables[0]['Symbol'].tolist()[:MAX_TICKERS_PER_EXCHANGE])
    
    # Remove duplicates and limit total tickers
    unique_tickers = list(set(all_tickers))
    return unique_tickers[:MAX_TICKERS_PER_EXCHANGE * len(EXCHANGES)]

def fetch_stock_data(symbol, days=365):
    if symbol in DATA_CACHE['stock_data']:
        cached_data = DATA_CACHE['stock_data'][symbol]
        if datetime.now() - cached_data['timestamp'] < timedelta(days=1):
            return cached_data['data']
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": POLYGON_API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            results = response.json().get('results', [])
            if results:
                df = pd.DataFrame(results)
                df['date'] = pd.to_datetime(df['t'], unit='ms')
                result = df.set_index('date')['c']
                # Update cache
                DATA_CACHE['stock_data'][symbol] = {
                    'data': result,
                    'timestamp': datetime.now()
                }
                return result
    except Exception as e:
        print(f"Error fetching {symbol}: {str(e)}")
    return None

def get_market_cap(symbol):
    url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
    params = {"apiKey": POLYGON_API_KEY}
    time.sleep(RATE_LIMIT)
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json().get('results', {}).get('market_cap', 0)
    except:
        return 0
    return 0

# ========================
# Integrated Trading System
# ========================
class TradingSystem:
    def __init__(self):
        self.sector_system = SectorRegimeSystem()
        self.trend_scanner = PolygonTrendScanner(max_tickers=MAX_TICKERS_TO_SCAN)
        self.active_positions = {}
        self.last_scan_date = None
    
    def run_weekly_scan(self):
        print("\nRunning weekly ticker scan...")
        results = self.trend_scanner.scan_tickers()
        
        if not results.empty:
            top_3 = results.head(3)
            print(f"Top 3 Tickers:\n{top_3[['Ticker', 'Score', 'Initial_Stop']]}")
            return top_3
        return pd.DataFrame()

    def calculate_position_size(self, price, portfolio_value):
        return max(1, int(ALLOCATION_PER_TICKER / price))

    def place_alpaca_order(self, ticker, side, qty, stop_price=None):
        try:
            if side == 'buy':
                order = MarketOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                trading_client.submit_order(order)
                print(f"Placed BUY order for {qty} shares of {ticker}")
                
                # Place associated stop loss
                if stop_price:
                    stop_order = StopOrderRequest(
                        symbol=ticker,
                        qty=qty,
                        side=OrderSide.SELL,
                        stop_price=stop_price,
                        time_in_force=TimeInForce.GTC
                    )
                    trading_client.submit_order(stop_order)
                    print(f"Placed STOP LOSS at ${stop_price:.2f} for {ticker}")
                    
        except Exception as e:
            print(f"Order failed for {ticker}: {str(e)}")

    def update_stop_losses(self):
        for ticker, data in self.active_positions.items():
            try:
                # Get latest price data
                latest = self.get_latest_bar(ticker)
                if not latest:
                    continue
                
                # Add ADX for stop loss calculation
                latest['adx'] = self.get_adx(ticker)
                
                # Update stop loss
                new_stop = data['stop_loss'].update(latest)
                
                # Cancel old stop and place new one
                self.place_alpaca_order(ticker, 'sell', data['qty'], new_stop)
                
            except Exception as e:
                print(f"Stop update failed for {ticker}: {str(e)}")
    
    def get_latest_bar(self, ticker):
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev"
        params = {'adjusted': 'true', 'apiKey': POLYGON_API_KEY}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            result = response.json().get('results', [])
            if result:
                return {
                    'open': result[0]['o'],
                    'high': result[0]['h'],
                    'low': result[0]['l'],
                    'close': result[0]['c'],
                    'volume': result[0]['v']
                }
        return None
    
    def get_adx(self, ticker):
        url = f"https://api.polygon.io/v1/indicators/adx/{ticker}"
        params = {
            'timespan': 'day',
            'window': 14,
            'apiKey': POLYGON_API_KEY
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and 'values' in data['results']:
                return data['results']['values'][0]['value']
        return None

    def execute_weekly_trades(self):
        # 1. Get top 3 tickers
        top_tickers = self.run_weekly_scan()
        if top_tickers.empty:
            print("No suitable tickers found this week")
            return
        
        # 2. Get portfolio value
        try:
            portfolio_value = float(trading_client.get_account().equity)
        except:
            portfolio_value = 100000  # Default if API fails
        
        # 3. Place trades with stop losses
        for _, row in top_tickers.iterrows():
            ticker = row['Ticker']
            price = row['Price']
            stop_price = row['Initial_Stop']
            qty = self.calculate_position_size(price, portfolio_value)
            
            # Place order
            self.place_alpaca_order(ticker, 'buy', qty, stop_price)
            
            # Initialize stop loss tracker
            self.active_positions[ticker] = {
                'qty': qty,
                'entry_price': price,
                'stop_loss': SmartStopLoss(
                    entry_price=price,
                    atr=row['ATR'],
                    adx=row['ADX']
                )
            }

    def run_market_regime_analysis(self):
        try:
            print("\nRunning market regime analysis...")
            tickers = get_all_tickers()
            self.sector_system.map_tickers_to_sectors(tickers)
            self.sector_system.calculate_sector_weights()
            self.sector_system.build_sector_composites()
            
            market_index = self.sector_system.overall_analyzer.prepare_market_data(tickers)
            market_results = self.sector_system.overall_analyzer.analyze_regime(market_index)
            
            current_regime = market_results['regimes'][-1]
            print(f"Current Market Regime: {current_regime}")
            
            # Update scanner parameters based on regime
            if "Bull" in current_regime:
                self.trend_scanner.base_multiplier = 1.2
                print("Bull market detected: Using tighter stops (1.2x ATR)")
            elif "Bear" in current_regime:
                self.trend_scanner.base_multiplier = 2.0
                print("Bear market detected: Using wider stops (2.0x ATR)")
            
            return current_regime
            
        except Exception as e:
            print(f"Regime analysis failed: {str(e)}")
            return "Neutral"

    def monitor_and_update(self):
        print("Starting Trading System...")
        print("Press Ctrl+C to stop")
        
        # Initial market regime analysis
        self.run_market_regime_analysis()
        
        while True:
            try:
                current_time = datetime.now()
                weekday = current_time.weekday()
                hour = current_time.hour
                
                # Run weekly trades on Sundays at 4 PM
                if weekday == 6 and hour == 16:
                    if not self.last_scan_date or (current_time - self.last_scan_date).days >= 7:
                        print("\n=== EXECUTING WEEKLY TRADES ===")
                        self.execute_weekly_trades()
                        self.last_scan_date = current_time
                
                # Update stops hourly during market hours
                if 9 <= hour <= 16:
                    if self.active_positions:
                        print("\nUpdating stop losses...")
                        self.update_stop_losses()
                
                # Run market regime analysis at 5 AM daily
                if hour == 5:
                    self.run_market_regime_analysis()
                
                # Sleep for 1 hour
                time.sleep(3600)
                
            except KeyboardInterrupt:
                print("\nExiting trading system...")
                break
            except Exception as e:
                print(f"System error: {str(e)}")
                time.sleep(300)

# ===============
# Main Execution
# ===============
if __name__ == "__main__":
    trading_system = TradingSystem()
    trading_system.monitor_and_update()