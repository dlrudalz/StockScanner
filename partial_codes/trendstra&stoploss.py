import sys
import math
import time
import threading
import queue
import json
import requests
import pytz
import warnings
from datetime import datetime, timedelta, timezone as tz
import pandas as pd
import numpy as np
import pandas_ta as ta
from websocket import create_connection
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QPushButton, 
    QTabWidget, QCheckBox, QMessageBox, QTextEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QColor, QBrush, QFont
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import concurrent.futures

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated")

# Polygon.io configuration
POLYGON_API_KEY = "OZzn0oK0H2yG6rpIvVhGfgXgnUTrL31z"
REST_API_URL = "https://api.polygon.io"
WEBSOCKET_URL = "wss://socket.polygon.io/stocks"

class PolygonDataHandler:
    """Handles Polygon.io REST API and WebSocket data with parallel historical data loading"""
    def __init__(self, tickers):
        self.tickers = tickers
        self.api_key = POLYGON_API_KEY
        self.historical_data = {t: pd.DataFrame() for t in tickers}
        self.realtime_data = {t: None for t in tickers}
        self.data_queue = queue.Queue()
        self.running = True
        self.thread = None
        self.data_lock = threading.Lock()  # For thread-safe data access
        print("Data handler initialized (data not loaded yet)")
    
    def load_historical_data(self):
        """Load initial historical data for all tickers in parallel"""
        print("Parallel loading initial historical data...")
        batches = [self.tickers[i:i+5] for i in range(0, len(self.tickers), 5)]
        
        for i, batch in enumerate(batches):
            print(f"Processing batch {i+1}/{len(batches)}: {batch}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = {executor.submit(self.load_ticker_data, ticker): ticker for ticker in batch}
                
                for future in concurrent.futures.as_completed(futures):
                    ticker = futures[future]
                    try:
                        result = future.result()
                        if not result.empty:
                            with self.data_lock:
                                self.historical_data[ticker] = result
                            print(f"Loaded {len(result)} historical bars for {ticker}")
                    except Exception as e:
                        print(f"Error loading data for {ticker}: {str(e)}")
            
            # Respect API rate limits (5 requests per minute)
            if i < len(batches) - 1:
                print("Waiting 60 seconds for rate limit reset...")
                time.sleep(60)
    
    def load_ticker_data(self, ticker):
        """Load data for a single ticker (thread worker function)"""
        try:
            to_date = datetime.now(tz.utc) - timedelta(days=1)
            from_date = to_date - timedelta(days=30)
            
            url = f"{REST_API_URL}/v2/aggs/ticker/{ticker}/range/1/minute/" \
                  f"{from_date.strftime('%Y-%m-%d')}/{to_date.strftime('%Y-%m-%d')}" \
                  f"?adjusted=true&sort=asc&limit=50000&apiKey={self.api_key}"
            
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'OK' and data['resultsCount'] > 0:
                    df = pd.DataFrame(data['results'])
                    df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True)
                    df.set_index('timestamp', inplace=True)
                    df.rename(columns={
                        'o': 'Open', 'h': 'High', 'l': 'Low', 
                        'c': 'Close', 'v': 'Volume'
                    }, inplace=True)
                    return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            return pd.DataFrame()
        except Exception as e:
            print(f"Error in thread for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def run_websocket(self):
        """Connect to Polygon WebSocket and handle real-time data"""
        print("WebSocket thread started")
        while self.running:
            try:
                ws = create_connection(WEBSOCKET_URL)
                ws.send(json.dumps({"action": "auth", "params": self.api_key}))
                params = ",".join([f"A.{t}" for t in self.tickers])
                ws.send(json.dumps({"action": "subscribe", "params": params}))
                print("WebSocket connected and authenticated")
                
                while self.running:
                    try:
                        message = ws.recv()
                        if message:
                            data = json.loads(message)
                            for event in data:
                                self.process_websocket_event(event)
                    except Exception as e:
                        print(f"WebSocket error: {str(e)}")
                        break
            except Exception as e:
                print(f"Connection error: {str(e)}")
                time.sleep(5)
    
    def process_websocket_event(self, event):
        """Process real-time WebSocket events"""
        try:
            event_type = event.get('ev')
            ticker = event.get('sym')
            
            if not ticker or ticker not in self.tickers:
                return
                
            if event_type == 'AM':  # Aggregate Minute
                timestamp = datetime.fromtimestamp(event['s'] / 1000.0, tz=tz.utc)
                new_bar = pd.DataFrame({
                    'Open': [event['o']], 'High': [event['h']], 
                    'Low': [event['l']], 'Close': [event['c']], 
                    'Volume': [event['v']]
                }, index=[timestamp])
                
                with self.data_lock:
                    if not self.historical_data[ticker].empty:
                        # Append new bar to historical data
                        self.historical_data[ticker] = pd.concat([self.historical_data[ticker], new_bar])
                        # Remove duplicates and keep last
                        self.historical_data[ticker] = self.historical_data[ticker][~self.historical_data[ticker].index.duplicated(keep='last')]
                        # Keep only the last 1000 bars
                        self.historical_data[ticker] = self.historical_data[ticker].iloc[-1000:]
                    else:
                        self.historical_data[ticker] = new_bar
                
                self.realtime_data[ticker] = {
                    'Open': event['o'], 'High': event['h'], 'Low': event['l'],
                    'Close': event['c'], 'Volume': event['v'],
                    'timestamp': timestamp
                }
                
                self.data_queue.put((ticker, self.realtime_data[ticker]))
        except Exception as e:
            print(f"Event processing error: {str(e)}")
    
    def start(self):
        """Start WebSocket connection and load historical data"""
        if self.thread and self.thread.is_alive():
            print("Data feed already running")
            return
            
        # First load historical data
        self.load_historical_data()
        
        # Then start WebSocket
        self.running = True
        self.thread = threading.Thread(target=self.run_websocket)
        self.thread.daemon = True
        self.thread.start()
        print("Data feed started")
    
    def stop(self):
        """Stop WebSocket connection"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        print("Data feed stopped")
    
    def get_latest(self, ticker):
        """Get latest data point with fallback to historical data"""
        # First try real-time data
        if ticker in self.realtime_data and self.realtime_data[ticker] is not None:
            return self.realtime_data[ticker]
        
        # Fallback to historical data if no real-time data available
        with self.data_lock:
            if ticker in self.historical_data and not self.historical_data[ticker].empty:
                last_row = self.historical_data[ticker].iloc[-1]
                return {
                    'Open': last_row['Open'],
                    'High': last_row['High'],
                    'Low': last_row['Low'],
                    'Close': last_row['Close'],
                    'Volume': last_row['Volume'],
                    'timestamp': self.historical_data[ticker].index[-1]
                }
        
        return None
    
    def get_historical(self, ticker, period=100):
        with self.data_lock:
            if ticker not in self.historical_data:
                return None
            df = self.historical_data[ticker]
            return df.iloc[-period:] if len(df) >= period else df


class SmartStopLoss:
    """Robust stop loss system with error prevention"""
    def __init__(self, entry_price, atr, adx, volatility_factor=1.5, 
                 hard_stop_percent=0.08, profit_target_ratio=3.0):
        self.entry_price = entry_price
        self.atr = atr
        self.adx = adx
        self.volatility_factor = volatility_factor
        self.hard_stop_percent = hard_stop_percent
        self.profit_target_ratio = profit_target_ratio
        
        self.initial_stop = self.calculate_initial_stop()
        self.trailing_stop = self.initial_stop
        self.hard_stop = entry_price * (1 - hard_stop_percent)
        self.profit_target = entry_price + (entry_price - self.initial_stop) * profit_target_ratio
        self.profit_target_2 = entry_price + (entry_price - self.initial_stop) * (profit_target_ratio * 2)
        
        self.history = [{
            'timestamp': datetime.now(tz.utc),
            'price': entry_price,
            'trailing_stop': self.trailing_stop,
            'hard_stop': self.hard_stop,
            'initial_stop': self.initial_stop,
            'profit_target': self.profit_target
        }]
        
    def calculate_initial_stop(self):
        base_stop = self.entry_price - self.atr * self.volatility_factor
        if self.adx > 40:  # Strong trend
            return base_stop * 0.95
        elif self.adx < 20:  # Weak trend
            return base_stop * 1.05
        return base_stop
    
    def update_trailing_stop(self, current_price, timestamp):
        new_stop = current_price - self.atr * self.volatility_factor * 0.8
        if new_stop > self.trailing_stop:
            self.trailing_stop = new_stop
        self.trailing_stop = max(self.trailing_stop, self.hard_stop)
        
        self.history.append({
            'timestamp': timestamp,
            'price': current_price,
            'trailing_stop': self.trailing_stop,
            'hard_stop': self.hard_stop,
            'initial_stop': self.initial_stop,
            'profit_target': self.profit_target
        })
        return self.trailing_stop
    
    def check_stop_hit(self, current_price):
        if current_price <= self.trailing_stop:
            return "trailing_stop"
        if current_price <= self.hard_stop:
            return "hard_stop"
        return None
    
    def detect_market_regime(self):
        if self.adx > 40:
            return "strong_trend"
        elif self.adx > 25:
            return "trending"
        elif self.adx > 20:
            return "transition"
        else:
            return "choppy"


# Sector Rotation System
class SectorSystem:
    """Tracks sector rotation and market regimes"""
    def __init__(self):
        self.sector_scores = {
            "Technology": 75,
            "Healthcare": 65,
            "Financial": 60,
            "Consumer": 70,
            "Energy": 55,
            "Industrial": 62,
            "Utilities": 58
        }
        
        self.sector_mappings = {
            "Technology": ["AAPL", "MSFT", "GOOG", "NVDA", "INTC", "AMD"],
            "Financial": ["JPM", "V", "MA", "GS", "BAC"],
            "Consumer": ["AMZN", "DIS", "META", "NFLX", "KO", "PEP"],
            "Healthcare": ["JNJ", "PFE", "UNH", "MRK", "ABT"],
            "Energy": ["XOM", "CVX", "COP", "SLB"],
            "Industrial": ["BA", "CAT", "HON", "MMM"],
            "Utilities": ["NEE", "DUK", "SO", "D"]
        }
        
        # Market regime analyzer
        self.overall_analyzer = MarketRegimeAnalyzer()
    
    def get_ticker_sector(self, ticker):
        """Get sector for a ticker"""
        for sector, tickers in self.sector_mappings.items():
            if ticker in tickers:
                return sector
        return "Unknown"


class MarketRegimeAnalyzer:
    """Analyzes overall market regime"""
    def __init__(self):
        self.state_labels = "Bull Market"  # Placeholder - implement actual analysis
        self.strength_index = 75


class TradingSystem(QThread):
    """Complete trading system with enhanced error handling and scoring"""
    # Stock replacement configuration
    POSITION_EVALUATION_INTERVAL = 3600  # Evaluate positions hourly
    MIN_HOLDING_DAYS = 5  # Minimum days before considering replacement
    SCORE_DEGRADATION_THRESHOLD = 0.8  # 20% score drop triggers review
    RELATIVE_STRENGTH_MARGIN = 0.15  # 15% better score required for replacement
    MIN_SCORE_FOR_ENTRY = 70  # Minimum score to enter a position
    
    update_signal = pyqtSignal(dict)
    log_signal = pyqtSignal(str)
    
    def __init__(self, tickers, capital=100000, risk_per_trade=0.01):
        super().__init__()
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.tickers = tickers
        self.positions = {}
        self.trade_log = []
        self.data_handler = None  # Will be created when started
        self.running = False  # Not running initially
        self.eastern = pytz.timezone('US/Eastern')
        self.evaluation_interval = 30
        self.last_evaluation_time = time.time()
        self.last_opportunities = []
        
        # Replacement system attributes
        self.position_evaluation_times = {}
        self.runner_ups = []  # Track top opportunities
        self.sector_system = SectorSystem()  # Sector rotation system
        self.executed_tickers = set()  # Track executed trades
        self.last_evaluation_timestamp = 0
        
        print("Trading system initialized (not started)")
        
    def run(self):
        """Main trading system loop"""
        if not self.running:
            return
            
        self.log_signal.emit("Trading system started")
        self.data_handler = PolygonDataHandler(self.tickers)
        self.data_handler.start()
        
        while self.running:
            try:
                # Process data points
                while not self.data_handler.data_queue.empty() and self.running:
                    ticker, data = self.data_handler.data_queue.get()
                    self.update_positions(ticker, data)
                
                # Close old positions
                self.close_old_positions()
                
                # Evaluate opportunities
                if self.should_evaluate_opportunities():
                    self.evaluate_opportunities()
                    # Enter positions for top opportunities
                    self.enter_top_opportunities()
                
                # Evaluate positions for replacement
                current_time = time.time()
                if current_time - self.last_evaluation_timestamp > self.POSITION_EVALUATION_INTERVAL:
                    self.evaluate_and_replace_positions()
                    self.last_evaluation_timestamp = current_time
                
                # Emit state update
                self.update_signal.emit(self.get_current_state())
                time.sleep(0.5)
                
            except Exception as e:
                self.log_signal.emit(f"System error: {str(e)}")
        
        # Clean up when stopping
        if self.data_handler:
            self.data_handler.stop()
        self.log_signal.emit("Trading system stopped")
    
    def update_positions(self, ticker, data):
        """Update all positions for a ticker"""
        for position_ticker, position in list(self.positions.items()):
            if position_ticker == ticker:
                current_price = data['Close']
                stop_system = position['stop_system']
                new_stop = stop_system.update_trailing_stop(
                    current_price=current_price,
                    timestamp=data['timestamp']
                )
                stop_trigger = stop_system.check_stop_hit(current_price)
                if stop_trigger:
                    self.exit_position(ticker, current_price, stop_trigger)
                elif current_price >= stop_system.profit_target_2:
                    self.exit_position(ticker, current_price, "profit_target_2")
                elif current_price >= stop_system.profit_target:
                    self.partial_exit(ticker, 0.5, current_price, "profit_target_1")
    
    def close_old_positions(self):
        """Close positions older than 4 hours"""
        for ticker in list(self.positions.keys()):
            position = self.positions[ticker]
            duration = (datetime.now(tz.utc) - position['entry_time']).seconds / 60
            if duration > 240:  # 4 hours
                data = self.data_handler.get_latest(ticker)
                if data:
                    self.exit_position(ticker, data['Close'], "time expiration")
    
    def should_evaluate_opportunities(self):
        """Determine if we should evaluate new opportunities"""
        current_time = time.time()
        return (
            (current_time - self.last_evaluation_time > self.evaluation_interval) and
            (len(self.positions) < 5) and 
            self.is_market_open()
        )
    
    def evaluate_opportunities(self):
        """Evaluate and rank trading opportunities using parallel scanning"""
        self.last_evaluation_time = time.time()
        opportunities = []
        tickers_to_score = [t for t in self.tickers if t not in self.positions]
        
        if not tickers_to_score:
            self.log_signal.emit("No tickers to evaluate (all in positions)")
            return
            
        # Use parallel processing for scoring
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {
                executor.submit(self.score_trade_opportunity, ticker): ticker
                for ticker in tickers_to_score
            }
            
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    if result:
                        opportunities.append(result)
                except Exception as e:
                    self.log_signal.emit(f"Scoring error for {ticker}: {str(e)}")
        
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        self.last_opportunities = opportunities[:3]
        self.runner_ups = opportunities[:10]  # Store top 10 for replacement
        self.log_signal.emit(f"Parallel evaluated {len(opportunities)} opportunities")
    
    def enter_top_opportunities(self):
        """Enter positions for top opportunities that meet criteria"""
        if not self.last_opportunities:
            self.log_signal.emit("No opportunities to enter")
            return
            
        for opp in self.last_opportunities[:1]:  # Only take the top opportunity
            if opp['score'] >= self.MIN_SCORE_FOR_ENTRY and opp['ticker'] not in self.positions:
                # Get fresh data for entry
                data = self.data_handler.get_latest(opp['ticker'])
                if not data:
                    self.log_signal.emit(f"No data for {opp['ticker']}")
                    continue
                    
                # Get historical data first
                df = self.data_handler.get_historical(opp['ticker'], 50)
                if df.empty:
                    self.log_signal.emit(f"Insufficient data for {opp['ticker']}")
                    continue
                    
                current_price = data['Close']
                
                # Recalculate indicators for fresh data
                atr = df.ta.atr(length=14).iloc[-1]
                adx = df.ta.adx(length=14)['ADX_14'].iloc[-1]
                
                self.enter_position(
                    opp['ticker'],
                    current_price,
                    atr,
                    adx,
                    opp['score']  # Store original score
                )
            else:
                if opp['ticker'] in self.positions:
                    self.log_signal.emit(f"Already in position for {opp['ticker']}")
                else:
                    self.log_signal.emit(f"Score too low for {opp['ticker']}: {opp['score']} < {self.MIN_SCORE_FOR_ENTRY}")
    
    def is_market_open(self):
        """Check if market is open"""
        now = datetime.now(self.eastern)
        if now.weekday() >= 5:  # Weekend
            return False
            
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now <= market_close
    
    def get_current_state(self):
        """Get current system state for UI"""
        total_profit = sum(trade['profit'] for trade in self.trade_log if trade['profit'] is not None)
        winning_trades = sum(1 for trade in self.trade_log if trade['profit'] and trade['profit'] > 0)
        win_rate = winning_trades / len(self.trade_log) * 100 if self.trade_log else 0
        
        # Prepare active positions
        active_positions = []
        for ticker, pos in self.positions.items():
            current_data = self.data_handler.get_latest(ticker) if self.data_handler else None
            current_price = current_data['Close'] if current_data and current_data else pos['entry_price']
            gain = (current_price / pos['entry_price'] - 1) * 100
            risk = (pos['entry_price'] - pos['stop_system'].trailing_stop) / pos['entry_price'] * 100
            regime = pos['stop_system'].detect_market_regime()
            
            active_positions.append({
                'ticker': ticker, 'shares': pos['shares'],
                'entry_price': pos['entry_price'], 'current_price': current_price,
                'gain': gain, 'trailing_stop': pos['stop_system'].trailing_stop,
                'hard_stop': pos['stop_system'].hard_stop, 
                'profit_target': pos['stop_system'].profit_target,
                'risk': risk, 'regime': regime,
                'original_score': pos.get('original_score', 0)
            })
        
        # Prepare recent trades (last 5, most recent first)
        recent_trades = self.trade_log[-5:][::-1]
        
        # Prepare opportunities
        opportunities = []
        for opp in self.last_opportunities:
            status = "ENTERED" if opp['ticker'] in self.positions else "PASSED"
            opportunities.append({
                'ticker': opp['ticker'], 'score': opp['score'],
                'price': opp['price'], 'adx': opp['adx'],
                'atr': opp['atr'], 'rsi': opp['rsi'],
                'volume': opp['volume'], 'status': status
            })
        
        # Prepare runner-ups
        runner_ups = []
        for opp in self.runner_ups:
            runner_ups.append({
                'ticker': opp['ticker'], 'score': opp['score'],
                'price': opp['price'], 'adx': opp['adx'],
                'atr': opp['atr'], 'rsi': opp['rsi'],
                'volume': opp['volume']
            })
        
        return {
            'timestamp': datetime.now(tz.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
            'market_open': self.is_market_open(),
            'capital': self.capital,
            'positions_count': len(self.positions),
            'trade_count': len(self.trade_log),
            'total_profit': total_profit,
            'win_rate': win_rate,
            'active_positions': active_positions,
            'recent_trades': recent_trades,
            'top_opportunities': opportunities,
            'runner_ups': runner_ups
        }
    
    def score_trade_opportunity(self, ticker):
        """Score trade opportunity (0-100 scale)"""
        try:
            if not self.data_handler:
                return None
                
            if not self.is_market_open():
                return None
                
            data = self.data_handler.get_latest(ticker)
            if not data:
                self.log_signal.emit(f"No data available for {ticker}")
                return None
                
            df = self.data_handler.get_historical(ticker, 50)
            if df is None or df.empty or len(df) < 14:
                self.log_signal.emit(f"Insufficient data for {ticker}")
                return None
                
            # Calculate indicators
            atr = df.ta.atr(length=14).iloc[-1]
            adx = df.ta.adx(length=14)['ADX_14'].iloc[-1]
            rsi = df.ta.rsi(length=14).iloc[-1]
            
            price = data['Close']
            volume = data['Volume']
            
            # Validate indicators
            if any(math.isnan(x) for x in [atr, adx, rsi, price]):
                self.log_signal.emit(f"NaN values in indicators for {ticker}")
                return None
                
            avg_volume = df['Volume'].rolling(14).mean().iloc[-1]
            if math.isnan(avg_volume) or avg_volume <= 0:
                avg_volume = df['Volume'].mean()
                
            if volume <= 0:
                volume = avg_volume
                
            # Create stop system for scoring
            stop_system = SmartStopLoss(
                entry_price=price, atr=atr, adx=adx,
                volatility_factor=1.5, hard_stop_percent=0.08,
                profit_target_ratio=3.0
            )
            
            # 1. Trend Strength Score (ADX-based)
            adx_score = min(100, max(0, (adx - 20) * 5))
            
            # 2. Volatility Quality Score (ATR-based)
            atr_pct = atr / price
            if atr_pct < 0.015:
                atr_score = 20 + (atr_pct / 0.015) * 30
            elif atr_pct > 0.03:
                atr_score = 80 - min(30, (atr_pct - 0.03) * 1000)
            else:
                atr_score = 50 + (atr_pct - 0.015) * 2000
                
            # 3. Risk-Reward Score
            risk = price - stop_system.initial_stop
            reward = stop_system.profit_target - price
            rr_ratio = reward / risk if risk > 0 else 0
            rr_score = min(100, rr_ratio * 25)
            
            # 4. Volume Confirmation Score
            volume_ratio = volume / avg_volume
            volume_score = min(100, volume_ratio * 50)
            
            # 5. Momentum Score (RSI-based)
            if rsi > 70:
                rsi_score = 100 - min(30, (rsi - 70) * 2)
            elif rsi < 30:
                rsi_score = 100 - min(30, (30 - rsi) * 2)
            else:
                rsi_score = 80 - abs(rsi - 50)
            
            # Weighted composite score
            composite_score = (
                0.30 * adx_score + 0.25 * atr_score + 
                0.20 * rr_score + 0.15 * volume_score + 
                0.10 * rsi_score
            )
            
            self.log_signal.emit(
                f"Scored {ticker}: {composite_score:.1f} (ADX: {adx:.1f}, ATR: {atr:.2f}, RSI: {rsi:.1f})"
            )
            return {
                'ticker': ticker, 'score': composite_score,
                'price': price, 'atr': atr, 'adx': adx,
                'rsi': rsi, 'volume': volume,
                'risk_reward': rr_ratio
            }
        except Exception as e:
            self.log_signal.emit(f"Scoring error for {ticker}: {str(e)}")
            return None
    
    def enter_position(self, ticker, price, atr, adx, original_score):
        """Enter new position with original score"""
        try:
            if ticker in self.positions:
                self.log_signal.emit(f"Already in position for {ticker}")
                return
                
            stop_system = SmartStopLoss(
                entry_price=price, atr=atr, adx=adx,
                volatility_factor=1.5, hard_stop_percent=0.08,
                profit_target_ratio=3.0
            )
            
            risk_per_share = price - stop_system.initial_stop
            if risk_per_share <= 0:
                self.log_signal.emit(f"Invalid risk for {ticker}: risk_per_share={risk_per_share}")
                return
                
            position_size = math.floor((self.capital * self.risk_per_trade) / risk_per_share)
            if position_size <= 0:
                self.log_signal.emit(f"Invalid position size for {ticker}: {position_size}")
                return
                
            self.positions[ticker] = {
                'entry_price': price,
                'entry_time': datetime.now(tz.utc),
                'shares': position_size,
                'stop_system': stop_system,
                'original_score': original_score  # Store for future comparison
            }
            
            self.trade_log.append({
                'ticker': ticker, 'entry': price,
                'entry_time': datetime.now(tz.utc),
                'exit': None, 'exit_time': None,
                'profit': None, 'percent_gain': None,
                'duration': None, 'exit_reason': None,
                'shares': position_size
            })
            
            self.executed_tickers.add(ticker)
            self.log_signal.emit(f"Entered {ticker} at ${price:.2f} - {position_size} shares")
        except Exception as e:
            self.log_signal.emit(f"Entry error for {ticker}: {str(e)}")

    def exit_position(self, ticker, exit_price, reason):
        """Fully exit position"""
        try:
            if ticker not in self.positions:
                self.log_signal.emit(f"Exit failed: no position for {ticker}")
                return
                
            position = self.positions.pop(ticker)
            entry_price = position['entry_price']
            shares = position['shares']
            entry_time = position['entry_time']
            
            profit = (exit_price - entry_price) * shares
            percent_gain = (exit_price / entry_price - 1) * 100
            duration = (datetime.now(tz.utc) - entry_time).total_seconds() / 60
            
            self.capital += profit
            
            # Find the open trade for this position
            for trade in reversed(self.trade_log):
                if trade['ticker'] == ticker and trade['exit'] is None:
                    trade['exit'] = exit_price
                    trade['exit_time'] = datetime.now(tz.utc)
                    trade['profit'] = profit
                    trade['percent_gain'] = percent_gain
                    trade['duration'] = duration
                    trade['exit_reason'] = reason
                    break
                    
            self.log_signal.emit(
                f"Exited {ticker} at ${exit_price:.2f} (Reason: {reason}, Profit: ${profit:.2f})"
            )
        except Exception as e:
            self.log_signal.emit(f"Exit error for {ticker}: {str(e)}")

    def partial_exit(self, ticker, percent, exit_price, reason):
        """Partially exit position"""
        try:
            if ticker not in self.positions:
                self.log_signal.emit(f"Partial exit failed: no position for {ticker}")
                return
                
            position = self.positions[ticker]
            if percent <= 0 or percent >= 1:
                self.log_signal.emit(f"Invalid partial exit percent for {ticker}: {percent}")
                return
                
            shares_to_sell = math.floor(position['shares'] * percent)
            if shares_to_sell <= 0:
                self.log_signal.emit(f"Invalid shares to sell for {ticker}: {shares_to_sell}")
                return
                
            profit = (exit_price - position['entry_price']) * shares_to_sell
            percent_gain = (exit_price / position['entry_price'] - 1) * 100
            
            self.capital += profit
            position['shares'] -= shares_to_sell
            
            self.trade_log.append({
                'ticker': ticker,
                'entry': position['entry_price'],
                'entry_time': position['entry_time'],
                'exit': exit_price,
                'exit_time': datetime.now(tz.utc),
                'profit': profit,
                'percent_gain': percent_gain,
                'duration': (datetime.now(tz.utc) - position['entry_time']).total_seconds() / 60,
                'exit_reason': reason,
                'shares': shares_to_sell
            })
            
            self.log_signal.emit(
                f"Partial exit {ticker} {shares_to_sell} shares at ${exit_price:.2f} (Profit: ${profit:.2f})"
            )
            
            if position['shares'] <= 0:
                self.positions.pop(ticker)
                self.log_signal.emit(f"Fully exited {ticker} via partial exits")
                
        except Exception as e:
            self.log_signal.emit(f"Partial exit error for {ticker}: {str(e)}")
    
    # Stock Replacement System -------------------------------------------------
    def evaluate_and_replace_positions(self):
        """Evaluate active positions and replace weak ones"""
        if not self.positions or not self.runner_ups:
            return
            
        self.log_signal.emit("Evaluating position strength...")
        current_prices = self.get_current_prices()
        
        for ticker, position in list(self.positions.items()):
            # Skip recently opened positions
            holding_days = (datetime.now(tz.utc) - position['entry_time']).days
            if holding_days < self.MIN_HOLDING_DAYS:
                continue
                
            # Calculate current score
            current_score = self.calculate_current_score(ticker, position, current_prices[ticker])
            
            # Calculate score degradation
            original_score = position.get('original_score', current_score)
            if original_score <= 0:
                continue
                
            score_ratio = current_score / original_score
            
            if score_ratio < self.SCORE_DEGRADATION_THRESHOLD:
                self.log_signal.emit(
                    f"Position degradation: {ticker} score {original_score:.1f} -> {current_score:.1f} "
                    f"({score_ratio*100:.1f}%)"
                )
                self.find_replacement(ticker, current_score, current_prices[ticker])
    
    def get_current_prices(self):
        """Get current prices for all active positions"""
        prices = {}
        for ticker in self.positions:
            if not self.data_handler:
                prices[ticker] = self.positions[ticker]['entry_price']
                continue
                
            latest = self.data_handler.get_latest(ticker)
            prices[ticker] = latest['Close'] if latest else self.positions[ticker]['entry_price']
        return prices
    
    def calculate_current_score(self, ticker, position, current_price):
        """Calculate current position strength score"""
        try:
            if not self.data_handler:
                return 0
                
            # Get updated technical data
            df = self.data_handler.get_historical(ticker, 50)
            if df is None or df.empty:
                return 0
                
            # Calculate indicators
            adx = df.ta.adx(length=14)['ADX_14'].iloc[-1]
            rsi = df.ta.rsi(length=14).iloc[-1]
            volume = df['Volume'].iloc[-1]
            avg_volume = df['Volume'].rolling(14).mean().iloc[-1]
            
            # Get current market regime
            regime = self.sector_system.overall_analyzer.state_labels
            
            # Position performance metrics
            price_change = ((current_price - position['entry_price']) / position['entry_price']) * 100
            volume_ratio = volume / avg_volume
            
            # Regime-based weighting
            regime_factor = 1.2 if "Bull" in regime else 0.8
            
            # Calculate composite score
            score = (
                0.4 * min(100, adx) + 
                0.3 * max(0, price_change) + 
                0.2 * min(100, volume_ratio * 50) + 
                0.1 * regime_factor * 100
            )
            return max(0, min(100, score))
            
        except Exception as e:
            self.log_signal.emit(f"Score calculation failed for {ticker}: {str(e)}")
            return 0
    
    def find_replacement(self, weak_ticker, weak_score, current_price):
        """Find suitable replacement for weak position"""
        self.log_signal.emit(f"Seeking replacement for {weak_ticker} (Score: {weak_score:.1f})")
        best_candidate = None
        best_score = weak_score
        
        # Check runner-ups first
        for candidate in self.runner_ups:
            ticker = candidate['ticker']
            if ticker in self.positions or ticker in self.executed_tickers:
                continue
                
            # Get updated candidate data
            candidate_score = self.score_trade_opportunity(ticker)
            if not candidate_score:
                continue
                
            # Check if significantly better
            if candidate_score['score'] > best_score * (1 + self.RELATIVE_STRENGTH_MARGIN):
                best_candidate = candidate_score
                best_score = candidate_score['score']
        
        # If no suitable runner-up, scan new candidates
        if not best_candidate:
            best_candidate = self.find_replacement_from_scan(weak_score)
        
        # Execute replacement
        if best_candidate:
            self.log_signal.emit(f"Replacing {weak_ticker} with {best_candidate['ticker']}")
            self.execute_replacement(weak_ticker, best_candidate)
    
    def find_replacement_from_scan(self, min_score):
        """Scan for new replacement candidates"""
        self.log_signal.emit("Scanning for new replacement candidates...")
        opportunities = []
        for ticker in self.tickers:
            if ticker in self.positions or ticker in self.executed_tickers:
                continue
            score_result = self.score_trade_opportunity(ticker)
            if score_result and score_result['score'] > min_score * (1 + self.RELATIVE_STRENGTH_MARGIN):
                opportunities.append(score_result)
        
        if not opportunities:
            return None
            
        # Add sector strength to score
        sector_scores = self.sector_system.sector_scores
        for opp in opportunities:
            sector = self.get_ticker_sector(opp['ticker'])
            sector_strength = sector_scores.get(sector, 50)
            opp['score'] *= (1 + sector_strength / 200)
        
        # Return best candidate
        return max(opportunities, key=lambda x: x['score'])
    
    def execute_replacement(self, old_ticker, new_candidate):
        """Execute the replacement trade"""
        # Close old position
        position = self.positions.get(old_ticker)
        if position:
            current_price = self.data_handler.get_latest(old_ticker)['Close']
            self.exit_position(old_ticker, current_price, "Replaced by stronger candidate")
            
            # Place new trade
            new_data = self.data_handler.get_latest(new_candidate['ticker'])
            if new_data:
                df = self.data_handler.get_historical(new_candidate['ticker'], 50)
                if df is None or df.empty:
                    return
                
                # Recalculate indicators for fresh data
                atr = df.ta.atr(length=14).iloc[-1]
                adx = df.ta.adx(length=14)['ADX_14'].iloc[-1]
                
                current_price = new_data['Close']
                
                self.enter_position(
                    new_candidate['ticker'],
                    current_price,
                    atr,
                    adx,
                    new_candidate['score']  # Store original score
                )
                self.log_signal.emit(f"Replaced {old_ticker} with {new_candidate['ticker']}")
    
    def get_ticker_sector(self, ticker):
        """Get sector for a ticker"""
        return self.sector_system.get_ticker_sector(ticker)
    # End Stock Replacement System ---------------------------------------------
    
    def start_system(self):
        """Start the trading system"""
        if not self.running:
            self.running = True
            self.start()  # Start the QThread
            return True
        return False
    
    def stop_system(self):
        """Stop the trading system"""
        if self.running:
            self.log_signal.emit("Stopping trading system...")
            self.running = False
            self.wait(5000)  # Wait up to 5 seconds for thread to finish
            return True
        return False


class PositionPlot(FigureCanvas):
    """Position visualization widget"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout()
        
    def plot_position(self, ticker, position, historical_data):
        """Plot position with stops and price history"""
        self.ax.clear()
        
        if historical_data is None or historical_data.empty:
            return
            
        df = historical_data.iloc[-50:]
        stop_system = position['stop_system']
        stop_history = pd.DataFrame(stop_system.history)
        stop_history.set_index('timestamp', inplace=True)
        
        # Plot price and stops
        df['Close'].plot(ax=self.ax, label='Price', color='blue', linewidth=2)
        stop_history['initial_stop'].plot(ax=self.ax, label='Initial Stop', color='red', linestyle='--')
        stop_history['trailing_stop'].plot(ax=self.ax, label='Trailing Stop', color='orange', linewidth=2)
        stop_history['hard_stop'].plot(ax=self.ax, label='Hard Stop', color='darkred', linestyle=':')
        stop_history['profit_target'].plot(ax=self.ax, label='Profit Target', color='green', linestyle='--')
        
        # Mark entry point
        self.ax.axhline(y=position['entry_price'], color='gray', linestyle='-', alpha=0.5)
        self.ax.annotate('Entry', (stop_history.index[0], position['entry_price']),
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        # Formatting
        self.ax.set_title(f'{ticker} Position Analysis')
        self.ax.set_ylabel('Price')
        self.ax.legend()
        self.ax.grid(True)
        
        # Add regime information
        current_regime = stop_system.detect_market_regime()
        self.ax.annotate(f"Regime: {current_regime.upper()}", 
                        xy=(0.02, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8))
        
        self.draw()


class TradingDashboard(QMainWindow):
    """Interactive trading dashboard with terminal log and runner-ups tab"""
    def __init__(self):
        super().__init__()
        self.tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "DIS"]
        self.trading_system = TradingSystem(self.tickers)
        self.setWindowTitle("Real-Time Trading Dashboard")
        self.setGeometry(100, 100, 1600, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        header_layout = QHBoxLayout()
        self.timestamp_label = QLabel("Timestamp: Initializing...")
        self.market_status_label = QLabel("Market Status: Checking...")
        header_layout.addWidget(self.timestamp_label)
        header_layout.addStretch()
        header_layout.addWidget(self.market_status_label)
        main_layout.addLayout(header_layout)
        
        # Account summary
        summary_layout = QHBoxLayout()
        self.capital_label = QLabel("Capital: $100,000.00")
        self.positions_label = QLabel("Active Positions: 0")
        self.trades_label = QLabel("Total Trades: 0")
        self.profit_label = QLabel("Total Profit: $0.00")
        self.win_rate_label = QLabel("Win Rate: 0.0%")
        
        summary_layout.addWidget(self.capital_label)
        summary_layout.addWidget(self.positions_label)
        summary_layout.addWidget(self.trades_label)
        summary_layout.addWidget(self.profit_label)
        summary_layout.addWidget(self.win_rate_label)
        main_layout.addLayout(summary_layout)
        
        # Control panel
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("Start System")
        self.stop_button = QPushButton("Stop System")
        
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addStretch()
        main_layout.addLayout(control_layout)
        
        # Tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Positions tab
        positions_tab = QWidget()
        positions_layout = QVBoxLayout(positions_tab)
        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(11)  # Added original score column
        self.positions_table.setHorizontalHeaderLabels([
            "Ticker", "Shares", "Entry", "Current", "Gain%", 
            "Trail Stop", "Hard Stop", "Profit Tgt", "Risk%", "Regime", "Score"
        ])
        self.positions_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        positions_layout.addWidget(self.positions_table)
        self.tabs.addTab(positions_tab, "Active Positions")
        
        # Trades tab
        trades_tab = QWidget()
        trades_layout = QVBoxLayout(trades_tab)
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(7)
        self.trades_table.setHorizontalHeaderLabels([
            "Ticker", "Entry", "Exit", "Profit", "Gain%", "Duration", "Reason"
        ])
        self.trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        trades_layout.addWidget(self.trades_table)
        self.tabs.addTab(trades_tab, "Recent Trades")
        
        # Opportunities tab
        opportunities_tab = QWidget()
        opportunities_layout = QVBoxLayout(opportunities_tab)
        self.opportunities_table = QTableWidget()
        self.opportunities_table.setColumnCount(8)
        self.opportunities_table.setHorizontalHeaderLabels([
            "Ticker", "Score", "Price", "ADX", "ATR", "RSI", "Volume", "Status"
        ])
        self.opportunities_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        opportunities_layout.addWidget(self.opportunities_table)
        self.tabs.addTab(opportunities_tab, "Trade Opportunities")
        
        # Runner-Ups tab
        runner_ups_tab = QWidget()
        runner_ups_layout = QVBoxLayout(runner_ups_tab)
        self.runner_ups_table = QTableWidget()
        self.runner_ups_table.setColumnCount(7)
        self.runner_ups_table.setHorizontalHeaderLabels([
            "Ticker", "Score", "Price", "ADX", "ATR", "RSI", "Volume"
        ])
        self.runner_ups_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        runner_ups_layout.addWidget(self.runner_ups_table)
        self.tabs.addTab(runner_ups_tab, "Runner-Ups")
        
        # Plots tab
        plots_tab = QWidget()
        plots_layout = QVBoxLayout(plots_tab)
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        plots_layout.addWidget(self.plot_container)
        self.tabs.addTab(plots_tab, "Position Analysis")
        
        # Terminal log tab
        terminal_tab = QWidget()
        terminal_layout = QVBoxLayout(terminal_tab)
        self.terminal_output = QTextEdit()
        self.terminal_output.setReadOnly(True)
        self.terminal_output.setFont(QFont("Courier", 10))
        self.terminal_output.setStyleSheet("background-color: black; color: #00FF00;")
        terminal_layout.addWidget(self.terminal_output)
        self.tabs.addTab(terminal_tab, "Terminal Log")
        
        # Connect signals
        self.start_button.clicked.connect(self.start_system)
        self.stop_button.clicked.connect(self.stop_system)
        self.trading_system.update_signal.connect(self.update_ui)
        self.trading_system.log_signal.connect(self.log_message)
        
        # Initial state
        self.stop_button.setEnabled(False)
        
        # Initialize log
        self.log_message("Trading Dashboard Initialized")
        self.log_message(f"Tracking Tickers: {', '.join(self.tickers)}")
        self.log_message("Press 'Start System' to begin trading")
        
        # Set initial UI state
        self.update_ui({
            'timestamp': datetime.now(tz.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
            'market_open': False,
            'capital': self.trading_system.capital,
            'positions_count': 0,
            'trade_count': 0,
            'total_profit': 0,
            'win_rate': 0,
            'active_positions': [],
            'recent_trades': [],
            'top_opportunities': [],
            'runner_ups': []
        })
        
    def log_message(self, message):
        """Add a message to the terminal log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.terminal_output.append(f"[{timestamp}] {message}")
        
        # Auto-scroll to bottom
        scrollbar = self.terminal_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def start_system(self):
        """Start the trading system"""
        if not self.trading_system.isRunning():
            if self.trading_system.start_system():
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(True)
                self.log_message("Trading system started")
                return
        self.log_message("System already running")
        
    def stop_system(self):
        """Stop the trading system"""
        if self.trading_system.isRunning():
            if self.trading_system.stop_system():
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                self.log_message("Trading system stopped")
                return
        self.log_message("System not running")
    
    def update_ui(self, state):
        """Update all UI elements with current state"""
        try:
            # Update header
            self.timestamp_label.setText(f"Timestamp: {state['timestamp']}")
            market_status = "OPEN" if state['market_open'] else "CLOSED"
            self.market_status_label.setText(f"Market Status: {market_status}")
            
            # Update account summary
            self.capital_label.setText(f"Capital: ${state['capital']:,.2f}")
            self.positions_label.setText(f"Active Positions: {state['positions_count']}")
            self.trades_label.setText(f"Total Trades: {state['trade_count']}")
            self.profit_label.setText(f"Total Profit: ${state['total_profit']:,.2f}")
            self.win_rate_label.setText(f"Win Rate: {state['win_rate']:.1f}%")
            
            # Update positions table
            self.positions_table.setRowCount(len(state['active_positions']))
            for row, pos in enumerate(state['active_positions']):
                self.positions_table.setItem(row, 0, QTableWidgetItem(pos['ticker']))
                self.positions_table.setItem(row, 1, QTableWidgetItem(str(pos['shares'])))
                self.positions_table.setItem(row, 2, QTableWidgetItem(f"{pos['entry_price']:.2f}"))
                self.positions_table.setItem(row, 3, QTableWidgetItem(f"{pos['current_price']:.2f}"))
                
                gain_item = QTableWidgetItem(f"{pos['gain']:.2f}%")
                gain_item.setForeground(QBrush(QColor('green') if pos['gain'] > 0 else QColor('red')))
                self.positions_table.setItem(row, 4, gain_item)
                
                self.positions_table.setItem(row, 5, QTableWidgetItem(f"{pos['trailing_stop']:.2f}"))
                self.positions_table.setItem(row, 6, QTableWidgetItem(f"{pos['hard_stop']:.2f}"))
                self.positions_table.setItem(row, 7, QTableWidgetItem(f"{pos['profit_target']:.2f}"))
                self.positions_table.setItem(row, 8, QTableWidgetItem(f"{pos['risk']:.2f}%"))
                self.positions_table.setItem(row, 9, QTableWidgetItem(pos['regime']))
                self.positions_table.setItem(row, 10, QTableWidgetItem(f"{pos.get('original_score', 0):.1f}"))
            
            # Update trades table
            self.trades_table.setRowCount(len(state['recent_trades']))
            for row, trade in enumerate(state['recent_trades']):
                self.trades_table.setItem(row, 0, QTableWidgetItem(trade['ticker']))
                self.trades_table.setItem(row, 1, QTableWidgetItem(f"{trade['entry']:.2f}"))
                self.trades_table.setItem(row, 2, QTableWidgetItem(f"{trade['exit']:.2f}" if trade['exit'] else ""))
                
                profit = trade['profit'] or 0
                profit_item = QTableWidgetItem(f"{profit:+.2f}")
                profit_item.setForeground(QBrush(QColor('green') if profit > 0 else QColor('red')))
                self.trades_table.setItem(row, 3, profit_item)
                
                gain = trade['percent_gain'] or 0
                gain_item = QTableWidgetItem(f"{gain:+.2f}%")
                gain_item.setForeground(QBrush(QColor('green') if gain > 0 else QColor('red')))
                self.trades_table.setItem(row, 4, gain_item)
                
                self.trades_table.setItem(row, 5, QTableWidgetItem(f"{trade['duration'] or 0:.1f}m"))
                self.trades_table.setItem(row, 6, QTableWidgetItem(trade['exit_reason'] or ""))
            
            # Update opportunities table
            self.opportunities_table.setRowCount(len(state['top_opportunities']))
            for row, opp in enumerate(state['top_opportunities']):
                self.opportunities_table.setItem(row, 0, QTableWidgetItem(opp['ticker']))
                self.opportunities_table.setItem(row, 1, QTableWidgetItem(f"{opp['score']:.1f}"))
                self.opportunities_table.setItem(row, 2, QTableWidgetItem(f"{opp['price']:.2f}"))
                self.opportunities_table.setItem(row, 3, QTableWidgetItem(f"{opp['adx']:.1f}"))
                self.opportunities_table.setItem(row, 4, QTableWidgetItem(f"{opp['atr']:.2f}"))
                self.opportunities_table.setItem(row, 5, QTableWidgetItem(f"{opp['rsi']:.1f}"))
                self.opportunities_table.setItem(row, 6, QTableWidgetItem(f"{opp['volume']:,.0f}"))
                
                status_item = QTableWidgetItem(opp['status'])
                status_item.setForeground(QBrush(QColor('green') if opp['status'] == "ENTERED" else QColor('black')))
                self.opportunities_table.setItem(row, 7, status_item)
            
            # Update runner-ups table
            self.runner_ups_table.setRowCount(len(state['runner_ups']))
            for row, opp in enumerate(state['runner_ups']):
                self.runner_ups_table.setItem(row, 0, QTableWidgetItem(opp['ticker']))
                self.runner_ups_table.setItem(row, 1, QTableWidgetItem(f"{opp['score']:.1f}"))
                self.runner_ups_table.setItem(row, 2, QTableWidgetItem(f"{opp['price']:.2f}"))
                self.runner_ups_table.setItem(row, 3, QTableWidgetItem(f"{opp['adx']:.1f}"))
                self.runner_ups_table.setItem(row, 4, QTableWidgetItem(f"{opp['atr']:.2f}"))
                self.runner_ups_table.setItem(row, 5, QTableWidgetItem(f"{opp['rsi']:.1f}"))
                self.runner_ups_table.setItem(row, 6, QTableWidgetItem(f"{opp['volume']:,.0f}"))
            
            # Update position plots
            self.update_plots()
            
        except Exception as e:
            self.log_message(f"UI update error: {str(e)}")
    
    def update_plots(self):
        """Update position analysis plots"""
        # Clear existing plots
        for i in reversed(range(self.plot_layout.count())):
            widget = self.plot_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        # Create new plots only if system is running
        if self.trading_system.running:
            for ticker, position in self.trading_system.positions.items():
                if self.trading_system.data_handler:
                    historical_data = self.trading_system.data_handler.get_historical(ticker, 50)
                    if historical_data is None or historical_data.empty:
                        continue
                        
                    plot = PositionPlot(self.plot_container, width=10, height=4, dpi=100)
                    plot.plot_position(ticker, position, historical_data)
                    self.plot_layout.addWidget(plot)
        
        # Add placeholder if no positions
        if not self.trading_system.positions or not self.trading_system.running:
            label = QLabel("No active positions to display")
            label.setAlignment(Qt.AlignCenter)
            label.setFont(QFont("Arial", 16))
            self.plot_layout.addWidget(label)
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop the trading system
        if self.trading_system.isRunning():
            self.trading_system.stop_system()
            self.trading_system.wait(3000)  # Wait up to 3 seconds
        
        # Close the application
        event.accept()


# Run the application
if __name__ == "__main__":
    # Create application
    app = QApplication(sys.argv)
    
    # Create and show dashboard
    dashboard = TradingDashboard()
    dashboard.show()
    
    # Start application
    sys.exit(app.exec_())