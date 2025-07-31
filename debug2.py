import requests
import pandas as pd
import numpy as np
import time
import math
import os
import json
import warnings
import pickle
import discord
from discord import Webhook
import aiohttp
import asyncio
from datetime import datetime, timedelta
from threading import Thread
from queue import Queue
from hmmlearn import hmm
from websocket import create_connection, WebSocketConnectionClosedException
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pathlib import Path
from config import POLYGON_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY, DISCORD_WEBHOOK_URL
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

# ======================
# Combined Configuration
# ======================
EXCHANGES = ["XNYS", "XNAS", "XASE"]  # NYSE, NASDAQ, AMEX
MAX_TICKERS_PER_EXCHANGE = 200
RATE_LIMIT = 0.00001  # seconds between requests
MIN_DAYS_DATA = 200  # Minimum days of data required for analysis
N_STATES = 3  # Bull/Neutral/Bear regimes
SECTOR_SAMPLE_SIZE = 50  # Stocks per sector for composite
TRANSITION_WINDOW = 30  # Days to analyze around regime transitions
ALLOCATION_PER_TICKER = 100  # $10,000 per position
MAX_TICKERS_TO_SCAN = None  # Limit for weekly scanner
STATE_FILE = 'trading_system_state.pkl'  # File to save system state
TRANSACTION_LOG_FILE = 'trading_transactions.log'
AUTO_SAVE_INTERVAL = 300  # 5 minutes in seconds
N_STATES = 3  # Bull/Neutral/Bear regimes

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
# Discord Notifier
# =====================
class DiscordNotifier:
    def __init__(self):
        self.webhook_url = DISCORD_WEBHOOK_URL
        
    async def send_embed(self, title, description, color=0x00ff00, fields=None):
        try:
            embed = discord.Embed(
                title=title,
                description=description,
                color=color,
                timestamp=datetime.now()
            )
            
            if fields:
                for name, value, inline in fields:
                    embed.add_field(name=name, value=value, inline=inline)
            
            async with aiohttp.ClientSession() as session:
                webhook = Webhook.from_url(self.webhook_url, session=session)
                await webhook.send(embed=embed, username="Trading System Bot")
        except Exception as e:
            print(f"Discord notification failed: {str(e)}")
    
    async def send_order_notification(self, order_type, ticker, qty, price, stop_loss=None, take_profit=None, hard_stop=None):
        """Send notification about an order"""
        color = 0x00ff00 if order_type.lower() == "buy" else 0xff0000
        fields = [
            ("Ticker", ticker, True),
            ("Quantity", str(qty), True),
            ("Price", f"${price:.2f}", True)
        ]
        
        if stop_loss:
            fields.append(("Stop Loss", f"${stop_loss:.2f}", True))
        if take_profit:
            fields.append(("Take Profit", f"${take_profit:.2f}", True))
        if hard_stop:
            fields.append(("Hard Stop", f"${hard_stop:.2f}", True))
            
        await self.send_embed(
            title=f"{order_type.upper()} Order Executed",
            description=f"New {order_type} order placed",
            color=color,
            fields=fields
        )
    
    async def send_position_update(self, ticker, entry_price, current_price, stop_loss, pnl=None):
        """Send position update notification"""
        fields = [
            ("Ticker", ticker, True),
            ("Entry Price", f"${entry_price:.2f}", True),
            ("Current Price", f"${current_price:.2f}", True),
            ("Stop Loss", f"${stop_loss:.2f}", True)
        ]
        
        if pnl is not None:
            pnl_percent = ((current_price - entry_price) / entry_price) * 100
            fields.append(("P&L", f"${pnl:.2f} ({pnl_percent:.2f}%)", True))
            
        await self.send_embed(
            title=f"Position Update: {ticker}",
            description="Position details update",
            color=0xffff00,
            fields=fields
        )
    
    async def send_stop_loss_update(self, ticker, old_stop, new_stop, reason=None):
        """Send stop loss adjustment notification"""
        description = f"Stop loss adjusted for {ticker}"
        if reason:
            description += f"\n**Reason:** {reason}"
            
        await self.send_embed(
            title="Stop Loss Updated",
            description=description,
            color=0xffa500,
            fields=[
                ("Ticker", ticker, True),
                ("Old Stop", f"${old_stop:.2f}", True),
                ("New Stop", f"${new_stop:.2f}", True)
            ]
        )
    
    async def send_hard_stop_triggered(self, ticker, entry_price, exit_price):
        """Notification when hard stop is triggered"""
        loss = entry_price - exit_price
        loss_percent = (loss / entry_price) * 100
        
        await self.send_embed(
            title="HARD STOP TRIGGERED",
            description=f"Position exited due to hard stop breach",
            color=0xff0000,
            fields=[
                ("Ticker", ticker, True),
                ("Entry Price", f"${entry_price:.2f}", True),
                ("Exit Price", f"${exit_price:.2f}", True),
                ("Loss", f"${loss:.2f} ({loss_percent:.2f}%)", True)
            ]
        )
    
    async def send_system_alert(self, message, is_error=False):
        """Send alert with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await self.send_embed(
                    title="SYSTEM ALERT" if is_error else "System Notification",
                    description=message,
                    color=0xff0000 if is_error else 0x0000ff
                )
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Final attempt failed to send Discord notification: {str(e)}")
                else:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def send_scan_results(self, results_df, total_scanned, new_tickers):
        """Send formatted scan results to Discord"""
        if results_df.empty:
            message = f"**Weekly Scan Completed**\nScanned {total_scanned} tickers\nNo qualified results found"
            await self.send_system_alert(message)
            return
            
        message = (
            f"**Weekly Scan Results:**\n"
            f"Scanned {total_scanned} tickers\n"
            f"Found {len(results_df)} qualified stocks\n"
        )
        
        if new_tickers:
            if len(new_tickers) <= 5:
                new_list = ", ".join(new_tickers)
                message += f"**New this week:** {new_list}\n\n"
            else:
                message += f"**New this week:** {len(new_tickers)} tickers\n\n"
        
        message += "**Top Candidates:**\n"
        for _, row in results_df.iterrows():
            message += (
                f"\n**{row['Rank']}. {row['Ticker']}** "
                f"(Score: {row['Score']}, Price: ${row['Price']:.2f})\n"
                f"ADX: {row['ADX']:.1f}, ATR: {row['ATR']:.2f}, "
                f"Stop: ${row['Initial_Stop']:.2f}\n"
            )
        
        await self.send_system_alert(message)
    
    async def send_position_notification(self, ticker, qty, entry_price, current_price, 
                                      stop_loss, take_profit, hard_stop, atr, adx):
        """Send detailed notification about a position"""
        risk_per_share = entry_price - stop_loss
        risk_percent = (risk_per_share / entry_price) * 100
        reward_per_share = take_profit - entry_price
        reward_percent = (reward_per_share / entry_price) * 100
        risk_reward_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0
        
        fields = [
            ("Ticker", ticker, True),
            ("Quantity", str(qty), True),
            ("Entry Price", f"${entry_price:.2f}", True),
            ("Current Price", f"${current_price:.2f}", True),
            ("Stop Loss", f"${stop_loss:.2f} ({risk_percent:.1f}%)", True),
            ("Take Profit", f"${take_profit:.2f} ({reward_percent:.1f}%)", True),
            ("Hard Stop", f"${hard_stop:.2f}", True),
            ("Risk/Reward", f"{risk_reward_ratio:.2f}:1", True),
            ("ATR", f"${atr:.2f}", True),
            ("ADX", f"{adx:.1f}", True)
        ]
        
        await self.send_embed(
            title=f"POSITION DETAILS: {ticker}",
            description="Complete position information",
            color=0x00ff00,
            fields=fields
        )
    
    async def send_active_positions_summary(self, positions):
        """Send summary of all active positions"""
        if not positions:
            await self.send_system_alert("No active positions currently")
            return
            
        message = "**Active Positions Summary**\n\n"
        total_pnl = 0
        total_invested = 0
        
        for ticker, position in positions.items():
            latest = get_latest_bar(ticker)
            current_price = latest['close'] if latest else position['entry_price']
            pnl = (current_price - position['entry_price']) * position['qty']
            pnl_percent = ((current_price - position['entry_price']) / position['entry_price']) * 100
            total_pnl += pnl
            total_invested += position['entry_price'] * position['qty']
            
            message += (
                f"**{ticker}** - "
                f"Qty: {position['qty']}, "
                f"Entry: ${position['entry_price']:.2f}, "
                f"Current: ${current_price:.2f}, "
                f"P&L: ${pnl:.2f} ({pnl_percent:.2f}%)\n"
                f"Stop: ${position['stop_loss'].current_stop:.2f}, "
                f"Target: ${position['stop_loss'].profit_target.current_target:.2f}\n\n"
            )
        
        total_return = (total_pnl / total_invested) * 100 if total_invested > 0 else 0
        message += (
            f"**Total Positions:** {len(positions)}\n"
            f"**Total P&L:** ${total_pnl:.2f} ({total_return:.2f}%)"
        )
        
        await self.send_system_alert(message)

# =====================
# Transaction Logger
# =====================
class TransactionLogger:
    def __init__(self, log_file=TRANSACTION_LOG_FILE):
        self.log_file = log_file
        Path(log_file).touch(exist_ok=True)
        
    def log(self, action, data):
        """Logs a transaction with timestamp"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "data": data
        }
        with open(self.log_file, "a") as f:
            json.dump(entry, f)
            f.write("\n")
            
    def replay_log(self, trading_system):
        """Replays transactions to rebuild state"""
        if not os.path.exists(self.log_file):
            return
            
        with open(self.log_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    self._process_entry(entry, trading_system)
                except json.JSONDecodeError:
                    continue
                    
    def _process_entry(self, entry, trading_system):
        """Process a single log entry"""
        action = entry["action"]
        data = entry["data"]
        
        if action == "place_order":
            ticker = data["ticker"]
            trading_system.active_positions[ticker] = {
                'qty': data['qty'],
                'entry_price': data['price'],
                'entry_time': entry["timestamp"],
                'stop_loss': SmartStopLoss(
                    entry_price=data['price'],
                    atr=data['atr'],
                    adx=data['adx']
                )
            }
            
        elif action == "update_stop":
            ticker = data["ticker"]
            if ticker in trading_system.active_positions:
                stop_loss = trading_system.active_positions[ticker]['stop_loss']
                stop_loss.current_stop = data['new_stop']
                stop_loss.highest_high = data['highest_high']
                stop_loss.activated = data['activated']
                if 'hard_stop' in data:
                    stop_loss.hard_stop = data['hard_stop']
                
        elif action == "close_position":
            ticker = data["ticker"]
            trading_system.active_positions.pop(ticker, None)

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
        for symbol in tqdm(tickers, desc="Mapping tickers to sectors"):  # Single progress bar here
            url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
            params = {"apiKey": POLYGON_API_KEY}
            time.sleep(RATE_LIMIT)
            
            try:
                response = requests.get(url, params=params, timeout=10)
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
        
        # Single progress bar here
        for sector, tickers in tqdm(self.sector_mappings.items(), desc="Building composites"):
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

class SmartProfitTarget:
    def __init__(self, entry_price, initial_target, atr, adx):
        self.entry = entry_price
        self.base_target = initial_target
        self.atr = atr
        self.adx = adx
        self.current_target = initial_target
        self.strength_factor = 1.0
        self.breached_levels = 0
        self.last_high = entry_price
        
    def update(self, current_bar):
        current_high = current_bar['high']
        current_close = current_bar['close']
        
        # Calculate trend strength
        adx_strength = min(2.0, self.adx / 30)  # 1.0 = ADX 30, 2.0 = ADX 60
        volume_ratio = current_bar.get('volume', 1e6) / current_bar.get('avg_volume', 1e6)
        volatility_factor = max(0.8, min(1.5, self.atr / (current_close * 0.01)))
        
        # Dynamic strength adjustment
        self.strength_factor = 1.0 + (adx_strength * min(2.0, volume_ratio) * volatility_factor)
        
        # Check if we've breached a target level
        if current_high > self.current_target:
            self.breached_levels += 1
            self.last_high = current_high
            
        # Calculate new target
        if self.breached_levels > 0:
            # Extend target in strong trends
            extension_factor = 1 + (0.25 * self.breached_levels)
            new_base = self.entry + (extension_factor * (self.base_target - self.entry))
            self.current_target = new_base * self.strength_factor
        else:
            # Maintain base target
            self.current_target = self.base_target * self.strength_factor
            
        return self.current_target

    def should_take_profit(self, current_bar):
        current_close = current_bar['close']
        rsi = current_bar.get('rsi', 50)
        
        # Basic profit taking condition
        if current_close >= self.current_target:
            return True
            
        # Hold conditions for strong trends
        if self.strength_factor > 1.5:
            # Only take profit if RSI > 70 and closing near highs
            if rsi < 70 or current_close < (current_bar['high'] * 0.99):
                return False
                
        return current_close >= self.current_target

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
        
        # New hard stop loss parameters
        self.hard_stop = entry_price - (base_multiplier * 1.8 * atr)  # Wider buffer
        self.hard_stop_triggered = False
        self.trend_strength = 1.0  # Measures confidence in trend continuation
        
        # Profit target system
        initial_target = entry_price + 2 * (entry_price - self.current_stop)
        self.profit_target = SmartProfitTarget(
            entry_price=entry_price,
            initial_target=initial_target,
            atr=atr,
            adx=adx
        )

    def update(self, current_bar):
        current_high = current_bar['high']
        current_low = current_bar['low']
        current_close = current_bar['close']
        current_adx = current_bar.get('adx', self.base_adx)
        
        # Update trend strength (combines ADX and volume)
        adx_strength = min(1.0, current_adx / 50)
        volume_ratio = current_bar.get('volume', 1e6) / current_bar.get('avg_volume', 1e6)
        self.trend_strength = max(0.5, min(2.0, adx_strength * min(1.5, volume_ratio)))
        
        # Update highest high
        if current_high > self.highest_high:
            self.highest_high = current_high
            self.consecutive_confirmations = 0

        # Calculate growth potential
        self.growth_potential = max(0.5, min(2.0, adx_strength * min(1.5, volume_ratio)))
        
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
        
        # Check if we need to trigger hard stop
        if current_low <= self.hard_stop:
            self.hard_stop_triggered = True
            
        # Update profit target
        self.profit_target.update(current_bar)
                
        return self.current_stop

    def sync_with_market(self, ticker):
        """Sync stop with latest market data after restart"""
        latest = get_latest_bar(ticker)
        if not latest:
            return
            
        # Update highest high if market moved
        if latest['high'] > self.highest_high:
            self.highest_high = latest['high']
            
        # Recalculate stop based on current market
        self.update(latest)

    def should_hold(self, current_bar):
        current_low = current_bar['low']
        current_close = current_bar['close']
        rsi = current_bar.get('rsi', 50)
        volatility_ratio = current_bar.get('volatility_ratio', 1.0)
        
        # 0. Never hold if hard stop triggered
        if self.hard_stop_triggered:
            return False
            
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
        # Always exit if hard stop triggered
        if self.hard_stop_triggered:
            return True
            
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
            'consecutive_confirmations': self.consecutive_confirmations,
            'hard_stop': self.hard_stop,
            'hard_stop_triggered': self.hard_stop_triggered,
            'profit_target': self.profit_target.current_target
        }

    def get_bracket_orders(self, entry_price, qty):
        """Generate bracket order details with properly rounded prices"""
        stop_price = self.current_stop
        
        # Normalize prices according to exchange rules
        def normalize_price(price):
            """Round price to proper increment based on price level"""
            if price < 1.00:
                return round(price, 4)  # $0.0001 increments for stocks < $1
            elif price < 10.00:
                return round(price, 3)  # $0.001 increments for stocks $1-$10
            else:
                return round(price, 2)  # $0.01 increments for stocks > $10
        
        normalized_entry = normalize_price(entry_price)
        normalized_stop = normalize_price(stop_price)
        normalized_hard_stop = normalize_price(self.hard_stop)
        normalized_profit_target = normalize_price(self.profit_target.current_target)
        
        # Ensure take profit is above current price
        if normalized_profit_target <= normalized_entry:
            normalized_profit_target = normalize_price(normalized_entry * 1.01)  # Minimum 1% profit
        
        # Ensure stop is below current price
        if normalized_stop >= normalized_entry:
            normalized_stop = normalize_price(normalized_entry * 0.99)  # Minimum 1% stop
            
        if normalized_hard_stop >= normalized_entry:
            normalized_hard_stop = normalize_price(normalized_entry * 0.98)  # Minimum 2% stop
        
        return {
            "stop_loss": {
                "stop_price": normalized_stop,
                "limit_price": normalize_price(normalized_stop * 0.98)  # Add limit price for stop-limit
            },
            "take_profit": {
                "limit_price": normalized_profit_target
            },
            "hard_stop": {
                "stop_price": normalized_hard_stop
            }
        }

    def get_serializable_state(self):
        """Return a dictionary of the current state that can be serialized"""
        return {
            'entry': self.entry,
            'initial_atr': self.initial_atr,
            'base_adx': self.base_adx,
            'activation_percent': self.activation_percent,
            'base_multiplier': self.base_multiplier,
            'activated': self.activated,
            'highest_high': self.highest_high,
            'current_stop': self.current_stop,
            'growth_potential': self.growth_potential,
            'consecutive_confirmations': self.consecutive_confirmations,
            'last_direction': self.last_direction,
            'previous_close': self.previous_close,
            'hard_stop': self.hard_stop,
            'hard_stop_triggered': self.hard_stop_triggered,
            'trend_strength': self.trend_strength,
            'profit_target_state': {
                'entry_price': self.profit_target.entry,
                'initial_target': self.profit_target.base_target,
                'atr': self.profit_target.atr,
                'adx': self.profit_target.adx,
                'current_target': self.profit_target.current_target,
                'strength_factor': self.profit_target.strength_factor,
                'breached_levels': self.profit_target.breached_levels,
                'last_high': self.profit_target.last_high
            }
        }
    
    @classmethod
    def from_serialized_state(cls, state):
        """Recreate a SmartStopLoss instance from serialized state"""
        instance = cls(
            entry_price=state['entry'],
            atr=state['initial_atr'],
            adx=state['base_adx'],
            activation_percent=state['activation_percent'],
            base_multiplier=state['base_multiplier']
        )
        
        # Restore all state variables with fallbacks
        instance.activated = state.get('activated', False)
        instance.highest_high = state.get('highest_high', state['entry'])
        instance.current_stop = state.get('current_stop', 
            state['entry'] - (state['base_multiplier'] * state['initial_atr']))
        instance.growth_potential = state.get('growth_potential', 1.0)
        instance.consecutive_confirmations = state.get('consecutive_confirmations', 0)
        instance.last_direction = state.get('last_direction', "up")
        instance.previous_close = state.get('previous_close', state['entry'])
        
        # New attributes with defaults
        instance.hard_stop = state.get('hard_stop', 
            state['entry'] - (state['base_multiplier'] * 1.8 * state['initial_atr']))
        instance.hard_stop_triggered = state.get('hard_stop_triggered', False)
        instance.trend_strength = state.get('trend_strength', 1.0)
        
        # Profit target reconstruction
        if 'profit_target_state' in state:
            profit_state = state['profit_target_state']
            instance.profit_target = SmartProfitTarget(
                entry_price=profit_state['entry_price'],
                initial_target=profit_state['initial_target'],
                atr=profit_state['atr'],
                adx=profit_state['adx']
            )
            # Restore profit target state
            instance.profit_target.current_target = profit_state['current_target']
            instance.profit_target.strength_factor = profit_state.get('strength_factor', 1.0)
            instance.profit_target.breached_levels = profit_state.get('breached_levels', 0)
            instance.profit_target.last_high = profit_state.get('last_high', profit_state['entry_price'])
        else:
            # Reconstruct profit target from available data
            initial_target = state['entry'] + 2*(state['entry'] - instance.current_stop)
            instance.profit_target = SmartProfitTarget(
                entry_price=state['entry'],
                initial_target=initial_target,
                atr=state['initial_atr'],
                adx=state['base_adx']
            )
        
        return instance

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
            response = requests.get(url, params=params, timeout=10)
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
        """Scan all tickers with detailed progress logging"""
        results = []
        total_tickers = len(self.tickers)
        qualified_count = 0
        
        self.log_debug(f"Starting ticker scan (total: {total_tickers})")
        self.log_debug(f"Sample tickers: {self.tickers[:5]}...")
        self.log_debug(f"Date range: {self.start_date} to {self.end_date}")

        # Initialize progress bar with custom formatting
        progress_bar = tqdm(
            self.tickers,
            desc="📊 Scanning tickers",
            bar_format='{l_bar}{bar:50}{r_bar}',
            colour='green'
        )

        for ticker in progress_bar:
            try:
                # Update progress description
                progress_bar.set_postfix({
                    'current': ticker,
                    'qualified': qualified_count
                })
                
                # 1. Fetch data
                data = self.get_polygon_data(ticker)
                if data is None:
                    self.log_debug(f"{ticker}: No data returned", "DEBUG")
                    continue
                    
                if len(data) < MIN_DAYS_DATA:
                    self.log_debug(f"{ticker}: Insufficient data ({len(data)} days)", "DEBUG")
                    continue

                # 2. Calculate indicators
                indicators = self.calculate_indicators(data)
                if indicators is None:
                    self.log_debug(f"{ticker}: Invalid indicators", "DEBUG")
                    continue

                # 3. Apply filters
                above_sma50 = indicators['Close'] > indicators['SMA_50']
                above_sma200 = indicators['Close'] > indicators['SMA_200']
                strong_adx = indicators['ADX'] > 25
                
                if not (above_sma50 and above_sma200 and strong_adx):
                    self.log_debug(
                        f"{ticker}: Filtered out | "
                        f"SMA50: {above_sma50} | "
                        f"SMA200: {above_sma200} | "
                        f"ADX: {strong_adx}",
                        "DEBUG"
                    )
                    continue

                # 4. Calculate score
                score = self._calculate_score(indicators)
                stop_system = SmartStopLoss(
                    entry_price=indicators['Close'],
                    atr=indicators['ATR'],
                    adx=indicators['ADX']
                )

                # 5. Store results
                result = {
                    'Ticker': ticker,
                    'Score': round(score, 1),
                    'Price': round(indicators['Close'], 2),
                    'ADX': round(indicators['ADX'], 1),
                    'SMA50_Distance%': round(indicators['Distance_from_SMA50'], 1),
                    'SMA200_Distance%': round(indicators['Distance_from_SMA200'], 1),
                    'Volume': int(indicators['Volume']),
                    'ATR': round(indicators['ATR'], 2),
                    'ATR_Ratio': round((indicators['Close'] - indicators['SMA_50'])/indicators['ATR'], 1),
                    '10D_Change%': round(indicators['10D_Change'], 1),
                    'Initial_Stop': round(stop_system.current_stop, 2),
                    'Hard_Stop': round(stop_system.hard_stop, 2),
                    'Take_Profit': round(stop_system.profit_target.current_target, 2),
                    'Risk_per_Share': round(indicators['Close'] - stop_system.current_stop, 2),
                    'Risk_Percent': round(((indicators['Close'] - stop_system.current_stop)/indicators['Close'])*100, 2),
                    'ATR_Multiplier': self.base_multiplier,
                    'Activation_Percent': 5.0
                }
                
                qualified_count += 1
                self.log_debug(
                    f"✅ Qualified: {ticker} | "
                    f"Score: {score:.1f} | "
                    f"Price: ${indicators['Close']:.2f} | "
                    f"ADX: {indicators['ADX']:.1f} | "
                    f"Risk: {result['Risk_Percent']:.1f}%",
                    "SUCCESS"
                )
                
                results.append(result)

            except requests.exceptions.RequestException as e:
                self.log_debug(f"⚠️ Network error on {ticker}: {str(e)}", "WARNING")
                time.sleep(5)  # Backoff on rate limits
            except Exception as e:
                self.log_debug(f"❌ Error processing {ticker}: {str(e)}", "ERROR")

        progress_bar.close()
        
        if results:
            df_results = pd.DataFrame(results)
            df_results['Rank'] = df_results['Score'].rank(ascending=False, method='min').astype(int)
            final_results = df_results.sort_values('Score', ascending=False).reset_index(drop=True)
            
            self.log_debug(f"Scan complete. Qualified {len(final_results)}/{total_tickers} tickers")
            self.log_debug("Top candidates:\n" + final_results.head(5).to_string())
            return final_results
        
        self.log_debug("Scan completed with no qualified stocks", "WARNING")
        return pd.DataFrame()
    
    def _calculate_score(self, indicators):
        """Calculate composite score (0-100) with logging"""
        try:
            adx_component = min(40, (indicators['ADX'] / 50) * 40)
            sma50_component = min(30, max(0, indicators['Distance_from_SMA50']) * 0.3)
            volume_component = min(20, math.log10(max(1, indicators['Volume']/10000)))
            momentum_component = min(10, max(0, indicators['10D_Change']))
            
            score = min(100, adx_component + sma50_component + volume_component + momentum_component)
            
            self.log_debug(
                f"Score components | "
                f"ADX: {adx_component:.1f} | "
                f"SMA50: {sma50_component:.1f} | "
                f"Volume: {volume_component:.1f} | "
                f"Momentum: {momentum_component:.1f}",
                "DEBUG"
            )
            
            return score
        except Exception as e:
            self.log_debug(f"Score calculation failed: {str(e)}", "ERROR")
            return 0

class StateManager:
    def __init__(self, state_file=STATE_FILE):
        self.state_file = state_file
        self.last_save = None
        self.transaction_log = TransactionLogger()
        
    def save_state(self, trading_system):
        """Save the current state of the trading system"""
        serializable_positions = {}
        for ticker, data in trading_system.active_positions.items():
            serializable_positions[ticker] = {
                'qty': data['qty'],
                'entry_price': data['entry_price'],
                'stop_loss_state': data['stop_loss'].get_serializable_state(),
                'entry_time': data.get('entry_time', datetime.now().isoformat())
            }
        
        state = {
            'active_positions': serializable_positions,
            'last_scan_date': trading_system.last_scan_date,
            'last_regime': trading_system.sector_system.overall_analyzer.state_labels,
            'saved_at': datetime.now(),
            'current_top_ticker': trading_system.current_top_ticker,
            'top_ticker_score': trading_system.top_ticker_score,
            'runner_ups': trading_system.runner_ups,
            'monthly_trade_count': trading_system.monthly_trade_count,
            'last_month_checked': trading_system.last_month_checked,
            'executed_tickers': list(trading_system.executed_tickers)
        }
        
        try:
            with open(self.state_file, 'wb') as f:
                pickle.dump(state, f)
            self.last_save = datetime.now()
            print(f"State saved at {self.last_save}")
            
            # Rotate logs on successful save
            self._rotate_logs()
            return True
        except Exception as e:
            print(f"Error saving state: {str(e)}")
            return False
    
    def _rotate_logs(self):
        """Rotate logs to prevent them from growing too large"""
        if os.path.exists(TRANSACTION_LOG_FILE):
            log_size = os.path.getsize(TRANSACTION_LOG_FILE)
            if log_size > 1_000_000:  # 1MB
                backup_name = f"{TRANSACTION_LOG_FILE}.bak"
                os.replace(TRANSACTION_LOG_FILE, backup_name)
                
    def load_state(self, trading_system):
        """Load state without duplicate messages"""
        if not os.path.exists(self.state_file):
            return None
            
        try:
            with open(self.state_file, 'rb') as f:
                state = pickle.load(f)
                
            # Add this line to prevent duplicate reconciliation
            trading_system.active_positions = {}
                
            return state
        except Exception as e:
            print(f"Error loading state: {str(e)}")
            return None

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
            response = requests.get(url, params=params, timeout=10)
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
        with tqdm(desc=f"Fetching {symbol}", bar_format='{l_bar}{bar:20}{r_bar}', leave=False) as pbar:
            response = requests.get(url, params=params, timeout=15)
            pbar.update(1)
            
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
    except requests.exceptions.Timeout:
        print(f"Timeout fetching {symbol}, skipping")
        return None
    except Exception as e:
        print(f"Error fetching {symbol}: {str(e)}")
    return None

def get_market_cap(symbol):
    url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
    params = {"apiKey": POLYGON_API_KEY}
    time.sleep(RATE_LIMIT)
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json().get('results', {}).get('market_cap', 0)
    except:
        return 0
    return 0

def get_latest_bar(ticker, retries=3):
    for i in range(retries):
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev"
            params = {'adjusted': 'true', 'apiKey': POLYGON_API_KEY}
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                result = response.json().get('results', [])
                if result:
                    return {
                        'open': result[0]['o'],
                        'high': result[0]['h'],
                        'low': result[0]['l'],
                        'close': result[0]['c'],
                        'volume': result[0].get('v', 1e6)
                    }
            elif response.status_code == 429:
                wait = int(response.headers.get('Retry-After', 30))
                print(f"Rate limited. Waiting {wait} seconds")
                time.sleep(wait)
            time.sleep(1)
        except Exception as e:
            time.sleep(2 ** i)  # Exponential backoff
    return None

# ========================
# Integrated Trading System
# ========================
class TradingSystem:
    def __init__(self):
        self.sector_system = SectorRegimeSystem()
        self.trend_scanner = PolygonTrendScanner(max_tickers=MAX_TICKERS_TO_SCAN)
        self.active_positions = {}
        self.last_scan_date = None
        self.last_state_save = datetime.now()
        self.state_manager = StateManager()
        self.notifier = DiscordNotifier()
        self.transaction_log = TransactionLogger()
        self.debug_log = []
        self.debug_file = 'trading_debug.log'

        # Initialize HMM model
        self.model = hmm.GaussianHMM(
            n_components=N_STATES,
            covariance_type="diag",
            n_iter=2000,
            tol=1e-4,
            init_params='stmc',
            params='stmc',
            random_state=42
        )
        self.feature_scaler = StandardScaler()
        
        # Strategy attributes
        self.current_top_ticker = None
        self.top_ticker_score = 0
        self.runner_ups = []
        self.monthly_trade_count = 0
        self.last_month_checked = None
        self.executed_tickers = set()
        self.max_positions = 3
        self.check_interval = 3600
        
        # Initialize
        asyncio.create_task(self.initialize())

    def log_debug(self, message, level="INFO"):
        """Log debug messages with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.debug_log.append(log_entry)
        print(log_entry)  # Also print to console
        
        # Write to log file
        with open(self.debug_file, 'a') as f:
            f.write(log_entry + "\n")

    async def initialize(self):
        """Initialize trading system with validation checks and debug logging"""
        try:
            self.log_debug("Initializing trading system...")
            
            # 1. Verify API keys
            if not POLYGON_API_KEY:
                raise ValueError("Missing Polygon API key")
            if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
                raise ValueError("Missing Alpaca API credentials")
            self.log_debug("API credentials validated")

            # 2. Load tickers
            self.log_debug("Loading ticker list...")
            tickers = await asyncio.to_thread(get_all_tickers)
            
            if not tickers:
                raise ValueError("No tickers loaded from data source")
            
            self.trend_scanner.tickers = tickers
            self.log_debug(f"Loaded {len(tickers)} tickers (sample: {tickers[:3]}...)")

            # 3. Load previous state
            self.log_debug("Attempting to load previous state...")
            state = await asyncio.to_thread(self.state_manager.load_state, self)
            
            if state:
                self.log_debug(f"Loaded state from {state.get('saved_at')}")
                self.log_debug(f"Active positions: {len(state.get('active_positions', {}))}")
            else:
                self.log_debug("No previous state found")

            # 4. Initialize components
            self.sector_system = SectorRegimeSystem()
            self.log_debug("Sector system initialized")
            
            # 5. Initial market analysis
            self.log_debug("Running initial market analysis...")
            current_regime = await self.run_market_regime_analysis()
            self.log_debug(f"Initial market regime: {current_regime}")

            # 6. Initial scan if no candidates
            if not self.runner_ups:
                self.log_debug("No existing candidates found - running initial scan")
                await self.run_weekly_scan()

            self._initialized = True
            self.log_debug("Initialization complete")
            await self.notifier.send_system_alert("System initialized successfully")

        except Exception as e:
            error_msg = f"Initialization failed: {str(e)}"
            self.log_debug(error_msg, "CRITICAL")
            await self.notifier.send_system_alert(error_msg, is_error=True)
            raise

    async def load_previous_state(self):
        """Load and verify previous state"""
        state = self.state_manager.load_state(self)
        if state:
            print(f"\nLoaded state from {state['saved_at']}")  # \n for newline
            print("Reconciling with Alpaca...")
            for ticker, data in state.get('active_positions', {}).items():
                try:
                    stop_loss = SmartStopLoss.from_serialized_state(data['stop_loss_state'])
                    await self.verify_and_load_position(ticker, data, stop_loss)
                except Exception as e:
                    print(f"Error loading position {ticker}: {str(e)}")
            
            # Load other state variables
            self.load_state_variables(state)
            
            print("State loaded, reconciling with Alpaca...")
            await self.reconcile_all_positions()

    async def verify_and_load_position(self, ticker, data, stop_loss):
        """Verify and load a position with market data"""
        latest = get_latest_bar(ticker)
        if latest:
            stop_loss.sync_with_market(ticker)
            self.active_positions[ticker] = {
                'qty': data['qty'],
                'entry_price': data['entry_price'],
                'entry_time': data.get('entry_time', datetime.now().isoformat()),
                'stop_loss': stop_loss
            }
        else:
            print(f"Could not verify position {ticker} - market data unavailable")

    def load_state_variables(self, state):
        """Load non-position state variables"""
        self.last_scan_date = state.get('last_scan_date')
        if 'last_regime' in state:
            self.sector_system.overall_analyzer.state_labels = state['last_regime']
        
        self.current_top_ticker = state.get('current_top_ticker')
        self.top_ticker_score = state.get('top_ticker_score', 0)
        self.runner_ups = state.get('runner_ups', [])
        self.monthly_trade_count = state.get('monthly_trade_count', 0)
        self.last_month_checked = state.get('last_month_checked')
        self.executed_tickers = set(state.get('executed_tickers', []))

    async def reconcile_all_positions(self):
        """Reconcile all positions with Alpaca"""
        reconciliation_count = 0
        for ticker in list(self.active_positions.keys()):
            if await self.reconcile_ticker_state(ticker):
                reconciliation_count += 1
        
        if reconciliation_count > 0:
            await self.notifier.send_system_alert(
                f"Reconciled {reconciliation_count} positions with Alpaca"
            )
            self.save_current_state()

    async def check_ticker_status_thoroughly(self, ticker):
        """Comprehensive status check for a ticker"""
        status = {
            'ticker': ticker,
            'has_position': False,
            'position_details': None,
            'has_open_orders': False,
            'open_orders': [],
            'recent_filled_orders': [],
            'recent_canceled_orders': [],
            'all_clear': True,
            'notes': []
        }

        try:
            # Check positions
            await self.check_positions(ticker, status)
            
            # Check orders
            await self.check_orders(ticker, status)
            
            # Final determination
            if status['all_clear']:
                status['notes'].append("No active positions or orders found")

        except Exception as e:
            status['error'] = f"Comprehensive check failed: {str(e)}"
            status['all_clear'] = False

        return status

    async def check_positions(self, ticker, status):
        """Check position status"""
        try:
            positions = trading_client.get_all_positions()
            for position in positions:
                if position.symbol == ticker:
                    status.update({
                        'has_position': True,
                        'all_clear': False,
                        'position_details': self.extract_position_details(position)
                    })
        except Exception as e:
            status['notes'].append(f"Position check error: {str(e)}")

    def extract_position_details(self, position):
        """Extract position details from Alpaca position object"""
        return {
            'qty': float(position.qty),
            'avg_entry_price': float(position.avg_entry_price),
            'market_value': float(position.market_value),
            'current_price': float(position.current_price),
            'unrealized_pl': float(position.unrealized_pl),
            'side': position.side
        }

    async def check_orders(self, ticker, status):
        """Check order status"""
        try:
            # Open orders
            await self.check_open_orders(ticker, status)
            
            # Recent filled orders
            await self.check_recent_orders(ticker, status, 'filled')
            
            # Recent canceled orders
            await self.check_recent_orders(ticker, status, 'canceled')
        except Exception as e:
            status['notes'].append(f"Order check error: {str(e)}")

    async def check_open_orders(self, ticker, status):
        """Check for open orders"""
        try:
            orders = trading_client.get_orders(
                order_status='open',  # Changed from 'status'
                symbol=ticker,
                limit=50
            )
            if orders:
                status.update({
                    'has_open_orders': True,
                    'all_clear': False,
                    'open_orders': [self.extract_order_details(order) for order in orders]
                })
        except Exception as e:
            status['notes'].append(f"Order check error: {str(e)}")

    async def check_recent_orders(self, ticker, status, status_type):
        """Check recently filled or canceled orders"""
        orders = trading_client.get_orders(
            status=status_type,
            symbol=ticker,
            after=datetime.now() - timedelta(hours=24),
            limit=50
        )
        if orders:
            status[f'recent_{status_type}_orders'] = [
                self.extract_order_details(order) for order in orders
            ]

    def extract_order_details(self, order):
        """Extract order details from Alpaca order object"""
        return {
            'id': order.id,
            'qty': float(order.qty),
            'filled_qty': float(order.filled_qty) if hasattr(order, 'filled_qty') else None,
            'type': order.order_type,
            'side': order.side,
            'status': order.status,
            'created_at': order.created_at.isoformat(),
            'filled_at': order.filled_at.isoformat() if hasattr(order, 'filled_at') else None,
            'filled_avg_price': float(order.filled_avg_price) if hasattr(order, 'filled_avg_price') else None,
            'limit_price': float(order.limit_price) if hasattr(order, 'limit_price') else None,
            'stop_price': float(order.stop_price) if hasattr(order, 'stop_price') else None
        }

    async def reconcile_ticker_state(self, ticker):
        """Reconcile state for a single ticker"""
        saved_state = self.active_positions.get(ticker)
        actual_status = await self.check_ticker_status_thoroughly(ticker)
        reconciliation_needed = False

        # Case 1: Phantom position
        if saved_state and not actual_status['has_position']:
            reconciliation_needed = await self.handle_phantom_position(ticker, saved_state)

        # Case 2: Unexpected open orders
        if actual_status['has_open_orders']:
            reconciliation_needed = await self.handle_unexpected_orders(ticker, actual_status['open_orders']) or reconciliation_needed

        # Case 3: Missing positions from fills
        if saved_state and actual_status['recent_filled_orders']:
            reconciliation_needed = await self.handle_missing_positions(ticker, actual_status['recent_filled_orders']) or reconciliation_needed

        return reconciliation_needed

    async def handle_phantom_position(self, ticker, saved_state):
        """Handle position that exists in state but not in Alpaca"""
        print(f"Removing phantom position for {ticker}")
        await self.notifier.send_system_alert(
            f"Removing {ticker} from active positions (not found in Alpaca)"
        )
        
        del self.active_positions[ticker]
        
        self.transaction_log.log(
            action="reconcile_position",
            data={
                "ticker": ticker,
                "action": "removed",
                "reason": "position_not_found",
                "saved_qty": saved_state['qty'],
                "saved_entry_price": saved_state['entry_price']
            }
        )
        return True

    async def handle_unexpected_orders(self, ticker, open_orders):
        """Handle orders that exist in Alpaca but not in state"""
        reconciliation_needed = False
        for order in open_orders:
            if order['id'] not in [o.id for o in self.active_positions.get(ticker, {}).get('open_orders', [])]:
                try:
                    trading_client.cancel_order_by_id(order['id'])
                    print(f"Cancelled unexpected order {order['id']} for {ticker}")
                    
                    await self.notifier.send_system_alert(
                        f"Cancelled unexpected order for {ticker} (ID: {order['id']})"
                    )
                    
                    self.transaction_log.log(
                        action="reconcile_order",
                        data={
                            "ticker": ticker,
                            "order_id": order['id'],
                            "action": "cancelled",
                            "reason": "unexpected_order"
                        }
                    )
                    reconciliation_needed = True
                except Exception as e:
                    print(f"Failed to cancel order {order['id']}: {str(e)}")
        return reconciliation_needed

    async def handle_missing_positions(self, ticker, filled_orders):
        """Handle positions that should exist based on fills"""
        reconciliation_needed = False
        for order in filled_orders:
            if order['side'] == 'buy' and not self.active_positions.get(ticker):
                print(f"Adding missing position for {ticker} from fill")
                self.active_positions[ticker] = {
                    'qty': order['filled_qty'],
                    'entry_price': order['filled_avg_price'],
                    'entry_time': order['filled_at'],
                    'stop_loss': SmartStopLoss(
                        entry_price=order['filled_avg_price'],
                        atr=self.calculate_current_atr(ticker),
                        adx=self.calculate_current_adx(ticker)
                    )
                }
                
                await self.notifier.send_system_alert(
                    f"Added missing position for {ticker} from filled order"
                )
                
                self.transaction_log.log(
                    action="reconcile_position",
                    data={
                        "ticker": ticker,
                        "action": "added",
                        "reason": "filled_order_missing_position",
                        "qty": order['filled_qty'],
                        "entry_price": order['filled_avg_price']
                    }
                )
                reconciliation_needed = True
        return reconciliation_needed

    def calculate_current_atr(self, ticker):
        """Calculate current ATR for a ticker"""
        # Implementation depends on your data source
        return 2.0  # Example value

    def calculate_current_adx(self, ticker):
        """Calculate current ADX for a ticker"""
        # Implementation depends on your data source
        return 25.0  # Example value

    async def execute_trades_safely(self):
        """Execute trades with state verification"""
        await self.reconcile_all_positions()
        
        if len(self.active_positions) >= self.max_positions:
            await self.notifier.send_system_alert(
                "Max positions reached. No new trades."
            )
            return
        
        next_ticker = await self.select_next_ticker()
        if next_ticker:
            await self.place_trade(next_ticker)

    async def select_next_ticker(self):
        if not self.runner_ups:
            self.log_debug("No runner-ups available for selection")
            return None
            
        self.log_debug(f"Evaluating {len(self.runner_ups)} candidates for trade")
        
        for i, candidate in enumerate(self.runner_ups):
            ticker = candidate['Ticker']
            self.log_debug(f"Checking candidate #{i+1}: {ticker} (Score: {candidate['Score']})")
            
            status = await self.check_ticker_status_thoroughly(ticker)
            self.log_debug(f"Status for {ticker}: {'Clear' if status['all_clear'] else 'Blocked'}")
            
            if status['all_clear'] and ticker not in self.executed_tickers:
                self.log_debug(f"Selected {ticker} for trading")
                return candidate
                
        self.log_debug("No tradeable candidates found in runner-ups")
        return None

    async def run_weekly_scan(self):
        """Run weekly scanner with comprehensive logging and error handling"""
        scan_start = datetime.now()
        self.log_debug(f"Starting weekly scan at {scan_start}")
        
        try:
            # 1. Validate prerequisites
            if not hasattr(self, 'trend_scanner') or not self.trend_scanner.tickers:
                self.log_debug("Scanner not properly initialized - reloading tickers", "WARNING")
                self.trend_scanner.tickers = await asyncio.to_thread(get_all_tickers)
                
                if not self.trend_scanner.tickers:
                    raise ValueError("Failed to load tickers for scanning")

            # 2. Run scan
            self.log_debug(f"Scanning {len(self.trend_scanner.tickers)} tickers...")
            results = await asyncio.to_thread(self.trend_scanner.scan_tickers)
            
            # 3. Process results
            if results.empty:
                self.log_debug("Scanner returned empty results", "WARNING")
                await self.notifier.send_system_alert("Weekly scan completed with no qualified stocks")
                return results

            self.log_debug(f"Scan found {len(results)} qualified stocks")
            self.log_debug("Top 5 candidates:\n" + results.head(5).to_string(), "DEBUG")

            # 4. Update system state
            self.process_scan_results(results)
            self.last_scan_date = datetime.now()
            
            # 5. Notify
            new_tickers = [t for t in results['Ticker'] if t not in self.executed_tickers]
            await self.notifier.send_scan_results(results, len(self.trend_scanner.tickers), new_tickers)
            
            scan_duration = (datetime.now() - scan_start).total_seconds()
            self.log_debug(f"Scan completed in {scan_duration:.2f} seconds")
            
            return results

        except requests.exceptions.RequestException as e:
            error_msg = f"Network error during scan: {str(e)}"
            self.log_debug(error_msg, "ERROR")
            await self.notifier.send_system_alert(error_msg, is_error=True)
            return pd.DataFrame()
            
        except Exception as e:
            error_msg = f"Unexpected scan error: {str(e)}"
            self.log_debug(error_msg, "CRITICAL")
            await self.notifier.send_system_alert(error_msg, is_error=True)
            return pd.DataFrame()

    def process_scan_results(self, results):
        """Process scan results"""
        top_3 = results.head(3)
        self.current_top_ticker = top_3.iloc[0]['Ticker']
        self.top_ticker_score = top_3.iloc[0]['Score']
        self.runner_ups = top_3.iloc[1:].to_dict('records')

    async def notify_scan_results(self, results):
        """Notify about scan results"""
        total_scanned = len(self.trend_scanner.tickers)
        new_tickers = self.identify_new_tickers(results)
        
        await self.notifier.send_scan_results(
            results.head(5),
            total_scanned,
            new_tickers
        )

    def identify_new_tickers(self, results):
        """Identify new tickers compared to previous scan"""
        current_tickers = set(results['Ticker'])
        if hasattr(self, 'previous_scan_tickers'):
            return list(current_tickers - set(self.previous_scan_tickers))
        self.previous_scan_tickers = current_tickers
        return []

    async def place_trade(self, ticker, row=None):
        self.log_debug(f"Attempting to place trade for {ticker}")
        
        if not row:
            row = self.get_ticker_data(ticker)
            if not row:
                self.log_debug(f"No data found for {ticker}", "WARNING")
                return False
        
        qty = self.calculate_position_size(row['Price'])
        self.log_debug(f"Calculated position size: {qty} shares at ${row['Price']:.2f}")
        
        if await self.place_bracket_order(ticker, qty, row):
            self.log_debug(f"Successfully placed trade for {ticker}")
            self.record_new_position(ticker, row, qty)
            await self.notify_new_position(ticker, row, qty)
            return True
            
        self.log_debug(f"Failed to place trade for {ticker}", "WARNING")
        return False

    def get_ticker_data(self, ticker):
        """Get data for a specific ticker"""
        for candidate in self.runner_ups:
            if candidate['Ticker'] == ticker:
                return candidate
        return None

    def calculate_position_size(self, price):
        """Calculate position size based on allocation"""
        return max(1, int(ALLOCATION_PER_TICKER / price))

    def record_new_position(self, ticker, row, qty):
        """Record a new position in state"""
        self.active_positions[ticker] = {
            'qty': qty,
            'entry_price': row['Price'],
            'entry_time': datetime.now().isoformat(),
            'stop_loss': SmartStopLoss(
                entry_price=row['Price'],
                atr=row['ATR'],
                adx=row['ADX']
            )
        }
        self.executed_tickers.add(ticker)
        self.monthly_trade_count += 1

    async def notify_new_position(self, ticker, row, qty):
        """Notify about a new position"""
        stop_loss = self.active_positions[ticker]['stop_loss']
        
        await self.notifier.send_position_notification(
            ticker=ticker,
            qty=qty,
            entry_price=row['Price'],
            current_price=row['Price'],
            stop_loss=stop_loss.current_stop,
            take_profit=stop_loss.profit_target.current_target,
            hard_stop=stop_loss.hard_stop,
            atr=row['ATR'],
            adx=row['ADX']
        )
        
        await self.notifier.send_order_notification(
            order_type="buy",
            ticker=ticker,
            qty=qty,
            price=row['Price'],
            stop_loss=stop_loss.current_stop,
            take_profit=stop_loss.profit_target.current_target,
            hard_stop=stop_loss.hard_stop
        )

    async def place_bracket_order(self, ticker, qty, row):
        """Place bracket order with verification"""
        try:
            # Initialize stop system
            stop_system = SmartStopLoss(
                entry_price=row['Price'],
                atr=row['ATR'],
                adx=row['ADX']
            )
            
            # Get bracket details
            bracket_details = stop_system.get_bracket_orders(row['Price'], qty)
            
            # Submit order
            order_response = self.submit_bracket_order(ticker, qty, bracket_details)
            
            if order_response:
                self.place_hard_stop_order(ticker, qty, bracket_details["hard_stop"]["stop_price"])
                return True
            return False
        except Exception as e:
            error_msg = f"Bracket order failed for {ticker}: {str(e)}"
            print(error_msg)
            await self.notifier.send_system_alert(error_msg, is_error=True)
            return False

    def submit_bracket_order(self, ticker, qty, bracket_details):
        """Submit bracket order to Alpaca"""
        triple_barrier_order = {
            "symbol": ticker,
            "qty": str(qty),
            "side": "buy",
            "type": "market",
            "time_in_force": "gtc",
            "order_class": "bracket",
            "stop_loss": {
                "stop_price": bracket_details["stop_loss"]["stop_price"],
                "limit_price": bracket_details["stop_loss"]["limit_price"]
            },
            "take_profit": {
                "limit_price": bracket_details["take_profit"]["limit_price"]
            }
        }
        
        headers = {
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
        }
        
        response = requests.post(
            "https://paper-api.alpaca.markets/v2/orders",
            headers=headers,
            json=triple_barrier_order,
            timeout=15
        )
        
        if response.status_code == 200:
            print(f"Submitted bracket order for {ticker}")
            return True
        else:
            print(f"Order failed for {ticker}: {response.text}")
            return False

    def place_hard_stop_order(self, ticker, qty, stop_price):
        """Place hard stop order"""
        try:
            stop_order = StopOrderRequest(
                symbol=ticker,
                qty=qty,
                side=OrderSide.SELL,
                stop_price=stop_price,
                time_in_force=TimeInForce.GTC
            )
            trading_client.submit_order(stop_order)
            print(f"Placed hard stop at ${stop_price:.2f}")
        except Exception as e:
            print(f"Failed to place hard stop: {str(e)}")

    async def update_stop_losses(self):
        """Update all stop losses"""
        for ticker, data in list(self.active_positions.items()).copy():
            try:
                await self.update_single_stop_loss(ticker, data)
            except Exception as e:
                print(f"Stop update failed for {ticker}: {str(e)}")
                await self.handle_stop_update_error(ticker, e)
        
        self.save_current_state()

    async def update_single_stop_loss(self, ticker, data):
        """Update stop loss for a single position"""
        latest = self.get_latest_bar(ticker)
        if not latest:
            print(f"No data for {ticker}, skipping update")
            return
        
        # Prepare market data
        latest = self.enrich_market_data(ticker, latest, data)
        
        # Update stop
        old_stop = data['stop_loss'].current_stop
        new_stop = data['stop_loss'].update(latest)
        
        # Check for hard stop
        if data['stop_loss'].hard_stop_triggered:
            await self.close_position(ticker, "Hard stop triggered")
            return
        
        # Log and notify if stop changed
        if new_stop != old_stop:
            await self.process_stop_change(ticker, data, old_stop, new_stop)

    def enrich_market_data(self, ticker, latest, data):
        """Enrich market data with additional indicators"""
        if 'adx' not in latest:
            latest['adx'] = self.get_adx(ticker) or data['stop_loss'].base_adx
        if 'volume' not in latest:
            latest['volume'] = 1e6
        latest['avg_volume'] = self.get_avg_volume(ticker) or 1e6
        return latest

    async def process_stop_change(self, ticker, data, old_stop, new_stop):
        """Process a stop loss change"""
        print(f"Updated stop for {ticker}: {new_stop:.2f}")
        reason = "Trailing stop activated" if data['stop_loss'].activated else "Price moved favorably"
        
        # Log the change
        self.log_stop_change(ticker, old_stop, new_stop, data)
        
        # Notify
        await self.notify_stop_change(ticker, old_stop, new_stop, reason, data)
        
        # Manage orders
        self.manage_stop_orders(ticker, data['qty'], new_stop, data['stop_loss'].hard_stop)

    def log_stop_change(self, ticker, old_stop, new_stop, data):
        """Log a stop loss change"""
        log_data = {
            "ticker": ticker,
            "old_stop": old_stop,
            "new_stop": new_stop,
            "highest_high": data['stop_loss'].highest_high,
            "activated": data['stop_loss'].activated
        }
        if data['stop_loss'].hard_stop != old_stop:
            log_data["hard_stop"] = data['stop_loss'].hard_stop
            
        self.transaction_log.log(
            action="update_stop",
            data=log_data
        )

    async def notify_stop_change(self, ticker, old_stop, new_stop, reason, data):
        """Notify about a stop loss change"""
        await self.notifier.send_stop_loss_update(
            ticker=ticker,
            old_stop=old_stop,
            new_stop=new_stop,
            reason=reason
        )
        
        await self.notifier.send_position_update(
            ticker=ticker,
            entry_price=data['entry_price'],
            current_price=data['stop_loss'].previous_close,
            stop_loss=new_stop,
            pnl=(data['stop_loss'].previous_close - data['entry_price']) * data['qty']
        )

    def manage_stop_orders(self, ticker, qty, new_stop, hard_stop):
        """Manage stop orders for a position"""
        self.cancel_existing_orders(ticker)
        self.place_stop_order(ticker, qty, new_stop)
        self.place_hard_stop_order(ticker, qty, hard_stop)

    def cancel_existing_orders(self, ticker):
        """Cancel existing orders for a ticker"""
        try:
            orders = trading_client.get_orders(status='open', symbol=ticker)
            for order in orders:
                if order.order_type != 'stop':
                    trading_client.cancel_order_by_id(order.id)
            print(f"Cancelled existing orders for {ticker}")
        except Exception as e:
            print(f"Failed to cancel orders for {ticker}: {str(e)}")

    def place_stop_order(self, ticker, qty, stop_price):
        """Place a stop order"""
        try:
            stop_order = StopOrderRequest(
                symbol=ticker,
                qty=qty,
                side=OrderSide.SELL,
                stop_price=stop_price,
                time_in_force=TimeInForce.GTC
            )
            trading_client.submit_order(stop_order)
            print(f"Placed new STOP at ${stop_price:.2f} for {ticker}")
        except Exception as e:
            print(f"Failed to place stop for {ticker}: {str(e)}")

    async def close_position(self, ticker, reason="Hard stop triggered"):
        """Close a position completely"""
        if ticker not in self.active_positions:
            return False
            
        position = self.active_positions[ticker]
        qty = position['qty']
        
        try:
            # Submit market sell order
            market_order = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            trading_client.submit_order(market_order)
            
            # Get exit price
            latest = self.get_latest_bar(ticker)
            exit_price = latest['close'] if latest else position['entry_price']
            
            # Log the closure
            self.transaction_log.log(
                action="close_position",
                data={"ticker": ticker}
            )
            
            # Send notification
            if reason == "Hard stop triggered":
                await self.notifier.send_hard_stop_triggered(
                    ticker=ticker,
                    entry_price=position['entry_price'],
                    exit_price=exit_price
                )
            else:
                await self.notifier.send_system_alert(
                    f"Closed position for {ticker}: {reason}\n"
                    f"Entry: ${position['entry_price']:.2f}, Exit: ${exit_price:.2f}"
                )
            
            # Remove from active positions
            del self.active_positions[ticker]
            print(f"Closed position for {ticker}: {reason}")
            return True
        except Exception as e:
            print(f"Error closing position for {ticker}: {str(e)}")
            return False

    async def handle_stop_update_error(self, ticker, error):
        """Handle stop loss update errors"""
        await self.notifier.send_system_alert(
            f"Stop update failed for {ticker}: {str(error)}",
            is_error=True
        )
        if ticker in self.active_positions:
            del self.active_positions[ticker]

    def get_latest_bar(self, ticker, retries=3):
        """Get latest price data"""
        return get_latest_bar(ticker, retries)

    def get_adx(self, ticker, retries=2):
        """Get ADX for a ticker"""
        for i in range(retries):
            try:
                url = f"https://api.polygon.io/v1/indicators/adx/{ticker}"
                params = {'timespan': 'day', 'window': 14, 'apiKey': POLYGON_API_KEY}
                response = requests.get(url, params=params, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'results' in data and 'values' in data['results']:
                        return data['results']['values'][0]['value']
            except Exception as e:
                time.sleep(0.5 * (i+1))
        return None

    def get_avg_volume(self, ticker, days=30):
        """Get average volume for a ticker"""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
            params = {'adjusted': 'true', 'apiKey': POLYGON_API_KEY}
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and data['results']:
                    volumes = [r['v'] for r in data['results']]
                    return sum(volumes) / len(volumes)
        except:
            return None
        return None

    def save_current_state(self):
        """Save the current state"""
        return self.state_manager.save_state(self)

    async def run_market_regime_analysis(self):
        """Asynchronously analyze market regime with caching and duplicate prevention"""
        # Cache check (only run once every 12 hours)
        if hasattr(self, '_last_regime_run') and \
        (datetime.now() - self._last_regime_run).total_seconds() < 43200:  # 12 hours
            return getattr(self, 'current_regime', 'Neutral')

        try:
            await self.notifier.send_system_alert("Starting market regime analysis...")
            analysis_start = datetime.now()

            # 1. Get tickers (async threaded)
            print("\nFetching tickers...")
            tickers = await asyncio.to_thread(get_all_tickers)
            
            # 2. Map to sectors (async with concurrency control)
            print("\nMapping tickers to sectors...")
            sector_mappings = {}
            
            async def process_ticker(symbol):
                url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
                params = {"apiKey": POLYGON_API_KEY}
                await asyncio.sleep(RATE_LIMIT)
                
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, params=params, timeout=10) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                sector = data.get('results', {}).get('sic_description', 
                                        data.get('primary_exchange', 'Unknown'))
                                async with asyncio.Lock():
                                    sector_mappings.setdefault(sector, []).append(symbol)
                except Exception as e:
                    print(f"Sector mapping error for {symbol}: {str(e)}")

            # Process with concurrency limit
            semaphore = asyncio.Semaphore(10)
            async with semaphore:
                tasks = [process_ticker(symbol) for symbol in tickers]
                for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), 
                            desc="Mapping sectors", unit="ticker"):
                    await f

            # Filter small sectors
            self.sector_mappings = {k: v for k, v in sector_mappings.items() 
                                if k != 'Unknown' and len(v) > 10}

            # 3. Calculate sector weights (async)
            print("\nCalculating sector weights...")
            sector_weights = {}
            total_mcap = 0
            
            async def calculate_sector_weight(sector, symbols):
                sector_mcap = 0
                for symbol in symbols[:100]:  # Limit to top 100 per sector
                    try:
                        mcap = await asyncio.to_thread(get_market_cap, symbol)
                        sector_mcap += mcap if mcap else 0
                    except Exception as e:
                        print(f"Market cap error for {symbol}: {str(e)}")
                return sector, sector_mcap

            weight_tasks = [calculate_sector_weight(sector, symbols) 
                        for sector, symbols in self.sector_mappings.items()]
            for f in tqdm(asyncio.as_completed(weight_tasks), total=len(weight_tasks),
                        desc="Calculating weights", unit="sector"):
                sector, mcap = await f
                sector_weights[sector] = mcap
                total_mcap += mcap

            self.sector_weights = {s: mcap/total_mcap if total_mcap > 0 else 1/len(sector_weights)
                                for s, mcap in sector_weights.items()}

            # 4. Build sector composites (async)
            print("\nBuilding sector composites...")
            self.sector_composites = {}
            
            async def build_composite(sector, symbols):
                prices = []
                valid_symbols = []
                
                for symbol in symbols[:SECTOR_SAMPLE_SIZE]:
                    try:
                        price_data = await asyncio.to_thread(fetch_stock_data, symbol)
                        if price_data is not None and len(price_data) >= MIN_DAYS_DATA:
                            prices.append(price_data)
                            valid_symbols.append(symbol)
                    except Exception as e:
                        print(f"Data error for {symbol}: {str(e)}")
                
                if prices:
                    composite = pd.concat(prices, axis=1)
                    composite.columns = valid_symbols
                    return sector, composite.mean(axis=1).dropna()
                return sector, None

            composite_tasks = [build_composite(sector, symbols) 
                            for sector, symbols in self.sector_mappings.items()]
            for f in tqdm(asyncio.as_completed(composite_tasks), total=len(composite_tasks),
                        desc="Building composites", unit="sector"):
                sector, composite = await f
                if composite is not None:
                    self.sector_composites[sector] = composite

            # 5. Prepare market data (async)
            print("\nPreparing market data...")
            market_index = await asyncio.to_thread(
                self.sector_system.overall_analyzer.prepare_market_data,
                tickers
            )

            # 6. Analyze regime (async threaded for CPU-bound work)
            print("\nAnalyzing market regime...")
            regime_results = await asyncio.to_thread(
                self.sector_system.overall_analyzer.analyze_regime,
                market_index
            )

            current_regime = regime_results['regimes'][-1]
            print(f"\nCurrent Market Regime: {current_regime}")
            
            # Update scanner parameters
            if "Bull" in current_regime:
                self.trend_scanner.base_multiplier = 1.2
                print("Bull market detected: Using tighter stops (1.2x ATR)")
            elif "Bear" in current_regime:
                self.trend_scanner.base_multiplier = 2.0
                print("Bear market detected: Using wider stops (2.0x ATR)")

            # Cache results
            self._last_regime_run = datetime.now()
            self.current_regime = current_regime
            
            # Final notification
            duration = (datetime.now() - analysis_start).total_seconds() / 60
            await self.notifier.send_system_alert(
                f"Regime analysis completed in {duration:.1f} minutes\n"
                f"Current regime: {current_regime}"
            )
            
            return current_regime

        except Exception as e:
            error_msg = f"Regime analysis failed: {str(e)}"
            print(error_msg)
            await self.notifier.send_system_alert(error_msg, is_error=True)
            return "Neutral"

    async def async_map_tickers_to_sectors(self, tickers):
        """Async version of sector mapping"""
        self.sector_mappings = {}
        
        async def process_ticker(symbol):
            url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
            params = {"apiKey": POLYGON_API_KEY}
            await asyncio.sleep(RATE_LIMIT)
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            results = data.get('results', {})
                            sector = results.get('sic_description', 'Unknown')
                            if sector == 'Unknown':
                                sector = results.get('primary_exchange', 'Unknown')
                            async with asyncio.Lock():
                                self.sector_mappings.setdefault(sector, []).append(symbol)
            except Exception as e:
                print(f"Sector mapping failed for {symbol}: {str(e)}")
        
        # Process tickers with limited concurrency
        semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
        async with semaphore:
            tasks = [process_ticker(symbol) for symbol in tickers]
            for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), 
                        desc="Mapping sectors"):
                await f
        
        # Filter results
        self.sector_mappings = {k: v for k, v in self.sector_mappings.items() 
                            if k != 'Unknown' and len(v) > 10}
        return self.sector_mappings

    async def async_calculate_sector_weights(self, sector_mappings):
        """Async version of sector weight calculation"""
        total_mcap = 0
        sector_mcaps = {}
        
        async def process_sector(sector, tickers):
            sector_mcap = 0
            for symbol in tickers[:100]:  # Limit to top 100 per sector
                try:
                    mcap = await asyncio.to_thread(get_market_cap, symbol)
                    sector_mcap += mcap if mcap else 0
                except Exception as e:
                    print(f"Error getting market cap for {symbol}: {str(e)}")
            return sector, sector_mcap
        
        tasks = [process_sector(sector, tickers) 
                for sector, tickers in sector_mappings.items()]
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                    desc="Calculating weights"):
            sector, mcap = await f
            sector_mcaps[sector] = mcap
            total_mcap += mcap
        
        self.sector_weights = {sector: mcap/total_mcap if total_mcap > 0 else 1/len(sector_mcaps)
                            for sector, mcap in sector_mcaps.items()}
        return self.sector_weights

    async def async_build_sector_composites(self, sector_mappings, sample_size=50):
        """Async version of building sector composites"""
        self.sector_composites = {}
        
        async def process_sector(sector, tickers):
            prices_data = []
            valid_tickers = []
            
            for symbol in tickers[:sample_size]:
                try:
                    prices = await asyncio.to_thread(fetch_stock_data, symbol)
                    if prices is not None and len(prices) >= MIN_DAYS_DATA:
                        prices_data.append(prices)
                        valid_tickers.append(symbol)
                except Exception as e:
                    print(f"Error processing {symbol}: {str(e)}")
            
            if prices_data:
                composite = pd.concat(prices_data, axis=1)
                composite.columns = valid_tickers
                return sector, composite.mean(axis=1).dropna()
            return sector, None
        
        tasks = [process_sector(sector, tickers) 
                for sector, tickers in sector_mappings.items()]
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                    desc="Building composites"):
            sector, composite = await f
            if composite is not None:
                self.sector_composites[sector] = composite
        
        return self.sector_composites

    async def async_prepare_market_data(self, tickers, sample_size=100):
        """Async version of preparing market data"""
        prices_data = []
        valid_tickers = []
        
        async def process_ticker(symbol):
            prices = await asyncio.to_thread(fetch_stock_data, symbol)
            if prices is not None and len(prices) >= MIN_DAYS_DATA:
                return symbol, prices
            return None
        
        tasks = [process_ticker(symbol) for symbol in tickers[:sample_size]]
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                    desc="Preparing market data"):
            result = await f
            if result:
                symbol, prices = result
                prices_data.append(prices)
                valid_tickers.append(symbol)
        
        if not prices_data:
            raise ValueError("Insufficient data to create market composite")
        
        composite = pd.concat(prices_data, axis=1)
        composite.columns = valid_tickers
        return composite.mean(axis=1).dropna()

    async def async_analyze_regime(self, index_data, n_states=None):
        """Async version of regime analysis"""
        if n_states is None:
            n_states = N_STATES  # Use the global constant
        
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
        scaled_features = await asyncio.to_thread(
            self.feature_scaler.fit_transform, features
        )
        
        # Create model (use class instance's model if same n_states, else create new)
        if n_states == N_STATES:
            model = self.model
        else:
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
            await asyncio.to_thread(model.fit, scaled_features)
        
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
            state_labels = {i: f'State {i+1}' for i in range(n_states)}
        
        # Predict regimes
        states = await asyncio.to_thread(model.predict, scaled_features)
        state_probs = await asyncio.to_thread(model.predict_proba, scaled_features)
        
        return {
            'model': model,
            'regimes': [state_labels[s] for s in states],
            'probabilities': state_probs,
            'features': features,
            'index_data': index_data[features.index[0]:],
            'state_labels': state_labels
        }
    
    async def monitor_and_update(self):
        await self.initialize()
        
        while True:
            now = datetime.now()
            self.log_debug(f"Starting monitoring cycle at {now.strftime('%H:%M:%S')}")
            
            # Force a scan if we have no candidates
            if not self.runner_ups:
                self.log_debug("No candidates available - forcing scan")
                await self.run_weekly_scan()
            
            # Market regime analysis
            current_regime = await self.run_market_regime_analysis()
            self.log_debug(f"Current market regime: {current_regime}")
            
            # Trade execution (run every hour regardless of regime)
            if now.minute == 0:  # Top of hour
                self.log_debug("Starting hourly trade execution check")
                await self.execute_trades_safely()
                await self.update_stop_losses()
            
            # Weekly scan (Sunday 4PM)
            if (now.weekday() == 6 and now.hour == 16) or not self.runner_ups:
                self.log_debug("Starting weekly scan (Sunday 4PM or no candidates)")
                await self.run_weekly_scan()
            
            # State saving
            if now.minute % 5 == 0:
                self.save_current_state()
            
            await asyncio.sleep(60)

    async def shutdown(self):
        """Graceful shutdown procedure"""
        try:
            # 1. Send notification
            await self.notifier.send_system_alert("Trading system shutting down gracefully")
            
            # 2. Cancel all open orders
            await self.cancel_all_orders()
            
            # 3. Save final state
            self.save_current_state()
            
            # 4. Close any remaining connections
            await self.close_connections()
            
            print("Trading system shutdown complete")
            
        except Exception as e:
            print(f"Error during shutdown: {str(e)}")
        finally:
            # Ensure the program exits
            asyncio.get_event_loop().stop()

    async def cancel_all_orders(self):
        """Cancel all open orders on shutdown"""
        try:
            orders = trading_client.get_orders(status='open')
            for order in orders:
                try:
                    trading_client.cancel_order_by_id(order.id)
                    print(f"Cancelled order {order.id}")
                    await asyncio.sleep(0.1)  # Rate limiting
                except Exception as e:
                    print(f"Failed to cancel order {order.id}: {str(e)}")
        except Exception as e:
            print(f"Error cancelling orders: {str(e)}")

    async def close_connections(self):
        """Close any open network connections"""
        # Add any connection cleanup needed
        pass

    async def handle_error(self, error):
        """Handle system errors"""
        error_msg = f"System error: {str(error)}"
        print(error_msg)
        await self.notifier.send_system_alert(error_msg, is_error=True)
        self.save_current_state()
    
# ===============
# Main Execution
# ===============
async def main():
    trading_system = TradingSystem()
    try:
        await trading_system.monitor_and_update()
    except asyncio.CancelledError:
        await trading_system.shutdown()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        await trading_system.notifier.send_system_alert(f"Fatal error: {str(e)}", is_error=True)
        await trading_system.shutdown()
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Application terminated by user")