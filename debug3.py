import asyncio
import json
import math
import os
import pickle
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue
from threading import Thread

import aiohttp
import discord
import numpy as np
import pandas as pd
import requests
from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest, TrailingStopOrderRequest
from alpaca.trading.models import Position
from discord import Webhook
from hmmlearn import hmm
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    DISCORD_WEBHOOK_URL,
    POLYGON_API_KEY,
)

# ======================
# Combined Configuration
# ======================
EXCHANGES = ["XNYS", "XNAS", "XASE"]  # NYSE, NASDAQ, AMEX
MAX_TICKERS_PER_EXCHANGE = 200
RATE_LIMIT = 0.0001  # seconds between requests
MIN_DAYS_DATA = 200  # Minimum days of data required for analysis
N_STATES = 3  # Bull/Neutral/Bear regimes
SECTOR_SAMPLE_SIZE = 50  # Stocks per sector for composite
ALLOCATION_PER_TICKER = 100  # $10,000 per position
MAX_TICKERS_TO_SCAN = 300  # Limit for weekly scanner
STATE_FILE = "trading_system_state.pkl"  # File to save system state
TRANSACTION_LOG_FILE = "trading_transactions.log"
AUTO_SAVE_INTERVAL = 300  # 5 minutes in seconds
MAX_PORTFOLIO_ALLOCATION = 0.9  # Use maximum 90% of buying power

# Global Cache
DATA_CACHE = {
    "tickers": None,
    "sector_mappings": None,
    "last_updated": None,
    "stock_data": {},
    "last_regime": None,
    "restricted_tickers": set(),
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

    async def send_embed(self, title, description, color=0x00FF00, fields=None):
        """Send an embed message to Discord"""
        embed = discord.Embed(
            title=title, description=description, color=color, timestamp=datetime.now()
        )

        if fields:
            for name, value, inline in fields:
                embed.add_field(name=name, value=value, inline=inline)

        async with aiohttp.ClientSession() as session:
            webhook = Webhook.from_url(self.webhook_url, session=session)
            try:
                await webhook.send(embed=embed, username="Trading System Bot")
            except Exception as e:
                print(f"Error sending Discord notification: {str(e)}")

    async def send_order_notification(
        self, order_type, ticker, qty, price, stop_loss=None, take_profit=None, hard_stop=None
    ):
        """Send notification about an order"""
        color = 0x00FF00 if order_type.lower() == "buy" else 0xFF0000
        fields = [
            ("Ticker", ticker, True),
            ("Quantity", str(qty), True),
            ("Price", f"${price:.2f}", True),
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
            fields=fields,
        )

    async def send_position_update(self, ticker, entry_price, current_price, stop_loss, pnl=None):
        """Send position update notification"""
        fields = [
            ("Ticker", ticker, True),
            ("Entry Price", f"${entry_price:.2f}", True),
            ("Current Price", f"${current_price:.2f}", True),
            ("Stop Loss", f"${stop_loss:.2f}", True),
        ]

        if pnl is not None:
            pnl_percent = ((current_price - entry_price) / entry_price) * 100
            fields.append(("P&L", f"${pnl:.2f} ({pnl_percent:.2f}%)", True))

        await self.send_embed(
            title=f"Position Update: {ticker}",
            description="Position details update",
            color=0xFFFF00,
            fields=fields,
        )

    async def send_stop_loss_update(self, ticker, old_stop, new_stop, reason=None):
        """Send stop loss adjustment notification"""
        description = f"Stop loss adjusted for {ticker}"
        if reason:
            description += f"\n**Reason:** {reason}"

        await self.send_embed(
            title="Stop Loss Updated",
            description=description,
            color=0xFFA500,
            fields=[
                ("Ticker", ticker, True),
                ("Old Stop", f"${old_stop:.2f}", True),
                ("New Stop", f"${new_stop:.2f}", True),
            ],
        )

    async def send_hard_stop_triggered(self, ticker, entry_price, exit_price):
        """Notification when hard stop is triggered"""
        loss = entry_price - exit_price
        loss_percent = (loss / entry_price) * 100

        await self.send_embed(
            title="HARD STOP TRIGGERED",
            description="Position exited due to hard stop breach",
            color=0xFF0000,
            fields=[
                ("Ticker", ticker, True),
                ("Entry Price", f"${entry_price:.2f}", True),
                ("Exit Price", f"${exit_price:.2f}", True),
                ("Loss", f"${loss:.2f} ({loss_percent:.2f}%)", True),
            ],
        )

    async def send_system_alert(self, message, is_error=False):
        """Send system alert/error notification"""
        await self.send_embed(
            title="SYSTEM ALERT" if is_error else "System Notification",
            description=message,
            color=0xFF0000 if is_error else 0x0000FF,
        )

    async def send_scan_results(self, results_df):
        """Send formatted scan results to Discord"""
        if results_df.empty:
            await self.send_system_alert("Scan completed with no results")
            return

        message = "**Weekly Scan Results:**\n"
        for _, row in results_df.iterrows():
            message += (
                f"\n**{row['Rank']}. {row['Ticker']}** "
                f"(Score: {row['Score']}, Price: ${row['Price']:.2f})\n"
                f"ADX: {row['ADX']:.1f}, ATR: {row['ATR']:.2f}, "
                f"Stop: ${row['Initial_Stop']:.2f}\n"
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
            "data": data,
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
                "qty": data["qty"],
                "entry_price": data["price"],
                "entry_time": entry["timestamp"],
                "stop_loss": SmartStopLoss(
                    entry_price=data["price"],
                    atr=data["atr"],
                    adx=data["adx"],
                ),
            }

        elif action == "update_stop":
            ticker = data["ticker"]
            if ticker in trading_system.active_positions:
                stop_loss = trading_system.active_positions[ticker]["stop_loss"]
                stop_loss.current_stop = data["new_stop"]
                stop_loss.highest_high = data["highest_high"]
                stop_loss.activated = data["activated"]
                if "hard_stop" in data:
                    stop_loss.hard_stop = data["hard_stop"]

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
            init_params="stmc",
            params="stmc",
            random_state=42,
        )
        self.state_labels = {}
        self.feature_scaler = StandardScaler()

    def prepare_market_data(self, tickers, sample_size=100):
        prices_data = []
        valid_tickers = []

        print("\nBuilding market composite from multiple exchanges...")

        # Parallel processing for data fetching
        def fetch_and_validate(symbol):
            prices = fetch_stock_data(symbol)
            if prices is not None and len(prices) >= MIN_DAYS_DATA:
                return symbol, prices
            return None

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(fetch_and_validate, symbol): symbol
                for symbol in tickers[:sample_size]
            }

            for future in tqdm(
                as_completed(futures),
                total=min(sample_size, len(tickers)),
                desc="Fetching Market Data",
            ):
                symbol = futures[future]
                try:
                    result = future.result()
                    if result:
                        valid_tickers.append(result[0])
                        prices_data.append(result[1])
                except Exception as e:
                    print(f"Error processing {symbol}: {str(e)}")

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
        features = pd.DataFrame(
            {
                "returns": log_returns,
                "volatility": log_returns.rolling(21).std(),
                "momentum": log_returns.rolling(14).mean(),
            }
        ).dropna()

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
            random_state=42,
        )

        # Fit model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(scaled_features)

        # Label states
        state_means = sorted(
            [(i, np.mean(model.means_[i])) for i in range(model.n_components)],
            key=lambda x: x[1],
        )

        state_labels = {}
        if n_states == 3:
            state_labels = {
                state_means[0][0]: "Bear",
                state_means[1][0]: "Neutral",
                state_means[2][0]: "Bull",
            }
        elif n_states == 4:
            state_labels = {
                state_means[0][0]: "Severe Bear",
                state_means[1][0]: "Mild Bear",
                state_means[2][0]: "Mild Bull",
                state_means[3][0]: "Strong Bull",
            }
        else:
            for i in range(n_states):
                state_labels[i] = f"State {i+1}"

        # Predict regimes
        states = model.predict(scaled_features)
        state_probs = model.predict_proba(scaled_features)

        return {
            "model": model,
            "regimes": [state_labels[s] for s in states],
            "probabilities": state_probs,
            "features": features,
            "index_data": index_data[features.index[0] :],
            "state_labels": state_labels,
        }


class SectorRegimeSystem:
    def __init__(self):
        self.sector_mappings = {}
        self.sector_composites = {}
        self.sector_analyzers = {}
        self.overall_analyzer = MarketRegimeAnalyzer()
        self.sector_weights = {}
        self.sector_scores = {}

    def map_tickers_to_sectors(self, tickers):
        self.sector_mappings = {}

        # Parallel sector mapping
        def map_single_ticker(symbol):
            url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
            params = {"apiKey": POLYGON_API_KEY}
            try:
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json().get("results", {})
                    sector = data.get("sic_description", "Unknown")
                    if sector == "Unknown":
                        sector = data.get("primary_exchange", "Unknown")
                    return symbol, sector
            except Exception as e:
                print(f"Sector mapping failed for {symbol}: {str(e)}")
            return symbol, "Unknown"

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(map_single_ticker, symbol): symbol for symbol in tickers}

            for future in tqdm(
                as_completed(futures), total=len(tickers), desc="Mapping Sectors"
            ):
                try:
                    symbol, sector = future.result()
                    if sector != "Unknown":
                        self.sector_mappings.setdefault(sector, []).append(symbol)
                except Exception as e:
                    print(f"Error processing sector mapping: {str(e)}")

        # Remove unknown sectors and small sectors
        self.sector_mappings = {
            k: v for k, v in self.sector_mappings.items() if k != "Unknown" and len(v) > 10
        }
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

        self.sector_weights = {
            sector: mcap / total_mcap if total_mcap > 0 else 1 / len(sector_mcaps)
            for sector, mcap in sector_mcaps.items()
        }
        return self.sector_weights

    def build_sector_composites(self, sample_size=50):
        print("\nBuilding sector composites...")
        self.sector_composites = {}

        # Process each sector in parallel
        def build_sector_composite(sector_data):
            sector, tickers = sector_data
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
                return sector, composite.mean(axis=1).dropna()
            return sector, None

        sector_data = list(self.sector_mappings.items())

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(build_sector_composite, data): data for data in sector_data
            }

            for future in tqdm(
                as_completed(futures), total=len(sector_data), desc="Building Composites"
            ):
                try:
                    sector, composite = future.result()
                    if composite is not None:
                        self.sector_composites[sector] = composite
                except Exception as e:
                    print(f"Error building composite: {str(e)}")

    def analyze_sector_regimes(self, n_states=3):
        print("\nAnalyzing sector regimes...")
        self.sector_analyzers = {}

        # Process each sector in parallel
        def analyze_single_sector(sector_data):
            sector, composite = sector_data
            try:
                analyzer = MarketRegimeAnalyzer()
                results = analyzer.analyze_regime(composite, n_states=n_states)
                return sector, {
                    "results": results,
                    "composite": composite,
                    "volatility": composite.pct_change().std(),
                }
            except Exception as e:
                print(f"Error analyzing {sector}: {str(e)}")
                return sector, None

        sector_data = list(self.sector_composites.items())

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(analyze_single_sector, data): data for data in sector_data
            }

            for future in tqdm(
                as_completed(futures), total=len(sector_data), desc="Analyzing Sectors"
            ):
                try:
                    sector, data = future.result()
                    if data is not None:
                        self.sector_analyzers[sector] = data
                except Exception as e:
                    print(f"Error processing sector analysis: {str(e)}")

    def calculate_sector_scores(self, market_regime):
        self.sector_scores = {}

        if not self.sector_analyzers:
            return pd.Series()

        for sector, data in self.sector_analyzers.items():
            try:
                if "results" not in data or "probabilities" not in data["results"]:
                    continue

                current_probs = data["results"]["probabilities"][-1]
                state_labels = data["results"].get("state_labels", {})

                # Calculate bull/bear probabilities
                bull_prob = (
                    sum(
                        current_probs[i]
                        for i, label in state_labels.items()
                        if "Bull" in label
                    )
                    if state_labels
                    else 0
                )
                bear_prob = (
                    sum(
                        current_probs[i]
                        for i, label in state_labels.items()
                        if "Bear" in label
                    )
                    if state_labels
                    else 0
                )

                # Calculate momentum
                momentum = (
                    data["composite"].pct_change(21).iloc[-1]
                    if len(data["composite"]) > 21
                    else 0
                )

                # Base score
                base_score = bull_prob - bear_prob + (momentum * 10)

                # Apply regime-specific adjustments
                if market_regime == "Bull":
                    beta_factor = 1 + (data["volatility"] / 0.02)
                    base_score *= beta_factor
                elif market_regime == "Bear":
                    volatility_factor = 1 + (0.04 - min(data["volatility"], 0.04)) / 0.02
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
        current_high = current_bar["high"]
        current_close = current_bar["close"]

        # Calculate trend strength
        adx_strength = min(2.0, self.adx / 30)  # 1.0 = ADX 30, 2.0 = ADX 60
        volume_ratio = current_bar.get("volume", 1e6) / current_bar.get("avg_volume", 1e6)
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
        current_close = current_bar["close"]
        rsi = current_bar.get("rsi", 50)

        # Basic profit taking condition
        if current_close >= self.current_target:
            return True

        # Hold conditions for strong trends
        if self.strength_factor > 1.5:
            # Only take profit if RSI > 70 and closing near highs
            if rsi < 70 or current_close < (current_bar["high"] * 0.99):
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
            adx=adx,
        )

    def update(self, current_bar):
        current_high = current_bar["high"]
        current_low = current_bar["low"]
        current_close = current_bar["close"]
        current_adx = current_bar.get("adx", self.base_adx)

        # Update trend strength (combines ADX and volume)
        adx_strength = min(1.0, current_adx / 50)
        volume_ratio = current_bar.get("volume", 1e6) / current_bar.get("avg_volume", 1e6)
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
        if current_low <= self.hard_stop * 0.995:  # 0.5% buffer to prevent false triggers
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
        if latest["high"] > self.highest_high:
            self.highest_high = latest["high"]

        # Recalculate stop based on current market
        self.update(latest)

    def should_hold(self, current_bar):
        current_low = current_bar["low"]
        current_close = current_bar["close"]
        rsi = current_bar.get("rsi", 50)
        volatility_ratio = current_bar.get("volatility_ratio", 1.0)

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

        current_low = current_bar["low"]
        current_close = current_bar["close"]
        rsi = current_bar.get("rsi", 50)
        volatility_ratio = current_bar.get("volatility_ratio", 1.0)

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
            if "volume" in current_bar and "avg_volume" in current_bar:
                volume_ratio = current_bar["volume"] / current_bar["avg_volume"]
                if volume_ratio > 1.2:
                    return True

        return False

    def get_status(self):
        return {
            "current_stop": self.current_stop,
            "growth_potential": self.growth_potential,
            "activated": self.activated,
            "consecutive_confirmations": self.consecutive_confirmations,
            "hard_stop": self.hard_stop,
            "hard_stop_triggered": self.hard_stop_triggered,
            "profit_target": self.profit_target.current_target,
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
                "limit_price": normalize_price(normalized_stop * 0.98),  # Add limit price for stop-limit
            },
            "take_profit": {"limit_price": normalized_profit_target},
            "hard_stop": {"stop_price": normalized_hard_stop},
        }

    def get_serializable_state(self):
        """Return a dictionary of the current state that can be serialized"""
        return {
            "entry": self.entry,
            "initial_atr": self.initial_atr,
            "base_adx": self.base_adx,
            "activation_percent": self.activation_percent,
            "base_multiplier": self.base_multiplier,
            "activated": self.activated,
            "highest_high": self.highest_high,
            "current_stop": self.current_stop,
            "growth_potential": self.growth_potential,
            "consecutive_confirmations": self.consecutive_confirmations,
            "last_direction": self.last_direction,
            "previous_close": self.previous_close,
            "hard_stop": self.hard_stop,
            "hard_stop_triggered": self.hard_stop_triggered,
            "trend_strength": self.trend_strength,
            "profit_target_state": {
                "entry_price": self.profit_target.entry,
                "initial_target": self.profit_target.base_target,
                "atr": self.profit_target.atr,
                "adx": self.profit_target.adx,
                "current_target": self.profit_target.current_target,
                "strength_factor": self.profit_target.strength_factor,
                "breached_levels": self.profit_target.breached_levels,
                "last_high": self.profit_target.last_high,
            },
        }

    @classmethod
    def from_serialized_state(cls, state):
        """Recreate a SmartStopLoss instance from serialized state"""
        instance = cls(
            entry_price=state["entry"],
            atr=state["initial_atr"],
            adx=state["base_adx"],
            activation_percent=state["activation_percent"],
            base_multiplier=state["base_multiplier"],
        )

        # Restore all state variables
        instance.activated = state["activated"]
        instance.highest_high = state["highest_high"]
        instance.current_stop = state["current_stop"]
        instance.growth_potential = state["growth_potential"]
        instance.consecutive_confirmations = state["consecutive_confirmations"]
        instance.last_direction = state["last_direction"]
        instance.previous_close = state["previous_close"]
        instance.hard_stop = state["hard_stop"]
        instance.hard_stop_triggered = state["hard_stop_triggered"]
        instance.trend_strength = state["trend_strength"]

        # Restore profit target
        profit_state = state["profit_target_state"]
        instance.profit_target = SmartProfitTarget(
            entry_price=profit_state["entry_price"],
            initial_target=profit_state["initial_target"],
            atr=profit_state["atr"],
            adx=profit_state["adx"],
        )
        instance.profit_target.current_target = profit_state["current_target"]
        instance.profit_target.strength_factor = profit_state["strength_factor"]
        instance.profit_target.breached_levels = profit_state["breached_levels"]
        instance.profit_target.last_high = profit_state["last_high"]

        return instance


class PolygonTrendScanner:
    def __init__(self, max_tickers=None):
        self.api_key = POLYGON_API_KEY
        self.base_url = "https://api.polygon.io/v2"
        self.tickers = self.load_tickers(max_tickers)
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        self.start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        self.base_multiplier = 1.5  # Default stop loss multiplier

    def load_tickers(self, max_tickers=None):
        if DATA_CACHE.get("tickers"):
            tickers = DATA_CACHE["tickers"]
        else:
            tickers = get_all_tickers()
            DATA_CACHE["tickers"] = tickers

        if max_tickers is not None and max_tickers > 0:
            return tickers[:max_tickers]
        return tickers

    def get_polygon_data(self, ticker):
        url = f"{self.base_url}/aggs/ticker/{ticker}/range/1/day/{self.start_date}/{self.end_date}"
        params = {"adjusted": "true", "apiKey": self.api_key}

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "results" not in data or len(data["results"]) < 200:
                return None

            df = pd.DataFrame(data["results"])
            df["date"] = pd.to_datetime(df["t"], unit="ms")
            df.set_index("date", inplace=True)
            df.rename(
                columns={
                    "o": "Open",
                    "h": "High",
                    "l": "Low",
                    "c": "Close",
                    "v": "Volume",
                },
                inplace=True,
            )

            return df[["Open", "High", "Low", "Close", "Volume"]]

        except Exception as e:
            return None

    def calculate_indicators(self, df):
        if df is None or len(df) < 200:
            return None

        try:
            latest = df.iloc[-1].copy()
            sma_50 = df["Close"].rolling(50).mean().iloc[-1]
            sma_200 = df["Close"].rolling(200).mean().iloc[-1]

            distance_sma50 = ((latest["Close"] - sma_50) / sma_50) * 100
            distance_sma200 = ((latest["Close"] - sma_200) / sma_200) * 100

            # Calculate True Range and ATR
            df["prev_close"] = df["Close"].shift(1)
            df["H-L"] = df["High"] - df["Low"]
            df["H-PC"] = abs(df["High"] - df["prev_close"])
            df["L-PC"] = abs(df["Low"] - df["prev_close"])
            df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
            atr = df["TR"].rolling(14).mean().iloc[-1]

            # Calculate ADX
            plus_dm = df["High"].diff()
            minus_dm = -df["Low"].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0

            atr_14 = df["TR"].rolling(14).mean()
            plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
            minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(14).mean().iloc[-1]

            # Calculate 10-day change
            if len(df) >= 10:
                ten_day_change = ((latest["Close"] / df["Close"].iloc[-10]) - 1) * 100
            else:
                ten_day_change = 0

            # Calculate average volume
            avg_volume = df["Volume"].rolling(30).mean().iloc[-1]

            return {
                "Close": float(latest["Close"]),
                "SMA_50": float(sma_50),
                "SMA_200": float(sma_200),
                "Distance_from_SMA50": float(distance_sma50),
                "Distance_from_SMA200": float(distance_sma200),
                "Volume": float(latest["Volume"]),
                "ATR": float(atr),
                "ADX": float(adx),
                "10D_Change": float(ten_day_change),
                "AvgVolume": float(avg_volume),
            }

        except Exception as e:
            return None
        finally:
            cols_to_drop = ["prev_close", "H-L", "H-PC", "L-PC", "TR"]
            for col in cols_to_drop:
                if col in df.columns:
                    df.drop(col, axis=1, inplace=True)

    def scan_tickers(self):
        """Scan tickers in parallel using thread pooling"""
        results = []

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {
                executor.submit(self.process_ticker, ticker): ticker
                for ticker in self.tickers
            }

            # Process results as they complete
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"Error processing {ticker}: {str(e)}")

        # Create DataFrame and sort
        if results:
            df_results = pd.DataFrame(results)
            df_results["Rank"] = df_results["Score"].rank(ascending=False, method="min").astype(int)
            return df_results.sort_values("Score", ascending=False).reset_index(drop=True)[
                [
                    "Rank",
                    "Ticker",
                    "Score",
                    "Price",
                    "ADX",
                    "SMA50_Distance%",
                    "SMA200_Distance%",
                    "10D_Change%",
                    "Volume",
                    "ATR",
                    "ATR_Ratio",
                    "Initial_Stop",
                    "Hard_Stop",
                    "Take_Profit",
                    "Risk_per_Share",
                    "Risk_Percent",
                    "ATR_Multiplier",
                    "Activation_Percent",
                ]
            ]
        return pd.DataFrame()

    def process_ticker(self, ticker):
        """Process a single ticker (to be run in parallel)"""
        data = self.get_polygon_data(ticker)
        if data is None:
            return None

        indicators = self.calculate_indicators(data)

        if indicators is None:
            return None

        try:
            # Trend conditions
            above_sma50 = indicators["Close"] > indicators["SMA_50"]
            above_sma200 = indicators["Close"] > indicators["SMA_200"]
            strong_adx = indicators["ADX"] > 25

            if above_sma50 and above_sma200 and strong_adx:
                # Calculate composite score (0-100)
                adx_component = min(40, (indicators["ADX"] / 50) * 40)
                sma50_component = min(30, max(0, indicators["Distance_from_SMA50"]) * 0.3)
                volume_component = min(20, math.log10(max(1, indicators["Volume"] / 10000)))
                momentum_component = min(10, max(0, indicators["10D_Change"]))

                score = min(
                    100,
                    adx_component + sma50_component + volume_component + momentum_component,
                )

                # Initialize stop loss system
                stop_system = SmartStopLoss(
                    entry_price=indicators["Close"],
                    atr=indicators["ATR"],
                    adx=indicators["ADX"],
                    activation_percent=0.05,
                    base_multiplier=self.base_multiplier,
                )

                # Calculate risk metrics
                risk_per_share = indicators["Close"] - stop_system.current_stop
                risk_percent = (risk_per_share / indicators["Close"]) * 100

                return {
                    "Ticker": ticker,
                    "Score": round(score, 1),
                    "Price": round(indicators["Close"], 2),
                    "ADX": round(indicators["ADX"], 1),
                    "SMA50_Distance%": round(indicators["Distance_from_SMA50"], 1),
                    "SMA200_Distance%": round(indicators["Distance_from_SMA200"], 1),
                    "Volume": int(indicators["Volume"]),
                    "ATR": round(indicators["ATR"], 2),
                    "ATR_Ratio": round(
                        (indicators["Close"] - indicators["SMA_50"]) / indicators["ATR"], 1
                    ),
                    "10D_Change%": round(indicators["10D_Change"], 1),
                    "Initial_Stop": round(stop_system.current_stop, 2),
                    "Hard_Stop": round(stop_system.hard_stop, 2),
                    "Take_Profit": round(stop_system.profit_target.current_target, 2),
                    "Risk_per_Share": round(risk_per_share, 2),
                    "Risk_Percent": round(risk_percent, 2),
                    "ATR_Multiplier": self.base_multiplier,
                    "Activation_Percent": 5.0,
                }
        except Exception as e:
            return None
        return None


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
                "qty": data["qty"],
                "entry_price": data["entry_price"],
                "stop_loss_state": data["stop_loss"].get_serializable_state(),
                "entry_time": data.get("entry_time", datetime.now().isoformat()),
                "order_ids": data.get("order_ids", {}),
            }

        state = {
            "active_positions": serializable_positions,
            "last_scan_date": trading_system.last_scan_date,
            "last_regime": trading_system.sector_system.overall_analyzer.state_labels,
            "saved_at": datetime.now(),
            "current_top_ticker": trading_system.current_top_ticker,
            "top_ticker_score": trading_system.top_ticker_score,
            "runner_ups": trading_system.runner_ups,
            "monthly_trade_count": trading_system.monthly_trade_count,
            "last_month_checked": trading_system.last_month_checked,
            "executed_tickers": list(trading_system.executed_tickers),
        }

        try:
            with open(self.state_file, "wb") as f:
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

    def load_state(self, trading_system_instance):
        """Load the saved state and replay transactions"""
        state = None
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "rb") as f:
                    state = pickle.load(f)
                print(f"Loaded state from {state['saved_at']}")
            except Exception as e:
                print(f"Error loading state: {str(e)}")

        # Always replay transaction log to catch recent changes
        self.transaction_log.replay_log(trading_system_instance)
        return state

    def get_order_details(self, order_id):
        """Get order details from Alpaca by order ID"""
        try:
            order = trading_client.get_order_by_id(order_id)
            return {
                "symbol": order.symbol,
                "qty": float(order.qty),
                "filled_qty": float(order.filled_qty),
                "status": order.status.value,
            }
        except Exception as e:
            print(f"Error fetching order {order_id}: {str(e)}")
            return None


# ========================
# Utility Functions
# ========================
def get_all_tickers():
    all_tickers = []
    restricted_types = {"BOND", "WARRANT", "RIGHT", "UNIT", "ETF", "ETN", "PFD"}  # Non-stock security types

    for exchange in EXCHANGES:
        url = "https://api.polygon.io/v3/reference/tickers"
        params = {
            "exchange": exchange,
            "market": "stocks",
            "active": "true",
            "limit": MAX_TICKERS_PER_EXCHANGE,
            "apiKey": POLYGON_API_KEY,
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            if "results" in data and data["results"]:
                # Filter out non-stock securities
                valid_tickers = [
                    (t["ticker"], t.get("market_cap", 0))
                    for t in data["results"]
                    if t.get("type") not in restricted_types  # Exclude non-stock types
                    and not any(t["ticker"].endswith(ext) for ext in [".WS", ".WT", ".U", ".RT", ".WI"])  # Exclude warrants
                ]
                
                # Sort by market cap and add to list
                tickers = sorted(valid_tickers, key=lambda x: x[1], reverse=True)
                exchange_tickers = [t[0] for t in tickers]
                all_tickers.extend(exchange_tickers)
        except Exception as e:
            # Fallback to index components if primary method fails
            if exchange == "XNYS":
                tables = pd.read_html("https://en.wikipedia.org/wiki/NYSE_Composite")
                tickers = tables[2]["Symbol"].tolist()[:MAX_TICKERS_PER_EXCHANGE]
                all_tickers.extend([t for t in tickers if "." not in t and "-" not in t])
            elif exchange == "XNAS":
                tables = pd.read_html("https://en.wikipedia.org/wiki/NASDAQ-100")
                tickers = tables[4]["Ticker"].tolist()[:MAX_TICKERS_PER_EXCHANGE]
                all_tickers.extend([t for t in tickers if "." not in t and "-" not in t])
            elif exchange == "XASE":
                tables = pd.read_html(
                    "https://en.wikipedia.org/wiki/List_of_American_Stock_Exchange_companies"
                )
                tickers = tables[0]["Symbol"].tolist()[:MAX_TICKERS_PER_EXCHANGE]
                all_tickers.extend([t for t in tickers if "." not in t and "-" not in t])

    # Remove duplicates and limit total tickers
    unique_tickers = list(set(all_tickers))
    return unique_tickers[: MAX_TICKERS_PER_EXCHANGE * len(EXCHANGES)]


def fetch_stock_data(symbol, days=365):
    if symbol in DATA_CACHE["stock_data"]:
        cached_data = DATA_CACHE["stock_data"][symbol]
        if datetime.now() - cached_data["timestamp"] < timedelta(days=1):
            return cached_data["data"]

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": POLYGON_API_KEY,
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                df = pd.DataFrame(results)
                df["date"] = pd.to_datetime(df["t"], unit="ms")
                result = df.set_index("date")["c"]
                # Update cache
                DATA_CACHE["stock_data"][symbol] = {"data": result, "timestamp": datetime.now()}
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
            return response.json().get("results", {}).get("market_cap", 0)
    except Exception:
        return 0
    return 0


def get_latest_bar(ticker, retries=3):
    for i in range(retries):
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev"
            params = {"adjusted": "true", "apiKey": POLYGON_API_KEY}
            response = requests.get(url, params=params, timeout=5)

            if response.status_code == 200:
                result = response.json().get("results", [])
                if result:
                    return {
                        "open": result[0]["o"],
                        "high": result[0]["h"],
                        "low": result[0]["l"],
                        "close": result[0]["c"],
                        "volume": result[0].get("v", 1e6),
                    }
            elif response.status_code == 429:
                wait = int(response.headers.get("Retry-After", 30))
                print(f"Rate limited. Waiting {wait} seconds")
                time.sleep(wait)
            time.sleep(1)
        except Exception:
            time.sleep(2**i)  # Exponential backoff
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
        self.state_manager = StateManager()
        self.notifier = DiscordNotifier()
        self.transaction_log = TransactionLogger()

        # New strategy attributes
        self.current_top_ticker = None
        self.top_ticker_score = 0
        self.runner_ups = []  # List of dictionaries for runner-up tickers
        self.monthly_trade_count = 0
        self.last_month_checked = None
        self.executed_tickers = set()  # Track all executed tickers

        # Load previous state if available
        self.load_previous_state()

    def normalize_price(self, price):
        """Round price to proper increment based on price level"""
        if price < 1.00:
            return round(price, 4)  # $0.0001 increments for stocks < $1
        elif price < 10.00:
            return round(price, 3)  # $0.001 increments for stocks $1-$10
        else:
            return round(price, 2)  # $0.01 increments for stocks > $10

    def reset_monthly_count(self):
        """Reset trade count at start of each month"""
        now = datetime.now()
        if not self.last_month_checked or now.month != self.last_month_checked.month:
            self.monthly_trade_count = 0
            self.last_month_checked = now
            self.executed_tickers = set()
            print("Monthly trade count reset")

    def load_previous_state(self):
        """Load previous state from file and sync with market"""
        state = self.state_manager.load_state(self)
        if state:
            # Recreate positions and sync with market
            for ticker, data in state.get("active_positions", {}).items():
                try:
                    stop_loss = SmartStopLoss.from_serialized_state(data["stop_loss_state"])
                    stop_loss.sync_with_market(ticker)  # Sync with current market

                    self.active_positions[ticker] = {
                        "qty": data["qty"],
                        "entry_price": data["entry_price"],
                        "entry_time": data.get("entry_time", datetime.now().isoformat()),
                        "stop_loss": stop_loss,
                        "order_ids": data.get("order_ids", {}),
                    }
                except Exception as e:
                    print(f"Error recreating position for {ticker}: {str(e)}")

            self.last_scan_date = state.get("last_scan_date")
            if "last_regime" in state:
                self.sector_system.overall_analyzer.state_labels = state["last_regime"]

            # Load new strategy state
            self.current_top_ticker = state.get("current_top_ticker")
            self.top_ticker_score = state.get("top_ticker_score", 0)
            self.runner_ups = state.get("runner_ups", [])
            self.monthly_trade_count = state.get("monthly_trade_count", 0)
            self.last_month_checked = state.get("last_month_checked")
            self.executed_tickers = set(state.get("executed_tickers", []))

            print("Previous state loaded successfully")

            # Verify positions with broker and check for new orders
            self.reconcile_positions()

    def reconcile_positions(self):
        """Verify our active positions match what's actually with the broker - LONG ONLY"""
        try:
            # Get actual positions from Alpaca - FILTER FOR LONG POSITIONS ONLY
            all_positions = trading_client.get_all_positions()
            actual_positions = [p for p in all_positions if float(p.qty) > 0]
            actual_symbols = {p.symbol for p in actual_positions}

            # CORRECTED: Get open orders without using 'status' parameter
            all_orders = trading_client.get_orders()
            open_orders = [o for o in all_orders if o.status == "open"]
            active_order_ids = {
                o.id for o in open_orders if o.order_type == "stop"
            }  # Focus on stop orders

            # Check our recorded positions
            for symbol in list(self.active_positions.keys()):
                if symbol not in actual_symbols:
                    print(f"Position {symbol} not found with broker - removing from state")
                    self.transaction_log.log(
                        action="close_position",
                        data={"ticker": symbol},
                    )
                    del self.active_positions[symbol]

            # Check for active orders not in our state
            for order_id in active_order_ids:
                if not any(
                    order_id in pos.get("order_ids", {}).values() for pos in self.active_positions.values()
                ):
                    order_details = self.state_manager.get_order_details(order_id)
                    if order_details and order_details["status"] == "filled":
                        print(f"Found filled order {order_id} not in state")
                        self.process_new_position_from_order(order_id, order_details)

            # Save cleaned state
            self.save_current_state()

        except Exception as e:
            print(f"Position reconciliation failed: {str(e)}")

    def process_new_position_from_order(self, order_id, order_details):
        """Process a position discovered from an active order - LONG ONLY"""
        try:
            symbol = order_details["symbol"]
            qty = float(order_details["filled_qty"])
            
            # Skip if quantity is negative (short position)
            if qty < 0:
                print(f"Skipping short position: {symbol}")
                return

            # Skip if we already have this position
            if symbol in self.active_positions:
                return

            print(f"Processing new position from order: {symbol} {qty} shares")

            # Get latest market data
            latest = self.get_latest_bar(symbol)
            if not latest:
                print(f"Couldn't get data for {symbol}")
                return

            # Create new position entry
            self.active_positions[symbol] = {
                "qty": qty,
                "entry_price": latest["close"],
                "entry_time": datetime.now().isoformat(),
                "order_ids": {"stop": None, "hard_stop": None},
                "stop_loss": SmartStopLoss(
                    entry_price=latest["close"],
                    atr=self.calculate_atr(symbol),
                    adx=self.get_adx(symbol),
                ),
            }

            # Log the new position
            self.transaction_log.log(
                action="place_order",
                data={
                    "ticker": symbol,
                    "qty": qty,
                    "price": latest["close"],
                    "atr": self.active_positions[symbol]["stop_loss"].initial_atr,
                    "adx": self.active_positions[symbol]["stop_loss"].base_adx,
                },
            )

            print(f"Added new position for {symbol} to state")

        except Exception as e:
            print(f"Error processing new position: {str(e)}")

    def calculate_atr(self, symbol, period=14):
        """Calculate ATR for a symbol"""
        try:
            data = self.trend_scanner.get_polygon_data(symbol)
            if data is None or len(data) < period:
                return 0.0

            df = data.copy()
            df["prev_close"] = df["Close"].shift(1)
            df["H-L"] = df["High"] - df["Low"]
            df["H-PC"] = abs(df["High"] - df["prev_close"])
            df["L-PC"] = abs(df["Low"] - df["prev_close"])
            df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
            return df["TR"].rolling(period).mean().iloc[-1]
        except Exception:
            return 0.0

    def save_current_state(self):
        """Save the current state"""
        return self.state_manager.save_state(self)

    async def run_weekly_scan(self):
        print("\nRunning weekly ticker scan...")
        results = self.trend_scanner.scan_tickers()

        if results.empty:
            print("No suitable tickers found this week")
            await self.notifier.send_system_alert("Weekly scan completed with no results")
            return pd.DataFrame()

        # Add rank column
        results["Rank"] = results["Score"].rank(ascending=False, method="min").astype(int)
        return results.sort_values("Score", ascending=False)

    async def execute_weekly_trades(self):
        self.reset_monthly_count()
        results = await self.run_weekly_scan()

        if results.empty:
            return

        # Get account buying power first
        try:
            account = trading_client.get_account()
            buying_power = float(account.buying_power)
            print(f"Current buying power: ${buying_power:.2f}")
        except Exception as e:
            print(f"Error checking buying power: {str(e)}")
            buying_power = 0

        # Get top 3 candidates
        top_3 = results.head(3)

        # Only execute if we have sufficient buying power
        if buying_power > ALLOCATION_PER_TICKER * 0.1:  # At least 10% of allocation
            # Execute top ticker if we have trades remaining
            if self.monthly_trade_count < 3:
                top_row = top_3.iloc[0]
                ticker = top_row["Ticker"]

                if ticker not in self.executed_tickers:
                    if await self.place_trade(ticker, top_row):
                        self.current_top_ticker = ticker
                        self.top_ticker_score = top_row["Score"]
                        self.monthly_trade_count += 1
                        self.executed_tickers.add(ticker)
                        print(f"Executed top ticker: {ticker} (Score: {self.top_ticker_score})")

            # Store runner-ups with their full data
            self.runner_ups = top_3.iloc[1:].to_dict("records")
            print(f"Storing {len(self.runner_ups)} runner-up tickers")

            # Send notification of scan results
            await self.notifier.send_scan_results(top_3)
        else:
            error_msg = (
                f"Insufficient buying power (${buying_power:.2f}) "
                f"for minimum trade size (${ALLOCATION_PER_TICKER * 0.1:.2f})"
            )
            print(error_msg)
            await self.notifier.send_system_alert(error_msg, is_error=True)

        self.save_current_state()
        return top_3

    def cancel_existing_orders(self, ticker):
        """Cancel any existing orders for a ticker"""
        try:
            orders = trading_client.get_orders(status='open')
            for order in orders:
                if order.symbol == ticker:
                    trading_client.cancel_order_by_id(order.id)
                    print(f"Cancelled existing order for {ticker}")
        except Exception as e:
            print(f"Error cancelling orders: {str(e)}")

    async def place_trade(self, ticker, row):
        """Place a trade for a single ticker with buying power checks - LONG ONLY"""
        try:
            # First verify we don't have a short position for this ticker
            positions = trading_client.get_all_positions()
            existing_short = any(
                p for p in positions 
                if p.symbol == ticker and float(p.qty) < 0
            )
            
            if existing_short:
                print(f"Skipping {ticker} because short position exists")
                await self.notifier.send_system_alert(
                    f"Skipping {ticker} due to existing short position", 
                    is_error=True
                )
                return False

            # Verify we can afford this trade
            account = trading_client.get_account()
            buying_power = float(account.buying_power)
            
            # Calculate position size with buying power constraints
            qty = self.calculate_position_size(row["Price"])
            estimated_cost = qty * row["Price"]
            
            if estimated_cost > buying_power:
                error_msg = (
                    f"Cannot trade {ticker}: Need ${estimated_cost:.2f} "
                    f"but only ${buying_power:.2f} available"
                )
                print(error_msg)
                await self.notifier.send_system_alert(error_msg, is_error=True)
                return False
                
            if await self.place_bracket_order(ticker, qty, row):
                self.active_positions[ticker] = {
                    "qty": qty,
                    "entry_price": row["Price"],
                    "entry_time": datetime.now().isoformat(),
                    "stop_loss": SmartStopLoss(
                        entry_price=row["Price"],
                        atr=row["ATR"],
                        adx=row["ADX"],
                    ),
                    "buying_power_used": estimated_cost,
                }
                return True
            return False
        except Exception as e:
            print(f"Error placing trade for {ticker}: {str(e)}")
            await self.notifier.send_system_alert(
                f"Trade placement failed for {ticker}: {str(e)}", 
                is_error=True
            )
            return False

    async def place_bracket_order(self, ticker, qty, row):
        """Place bracket order for LONG position only"""
        try:
            # First check buying power
            account = trading_client.get_account()
            buying_power = float(account.buying_power)
            order_cost = row["Price"] * qty
            
            if order_cost > buying_power:
                error_msg = (
                    f"Cancelled order for {ticker}: Not enough buying power\n"
                    f"Needed: ${order_cost:.2f}, Available: ${buying_power:.2f}\n"
                    f"Consider reducing ALLOCATION_PER_TICKER in config"
                )
                print(error_msg)
                await self.notifier.send_system_alert(error_msg, is_error=True)
                return False
                
            # Skip restricted securities
            if self.is_restricted_security(ticker):
                print(f"Skipping restricted security: {ticker}")
                return False
                
            # Cancel any existing orders first
            self.cancel_existing_orders(ticker)
                
            # Initialize stop loss system
            stop_system = SmartStopLoss(
                entry_price=row["Price"],
                atr=row["ATR"],
                adx=row["ADX"]
            )
            
            # Get bracket details with properly formatted prices
            bracket_details = stop_system.get_bracket_orders(row["Price"], qty)
            
            # Submit via REST API
            headers = {
                "APCA-API-KEY-ID": ALPACA_API_KEY,
                "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
            }
            
            # Create bracket order request
            order_request = {
                "symbol": ticker,
                "qty": str(qty),
                "side": "buy",
                "type": "market",
                "time_in_force": "gtc",
                "order_class": "bracket",
                "take_profit": {
                    "limit_price": str(bracket_details["take_profit"]["limit_price"])
                },
                "stop_loss": {
                    "stop_price": str(bracket_details["stop_loss"]["stop_price"]),
                    "limit_price": str(bracket_details["stop_loss"]["limit_price"])
                }
            }
            
            # Add hard stop as separate order
            hard_stop_order = {
                "symbol": ticker,
                "qty": str(qty),
                "side": "sell",
                "type": "stop",
                "time_in_force": "gtc",
                "stop_price": str(bracket_details["hard_stop"]["stop_price"])
            }
            
            # 1. Place the bracket order
            response = requests.post(
                "https://paper-api.alpaca.markets/v2/orders",
                headers=headers,
                json=order_request,
                timeout=15
            )
            
            if response.status_code != 200:
                error_msg = f"Bracket order failed for {ticker}: {response.text}"
                print(error_msg)
                await self.notifier.send_system_alert(error_msg, is_error=True)
                return False
                
            bracket_response = response.json()
            bracket_order_id = bracket_response["id"]
            print(f"Submitted bracket order for {ticker}")
            
            # 2. Place hard stop order separately
            response = requests.post(
                "https://paper-api.alpaca.markets/v2/orders",
                headers=headers,
                json=hard_stop_order,
                timeout=15
            )
            
            if response.status_code != 200:
                print(f"Hard stop order failed for {ticker}: {response.text}")
                # Cancel bracket order if hard stop fails
                trading_client.cancel_order_by_id(bracket_order_id)
                return False
                
            hard_stop_response = response.json()
            hard_stop_id = hard_stop_response["id"]
            print(f"Placed hard stop at ${bracket_details['hard_stop']['stop_price']:.2f}")
            
            # Get child order IDs from bracket order
            bracket_orders = trading_client.get_orders_by_id(bracket_order_id)
            child_orders = bracket_orders.legs
            
            # Store order IDs for later management
            order_ids = {
                "bracket": bracket_order_id,
                "take_profit": None,
                "stop_loss": None,
                "hard_stop": hard_stop_id
            }
            
            # Identify child orders
            for order in child_orders:
                if order.side == "sell" and order.order_type == "limit":
                    order_ids["take_profit"] = order.id
                elif order.side == "sell" and order.order_type == "stop_limit":
                    order_ids["stop_loss"] = order.id
            
            # Update position with order IDs
            if ticker in self.active_positions:
                self.active_positions[ticker]["order_ids"] = order_ids
            
            # Send Discord notification
            await self.notifier.send_order_notification(
                order_type="buy",
                ticker=ticker,
                qty=qty,
                price=row["Price"],
                stop_loss=bracket_details["stop_loss"]["stop_price"],
                take_profit=bracket_details["take_profit"]["limit_price"],
                hard_stop=bracket_details["hard_stop"]["stop_price"]
            )
            
            # Log before placing order
            self.transaction_log.log(
                action="place_order",
                data={
                    "ticker": ticker,
                    "qty": qty,
                    "price": row["Price"],
                    "atr": row["ATR"],
                    "adx": row["ADX"],
                    "initial_stop": row["Initial_Stop"],
                    "hard_stop": row["Hard_Stop"],
                    "take_profit": row["Take_Profit"],
                    "order_ids": order_ids,
                    "buying_power_used": order_cost,
                    "remaining_buying_power": buying_power - order_cost
                }
            )
            
            return True
            
        except Exception as e:
            error_msg = f"Order placement failed for {ticker}: {str(e)}"
            print(error_msg)
            await self.notifier.send_system_alert(error_msg, is_error=True)
            return False

    def is_restricted_security(self, ticker):
        """Check if security has trading restrictions"""
        # Skip warrants and special securities
        if any(ticker.endswith(ext) for ext in [".WS", ".WT", ".U", ".RT", ".WI"]):
            return True

        # Skip units and special symbols
        if "." in ticker or "-" in ticker or " " in ticker:
            return True

        # Check cached restricted tickers
        if ticker in DATA_CACHE.get("restricted_tickers", set()):
            return True

        return False

    async def close_position(self, ticker, reason="Hard stop triggered"):
        """Close a position immediately - LONG ONLY"""
        if ticker not in self.active_positions:
            return False

        position = self.active_positions[ticker]
        qty = position["qty"]

        try:
            # First verify we have a long position to close
            positions = trading_client.get_all_positions()
            existing_position = next(
                (p for p in positions 
                if p.symbol == ticker and float(p.qty) > 0),
                None
            )
            
            if not existing_position:
                print(f"No long position found for {ticker} to close")
                return False

            # Submit market sell order
            market_order = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            trading_client.submit_order(market_order)

            # Get exit price
            latest = get_latest_bar(ticker)
            exit_price = latest["close"] if latest else position["entry_price"]

            # Log the closure
            self.transaction_log.log(
                action="close_position",
                data={"ticker": ticker},
            )

            # Send notification
            if reason == "Hard stop triggered":
                await self.notifier.send_hard_stop_triggered(
                    ticker=ticker,
                    entry_price=position["entry_price"],
                    exit_price=exit_price,
                )

            # Remove from active positions
            del self.active_positions[ticker]
            print(f"Closed position for {ticker}: {reason}")
            return True
        except Exception as e:
            print(f"Error closing position for {ticker}: {str(e)}")
            return False

    async def update_stop_losses(self):
        # Create a queue for position updates
        update_queue = Queue()
        results = {}

        # Worker function for parallel updates
        def update_worker():
            while True:
                ticker, data = update_queue.get()
                if ticker is None:  # Poison pill
                    break

                try:
                    # Get latest price data with retries
                    latest = self.get_latest_bar(ticker)
                    if not latest:
                        print(f"No data for {ticker}, skipping update")
                        update_queue.task_done()
                        continue

                    # Handle missing ADX
                    if "adx" not in latest:
                        latest["adx"] = self.get_adx(ticker) or data["stop_loss"].base_adx

                    # Handle missing volume
                    if "volume" not in latest:
                        latest["volume"] = 1e6

                    # Add average volume
                    latest["avg_volume"] = self.get_avg_volume(ticker) or 1e6

                    # Get current stop before update
                    old_stop = data["stop_loss"].current_stop
                    old_hard_stop = data["stop_loss"].hard_stop

                    # Update stop loss
                    new_stop = data["stop_loss"].update(latest)

                    # Check if hard stop was triggered
                    if data["stop_loss"].hard_stop_triggered:
                        results[ticker] = ("close", "Hard stop triggered")
                        update_queue.task_done()
                        continue

                    # Log the stop update
                    log_data = {
                        "ticker": ticker,
                        "old_stop": old_stop,
                        "new_stop": new_stop,
                        "highest_high": data["stop_loss"].highest_high,
                        "activated": data["stop_loss"].activated,
                    }

                    # Only include hard stop if it changed
                    if data["stop_loss"].hard_stop != old_hard_stop:
                        log_data["hard_stop"] = data["stop_loss"].hard_stop

                    self.transaction_log.log(
                        action="update_stop",
                        data=log_data,
                    )

                    # Store result for later processing
                    results[ticker] = ("update", data, latest, old_stop, new_stop)

                except Exception as e:
                    print(f"Stop update failed for {ticker}: {str(e)}")
                    results[ticker] = ("error", e)
                finally:
                    update_queue.task_done()

        # Create worker threads
        num_workers = min(10, max(1, len(self.active_positions)))
        workers = []
        for _ in range(num_workers):
            t = Thread(target=update_worker)
            t.daemon = True
            t.start()
            workers.append(t)

        # Add positions to queue
        for ticker, data in list(self.active_positions.items()).copy():
            update_queue.put((ticker, data))

        # Wait for completion
        update_queue.join()

        # Signal workers to exit
        for _ in range(num_workers):
            update_queue.put((None, None))

        # Process results
        for ticker, result in results.items():
            action = result[0]

            if action == "close":
                await self.close_position(ticker, reason=result[1])
            elif action == "update":
                data, latest, old_stop, new_stop = result[1:]

                # Only send notification if stop actually changed
                if new_stop != old_stop:
                    print(f"Updated stop for {ticker}: {new_stop:.2f}")

                    # Determine reason for stop adjustment
                    reason = "Price moved in favorable direction"
                    if data["stop_loss"].activated:
                        reason = "Trailing stop activated"

                    # Send Discord notification
                    await self.notifier.send_stop_loss_update(
                        ticker=ticker,
                        old_stop=old_stop,
                        new_stop=new_stop,
                        reason=reason,
                    )

                # Atomically replace stop orders
                await self.replace_stop_orders(
                    ticker=ticker,
                    qty=data["qty"],
                    new_stop=new_stop,
                    new_hard_stop=data["stop_loss"].hard_stop
                )

                # Send position update
                await self.notifier.send_position_update(
                    ticker=ticker,
                    entry_price=data["entry_price"],
                    current_price=latest["close"],
                    stop_loss=new_stop,
                    pnl=(latest["close"] - data["entry_price"]) * data["qty"],
                )
            elif action == "error":
                error = result[1]
                await self.notifier.send_system_alert(
                    f"Stop update failed for {ticker}: {str(error)}", is_error=True
                )
                # Remove problematic position
                if ticker in self.active_positions:
                    del self.active_positions[ticker]

        # Save state after updating all stops
        self.save_current_state()

    async def replace_stop_orders(self, ticker, qty, new_stop, new_hard_stop):
        """Atomically replace stop orders by placing new ones first"""
        try:
            # Cancel any existing orders before placing new ones
            self.cancel_existing_orders(ticker)
            
            # Place new orders FIRST to minimize exposure
            new_stop_id = self.place_stop_order(ticker, qty, new_stop)
            new_hard_stop_id = self.place_hard_stop_order(ticker, qty, new_hard_stop)
            
            # Only cancel old orders after new ones are placed
            if ticker in self.active_positions:
                position = self.active_positions[ticker]
                old_stop_id = position["order_ids"].get("stop")
                old_hard_stop_id = position["order_ids"].get("hard_stop")
                
                # Update position with new order IDs
                position["order_ids"] = {
                    "stop": new_stop_id,
                    "hard_stop": new_hard_stop_id
                }
                
                # Cancel old orders
                if old_stop_id:
                    self.cancel_order_by_id(old_stop_id)
                if old_hard_stop_id:
                    self.cancel_order_by_id(old_hard_stop_id)
                    
            return True
        except Exception as e:
            print(f"Failed to replace stops for {ticker}: {str(e)}")
            # Attempt to cancel any partially created orders
            if new_stop_id:
                self.cancel_order_by_id(new_stop_id)
            if new_hard_stop_id:
                self.cancel_order_by_id(new_hard_stop_id)
            return False

    def cancel_order_by_id(self, order_id):
        """Cancel an order by its ID"""
        try:
            trading_client.cancel_order_by_id(order_id)
            print(f"Cancelled order {order_id}")
            return True
        except APIError as e:
            if e.status_code == 404:  # Order not found
                print(f"Order {order_id} not found - may have been filled")
                return False
            print(f"Failed to cancel order {order_id}: {str(e)}")
            return False
        except Exception as e:
            print(f"Error canceling order {order_id}: {str(e)}")
            return False

    def place_stop_order(self, ticker, qty, stop_price, retries=3):
        """Place a stop order with retry logic"""
        for attempt in range(retries):
            try:
                normalized_stop = self.normalize_price(stop_price)
                stop_order = StopOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=OrderSide.SELL,
                    stop_price=normalized_stop,
                    time_in_force=TimeInForce.GTC,
                )
                response = trading_client.submit_order(stop_order)
                print(f"Placed new STOP at ${normalized_stop:.4f} for {ticker}")
                return response.id  # Return order ID
            except APIError as e:
                if e.status_code == 429:  # Rate limit
                    wait = int(e.headers.get('Retry-After', 30))
                    print(f"Rate limited. Waiting {wait} seconds")
                    time.sleep(wait)
                else:
                    raise
            except Exception as e:
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Retrying stop order for {ticker} in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
        return None

    def place_hard_stop_order(self, ticker, qty, stop_price, retries=3):
        """Place a hard stop order with retry logic"""
        for attempt in range(retries):
            try:
                normalized_stop = self.normalize_price(stop_price)
                stop_order = StopOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=OrderSide.SELL,
                    stop_price=normalized_stop,
                    time_in_force=TimeInForce.GTC,
                )
                response = trading_client.submit_order(stop_order)
                print(f"Placed new HARD STOP at ${normalized_stop:.4f} for {ticker}")
                return response.id  # Return order ID
            except APIError as e:
                if e.status_code == 429:  # Rate limit
                    wait = int(e.headers.get('Retry-After', 30))
                    print(f"Rate limited. Waiting {wait} seconds")
                    time.sleep(wait)
                else:
                    raise
            except Exception as e:
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Retrying hard stop for {ticker} in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
        return None

    def get_latest_bar(self, ticker, retries=3):
        return get_latest_bar(ticker, retries)

    def get_adx(self, ticker, retries=2):
        for i in range(retries):
            try:
                url = f"https://api.polygon.io/v1/indicators/adx/{ticker}"
                params = {"timespan": "day", "window": 14, "apiKey": POLYGON_API_KEY}
                response = requests.get(url, params=params, timeout=5)

                if response.status_code == 200:
                    data = response.json()
                    if "results" in data and "values" in data["results"]:
                        return data["results"]["values"][0]["value"]
            except Exception:
                time.sleep(0.5 * (i + 1))
        return None

    def get_avg_volume(self, ticker, days=30):
        try:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
            params = {"adjusted": "true", "apiKey": POLYGON_API_KEY}
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if "results" in data and data["results"]:
                    volumes = [r["v"] for r in data["results"]]
                    return sum(volumes) / len(volumes)
        except Exception:
            return None
        return None

    def calculate_position_size(self, price):
        """Calculate position size based on available buying power - LONG ONLY"""
        try:
            # Get account information
            account = trading_client.get_account()
            buying_power = float(account.buying_power)
            
            # Get existing LONG positions market value
            positions = trading_client.get_all_positions()
            long_market_value = sum(
                float(p.market_value) for p in positions if float(p.qty) > 0
            )
            
            # Calculate available buying power for new positions
            available_bp = buying_power - long_market_value
            
            # Calculate maximum possible allocation
            max_allocation = min(
                ALLOCATION_PER_TICKER, 
                available_bp * MAX_PORTFOLIO_ALLOCATION
            )
            
            # Calculate position size
            position_size = max(1, int(max_allocation / price))
            
            # Verify we're not exceeding buying power
            estimated_cost = position_size * price
            if estimated_cost > available_bp:
                # Reduce position size if needed
                position_size = max(1, int(available_bp * MAX_PORTFOLIO_ALLOCATION / price))
                print(f"Reduced position size to {position_size} due to buying power constraints")
            
            return position_size
            
        except Exception as e:
            print(f"Error calculating position size: {str(e)}")
            # Fallback to basic calculation
            return max(1, int(ALLOCATION_PER_TICKER / price))

    async def check_runner_ups(self):
        """Check if we need to fill positions or monitor runner-ups"""
        # First check if we need to fill empty positions
        if not self.active_positions and self.monthly_trade_count < 3:
            print("No active positions - scanning for new candidates")
            await self.scan_and_fill_empty_positions()
            return

        # Then proceed with runner-up monitoring
        if not self.runner_ups or self.monthly_trade_count >= 3:
            return

        print("Checking runner-up tickers for improvement...")
        new_runner_ups = []
        executed_count = 0

        for candidate in self.runner_ups:
            try:
                ticker = candidate["Ticker"]

                # Skip if already executed
                if ticker in self.executed_tickers:
                    continue

                # Get updated data
                updated_data = self.get_updated_ticker_data(ticker)
                if not updated_data:
                    new_runner_ups.append(candidate)  # Keep original data
                    continue

                # Recalculate score
                new_score = self.calculate_ticker_score(updated_data)

                # Check if score matches/exceeds original top score
                if new_score >= self.top_ticker_score:
                    # Place trade if not already executed and within limits
                    if self.monthly_trade_count < 3 and ticker not in self.executed_tickers:
                        if await self.place_trade(ticker, updated_data):
                            print(
                                f"Executed runner-up: {ticker} (New: {new_score} vs Original: {self.top_ticker_score})"
                            )
                            self.monthly_trade_count += 1
                            self.executed_tickers.add(ticker)
                            executed_count += 1
                            continue  # Skip adding to new runner-ups

                # Update candidate with new score and keep for next check
                candidate["Score"] = new_score
                candidate.update(updated_data)
                new_runner_ups.append(candidate)

            except Exception as e:
                print(f"Error checking runner-up {ticker}: {str(e)}")
                new_runner_ups.append(candidate)  # Keep original on error

        # Update runner-up list
        self.runner_ups = new_runner_ups

        if executed_count > 0:
            print(f"Executed {executed_count} runner-up tickers")
            await self.notifier.send_system_alert(
                f"Executed {executed_count} runner-up tickers with improved scores"
            )
            self.save_current_state()

    async def scan_and_fill_empty_positions(self):
        """Scan for new candidates when there are no active positions"""
        print("Running fill scan for empty positions...")
        # Create a new scanner with reduced ticker count for efficiency
        fill_scanner = PolygonTrendScanner(MAX_TICKERS_TO_SCAN)
        
        # Add progress bar for scanning
        print("Scanning for new position candidates...")
        
        # Initialize progress bar
        total_tickers = len(fill_scanner.tickers)
        progress_bar = tqdm(total=total_tickers, desc="Scanning Tickers", unit="ticker")
        
        results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(fill_scanner.process_ticker, ticker): ticker 
                for ticker in fill_scanner.tickers
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"Error processing {ticker}: {str(e)}")
                finally:
                    # Update progress bar after each ticker is processed
                    progress_bar.update(1)
        
        # Close progress bar
        progress_bar.close()
        
        # Create DataFrame and sort
        if results:
            df_results = pd.DataFrame(results)
            df_results["Rank"] = df_results["Score"].rank(ascending=False, method="min").astype(int)
            results_df = df_results.sort_values("Score", ascending=False).reset_index(drop=True)
        else:
            results_df = pd.DataFrame()
        
        if results_df.empty:
            print("No suitable tickers found in fill scan")
            return
            
        # Get top candidate
        top_row = results_df.iloc[0]
        ticker = top_row["Ticker"]
        
        if await self.place_trade(ticker, top_row):
            print(f"Filled empty position with: {ticker}")
            self.current_top_ticker = ticker
            self.top_ticker_score = top_row["Score"]
            self.monthly_trade_count += 1
            self.executed_tickers.add(ticker)
            
            # Store runner-ups from this scan
            self.runner_ups = results_df.iloc[1:3].to_dict("records")
            print(f"Stored {len(self.runner_ups)} new runner-ups")
            
            await self.notifier.send_system_alert(
                f"Filled empty position with {ticker} (Score: {top_row['Score']})"
            )
            self.save_current_state()

    def get_updated_ticker_data(self, ticker):
        """Get updated data for a ticker"""
        data = self.trend_scanner.get_polygon_data(ticker)
        if data is None:
            return None

        indicators = self.trend_scanner.calculate_indicators(data)
        if indicators is None:
            return None

        return {
            "Ticker": ticker,
            "Score": 0,  # Will be calculated separately
            "Price": indicators["Close"],
            "ATR": indicators["ATR"],
            "ADX": indicators["ADX"],
            "SMA_50": indicators["SMA_50"],
            "SMA_200": indicators["SMA_200"],
            "Volume": indicators["Volume"],
            "10D_Change": indicators["10D_Change"],
            "Initial_Stop": indicators["Close"]
            - (self.trend_scanner.base_multiplier * indicators["ATR"]),
        }

    def calculate_ticker_score(self, data):
        """Calculate score using same method as scanner"""
        try:
            # Same calculation as in PolygonTrendScanner.scan_tickers()
            adx_component = min(40, (data["ADX"] / 50) * 40)
            sma50_component = min(
                30, max(0, ((data["Price"] - data["SMA_50"]) / data["SMA_50"]) * 100) * 0.3
            )
            volume_component = min(20, math.log10(max(1, data["Volume"] / 10000)))
            momentum_component = min(10, max(0, data["10D_Change"]))

            return min(
                100, adx_component + sma50_component + volume_component + momentum_component
            )
        except Exception:
            return 0

    async def run_market_regime_analysis(self):
        try:
            print("\nRunning market regime analysis...")
            tickers = get_all_tickers()
            self.sector_system.map_tickers_to_sectors(tickers)
            self.sector_system.calculate_sector_weights()
            self.sector_system.build_sector_composites()

            market_index = self.sector_system.overall_analyzer.prepare_market_data(tickers)
            market_results = self.sector_system.overall_analyzer.analyze_regime(market_index)

            current_regime = market_results["regimes"][-1]
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

    async def initialize(self):
        """Initialize async components"""
        await self.notifier.send_system_alert("Trading system starting up...")

    async def shutdown(self):
        """Clean up resources"""
        await self.notifier.send_system_alert("Trading system shutting down...")

    async def monitor_and_update(self):
        print("Starting Trading System...")
        print("Press Ctrl+C to stop")

        # Initial market regime analysis
        current_regime = await self.run_market_regime_analysis()
        regime_message = f"Initialization complete. Current regime: {current_regime}"
        print(f"\n{regime_message}")
        await self.notifier.send_system_alert(regime_message)

        print("System now monitoring for scheduled tasks...")
        last_state_save = datetime.now()
        last_runner_up_check = None

        while True:
            try:
                current_time = datetime.now()
                weekday = current_time.weekday()
                hour = current_time.hour
                minute = current_time.minute

                # Save state every 5 minutes
                if (current_time - last_state_save).total_seconds() > AUTO_SAVE_INTERVAL:
                    self.save_current_state()
                    last_state_save = current_time
                    print("System state saved")

                # Print status message
                print(f"\n[{current_time.strftime('%Y-%m-%d %H:%M')}] Checking tasks... ", end="")
                task_performed = False

                # Run weekly trades on Sundays at 4 PM
                if weekday == 6 and hour == 16 and minute < 10:  # Run once at 4:00 PM
                    if not self.last_scan_date or (current_time - self.last_scan_date).days >= 7:
                        print("Executing weekly trades")
                        await self.notifier.send_system_alert("Executing weekly trades")
                        await self.execute_weekly_trades()
                        self.last_scan_date = current_time
                        task_performed = True

                # Update stops hourly during market hours (9AM-4PM)
                if 9 <= hour <= 16 and not task_performed:
                    if self.active_positions:
                        print("Updating stop losses")
                        await self.update_stop_losses()
                        task_performed = True

                # Check positions and runner-ups every 2 hours during market hours
                if (
                    9 <= hour <= 16
                    and not task_performed
                    and (
                        last_runner_up_check is None
                        or (current_time - last_runner_up_check).total_seconds() > 7200
                    )
                ):  # 2 hours
                    print("Checking positions and runner-ups")
                    await self.check_runner_ups()  # This handles both empty positions and runner-ups
                    last_runner_up_check = current_time
                    task_performed = True

                # Run market regime analysis at 5 AM daily
                if hour == 5 and minute < 10 and not task_performed:
                    print("Running market regime analysis")
                    current_regime = await self.run_market_regime_analysis()
                    await self.notifier.send_system_alert(
                        f"Market regime analysis complete. Current regime: {current_regime}"
                    )
                    task_performed = True

                # Emergency fill check if we somehow have no positions during market hours
                if (
                    9 <= hour <= 16
                    and not self.active_positions
                    and self.monthly_trade_count < 3
                    and not task_performed
                ):
                    print("Emergency fill check - no active positions")
                    await self.scan_and_fill_empty_positions()
                    task_performed = True

                if not task_performed:
                    print("No scheduled tasks")

                # Calculate sleep time until next check
                now = datetime.now()
                if 9 <= now.hour <= 16:  # Market hours - check hourly
                    sleep_time = 3600  # 1 hour
                else:  # Outside market hours - check every 4 hours
                    sleep_time = 14400  # 4 hours

                next_check = now + timedelta(seconds=sleep_time)
                print(f"Next check at {next_check.strftime('%H:%M')}")
                await asyncio.sleep(sleep_time)

            except KeyboardInterrupt:
                print("\nExiting trading system...")
                # Save state before exiting
                self.save_current_state()
                break
            except Exception as e:
                print(f"System error: {str(e)}")
                await self.notifier.send_system_alert(f"System error: {str(e)}", is_error=True)
                # Save state on error
                self.save_current_state()
                await asyncio.sleep(300)  # Wait 5 minutes before retrying


# ===============
# Main Execution
# ===============
async def main():
    trading_system = TradingSystem()
    await trading_system.initialize()

    try:
        await trading_system.monitor_and_update()
    except Exception as e:
        await trading_system.notifier.send_system_alert(f"System crashed: {str(e)}", is_error=True)
    finally:
        await trading_system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())