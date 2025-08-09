import numpy as np
if not hasattr(np, 'NaN'):
    np.isNaN = np.isnan  # Create alias for older numpy versions
import sys
import math
import time
import threading
import queue
import json
import requests
import pytz
import warnings
import os
import logging
import pandas as pd
from datetime import datetime, timedelta, timezone as tz
import pandas_ta as ta
from websocket import create_connection
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QPushButton, 
    QTabWidget, QTextEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QBrush, QFont
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import concurrent.futures
import talib
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.exceptions import ConvergenceWarning
from concurrent.futures import ThreadPoolExecutor
import random

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure logging for MarketRegimeAnalyzer
analyzer_logger = logging.getLogger("MarketRegimeAnalyzer")
analyzer_logger.setLevel(logging.DEBUG)
analyzer_fh = logging.FileHandler('market_regime_analyzer.log')
analyzer_fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
analyzer_fh.setFormatter(formatter)
analyzer_logger.addHandler(analyzer_fh)

# ======================== MARKET REGIME CODE ======================== #
class MarketRegimeAnalyzer:
    def __init__(self, n_states=4, polygon_api_key="OZzn0oK0H2yG6rpIvVhGfgXgnUTrL31z", testing_mode=False):
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=20000,       # Significantly increased iterations
            tol=1e-3,           # More tolerant convergence threshold
            init_params="se",
            params="stmc",
            random_state=42,
            min_covar=0.2,      # Increased regularization
            implementation='scaling',  # Logarithmic implementation for stability
            verbose=True        # Enable verbose output
        )
        self.state_labels = {}
        self.feature_scaler = RobustScaler()  # More robust to outliers
        self.polygon_api_key = polygon_api_key
        self.data_cache = {}
        os.makedirs("data_cache", exist_ok=True)
        self.logger = analyzer_logger
        self.testing_mode = testing_mode

    def prepare_market_data(self, tickers, sample_size=100, min_days_data=200):
        if self.testing_mode:
            return self.generate_test_market_data()
            
        prices_data = []
        valid_tickers = []
        mcaps = {}  # For market cap weighting

        # First pass: Get market caps for weighting
        self.logger.info("Collecting market caps...")
        for ticker in tickers[:sample_size]:
            mcaps[ticker] = self.get_market_cap(ticker) or 1

        total_mcap = sum(mcaps.values())
        
        # Second pass: Get price data with weighting
        self.logger.info("Building market composite...")
        for symbol in tickers[:sample_size]:
            prices = self.fetch_stock_data(symbol)
            if prices is not None and len(prices) >= min_days_data:
                weight = mcaps.get(symbol, 1) / total_mcap
                prices_data.append(prices * weight)  # Apply market cap weighting
                valid_tickers.append(symbol)

        if not prices_data:
            raise ValueError("Insufficient data to create market composite")
            
        # Align and combine data
        composite = pd.concat(prices_data, axis=1)
        composite.columns = valid_tickers
        composite = composite.fillna(method='ffill').fillna(method='bfill')
        return composite.sum(axis=1).dropna()  # Sum of weighted prices

    def analyze_regime(self, index_data, n_states=None):
        if n_states is None:
            n_states = self.model.n_components

        # Log feature preparation
        self.logger.info(f"Preparing features for regime analysis with {n_states} states...")
        
        # Calculate advanced features (using only closing prices)
        log_returns = np.log(index_data).diff().dropna()
        features = pd.DataFrame({
            "returns": log_returns,
            "volatility": log_returns.rolling(21).std(),
            "momentum": log_returns.rolling(14).mean(),
            "rsi": talib.RSI(index_data, timeperiod=14).dropna(),
            "macd": talib.MACD(index_data)[0].dropna(),
        }).dropna()

        if len(features) < 100:  # Increased minimum for better model stability
            self.logger.warning(f"Insufficient feature data: only {len(features)} samples")
            raise ValueError(f"Only {len(features)} days of feature data")
            
        # Robust scaling
        self.logger.info("Scaling features...")
        scaled_features = self.feature_scaler.fit_transform(features)
        
        # Clip extreme values to prevent numerical instability
        scaled_features = np.clip(scaled_features, -10, 10)
        
        # FIXED: Ensure we're using scalar values for logging
        min_val = np.min(scaled_features)
        max_val = np.max(scaled_features)
        self.logger.info(f"Feature range after scaling/clipping: Min={min_val:.4f}, Max={max_val:.4f}")

        # Create and fit model with multiple initializations
        best_model = None
        best_score = -np.inf
        convergence_success = False
        
        self.logger.info(f"Starting HMM fitting with {n_states} states (3 attempts)...")
        
        # Try 3 different random initializations
        for i in range(3):
            self.logger.info(f"Attempt {i+1}/3 with random_state={42+i}")
            model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=20000,       # Significantly increased iterations
                tol=1e-3,           # More tolerant convergence threshold
                init_params="se",
                params="stmc",
                random_state=42 + i,  # Different seed each try
                min_covar=0.2,      # Increased regularization
                implementation='scaling',  # Changed to scaling implementation
                verbose=True
            )

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.logger.info(f"Fitting HMM model (attempt {i+1})...")
                    model.fit(scaled_features)
                    
                # Get convergence info
                converged = model.monitor_.converged
                iterations = model.monitor_.iter
                history = model.monitor_.history
                
                # Convert history to list if it's a special type
                try:
                    history_list = list(history)
                except TypeError:
                    history_list = [history] if isinstance(history, (int, float)) else []
                
                final_log_likelihood = history_list[-1] if history_list else float('-inf')
                
                self.logger.info(
                    f"Attempt {i+1}: Converged={converged}, Iterations={iterations}, "
                    f"Final Log-Likelihood={final_log_likelihood:.4f}"
                )
                
                # Log convergence history if available
                if history_list:
                    # Log first, last, and min/max values
                    self.logger.debug(f"Log-likelihood history (first 5): {history_list[:5]}")
                    self.logger.debug(f"Log-likelihood history (last 5): {history_list[-5:]}")
                    
                    # FIXED: Ensure scalar values for logging
                    min_hist = min(history_list)
                    max_hist = max(history_list)
                    delta_hist = history_list[-1] - history_list[0]
                    self.logger.debug(
                        f"Log-likelihood range: Min={min_hist:.4f}, Max={max_hist:.4f}, "
                        f"Delta={delta_hist:.4f}"
                    )
                    
                    # Log convergence issues
                    if any(np.nan(x) for x in history_list) or any(np.isinf(x) for x in history_list):
                        self.logger.warning("History contains NaN/Inf values!")
                        
                    decreasing_count = 0
                    for i in range(1, len(history_list)):
                        if history_list[i] < history_list[i-1]:
                            decreasing_count += 1
                    if decreasing_count > 0:
                        self.logger.warning(f"Log-likelihood decreased {decreasing_count} times during fitting")

                # Check if model converged
                if converged:
                    convergence_success = True
                    model_score = model.score(scaled_features)
                    self.logger.info(f"Converged model score: {model_score:.4f}")
                    
                    if model_score > best_score:
                        best_model = model
                        best_score = model_score
                        self.logger.info(f"New best model found (score={model_score:.4f})")
                # Allow small negative deltas
                elif best_model is not None:
                    model_score = model.score(scaled_features)
                    score_diff = best_score - model_score
                    
                    if score_diff < 1e-4:
                        convergence_success = True
                        best_model = model
                        best_score = model_score
                        self.logger.info(
                            f"Accepting model with small delta ({score_diff:.6f}), "
                            f"new score={model_score:.4f}"
                        )
                    else:
                        self.logger.warning(
                            f"Model not converged and score too low (delta={score_diff:.6f})"
                        )
            except Exception as e:
                self.logger.error(f"HMM fitting attempt {i+1} failed: {str(e)}", exc_info=True)
        
        if convergence_success:
            self.logger.info(f"Best model score: {best_score:.4f}")
            model = best_model

        # Label states based on volatility and returns
        state_stats = []
        for i in range(model.n_components):
            # FIXED: Ensure scalar values
            state_return = float(model.means_[i][0])
            state_vol = float(model.means_[i][1])
            state_stats.append((i, state_return, state_vol))
            
        # Log state characteristics before sorting
        self.logger.info("State characteristics before sorting:")
        for i, (state_idx, ret, vol) in enumerate(state_stats):
            self.logger.info(f"State {i}: Return={ret:.6f}, Volatility={vol:.6f}")
            
        # Sort by return then volatility
        state_stats.sort(key=lambda x: (x[1], -x[2]))
        
        # Create labels based on sorted states
        if n_states == 3:
            state_labels = {state_stats[0][0]: "Bear", state_stats[1][0]: "Neutral", state_stats[2][0]: "Bull"}
        elif n_states == 4:
            state_labels = {
                state_stats[0][0]: "Severe Bear",
                state_stats[1][0]: "Mild Bear",
                state_stats[2][0]: "Mild Bull",
                state_stats[3][0]: "Strong Bull",
            }
        else:
            state_labels = {i: f"State {i+1}" for i in range(n_states)}
        
        # Log final state labeling
        self.logger.info("Final state labeling:")
        for state_idx, label in state_labels.items():
            # Find matching state stats
            ret, vol = next((s_ret, s_vol) for s_idx, s_ret, s_vol in state_stats if s_idx == state_idx)
            self.logger.info(f"State {state_idx}: {label} (Return={ret:.6f}, Volatility={vol:.6f})")

        # Predict regimes
        states = model.predict(scaled_features)
        state_probs = model.predict_proba(scaled_features)
        
        # Calculate state durations
        state_durations = self.calculate_state_durations(states)
        self.logger.info(f"State durations: {state_durations}")

        return {
            "model": model,
            "regimes": [state_labels[s] for s in states],
            "probabilities": state_probs,
            "features": features,
            "index_data": index_data[features.index[0] :],
            "state_labels": state_labels,
            "state_durations": state_durations
        }
        
    def calculate_state_durations(self, states):
        """Calculate average duration per state"""
        durations = {state: [] for state in set(states)}
        current_state = states[0]
        current_duration = 1
        
        for i in range(1, len(states)):
            if states[i] == current_state:
                current_duration += 1
            else:
                durations[current_state].append(current_duration)
                current_state = states[i]
                current_duration = 1
                
        durations[current_state].append(current_duration)
        
        # Calculate average durations
        avg_durations = {}
        for state, durs in durations.items():
            avg_durations[state] = sum(durs) / len(durs) if durs else 0
            
        return avg_durations

    def fetch_stock_data(self, symbol, days=365):
        # Check cache first
        cache_file = f"data_cache/{symbol}_{days}.pkl"
        if os.path.exists(cache_file):
            return pd.read_pickle(cache_file)
            
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.polygon_api_key,
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            
            # Handle rate limits
            if response.status_code == 429:
                time.sleep(0.5)
                return self.fetch_stock_data(symbol, days)
                
            if response.status_code != 200:
                return None

            data = response.json()
            if 'results' not in data or not data['results']:
                return None
                
            df = pd.DataFrame(data['results'])
            df["date"] = pd.to_datetime(df["t"], unit="ms")
            result = df.set_index("date")["c"]
            
            # Cache result
            result.to_pickle(cache_file)
            return result
            
        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout fetching {symbol}, skipping")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching {symbol}: {str(e)}")
            return None
            
    def get_market_cap(self, symbol):
        cache_file = f"data_cache/mcap_{symbol}.pkl"
        if os.path.exists(cache_file):
            return pd.read_pickle(cache_file)
            
        url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
        params = {"apiKey": self.polygon_api_key}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 429:
                time.sleep(0.5)
                return self.get_market_cap(symbol)
                
            if response.status_code == 200:
                data = response.json().get("results", {})
                mcap = data.get("market_cap", 0)
                pd.Series([mcap]).to_pickle(cache_file)
                return mcap
            return 0
        except Exception as e:
            self.logger.error(f"Error getting market cap for {symbol}: {str(e)}")
            return 0
            
    def generate_test_market_data(self):
        """Generate simulated market data for testing"""
        start_date = datetime.now() - timedelta(days=365)
        dates = pd.date_range(start_date, datetime.now(), freq='D')
        
        # Create a random walk with trends
        prices = [100]
        for i in range(1, len(dates)):
            change = random.uniform(-0.02, 0.03)
            prices.append(prices[-1] * (1 + change))
            
        return pd.Series(prices, index=dates)


# ======================== SECTOR REGIME SYSTEM ======================== #
class SectorRegimeSystem:
    def __init__(self, polygon_api_key="OZzn0oK0H2yG6rpIvVhGfgXgnUTrL31z", testing_mode=False):
        self.sector_mappings = {}
        self.sector_composites = {}
        self.sector_analyzers = {}
        self.overall_analyzer = MarketRegimeAnalyzer(polygon_api_key=polygon_api_key, testing_mode=testing_mode)
        self.sector_weights = {}
        self.sector_scores = {}
        self.polygon_api_key = polygon_api_key
        self.current_regime = None
        self.logger = logging.getLogger("SectorRegimeSystem")
        self.logger.setLevel(logging.INFO)
        self.testing_mode = testing_mode

    def map_tickers_to_sectors(self, tickers):
        if self.testing_mode:
            return self.generate_test_sector_mappings(tickers)
            
        self.sector_mappings = {}
        cache_file = "data_cache/sector_mappings.pkl"
        
        # Try loading from cache
        if os.path.exists(cache_file):
            try:
                self.sector_mappings = pd.read_pickle(cache_file)
                return self.sector_mappings
            except:
                pass

        # Enhanced sector mapping function
        def map_single_ticker(symbol):
            cache_file = f"data_cache/sector_{symbol}.pkl"
            if os.path.exists(cache_file):
                return symbol, pd.read_pickle(cache_file)
                
            url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
            params = {"apiKey": self.polygon_api_key}
            try:
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 429:
                    time.sleep(0.5)
                    return map_single_ticker(symbol)
                    
                if response.status_code == 200:
                    data = response.json().get("results", {})
                    
                    # Priority-based sector mapping
                    sector = data.get("sic_description", "")
                    if not sector or sector == "Unknown":
                        sector = data.get("sector", "")
                    if not sector or sector == "Unknown":
                        sector = data.get("industry", "")
                    if not sector or sector == "Unknown":
                        sector = data.get("primary_exchange", "Unknown")
                    
                    # Clean sector name
                    sector = self.clean_sector_name(sector)
                    
                    # Cache result
                    pd.Series([sector]).to_pickle(cache_file)
                    return symbol, sector
            except Exception as e:
                self.logger.error(f"Sector mapping failed for {symbol}: {str(e)}")
            return symbol, "Unknown"

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(map_single_ticker, symbol): symbol for symbol in tickers}

            for future in concurrent.futures.as_completed(futures):
                try:
                    symbol, sector = future.result()
                    if sector != "Unknown":
                        self.sector_mappings.setdefault(sector, []).append(symbol)
                except Exception as e:
                    self.logger.error(f"Error processing sector mapping: {str(e)}")

        # Remove unknown sectors and small sectors
        self.sector_mappings = {
            k: v for k, v in self.sector_mappings.items() 
            if k != "Unknown" and len(v) > 10
        }
        
        # Save to cache
        pd.to_pickle(self.sector_mappings, cache_file)
        return self.sector_mappings

    def clean_sector_name(self, sector):
        """Normalize and clean sector names"""
        if not sector or sector == "Unknown":
            return "Unknown"
            
        # Remove exchange prefixes (like XNAS, XNYS, etc.)
        if sector.startswith("X"):
            return "Unknown"
            
        # Standardize common sector names
        sector_lower = sector.lower()
        if "tech" in sector_lower or "software" in sector_lower or "hardware" in sector_lower:
            return "Technology"
        if "health" in sector_lower or "medical" in sector_lower or "pharma" in sector_lower:
            return "Healthcare"
        if "financial" in sector_lower or "finance" in sector_lower or "bank" in sector_lower or "credit" in sector_lower:
            return "Financial Services"
        if "consumer" in sector_lower or "retail" in sector_lower:
            if "non-cyclical" in sector_lower or "staples" in sector_lower:
                return "Consumer Defensive"
            return "Consumer Cyclical"
        if "industrial" in sector_lower or "machinery" in sector_lower:
            return "Industrials"
        if "energy" in sector_lower or "oil" in sector_lower or "gas" in sector_lower:
            return "Energy"
        if "utility" in sector_lower:
            return "Utilities"
        if "communication" in sector_lower or "telecom" in sector_lower:
            return "Communication Services"
        if "material" in sector_lower or "chemical" in sector_lower:
            return "Basic Materials"
        if "real estate" in sector_lower or "reit" in sector_lower:
            return "Real Estate"
        return sector  # Return original if no match

    def calculate_sector_weights(self):
        if self.testing_mode:
            return self.generate_test_sector_weights()
            
        total_mcap = 0
        sector_mcaps = {}
        
        # Use cached data if available
        self.logger.info("Calculating sector weights...")
        for sector, tickers in self.sector_mappings.items():
            sector_mcap = 0
            # Get market caps for first 30 tickers in sector
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_ticker = {
                    executor.submit(self.overall_analyzer.get_market_cap, symbol): symbol
                    for symbol in tickers[:30]
                }
                for future in concurrent.futures.as_completed(future_to_ticker):
                    symbol = future_to_ticker[future]
                    try:
                        mcap = future.result()
                        sector_mcap += mcap if mcap else 0
                    except:
                        pass

            sector_mcaps[sector] = sector_mcap
            total_mcap += sector_mcap
        
        self.sector_weights = {
            sector: mcap / total_mcap if total_mcap > 0 else 1 / len(sector_mcaps)
            for sector, mcap in sector_mcaps.items()
        }
        return self.sector_weights

    def build_sector_composites(self, sample_size=30):
        if self.testing_mode:
            return self.generate_test_sector_composites()
            
        self.sector_composites = {}
        cache_file = "data_cache/sector_composites.pkl"
        
        # Try loading from cache
        if os.path.exists(cache_file):
            try:
                self.sector_composites = pd.read_pickle(cache_file)
                return self.sector_composites
            except:
                pass

        # Process each sector
        self.logger.info("Building sector composites...")
        for sector, tickers in self.sector_mappings.items():
            prices_data = []
            mcaps = {}
            
            # Get market caps first
            for symbol in tickers[:sample_size]:
                mcaps[symbol] = self.overall_analyzer.get_market_cap(symbol) or 1
            total_mcap = sum(mcaps.values())
            
            # Get price data with weighting
            for symbol in tickers[:sample_size]:
                try:
                    prices = self.overall_analyzer.fetch_stock_data(symbol)
                    if prices is not None and len(prices) >= 200:
                        weight = mcaps[symbol] / total_mcap
                        prices_data.append(prices * weight)
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {str(e)}")
            
            if prices_data:
                composite = pd.concat(prices_data, axis=1)
                composite = composite.fillna(method='ffill').fillna(method='bfill')
                self.sector_composites[sector] = composite.sum(axis=1).dropna()
        
        # Save to cache
        pd.to_pickle(self.sector_composites, cache_file)
        return self.sector_composites

    def analyze_sector_regimes(self, n_states=4):
        if self.testing_mode:
            return self.generate_test_sector_regimes()
            
        self.sector_analyzers = {}
        
        # Get overall market regime first
        if not hasattr(self, 'market_composite'):
            tickers = [t for sublist in self.sector_mappings.values() for t in sublist]
            self.logger.info("Creating market composite...")
            self.market_composite = self.overall_analyzer.prepare_market_data(tickers[:100])
        self.logger.info("Analyzing overall market regime...")
        market_result = self.overall_analyzer.analyze_regime(self.market_composite)
        self.current_regime = market_result["regimes"][-1]
        
        # Process each sector
        self.logger.info("Analyzing sector regimes...")
        for sector, composite in self.sector_composites.items():
            try:
                analyzer = MarketRegimeAnalyzer(polygon_api_key=self.polygon_api_key, testing_mode=self.testing_mode)
                results = analyzer.analyze_regime(composite, n_states=n_states)
                self.sector_analyzers[sector] = {
                    "results": results,
                    "composite": composite,
                    "volatility": composite.pct_change().std(),
                    "analyzer": analyzer
                }
            except Exception as e:
                self.logger.error(f"Error analyzing {sector}: {str(e)}")
                
        return self.sector_analyzers

    def calculate_sector_scores(self):
        if self.testing_mode:
            return self.generate_test_sector_scores()
            
        self.sector_scores = {}
        if not self.sector_analyzers:
            return pd.Series()

        # First, calculate raw scores per sector
        for sector, data in self.sector_analyzers.items():
            try:
                if "results" not in data:
                    continue
                    
                # Get latest probabilities
                current_probs = data["results"]["probabilities"][-1]
                state_labels = data["results"].get("state_labels", {})
                
                # Calculate sector momentum
                momentum = data["composite"].pct_change(21).iloc[-1] if len(data["composite"]) > 21 else 0
                
                # Calculate bull/bear probabilities
                bull_prob = sum(
                    current_probs[i] 
                    for i, label in state_labels.items() 
                    if "Bull" in label
                )
                bear_prob = sum(
                    current_probs[i] 
                    for i, label in state_labels.items() 
                    if "Bear" in label
                )
                
                # Base score (simplified)
                base_score = bull_prob - bear_prob
                
                # Apply momentum adjustment
                momentum_factor = 1 + (momentum * 5)  # Amplify momentum effect
                adjusted_score = base_score * momentum_factor
                
                # Apply sector weight
                weight = self.sector_weights.get(sector, 0.01)
                self.sector_scores[sector] = adjusted_score * (1 + weight)
                
            except Exception as e:
                self.logger.error(f"Error calculating score for {sector}: {str(e)}")
                self.sector_scores[sector] = 0

        # Clean and merge sector scores
        cleaned_scores = {}
        for raw_sector, score in self.sector_scores.items():
            clean_sector = self.clean_sector_name(raw_sector)
            if clean_sector not in cleaned_scores:
                cleaned_scores[clean_sector] = score
            else:
                # Take the max score for merged sectors
                cleaned_scores[clean_sector] = max(cleaned_scores[clean_sector], score)
                
        # Filter out "Unknown" sectors
        cleaned_scores = {k: v for k, v in cleaned_scores.items() if k != "Unknown"}
        
        self.sector_scores = cleaned_scores
        return pd.Series(self.sector_scores).sort_values(ascending=False)


# ======================== ASSET ALLOCATIONS ======================== #
ASSET_ALLOCATIONS = {
    "Bear": {
        "defensive_stocks": 70,    # Utilities, consumer staples
        "dividend_stocks": 30       # High-yield, stable companies
    },
    "Severe Bear": {
        "inverse_etfs": 40,         # Hedging instruments
        "defensive_stocks": 40,     # Essential services
        "cash": 20                  # Preserve capital
    },
    "Bull": {
        "growth_stocks": 60,        # High-growth companies
        "tech_stocks": 30,          # Technology sector
        "small_caps": 10            # Aggressive growth potential
    },
    "Strong Bull": {
        "growth_stocks": 75,        # Maximize growth exposure
        "tech_stocks": 20,          # Leading innovators
        "small_caps": 5             # Speculative growth
    },
    "Neutral": {
        "value_stocks": 50,         # Undervalued companies
        "dividend_stocks": 40,      # Income generation
        "cash": 10                  # Dry powder for opportunities
    },
    "Mild Bear": {
        "defensive_stocks": 60,     # Stable sectors
        "dividend_stocks": 40       # Income with lower volatility
    },
    "Mild Bull": {
        "growth_stocks": 60,        # Growth orientation
        "value_stocks": 35,         # Blend with value
        "cash": 5                   # Minimal cash reserve
    }
}
# ======================== END MARKET REGIME CODE ======================== #

# Polygon.io configuration
POLYGON_API_KEY = "OZzn0oK0H2yG6rpIvVhGfgXgnUTrL31z"
REST_API_URL = "https://api.polygon.io"
WEBSOCKET_URL = "wss://socket.polygon.io/stocks"

class SmartStopLoss:
    """Enhanced stop loss system with volatility and regime awareness + backward compatibility"""
    def __init__(self, entry_price, atr, adx, market_volatility=None, regime=None,
                 base_vol_factor=1.5, base_hard_stop=0.08, profit_target_ratio=3.0):
        self.entry_price = entry_price
        self.atr = atr
        self.adx = adx
        self.regime = regime or "Neutral"  # Default to neutral
        self.creation_time = datetime.now(tz.utc)
        
        # Backward compatibility handling
        if market_volatility is None:
            # Default to historical average volatility
            market_volatility = 0.15
        
        # 1. Volatility-based adjustments
        vol_deviation = market_volatility - 0.15  # Deviation from historical average
        volatility_factor = base_vol_factor
        hard_stop_percent = base_hard_stop
        
        # Only apply enhancements if market data is provided
        if market_volatility is not None:
            volatility_factor = base_vol_factor * (1 + vol_deviation * 3)
            hard_stop_percent = base_hard_stop * (1 + vol_deviation * 2)
        
        # 2. Regime-based adjustments (only if regime provided)
        if regime is not None:
            if "Bear" in regime:
                volatility_factor *= 1.3  # Wider stops in bear markets
                hard_stop_percent *= 0.9  # Slightly tighter hard stop
                profit_target_ratio *= 0.8  # Conservative targets
            elif "Bull" in regime:
                volatility_factor *= 0.8  # Tighter stops in bull markets
                hard_stop_percent *= 1.1  # More room in strong trends
                profit_target_ratio *= 1.2  # Ambitious targets
        
        # 3. ADX-based profit target scaling
        if adx > 40:  # Strong trend
            profit_target_ratio *= 1.3
        elif adx < 20:  # Weak trend
            profit_target_ratio *= 0.7
            
        # Clamp values to safe ranges
        self.volatility_factor = max(1.0, min(2.5, volatility_factor))
        self.hard_stop_percent = max(0.05, min(0.15, hard_stop_percent))
        self.profit_target_ratio = max(1.5, min(5.0, profit_target_ratio))
        
        # Initialize stops
        self.initial_stop = self.calculate_initial_stop()
        self.trailing_stop = self.initial_stop
        self.hard_stop = entry_price * (1 - self.hard_stop_percent)
        self.profit_target = entry_price + (entry_price - self.initial_stop) * self.profit_target_ratio
        self.profit_target_2 = entry_price + (entry_price - self.initial_stop) * (self.profit_target_ratio * 1.8)
        
        # Store history
        self.history = [{
            'timestamp': self.creation_time,
            'price': entry_price,
            'trailing_stop': self.trailing_stop,
            'hard_stop': self.hard_stop,
            'initial_stop': self.initial_stop,
            'profit_target': self.profit_target,
            'volatility_factor': self.volatility_factor,
            'regime': self.regime
        }]
        
    def calculate_initial_stop(self):
        base_stop = self.entry_price - self.atr * self.volatility_factor
        
        # ADX-based refinement (always applies)
        if self.adx > 40:  # Strong trend
            return base_stop * 0.95
        elif self.adx < 20:  # Weak trend
            return base_stop * 1.05
        return base_stop
    
    def update_trailing_stop(self, current_price, timestamp):
        # Base sensitivity
        sensitivity = 0.8
        
        # Regime-based adjustments (only if regime provided)
        if self.regime is not None:
            if "Bear" in self.regime:
                sensitivity = 0.65  # More responsive in bear markets
            elif "Bull" in self.regime and self.adx > 30:
                sensitivity = 0.95  # Less responsive in strong trends
        
        # Calculate new stop
        new_stop = current_price - self.atr * self.volatility_factor * sensitivity
        
        # Time-based relaxation for proven winners (always applies)
        holding_days = (datetime.now(tz.utc) - self.creation_time).days
        if holding_days > 3 and current_price > self.profit_target:
            # Loosen stop by 0.5% per day after 3 days
            relaxation = 1 - (0.005 * (holding_days - 3))
            new_stop *= relaxation
        
        # Update trailing stop if new stop is higher
        if new_stop > self.trailing_stop:
            self.trailing_stop = new_stop
        
        # Never go below hard stop
        self.trailing_stop = max(self.trailing_stop, self.hard_stop)
        
        # Record update
        self.history.append({
            'timestamp': timestamp,
            'price': current_price,
            'trailing_stop': self.trailing_stop,
            'hard_stop': self.hard_stop,
            'initial_stop': self.initial_stop,
            'profit_target': self.profit_target,
            'volatility_factor': self.volatility_factor,
            'regime': self.regime
        })
        
        return self.trailing_stop
    
    # Maintain all original methods unchanged
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

class AdaptiveLookbackSystem:
    """Dynamic data length adjustment based on position age and market conditions"""
    def __init__(self, base_lookback=35, max_extended=180):
        self.base = base_lookback
        self.max_extended = max_extended
        self.position_age = {}  # Ticker -> days held (calendar days)

    def get_lookback(self, ticker, volatility, trend_strength, is_open_position=False):
        """Calculate adaptive lookback in days"""
        lookback = self.base
        
        # 1. Adjust for holding period of existing positions
        if is_open_position:
            days_held = self.position_age.get(ticker, 0)
            # Extend lookback by 2 days for every day held, capped at max_extended
            lookback = min(self.max_extended, self.base + days_held * 2)
        
        # 2. Volatility scaling: volatility is a float (e.g., 0.15 for 15%)
        vol_deviation = volatility - 0.15  # Deviation from historical average
        volatility_factor = 1 + vol_deviation
        volatility_factor = max(0.7, min(1.5, volatility_factor))
        lookback = lookback * volatility_factor
        
        # 3. Trend strength adjustment (ADX: 0-100)
        if trend_strength > 40:  # Strong trend
            lookback = max(14, lookback * 0.8)  # Reduce lookback to focus on recent momentum
        elif trend_strength < 20:  # Weak trend (choppy)
            lookback = min(self.max_extended, lookback * 1.3)  # Extend lookback to filter noise
        
        # 4. Ensure we have at least 14 days for indicators
        lookback = max(14, lookback)
        return int(round(lookback))

class PolygonDataHandler:
    """Handles Polygon.io REST API and WebSocket data with parallel historical data loading"""
    def __init__(self, tickers, testing_mode=False):
        self.tickers = tickers
        self.api_key = POLYGON_API_KEY
        self.historical_data = {t: pd.DataFrame() for t in tickers}
        self.realtime_data = {t: None for t in tickers}
        self.data_queue = queue.Queue()
        self.running = True
        self.thread = None
        self.data_lock = threading.Lock()  # For thread-safe data access
        self.logger = logging.getLogger("PolygonDataHandler")
        self.logger.setLevel(logging.INFO)
        self.testing_mode = testing_mode
    
    def load_historical_data(self):
        """Load initial historical data for all tickers in parallel"""
        if self.testing_mode:
            self.generate_test_historical_data()
            return
            
        self.logger.info("Parallel loading initial historical data...")
        batches = [self.tickers[i:i+5] for i in range(0, len(self.tickers), 5)]
        
        for i, batch in enumerate(batches):
            self.logger.info(f"Processing batch {i+1}/{len(batches)}: {batch}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = {executor.submit(self.load_ticker_data, ticker): ticker for ticker in batch}
                
                for future in concurrent.futures.as_completed(futures):
                    ticker = futures[future]
                    try:
                        result = future.result()
                        if not result.empty:
                            with self.data_lock:
                                self.historical_data[ticker] = result
                            self.logger.info(f"Loaded {len(result)} historical bars for {ticker}")
                    except Exception as e:
                        self.logger.error(f"Error loading data for {ticker}: {str(e)}")
    
    def load_ticker_data(self, ticker):
        """Load data for a single ticker (thread worker function)"""
        if self.testing_mode:
            return self.generate_test_ticker_data(ticker)
            
        try:
            to_date = datetime.now(tz.utc) - timedelta(days=1)
            from_date = to_date - timedelta(days=30)
            
            url = f"{REST_API_URL}/v2/aggs/ticker/{ticker}/range/1/minute/" \
                  f"{from_date.strftime('%Y-%m-%d')}/{to_date.strftime('%Y-%m-%d')}" \
                  f"?adjusted=true&sort=asc&limit=50000&apiKey={self.api_key}"
            
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'OK' and data.get('resultsCount', 0) > 0:
                    df = pd.DataFrame(data['results'])
                    df['timestamp'] = pd.to_datetime(df["t"], unit='ms', utc=True)
                    df.set_index('timestamp', inplace=True)
                    # FIXED: Use lowercase column names
                    df.rename(columns={
                        'o': 'open', 'h': 'high', 'l': 'low', 
                        'c': 'close', 'v': 'volume'
                    }, inplace=True)
                    return df[['open', 'high', 'low', 'close', 'volume']]
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error in thread for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def run_websocket(self):
        """Connect to Polygon WebSocket and handle real-time data"""
        if self.testing_mode:
            self.run_test_websocket()
            return
            
        self.logger.info("WebSocket thread started")
        while self.running:
            try:
                ws = create_connection(WEBSOCKET_URL)
                ws.send(json.dumps({"action": "auth", "params": self.api_key}))
                params = ",".join([f"A.{t}" for t in self.tickers])
                ws.send(json.dumps({"action": "subscribe", "params": params}))
                self.logger.info("WebSocket connected and authenticated")
                
                while self.running:
                    try:
                        message = ws.recv()
                        if message:
                            data = json.loads(message)
                            for event in data:
                                self.process_websocket_event(event)
                    except Exception as e:
                        self.logger.error(f"WebSocket error: {str(e)}")
                        break
            except Exception as e:
                self.logger.error(f"Connection error: {str(e)}")
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
                # FIXED: Use lowercase column names
                new_bar = pd.DataFrame({
                    'open': [event['o']], 'high': [event['h']], 
                    'low': [event['l']], 'close': [event['c']], 
                    'volume': [event['v']]
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
                    'open': event['o'], 'high': event['h'], 'low': event['l'],
                    'close': event['c'], 'volume': event['v'],
                    'timestamp': timestamp
                }
                
                self.data_queue.put((ticker, self.realtime_data[ticker]))
        except Exception as e:
            self.logger.error(f"Event processing error: {str(e)}")
    
    def start(self):
        """Start WebSocket connection and load historical data"""
        if not self.thread or not self.thread.is_alive():
            # First load historical data
            self.load_historical_data()
            
            # Then start WebSocket
            self.running = True
            self.thread = threading.Thread(target=self.run_websocket)
            self.thread.daemon = True
            self.thread.start()
            self.logger.info("Data feed started")
        else:
            self.logger.info("Data feed already running")
    
    def stop(self):
        """Stop WebSocket connection"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        self.logger.info("Data feed stopped")
    
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
                    'open': last_row['open'],
                    'high': last_row['high'],
                    'low': last_row['low'],
                    'close': last_row['close'],
                    'volume': last_row['volume'],
                    'timestamp': self.historical_data[ticker].index[-1]
                }
        
        return None
    
    def get_historical(self, ticker, period=100):
        with self.data_lock:
            if ticker not in self.historical_data:
                return None
            df = self.historical_data[ticker]
            return df.iloc[-period:] if len(df) >= period else df
            
    # ======= TESTING MODE METHODS ======= #
    def generate_test_historical_data(self):
        """Generate simulated historical data for testing"""
        self.logger.info("Generating test historical data...")
        for ticker in self.tickers:
            start_date = datetime.now(tz.utc) - timedelta(days=30)
            dates = pd.date_range(start_date, datetime.now(tz.utc), freq='1min')
            
            # Create a random walk with volatility
            prices = [random.uniform(100, 200)]
            for i in range(1, len(dates)):
                change = random.uniform(-0.001, 0.002)
                prices.append(prices[-1] * (1 + change))
                
            # FIXED: Use lowercase column names
            df = pd.DataFrame({
                'open': prices,
                'high': [p * 1.001 for p in prices],
                'low': [p * 0.999 for p in prices],
                'close': prices,
                'volume': [random.randint(1000, 10000) for _ in prices]
            }, index=dates)
            
            self.historical_data[ticker] = df
            
    def generate_test_ticker_data(self, ticker):
        """Generate test data for a single ticker"""
        start_date = datetime.now(tz.utc) - timedelta(days=30)
        dates = pd.date_range(start_date, datetime.now(tz.utc), freq='1min')
        
        # Create a random walk with volatility
        prices = [random.uniform(100, 200)]
        for i in range(1, len(dates)):
            change = random.uniform(-0.001, 0.002)
            prices.append(prices[-1] * (1 + change))
            
        # FIXED: Use lowercase column names
        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.001 for p in prices],
            'low': [p * 0.999 for p in prices],
            'close': prices,
            'volume': [random.randint(1000, 10000) for _ in prices]
        }, index=dates)
        
        return df
        
    def run_test_websocket(self):
        """Simulate real-time data in testing mode"""
        self.logger.info("Starting test WebSocket simulator")
        while self.running:
            for ticker in self.tickers:
                # Generate a new data point
                last_data = self.get_latest(ticker)
                if last_data:
                    last_price = last_data['close']
                else:
                    last_price = random.uniform(100, 200)
                    
                # Create small price change
                price_change = random.uniform(-0.005, 0.01)
                new_price = last_price * (1 + price_change)
                
                new_data = {
                    'open': new_price,
                    'high': new_price * 1.001,
                    'low': new_price * 0.999,
                    'close': new_price,
                    'volume': random.randint(1000, 10000),
                    'timestamp': datetime.now(tz.utc)
                }
                
                # Update historical data
                with self.data_lock:
                    # FIXED: Use lowercase column names
                    new_bar = pd.DataFrame({
                        'open': [new_data['open']], 
                        'high': [new_data['high']], 
                        'low': [new_data['low']], 
                        'close': [new_data['close']], 
                        'volume': [new_data['volume']]
                    }, index=[new_data['timestamp']])
                    
                    if not self.historical_data[ticker].empty:
                        self.historical_data[ticker] = pd.concat([self.historical_data[ticker], new_bar])
                        self.historical_data[ticker] = self.historical_data[ticker].iloc[-1000:]
                    else:
                        self.historical_data[ticker] = new_bar
                
                # Update realtime data
                self.realtime_data[ticker] = new_data
                self.data_queue.put((ticker, new_data))
                
            # Sleep for a short time to simulate real-time updates
            time.sleep(1)


class TradingSystem(QThread):
    """Complete trading system with adaptive lookback and position scaling"""
    # Stock replacement configuration
    POSITION_EVALUATION_INTERVAL = 3600  # Evaluate positions hourly
    MIN_HOLDING_DAYS = 5  # Minimum days before considering replacement
    SCORE_DEGRADATION_THRESHOLD = 0.8  # 20% score drop triggers review
    RELATIVE_STRENGTH_MARGIN = 0.15  # 15% better score required for replacement
    MIN_SCORE_FOR_ENTRY = 70  # Minimum score to enter a position
    
    update_signal = pyqtSignal(dict)
    log_signal = pyqtSignal(str)
    scan_requested = pyqtSignal()  # Signal for manual scanning
    
    def __init__(self, tickers, capital=100000, risk_per_trade=0.01, testing_mode=False):
        super().__init__()
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.tickers = tickers
        self.positions = {}
        self.trade_log = []
        self.data_handler = None
        self.running = False
        self.eastern = pytz.timezone('US/Eastern')
        self.evaluation_interval = 30
        self.last_evaluation_time = time.time()
        self.last_opportunities = []
        self.testing_mode = testing_mode
        
        # Manual scan flag
        self.scan_requested_flag = False
        
        # Replacement system attributes
        self.position_evaluation_times = {}
        self.runner_ups = []  # Track top opportunities
        self.sector_system = SectorRegimeSystem(testing_mode=testing_mode)
        self.executed_tickers = set()
        self.last_evaluation_timestamp = 0
        
        # Market regime tracking
        self.market_regime = "Neutral"
        self.sector_scores = {}
        self.regime_last_updated = 0
        self.regime_analysis_interval = 3600 * 6  # 6 hours
        self.market_volatility = 0.15  # Default volatility
        
        # Adaptive lookback system
        self.lookback_system = AdaptiveLookbackSystem(base_lookback=35, max_extended=180)
        self.last_update_date = datetime.now(tz.utc).date()
        
        self.logger = logging.getLogger("TradingSystem")
        self.logger.setLevel(logging.INFO)
        
        # Connect signals
        self.scan_requested.connect(self.force_scan)
        
        print(f"Trading system initialized (Testing Mode: {testing_mode})")
        
    def ensure_scalar(self, value):
        """Convert Series to scalar if needed"""
        if isinstance(value, pd.Series):
            return value.iloc[-1] if not value.empty else 0
        if isinstance(value, pd.DataFrame):
            return value.iloc[-1, -1] if not value.empty else 0
        return value
        
    def force_scan(self):
        """Set flag for manual scan"""
        self.scan_requested_flag = True
        self.log_signal.emit("Manual scan requested by user")
        
    def run(self):
        """Main trading system loop"""
        if not self.running:
            return
            
        self.log_signal.emit("Trading system started")
        self.data_handler = PolygonDataHandler(self.tickers, testing_mode=self.testing_mode)
        self.data_handler.start()
        
        while self.running:
            try:
                # Process manual scan request
                if self.scan_requested_flag:
                    self.scan_requested_flag = False
                    self.evaluate_opportunities()
                    self.enter_top_opportunities()
                    self.log_signal.emit("Manual scan completed")
                
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
                
                # Update market regime periodically
                if current_time - self.regime_last_updated > self.regime_analysis_interval:
                    self.update_market_regime()
                    self.regime_last_updated = current_time
                
                # Daily position age update
                now = datetime.now(tz.utc)
                current_date = now.date()
                if current_date != self.last_update_date:
                    self.daily_update()
                    self.last_update_date = current_date
                
                # Emit state update
                self.update_signal.emit(self.get_current_state())
                time.sleep(0.5)
                
            except Exception as e:
                self.log_signal.emit(f"System error: {str(e)}")
        
        # Clean up when stopping
        if self.data_handler:
            self.data_handler.stop()
        self.log_signal.emit("Trading system stopped")
    
    def daily_update(self):
        """Increment holding days for all positions"""
        for ticker in list(self.lookback_system.position_age.keys()):
            self.lookback_system.position_age[ticker] += 1
        self.log_signal.emit("Daily position age updated")
    
    def update_market_regime(self):
        """Update market regime and sector scores"""
        try:
            self.log_signal.emit("Starting market regime analysis...")
            
            # Update sector mappings
            self.sector_system.map_tickers_to_sectors(self.tickers)
            
            # Calculate sector weights
            self.sector_system.calculate_sector_weights()
            
            # Build sector composites
            self.sector_system.build_sector_composites()
            
            # Analyze sector regimes
            self.sector_system.analyze_sector_regimes()
            
            # Calculate sector scores
            scores = self.sector_system.calculate_sector_scores()
            
            # Update market regime
            self.market_regime = self.sector_system.current_regime
            self.sector_scores = scores.to_dict()
            
            # Update market volatility from composite - NEW IMPROVED VERSION
            if not self.sector_system.market_composite.empty:
                try:
                    # Create a proper OHLC DataFrame - CRITICAL FIX
                    composite_df = pd.DataFrame({
                        'open': self.sector_system.market_composite,
                        'high': self.sector_system.market_composite,
                        'low': self.sector_system.market_composite,
                        'close': self.sector_system.market_composite
                    })
                    
                    # Compute 14-day ATR using the properly structured DataFrame
                    atr = composite_df.ta.atr(length=14).iloc[-1]
                    current_price = composite_df['close'].iloc[-1]
                    self.market_volatility = self.ensure_scalar(atr / current_price)
                except Exception as e:
                    self.log_signal.emit(f"Volatility update error: {str(e)}")
                    self.market_volatility = 0.15  # fallback
            
            # Ensure volatility is scalar before formatting
            vol_scalar = self.ensure_scalar(self.market_volatility)
            
            self.log_signal.emit(
                f"Market regime updated: {self.market_regime} "
                f"(Volatility: {vol_scalar*100:.2f}%)"
            )
            self.log_signal.emit("Top sectors:")
            for sector, score in sorted(self.sector_scores.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)[:3]:
                self.log_signal.emit(f"  {sector}: {score:.4f}")
                
            # Send Discord notification
            if not self.testing_mode:
                self.send_discord_notification()
            
        except Exception as e:
            self.log_signal.emit(f"Market regime analysis failed: {str(e)}")
    
    def send_discord_notification(self):
        """Send market analysis summary to Discord"""
        DISCORD_WEBHOOK_URL = "YOUR_DISCORD_WEBHOOK_URL"
        if not DISCORD_WEBHOOK_URL:
            return
            
        try:
            regime = self.market_regime
            sector_scores = self.sector_scores
            
            # Prepare message content
            content = f"**Market Analysis Update**\nCurrent Regime: **{regime}**"
            
            # Prepare fields for embed
            fields = []
            
            # Asset allocation
            allocation = ASSET_ALLOCATIONS.get(regime, {})
            alloc_text = "\n".join([f"- {asset.replace('_', ' ').title()}: {pct}%" for asset, pct in allocation.items()])
            fields.append({"name": "Recommended Allocation", "value": alloc_text, "inline": True})
            
            # Top sectors
            top_sectors = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            sectors_text = "\n".join([f"- {sector}: {score:.2f}" for sector, score in top_sectors])
            fields.append({"name": "Top Performing Sectors", "value": sectors_text, "inline": True})
            
            # Create embed
            embed = {
                "title": "Market Analysis Report",
                "color": 0x3498db,
                "fields": fields,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Send to Discord
            payload = {
                "content": content,
                "embeds": [embed]
            }
            
            headers = {"Content-Type": "application/json"}
            response = requests.post(DISCORD_WEBHOOK_URL, data=json.dumps(payload), headers=headers)
            
            if response.status_code != 204:
                self.log_signal.emit(f"Discord notification failed: {response.status_code} {response.text}")
            else:
                self.log_signal.emit("Sent market analysis to Discord")
                
        except Exception as e:
            self.log_signal.emit(f"Discord notification failed: {e}")
    
    def update_positions(self, ticker, data):
        """Update all positions for a ticker"""
        for position_ticker, position in list(self.positions.items()):
            if position_ticker == ticker:
                current_price = data['close']
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
                    self.exit_position(ticker, data['close'], "time expiration")
    
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
            # Create a mapping of futures to tickers
            future_to_ticker = {
                executor.submit(self.score_trade_opportunity, ticker): ticker
                for ticker in tickers_to_score
            }
            
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]  # Get the ticker from the mapping
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
                    
                current_price = data['close']
                
                # Recalculate indicators for fresh data
                # FIXED: Remove explicit column mappings
                atr = self.ensure_scalar(df.ta.atr(length=14).iloc[-1])
                
                # ADX calculation with fallback
                try:
                    adx = df.ta.adx(length=14)['ADX_14'].iloc[-1]
                    adx = self.ensure_scalar(adx)
                except Exception as e:
                    self.log_signal.emit(f"ADX calculation failed for {opp['ticker']}: {str(e)}")
                    adx = 20.0  # Default value
                
                # Create stop system with enhancements if data available
                if hasattr(self, 'market_volatility') and hasattr(self, 'market_regime'):
                    stop_system = SmartStopLoss(
                        entry_price=current_price,
                        atr=atr,
                        adx=adx,
                        market_volatility=self.market_volatility,
                        regime=self.market_regime
                    )
                else:
                    # Fallback to original behavior
                    stop_system = SmartStopLoss(
                        entry_price=current_price,
                        atr=atr,
                        adx=adx
                    )
                
                self.enter_position(
                    opp['ticker'],
                    current_price,
                    atr,
                    adx,
                    opp['score'],  # Store original score
                    stop_system  # Pass the stop system
                )
            else:
                if opp['ticker'] in self.positions:
                    self.log_signal.emit(f"Already in position for {opp['ticker']}")
                else:
                    self.log_signal.emit(f"Score too low for {opp['ticker']}: {opp['score']} < {self.MIN_SCORE_FOR_ENTRY}")
    
    def is_market_open(self):
        """Check if market is open"""
        if self.testing_mode:
            return True  # Always open in testing mode
            
        now = datetime.now(self.eastern)
        if now.weekday() >= 5:  # Weekend
            return False
            
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now <= market_close
    
    def get_current_state(self):
        """Get current system state for UI"""
        total_profit = sum(trade['profit'] for trade in self.trade_log if trade['profit'] is not None)
        total_profit = self.ensure_scalar(total_profit)
        
        winning_trades = sum(1 for trade in self.trade_log if trade['profit'] and trade['profit'] > 0)
        win_rate = winning_trades / len(self.trade_log) * 100 if self.trade_log else 0
        win_rate = self.ensure_scalar(win_rate)
        
        # Prepare active positions
        active_positions = []
        for ticker, pos in self.positions.items():
            current_data = self.data_handler.get_latest(ticker) if self.data_handler else None
            current_price = current_data['close'] if current_data and current_data else pos['entry_price']
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
                'original_score': pos.get('original_score', 0),
                'days_held': self.lookback_system.position_age.get(ticker, 0)
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
                'volume': opp['volume'], 'status': status,
                'lookback': opp.get('lookback_days', 35)
            })
        
        # Prepare runner-ups
        runner_ups = []
        for opp in self.runner_ups:
            runner_ups.append({
                'ticker': opp['ticker'], 'score': opp['score'],
                'price': opp['price'], 'adx': opp['adx'],
                'atr': opp['atr'], 'rsi': opp['rsi'],
                'volume': opp['volume'],
                'lookback': opp.get('lookback_days', 35)
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
            'runner_ups': runner_ups,
            'market_regime': self.market_regime,
            'sector_scores': self.sector_scores,
            'market_volatility': self.ensure_scalar(self.market_volatility),
            'testing_mode': self.testing_mode
        }
    
    def score_trade_opportunity(self, ticker):
        """Score trade opportunity (0-100 scale) with adaptive lookback"""
        try:
            if not self.data_handler:
                return None
                
            if not self.is_market_open() and not self.testing_mode:
                return None
                
            # Check if this is for an open position
            is_open_position = ticker in self.positions
            
            # Step 1: Get base data (35 days) for initial ADX calculation
            base_lookback_days = 35
            base_lookback_minutes = base_lookback_days * 390  # 390 minutes per trading day
            base_data = self.data_handler.get_historical(ticker, base_lookback_minutes)
            if base_data is None or base_data.empty:
                self.log_signal.emit(f"Insufficient base data for {ticker}")
                return None
                
            # Calculate base ADX from base_data
            try:
                # FIXED: Remove explicit column mappings
                base_adx = base_data.ta.adx(length=14)['ADX_14'].iloc[-1]
                base_adx = self.ensure_scalar(base_adx)
                # If NaN, use 20
                if np.isnan(base_adx):
                    base_adx = 20.0
            except:
                base_adx = 20.0
                
            # Get adaptive lookback
            lookback_days = self.lookback_system.get_lookback(
                ticker, 
                self.market_volatility, 
                base_adx,
                is_open_position
            )
            
            # Get data for the adaptive lookback period
            if lookback_days != base_lookback_days:
                # Fetch data for the new lookback period
                data = self.data_handler.get_historical(ticker, lookback_days * 390)
            else:
                data = base_data
                
            if data is None or data.empty:
                return None
                
            # Recalculate indicators with the adaptive lookback data
            # FIXED: Remove explicit column mappings
            atr = self.ensure_scalar(data.ta.atr(length=14).iloc[-1])
            
            # ADX calculation with fallback
            try:
                # FIXED: Remove explicit column mappings
                adx = data.ta.adx(length=14)['ADX_14'].iloc[-1]
                adx = self.ensure_scalar(adx)
            except Exception as e:
                self.log_signal.emit(f"ADX calculation failed for {ticker}: {str(e)}")
                adx = 20.0  # Default value
                
            # FIXED: Remove explicit column mappings
            rsi = self.ensure_scalar(data.ta.rsi(length=14).iloc[-1])
            
            # Get the latest data point for price and volume
            latest = self.data_handler.get_latest(ticker)
            if not latest:
                return None
            price = latest['close']
            volume = latest['volume']
            
            # Calculate average volume for the period in data
            avg_volume = data['volume'].rolling(14).mean().iloc[-1]
            avg_volume = self.ensure_scalar(avg_volume)
            if np.isnan(avg_volume) or avg_volume <= 0:
                avg_volume = self.ensure_scalar(data['volume'].mean())
                
            if volume <= 0:
                volume = avg_volume
                
            # Create stop system for scoring
            if hasattr(self, 'market_volatility') and hasattr(self, 'market_regime'):
                stop_system = SmartStopLoss(
                    entry_price=price,
                    atr=atr,
                    adx=adx,
                    market_volatility=self.market_volatility,
                    regime=self.market_regime
                )
            else:
                # Fallback to original behavior
                stop_system = SmartStopLoss(
                    entry_price=price,
                    atr=atr,
                    adx=adx
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
            
            # 6. Sector Strength Adjustment
            sector = self.get_ticker_sector(ticker)
            sector_score = self.sector_scores.get(sector, 50)  # Default to 50 if sector not found
            sector_factor = 1 + (sector_score - 50) / 100  # Convert 0-100 score to 0.5-1.5 factor
            
            # 7. Market Regime Adjustment
            regime_factor = self.get_regime_factor()
            
            # Weighted composite score
            composite_score = (
                0.30 * adx_score + 0.25 * atr_score + 
                0.20 * rr_score + 0.15 * volume_score + 
                0.10 * rsi_score
            ) * sector_factor * regime_factor
            
            # Cap score at 100
            composite_score = min(100, composite_score)
            
            self.log_signal.emit(
                f"Scored {ticker}: {composite_score:.1f} (Lookback: {lookback_days}d, Sector: {sector} {sector_score:.1f})"
            )
            return {
                'ticker': ticker, 'score': composite_score,
                'price': price, 'atr': atr, 'adx': adx,
                'rsi': rsi, 'volume': volume,
                'risk_reward': rr_ratio,
                'lookback_days': lookback_days
            }
        except Exception as e:
            self.log_signal.emit(f"Scoring error for {ticker}: {str(e)}")
            return None
    
    def get_ticker_sector(self, ticker):
        """Get sector for a ticker"""
        for sector, tickers in self.sector_system.sector_mappings.items():
            if ticker in tickers:
                return sector
        return "Unknown"
    
    def get_regime_factor(self):
        """Get position sizing factor based on market regime"""
        if "Bull" in self.market_regime:
            return 1.2  # More aggressive in bull markets
        elif "Bear" in self.market_regime:
            return 0.8  # More conservative in bear markets
        return 1.0  # Neutral in other regimes
    
    def enter_position(self, ticker, price, atr, adx, original_score, stop_system):
        """Enter new position with original score and stop system"""
        try:
            if ticker in self.positions:
                self.log_signal.emit(f"Already in position for {ticker}")
                return
                
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
            
            # Initialize position age
            self.lookback_system.position_age[ticker] = 0
            
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
            
            # Remove from position age tracking
            if ticker in self.lookback_system.position_age:
                del self.lookback_system.position_age[ticker]
            
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
            holding_days = self.lookback_system.position_age.get(ticker, 0)
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
            prices[ticker] = latest['close'] if latest else self.positions[ticker]['entry_price']
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
            try:
                # FIXED: Remove explicit column mappings
                adx = df.ta.adx(length=14)['ADX_14'].iloc[-1]
                adx = self.ensure_scalar(adx)
            except Exception as e:
                self.log_signal.emit(f"ADX calculation failed for {ticker}: {str(e)}")
                adx = 20.0  # Default value
                
            # FIXED: Remove explicit column mappings
            rsi = self.ensure_scalar(df.ta.rsi(length=14).iloc[-1])
            volume = self.ensure_scalar(df['volume'].iloc[-1])
            avg_volume = df['volume'].rolling(14).mean().iloc[-1]
            avg_volume = self.ensure_scalar(avg_volume)
            
            # Position performance metrics
            price_change = ((current_price - position['entry_price']) / position['entry_price']) * 100
            volume_ratio = volume / avg_volume
            
            # Regime-based weighting
            regime_factor = self.get_regime_factor()
            
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
        sector_scores = self.sector_scores
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
            current_price = self.data_handler.get_latest(old_ticker)['close']
            self.exit_position(old_ticker, current_price, "Replaced by stronger candidate")
            
            # Place new trade
            new_data = self.data_handler.get_latest(new_candidate['ticker'])
            if new_data:
                df = self.data_handler.get_historical(new_candidate['ticker'], 50)
                if df is None or df.empty:
                    return
                
                # Recalculate indicators for fresh data
                # FIXED: Remove explicit column mappings
                atr = self.ensure_scalar(df.ta.atr(length=14).iloc[-1])
                
                # ADX calculation with fallback
                try:
                    # FIXED: Remove explicit column mappings
                    adx = df.ta.adx(length=14)['ADX_14'].iloc[-1]
                    adx = self.ensure_scalar(adx)
                except Exception as e:
                    self.log_signal.emit(f"ADX calculation failed for {new_candidate['ticker']}: {str(e)}")
                    adx = 20.0  # Default value
                
                current_price = new_data['close']
                
                # Create stop system with enhancements if data available
                if hasattr(self, 'market_volatility') and hasattr(self, 'market_regime'):
                    stop_system = SmartStopLoss(
                        entry_price=current_price,
                        atr=atr,
                        adx=adx,
                        market_volatility=self.market_volatility,
                        regime=self.market_regime
                    )
                else:
                    # Fallback to original behavior
                    stop_system = SmartStopLoss(
                        entry_price=current_price,
                        atr=atr,
                        adx=adx
                    )
                
                self.enter_position(
                    new_candidate['ticker'],
                    current_price,
                    atr,
                    adx,
                    new_candidate['score'],  # Store original score
                    stop_system  # Pass the stop system
                )
                self.log_signal.emit(f"Replaced {old_ticker} with {new_candidate['ticker']}")
    
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
        df['close'].plot(ax=self.ax, label='Price', color='blue', linewidth=2)
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
    """Interactive trading dashboard with 'Scan Now' button"""
    def __init__(self, testing_mode=False):
        super().__init__()
        self.tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "DIS"]
        self.trading_system = TradingSystem(self.tickers, testing_mode=testing_mode)
        self.setWindowTitle("Real-Time Trading Dashboard" + (" - TESTING MODE" if testing_mode else ""))
        self.setGeometry(100, 100, 1600, 900)
        self.testing_mode = testing_mode
        
        # Configure logging
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger("TradingDashboard")
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        header_layout = QHBoxLayout()
        self.timestamp_label = QLabel("Timestamp: Initializing...")
        self.market_status_label = QLabel("Market Status: Checking...")
        self.regime_label = QLabel("Market Regime: Unknown")
        self.volatility_label = QLabel("Market Volatility: 0.00%")
        header_layout.addWidget(self.timestamp_label)
        header_layout.addStretch()
        header_layout.addWidget(self.market_status_label)
        header_layout.addWidget(self.regime_label)
        header_layout.addWidget(self.volatility_label)
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
        self.scan_button = QPushButton("Scan Now")  # ADDED SCAN BUTTON
        self.testing_label = QLabel("TESTING MODE" if testing_mode else "LIVE MODE")
        self.testing_label.setStyleSheet("color: red; font-weight: bold;" if testing_mode else "color: green; font-weight: bold;")
        
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.scan_button)  # ADDED TO LAYOUT
        control_layout.addStretch()
        control_layout.addWidget(self.testing_label)
        main_layout.addLayout(control_layout)
        
        # Tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Positions tab
        positions_tab = QWidget()
        positions_layout = QVBoxLayout(positions_tab)
        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(13)  # Added original score and sector columns
        self.positions_table.setHorizontalHeaderLabels([
            "Ticker", "Shares", "Entry", "Current", "Gain%", 
            "Trail Stop", "Hard Stop", "Profit Tgt", "Risk%", "Regime", "Score", "Sector", "Days Held"
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
        self.opportunities_table.setColumnCount(9)
        self.opportunities_table.setHorizontalHeaderLabels([
            "Ticker", "Score", "Price", "ADX", "ATR", "RSI", "Volume", "Status", "Lookback"
        ])
        self.opportunities_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        opportunities_layout.addWidget(self.opportunities_table)
        self.tabs.addTab(opportunities_tab, "Trade Opportunities")
        
        # Runner-Ups tab
        runner_ups_tab = QWidget()
        runner_ups_layout = QVBoxLayout(runner_ups_tab)
        self.runner_ups_table = QTableWidget()
        self.runner_ups_table.setColumnCount(8)
        self.runner_ups_table.setHorizontalHeaderLabels([
            "Ticker", "Score", "Price", "ADX", "ATR", "RSI", "Volume", "Lookback"
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
        
        # Sector tab
        self.sector_tab = QWidget()
        self.sector_layout = QVBoxLayout(self.sector_tab)
        self.sector_table = QTableWidget()
        self.sector_table.setColumnCount(2)
        self.sector_table.setHorizontalHeaderLabels(["Sector", "Score"])
        self.sector_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.sector_layout.addWidget(self.sector_table)
        
        # Allocation recommendations
        self.alloc_label = QLabel("Recommended Asset Allocation:")
        self.alloc_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.sector_layout.addWidget(self.alloc_label)
        
        self.alloc_text = QTextEdit()
        self.alloc_text.setReadOnly(True)
        self.alloc_text.setFont(QFont("Arial", 10))
        self.sector_layout.addWidget(self.alloc_text)
        
        self.tabs.addTab(self.sector_tab, "Market Analysis")
        
        # Connect signals
        self.start_button.clicked.connect(self.start_system)
        self.stop_button.clicked.connect(self.stop_system)
        self.scan_button.clicked.connect(self.request_scan)  # Connect scan button
        self.trading_system.update_signal.connect(self.update_ui)
        self.trading_system.log_signal.connect(self.log_message)
        
        # Initial state
        self.stop_button.setEnabled(False)
        self.scan_button.setEnabled(False)  # Disable scan button initially
        
        # Initialize log
        self.log_message("Trading Dashboard Initialized")
        self.log_message(f"Tracking Tickers: {', '.join(self.tickers)}")
        self.log_message(f"Operating in {'TESTING' if testing_mode else 'LIVE'} mode")
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
            'runner_ups': [],
            'market_regime': "Neutral",
            'sector_scores': {},
            'market_volatility': 0.15,
            'testing_mode': testing_mode
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
                self.scan_button.setEnabled(True)  # Enable scan button
                self.log_message("Trading system started")
                return
        self.log_message("System already running")
        
    def stop_system(self):
        """Stop the trading system"""
        if self.trading_system.isRunning():
            if self.trading_system.stop_system():
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                self.scan_button.setEnabled(False)  # Disable scan button
                self.log_message("Trading system stopped")
                return
        self.log_message("System not running")
    
    def request_scan(self):
        """Request manual scan"""
        if self.trading_system.running:
            self.trading_system.scan_requested.emit()
            self.log_message("Manual scan requested")
        else:
            self.log_message("System not running - cannot scan")
    
    def update_ui(self, state):
        """Update all UI elements with current state"""
        try:
            # Update header
            self.timestamp_label.setText(f"Timestamp: {state['timestamp']}")
            market_status = "OPEN" if state['market_open'] else "CLOSED"
            self.market_status_label.setText(f"Market Status: {market_status}")
            self.regime_label.setText(f"Market Regime: {state['market_regime']}")
            
            # Ensure market_volatility is scalar
            volatility = state['market_volatility']
            if isinstance(volatility, pd.Series):
                volatility = volatility.iloc[-1] if not volatility.empty else 0.15
            self.volatility_label.setText(f"Market Volatility: {volatility*100:.2f}%")
            
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
                
                # Add sector information
                sector = self.trading_system.get_ticker_sector(pos['ticker'])
                self.positions_table.setItem(row, 11, QTableWidgetItem(sector))
                
                # Add days held
                self.positions_table.setItem(row, 12, QTableWidgetItem(str(pos.get('days_held', 0))))
            
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
                
                self.opportunities_table.setItem(row, 8, QTableWidgetItem(f"{opp.get('lookback', 35)}d"))
            
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
                self.runner_ups_table.setItem(row, 7, QTableWidgetItem(f"{opp.get('lookback', 35)}d"))
            
            # Update position plots
            self.update_plots()
            
            # Update sector analysis tab
            self.update_sector_tab(state)
            
        except Exception as e:
            self.log_message(f"UI update error: {str(e)}")
    
    def update_sector_tab(self, state):
        """Update the sector analysis tab"""
        # Update sector scores
        self.sector_table.setRowCount(len(state['sector_scores']))
        for row, (sector, score) in enumerate(state['sector_scores'].items()):
            self.sector_table.setItem(row, 0, QTableWidgetItem(sector))
            score_item = QTableWidgetItem(f"{score:.2f}")
            # Color code based on score
            if score > 70:
                score_item.setForeground(QBrush(QColor('green')))
            elif score < 30:
                score_item.setForeground(QBrush(QColor('red')))
            else:
                score_item.setForeground(QBrush(QColor('orange')))
            self.sector_table.setItem(row, 1, score_item)
        
        # Update allocation recommendations
        regime = state['market_regime']
        allocation = ASSET_ALLOCATIONS.get(regime, {})
        alloc_text = "\n".join([f"• {asset.replace('_', ' ').title()}: {pct}%" for asset, pct in allocation.items()])
        self.alloc_text.setHtml(f"""
            <h3>Recommended Allocation for {regime} Market:</h3>
            <p>{alloc_text}</p>
        """)
    
    def update_plots(self):
        """Update position analysis plots"""
        # Clear existing plots
        for i in reversed(range(self.plot_layout.count())):
            widget = self.plot_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        # Create new plots only if system is running
        if self.trading_system.running and self.trading_system.data_handler:
            for ticker, position in self.trading_system.positions.items():
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
    
    # Check for testing mode flag
    testing_mode = "--testing" in sys.argv
    
    # Create and show dashboard
    dashboard = TradingDashboard(testing_mode=testing_mode)
    dashboard.show()
    
    # Start application
    sys.exit(app.exec_())