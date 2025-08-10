import numpy as np
if not hasattr(np, 'NaN'):
    np.NaN = np.nan
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
    QTabWidget, QTextEdit, QGroupBox, QFormLayout, QDateTimeEdit,
    QAbstractItemView, QDialog, QProgressBar, QLineEdit, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDateTime
from PyQt5.QtGui import QColor, QBrush, QFont
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.dates as mdates
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import concurrent.futures
import talib
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.exceptions import ConvergenceWarning
from concurrent.futures import ThreadPoolExecutor
import random
import pandas_market_calendars as mcal
from pandas.tseries.offsets import CustomBusinessDay

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

# Polygon.io configuration
POLYGON_API_KEY = "OZzn0oK0H2yG6rpIvVhGfgXgnUTrL31z"
REST_API_URL = "https://api.polygon.io"
WEBSOCKET_URL = "wss://socket.polygon.io/stocks"

# ======================== MARKET REGIME CODE ======================== #
class MarketRegimeAnalyzer:
    def __init__(self, n_states=4, polygon_api_key=POLYGON_API_KEY, testing_mode=False):
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=20000,
            tol=1e-3,
            init_params="se",
            params="stmc",
            random_state=42,
            min_covar=0.2,
            implementation='scaling',
            verbose=True
        )
        self.state_labels = {}
        self.feature_scaler = RobustScaler()
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
        mcaps = {}

        self.logger.info("Collecting market caps...")
        for ticker in tickers[:sample_size]:
            mcaps[ticker] = self.get_market_cap(ticker) or 1

        total_mcap = sum(mcaps.values())
        
        self.logger.info("Building market composite...")
        for symbol in tickers[:sample_size]:
            prices = self.fetch_stock_data(symbol)
            if prices is not None and len(prices) >= min_days_data:
                weight = mcaps.get(symbol, 1) / total_mcap
                prices_data.append(prices * weight)
                valid_tickers.append(symbol)

        if not prices_data:
            raise ValueError("Insufficient data to create market composite")
            
        composite = pd.concat(prices_data, axis=1)
        composite.columns = valid_tickers
        composite = composite.fillna(method='ffill').fillna(method='bfill')
        return composite.sum(axis=1).dropna()

    def analyze_regime(self, index_data, n_states=None):
        if n_states is None:
            n_states = self.model.n_components

        self.logger.info(f"Preparing features for regime analysis with {n_states} states...")
        
        log_returns = np.log(index_data).diff().dropna()
        features = pd.DataFrame({
            "returns": log_returns,
            "volatility": log_returns.rolling(21).std(),
            "momentum": log_returns.rolling(14).mean(),
            "rsi": talib.RSI(index_data, timeperiod=14).dropna(),
            "macd": talib.MACD(index_data)[0].dropna(),
        }).dropna()

        if len(features) < 100:
            self.logger.warning(f"Insufficient feature data: only {len(features)} samples")
            raise ValueError(f"Only {len(features)} days of feature data")
            
        self.logger.info("Scaling features...")
        scaled_features = self.feature_scaler.fit_transform(features)
        scaled_features = np.clip(scaled_features, -10, 10)
        
        min_val = np.min(scaled_features)
        max_val = np.max(scaled_features)
        self.logger.info(f"Feature range after scaling/clipping: Min={min_val:.4f}, Max={max_val:.4f}")

        best_model = None
        best_score = -np.inf
        convergence_success = False
        
        self.logger.info(f"Starting HMM fitting with {n_states} states (3 attempts)...")
        
        for i in range(3):
            self.logger.info(f"Attempt {i+1}/3 with random_state={42+i}")
            model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=20000,
                tol=1e-3,
                init_params="se",
                params="stmc",
                random_state=42 + i,
                min_covar=0.2,
                implementation='scaling',
                verbose=True
            )

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.logger.info(f"Fitting HMM model (attempt {i+1})...")
                    model.fit(scaled_features)
                    
                converged = model.monitor_.converged
                iterations = model.monitor_.iter
                history = model.monitor_.history
                
                try:
                    history_list = list(history)
                except TypeError:
                    history_list = [history] if isinstance(history, (int, float)) else []
                
                final_log_likelihood = history_list[-1] if history_list else float('-inf')
                
                self.logger.info(
                    f"Attempt {i+1}: Converged={converged}, Iterations={iterations}, "
                    f"Final Log-Likelihood={final_log_likelihood:.4f}"
                )
                
                if history_list:
                    self.logger.debug(f"Log-likelihood history (first 5): {history_list[:5]}")
                    self.logger.debug(f"Log-likelihood history (last 5): {history_list[-5:]}")
                    
                    min_hist = min(history_list)
                    max_hist = max(history_list)
                    delta_hist = history_list[-1] - history_list[0]
                    self.logger.debug(
                        f"Log-likelihood range: Min={min_hist:.4f}, Max={max_hist:.4f}, "
                        f"Delta={delta_hist:.4f}"
                    )
                    
                    if any(np.isnan(x) for x in history_list) or any(np.isinf(x) for x in history_list):
                        self.logger.warning("History contains NaN/Inf values!")
                        
                    decreasing_count = 0
                    for i in range(1, len(history_list)):
                        if history_list[i] < history_list[i-1]:
                            decreasing_count += 1
                    if decreasing_count > 0:
                        self.logger.warning(f"Log-likelihood decreased {decreasing_count} times during fitting")

                if converged:
                    convergence_success = True
                    model_score = model.score(scaled_features)
                    self.logger.info(f"Converged model score: {model_score:.4f}")
                    
                    if model_score > best_score:
                        best_model = model
                        best_score = model_score
                        self.logger.info(f"New best model found (score={model_score:.4f})")
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
        else:
            self.logger.error("All HMM attempts failed to converge, using simple detection")
            return self.simple_regime_detection(features)

        state_stats = []
        for i in range(model.n_components):
            state_return = float(model.means_[i][0])
            state_vol = float(model.means_[i][1])
            state_stats.append((i, state_return, state_vol))
            
        self.logger.info("State characteristics before sorting:")
        for i, (state_idx, ret, vol) in enumerate(state_stats):
            self.logger.info(f"State {i}: Return={ret:.6f}, Volatility={vol:.6f}")
            
        state_stats.sort(key=lambda x: (x[1], -x[2]))
        
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
        
        self.logger.info("Final state labeling:")
        for state_idx, label in state_labels.items():
            ret, vol = next((s_ret, s_vol) for s_idx, s_ret, s_vol in state_stats if s_idx == state_idx)
            self.logger.info(f"State {state_idx}: {label} (Return={ret:.6f}, Volatility={vol:.6f})")

        states = model.predict(scaled_features)
        state_probs = model.predict_proba(scaled_features)
        
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
        
        avg_durations = {}
        for state, durs in durations.items():
            avg_durations[state] = sum(durs) / len(durs) if durs else 0
            
        return avg_durations

    def fetch_stock_data(self, symbol, days=365):
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
        start_date = datetime.now() - timedelta(days=365)
        dates = pd.date_range(start_date, datetime.now(), freq='D')
        
        prices = [100]
        for i in range(1, len(dates)):
            change = random.uniform(-0.02, 0.03)
            prices.append(prices[-1] * (1 + change))
            
        return pd.Series(prices, index=dates)


# ======================== SECTOR REGIME SYSTEM ======================== #
class SectorRegimeSystem:
    def __init__(self, polygon_api_key=POLYGON_API_KEY, testing_mode=False):
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
        
        if os.path.exists(cache_file):
            try:
                self.sector_mappings = pd.read_pickle(cache_file)
                return self.sector_mappings
            except:
                pass

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
                    
                    sector = data.get("sic_description", "")
                    if not sector or sector == "Unknown":
                        sector = data.get("sector", "")
                    if not sector or sector == "Unknown":
                        sector = data.get("industry", "")
                    if not sector or sector == "Unknown":
                        sector = data.get("primary_exchange", "Unknown")
                    
                    sector = self.clean_sector_name(sector)
                    
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

        self.sector_mappings = {
            k: v for k, v in self.sector_mappings.items() 
            if k != "Unknown" and len(v) > 10
        }
        
        pd.to_pickle(self.sector_mappings, cache_file)
        return self.sector_mappings

    def clean_sector_name(self, sector):
        if not sector or sector == "Unknown":
            return "Unknown"
            
        if sector.startswith("X"):
            return "Unknown"
            
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
        return sector

    def calculate_sector_weights(self):
        if self.testing_mode:
            return self.generate_test_sector_weights()
            
        total_mcap = 0
        sector_mcaps = {}
        
        self.logger.info("Calculating sector weights...")
        for sector, tickers in self.sector_mappings.items():
            sector_mcap = 0
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
        
        if os.path.exists(cache_file):
            try:
                self.sector_composites = pd.read_pickle(cache_file)
                return self.sector_composites
            except:
                pass

        self.logger.info("Building sector composites...")
        for sector, tickers in self.sector_mappings.items():
            prices_data = []
            mcaps = {}
            
            for symbol in tickers[:sample_size]:
                mcaps[symbol] = self.overall_analyzer.get_market_cap(symbol) or 1
            total_mcap = sum(mcaps.values())
            
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
        
        pd.to_pickle(self.sector_composites, cache_file)
        return self.sector_composites

    def analyze_sector_regimes(self, n_states=4):
        if self.testing_mode:
            return self.generate_test_sector_regimes()
            
        self.sector_analyzers = {}
        
        if not hasattr(self, 'market_composite'):
            tickers = [t for sublist in self.sector_mappings.values() for t in sublist]
            self.logger.info("Creating market composite...")
            self.market_composite = self.overall_analyzer.prepare_market_data(tickers[:100])
        self.logger.info("Analyzing overall market regime...")
        market_result = self.overall_analyzer.analyze_regime(self.market_composite)
        self.current_regime = market_result["regimes"][-1]
        
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

        for sector, data in self.sector_analyzers.items():
            try:
                if "results" not in data:
                    continue
                    
                current_probs = data["results"]["probabilities"][-1]
                state_labels = data["results"].get("state_labels", {})
                
                momentum = data["composite"].pct_change(21).iloc[-1] if len(data["composite"]) > 21 else 0
                
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
                
                base_score = bull_prob - bear_prob
                momentum_factor = 1 + (momentum * 5)
                adjusted_score = base_score * momentum_factor
                weight = self.sector_weights.get(sector, 0.01)
                self.sector_scores[sector] = adjusted_score * (1 + weight)
                
            except Exception as e:
                self.logger.error(f"Error calculating score for {sector}: {str(e)}")
                self.sector_scores[sector] = 0

        cleaned_scores = {}
        for raw_sector, score in self.sector_scores.items():
            clean_sector = self.clean_sector_name(raw_sector)
            if clean_sector not in cleaned_scores:
                cleaned_scores[clean_sector] = score
            else:
                cleaned_scores[clean_sector] = max(cleaned_scores[clean_sector], score)
                
        cleaned_scores = {k: v for k, v in cleaned_scores.items() if k != "Unknown"}
        
        self.sector_scores = cleaned_scores
        return pd.Series(self.sector_scores).sort_values(ascending=False)


# ======================== ASSET ALLOCATIONS ======================== #
ASSET_ALLOCATIONS = {
    "Bear": {
        "defensive_stocks": 70,
        "dividend_stocks": 30
    },
    "Severe Bear": {
        "inverse_etfs": 40,
        "defensive_stocks": 40,
        "cash": 20
    },
    "Bull": {
        "growth_stocks": 60,
        "tech_stocks": 30,
        "small_caps": 10
    },
    "Strong Bull": {
        "growth_stocks": 75,
        "tech_stocks": 20,
        "small_caps": 5
    },
    "Neutral": {
        "value_stocks": 50,
        "dividend_stocks": 40,
        "cash": 10
    },
    "Mild Bear": {
        "defensive_stocks": 60,
        "dividend_stocks": 40
    },
    "Mild Bull": {
        "growth_stocks": 60,
        "value_stocks": 35,
        "cash": 5
    }
}
# ======================== END MARKET REGIME CODE ======================== #

class SmartStopLoss:
    def __init__(self, entry_price, atr, adx, market_volatility=None, regime=None,
                 base_vol_factor=1.5, base_hard_stop=0.08, profit_target_ratio=3.0):
        self.entry_price = entry_price
        self.atr = atr
        self.adx = adx
        self.regime = regime or "Neutral"
        self.creation_time = datetime.now(tz.utc)
        
        if market_volatility is None:
            market_volatility = 0.15
        
        vol_deviation = market_volatility - 0.15
        volatility_factor = base_vol_factor
        hard_stop_percent = base_hard_stop
        
        if market_volatility is not None:
            volatility_factor = base_vol_factor * (1 + vol_deviation * 3)
            hard_stop_percent = base_hard_stop * (1 + vol_deviation * 2)
        
        if regime is not None:
            if "Bear" in regime:
                volatility_factor *= 1.3
                hard_stop_percent *= 0.9
                profit_target_ratio *= 0.8
            elif "Bull" in regime:
                volatility_factor *= 0.8
                hard_stop_percent *= 1.1
                profit_target_ratio *= 1.2
        
        if adx > 40:
            profit_target_ratio *= 1.3
        elif adx < 20:
            profit_target_ratio *= 0.7
            
        self.volatility_factor = max(1.0, min(2.5, volatility_factor))
        self.hard_stop_percent = max(0.05, min(0.15, hard_stop_percent))
        self.profit_target_ratio = max(1.5, min(5.0, profit_target_ratio))
        
        self.initial_stop = self.calculate_initial_stop()
        self.trailing_stop = self.initial_stop
        self.hard_stop = entry_price * (1 - self.hard_stop_percent)
        self.profit_target = entry_price + (entry_price - self.initial_stop) * self.profit_target_ratio
        self.profit_target_2 = entry_price + (entry_price - self.initial_stop) * (self.profit_target_ratio * 1.8)
        
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
        
        if self.adx > 40:
            return base_stop * 0.95
        elif self.adx < 20:
            return base_stop * 1.05
        return base_stop
    
    def update_trailing_stop(self, current_price, timestamp):
        sensitivity = 0.8
        
        if self.regime is not None:
            if "Bear" in self.regime:
                sensitivity = 0.65
            elif "Bull" in self.regime and self.adx > 30:
                sensitivity = 0.95
        
        new_stop = current_price - self.atr * self.volatility_factor * sensitivity
        
        holding_days = (datetime.now(tz.utc) - self.creation_time).days
        if holding_days > 3 and current_price > self.profit_target:
            relaxation = 1 - (0.005 * (holding_days - 3))
            new_stop *= relaxation
        
        if new_stop > self.trailing_stop:
            self.trailing_stop = new_stop
        
        self.trailing_stop = max(self.trailing_stop, self.hard_stop)
        
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
    def __init__(self, base_lookback=35, max_extended=180):
        self.base = base_lookback
        self.max_extended = max_extended
        self.position_age = {}

    def get_lookback(self, ticker, volatility, trend_strength, is_open_position=False):
        lookback = self.base
        
        if is_open_position:
            days_held = self.position_age.get(ticker, 0)
            lookback = min(self.max_extended, self.base + days_held * 2)
        
        vol_deviation = volatility - 0.15
        volatility_factor = 1 + vol_deviation
        volatility_factor = max(0.7, min(1.5, volatility_factor))
        lookback = lookback * volatility_factor
        
        if trend_strength > 40:
            lookback = max(14, lookback * 0.8)
        elif trend_strength < 20:
            lookback = min(self.max_extended, lookback * 1.3)
        
        lookback = max(14, lookback)
        return int(round(lookback))

class DataHandler:
    """Base class for data handlers"""
    def __init__(self, tickers):
        self.tickers = tickers
        self.historical_data = {t: pd.DataFrame() for t in tickers}
        self.realtime_data = {t: None for t in tickers}
        self.data_queue = queue.Queue()
        self.running = True
        self.thread = None
        self.data_lock = threading.Lock()
        self.logger = logging.getLogger("DataHandler")
        self.logger.setLevel(logging.INFO)
    
    def start(self):
        raise NotImplementedError("Subclasses must implement start method")
    
    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        self.logger.info("Data feed stopped")
    
    def get_latest(self, ticker):
        if ticker in self.realtime_data and self.realtime_data[ticker] is not None:
            return self.realtime_data[ticker]
        
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

class PolygonDataHandler(DataHandler):
    """Handles Polygon.io REST API and WebSocket data"""
    def __init__(self, tickers, testing_mode=False):
        super().__init__(tickers)
        self.api_key = POLYGON_API_KEY
        self.testing_mode = testing_mode
    
    def load_historical_data(self):
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
        try:
            event_type = event.get('ev')
            ticker = event.get('sym')
            
            if not ticker or ticker not in self.tickers:
                return
                
            if event_type == 'AM':  # Aggregate Minute
                timestamp = datetime.fromtimestamp(event['s'] / 1000.0, tz=tz.utc)
                new_bar = pd.DataFrame({
                    'open': [event['o']], 'high': [event['h']], 
                    'low': [event['l']], 'close': [event['c']], 
                    'volume': [event['v']]
                }, index=[timestamp])
                
                with self.data_lock:
                    if not self.historical_data[ticker].empty:
                        self.historical_data[ticker] = pd.concat([self.historical_data[ticker], new_bar])
                        self.historical_data[ticker] = self.historical_data[ticker][~self.historical_data[ticker].index.duplicated(keep='last')]
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
        if not self.thread or not self.thread.is_alive():
            self.load_historical_data()
            self.running = True
            self.thread = threading.Thread(target=self.run_websocket)
            self.thread.daemon = True
            self.thread.start()
            self.logger.info("Data feed started")
        else:
            self.logger.info("Data feed already running")
    
    def generate_test_historical_data(self):
        self.logger.info("Generating test historical data...")
        for ticker in self.tickers:
            start_date = datetime.now(tz.utc) - timedelta(days=30)
            dates = pd.date_range(start_date, datetime.now(tz.utc), freq='1min')
            
            prices = [100]
            for i in range(1, len(dates)):
                change = random.uniform(-0.001, 0.002)
                prices.append(prices[-1] * (1 + change))
                
            df = pd.DataFrame({
                'open': prices,
                'high': [p * 1.001 for p in prices],
                'low': [p * 0.999 for p in prices],
                'close': prices,
                'volume': [random.randint(1000, 10000) for _ in prices]
            }, index=dates)
            
            self.historical_data[ticker] = df
            
    def generate_test_ticker_data(self, ticker):
        start_date = datetime.now(tz.utc) - timedelta(days=30)
        dates = pd.date_range(start_date, datetime.now(tz.utc), freq='1min')
        
        prices = [100]
        for i in range(1, len(dates)):
            change = random.uniform(-0.001, 0.002)
            prices.append(prices[-1] * (1 + change))
            
        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.001 for p in prices],
            'low': [p * 0.999 for p in prices],
            'close': prices,
            'volume': [random.randint(1000, 10000) for _ in prices]
        }, index=dates)
        
        return df
        
    def run_test_websocket(self):
        self.logger.info("Starting test WebSocket simulator")
        while self.running:
            for ticker in self.tickers:
                last_data = self.get_latest(ticker)
                if last_data:
                    last_price = last_data['close']
                else:
                    last_price = random.uniform(100, 200)
                    
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
                
                with self.data_lock:
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
                
                self.realtime_data[ticker] = new_data
                self.data_queue.put((ticker, new_data))
                
            time.sleep(1)

class BacktestDataHandler(DataHandler):
    """Backtest data handler that uses your exact data fetching method"""
    def __init__(self, tickers, start_date, end_date):
        super().__init__(tickers)
        self.start_date = start_date
        self.end_date = end_date
        self.current_time = start_date
        self.market_calendar = mcal.get_calendar('NYSE')
        self.schedule = self.market_calendar.schedule(start_date, end_date)
        self.all_data = {}
        
    def load_all_data(self):
        """Load all historical data using your original method"""
        self.logger.info(f"Loading historical data for backtesting ({self.start_date} to {self.end_date})")
        for ticker in self.tickers:
            df = self.load_ticker_data(ticker)
            
            if df is None or df.empty:
                continue
                
            # Convert to UTC if necessary
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
                
            df = df.loc[self.start_date:self.end_date]
            
            if not df.empty:
                self.all_data[ticker] = df
                self.logger.info(f"Loaded {len(df)} data points for {ticker}")
            else:
                self.logger.warning(f"No data found for {ticker} in backtest period")
    
    def load_ticker_data(self, ticker):
        """Your exact data fetching method for both live and backtesting"""
        cache_file = f"data_cache/{ticker}_backtest.pkl"
        if os.path.exists(cache_file):
            return pd.read_pickle(cache_file)
            
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/" \
              f"{self.start_date.strftime('%Y-%m-%d')}/{self.end_date.strftime('%Y-%m-%d')}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": POLYGON_API_KEY,
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 429:
                time.sleep(0.5)
                return self.load_ticker_data(ticker)
                
            if response.status_code != 200:
                return None

            data = response.json()
            if 'results' not in data or not data['results']:
                return None
                
            df = pd.DataFrame(data['results'])
            df["date"] = pd.to_datetime(df["t"], unit="ms")
            result = df.set_index("date")[["o", "h", "l", "c", "v"]]
            result.columns = ['open', 'high', 'low', 'close', 'volume']
            
            result.to_pickle(cache_file)
            return result
            
        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout fetching {ticker}, skipping")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching {ticker}: {str(e)}")
            return None
    
    def run_websocket(self):
        """Simulate real-time data flow using historical data"""
        self.logger.info("Starting backtest data simulation")
        business_days = self.schedule.index
        
        all_timestamps = []
        for date in business_days:
            market_open = self.schedule.loc[date].market_open.astimezone(pytz.utc)
            market_close = self.schedule.loc[date].market_close.astimezone(pytz.utc)
            current = market_open
            while current <= market_close:
                all_timestamps.append(current)
                current += timedelta(minutes=1)
        
        for timestamp in all_timestamps:
            if not self.running:
                break
                
            self.current_time = timestamp
            for ticker in self.tickers:
                if ticker in self.all_data:
                    try:
                        # Get daily data and resample to minute
                        daily_data = self.all_data[ticker]
                        if timestamp in daily_data.index:
                            data_point = daily_data.loc[timestamp]
                        else:
                            # Find the most recent data point
                            previous_data = daily_data[daily_data.index <= timestamp]
                            if not previous_data.empty:
                                data_point = previous_data.iloc[-1]
                            else:
                                continue
                        
                        realtime_data = {
                            'open': data_point['open'],
                            'high': data_point['high'],
                            'low': data_point['low'],
                            'close': data_point['close'],
                            'volume': data_point['volume'],
                            'timestamp': timestamp
                        }
                        
                        # Update historical data
                        if not self.historical_data[ticker].empty:
                            new_bar = pd.DataFrame([realtime_data], index=[timestamp])
                            self.historical_data[ticker] = pd.concat([self.historical_data[ticker], new_bar])
                        else:
                            self.historical_data[ticker] = pd.DataFrame([realtime_data], index=[timestamp])
                        
                        # Update realtime data
                        self.realtime_data[ticker] = realtime_data
                        self.data_queue.put((ticker, realtime_data))
                    except KeyError:
                        pass
            
            time.sleep(0.01)
    
    def start(self):
        self.load_all_data()
        self.running = True
        self.thread = threading.Thread(target=self.run_websocket)
        self.thread.daemon = True
        self.thread.start()
        self.logger.info("Backtest data simulation started")

# ======================== TRADING SYSTEM ======================== #
class TradingSystem(QThread):
    """Complete trading system for both live and backtest"""
    POSITION_EVALUATION_INTERVAL = 3600
    MIN_HOLDING_DAYS = 5
    SCORE_DEGRADATION_THRESHOLD = 0.8
    RELATIVE_STRENGTH_MARGIN = 0.15
    MIN_SCORE_FOR_ENTRY = 70
    
    update_signal = pyqtSignal(dict)
    log_signal = pyqtSignal(str)
    scan_requested = pyqtSignal()
    
    def __init__(self, tickers, capital=100000, risk_per_trade=0.01, 
                 testing_mode=False, backtest_mode=False, 
                 start_date=None, end_date=None):
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
        self.backtest_mode = backtest_mode
        self.scan_requested_flag = False
        self.position_evaluation_times = {}
        self.runner_ups = []
        self.sector_system = SectorRegimeSystem(testing_mode=testing_mode)
        self.executed_tickers = set()
        self.last_evaluation_timestamp = 0
        self.market_regime = "Neutral"
        self.sector_scores = {}
        self.regime_last_updated = 0
        self.regime_analysis_interval = 3600 * 6
        self.market_volatility = 0.15
        self.lookback_system = AdaptiveLookbackSystem(base_lookback=35, max_extended=180)
        self.last_update_date = datetime.now(tz.utc).date()
        self.logger = logging.getLogger("TradingSystem")
        self.logger.setLevel(logging.INFO)
        self.scan_requested.connect(self.force_scan)
        
        # Backtest specific initialization
        if backtest_mode:
            if not start_date or not end_date:
                raise ValueError("Start and end dates required for backtest")
                
            # Ensure timezone-aware dates
            if start_date.tzinfo is None:
                start_date = pytz.utc.localize(start_date)
            if end_date.tzinfo is None:
                end_date = pytz.utc.localize(end_date)
                
            self.start_date = start_date
            self.end_date = end_date
            self.clock = BacktestClock(start_date, end_date)
            self.data_handler = BacktestDataHandler(tickers, start_date, end_date)
        else:
            self.data_handler = PolygonDataHandler(tickers, testing_mode=testing_mode)
        
        print(f"Trading system initialized (Testing Mode: {testing_mode}, Backtest Mode: {backtest_mode})")
        
    def ensure_scalar(self, value):
        if isinstance(value, pd.Series):
            return value.iloc[-1] if not value.empty else 0
        if isinstance(value, pd.DataFrame):
            return value.iloc[-1, -1] if not value.empty else 0
        return value
        
    def force_scan(self):
        self.scan_requested_flag = True
        self.log_signal.emit("Manual scan requested by user")
        
    def run(self):
        if not self.running:
            return
            
        self.log_signal.emit("Trading system started")
        self.data_handler.start()
        self.update_market_regime()
        self.regime_last_updated = time.time()
        
        while self.running:
            try:
                if self.scan_requested_flag:
                    self.scan_requested_flag = False
                    self.evaluate_opportunities()
                    self.enter_top_opportunities()
                    self.log_signal.emit("Manual scan completed")
                
                while not self.data_handler.data_queue.empty() and self.running:
                    ticker, data = self.data_handler.data_queue.get()
                    self.update_positions(ticker, data)
                
                self.close_old_positions()
                
                if self.should_evaluate_opportunities():
                    self.evaluate_opportunities()
                    self.enter_top_opportunities()
                
                current_time = time.time()
                if current_time - self.last_evaluation_timestamp > self.POSITION_EVALUATION_INTERVAL:
                    self.evaluate_and_replace_positions()
                    self.last_evaluation_timestamp = current_time
                
                if current_time - self.regime_last_updated > self.regime_analysis_interval:
                    self.update_market_regime()
                    self.regime_last_updated = current_time
                
                now = datetime.now(tz.utc)
                current_date = now.date()
                if current_date != self.last_update_date:
                    self.daily_update()
                    self.last_update_date = current_date
                
                self.update_signal.emit(self.get_current_state())
                
                # Backtest specific: Advance clock
                if self.backtest_mode:
                    # Capture backtest results when complete
                    results = {
                        'initial_capital': self.initial_capital,
                        'capital': self.capital,
                        'total_profit': self.total_profit,
                        'trade_count': len(self.trade_log),
                        'win_rate': self.win_rate,
                        'trade_log': self.trade_log,
                        'equity_curve': self.equity_curve,
                        'backtest_mode': True
                    }
                self.update_signal.emit(results)
                
            except Exception as e:
                self.log_signal.emit(f"System error: {str(e)}")
        
        if self.data_handler:
            self.data_handler.stop()
        self.log_signal.emit("Trading system stopped")
    
    def daily_update(self):
        for ticker in self.positions.keys():
            if ticker in self.lookback_system.position_age:
                self.lookback_system.position_age[ticker] += 1
            else:
                self.lookback_system.position_age[ticker] = 1
        self.log_signal.emit("Daily position age updated")
    
    def update_market_regime(self):
        try:
            self.log_signal.emit("Starting market regime analysis...")
            self.sector_system.map_tickers_to_sectors(self.tickers)
            self.sector_system.calculate_sector_weights()
            self.sector_system.build_sector_composites()
            self.sector_system.analyze_sector_regimes()
            scores = self.sector_system.calculate_sector_scores()
            self.market_regime = self.sector_system.current_regime
            self.sector_scores = scores.to_dict()
            
            if not self.sector_system.market_composite.empty:
                try:
                    composite_df = pd.DataFrame({
                        'open': self.sector_system.market_composite,
                        'high': self.sector_system.market_composite,
                        'low': self.sector_system.market_composite,
                        'close': self.sector_system.market_composite
                    })
                    atr = composite_df.ta.atr(length=14).iloc[-1]
                    current_price = composite_df['close'].iloc[-1]
                    self.market_volatility = self.ensure_scalar(atr / current_price)
                except Exception as e:
                    self.log_signal.emit(f"Volatility update error: {str(e)}")
                    self.market_volatility = 0.15
            
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
                
            if not self.testing_mode and not self.backtest_mode:
                self.send_discord_notification()
            
        except Exception as e:
            self.log_signal.emit(f"Market regime analysis failed: {str(e)}")
    
    def send_discord_notification(self):
        DISCORD_WEBHOOK_URL = "YOUR_DISCORD_WEBHOOK_URL"
        if not DISCORD_WEBHOOK_URL:
            return
            
        try:
            regime = self.market_regime
            sector_scores = self.sector_scores
            
            content = f"**Market Analysis Update**\nCurrent Regime: **{regime}**"
            
            fields = []
            
            allocation = ASSET_ALLOCATIONS.get(regime, {})
            alloc_text = "\n".join([f"- {asset.replace('_', ' ').title()}: {pct}%" for asset, pct in allocation.items()])
            fields.append({"name": "Recommended Allocation", "value": alloc_text, "inline": True})
            
            top_sectors = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            sectors_text = "\n".join([f"- {sector}: {score:.2f}" for sector, score in top_sectors])
            fields.append({"name": "Top Performing Sectors", "value": sectors_text, "inline": True})
            
            embed = {
                "title": "Market Analysis Report",
                "color": 0x3498db,
                "fields": fields,
                "timestamp": datetime.utcnow().isoformat()
            }
            
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
        for ticker in list(self.positions.keys()):
            position = self.positions[ticker]
            if self.backtest_mode:
                duration = (self.clock.current_time - position['entry_time']).seconds / 60
            else:
                duration = (datetime.now(tz.utc) - position['entry_time']).seconds / 60
            if duration > 240:
                data = self.data_handler.get_latest(ticker)
                if data:
                    self.exit_position(ticker, data['close'], "time expiration")
    
    def should_evaluate_opportunities(self):
        current_time = time.time()
        return (
            (current_time - self.last_evaluation_time > self.evaluation_interval) and
            (len(self.positions) < 5) and 
            self.is_market_open()
        )
    
    def evaluate_opportunities(self):
        self.last_evaluation_time = time.time()
        opportunities = []
        tickers_to_score = [t for t in self.tickers if t not in self.positions]
        
        if not tickers_to_score:
            self.log_signal.emit("No tickers to evaluate (all in positions)")
            return
            
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
        self.runner_ups = opportunities[:10]
        self.log_signal.emit(f"Parallel evaluated {len(opportunities)} opportunities")
    
    def enter_top_opportunities(self):
        if not self.last_opportunities:
            self.log_signal.emit("No opportunities to enter")
            return
            
        for opp in self.last_opportunities[:1]:
            if opp['score'] >= self.MIN_SCORE_FOR_ENTRY and opp['ticker'] not in self.positions:
                data = self.data_handler.get_latest(opp['ticker'])
                if not data:
                    self.log_signal.emit(f"No data for {opp['ticker']}")
                    continue
                    
                df = self.data_handler.get_historical(opp['ticker'], 50)
                if df.empty:
                    self.log_signal.emit(f"Insufficient data for {opp['ticker']}")
                    continue
                    
                current_price = data['close']
                atr = self.ensure_scalar(df.ta.atr(length=14).iloc[-1])
                
                try:
                    adx = df.ta.adx(length=14)['ADX_14'].iloc[-1]
                    adx = self.ensure_scalar(adx)
                except Exception as e:
                    self.log_signal.emit(f"ADX calculation failed for {opp['ticker']}: {str(e)}")
                    adx = 20.0
                
                if hasattr(self, 'market_volatility') and hasattr(self, 'market_regime'):
                    stop_system = SmartStopLoss(
                        entry_price=current_price,
                        atr=atr,
                        adx=adx,
                        market_volatility=self.market_volatility,
                        regime=self.market_regime
                    )
                else:
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
                    opp['score'],
                    stop_system
                )
            else:
                if opp['ticker'] in self.positions:
                    self.log_signal.emit(f"Already in position for {opp['ticker']}")
                else:
                    self.log_signal.emit(f"Score too low for {opp['ticker']}: {opp['score']} < {self.MIN_SCORE_FOR_ENTRY}")
    
    def is_market_open(self):
        if self.testing_mode or self.backtest_mode:
            return True
            
        now = datetime.now(self.eastern)
        if now.weekday() >= 5:
            return False
            
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now <= market_close
    
    def get_current_state(self):
        total_profit = sum(trade['profit'] for trade in self.trade_log if trade['profit'] is not None)
        total_profit = self.ensure_scalar(total_profit)
        
        winning_trades = sum(1 for trade in self.trade_log if trade['profit'] and trade['profit'] > 0)
        win_rate = winning_trades / len(self.trade_log) * 100 if self.trade_log else 0
        win_rate = self.ensure_scalar(win_rate)
        
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
        
        recent_trades = self.trade_log[-5:][::-1]
        
        opportunities = []
        for opp in self.last_opportunities:
            status = "ENTERED" if opp['ticker'] in self.positions else "PASSED"
            opportunities.append({
                'ticker': opp['ticker'], 'score': opp['score'],
                'price': opp['price'], 'atr': opp['atr'], 'adx': opp['adx'],
                'rsi': opp['rsi'], 'volume': opp['volume'], 'status': status,
                'lookback': opp.get('lookback_days', 35)
            })
        
        runner_ups = []
        for opp in self.runner_ups:
            runner_ups.append({
                'ticker': opp['ticker'], 'score': opp['score'],
                'price': opp['price'], 'adx': opp['adx'],
                'atr': opp['atr'], 'rsi': opp['rsi'],
                'volume': opp['volume'],
                'lookback': opp.get('lookback_days', 35)
            })
        
        timestamp = datetime.now(tz.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        if self.backtest_mode:
            timestamp = self.clock.current_time.strftime('%Y-%m-%d %H:%M:%S %Z')
        
        return {
            'timestamp': timestamp,
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
            'testing_mode': self.testing_mode,
            'backtest_mode': self.backtest_mode
        }
    
    def score_trade_opportunity(self, ticker):
        try:
            if not self.data_handler:
                return None
                
            if not self.is_market_open() and not (self.testing_mode or self.backtest_mode):
                return None
                
            is_open_position = ticker in self.positions
            
            base_lookback_days = 35
            base_lookback_minutes = base_lookback_days * 390
            base_data = self.data_handler.get_historical(ticker, base_lookback_minutes)
            if base_data is None or base_data.empty:
                self.log_signal.emit(f"Insufficient base data for {ticker}")
                return None
                
            try:
                base_adx = base_data.ta.adx(length=14)['ADX_14'].iloc[-1]
                base_adx = self.ensure_scalar(base_adx)
                if np.isnan(base_adx):
                    base_adx = 20.0
            except:
                base_adx = 20.0
                
            lookback_days = self.lookback_system.get_lookback(
                ticker, 
                self.market_volatility, 
                base_adx,
                is_open_position
            )
            
            if lookback_days != base_lookback_days:
                data = self.data_handler.get_historical(ticker, lookback_days * 390)
            else:
                data = base_data
                
            if data is None or data.empty:
                return None
                
            atr = self.ensure_scalar(data.ta.atr(length=14).iloc[-1])
            
            try:
                adx = data.ta.adx(length=14)['ADX_14'].iloc[-1]
                adx = self.ensure_scalar(adx)
            except Exception as e:
                self.log_signal.emit(f"ADX calculation failed for {ticker}: {str(e)}")
                adx = 20.0
                
            rsi = self.ensure_scalar(data.ta.rsi(length=14).iloc[-1])
            
            latest = self.data_handler.get_latest(ticker)
            if not latest:
                return None
            price = latest['close']
            volume = latest['volume']
            
            avg_volume = data['volume'].rolling(14).mean().iloc[-1]
            avg_volume = self.ensure_scalar(avg_volume)
            if np.isnan(avg_volume) or avg_volume <= 0:
                avg_volume = self.ensure_scalar(data['volume'].mean())
                
            if volume <= 0:
                volume = avg_volume
                
            if hasattr(self, 'market_volatility') and hasattr(self, 'market_regime'):
                stop_system = SmartStopLoss(
                    entry_price=price,
                    atr=atr,
                    adx=adx,
                    market_volatility=self.market_volatility,
                    regime=self.market_regime
                )
            else:
                stop_system = SmartStopLoss(
                    entry_price=price,
                    atr=atr,
                    adx=adx
                )
            
            adx_score = min(100, max(0, (adx - 20) * 5))
            
            atr_pct = atr / price
            if atr_pct < 0.015:
                atr_score = 20 + (atr_pct / 0.015) * 30
            elif atr_pct > 0.03:
                atr_score = 80 - min(30, (atr_pct - 0.03) * 1000)
            else:
                atr_score = 50 + (atr_pct - 0.015) * 2000
                
            risk = price - stop_system.initial_stop
            reward = stop_system.profit_target - price
            rr_ratio = reward / risk if risk > 0 else 0
            rr_score = min(100, rr_ratio * 25)
            
            volume_ratio = volume / avg_volume
            volume_score = min(100, volume_ratio * 50)
            
            if rsi > 70:
                rsi_score = 100 - min(30, (rsi - 70) * 2)
            elif rsi < 30:
                rsi_score = 100 - min(30, (30 - rsi) * 2)
            else:
                rsi_score = 80 - abs(rsi - 50)
            
            sector = self.get_ticker_sector(ticker)
            sector_score = self.sector_scores.get(sector, 50)
            sector_factor = 1 + (sector_score - 50) / 100
            
            regime_factor = self.get_regime_factor()
            
            composite_score = (
                0.30 * adx_score + 0.25 * atr_score + 
                0.20 * rr_score + 0.15 * volume_score + 
                0.10 * rsi_score
            ) * sector_factor * regime_factor
            
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
        for sector, tickers in self.sector_system.sector_mappings.items():
            if ticker in tickers:
                return sector
        return "Unknown"
    
    def get_regime_factor(self):
        if "Bull" in self.market_regime:
            return 1.2
        elif "Bear" in self.market_regime:
            return 0.8
        return 1.0
    
    def enter_position(self, ticker, price, atr, adx, original_score, stop_system):
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
                
            entry_time = datetime.now(tz.utc)
            if self.backtest_mode:
                entry_time = self.clock.current_time
                
            self.positions[ticker] = {
                'entry_price': price,
                'entry_time': entry_time,
                'shares': position_size,
                'stop_system': stop_system,
                'original_score': original_score
            }
            
            self.lookback_system.position_age[ticker] = 0
            
            self.trade_log.append({
                'ticker': ticker, 'entry': price,
                'entry_time': entry_time,
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
            
            if self.backtest_mode:
                duration = (self.clock.current_time - entry_time).total_seconds() / 60
            else:
                duration = (datetime.now(tz.utc) - entry_time).total_seconds() / 60
                
            self.capital += profit
            
            if ticker in self.lookback_system.position_age:
                del self.lookback_system.position_age[ticker]
            
            for trade in reversed(self.trade_log):
                if trade['ticker'] == ticker and trade['exit'] is None:
                    trade['exit'] = exit_price
                    if self.backtest_mode:
                        trade['exit_time'] = self.clock.current_time
                    else:
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
            
            if self.backtest_mode:
                exit_time = self.clock.current_time
                duration = (exit_time - position['entry_time']).total_seconds() / 60
            else:
                exit_time = datetime.now(tz.utc)
                duration = (exit_time - position['entry_time']).total_seconds() / 60
            
            self.trade_log.append({
                'ticker': ticker,
                'entry': position['entry_price'],
                'entry_time': position['entry_time'],
                'exit': exit_price,
                'exit_time': exit_time,
                'profit': profit,
                'percent_gain': percent_gain,
                'duration': duration,
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
    
    def evaluate_and_replace_positions(self):
        if not self.positions or not self.runner_ups:
            return
            
        self.log_signal.emit("Evaluating position strength...")
        current_prices = self.get_current_prices()
        
        for ticker, position in list(self.positions.items()):
            holding_days = self.lookback_system.position_age.get(ticker, 0)
            if holding_days < self.MIN_HOLDING_DAYS:
                continue
                
            current_score = self.calculate_current_score(ticker, position, current_prices[ticker])
            
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
        prices = {}
        for ticker in self.positions:
            if not self.data_handler:
                prices[ticker] = self.positions[ticker]['entry_price']
                continue
                
            latest = self.data_handler.get_latest(ticker)
            prices[ticker] = latest['close'] if latest else self.positions[ticker]['entry_price']
        return prices
    
    def calculate_current_score(self, ticker, position, current_price):
        try:
            if not self.data_handler:
                return 0
                
            df = self.data_handler.get_historical(ticker, 50)
            if df is None or df.empty:
                return 0
                
            try:
                adx = df.ta.adx(length=14)['ADX_14'].iloc[-1]
                adx = self.ensure_scalar(adx)
            except Exception as e:
                self.log_signal.emit(f"ADX calculation failed for {ticker}: {str(e)}")
                adx = 20.0
                
            rsi = self.ensure_scalar(df.ta.rsi(length=14).iloc[-1])
            volume = self.ensure_scalar(df['volume'].iloc[-1])
            avg_volume = df['volume'].rolling(14).mean().iloc[-1]
            avg_volume = self.ensure_scalar(avg_volume)
            
            price_change = ((current_price - position['entry_price']) / position['entry_price']) * 100
            volume_ratio = volume / avg_volume
            
            regime_factor = self.get_regime_factor()
            
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
        self.log_signal.emit(f"Seeking replacement for {weak_ticker} (Score: {weak_score:.1f})")
        best_candidate = None
        best_score = weak_score
        
        for candidate in self.runner_ups:
            ticker = candidate['ticker']
            if ticker in self.positions or ticker in self.executed_tickers:
                continue
                
            candidate_score = self.score_trade_opportunity(ticker)
            if not candidate_score:
                continue
                
            if candidate_score['score'] > best_score * (1 + self.RELATIVE_STRENGTH_MARGIN):
                best_candidate = candidate_score
                best_score = candidate_score['score']
        
        if not best_candidate:
            best_candidate = self.find_replacement_from_scan(weak_score)
        
        if best_candidate:
            self.log_signal.emit(f"Replacing {weak_ticker} with {best_candidate['ticker']}")
            self.execute_replacement(weak_ticker, best_candidate)
    
    def find_replacement_from_scan(self, min_score):
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
            
        sector_scores = self.sector_scores
        for opp in opportunities:
            sector = self.get_ticker_sector(opp['ticker'])
            sector_strength = sector_scores.get(sector, 50)
            opp['score'] *= (1 + sector_strength / 200)
        
        return max(opportunities, key=lambda x: x['score'])
    
    def execute_replacement(self, old_ticker, new_candidate):
        position = self.positions.get(old_ticker)
        if position:
            current_price = self.data_handler.get_latest(old_ticker)['close']
            self.exit_position(old_ticker, current_price, "Replaced by stronger candidate")
            
            new_data = self.data_handler.get_latest(new_candidate['ticker'])
            if new_data:
                df = self.data_handler.get_historical(new_candidate['ticker'], 50)
                if df is None or df.empty:
                    return
                
                atr = self.ensure_scalar(df.ta.atr(length=14).iloc[-1])
                
                try:
                    adx = df.ta.adx(length=14)['ADX_14'].il极客时间[-1]
                    adx = self.ensure_scalar(adx)
                except Exception as e:
                    self.log_signal.emit(f"ADX calculation failed for {new_candidate['ticker']}: {str(e)}")
                    adx = 20.0
                
                current_price = new_data['close']
                
                if hasattr(self, 'market_volatility') and hasattr(self, 'market_regime'):
                    stop_system = SmartStopLoss(
                        entry_price=current_price,
                        atr=atr,
                        adx=adx,
                        market_volatility=self.market_volatility,
                        regime=self.market_regime
                    )
                else:
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
                    new_candidate['score'],
                    stop_system
                )
                self.log_signal.emit(f"Replaced {old_ticker} with {new_candidate['ticker']}")
    
    def start_system(self):
        if not self.running:
            self.running = True
            self.start()
            return True
        return False
    
    def stop_system(self):
        if self.running:
            self.log_signal.emit("Stopping trading system...")
            self.running = False
            self.wait(5000)
            return True
        return False

# ======================== BACKTEST CLOCK ======================== #
class BacktestClock:
    def __init__(self, start_date, end_date):
        # Ensure timezone-aware dates
        if start_date.tzinfo is None:
            start_date = pytz.utc.localize(start_date)
        if end_date.tzinfo is None:
            end_date = pytz.utc.localize(end_date)
            
        # Convert to Eastern time
        eastern = pytz.timezone('US/Eastern')
        self.start_date = start_date.astimezone(eastern)
        self.end_date = end_date.astimezone(eastern)
        
        self.market_calendar = mcal.get_calendar('NYSE')
        self.schedule = self.market_calendar.schedule(
            self.start_date, 
            self.end_date
        )
        self.market_days = self.schedule.index
        self.current_time = self.start_date
        
    def advance(self, minutes=1):
        self.current_time += timedelta(minutes=minutes)
        
        # If we've moved to a new day, update market open/close times
        if self.current_time.date() != (self.current_time - timedelta(minutes=minutes)).date():
            day_schedule = self.schedule.loc[self.schedule.index.date == self.current_time.date()]
            if not day_schedule.empty:
                self.market_open = day_schedule.iloc[0].market_open
                self.market_close = day_schedule.iloc[0].market_close
                
    def is_market_open(self):
        """Check if market is open at current simulated time"""
        if not hasattr(self, 'market_open'):
            return False
        return self.market_open <= self.current_time <= self.market_close

# ======================== TRADING DASHBOARD ======================== #
class PositionPlot(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.f极客时间.add_subplot(111)
        self.fig.tight_layout()
        
    def plot_position(self, ticker, position, historical_data):
        self.ax.clear()
        
        if historical_data is None or historical_data.empty:
            return
            
        df = historical_data.iloc[-50:]
        stop_system = position['stop_system']
        stop_history = pd.DataFrame(stop_system.history)
        stop_history.set_index('timestamp', inplace=True)
        
        df['close'].plot(ax=self.ax, label='Price', color='blue', linewidth=2)
        stop_history['initial_stop'].plot(ax=self.ax, label='Initial Stop', color='red', linestyle='--')
        stop_history['trailing_stop'].plot(ax=self.ax, label='Trailing Stop', color='orange', linewidth=2)
        stop_history['hard_stop'].plot(ax=self.ax, label='Hard Stop', color='darkred', linestyle=':')
        stop_history['profit_target'].plot(ax=self.ax, label='Profit Target', color='green', linestyle='--')
        
        self.ax.axhline(y=position['entry_price'], color='gray', linestyle='-', alpha=0.5)
        self.ax.annotate('Entry', (stop_history.index[0], position['entry_price']),
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        self.ax.set_title(f'{ticker} Position Analysis')
        self.ax.set_ylabel('Price')
        self.ax.legend()
        self.ax.grid(True)
        
        current_regime = stop_system.detect_market_regime()
        self.ax.annotate(f"Regime: {current_regime.upper()}", 
                        xy=(0.02, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8))
        
        self.draw()

class BacktestProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Running Backtest")
        self.setGeometry(200, 200, 400, 100)
        layout = QVBoxLayout()
        
        self.progress_label = QLabel("Starting backtest...")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)
    
    def update_progress_from_log(self, message):
        try:
            if "Backtest progress:" in message:
                progress_str = message.split(":")[1].replace("%", "").strip()
                progress_value = float(progress_str)
                self.progress_bar.setValue(int(progress_value))
                self.progress_label.setText(message)
        except Exception as e:
            print(f"Error updating progress from log: {str(e)}")

class TradingDashboard(QMainWindow):
    def __init__(self, testing_mode=False):
        super().__init__()
        self.tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "DIS"]
        self.trading_system = None
        self.setWindowTitle("Real-Time Trading Dashboard" + (" - TESTING MODE" if testing_mode else ""))
        self.setGeometry(100, 100, 1600, 900)
        self.testing_mode = testing_mode
        
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger("TradingDashboard")
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
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
        
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("Start System")
        self.stop_button = QPushButton("Stop System")
        self.scan_button = QPushButton("Scan Now")
        
        self.capital_label_ctrl = QLabel("Capital:")
        self.capital_input = QLineEdit("100000")
        self.capital_input.setFixedWidth(100)
        
        self.risk_label_ctrl = QLabel("Risk:")
        self.risk_input = QLineEdit("0.01")
        self.risk_input.setFixedWidth(50)
        
        self.testing_label = QLabel("TESTING MODE" if testing_mode else "LIVE MODE")
        self.testing_label.setStyleSheet("color: red; font-weight: bold;" if testing_mode else "color: green; font-weight: bold;")
        
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.scan_button)
        control_layout.addWidget(self.capital_label_ctrl)
        control_layout.addWidget(self.capital_input)
        control_layout.addWidget(self.risk_label_ctrl)
        control_layout.addWidget(self.risk_input)
        control_layout.addStretch()
        control_layout.addWidget(self.testing_label)
        main_layout.addLayout(control_layout)
        
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        positions_tab = QWidget()
        positions_layout = QVBoxLayout(positions_tab)
        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(13)
        self.positions_table.setHorizontalHeaderLabels([
            "Ticker", "Shares", "Entry", "Current", "Gain%", 
            "Trail Stop", "Hard Stop", "Profit Tgt", "Risk%", "Regime", "Score", "Sector", "Days Held"
        ])
        self.positions_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        positions_layout.addWidget(self.positions_table)
        self.tabs.addTab(positions_tab, "Active Positions")
        
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
        
        plots_tab = QWidget()
        plots_layout = QVBoxLayout(plots_tab) 
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        plots_layout.addWidget(self.plot_container)
        self.tabs.addTab(plots_tab, "Position Analysis")
        
        terminal_tab = QWidget()
        terminal_layout = QVBoxLayout(terminal_tab)
        self.terminal_output = QTextEdit()
        self.terminal_output.setReadOnly(True)
        self.terminal_output.setFont(QFont("Courier", 10))
        self.terminal_output.setStyleSheet("background-color: black; color: #00FF00;")
        terminal_layout.addWidget(self.terminal_output)
        self.tabs.addTab(terminal_tab, "Terminal Log")
        
        self.sector_tab = QWidget()
        self.sector_layout = QVBoxLayout(self.sector_tab)
        self.sector_table = QTableWidget()
        self.sector_table.setColumnCount(2)
        self.sector_table.setHorizontalHeaderLabels(["Sector", "Score"])
        self.sector_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.sector_layout.addWidget(self.sector_table)
        
        self.alloc_label = QLabel("Recommended Asset Allocation:")
        self.alloc_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.sector_layout.addWidget(self.alloc_label)
        
        self.alloc_text = QTextEdit()
        self.alloc_text.setReadOnly(True)
        self.alloc_text.setFont(QFont("Arial", 10))
        self.sector_layout.addWidget(self.alloc_text)
        
        self.tabs.addTab(self.sector_tab, "Market Analysis")
        
        self.backtest_tab = QWidget()
        self.backtest_layout = QVBoxLayout(self.backtest_tab)
        
        config_group = QGroupBox("Backtest Configuration")
        config_layout = QFormLayout()
        
        self.backtest_ticker_input = QLineEdit("AAPL,MSFT,GOOG,AMZN,TSLA")
        self.backtest_start_date = QDateTimeEdit()
        self.backtest_start_date.setDateTime(QDateTime.currentDateTime().addMonths(-1))
        self.backtest_end_date = QDateTimeEdit()
        self.backtest_end_date.setDateTime(QDateTime.currentDateTime())
        self.backtest_capital_input = QLineEdit("100000")
        self.backtest_risk_input = QLineEdit("0.01")
        
        config_layout.addRow("Tickers (comma separated):", self.backtest_ticker_input)
        config_layout.addRow("Start Date:", self.backtest_start_date)
        config_layout.addRow("End Date:", self.backtest_end_date)
        config_layout.addRow("Starting Capital ($):", self.backtest_capital_input)
        config_layout.addRow("Risk per Trade:", self.backtest_risk_input)
        
        self.run_backtest_button = QPushButton("Run Backtest")
        self.run_backtest_button.clicked.connect(self.run_backtest)
        config_layout.addRow(self.run_backtest_button)
        
        config_group.setLayout(config_layout)
        self.backtest_layout.addWidget(config_group)
        
        self.backtest_results_tabs = QTabWidget()
        self.backtest_layout.addWidget(self.backtest_results_tabs)
        
        self.backtest_equity_tab = QWidget()
        self.backtest_equity_layout = QVBoxLayout(self.backtest_equity_tab)
        self.backtest_equity_plot = FigureCanvas(Figure(figsize=(10, 6)))
        self.backtest_equity_layout.addWidget(self.backtest_equity_plot)
        self.backtest_results_tabs.addTab(self.backtest_equity_tab, "Equity Curve")
        
        self.backtest_performance_tab = QWidget()
        self.backtest_performance_layout = QVBoxLayout(self.backtest_performance_tab)
        self.backtest_performance_table = QTableWidget()
        self.backtest_performance_table.setColumnCount(4)
        self.backtest_performance_table.setHorizontalHeaderLabels(["Metric", "Value", "", ""])
        self.backtest_performance_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.backtest_performance_layout.addWidget(self.backtest_performance_table)
        self.backtest_results_tabs.addTab(self.backtest_performance_tab, "Performance Metrics")
        
        self.backtest_regime_tab = QWidget()
        self.backtest_regime_layout = QVBoxLayout(self.backtest_regime_tab)
        self.backtest_regime_table = QTableWidget()
        self.backtest_regime_table.setColumnCount(5)
        self.backtest_regime_table.setHorizontalHeaderLabels([
            "Regime", "Trades", "Win Rate", "Profit", "Avg Profit"
        ])
        self.backtest_regime_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.backtest_regime_layout.addWidget(self.backtest_regime_table)
        self.backtest_results_tabs.addTab(self.backtest_regime_tab, "Regime Performance")
        
        self.backtest_sector_tab = QWidget()
        self.backtest_sector_layout = QVBoxLayout(self.backtest_sector_tab)
        self.backtest_sector_table = QTableWidget()
        self.backtest_sector_table.setColumnCount(5)
        self.backtest_sector_table.setHorizontalHeaderLabels([
            "Sector", "Trades", "Win Rate", "Profit", "Avg Profit"
        ])
        self.backtest_sector_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.backtest_sector_layout.addWidget(self.backtest_sector_table)
        self.backtest_results_tabs.addTab(self.backtest_sector_tab, "Sector Performance")
        
        self.backtest_trades_tab = QWidget()
        self.backtest_trades_layout = QVBoxLayout(self.backtest_trades_tab)
        self.backtest_trades_table = QTableWidget()
        self.backtest_trades_table.setColumnCount(8)
        self.backtest_trades_table.setHorizontalHeaderLabels([
            "Ticker", "Entry", "Exit", "Profit", "Gain%", "Duration", "Reason", "Regime"
        ])
        self.backtest_trades_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.backtest_trades_layout.addWidget(self.backtest_trades_table)
        self.backtest_results_tabs.addTab(self.backtest_trades_tab, "Trade Log")
        
        self.tabs.addTab(self.backtest_tab, "Backtest")
        
        self.start_button.clicked.connect(self.start_system)
        self.stop_button.clicked.connect(self.stop_system)
        self.scan_button.clicked.connect(self.request_scan)
        
        self.stop_button.setEnabled(False)
        self.scan_button.setEnabled(False)
        self.backtest_results_tabs.setEnabled(False)
        
        self.log_message("Trading Dashboard Initialized")
        self.log_message(f"Tracking Tickers: {', '.join(self.tickers)}")
        self.log_message(f"Operating in {'TESTING' if testing_mode else 'LIVE'} mode")
        self.log_message("Press 'Start System' to begin trading")
        
        self.update_ui({
            'timestamp': datetime.now(tz.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
            'market_open': False,
            'capital': 100000,
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
            'testing_mode': testing_mode,
            'backtest_mode': False
        })
        
    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.terminal_output.append(f"[{timestamp}] {message}")
        scrollbar = self.terminal_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def start_system(self):
        try:
            capital = float(self.capital_input.text())
            risk = float(self.risk_input.text())
        except ValueError:
            self.log_message("Invalid capital or risk value. Using defaults.")
            capital = 100000
            risk = 0.01
        
        self.trading_system = TradingSystem(
            self.tickers, 
            capital=capital, 
            risk_per_trade=risk,
            testing_mode=self.testing_mode
        )
        
        self.trading_system.update_signal.connect(self.update_ui)
        self.trading_system.log_signal.connect(self.log_message)
        
        if not self.trading_system.isRunning():
            if self.trading_system.start_system():
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(True)
                self.scan_button.setEnabled(True)
                self.log_message("Trading system started")
                return
        self.log_message("System already running")
    
    def stop_system(self):
        if self.trading_system and self.trading_system.isRunning():
            if self.trading_system.stop_system():
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                self.scan_button.setEnabled(False)
                self.log_message("Trading system stopped")
                return
        self.log_message("System not running")
    
    def request_scan(self):
        if self.trading_system and self.trading_system.running:
            self.trading_system.scan_requested.emit()
            self.log_message("Manual scan requested")
        else:
            self.log_message("System not running - cannot scan")
    
    def update_ui(self, state):
        try:
            self.timestamp_label.setText(f"Timestamp: {state['timestamp']}")
            market_status = "OPEN" if state['market_open'] else "CLOSED"
            self.market_status_label.setText(f"Market Status: {market_status}")
            self.regime_label.setText(f"Market Regime: {state['market_regime']}")
            
            volatility = state['market_volatility']
            if isinstance(volatility, pd.Series):
                volatility = volatility.iloc[-1] if not volatility.empty else 0.15
            self.volatility_label.setText(f"Market Volatility: {volatility*100:.2f}%")
            
            self.capital_label.setText(f"Capital: ${state['capital']:,.2f}")
            self.positions_label.setText(f"Active Positions: {state['positions_count']}")
            self.trades_label.setText(f"Total Trades: {state['trade_count']}")
            self.profit_label.setText(f"Total Profit: ${state['total_profit']:,.2f}")
            self.win_rate_label.setText(f"Win Rate: {state['win_rate']:.1f}%")
            
            # Handle live trading updates
            if not state['backtest_mode']:
                self.update_positions_table(state)
                self.update_trades_table(state)
                self.update_opportunities_table(state)
                self.update_runner_ups_table(state)
                self.update_plots()
                self.update_sector_tab(state)
            # Handle backtest results
            else:
                # Only update backtest results when backtest is complete
                if not self.trading_system.running:
                    self.display_backtest_results(state)
                    
        except Exception as e:
            self.log_message(f"UI update error: {str(e)}")

    def display_backtest_results(self, state):
        """Update backtest results tabs with final statistics"""
        try:
            # Update performance metrics table
            self.backtest_performance_table.setRowCount(5)
            metrics = [
                ("Starting Capital", f"${state['initial_capital']:,.2f}"),
                ("Ending Capital", f"${state['capital']:,.2f}"),
                ("Total Profit", f"${state['total_profit']:,.2f}"),
                ("Total Trades", str(state['trade_count'])),
                ("Win Rate", f"{state['win_rate']:.1f}%")
            ]
            
            for row, (metric, value) in enumerate(metrics):
                self.backtest_performance_table.setItem(row, 0, QTableWidgetItem(metric))
                self.backtest_performance_table.setItem(row, 1, QTableWidgetItem(value))
                
            # Update trade log table
            trades = state['trade_log'][-100:]  # Show last 100 trades
            self.backtest_trades_table.setRowCount(len(trades))
            for row, trade in enumerate(trades):
                self.backtest_trades_table.setItem(row, 0, QTableWidgetItem(trade['ticker']))
                self.backtest_trades_table.setItem(row, 1, QTableWidgetItem(f"{trade['entry']:.2f}"))
                self.backtest_trades_table.setItem(row, 2, QTableWidgetItem(f"{trade['exit']:.2f}" if trade['exit'] else ""))
                
                profit = trade['profit'] or 0
                profit_item = QTableWidgetItem(f"{profit:+.2f}")
                profit_item.setForeground(QBrush(QColor('green') if profit > 0 else QColor('red')))
                self.backtest_trades_table.setItem(row, 3, profit_item)
                
                gain = trade['percent_gain'] or 0
                gain_item = QTableWidgetItem(f"{gain:+.2f}%")
                gain_item.setForeground(QBrush(QColor('green') if gain > 0 else QColor('red')))
                self.backtest_trades_table.setItem(row, 4, gain_item)
                
                self.backtest_trades_table.setItem(row, 5, QTableWidgetItem(f"{trade['duration'] or 0:.1f}m"))
                self.backtest_trades_table.setItem(row, 6, QTableWidgetItem(trade['exit_reason'] or ""))
                self.backtest_trades_table.setItem(row, 7, QTableWidgetItem(trade.get('regime', '')))
            
            # Update equity curve plot
            self.update_equity_curve(state.get('equity_curve', []))
            
            # Enable backtest results tabs
            self.backtest_results_tabs.setEnabled(True)
            self.run_backtest_button.setEnabled(True)
            
        except Exception as e:
            self.log_message(f"Backtest results error: {str(e)}")
    
    def update_positions_table(self, state):
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
            self.positions_table.setItem(row, 7, QTableWidgetItem(f"{pos['profit_target']:.2极客时间}"))
            self.positions_table.setItem(row, 8, QTableWidgetItem(f"{pos['risk']:.2f}%"))
            self.positions_table.setItem(row, 9, QTableWidgetItem(pos['regime']))
            self.positions_table.setItem(row, 10, QTableWidgetItem(f"{pos.get('original_score', 0):.1f}"))
            
            if hasattr(self, 'trading_system') and self.trading_system:
                sector = self.trading_system.get_ticker_sector(pos['ticker'])
                self.positions_table.setItem(row, 11, QTableWidgetItem(sector))
            else:
                self.positions_table.setItem(row, 11, QTableWidgetItem("Unknown"))
            
            self.positions_table.setItem(row, 12, QTableWidgetItem(str(pos.get('days_held', 0))))

    def update_equity_curve(self, equity_data):
        """Plot the backtest equity curve"""
        try:
            ax = self.backtest_equity_plot.figure.subplots()
            ax.clear()
            
            if equity_data:
                dates = [point[0] for point in equity_data]
                values = [point[1] for point in equity_data]
                ax.plot(dates, values, 'b-', linewidth=2)
                ax.set_title('Equity Curve')
                ax.set_xlabel('Date')
                ax.set_ylabel('Account Value ($)')
                ax.grid(True)
                
                # Format dates nicely
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                self.backtest_equity_plot.figure.autofmt_xdate()
            
            self.backtest_equity_plot.draw()
        except Exception as e:
            self.log_message(f"Equity curve error: {str(e)}")

    def update_trades_table(self, state):
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
    
    def update_opportunities_table(self, state):
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
    
    def update_runner_ups_table(self, state):
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
    
    def update_plots(self):
        for i in reversed(range(self.plot_layout.count())):
            widget = self.plot_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        # Safe access to trading_system
        trading_system = getattr(self, 'trading_system', None)
        
        # Check if trading_system exists and is running with data handler
        if trading_system and trading_system.running and trading_system.data_handler:
            # Check if positions exist before iterating
            if hasattr(trading_system, 'positions') and trading_system.positions:
                for ticker, position in trading_system.positions.items():
                    historical_data = trading_system.data_handler.get_historical(ticker, 50)
                    if historical_data is None or historical_data.empty:
                        continue
                    plot = PositionPlot(self.plot_container, width=10, height=4, dpi=100)
                    plot.plot_position(ticker, position, historical_data)
                    self.plot_layout.addWidget(plot)
                return  # Exit after successful plot
        
        # Show message if no positions or system not ready
        label = QLabel("No active positions to display")
        label.setAlignment(Qt.AlignCenter)
        label.setFont(QFont("Arial", 16))
        self.plot_layout.addWidget(label)
    
    def update_sector_tab(self, state):
        self.sector_table.setRowCount(len(state['sector_scores']))
        for row, (sector, score) in enumerate(state['sector_scores'].items()):
            self.sector_table.setItem(row, 0, QTableWidgetItem(sector))
            score_item = QTableWidgetItem(f"{score:.2f}")
            if score > 70:
                score_item.setForeground(QBrush(QColor('green')))
            elif score < 30:
                score_item.setForeground(QBrush(QColor('red')))
            else:
                score_item.setForeground(QBrush(QColor('orange')))
            self.sector_table.setItem(row, 1, score_item)
        
        regime = state['market_regime']
        allocation = ASSET_ALLOCATIONS.get(regime, {})
        alloc_text = "\n".join([f"• {asset.replace('_', ' ').title()}: {pct}%" for asset, pct in allocation.items()])
        self.alloc_text.setHtml(f"""
            <h3>Recommended Allocation for {regime} Market:</h3>
            <p>{alloc_text}</p>
        """)
    
    def run_backtest(self):
        tickers = [t.strip() for t in self.backtest_ticker_input.text().split(",")]
        if not tickers:
            QMessageBox.warning(self, "Warning", "Please enter at least one ticker")
            return
            
        start_date = self.backtest_start_date.dateTime().toPyDateTime()
        end_date = self.backtest_end_date.dateTime().toPyDateTime()
        
        try:
            capital = float(self.backtest_capital_input.text())
            risk = float(self.backtest_risk_input.text())
        except ValueError:
            self.log_message("Invalid capital or risk value. Using defaults.")
            capital = 100000
            risk = 0.01
        
        if start_date >= end_date:
            QMessageBox.warning(self, "Warning", "Start date must be before end date")
            return
            
        if capital <= 0:
            QMessageBox.warning(self, "Warning", "Capital must be positive")
            return
            
        if risk <= 0 or risk > 0.1:
            QMessageBox.warning(self, "Warning", "Risk per trade must be between 0 and 0.1")
            return
        
        # Localize naive datetimes to UTC
        if start_date.tzinfo is None:
            start_date = pytz.utc.localize(start_date)
        if end_date.tzinfo is None:
            end_date = pytz.utc.localize(end_date)
        
        self.trading_system = TradingSystem(
            tickers=tickers,
            capital=capital,
            risk_per_trade=risk,
            testing_mode=self.testing_mode,
            backtest_mode=True,
            start_date=start_date,
            end_date=end_date
        )
        
        self.trading_system.update_signal.connect(self.update_ui)
        self.trading_system.log_signal.connect(self.log_message)
        
        self.progress_dialog = BacktestProgressDialog(self)
        self.trading_system.log_signal.connect(self.progress_dialog.update_progress_from_log)
        self.progress_dialog.show()
        
        if self.trading_system.start_system():
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.scan_button.setEnabled(True)
            self.log_message("Backtest started")
            self.run_backtest_button.setEnabled(False)
    
    def closeEvent(self, event):
        if hasattr(self, 'trading_system') and self.trading_system and self.trading_system.isRunning():
            self.trading_system.stop_system()
            self.trading_system.wait(3000)
        event.accept()


# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    testing_mode = "--testing" in sys.argv
    dashboard = TradingDashboard(testing_mode=testing_mode)
    dashboard.show()
    sys.exit(app.exec_())