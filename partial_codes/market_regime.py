import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
import math
import time
from datetime import datetime, timedelta
import os
import joblib
import talib  # Added for advanced technical indicators

# Suppress warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

class MarketRegimeAnalyzer:
    def __init__(self, n_states=3, polygon_api_key=None):
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=2000,
            tol=1e-4,
            init_params="se",  # Critical fix: Proper initialization
            params="stmc",
            random_state=42,
        )
        self.state_labels = {}
        self.feature_scaler = StandardScaler()
        self.polygon_api_key = polygon_api_key
        self.data_cache = {}  # For caching API responses
        os.makedirs("data_cache", exist_ok=True)

    def prepare_market_data(self, tickers, sample_size=100, min_days_data=200):
        prices_data = []
        valid_tickers = []
        mcaps = {}  # For market cap weighting

        print("\nBuilding market composite from multiple exchanges...")

        # First pass: Get market caps for weighting
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {
                executor.submit(self.get_market_cap, ticker): ticker 
                for ticker in tickers[:sample_size]
            }
            for future in tqdm(as_completed(future_to_ticker), total=min(sample_size, len(tickers)), desc="Fetching Market Caps"):
                ticker = future_to_ticker[future]
                try:
                    mcaps[ticker] = future.result() or 1  # Default to 1 if None
                except:
                    mcaps[ticker] = 1

        total_mcap = sum(mcaps.values())
        
        # Second pass: Get price data with market cap weighting
        def fetch_and_validate(symbol):
            prices = self.fetch_stock_data(symbol)
            if prices is not None and len(prices) >= min_days_data:
                weight = mcaps.get(symbol, 1) / total_mcap
                return symbol, prices * weight  # Apply market cap weighting
            return None

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(fetch_and_validate, symbol): symbol
                for symbol in tickers[:sample_size]
            }

            for future in tqdm(as_completed(futures), total=min(sample_size, len(tickers)), desc="Fetching Price Data"):
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
            
        # Align and combine data
        composite = pd.concat(prices_data, axis=1)
        composite.columns = valid_tickers
        composite = composite.fillna(method='ffill').fillna(method='bfill')
        return composite.sum(axis=1).dropna()  # Sum of weighted prices

    def analyze_regime(self, index_data, n_states=None):
        if n_states is None:
            n_states = self.model.n_components

        # Calculate advanced features
        log_returns = np.log(index_data).diff().dropna()
        features = pd.DataFrame({
            "returns": log_returns,
            "volatility": log_returns.rolling(21).std(),
            "momentum": log_returns.rolling(14).mean(),
            "rsi": talib.RSI(index_data, timeperiod=14),
            "macd": talib.MACD(index_data)[0],  # MACD line
            "adx": talib.ADX(index_data, index_data, index_data, timeperiod=14)
        }).dropna()

        if len(features) < 100:  # Increased minimum for better model stability
            raise ValueError(f"Only {len(features)} days of feature data")
            
        # Scale features
        scaled_features = self.feature_scaler.fit_transform(features)

        # Create and fit model
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=2000,
            tol=1e-4,
            init_params="se",  # Critical fix
            params="stmc",
            random_state=42,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(scaled_features)

        # Label states based on volatility and returns
        state_stats = []
        for i in range(model.n_components):
            state_return = model.means_[i][0]  # Returns feature
            state_vol = model.means_[i][1]    # Volatility feature
            state_stats.append((i, state_return, state_vol))
            
        # Sort by return then volatility
        state_stats.sort(key=lambda x: (x[1], -x[2]))
        
        # Create labels based on sorted states
        if n_states == 3:
            state_labels = {
                state_stats[0][0]: "Bear",
                state_stats[1][0]: "Neutral",
                state_stats[2][0]: "Bull",
            }
        elif n_states == 4:
            state_labels = {
                state_stats[0][0]: "Severe Bear",
                state_stats[1][0]: "Mild Bear",
                state_stats[2][0]: "Mild Bull",
                state_stats[3][0]: "Strong Bull",
            }
        else:
            state_labels = {i: f"State {i+1}" for i in range(n_states)}

        # Predict regimes
        states = model.predict(scaled_features)
        state_probs = model.predict_proba(scaled_features)
        
        # Calculate state durations
        state_durations = self.calculate_state_durations(states)

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
            time.sleep(0.15)  # Respect rate limits (6-7 requests/sec)
            response = requests.get(url, params=params, timeout=15)
            
            # Handle rate limits
            if response.status_code == 429:
                time.sleep(30)
                return self.fetch_stock_data(symbol, days)
                
            if response.status_code != 200:
                return None

            results = response.json().get("results", [])
            if not results:
                return None
                
            df = pd.DataFrame(results)
            df["date"] = pd.to_datetime(df["t"], unit="ms")
            result = df.set_index("date")["c"]
            
            # Cache result
            result.to_pickle(cache_file)
            return result
            
        except requests.exceptions.Timeout:
            print(f"Timeout fetching {symbol}, skipping")
            return None
        except Exception as e:
            print(f"Error fetching {symbol}: {str(e)}")
            return None
            
    def get_market_cap(self, symbol):
        cache_file = f"data_cache/mcap_{symbol}.pkl"
        if os.path.exists(cache_file):
            return joblib.load(cache_file)
            
        url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
        params = {"apiKey": self.polygon_api_key}
        
        try:
            time.sleep(0.15)  # Rate limiting
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 429:
                time.sleep(30)
                return self.get_market_cap(symbol)
                
            if response.status_code == 200:
                data = response.json().get("results", {})
                mcap = data.get("market_cap", 0)
                joblib.dump(mcap, cache_file)
                return mcap
        except Exception:
            pass
            
        return 0


class SectorRegimeSystem:
    def __init__(self, polygon_api_key=None):
        self.sector_mappings = {}
        self.sector_composites = {}
        self.sector_analyzers = {}
        self.overall_analyzer = MarketRegimeAnalyzer(polygon_api_key=polygon_api_key)
        self.sector_weights = {}
        self.sector_scores = {}
        self.polygon_api_key = polygon_api_key
        self.current_regime = None

    def map_tickers_to_sectors(self, tickers):
        self.sector_mappings = {}
        cache_file = "data_cache/sector_mappings.pkl"
        
        # Try loading from cache
        if os.path.exists(cache_file):
            self.sector_mappings = joblib.load(cache_file)
            return self.sector_mappings

        # Parallel sector mapping
        def map_single_ticker(symbol):
            cache_file = f"data_cache/sector_{symbol}.pkl"
            if os.path.exists(cache_file):
                return symbol, joblib.load(cache_file)
                
            url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
            params = {"apiKey": self.polygon_api_key}
            try:
                time.sleep(0.15)
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 429:
                    time.sleep(30)
                    return map_single_ticker(symbol)
                    
                if response.status_code == 200:
                    data = response.json().get("results", {})
                    sector = data.get("sic_description", "Unknown")
                    if sector == "Unknown":
                        sector = data.get("primary_exchange", "Unknown")
                    
                    # Cache result
                    joblib.dump(sector, cache_file)
                    return symbol, sector
            except Exception as e:
                print(f"Sector mapping failed for {symbol}: {str(e)}")
            return symbol, "Unknown"

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(map_single_ticker, symbol): symbol for symbol in tickers}

            for future in tqdm(as_completed(futures), total=len(tickers), desc="Mapping Sectors"):
                try:
                    symbol, sector = future.result()
                    if sector != "Unknown":
                        self.sector_mappings.setdefault(sector, []).append(symbol)
                except Exception as e:
                    print(f"Error processing sector mapping: {str(e)}")

        # Remove unknown sectors and small sectors
        self.sector_mappings = {
            k: v for k, v in self.sector_mappings.items() 
            if k != "Unknown" and len(v) > 10
        }
        
        # Save to cache
        joblib.dump(self.sector_mappings, cache_file)
        return self.sector_mappings

    def calculate_sector_weights(self):
        total_mcap = 0
        sector_mcaps = {}
        
        # Use cached data if available
        for sector, tickers in self.sector_mappings.items():
            sector_mcap = 0
            # Get market caps for first 30 tickers in sector
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_ticker = {
                    executor.submit(self.overall_analyzer.get_market_cap, symbol): symbol
                    for symbol in tickers[:30]
                }
                for future in as_completed(future_to_ticker):
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
        print("\nBuilding sector composites...")
        self.sector_composites = {}
        cache_file = "data_cache/sector_composites.pkl"
        
        # Try loading from cache
        if os.path.exists(cache_file):
            self.sector_composites = joblib.load(cache_file)
            return self.sector_composites

        # Process each sector
        for sector, tickers in tqdm(self.sector_mappings.items(), desc="Sectors"):
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
                    print(f"Error processing {symbol}: {str(e)}")
            
            if prices_data:
                composite = pd.concat(prices_data, axis=1)
                composite = composite.fillna(method='ffill').fillna(method='bfill')
                self.sector_composites[sector] = composite.sum(axis=1).dropna()
        
        # Save to cache
        joblib.dump(self.sector_composites, cache_file)
        return self.sector_composites

    def analyze_sector_regimes(self, n_states=3):
        print("\nAnalyzing sector regimes...")
        self.sector_analyzers = {}
        
        # Get overall market regime first
        if not hasattr(self, 'market_composite'):
            tickers = [t for sublist in self.sector_mappings.values() for t in sublist]
            self.market_composite = self.overall_analyzer.prepare_market_data(tickers[:100])
        market_result = self.overall_analyzer.analyze_regime(self.market_composite)
        self.current_regime = market_result["regimes"][-1]
        
        # Process each sector
        for sector, composite in tqdm(self.sector_composites.items(), desc="Analyzing Sectors"):
            try:
                analyzer = MarketRegimeAnalyzer(polygon_api_key=self.polygon_api_key)
                results = analyzer.analyze_regime(composite, n_states=n_states)
                self.sector_analyzers[sector] = {
                    "results": results,
                    "composite": composite,
                    "volatility": composite.pct_change().std(),
                    "analyzer": analyzer
                }
            except Exception as e:
                print(f"Error analyzing {sector}: {str(e)}")
                
        return self.sector_analyzers

    def calculate_sector_scores(self):
        self.sector_scores = {}
        if not self.sector_analyzers:
            return pd.Series()

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
                print(f"Error calculating score for {sector}: {str(e)}")
                self.sector_scores[sector] = 0

        return pd.Series(self.sector_scores).sort_values(ascending=False)

    def get_regime_aware_screener(self):
        """Regime-based screening recommendations"""
        if not self.current_regime:
            return {}
            
        screening_rules = {
            "Bull": {
                "filters": [
                    {"indicator": "RSI", "min": 50, "max": 70},
                    {"indicator": "MACD", "signal": "positive"},
                    {"indicator": "Volume", "min": 1.5, "lookback": 21}
                ],
                "types": ["Breakouts", "Momentum"]
            },
            "Bear": {
                "filters": [
                    {"indicator": "RSI", "min": 30, "max": 50},
                    {"indicator": "Volatility", "max": 0.4},
                    {"indicator": "ShortFloat", "min": 0.1}
                ],
                "types": ["ShortSqueeze", "Oversold"]
            },
            "Severe Bear": {
                "filters": [
                    {"indicator": "RSI", "max": 30},
                    {"indicator": "DebtToEquity", "max": 0.7}
                ],
                "types": ["Oversold", "Fundamentals"]
            },
            "Strong Bull": {
                "filters": [
                    {"indicator": "RSI", "max": 80},
                    {"indicator": "Volume", "min": 2.0, "lookback": 50},
                    {"indicator": "EarningsGrowth", "min": 0.2}
                ],
                "types": ["Momentum", "Growth"]
            }
        }
        
        return screening_rules.get(self.current_regime, {})

    def validate_model(self, data, windows=36, min_test_size=10):
        """Walk-forward validation for regime model"""
        print("\nRunning model validation...")
        accuracy = []
        state_transitions = []
        
        for i in range(windows, len(data) - min_test_size):
            # Split data
            train_data = data.iloc[:i]
            test_data = data.iloc[i:i+min_test_size]
            
            # Train model
            analyzer = MarketRegimeAnalyzer()
            train_result = analyzer.analyze_regime(train_data)
            
            # Test model - CORRECTED LINE BELOW
            test_features = analyzer.feature_scaler.transform(
                analyzer.prepare_features(test_data))
            test_states = analyzer.model.predict(test_features)
            
            # Get actual test states (using same model)
            full_result = analyzer.analyze_regime(pd.concat([train_data, test_data]))
            actual_states = full_result["regimes"][-min_test_size:]
            
            # Calculate accuracy
            predicted = [train_result["state_labels"][s] for s in test_states]
            acc = sum(p == a for p, a in zip(predicted, actual_states)) / min_test_size
            accuracy.append(acc)
            
            # Track state transitions
            state_transitions.append((train_result["regimes"][-1], predicted[0]))
        
        # Calculate validation metrics
        avg_accuracy = np.mean(accuracy) if accuracy else 0
        transition_matrix = pd.crosstab(
            [t[0] for t in state_transitions],
            [t[1] for t in state_transitions],
            normalize='index'
        )
        
        print(f"Model Validation Accuracy: {avg_accuracy:.2%}")
        print("\nState Transition Probabilities:")
        print(transition_matrix)
        
        return {
            "accuracy": avg_accuracy,
            "transition_matrix": transition_matrix
        }
        
    def prepare_features(self, price_data):
        """Feature preparation for validation"""
        log_returns = np.log(price_data).diff().dropna()
        features = pd.DataFrame({
            "returns": log_returns,
            "volatility": log_returns.rolling(21).std(),
            "momentum": log_returns.rolling(14).mean(),
            "rsi": talib.RSI(price_data, timeperiod=14),
            "macd": talib.MACD(price_data)[0]
        }).dropna()
        return features