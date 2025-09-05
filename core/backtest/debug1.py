import numpy as np
import pandas as pd
import asyncio
import aiohttp
import time
import os
import logging
import json
import threading
from datetime import datetime, timedelta
from urllib.parse import urlencode
from threading import Lock, Event, RLock
from collections import defaultdict
import sys
import signal
from tzlocal import get_localzone
import contextlib
from typing import List, Dict, Optional, Any, Tuple
import argparse
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from scipy.stats import norm, t
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy import signal as sp_signal

import psycopg2
from psycopg2.extras import RealDictCursor
import psycopg2.pool


# ======================== CONFIGURATION ======================== #
class Config:
    # API Configuration
    POLYGON_API_KEY = "ld1Poa63U6t4Y2MwOCA2JeKQyHVrmyg8"
    
    # Scanner Configuration - Using ONLY Nasdaq Composite
    COMPOSITE_INDICES = ["^IXIC"]  # NASDAQ Composite only
    MAX_CONCURRENT_REQUESTS = 100
    RATE_LIMIT_DELAY = 0.02
    SCAN_TIME = "08:30"
    
    # Market Regime Configuration
    REGIME_SCAN_INTERVAL = 3600  # 1 hour in seconds
    HISTORICAL_DATA_DAYS = 365  # 1 year of historical data
    HMM_N_COMPONENTS = 3  # Number of market regimes (bull, bear, sideways)
    HMM_N_ITER = 100  # Number of HMM iterations
    HMM_COVARIANCE_TYPE = "diag"  # Covariance type
    
    # Enhanced Model Configuration
    USE_ENSEMBLE_MODEL = True  # Use ensemble of HMM and GMM
    MODEL_VERSION = "enhanced_v4.0"  # Model version identifier
    USE_ANOMALY_DETECTION = True  # Detect anomalous market conditions
    N_CLUSTERS = 4  # Allow for more nuanced regime detection
    PCA_COMPONENTS = 10  # Reduce dimensionality for better clustering
    ROLLING_TRAINING_WINDOW = 252  # Use 1 year of data for training (approx 252 trading days)

    # Database Configuration - PostgreSQL
    POSTGRES_HOST = "localhost"
    POSTGRES_PORT = 5432
    POSTGRES_DB = "stock_scanner"
    POSTGRES_USER = "hodumaru"
    POSTGRES_PASSWORD = "Leetkd214"
    
    # Backtesting Configuration
    BACKTEST_HISTORICAL_DAYS = 30  # Days of historical data to use for backtesting
    
    # Logging Configuration
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # New Configuration for Enhanced Features
    USE_DEEP_LEARNING = True
    DEEP_LEARNING_HIDDEN_LAYERS = [64, 32]
    BAYESIAN_SAMPLES = 1000
    MICROSTRUCTURE_FEATURES = True
    UNCERTAINTY_ESTIMATION = True
    REGIME_TRANSITION_PROBABILITIES = True

# Initialize configuration
config = Config()

# ======================== LOGGING SETUP ======================== #
def setup_logging():
    """Configure logging with file and console handlers"""
    os.makedirs("logs", exist_ok=True)
    
    logger = logging.getLogger("MarketRegimeScanner")
    logger.setLevel(config.LOG_LEVEL)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter(config.LOG_FORMAT)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/market_regime_scanner_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(config.LOG_LEVEL)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(config.LOG_LEVEL)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# ======================== DEEP LEARNING MODELS ======================== #
class MarketRegimeLSTM(nn.Module):
    """LSTM-based market regime classifier"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(MarketRegimeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(self.dropout(out[:, -1, :]))
        return out

class MarketRegimeCNN(nn.Module):
    """CNN-based market regime classifier"""
    def __init__(self, input_channels, num_classes):
        super(MarketRegimeCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.transpose(1, 2)  # Convert to (batch, channels, time)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ======================== BAYESIAN MODELS ======================== #
class BayesianMarketRegime:
    """Bayesian model for market regime detection with uncertainty estimation"""
    def __init__(self, n_regimes=4):
        self.n_regimes = n_regimes
        self.models = []
        
    def fit(self, X, n_models=10):
        """Train ensemble of models for uncertainty estimation"""
        from sklearn.utils import resample
        
        self.models = []
        successful_models = 0
        
        for i in range(n_models):
            try:
                # Bootstrap sample
                X_sample = resample(X)
                
                # Skip if sample is too small
                if len(X_sample) < 20:
                    continue
                
                # Train diverse models
                model = GaussianMixture(
                    n_components=self.n_regimes, 
                    covariance_type='full', 
                    random_state=np.random.randint(1000),
                    max_iter=200,
                    n_init=3
                )
                model.fit(X_sample)
                self.models.append(model)
                successful_models += 1
                
            except Exception as e:
                logger.debug(f"Bayesian model {i+1} failed: {e}")
                continue
        
        # Ensure we have at least one model
        if successful_models == 0:
            logger.warning("All Bayesian models failed, creating fallback model")
            try:
                model = GaussianMixture(
                    n_components=self.n_regimes,
                    covariance_type='full',
                    random_state=42,
                    max_iter=200
                )
                model.fit(X)
                self.models.append(model)
            except Exception as e:
                logger.error(f"Fallback Bayesian model also failed: {e}")
    
    def predict_proba(self, X):
        """Get probabilistic predictions with uncertainty"""
        if not self.models:
            # Return default predictions if no models are available
            default_probs = np.ones((len(X), self.n_regimes)) / self.n_regimes
            return default_probs, np.ones(len(X)) * 0.5, []
        
        all_probs = []
        for model in self.models:
            try:
                probs = model.predict_proba(X)
                all_probs.append(probs)
            except:
                # Skip failed predictions
                continue
        
        if not all_probs:
            # Return default if all predictions failed
            default_probs = np.ones((len(X), self.n_regimes)) / self.n_regimes
            return default_probs, np.ones(len(X)) * 0.5, []
        
        # Mean and standard deviation across ensemble
        mean_probs = np.mean(all_probs, axis=0)
        std_probs = np.std(all_probs, axis=0)
        
        # Uncertainty measure
        uncertainty = np.mean(std_probs, axis=1)
        
        return mean_probs, uncertainty, all_probs
    
class BayesianTransitionModel:
    """Bayesian model for regime transition probabilities"""
    def __init__(self, n_regimes=4):
        self.n_regimes = n_regimes
        self.transition_counts = np.ones((n_regimes, n_regimes))  # Dirichlet prior
        self.transition_probs = np.ones((n_regimes, n_regimes)) / n_regimes
        
    def update(self, previous_regime, current_regime):
        """Update transition probabilities"""
        # Ensure regimes are within valid range
        if 0 <= previous_regime < self.n_regimes and 0 <= current_regime < self.n_regimes:
            self.transition_counts[previous_regime, current_regime] += 1
            self.transition_probs = self.transition_counts / self.transition_counts.sum(axis=1, keepdims=True)
        
    def get_transition_prob(self, previous_regime, current_regime):
        """Get transition probability"""
        if 0 <= previous_regime < self.n_regimes and 0 <= current_regime < self.n_regimes:
            return self.transition_probs[previous_regime, current_regime]
        return 1.0 / self.n_regimes  # Default equal probability
    
    def get_transition_uncertainty(self, previous_regime):
        """Get uncertainty about transitions from a regime"""
        if 0 <= previous_regime < self.n_regimes:
            counts = self.transition_counts[previous_regime]
            total = counts.sum()
            # Higher entropy means more uncertainty
            if total > 0:
                probs = counts / total
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                return entropy / np.log(self.n_regimes)  # Normalized entropy
        return 1.0  # Maximum uncertainty

# ======================== DATABASE MANAGER (FULLY OPTIMIZED) ======================== #
class DatabaseManager:
    def __init__(self):
        self.pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            database=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD
        )
        self._init_database()
    
    def _get_connection_from_pool(self):
        return self.pool.getconn()
                
    def _return_connection_to_pool(self, conn):
        self.pool.putconn(conn)
    
    @contextlib.contextmanager
    def get_connection(self):
        conn = self._get_connection_from_pool()
        try:
            yield conn
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            conn.close()
            raise
        finally:
            self._return_connection_to_pool(conn)
    
    def close_all_connections(self):
        self.pool.closeall()
    
    def _init_database(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create tickers table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tickers (
                    ticker TEXT PRIMARY KEY,
                    name TEXT,
                    primary_exchange TEXT,
                    last_updated_utc TEXT,
                    type TEXT,
                    market TEXT,
                    locale TEXT,
                    currency_name TEXT,
                    active INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create historical_tickers table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS historical_tickers (
                    id SERIAL PRIMARY KEY,
                    ticker TEXT,
                    name TEXT,
                    primary_exchange TEXT,
                    last_updated_utc TEXT,
                    type TEXT,
                    market TEXT,
                    locale TEXT,
                    currency_name TEXT,
                    active INTEGER,
                    change_type TEXT,
                    change_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create market_regimes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_regimes (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL UNIQUE,
                    regime INTEGER NOT NULL,
                    regime_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    features TEXT,
                    model_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create regime_statistics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS regime_statistics (
                    id SERIAL PRIMARY KEY,
                    regime INTEGER NOT NULL,
                    start_date TIMESTAMP NOT NULL,
                    end_date TIMESTAMP,
                    duration_days INTEGER,
                    return_pct REAL,
                    volatility REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tickers_exchange ON tickers(primary_exchange)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tickers_active ON tickers(active)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_historical_tickers_date ON historical_tickers(change_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_regimes_timestamp ON market_regimes(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_regimes_regime ON market_regimes(regime)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_statistics_regime ON regime_statistics(regime)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_statistics_date ON regime_statistics(start_date)')
            
            conn.commit()

    def _convert_numpy_types(self, params):
        """Convert numpy data types to native Python types for database compatibility"""
        converted_params = []
        for param in params:
            if isinstance(param, np.integer):
                converted_params.append(int(param))
            elif isinstance(param, np.floating):
                converted_params.append(float(param))
            else:
                converted_params.append(param)
        return tuple(converted_params)

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query, params)
                    return cursor.fetchall()
        except psycopg2.Error as e:
            logger.error(f"Database query error: {e}, Query: {query}, Params: {params}")
            return []
            
    def execute_write(self, query: str, params: tuple = ()) -> int:
        try:
            converted_params = self._convert_numpy_types(params)
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, converted_params)
                    conn.commit()
                    return cursor.rowcount
        except psycopg2.Error as e:
            logger.error(f"Database write error: {e}, Query: {query}, Params: {converted_params}")
            return 0

    # Update all SQL queries to use PostgreSQL syntax
    def upsert_tickers(self, tickers: List[Dict]) -> Tuple[int, int]:
        if not tickers:
            return 0, 0
            
        inserted = 0
        updated = 0
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                # Get existing tickers
                ticker_symbols = [t['ticker'] for t in tickers]
                placeholders = ','.join(['%s'] * len(ticker_symbols))
                
                cursor.execute(
                    f"SELECT ticker FROM tickers WHERE ticker IN ({placeholders})", 
                    ticker_symbols
                )
                existing_tickers = {row[0] for row in cursor.fetchall()}
                
                # Separate into inserts and updates
                inserts = []
                updates = []
                
                for ticker_data in tickers:
                    if ticker_data['ticker'] in existing_tickers:
                        updates.append(ticker_data)
                    else:
                        inserts.append(ticker_data)
                
                # Use transaction
                cursor.execute("BEGIN")
                
                try:
                    # Bulk insert new tickers
                    if inserts:
                        insert_values = [
                            (
                                t['ticker'], t.get('name'), t.get('primary_exchange'), 
                                t.get('last_updated_utc'), t.get('type'), t.get('market'),
                                t.get('locale'), t.get('currency_name'), 1
                            ) for t in inserts
                        ]
                        
                        cursor.executemany('''
                            INSERT INTO tickers 
                            (ticker, name, primary_exchange, last_updated_utc, 
                             type, market, locale, currency_name, active)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ''', insert_values)
                        
                        inserted = len(inserts)
                        
                        # Bulk insert historical records
                        historical_inserts = [
                            (
                                t['ticker'], t.get('name'), t.get('primary_exchange'), 
                                t.get('last_updated_utc'), t.get('type'), t.get('market'),
                                t.get('locale'), t.get('currency_name'), 1, 'added'
                            ) for t in inserts
                        ]
                        
                        cursor.executemany('''
                            INSERT INTO historical_tickers 
                            (ticker, name, primary_exchange, last_updated_utc, 
                             type, market, locale, currency_name, active, change_type)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ''', historical_inserts)
                    
                    # Bulk update existing tickers
                    if updates:
                        update_values = [
                            (
                                t.get('name'), t.get('primary_exchange'), 
                                t.get('last_updated_utc'), t.get('type'), t.get('market'),
                                t.get('locale'), t.get('currency_name'), 1,
                                t['ticker']
                            ) for t in updates
                        ]
                        
                        cursor.executemany('''
                            UPDATE tickers 
                            SET name = %s, primary_exchange = %s, last_updated_utc = %s, 
                                type = %s, market = %s, locale = %s, currency_name = %s, 
                                active = %s, updated_at = CURRENT_TIMESTAMP
                            WHERE ticker = %s
                        ''', update_values)
                        
                        updated = len(updates)
                        
                        # Bulk insert historical records for updates
                        historical_updates = [
                            (
                                t['ticker'], t.get('name'), t.get('primary_exchange'), 
                                t.get('last_updated_utc'), t.get('type'), t.get('market'),
                                t.get('locale'), t.get('currency_name'), 1, 'updated'
                            ) for t in updates
                        ]
                        
                        cursor.executemany('''
                            INSERT INTO historical_tickers 
                            (ticker, name, primary_exchange, last_updated_utc, 
                             type, market, locale, currency_name, active, change_type)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ''', historical_updates)
                    
                    cursor.execute("COMMIT")
                    
                except psycopg2.Error as e:
                    cursor.execute("ROLLBACK")
                    logger.error(f"Transaction failed during ticker upsert: {e}")
                    raise
            
        return inserted, updated

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key"""
        result = self.execute_query(
            "SELECT value FROM metadata WHERE key = %s",
            (key,)
        )
        
        if result:
            value = result[0]['value']
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        return default
    
    def update_metadata(self, key: str, value: Any) -> None:
        """Update metadata key-value pair"""
        self.execute_write(
            "INSERT INTO metadata (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = CURRENT_TIMESTAMP",
            (key, json.dumps(value) if isinstance(value, (list, dict)) else str(value))
        )

    def get_all_active_tickers(self) -> List[Dict]:
        """Get all active tickers from the database"""
        return self.execute_query(
            "SELECT * FROM tickers WHERE active = 1 ORDER BY ticker"
        )

    def search_tickers(self, search_term: str, limit: int = 50) -> List[Dict]:
        """Search tickers by name or symbol"""
        return self.execute_query(
            "SELECT * FROM tickers WHERE (ticker LIKE %s OR name LIKE %s) AND active = 1 ORDER BY ticker LIMIT %s",
            (f"%{search_term}%", f"%{search_term}%", limit)
        )

    def get_ticker_history(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Get historical changes for a ticker"""
        return self.execute_query(
            "SELECT * FROM historical_tickers WHERE ticker = %s ORDER by change_date DESC LIMIT %s",
            (ticker, limit)
        )

    def mark_tickers_inactive(self, tickers: List[str]) -> int:
        """Mark tickers as inactive using bulk operations"""
        if not tickers:
            return 0
            
        marked = 0
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                # Get current data for tickers to be marked inactive
                placeholders = ','.join(['%s'] * len(tickers))
                cursor.execute(
                    f"SELECT * FROM tickers WHERE ticker IN ({placeholders}) AND active = 1", 
                    tickers
                )
                rows = cursor.fetchall()
                
                if not rows:
                    return 0
                    
                # Use transaction for better performance
                cursor.execute("BEGIN")
                
                try:
                    # Bulk update to mark as inactive
                    update_params = [(row[0],) for row in rows]  # Assuming ticker is the first column
                    cursor.executemany(
                        "UPDATE tickers SET active = 0, updated_at = CURRENT_TIMESTAMP WHERE ticker = %s",
                        update_params
                    )
                    
                    marked = cursor.rowcount
                    
                    # Bulk insert historical records
                    historical_data = [
                        (
                            row[0], row[1], row[2],  # ticker, name, primary_exchange
                            row[3], row[4], row[5],  # last_updated_utc, type, market
                            row[6], row[7], 0, 'removed'  # locale, currency_name, active, change_type
                        ) for row in rows
                    ]
                    
                    cursor.executemany('''
                        INSERT INTO historical_tickers 
                        (ticker, name, primary_exchange, last_updated_utc, 
                        type, market, locale, currency_name, active, change_type)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ''', historical_data)
                    
                    cursor.execute("COMMIT")
                    
                except psycopg2.Error as e:
                    cursor.execute("ROLLBACK")
                    logger.error(f"Transaction failed during mark inactive: {e}")
                    raise
                
        return marked

    def get_ticker_details(self, ticker: str) -> Optional[Dict]:
        """Get details for a specific ticker"""
        result = self.execute_query(
            "SELECT * FROM tickers WHERE ticker = %s", 
            (ticker,)
        )
        return result[0] if result else None

    def save_market_regime(self, timestamp: datetime, regime: int, regime_label: str, confidence: float, 
                        features: Dict, model_version: str) -> int:
        return self.execute_write(
            '''INSERT INTO market_regimes 
            (timestamp, regime, regime_label, confidence, features, model_version) 
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (timestamp) DO UPDATE SET
            regime = EXCLUDED.regime,
            regime_label = EXCLUDED.regime_label,
            confidence = EXCLUDED.confidence,
            features = EXCLUDED.features,
            model_version = EXCLUDED.model_version''',
            (timestamp, regime, regime_label, confidence, json.dumps(features), model_version)
        )

    def get_latest_regime(self) -> Optional[Dict]:
        """Get the latest market regime from database"""
        result = self.execute_query(
            "SELECT * FROM market_regimes ORDER BY timestamp DESC LIMIT 1"
        )
        return result[0] if result else None

    def get_regimes_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get market regimes within a date range"""
        return self.execute_query(
            "SELECT * FROM market_regimes WHERE timestamp BETWEEN %s AND %s ORDER BY timestamp",
            (start_date, end_date)
        )

    def save_regime_statistics(self, regime: int, start_date: datetime, end_date: datetime, 
                            duration_days: int, return_pct: float, volatility: float) -> int:
        """Save regime statistics to database"""
        return self.execute_write(
            '''INSERT INTO regime_statistics 
            (regime, start_date, end_date, duration_days, return_pct, volatility) 
            VALUES (%s, %s, %s, %s, %s, %s)''',
            (regime, start_date, end_date, duration_days, return_pct, volatility)
        )

    def get_regime_statistics(self, regime: Optional[int] = None) -> List[Dict]:
        """Get regime statistics, optionally filtered by regime"""
        if regime is not None:
            return self.execute_query(
                "SELECT * FROM regime_statistics WHERE regime = %s ORDER BY start_date",
                (regime,)
            )
        else:
            return self.execute_query(
                "SELECT * FROM regime_statistics ORDER BY start_date"
            )
    

# ======================== BACKTEST DATABASE MANAGER (OPTIMIZED) ======================== #
class BacktestDatabaseManager:
    def __init__(self):
        self.pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=5,
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            database=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD
        )
        self._init_database()
    
    def _get_connection_from_pool(self):
        return self.pool.getconn()
                
    def _return_connection_to_pool(self, conn):
        self.pool.putconn(conn)
    
    @contextlib.contextmanager
    def get_connection(self):
        conn = self._get_connection_from_pool()
        try:
            yield conn
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            conn.close()
            raise
        finally:
            self._return_connection_to_pool(conn)
    
    def close_all_connections(self):
        self.pool.closeall()
    
    def _init_database(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create backtest_tickers table for historical data with year column
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtest_tickers (
                    date TEXT,
                    year INTEGER,
                    ticker TEXT,
                    name TEXT,
                    primary_exchange TEXT,
                    last_updated_utc TEXT,
                    type TEXT,
                    market TEXT,
                    locale TEXT,
                    currency_name TEXT,
                    PRIMARY KEY (date, ticker)
                )
            ''')
            
            # Create backtest_final_results table for storing final backtest results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtest_final_results (
                    id SERIAL PRIMARY KEY,
                    run_id TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    start_year INTEGER,
                    end_year INTEGER,
                    ticker TEXT,
                    name TEXT,
                    primary_exchange TEXT,
                    last_updated_utc TEXT,
                    type TEXT,
                    market TEXT,
                    locale TEXT,
                    currency_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create backtest_market_regimes table for historical backtesting with unique constraint
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtest_market_regimes (
                    id SERIAL PRIMARY KEY,
                    backtest_date TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    regime INTEGER NOT NULL,
                    regime_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    features TEXT,
                    model_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (backtest_date, timestamp)
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_backtest_tickers_date ON backtest_tickers(date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_backtest_tickers_year ON backtest_tickers(year)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_backtest_final_dates ON backtest_final_results(start_date, end_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_backtest_final_years ON backtest_final_results(start_year, end_year)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_backtest_final_ticker ON backtest_final_results(ticker)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_backtest_final_run_id ON backtest_final_results(run_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_backtest_regimes_date ON backtest_market_regimes(backtest_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_backtest_regimes_timestamp ON backtest_market_regimes(timestamp)')
            
            conn.commit()

    def _convert_numpy_types(self, params):
        """Convert numpy data types to native Python types for database compatibility"""
        converted_params = []
        for param in params:
            if isinstance(param, np.integer):
                converted_params.append(int(param))
            elif isinstance(param, np.floating):
                converted_params.append(float(param))
            else:
                converted_params.append(param)
        return tuple(converted_params)
            
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query, params)
                    return cursor.fetchall()
        except psycopg2.Error as e:
            logger.error(f"Database query error: {e}, Query: {query}, Params: {params}")
            return []
            
    def execute_write(self, query: str, params: tuple = ()) -> int:
        try:
            converted_params = self._convert_numpy_types(params)
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, converted_params)
                    conn.commit()
                    return cursor.rowcount
        except psycopg2.Error as e:
            logger.error(f"Database write error: {e}, Query: {query}, Params: {converted_params}")
            return 0
            
    def upsert_backtest_tickers(self, tickers: List[Dict], date_str: str) -> int:
        if not tickers:
            return 0
            
        year = int(date_str.split('-')[0])  # Extract year from date
        inserted = 0
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                # Prepare data for bulk insert
                data_tuples = [
                    (
                        date_str, year, t['ticker'], t.get('name'),
                        t.get('primary_exchange'), t.get('last_updated_utc'),
                        t.get('type'), t.get('market'), t.get('locale'),
                        t.get('currency_name')
                    ) for t in tickers
                ]
                
                cursor.executemany('''
                    INSERT INTO backtest_tickers 
                    (date, year, ticker, name, primary_exchange, last_updated_utc, 
                     type, market, locale, currency_name)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date, ticker) DO UPDATE SET
                    name = EXCLUDED.name,
                    primary_exchange = EXCLUDED.primary_exchange,
                    last_updated_utc = EXCLUDED.last_updated_utc,
                    type = EXCLUDED.type,
                    market = EXCLUDED.market,
                    locale = EXCLUDED.locale,
                    currency_name = EXCLUDED.currency_name
                ''', data_tuples)
                
                inserted = cursor.rowcount
                conn.commit()
            
        return inserted
        
    def get_backtest_tickers(self, date_str: str) -> List[Dict]:
        return self.execute_query(
            "SELECT * FROM backtest_tickers WHERE date = %s ORDER BY ticker",
            (date_str,)
        )
        
    def get_backtest_tickers_by_year(self, year: int) -> List[Dict]:
        return self.execute_query(
            "SELECT * FROM backtest_tickers WHERE year = %s ORDER BY date, ticker",
            (year,)
        )
        
    def get_backtest_dates(self) -> List[str]:
        result = self.execute_query(
            "SELECT DISTINCT date FROM backtest_tickers ORDER BY date"
        )
        return [row['date'] for row in result]
        
    def get_backtest_years(self) -> List[int]:
        result = self.execute_query(
            "SELECT DISTINCT year FROM backtest_tickers ORDER BY year"
        )
        return [row['year'] for row in result]
        
    def upsert_backtest_final_results(self, tickers_data: List[Dict], start_date: str, end_date: str, run_id: str = "default") -> int:
        if not tickers_data:
            return 0
            
        inserted = 0
        start_year = int(start_date.split('-')[0])
        end_year = int(end_date.split('-')[0])
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                # Only delete existing results for this specific run_id and date range
                cursor.execute(
                    "DELETE FROM backtest_final_results WHERE start_date = %s AND end_date = %s AND run_id = %s",
                    (start_date, end_date, run_id)
                )
                
                # Prepare data for bulk insert
                data_tuples = [
                    (
                        run_id, start_date, end_date, start_year, end_year,
                        t['ticker'], t.get('name'), t.get('primary_exchange'),
                        t.get('last_updated_utc'), t.get('type'), t.get('market'),
                        t.get('locale'), t.get('currency_name')
                    ) for t in tickers_data
                ]
                
                cursor.executemany('''
                    INSERT INTO backtest_final_results 
                    (run_id, start_date, end_date, start_year, end_year, ticker, name, primary_exchange, last_updated_utc, 
                     type, market, locale, currency_name)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', data_tuples)
                
                inserted = cursor.rowcount
                conn.commit()
            
        return inserted
        
    def get_backtest_final_results(self, start_date: str, end_date: str, run_id: str = "default") -> List[Dict]:
        return self.execute_query(
            "SELECT * FROM backtest_final_results WHERE start_date = %s AND end_date = %s AND run_id = %s ORDER BY ticker",
            (start_date, end_date, run_id)
        )
        
    def get_backtest_final_results_by_year(self, year: int, run_id: str = "default") -> List[Dict]:
        return self.execute_query(
            "SELECT * FROM backtest_final_results WHERE start_year <= %s AND end_year >= %s AND run_id = %s ORDER BY ticker",
            (year, year, run_id)
        )
        
    def get_all_backtest_runs(self) -> List[Dict]:
        return self.execute_query(
            "SELECT DISTINCT run_id, start_date, end_date, start_year, end_year FROM backtest_final_results ORDER BY start_date, end_date"
        )
        
    def get_backtest_runs_by_date_range(self, start_date: str, end_date: str) -> List[Dict]:
        return self.execute_query(
            "SELECT DISTINCT run_id, start_date, end_date, start_year, end_year FROM backtest_final_results WHERE start_date = %s AND end_date = %s ORDER BY run_id",
            (start_date, end_date)
        )
    
    def save_backtest_market_regime(self, backtest_date: str, timestamp: datetime, regime: int, 
                                regime_label: str, confidence: float, features: Dict, model_version: str) -> int:
        return self.execute_write(
            '''INSERT INTO backtest_market_regimes 
            (backtest_date, timestamp, regime, regime_label, confidence, features, model_version) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (backtest_date, timestamp) DO UPDATE SET
            regime = EXCLUDED.regime,
            regime_label = EXCLUDED.regime_label,
            confidence = EXCLUDED.confidence,
            features = EXCLUDED.features,
            model_version = EXCLUDED.model_version''',
            (backtest_date, timestamp, regime, regime_label, confidence, json.dumps(features), model_version)
        )
        
    def get_backtest_regimes_by_date(self, backtest_date: str) -> List[Dict]:
        return self.execute_query(
            "SELECT * FROM backtest_market_regimes WHERE backtest_date = %s ORDER BY timestamp",
            (backtest_date,)
        )
        
    def get_backtest_dates(self) -> List[str]:
        result = self.execute_query(
            "SELECT DISTINCT backtest_date FROM backtest_market_regimes ORDER BY backtest_date"
        )
        return [row['backtest_date'] for row in result]

    def get_backtest_tickers(self, date_str: str) -> List[Dict]:
        """Get tickers for a specific backtest date"""
        return self.execute_query(
            "SELECT * FROM backtest_tickers WHERE date = %s ORDER BY ticker",
            (date_str,)
    )

    def get_backtest_tickers_by_year(self, year: int) -> List[Dict]:
        """Get tickers for a specific backtest year"""
        return self.execute_query(
            "SELECT * FROM backtest_tickers WHERE year = %s ORDER BY date, ticker",
            (year,)
    )

    def get_backtest_dates(self) -> List[str]:
        """Get all available backtest dates"""
        result = self.execute_query(
            "SELECT DISTINCT date FROM backtest_tickers ORDER BY date"
        )
        return [row['date'] for row in result]

    def get_backtest_years(self) -> List[int]:
        """Get all available backtest years"""
        result = self.execute_query(
            "SELECT DISTINCT year FROM backtest_tickers ORDER BY year"
        )
        return [row['year'] for row in result]

    def get_backtest_final_results(self, start_date: str, end_date: str, run_id: str = "default") -> List[Dict]:
        """Get final backtest results for a specific date range"""
        return self.execute_query(
            "SELECT * FROM backtest_final_results WHERE start_date = %s AND end_date = %s AND run_id = %s ORDER BY ticker",
            (start_date, end_date, run_id)
    )

    def get_backtest_final_results_by_year(self, year: int, run_id: str = "default") -> List[Dict]:
        """Get final backtest results for a specific year"""
        return self.execute_query(
            "SELECT * FROM backtest_final_results WHERE start_year <= %s AND end_year >= %s AND run_id = %s ORDER BY ticker",
            (year, year, run_id)
    )

    def get_all_backtest_runs(self) -> List[Dict]:
        """Get all backtest runs with their date ranges"""
        return self.execute_query(
            "SELECT DISTINCT run_id, start_date, end_date, start_year, end_year FROM backtest_final_results ORDER BY start_date, end_date"
        )

    def get_backtest_runs_by_date_range(self, start_date: str, end_date: str) -> List[Dict]:
        """Get all backtest runs for a specific date range"""
        return self.execute_query(
            "SELECT DISTINCT run_id, start_date, end_date, start_year, end_year FROM backtest_final_results WHERE start_date = %s AND end_date = %s ORDER BY run_id",
            (start_date, end_date)
    )

# ======================== TICKER SCANNER ======================== #
class PolygonTickerScanner:
    def __init__(self):
        self.api_key = config.POLYGON_API_KEY
        self.base_url = "https://api.polygon.io/v3/reference/tickers"
        # Use ONLY Nasdaq Composite
        self.composite_indices = config.COMPOSITE_INDICES
        self.active = False
        self.cache_lock = RLock()
        self.refresh_lock = Lock()
        self.known_missing_tickers = set()
        self.initial_refresh_complete = Event()
        self.last_refresh_time = 0
        self.ticker_cache = pd.DataFrame(columns=[
            "ticker", "name", "primary_exchange", "last_updated_utc", "type", "market", "locale"
        ])
        self.current_tickers_set = set()
        self.local_tz = get_localzone()
        self.semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
        self.db = DatabaseManager()  # Updated to use PostgreSQL
        self.shutdown_requested = False
        # Backtesting attributes
        self.backtest_mode = False
        self.backtest_date = None
        logger.info(f"Using local timezone: {self.local_tz}")
        logger.info(f"Using PostgreSQL database: {config.POSTGRES_DB}")
        logger.info(f"Using composite indices: {', '.join(self.composite_indices)}")
        
    def _init_cache(self):
        """Initialize or load ticker cache from database"""
        self.last_refresh_time = self.db.get_metadata('last_refresh_time', 0)
        
        # Load active tickers from database
        db_tickers = self.db.get_all_active_tickers()
        
        if db_tickers:
            self.ticker_cache = pd.DataFrame(db_tickers)
            logger.info(f"Loaded {len(self.ticker_cache)} tickers from database")
        else:
            self.ticker_cache = pd.DataFrame(columns=[
                "ticker", "name", "primary_exchange", "last_updated_utc", "type", "market", "locale"
            ])
            logger.info("No tickers found in database")
        
        self.current_tickers_set = set(self.ticker_cache['ticker'].tolist()) if not self.ticker_cache.empty else set()
        
        # Load known missing tickers from database
        self.known_missing_tickers = set(self.db.get_metadata('known_missing_tickers', []))
        
        self.initial_refresh_complete.set()

    async def _call_polygon_api(self, session, url):
        """Make API call with retry logic and rate limiting"""
        # Check for shutdown before making the request
        if self.shutdown_requested:
            return None
            
        async with self.semaphore:
            try:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:
                        retry_after = int(response.headers.get('Retry-After', 1))
                        logger.warning(f"Rate limit hit, retrying after {retry_after} seconds")
                        await asyncio.sleep(retry_after)
                        return await self._call_polygon_api(session, url)
                    else:
                        logger.error(f"API request failed: {response.status}")
                        return None
            except asyncio.TimeoutError:
                logger.warning(f"Timeout for URL: {url}")
                return None
            except asyncio.CancelledError:
                logger.info("API request cancelled")
                return None
            except Exception as e:
                logger.error(f"API request exception: {e}")
                return None

    async def _fetch_composite_tickers(self, session, composite_index):
        """Fetch all tickers for a specific composite index"""
        logger.info(f"Fetching tickers for composite index {composite_index}")
        all_results = []
        next_url = None
        
        # Use current date or backtest date
        if self.backtest_mode and self.backtest_date:
            date_param = self.backtest_date
            logger.info(f"Using historical date: {date_param}")
        else:
            date_param = datetime.now().strftime("%Y-%m-%d")
            
        # Different API endpoint for composite indices
        if composite_index == "^IXIC":  # NASDAQ Composite
            exchange = "XNAS"
        else:
            logger.error(f"Unknown composite index: {composite_index}")
            return []
            
        params = {
            "market": "stocks",
            "exchange": exchange,
            "active": "true",
            "limit": 1000,  # Maximum allowed by Polygon
            "apiKey": self.api_key,
            "date": date_param  # Add date parameter for historical data
        }
        
        # Initial URL construction
        url = f"{self.base_url}?{urlencode(params)}"
        page_count = 0
        
        while url and not self.shutdown_requested:
            data = await self._call_polygon_api(session, url)
            if not data or self.shutdown_requested:
                break
                
            results = data.get("results", [])
            # Filter for common stocks only and add composite index info
            stock_results = [
                {**r, "composite_index": composite_index} 
                for r in results 
                if r.get('type', '').upper() == 'CS'
            ]
            all_results.extend(stock_results)
            
            next_url = data.get("next_url", None)
            url = f"{next_url}&apiKey={self.api_key}" if next_url else None
            page_count += 1
            
            # Minimal delay for premium API access
            await asyncio.sleep(config.RATE_LIMIT_DELAY)
        
        if self.shutdown_requested:
            logger.info(f"Shutdown requested, aborting {composite_index} fetch")
            return []
            
        logger.info(f"Completed {composite_index}: {len(all_results)} stocks across {page_count} pages")
        return all_results

    async def _refresh_all_tickers_async(self):
        """Refresh all tickers with parallel composite index processing"""
        start_time = time.time()
        
        if self.backtest_mode and self.backtest_date:
            logger.info(f"Starting historical ticker refresh for {self.backtest_date}")
        else:
            logger.info("Starting full ticker refresh")
        
        # Check for shutdown before starting
        if self.shutdown_requested:
            logger.info("Shutdown requested, aborting refresh")
            return False
            
        async with aiohttp.ClientSession() as session:
            # Fetch all composite indices in parallel
            tasks = [self._fetch_composite_tickers(session, idx) for idx in self.composite_indices]
            composite_results = await asyncio.gather(*tasks)
            
            # Check for shutdown after fetching
            if self.shutdown_requested:
                logger.info("Shutdown requested during data processing")
                return False
                
            # Flatten results
            all_results = []
            for results in composite_results:
                if results:
                    all_results.extend(results)
        
        if not all_results:
            logger.warning("Refresh fetched no results")
            return False
            
        # Create DataFrame with only necessary columns
        new_df = pd.DataFrame(all_results)[["ticker", "name", "primary_exchange", "last_updated_utc", "type", "market", "locale", "currency_name"]]
        new_tickers = set(new_df['ticker'].tolist())
        
        with self.cache_lock:
            # For backtest mode, we don't update the main database
            if self.backtest_mode and self.backtest_date:
                # Store backtest results
                tickers_data = new_df.to_dict('records')
                inserted = self.db.upsert_backtest_tickers(tickers_data, self.backtest_date)
                logger.info(f"Stored {inserted} tickers in backtest database for {self.backtest_date}")
            else:
                # Original logic for live mode
                old_tickers = set(self.current_tickers_set)
                added = new_tickers - old_tickers
                removed = old_tickers - new_tickers
                
                # Convert DataFrame to list of dictionaries for database storage
                tickers_data = new_df.to_dict('records')
                
                # Update database
                inserted, updated = self.db.upsert_tickers(tickers_data)
                
                # Mark removed tickers as inactive
                if removed:
                    marked_inactive = self.db.mark_tickers_inactive(list(removed))
                    logger.info(f"Marked {marked_inactive} tickers as inactive")
                
                # Update in-memory cache
                self.ticker_cache = new_df
                self.current_tickers_set = new_tickers
                
                # Update known missing tickers
                rediscovered = added & self.known_missing_tickers
                if rediscovered:
                    self.known_missing_tickers -= rediscovered
                    self.db.update_metadata('known_missing_tickers', list(self.known_missing_tickers))
            
        self.last_refresh_time = time.time()
        
        if not self.backtest_mode:
            self.db.update_metadata('last_refresh_time', self.last_refresh_time)
        
        elapsed = time.time() - start_time
        logger.info(f"Ticker refresh completed in {elapsed:.2f}s")
        
        if not self.backtest_mode:
            logger.info(f"Total: {len(new_df)} | Added: {len(added)} | Removed: {len(removed)}")
            logger.info(f"Database: {inserted} inserted, {updated} updated")
        else:
            logger.info(f"Historical data: {len(new_df)} tickers for {self.backtest_date}")
            
        return True

    async def refresh_all_tickers(self):
        """Public async method to refresh tickers"""
        with self.refresh_lock:
            return await self._refresh_all_tickers_async()

    def start(self):
        if not self.active:
            self.active = True
            self.shutdown_requested = False
            self._init_cache()
            self.initial_refresh_complete.set()
            logger.info("Ticker scanner started")

    def stop(self):
        self.active = False
        self.shutdown_requested = True
        logger.info("Ticker scanner stopped")
        
    async def shutdown(self):
        """Cleanup resources"""
        self.stop()
        self.db.close_all_connections()
        logger.info("Ticker scanner shutdown complete")

    def get_current_tickers_list(self):
        with self.cache_lock:
            return self.ticker_cache['ticker'].tolist()

    def get_ticker_details(self, ticker):
        """Get details for a specific ticker from cache"""
        with self.cache_lock:
            result = self.ticker_cache[self.ticker_cache['ticker'] == ticker]
            return result.to_dict('records')[0] if not result.empty else None
            
    def search_tickers_db(self, search_term: str, limit: int = 50) -> List[Dict]:
        """Search tickers in database by name or symbol"""
        return self.db.search_tickers(search_term, limit)
        
    def get_ticker_history_db(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Get historical changes for a ticker from database"""
        return self.db.get_ticker_history(ticker, limit)
        
    def get_backtest_tickers(self, date_str: str) -> List[Dict]:
        """Get tickers for a specific backtest date"""
        return self.db.get_backtest_tickers(date_str)
        
    def get_backtest_tickers_by_year(self, year: int) -> List[Dict]:
        """Get tickers for a specific backtest year"""
        return self.db.get_backtest_tickers_by_year(year)
        
    def get_backtest_dates(self) -> List[str]:
        """Get all available backtest dates"""
        return self.db.get_backtest_dates()
        
    def get_backtest_years(self) -> List[int]:
        """Get all available backtest years"""
        return self.db.get_backtest_years()
        
    def get_backtest_final_results(self, start_date: str, end_date: str, run_id: str = "default") -> List[Dict]:
        """Get final backtest results for a specific date range"""
        return self.db.get_backtest_final_results(start_date, end_date, run_id)
        
    def get_backtest_final_results_by_year(self, year: int, run_id: str = "default") -> List[Dict]:
        """Get final backtest results for a specific year"""
        return self.db.get_backtest_final_results_by_year(year, run_id)
        
    def get_all_backtest_runs(self) -> List[Dict]:
        """Get all backtest runs with their date ranges"""
        return self.db.get_all_backtest_runs()
        
    def get_backtest_runs_by_date_range(self, start_date: str, end_date: str) -> List[Dict]:
        """Get all backtest runs for a specific date range"""
        return self.db.get_backtest_runs_by_date_range(start_date, end_date)
    
# ======================== MARKET REGIME SCANNER ======================== #
class MarketRegimeScanner:
    def __init__(self, ticker_scanner: PolygonTickerScanner):
        self.ticker_scanner = ticker_scanner
        self.api_key = config.POLYGON_API_KEY
        self.base_url = "https://api.polygon.io/v2/aggs/ticker"
        self.active = False
        self.shutdown_requested = False
        self.local_tz = get_localzone()
        self.semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
        self.db = DatabaseManager()  # Use main database for regular data
        self.backtest_db = BacktestDatabaseManager()  # Use backtest database for backtest data
        self.hmm_model = None
        self.gmm_model = None
        self.rf_model = None
        self.anomaly_detector = None
        self.pca = None
        self.kmeans = None
        self.training_data = None
        self.scaler = StandardScaler()
        self.model_version = config.MODEL_VERSION
        self.min_data_points = 100  # Minimum data points required for reliable analysis
        self.nan_threshold = 0.1    # Maximum allowed NaN percentage
        
        # New models for enhanced regime detection
        self.lstm_model = None
        self.cnn_model = None
        self.bayesian_model = None
        self.transition_model = BayesianTransitionModel()
        self.previous_regime = None  # This should already be here
        
        # Initialize with a default regime to avoid None issues
        self.default_regime = 1  # Sideways market as default
        
        # Backtesting attributes
        self.backtest_mode = False
        self.backtest_date = None
        
        # Focus on Nasdaq Composite for regime analysis
        self.market_indices = {
            "^IXIC": "COMP",    # NASDAQ Composite (primary focus)
        }
        
        logger.info(f"Using PostgreSQL database: {config.POSTGRES_DB}")
        
    async def _fetch_historical_data(self, session, ticker: str, days: int, end_date: datetime = None) -> Optional[pd.DataFrame]:
        """Fetch historical data for a ticker with optional end date for backtesting"""
        if end_date is None:
            end_date = datetime.now()
        
        start_date = end_date - timedelta(days=days)
        
        # Format dates for Polygon API
        end_date_str = end_date.strftime("%Y-%m-%d")
        start_date_str = start_date.strftime("%Y-%m-%d")
        
        # Use the correct symbol for Polygon API with I: prefix for indices
        polygon_symbol = self.market_indices.get(ticker, ticker)
        
        # Construct the API URL
        url = f"{self.base_url}/{polygon_symbol}/range/1/day/{start_date_str}/{end_date_str}?apiKey={self.api_key}"
        
        logger.debug(f"Fetching historical data from: {url}")
        
        async with self.semaphore:
            try:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('resultsCount', 0) > 0:
                            df = pd.DataFrame(data['results'])
                            df['t'] = pd.to_datetime(df['t'], unit='ms')
                            df.set_index('t', inplace=True)
                            df['ticker'] = ticker
                            return df
                        else:
                            logger.warning(f"No results for {ticker} (Polygon: {polygon_symbol}): {data}")
                            return None
                    elif response.status == 429:
                        retry_after = int(response.headers.get('Retry-After', 1))
                        logger.warning(f"Rate limit hit for {ticker}, retrying after {retry_after} seconds")
                        await asyncio.sleep(retry_after)
                        return await self._fetch_historical_data(session, ticker, days, end_date)
                    else:
                        error_text = await response.text()
                        logger.warning(f"Failed to fetch data for {ticker} (Polygon: {polygon_symbol}): {response.status} - {error_text}")
                        return None
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching data for {ticker}")
                return None
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
                return None
            
    def robust_feature_combination(self, basic_features, quant_features):
        """Robust method to combine features with enhanced NaN handling"""
        # Align indices first
        aligned_features = pd.concat([basic_features, quant_features], axis=1)
        
        # More aggressive NaN handling
        aligned_features = aligned_features.replace([np.inf, -np.inf], np.nan)
        
        # Drop columns with too many NaNs
        col_nan_threshold = len(aligned_features) * 0.3  # Keep columns with less than 30% NaN
        aligned_features = aligned_features.dropna(axis=1, thresh=col_nan_threshold)
        
        # Drop rows with too many NaNs
        row_nan_threshold = len(aligned_features.columns) * 0.7  # Keep rows with at least 70% data
        aligned_features = aligned_features.dropna(axis=0, thresh=row_nan_threshold)
        
        # Fill remaining NaNs with more sophisticated methods
        for col in aligned_features.columns:
            if aligned_features[col].isnull().any():
                # Use forward fill, then backward fill, then median
                aligned_features[col] = aligned_features[col].fillna(method='ffill').fillna(method='bfill')
                
                # If still NaN, use rolling median
                if aligned_features[col].isnull().any():
                    aligned_features[col] = aligned_features[col].fillna(
                        aligned_features[col].rolling(5, min_periods=1).median())
                    
                # If still NaN, use overall median
                if aligned_features[col].isnull().any():
                    aligned_features[col] = aligned_features[col].fillna(aligned_features[col].median())
        
        return aligned_features
    
    def validate_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Validate and ensure feature quality before training/prediction"""
        if features.empty:
            logger.error("Empty features DataFrame")
            return features
        
        # Check for and handle infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Calculate NaN percentage
        nan_percentage = features.isnull().sum().sum() / (features.shape[0] * features.shape[1])
        
        if nan_percentage > self.nan_threshold:
            logger.warning(f"High NaN percentage: {nan_percentage:.2%}. Applying aggressive cleaning.")
            
            # Drop columns with too many NaNs
            features = features.dropna(axis=1, thresh=int(features.shape[0] * 0.7))
            
            # Drop rows with too many NaNs
            features = features.dropna(axis=0, thresh=int(features.shape[1] * 0.7))
        
        # Fill remaining NaNs
        for col in features.columns:
            if features[col].isnull().any():
                # Try forward fill first
                features[col] = features[col].fillna(method='ffill')
                
                # Then backward fill
                features[col] = features[col].fillna(method='bfill')
                
                # Then use interpolation
                features[col] = features[col].interpolate()
                
                # Finally, use median if still NaN
                if features[col].isnull().any():
                    features[col] = features[col].fillna(features[col].median())
        
        logger.info(f"Feature validation completed. Final shape: {features.shape}, NaNs: {features.isnull().sum().sum()}")
        return features
                
    async def fetch_market_data(self, days: int = config.HISTORICAL_DATA_DAYS, end_date: datetime = None) -> pd.DataFrame:
        """Fetch historical market data for all market indices with enhanced validation"""
        logger.info(f"Fetching {days} days of market data for regime analysis")
        
        # Get the market indices
        tickers = list(self.market_indices.keys())
        
        all_data = []
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_historical_data(session, ticker, days, end_date) for ticker in tickers]
            results = await asyncio.gather(*tasks)
            
            for result in results:
                if result is not None and not result.empty:
                    all_data.append(result)
        
        # Data validation
        if not all_data:
            logger.error("No market data fetched")
            return pd.DataFrame()
        
        # Combine all data
        combined_data = pd.concat(all_data)
        
        # Ensure no duplicate indices
        combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
        
        # Ensure data is sorted by date
        combined_data = combined_data.sort_index()
        
        # Check if we have sufficient data
        if len(combined_data) < self.min_data_points:
            logger.warning(f"Insufficient data points: {len(combined_data)}. Minimum required: {self.min_data_points}")
            # Consider implementing a fallback or expanding the date range
        
        # Pivot to get OHLCV data for each ticker
        ohlcv_data = combined_data.pivot_table(index='t', columns='ticker', 
                                            values=['o', 'h', 'l', 'c', 'v'])
        
        # Additional data validation
        if ohlcv_data.isnull().sum().sum() > len(ohlcv_data) * 0.5:  # If more than 50% NaN
            logger.error("Poor data quality: more than 50% NaN values")
            # Consider implementing a fallback data source or retry mechanism
        
        logger.info(f"Fetched market data with shape: {ohlcv_data.shape}")
        return ohlcv_data
            
    def calculate_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate more sophisticated features for regime detection with enhanced NaN handling"""
        # Extract close prices for returns calculation
        close_prices = data.xs('c', axis=1, level=0) if isinstance(data.columns, pd.MultiIndex) else data
        
        features = pd.DataFrame(index=close_prices.index)
        
        for ticker in close_prices.columns:
            # Use more robust pct_change with better NaN handling
            returns = close_prices[ticker].pct_change().fillna(0)
            
            # Ensure we're working with the same index
            if len(returns) != len(features.index):
                returns = returns.reindex(features.index, fill_value=0)
            
            features[f'{ticker}_return'] = returns
            
            # Advanced volatility measures with increased min_periods
            volatility_20 = returns.rolling(window=20, min_periods=10).std()  # Increased from 5 to 10
            volatility_50 = returns.rolling(window=50, min_periods=20).std()  # Increased from 10 to 20
            
            # Safe division to avoid NaN/Inf
            volatility_ratio = pd.Series(1.0, index=features.index)  # Default to 1.0
            valid_mask = (volatility_50 > 1e-10) & (~volatility_20.isnull()) & (~volatility_50.isnull())
            volatility_ratio[valid_mask] = volatility_20[valid_mask] / volatility_50[valid_mask]
            
            features[f'{ticker}_volatility_ratio'] = volatility_ratio.fillna(1.0)
            
            # Get OHLC data if available
            if isinstance(data.columns, pd.MultiIndex):
                try:
                    highs = data.xs('h', axis=1, level=0)[ticker]
                    lows = data.xs('l', axis=1, level=0)[ticker]
                    opens = data.xs('o', axis=1, level=0)[ticker]
                    volumes = data.xs('v', axis=1, level=0)[ticker]
                    
                    # True range calculation with enhanced NaN handling
                    tr1 = highs - lows
                    tr2 = abs(highs - close_prices[ticker].shift(1).fillna(method='bfill'))
                    tr3 = abs(lows - close_prices[ticker].shift(1).fillna(method='bfill'))
                    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    features[f'{ticker}_atr'] = true_range.rolling(14, min_periods=7).mean().fillna(method='bfill')
                    
                    # Volume features with enhanced NaN handling
                    features[f'{ticker}_volume_ma'] = volumes.rolling(20, min_periods=10).mean().fillna(method='bfill')
                    vol_ma_50 = volumes.rolling(50, min_periods=25).mean()
                    features[f'{ticker}_volume_ratio'] = (volumes / vol_ma_50).fillna(1.0)
                    
                except KeyError:
                    logger.warning(f"OHLCV data not available for {ticker}")
                    # Fill with default values
                    features[f'{ticker}_atr'] = pd.Series(0, index=features.index)
                    features[f'{ticker}_volume_ma'] = pd.Series(0, index=features.index)
                    features[f'{ticker}_volume_ratio'] = pd.Series(1.0, index=features.index)
            
            # Advanced momentum indicators with increased min_periods
            moving_avg_20 = close_prices[ticker].rolling(window=20, min_periods=10).mean().reindex(features.index)
            moving_avg_50 = close_prices[ticker].rolling(window=50, min_periods=25).mean().reindex(features.index)
            
            # Safe division for momentum ratio
            momentum_ratio = pd.Series(1.0, index=features.index)  # Default to 1.0
            valid_mask = (moving_avg_50 > 1e-10) & (~moving_avg_20.isnull()) & (~moving_avg_50.isnull())
            momentum_ratio[valid_mask] = moving_avg_20[valid_mask] / moving_avg_50[valid_mask]
            
            features[f'{ticker}_momentum_ratio'] = momentum_ratio.fillna(1.0)
            
            # Volatility clustering feature with error handling and increased min_periods
            vol_clustering = returns.rolling(window=20, min_periods=10).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 2 and not np.isclose(x.std(), 0) else 0, 
                raw=False
            )
            features[f'{ticker}_volatility_clustering'] = vol_clustering.reindex(features.index).fillna(0)
            
            # Add RSI with safe division and increased min_periods
            delta = close_prices[ticker].diff().reindex(features.index).fillna(0)
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=7).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=7).mean()
            
            # Safe RSI calculation
            rs = pd.Series(1.0, index=features.index)  # Default to 1.0 (RSI 50)
            valid_mask = (loss > 1e-10) & (~gain.isnull()) & (~loss.isnull())
            rs[valid_mask] = gain[valid_mask] / loss[valid_mask]
            rsi = np.where(rs > 0, 100 - (100 / (1 + rs)), 50)
            features[f'{ticker}_rsi'] = pd.Series(rsi, index=features.index).fillna(50)
            
            # Add MACD with error handling and increased min_periods
            try:
                exp12 = close_prices[ticker].ewm(span=12, adjust=False, min_periods=6).mean().reindex(features.index)
                exp26 = close_prices[ticker].ewm(span=26, adjust=False, min_periods=13).mean().reindex(features.index)
                macd = exp12 - exp26
                features[f'{ticker}_macd'] = macd.fillna(0)
                features[f'{ticker}_macd_signal'] = macd.ewm(span=9, adjust=False, min_periods=5).mean().fillna(0)
            except:
                features[f'{ticker}_macd'] = pd.Series(0, index=features.index)
                features[f'{ticker}_macd_signal'] = pd.Series(0, index=features.index)
            
            # Add Bollinger Bands with safe calculations and increased min_periods
            rolling_mean = close_prices[ticker].rolling(window=20, min_periods=10).mean().reindex(features.index)
            rolling_std = close_prices[ticker].rolling(window=20, min_periods=10).std().reindex(features.index)
            
            features[f'{ticker}_bollinger_upper'] = (rolling_mean + (rolling_std * 2)).fillna(method='bfill')
            features[f'{ticker}_bollinger_lower'] = (rolling_mean - (rolling_std * 2)).fillna(method='bfill')
            
            # Safe Bollinger % calculation
            denominator = features[f'{ticker}_bollinger_upper'] - features[f'{ticker}_bollinger_lower']
            bollinger_pct = pd.Series(0.5, index=features.index)  # Default to 0.5 (middle of band)
            valid_mask = (denominator > 1e-10) & (~denominator.isnull())
            bollinger_pct[valid_mask] = (
                (close_prices[ticker].reindex(features.index)[valid_mask] - 
                features[f'{ticker}_bollinger_lower'][valid_mask]) / 
                denominator[valid_mask]
            )
            features[f'{ticker}_bollinger_pct'] = bollinger_pct.fillna(0.5)
        
        # Enhanced NaN handling
        nan_counts = features.isnull().sum()
        if nan_counts.any():
            logger.warning(f"NaN counts before final cleaning: {nan_counts.sum()}")
            
            # Replace inf/-inf with NaN first
            features = features.replace([np.inf, -np.inf], np.nan)
            
            # Drop rows with excessive NaNs (more than 30% of columns)
            features = features.dropna(thresh=len(features.columns) * 0.7)
            
            # Fill remaining NaNs with appropriate values using multiple strategies
            for col in features.columns:
                if features[col].isnull().any():
                    # First try forward fill
                    features[col] = features[col].fillna(method='ffill')
                    
                    # Then backward fill
                    features[col] = features[col].fillna(method='bfill')
                    
                    # Then use interpolation
                    features[col] = features[col].interpolate()
                    
                    # Finally, use median for numeric columns
                    if features[col].isnull().any():
                        if features[col].dtype in [np.float64, np.int64]:
                            median_val = features[col].median()
                            if not np.isnan(median_val):
                                features[col] = features[col].fillna(median_val)
                            else:
                                features[col] = features[col].fillna(0)
                        else:
                            features[col] = features[col].fillna(0)
        
        logger.info(f"Calculated advanced features with shape: {features.shape}, NaNs: {features.isnull().sum().sum()}")
        return features

    def calculate_quant_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Advanced quantitative features for regime detection with NaN handling"""
        # Extract close prices for returns calculation
        close_prices = data.xs('c', axis=1, level=0) if isinstance(data.columns, pd.MultiIndex) else data
        
        features = pd.DataFrame(index=close_prices.index)
        
        for ticker in close_prices.columns:
            returns = close_prices[ticker].pct_change().dropna()
            
            # Volatility features
            features[f'{ticker}_realized_vol'] = returns.rolling(20, min_periods=5).std()
            
            # Parkinson volatility estimator (if we have OHLC data)
            if isinstance(data.columns, pd.MultiIndex):
                try:
                    highs = data.xs('h', axis=1, level=0)[ticker]
                    lows = data.xs('l', axis=1, level=0)[ticker]
                    parkinson = np.log(highs / lows)
                    features[f'{ticker}_parkinson_vol'] = parkinson.rolling(20, min_periods=5).std() * (1/(4*np.log(2)))
                except KeyError:
                    logger.warning(f"High/Low data not available for {ticker}")
            
            # Advanced momentum
            features[f'{ticker}_ts_momentum'] = close_prices[ticker].pct_change(20)
            features[f'{ticker}_cross_sectional_mom'] = (
                returns.rolling(20, min_periods=5).mean() / 
                returns.rolling(20, min_periods=5).std()
            ).fillna(0)
            
            # Mean reversion features
            features[f'{ticker}_hurst'] = self.calculate_hurst_exponent(close_prices[ticker])
            features[f'{ticker}_half_life'] = self.calculate_mean_reversion_half_life(close_prices[ticker])
            
            # Jump detection
            features[f'{ticker}_jump_ratio'] = self.calculate_jump_ratio(returns)
            
            # Tail risk measures
            features[f'{ticker}_var_95'] = returns.rolling(100, min_periods=20).apply(
                lambda x: np.percentile(x, 5) if len(x) > 0 else np.nan
            )
            features[f'{ticker}_expected_shortfall'] = returns.rolling(100, min_periods=20).apply(
                lambda x: x[x <= np.percentile(x, 5)].mean() if len(x[x <= np.percentile(x, 5)]) > 0 else 0
            )
            
            # Microstructure features (if available)
            if isinstance(data.columns, pd.MultiIndex):
                try:
                    volumes = data.xs('v', axis=1, level=0)[ticker]
                    features[f'{ticker}_volume_zscore'] = (
                        volumes - volumes.rolling(50, min_periods=10).mean()
                    ) / volumes.rolling(50, min_periods=10).std()
                    
                    # Volume-price relationship
                    volume_pct_change = volumes.pct_change()
                    features[f'{ticker}_volume_price_correlation'] = returns.rolling(
                        20, min_periods=5
                    ).corr(volume_pct_change)
                except KeyError:
                    logger.warning(f"Volume data not available for {ticker}")
        
        # Market structure features
        returns_matrix = close_prices.pct_change().dropna()
        if len(returns_matrix) > 0:
            features['dispersion'] = returns_matrix.std(axis=1)
            features['skewness'] = returns_matrix.skew(axis=1)
            features['kurtosis'] = returns_matrix.kurtosis(axis=1)
        
        # Enhanced NaN handling
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().bfill().fillna(0)  # Forward then backward fill then zero fill
        
        return features
    
    def calculate_hurst_exponent(self, series, max_lag=50):
        """Calculate Hurst exponent for mean reversion detection"""
        lags = range(2, max_lag)
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    
    def calculate_mean_reversion_half_life(self, series):
        """Calculate half-life of mean reversion"""
        series = series.dropna()
        lagged = series.shift(1)
        delta = series - lagged
        beta = np.polyfit(lagged[1:], delta[1:], 1)[0]
        half_life = -np.log(2) / beta
        return half_life
    
    def calculate_jump_ratio(self, returns, window=20):
        """Calculate jump ratio to detect large price movements"""
        returns = returns.dropna()
        jump_threshold = returns.rolling(window).std() * 3
        jumps = (np.abs(returns) > jump_threshold).astype(int)
        return jumps.rolling(window).mean()
        
    def train_enhanced_ensemble(self, features: pd.DataFrame):
        """Train an enhanced ensemble of models for regime detection with robust NaN handling"""
        # Comprehensive data validation
        if features.empty:
            logger.error("No features available for training")
            return
        
        # Check for and handle NaN values
        nan_count = features.isnull().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in features, cleaning...")
            
            # Drop rows with excessive NaNs
            features = features.dropna(thresh=len(features.columns) * 0.8)
            
            # Fill remaining NaNs with appropriate values
            for col in features.columns:
                if features[col].isnull().any():
                    if features[col].dtype in [np.float64, np.int64]:
                        # Use median for numeric columns
                        median_val = features[col].median()
                        if not np.isnan(median_val):
                            features[col] = features[col].fillna(median_val)
                        else:
                            features[col] = features[col].fillna(0)
                    else:
                        features[col] = features[col].fillna(0)
        
        # Check if we still have enough data
        if len(features) < 50:
            logger.error(f"Insufficient data for training: {len(features)} samples")
            return
        
        scaled_features = self.scaler.fit_transform(features)
        
        # Store training data for rolling updates
        self.training_data = scaled_features
        
        # Dimensionality reduction
        self.pca = PCA(n_components=min(config.PCA_COMPONENTS, scaled_features.shape[1]))
        pca_features = self.pca.fit_transform(scaled_features)
        
        # HMM Model
        self.hmm_model = hmm.GaussianHMM(
            n_components=config.HMM_N_COMPONENTS,
            covariance_type=config.HMM_COVARIANCE_TYPE,
            n_iter=config.HMM_N_ITER,
            random_state=42
        )
        self.hmm_model.fit(pca_features)
        
        # GMM Model
        self.gmm_model = GaussianMixture(
            n_components=config.HMM_N_COMPONENTS,
            covariance_type=config.HMM_COVARIANCE_TYPE,
            max_iter=config.HMM_N_ITER,
            random_state=42
        )
        self.gmm_model.fit(pca_features)
        
        # K-means clustering
        self.kmeans = KMeans(n_clusters=config.N_CLUSTERS, random_state=42)
        self.kmeans.fit(pca_features)
        
        # Create consensus labels
        hmm_labels = self.hmm_model.predict(pca_features)
        gmm_labels = self.gmm_model.predict(pca_features)
        
        consensus_labels = []
        for i in range(len(hmm_labels)):
            votes = [hmm_labels[i], gmm_labels[i], self.kmeans.labels_[i]]
            consensus_labels.append(max(set(votes), key=votes.count))
        
        # Random Forest classifier
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(scaled_features, consensus_labels)
        
        # Anomaly detection
        self.anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
        self.anomaly_detector.fit(scaled_features)
        
        # Train deep learning models if enabled
        if config.USE_DEEP_LEARNING:
            self._train_deep_learning_models(scaled_features, consensus_labels)
        
        # Train Bayesian model for uncertainty estimation
        if config.UNCERTAINTY_ESTIMATION:
            try:
                self.bayesian_model = BayesianMarketRegime(n_regimes=config.HMM_N_COMPONENTS)
                self.bayesian_model.fit(scaled_features)
            except Exception as e:
                logger.warning(f"Bayesian model training failed: {e}")
                self.bayesian_model = None
        
        logger.info("Enhanced ensemble model training completed")
        logger.info(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
    def _train_deep_learning_models(self, features, labels):
        """Train deep learning models for regime detection"""
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(features)
        y_tensor = torch.LongTensor(labels)
        
        # Reshape for sequence models (add time dimension)
        seq_length = 10  # Use 10-day sequences
        n_samples = len(features) - seq_length + 1
        X_seq = np.zeros((n_samples, seq_length, features.shape[1]))
        y_seq = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            X_seq[i] = features[i:i+seq_length]
            y_seq[i] = labels[i+seq_length-1]
        
        X_seq_tensor = torch.FloatTensor(X_seq)
        y_seq_tensor = torch.LongTensor(y_seq)
        
        # Train LSTM model
        self.lstm_model = MarketRegimeLSTM(
            input_size=features.shape[1],
            hidden_size=64,
            num_layers=2,
            num_classes=len(np.unique(labels))
        )
        
        # Train CNN model
        self.cnn_model = MarketRegimeCNN(
            input_channels=features.shape[1],
            num_classes=len(np.unique(labels))
        )
        
        # Simple training loop (in practice, you'd want proper validation)
        criterion = nn.CrossEntropyLoss()
        optimizer_lstm = optim.Adam(self.lstm_model.parameters(), lr=0.001)
        optimizer_cnn = optim.Adam(self.cnn_model.parameters(), lr=0.001)
        
        # Train LSTM
        for epoch in range(10):  # Just a few epochs for demonstration
            optimizer_lstm.zero_grad()
            outputs = self.lstm_model(X_seq_tensor)
            loss = criterion(outputs, y_seq_tensor)
            loss.backward()
            optimizer_lstm.step()
            
        # Train CNN
        for epoch in range(10):
            optimizer_cnn.zero_grad()
            outputs = self.cnn_model(X_seq_tensor)
            loss = criterion(outputs, y_seq_tensor)
            loss.backward()
            optimizer_cnn.step()
            
        logger.info("Deep learning models trained")
        
    def predict_enhanced_regime(self, features: pd.DataFrame) -> Tuple[int, float, Dict]:
        """Predict using enhanced ensemble approach with NaN handling"""
        # Check for and handle NaN values in prediction data
        if features.isnull().any().any():
            logger.warning(f"Prediction features contain {features.isnull().sum().sum()} NaN values, filling with zeros")
            features = features.fillna(0)
        
        scaled_features = self.scaler.transform(features)
        pca_features = self.pca.transform(scaled_features)
        
        # Get predictions from all models
        hmm_pred = self.hmm_model.predict(pca_features[-5:])  # Use last 5 days
        gmm_pred = self.gmm_model.predict(pca_features[-5:])
        kmeans_pred = self.kmeans.predict(pca_features[-5:])
        rf_pred = self.rf_model.predict(scaled_features[-5:])
        anomaly_score = self.anomaly_detector.score_samples(scaled_features[-5:])
        
        # Get probabilities for confidence calculation
        hmm_probs = self.hmm_model.predict_proba(pca_features[-5:])
        gmm_probs = self.gmm_model.predict_proba(pca_features[-5:])
        rf_probs = self.rf_model.predict_proba(scaled_features[-5:])
        
        # Use the most recent predictions
        recent_hmm = hmm_pred[-1]
        recent_gmm = gmm_pred[-1]
        recent_kmeans = kmeans_pred[-1]
        recent_rf = rf_pred[-1]
        
        # Calculate confidence based on model agreement and probabilities
        model_agreement = np.mean([
            recent_hmm == recent_gmm,
            recent_hmm == recent_kmeans,
            recent_hmm == recent_rf,
            recent_gmm == recent_kmeans,
            recent_gmm == recent_rf,
            recent_kmeans == recent_rf
        ])
        
        # Average the probabilities from models that support the final decision
        final_regime = recent_hmm  # Start with HMM as base
        regime_votes = [recent_hmm, recent_gmm, recent_kmeans, recent_rf]
        final_regime = max(set(regime_votes), key=regime_votes.count)
        
        supporting_probs = []
        if recent_hmm == final_regime:
            supporting_probs.append(np.max(hmm_probs, axis=1)[-1])
        if recent_gmm == final_regime:
            supporting_probs.append(np.max(gmm_probs, axis=1)[-1])
        if recent_rf == final_regime:
            supporting_probs.append(np.max(rf_probs, axis=1)[-1])
        
        confidence = np.mean(supporting_probs) if supporting_probs else 0.5
        confidence = confidence * (0.7 + 0.3 * model_agreement)  # Scale by agreement
        
        # Adjust confidence for anomalies
        anomaly_factor = 1.0 - min(1.0, max(0.0, (1.0 - np.mean(anomaly_score)) * 2))  # Map [-1, 1] to [0, 1]
        confidence = confidence * (0.8 + 0.2 * anomaly_factor)  # Reduce confidence in anomalous conditions
        
        # Bayesian uncertainty estimation
        bayesian_uncertainty = 0.5  # Default uncertainty
        if config.UNCERTAINTY_ESTIMATION and self.bayesian_model:
            try:
                bayesian_probs, bayesian_uncert, _ = self.bayesian_model.predict_proba(scaled_features[-1:])
                bayesian_confidence = 1.0 - bayesian_uncert[0]
                # Blend traditional confidence with Bayesian confidence
                confidence = 0.7 * confidence + 0.3 * bayesian_confidence
                bayesian_uncertainty = bayesian_uncert[0]
            except Exception as e:
                logger.warning(f"Bayesian prediction failed: {e}")
        
        # Deep learning predictions
        if config.USE_DEEP_LEARNING and self.lstm_model and self.cnn_model:
            # Prepare sequence data for deep learning models
            seq_length = 10
            if len(scaled_features) >= seq_length:
                try:
                    X_seq = scaled_features[-seq_length:].reshape(1, seq_length, -1)
                    X_seq_tensor = torch.FloatTensor(X_seq)
                    
                    # LSTM prediction
                    lstm_output = self.lstm_model(X_seq_tensor)
                    lstm_probs = torch.softmax(lstm_output, dim=1).detach().numpy()[0]
                    lstm_pred = np.argmax(lstm_probs)
                    
                    # CNN prediction
                    cnn_output = self.cnn_model(X_seq_tensor)
                    cnn_probs = torch.softmax(cnn_output, dim=1).detach().numpy()[0]
                    cnn_pred = np.argmax(cnn_probs)
                    
                    # Add to voting
                    regime_votes.extend([lstm_pred, cnn_pred])
                    
                    # Update final regime based on all models
                    final_regime = max(set(regime_votes), key=regime_votes.count)
                    
                    # Update confidence with deep learning models
                    if lstm_pred == final_regime:
                        supporting_probs.append(lstm_probs[lstm_pred])
                    if cnn_pred == final_regime:
                        supporting_probs.append(cnn_probs[cnn_pred])
                    
                    if supporting_probs:
                        confidence = np.mean(supporting_probs)
                except Exception as e:
                    logger.warning(f"Deep learning prediction failed: {e}")
        
        # Initialize transition probability variables with defaults
        transition_prob = 0.5  # Default probability
        transition_uncertainty = 1.0  # Maximum uncertainty
        
        # Transition probability adjustment (only if we have a previous regime)
        if config.REGIME_TRANSITION_PROBABILITIES:
            if self.previous_regime is not None:
                try:
                    transition_prob = self.transition_model.get_transition_prob(self.previous_regime, final_regime)
                    transition_uncertainty = self.transition_model.get_transition_uncertainty(self.previous_regime)
                    
                    # Adjust confidence based on transition probability
                    # Low probability transitions reduce confidence
                    confidence = confidence * (0.5 + 0.5 * transition_prob)
                    
                    # High uncertainty about transitions also reduces confidence
                    confidence = confidence * (1.0 - 0.3 * transition_uncertainty)
                    
                    # Update transition model
                    self.transition_model.update(self.previous_regime, final_regime)
                except Exception as e:
                    logger.warning(f"Transition probability calculation failed: {e}")
            else:
                # First prediction, initialize with default values
                transition_prob = 0.25  # Equal probability for 4 regimes
                transition_uncertainty = 1.0  # Maximum uncertainty
        
        # Store current regime for next prediction
        self.previous_regime = final_regime
        
        # Prepare feature values for storage
        feature_values = {
            col: features[col].iloc[-1] for col in features.columns
        }
        feature_values['anomaly_score'] = np.mean(anomaly_score)
        feature_values['model_agreement'] = model_agreement
        
        # Add uncertainty measures
        if config.UNCERTAINTY_ESTIMATION:
            feature_values['bayesian_uncertainty'] = bayesian_uncertainty
        
        # Add transition probabilities if available
        if config.REGIME_TRANSITION_PROBABILITIES:
            feature_values['transition_probability'] = transition_prob
            feature_values['transition_uncertainty'] = transition_uncertainty
        
        return final_regime, confidence, feature_values
        
    def interpret_regime(self, regime: int, features: Dict) -> Dict:
        """Enhanced regime interpretation with contextual information"""
        base_interpretation = {
            0: {"label": "Bear Market", "color": "red", "description": "Declining prices, high volatility"},
            1: {"label": "Sideways Market", "color": "gray", "description": "Range-bound prices, moderate volatility"},
            2: {"label": "Bull Market", "color": "green", "description": "Rising prices, low to moderate volatility"},
            3: {"label": "High Volatility", "color": "orange", "description": "Extreme price movements in both directions"}
        }
        
        # Default to unknown if regime not in interpretation
        interpretation = base_interpretation.get(regime, {
            "label": f"Unknown Regime {regime}",
            "color": "purple",
            "description": "Unclassified market regime"
        })
        
        # Add contextual details based on features
        volatility = features.get('^IXIC_volatility_ratio', 1.0)
        momentum = features.get('^IXIC_momentum_ratio', 1.0)
        rsi = features.get('^IXIC_rsi', 50)
        anomaly_score = features.get('anomaly_score', 0)
        
        # Enhanced description based on market conditions
        details = []
        
        if volatility > 1.5:
            details.append("High volatility environment")
        elif volatility < 0.8:
            details.append("Low volatility environment")
            
        if momentum > 1.05:
            details.append("Strong upward momentum")
        elif momentum < 0.95:
            details.append("Strong downward momentum")
            
        if rsi > 70:
            details.append("Overbought conditions")
        elif rsi < 30:
            details.append("Oversold conditions")
            
        if anomaly_score < -0.5:
            details.append("Anomalous market behavior detected")
            
        # Add uncertainty information
        if 'bayesian_uncertainty' in features:
            uncertainty = features['bayesian_uncertainty']
            if uncertainty > 0.7:
                details.append("High prediction uncertainty")
            elif uncertainty > 0.4:
                details.append("Moderate prediction uncertainty")
            else:
                details.append("Low prediction uncertainty")
                
        # Add transition information
        if 'transition_probability' in features:
            trans_prob = features['transition_probability']
            if trans_prob < 0.2:
                details.append("Rare regime transition")
            elif trans_prob < 0.4:
                details.append("Uncommon regime transition")
                
        if details:
            interpretation["details"] = ", ".join(details)
        
        return interpretation
            
    async def scan_market_regime(self):
        """Perform a complete market regime scan with enhanced data validation"""
        if self.shutdown_requested:
            return False
            
        logger.info("Starting market regime scan")
        start_time = time.time()
        
        try:
            # Determine end date for data fetching
            end_date = None
            if self.backtest_mode and self.backtest_date:
                end_date = datetime.strptime(self.backtest_date, "%Y-%m-%d")
            
            # Fetch market data
            market_data = await self.fetch_market_data(end_date=end_date)
            if market_data.empty:
                logger.error("No market data available for regime analysis")
                return False
                
            # Data validation
            if market_data.isnull().sum().sum() > len(market_data) * 0.5:  # If more than 50% NaN
                logger.error("Insufficient quality data for regime detection")
                return False
                
            # Calculate features
            basic_features = self.calculate_advanced_features(market_data)
            quant_features = self.calculate_quant_features(market_data)
            
            # Validate features before combining
            basic_features = self.validate_features(basic_features)
            quant_features = self.validate_features(quant_features)
            
            # Check if we have enough data
            if basic_features.empty or quant_features.empty or len(basic_features) < 50:
                logger.error("Not enough data to calculate features")
                return False
                
            # Combine features with robust method
            features = self.robust_feature_combination(basic_features, quant_features)
            
            # Final validation
            features = self.validate_features(features)
            
            # Final NaN check and handling
            if features.isnull().any().any():
                logger.warning(f"Final features contain {features.isnull().sum().sum()} NaN values, cleaning...")
                features = features.replace([np.inf, -np.inf], np.nan)
                features = features.ffill().bfill().fillna(0)
                
            if features.empty:
                logger.error("No features calculated for regime analysis")
                return False
                
            # Train models if not already trained
            if self.hmm_model is None:
                self.train_enhanced_ensemble(features)
                
            # Predict current regime
            regime, confidence, feature_values = self.predict_enhanced_regime(features)
                
            interpretation = self.interpret_regime(regime, feature_values)
            regime_label = interpretation["label"]
            
            # Save to appropriate database
            timestamp = datetime.now()
            
            if self.backtest_mode and self.backtest_date:
                self.backtest_db.save_backtest_market_regime(
                    self.backtest_date, timestamp, regime, regime_label, confidence, feature_values, self.model_version
                )
            else:
                self.db.save_market_regime(
                    timestamp, regime, regime_label, confidence, feature_values, self.model_version
                )
                logger.info(f"Current market regime: {regime_label} (confidence: {confidence:.2f})")
            
            elapsed = time.time() - start_time
            logger.info(f"Market regime scan completed in {elapsed:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during market regime scan: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    def start(self):
        if not self.active:
            self.active = True
            self.shutdown_requested = False
            logger.info("Market regime scanner started")
            
    def stop(self):
        self.active = False
        self.shutdown_requested = True
        logger.info("Market regime scanner stopped")
        
    async def shutdown(self):
        """Cleanup resources"""
        self.stop()
        self.db.close_all_connections()
        self.backtest_db.close_all_connections()
        logger.info("Market regime scanner shutdown complete")
        
    def get_backtest_regimes_by_date(self, backtest_date: str) -> List[Dict]:
        """Get backtest market regimes for a specific date"""
        return self.backtest_db.get_backtest_regimes_by_date(backtest_date)
        
    def get_backtest_dates(self) -> List[str]:
        """Get all available backtest dates"""
        return self.backtest_db.get_backtest_dates()

# ======================== BACKTESTER ======================== #
class Backtester:
    def __init__(self, ticker_scanner, regime_scanner=None):
        self.ticker_scanner = ticker_scanner
        self.regime_scanner = regime_scanner
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    async def run_backtest(self, start_date, end_date=None, test_tickers=True, test_regime=True):
        """
        Run backtest for a specific date range
        Args:
            start_date: datetime object or string in YYYY-MM-DD format
            end_date: datetime object or string in YYYY-MM-DD format (optional)
            test_tickers: Whether to test ticker fetching
            test_regime: Whether to test market regime detection
        """
        if end_date is None:
            end_date = start_date
            
        # Convert to datetime if strings are provided
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
        logger.info(f"Starting backtest from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"Testing: Tickers={test_tickers}, Market Regime={test_regime}")
        logger.info(f"Run ID: {self.run_id}")
        
        # Dictionary to track ticker availability across all dates
        ticker_availability = defaultdict(list)
        
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Only weekdays
                date_str = current_date.strftime("%Y-%m-%d")
                
                if test_tickers:
                    # Check if we already have ticker data for this date
                    existing_ticker_data = self.ticker_scanner.db.get_backtest_tickers(date_str)
                    if not existing_ticker_data:
                        await self.run_single_day_ticker_backtest(current_date)
                
                if test_regime and self.regime_scanner:
                    # Check if we already have regime data for this date
                    existing_regime_data = self.regime_scanner.get_backtest_regimes_by_date(date_str)
                    if not existing_regime_data:
                        await self.run_single_day_regime_backtest(current_date)
            
            current_date += timedelta(days=1)
        
        # If testing tickers, filter tickers to only those available for the entire period
        if test_tickers:
            # Get all dates in the range
            all_dates = [d.strftime("%Y-%m-%d") for d in self._date_range(start_date, end_date) if d.weekday() < 5]
            
            # Track which tickers were available on each date
            for date_str in all_dates:
                ticker_data = self.ticker_scanner.db.get_backtest_tickers(date_str)
                if ticker_data:
                    for ticker in ticker_data:
                        ticker_availability[ticker['ticker']].append(date_str)
            
            # Filter tickers to only those available for the entire period
            full_period_tickers = []
            for ticker, available_dates in ticker_availability.items():
                # Check if ticker was available for all dates in the range
                if len(available_dates) == len(all_dates):
                    # Get the most recent data for this ticker
                    latest_date = max(available_dates)
                    ticker_data = self.ticker_scanner.db.execute_query(
                        "SELECT * FROM backtest_tickers WHERE ticker = ? AND date = ?",
                        (ticker, latest_date)
                    )
                    if ticker_data:
                        full_period_tickers.append(ticker_data[0])
            
            # Save the filtered results to the database
            if full_period_tickers:
                start_str = start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")
                inserted = self.ticker_scanner.db.upsert_backtest_final_results(full_period_tickers, start_str, end_str, self.run_id)
                logger.info(f"Saved {inserted} tickers to database that were active throughout the entire period")
            else:
                logger.warning("No tickers were active throughout the entire period")
            
        logger.info("Backtest completed")
        
    def _date_range(self, start_date, end_date):
        """Generate a range of dates between start_date and end_date"""
        for n in range(int((end_date - start_date).days) + 1):
            yield start_date + timedelta(n)
        
    async def run_single_day_ticker_backtest(self, target_date):
        """
        Run ticker backtest for a single day
        """
        logger.info(f"Running ticker backtest for {target_date.strftime('%Y-%m-%d')}")
        
        # Format date for API
        date_str = target_date.strftime("%Y-%m-%d")
        
        # Check if we already have data for this date
        existing_data = self.ticker_scanner.db.get_backtest_tickers(date_str)
        if existing_data:
            logger.info(f"Using cached ticker data for {date_str} ({len(existing_data)} tickers)")
            return True
        
        # Temporarily set scanner to backtest mode
        original_mode = self.ticker_scanner.backtest_mode
        self.ticker_scanner.backtest_mode = True
        self.ticker_scanner.backtest_date = date_str
        
        try:
            # Use the existing refresh method but with historical date
            success = await self.ticker_scanner.refresh_all_tickers()
            if success:
                logger.info(f"Successfully fetched ticker data for {date_str}")
                return True
            else:
                logger.warning(f"Failed to fetch ticker data for {date_str}")
                return False
        except Exception as e:
            logger.error(f"Error during ticker backtest for {date_str}: {e}")
            return False
        finally:
            # Restore original mode
            self.ticker_scanner.backtest_mode = original_mode
            self.ticker_scanner.backtest_date = None
            
    async def run_single_day_regime_backtest(self, target_date):
        """
        Run market regime backtest for a single day
        """
        if not self.regime_scanner:
            logger.error("No regime scanner provided for backtesting")
            return False
            
        logger.info(f"Running market regime backtest for {target_date.strftime('%Y-%m-%d')}")
        
        # Format date for API
        date_str = target_date.strftime("%Y-%m-%d")
        
        # Check if we already have data for this date
        existing_data = self.regime_scanner.get_backtest_regimes_by_date(date_str)
        if existing_data:
            logger.info(f"Using cached regime data for {date_str}")
            return True
        
        # Temporarily set scanner to backtest mode
        original_mode = self.regime_scanner.backtest_mode
        self.regime_scanner.backtest_mode = True
        self.regime_scanner.backtest_date = date_str
        
        try:
            # Use the existing scan method but with historical date
            success = await self.regime_scanner.scan_market_regime()
            if success:
                logger.info(f"Successfully fetched regime data for {date_str}")
                return True
            else:
                logger.warning(f"Failed to fetch regime data for {date_str}")
                return False
        except Exception as e:
            logger.error(f"Error during regime backtest for {date_str}: {e}")
            return False
        finally:
            # Restore original mode
            self.regime_scanner.backtest_mode = original_mode
            self.regime_scanner.backtest_date = None

# ======================== SCHEDULER ======================== #
async def run_scheduled_ticker_refresh(scanner):
    """Run immediate scan on startup and then daily at scheduled time"""
    # Run immediate scan on startup
    logger.info("Starting immediate ticker scan on startup")
    try:
        success = await scanner.refresh_all_tickers()
        if success:
            logger.info("Initial ticker scan completed successfully")
        else:
            logger.warning("Initial ticker scan encountered errors")
    except asyncio.CancelledError:
        logger.info("Initial ticker scan cancelled")
        return
    except Exception as e:
        logger.error(f"Error during initial ticker scan: {e}")
    
    # Continue with daily scans
    while scanner.active and not scanner.shutdown_requested:
        now = datetime.now(scanner.local_tz)
        
        # Calculate next run time (today at 8:30 AM)
        target_time = datetime.strptime(config.SCAN_TIME, "%H:%M").time()
        target_datetime = now.replace(
            hour=target_time.hour,
            minute=target_time.minute,
            second=0,
            microsecond=0
        )
        
        # If we already passed today's scheduled time, set for tomorrow
        if now > target_datetime:
            target_datetime += timedelta(days=1)
        
        sleep_seconds = (target_datetime - now).total_seconds()
        hours = sleep_seconds // 3600
        minutes = (sleep_seconds % 3600) // 60

        logger.info(f"Next ticker refresh scheduled at {target_datetime} ({hours} hours and {minutes} minutes from now)")
        
        # Wait until scheduled time, but check every second if we should stop
        while sleep_seconds > 0 and scanner.active and not scanner.shutdown_requested:
            try:
                # Sleep in small increments to be responsive to shutdown requests
                await asyncio.sleep(min(1, sleep_seconds))
                sleep_seconds -= 1
            except asyncio.CancelledError:
                logger.info("Sleep interrupted by shutdown")
                return
            
        if not scanner.active or scanner.shutdown_requested:
            break
            
        # Run the refresh
        logger.info("Starting scheduled ticker refresh")
        try:
            success = await scanner.refresh_all_tickers()
            if success:
                logger.info("Scheduled ticker refresh completed successfully")
            else:
                logger.warning("Scheduled ticker refresh encountered errors")
        except asyncio.CancelledError:
            logger.info("Ticker refresh cancelled")
            return
        except Exception as e:
            logger.error(f"Error during scheduled ticker refresh: {e}")

async def run_scheduled_regime_scan(regime_scanner, wait_for_ticker=True):
    """Run immediate regime scan on startup and then at scheduled intervals aligned with local clock"""
    # Wait for ticker scan to complete if requested
    if wait_for_ticker:
        logger.info("Waiting for initial ticker scan to complete before starting market regime scan")
        await asyncio.sleep(5)  # Give ticker scan a head start
    
    # Run immediate scan on startup
    logger.info("Starting immediate market regime scan")
    try:
        success = await regime_scanner.scan_market_regime()
        if success:
            logger.info("Initial market regime scan completed successfully")
        else:
            logger.warning("Initial market regime scan encountered errors")
    except asyncio.CancelledError:
        logger.info("Initial market regime scan cancelled")
        return
    except Exception as e:
        logger.error(f"Error during initial market regime scan: {e}")
    
    # Continue with scheduled scans aligned to local clock
    while regime_scanner.active and not regime_scanner.shutdown_requested:
        now = datetime.now(regime_scanner.local_tz)
        
        # Calculate next full hour
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        sleep_seconds = (next_hour - now).total_seconds()
        
        hours = sleep_seconds // 3600
        minutes = (sleep_seconds % 3600) // 60

        logger.info(f"Next market regime scan at {next_hour.strftime('%H:%M:%S')} ({hours} hours and {minutes} minutes from now)")
        
        # Wait until scheduled time, but check every second if we should stop
        while sleep_seconds > 0 and regime_scanner.active and not regime_scanner.shutdown_requested:
            try:
                # Sleep in small increments to be responsive to shutdown requests
                await asyncio.sleep(min(1, sleep_seconds))
                sleep_seconds -= 1
            except asyncio.CancelledError:
                logger.info("Sleep interrupted by shutdown")
                return
            
        if not regime_scanner.active or regime_scanner.shutdown_requested:
            break
            
        # Run the scan
        logger.info("Starting scheduled market regime scan")
        try:
            success = await regime_scanner.scan_market_regime()
            if success:
                logger.info("Scheduled market regime scan completed successfully")
            else:
                logger.warning("Scheduled market regime scan encountered errors")
        except asyncio.CancelledError:
            logger.info("Market regime scan cancelled")
            return
        except Exception as e:
            logger.error(f"Error during scheduled market regime scan: {e}")

# ======================== MAIN EXECUTION ======================== #
async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stock Ticker Fetcher and Market Regime Scanner with Backtesting')
    parser.add_argument('--search', type=str, help='Search for a ticker by name or symbol')
    parser.add_argument('--history', type=str, help='Get history for a specific ticker')
    parser.add_argument('--list', action='store_true', help='List all active tickers')
    parser.add_argument('--regime', action='store_true', help='Get current market regime')
    parser.add_argument('--regime-history', type=int, nargs='?', const=7, help='Get market regime history for past N days (default: 7)')
    
    # Backtesting arguments
    parser.add_argument('--backtest', type=str, help='Run backtest for a specific date (YYYY-MM-DD)')
    parser.add_argument('--backtest-range', type=str, help='Run backtest for a date range (YYYY-MM-DD:YYYY-MM-DD)')
    parser.add_argument('--backtest-year', type=int, help='Run backtest for a specific year')
    parser.add_argument('--backtest-tickers-only', action='store_true', help='Run backtest for tickers only')
    parser.add_argument('--backtest-regime-only', action='store_true', help='Run backtest for market regime only')
    parser.add_argument('--list-backtests', action='store_true', help='List available backtest dates')
    parser.add_argument('--list-backtest-years', action='store_true', help='List available backtest years')
    parser.add_argument('--list-backtest-runs', action='store_true', help='List available backtest runs')
    parser.add_argument('--show-backtest-results', type=str, help='Show results for a specific backtest run (format: YYYY-MM-DD:YYYY-MM-DD)')
    parser.add_argument('--show-year-results', type=int, help='Show results for a specific year')
    parser.add_argument('--show-regime-backtest', type=str, help='Show regime backtest results for a specific date (YYYY-MM-DD)')
    parser.add_argument('--list-regime-backtests', action='store_true', help='List available regime backtest dates')
    parser.add_argument('--run-id', type=str, default="default", help='Specify a run ID for backtest results')
    
    args = parser.parse_args()
    
    ticker_scanner = PolygonTickerScanner()
    regime_scanner = MarketRegimeScanner(ticker_scanner)
    
    # Check if we're running in backtest mode
    if (args.backtest or args.backtest_range or args.backtest_year or 
        args.list_backtests or args.list_backtest_years or args.list_backtest_runs or 
        args.show_backtest_results or args.show_year_results or
        args.show_regime_backtest or args.list_regime_backtests or
        args.backtest_tickers_only or args.backtest_regime_only):
        
        if args.list_backtests:
            # List available backtest dates
            dates = ticker_scanner.get_backtest_dates()
            if dates:
                print("Available backtest dates:")
                for date in dates:
                    print(f"  {date}")
            else:
                print("No backtest data available")
            return
        
        if args.list_backtest_years:
            # List available backtest years
            years = ticker_scanner.get_backtest_years()
            if years:
                print("Available backtest years:")
                for year in years:
                    print(f"  {year}")
            else:
                print("No backtest data available")
            return
            
        if args.list_backtest_runs:
            # List available backtest runs
            runs = ticker_scanner.get_all_backtest_runs()
            if runs:
                print("Available backtest runs:")
                for run in runs:
                    print(f"  {run['run_id']}: {run['start_date']} to {run['end_date']} (Years: {run['start_year']}-{run['end_year']})")
            else:
                print("No backtest runs available")
            return
            
        if args.show_backtest_results:
            # Show results for a specific backtest run
            start_date, end_date = args.show_backtest_results.split(':')
            results = ticker_scanner.get_backtest_final_results(start_date, end_date, args.run_id)
            if results:
                print(f"Backtest results for {start_date} to {end_date} (Run ID: {args.run_id}):")
                print(f"Found {len(results)} tickers that were active throughout the period")
                for result in results[:10]:  # Show first 10 results
                    print(f"  {result['ticker']}: {result['name']}")
                if len(results) > 10:
                    print(f"  ... and {len(results) - 10} more")
            else:
                print(f"No results found for {start_date} to {end_date} with run ID {args.run_id}")
            return
            
        if args.show_year_results:
            # Show results for a specific year
            year = args.show_year_results
            results = ticker_scanner.get_backtest_final_results_by_year(year, args.run_id)
            if results:
                print(f"Backtest results for year {year} (Run ID: {args.run_id}):")
                print(f"Found {len(results)} tickers that were active in this year")
                for result in results[:10]:  # Show first 10 results
                    print(f"  {result['ticker']}: {result['name']} (From {result['start_date']} to {result['end_date']})")
                if len(results) > 10:
                    print(f"  ... and {len(results) - 10} more")
            else:
                print(f"No results found for year {year} with run ID {args.run_id}")
            return
                
        if args.show_regime_backtest:
            # Show regime backtest results for a specific date
            date_str = args.show_regime_backtest
            results = regime_scanner.get_backtest_regimes_by_date(date_str)
            if results:
                print(f"Regime backtest results for {date_str}:")
                for result in results:
                    interpretation = regime_scanner.interpret_regime(result['regime'], json.loads(result['features']))
                    print(f"  {result['timestamp']}: {interpretation['label']} (confidence: {result['confidence']:.2f})")
            else:
                print(f"No regime backtest results found for {date_str}")
            return
                
        if args.list_regime_backtests:
            # List available regime backtest dates
            dates = regime_scanner.get_backtest_dates()
            if dates:
                print("Available regime backtest dates:")
                for date in dates:
                    print(f"  {date}")
            else:
                print("No regime backtest data available")
            return
        
        backtester = Backtester(ticker_scanner, regime_scanner)
        backtester.run_id = args.run_id  # Use the provided run ID
        
        # Determine what to test
        test_tickers = not args.backtest_regime_only
        test_regime = not args.backtest_tickers_only
        
        if args.backtest:
            # Single date backtest
            if test_tickers:
                await backtester.run_single_day_ticker_backtest(datetime.strptime(args.backtest, "%Y-%m-%d"))
            if test_regime:
                await backtester.run_single_day_regime_backtest(datetime.strptime(args.backtest, "%Y-%m-%d"))
        elif args.backtest_range:
            # Date range backtest
            start_str, end_str = args.backtest_range.split(':')
            start_date = datetime.strptime(start_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_str, "%Y-%m-%d")
            await backtester.run_backtest(start_date, end_date, test_tickers, test_regime)
        elif args.backtest_year:
            # Year backtest
            start_date = datetime(args.backtest_year, 1, 1)
            end_date = datetime(args.backtest_year, 12, 31)
            await backtester.run_backtest(start_date, end_date, test_tickers, test_regime)
    else:
        # Handle other command line arguments
        if args.search:
            results = ticker_scanner.search_tickers_db(args.search)
            if results:
                print(f"Found {len(results)} matching tickers:")
                for result in results:
                    print(f"{result['ticker']}: {result['name']} ({result['primary_exchange']})")
            else:
                print("No matching tickers found")
            return
        
        if args.history:
            results = ticker_scanner.get_ticker_history_db(args.history)
            if results:
                print(f"History for {args.history}:")
                for result in results:
                    print(f"{result['change_date']}: {result['change_type']}")
            else:
                print(f"No history found for {args.history}")
            return
        
        if args.list:
            results = ticker_scanner.db.get_all_active_tickers()
            if results:
                print(f"Found {len(results)} active tickers:")
                for result in results:
                    print(f"{result['ticker']}: {result['name']} ({result['primary_exchange']})")
            else:
                print("No active tickers found")
            return
            
        if args.regime:
            regime_scanner.start()
            await regime_scanner.scan_market_regime()
            latest_regime = regime_scanner.regime_db.get_latest_regime()
            if latest_regime:
                interpretation = regime_scanner.interpret_regime(latest_regime['regime'], json.loads(latest_regime['features']))
                print(f"Current market regime: {interpretation['label']}")
                print(f"Confidence: {latest_regime['confidence']:.2f}")
                print(f"Description: {interpretation['description']}")
                if 'details' in interpretation:
                    print(f"Details: {interpretation['details']}")
                print(f"Timestamp: {latest_regime['timestamp']}")
            else:
                print("No market regime data available")
            await regime_scanner.shutdown()
            return
            
        if args.regime_history is not None:
            days = args.regime_history if args.regime_history > 0 else 7
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            results = regime_scanner.regime_db.get_regimes_by_date_range(start_date, end_date)
            if results:
                print(f"Market regime history for the past {days} days:")
                for result in results:
                    interpretation = regime_scanner.interpret_regime(result['regime'], json.loads(result['features']))
                    print(f"{result['timestamp']}: {interpretation['label']} (confidence: {result['confidence']:.2f})")
            else:
                print("No market regime history available")
            return
        
        # Normal operation - run both scanners
        ticker_scanner.start()
        regime_scanner.start()
        
        # Wait for initial cache load
        await asyncio.get_event_loop().run_in_executor(None, ticker_scanner.initial_refresh_complete.wait)
        
        # Create tasks for the schedulers - ticker first, then regime with a delay
        ticker_scheduler_task = asyncio.create_task(run_scheduled_ticker_refresh(ticker_scanner))
        
        # Wait a moment for the ticker scan to start
        await asyncio.sleep(2)
        
        regime_scheduler_task = asyncio.create_task(run_scheduled_regime_scan(regime_scanner))
        
        # Set up signal handlers
        loop = asyncio.get_event_loop()
        stop_event = asyncio.Event()
        
        def signal_handler():
            """Handle shutdown signals immediately"""
            print("\nReceived interrupt signal, shutting down...")
            ticker_scanner.stop()
            regime_scanner.stop()
            stop_event.set()
            # Cancel all tasks
            for task in asyncio.all_tasks(loop):
                if task is not asyncio.current_task():
                    task.cancel()
        
        # Register signal handlers
        if sys.platform != "win32":
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, signal_handler)
        else:
            # Windows signal handling
            signal.signal(signal.SIGINT, lambda s, f: signal_handler())
        
        try:
            # Create a task for the stop_event.wait() coroutine
            stop_task = asyncio.create_task(stop_event.wait())
            
            # Wait for either shutdown event or task completion
            done, pending = await asyncio.wait(
                [ticker_scheduler_task, regime_scheduler_task, stop_task], 
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel the scheduler tasks if they're still running
            if not ticker_scheduler_task.done():
                ticker_scheduler_task.cancel()
                try:
                    await ticker_scheduler_task
                except asyncio.CancelledError:
                    pass
                    
            if not regime_scheduler_task.done():
                regime_scheduler_task.cancel()
                try:
                    await regime_scheduler_task
                except asyncio.CancelledError:
                    pass
                    
            # Cancel the stop task if it's still running
            if not stop_task.done():
                stop_task.cancel()
                try:
                    await stop_task
                except asyncio.CancelledError:
                    pass
                    
        except asyncio.CancelledError:
            logger.info("Main task cancelled")
        finally:
            # Shutdown the scanners
            await ticker_scanner.shutdown()
            await regime_scanner.shutdown()

if __name__ == "__main__":
    # Windows event loop policy
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")