import math
import requests
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
from datetime import datetime, timedelta
from position_manager import SmartStopLoss, SmartProfitTarget
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class PolygonTrendScanner:
    def __init__(self, max_tickers=None, polygon_api_key=None):
        self.api_key = polygon_api_key
        self.base_url = "https://api.polygon.io/v2"
        self.tickers = self.load_tickers(max_tickers)
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        self.start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        self.base_multiplier = 1.5
        self.session = self._create_session()
        
    def _create_session(self):
        """Create requests session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        return session

    def load_tickers(self, max_tickers=None):
        # Use a more comprehensive ticker list in production
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "DIS"][:max_tickers]

    def get_polygon_data(self, ticker):
        url = f"{self.base_url}/aggs/ticker/{ticker}/range/1/day/{self.start_date}/{self.end_date}"
        params = {"adjusted": "true", "apiKey": self.api_key}

        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            if data.get("resultsCount", 0) < 200:
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
            print(f"Error fetching {ticker}: {str(e)}")
            return None

    def calculate_indicators(self, df):
        if df is None or len(df) < 200:
            return None

        try:
            # Copy to avoid SettingWithCopyWarning
            df = df.copy()
            latest = df.iloc[-1].copy()
            
            # Calculate SMAs
            df['SMA_50'] = df["Close"].rolling(50).mean()
            df['SMA_200'] = df["Close"].rolling(200).mean()
            sma_50 = df['SMA_50'].iloc[-1]
            sma_200 = df['SMA_200'].iloc[-1]

            distance_sma50 = ((latest["Close"] - sma_50) / sma_50) * 100
            distance_sma200 = ((latest["Close"] - sma_200) / sma_200) * 100

            # Calculate True Range and ATR
            df["prev_close"] = df["Close"].shift(1)
            df["H-L"] = df["High"] - df["Low"]
            df["H-PC"] = abs(df["High"] - df["prev_close"])
            df["L-PC"] = abs(df["Low"] - df["prev_close"])
            df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
            atr = df["TR"].rolling(14).mean().iloc[-1]

            # Calculate ADX with smoothing
            df['UpMove'] = df['High'].diff()
            df['DownMove'] = -df['Low'].diff()
            df['PlusDM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
            df['MinusDM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)
            
            # Smooth DM values
            df['PlusDM'] = df['PlusDM'].ewm(alpha=1/14, adjust=False).mean()
            df['MinusDM'] = df['MinusDM'].ewm(alpha=1/14, adjust=False).mean()
            
            # Calculate DX and ADX
            df['PlusDI'] = 100 * (df['PlusDM'] / df['TR'].ewm(alpha=1/14, adjust=False).mean())
            df['MinusDI'] = 100 * (df['MinusDM'] / df['TR'].ewm(alpha=1/14, adjust=False).mean())
            df['DX'] = 100 * abs(df['PlusDI'] - df['MinusDI']) / (df['PlusDI'] + df['MinusDI'])
            adx = df['DX'].rolling(14).mean().iloc[-1]

            # Calculate RSI
            delta = df['Close'].diff(1)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]

            # Volume analysis
            avg_volume = df["Volume"].rolling(30).mean().iloc[-1]
            volume_ratio = latest["Volume"] / avg_volume if avg_volume > 0 else 1

            return {
                "Close": float(latest["Close"]),
                "SMA_50": float(sma_50),
                "SMA_200": float(sma_200),
                "Distance_SMA50": float(distance_sma50),
                "Distance_SMA200": float(distance_sma200),
                "Volume": float(latest["Volume"]),
                "AvgVolume": float(avg_volume),
                "Volume_Ratio": float(volume_ratio),
                "ATR": float(atr),
                "ADX": float(adx),
                "RSI": float(rsi),
            }

        except Exception as e:
            print(f"Indicator error: {str(e)}")
            return None

    def calculate_score(self, indicators):
        """Enhanced scoring system with multiple factors"""
        try:
            # Trend strength components (max 60 points)
            adx_strength = min(30, indicators["ADX"] * 0.6)  # 50 ADX = 30 points
            sma_distance = min(20, max(0, indicators["Distance_SMA50"]) * 0.4)  # 50% above SMA50 = 20 points
            trend_consistency = min(10, (indicators["Distance_SMA50"] - indicators["Distance_SMA200"]) * 0.2)
            
            # Volume/momentum components (max 40 points)
            volume_score = min(20, (indicators["Volume_Ratio"] - 1) * 10)  # 3x volume = 20 points
            rsi_score = 10 - abs(indicators["RSI"] - 50) * 0.2  # Centered around 50 RSI
            
            return min(100, sum([
                adx_strength,
                sma_distance,
                trend_consistency,
                volume_score,
                rsi_score
            ]))
        
        except Exception as e:
            print(f"Scoring error: {str(e)}")
            return 0

    def scan_tickers(self):
        """Scan tickers with progress tracking"""
        results = []
        tickers = self.tickers
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(self.process_ticker, t): t for t in tickers}
            
            for future in tqdm(as_completed(futures), total=len(tickers)):
                if result := future.result():
                    results.append(result)
        
        if not results:
            return pd.DataFrame()
            
        df_results = pd.DataFrame(results)
        df_results["Rank"] = df_results["Score"].rank(ascending=False).astype(int)
        return df_results.sort_values("Rank")[[
            "Rank", "Ticker", "Score", "Price", "ADX", "RSI", 
            "SMA50_Distance%", "SMA200_Distance%", "Volume_Ratio",
            "ATR", "Initial_Stop", "Hard_Stop", "Take_Profit", 
            "Risk_per_Share", "Risk_Percent"
        ]]

    def process_ticker(self, ticker):
        """Process a single ticker with enhanced filters"""
        try:
            data = self.get_polygon_data(ticker)
            if data is None: 
                return None

            indicators = self.calculate_indicators(data)
            if not indicators: 
                return None

            # Core trend filters
            above_sma50 = indicators["Close"] > indicators["SMA_50"]
            above_sma200 = indicators["Close"] > indicators["SMA_200"]
            valid_adx = indicators["ADX"] > 25
            valid_rsi = 40 < indicators["RSI"] < 70
            volume_ok = indicators["Volume_Ratio"] > 0.8
            
            if not all([above_sma50, above_sma200, valid_adx, valid_rsi, volume_ok]):
                return None

            # Calculate composite score
            score = self.calculate_score(indicators)
            if score < 40:  # Minimum quality threshold
                return None

            # Position management
            stop_system = SmartStopLoss(
                entry_price=indicators["Close"],
                atr=indicators["ATR"],
                adx=indicators["ADX"],
                activation_percent=0.03,  # Tighter activation
                base_multiplier=self.base_multiplier,
            )
            
            risk_per_share = indicators["Close"] - stop_system.current_stop
            risk_percent = (risk_per_share / indicators["Close"]) * 100

            return {
                "Ticker": ticker,
                "Score": round(score, 1),
                "Price": round(indicators["Close"], 2),
                "ADX": round(indicators["ADX"], 1),
                "RSI": round(indicators["RSI"], 1),
                "SMA50_Distance%": round(indicators["Distance_SMA50"], 1),
                "SMA200_Distance%": round(indicators["Distance_SMA200"], 1),
                "Volume_Ratio": round(indicators["Volume_Ratio"], 2),
                "ATR": round(indicators["ATR"], 2),
                "Initial_Stop": round(stop_system.current_stop, 2),
                "Hard_Stop": round(stop_system.hard_stop, 2),
                "Take_Profit": round(stop_system.profit_target.current_target, 2),
                "Risk_per_Share": round(risk_per_share, 2),
                "Risk_Percent": round(risk_percent, 2),
            }
            
        except Exception as e:
            print(f"Processing error ({ticker}): {str(e)}")
            return None