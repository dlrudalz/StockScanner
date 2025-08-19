import numpy as np
import pandas as pd
import requests
import math
from datetime import datetime, timedelta
import time
from config import POLYGON_API_KEY

class EnhancedBreakoutStrategy:
    """
    Advanced breakout detection system with upside potential assessment
    Combines: Volume surge, volatility contraction, RSI confirmation, and relative strength
    """
    
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
        # Existing update logic remains the same
        # ...
        return self.current_stop

class PolygonBreakoutScanner:
    def __init__(self, max_tickers=None):
        self.api_key = POLYGON_API_KEY
        self.base_url = "https://api.polygon.io/v2"
        self.tickers = self.load_tickers(max_tickers)
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Pre-cache SPY data for relative strength
        self.spy_data = self.get_polygon_data("SPY")
        self.spy_3mo_return = self.calculate_3mo_return(self.spy_data) if self.spy_data is not None else 0

    def load_tickers(self, max_tickers=None):
        # Existing implementation remains
        # ...

    def get_polygon_data(self, ticker):
        # Existing implementation remains
        # ...

    def calculate_3mo_return(self, df):
        """Calculate 3-month return percentage"""
        if len(df) < 63:
            return 0
        return ((df['Close'].iloc[-1] / df['Close'].iloc[-63]) - 1) * 100

    def calculate_rsi(self, prices, window=14):
        """Calculate RSI with Wilder's smoothing"""
        delta = prices.diff().dropna()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_indicators(self, df):
        if df is None or len(df) < 200:
            return None
            
        try:
            # Basic price data
            latest = df.iloc[-1].copy()
            high = latest['High']
            low = latest['Low']
            close = latest['Close']
            volume = latest['Volume']
            
            # Moving averages
            sma_50 = df['Close'].rolling(50).mean().iloc[-1]
            sma_200 = df['Close'].rolling(200).mean().iloc[-1]
            distance_sma50 = ((close - sma_50) / sma_50) * 100
            distance_sma200 = ((close - sma_200) / sma_200) * 100
            
            # Volatility metrics
            df['Range'] = df['High'] - df['Low']
            current_range = high - low
            avg_range_10d = df['Range'].rolling(10).mean().iloc[-1]
            volatility_ratio = current_range / avg_range_10d if avg_range_10d > 0 else 1.0
            
            # ATR calculation
            df['prev_close'] = df['Close'].shift(1)
            df['H-L'] = df['High'] - df['Low']
            df['H-PC'] = abs(df['High'] - df['prev_close'])
            df['L-PC'] = abs(df['Low'] - df['prev_close'])
            df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            atr = df['TR'].rolling(14).mean().iloc[-1]
            
            # ADX calculation (existing)
            # ...
            
            # Momentum indicators
            rsi = self.calculate_rsi(df['Close']).iloc[-1]
            if len(df) >= 10:
                ten_day_change = ((close / df['Close'].iloc[-10]) - 1) * 100
            else:
                ten_day_change = 0
                
            # Volume analysis
            avg_volume = df['Volume'].rolling(30).mean().iloc[-1]
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            
            # Breakout detection metrics
            consolidation_high = df['High'].rolling(20).max().iloc[-2]  # Previous period high
            breakout_confirmed = close > consolidation_high * 1.03  # 3% above resistance
            
            # Relative strength
            stock_3mo_return = self.calculate_3mo_return(df)
            relative_strength = stock_3mo_return - self.spy_3mo_return
            
            return {
                'Close': float(close),
                'High': float(high),
                'Low': float(low),
                'Volume': float(volume),
                'SMA_50': float(sma_50),
                'SMA_200': float(sma_200),
                'Distance_SMA50': float(distance_sma50),
                'Distance_SMA200': float(distance_sma200),
                'ATR': float(atr),
                'ADX': float(adx),
                'RSI': float(rsi),
                '10D_Change': float(ten_day_change),
                'AvgVolume': float(avg_volume),
                'Volume_Ratio': float(volume_ratio),
                'Consolidation_High': float(consolidation_high),
                'Breakout_Confirmed': breakout_confirmed,
                'Current_Range': float(current_range),
                'Volatility_Ratio': float(volatility_ratio),
                'Relative_Strength': float(relative_strength)
            }
            
        except Exception as e:
            print(f"Indicator calculation error: {str(e)}")
            return None

    def calculate_upside_score(self, indicators):
        """
        Calculate composite upside potential score (0-100)
        Components:
        - Breakout confirmation: 30%
        - Relative strength: 25%
        - Volume surge: 20%
        - Volatility contraction: 15%
        - RSI positioning: 10%
        """
        # Breakout confirmation (30 points)
        breakout_score = 30 if indicators['Breakout_Confirmed'] else 0
        
        # Relative strength (25 points)
        rs_score = min(25, max(0, indicators['Relative_Strength'] * 0.5))  # 1 point per 2% outperformance
        
        # Volume surge (20 points)
        vol_ratio = indicators['Volume_Ratio']
        volume_score = 0
        if vol_ratio > 2.0:
            volume_score = min(20, (vol_ratio - 2.0) * 10)  # 2x = 0, 3x = 10, 4x = 20
        
        # Volatility contraction (15 points)
        vol_contraction = indicators['Volatility_Ratio']
        volatility_score = 15 * max(0, 1 - min(1, vol_contraction/0.7))  # Full points at <0.3 ratio
        
        # RSI positioning (10 points)
        rsi = indicators['RSI']
        rsi_score = 0
        if 40 <= rsi <= 80:
            # Peak at 60 RSI
            rsi_score = 10 * (1 - abs(rsi - 60)/20)
        
        return min(100, breakout_score + rs_score + volume_score + volatility_score + rsi_score)

    def scan_tickers(self):
        results = []
        
        for i, ticker in enumerate(self.tickers):
            if i > 0 and i % 5 == 0:
                time.sleep(1)
                
            print(f"Processing {ticker} ({i+1}/{len(self.tickers)})...", end='\r')
            
            data = self.get_polygon_data(ticker)
            if data is None:
                continue
                
            indicators = self.calculate_indicators(data)
            if indicators is None:
                continue
                
            try:
                # Core breakout conditions
                above_sma50 = indicators['Close'] > indicators['SMA_50']
                above_sma200 = indicators['Close'] > indicators['SMA_200']
                strong_adx = indicators['ADX'] > 25
                volume_surge = indicators['Volume_Ratio'] > 2.0
                volatility_contraction = indicators['Volatility_Ratio'] < 0.7
                valid_rsi = 40 <= indicators['RSI'] <= 80
                
                # Must-pass filters
                if not (above_sma50 and above_sma200 and strong_adx and volume_surge 
                        and volatility_contraction and valid_rsi):
                    continue
                
                # Calculate upside potential score
                upside_score = self.calculate_upside_score(indicators)
                
                # Initialize breakout system
                breakout_system = EnhancedBreakoutStrategy(
                    entry_price=indicators['Close'],
                    atr=indicators['ATR'],
                    adx=indicators['ADX']
                )
                
                # Risk metrics
                risk_per_share = indicators['Close'] - breakout_system.current_stop
                risk_percent = (risk_per_share / indicators['Close']) * 100
                
                # Calculate profit potential
                profit_target = indicators['Close'] + (3 * indicators['ATR'])
                reward_risk_ratio = (profit_target - indicators['Close']) / risk_per_share
                
                results.append({
                    'Ticker': ticker,
                    'Upside_Score': round(upside_score, 1),
                    'Price': round(indicators['Close'], 2),
                    'Breakout_Level': round(indicators['Consolidation_High'], 2),
                    'Breakout_Confirmed': indicators['Breakout_Confirmed'],
                    'ADX': round(indicators['ADX'], 1),
                    'RSI': round(indicators['RSI'], 1),
                    'Relative_Strength': round(indicators['Relative_Strength'], 1),
                    'Vol_Ratio': round(indicators['Volatility_Ratio'], 2),
                    'Volume_Ratio': round(indicators['Volume_Ratio'], 1),
                    'SMA50_Dist%': round(indicators['Distance_SMA50'], 1),
                    'ATR': round(indicators['ATR'], 2),
                    '10D_Change%': round(indicators['10D_Change'], 1),
                    'Profit_Target': round(profit_target, 2),
                    'Reward_Risk_Ratio': round(reward_risk_ratio, 1),
                    'Upside_Potential%': round((profit_target/indicators['Close'] - 1)*100, 1)
                })
                    
            except Exception as e:
                print(f"Error evaluating {ticker}: {str(e)}")
                continue
        
        if results:
            df_results = pd.DataFrame(results)
            df_results['Rank'] = df_results['Upside_Score'].rank(ascending=False, method='min').astype(int)
            return df_results.sort_values('Upside_Score', ascending=False)
        return pd.DataFrame()
    
    def save_results(self, df, filename='breakout_signals.csv'):
        df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")

if __name__ == '__main__':
    scanner = PolygonBreakoutScanner(max_tickers=500)  # Limit to 500 for demo
    print(f"Scanning {len(scanner.tickers)} tickers for breakout opportunities...")
    
    results = scanner.scan_tickers()
    
    if not results.empty:
        print("\nTop Breakout Candidates:")
        print(results.head(20))
        scanner.save_results(results)
    else:
        print("\nNo breakout opportunities found.")