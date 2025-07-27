import requests
import pandas as pd
import math
from datetime import datetime, timedelta
import time
from config import POLYGON_API_KEY

class SmartStopLoss:
    """
    Advanced dynamic trailing stop loss system with false trigger protection.
    
    Features:
    - ATR-based volatility adjustment
    - ADX-based trend strength adaptation
    - Volume confirmation
    - Multi-layer false trigger protection
    - Growth potential assessment
    - Closing price confirmation
    
    Usage:
    >>> stop_manager = SmartStopLoss(entry_price=100.0, atr=2.5, adx=30)
    >>> for new_bar in data_stream:
    >>>     current_stop = stop_manager.update(new_bar)
    >>>     if stop_manager.should_exit(new_bar):
    >>>         exit_position()
    """
    
    def __init__(self, entry_price, atr, adx, activation_percent=0.05, base_multiplier=1.5):
        """
        Initialize stop loss manager
        
        :param entry_price: Entry price of position
        :param atr: Average True Range at entry
        :param adx: ADX (Average Directional Index) at entry
        :param activation_percent: Profit % before trailing activates (default: 5%)
        :param base_multiplier: Base ATR multiplier for stops (default: 1.5)
        """
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
        """
        Update stop loss with new price data
        
        :param current_bar: Dictionary containing:
            - 'high': Current period high
            - 'low': Current period low
            - 'close': Current closing price
            - 'volume': Current volume
            - 'adx': Current ADX value
            - 'rsi': Current RSI value (optional)
            - 'avg_volume': Average volume (e.g., 30-day)
        :return: Current stop loss price
        """
        # Update tracking variables
        current_high = current_bar['high']
        current_low = current_bar['low']
        current_close = current_bar['close']
        current_adx = current_bar['adx']
        
        # Calculate volume ratio
        volume_ratio = current_bar['volume'] / current_bar.get('avg_volume', 1e6)  # Default 1M if not provided
        
        # Update highest high
        if current_high > self.highest_high:
            self.highest_high = current_high
            self.consecutive_confirmations = 0  # Reset on new highs

        # Calculate growth potential
        adx_strength = min(1.0, current_adx / 50)  # Cap at 1.0 for ADX > 50
        volume_boost = min(1.5, max(0.5, volume_ratio / 1.2))  # Cap at 1.5x
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
            adx_factor = 1.0 + (min(current_adx, 60) / 100)  # Cap at 1.6
            
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
        """
        Determine if we should override a potential exit signal
        
        :param current_bar: Dictionary with current period data
        :return: True if we should hold despite stop proximity
        """
        current_low = current_bar['low']
        current_close = current_bar['close']
        rsi = current_bar.get('rsi', 50)  # Default to neutral if not provided
        volatility_ratio = current_bar.get('volatility_ratio', 1.0)  # Current ATR / Initial ATR
        
        # 1. Strong momentum override
        price_change = (current_close / self.previous_close - 1) * 100
        if price_change > 3:  # 3%+ daily gain
            return True
            
        # 2. Volatility contraction protection
        if volatility_ratio < 0.7:  # Volatility decreased >30%
            return True
            
        # 3. ADX strengthening override
        if self.growth_potential > 1.5:
            return True
            
        # 4. Oversold bounce prevention
        if rsi < 40 and (current_close > current_low * 1.02):  # Recovered from lows
            return True
            
        # 5. Confirmation sequence requirement
        if self.consecutive_confirmations < 2:  # Require multiple signals
            self.consecutive_confirmations += 1
            return True
            
        return False

    def should_exit(self, current_bar):
        """
        Comprehensive exit decision with confirmation layers
        
        :param current_bar: Dictionary with current period data
        :return: True if exit conditions are met
        """
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
                if volume_ratio > 1.2:  # High volume breakdown
                    return True
                    
        return False

    def get_status(self):
        """Return current stop loss status"""
        return {
            'current_stop': self.current_stop,
            'growth_potential': self.growth_potential,
            'activated': self.activated,
            'distance_to_stop': (self.current_stop / self.previous_close - 1) * 100,
            'consecutive_confirmations': self.consecutive_confirmations
        }

class PolygonTrendScanner:
    def __init__(self, max_tickers=None):
        self.api_key = POLYGON_API_KEY
        self.base_url = "https://api.polygon.io/v2"
        self.tickers = self.load_tickers(max_tickers)
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    def load_tickers(self, max_tickers=None):
        try:
            with open('all_tickers.txt', 'r') as f:
                tickers = [line.strip() for line in f if line.strip()]
                if not tickers:
                    print("Warning: all_tickers.txt is empty")
                    return []
                
                # Apply max_tickers limit if specified
                if max_tickers is not None and max_tickers > 0:
                    tickers = tickers[:max_tickers]
                
                return tickers
        except FileNotFoundError:
            print("Error: all_tickers.txt not found in the current directory")
            return []
        except Exception as e:
            print(f"Error loading tickers: {str(e)}")
            return []

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
            print(f"Error fetching {ticker}: {str(e)}")
            return None

    def calculate_indicators(self, df):
        if df is None or len(df) < 200:
            return None
            
        try:
            # Calculate basic indicators
            latest = df.iloc[-1].copy()
            sma_50 = df['Close'].rolling(50).mean().iloc[-1]
            sma_200 = df['Close'].rolling(200).mean().iloc[-1]
            
            # Calculate distance from SMAs
            distance_sma50 = ((latest['Close'] - sma_50) / sma_50) * 100
            distance_sma200 = ((latest['Close'] - sma_200) / sma_200) * 100
            
            # Calculate True Range and ATR
            df['prev_close'] = df['Close'].shift(1)
            df['H-L'] = df['High'] - df['Low']
            df['H-PC'] = abs(df['High'] - df['prev_close'])
            df['L-PC'] = abs(df['Low'] - df['prev_close'])
            df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            atr = df['TR'].rolling(14).mean().iloc[-1]
            
            # Calculate ADX components
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
            print(f"Indicator calculation error: {str(e)}")
            return None
        finally:
            # Clean up intermediate columns
            cols_to_drop = ['prev_close', 'H-L', 'H-PC', 'L-PC', 'TR']
            for col in cols_to_drop:
                if col in df.columns:
                    df.drop(col, axis=1, inplace=True)

    def scan_tickers(self):
        """Scan all tickers and return DataFrame sorted by trend score"""
        results = []
        
        for i, ticker in enumerate(self.tickers):
            # Respect Polygon's rate limits (5 requests per second)
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
                        base_multiplier=1.5
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
                        'ATR_Multiplier': 1.5,
                        'Activation_Percent': 5.0
                    })
                    
            except Exception as e:
                print(f"Error evaluating {ticker}: {str(e)}")
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
    
    def save_results(self, df, filename='polygon_trend_signals.csv'):
        df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")

if __name__ == '__main__':
    # Get user input for number of tickers to scan
    while True:
        try:
            max_tickers_input = input("Enter number of tickers to scan (leave empty for all): ")
            if not max_tickers_input.strip():
                max_tickers = None
                break
            max_tickers = int(max_tickers_input)
            if max_tickers > 0:
                break
            print("Please enter a positive number or leave empty for all tickers.")
        except ValueError:
            print("Please enter a valid number or leave empty for all tickers.")
    
    scanner = PolygonTrendScanner(max_tickers=max_tickers)
    print(f"Scanning {len(scanner.tickers)} tickers using Polygon.io API...")
    
    results = scanner.scan_tickers()
    
    if not results.empty:
        print("\nTop Trend Following Candidates:")
        print(results.head(20))
        scanner.save_results(results)
    else:
        print("\nNo stocks meet the criteria currently.")