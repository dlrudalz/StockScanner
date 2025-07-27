# stoploss.py
import pandas as pd

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


# ----------------------------
# Helper Functions for Backtesting
# ----------------------------

def calculate_technical_indicators(df, atr_period=14, adx_period=14, rsi_period=14):
    """
    Calculate technical indicators needed for stop loss system
    
    :param df: DataFrame with OHLCV data
    :return: DataFrame with added technical columns
    """
    # Calculate True Range
    df['prev_close'] = df['Close'].shift(1)
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['prev_close'])
    df['L-PC'] = abs(df['Low'] - df['prev_close'])
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    # Calculate ATR
    df['ATR'] = df['TR'].rolling(atr_period).mean()
    
    # Calculate ADX components
    df['UpMove'] = df['High'] - df['High'].shift(1)
    df['DownMove'] = df['Low'].shift(1) - df['Low']
    df['PlusDM'] = df.apply(lambda x: x['UpMove'] if x['UpMove'] > x['DownMove'] and x['UpMove'] > 0 else 0, axis=1)
    df['MinusDM'] = df.apply(lambda x: x['DownMove'] if x['DownMove'] > x['UpMove'] and x['DownMove'] > 0 else 0, axis=1)
    
    df['PlusDI'] = 100 * (df['PlusDM'].rolling(adx_period).mean() / df['ATR'])
    df['MinusDI'] = 100 * (df['MinusDM'].rolling(adx_period).mean() / df['ATR'])
    df['DX'] = 100 * abs(df['PlusDI'] - df['MinusDI']) / (df['PlusDI'] + df['MinusDI'])
    df['ADX'] = df['DX'].rolling(adx_period).mean()
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Clean up intermediate columns
    df.drop(['prev_close', 'H-L', 'H-PC', 'L-PC', 'TR', 'UpMove', 'DownMove', 
             'PlusDM', 'MinusDM', 'PlusDI', 'MinusDI', 'DX'], axis=1, inplace=True, errors='ignore')
    
    # Calculate average volume
    df['AvgVolume'] = df['Volume'].rolling(30).mean()
    
    return df

def backtest_stop_loss(entry_price, entry_date, df, initial_atr, initial_adx):
    """
    Backtest stop loss system on historical data
    
    :param entry_price: Entry price of position
    :param entry_date: Entry date (datetime or string)
    :param df: DataFrame with historical data (must include post-entry data)
    :param initial_atr: ATR at entry
    :param initial_adx: ADX at entry
    :return: Tuple (exit_price, exit_date, exit_reason)
    """
    # Convert entry_date if needed
    if isinstance(entry_date, str):
        entry_date = pd.to_datetime(entry_date)
    
    # Filter to data after entry
    df = df[df.index >= entry_date].copy()
    
    # Initialize stop loss
    stop_manager = SmartStopLoss(
        entry_price=entry_price,
        atr=initial_atr,
        adx=initial_adx,
        activation_percent=0.05,
        base_multiplier=1.5
    )
    
    # Track previous close for momentum calculation
    prev_close = entry_price
    
    for idx, row in df.iterrows():
        # Prepare current bar data
        current_bar = {
            'date': idx,
            'high': row['High'],
            'low': row['Low'],
            'close': row['Close'],
            'volume': row['Volume'],
            'adx': row['ADX'],
            'rsi': row['RSI'],
            'avg_volume': row['AvgVolume'],
            'volatility_ratio': row['ATR'] / initial_atr
        }
        
        # Update stop loss
        current_stop = stop_manager.update(current_bar)
        
        # Check exit conditions
        if stop_manager.should_exit(current_bar):
            return (row['Close'], idx, 'stop_loss')
            
        # Update previous close for next iteration
        prev_close = row['Close']
        
        # Optional: Check profit target or other exit conditions
        if row['Close'] > entry_price * 1.20:  # 20% profit target
            return (row['Close'], idx, 'profit_target')
            
    # If no exit triggered, exit at last close
    return (df.iloc[-1]['Close'], df.index[-1], 'no_exit')



# #integration tip
# # In your PolygonTrendScanner class
# from stoploss import SmartStopLoss

# # When processing results
# for ticker in selected_tickers:
#     entry_price = indicators['Close']
#     atr = indicators['ATR']
#     adx = indicators['ADX']
    
#     stop_system = SmartStopLoss(
#         entry_price=entry_price,
#         atr=atr,
#         adx=adx
#     )
    
#     # Add stop parameters to results
#     results.append({
#         'Ticker': ticker,
#         # ... other fields ...
#         'Initial_Stop': stop_system.current_stop,
#         'Activation_%': 5.0,
#         'ATR_Multiplier': 1.5
#     })

# #parameter tuning
# # Test different parameter combinations
# param_grid = {
#     'activation_percent': [0.03, 0.05, 0.08],
#     'base_multiplier': [1.2, 1.5, 2.0],
#     'adx_threshold': [25, 30, 35]
# }

# # Backtest and evaluate performance