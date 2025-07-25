import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from config import POLYGON_API_KEY  # Import directly from config.py

class PolygonTrendScanner:
    def __init__(self):
        self.api_key = POLYGON_API_KEY  # Use the imported key
        self.base_url = "https://api.polygon.io/v2"
        self.tickers = self.load_tickers()
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
    def load_tickers(self):
        with open('all_tickers.txt', 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
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
            
        latest = df.iloc[-1].copy()
        
        # Calculate SMAs
        latest['SMA_50'] = df['Close'].rolling(50).mean().iloc[-1]
        latest['SMA_200'] = df['Close'].rolling(200).mean().iloc[-1]
        
        # Calculate ADX
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        plus_dm = df['High'].diff()
        minus_dm = -df['Low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        atr = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        latest['ADX'] = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).rolling(14).mean().iloc[-1]
        
        # Calculate ATR
        latest['ATR'] = tr.rolling(14).mean().iloc[-1]
        
        return latest
    
    def scan_tickers(self):
        results = []
        for i, ticker in enumerate(self.tickers):
            # Respect Polygon's rate limits (5 requests per second)
            if i > 0 and i % 5 == 0:
                time.sleep(1)
                
            print(f"Processing {ticker} ({i+1}/{len(self.tickers)})...", end='\r')
            
            data = self.get_polygon_data(ticker)
            latest = self.calculate_indicators(data)
            
            if latest is None:
                continue
                
            # Check entry conditions
            condition1 = latest['Close'] > latest['SMA_50'] and latest['Close'] > latest['SMA_200']
            condition2 = latest['ADX'] > 25
            
            if condition1 and condition2:
                results.append({
                    'Ticker': ticker,
                    'Price': latest['Close'],
                    'SMA_50': latest['SMA_50'],
                    'SMA_200': latest['SMA_200'],
                    'ADX': latest['ADX'],
                    'ATR': latest['ATR'],
                    'Distance_from_SMA50': (latest['Close'] - latest['SMA_50']) / latest['SMA_50'] * 100,
                    'Volume': latest['Volume']
                })
        
        # Sort by strongest ADX (best trends)
        results = sorted(results, key=lambda x: x['ADX'], reverse=True)
        return pd.DataFrame(results)
    
    def save_results(self, df, filename='polygon_trend_signals.csv'):
        df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")

if __name__ == '__main__':
    scanner = PolygonTrendScanner()
    print(f"Scanning {len(scanner.tickers)} tickers using Polygon.io API...")
    results = scanner.scan_tickers()
    
    if not results.empty:
        print("\nTop Trend Following Candidates:")
        print(results[['Ticker', 'Price', 'ADX', 'Distance_from_SMA50', 'Volume']].head(20))
        scanner.save_results(results)
    else:
        print("\nNo stocks meet the criteria currently.")