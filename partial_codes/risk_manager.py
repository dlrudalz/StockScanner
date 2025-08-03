from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest, TrailingStopOrderRequest
import time

class RiskManager:
    def __init__(self, api_key, secret_key, paper=True):
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)
        
    def place_hard_stop(self, ticker, qty, stop_price):
        try:
            stop_order = StopOrderRequest(
                symbol=ticker,
                qty=qty,
                side=OrderSide.SELL,
                stop_price=stop_price,
                time_in_force=TimeInForce.GTC,
            )
            return self.trading_client.submit_order(stop_order)
        except Exception as e:
            print(f"Error placing hard stop: {str(e)}")
            return None

    def execute_hard_stop(self, ticker, qty):
        try:
            market_order = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            return self.trading_client.submit_order(market_order)
        except Exception as e:
            print(f"Error executing hard stop: {str(e)}")
            return None

    def calculate_position_size(self, price, account_buying_power, allocation_per_ticker=100, max_portfolio_allocation=0.9):
        """Calculate position size based on account buying power"""
        max_alloc = min(allocation_per_ticker, account_buying_power * max_portfolio_allocation)
        position_size = max(1, int(max_alloc / price))
        
        # Verify we're not exceeding buying power
        estimated_cost = position_size * price
        if estimated_cost > account_buying_power:
            position_size = max(1, int(account_buying_power * max_portfolio_allocation / price))
            print(f"Reduced position size to {position_size} due to buying power constraints")
            
        return position_size
