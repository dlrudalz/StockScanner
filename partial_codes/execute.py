# ======================== POSITION SIZING & TRADE EXECUTION ======================== #
class PositionManager:
    def __init__(self, trading_system):
        self.trading_system = trading_system
        self.position_risk = {}  # Track risk per position
        self.max_portfolio_risk = 0.15  # Max 15% of portfolio at risk
        self.sector_exposure = {}  # Track sector exposure
        self.volatility_factor = 1.0  # Adjust based on market volatility
        self.last_vix = 0.0
        self.order_history = []
        
    async def calculate_position_size(self, symbol, entry_price, stop_loss, atr, adx):
        """Calculate position size with multiple risk constraints"""
        # 1. Basic risk-per-trade calculation
        risk_per_share = entry_price - stop_loss
        max_risk_amount = self.trading_system.account_value * config.MAX_RISK_PERCENT
        base_size = max_risk_amount / risk_per_share if risk_per_share > 0 else 0
        
        # 2. Volatility scaling (VIX-based)
        await self.update_volatility_factor()
        volatility_size = base_size * self.volatility_factor
        
        # 3. Liquidity constraint (max 5% of average daily volume)
        avg_volume = await self.get_average_volume(symbol)
        volume_constrained_size = min(volatility_size, avg_volume * 0.05) if avg_volume else volatility_size
        
        # 4. Sector exposure constraint (max 25% per sector)
        sector = await self.get_sector(symbol)
        sector_exposure = self.sector_exposure.get(sector, 0)
        sector_constrained_size = min(volume_constrained_size, 
                                    (0.25 * self.trading_system.account_value - sector_exposure) / entry_price)
        
        # 5. Portfolio risk constraint
        current_risk = self.calculate_portfolio_risk()
        portfolio_constrained_size = min(sector_constrained_size, 
                                        (self.max_portfolio_risk * self.trading_system.account_value - current_risk) / risk_per_share)
        
        # Final sizing with integer shares
        position_size = max(1, int(portfolio_constrained_size))
        
        # Store risk metrics
        self.position_risk[symbol] = {
            'risk_per_share': risk_per_share,
            'position_value': position_size * entry_price,
            'risk_amount': position_size * risk_per_share,
            'sector': sector
        }
        
        # Update sector exposure
        if sector:
            self.sector_exposure[sector] = self.sector_exposure.get(sector, 0) + (position_size * entry_price)
        
        return position_size
    
    def calculate_portfolio_risk(self):
        """Calculate total portfolio risk"""
        return sum(pos['risk_amount'] for pos in self.position_risk.values())
    
    async def update_volatility_factor(self):
        """Update volatility scaling based on VIX"""
        # Only update once per hour
        if time.time() - getattr(self, 'last_vix_update', 0) < 3600:
            return
            
        try:
            vix_url = "https://api.polygon.io/v2/aggs/ticker/VIX/range/1/day?adjusted=true&sort=asc&limit=1"
            async with self.trading_system.session.get(vix_url) as response:
                data = await response.json()
                vix = data['results'][0]['c'] if data['resultsCount'] > 0 else 20
                
            # Normalize VIX to scaling factor (15-25 is normal range)
            if vix < 15:
                self.volatility_factor = 1.2  # Increase size in low volatility
            elif vix < 25:
                self.volatility_factor = 1.0
            elif vix < 35:
                self.volatility_factor = 0.8
            else:
                self.volatility_factor = 0.6  # Reduce size in high volatility
                
            self.last_vix = vix
            self.last_vix_update = time.time()
            logger.info(f"Updated volatility factor: VIX={vix}, Factor={self.volatility_factor}")
        except Exception as e:
            logger.error(f"VIX update failed: {str(e)}")
    
    async def get_average_volume(self, symbol):
        """Get 30-day average volume"""
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day?adjusted=true&sort=desc&limit=30"
            async with self.trading_system.session.get(url) as response:
                data = await response.json()
                if data['resultsCount'] > 0:
                    volumes = [r['v'] for r in data['results']]
                    return sum(volumes) / len(volumes)
        except Exception:
            return None
    
    async def get_sector(self, symbol):
        """Get stock sector from Polygon"""
        try:
            url = f"https://api.polygon.io/v3/reference/tickers/{symbol}?apiKey={config.POLYGON_API_KEY}"
            async with self.trading_system.session.get(url) as response:
                data = await response.json()
                return data['results'].get('sic_description', 'Unknown')
        except Exception:
            return 'Unknown'
    
    def estimate_slippage(self, symbol, quantity, order_type):
        """Estimate slippage based on liquidity and order size"""
        # Simple model: 0.1% for liquid stocks, 0.5% for illiquid
        liquidity = self.get_liquidity_tier(symbol)
        slippage_rates = {'liquid': 0.001, 'medium': 0.003, 'illiquid': 0.005}
        return slippage_rates.get(liquidity, 0.005)
    
    def get_liquidity_tier(self, symbol):
        """Classify stock liquidity based on average volume"""
        # In production, use more sophisticated classification
        avg_volume = self.position_risk[symbol].get('avg_volume', 0)
        if avg_volume > 1000000:
            return 'liquid'
        elif avg_volume > 500000:
            return 'medium'
        return 'illiquid'
    
    async def execute_trade(self, symbol, side, quantity, price, stop_loss, profit_target):
        """Execute trades with robust order management"""
        try:
            # 1. Determine optimal order type
            liquidity = self.get_liquidity_tier(symbol)
            order_type = 'limit' if liquidity in ['liquid', 'medium'] else 'vwap'
            
            # 2. Calculate limit price with buffer
            current_price = await self.get_realtime_price(symbol)
            if side == "BUY":
                limit_price = current_price * 1.005  # Pay up 0.5% for fills
            else:
                limit_price = current_price * 0.995  # Sell down 0.5%
            
            # 3. Estimate slippage
            slippage = self.estimate_slippage(symbol, quantity, order_type)
            
            # 4. Place primary order
            order_id = await self.place_order(symbol, side, quantity, order_type, limit_price)
            
            # 5. Place bracket orders (OCO - One Cancels Other)
            bracket_id = await self.place_bracket_order(
                symbol, 
                side, 
                quantity, 
                limit_price, 
                stop_loss, 
                profit_target
            )
            
            # 6. Record trade details
            self.order_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'limit_price': limit_price,
                'stop_loss': stop_loss,
                'profit_target': profit_target,
                'order_id': order_id,
                'bracket_id': bracket_id,
                'slippage_est': slippage,
                'status': 'pending'
            })
            
            logger.info(f"Order placed: {side} {quantity} {symbol} @ {limit_price:.2f}")
            return True
        except Exception as e:
            logger.error(f"Trade execution failed: {str(e)}")
            return False
    
    async def place_order(self, symbol, side, quantity, order_type, price=None):
        """Place order with broker API (placeholder)"""
        # In production, integrate with actual broker API
        logger.info(f"Placing {order_type} order: {side} {quantity} {symbol} @ {price or 'market'}")
        return f"ORDER-{int(time.time() * 1000)}"
    
    async def place_bracket_order(self, symbol, side, quantity, price, stop_loss, profit_target):
        """Place bracket order for stop loss and take profit"""
        # This would be implemented with broker's OCO order functionality
        logger.info(f"Placing bracket order: SL={stop_loss:.2f}, PT={profit_target:.2f}")
        return f"BRACKET-{int(time.time() * 1000)}"
    
    async def monitor_orders(self):
        """Monitor open orders and handle partial fills/timeouts"""
        active_orders = [o for o in self.order_history if o['status'] in ['pending', 'partial']]
        
        for order in active_orders:
            # Check order status with broker
            order_age = (datetime.now() - order['timestamp']).total_seconds() / 60
            
            # Handle timeouts
            if order_age > 5:  # 5 minutes without fill
                await self.handle_order_timeout(order)
            
            # Handle partial fills
            # (Implementation would depend on broker API)
            
    async def handle_order_timeout(self, order):
        """Handle orders that haven't filled within expected time"""
        logger.warning(f"Order {order['order_id']} timed out - adjusting")
        
        # Cancel original order
        await self.cancel_order(order['order_id'])
        
        # Determine new price based on current market
        current_price = await self.get_realtime_price(order['symbol'])
        
        # Adjust price based on side
        if order['side'] == "BUY":
            new_price = current_price * 1.01  # Increase bid by 1%
        else:
            new_price = current_price * 0.99  # Reduce ask by 1%
        
        # Place new order
        new_order_id = await self.place_order(
            order['symbol'],
            order['side'],
            order['quantity'],
            'limit',
            new_price
        )
        
        # Update order record
        order['limit_price'] = new_price
        order['order_id'] = new_order_id
        order['timestamp'] = datetime.now()
        order['status'] = 'adjusted'
        
    async def cancel_order(self, order_id):
        """Cancel order with broker"""
        logger.info(f"Cancelling order {order_id}")
        # Broker API implementation would go here
        return True

# ======================== TRADING SYSTEM ENHANCEMENTS ======================== #
class QuantTradingSystem:
    def __init__(self, ticker_scanner):
        # ... existing initialization ...
        self.position_manager = PositionManager(self)
        
    async def process_ticker(self, ticker):
        # ... existing processing ...
        
        # Enhanced position sizing
        position_size = await self.position_manager.calculate_position_size(
            ticker,
            indicators['Close'],
            current_stop,
            indicators['ATR'],
            indicators['ADX']
        )
        
        # Adjust for account size
        max_position_value = self.account_value * 0.05  # Max 5% per position
        position_value = indicators['Close'] * position_size
        if position_value > max_position_value:
            position_size = int(max_position_value / indicators['Close'])
        
        # ... rest of processing ...
        
        return {
            # ... existing fields ...
            'Position_Size': position_size,
            'Position_Value': round(indicators['Close'] * position_size, 2),
            'Risk_Amount%': round((risk_per_share * position_size) / self.account_value * 100, 2)
        }, None
        
    async def execute_trade(self, symbol, side, quantity, price, stop_loss, profit_target):
        """Execute trades through position manager"""
        return await self.position_manager.execute_trade(
            symbol, side, quantity, price, stop_loss, profit_target
        )
        
    async def monitor_positions(self):
        """Enhanced position monitoring"""
        # 1. Monitor open orders
        await self.position_manager.monitor_orders()
        
        # 2. Monitor existing positions
        # ... existing position monitoring ...
        
    async def place_trade(self, signal):
        """Place trade based on breakout signal"""
        if self.shutting_down:
            return False
            
        # Calculate position size
        position_size = signal['Position_Size']
        if position_size < 1:
            logger.warning(f"Position size too small for {signal['Ticker']}: {position_size}")
            return False
            
        # Execute trade
        return await self.execute_trade(
            signal['Ticker'],
            "BUY",
            position_size,
            signal['Price'],
            signal['Stop_Loss'],
            signal['Profit_Target']
        )

# ======================== SCANNER THREAD ENHANCEMENTS ======================== #
class ScannerThread(QThread):
    # ... existing code ...
    
    async def process_signal(self, signal):
        """Process a breakout signal and place trade"""
        # 1. Validate market conditions
        if not self.validate_market_conditions():
            self.log_message.emit(f"Skipping trade in current market conditions")
            return
            
        # 2. Check existing exposure
        if not self.validate_position_risk(signal):
            self.log_message.emit(f"Risk limits exceeded for {signal['Ticker']}")
            return
            
        # 3. Place trade
        if await self.trading_system.place_trade(signal):
            self.log_message.emit(f"Trade executed: {signal['Ticker']} {signal['Position_Size']} shares")
            # Add to open positions
            self.trading_system.open_positions.append({
                'symbol': signal['Ticker'],
                'entry_price': signal['Price'],
                'quantity': signal['Position_Size'],
                'stop_loss': signal['Stop_Loss'],
                'profit_target': signal['Profit_Target']
            })
            self.position_update.emit(self.trading_system.open_positions)
    
    def validate_market_conditions(self):
        """Check if market conditions are suitable for trading"""
        # 1. Market phase
        phase = self.calendar.get_market_phase()
        if phase not in ["pre_market", "market_hours"]:
            return False
            
        # 2. Volatility filter
        vix = self.trading_system.position_manager.last_vix
        if vix > 35:  # Extreme volatility
            return False
            
        # 3. Trend filter
        if self.trading_system.spy_sma_50 < self.trading_system.spy_sma_200:
            return False  # Bear market
            
        return True
    
    def validate_position_risk(self, signal):
        """Validate position against risk constraints"""
        # 1. Single position risk
        if signal['Risk_Amount%'] > config.MAX_RISK_PERCENT * 100:
            return False
            
        # 2. Portfolio risk
        portfolio_risk = self.trading_system.position_manager.calculate_portfolio_risk()
        new_risk = portfolio_risk + (signal['Risk_Amount%'] / 100 * self.trading_system.account_value)
        if new_risk > self.trading_system.position_manager.max_portfolio_risk * self.trading_system.account_value:
            return False
            
        # 3. Sector exposure
        sector = self.trading_system.position_manager.position_risk.get(signal['Ticker'], {}).get('sector')
        if sector:
            sector_value = self.trading_system.position_manager.sector_exposure.get(sector, 0)
            if sector_value + signal['Position_Value'] > 0.25 * self.trading_system.account_value:
                return False
                
        return True

# ======================== UI ENHANCEMENTS ======================== #
class ScannerUI(QMainWindow):
    # ... existing code ...
    
    def create_risk_dashboard(self):
        """Add risk management dashboard"""
        risk_tab = QWidget()
        risk_layout = QVBoxLayout(risk_tab)
        
        # Risk metrics
        metrics_layout = QGridLayout()
        
        self.portfolio_risk_label = QLabel("Portfolio Risk: 0.0%")
        self.sector_exposure_label = QLabel("Sector Exposure: Not loaded")
        self.vix_label = QLabel("VIX: 0.0")
        self.volatility_factor_label = QLabel("Volatility Factor: 1.0x")
        
        metrics_layout.addWidget(QLabel("Portfolio Risk:"), 0, 0)
        metrics_layout.addWidget(self.portfolio_risk_label, 0, 1)
        metrics_layout.addWidget(QLabel("Sector Exposure:"), 1, 0)
        metrics_layout.addWidget(self.sector_exposure_label, 1, 1)
        metrics_layout.addWidget(QLabel("VIX:"), 2, 0)
        metrics_layout.addWidget(self.vix_label, 2, 1)
        metrics_layout.addWidget(QLabel("Volatility Factor:"), 3, 0)
        metrics_layout.addWidget(self.volatility_factor_label, 3, 1)
        
        # Position risk table
        risk_layout.addLayout(metrics_layout)
        risk_layout.addWidget(QLabel("Position Risk Details:"))
        
        self.risk_table = QTableView()
        self.risk_table.setSortingEnabled(True)
        self.risk_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        risk_layout.addWidget(self.risk_table)
        
        self.tab_widget.addTab(risk_tab, "Risk Management")
    
    def update_risk_dashboard(self):
        """Update risk management dashboard"""
        if not self.scanner_thread or not self.scanner_thread.trading_system:
            return
            
        pm = self.scanner_thread.trading_system.position_manager
        
        # Portfolio risk
        portfolio_risk = pm.calculate_portfolio_risk()
        portfolio_risk_pct = (portfolio_risk / self.scanner_thread.trading_system.account_value) * 100
        self.portfolio_risk_label.setText(f"Portfolio Risk: {portfolio_risk_pct:.1f}%")
        
        # Sector exposure
        sector_text = "\n".join([f"{s}: {v/self.scanner_thread.trading_system.account_value:.1%}" 
                               for s, v in pm.sector_exposure.items()])
        self.sector_exposure_label.setText(sector_text)
        
        # Volatility metrics
        self.vix_label.setText(f"VIX: {pm.last_vix:.1f}")
        self.volatility_factor_label.setText(f"Volatility Factor: {pm.volatility_factor:.2f}x")
        
        # Position risk details
        risk_data = []
        for symbol, risk in pm.position_risk.items():
            risk_data.append({
                'Symbol': symbol,
                'Sector': risk.get('sector', 'Unknown'),
                'Position Value': f"${risk['position_value']:,.2f}",
                'Risk Amount': f"${risk['risk_amount']:,.2f}",
                'Risk %': f"{risk['risk_amount']/self.scanner_thread.trading_system.account_value:.2%}"
            })
            
        if risk_data:
            df = pd.DataFrame(risk_data)
            model = PandasModel(df)
            self.risk_table.setModel(model)