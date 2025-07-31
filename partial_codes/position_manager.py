class SmartProfitTarget:
    def __init__(self, entry_price, initial_target, atr, adx):
        self.entry = entry_price
        self.base_target = initial_target
        self.atr = atr
        self.adx = adx
        self.current_target = initial_target
        self.strength_factor = 1.0
        self.breached_levels = 0
        self.last_high = entry_price

    def update(self, current_bar):
        current_high = current_bar["high"]
        current_close = current_bar["close"]

        # Calculate trend strength
        adx_strength = min(2.0, self.adx / 30)  # 1.0 = ADX 30, 2.0 = ADX 60
        volume_ratio = current_bar.get("volume", 1e6) / current_bar.get("avg_volume", 1e6)
        volatility_factor = max(0.8, min(1.5, self.atr / (current_close * 0.01)))

        # Dynamic strength adjustment
        self.strength_factor = 1.0 + (adx_strength * min(2.0, volume_ratio) * volatility_factor)

        # Check if we've breached a target level
        if current_high > self.current_target:
            self.breached_levels += 1
            self.last_high = current_high

        # Calculate new target
        if self.breached_levels > 0:
            # Extend target in strong trends
            extension_factor = 1 + (0.25 * self.breached_levels)
            new_base = self.entry + (extension_factor * (self.base_target - self.entry))
            self.current_target = new_base * self.strength_factor
        else:
            # Maintain base target
            self.current_target = self.base_target * self.strength_factor

        return self.current_target

    def should_take_profit(self, current_bar):
        current_close = current_bar["close"]
        rsi = current_bar.get("rsi", 50)

        # Basic profit taking condition
        if current_close >= self.current_target:
            return True

        # Hold conditions for strong trends
        if self.strength_factor > 1.5:
            # Only take profit if RSI > 70 and closing near highs
            if rsi < 70 or current_close < (current_bar["high"] * 0.99):
                return False

        return current_close >= self.current_target


class SmartStopLoss:
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

        # New hard stop loss parameters
        self.hard_stop = entry_price - (base_multiplier * 1.8 * atr)  # Wider buffer
        self.hard_stop_triggered = False
        self.trend_strength = 1.0  # Measures confidence in trend continuation

        # Profit target system
        initial_target = entry_price + 2 * (entry_price - self.current_stop)
        self.profit_target = SmartProfitTarget(
            entry_price=entry_price,
            initial_target=initial_target,
            atr=atr,
            adx=adx,
        )

    def update(self, current_bar):
        current_high = current_bar["high"]
        current_low = current_bar["low"]
        current_close = current_bar["close"]
        current_adx = current_bar.get("adx", self.base_adx)

        # Update trend strength (combines ADX and volume)
        adx_strength = min(1.0, current_adx / 50)
        volume_ratio = current_bar.get("volume", 1e6) / current_bar.get("avg_volume", 1e6)
        self.trend_strength = max(0.5, min(2.0, adx_strength * min(1.5, volume_ratio)))

        # Update highest high
        if current_high > self.highest_high:
            self.highest_high = current_high
            self.consecutive_confirmations = 0

        # Calculate growth potential
        self.growth_potential = max(0.5, min(2.0, adx_strength * min(1.5, volume_ratio)))

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
            adx_factor = 1.0 + (min(current_adx, 60) / 100)

            # Combine with growth potential
            dynamic_multiplier = self.base_multiplier * adx_factor * self.growth_potential

            # Set bounds for multiplier
            dynamic_multiplier = max(0.5, min(3.0, dynamic_multiplier))

            # Calculate new stop
            new_stop = self.highest_high - (dynamic_multiplier * self.initial_atr)

            # Only move stop up, never down
            if new_stop > self.current_stop:
                self.current_stop = new_stop

        # Check if we need to trigger hard stop
        if current_low <= self.hard_stop * 0.995:  # 0.5% buffer to prevent false triggers
            self.hard_stop_triggered = True

        # Update profit target
        self.profit_target.update(current_bar)

        return self.current_stop

    def sync_with_market(self, ticker):
        """Sync stop with latest market data after restart"""
        latest = get_latest_bar(ticker)
        if not latest:
            return

        # Update highest high if market moved
        if latest["high"] > self.highest_high:
            self.highest_high = latest["high"]

        # Recalculate stop based on current market
        self.update(latest)

    def should_hold(self, current_bar):
        current_low = current_bar["low"]
        current_close = current_bar["close"]
        rsi = current_bar.get("rsi", 50)
        volatility_ratio = current_bar.get("volatility_ratio", 1.0)

        # 0. Never hold if hard stop triggered
        if self.hard_stop_triggered:
            return False

        # 1. Strong momentum override
        price_change = (current_close / self.previous_close - 1) * 100
        if price_change > 3:
            return True

        # 2. Volatility contraction protection
        if volatility_ratio < 0.7:
            return True

        # 3. ADX strengthening override
        if self.growth_potential > 1.5:
            return True

        # 4. Oversold bounce prevention
        if rsi < 40 and (current_close > current_low * 1.02):
            return True

        # 5. Confirmation sequence requirement
        if self.consecutive_confirmations < 2:
            self.consecutive_confirmations += 1
            return True

        return False

    def should_exit(self, current_bar):
        # Always exit if hard stop triggered
        if self.hard_stop_triggered:
            return True

        current_low = current_bar["low"]
        current_close = current_bar["close"]
        rsi = current_bar.get("rsi", 50)
        volatility_ratio = current_bar.get("volatility_ratio", 1.0)

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
            if "volume" in current_bar and "avg_volume" in current_bar:
                volume_ratio = current_bar["volume"] / current_bar["avg_volume"]
                if volume_ratio > 1.2:
                    return True

        return False

    def get_status(self):
        return {
            "current_stop": self.current_stop,
            "growth_potential": self.growth_potential,
            "activated": self.activated,
            "consecutive_confirmations": self.consecutive_confirmations,
            "hard_stop": self.hard_stop,
            "hard_stop_triggered": self.hard_stop_triggered,
            "profit_target": self.profit_target.current_target,
        }

    def get_bracket_orders(self, entry_price, qty):
        """Generate bracket order details with properly rounded prices"""
        stop_price = self.current_stop

        # Normalize prices according to exchange rules
        def normalize_price(price):
            """Round price to proper increment based on price level"""
            if price < 1.00:
                return round(price, 4)  # $0.0001 increments for stocks < $1
            elif price < 10.00:
                return round(price, 3)  # $0.001 increments for stocks $1-$10
            else:
                return round(price, 2)  # $0.01 increments for stocks > $10

        normalized_entry = normalize_price(entry_price)
        normalized_stop = normalize_price(stop_price)
        normalized_hard_stop = normalize_price(self.hard_stop)
        normalized_profit_target = normalize_price(self.profit_target.current_target)

        # Ensure take profit is above current price
        if normalized_profit_target <= normalized_entry:
            normalized_profit_target = normalize_price(normalized_entry * 1.01)  # Minimum 1% profit

        # Ensure stop is below current price
        if normalized_stop >= normalized_entry:
            normalized_stop = normalize_price(normalized_entry * 0.99)  # Minimum 1% stop

        if normalized_hard_stop >= normalized_entry:
            normalized_hard_stop = normalize_price(normalized_entry * 0.98)  # Minimum 2% stop

        return {
            "stop_loss": {
                "stop_price": normalized_stop,
                "limit_price": normalize_price(normalized_stop * 0.98),  # Add limit price for stop-limit
            },
            "take_profit": {"limit_price": normalized_profit_target},
            "hard_stop": {"stop_price": normalized_hard_stop},
        }

    def get_serializable_state(self):
        """Return a dictionary of the current state that can be serialized"""
        return {
            "entry": self.entry,
            "initial_atr": self.initial_atr,
            "base_adx": self.base_adx,
            "activation_percent": self.activation_percent,
            "base_multiplier": self.base_multiplier,
            "activated": self.activated,
            "highest_high": self.highest_high,
            "current_stop": self.current_stop,
            "growth_potential": self.growth_potential,
            "consecutive_confirmations": self.consecutive_confirmations,
            "last_direction": self.last_direction,
            "previous_close": self.previous_close,
            "hard_stop": self.hard_stop,
            "hard_stop_triggered": self.hard_stop_triggered,
            "trend_strength": self.trend_strength,
            "profit_target_state": {
                "entry_price": self.profit_target.entry,
                "initial_target": self.profit_target.base_target,
                "atr": self.profit_target.atr,
                "adx": self.profit_target.adx,
                "current_target": self.profit_target.current_target,
                "strength_factor": self.profit_target.strength_factor,
                "breached_levels": self.profit_target.breached_levels,
                "last_high": self.profit_target.last_high,
            },
        }

    @classmethod
    def from_serialized_state(cls, state):
        """Recreate a SmartStopLoss instance from serialized state"""
        instance = cls(
            entry_price=state["entry"],
            atr=state["initial_atr"],
            adx=state["base_adx"],
            activation_percent=state["activation_percent"],
            base_multiplier=state["base_multiplier"],
        )

        # Restore all state variables
        instance.activated = state["activated"]
        instance.highest_high = state["highest_high"]
        instance.current_stop = state["current_stop"]
        instance.growth_potential = state["growth_potential"]
        instance.consecutive_confirmations = state["consecutive_confirmations"]
        instance.last_direction = state["last_direction"]
        instance.previous_close = state["previous_close"]
        instance.hard_stop = state["hard_stop"]
        instance.hard_stop_triggered = state["hard_stop_triggered"]
        instance.trend_strength = state["trend_strength"]

        # Restore profit target
        profit_state = state["profit_target_state"]
        instance.profit_target = SmartProfitTarget(
            entry_price=profit_state["entry_price"],
            initial_target=profit_state["initial_target"],
            atr=profit_state["atr"],
            adx=profit_state["adx"],
        )
        instance.profit_target.current_target = profit_state["current_target"]
        instance.profit_target.strength_factor = profit_state["strength_factor"]
        instance.profit_target.breached_levels = profit_state["breached_levels"]
        instance.profit_target.last_high = profit_state["last_high"]

        return instance

# Utility function used in SmartStopLoss
def get_latest_bar(ticker, retries=3):
    for i in range(retries):
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev"
            params = {"adjusted": "true", "apiKey": "YOUR_POLYGON_API_KEY"}
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                result = response.json().get("results", [])
                if result:
                    return {
                        "open": result[0]["o"],
                        "high": result[0]["h"],
                        "low": result[0]["l"],
                        "close": result[0]["c"],
                        "volume": result[0].get("v", 1e6),
                    }
        except Exception:
            time.sleep(2**i)  # Exponential backoff
    return None