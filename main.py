#!/usr/bin/env python3
"""
Crypto Trading Bot with Moving Average Crossover + RSI Strategy
Designed for Kraken exchange and AWS Lightsail deployment
"""

import ccxt
import pandas as pd
import numpy as np
import ta
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
import time
import logging
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json
import requests

# Load environment variables
load_dotenv()

class CryptoTradingBot:
    def __init__(self):
        self.setup_logging()
        self.setup_exchange()
        self.load_config()
        self.positions = {}
        self.trade_history = []
        self.daily_pnl = 0.0
        self.session_start_balance = 0.0
        self.peak_balance = 0.0
        self.emergency_stop = False
        self.discord_webhook_url = "https://discordapp.com/api/webhooks/1403453073485070527/P50wqbjUl9OHODBe1IoRNzT7jQ_ObpRhQQE-K50AfRB0m-utkNApo2UKA_HnMnF3jy0W"
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def send_discord_notification(self, message, color=None):
        """Send notification to Discord webhook"""
        try:
            if not self.discord_webhook_url:
                return
                
            # Create embed for better formatting
            embed = {
                "title": "🤖 Crypto Trading Bot",
                "description": message,
                "timestamp": datetime.now().isoformat(),
                "color": color or 0x00ff00  # Default green
            }
            
            payload = {
                "embeds": [embed]
            }
            
            response = requests.post(self.discord_webhook_url, json=payload, timeout=10)
            if response.status_code == 204:
                self.logger.info("Discord notification sent successfully")
            else:
                self.logger.warning(f"Discord notification failed: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error sending Discord notification: {e}")
        
    def setup_exchange(self):
        """Initialize Kraken exchange connection"""
        try:
            api_key = os.getenv('KRAKEN_API_KEY')
            secret = os.getenv('KRAKEN_SECRET')
            if not api_key or not secret:
                raise ValueError("KRAKEN_API_KEY and KRAKEN_SECRET must be set in environment variables")
            self.exchange = ccxt.kraken({
                'apiKey': api_key,
                'secret': secret,
                'sandbox': os.getenv('KRAKEN_SANDBOX', 'False').lower() == 'true',
                'enableRateLimit': True,
                'rateLimit': 1000,  # Kraken allows 1 request per second
            })
            
            # Test connection
            balance = self.exchange.fetch_balance()
            self.logger.info("Successfully connected to Kraken")
            self.logger.info(f"Account balance: {balance['total']}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Kraken: {e}")
            raise
            
    def load_config(self):
        """Load trading configuration"""
        self.config = {
            'symbols': ['BTC/USD', 'ETH/USD', 'ADA/USD'],  # Trading pairs
            'timeframe': '1h',  # 1 hour candles
            'short_ma_period': 10,  # Short moving average
            'long_ma_period': 30,   # Long moving average
            'rsi_period': 14,       # RSI period
            'rsi_entry_threshold': 65,  # RSI entry filter (buy when RSI < 65) - more selective
            'rsi_exit_threshold': 75,   # RSI exit filter (sell when RSI > 75)
            'position_size_pct': 0.3333,  # 33.33% (one-third) of portfolio per trade
            'stop_loss_pct': 0.04,      # 4% stop loss (within 3-5% range)
            'take_profit_pct': 0.03,    # 3% take profit (faster profit capture)
            'trailing_stop_pct': 0.015, # 1.5% trailing stop to protect profits
            'volume_ma_period': 20,     # Volume moving average for confirmation
            'volume_threshold': 0.8,    # Volume must be 0.8x above average (more realistic)
            'momentum_periods': 2,      # Price must be above both MAs for X periods (reduced)
            'max_hold_hours': 24,       # Maximum hold time in hours
            'max_drawdown_pct': 0.10,   # 10% max portfolio drawdown (emergency stop)
            'daily_loss_limit_pct': 0.05, # 5% daily loss limit
            'correlation_threshold': 0.7,  # Don't buy correlated assets (0.7 = 70% correlation)
            'min_trade_amount': 15,     # Minimum trade amount in USD (matches Bitcoin minimum)
            'check_interval': 300,      # Check every 5 minutes
            'paper_trading': os.getenv('PAPER_TRADING', 'True').lower() == 'true',  # Paper trading mode
        }
        
    def get_historical_data(self, symbol, timeframe='1h', limit=100):
        """Fetch historical OHLCV data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None
        
    def generate_signals(self, df):
        """Generate enhanced buy/sell signals with volume and momentum confirmation"""
        if len(df) < 2:
            return {'buy': False, 'sell': False, 'price': 0, 'rsi': 0, 'ma_short': 0, 'ma_long': 0}
            
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Moving average crossover signals
        ma_cross_up = (previous['ma_short'] <= previous['ma_long']) and (latest['ma_short'] > latest['ma_long'])
        
        # Trend analysis for safer entries
        ma_uptrend = latest['ma_short'] > latest['ma_long']
        ma_gap_pct = (latest['ma_short'] - latest['ma_long']) / latest['ma_long'] * 100
        meaningful_trend = ma_gap_pct >= 0.25  # At least 0.25% gap between MAs
        
        # Price extension check (don't buy if too far from MA)
        price_ext_pct = (latest['close'] - latest['ma_short']) / latest['ma_short'] * 100
        price_not_extended = price_ext_pct <= 3.0  # Price within 3% of short MA
        
        # Volume confirmation
        volume_confirmed = latest['volume_ratio'] >= self.config['volume_threshold']
        
        # Momentum confirmation - price above both MAs for required periods
        momentum_confirmed = latest['momentum_count'] >= self.config['momentum_periods']
        
        # Safe uptrend: meaningful trend + price not extended + good momentum
        safe_uptrend = (ma_uptrend and meaningful_trend and price_not_extended and 
                       momentum_confirmed and volume_confirmed)
        
        # Enhanced buy signal: (MA crossover OR safe uptrend) + RSI + volume + momentum
        buy_signal = ((ma_cross_up or safe_uptrend) and 
                     latest['rsi'] < self.config['rsi_entry_threshold'] and
                     volume_confirmed and
                     momentum_confirmed)
        
        # Sell signal: MA crossover down (short MA crosses below long MA) OR RSI > threshold OR weak trend
        ma_cross_down = (previous['ma_short'] >= previous['ma_long']) and (latest['ma_short'] < latest['ma_long'])
        weak_trend = (latest['ma_short'] - latest['ma_long']) / latest['ma_long'] < 0.001  # 0.1% gap - trend weakening
        sell_signal = ma_cross_down or latest['rsi'] > self.config['rsi_exit_threshold'] or weak_trend
        
        return {
            'buy': buy_signal,
            'sell': sell_signal,
            'price': latest['close'],
            'rsi': latest['rsi'],
            'ma_short': latest['ma_short'],
            'ma_long': latest['ma_long'],
            'ma_cross_up': ma_cross_up,
            'ma_cross_down': ma_cross_down,
            'safe_uptrend': safe_uptrend,
            'ma_uptrend': ma_uptrend,
            'price_not_extended': price_not_extended,
            'meaningful_trend': meaningful_trend,
            'volume_confirmed': volume_confirmed,
            'momentum_confirmed': momentum_confirmed,
            'volume_ratio': latest['volume_ratio'],
            'momentum_count': latest['momentum_count'],
            'ma_gap_pct': ma_gap_pct,
            'price_ext_pct': price_ext_pct
        }
        
    def calculate_position_size(self, symbol, price):
        """Calculate position size based on portfolio percentage"""
        try:
            balance = self.exchange.fetch_balance()
            # Use 'USD' key for USD balances (ccxt normalizes this)
            usd_key = 'USD'
            available_balance = balance['total'].get(usd_key, 0)
            
            # Log available balance for debugging
            self.logger.info(f"Available USD balance: ${available_balance:.2f}")
                
            position_value = available_balance * self.config['position_size_pct']
            quantity = position_value / price
            
            # Check minimum trade amount
            if position_value < self.config['min_trade_amount']:
                self.logger.warning(f"Position value ${position_value:.2f} below minimum ${self.config['min_trade_amount']}")
                return 0
            
            # Check if we have sufficient balance (with small buffer for fees)
            if position_value > (available_balance * 0.98):  # Leave 2% buffer for fees
                self.logger.warning(f"Insufficient balance: need ${position_value:.2f}, have ${available_balance:.2f}")
                return 0
                
            self.logger.info(f"Position size: ${position_value:.2f} ({quantity:.6f} {symbol.split('/')[0]})")
            return quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0
            
    def place_order(self, symbol, side, amount, price=None):
        """Place market or limit order"""
        try:
            if amount <= 0:
                self.logger.warning(f"Invalid amount for {symbol}: {amount}")
                return None
            
            # Round amount to appropriate precision for the trading pair
            # Most crypto pairs need 6-8 decimal places
            if symbol.startswith('BTC'):
                amount = round(amount, 8)  # BTC uses 8 decimals
            elif symbol.startswith('ETH'):
                amount = round(amount, 6)  # ETH uses 6 decimals
            else:
                amount = round(amount, 6)  # Default to 6 decimals for other coins
            
            # Double-check amount is still valid after rounding
            if amount <= 0:
                self.logger.warning(f"Amount too small after rounding for {symbol}: {amount}")
                return None
            
            # Paper trading mode - simulate orders without real trades
            if self.config['paper_trading']:
                self.logger.info(f"[PAPER TRADING] {side.upper()} {amount:.6f} {symbol} at ${price:.2f}")
                
                # Send Discord notification for paper trades
                self.send_discord_notification(
                    f"📝 **PAPER TRADE**\n"
                    f"**{side.upper()}** {amount:.6f} {symbol}\n"
                    f"Price: ${price:.2f}\n"
                    f"Value: ${amount * price:.2f}",
                    color=0xffaa00  # Orange for paper trades
                )
                
                # Create fake order for logging
                order = {
                    'id': f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'price': price,
                    'status': 'closed',
                    'paper_trade': True
                }
            else:
                # Real trading - place actual market order
                order = self.exchange.create_market_order(symbol, side, amount)
                self.logger.info(f"[LIVE TRADING] Order placed: {side} {amount:.6f} {symbol} - Order ID: {order['id']}")
                
                # Send Discord notification for live trades
                trade_value = amount * price
                self.send_discord_notification(
                    f"💰 **LIVE TRADE EXECUTED**\n"
                    f"**{side.upper()}** {amount:.6f} {symbol}\n"
                    f"Price: ${price:.2f}\n"
                    f"Value: ${trade_value:.2f}\n"
                    f"Order ID: {order['id']}",
                    color=0x00ff00 if side == 'buy' else 0xff0000  # Green for buy, red for sell
                )
                
            # Log trade
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': price,
                'order_id': order['id'],
                'paper_trade': self.config['paper_trading']
            }
            self.trade_history.append(trade_record)
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error placing {side} order for {symbol}: {e}")
            return None
            
    def check_stop_loss_take_profit(self, symbol, position):
        """Check stop loss, take profit, trailing stop, and time-based exit conditions"""
        try:
            current_price = self.exchange.fetch_ticker(symbol)['last']
            entry_price = position['entry_price']
            side = position['side']
            
            # Check time-based exit (24-hour max hold)
            position_time = datetime.fromisoformat(position['timestamp'])
            hours_held = (datetime.now() - position_time).total_seconds() / 3600
            
            if hours_held >= self.config['max_hold_hours']:
                pnl_pct = (current_price - entry_price) / entry_price
                self.logger.info(f"Time-based exit for {symbol}: held {hours_held:.1f}h, P&L: {pnl_pct:.2%}")
                
                # Send Discord notification for time exit
                self.send_discord_notification(
                    f"⏰ **TIME-BASED EXIT**\n"
                    f"Symbol: {symbol}\n"
                    f"Entry Price: ${entry_price:.2f}\n"
                    f"Current Price: ${current_price:.2f}\n"
                    f"Time Held: {hours_held:.1f} hours\n"
                    f"Final P&L: {pnl_pct:.2%}",
                    color=0x800080  # Purple for time exit
                )
                return 'sell'
            
            if side == 'buy':
                # Calculate P&L percentage
                pnl_pct = (current_price - entry_price) / entry_price
                
                # Update highest price seen for trailing stop
                highest_price = position.get('highest_price', entry_price)
                if current_price > highest_price:
                    highest_price = current_price
                    position['highest_price'] = highest_price
                    self.logger.info(f"{symbol} - New high: ${highest_price:.2f}, P&L: {pnl_pct:.2%}")
                
                # Calculate trailing stop price
                trailing_stop_price = highest_price * (1 - self.config['trailing_stop_pct'])
                
                # Check stop loss (hard stop)
                if pnl_pct <= -self.config['stop_loss_pct']:
                    self.logger.info(f"Stop loss triggered for {symbol}: {pnl_pct:.3f}%")
                    
                    # Send Discord notification for stop loss
                    self.send_discord_notification(
                        f"🛑 **STOP LOSS TRIGGERED**\n"
                        f"Symbol: {symbol}\n"
                        f"Entry Price: ${entry_price:.2f}\n"
                        f"Current Price: ${current_price:.2f}\n"
                        f"Loss: {pnl_pct:.2%}",
                        color=0xff0000  # Red for stop loss
                    )
                    return 'sell'
                
                # Check trailing stop (protect profits)
                if current_price <= trailing_stop_price and highest_price > entry_price:
                    trailing_pnl = (current_price - entry_price) / entry_price
                    self.logger.info(f"Trailing stop triggered for {symbol}: {trailing_pnl:.3f}%")
                    
                    # Send Discord notification for trailing stop
                    self.send_discord_notification(
                        f"📉 **TRAILING STOP TRIGGERED**\n"
                        f"Symbol: {symbol}\n"
                        f"Entry Price: ${entry_price:.2f}\n"
                        f"High Price: ${highest_price:.2f}\n"
                        f"Current Price: ${current_price:.2f}\n"
                        f"Final P&L: {trailing_pnl:.2%}",
                        color=0xffa500  # Orange for trailing stop
                    )
                    return 'sell'
                    
                # Check take profit (quick profit capture)
                if pnl_pct >= self.config['take_profit_pct']:
                    self.logger.info(f"Take profit triggered for {symbol}: {pnl_pct:.3f}%")
                    
                    # Send Discord notification for take profit
                    self.send_discord_notification(
                        f"🎯 **TAKE PROFIT TRIGGERED**\n"
                        f"Symbol: {symbol}\n"
                        f"Entry Price: ${entry_price:.2f}\n"
                        f"Current Price: ${current_price:.2f}\n"
                        f"Profit: {pnl_pct:.2%}",
                        color=0x00ff00  # Green for take profit
                    )
                    return 'sell'
                    
        except Exception as e:
            self.logger.error(f"Error checking stop loss/take profit for {symbol}: {e}")
            
        return None
        
    def check_risk_management(self):
        """Check risk management rules and emergency stops"""
        try:
            # Get current balance
            balance = self.exchange.fetch_balance()
            current_balance = balance['total'].get('USD', 0)
            
            # Initialize session tracking on first run
            if self.session_start_balance == 0:
                self.session_start_balance = current_balance
                self.peak_balance = current_balance
                self.logger.info(f"Session started with balance: ${current_balance:.2f}")
                return True
            
            # Update peak balance
            if current_balance > self.peak_balance:
                self.peak_balance = current_balance
            
            # Calculate drawdown from peak
            drawdown_pct = (self.peak_balance - current_balance) / self.peak_balance
            
            # Calculate daily P&L
            daily_pnl_pct = (current_balance - self.session_start_balance) / self.session_start_balance
            
            # Log risk metrics
            self.logger.info(f"Risk Check - Balance: ${current_balance:.2f}, Peak: ${self.peak_balance:.2f}, "
                           f"Drawdown: {drawdown_pct:.2%}, Daily P&L: {daily_pnl_pct:.2%}")
            
            # Check maximum drawdown (emergency stop)
            if drawdown_pct >= self.config['max_drawdown_pct']:
                self.emergency_stop = True
                self.logger.error(f"EMERGENCY STOP: Maximum drawdown reached ({drawdown_pct:.2%})")
                
                # Send urgent Discord notification
                self.send_discord_notification(
                    f"🚨 **EMERGENCY STOP ACTIVATED** 🚨\n"
                    f"Maximum drawdown exceeded: {drawdown_pct:.2%}\n"
                    f"Peak Balance: ${self.peak_balance:.2f}\n"
                    f"Current Balance: ${current_balance:.2f}\n"
                    f"**BOT TRADING DISABLED**",
                    color=0xff0000  # Red for emergency
                )
                return False
            
            # Check daily loss limit
            if daily_pnl_pct <= -self.config['daily_loss_limit_pct']:
                self.logger.warning(f"Daily loss limit reached ({daily_pnl_pct:.2%})")
                
                # Send Discord notification
                self.send_discord_notification(
                    f"⚠️ **DAILY LOSS LIMIT REACHED** ⚠️\n"
                    f"Daily Loss: {daily_pnl_pct:.2%}\n"
                    f"Session Start: ${self.session_start_balance:.2f}\n"
                    f"Current Balance: ${current_balance:.2f}\n"
                    f"No new trades until tomorrow",
                    color=0xff6600  # Orange for warning
                )
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in risk management check: {e}")
            return True  # Allow trading if risk check fails
    
    def check_correlation(self, new_symbol):
        """Check if new symbol is too correlated with existing positions"""
        if not self.positions:
            return True  # No existing positions, no correlation risk
        
        # Simple correlation check based on asset type
        # More sophisticated correlation would require price data analysis
        crypto_pairs = {
            'BTC/USD': 'bitcoin',
            'ETH/USD': 'ethereum', 
            'ADA/USD': 'altcoin',
            'LINK/USD': 'altcoin',
            'DOT/USD': 'altcoin'
        }
        
        new_type = crypto_pairs.get(new_symbol, 'unknown')
        
        # Count positions of same type
        same_type_count = 0
        for symbol in self.positions.keys():
            if crypto_pairs.get(symbol, 'unknown') == new_type and new_type == 'altcoin':
                same_type_count += 1
        
        # Don't allow more than 1 altcoin position at once
        if new_type == 'altcoin' and same_type_count >= 1:
            self.logger.warning(f"Correlation risk: Already holding altcoin position, skipping {new_symbol}")
            return False
        
        return True
    
    def get_portfolio_exposure(self):
        """Calculate current portfolio exposure"""
        try:
            balance = self.exchange.fetch_balance()
            total_usd = balance['total'].get('USD', 0)
            
            exposure = 0
            for symbol, position in self.positions.items():
                current_price = self.exchange.fetch_ticker(symbol)['last']
                position_value = position['amount'] * current_price
                exposure += position_value
            
            exposure_pct = (exposure / total_usd) * 100 if total_usd > 0 else 0
            
            self.logger.info(f"Portfolio exposure: ${exposure:.2f} ({exposure_pct:.1f}% of balance)")
            return exposure_pct
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio exposure: {e}")
            return 0
        
    def reset_emergency_stop(self):
        """Manually reset emergency stop (use with caution)"""
        self.emergency_stop = False
        self.logger.info("Emergency stop manually reset")
        self.send_discord_notification(
            "🟢 **EMERGENCY STOP RESET**\n"
            "Trading has been manually re-enabled\n"
            "⚠️ Monitor carefully",
            color=0x00ff00
        )
        
    def calculate_indicators(self, df):
        """Calculate moving averages, RSI, volume, and momentum indicators on the DataFrame"""
        # Moving averages
        df['ma_short'] = SMAIndicator(df['close'], window=self.config['short_ma_period']).sma_indicator()
        df['ma_long'] = SMAIndicator(df['close'], window=self.config['long_ma_period']).sma_indicator()
        
        # RSI - using ta library's RSI function
        df['rsi'] = RSIIndicator(df['close'], window=self.config['rsi_period']).rsi()
        
        # Volume moving average for confirmation
        df['volume_ma'] = SMAIndicator(df['volume'], window=self.config['volume_ma_period']).sma_indicator()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Momentum: price above both MAs
        df['above_mas'] = (df['close'] > df['ma_short']) & (df['close'] > df['ma_long'])
        
        # Count consecutive periods above MAs
        df['momentum_count'] = 0
        for i in range(1, len(df)):
            if df['above_mas'].iloc[i]:
                if df['above_mas'].iloc[i-1]:
                    df.loc[df.index[i], 'momentum_count'] = df['momentum_count'].iloc[i-1] + 1
                else:
                    df.loc[df.index[i], 'momentum_count'] = 1
            else:
                df.loc[df.index[i], 'momentum_count'] = 0
        
        return df

    def process_symbol(self, symbol):
        """Process trading logic for a single symbol"""
        try:
            self.logger.info(f"Processing {symbol}")
            
            # Get historical data
            df = self.get_historical_data(symbol, self.config['timeframe'])
            if df is None or len(df) < self.config['long_ma_period']:
                self.logger.warning(f"Insufficient data for {symbol}")
                return
                
            # Calculate indicators and signals
            df = self.calculate_indicators(df)
            signals = self.generate_signals(df)
            
            self.logger.info(f"{symbol} - Price: ${signals['price']:.2f}, RSI: {signals['rsi']:.1f}, "
                           f"MA Short: ${signals['ma_short']:.2f}, MA Long: ${signals['ma_long']:.2f}")
            self.logger.info(f"{symbol} DEBUG - MA Gap: {signals.get('ma_gap_pct', 0):.2f}%, "
                           f"Price Ext: {signals.get('price_ext_pct', 0):.2f}%, "
                           f"Volume: {signals.get('volume_ratio', 0):.2f}x, "
                           f"Momentum: {signals.get('momentum_count', 0)} periods, "
                           f"Vol OK: {signals.get('volume_confirmed', False)}, "
                           f"Mom OK: {signals.get('momentum_confirmed', False)}, "
                           f"Safe Uptrend: {signals.get('safe_uptrend', False)}, "
                           f"Buy: {signals.get('buy', False)}")
            
            # Check if we have an open position
            has_position = symbol in self.positions
            
            # Log position status for debugging
            if has_position:
                position = self.positions[symbol]
                current_price = signals['price']
                entry_price = position['entry_price']
                pnl_pct = (current_price - entry_price) / entry_price
                self.logger.info(f"{symbol} - POSITION ACTIVE: Entry ${entry_price:.2f}, Current ${current_price:.2f}, P&L: {pnl_pct:.2%}")
            else:
                self.logger.info(f"{symbol} - NO POSITION")
            
            if has_position:
                # Check stop loss / take profit
                sl_tp_action = self.check_stop_loss_take_profit(symbol, self.positions[symbol])
                if sl_tp_action == 'sell':
                    amount = self.positions[symbol]['amount']
                    order = self.place_order(symbol, 'sell', amount, signals['price'])
                    if order:
                        self.logger.info(f"POSITION CLOSED for {symbol} via stop/take profit")
                        del self.positions[symbol]
                        self.save_state()  # Save immediately after position removal
                        return
                        
                # Check sell signal
                if signals['sell']:
                    self.logger.info(f"Sell signal for {symbol}")
                    
                    # Send Discord notification for sell signal
                    amount = self.positions[symbol]['amount']
                    entry_price = self.positions[symbol]['entry_price']
                    current_price = signals['price']
                    pnl_pct = (current_price - entry_price) / entry_price
                    
                    self.send_discord_notification(
                        f"📉 **SELL SIGNAL DETECTED**\n"
                        f"Symbol: {symbol}\n"
                        f"Entry Price: ${entry_price:.2f}\n"
                        f"Current Price: ${current_price:.2f}\n"
                        f"P&L: {pnl_pct:.2%}\n"
                        f"RSI: {signals['rsi']:.1f}",
                        color=0xff6600  # Orange for sell signal
                    )
                    
                    order = self.place_order(symbol, 'sell', amount, signals['price'])
                    if order:
                        self.logger.info(f"POSITION CLOSED for {symbol} via sell signal")
                        del self.positions[symbol]
                        self.save_state()  # Save immediately after position removal
                        
            else:
                # Check buy signal
                if signals['buy']:
                    # Risk management checks before buying
                    if self.emergency_stop:
                        self.logger.warning(f"Emergency stop active - skipping buy signal for {symbol}")
                        return
                    
                    if not self.check_risk_management():
                        self.logger.warning(f"Risk management check failed - skipping buy signal for {symbol}")
                        return
                    
                    if not self.check_correlation(symbol):
                        self.logger.warning(f"Correlation check failed - skipping buy signal for {symbol}")
                        return
                    
                    # Check portfolio exposure
                    exposure = self.get_portfolio_exposure()
                    if exposure > 80:  # Don't exceed 80% portfolio exposure
                        self.logger.warning(f"Portfolio exposure too high ({exposure:.1f}%) - skipping buy signal for {symbol}")
                        return
                    
                    self.logger.info(f"Buy signal for {symbol}")
                    
                    # Send Discord notification for buy signal
                    self.send_discord_notification(
                        f"📈 **BUY SIGNAL DETECTED**\n"
                        f"Symbol: {symbol}\n"
                        f"Price: ${signals['price']:.2f}\n"
                        f"RSI: {signals['rsi']:.1f}\n"
                        f"MA Short: ${signals['ma_short']:.2f}\n"
                        f"MA Long: ${signals['ma_long']:.2f}",
                        color=0x00aa00  # Green for buy signal
                    )
                    
                    amount = self.calculate_position_size(symbol, signals['price'])
                    if amount > 0:
                        order = self.place_order(symbol, 'buy', amount, signals['price'])
                        if order:
                            # Create position record
                            position_data = {
                                'side': 'buy',
                                'amount': amount,
                                'entry_price': signals['price'],
                                'highest_price': signals['price'],  # Initialize trailing stop
                                'timestamp': datetime.now().isoformat()
                            }
                            self.positions[symbol] = position_data
                            
                            # Validate position was saved
                            self.logger.info(f"POSITION CREATED for {symbol}: {position_data}")
                            self.logger.info(f"Active positions: {list(self.positions.keys())}")
                            
                            # Force save state immediately after position creation
                            self.save_state()
                            
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}")
            
    def save_state(self):
        """Save bot state to file"""
        state = {
            'positions': self.positions,
            'trade_history': self.trade_history[-100:],  # Keep last 100 trades
            'emergency_stop': self.emergency_stop,
            'session_start_balance': self.session_start_balance,
            'peak_balance': self.peak_balance,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('bot_state.json', 'w') as f:
            json.dump(state, f, indent=2)
            
    def load_state(self):
        """Load bot state from file"""
        try:
            with open('bot_state.json', 'r') as f:
                state = json.load(f)
                self.positions = state.get('positions', {})
                self.trade_history = state.get('trade_history', [])
                self.emergency_stop = state.get('emergency_stop', False)
                self.session_start_balance = state.get('session_start_balance', 0.0)
                self.peak_balance = state.get('peak_balance', 0.0)
                
                if self.emergency_stop:
                    self.logger.warning("Emergency stop was previously activated")
                
                self.logger.info("Bot state loaded successfully")
        except FileNotFoundError:
            self.logger.info("No previous state found, starting fresh")
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
            
    def run(self):
        """Main bot loop"""
        if self.config['paper_trading']:
            self.logger.info("[PAPER] Starting Crypto Trading Bot in PAPER TRADING MODE (no real money)")
            # Send Discord startup notification
            self.send_discord_notification(
                f"🟡 **BOT STARTED - PAPER TRADING MODE**\n"
                f"Symbols: {', '.join(self.config['symbols'])}\n"
                f"Check Interval: {self.config['check_interval']}s\n"
                f"⚠️ No real money will be used",
                color=0xffaa00  # Orange for paper trading
            )
        else:
            self.logger.info("[LIVE] Starting Crypto Trading Bot in LIVE TRADING MODE (real money)")
            # Send Discord startup notification
            self.send_discord_notification(
                f"🟢 **BOT STARTED - LIVE TRADING MODE**\n"
                f"Symbols: {', '.join(self.config['symbols'])}\n"
                f"Check Interval: {self.config['check_interval']}s\n"
                f"💰 Real money trading active!",
                color=0x00ff00  # Green for live trading
            )
            
        self.load_state()
        
        # Log current positions on startup
        if self.positions:
            self.logger.info(f"Resuming with {len(self.positions)} active positions: {list(self.positions.keys())}")
            for symbol, position in self.positions.items():
                self.logger.info(f"  {symbol}: Entry ${position['entry_price']:.2f} at {position['timestamp']}")
        else:
            self.logger.info("Starting with no active positions")
        
        try:
            while True:
                self.logger.info("=== Starting trading cycle ===")
                
                # Emergency stop check
                if self.emergency_stop:
                    self.logger.error("Emergency stop is active - trading disabled")
                    time.sleep(self.config['check_interval'])
                    continue
                
                # Perform risk management check at start of each cycle
                if not self.check_risk_management():
                    self.logger.warning("Risk management check failed - skipping this cycle")
                    time.sleep(self.config['check_interval'])
                    continue
                
                for symbol in self.config['symbols']:
                    self.process_symbol(symbol)
                    time.sleep(2)  # Rate limiting between symbols
                    
                self.save_state()
                
                self.logger.info(f"Cycle complete. Sleeping for {self.config['check_interval']} seconds...")
                time.sleep(self.config['check_interval'])
                
        except KeyboardInterrupt:
            self.logger.info("Bot stopped by user")
            self.send_discord_notification(
                "🔴 **BOT STOPPED**\n"
                "Bot manually stopped by user",
                color=0xff0000  # Red for stop
            )
        except Exception as e:
            self.logger.error(f"Bot error: {e}")
            self.send_discord_notification(
                f"❌ **BOT ERROR**\n"
                f"Error: {str(e)}\n"
                f"Bot may have crashed!",
                color=0xff0000  # Red for error
            )
            raise
        finally:
            self.save_state()
            self.logger.info("Bot shutdown complete")

if __name__ == "__main__":
    bot = CryptoTradingBot()
    bot.run()