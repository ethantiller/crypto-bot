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
            'rsi_entry_threshold': 70,  # RSI entry filter (buy when RSI < 70)
            'rsi_exit_threshold': 75,   # RSI exit filter (sell when RSI > 75)
            'position_size_pct': 0.3333,  # 33.33% (one-third) of portfolio per trade
            'stop_loss_pct': 0.04,      # 4% stop loss (within 3-5% range)
            'take_profit_pct': 0.10,    # 10% take profit (within 8-12% range)
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
        """Generate buy/sell signals based on strategy"""
        if len(df) < 2:
            return {'buy': False, 'sell': False, 'price': 0, 'rsi': 0, 'ma_short': 0, 'ma_long': 0}
            
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Buy signal: MA crossover up (short MA crosses above long MA) AND RSI < threshold
        ma_cross_up = (previous['ma_short'] <= previous['ma_long']) and (latest['ma_short'] > latest['ma_long'])
        
        # Trend-following logic with safety checks
        ma_uptrend = latest['ma_short'] > latest['ma_long']
        # Safety: Don't buy if price is too extended above short MA (avoid buying tops)
        price_not_extended = latest['close'] <= (latest['ma_short'] * 1.03)  # Within 3% of short MA
        # Safety: Ensure we have a meaningful trend (short MA sufficiently above long MA)
        meaningful_trend = (latest['ma_short'] - latest['ma_long']) / latest['ma_long'] >= 0.0025  # 0.25% gap minimum
        
        # Debug trend-following conditions
        ma_gap_pct = (latest['ma_short'] - latest['ma_long']) / latest['ma_long'] * 100
        price_ext_pct = (latest['close'] - latest['ma_short']) / latest['ma_short'] * 100
        
        # Combined buy signal with safety checks
        safe_uptrend = ma_uptrend and price_not_extended and meaningful_trend
        buy_signal = (ma_cross_up or safe_uptrend) and latest['rsi'] < self.config['rsi_entry_threshold']
        
        # Sell signal: MA crossover down (short MA crosses below long MA) OR RSI > threshold
        ma_cross_down = (previous['ma_short'] >= previous['ma_long']) and (latest['ma_short'] < latest['ma_long'])
        sell_signal = ma_cross_down or latest['rsi'] > self.config['rsi_exit_threshold']
        
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
        """Check stop loss and take profit conditions"""
        try:
            current_price = self.exchange.fetch_ticker(symbol)['last']
            entry_price = position['entry_price']
            side = position['side']
            
            if side == 'buy':
                # Calculate P&L percentage
                pnl_pct = (current_price - entry_price) / entry_price
                
                # Check stop loss
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
                    
                # Check take profit
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
        
    def calculate_indicators(self, df):
        """Calculate moving averages, RSI, and signals on the DataFrame"""
        # Moving averages
        df['ma_short'] = SMAIndicator(df['close'], window=self.config['short_ma_period']).sma_indicator()
        df['ma_long'] = SMAIndicator(df['close'], window=self.config['long_ma_period']).sma_indicator()
        
        # RSI - using ta library's RSI function
        df['rsi'] = RSIIndicator(df['close'], window=self.config['rsi_period']).rsi()
        
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
                           f"Uptrend: {signals.get('ma_uptrend', False)}, "
                           f"Not Extended: {signals.get('price_not_extended', False)}, "
                           f"Meaningful: {signals.get('meaningful_trend', False)}, "
                           f"Safe Uptrend: {signals.get('safe_uptrend', False)}, "
                           f"Buy: {signals.get('buy', False)}")
            
            # Check if we have an open position
            has_position = symbol in self.positions
            
            if has_position:
                # Check stop loss / take profit
                sl_tp_action = self.check_stop_loss_take_profit(symbol, self.positions[symbol])
                if sl_tp_action == 'sell':
                    amount = self.positions[symbol]['amount']
                    order = self.place_order(symbol, 'sell', amount, signals['price'])
                    if order:
                        del self.positions[symbol]
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
                        del self.positions[symbol]
                        
            else:
                # Check buy signal
                if signals['buy']:
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
                            self.positions[symbol] = {
                                'side': 'buy',
                                'amount': amount,
                                'entry_price': signals['price'],
                                'timestamp': datetime.now().isoformat()
                            }
                            
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}")
            
    def save_state(self):
        """Save bot state to file"""
        state = {
            'positions': self.positions,
            'trade_history': self.trade_history[-100:],  # Keep last 100 trades
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
        
        try:
            while True:
                self.logger.info("=== Starting trading cycle ===")
                
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