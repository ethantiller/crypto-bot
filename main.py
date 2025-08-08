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
            'rsi_entry_threshold': 65,  # RSI entry filter (buy when RSI < 65)
            'rsi_exit_threshold': 70,   # RSI exit filter (sell when RSI > 70)
            'position_size_pct': 0.075,  # 7.5% of portfolio per trade (within 5-10% range)
            'stop_loss_pct': 0.04,      # 4% stop loss (within 3-5% range)
            'take_profit_pct': 0.10,    # 10% take profit (within 8-12% range)
            'min_trade_amount': 10,     # Minimum trade amount in USD
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
        buy_signal = ma_cross_up and latest['rsi'] < self.config['rsi_entry_threshold']
        
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
            'ma_cross_down': ma_cross_down
        }
        
    def calculate_position_size(self, symbol, price):
        """Calculate position size based on portfolio percentage"""
        try:
            balance = self.exchange.fetch_balance()
            # Kraken uses 'ZUSD' for USD balances
            usd_key = 'ZUSD'
            available_balance = balance['total'].get(usd_key, 0)
                
            position_value = available_balance * self.config['position_size_pct']
            quantity = position_value / price
            
            # Check minimum trade amount
            if position_value < self.config['min_trade_amount']:
                self.logger.warning(f"Position value ${position_value:.2f} below minimum ${self.config['min_trade_amount']}")
                return 0
                
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
            
            # Paper trading mode - simulate orders without real trades
            if self.config['paper_trading']:
                self.logger.info(f"[PAPER TRADING] {side.upper()} {amount:.6f} {symbol} at ${price:.2f}")
                
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
                    return 'sell'
                    
                # Check take profit
                if pnl_pct >= self.config['take_profit_pct']:
                    self.logger.info(f"Take profit triggered for {symbol}: {pnl_pct:.3f}%")
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
                           f"MA Short: ${signals['ma_short']:.2f}, MA Long: ${signals['ma_long']:.2f}, "
                           f"Cross Up: {signals.get('ma_cross_up', False)}, Cross Down: {signals.get('ma_cross_down', False)}")
            
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
                    amount = self.positions[symbol]['amount']
                    order = self.place_order(symbol, 'sell', amount, signals['price'])
                    if order:
                        del self.positions[symbol]
                        
            else:
                # Check buy signal
                if signals['buy']:
                    self.logger.info(f"Buy signal for {symbol}")
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
        else:
            self.logger.info("[LIVE] Starting Crypto Trading Bot in LIVE TRADING MODE (real money)")
            
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
        except Exception as e:
            self.logger.error(f"Bot error: {e}")
            raise
        finally:
            self.save_state()
            self.logger.info("Bot shutdown complete")

if __name__ == "__main__":
    bot = CryptoTradingBot()
    bot.run()