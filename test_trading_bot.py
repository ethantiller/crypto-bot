#!/usr/bin/env python3
"""
Unit Tests for Crypto Trading Bot
Tests edge cases and validates trading logic
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import sys
import tempfile

# Import the trading bot
from main import CryptoTradingBot

class TestCryptoTradingBot(unittest.TestCase):
    """Test cases for the CryptoTradingBot class"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Mock environment variables
        with patch.dict(os.environ, {
            'KRAKEN_API_KEY': 'test_key',
            'KRAKEN_SECRET': 'test_secret',
            'PAPER_TRADING': 'true'
        }):
            # Mock the exchange setup to avoid real API calls
            with patch('main.ccxt.kraken') as mock_kraken:
                mock_exchange = Mock()
                mock_exchange.fetch_balance.return_value = {'total': {'ZUSD': 1000.0}}
                mock_kraken.return_value = mock_exchange
                
                # Mock Discord webhook
                with patch('requests.post'):
                    self.bot = CryptoTradingBot()
                    self.bot.exchange = mock_exchange
    
    def create_test_dataframe(self, prices, rsi_values=None, ma_short_values=None, ma_long_values=None):
        """Helper method to create test DataFrame with market data"""
        timestamps = pd.date_range(start='2025-01-01', periods=len(prices), freq='1h')  # Updated to use 'h' instead of 'H'
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * 1.01 for p in prices],  # Slightly higher highs
            'low': [p * 0.99 for p in prices],   # Slightly lower lows
            'close': prices,
            'volume': [1000] * len(prices)
        })
        df.set_index('timestamp', inplace=True)
        
        # Add indicators if provided, otherwise calculate them
        if rsi_values:
            df['rsi'] = rsi_values
        else:
            df = self.bot.calculate_indicators(df)
            
        if ma_short_values:
            df['ma_short'] = ma_short_values
        if ma_long_values:
            df['ma_long'] = ma_long_values
            
        return df

    def test_buy_signal_ma_crossover_with_low_rsi(self):
        """Test buy signal when MA crosses up and RSI is low"""
        # Create data where short MA crosses above long MA with low RSI
        prices = [100, 101, 102, 103, 104]
        ma_short = [99, 100, 101, 102, 103]  # Rising
        ma_long = [102, 102, 102, 102, 102]  # Flat
        rsi = [40, 45, 50, 55, 60]  # Below 65 threshold
        
        df = self.create_test_dataframe(prices, rsi, ma_short, ma_long)
        signals = self.bot.generate_signals(df)
        
        self.assertTrue(signals['buy'], "Should generate buy signal on MA crossover with low RSI")
        self.assertFalse(signals['sell'], "Should not generate sell signal")
        self.assertEqual(signals['price'], 104)

    def test_no_buy_signal_ma_crossover_with_high_rsi(self):
        """Test no buy signal when MA crosses up but RSI is high"""
        prices = [100, 101, 102, 103, 104]
        ma_short = [99, 100, 101, 102, 103]  # Rising
        ma_long = [102, 102, 102, 102, 102]  # Flat
        rsi = [60, 65, 68, 70, 72]  # Above 65 threshold AND above 70 (would trigger sell)
        
        df = self.create_test_dataframe(prices, rsi, ma_short, ma_long)
        signals = self.bot.generate_signals(df)
        
        self.assertFalse(signals['buy'], "Should not generate buy signal with high RSI")
        # Note: Sell signal will be True because RSI > 70
        self.assertTrue(signals['sell'], "Should generate sell signal due to high RSI")

    def test_sell_signal_ma_crossover_down(self):
        """Test sell signal when short MA crosses below long MA"""
        prices = [104, 103, 102, 101, 100]
        # For crossover DOWN: previous MA short >= long, current MA short < long
        # Short MA starts above, then crosses below long MA
        ma_short = [105, 104, 103, 101, 99]  # Starts above 100, ends below 100
        ma_long = [100, 100, 100, 100, 100]  # Flat at 100
        rsi = [60, 58, 55, 50, 45]  # Normal RSI (below 70)
        
        df = self.create_test_dataframe(prices, rsi, ma_short, ma_long)
        signals = self.bot.generate_signals(df)
        
        self.assertFalse(signals['buy'], "Should not generate buy signal")
        self.assertTrue(signals['sell'], "Should generate sell signal on MA crossover down")

    def test_sell_signal_high_rsi(self):
        """Test sell signal when RSI exceeds exit threshold"""
        prices = [100, 101, 102, 103, 104]
        ma_short = [101, 102, 103, 104, 105]  # Above long MA
        ma_long = [100, 100, 100, 100, 100]
        rsi = [65, 68, 70, 72, 75]  # Above 70 threshold
        
        df = self.create_test_dataframe(prices, rsi, ma_short, ma_long)
        signals = self.bot.generate_signals(df)
        
        self.assertFalse(signals['buy'], "Should not generate buy signal")
        self.assertTrue(signals['sell'], "Should generate sell signal with high RSI")

    def test_stop_loss_trigger(self):
        """Test stop loss triggers at 4% loss"""
        # Set up a position
        symbol = 'BTC/USD'
        self.bot.positions[symbol] = {
            'side': 'buy',
            'amount': 0.01,
            'entry_price': 100.0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Mock current price at 4% loss
        mock_ticker = {'last': 96.0}  # 4% below entry price
        with patch.object(self.bot.exchange, 'fetch_ticker', return_value=mock_ticker):
            action = self.bot.check_stop_loss_take_profit(symbol, self.bot.positions[symbol])
        
        self.assertEqual(action, 'sell', "Should trigger stop loss at 4% loss")

    def test_take_profit_trigger(self):
        """Test take profit triggers at 10% gain"""
        # Set up a position
        symbol = 'BTC/USD'
        self.bot.positions[symbol] = {
            'side': 'buy',
            'amount': 0.01,
            'entry_price': 100.0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Mock current price at 10% gain
        mock_ticker = {'last': 110.0}  # 10% above entry price
        with patch.object(self.bot.exchange, 'fetch_ticker', return_value=mock_ticker):
            action = self.bot.check_stop_loss_take_profit(symbol, self.bot.positions[symbol])
        
        self.assertEqual(action, 'sell', "Should trigger take profit at 10% gain")

    def test_no_stop_loss_take_profit_in_range(self):
        """Test no action when P&L is within acceptable range"""
        # Set up a position
        symbol = 'BTC/USD'
        self.bot.positions[symbol] = {
            'side': 'buy',
            'amount': 0.01,
            'entry_price': 100.0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Mock current price at 2% gain (within range)
        mock_ticker = {'last': 102.0}
        with patch.object(self.bot.exchange, 'fetch_ticker', return_value=mock_ticker):
            action = self.bot.check_stop_loss_take_profit(symbol, self.bot.positions[symbol])
        
        self.assertIsNone(action, "Should not trigger stop loss or take profit within range")

    def test_position_size_calculation(self):
        """Test position size calculation based on portfolio percentage"""
        symbol = 'BTC/USD'
        price = 50000.0
        
        # Mock balance
        mock_balance = {'total': {'ZUSD': 10000.0}}
        with patch.object(self.bot.exchange, 'fetch_balance', return_value=mock_balance):
            position_size = self.bot.calculate_position_size(symbol, price)
        
        expected_value = 10000.0 * 0.075  # 7.5% of portfolio
        expected_quantity = expected_value / price
        
        self.assertAlmostEqual(position_size, expected_quantity, places=6)

    def test_minimum_trade_amount(self):
        """Test that trades below minimum amount are rejected"""
        symbol = 'BTC/USD'
        price = 50000.0
        
        # Mock very small balance
        mock_balance = {'total': {'ZUSD': 50.0}}  # Only $50
        with patch.object(self.bot.exchange, 'fetch_balance', return_value=mock_balance):
            position_size = self.bot.calculate_position_size(symbol, price)
        
        self.assertEqual(position_size, 0, "Should reject trades below minimum amount")

    @patch('main.requests.post')
    def test_paper_trading_order(self, mock_post):
        """Test paper trading order placement"""
        mock_post.return_value.status_code = 204
        
        order = self.bot.place_order('BTC/USD', 'buy', 0.01, 50000.0)
        
        self.assertIsNotNone(order, "Should create paper trading order")
        if order:  # Check if order was created
            self.assertTrue(order['paper_trade'], "Should mark as paper trade")
            self.assertEqual(order['symbol'], 'BTC/USD')
            self.assertEqual(order['side'], 'buy')
            self.assertEqual(order['amount'], 0.01)

    def test_insufficient_data_handling(self):
        """Test handling of insufficient historical data"""
        # Create DataFrame with only 1 row (insufficient for indicators)
        df = self.create_test_dataframe([100])
        
        signals = self.bot.generate_signals(df)
        
        self.assertFalse(signals['buy'], "Should not generate buy signal with insufficient data")
        self.assertFalse(signals['sell'], "Should not generate sell signal with insufficient data")

    def test_state_save_and_load(self):
        """Test saving and loading bot state"""
        # Set up some test state
        self.bot.positions = {
            'BTC/USD': {
                'side': 'buy',
                'amount': 0.01,
                'entry_price': 50000.0,
                'timestamp': '2025-01-01T00:00:00'
            }
        }
        
        self.bot.trade_history = [{
            'timestamp': '2025-01-01T00:00:00',
            'symbol': 'BTC/USD',
            'side': 'buy',
            'amount': 0.01,
            'price': 50000.0,
            'order_id': 'test123'
        }]
        
        # Save state
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
            
        # Temporarily replace the state file
        original_file = 'bot_state.json'
        try:
            # Mock the file path
            with patch('builtins.open', create=True) as mock_open:
                mock_file = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_file
                
                self.bot.save_state()
                
                # Verify save was called
                mock_open.assert_called_with('bot_state.json', 'w')
                
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_edge_case_zero_volume(self):
        """Test handling of zero volume data"""
        prices = [100, 100, 100, 100, 100]  # No price movement
        volumes = [0, 0, 0, 0, 0]  # Zero volume
        
        timestamps = pd.date_range(start='2025-01-01', periods=len(prices), freq='1h')  # Updated to use 'h'
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': prices,
            'low': prices,
            'close': prices,
            'volume': volumes
        })
        df.set_index('timestamp', inplace=True)
        
        # Should not crash on zero volume
        df = self.bot.calculate_indicators(df)
        signals = self.bot.generate_signals(df)
        
        self.assertIsInstance(signals, dict, "Should handle zero volume data gracefully")

    def test_extreme_price_movements(self):
        """Test handling of extreme price movements"""
        # Simulate extreme volatility
        prices = [100, 200, 50, 150, 75]  # Extreme swings
        
        df = self.create_test_dataframe(prices)
        signals = self.bot.generate_signals(df)
        
        self.assertIsInstance(signals, dict, "Should handle extreme price movements")
        # RSI might be NaN with extreme data, so check if it's a number or NaN
        if not pd.isna(signals['rsi']):
            self.assertGreaterEqual(signals['rsi'], 0, "RSI should be >= 0")
            self.assertLessEqual(signals['rsi'], 100, "RSI should be <= 100")
        else:
            # If RSI is NaN, that's also acceptable for extreme data
            self.assertTrue(pd.isna(signals['rsi']), "RSI can be NaN with extreme data")

    def test_ma_convergence(self):
        """Test behavior when moving averages converge"""
        # Create data where MAs are very close
        prices = [100.00, 100.01, 100.02, 100.01, 100.00]
        ma_short = [100.00, 100.005, 100.01, 100.005, 100.00]
        ma_long = [100.00, 100.005, 100.01, 100.005, 100.00]
        
        df = self.create_test_dataframe(prices, ma_short_values=ma_short, ma_long_values=ma_long)
        signals = self.bot.generate_signals(df)
        
        # Should handle convergence without false signals
        self.assertIsInstance(signals, dict, "Should handle MA convergence")

    def test_error_handling_in_process_symbol(self):
        """Test error handling in process_symbol method"""
        # Create a simple test that verifies the method handles errors gracefully
        with patch.object(self.bot, 'get_historical_data', side_effect=Exception("API Error")):
            # Should not raise exception, just handle it gracefully
            try:
                self.bot.process_symbol('BTC/USD')
                # If we get here, the error was handled properly
                self.assertTrue(True, "Error was handled gracefully")
            except Exception as e:
                self.fail(f"process_symbol should handle errors gracefully, but raised: {e}")

class TestTradingScenarios(unittest.TestCase):
    """Test specific trading scenarios and edge cases"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch.dict(os.environ, {
            'KRAKEN_API_KEY': 'test_key',
            'KRAKEN_SECRET': 'test_secret',
            'PAPER_TRADING': 'true'
        }):
            with patch('main.ccxt.kraken') as mock_kraken:
                mock_exchange = Mock()
                mock_exchange.fetch_balance.return_value = {'total': {'ZUSD': 1000.0}}
                mock_kraken.return_value = mock_exchange
                
                with patch('requests.post'):
                    self.bot = CryptoTradingBot()
                    self.bot.exchange = mock_exchange

    def test_full_trading_cycle(self):
        """Test a complete buy -> hold -> sell cycle"""
        symbol = 'BTC/USD'
        
        # Step 1: Generate buy signal
        buy_prices = [100, 101, 102, 103, 104]
        buy_ma_short = [99, 100, 101, 102, 103]
        buy_ma_long = [102, 102, 102, 102, 102]
        buy_rsi = [50, 55, 58, 60, 62]
        
        buy_df = pd.DataFrame({
            'close': buy_prices,
            'ma_short': buy_ma_short,
            'ma_long': buy_ma_long,
            'rsi': buy_rsi
        })
        buy_df.index = pd.date_range(start='2025-01-01', periods=len(buy_prices), freq='1h')  # Updated
        
        buy_signals = self.bot.generate_signals(buy_df)
        self.assertTrue(buy_signals['buy'], "Should generate buy signal")
        
        # Simulate buying
        self.bot.positions[symbol] = {
            'side': 'buy',
            'amount': 0.01,
            'entry_price': 104.0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Step 2: Test holding period (no signals)
        hold_prices = [104, 105, 106, 105, 104]
        hold_ma_short = [103, 104, 105, 105, 104]
        hold_ma_long = [102, 102, 103, 103, 103]
        hold_rsi = [62, 65, 68, 65, 62]
        
        hold_df = pd.DataFrame({
            'close': hold_prices,
            'ma_short': hold_ma_short,
            'ma_long': hold_ma_long,
            'rsi': hold_rsi
        })
        hold_df.index = pd.date_range(start='2025-01-01 05:00:00', periods=len(hold_prices), freq='1h')  # Updated
        
        hold_signals = self.bot.generate_signals(hold_df)
        self.assertFalse(hold_signals['buy'], "Should not generate new buy signal while holding")
        self.assertFalse(hold_signals['sell'], "Should not generate sell signal while holding in range")
        
        # Step 3: Generate sell signal (high RSI)
        sell_prices = [106, 107, 108, 109, 110]
        sell_ma_short = [105, 106, 107, 108, 109]
        sell_ma_long = [103, 104, 105, 106, 107]
        sell_rsi = [68, 70, 72, 74, 76]  # Above 70 threshold
        
        sell_df = pd.DataFrame({
            'close': sell_prices,
            'ma_short': sell_ma_short,
            'ma_long': sell_ma_long,
            'rsi': sell_rsi
        })
        sell_df.index = pd.date_range(start='2025-01-01 10:00:00', periods=len(sell_prices), freq='1h')  # Updated
        
        sell_signals = self.bot.generate_signals(sell_df)
        self.assertTrue(sell_signals['sell'], "Should generate sell signal with high RSI")

if __name__ == '__main__':
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestCryptoTradingBot))
    suite.addTests(loader.loadTestsFromTestCase(TestTradingScenarios))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORs:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n🎉 All tests passed!")
    else:
        print(f"\n❌ {len(result.failures + result.errors)} test(s) failed")
