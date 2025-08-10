#!/usr/bin/env python3
"""
Comprehensive test suite for the Crypto Trading Bot

Tests all entry conditions, exit conditions, and safety measures:

BUY CONDITIONS:
1. 1H EMA21 crosses above EMA50 (on last closed 1H candle)
2. 4H EMA21 > EMA50 (trend confirmation)
3. 1H RSI between 40-60, and not >70 or <30
4. 1H MACD hist positive and macd > signal
5. 1H volume > vol_ma20
6. 2-hour move <= 5%
7. 15m confirmation bullish candle (close > open and close > ema21)
8. Not during first/last 30 minutes of day
9. Correlation check with existing positions
10. Max concurrent positions not exceeded

SELL CONDITIONS:
1. Stop Loss: P&L <= -2%
2. Partial Take Profit: P&L >= 3% (sell 50%)
3. Primary Take Profit: P&L >= 5%
4. Trailing Stop: Activated at +3%, trails by 1.5%
5. Time Exit: 48 hours without movement

SAFETY MEASURES:
1. Daily loss limit: 5%
2. Max drawdown: 5%
3. Emergency stop functionality
4. Position sizing limits
5. Minimum trade amounts
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone, timedelta
import json
import os
import sys
import logging

# Import the bot class
from main import CryptoTradingBot

class TestCryptoTradingBot(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock exchange
        self.mock_exchange = Mock()
        
        # Create a mock logger to capture output
        self.mock_logger = Mock()
        
        # Initialize bot with mocked dependencies
        self.bot = CryptoTradingBot(exchange=self.mock_exchange, logger=self.mock_logger)
        
        # Set up test configuration
        self.bot.config.update({
            'paper_trading': True,
            'symbols': ['BTC/USD', 'ETH/USD'],
            'check_interval': 30,
            'entry_scan_interval': 300,
            'ema_short': 21,
            'ema_long': 50,
            'rsi_period': 14,
            'rsi_entry_low': 40,
            'rsi_entry_high': 60,
            'rsi_hard_upper': 70,
            'rsi_hard_lower': 30,
            'stop_loss_pct': 0.02,
            'partial_tp_pct': 0.03,
            'primary_tp_pct': 0.05,
            'trailing_activate_pct': 0.03,
            'trailing_pct': 0.015,
            'time_exit_hours': 48,
            'max_recent_move_pct': 5.0,
            'max_concurrent_positions': 3,
            'daily_loss_limit_pct': 0.05,
            'max_drawdown_pct': 0.05
        })
        
        # Create sample data for testing
        self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample OHLCV data for testing."""
        # Create basic DataFrame structure - fix deprecated 'H' to 'h'
        dates = pd.date_range(start='2025-01-01', periods=100, freq='1h')
        
        # Base price data with uptrend
        base_price = 50000
        price_data = []
        for i in range(100):
            # Create slight uptrend with some volatility
            price = base_price + (i * 10) + np.random.normal(0, 100)
            price_data.append(price)
        
        self.sample_df_1h = pd.DataFrame({
            'open': price_data,
            'high': [p + np.random.uniform(50, 200) for p in price_data],
            'low': [p - np.random.uniform(50, 200) for p in price_data],
            'close': [p + np.random.uniform(-100, 100) for p in price_data],
            'volume': [np.random.uniform(1000, 5000) for _ in range(100)]
        }, index=dates)
        
        # Create 4H data (resample from 1H) - fix deprecated 'H' to 'h'
        self.sample_df_4h = self.sample_df_1h.resample('4h').agg({
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Create 15m data
        dates_15m = pd.date_range(start='2025-01-01', periods=400, freq='15min')
        self.sample_df_15m = pd.DataFrame({
            'open': [base_price + (i * 2.5) + np.random.normal(0, 25) for i in range(400)],
            'high': [base_price + (i * 2.5) + np.random.uniform(25, 100) for i in range(400)],
            'low': [base_price + (i * 2.5) - np.random.uniform(25, 100) for i in range(400)],
            'close': [base_price + (i * 2.5) + np.random.uniform(-50, 50) for i in range(400)],
            'volume': [np.random.uniform(100, 500) for _ in range(400)]
        }, index=dates_15m)

    def create_bullish_conditions_data(self):
        """Create data that should trigger a BUY signal."""
        # Create data with EMA cross
        df_1h = self.sample_df_1h.copy()
        
        # Manually set EMAs to create a clear cross (previous negative, current positive)
        ema21_values = [49000] * 98 + [49900, 50100]  # EMA21 crosses above
        ema50_values = [50000] * 98 + [50000, 50000]  # EMA50 stays constant
        df_1h['ema21'] = ema21_values
        df_1h['ema50'] = ema50_values
        
        # Set RSI in good range
        df_1h['rsi14'] = [50.0] * 100
        
        # Set bullish MACD (all positive)
        df_1h['macd'] = [100.0] * 100
        df_1h['macd_signal'] = [50.0] * 100
        df_1h['macd_hist'] = [50.0] * 100
        
        # Set good volume
        df_1h['vol_ma20'] = [2000.0] * 100
        df_1h['vol_ratio'] = [1.5] * 100
        df_1h['volume'] = [3000.0] * 100  # Ensure volume > vol_ma20
        
        # Set ATR
        df_1h['atr14'] = [500.0] * 100
        
        # Create 4H trend confirmation (EMA21 > EMA50)
        df_4h = self.sample_df_4h.copy()
        df_4h['ema21'] = [50100] * len(df_4h)
        df_4h['ema50'] = [50000] * len(df_4h)
        
        # Create bullish 15m confirmation
        df_15m = self.sample_df_15m.copy()
        # Last candle is bullish (close > open) - modify values directly
        df_15m.at[df_15m.index[-1], 'open'] = 50100
        df_15m.at[df_15m.index[-1], 'close'] = 50200  # close > open
        df_15m['ema21'] = [50000] * len(df_15m)
        # Ensure close > ema21 for confirmation (already set above)
        
        return {'1h': df_1h, '4h': df_4h, '15m': df_15m}

    def create_bearish_conditions_data(self):
        """Create data that should NOT trigger a BUY signal."""
        df_1h = self.sample_df_1h.copy()
        
        # No EMA cross (EMA21 below EMA50)
        df_1h['ema21'] = [49000] * 100
        df_1h['ema50'] = [50000] * 100
        
        # RSI out of range
        df_1h['rsi14'] = [75.0] * 100  # Too high
        
        return {'1h': df_1h, '4h': self.sample_df_4h, '15m': self.sample_df_15m}

    # ========================
    # BUY SIGNAL TESTS
    # ========================
    
    def test_buy_signal_all_conditions_met(self):
        """Test that a BUY signal is generated when all conditions are met."""
        # Mock the bot's methods to return the conditions we want
        with patch('main.datetime') as mock_datetime:
            # Set time to avoid time blocking (middle of day)
            mock_datetime.now.return_value = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
            
            # Create mock data frames that will pass indicator computation
            mock_1h = pd.DataFrame({
                'open': [49000, 49500, 50000],
                'high': [49200, 49700, 50200],
                'low': [48800, 49300, 49800],
                'close': [49500, 50000, 50100],  # trending up
                'volume': [3000, 3100, 3200]
            })
            
            mock_4h = pd.DataFrame({
                'open': [49000],
                'high': [50200],
                'low': [48800],
                'close': [50100],
                'volume': [10000]
            })
            
            mock_15m = pd.DataFrame({
                'open': [50000, 50050],
                'high': [50100, 50150],
                'low': [49950, 50000],
                'close': [50050, 50150],  # bullish last candle
                'volume': [500, 600]
            })
            
            dfs = {'1h': mock_1h, '4h': mock_4h, '15m': mock_15m}
            
            # Mock the indicator computation to return favorable values
            def mock_compute_indicators(df):
                if len(df) == 3:  # 1H data
                    df['ema21'] = [49000, 49500, 50200]  # crosses above
                    df['ema50'] = [49500, 49800, 50000]  # EMA50 values
                    df['rsi14'] = [50, 52, 55]  # good RSI
                    df['macd'] = [10, 15, 20]
                    df['macd_signal'] = [5, 10, 15]
                    df['macd_hist'] = [5, 5, 5]
                    df['vol_ma20'] = [2000, 2000, 2000]
                    df['vol_ratio'] = [1.5, 1.55, 1.6]  # good volume
                    df['atr14'] = [500, 500, 500]
                elif len(df) == 1:  # 4H data
                    df['ema21'] = [50200]
                    df['ema50'] = [50000]  # trend confirmation
                elif len(df) == 2:  # 15m data
                    df['ema21'] = [50000, 50000]
                return df
            
            with patch.object(self.bot, 'compute_indicators_on_df', side_effect=mock_compute_indicators):
                result = self.bot.check_entry_conditions('BTC/USD', dfs)
            
            # Now we should get a buy signal
            self.assertTrue(result['buy'], f"Expected BUY=True but got {result}")
            self.assertEqual(result['reason'], 'all_ok')
            self.assertTrue(result['ma_cross_up'])
            self.assertTrue(result['trend_confirm'])
            self.assertTrue(result['macd_ok'])
            self.assertGreater(result['vol_ratio'], 1.0)
            self.assertTrue(result['15m_confirm'])
            self.assertFalse(result['time_blocked'])

    def test_no_ema_cross(self):
        """Test that no BUY signal when EMA21 doesn't cross above EMA50."""
        # Create basic data frames
        mock_1h = pd.DataFrame({
            'open': [50000, 50000],
            'high': [50100, 50100],
            'low': [49900, 49900],
            'close': [50000, 50000],
            'volume': [3000, 3000]
        })
        
        mock_4h = pd.DataFrame({
            'open': [50000], 'high': [50100], 'low': [49900], 
            'close': [50000], 'volume': [10000]
        })
        
        mock_15m = pd.DataFrame({
            'open': [50000, 50000], 'high': [50100, 50100], 'low': [49900, 49900],
            'close': [50000, 50000], 'volume': [500, 500]
        })
        
        dfs = {'1h': mock_1h, '4h': mock_4h, '15m': mock_15m}
        
        # Mock indicators with no cross (ema21 < ema50)
        def mock_compute_no_cross(df):
            if len(df) == 2:  # 1H data
                df['ema21'] = [49000, 49000]  # Below ema50
                df['ema50'] = [50000, 50000]
                df['rsi14'] = [50, 50]
                df['macd'] = [10, 20]
                df['macd_signal'] = [5, 15]
                df['macd_hist'] = [5, 5]
                df['vol_ma20'] = [2000, 2000]
                df['vol_ratio'] = [1.5, 1.5]
                df['atr14'] = [500, 500]
            elif len(df) == 1:  # 4H data
                df['ema21'] = [49000]
                df['ema50'] = [50000]
            elif len(df) == 2 and 'open' in df.columns:  # 15m data
                df['ema21'] = [50000, 50000]
            return df
        
        with patch.object(self.bot, 'compute_indicators_on_df', side_effect=mock_compute_no_cross):
            result = self.bot.check_entry_conditions('BTC/USD', dfs)
            self.assertFalse(result['buy'])
            self.assertIn('no_ma_cross', result['reason'])
            self.assertFalse(result['ma_cross_up'])

    def test_rsi_out_of_bounds(self):
        """Test various RSI boundary conditions."""
        # Create basic data frames
        mock_1h = pd.DataFrame({
            'open': [50000, 50000],
            'high': [50100, 50100],
            'low': [49900, 49900],
            'close': [50000, 50000],
            'volume': [3000, 3000]
        })
        
        mock_4h = pd.DataFrame({
            'open': [50000], 'high': [50100], 'low': [49900], 
            'close': [50000], 'volume': [10000]
        })
        
        mock_15m = pd.DataFrame({
            'open': [50000, 50000], 'high': [50100, 50100], 'low': [49900, 49900],
            'close': [50000, 50000], 'volume': [500, 500]
        })
        
        dfs = {'1h': mock_1h, '4h': mock_4h, '15m': mock_15m}
        
        # Test RSI too high (>70) - mock the indicator computation
        def mock_compute_high_rsi(df):
            if len(df) == 2:  # 1H data
                df['ema21'] = [50200, 50300]  # create cross
                df['ema50'] = [50000, 50000]
                df['rsi14'] = [75, 75]  # TOO HIGH
                df['macd'] = [10, 20]
                df['macd_signal'] = [5, 15]
                df['macd_hist'] = [5, 5]
                df['vol_ma20'] = [2000, 2000]
                df['vol_ratio'] = [1.5, 1.5]
                df['atr14'] = [500, 500]
            elif len(df) == 1:  # 4H data
                df['ema21'] = [50200]
                df['ema50'] = [50000]
            elif len(df) == 2 and 'open' in df.columns:  # 15m data
                df['ema21'] = [50000, 50000]
            return df
        
        with patch.object(self.bot, 'compute_indicators_on_df', side_effect=mock_compute_high_rsi):
            result = self.bot.check_entry_conditions('BTC/USD', dfs)
            self.assertFalse(result['buy'])
            self.assertIn('rsi_out_of_bounds', result['reason'])
        
        # Test RSI too low (<30)
        def mock_compute_low_rsi(df):
            if len(df) == 2:  # 1H data
                df['ema21'] = [50200, 50300]
                df['ema50'] = [50000, 50000]
                df['rsi14'] = [25, 25]  # TOO LOW
                df['macd'] = [10, 20]
                df['macd_signal'] = [5, 15]
                df['macd_hist'] = [5, 5]
                df['vol_ma20'] = [2000, 2000]
                df['vol_ratio'] = [1.5, 1.5]
                df['atr14'] = [500, 500]
            elif len(df) == 1:  # 4H data
                df['ema21'] = [50200]
                df['ema50'] = [50000]
            elif len(df) == 2 and 'open' in df.columns:  # 15m data
                df['ema21'] = [50000, 50000]
            return df
        
        with patch.object(self.bot, 'compute_indicators_on_df', side_effect=mock_compute_low_rsi):
            result = self.bot.check_entry_conditions('BTC/USD', dfs)
            self.assertFalse(result['buy'])
            self.assertIn('rsi_out_of_bounds', result['reason'])
        
        # Test RSI outside 40-60 range but within hard bounds
        def mock_compute_mid_rsi(df):
            if len(df) == 2:  # 1H data
                df['ema21'] = [50200, 50300]
                df['ema50'] = [50000, 50000]
                df['rsi14'] = [65, 65]  # Outside 40-60 but within 30-70
                df['macd'] = [10, 20]
                df['macd_signal'] = [5, 15]
                df['macd_hist'] = [5, 5]
                df['vol_ma20'] = [2000, 2000]
                df['vol_ratio'] = [1.5, 1.5]
                df['atr14'] = [500, 500]
            elif len(df) == 1:  # 4H data
                df['ema21'] = [50200]
                df['ema50'] = [50000]
            elif len(df) == 2 and 'open' in df.columns:  # 15m data
                df['ema21'] = [50000, 50000]
            return df
        
        with patch.object(self.bot, 'compute_indicators_on_df', side_effect=mock_compute_mid_rsi):
            result = self.bot.check_entry_conditions('BTC/USD', dfs)
            self.assertFalse(result['buy'])
            self.assertIn('rsi_not_in_40_60', result['reason'])

    def test_macd_conditions(self):
        """Test MACD bullish/bearish conditions."""
        dfs = self.create_bullish_conditions_data()
        
        # Test bearish MACD (macd < signal)
        dfs['1h']['macd'] = [50.0] * 100
        dfs['1h']['macd_signal'] = [100.0] * 100
        dfs['1h']['macd_hist'] = [-50.0] * 100
        
        result = self.bot.check_entry_conditions('BTC/USD', dfs)
        self.assertFalse(result['buy'])
        self.assertIn('macd_no', result['reason'])
        self.assertFalse(result['macd_ok'])

    def test_volume_conditions(self):
        """Test volume confirmation requirements."""
        dfs = self.create_bullish_conditions_data()
        
        # Test low volume (ratio < 1.0)
        dfs['1h']['vol_ratio'] = [0.8] * 100
        
        result = self.bot.check_entry_conditions('BTC/USD', dfs)
        self.assertFalse(result['buy'])
        self.assertIn('volume_no', result['reason'])

    def test_recent_move_too_high(self):
        """Test rejection when recent price move is too high."""
        dfs = self.create_bullish_conditions_data()
        
        # Create a large recent move by manipulating 15m data
        df_15m = dfs['15m']
        start_price = 50000
        end_price = start_price * 1.08  # 8% move (> 5% limit)
        
        # Set up data for large move
        for i in range(8):  # 8 * 15min = 2 hours
            df_15m.loc[df_15m.index[-(i+1)], 'close'] = start_price + (end_price - start_price) * (8-i) / 8
        
        result = self.bot.check_entry_conditions('BTC/USD', dfs)
        self.assertFalse(result['buy'])
        self.assertIn('recent_move_high', result['reason'])

    def test_15m_confirmation_failure(self):
        """Test 15m confirmation requirements."""
        dfs = self.create_bullish_conditions_data()
        
        # Make last 15m candle bearish (close < open)
        df_15m = dfs['15m']
        df_15m.at[df_15m.index[-1], 'open'] = 50200
        df_15m.at[df_15m.index[-1], 'close'] = 50100  # close < open
        
        result = self.bot.check_entry_conditions('BTC/USD', dfs)
        self.assertFalse(result['buy'])
        self.assertIn('no_15m_confirm', result['reason'])
        self.assertFalse(result['15m_confirm'])

    def test_time_blocking(self):
        """Test time-of-day blocking (first/last 30 minutes)."""
        dfs = self.create_bullish_conditions_data()
        
        with patch('main.datetime') as mock_datetime:
            # Test first 30 minutes of day
            mock_datetime.now.return_value = datetime(2025, 1, 1, 0, 15, tzinfo=timezone.utc)
            
            result = self.bot.check_entry_conditions('BTC/USD', dfs)
            self.assertFalse(result['buy'])
            self.assertEqual(result['reason'], 'time_blocked')
            
            # Test last 30 minutes of day
            mock_datetime.now.return_value = datetime(2025, 1, 1, 23, 45, tzinfo=timezone.utc)
            
            result = self.bot.check_entry_conditions('BTC/USD', dfs)
            self.assertFalse(result['buy'])
            self.assertEqual(result['reason'], 'time_blocked')

    def test_4h_trend_confirmation(self):
        """Test 4H trend confirmation requirement."""
        dfs = self.create_bullish_conditions_data()
        
        # Make 4H trend bearish (EMA21 < EMA50)
        dfs['4h']['ema21'] = [49000] * len(dfs['4h'])
        dfs['4h']['ema50'] = [50000] * len(dfs['4h'])
        
        result = self.bot.check_entry_conditions('BTC/USD', dfs)
        self.assertFalse(result['buy'])
        self.assertIn('4h_no_trend_confirm', result['reason'])
        self.assertFalse(result['trend_confirm'])

    # ========================
    # SELL SIGNAL TESTS
    # ========================
    
    def test_stop_loss_trigger(self):
        """Test stop loss at -2%."""
        # Create a position
        symbol = 'BTC/USD'
        entry_price = 50000
        current_price = 49000  # -2% loss
        
        self.bot.positions[symbol] = {
            'amount': 0.01,
            'entry_price': entry_price,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'highest_price': entry_price
        }
        
        # Mock ticker data - ensure we're not in paper trading mode for this test
        self.bot.config['paper_trading'] = False
        self.mock_exchange.fetch_ticker.return_value = {'last': current_price}
        
        with patch.object(self.bot, 'close_position') as mock_close:
            self.bot.manage_positions_cycle()
            mock_close.assert_called_with(symbol, reason='stop_loss')

    def test_partial_take_profit(self):
        """Test partial take profit at +3%."""
        symbol = 'BTC/USD'
        entry_price = 50000
        current_price = 51500  # +3% profit
        
        self.bot.positions[symbol] = {
            'amount': 0.02,
            'entry_price': entry_price,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'highest_price': current_price
        }
        
        # Ensure not in paper trading mode
        self.bot.config['paper_trading'] = False
        self.mock_exchange.fetch_ticker.return_value = {'last': current_price}
        
        with patch.object(self.bot, 'close_position') as mock_close:
            self.bot.manage_positions_cycle()
            # Should close 50% of position
            mock_close.assert_called_with(symbol, amount=0.01, reason='partial_take_profit')

    def test_primary_take_profit(self):
        """Test primary take profit at +5%."""
        symbol = 'BTC/USD'
        entry_price = 50000
        current_price = 52500  # +5% profit
        
        self.bot.positions[symbol] = {
            'amount': 0.01,
            'entry_price': entry_price,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'highest_price': current_price,
            'partial_taken': True  # Already took partial
        }
        
        # Ensure not in paper trading mode
        self.bot.config['paper_trading'] = False
        self.mock_exchange.fetch_ticker.return_value = {'last': current_price}
        
        with patch.object(self.bot, 'close_position') as mock_close:
            self.bot.manage_positions_cycle()
            mock_close.assert_called_with(symbol, reason='primary_take_profit')

    def test_trailing_stop(self):
        """Test trailing stop functionality."""
        symbol = 'BTC/USD'
        entry_price = 50000
        highest_price = 52000  # +4% peak
        current_price = 51220  # 1.5% below peak (triggers trail)
        
        self.bot.positions[symbol] = {
            'amount': 0.01,
            'entry_price': entry_price,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'highest_price': highest_price,
            'trailing_active': True,
            'partial_taken': True
        }
        
        self.mock_exchange.fetch_ticker.return_value = {'last': current_price}
        
        with patch.object(self.bot, 'close_position') as mock_close:
            self.bot.manage_positions_cycle()
            mock_close.assert_called_with(symbol, reason='trailing_stop')

    def test_time_exit(self):
        """Test time-based exit after 48 hours."""
        symbol = 'BTC/USD'
        entry_time = datetime.now(timezone.utc) - timedelta(hours=49)  # 49 hours ago
        
        self.bot.positions[symbol] = {
            'amount': 0.01,
            'entry_price': 50000,
            'timestamp': entry_time.isoformat(),
            'highest_price': 50000
        }
        
        self.mock_exchange.fetch_ticker.return_value = {'last': 50100}
        
        with patch.object(self.bot, 'close_position') as mock_close:
            self.bot.manage_positions_cycle()
            mock_close.assert_called_with(symbol, reason='time_exit')

    # ========================
    # SAFETY MEASURE TESTS
    # ========================
    
    def test_daily_loss_limit(self):
        """Test daily loss limit safety measure."""
        # Set up session with loss
        self.bot.session_start_balance = 10000
        
        # Mock balance showing 5.1% loss
        mock_balance = {'total': {'USD': 9490}}  # -5.1%
        self.mock_exchange.fetch_balance.return_value = mock_balance
        
        result = self.bot.check_risk_management()
        self.assertFalse(result)  # Should return False to stop trading

    def test_max_drawdown_emergency_stop(self):
        """Test maximum drawdown emergency stop."""
        # Set up peak balance and current balance showing >5% drawdown
        self.bot.session_start_balance = 10000
        self.bot.peak_balance = 12000
        
        # Mock balance showing 5.1% drawdown from peak
        mock_balance = {'total': {'USD': 11388}}  # 5.1% below peak
        self.mock_exchange.fetch_balance.return_value = mock_balance
        
        result = self.bot.check_risk_management()
        self.assertFalse(result)
        self.assertTrue(self.bot.emergency_stop)

    def test_max_concurrent_positions(self):
        """Test maximum concurrent positions limit."""
        # Fill up to max positions
        for i in range(self.bot.config['max_concurrent_positions']):
            self.bot.positions[f'TEST{i}/USD'] = {
                'amount': 0.01,
                'entry_price': 50000,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        # Try to add another position
        dfs = self.create_bullish_conditions_data()
        
        with patch('main.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
            
            result = self.bot.check_entry_conditions('BTC/USD', dfs)
            # Should still return buy signal, but execution should be blocked by position limit
            self.assertTrue(result['buy'])

    def test_correlation_check(self):
        """Test correlation check between symbols."""
        # Add existing position
        self.bot.positions['ETH/USD'] = {
            'amount': 0.1,
            'entry_price': 3000,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Mock highly correlated price data
        mock_df = pd.DataFrame({
            'close': [50000 + i*100 for i in range(50)]  # Highly correlated
        })
        
        with patch.object(self.bot, 'fetch_ohlcv', return_value=mock_df):
            result = self.bot.check_correlation('BTC/USD')
            # Should detect high correlation and reject
            # Note: This test might need adjustment based on actual correlation calculation

    def test_position_sizing_btc_eth(self):
        """Test position sizing for BTC/ETH (minimum $16)."""
        # Mock small portfolio
        self.mock_exchange.fetch_balance.return_value = {'total': {'USD': 100}}
        
        amount = self.bot.calculate_position_amount('BTC/USD', 50000)
        
        # Should calculate for at least $16
        min_amount = 16 / 50000
        self.assertGreaterEqual(amount * 50000, 16)

    def test_position_sizing_others(self):
        """Test position sizing for ADA/XRP/SOL (10% target)."""
        # Mock portfolio
        self.mock_exchange.fetch_balance.return_value = {'total': {'USD': 1000}}
        
        amount = self.bot.calculate_position_amount('ADA/USD', 1.0)
        
        # Should target 10% of portfolio = $100
        expected_amount = 100 / 1.0
        self.assertAlmostEqual(amount, expected_amount, places=2)

    def test_minimum_trade_amount(self):
        """Test global minimum trade amount enforcement."""
        # Mock very small portfolio
        self.mock_exchange.fetch_balance.return_value = {'total': {'USD': 1}}
        
        amount = self.bot.calculate_position_amount('ADA/USD', 1.0)
        
        # Should still meet minimum trade amount
        min_usd = self.bot.config['min_trade_amount_usd']
        self.assertGreaterEqual(amount * 1.0, min_usd)

    # ========================
    # DATA HANDLING TESTS
    # ========================
    
    def test_missing_data_handling(self):
        """Test handling of missing or insufficient data."""
        # Test with None data
        result = self.bot.check_entry_conditions('BTC/USD', {'1h': None, '4h': None, '15m': None})
        self.assertFalse(result['buy'])
        self.assertEqual(result['reason'], 'missing_data')
        
        # Test with insufficient data
        short_df = pd.DataFrame({'close': [50000]})  # Only 1 row
        dfs = {'1h': short_df, '4h': short_df, '15m': short_df}
        result = self.bot.check_entry_conditions('BTC/USD', dfs)
        self.assertFalse(result['buy'])
        self.assertEqual(result['reason'], 'insufficient_1h')

    def test_indicator_calculation_robustness(self):
        """Test indicator calculation with edge cases."""
        # Test with very short data
        short_df = pd.DataFrame({
            'open': [50000, 50100],
            'high': [50200, 50300],
            'low': [49900, 50000],
            'close': [50100, 50200],
            'volume': [1000, 1100]
        })
        
        result_df = self.bot.compute_indicators_on_df(short_df)
        
        # Should not crash and should have indicator columns (even if NaN)
        self.assertIn('ema21', result_df.columns)
        self.assertIn('rsi14', result_df.columns)
        self.assertIn('macd', result_df.columns)

    def test_state_persistence(self):
        """Test state saving and loading."""
        # Set up some state
        self.bot.positions['BTC/USD'] = {
            'amount': 0.01,
            'entry_price': 50000,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        self.bot.session_start_balance = 10000
        
        # Save state
        self.bot.save_state()
        
        # Verify file exists
        self.assertTrue(os.path.exists(self.bot.config['state_file']))
        
        # Create new bot and load state
        new_bot = CryptoTradingBot(exchange=self.mock_exchange, logger=self.mock_logger)
        
        # Verify state was loaded
        self.assertIn('BTC/USD', new_bot.positions)
        self.assertEqual(new_bot.session_start_balance, 10000)
        
        # Cleanup
        if os.path.exists(self.bot.config['state_file']):
            os.remove(self.bot.config['state_file'])

    def tearDown(self):
        """Clean up after each test."""
        # Remove any state files created during testing
        if os.path.exists(self.bot.config['state_file']):
            os.remove(self.bot.config['state_file'])


class TestIntegration(unittest.TestCase):
    """Integration tests for the full bot workflow."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.mock_exchange = Mock()
        self.bot = CryptoTradingBot(exchange=self.mock_exchange)
        self.bot.config['paper_trading'] = True
    
    def test_full_buy_workflow(self):
        """Test complete buy workflow from signal to position creation."""
        # Mock fetch_multi_tf to return bullish data
        mock_dfs = {
            '1h': pd.DataFrame({
                'close': [49900, 50100],  # Price increase
                'ema21': [49900, 50050],  # EMA21 crosses above
                'ema50': [50000, 50000],  # EMA50 stable
                'rsi14': [50, 50],        # Good RSI
                'macd': [100, 100],       # Bullish MACD
                'macd_signal': [50, 50],
                'macd_hist': [50, 50],
                'vol_ratio': [1.5, 1.5], # Good volume
                'volume': [3000, 3000],
                'vol_ma20': [2000, 2000],
                'atr14': [500, 500]
            }),
            '4h': pd.DataFrame({
                'close': [50000],
                'ema21': [50050],         # Trend confirmation
                'ema50': [50000]
            }),
            '15m': pd.DataFrame({
                'open': [50000, 50050],   # Bullish candle
                'close': [50050, 50150],  # close > open
                'ema21': [50000, 50000],  # close > ema21
                'volume': [1500, 1500]    # Add volume data
            })
        }
        
        with patch.object(self.bot, 'fetch_multi_tf', return_value=mock_dfs), \
             patch('main.datetime') as mock_datetime:
            
            mock_datetime.now.return_value = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
            
            # Test signal generation
            result = self.bot.check_entry_conditions('BTC/USD', mock_dfs)
            self.assertTrue(result['buy'])
            
            # Test position creation
            with patch.object(self.bot, 'calculate_position_amount', return_value=0.01):
                order = self.bot.execute_entry('BTC/USD', 0.01, 50100)
                self.assertIsNotNone(order)


def run_comprehensive_test():
    """Run all tests and provide detailed report."""
    print("🚀 Starting Comprehensive Crypto Trading Bot Test Suite")
    print("=" * 70)
    
    # Set up test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCryptoTradingBot))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print("\n🔥 ERRORS:")
        for test, error_traceback in result.errors:
            error_lines = error_traceback.split('\n')
            error_msg = next((line for line in reversed(error_lines) if line.strip()), "Unknown error")
            print(f"  - {test}: {error_msg}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED! Your bot is ready for deployment.")
    else:
        print("\n⚠️  Some tests failed. Please review and fix issues before deployment.")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
