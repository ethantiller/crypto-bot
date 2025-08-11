#!/usr/bin/env python3
"""
Full Crypto Trading Bot implementing MODERATE STRATEGY:
- EMA21 > EMA50 on 1H (no cross required) with 4H confirmation
- RSI14 between 30-70 (relaxed from 40-60), MACD(12,26,9), Volume >80% MA20, ATR14
- No 15m confirmation required (moderate)
- Position sizing with BTC/ETH min $16, ADA/XRP/SOL target 15% each
- Max 5 concurrent positions (increased from 3), larger position caps 25%
- Relaxed stop/take/partial/trailing levels for moderate risk
- No time window restrictions (24/7 trading)
- Discord notifications, bot_state.json persistence, comprehensive logging
- Market orders for execution (paper trading supported)
"""

import ccxt
import pandas as pd
import numpy as np
import ta
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
import time
import logging
import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import json
import requests
import math
import traceback

# Load environment variables from .env
load_dotenv()

class CryptoTradingBot:
    def __init__(self, exchange=None, logger=None):
        # Allow injecting logger/exchange for tests
        if logger is not None:
            self.logger = logger
        else:
            self.setup_logging()
        self.load_config()
        if exchange is not None:
            self.exchange = exchange
            if hasattr(self.exchange, 'fetch_balance'):
                self.logger.info("Using injected exchange instance")
        else:
            self.setup_exchange()
        self.positions = {}         # active positions {symbol: {...}}
        self.trade_history = []     # list of trade dicts
        self.session_start_balance = 0.0
        self.peak_balance = 0.0
        self.emergency_stop = False
        self.load_state()

    # ---------------------------
    # Setup & Utilities
    # ---------------------------
    def setup_logging(self):
        # Respect LOG_LEVEL from environment; default INFO
        log_level_name = os.getenv('LOG_LEVEL', 'INFO').upper()
        log_level = getattr(logging, log_level_name, logging.INFO)

        # Avoid duplicate handlers if re-initialized
        root_logger = logging.getLogger()
        if root_logger.handlers:
            for h in list(root_logger.handlers):
                root_logger.removeHandler(h)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('trading_bot.log', encoding='utf-8')
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        root_logger.setLevel(log_level)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(stream_handler)

        # Quiet very chatty libs
        logging.getLogger('ccxt').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized at level {log_level_name}")

    def load_config(self):
        # Primary config - adjust as necessary
        self.config = {
            'symbols': ['BTC/USD', 'ETH/USD', 'ADA/USD', 'XRP/USD', 'SOL/USD'],
            'check_interval': 30,  # seconds between position management cycles (30 seconds)
            'entry_scan_interval': 300,  # seconds between entry signal scans (5 minutes)
            'rate_limit_sleep': 1.1,  # seconds to sleep between exchange requests
            'paper_trading': os.getenv('PAPER_TRADING', 'True').lower() == 'true',
            'kraken_sandbox': os.getenv('KRAKEN_SANDBOX', 'False').lower() == 'true',
            # Indicators / Timeframes
            'tf_execution': '15m',
            'tf_signal': '1h',
            'tf_trend': '4h',
            # Indicators parameters
            'ema_short': 21,
            'ema_long': 50,
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'vol_ma_period': 20,
            'atr_period': 14,
            # Entry filters - MODERATE SETTINGS
            'rsi_entry_low': 30,         # Relaxed from 40 to 30
            'rsi_entry_high': 70,        # Relaxed from 60 to 70
            'rsi_hard_upper': 80,        # Relaxed from 70 to 80
            'rsi_hard_lower': 20,        # Relaxed from 30 to 20
            'max_recent_move_pct': 8.0,  # Relaxed from 5% to 8%
            # Position sizing rules - MODERATE SETTINGS
            'btc_eth_min_value': 16.0,   # minimum $16 position for BTC and ETH
            'others_target_pct': 0.15,   # Increased from 10% to 15% for alt coins
            'max_concurrent_positions': 5, # Increased from 3 to 5
            'per_asset_cap_pct': 0.25,   # Increased from 20% to 25% cap
            # Risk management - MODERATE SETTINGS
            'stop_loss_pct': 0.03,       # Relaxed from 2% to 3%
            'partial_tp_pct': 0.05,      # Relaxed from 3% to 5%
            'primary_tp_pct': 0.07,      # Relaxed from 5% to 7%
            'trailing_activate_pct': 0.05, # Relaxed from 3% to 5%
            'trailing_pct': 0.02,        # Relaxed from 1.5% to 2%
            'time_exit_hours': 72,       # Increased from 48 to 72 hours
            # Safety
            'daily_loss_limit_pct': 0.05,  # 5% daily loss limit -> stop trading
            'max_drawdown_pct': 0.05,      # 5% peak drawdown emergency stop
            'min_trade_amount_usd': 5.0,  # global absolute minimum trade (USD)
            # correlation settings
            'correlation_lookback_4h_hours': 30 * 24,  # fallback if needed
            'correlation_threshold': 0.8,
            # logging / state
            'state_file': 'bot_state.json',
            'log_file': 'trading_bot.log',
            # Discord webhook from your previous code (keeps your webhook)
            'discord_webhook_url': os.getenv('DISCORD_WEBHOOK_URL') or
                                   "https://discordapp.com/api/webhooks/1403453073485070527/P50wqbjUl9OHODBe1IoRNzT7jQ_ObpRhQQE-K50AfRB0m-utkNApo2UKA_HnMnF3jy0W"
        }

    def setup_exchange(self):
        try:
            api_key = os.getenv('KRAKEN_API_KEY')
            secret = os.getenv('KRAKEN_SECRET')
            if not api_key or not secret:
                self.logger.warning("Kraken API key/secret not found in environment; running in paper trading only.")
            self.exchange = ccxt.kraken({
                'apiKey': api_key,
                'secret': secret,
                'enableRateLimit': True,
                'rateLimit': 1000,
                'timeout': 30_000,
            })
            # Kraken sandbox: ccxt uses 'urls': {'api': {...}} but it's fine if not used; we respect config
            if self.config['kraken_sandbox']:
                self.logger.info("Sandbox mode enabled (note: confirm sandbox urls if needed).")
            # test fetch balance if keys exist
            if api_key and secret:
                try:
                    balance = self.exchange.fetch_balance()
                    self.logger.info("Connected to Kraken; balance keys present.")
                except Exception as e:
                    self.logger.warning(f"Couldn't fetch balance from Kraken: {e}")
        except Exception as e:
            self.logger.error(f"Exchange setup error: {e}")
            raise

    # ---------------------------
    # Discord notifications
    # ---------------------------
    def send_discord_notification(self, message, color=None):
        try:
            webhook = self.config.get('discord_webhook_url')
            if not webhook:
                return
            embed = {
                "title": "🤖 Crypto Trading Bot",
                "description": message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "color": color or 0x00ff00
            }
            payload = {"embeds": [embed]}
            # Note: Kraken webhook previously used returns 204 for success
            try:
                resp = requests.post(webhook, json=payload, timeout=10)
                if resp.status_code in (200, 204):
                    self.logger.info("Discord notification sent")
                else:
                    self.logger.warning(f"Discord webhook responded {resp.status_code}")
            except Exception as e:
                self.logger.warning(f"Discord send error: {e}")
        except Exception as e:
            self.logger.error(f"send_discord_notification error: {e}")

    # ---------------------------
    # Persistence
    # ---------------------------
    def save_state(self):
        try:
            state = {
                'positions': self.positions,
                'trade_history': self.trade_history[-500:],  # keep last 500
                'session_start_balance': self.session_start_balance,
                'peak_balance': self.peak_balance,
                'emergency_stop': self.emergency_stop,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            with open(self.config['state_file'], 'w') as f:
                json.dump(state, f, indent=2, default=str)
            self.logger.info("State saved")
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")

    def load_state(self):
        try:
            if os.path.exists(self.config['state_file']):
                with open(self.config['state_file'], 'r') as f:
                    state = json.load(f)
                self.positions = state.get('positions', {})
                self.trade_history = state.get('trade_history', [])
                self.session_start_balance = float(state.get('session_start_balance', 0.0) or 0.0)
                self.peak_balance = float(state.get('peak_balance', 0.0) or 0.0)
                self.emergency_stop = state.get('emergency_stop', False)
                self.logger.info("Loaded saved state")
            else:
                self.logger.info("No saved state found; starting fresh")
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")

    # ---------------------------
    # Data fetch & indicators
    # ---------------------------
    def fetch_ohlcv(self, symbol, timeframe, limit=500):
        """Fetch OHLCV and convert to DataFrame"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"fetch_ohlcv error {symbol} {timeframe}: {e}")
            return None

    def fetch_multi_tf(self, symbol, limit_1m=1200):
        """
        Fetch 1m and resample to 15m, 1h, 4h.
        Use limit_1m to gather sufficient history (e.g. 1200 minutes ~ 20 hours).
        """
        try:
            self.logger.debug(f"Fetching 1m data for {symbol} limit={limit_1m}")
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1m', limit=limit_1m)
            df1 = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df1['timestamp'] = pd.to_datetime(df1['timestamp'], unit='ms')
            df1.set_index('timestamp', inplace=True)
            # Resample (use non-deprecated aliases)
            df15 = df1.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
            df1h = df1.resample('1h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
            df4h = df1.resample('4h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
            self.logger.debug(f"Resampled {symbol}: 1m={len(df1)}, 15m={len(df15)}, 1h={len(df1h)}, 4h={len(df4h)}")
            return {'1m': df1, '15m': df15, '1h': df1h, '4h': df4h}
        except Exception as e:
            self.logger.error(f"fetch_multi_tf error for {symbol}: {e}")
            return None

    def compute_indicators_on_df(self, df):
        """Compute EMA21/50, RSI14, MACD, vol MA20, ATR14 for a DataFrame"""
        if df is None:
            return df
        try:
            df = df.copy()
            # Pre-create columns with NaN defaults
            for col in ['ema21','ema50','rsi14','macd','macd_signal','macd_hist','vol_ma20','vol_ratio','atr14']:
                if col not in df.columns:
                    df[col] = np.nan

            computed = []
            # EMA21
            if len(df) >= self.config['ema_short']:
                df['ema21'] = EMAIndicator(df['close'], window=self.config['ema_short']).ema_indicator()
                computed.append('ema21')
            # EMA50
            if len(df) >= self.config['ema_long']:
                df['ema50'] = EMAIndicator(df['close'], window=self.config['ema_long']).ema_indicator()
                computed.append('ema50')
            # RSI14
            if len(df) >= self.config['rsi_period'] + 1:
                df['rsi14'] = RSIIndicator(df['close'], window=self.config['rsi_period']).rsi()
                computed.append('rsi14')
            # MACD
            macd_min_len = max(self.config['macd_slow'], self.config['macd_signal']) + 1
            if len(df) >= macd_min_len:
                macd = MACD(df['close'], window_slow=self.config['macd_slow'],
                            window_fast=self.config['macd_fast'],
                            window_sign=self.config['macd_signal'])
                df['macd'] = macd.macd()
                df['macd_signal'] = macd.macd_signal()
                df['macd_hist'] = df['macd'] - df['macd_signal']
                computed.append('macd')
            # Volume MA and ratio (safe with min_periods=1)
            df['vol_ma20'] = df['volume'].rolling(window=self.config['vol_ma_period'], min_periods=1).mean()
            df['vol_ratio'] = df['volume'] / df['vol_ma20'].replace(0, np.nan)
            computed.append('vol')
            # ATR14
            if len(df) >= self.config['atr_period'] + 1:
                atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=self.config['atr_period'])
                df['atr14'] = atr.average_true_range()
                computed.append('atr14')
            self.logger.debug(f"Indicators computed (len={len(df)}): {computed}")
            # Fill forward or drop NaN when needed
            return df
        except Exception as e:
            self.logger.error(f"compute_indicators_on_df error: {e}")
            return df

    # ---------------------------
    # Entry logic
    # ---------------------------

    def recent_pct_move(self, df_15m, lookback_minutes=120):
        """
        Compute % move over last `lookback_minutes` using 15m data
        """
        try:
            if df_15m is None or len(df_15m) == 0:
                return 0.0
            lookback_bars = max(1, int(lookback_minutes / 15))
            recent = df_15m['close'].iloc[-lookback_bars:]
            pct = (recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0] * 100.0
            return abs(pct)
        except Exception as e:
            self.logger.debug(f"recent_pct_move error: {e}")
            return 0.0

    def check_15m_confirmation(self, df15):
        """
        15m confirmation candle rule: last 15m candle close > open (bullish),
        and close > ema21(15m) (momentum). Returns True/False.
        """
        try:
            if df15 is None or len(df15) < 2:
                return False
            # ensure ema21 exists before reading last row; be robust with short history
            if 'ema21' not in df15.columns:
                if len(df15) >= self.config['ema_short']:
                    df15['ema21'] = EMAIndicator(df15['close'], window=self.config['ema_short']).ema_indicator()
                else:
                    # Fallback with EWM to avoid ta window restrictions
                    df15['ema21'] = df15['close'].ewm(span=self.config['ema_short'], adjust=False, min_periods=1).mean()
            last = df15.iloc[-1]
            last_ema21 = df15['ema21'].iloc[-1]
            bullish = last['close'] > last['open']
            close_above_ema = last['close'] > last_ema21
            self.logger.debug(f"15m confirm check: open={last['open']:.4f} close={last['close']:.4f} ema21={last_ema21:.4f} -> bullish={bullish}, above_ema={close_above_ema}")
            return bullish and close_above_ema
        except Exception as e:
            self.logger.debug(f"check_15m_confirmation error: {e}")
            return False

    def log_detailed_analysis(self, symbols_data, cycle_ts):
        """Log detailed analysis for each cryptocurrency showing all indicators and conditions"""
        try:
            portfolio_usd = self.get_portfolio_usd()
        except:
            portfolio_usd = 95.0  # Fallback
            
        self.logger.info(f"🔍 DETAILED CRYPTO ANALYSIS @ {cycle_ts}")
        self.logger.info(f"   💰 Balance: ${portfolio_usd:.2f} | Active Positions: {len(self.positions)} | Next Scan: {self.config['entry_scan_interval']}s")
        self.logger.info(f"   � Open Positions: {list(self.positions.keys()) if self.positions else 'None'}")
        self.logger.info(f"   {'='*80}")
        
        for data in symbols_data:
            symbol = data.get('symbol', 'UNKNOWN')
            price = data.get('price', 0)
            
            # Extract all condition data
            ema21_1h = data.get('ema21_1h', 0)
            ema50_1h = data.get('ema50_1h', 0)
            ema21_4h = data.get('ema21_4h', 0)
            ema50_4h = data.get('ema50_4h', 0)
            rsi = data.get('rsi', 0)
            macd_hist = data.get('macd_hist', 0)
            macd_delta = data.get('macd_delta', 0)
            vol_ratio = data.get('vol_ratio', 0)
            recent_move = data.get('recent_move_pct', 0)
            
            # Condition checks
            ma_cross = data.get('ma_cross_up', False)
            trend_confirm = data.get('trend_confirm', False)
            rsi_ok = self.config['rsi_entry_low'] <= rsi <= self.config['rsi_entry_high']
            macd_ok = data.get('macd_ok', False)
            vol_ok = data.get('vol_ok', False)
            move_ok = recent_move <= self.config['max_recent_move_pct']
            confirm15 = data.get('15m_confirm', True)
            time_ok = not data.get('time_blocked', False)
            
            # Overall signal
            buy_signal = data.get('buy', False)
            reason = data.get('reason', 'unknown')
            
            # Format the line
            signal_icon = "🟢 BUY" if buy_signal else "🔴 NO BUY"
            
            self.logger.info(f"   {symbol:8s} | {signal_icon:8s} | ${price:8.2f} | "
                           f"EMA21/50(1h): {ema21_1h:7.2f}/{ema50_1h:7.2f} {'✅' if ma_cross else '❌'} | "
                           f"EMA21/50(4h): {ema21_4h:7.2f}/{ema50_4h:7.2f} {'✅' if trend_confirm else '❌'}")
            
            spacing = " " * 8
            self.logger.info(f"   {spacing} | RSI: {rsi:5.1f} {'✅' if rsi_ok else '❌'} | "
                           f"MACD: H={macd_hist:6.4f} D={macd_delta:6.4f} {'✅' if macd_ok else '❌'} | "
                           f"Vol: {vol_ratio:4.2f}x {'✅' if vol_ok else '❌'} | "
                           f"Move: {recent_move:4.1f}% {'✅' if move_ok else '❌'}")
            
            if not buy_signal:
                self.logger.info(f"   {spacing} | Reason: {reason}")
            
            self.logger.info(f"   {'-'*80}")

    def log_condition_check(self, symbol, conditions):
        """Log detailed condition check results in a readable format - ENHANCED"""
        # Store this data for batch logging later
        if not hasattr(self, '_symbol_data_cache'):
            self._symbol_data_cache = []
        
        # Add enhanced data for batch logging
        enhanced_data = conditions.copy()
        enhanced_data.update({
            'ema21_1h': conditions.get('ema21_1h', 0),
            'ema50_1h': conditions.get('ema50_1h', 0),
            'ema21_4h': conditions.get('ema21_4h', 0),
            'ema50_4h': conditions.get('ema50_4h', 0),
            'vol_ok': conditions.get('vol_ratio', 0) > 0.8,
        })
        self._symbol_data_cache.append(enhanced_data)

    def check_entry_conditions(self, symbol, dfs):
        """
        Returns dict with signals and reason. Follows:
         - 1H EMA21 crosses above EMA50 (on last closed 1H candle)
         - 4H EMA21 > EMA50
         - 1H RSI between 40-60, and not >70 or <30
         - 1H MACD hist positive and macd > signal
         - 1H volume > vol_ma20
         - 2-hour move <= 5%
         - 15m confirmation bullish candle
         - Not during first/last 30 minutes of day (configurable)
        """
        try:
            df1h = dfs.get('1h')
            df4h = dfs.get('4h')
            df15 = dfs.get('15m')
            if df1h is None or df4h is None or df15 is None:
                return {'buy': False, 'reason': 'missing_data'}

            # ensure indicators present
            df1h = self.compute_indicators_on_df(df1h)
            df4h = self.compute_indicators_on_df(df4h)
            df15 = self.compute_indicators_on_df(df15)

            # Need at least two 1H candles to detect cross
            if len(df1h) < 2:
                return {'buy': False, 'reason': 'insufficient_1h'}

            prev = df1h.iloc[-2]
            last = df1h.iloc[-1]
            # EMA cross
            # Guard against missing columns or NaNs in EMA columns
            for col in ['ema21','ema50']:
                if col not in df1h.columns:
                    return {'buy': False, 'reason': 'insufficient_indicators'}
            if pd.isna(prev['ema21']) or pd.isna(prev['ema50']) or pd.isna(last['ema21']) or pd.isna(last['ema50']):
                return {'buy': False, 'reason': 'insufficient_indicators'}
            # EMA cross - MODERATE: Just require EMA21 > EMA50 (no fresh cross needed)
            ema_spread_prev = float(prev['ema21'] - prev['ema50'])
            ema_spread_now = float(last['ema21'] - last['ema50'])
            ma_cross_up = ema_spread_now > 0  # Changed: just need EMA21 > EMA50, no cross required
            # 4H confirmation
            last4h = df4h.iloc[-1]
            for col in ['ema21','ema50']:
                if col not in df4h.columns:
                    return {'buy': False, 'reason': 'insufficient_trend_indicators'}
            if pd.isna(last4h['ema21']) or pd.isna(last4h['ema50']):
                return {'buy': False, 'reason': 'insufficient_trend_indicators'}
            trend_spread = float(last4h['ema21'] - last4h['ema50'])
            trend_confirm = trend_spread > 0
            # RSI checks
            rsi = last.get('rsi14', None)
            if rsi is None or math.isnan(rsi):
                return {'buy': False, 'reason': 'no_rsi'}
            if rsi > self.config['rsi_hard_upper'] or rsi < self.config['rsi_hard_lower']:
                return {'buy': False, 'reason': f'rsi_out_of_bounds ({rsi:.1f})'}
            if not (self.config['rsi_entry_low'] <= rsi <= self.config['rsi_entry_high']):
                # allow slightly outside but still not in hard bounds — we'll still reject per spec
                return {'buy': False, 'reason': f'rsi_not_in_40_60 ({rsi:.1f})'}

            # MACD check
            macd = float(last.get('macd', 0.0) or 0.0)
            macd_sig = float(last.get('macd_signal', 0.0) or 0.0)
            macd_hist = float(last.get('macd_hist', macd - macd_sig) or 0.0)
            macd_delta = float(macd - macd_sig)
            macd_ok = (macd_delta > 0) and (macd_hist > 0)

            # Volume confirmation - MODERATE SETTING
            vol_ratio = float(last.get('vol_ratio', 0.0) or 0.0)
            vol_ok = vol_ratio > 0.8  # Relaxed from 1.0 to 0.8 (80% of average volume)

            # 2-hour move check (use 15m series)
            recent_move = self.recent_pct_move(df15, lookback_minutes=120)
            move_ok = recent_move <= self.config['max_recent_move_pct']

            # 15m confirmation - MODERATE: Skip this requirement
            confirm15 = True  # Changed: Always pass 15m confirmation for moderate strategy

            # time-of-day block: MODERATE - Remove time restrictions
            time_blocked = False  # Changed: Allow trading at all times for moderate strategy

            # Aggregate decision
            buy = ma_cross_up and trend_confirm and macd_ok and vol_ok and move_ok and confirm15

            reason_parts = []
            if not ma_cross_up:
                reason_parts.append('no_ma_cross')
            if not trend_confirm:
                reason_parts.append('4h_no_trend_confirm')
            if not macd_ok:
                reason_parts.append('macd_no')
            if not vol_ok:
                reason_parts.append('volume_no')
            if not move_ok:
                reason_parts.append(f'recent_move_high({recent_move:.2f}%)')
            if not confirm15:
                reason_parts.append('no_15m_confirm')
            if not (self.config['rsi_entry_low'] <= rsi <= self.config['rsi_entry_high']):
                reason_parts.append('rsi_not_40_60')

            conditions_result = {
                'buy': bool(buy),
                'reason': ' & '.join(reason_parts) if reason_parts else 'all_ok',
                'rsi': float(rsi),
                'rsi_dist_low': float(rsi - self.config['rsi_entry_low']),
                'rsi_dist_high': float(self.config['rsi_entry_high'] - rsi),
                'ma_cross_up': bool(ma_cross_up),
                'ema_spread_prev': float(ema_spread_prev),
                'ema_spread_now': float(ema_spread_now),
                'trend_confirm': bool(trend_confirm),
                'trend_spread': float(trend_spread),
                'macd_ok': bool(macd_ok),
                'macd_hist': float(macd_hist),
                'macd_delta': float(macd_delta),
                'vol_ratio': float(vol_ratio),
                'recent_move_pct': float(recent_move),
                '15m_confirm': bool(confirm15),
                'time_blocked': bool(time_blocked),
                'price': float(df15['close'].iloc[-1]),
                'atr_1h': float(df1h['atr14'].iloc[-1]) if 'atr14' in df1h.columns else None,
                # Enhanced data for detailed logging
                'ema21_1h': float(last['ema21']) if 'ema21' in df1h.columns else 0,
                'ema50_1h': float(last['ema50']) if 'ema50' in df1h.columns else 0,
                'ema21_4h': float(last4h['ema21']) if 'ema21' in df4h.columns else 0,
                'ema50_4h': float(last4h['ema50']) if 'ema50' in df4h.columns else 0,
                'vol_ok': bool(vol_ok),
            }
            
            # Log detailed condition check
            self.log_condition_check(symbol, conditions_result)
            
            return conditions_result

        except Exception as e:
            self.logger.error(f"check_entry_conditions error for {symbol}: {e}\n{traceback.format_exc()}")
            return {'buy': False, 'reason': 'exception'}

    # ---------------------------
    # Position sizing & execution
    # ---------------------------
    def get_portfolio_usd(self):
        """Get estimated portfolio USD total (uses 'total' balances)."""
        try:
            bal = self.exchange.fetch_balance()
            usd = 0.0
            # Try common USD keys
            usd_keys = ['USD', 'ZUSD', 'USDT']  # Kraken uses ZUSD sometimes; but prefer USD
            for k in usd_keys:
                if k in bal.get('total', {}) and bal['total'][k] is not None:
                    usd = float(bal['total'][k])
                    self.logger.debug(f"Portfolio USD detected via key {k}: ${usd:.2f}")
                    break
            # Fall back to summing known fiat + crypto via last prices (costly). For now return usd.
            return float(usd)
        except Exception as e:
            self.logger.warning(f"get_portfolio_usd fallback error: {e}")
            return 0.0

    def calculate_position_amount(self, symbol, price):
        """
        Determine trade amount (in units) to place:
        - For BTC & ETH: minimum position value $16
        - For ADA/XRP/SOL: target 10% of portfolio
        - Respect per-asset cap of 20% of portfolio
        - Respect min_trade_amount_usd config
        Returns amount in base currency units (e.g., BTC)
        """
        try:
            portfolio_usd = self.get_portfolio_usd()
            if portfolio_usd <= 0:
                # Paper trading: assume small wallet fallback
                portfolio_usd = 95.0  # as you stated
            symbol_base = symbol.split('/')[0].upper()

            # Determine target notional (USD)
            if symbol_base in ['BTC', 'ETH', 'XBT']:
                target_notional = max(self.config['btc_eth_min_value'], portfolio_usd * 0.05)  # ensure reasonable fraction too
                # But don't exceed per-asset cap
                cap_notional = portfolio_usd * self.config['per_asset_cap_pct']
                target_notional = min(target_notional, cap_notional)
            else:
                target_notional = portfolio_usd * self.config['others_target_pct']
                cap_notional = portfolio_usd * self.config['per_asset_cap_pct']
                target_notional = min(target_notional, cap_notional)

            # Enforce global minimum
            if target_notional < self.config['min_trade_amount_usd']:
                target_notional = self.config['min_trade_amount_usd']

            qty = target_notional / price
            # rounding depending on coin
            if symbol_base == 'BTC' or symbol_base == 'XBT':
                qty = round(qty, 8)
            elif symbol_base == 'ETH':
                qty = round(qty, 6)
            else:
                qty = round(qty, 6)
            if qty <= 0:
                return 0.0
            return float(qty)
        except Exception as e:
            self.logger.error(f"calculate_position_amount error: {e}")
            return 0.0

    def execute_entry(self, symbol, amount, price):
        """Place market buy order (or simulate in paper trading). Returns order dict or None"""
        try:
            if amount <= 0:
                self.logger.warning(f"Zero amount; skipping buy for {symbol}")
                return None
            # In paper trading mode, simulate
            if self.config['paper_trading']:
                order = {
                    'id': f"paper_{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                    'symbol': symbol,
                    'side': 'buy',
                    'filled': amount,
                    'price': price,
                    'status': 'closed',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                # notify
                self.send_discord_notification(f"📝 [PAPER] BUY {symbol} amount={amount} @ {price:.2f}", color=0xffaa00)
                self.trade_history.append({'timestamp': datetime.now(timezone.utc).isoformat(), 'symbol': symbol, 'side': 'buy', 'amount': amount, 'price': price, 'order': order})
                return order
            else:
                # Place market order on Kraken
                self.logger.info(f"Placing MARKET BUY {symbol} amount={amount}")
                # Use create_market_order where ccxt exchange supports (some need create_order with 'market')
                order = self.exchange.create_market_buy_order(symbol, amount)
                self.logger.info(f"Placed market buy order: {order.get('id', order)}")
                self.send_discord_notification(f"💰 [LIVE] BUY {symbol} amount={amount} @ market", color=0x00ff00)
                self.trade_history.append({'timestamp': datetime.now(timezone.utc).isoformat(), 'symbol': symbol, 'side': 'buy', 'amount': amount, 'price': price, 'order': order})
                return order
        except Exception as e:
            self.logger.error(f"execute_entry error for {symbol}: {e}\n{traceback.format_exc()}")
            return None

    # ---------------------------
    # Position management & exits
    # ---------------------------
    def close_position(self, symbol, amount=None, reason='manual'):
        """Close a position fully (or partially if amount provided)"""
        try:
            if symbol not in self.positions:
                self.logger.warning(f"Tried to close {symbol} but no active position")
                return None
            pos = self.positions[symbol]
            amount_to_sell = amount if amount is not None else pos['amount']
            last_price_raw = self.exchange.fetch_ticker(symbol)['last'] if not self.config['paper_trading'] else pos['entry_price'] * (1 + 0.01)
            if last_price_raw is None:
                self.logger.warning(f"No last price available for {symbol}, skipping close")
                return None
            current_price = float(last_price_raw)
            if self.config['paper_trading']:
                # simulate sell
                order = {'id': f"paper_sell_{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}", 'status': 'closed', 'filled': amount_to_sell, 'price': current_price}
                self.send_discord_notification(f"📝 [PAPER] SELL {symbol} amount={amount_to_sell} @ {current_price:.2f} (reason: {reason})", color=0xff6600)
            else:
                order = self.exchange.create_market_sell_order(symbol, amount_to_sell)
                self.send_discord_notification(f"💸 [LIVE] SELL {symbol} amount={amount_to_sell} (reason: {reason})", color=0xff0000)

            # compute P&L in approximate terms
            entry = pos['entry_price']
            pnl_pct = (current_price - entry) / entry
            self.logger.info(f"Closed {symbol} amount={amount_to_sell} entry={entry:.4f} exit={current_price:.4f} P&L={pnl_pct:.2%} (reason: {reason})")
            # log
            self.trade_history.append({'timestamp': datetime.now(timezone.utc).isoformat(), 'symbol': symbol, 'side': 'sell', 'amount': amount_to_sell, 'price': current_price, 'order': order, 'reason': reason})
            # reduce or remove position
            if amount is None or math.isclose(amount_to_sell, pos['amount'], rel_tol=1e-6):
                del self.positions[symbol]
            else:
                pos['amount'] = round(pos['amount'] - amount_to_sell, 8)
                self.positions[symbol] = pos
            self.save_state()
            return order
        except Exception as e:
            self.logger.error(f"close_position error for {symbol}: {e}\n{traceback.format_exc()}")
            return None

    def manage_positions_cycle(self):
        """Run checks for each open position to enforce SL, TP, trailing, and time-based exits"""
        try:
            to_close = []
            for symbol, pos in list(self.positions.items()):
                try:
                    # fetch last price
                    if not self.config['paper_trading']:
                        ticker = self.exchange.fetch_ticker(symbol)
                        last_price_raw = ticker.get('last')
                        if last_price_raw is None:
                            self.logger.warning(f"No last price available for {symbol}, skipping position management")
                            continue
                        last_price = float(last_price_raw)
                    else:
                        last_price = pos['entry_price'] * 1.01
                    entry_price = pos['entry_price']
                    pnl_pct = (last_price - entry_price) / entry_price
                    
                    # Calculate exit condition thresholds
                    sl_trigger = -self.config['stop_loss_pct']
                    partial_tp_trigger = self.config['partial_tp_pct']
                    primary_tp_trigger = self.config['primary_tp_pct']
                    trail_activate_trigger = self.config['trailing_activate_pct']
                    
                    trail_active = pos.get('trailing_active', False)
                    trail_highest = pos.get('highest_price', last_price)
                    trail_price = trail_highest * (1 - self.config['trailing_pct']) if trail_active else None
                    age_hours = (datetime.now(timezone.utc) - datetime.fromisoformat(pos['timestamp']).replace(tzinfo=timezone.utc)).total_seconds() / 3600
                    partial_taken = pos.get('partial_taken', False)
                    
                    # Enhanced position management logging
                    self.logger.info(f"🔍 POSITION CHECK for {symbol}:")
                    self.logger.info(f"   💰 Current: ${last_price:.4f} | Entry: ${entry_price:.4f} | P&L: {pnl_pct:.2%}")
                    self.logger.info(f"   ⏰ Age: {age_hours:.1f}h | Partial Taken: {'Yes' if partial_taken else 'No'}")
                    
                    # Stop Loss Check
                    sl_hit = pnl_pct <= sl_trigger
                    sl_status = "🚨 TRIGGERED" if sl_hit else "✅ OK"
                    self.logger.info(f"   🛑 Stop Loss ({sl_trigger:.1%}): {sl_status} | Current P&L: {pnl_pct:.2%}")
                    
                    # Partial TP Check
                    partial_ready = not partial_taken and pnl_pct >= partial_tp_trigger
                    partial_status = "💰 READY" if partial_ready else ("✅ TAKEN" if partial_taken else "⏳ WAITING")
                    self.logger.info(f"   🎯 Partial TP ({partial_tp_trigger:.1%}): {partial_status}")
                    
                    # Primary TP Check
                    primary_tp_hit = pnl_pct >= primary_tp_trigger
                    primary_status = "💰 TRIGGERED" if primary_tp_hit else "⏳ WAITING"
                    self.logger.info(f"   🎯 Primary TP ({primary_tp_trigger:.1%}): {primary_status}")
                    
                    # Trailing Stop Check
                    trail_will_activate = not trail_active and pnl_pct >= trail_activate_trigger
                    if trail_active:
                        trail_hit = trail_price is not None and last_price <= trail_price and trail_highest > entry_price
                        trail_status = "🚨 HIT" if trail_hit else f"✅ ACTIVE (Trail: ${trail_price:.4f})" if trail_price else "❌ ERROR"
                    elif trail_will_activate:
                        trail_status = "🟡 ACTIVATING"
                    else:
                        trail_status = "⏳ INACTIVE"
                    self.logger.info(f"   🔄 Trailing Stop: {trail_status} | Highest: ${trail_highest:.4f}")
                    
                    # Time Exit Check
                    time_exit_due = age_hours >= self.config['time_exit_hours']
                    time_status = "⏰ DUE" if time_exit_due else "✅ OK"
                    self.logger.info(f"   ⏰ Time Exit ({self.config['time_exit_hours']}h): {time_status}")
                    
                    # update highest price for trailing
                    if 'highest_price' not in pos or last_price > pos.get('highest_price', entry_price):
                        pos['highest_price'] = last_price
                    
                    # Execute exit conditions in priority order
                    # Stop loss
                    if sl_hit:
                        self.logger.info(f"🚨 EXECUTING: Stop loss for {symbol}")
                        self.close_position(symbol, reason='stop_loss')
                        continue
                    
                    # Partial TP at 3%: if still hasn't taken partial, take 50% and set trailing
                    if partial_ready:
                        half = round(pos['amount'] * 0.5, 8)
                        if half > 0:
                            self.logger.info(f"💰 EXECUTING: Partial TP for {symbol} (50% at {pnl_pct:.2%})")
                            self.close_position(symbol, amount=half, reason='partial_take_profit')
                            pos['partial_taken'] = True
                            # keep trailing active (set minimal)
                            pos['trailing_active'] = True
                            pos['trail_start_price'] = pos.get('highest_price', last_price)
                            self.positions[symbol] = pos
                            continue
                    
                    # Activate trailing when profit >= trailing_activate_pct
                    if trail_will_activate:
                        self.logger.info(f"🔄 ACTIVATING: Trailing stop for {symbol}")
                        pos['trailing_active'] = True
                        pos['trail_start_price'] = pos.get('highest_price', last_price)
                    
                    # Trailing stop enforcement
                    if trail_active:
                        highest = pos.get('highest_price', last_price)
                        trail_price = highest * (1 - self.config['trailing_pct'])
                        if last_price <= trail_price and highest > entry_price:
                            self.logger.info(f"🔄 EXECUTING: Trailing stop for {symbol} (${last_price:.4f} <= ${trail_price:.4f})")
                            self.close_position(symbol, reason='trailing_stop')
                            continue
                    
                    # Primary TP
                    if primary_tp_hit:
                        self.logger.info(f"🎯 EXECUTING: Primary TP for {symbol} at {pnl_pct:.2%}")
                        self.close_position(symbol, reason='primary_take_profit')
                        continue
                    
                    # Time-based exit
                    if time_exit_due:
                        self.logger.info(f"⏰ EXECUTING: Time exit for {symbol} after {age_hours:.1f}h")
                        self.close_position(symbol, reason='time_exit')
                        continue
                    
                    # update pos record
                    self.positions[symbol] = pos
                    self.logger.info(f"   ✅ Position {symbol} maintained")
                    self.logger.info(f"   {'='*50}")
                    
                except Exception as inner_e:
                    self.logger.error(f"Error managing position {symbol}: {inner_e}\n{traceback.format_exc()}")
            # Save if any closed
            self.save_state()
        except Exception as e:
            self.logger.error(f"manage_positions_cycle top error: {e}\n{traceback.format_exc()}")

    # ---------------------------
    # Correlation check (simple)
    # ---------------------------
    def check_correlation(self, candidate_symbol):
        """
        Compute correlation (Pearson) between candidate_symbol and existing position symbols on 4H returns.
        If correlation > threshold, refuse to open candidate if conflict arises.
        """
        try:
            active_symbols = list(self.positions.keys())
            if not active_symbols:
                return True
            # fetch 4h returns for candidate and each active pos
            candidate_df = self.fetch_ohlcv(candidate_symbol, '4h', limit=200)
            if candidate_df is None or len(candidate_df) < 10:
                return True
            candidate_returns = candidate_df['close'].pct_change().dropna()
            for sym in active_symbols:
                df = self.fetch_ohlcv(sym, '4h', limit=200)
                if df is None or len(df) < 10:
                    continue
                returns = df['close'].pct_change().dropna()
                # align lengths
                minlen = min(len(candidate_returns), len(returns))
                if minlen < 5:
                    continue
                corr = candidate_returns.iloc[-minlen:].corr(returns.iloc[-minlen:])
                if corr is not None and abs(corr) >= self.config['correlation_threshold']:
                    self.logger.warning(f"Correlation {corr:.2f} between {candidate_symbol} and {sym} exceeds threshold")
                    return False
            return True
        except Exception as e:
            self.logger.error(f"check_correlation error: {e}")
            return True

    # ---------------------------
    # Risk checks
    # ---------------------------
    def check_risk_management(self):
        """
        - Update session start / peak balances
        - Check daily loss and drawdown emergency stop
        """
        try:
            bal = None
            try:
                bal = self.exchange.fetch_balance()
            except Exception as e:
                self.logger.warning(f"Could not fetch balance: {e}")
                # in paper mode simulate
            current_usd = 0.0
            if bal:
                # try typical keys
                for key in ['USD', 'ZUSD', 'USDT']:
                    if key in bal.get('total', {}) and bal['total'][key] is not None:
                        current_usd = float(bal['total'][key])
                        break
            else:
                current_usd = self.session_start_balance if self.session_start_balance > 0 else 95.0

            if self.session_start_balance == 0.0:
                self.session_start_balance = current_usd
                self.peak_balance = current_usd
                self.logger.info(f"Session start balance set to ${self.session_start_balance:.2f}")

            if current_usd > self.peak_balance:
                self.peak_balance = current_usd

            drawdown_pct = (self.peak_balance - current_usd) / self.peak_balance if self.peak_balance > 0 else 0.0
            daily_pct = (current_usd - self.session_start_balance) / max(self.session_start_balance, 1.0)

            self.logger.info(f"Risk: balance=${current_usd:.2f}, peak=${self.peak_balance:.2f}, drawdown={drawdown_pct:.2%}, daily={daily_pct:.2%}")

            if drawdown_pct >= self.config['max_drawdown_pct']:
                self.emergency_stop = True
                self.send_discord_notification(f"🚨 EMERGENCY STOP: Peak drawdown {drawdown_pct:.2%} exceeded {self.config['max_drawdown_pct']:.2%}", color=0xff0000)
                return False
            if daily_pct <= -self.config['daily_loss_limit_pct']:
                self.logger.warning("Daily loss limit reached, pausing trading until next day")
                self.send_discord_notification(f"⚠️ DAILY LOSS LIMIT reached: {daily_pct:.2%}. No new trades today.", color=0xff6600)
                return False
            return True
        except Exception as e:
            self.logger.error(f"check_risk_management error: {e}\n{traceback.format_exc()}")
            return True

    # ---------------------------
    # Main loop
    # ---------------------------
    def run(self):
        """Main loop: signal detection, execution, position management"""
        try:
            # Startup message
            mode = "PAPER" if self.config['paper_trading'] else "LIVE"
            strategy = "MODERATE"
            self.logger.info(f"Starting CryptoTradingBot ({mode} - {strategy}) with symbols: {self.config['symbols']}")
            self.logger.info(f"📊 MODERATE STRATEGY CONFIG:")
            self.logger.info(f"   RSI Range: {self.config['rsi_entry_low']}-{self.config['rsi_entry_high']} (was 40-60)")
            self.logger.info(f"   Max Positions: {self.config['max_concurrent_positions']} (was 3)")
            self.logger.info(f"   Alt Coin Target: {self.config['others_target_pct']:.1%} (was 10%)")
            self.logger.info(f"   Stop Loss: {self.config['stop_loss_pct']:.1%} (was 2%)")
            self.logger.info(f"   Recent Move Limit: {self.config['max_recent_move_pct']:.1f}% (was 5%)")
            self.logger.info(f"   Volume Threshold: 80% of MA20 (was 100%)")
            self.logger.info(f"   15m Confirmation: DISABLED (was required)")
            self.logger.info(f"   Time Restrictions: DISABLED (24/7 trading)")
            self.logger.info(f"Position management every {self.config['check_interval']}s, Entry scans every {self.config['entry_scan_interval']}s")
            self.send_discord_notification(f"🟡 Bot started ({mode} - {strategy}). Symbols: {', '.join(self.config['symbols'])}", color=0x00aa00)

            # Calculate how many position management cycles before doing entry scan
            cycles_per_entry_scan = max(1, int(self.config['entry_scan_interval'] / self.config['check_interval']))
            cycle_count = 0

            while True:
                try:
                    cycle_ts = datetime.now(timezone.utc).isoformat(timespec='seconds')
                    is_entry_scan_cycle = (cycle_count % cycles_per_entry_scan) == 0
                    
                    if is_entry_scan_cycle:
                        self.logger.info(f"=== FULL CYCLE (Entry + Position Management) @ {cycle_ts} ===")
                    else:
                        self.logger.info(f"=== POSITION MANAGEMENT CYCLE @ {cycle_ts} ===")
                    
                    if self.emergency_stop:
                        self.logger.error("Emergency stop active - sleeping until manual reset")
                        time.sleep(self.config['check_interval'])
                        cycle_count += 1
                        continue

                    if not self.check_risk_management():
                        # skip cycle if risk rules fail
                        time.sleep(self.config['check_interval'])
                        cycle_count += 1
                        continue

                    # ALWAYS manage existing positions (check SL/TP/trailing/time exits)
                    if self.positions:
                        self.logger.info(f"Managing {len(self.positions)} open position(s): {list(self.positions.keys())}")
                        self.manage_positions_cycle()

                    # Only scan for new entries every entry_scan_interval
                    signals = []  # Initialize signals for summary
                    if is_entry_scan_cycle:
                        # Initialize symbol data cache for detailed logging
                        self._symbol_data_cache = []
                        
                        # enforce max concurrent positions
                        if len(self.positions) >= self.config['max_concurrent_positions']:
                            self.logger.info(f"Max concurrent positions reached ({len(self.positions)}) - skipping new entries")
                        else:
                            # Evaluate signals for all symbols and execute up to available slots
                            self.logger.info(f"📊 EVALUATING SIGNALS for {len(self.config['symbols'])} symbols...")
                            
                            for symbol in self.config['symbols']:
                                try:
                                    dfs = self.fetch_multi_tf(symbol, limit_1m=720)  # e.g., last 12 hours of 1m bars
                                    if dfs is None:
                                        time.sleep(self.config['rate_limit_sleep'])
                                        continue
                                    result = self.check_entry_conditions(symbol, dfs)
                                    # attach more info
                                    result['symbol'] = symbol
                                    result['current_price'] = result.get('price') or (dfs['15m']['close'].iloc[-1] if dfs['15m'] is not None else None)
                                    signals.append(result)
                                except Exception as e:
                                    self.logger.error(f"Signal evaluation error for {symbol}: {e}\n{traceback.format_exc()}")
                                time.sleep(self.config['rate_limit_sleep'])

                        # Log detailed analysis for all symbols
                        if hasattr(self, '_symbol_data_cache') and self._symbol_data_cache:
                            self.log_detailed_analysis(self._symbol_data_cache, cycle_ts)

                        # Continue with execution logic only if we have slots
                        if len(self.positions) < self.config['max_concurrent_positions']:
                            # Rank signals by simple priority: prefer ones with all_ok and higher vol_ratio
                            ranked = [s for s in signals if s.get('symbol') and s.get('buy')]
                            # also keep those with reason 'all_ok' first
                            ranked = sorted(ranked, key=lambda x: (0 if x.get('reason') == 'all_ok' else 1, -x.get('vol_ratio', 0)))
                            slots = self.config['max_concurrent_positions'] - len(self.positions)
                            for sig in ranked[:slots]:
                                sym = sig['symbol']
                                # Re-check correlation and exposure
                                if not self.check_correlation(sym):
                                    self.logger.info(f"Skipping {sym} due to correlation check")
                                    continue
                                # calculate amount in base units
                                price = float(sig['current_price']) if sig.get('current_price') else None
                                if not price or price <= 0:
                                    self.logger.warning(f"Invalid price for {sym}, skipping")
                                    continue
                                amount = self.calculate_position_amount(sym, price)
                                if amount <= 0:
                                    self.logger.info(f"Calculated amount for {sym} is zero, skipping")
                                    continue
                                # execute buy
                                self.logger.info(f"Executing entry for {sym}: amount={amount}, price={price}")
                                order = self.execute_entry(sym, amount, price)
                                if order:
                                    # create position record
                                    pos = {
                                        'side': 'buy',
                                        'amount': amount,
                                        'entry_price': price,
                                        'timestamp': datetime.now(timezone.utc).isoformat(),
                                        'highest_price': price,
                                        'partial_taken': False,
                                        'trailing_active': False
                                    }
                                    self.positions[sym] = pos
                                    self.logger.info(f"Position opened: {sym} amount={amount} entry={price}")
                                    self.save_state()
                                time.sleep(self.config['rate_limit_sleep'])

                    # End of cycle housekeeping
                    self.save_state()
                    
                    # Cycle Summary
                    if is_entry_scan_cycle:
                        buy_signals = [s for s in signals if s.get('buy')]
                        total_evaluated = len(signals)
                        self.logger.info(f"📋 FULL CYCLE SUMMARY:")
                        self.logger.info(f"   📊 Symbols Evaluated: {total_evaluated}")
                        self.logger.info(f"   🟢 Buy Signals: {len(buy_signals)}")
                        self.logger.info(f"   📈 Active Positions: {len(self.positions)} ({list(self.positions.keys())})")
                        self.logger.info(f"   ⏳ Next entry scan in {self.config['entry_scan_interval']}s")
                        self.logger.info(f"   {'='*60}")
                    else:
                        self.logger.info(f"📋 POSITION MANAGEMENT SUMMARY:")
                        self.logger.info(f"   📈 Active Positions: {len(self.positions)} ({list(self.positions.keys())})")
                        remaining_cycles = cycles_per_entry_scan - (cycle_count % cycles_per_entry_scan) - 1
                        self.logger.info(f"   ⏳ Next entry scan in {remaining_cycles * self.config['check_interval']}s ({remaining_cycles} cycles)")
                        self.logger.info(f"   {'='*60}")
                    
                    cycle_count += 1
                    time.sleep(self.config['check_interval'])

                except KeyboardInterrupt:
                    self.logger.info("Bot stopped by user interrupt")
                    self.send_discord_notification("🔴 Bot stopped by user", color=0xff0000)
                    break
                except Exception as e:
                    self.logger.error(f"Unexpected error in main loop: {e}\n{traceback.format_exc()}")
                    self.send_discord_notification(f"❌ Bot error: {e}", color=0xff0000)
                    cycle_count += 1
                    time.sleep(self.config['check_interval'])
        finally:
            self.save_state()
            self.logger.info("Bot shutdown - state saved")

if __name__ == "__main__":
    bot = CryptoTradingBot()
    bot.run()
