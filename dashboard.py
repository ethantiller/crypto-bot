#!/usr/bin/env python3
"""
Simple monitoring dashboard for the crypto trading bot
Run this to check bot status and performance
"""

import json
import os
from datetime import datetime, timedelta
import pandas as pd

class BotMonitor:
    def __init__(self):
        self.bot_dir = os.path.dirname(os.path.abspath(__file__))
        
    def load_bot_state(self):
        """Load current bot state"""
        try:
            with open(os.path.join(self.bot_dir, 'bot_state.json'), 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("No bot state file found")
            return None
        except Exception as e:
            print(f"Error loading bot state: {e}")
            return None
            
    def load_log_file(self):
        """Parse recent log entries"""
        log_file = os.path.join(self.bot_dir, 'trading_bot.log')
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                return lines[-50:]  # Last 50 lines
        except FileNotFoundError:
            print("No log file found")
            return []
        except Exception as e:
            print(f"Error reading log file: {e}")
            return []
            
    def display_positions(self, state):
        """Display current positions"""
        if not state or not state.get('positions'):
            print("No open positions")
            return
            
        print("\n=== CURRENT POSITIONS ===")
        for symbol, position in state['positions'].items():
            entry_time = datetime.fromisoformat(position['timestamp'])
            duration = datetime.now() - entry_time
            
            print(f"Symbol: {symbol}")
            print(f"  Side: {position['side']}")
            print(f"  Amount: {position['amount']:.6f}")
            print(f"  Entry Price: ${position['entry_price']:.2f}")
            print(f"  Entry Time: {entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Duration: {str(duration).split('.')[0]}")
            print("")
            
    def display_recent_trades(self, state):
        """Display recent trade history"""
        if not state or not state.get('trade_history'):
            print("No trade history found")
            return
            
        print("\n=== RECENT TRADES (Last 10) ===")
        recent_trades = state['trade_history'][-10:]
        
        for trade in reversed(recent_trades):
            timestamp = datetime.fromisoformat(trade['timestamp'])
            print(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} | "
                  f"{trade['side'].upper()} {trade['amount']:.6f} {trade['symbol']} | "
                  f"Price: ${trade.get('price', 'N/A')}")
                  
    def calculate_performance(self, state):
        """Calculate basic performance metrics"""
        if not state or not state.get('trade_history'):
            print("No trade history for performance calculation")
            return
            
        trades = state['trade_history']
        buy_trades = [t for t in trades if t['side'] == 'buy']
        sell_trades = [t for t in trades if t['side'] == 'sell']
        
        print("\n=== PERFORMANCE SUMMARY ===")
        print(f"Total Trades: {len(trades)}")
        print(f"Buy Orders: {len(buy_trades)}")
        print(f"Sell Orders: {len(sell_trades)}")
        
        if buy_trades:
            last_24h = datetime.now() - timedelta(hours=24)
            recent_trades = [t for t in trades 
                           if datetime.fromisoformat(t['timestamp']) > last_24h]
            print(f"Trades (24h): {len(recent_trades)}")
            
    def display_system_info(self):
        """Display system information"""
        print("\n=== SYSTEM INFO ===")
        
        # Check if bot service is running
        try:
            import subprocess
            result = subprocess.run(['systemctl', 'is-active', 'crypto-bot'], 
                                  capture_output=True, text=True)
            service_status = result.stdout.strip()
            print(f"Bot Service Status: {service_status}")
        except:
            print("Bot Service Status: Unknown (systemctl not available)")
            
        # Check disk space
        try:
            import shutil
            disk_usage = shutil.disk_usage('.')
            free_gb = disk_usage.free / (1024**3)
            print(f"Free Disk Space: {free_gb:.1f} GB")
        except:
            print("Disk Space: Unknown")
            
        # Check log file size
        try:
            log_size = os.path.getsize('trading_bot.log') / (1024**2)
            print(f"Log File Size: {log_size:.1f} MB")
        except:
            print("Log File Size: Unknown")
            
    def display_recent_logs(self):
        """Display recent log entries"""
        print("\n=== RECENT LOG ENTRIES ===")
        logs = self.load_log_file()
        
        for log in logs[-10:]:  # Last 10 entries
            print(log.strip())
            
    def run_monitor(self):
        """Run the monitoring dashboard"""
        print("=" * 60)
        print("CRYPTO TRADING BOT MONITOR")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        state = self.load_bot_state()
        
        self.display_system_info()
        self.display_positions(state)
        self.display_recent_trades(state)
        self.calculate_performance(state)
        self.display_recent_logs()
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    monitor = BotMonitor()
    monitor.run_monitor()