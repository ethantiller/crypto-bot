#!/usr/bin/env python3
"""
Script to fix emergency stop issue by correcting portfolio calculation
"""

import json
import os
from datetime import datetime, timezone

def fix_bot_state():
    """Fix the bot state file to remove emergency stop and correct portfolio calculation"""
    
    state_file = 'bot_state.json'
    backup_file = 'bot_state_backup.json'
    
    # Check if state file exists
    if not os.path.exists(state_file):
        print(f"❌ {state_file} not found")
        return False
    
    # Load current state
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
        print(f"✅ Loaded {state_file}")
    except Exception as e:
        print(f"❌ Error loading {state_file}: {e}")
        return False
    
    # Backup current state
    try:
        with open(backup_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        print(f"✅ Backed up current state to {backup_file}")
    except Exception as e:
        print(f"⚠️ Warning: Could not create backup: {e}")
    
    # Display current state
    print("\n📊 Current State:")
    print(f"   Emergency stop: {state.get('emergency_stop', False)}")
    print(f"   Peak balance: ${state.get('peak_balance', 0):.2f}")
    print(f"   Session start: ${state.get('session_start_balance', 0):.2f}")
    print(f"   Positions: {list(state.get('positions', {}).keys())}")
    
    # Calculate portfolio value if positions exist
    total_portfolio = state.get('session_start_balance', 92.70)
    if state.get('positions'):
        print("\n💰 Calculating portfolio value:")
        portfolio_value = 0.0
        
        for symbol, pos in state['positions'].items():
            amount = pos.get('amount', 0)
            entry_price = pos.get('entry_price', 0)
            value = amount * entry_price
            portfolio_value += value
            print(f"   {symbol}: {amount:.6f} @ ${entry_price:.2f} = ${value:.2f}")
        
        # Add estimated cash balance (you can adjust this)
        estimated_cash = 76.63  # From your logs
        total_portfolio = estimated_cash + portfolio_value
        print(f"   Estimated cash: ${estimated_cash:.2f}")
        print(f"   Total portfolio: ${total_portfolio:.2f}")
    
    # Fix the state
    changes_made = []
    
    if state.get('emergency_stop', False):
        state['emergency_stop'] = False
        changes_made.append("Disabled emergency stop")
    
    # Reset peak balance to current portfolio value or session start (whichever is higher)
    old_peak = state.get('peak_balance', 0)
    new_peak = max(total_portfolio, state.get('session_start_balance', 92.70))
    if abs(old_peak - new_peak) > 0.01:  # Only update if significantly different
        state['peak_balance'] = new_peak
        changes_made.append(f"Updated peak balance from ${old_peak:.2f} to ${new_peak:.2f}")
    
    # Update timestamp
    state['timestamp'] = datetime.now(timezone.utc).isoformat()
    changes_made.append("Updated timestamp")
    
    # Save fixed state
    try:
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        print(f"\n✅ Fixed state saved to {state_file}")
    except Exception as e:
        print(f"❌ Error saving fixed state: {e}")
        return False
    
    # Show changes
    if changes_made:
        print("\n🔧 Changes made:")
        for change in changes_made:
            print(f"   • {change}")
    else:
        print("\n✅ No changes needed - state looks good")
    
    print(f"\n📊 Fixed State:")
    print(f"   Emergency stop: {state['emergency_stop']}")
    print(f"   Peak balance: ${state['peak_balance']:.2f}")
    print(f"   Session start: ${state['session_start_balance']:.2f}")
    
    return True

if __name__ == "__main__":
    print("🤖 Crypto Trading Bot - Emergency Stop Fix")
    print("=" * 50)
    
    success = fix_bot_state()
    
    if success:
        print("\n🎉 Bot state fixed successfully!")
        print("\nYou can now restart your trading bot:")
        print("   python main.py")
    else:
        print("\n❌ Failed to fix bot state. Please check the errors above.")
