#!/usr/bin/env python3
"""
Quick test script to verify Kraken API can place real orders
Tests with a small $5 BTC purchase
"""

import ccxt
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_kraken_buy():
    try:
        # Initialize exchange
        exchange = ccxt.kraken({
            'apiKey': os.getenv('KRAKEN_API_KEY'),
            'secret': os.getenv('KRAKEN_SECRET'),
            'sandbox': False,  # REAL TRADING - not sandbox
            'enableRateLimit': True,
        })
        
        print("🔗 Connecting to Kraken...")
        
        # Test connection and get balance
        balance = exchange.fetch_balance()
        usd_balance = balance['total'].get('USD', 0)  # Use 'USD' not 'ZUSD'
        print(f"💰 USD Balance: ${usd_balance:.2f}")
        
        if usd_balance < 15:
            print("❌ Insufficient USD balance for $15 test trade")
            return
        
        # Get current BTC price
        ticker = exchange.fetch_ticker('BTC/USD')
        btc_price = ticker['last']
        print(f"📈 Current BTC price: ${btc_price:,.2f}")
        
        # Calculate amount for $15 purchase (Kraken minimum ~0.0001 BTC)
        usd_amount = 15.0
        btc_amount = usd_amount / btc_price
        print(f"🧮 Buying ${usd_amount} = {btc_amount:.8f} BTC")
        
        # Confirm before placing order
        confirm = input("\n⚠️  REAL MONEY TRADE! Type 'YES' to proceed with $15 BTC purchase: ")
        if confirm != 'YES':
            print("❌ Trade cancelled")
            return
        
        # Place market buy order
        print("📦 Placing market buy order...")
        order = exchange.create_market_order('BTC/USD', 'buy', btc_amount)
        
        print("✅ Order placed successfully!")
        print(f"Order ID: {order['id']}")
        print(f"Status: {order['status']}")
        print(f"Amount: {order['amount']} BTC")
        print(f"Symbol: {order['symbol']}")
        
        # Check updated balance
        new_balance = exchange.fetch_balance()
        new_usd = new_balance['total'].get('USD', 0)  # Use 'USD' not 'ZUSD'
        new_btc = new_balance['total'].get('BTC', 0)  # Use 'BTC' not 'XXBT'
        
        print(f"\n💰 Updated Balances:")
        print(f"USD: ${new_usd:.2f} (was ${usd_balance:.2f})")
        print(f"BTC: {new_btc:.8f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Common issues:")
        print("- Check API keys are correct")
        print("- Ensure API has trading permissions")
        print("- Verify sufficient balance")
        print("- Check if 2FA is blocking API trades")

if __name__ == "__main__":
    print("🧪 Kraken API Test - $15 BTC Purchase")
    print("=" * 40)
    test_kraken_buy()
