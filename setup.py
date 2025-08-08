#!/usr/bin/env python3
"""
Setup script for Crypto Trading Bot
Run this before starting the bot for the first time
"""

import os
import shutil
from pathlib import Path

def setup_env_file():
    """Setup environment file"""
    env_example = Path('.env.example')
    env_file = Path('.env')
    
    if not env_file.exists():
        if env_example.exists():
            shutil.copy(env_example, env_file)
            print("✅ Created .env file from template")
            print("❗ Please edit .env file and add your Kraken API credentials")
            return False
        else:
            print("❌ .env.example file not found")
            return False
    else:
        print("✅ .env file already exists")
        return True

def check_api_keys():
    """Check if API keys are configured"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('KRAKEN_API_KEY')
        secret = os.getenv('KRAKEN_SECRET')
        
        if not api_key or api_key == 'your_kraken_api_key_here':
            print("❌ KRAKEN_API_KEY not configured in .env file")
            return False
            
        if not secret or secret == 'your_kraken_secret_here':
            print("❌ KRAKEN_SECRET not configured in .env file")
            return False
            
        print("✅ API keys are configured")
        return True
        
    except ImportError:
        print("❌ python-dotenv not installed")
        return False

def test_kraken_connection():
    """Test connection to Kraken"""
    try:
        import ccxt
        from dotenv import load_dotenv
        load_dotenv()
        
        exchange = ccxt.kraken({
            'apiKey': os.getenv('KRAKEN_API_KEY'),
            'secret': os.getenv('KRAKEN_SECRET'),
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        # Test with a simple API call
        balance = exchange.fetch_balance()
        print("✅ Successfully connected to Kraken")
        print(f"✅ Account currencies: {list(balance['total'].keys())}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to Kraken: {e}")
        print("   Please check your API credentials and permissions")
        return False

def main():
    """Run setup checks"""
    print("🚀 Crypto Trading Bot Setup")
    print("=" * 40)
    
    # Check environment file
    env_ready = setup_env_file()
    
    if not env_ready:
        print("\n📝 Next steps:")
        print("1. Edit the .env file and add your Kraken API credentials")
        print("2. Run this setup script again to test the connection")
        print("\n💡 How to get Kraken API keys:")
        print("   1. Log into your Kraken account")
        print("   2. Go to Settings > API Management")
        print("   3. Create a new API key with trading permissions")
        print("   4. Copy the API Key and Private Key to your .env file")
        return
    
    # Check API keys
    if not check_api_keys():
        print("\n❗ Please configure your API keys in the .env file")
        return
    
    # Test connection
    if test_kraken_connection():
        print("\n🎉 Setup complete! You can now run the trading bot:")
        print("   python main.py")
        print("\n📊 To monitor the bot:")
        print("   python dashboard.py")
    else:
        print("\n❌ Setup failed. Please check your API configuration.")

if __name__ == "__main__":
    main()
