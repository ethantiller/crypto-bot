# Crypto Trading Bot

A Python-based cryptocurrency trading bot that implements a Moving Average Crossover strategy with RSI filtering on the Kraken exchange.

## Strategy Overview

**Moving Average Crossover + RSI Filter (Moderate Risk)**

- **Entry Signal**: Buy when short-term MA (10) crosses above long-term MA (30) AND RSI < 65
- **Exit Signal**: Sell when short-term MA crosses below long-term MA OR RSI > 70
- **Position Sizing**: 7.5% of portfolio per trade (within 5-10% range)
- **Stop Loss**: 4% below entry price (within 3-5% range)
- **Take Profit**: 10% above entry price (within 8-12% range)
- **Risk Management**: Prevents over-trading and reduces drawdowns

## Features

- ✅ Real-time trading on Kraken exchange
- ✅ Moving Average Crossover strategy with RSI confirmation
- ✅ Built-in risk management (stop-loss, take-profit)
- ✅ Position sizing based on portfolio percentage
- ✅ Trade logging and state persistence
- ✅ Monitoring dashboard
- ✅ Error handling and recovery

## Quick Start

### 1. Setup Environment

```bash
# Clone or navigate to the project directory
cd crypto-bot

# Create and activate virtual environment (if not already done)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Run setup script
python setup.py
```

This will create a `.env` file. Edit it with your Kraken API credentials:

```env
KRAKEN_API_KEY=your_actual_api_key
KRAKEN_SECRET=your_actual_secret_key
```

**Getting Kraken API Keys:**
1. Log into your Kraken account
2. Go to Settings → API Management
3. Create a new API key with these permissions:
   - Query Funds
   - Query Open Orders
   - Query Closed Orders
   - Query Trades History
   - Create & Modify Orders
   - Cancel Orders

### 3. Test Setup

```bash
# Test your configuration
python setup.py
```

### 4. Run the Bot

```bash
# Start trading (with real money - be careful!)
python main.py
```

### 5. Monitor Performance

```bash
# In another terminal, check bot status
python dashboard.py
```

## Configuration

Edit the configuration in `main.py` → `load_config()` method:

```python
self.config = {
    'symbols': ['BTC/USD', 'ETH/USD', 'ADA/USD'],  # Trading pairs
    'timeframe': '1h',              # Candlestick timeframe
    'short_ma_period': 10,          # Short MA period
    'long_ma_period': 30,           # Long MA period
    'rsi_period': 14,               # RSI calculation period
    'rsi_entry_threshold': 65,      # Buy when RSI < this
    'rsi_exit_threshold': 70,       # Sell when RSI > this
    'position_size_pct': 0.075,     # 7.5% of portfolio per trade
    'stop_loss_pct': 0.04,          # 4% stop loss
    'take_profit_pct': 0.10,        # 10% take profit
    'check_interval': 300,          # Check every 5 minutes
}
```

## File Structure

```
crypto-bot/
├── main.py              # Main trading bot
├── dashboard.py         # Monitoring dashboard
├── setup.py            # Setup and configuration script
├── requirements.txt    # Python dependencies
├── .env.example       # Environment variables template
├── .env               # Your API keys (created during setup)
├── bot_state.json     # Bot state persistence (auto-created)
├── trading_bot.log    # Trading logs (auto-created)
└── README.md          # This file
```

## Safety Features

- **Paper Trading**: Test your strategy without real money by modifying the exchange to sandbox mode
- **Stop Loss**: Automatic position exit at 4% loss
- **Take Profit**: Automatic position exit at 10% gain
- **Position Sizing**: Limits risk to 7.5% of portfolio per trade
- **Rate Limiting**: Respects Kraken's API rate limits
- **Error Recovery**: Handles network issues and API errors gracefully

## Monitoring

The bot creates several files for monitoring:

- `trading_bot.log`: Detailed operation logs
- `bot_state.json`: Current positions and trade history
- `dashboard.py`: Real-time monitoring script

## Important Notes

⚠️ **This bot trades with real money. Start with small amounts and monitor closely.**

- Always test with small amounts first
- Monitor the bot regularly, especially during high volatility
- Ensure you have sufficient funds for trading
- Be aware of Kraken's trading fees
- The strategy works best in trending markets

## Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Check your API keys in `.env`
   - Verify API permissions in Kraken
   - Check internet connection

2. **Insufficient Funds**
   - Ensure you have USD balance in Kraken
   - Check minimum trade amounts

3. **No Trading Activity**
   - Market might not be generating signals
   - Check if RSI and MA conditions are being met
   - Review logs for details

### Support

- Check `trading_bot.log` for detailed error messages
- Run `python dashboard.py` to see current status
- Ensure all dependencies are installed correctly

## License

This project is for educational purposes. Use at your own risk. Cryptocurrency trading involves substantial risk of loss.
