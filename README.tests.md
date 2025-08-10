Test suite for CryptoTradingBot

- Location: tests/test_main.py
- Framework: Python unittest (no extra deps)
- What it covers:
  - Indicator computation safety and presence of metrics
  - Entry condition evaluation shape and diagnostics
  - Position sizing returns positive amount
  - Paper trade execution returns an order
  - Trailing stop management closes position when price dips below trail

Run tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Optional with pytest:

```bash
pytest -q
```
