# Module 0 — Orientation

This module is short on purpose. By the end of it you will:

- Know **how to read** the rest of the course efficiently.
- Have a working **environment**, with the data adapter pulling real bars from Yahoo.
- Understand the **conventions** every chapter follows.

Pages in this module:

1. **[How to read this course](how-to-read.md)** — the meta-instructions.
2. **[Environment setup](setup.md)** — Python version, virtualenv, install, smoke test.
3. **[The universal data adapter](data-adapter.md)** — one seam, many vendors.
4. **[Course conventions](conventions.md)** — code style, math notation, the "production vs frontier" markers.

If you're impatient, the entire first-time path is:

```bash
git clone https://github.com/billyribeiro-ux/python-engine.git
cd python-engine
python -m venv .venv && source .venv/bin/activate
pip install -e ".[core,docs,dev]"
python -c "from engine.data import YFinanceFeed; print(YFinanceFeed().bars('SPY','2024-01-02','2024-01-15').tail())"
```

If that prints five rows of OHLCV data, you're set. Continue to **[How to read this course](how-to-read.md)**.
