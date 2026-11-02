# Vendored: tvDatafeed 2.1.0

- Upstream: https://github.com/rongardF/tvdatafeed, commit e6f6aaa (2022-09-13), MIT licence (LICENSE).
- Vendored 2 Nov 2026 because `pip install git+...` fails to build (old setup.py) and scheduled
  runs should not install code from GitHub. Source reviewed: it only connects to TradingView
  (`wss://data.tradingview.com`, `symbol-search.tradingview.com`, sign-in on `www.tradingview.com`
  only when a username/password is passed; SZones never passes one).
- One patch in `main.py`: the websocket Origin header is passed as `header=` with a dict.
  Upstream passed `headers=` with a JSON string, which websocket-client 1.x ignores.
- Needs `websocket-client` from PyPI (`pip install websocket-client`); pandas, requests and
  python-dateutil are already required by SZones' environment.
- Unofficial TradingView access: may break or be rate-limited at any time.
