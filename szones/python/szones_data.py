"""Market data for SZones with fallbacks, so the daily routine does not depend on Yahoo alone.

Every source is converted to the Yahoo chart-API JSON shape that szones_v33.load15/loadD
already read, so the rules engine is unchanged. Sources are tried in order; the first
one that returns enough clean bars wins, and the winner is recorded per counter.

  yahoo      query1 then query2.finance.yahoo.com, 15m (60d) + 1d (1y). Free, no key.
  eodhd      eodhd.com official API (paid, ~US$20/month EOD+intraday plan). Needs env
             EODHD_API_KEY. Non-US intraday comes as 5m bars (history from Oct 2020); they
             are resampled here to 15m. Symbols: <code>.KLSE. UNTESTED against live KLSE data.
  tradingview  TradingView's own feed via the unofficial `tvDatafeed` package, vendored and
             reviewed in vendor/tvDatafeed (needs pip websocket-client) (MYX:<name>,
             15m, up to 5000 bars, no login). Free but unofficial: may break or be rate-limited.
             UNTESTED against live MYX data.

  local      reuse <workdir>/data files already on disk (SZ_SOURCES=local for offline replays).

Corporate actions (since 6 Nov 2026): Yahoo does not always adjust its daily history for bonus issues,
splits or capital distributions (CAPITALA 24 Nov 2025: 0.945 -> 0.277; VITROX 2 May 2024: 7.45 -> 3.72).
repair_gaps() finds any overnight gap beyond 40% in a Yahoo/EODHD daily series and checks that day on
TradingView, whose history is adjusted: if TradingView shows no such gap, every earlier bar is scaled by
TradingView's ratio (prices x f, volume / f); if TradingView shows the same move it is real and kept; if
TradingView cannot be reached the gap is kept and flagged 'unverified'. fetch() reports it under 'adjusted'.

Order is set by env SZ_SOURCES (default "yahoo,eodhd,tradingview").
Hosts that must be allowed in a restricted network: query1/query2.finance.yahoo.com,
eodhd.com, data.tradingview.com (websocket) and www.tradingview.com.
"""
import datetime as dt
import json
import os
import subprocess
import sys

MYT = dt.timezone(dt.timedelta(hours=8))
ORDER = [s.strip() for s in os.environ.get('SZ_SOURCES', 'yahoo,eodhd,tradingview').split(',') if s.strip()]
MIN_15M_BARS = 26 * 20          # at least ~20 sessions of 15m bars (26 bars a session incl. lunch-gap bars)
MIN_1D_BARS = 60


def _curl(url, timeout=40):
    r = subprocess.run(['curl', '-sS', '-m', str(timeout), '-A', 'Mozilla/5.0', url],
                       capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError((r.stderr or 'empty response').strip()[:160])
    return r.stdout


def _yahoo_shape(rows, meta):
    """rows: [(unix_ts, o, h, l, c, v)] sorted by time."""
    return {'chart': {'result': [{
        'meta': meta,
        'timestamp': [r[0] for r in rows],
        'indicators': {'quote': [{k: [r[i] for r in rows] for i, k in
                                  enumerate(('open', 'high', 'low', 'close', 'volume'), start=1)}]}}]}}


def _count(doc):
    q = doc['chart']['result'][0]['indicators']['quote'][0]
    return sum(c is not None for c in q['close'])


# ── Yahoo ────────────────────────────────────────────────────────────────────
def yahoo(code, name, interval, long=False):
    rng = '60d' if interval == '15m' else ('5y' if long else '1y')
    last = None
    for host in ('query1', 'query2'):
        try:
            doc = json.loads(_curl(f'https://{host}.finance.yahoo.com/v8/finance/chart/{code}.KL'
                                   f'?interval={interval}&range={rng}'))
            if doc.get('chart', {}).get('result'):
                return doc
            last = RuntimeError(str(doc.get('chart', {}).get('error'))[:160])
        except Exception as e:          # noqa: BLE001 - try the next host
            last = e
    raise last or RuntimeError('no result')


# ── EODHD ────────────────────────────────────────────────────────────────────
def resample_15m(bars5):
    """bars5: [(unix_ts_start, o, h, l, c, v)] 5-minute bars -> 15-minute bars on MYT
    quarter-hours (09:00, 09:15, ...), matching Yahoo/TradingView 15m bar starts."""
    out = {}
    for ts, o, h, l, c, v in sorted(bars5):
        if None in (o, h, l, c):
            continue
        t = dt.datetime.fromtimestamp(ts, MYT)
        key = int(t.replace(minute=t.minute - t.minute % 15, second=0, microsecond=0).timestamp())
        if key not in out:
            out[key] = [key, o, h, l, c, v or 0]
        else:
            b = out[key]
            b[2], b[3], b[4], b[5] = max(b[2], h), min(b[3], l), c, b[5] + (v or 0)
    return [tuple(b) for b in sorted(out.values())]


def eodhd(code, name, interval, long=False):
    key = os.environ.get('EODHD_API_KEY')
    if not key:
        raise RuntimeError('EODHD_API_KEY not set')
    sym = f'{code}.KLSE'
    now = dt.datetime.now(dt.timezone.utc)
    if interval == '15m':
        frm = int((now - dt.timedelta(days=90)).timestamp())
        data = json.loads(_curl(f'https://eodhd.com/api/intraday/{sym}?interval=5m&from={frm}'
                                f'&api_token={key}&fmt=json'))
        rows = resample_15m([(d['timestamp'], d['open'], d['high'], d['low'], d['close'], d.get('volume'))
                             for d in data])
        last5 = max(d['timestamp'] for d in data) if data else 0
        meta = {'shortName': name, 'source': 'eodhd', 'regularMarketTime': last5 + 300}
    else:
        frm = (now - dt.timedelta(days=1900 if long else 370)).date().isoformat()
        data = json.loads(_curl(f'https://eodhd.com/api/eod/{sym}?from={frm}&api_token={key}&fmt=json'))
        rows = [(int(dt.datetime.fromisoformat(d['date']).replace(hour=9, tzinfo=MYT).timestamp()),
                 d['open'], d['high'], d['low'], d['close'], d.get('volume')) for d in data]
        meta = {'shortName': name, 'source': 'eodhd'}
    if not rows:
        raise RuntimeError('no bars')
    return _yahoo_shape(rows, meta)


# ── TradingView (unofficial) ─────────────────────────────────────────────────
def tradingview(code, name, interval, long=False):
    vendor = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor')
    if vendor not in sys.path:
        sys.path.insert(0, vendor)      # reviewed copy, see vendor/tvDatafeed/VENDORED.md
    try:
        from tvDatafeed import TvDatafeed, Interval
    except ImportError as e:
        raise RuntimeError(f'tvDatafeed unavailable ({e}); pip install websocket-client') from e
    tv = TvDatafeed()
    iv = {'5m': Interval.in_5_minute, '15m': Interval.in_15_minute, '30m': Interval.in_30_minute,
          '60m': Interval.in_1_hour, '1d': Interval.in_daily}[interval]
    df = tv.get_hist(symbol=name, exchange='MYX', interval=iv, n_bars=(1300 if long else 300) if interval == '1d' else 5000)
    if df is None or df.empty:
        raise RuntimeError(f'no bars for MYX:{name}')
    rows = []
    for t, r in df.iterrows():
        ts = t.to_pydatetime()
        # tvDatafeed returns naive datetimes in the machine's local zone
        ts = ts.astimezone(dt.timezone.utc) if ts.tzinfo else ts.replace(tzinfo=dt.datetime.now().astimezone().tzinfo)
        rows.append((int(ts.timestamp()), float(r['open']), float(r['high']), float(r['low']),
                     float(r['close']), float(r['volume'])))
    meta = {'shortName': name, 'source': 'tradingview'}
    if interval != '1d':
        meta['regularMarketTime'] = rows[-1][0] + 60 * int(interval[:-1])
    return _yahoo_shape(rows, meta)


SOURCES = {'yahoo': yahoo, 'eodhd': eodhd, 'tradingview': tradingview}

GAP = 0.6       # open below 60% (or above 1/0.6) of the prior close: too big for a normal session


def repair_gaps(doc, code, name):
    """Adjust unadjusted corporate-action gaps in a daily doc against TradingView. Returns (doc, notes)."""
    r = doc['chart']['result'][0]
    q = r['indicators']['quote'][0]
    ts = r['timestamp']
    day = lambda t: dt.datetime.fromtimestamp(t, MYT).date()
    idx = [i for i in range(len(ts)) if q['open'][i] is not None and q['close'][i] is not None]
    gaps = [(i0, i1) for i0, i1 in zip(idx, idx[1:]) if not GAP <= q['open'][i1] / q['close'][i0] <= 1 / GAP]
    if not gaps:
        return doc, []
    try:
        tv = tradingview(code, name, '1d', long=True)['chart']['result'][0]
        tq = tv['indicators']['quote'][0]
        tvd = {day(t): (tq['open'][k], tq['close'][k]) for k, t in enumerate(tv['timestamp']) if tq['close'][k] is not None}
    except Exception as e:              # noqa: BLE001 - keep the data, flag the gap
        return doc, [f'{day(ts[i1])}: gap x{q["open"][i1] / q["close"][i0]:.3f} unverified (TradingView: {str(e)[:60]})'
                     for i0, i1 in gaps]
    notes = []
    for i0, i1 in gaps:
        d0, d1 = day(ts[i0]), day(ts[i1])
        g_src = q['open'][i1] / q['close'][i0]
        if d0 not in tvd or d1 not in tvd:
            notes.append(f'{d1}: gap x{g_src:.3f} unverified (date not in TradingView history)')
            continue
        g_tv = tvd[d1][0] / tvd[d0][1]
        if not GAP <= g_tv <= 1 / GAP:
            notes.append(f'{d1}: gap x{g_src:.3f} kept (TradingView shows the same move, x{g_tv:.3f})')
            continue
        f = g_src / g_tv                # scale every bar before d1 so the day-over-day move matches TradingView
        for k in range(i1):
            for key in ('open', 'high', 'low', 'close'):
                if q[key][k] is not None:
                    q[key][k] = q[key][k] * f
            if q.get('volume') and q['volume'][k] is not None:
                q['volume'][k] = q['volume'][k] / f
        notes.append(f'{d1}: gap x{g_src:.3f} adjusted, earlier bars x{f:.4f} to match TradingView')
    if any('adjusted' in n for n in notes):
        r.setdefault('meta', {})['adjusted'] = notes
    return doc, notes


def fetch(code, name, outdir, mode='15m'):
    """Write <outdir>/<code>_15m.json and _1d.json from the first source that works.
    mode='daily': only ~5 years of daily bars (the weekly bias needs >55 weeks), written as
    _1d.json and _1440m.json (the engine's daily chart file).
    Returns {'used': {interval: source}, 'errors': {...}} plus 'adjusted': {interval: [notes]} when
    repair_gaps() found a corporate-action gap."""
    used, errors, adjusted = {}, {}, {}
    long = mode == 'daily'
    plan = (('1d', 300),) if long else (('15m', MIN_15M_BARS), ('1d', MIN_1D_BARS))
    for iv, need in plan:
        for src in ORDER:
            try:
                if src == 'local':      # replay files already in outdir (offline tests, re-runs)
                    doc = json.load(open(os.path.join(outdir, f'{code}_{iv}.json')))
                else:
                    doc = SOURCES[src](code, name, iv, long=long)
                n = _count(doc)
                if n < need:
                    raise RuntimeError(f'only {n} bars')
                if iv == '1d' and src in ('yahoo', 'eodhd'):
                    doc, notes = repair_gaps(doc, code, name)
                    if notes:
                        adjusted[iv] = notes
                json.dump(doc, open(os.path.join(outdir, f'{code}_{iv}.json'), 'w'))
                if long:
                    json.dump(doc, open(os.path.join(outdir, f'{code}_1440m.json'), 'w'))
                used[iv] = src
                break
            except Exception as e:      # noqa: BLE001 - fall through to the next source
                errors[f'{src}:{iv}'] = str(e)[:160]
        if iv not in used:              # never let a file from an earlier run pass as today's data
            stale = os.path.join(outdir, f'{code}_{iv}.json')
            if os.path.exists(stale):
                os.remove(stale)
    return dict(used=used, errors=errors, **({'adjusted': adjusted} if adjusted else {}))
