#!/usr/bin/env python3
"""SZones mid-day check: validate the pre-open plan against the morning session (Bursa lunch break).

  python3 szones_midday.py --date 2026-11-06 --workdir ../runs/midday_2026-11-06

Run in the lunch break (12:30-14:30 MYT). Reads the plan written before the open
(runs/premarket_<date>/, or runs/swing_<last close>/ if the pre-open run is missing) and Claude's calls
(reviews/<last close>.json), fetches today's 15-minute bars for those counters only (Yahoo, then the
fallback chain; about 15 minutes delayed) and checks the morning session 09:00-12:30:

  SIGNAL   bought at today's open: actual open vs the indicative entry, the stop and TP1 re-based on the
           actual open (1.8 / 1.0 ATR, Bursa ticks), and whether the morning already touched either.
  POSITION whether the morning low reached the stop or the high reached TP1/TP2, and open R at the
           last price.
  WATCH    where the last price sits against the buy-close range, and whether the day is green so far
           (a signal still needs a green DAILY close inside the range; nothing is decided at mid-day).

Nothing here changes the paper record: exits and signals are booked by the evening run on the daily
bar. Writes midday.json (rows + summary) and sources.json to the workdir, and brief/midday.json for the
SZones Brief dashboard.
"""
import argparse, concurrent.futures as cf, datetime as dt, glob, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, HERE)
import szones_data as SD                                   # noqa: E402
import szones_v33 as Z                                     # noqa: E402  (tick table, settings)

ap = argparse.ArgumentParser()
ap.add_argument('--date', required=True, help='trading day (MYT)')
ap.add_argument('--workdir', default=None)
ap.add_argument('--plan-dir', default=None, help='override the plan folder (tests)')
ap.add_argument('--shariah', default=os.path.join(ROOT, 'data', 'shariah_sc_list_2026-05-29.json'))
a = ap.parse_args()
T = dt.date.fromisoformat(a.date)
wd = os.path.abspath(a.workdir or os.path.join(ROOT, 'runs', f'midday_{T}'))
os.makedirs(os.path.join(wd, 'data'), exist_ok=True)
MYT = dt.timezone(dt.timedelta(hours=8))
SL_ATR, TP1_ATR, SCALE = 1.8, 1.0, Z.P['scale']
up = lambda x, t: math.ceil(round(x / t, 6)) * t
dn = lambda x, t: math.floor(round(x / t, 6)) * t

# ── the plan made before the open ─────────────────────────────────────────────
pre = os.path.join(ROOT, 'runs', f'premarket_{T}')
if a.plan_dir:
    plan_dir, plan_from = os.path.abspath(a.plan_dir), 'test plan'
elif os.path.exists(os.path.join(pre, 'summary.json')):
    plan_dir, plan_from = pre, 'pre-open check'
else:
    ev = sorted(d for d in glob.glob(os.path.join(ROOT, 'runs', 'swing_*')) if os.path.basename(d)[6:] < str(T))
    if not ev:
        sys.exit(f'ABORT: no pre-open or evening plan found for {T}.')
    plan_dir, plan_from = ev[-1], 'evening run (no pre-open run today)'
S = json.load(open(os.path.join(plan_dir, 'summary.json')))
session = S['session']
snaps = json.load(open(os.path.join(plan_dir, 'snapshots.json')))
positions = json.load(open(os.path.join(plan_dir, 'positions.json')))
rf = os.path.join(ROOT, 'reviews', f'{session}.json')
calls = {(r['ticker'], r['event']): r for r in json.load(open(rf))} if os.path.exists(rf) else {}

sh = json.load(open(a.shariah))
code_of = {v['name']: c for c, v in sh.items()}
names = sorted({p['ticker'] for p in positions} | {s['ticker'] for s in snaps})
if not names:
    print(f'Nothing to check: no positions, signals or watch-list setups in the plan for {T}.')

# ── today's 15-minute bars ────────────────────────────────────────────────────
src = {}
def get(nm):
    src[nm] = SD.fetch(code_of[nm], nm, os.path.join(wd, 'data'), mode='15m')
with cf.ThreadPoolExecutor(6) as ex:
    list(ex.map(get, [n for n in names if n in code_of]))
json.dump(src, open(os.path.join(wd, 'sources.json'), 'w'), indent=1)


def morning(nm):
    """Today's bars from the 15m file: open, high, low, last, last bar time, prior close (last bar before today)."""
    f = os.path.join(wd, 'data', f'{code_of[nm]}_15m.json')
    if not os.path.exists(f):
        return None
    r = json.load(open(f))['chart']['result'][0]
    q = r['indicators']['quote'][0]
    snap = lambda v: round(round(v / Z.tick(v)) * Z.tick(v), 4)      # Yahoo intraday prices carry float32 noise
    bars = [(dt.datetime.fromtimestamp(t, MYT), *(snap(q[k][i]) for k in ('open', 'high', 'low', 'close')))
            for i, t in enumerate(r['timestamp']) if None not in (q['open'][i], q['high'][i], q['low'][i], q['close'][i])]
    today = [b for b in bars if b[0].date() == T and b[0].time() < dt.time(12, 30)]
    before = [b for b in bars if b[0].date() < T]
    if not today:
        return None
    return dict(open=today[0][1], high=max(b[2] for b in today), low=min(b[3] for b in today), last=today[-1][4],
                last_time=min((today[-1][0] + dt.timedelta(minutes=15)).strftime('%H:%M'), '12:30'), bars=len(today),
                prev_close=before[-1][4] if before else None)


rows, missing = [], []
for nm in names:
    m = morning(nm) if nm in code_of else None
    if not m:
        missing.append(nm)
        continue
    pc = m['prev_close']
    base = dict(ticker=nm, open=m['open'], high=m['high'], low=m['low'], last=m['last'], last_time=m['last_time'],
                prev_close=pc, day_pct=round((m['last'] / pc - 1) * 100, 2) if pc else None, green=m['last'] > m['open'])
    for p in [p for p in positions if p['ticker'] == nm]:
        risk = p['entry'] - p['stop']
        rem = 1 - SCALE if p['tp1_hit'] else 1.0
        banked = SCALE * (p['tp1'] - p['entry']) / risk if p['tp1_hit'] else 0.0
        open_r = banked + rem * (m['last'] - p['entry']) / risk
        if m['low'] <= p['stop']:
            fill = min(m['open'], p['stop'])                      # gap-aware, as the engine fills stops
            open_r = banked + rem * (fill - p['entry']) / risk
            st, kind = 'Stop touched', 'bad'
            note = (f"Morning low {m['low']:g} reached the {p['stop']:g} stop" + (f" (opened below it at {m['open']:g})" if m['open'] < p['stop'] else '')
                    + f"; exit at {fill:g} = {open_r:+.2f}R in total, booked by the evening run.")
        elif not p['tp2_hit'] and m['high'] >= p['tp2']:
            st, kind, note = 'TP2 reached', 'ok', f"Morning high {m['high']:g} reached TP2 {p['tp2']:g}; the runner continues."
        elif not p['tp1_hit'] and m['high'] >= p['tp1']:
            st, kind, note = 'TP1 reached', 'ok', f"Morning high {m['high']:g} reached TP1 {p['tp1']:g}; half is banked at the close."
        else:
            room = (m['low'] - p['stop']) / risk
            st, kind = 'Holding', ('warn' if room < 0.25 else 'neutral')
            note = f"Lowest point stayed {room:.2f}R above the stop." + (' Close to the stop.' if room < 0.25 else '')
        rows.append(dict(base, kind='Position', grade=p['grade'], call=None, entry=p['entry'], stop=p['stop'], tp1=p['tp1'],
                         tp2=p['tp2'], lo=None, hi=None, status=st, status_kind=kind, note=note, open_r=round(open_r, 3),
                         since=p['entry_date']))
    for s in [s for s in snaps if s['ticker'] == nm]:
        q, c, atr = s['plan'], calls.get((nm, s['event']), {}), s['atr']
        tk = Z.tick(m['open'])
        if s['event'] == 'SIGNAL':
            e = m['open']; sl = dn(e - SL_ATR * atr, tk); t1 = max(up(e + TP1_ATR * atr, tk), e + tk)
            gap_atr = (e - q['indicative_entry']) / atr
            if m['low'] <= sl:
                st, kind = 'Stop touched', 'bad'
            elif m['high'] >= t1:
                st, kind = 'TP1 reached', 'ok'
            else:
                st, kind = 'Bought at open', 'neutral'
            note = (f"Filled at the {e:g} open ({'+' if gap_atr >= 0 else '−'}{abs(gap_atr):.2f} ATR vs the {q['indicative_entry']:g} plan). "
                    f"Stop {sl:g}, TP1 {t1:g} re-based on the open." + (' Gapped up more than 0.5 ATR: a worse entry than planned.' if gap_atr > 0.5 else ''))
            rows.append(dict(base, kind='Signal', grade=s['grade'], call=c.get('call'), entry=e, stop=sl, tp1=t1, tp2=None,
                             lo=None, hi=None, status=st, status_kind=kind, note=note, open_r=round((m['last'] - e) / (e - sl), 3)))
        else:
            lo, hi = q['trigger_close'], q['max_close']
            if lo - 1e-9 <= m['last'] <= hi + 1e-9:
                st, kind = ('On track' if base['green'] else 'In range, red so far'), ('ok' if base['green'] else 'warn')
                note = f"Last {m['last']:g} is inside {lo:g}–{hi:g}" + (' and above the open: a green close here would signal.' if base['green'] else ', but below the open: it needs to finish green.')
            elif m['last'] > hi:
                st, kind = 'Above range', 'neutral'
                note = f"Last {m['last']:g} is {(m['last'] / hi - 1) * 100:.1f}% above the {hi:g} top; it must come back into range by the close."
            else:
                st, kind = 'Below range', 'neutral'
                note = f"Last {m['last']:g} is {(1 - m['last'] / lo) * 100:.1f}% below the {lo:g} bottom; it must climb back into range by the close."
            if m['low'] <= q['sl']:
                note += f" The morning low already traded below the planned {q['sl']:g} stop."
            rows.append(dict(base, kind='Watch', grade=s['grade'], call=c.get('call'), entry=None, stop=q['sl'], tp1=q['tp1'],
                             tp2=None, lo=lo, hi=hi, status=st, status_kind=kind, note=note, open_r=None))

if names and len(missing) == len(names):
    if all(src.get(n, {}).get('used', {}).get('15m') for n in names if n in code_of):
        print(f'No bars for {T} before 12:30 on any counter: no session today (Bursa holiday?) or the data is not in yet.')
    else:
        errs = next((v['errors'] for v in src.values() if v.get('errors')), {})
        sys.exit(f'ABORT: no 15-minute data for any counter. e.g. {errs}')

order = {'Signal': 0, 'Position': 1, 'Watch': 2}
rows.sort(key=lambda r: (order[r['kind']], r['ticker']))
now = dt.datetime.now(MYT)
meta = dict(date=str(T), session=session, plan_from=plan_from, run_time=f'{now:%Y-%m-%d %H:%M} MYT',
            as_of=max((r['last_time'] for r in rows), default=None), missing=missing,
            data_source=sorted({v['used'].get('15m', 'NONE') for v in src.values()}))
json.dump(dict(meta=meta, rows=rows), open(os.path.join(wd, 'midday.json'), 'w'), indent=1)
os.makedirs(os.path.join(wd, 'brief'), exist_ok=True)
json.dump([dict(r, as_of=meta['as_of'], date=str(T), plan_from=plan_from) for r in rows],
          open(os.path.join(wd, 'brief', 'midday.json'), 'w'), indent=1)

print(f'=== Mid-day check {T}, morning session to {meta["as_of"]} MYT, against the {plan_from} (close of {session}) ===')
print('Data:', ', '.join(meta['data_source']), '(about 15 minutes delayed)' + (f' | no bars today: {", ".join(missing)}' if missing else ''))
for k in ('Signal', 'Position', 'Watch'):
    rs = [r for r in rows if r['kind'] == k]
    if not rs:
        continue
    print(f"\n{dict(Signal='Bought at the open', Position='Open positions', Watch='Watch list')[k]} ({len(rs)}):")
    for r in rs:
        extra = f"  open {r['open_r']:+.2f}R" if r['open_r'] is not None else ''
        call = f"  [Claude {r['call']}]" if r['call'] else ''
        print(f"  {r['ticker']:9} {r['status']:20} O {r['open']:g} H {r['high']:g} L {r['low']:g} last {r['last']:g}"
              f" ({r['day_pct']:+.2f}%){extra}{call}\n             {r['note']}")
print(f'\nWrote {len(rows)} rows -> {os.path.join(wd, "midday.json")}. Provisional: the evening run books signals and exits on the daily close.')
