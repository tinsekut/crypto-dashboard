#!/usr/bin/env python3
"""SZones daily swing routine for Shariah-compliant Bursa counters (paper record since 2 Nov 2026).

Same settings as the 2016-2026 daily backtest (CLAUDE.md section 3): daily chart, weekly bias,
Bursa ticks, chase guard 0.5 ATR, net TP1 >= 0.25R, TP1 1.0 / TP2 2.0 ATR (or BSL within 6 ATR),
SL 1.8 ATR, runner after TP2, entry at the NEXT day's open after a signal close, gap-aware stops,
0.25% round-trip cost, long only. Backtest: +0.003R per trade, 43% win, TP1 55%, median hold 13 days.

Usage (after 17:00 MYT):
  python3 szones_swing.py --date 2026-11-02 --workdir ../runs/swing_2026-11-02

Writes to the workdir: snapshots.json (SIGNAL = buy at next open, WATCH = could signal at the next
close), trades.json (closed trades, entry in the last 400 days), positions.json (open positions),
sources.json. Aborts without outputs if more than half the universe has no data.
"""
import argparse, json, math, os, sys, datetime as dt, concurrent.futures as cf
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
GO_LIVE = dt.date(2026, 11, 3)          # first session of the forward paper record
ap = argparse.ArgumentParser()
ap.add_argument('--date', required=True)
ap.add_argument('--workdir', default=None)
ap.add_argument('--shariah', default=os.path.join(HERE, '..', 'data', 'shariah_sc_list_2026-05-29.json'))
ap.add_argument('--min-adv', type=float, default=10e6, help='min 20-day avg turnover (RM)')
a = ap.parse_args()
D = dt.date.fromisoformat(a.date)
wd = os.path.abspath(a.workdir or os.path.join(os.getcwd(), f'swing_{a.date}'))
os.makedirs(os.path.join(wd, 'data'), exist_ok=True)
os.environ['SZ_CLOSED'] = '1'
sh = json.load(open(a.shariah))
codes = [c for c, i in sh.items() if i['status'] == 'Y' and c not in ('0138', '7033')]

sys.path.insert(0, HERE)
import szones_data as SD
src = {}
def get(c):
    src[c] = SD.fetch(c, sh[c]['name'], os.path.join(wd, 'data'), mode='daily')
with cf.ThreadPoolExecutor(6) as ex: list(ex.map(get, codes))
json.dump(src, open(os.path.join(wd, 'sources.json'), 'w'), indent=1)
print('Data sources (daily):', dict(Counter(v['used'].get('1d', 'NONE') for v in src.values())))
missing = [c for c, v in src.items() if '1d' not in v['used']]
if len(missing) > len(codes) // 2:
    sys.exit(f'ABORT: no daily data for {len(missing)} of {len(codes)} counters; no outputs written. '
             f'e.g. {src[missing[0]]["errors"]}')

os.chdir(wd)
import szones_v33 as Z
Z.P.update(ticks=True, chase=0.5, minNet=0.25, tp1=1.0, tp2=2.0, tf=1440, daily=True, flat=False,
           nextOpen=True, gapFill=True)
up = lambda x, t: math.ceil(round(x / t, 6)) * t
dn = lambda x, t: math.floor(round(x / t, 6)) * t
G = 'ABC'


def _f(x, nd=6):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


closed, opened, signals, watch, positions, stale = [], [], [], [], [], []
for c in codes:
    if c in missing:
        continue
    try:
        r = Z.run(c)
    except Exception as e:
        print('skip', sh[c]['name'], e); continue
    nm = sh[c]['name']; r['ticker'] = nm
    if r['last'][:10] != str(D):
        stale.append(f"{nm} (last bar {r['last'][:10]})")
    for t in Z.run.trades:
        if (D - t['t'].date()).days <= 400:
            closed.append(dict(ticker=nm, time=f"{t['t']:%Y-%m-%d}", exit_time=f"{t['xt']:%Y-%m-%d}", grade=G[t['grade']],
                               entry=_f(t['e']), sl=_f(t['stop']), tp1=_f(t['t1']), tp2=_f(t['t2']),
                               tp1_hit=bool(t['tp1']), exit=t['how'], exit_px=_f(t['xp']), r=round(float(t['R']), 4)))
    o = Z.run.open
    if o:
        rem = 1 - Z.P['scale'] if o['tp1'] else 1.0
        unreal = o['R'] + rem * (r['close'] - o['e']) / o['risk']
        positions.append(dict(ticker=nm, entry_date=f"{o['t']:%Y-%m-%d}", grade=G[o['grade']], entry=_f(o['e']),
                              stop=_f(o['stop']), tp1=_f(o['t1']), tp2=_f(o['t2']), tp1_hit=o['tp1'], tp2_hit=o['tp2'],
                              close=_f(r['close']), open_r=round(float(unreal), 3)))
    a_ = r['atr15']; tk = Z.tick(r['close'])          # atr15 = ATR of the chart timeframe (daily here)
    snapbase = dict(code=c, ticker=nm, tf='D', time=r['last'][:10], price=_f(r['close']), dir=1,
                    trend=dict(bias=r['zDir'], htf_bull=r['htf'] == 'bull', struct=r['sDir'], discount=r['disc'],
                               squeeze=r['squeeze']),
                    zones=dict(next=r['nextZ'], dist_z1_atr=_f(r['distZ1'], 3),
                               **{f'z{k + 1}': [_f(zb), _f(zt)] for k, (zt, zb) in enumerate(r['zones'])}),
                    smc=dict(maj=r['majDir'], msb=_f(r['msbLvl']), idm=_f(r['idm']), leg_ext=_f(r['legExt']),
                             ext=_f(r['ext']), bsl=_f(r['bsl']), ssl=_f(r['ssl'])),
                    atr=_f(a_), atr_pct=_f(r['atr15pct'], 3), adv_rm=round(r['adv']),
                    stats=dict(n=r['nTr'], avg_r=_f(r['avgR'], 3)), cost_pct=0.25, long_only=True)
    p = Z.run.pend
    if p and not o:
        # indicative levels from today's close; real levels come from tomorrow's open
        e = r['close']; sl = dn(e - 1.8 * p['a'], tk); t1 = max(up(e + 1.0 * p['a'], tk), e + tk)
        signals.append(dict(snapbase, event='SIGNAL', id=f'{nm}-{D}', grade=G[p['grade']], zone=p['zone'],
                            plan=dict(entry='next open', indicative_entry=_f(e), sl=_f(sl), tp1=_f(t1),
                                      tp2=_f(up(e + 2.0 * p['a'], tk)), chase_atr=_f(p['chase'], 3),
                                      net_tp1_r=_f(((t1 - e) - 0.0025 * e) / (e - sl), 3))))
    elif r['zDir'] == 1 and r['nextZ'] and not o and r['adv'] >= a.min_adv and not r['squeeze']:
        zt, zb = r['zones'][r['nextZ'] - 1]
        trig = up(zt, tk); trig = trig + tk if abs(trig - zt) < 1e-9 else trig
        sl = dn(trig - 1.8 * a_, tk); t1 = max(up(trig + 1.0 * a_, tk), trig + tk)
        net = ((t1 - trig) - 0.0025 * trig) / (trig - sl)
        maxE = dn(zt + 0.5 * a_, tk); g = r['plan']['expGrade']
        if g in 'AB' and net >= 0.25 and maxE >= trig - 1e-9 and r['distZ1'] is not None and r['distZ1'] <= 1.0:
            watch.append(dict(snapbase, event='WATCH', id=f'{nm}-{D}', grade=g, reasons=r['plan']['why'],
                              plan=dict(trigger_close=_f(trig), max_close=_f(maxE), entry='open after the signal close',
                                        sl=_f(sl), tp1=_f(t1), net_tp1_r=_f(net, 3))))

closed.sort(key=lambda t: t['time'])
print(f'\n=== {D}: daily swing (paper) ===')
if stale:
    print(f'WARNING: {len(stale)} counters have no bar for {D}: {", ".join(stale[:8])}')
print(f'\nSignals at today\'s close -> buy at the next open ({len(signals)}):')
for s in sorted(signals, key=lambda s: s['grade']):
    q = s['plan']
    print(f"  {s['ticker']:9} grade {s['grade']} zone {s['zone']} close {s['price']}  ~SL {q['sl']}  ~TP1 {q['tp1']}  ~TP2 {q['tp2']}  net TP1 {q['net_tp1_r']:+.2f}R")
print(f'\nOpen positions ({len(positions)}):')
for p in sorted(positions, key=lambda p: p['entry_date']):
    print(f"  {p['ticker']:9} since {p['entry_date']} grade {p['grade']} entry {p['entry']} stop {p['stop']} "
          f"TP1 {p['tp1']}{' (hit)' if p['tp1_hit'] else ''} TP2 {p['tp2']}{' (hit)' if p['tp2_hit'] else ''} close {p['close']} open {p['open_r']:+.2f}R"
          f"{'  [inherited: entered before the forward record]' if p['entry_date'] < str(GO_LIVE) else ''}")
if len(positions) > 3:
    print(f'  NOTE: {len(positions)} open; the manual caps live trading at 3 (the paper record tracks all).')
today_x = [t for t in closed if t['exit_time'] == str(D)]
print(f'\nClosed today ({len(today_x)}):')
for t in today_x:
    print(f"  {t['ticker']:9} entered {t['time']} grade {t['grade']} exit {t['exit']} @ {t['exit_px']}  {t['r']:+.2f}R")
fwd = [t['r'] for t in closed if t['time'] >= str(GO_LIVE)]
yr = [t['r'] for t in closed if (D - dt.date.fromisoformat(t['time'])).days <= 365]
print(f'\nForward paper record since {GO_LIVE}: {len(fwd)} closed trades, total {sum(fwd):+.2f}R'
      + (f', avg {sum(fwd)/len(fwd):+.3f}R' if fwd else ''))
print(f'Last 12 months (rules replayed): {len(yr)} trades, avg {sum(yr)/max(1, len(yr)):+.3f}R  |  backtest 2016-2026: +0.003R')
print(f'\nWatch: could signal at the next close ({len(watch)}):')
for w in sorted(watch, key=lambda w: w['zones']['dist_z1_atr']):
    q = w['plan']
    print(f"  {w['ticker']:9} grade {w['grade']} [{w['reasons']}] close {w['price']} dist {w['zones']['dist_z1_atr']:+.2f}ATR  "
          f"green close {q['trigger_close']}..{q['max_close']}  SL {q['sl']}  TP1 {q['tp1']}  net {q['net_tp1_r']:+.2f}R")
json.dump(signals + watch, open('snapshots.json', 'w'), indent=1)
json.dump(closed, open('trades.json', 'w'), indent=1)
json.dump(positions, open('positions.json', 'w'), indent=1)
print(f'\nWrote {len(signals)} SIGNAL + {len(watch)} WATCH snapshots for Claude review -> {os.path.join(wd, "snapshots.json")}')
