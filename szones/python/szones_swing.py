#!/usr/bin/env python3
"""SZones daily swing routine for Shariah-compliant Bursa counters (paper record since 2 Nov 2026).

Same settings as the 2016-2026 daily backtest (CLAUDE.md section 3), grades A and B by default (A only on 2 Nov): daily chart, weekly bias,
Bursa ticks, chase guard 0.5 ATR, net TP1 >= 0.25R, TP1 1.0 / TP2 2.0 ATR (or BSL within 6 ATR),
SL 1.8 ATR, runner after TP2, entry at the NEXT day's open after a signal close, gap-aware stops,
0.25% round-trip cost, long only. Backtest all grades: +0.003R per trade, 43% win, TP1 55%, median hold
13 days; grades A and B: +0.038R (t 0.85), 45% win, TP1 55%, 569 trades; A only: -0.024R, 212 trades.

Usage (after 17:00 MYT, evening):
  python3 szones_swing.py --date 2026-11-02 --workdir ../runs/swing_2026-11-02
Usage (before 09:00 MYT, pre-open check on the final data of the last session):
  python3 szones_swing.py --date 2026-11-03 --premarket --workdir ../runs/premarket_2026-11-03

Writes to the workdir: snapshots.json (SIGNAL = buy at next open, WATCH = could signal at the next
close), trades.json (closed trades, entry in the last 400 days), positions.json (open positions),
sources.json, summary.json (headline numbers for the brief). Aborts without outputs if more than half the
universe has no data.

--premarket: --date is the day of the coming open. The session analysed is the latest daily bar in the data
(the last close). Nothing trades overnight on a daily chart, so this re-checks the evening plan on final
data: it lists the orders for today's open and what changed against runs/swing_<session>/snapshots.json.
Writes snapshots.json, positions.json, summary.json, sources.json (no trades.json, so the scorecard is
unaffected).
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
ap.add_argument('--min-grade', default='B', choices=['A', 'B', 'C'],
                help='lowest grade traded (tIn: A only on 2 Nov 2026, A and B from 3 Nov; backtest A+B: +0.038R, t 0.85)')
ap.add_argument('--premarket', action='store_true', help='pre-open check before the next open (see above)')
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
           nextOpen=True, gapFill=True, minGrade='ABC'.index(a.min_grade))
up = lambda x, t: math.ceil(round(x / t, 6)) * t
dn = lambda x, t: math.floor(round(x / t, 6)) * t
G = 'ABC'


def _f(x, nd=6):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


closed, opened, signals, watch, positions, lasts = [], [], [], [], [], {}
for c in codes:
    if c in missing:
        continue
    try:
        r = Z.run(c)
    except Exception as e:
        print('skip', sh[c]['name'], e); continue
    nm = sh[c]['name']; r['ticker'] = nm
    lasts[nm] = r['last'][:10]
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
        if g in 'ABC'[:'ABC'.index(a.min_grade) + 1] and net >= 0.25 and maxE >= trig - 1e-9 and r['distZ1'] is not None and r['distZ1'] <= 1.0:
            watch.append(dict(snapbase, event='WATCH', id=f'{nm}-{D}', grade=g, reasons=r['plan']['why'],
                              plan=dict(trigger_close=_f(trig), max_close=_f(maxE), entry='open after the signal close',
                                        sl=_f(sl), tp1=_f(t1), net_tp1_r=_f(net, 3))))

closed.sort(key=lambda t: t['time'])
OPEN_DAY = D                             # the session whose open the plan is for
if a.premarket:
    D = dt.date.fromisoformat(max(lasts.values(), key=lambda d: (list(lasts.values()).count(d), d)))
    if D >= OPEN_DAY:
        sys.exit(f'ABORT: latest bar {D} is not before {OPEN_DAY}; run --premarket before the open.')
    for s in signals + watch:
        s['id'] = f"{s['ticker']}-{D}"
stale = [f'{nm} (last bar {d})' for nm, d in lasts.items() if d != str(D)]
edition = 'Pre-open' if a.premarket else 'Evening'
print(f'\n=== {edition} {"check for the open of " + str(OPEN_DAY) + ", on the close of " if a.premarket else ""}'
      f'{D}: daily swing (paper), grades traded: {"ABC"[:"ABC".index(a.min_grade) + 1]} ===')
if stale:
    print(f'WARNING: {len(stale)} counters have no bar for {D}: {", ".join(stale[:8])}')
print(f'\nSignals at the close of {D} -> buy at the open of {OPEN_DAY if a.premarket else "the next session"} ({len(signals)}):')
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
BT = {'A': '-0.024R (212 trades)', 'B': '+0.038R (569 trades)', 'C': '+0.003R (938 trades)'}[a.min_grade]
print(f'Last 12 months (rules replayed): {len(yr)} trades, avg {sum(yr)/max(1, len(yr)):+.3f}R  |  backtest 2016-2026: {BT}')
print(f'\nWatch: could signal at the next close ({len(watch)}):')
for w in sorted(watch, key=lambda w: w['zones']['dist_z1_atr']):
    q = w['plan']
    print(f"  {w['ticker']:9} grade {w['grade']} [{w['reasons']}] close {w['price']} dist {w['zones']['dist_z1_atr']:+.2f}ATR  "
          f"green close {q['trigger_close']}..{q['max_close']}  SL {q['sl']}  TP1 {q['tp1']}  net {q['net_tp1_r']:+.2f}R")
changes = None
if a.premarket:                          # compare with the evening plan for the same close
    ev = os.path.join(wd, '..', f'swing_{D}', 'snapshots.json')
    if os.path.exists(ev):
        old = {(s['ticker'], s['event']) for s in json.load(open(ev))}
        new = {(s['ticker'], s['event']) for s in signals + watch}
        changes = dict(added=sorted(f'{t} {e}' for t, e in new - old), dropped=sorted(f'{t} {e}' for t, e in old - new))
        print(f'\nAgainst the evening plan of {D}: added {changes["added"] or "none"}, dropped {changes["dropped"] or "none"}')
    else:
        print(f'\nNo evening run for {D} found ({ev}); treat this as the plan for {OPEN_DAY}.')
src_used = Counter(v['used'].get('1d', 'NONE') for v in src.values())
mt = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
json.dump(dict(edition=edition, session=str(D), for_open=str(OPEN_DAY) if a.premarket else None,
               run_time=f'{mt:%Y-%m-%d %H:%M} MYT', min_grade=a.min_grade,
               signals_next_open=len(signals), watch=len(watch), open_positions=len(positions), position_cap=3,
               closed_on_session=len(today_x), closed_on_session_r=round(sum(t['r'] for t in today_x), 3),
               forward_closed=len(fwd), forward_total_r=round(sum(fwd), 3), go_live=str(GO_LIVE),
               replay_12m_trades=len(yr), replay_12m_avg_r=round(sum(yr) / max(1, len(yr)), 3),
               data_source=src_used.most_common(1)[0][0], counters_with_data=len(codes) - len(missing),
               counters_total=len(codes), stale=stale, changes=changes),
          open('summary.json', 'w'), indent=1)
json.dump(signals + watch, open('snapshots.json', 'w'), indent=1)
if not a.premarket:
    json.dump(closed, open('trades.json', 'w'), indent=1)
json.dump(positions, open('positions.json', 'w'), indent=1)
print(f'\nWrote {len(signals)} SIGNAL + {len(watch)} WATCH snapshots for Claude review -> {os.path.join(wd, "snapshots.json")}')
