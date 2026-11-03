#!/usr/bin/env python3
"""Build the data files of the SZones Brief dashboard from one run.

  python3 szones_brief.py --run ../runs/swing_2026-11-03                # evening, after reviews/2026-11-03.json
  python3 szones_brief.py --run ../runs/premarket_2026-11-04            # pre-open check

Reads summary.json, positions.json and snapshots.json from the run folder, plus Claude's calls from
reviews/<session>.json (session = the close the run analysed). Writes <run>/brief/run_summary.json,
positions.json and watchlist.json, the three files the dashboard (CLAUDE.md section 7) reads. The
timeframe and grade test files are fixed backtests and are not rebuilt.
"""
import argparse, json, os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
A = ap.parse_args()
run = os.path.abspath(A.run)
S = json.load(open(os.path.join(run, 'summary.json')))
pos = json.load(open(os.path.join(run, 'positions.json')))
snaps = json.load(open(os.path.join(run, 'snapshots.json')))
rf = os.path.join(ROOT, 'reviews', f"{S['session']}.json")
calls = {(r['ticker'], r['event']): r for r in json.load(open(rf))} if os.path.exists(rf) else {}
GO = S['go_live']
# daily backtest 2016-2026 by lowest grade traded: avg R, trades, TP1 rate, label (CLAUDE.md section 3)
BT = {'A': (-0.024, 212, 0.47, 'A only'), 'B': (0.038, 569, 0.55, 'A and B'), 'C': (0.003, 938, 0.55, 'A, B and C')}

summary = [dict(
    edition=S['edition'], date=S['session'], for_open=S['for_open'] or 'next open', run_time=S['run_time'],
    signals_next_open=S['signals_next_open'], open_positions=S['open_positions'], position_cap=S['position_cap'],
    closed_today=S['closed_on_session'], forward_closed=S['forward_closed'], forward_total_r=S['forward_total_r'],
    replay_12m_trades=S['replay_12m_trades'], replay_12m_avg_r=S['replay_12m_avg_r'],
    grades=BT[S['min_grade']][3], backtest_avg_r=BT[S['min_grade']][0], backtest_trades=BT[S['min_grade']][1],
    base_tp1_rate=BT[S['min_grade']][2],
    data_source={'yahoo': 'Yahoo', 'eodhd': 'EODHD', 'tradingview': 'TradingView'}.get(S['data_source'], S['data_source']),
    counters_with_data=S['counters_with_data'], counters_total=S['counters_total'],
    stale=len(S['stale']), reviewed=sum(1 for s in snaps if (s['ticker'], s['event']) in calls))]

positions = [dict(ticker=p['ticker'], since=p['entry_date'], grade=p['grade'], entry=p['entry'], stop=p['stop'],
                  tp1=p['tp1'], tp2=p['tp2'], tp1_hit=p['tp1_hit'], tp2_hit=p['tp2_hit'], close=p['close'],
                  open_r=p['open_r'], inherited=p['entry_date'] < GO)
             for p in sorted(pos, key=lambda p: p['entry_date'])]

watch = []
for s in snaps:
    q, c = s['plan'], calls.get((s['ticker'], s['event']), {})
    sig = s['event'] == 'SIGNAL'
    watch.append(dict(
        ticker=s['ticker'], event=s['event'], grade=s['grade'], why=s.get('reasons', f"zone {s.get('zone')}"),
        close=s['price'], dist_z1_atr=s['zones']['dist_z1_atr'],
        trigger_close=q['indicative_entry'] if sig else q['trigger_close'], max_close=None if sig else q['max_close'],
        sl=q['sl'], tp1=q['tp1'], net_tp1_r=q['net_tp1_r'],
        call=c.get('call', 'not reviewed'), p_tp1=c.get('p_tp1'),
        reason=('BUY AT THE OPEN. ' if sig else '') + c.get('reason', 'No Claude review for this setup yet.')))
watch.sort(key=lambda w: (w['event'] != 'SIGNAL', -(w['dist_z1_atr'] or 0)))

out = os.path.join(run, 'brief')
os.makedirs(out, exist_ok=True)
for name, rows in (('run_summary', summary), ('positions', positions), ('watchlist', watch)):
    json.dump(rows, open(os.path.join(out, f'{name}.json'), 'w'), indent=1)
print(f"Brief files for the {S['edition'].lower()} run on the close of {S['session']}: {out} "
      f"({len(positions)} positions, {len(watch)} setups, {summary[0]['reviewed']} reviewed)")
