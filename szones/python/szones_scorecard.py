#!/usr/bin/env python3
"""Score Claude's reviews against what the rules actually did.

Inputs (paths relative to the szones folder):
  reviews/<D>.json   Claude's calls on the snapshots from the close of D
                     [{ticker, call: TAKE|SKIP|WAIT, p_tp1, next_move, key_level, invalidation, reason}, ...]
  runs/*/trades.json closed trades written by each run ({ticker, time = entry, tp1_hit, r, ...}).
                     All runs are merged (deduplicated by ticker + entry time), so multi-day swing
                     trades are scored once they close, whichever later run saw the exit.

A review counts once its ticker has a CLOSED trade entered within WINDOW calendar days after D
(daily swing: SIGNAL = next open, WATCH = the open after the next signal close; 5 days covers
weekends). The first such trade is the outcome. Reviews whose window has not passed, or whose trade
is still open, are pending. Claude helps only if TAKE beats all reviewed trades on average R over
50+ trades, and its Brier score on TP1 is lower than the base-rate Brier.
"""
import argparse, datetime as dt, glob, json, os

ap = argparse.ArgumentParser()
ap.add_argument('--root', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
ap.add_argument('--window', type=int, default=5, help='calendar days after the review to look for an entry')
A = ap.parse_args()
ROOT = A.root

trades, last_seen = {}, None
for f in sorted(glob.glob(os.path.join(ROOT, 'runs', '*', 'trades.json'))):
    run_date = os.path.basename(os.path.dirname(f)).split('_')[-1]
    last_seen = max(last_seen or run_date, run_date)
    for t in json.load(open(f)):
        trades[(t['ticker'], t['time'])] = t
open_now = []                       # open positions in the most recent run: their reviews stay pending
runs_with_pos = sorted(glob.glob(os.path.join(ROOT, 'runs', '*', 'positions.json')),
                       key=lambda f: os.path.basename(os.path.dirname(f)).split('_')[-1])
if runs_with_pos:
    open_now = json.load(open(runs_with_pos[-1]))
by_ticker = {}
for (tk, tm), t in sorted(trades.items(), key=lambda kv: kv[0][1]):
    by_ticker.setdefault(tk, []).append(t)

rows, pending, no_trade = [], 0, 0
for f in sorted(glob.glob(os.path.join(ROOT, 'reviews', '*.json'))):
    D = dt.date.fromisoformat(os.path.basename(f)[:-5])
    end = D + dt.timedelta(days=A.window)
    for rv in json.load(open(f)):
        hits = [t for t in by_ticker.get(rv['ticker'], [])
                if D < dt.date.fromisoformat(t['time'][:10]) <= end]
        if hits:
            t = hits[0]
            rows.append(dict(date=str(D), ticker=rv['ticker'], call=rv['call'], p=rv.get('p_tp1'),
                             r=t['r'], hit=1 if t['tp1_hit'] else 0))
        elif (last_seen is None or dt.date.fromisoformat(last_seen) <= end
              or any(p['ticker'] == rv['ticker'] and D < dt.date.fromisoformat(p['entry_date']) <= end for p in open_now)):
            pending += 1           # window still open, or the trade has not closed yet
        else:
            no_trade += 1          # no entry in the window: nothing to score


def grp(name, xs):
    if not xs:
        return f'{name:14} {0:>6}'
    R = [x['r'] for x in xs]
    hit = [x['hit'] for x in xs]
    base = sum(hit) / len(hit)
    ps = [x for x in xs if isinstance(x['p'], (int, float))]
    brier = sum((x['p'] - x['hit']) ** 2 for x in ps) / len(ps) if ps else float('nan')
    brier0 = sum((base - h) ** 2 for h in hit) / len(hit)
    return (f'{name:14} {len(xs):>6} {sum(R) / len(R):>+8.3f} {sum(r > 0 for r in R) / len(R):>7.0%} '
            f'{sum(hit) / len(hit):>7.0%} {brier:>9.3f} {brier0:>9.3f}')


print(f'{"Group":14} {"Trades":>6} {"Avg R":>8} {"Win":>7} {"TP1":>7} {"Brier":>9} {"Base":>9}')
print(grp('All reviewed', rows))
for c in ('TAKE', 'SKIP', 'WAIT'):
    print(grp(f'Claude {c}', [x for x in rows if x['call'] == c]))
print(f'\n{len(rows)} scored; {pending} pending (window open or trade still running); {no_trade} reviewed but never triggered.')
if len(rows) < 50:
    print('Fewer than 50 trades: too early to judge Claude.')
