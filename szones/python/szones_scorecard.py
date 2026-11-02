#!/usr/bin/env python3
"""Score Claude's watchlist reviews against what the rules actually did.

Inputs (paths relative to the szones folder):
  reviews/<D>.json      Claude's calls on the snapshots from the close of D
                        [{ticker, call: TAKE|SKIP|WAIT, p_tp1, next_move, key_level, invalidation, reason}, ...]
  runs/run_<N>/trades.json   the rules' trades in the next session N that has a run (handles holidays)

A review counts once its ticker trades in session N. Its first trade that session is the outcome.
Claude helps only if TAKE beats all reviewed trades on average R over 50+ trades,
and its Brier score on TP1 is lower than the base-rate Brier.
"""
import argparse, glob, json, os

ap = argparse.ArgumentParser()
ap.add_argument('--root', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
ROOT = ap.parse_args().root
runs = sorted(os.path.basename(os.path.dirname(t))[4:]
              for t in glob.glob(os.path.join(ROOT, 'runs', 'run_*', 'trades.json')))
rows, pending = [], 0
for f in sorted(glob.glob(os.path.join(ROOT, 'reviews', '*.json'))):
    D = os.path.basename(f)[:-5]
    nxt = [d for d in runs if d > D]
    if not nxt:
        pending += 1
        continue
    tf = os.path.join(ROOT, 'runs', f'run_{nxt[0]}', 'trades.json')
    first = {}
    for t in json.load(open(tf)):
        first.setdefault(t['ticker'], t)
    for rv in json.load(open(f)):
        t = first.get(rv['ticker'])
        if t:
            rows.append(dict(date=nxt[0], ticker=rv['ticker'], call=rv['call'], p=rv.get('p_tp1'),
                             r=t['r'], hit=1 if t['tp1_hit'] else 0))


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
print(f'\n{len(rows)} scored trades; {pending} review day(s) still waiting for their session.')
if len(rows) < 50:
    print('Fewer than 50 trades: too early to judge Claude.')
