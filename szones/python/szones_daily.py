#!/usr/bin/env python3
"""SZones v3.3.1 daily routine for Shariah-compliant Bursa counters (Bursa intraday preset, 15m).

Usage:
  python3 szones_daily.py --date 2026-11-02            # during the session or pre-market
  python3 szones_daily.py --date 2026-11-02 --closed   # after 17:00: keep the 16:45 bar, flatten trades

Steps: download Yahoo 15m (60d) + 1d (1y) data into <workdir>/data, run the v3.3 rules
(ticks, chase 0.5 ATR, net TP1 >= 0.25R, TP1 1.0 / TP2 2.0 ATR, long only, 0.25% cost),
print the day's signals with R after costs, the rolling 60-session average, and the next watchlist.
"""
import argparse, json, math, os, subprocess, sys, datetime as dt, concurrent.futures as cf
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--date', required=True)
ap.add_argument('--closed', action='store_true')
ap.add_argument('--workdir', default=None)
ap.add_argument('--shariah', default=os.path.join(HERE, '..', 'data', 'shariah_sc_list_2026-05-29.json'))
ap.add_argument('--min-adv', type=float, default=10e6, help='min 20-day avg turnover (RM) for the watchlist')
a = ap.parse_args()
D = dt.date.fromisoformat(a.date)
wd = a.workdir or os.path.join(os.getcwd(), f'run_{a.date}')
os.makedirs(os.path.join(wd, 'data'), exist_ok=True)
if a.closed: os.environ['SZ_CLOSED'] = '1'
sh = json.load(open(a.shariah))
codes = [c for c, i in sh.items() if i['status'] == 'Y' and c not in ('0138', '7033')]   # MYEG/DNEX: bad Yahoo data

def get(c):
    for iv, rg in (('15m', '60d'), ('1d', '1y')):
        subprocess.run(['curl', '-sS', '-m', '40', '-A', 'Mozilla/5.0',
                        f'https://query1.finance.yahoo.com/v8/finance/chart/{c}.KL?interval={iv}&range={rg}',
                        '-o', os.path.join(wd, 'data', f'{c}_{iv}.json')])
with cf.ThreadPoolExecutor(8) as ex: list(ex.map(get, codes))

os.chdir(wd); sys.path.insert(0, HERE)
import szones_v33 as Z
Z.P.update(ticks=True, chase=0.5, minNet=0.25, tp1=1.0, tp2=2.0)
up = lambda x, t: math.ceil(round(x / t, 6)) * t
dn = lambda x, t: math.floor(round(x / t, 6)) * t
today, snap, allR = [], [], []
for c in codes:
    try: r = Z.run(c)
    except Exception as e: print('skip', c, e); continue
    r['ticker'] = sh[c]['name']
    for t in Z.run.trades:
        allR.append(t['R'])
        if t['t'].date() == D:
            today.append((r['ticker'], t))
    a15 = r['atr15']; tk = Z.tick(r['close'])
    if r['zDir'] == 1 and r['nextZ']:
        zt, zb = r['zones'][r['nextZ'] - 1]
        trig = up(zt, tk); trig = trig + tk if abs(trig - zt) < 1e-9 else trig
        sl = dn(trig - 1.8 * a15, tk); t1 = max(up(trig + 1.0 * a15, tk), trig + tk)
        b = r['bsl']; t2 = b if (b and b > t1 and b - trig <= 6 * a15) else up(trig + 2.0 * a15, tk)
        r['p33'] = dict(trig=trig, maxE=dn(zt + 0.5 * a15, tk), sl=sl, t1=t1, t2=t2,
                        net=((t1 - trig) - 0.0025 * trig) / (trig - sl))
    snap.append(r)

print(f'\n=== Signals on {D} ({len(today)}) ===')
for nm, t in sorted(today, key=lambda x: x[1]['t']):
    print(f"{nm:9} {t['t']:%H:%M} grade {'ABC'[t['grade']]} entry {t['e']:.3f} exit={t.get('how')} tp1={t['tp1']} tp2={t['tp2']} R={t['R']:+.2f}")
print('Day total R:', round(sum(t['R'] for _, t in today), 2))
print(f'Rolling 60-session: {len(allR)} trades, avg {sum(allR)/max(1,len(allR)):+.3f}R')

print('\n=== Next watchlist (A/B, valid trigger, net TP1 >= 0.25R, ADV >= RM%.0fm) ===' % (a.min_adv / 1e6))
for r in sorted([r for r in snap if r.get('p33') and r['adv'] >= a.min_adv and not r['squeeze']], key=lambda r: r['distZ1']):
    p = r['p33']; g = r['plan']['expGrade']
    ok = p['net'] >= 0.25 and p['maxE'] >= p['trig'] - 1e-9 and g in 'AB'
    print(f"{'OK  ' if ok else 'skip'} {r['ticker']:9} close {r['close']:.3f} dist {r['distZ1']:+.2f}ATR grade {g} [{r['plan']['why']}] "
          f"trig {p['trig']:.3f}..{p['maxE']:.3f} SL {p['sl']:.3f} TP1 {p['t1']:.3f} TP2 {p['t2']:.3f} net {p['net']:+.2f}R")
json.dump(dict(today=[(n, {k: str(v) for k, v in t.items()}) for n, t in today], snap=snap), open('out.json', 'w'), default=str, indent=1)

# ── Claude review inputs (replaces the TradingView webhook bridge) ──────────
# snapshots.json: one WATCH snapshot per OK watchlist counter, same fields as the
# v3.3.2 Pine payload. Claude reviews these after the close, before the next session.
# trades.json: the day's trades, used to score those reviews the next evening.
def _f(x, nd=6):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)

def snapshot(r):
    p, pl = r['p33'], r['plan']
    zones = {f'z{k + 1}': [_f(zb), _f(zt)] for k, (zt, zb) in enumerate(r['zones'])}
    by = {g: [n, _f(m, 3)] for g, (n, m) in r['byGrade'].items()}
    poi = r['poi']
    return dict(
        v='3.3-py', event='WATCH', id=f"{r['ticker']}-{D}", ticker=r['ticker'], code=r['code'], tf='15',
        time=r['last'], price=_f(r['close']), dir=1, grade=pl['expGrade'], reasons=pl['why'],
        trend=dict(bias=r['zDir'], htf_bull=r['htf'] == 'bull', struct=r['sDir'], discount=r['disc'],
                   squeeze=r['squeeze'], daily_up=r['dailyUp'], daily_dn=r['dailyDn']),
        zones=dict(next=r['nextZ'], dist_z1_atr=_f(r['distZ1'], 3), **zones),
        plan=dict(trigger=_f(p['trig']), max_entry=_f(p['maxE']), sl=_f(p['sl']), tp1=_f(p['t1']),
                  tp2=_f(p['t2']), net_tp1_r=_f(p['net'], 3)),
        smc=dict(maj=r['majDir'], msb=_f(r['msbLvl']), poi=[_f(poi['bot']), _f(poi['top'])] if poi else None,
                 idm=_f(r['idm']), leg_ext=_f(r['legExt']), ext=_f(r['ext']), bsl=_f(r['bsl']), ssl=_f(r['ssl'])),
        levels=dict(pdh=_f(r['pdh']), pdl=_f(r['pdl'])), atr=_f(r['atr15']), atr_pct=_f(r['atr15pct'], 3),
        adv_rm=round(r['adv']), stats=dict(n=r['nTr'], avg_r=_f(r['avgR'], 3), **by),
        cost_pct=0.25, long_only=True)

wl = [r for r in snap if r.get('p33') and r['adv'] >= a.min_adv and not r['squeeze']
      and r['p33']['net'] >= 0.25 and r['p33']['maxE'] >= r['p33']['trig'] - 1e-9 and r['plan']['expGrade'] in 'AB']
json.dump([snapshot(r) for r in sorted(wl, key=lambda r: r['distZ1'])], open('snapshots.json', 'w'), indent=1)
json.dump([dict(ticker=nm, time=f"{t['t']:%Y-%m-%d %H:%M}", grade='ABC'[t['grade']], entry=t['e'], sl=t['stop'],
                tp1=t['t1'], tp2=t['t2'], tp1_hit=t['tp1'], exit=t.get('how'), r=round(t['R'], 4))
           for nm, t in sorted(today, key=lambda x: x[1]['t'])], open('trades.json', 'w'), indent=1)
print(f'\nWrote {len(wl)} snapshots to {os.path.abspath("snapshots.json")} for Claude review')
