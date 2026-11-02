"""SZones v3.3 core logic (Python port of the Pine indicator) for batch backtests and daily scans.

Set SZ_CLOSED=1 after the 17:00 close so the 16:45 bar is kept and open trades flatten.

Chart: 15m Bursa bars, HTF = 60m (resampled), the manual's intraday pairing.
Intraday adaptations (stated in the report):
  * long-only (Shariah: no short selling)
  * no new entries after 16:15 MYT, every open trade closed on the day's last bar
  * round-trip cost deducted from every trade (COST, fraction of entry price)
"""
import json, math, os, datetime as dt
import numpy as np

COST = 0.0025          # ~0.25% round trip: brokerage + clearing + stamp duty (assumption)
P = dict(emaF=21, emaS=55, htfEma=55, atr=14, z=(0.5, 1.0, 1.6, 2.4), zw=0.25, center=34,
         st=5, pd=50, fvgMin=0.2, fvgMax=5, liq=10, liqMax=10, obLook=10, obMax=3,
         maj=20, poiLook=15, poiMax=150, sl=1.8, tp1=0.6, tp2=1.2, maxLiq=6.0, scale=0.5,
         sweepLook=5, tf=15, htfMin=60, flat=True,   # tf = chart minutes; flat = exit at session end
         daily=False, nextOpen=False, gapFill=False)  # daily: weekly bias, no session rules; nextOpen: fill at next bar's open; gapFill: stop fills at open on a gap down


def tick(p):
    return 0.005 if p < 1 else 0.01 if p < 10 else 0.02 if p < 100 else 0.10


def ema(x, n):
    out = np.full(len(x), np.nan)
    a = 2 / (n + 1)
    if len(x) < n:
        return out
    out[n - 1] = np.mean(x[:n])
    for i in range(n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def rma(x, n):
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    out[n - 1] = np.mean(x[:n])
    for i in range(n, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


def atr_(h, l, c, n):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    tr[0] = h[0] - l[0]
    return rma(tr, n)


def pivots(x, L, high=True):
    """Value reported at bar i (confirmation bar) for pivot at i-L, like ta.pivothigh/low."""
    out = np.full(len(x), np.nan)
    for i in range(2 * L, len(x)):
        c = i - L
        left, right = x[c - L:c], x[c + 1:i + 1]
        v = x[c]
        if high and v > left.max() and v >= right.max():
            out[i] = v
        if not high and v < left.min() and v <= right.min():
            out[i] = v
    return out


def load15(code):
    d = json.load(open(f"data/{code}_{P['tf']}m.json"))['chart']['result'][0]
    q = d['indicators']['quote'][0]
    rows = []
    for t, o, h, l, c, v in zip(d['timestamp'], q['open'], q['high'], q['low'], q['close'], q['volume']):
        if None in (o, h, l, c):
            continue
        m = dt.datetime.fromtimestamp(t, dt.timezone(dt.timedelta(hours=8)))
        hm = m.hour * 60 + m.minute
        if 12 * 60 + 30 <= hm < 14 * 60 + 30:      # lunch break bars
            continue
        rows.append((m, o, h, l, c, v or 0))
    rmt = d['meta'].get('regularMarketTime')
    if not os.environ.get('SZ_CLOSED') and rows and rmt and rows[-1][0].timestamp() + P['tf'] * 60 > rmt + 60:
        rows.pop()
    return rows, d['meta']


def loadD(code):
    d = json.load(open(f'data/{code}_1d.json'))['chart']['result'][0]
    q = d['indicators']['quote'][0]
    rows = [(dt.datetime.fromtimestamp(t, dt.timezone(dt.timedelta(hours=8))).date(), o, h, l, c, v or 0)
            for t, o, h, l, c, v in zip(d['timestamp'], q['open'], q['high'], q['low'], q['close'], q['volume'])
            if None not in (o, h, l, c)]
    return rows


def _open(e, a, i, t, sig, S):
    """Levels from the actual fill price e; None if the cost filter rejects it."""
    tk = tick(e)
    if P.get('ticks'):
        sl = math.floor(round((e - a * P['sl']) / tk, 6)) * tk
        t1 = max(math.ceil(round((e + a * P['tp1']) / tk, 6)) * tk, e + P.get('minTk', 1) * tk)
    else:
        sl = e - a * P['sl']; t1 = e + a * P['tp1']
    risk = e - sl
    if risk <= 0 or (P.get('minNet') is not None and ((t1 - e) - COST * e) / risk < P['minNet']):
        return None
    liq = [y for y in S['hiL'] if y > t1 and y - e <= P['maxLiq'] * a]
    t2 = min(liq) if liq else (math.ceil(round((e + a * P['tp2']) / tk, 6)) * tk if P.get('ticks') else e + a * P['tp2'])
    return dict(i=i, t=t, e=e, stop=sl, risk=risk, t1=t1, t2=t2, tp1=False, tp2=False,
                R=0.0, grade=sig['grade'], zone=sig['zone'], maj=sig['maj'], chase=sig['chase'])


def run(code):
    rows, meta = load15(code)
    n = len(rows)
    T = [r[0] for r in rows]
    o, h, l, c, v = (np.array([r[k] for r in rows], float) for k in range(1, 6))
    atr = atr_(h, l, c, P['atr'])
    eF, eS, cen = ema(c, P['emaF']), ema(c, P['emaS']), ema(c, P['center'])

    # HTF 60m: resample, then each 15m bar sees the PREVIOUS completed hour (close[1] + lookahead_on)
    hkey = ([tuple(t.isocalendar()[:2]) for t in T] if P['daily'] else
            [(t.date(), (t.hour * 60 + t.minute) // P['htfMin']) for t in T])
    keys, hc = [], []
    for i, k in enumerate(hkey):
        if not keys or keys[-1] != k:
            keys.append(k); hc.append(c[i])
        else:
            hc[-1] = c[i]
    hc = np.array(hc); hE = ema(hc, P['htfEma'])
    kidx = {k: j for j, k in enumerate(keys)}
    htfC = np.array([hc[kidx[k] - 1] if kidx[k] >= 1 else np.nan for k in hkey])
    htfE = np.array([hE[kidx[k] - 1] if kidx[k] >= 1 else np.nan for k in hkey])

    pdHi = np.array([h[max(0, i - P['pd'] + 1):i + 1].max() for i in range(n)])
    pdLo = np.array([l[max(0, i - P['pd'] + 1):i + 1].min() for i in range(n)])
    swH, swL = pivots(h, P['liq']), pivots(l, P['liq'], False)
    stH, stL = pivots(h, P['st']), pivots(l, P['st'], False)
    mjH_, mjL_ = pivots(h, P['maj']), pivots(l, P['maj'], False)
    atrAvg = np.array([np.nanmean(atr[max(0, i - 99):i + 1]) if i >= 99 else np.nan for i in range(n)])
    lastBar = [False] * n if P['daily'] else [(i < n - 1 and T[i + 1].date() != T[i].date()) or (i == n - 1 and T[i].hour * 60 + T[i].minute + P['tf'] >= 16 * 60 + 50) for i in range(n)]

    S = dict(anchor=np.nan, zAtr=np.nan, zDirPrev=0, hiL=[], loL=[], bullSwBar=None, lastSH=np.nan, prevSH=np.nan,
             lastSL=np.nan, prevSL=np.nan, shB=True, slB=True, sDir=0, bullOB=[], bearOB=[], mjH=np.nan, mjHB=True,
             mjL=np.nan, mjLB=True, majDir=0, msbBar=None, msbLvl=np.nan, legExt=np.nan, idm=np.nan, idmTaken=False,
             bullF=[], bearF=[], poi=None)
    tr = None
    pend = None
    trades = []
    for i in range(n):
        if np.isnan(atr[i]) or np.isnan(eS[i]) or np.isnan(htfE[i]):
            continue
        a = atr[i]
        longB = eF[i] > eS[i] and htfC[i] > htfE[i]
        shortB = eF[i] < eS[i] and htfC[i] < htfE[i]
        zDir = 1 if longB else -1 if shortB else 0
        pull = (zDir == 1 and c[i] < cen[i]) or (zDir == -1 and c[i] > cen[i])
        hold = pull and zDir == S['zDirPrev'] and not np.isnan(S['anchor'])
        if not hold:
            S['anchor'], S['zAtr'] = cen[i], a
        S['zDirPrev'] = zDir
        hw = S['zAtr'] * P['zw'] / 2
        zones = [(S['anchor'] - zDir * S['zAtr'] * m + hw, S['anchor'] - zDir * S['zAtr'] * m - hw) for m in P['z']]
        inDisc = c[i] < (pdHi[i] + pdLo[i]) / 2

        # liquidity + sweeps
        S['hiL'] = [y for y in S['hiL'] if not h[i] > y]
        swept = [y for y in S['loL'] if l[i] < y and c[i] > y]
        S['loL'] = [y for y in S['loL'] if not l[i] < y]
        if swept:
            S['bullSwBar'] = i
        if not np.isnan(swH[i]):
            S['hiL'] = (S['hiL'] + [swH[i]])[-P['liqMax']:]
        if not np.isnan(swL[i]):
            S['loL'] = (S['loL'] + [swL[i]])[-P['liqMax']:]

        # internal structure
        if not np.isnan(stH[i]):
            S['prevSH'], S['lastSH'], S['shB'] = S['lastSH'], stH[i], False
        if not np.isnan(stL[i]):
            S['prevSL'], S['lastSL'], S['slB'] = S['lastSL'], stL[i], False
        bosUp = not S['shB'] and c[i] > S['lastSH']
        bosDn = not S['slB'] and c[i] < S['lastSL']
        if bosUp: S['shB'] = True
        if bosDn: S['slB'] = True
        if not np.isnan(S['lastSH']) and c[i] > S['lastSH']:
            S['sDir'] = 1
        elif not np.isnan(S['lastSL']) and c[i] < S['lastSL']:
            S['sDir'] = -1
        if bosUp:
            for k in range(1, P['obLook'] + 1):
                if i - k >= 0 and c[i - k] < o[i - k]:
                    S['bullOB'] = (S['bullOB'] + [(h[i - k], l[i - k])])[-P['obMax']:]; break
        if bosDn:
            for k in range(1, P['obLook'] + 1):
                if i - k >= 0 and c[i - k] > o[i - k]:
                    S['bearOB'] = (S['bearOB'] + [(h[i - k], l[i - k])])[-P['obMax']:]; break
        S['bullOB'] = [b for b in S['bullOB'] if not c[i] < b[1]]
        S['bearOB'] = [b for b in S['bearOB'] if not c[i] > b[0]]

        # major structure + leg + IDM
        if not np.isnan(mjH_[i]): S['mjH'], S['mjHB'] = mjH_[i], False
        if not np.isnan(mjL_[i]): S['mjL'], S['mjLB'] = mjL_[i], False
        msbUp = not S['mjHB'] and c[i] > S['mjH']
        msbDn = not S['mjLB'] and c[i] < S['mjL']
        if msbUp: S.update(mjHB=True, majDir=1, msbBar=i, msbLvl=S['mjH'])
        if msbDn: S.update(mjLB=True, majDir=-1, msbBar=i, msbLvl=S['mjL'])
        if msbUp or msbDn:
            S.update(legExt=l[i] if msbDn else h[i], idm=np.nan, idmTaken=False)
        elif S['majDir'] == -1:
            S['legExt'] = min(S['legExt'], l[i])
        elif S['majDir'] == 1:
            S['legExt'] = max(S['legExt'], h[i])
        idmNew = False
        if S['majDir'] == -1 and not np.isnan(stL[i]) and i - P['st'] > S['msbBar'] and stL[i] > S['legExt']:
            S.update(idm=stL[i], idmTaken=False); idmNew = True
        if S['majDir'] == 1 and not np.isnan(stH[i]) and i - P['st'] > S['msbBar'] and stH[i] < S['legExt']:
            S.update(idm=stH[i], idmTaken=False); idmNew = True
        if not idmNew and not np.isnan(S['idm']) and not S['idmTaken'] and (l[i] < S['idm'] if S['majDir'] == -1 else h[i] > S['idm']):
            S['idmTaken'] = True

        # FVGs
        if i >= 2:
            if l[i] > h[i - 2] and l[i] - h[i - 2] >= P['fvgMin'] * a and c[i - 1] > o[i - 1]:
                S['bullF'] = (S['bullF'] + [(l[i], h[i - 2])])[-P['fvgMax']:]
            if h[i] < l[i - 2] and l[i - 2] - h[i] >= P['fvgMin'] * a and c[i - 1] < o[i - 1]:
                S['bearF'] = (S['bearF'] + [(l[i - 2], h[i])])[-P['fvgMax']:]
        S['bullF'] = [f for f in S['bullF'] if not l[i] <= f[1]]
        S['bearF'] = [f for f in S['bearF'] if not h[i] >= f[0]]

        squeeze = bool(not np.isnan(S['prevSH']) and not np.isnan(S['prevSL']) and S['lastSH'] < S['prevSH']
                   and S['lastSL'] > S['prevSL'] and S['lastSL'] < c[i] < S['lastSH'] and a < atrAvg[i])

        # zone touch + grade (long side only)
        hit = 0
        for k in (3, 2, 1, 0):
            zt, zb = zones[k]
            if l[i] <= zt and h[i] >= zb:
                hit = k + 1; break
        zt, zb = zones[hit - 1] if hit else zones[0]
        obst = [max(b[1], c[i]) for b in S['bearF'] + S['bearOB'] if b[0] > c[i]]
        barrier = bool(obst) and min(obst) < c[i] + a * P['tp1']
        fvgConf = any(f[1] <= zt and f[0] >= zb for f in S['bullF'])
        score = int(S['sDir'] == 1) + int(inDisc) + int(fvgConf) - int(barrier) - int(squeeze)
        grade = 0 if score >= 2 else 1 if score == 1 else 2
        longSetup = zDir == 1 and hit > 0 and c[i] > o[i] and c[i] > zt
        hm = T[i].hour * 60 + T[i].minute

        # next-open fill of yesterday's signal (daily mode)
        if pend is not None and not tr:
            tr = _open(o[i], pend['a'], i, T[i], pend, S)
        pend = None

        # trade management (stop first)
        if tr:
            ex = None
            if l[i] <= tr['stop']:
                ex = min(tr['stop'], o[i]) if P['gapFill'] else tr['stop']; tr['how'] = 'BE' if tr['tp2'] else 'SL'
            else:
                if not tr['tp1'] and h[i] >= tr['t1']:
                    tr['tp1'] = True; tr['R'] += P['scale'] * (tr['t1'] - tr['e']) / tr['risk']
                hit2 = False
                if not tr['tp2'] and h[i] >= tr['t2']:
                    tr['tp2'] = True; hit2 = True; tr['stop'] = tr['e']
                if tr['tp2'] and not hit2 and c[i] < eF[i]:
                    ex = c[i]; tr['how'] = 'runner'
                elif not longB:
                    ex = c[i]; tr['how'] = 'bias'
                elif lastBar[i] and P['flat']:
                    ex = c[i]; tr['how'] = 'EOD'
            if ex is not None:
                rem = 1 - P['scale'] if tr['tp1'] else 1.0
                tr['R'] += rem * (ex - tr['e']) / tr['risk'] - COST * tr['e'] / tr['risk']
                tr['xt'] = T[i]; tr['xp'] = ex
                trades.append(tr); tr = None
        elif longSetup and (P['daily'] or (hm + P['tf'] <= 16 * 60 + 30 and not lastBar[i])):
            chase = (c[i] - zt) / a
            ok = True
            if P.get('msbGate') and S['majDir'] == -1: ok = False
            if P.get('chase') is not None and chase > P['chase']: ok = False
            if ok:
                sig = dict(grade=grade, zone=hit, maj=S['majDir'], chase=chase, a=a)
                if P['nextOpen']:
                    pend = sig
                else:
                    tr = _open(c[i], a, i, T[i], sig, S)

        # POI on MSB (analyst view)
        if msbUp or msbDn:
            d = -1 if msbDn else 1
            found = 0
            for k in range(1, P['poiLook'] + 1):
                if i - k >= 0 and ((c[i - k] > o[i - k]) if d == -1 else (c[i - k] < o[i - k])):
                    found = k; break
            if found:
                tp_, bt = h[i - found], l[i - found]; j = found + 1; m = 1
                while m < 3 and i - j >= 0 and ((c[i - j] > o[i - j]) if d == -1 else (c[i - j] < o[i - j])):
                    tp_, bt = max(tp_, h[i - j]), min(bt, l[i - j]); j += 1; m += 1
                S['poi'] = dict(top=tp_, bot=bt, mid=(tp_ + bt) / 2, dir=d, bar=i)
        p = S['poi']
        if p and ((p['dir'] == -1 and c[i] > p['top']) or (p['dir'] == 1 and c[i] < p['bot']) or i - p['bar'] > P['poiMax']):
            S['poi'] = None

    run.trades = trades
    run.open = tr
    # ── snapshot at last bar ──
    i = n - 1; a = atr[i]
    D = loadD(code)
    dc = np.array([r[4] for r in D]); dh = np.array([r[2] for r in D]); dl = np.array([r[3] for r in D])
    dv = np.array([r[5] for r in D])
    datr = atr_(dh, dl, dc, 14)
    adv = float(np.mean((dc * dv)[-20:]))
    dE20, dE50 = ema(dc, 20), ema(dc, 50)
    zl = [(round(zt, 4), round(zb, 4)) for zt, zb in zones]
    nextZ = None
    if zDir == 1:
        for k, (zt, zb) in enumerate(zones):
            if zb <= c[i] <= zt or zt < c[i]:
                nextZ = k + 1; break
    plan = None
    if nextZ:
        zt, zb = zones[nextZ - 1]; zm = (zt + zb) / 2
        t1 = zm + a * P['tp1']
        liq = [y for y in S['hiL'] if y > t1 and y - zm <= P['maxLiq'] * a]
        t2 = min(liq) if liq else zm + a * P['tp2']
        fvgC = any(f[1] <= zt and f[0] >= zb for f in S['bullF'])
        obst = [max(b[1], zm) for b in S['bearF'] + S['bearOB'] if b[0] > zm]
        barr = bool(obst) and min(obst) < t1
        pdMid = (pdHi[i] + pdLo[i]) / 2
        sc = int(S['sDir'] == 1) + int(zm < pdMid) + int(fvgC) - int(barr) - int(squeeze)
        plan = dict(zone=nextZ, zt=zt, zb=zb, zm=zm, sl=zm - a * P['sl'], tp1=t1, tp2=t2,
                    tp2src='BSL' if liq else 'ATR', expGrade='ABC'[0 if sc >= 2 else 1 if sc == 1 else 2],
                    why=' '.join(x for x, ok in (('BOS', S['sDir'] == 1), ('P/D', zm < pdMid), ('FVG', fvgC),
                                                 ('barrier', barr), ('squeeze', squeeze)) if ok) or 'zone only')
    Rs = np.array([t['R'] for t in trades])
    byg = {g: [t['R'] for t in trades if t['grade'] == gi] for gi, g in enumerate('ABC')}
    bsl = [y for y in S['hiL'] if y > c[i]]
    ssl = [y for y in S['loL'] if y < c[i]]
    p = S['poi']
    ext = None
    if S['majDir'] != 0 and not np.isnan(S['legExt']):
        ls = (p['bot'] if S['majDir'] == 1 else p['top']) if p and p['dir'] == S['majDir'] else S['msbLvl']
        ext = S['legExt'] + S['majDir'] * abs(ls - S['legExt']) * 0.618
    return dict(
        code=code, name=meta.get('shortName') or meta.get('longName'), last=T[-1].strftime('%Y-%m-%d %H:%M'),
        close=c[i], atr15=a, atr15pct=a / c[i] * 100, tick=tick(c[i]), tp1ticks=a * P['tp1'] / tick(c[i]),
        tp1pct=a * P['tp1'] / c[i] * 100, datrpct=datr[-1] / dc[-1] * 100, adv=adv,
        dailyUp=bool(dc[-1] > dE20[-1] > dE50[-1]), dailyDn=bool(dc[-1] < dE20[-1] < dE50[-1]),
        pdh=dh[-1], pdl=dl[-1],
        zDir=zDir, htf='bull' if htfC[i] > htfE[i] else 'bear', emaF=eF[i], emaS=eS[i], sDir=S['sDir'],
        disc=bool(c[i] < (pdHi[i] + pdLo[i]) / 2), squeeze=bool(squeeze), zones=zl, nextZ=nextZ, plan=plan,
        distZ1=(c[i] - zones[0][0]) / a if zDir == 1 else None,
        majDir=S['majDir'], msbLvl=S['msbLvl'], msbAgo=(i - S['msbBar']) if S['msbBar'] is not None else None,
        poi=p, idm=None if np.isnan(S['idm']) or S['idmTaken'] else S['idm'], legExt=S['legExt'], ext=ext,
        bsl=min(bsl) if bsl else None, ssl=max(ssl) if ssl else None,
        sweepAgo=(i - S['bullSwBar']) if S['bullSwBar'] is not None else None,
        nTr=len(trades), avgR=float(Rs.mean()) if len(Rs) else None, win=float((Rs > 0).mean() * 100) if len(Rs) else None,
        sumR=float(Rs.sum()) if len(Rs) else 0.0,
        grossAvgR=float(np.mean([t['R'] + COST * t['e'] / t['risk'] for t in trades])) if trades else None,
        byGrade={g: (len(r), float(np.mean(r)) if r else None) for g, r in byg.items()},
        exits={k: sum(1 for t in trades if t.get('how') == k) for k in ('SL', 'BE', 'runner', 'bias', 'EOD')},
        days=len(set(t.date() for t in T)))
