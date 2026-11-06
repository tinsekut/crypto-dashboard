# SZones — project handoff for Claude Code

**As of:** Fri 6 Nov 2026, ~11:45 MYT (handoff written 2 Nov 07:36 MYT; pre-open/evening routines and the SZones Brief dashboard added 2 Nov; grades A and B from 3 Nov; dashboard charts 5 Nov; mid-day check 6 Nov). Latest backtest data: Fri 30 Oct close.
**Owner:** tIn (Malaysia, Asia/Kuala_Lumpur). Trades Bursa Malaysia (Shariah-compliant only, long only) and gold.
**Use this file as `CLAUDE.md` in the project folder.** It is the single source of truth for the new session.

---

## 1. How to work with tIn

- Terse, directive. Proceed with independent judgment; don't ask for confirmation on routine steps.
- Wants production-ready deliverables, not explanations.
- **Pine scripts: always paste the full script inline** in the reply (he copies from chat into TradingView on iPad). Attaching a file alone is not enough.
- Shariah: only SC-listed Shariah-compliant counters; long only; no short selling.
- Be honest about evidence. Every rule change so far was backtested first; untested ideas are labelled "untested" in code and docs.
- Not financial advice: frame outputs as indicator readouts.

---

## 2. What SZones is

A Pine Script v6 TradingView indicator: trend-pullback zones (4 ATR bands off a 34-EMA anchor, EMA 21/55 trend + higher-timeframe filter), rejection-candle entries, SMC confluence grading (A/B/C), trade management (SL 1.8 ATR, TP1, TP2 at liquidity, runner, breakeven after TP2), an "Analyst view" (MSB, POI order block, IDM, price tags, scenario path) and a beginner dashboard.

### Version history

| Version | Key change |
|---|---|
| KZones v2 / v2.1 | Fixed original script (compile errors, bias logic, real trade state machine), beginner dashboard |
| SZones v3 | SMC layer (liquidity BSL/SSL, sweeps, FVG, OB, BOS/CHoCH, squeeze), grades, liquidity TP2 |
| v3.1 | Grading re-weighted to tested evidence (BOS, premium/discount, FVG); OB and sweeps info only |
| v3.2 | Analyst view: MSB, POI with 50% line, IDM, price tags, scenario path, optional POI limit entries |
| v3.3 | Execution guards: entry-relative tick-valid levels, chase guard 0.5 ATR, cost filter (net TP1 ≥ 0.25R), stats net of costs, long-only, session rules, "Bursa intraday (15m)" preset |
| v3.3.1 | Fixed RE10140 "too many plots (65 > 64)": zone borders fixed colour; signal/POI/sweep markers are labels |
| v3.3.2 | AI bridge: JSON webhook snapshots for a Claude relay. Not used since 2 Nov: Claude reviews from the Python port instead (section 6); keep the input off |
| v3.3.3 | Defaults only: daily swing, grade A only (long only, Bursa ticks, 0.25% cost, HTF W, TP1 1.0 / TP2 2.0, minimum grade A). Logic identical to v3.3.2. Use on a daily chart |
| **v3.3.4 (current)** | Dashboard: Compact (default) / Full layout; new Timing row — next signal check, entry at next open, typical TP1/TP2/exit date ranges (25th–75th percentile) from this chart's 5+ closed trades, else daily grade A backtest (TP1 2 bars [1–5] 47% reach, TP2 13 [6–20] 25%, exit 7 [3–23]). Weekends skipped, holidays not. Signal logic unchanged |

Current script: `pine/SZones_v3_3_4.pine` (v3.3.2 and v3.3.3 kept for reference). Its default Minimum grade is still "A only": since 3 Nov tIn sets "Minimum signal grade" to "B or better" in the indicator settings to match the paper record. Not compile-checked here (no Pine compiler in the container); tIn compiles on TradingView.

### Bursa intraday preset (what it forces)

Long only · Bursa tick table (<1.00: 0.005 · <10: 0.01 · <100: 0.02 · else 0.10) · 0.25% round-trip cost · TP1 1.0 ATR / TP2 2.0 ATR (or nearest BSL within 6 ATR) · entries 09:00–16:15 · flatten from the 16:45 bar · HTF for bias = 60 on a 15m chart.

---

## 3. Evidence base (do not lose these numbers)

**Confluence Signal Engine test (earlier claude.ai chat):** 275 signals, 37 crypto pairs, H1.
Structure aligned (BOS) +0.268R (only filter that held out-of-sample) · BOS + premium/discount +0.375R (best) · BOS + FVG +0.341R · OB within 1 ATR −0.115R (harmful) · after liquidity sweep −0.083R (harmful). Best filter narrowly failed Bonferroni (p 0.0046 vs 0.0042).

**Bursa 15m backtest (57 Shariah counters, 60 sessions, long only, flat by close, 0.25% cost):**

| Version | Trades | Avg R | Halves |
|---|---|---|---|
| v3.2 rules | 751 | −0.45 | −0.44 / −0.47 |
| + tick-valid levels | 720 | −0.39 | — |
| + chase guard 0.5 ATR | 258 | −0.25 | −0.24 / −0.27 |
| + cost filter + TP 1.0/2.0 (Bursa preset) | 234 | −0.21 (grade A −0.05) | −0.20 / −0.23 |
| Skip longs vs bearish MSB | 628 | −0.46 | no effect → info only |

Afternoon entries (after 14:30): −0.16R over 67 trades; 55 of 67 closed at session end.
By grade (v3.3 preset): A −0.05, B −0.28, C −0.23. B is not better than C → no min-grade change yet.

**Live paper record (v3.3.1 rules):**

| Day | Signals | Total R | Notes |
|---|---|---|---|
| Tue 27 Oct | 2 (v3.2 morning list) | −1.30 | DIALOG ~0R, MISC stopped. Under v3.3 both would be skipped (chased). PCHEM A long +, open at 15:15 |
| Thu 29 Oct | 10 | −2.79 | A/B only −2.20; morning −0.38, afternoon −2.42 |
| Fri 30 Oct | 6 | −0.84 | Revised from −0.51 once final data arrived (PPB stopped in the closing auction). A/B only +0.11 |

Rolling 60-session average: **−0.196R per trade (221 trades)**. Still negative → paper or very small size.

**Timeframe test (2 Nov 2026, TradingView data, 57 counters, Bursa preset, 0.25% cost, long only):** asked for "real time, shortest interval, maximum gain". Result: **shorter is worse; no timeframe is profitable after costs.**

| Chart (bias TF) | Sessions | Trades | Gross R/trade | Net R/trade | Net R/session | Halves |
|---|---|---|---|---|---|---|
| 5m (60) intraday | 159 | 282 | −0.27 | −0.52 | −0.92 | −0.55 / −0.49 |
| 15m (60) intraday | 285 | 1,003 | −0.07 | −0.27 | −0.97 | −0.29 / −0.26 |
| 30m (240) intraday | 491 | 1,573 | −0.01 | −0.18 | −0.56 | −0.20 / −0.16 |
| 60m (240) intraday | 780 | 1,727 | +0.00 | −0.13 | −0.30 | −0.14 / −0.13 |
| 60m (240) overnight holds | 780 | 1,317 | +0.03 | −0.11 | −0.19 | −0.15 / −0.07 |

5m has no edge even before costs, and costs (0.26R/trade at a ~1% stop) double the loss. Only 60m is about break-even gross. Same ranking on the common 159-session window. 4H HTF buckets are clock-aligned (approximation). Engine now takes `P['tf']`, `P['htfMin']`, `P['flat']`; tf=15 reproduces Fri 30 Oct exactly. Data delay (~15 min on all free feeds) also rules out true real-time on 5m.

**Daily swing test (2 Nov 2026, TradingView daily, 57 counters, trades 2016 → 30 Oct 2026, weekly bias, entry at next day's open, gap-aware stops, 0.25% cost, long only):** **break-even, no edge.**

| Variant | Trades | Avg R | Win | R/year | Halves |
|---|---|---|---|---|---|
| Daily preset, all signals | 938 | +0.003 (t = 0.10) | 43% | +0.3 | +0.04 / −0.03 |
| Max 3 open positions (manual rule) | 404 | +0.020 | 45% | +0.7 | +0.17 / −0.13 |
| Fill at signal close (optimistic) | 939 | −0.023 | 41% | −2.0 | −0.01 / −0.04 |
| TP 0.6 / 1.2 ATR | 893 | +0.009 | 44% | +0.7 | +0.06 / −0.04 |
| No chase guard / cost filter | 1,476 | +0.017 | 45% | +2.3 | +0.06 / −0.02 |

Median hold 13 days; +0.05% net per trade (+0.30% gross). By year: 2017 +26R carries the total; 2018 −12R, 2022 −9R, 2026 YTD −13R. By grade: A −0.05 (150), B +0.12 (331), C −0.06 (457) — A not better than C, so grading does not hold on daily; B's figure is one of many post-hoc cuts, not evidence. Buy-and-hold of the 48 counters listed since 2016: median +0.2%/yr, mean +4.2%/yr (price only). Universe is today's compliant, still-listed counters → survivorship bias flatters every figure here. Conclusion: longer holds stop the intraday bleed (cost per trade falls from 0.20R to ~0.06R) but the entry signal itself has no edge on Bursa at any timeframe tested. Engine: `P['daily']`, `P['nextOpen']`, `P['gapFill']`; trades record exit time `xt`.

**Daily grade filter (2 Nov 2026, same daily backtest, filter applied in the engine via `P['minGrade']`):** tIn chose grade A only on 2 Nov, then **grades A and B from 3 Nov 2026** (before the first forward session) for the paper record:

| Grades traded | Trades | Avg R | t | Win | TP1 | Halves |
|---|---|---|---|---|---|---|
| All | 938 | +0.003 | +0.10 | 43% | 55% | +0.04 / −0.03 |
| **A and B (live setting from 3 Nov)** | 569 | +0.038 | +0.85 | 45% | 55% | +0.00 / +0.07 |
| A only (2 Nov setting) | 212 | −0.024 | −0.30 | 38% | 47% | −0.09 / +0.04 |
| A only, max 3 open | 189 | −0.003 | −0.03 | 39% | 47% | −0.09 / +0.08 |

None is significant. Replaying the last 12 months with A and B through the live pipeline gave −0.03R over 62 trades (3 Nov). **Data warning (5 Nov):** that replay includes CAPITALA at −5.57R (entry 0.915, exit 0.277 on 24 Nov 2025): a one-day gap far past the stop, almost certainly a corporate action Yahoo does not adjust. Without it the 12-month replay is +0.048R over 59 trades (5 Nov). The dashboard flags any closed trade below −2R as suspect; the engine and figures are unchanged. Earlier, replaying the last 12 months through the live pipeline (Yahoo, 5-year history) gave A only −0.13R over 26 trades versus +0.05R over 29 on TradingView's 20-year series — the difference is within noise (standard error ≈ 0.2R) but shows the data-source sensitivity.

**Gold test (2 Nov 2026, OANDA:XAUUSD via TradingView, indicator defaults TP 0.6/1.2 ATR + chase 0.5 + net TP1 ≥ 0.25R, 0.02% cost, long only, next-bar fills, gap-aware stops, no Bursa session rules):** **no significant edge; intraday negative; the positive higher timeframes trail plain holding by a wide margin.**

| Chart (bias) | Window | Gold move | Trades | Avg R (gross) | t | R/month |
|---|---|---|---|---|---|---|
| 15m (60) | 17 Aug → 30 Oct 2026 | −12% | 27 | −0.24 (−0.18) | −2.3 | −3.3 |
| 1H (240) | 29 Dec 2025 → 30 Oct 2026 | −14% | 32 | −0.10 (−0.08) | −0.9 | −0.4 |
| 4H (D) | Aug 2023 → Oct 2026 | +101% | 47 | +0.10 (+0.12) | +0.6 | +0.14 |
| Daily (W) | Jul 2007 → Oct 2026 | +481% | 47 | +0.18 (+0.19) | +1.0 | +0.04 |

Sensitivity (TP 1.0/2.0, cost 0.05%, close fills) moves nothing materially; at 0.05% cost the 15m cost filter rejects every signal. At 1% risk per trade, Daily ≈ +0.5%/yr and 4H ≈ +1.7%/yr versus holding gold +9.5%/yr (19 yrs) and +24%/yr (3 yrs). Daily by grade: A +0.36 (9), B +0.16 (15), C +0.12 (23) — too few trades to mean anything. One instrument, 27–47 trades per test, and each window sits in one regime (intraday windows in a falling market, 4H/daily in a bull market), so none of this is conclusive. Engine settings used: `sess=False`, `tickSize=0.01`, `COST=0.0002`.

### Lessons already built into the rules
1. Levels must be entry-relative and tick-valid (27 Oct: morning sheet TP1 sat below DIALOG's actual entry).
2. Don't chase: skip closes > 0.5 ATR past the zone (largest single improvement).
3. Costs dominate small TP1s on Bursa 15m; skip if net TP1 < 0.25R.
4. **Never pre-place buy orders from the watchlist.** 30 Oct: VITROX, FRONTKN, INARI, GAMUDA traded through their triggers and stops; waiting for the green close avoided all four.
5. Some counters have no valid trigger because the tick is coarse vs ATR (HARTA, AXIATA, SUPERMX, IOIPG, MRDIY at times).
6. Don't tune on one day.

### Pine v6 gotchas hit in this project
- Max 64 plots; series colours count extra. Use labels/lines/boxes for markers.
- Avoid `+=` style compound operators (lint rule kept since an earlier failure); use `x := x + 1`.
- `bool x = na` is not allowed; use separate "has" flags.
- Continuation lines must not be indented at a multiple of 4 spaces.
- JSON numbers: format with `"0.######"` (not `"#.###"`, which can print `.5`).
- `str.format_time`, `str.replace_all`, `polyline`, `chart.point` are used and valid in v6.

---

## 4. Daily routine (Python)

`python/szones_v33.py` is a Python port of the v3.3 rules. `python/szones_daily.py` fetches data, runs the Bursa preset over the Shariah universe, prints the day's signals (R after costs), the rolling average, and the next watchlist.

```bash
cd python
python3 szones_daily.py --date 2026-11-02            # pre-market / intraday (drops the still-forming bar)
python3 szones_daily.py --date 2026-11-02 --closed   # after 17:00 MYT (keeps the 16:45 bar, flattens trades)
```

- **Data (since 2 Nov: fallback chain in `python/szones_data.py`):** each counter's 15m and 1d bars come from the first source that returns enough clean bars, converted to the Yahoo chart-JSON shape so `szones_v33.py` is unchanged. Order via env `SZ_SOURCES` (default `yahoo,eodhd,tradingview`); `local` replays files already in the run folder. The source per counter is written to `runs/run_D/sources.json` — name it in the report. If more than half the universe has no 15m data the run aborts and writes no outputs (so the scorecard never scores an empty day).
  1. **Yahoo** (free): `query1`/`query2.finance.yahoo.com/v8/finance/chart/{code}.KL`, 15m 60d + 1d 1y, ~15 min delayed.
  2. **EODHD** (official, paid ~US$20/month EOD+intraday): `eodhd.com/api/intraday/{code}.KLSE?interval=5m` resampled to 15m, `api/eod` for daily. Needs env `EODHD_API_KEY` (set as an environment secret, never in files or chat). Non-US intraday is 5m/1h from Oct 2020. Live KLSE coverage untested.
  3. **TradingView** (free, unofficial `tvDatafeed`, `MYX:<name>`, 15m up to 5000 bars ≈ 190 sessions): vendored and reviewed in `python/vendor/tvDatafeed` (see its VENDORED.md; one header patch). Needs `pip install -r requirements.txt` (websocket-client, from PyPI). Never install it from GitHub in scheduled runs. May break or be rate-limited. Verified on live MYX data 2 Nov (all 57 counters at 5m/15m/30m/60m/daily; some connections drop and succeed on retry).
  Researched and rejected (2 Nov): Twelve Data (Bursa only on Pro tier+, its own docs say XKLS may need a licence); Google Finance in Sheets (no confirmed Bursa support, history not readable by API).
  Network allowlist: `query1.finance.yahoo.com`, `query2.finance.yahoo.com`, `eodhd.com`, `data.tradingview.com` — **allowed by tIn on 2 Nov; all three sources reachable.**
  Verified 2 Nov: replay of Fri 30 Oct on live Yahoo data reproduced the recorded day exactly (6 signals, −0.84R, A/B +0.11R). TradingView returned 5,000 15m bars for DIALOG (back to 17 Dec 2025) and matched Yahoo closes on 1,411 of 1,414 shared bars — usable for longer backtests (open item 4).
- Lunch bars 12:30–14:30 are removed.
- **Universe:** `data/shariah_sc_list_2026-05-29.json` (SC list effective 29 May 2026, via a mirror site). 59 compliant in the file; MYEG (0138) and DNEX (7033) excluded for bad Yahoo data → 57. **Next SC list update: Fri 27 Nov 2026** — refresh the file then.
- **Non-compliant (excluded):** YTL Power, YTL, Yinson, Hap Seng, Astro, Maybank, Genting, Genting Malaysia, KLCC.
- **Watchlist filter:** grade A/B, valid trigger window (max entry ≥ trigger), net TP1 ≥ 0.25R, 20-day turnover ≥ RM10m, no squeeze.
- **Post-close report format used so far:** day verdict + KLCI context; signals table (time = signal-candle close, grade, entry, result, R); previous watchlist outcome; lesson; next watchlist table (close, trigger, SL, TP1/TP2, grade); grade-A deeper pullbacks; skip list; sources.

---

## 5. Mon 2 Nov — 15m pre-market plan (historical; superseded the same day by the daily swing routine, section 6)

Market: KLCI 1,615.60 (+4.04 w/w); analysts expect 1,600–1,630, mild upside bias. Events: Malaysia manufacturing PMI today; **BNM rate decision Wed 5 Nov**. Last week energy and plantation led, financials fell.

| Counter | Fri close | Trigger (green 15m close) | SL | TP1 / TP2 | Grade |
|---|---|---|---|---|---|
| TOPGLOV | 0.975 | 0.980 | 0.960 | 0.990 / 1.010 | A |
| SIME | 2.43 | 2.43 | 2.40 | 2.45 / 2.48 | A |
| TM | 7.93 | 7.92–7.93 | 7.84 | 7.97 / 8.04 | A |
| SDG | 5.83 | 5.82 | 5.76 | 5.85 / 5.88 | A (net only +0.26R) |
| DIALOG | 2.13 | 2.13 | 2.10 | 2.15 / 2.16 | B (cap 2.15–2.16) |

Deeper grade A only: PCHEM 4.63–4.64, SUNWAY 4.35, KLK 20.68–20.72. Skip: HARTA, SUPERMX, IOIPG, MRDIY (no valid trigger); TENAGA, TIMECOM, PETGAS, PMETAL (TP1 < costs); grade C.
Rules: no pre-placed orders; afternoon = TP1 scalp only, no entries after 16:15; halve size or sit out Wed afternoon (BNM).

### DIALOG (7277) notes
Daily uptrend (above 20/50/200 EMA; +13.8% in 20 days; 52-week high 2.36). Resistance 2.15–2.18, extension 2.24. POI 2.08–2.09 is the line in the sand. FY26 (June YE) core profit +39–46%, brokers BUY RM2.45–2.57. Q1 FY27 results likely mid/late Nov (last year 18 Nov).

---

## 6. Claude review (replaces the TradingView AI bridge, decided 2 Nov 2026)

tIn decided: **no TradingView integration.** Claude does the AI review itself from the Python port's output. The webhook relay was removed. Leave the Pine "Send JSON snapshots" input **off**; the v3.3.2 script needs no other change.

**Evening routine — DAILY SWING since 2 Nov 2026 (tIn switched from 15m; weekdays after 17:00 MYT, D = today):**

```bash
pip install -r requirements.txt                 # PyPI only
cd python
python3 szones_swing.py --date D --workdir ../runs/swing_D
python3 szones_scorecard.py
```

`szones_swing.py` uses exactly the daily backtest settings (section 3), **grades A and B** (default `--min-grade B`, tIn's choice on 3 Nov; A only on 2 Nov; see the grade-filter table): daily chart, weekly bias, chase 0.5 ATR, net TP1 ≥ 0.25R, TP1 1.0 / TP2 2.0 ATR, SL 1.8 ATR, runner, **entry at the next day's open** after a green signal close, gap-aware stops, 0.25% cost, long only, ~5 years of daily bars via the fallback chain (`mode='daily'`). No position cap in the record (backtest primary); note when more than 3 are open. Forward paper record counts only trades entered from **Tue 3 Nov 2026**; positions open on 2 Nov came from replaying the rules and are labelled "inherited".

1. Outputs in `runs/swing_D/`: `snapshots.json` (SIGNAL = fired at today's close, buy at the next open; WATCH = within 1 ATR of the next zone, A/B, valid trigger, net TP1 ≥ 0.25R, ADV ≥ RM10m, no squeeze — could signal at the next close), `trades.json` (closed trades, entries in the last 400 days), `positions.json` (open), `sources.json`.
2. Review every snapshot and write `reviews/D.json` **before** the next open: `[{ticker, event, call, p_tp1, next_move, horizon_bars, key_level, invalidation, reason}]`. Never edit a review after the next session opens.
3. Report: data source + stale counters; signals for the next open (with Claude's call); open positions with stop/targets and open R; trades closed today with R; forward record since 3 Nov and last-12-month replay; watch list (with Claude's call); scorecard. Indicator readouts, not advice.
4. Refresh the dashboard (see "Brief refresh" below) with `--run ../runs/swing_D`, edition Evening.
5. Commit `runs/swing_D/{snapshots,trades,positions,sources,summary}.json`, `runs/swing_D/brief/*.json` and `reviews/D.json`, push to the working branch. Raw data (`runs/*/data/`, `out.json`) is git-ignored.

**Pre-open routine (since 3 Nov 2026, tIn asked for two runs a day; weekdays 08:12 MYT, before Bursa's 08:30 pre-open; T = today):**

```bash
cd python
python3 szones_swing.py --date T --premarket --workdir ../runs/premarket_T
```

A daily chart cannot produce new signals overnight, so this run re-checks last evening's plan on the final data of the last close (session S = the latest daily bar) and lists the orders for today's open. It writes `snapshots.json`, `positions.json`, `summary.json`, `sources.json` (no `trades.json`, so the scorecard is untouched) and prints what changed against `runs/swing_S/snapshots.json`.
1. If `reviews/S.json` is missing (the evening run failed), review every snapshot now under the rules below and write it. If it exists and `summary.json` lists `changes.added`, append reviews for the added setups only. Never edit an existing review, and write nothing after 09:00.
2. Refresh the dashboard with `--run ../runs/premarket_T`, edition Pre-open.
3. Report briefly: orders for today's open (SIGNALs; real stop/targets are re-based on the actual open), changes against the evening plan, open positions with their stops, data source and stale counters. Indicator readouts, not advice. If T is a Bursa holiday the plan simply carries to the next session; say so if you know it.
4. Commit `runs/premarket_T/{snapshots,positions,summary,sources}.json`, `runs/premarket_T/brief/*.json` and any review file changed, push.

**Mid-day check (since 6 Nov 2026, tIn asked for a mid-day analysis validating the pre-open results; weekdays 12:45 MYT, in Bursa's 12:30–14:30 lunch break; T = today):**

```bash
cd python
python3 szones_midday.py --date T --workdir ../runs/midday_T
```

Reads the plan from `runs/premarket_T/` (else the latest `runs/swing_*` before T) and Claude's calls from `reviews/S.json`, fetches 15-minute bars for those counters only (fallback chain, about 15 minutes delayed) and checks the morning session 09:00–12:30. Signals bought at the open: actual open vs the indicative entry (in ATR), stop and TP1 re-based on the open, whether the morning touched either. Positions: stop touched (exit at the stop, or the open if it gapped below; R reported), TP1/TP2 reached, or holding with the room left above the stop. Watch list: last price against the buy-close range and whether the day is green so far ("On track" = inside the range and above the open). Prices are snapped to Bursa ticks (Yahoo intraday prices carry float noise). Nothing changes the paper record or the reviews: the evening run books signals and exits on the daily close. Writes `midday.json`, `sources.json` and `brief/midday.json`.
1. If it ABORTs or finds no bars for T, report why (holiday or data) and stop.
2. Upload `runs/midday_T/brief/midday.json` as an asset to the dashboard, ArtifactData `get` `datasets/midday`, `update` it pinned with `if_version` (new `source` url, `updated` = now, by "Claude, SZones mid-day check"), delete the replaced asset. Touch no other dataset.
3. Report briefly: per position, signal and watch-list counter the status and the note; what needs attention this afternoon (stops touched, positions near the stop, watch-list setups on track to signal at the close). Indicator readouts, not advice.
4. Commit `runs/midday_T/{midday,sources}.json` and `runs/midday_T/brief/midday.json`, push.

**Brief refresh (both routines).** Dashboard "SZones Brief": https://claude.ai/artifact/Kt4NUYMt5fGiE2U3j1v1cj (Dashboard artifact type; the page reads five data files from its asset store; page source kept in `dashboard/index.html`).
1. `python3 szones_brief.py --run <run folder>` writes `<run>/brief/{run_summary,positions,watchlist,closed_trades,calls}.json` (Claude's calls merged from `reviews/S.json`; closed trades from `runs/swing_S/trades.json`, entries in the last 12 months, with a running total; calls from `szones_scorecard.py --json`).
2. Upload each of the five files as an asset to the dashboard, one call per file (Artifact publish with `url` = the dashboard, `asset: true`, `file_path`). Note each new `/_blob/<id>` url.
3. ArtifactData `list` `datasets` (versions and old source urls), then one `batch` of `update`s to `datasets/run_summary`, `positions`, `watchlist`, `closed_trades`, `calls`, each pinned with `if_version`: `source` = `{kind: "file", url: <new blob url>, name: "<file>.json"}`, `updated` = `{at: <now, UTC ISO>, by: "Claude, SZones <edition> routine"}`.
4. Delete the five replaced assets (Artifact `delete`, `url` = the dashboard, `path` = the old blob id). Never touch `datasets/timeframe_test`, `datasets/grade_test` (fixed backtests) or `files/index.html` (the page) in a routine.

Dashboard sections (added 5 Nov at tIn's request for more visuals): headline tiles; mid-day check (6 Nov: morning low–high, open and last against stop, buy range and TP1, with status; dataset `midday`, refreshed only by the mid-day check); open positions ladder in R; watch list "distance to buy range" chart (% from close, buy-close range, stop, TP1) plus the levels table with Claude's calls; track record (running total of closed trades over 12 months with the forward-record start marked, and Claude's calls toward the 50 needed); evidence bars (timeframe, grade filter).

The 15m routine (`szones_daily.py`) still works but is no longer scheduled.

**Review rules (keep identical across sessions, or the scorecard means nothing):**
- Use only the numbers in the snapshot. No news, prices or fundamentals from outside it.
- Long only. SIGNAL: entry at the next open, stop/targets re-based on that open. WATCH: only if the next daily candle closes green between plan.trigger_close and plan.max_close; skip if net_tp1_r < 0.25.
- Base rate (daily backtest 2016–2026, grades A and B, 569 trades; live setting since 3 Nov): about 0R per trade after costs (+0.038R), win 45%, TP1 reached on 55%, median hold about 2 weeks. (Grade A only, used for the 2 Nov reviews: −0.024R, win 38%, TP1 47%.) Stay near it unless the snapshot gives concrete reasons (structure, discount/premium, FVG, barriers, squeeze, distance to liquidity, weekly bias).
- call: TAKE (would take it), SKIP (would not), WAIT (not enough to judge). p_tp1 in 0–1. reason ≤ 160 characters, plain English.

**Judging Claude:** `szones_scorecard.py` merges closed trades from every run, matches each review to that ticker's first trade entered within 5 calendar days after the review, and keeps it pending while the trade is open. Claude helps only if TAKE beats all reviewed trades on average R over 50+ trades, and its Brier score on TP1 is below the base-rate Brier. Until then the rules decide; Claude's call is logged beside them.

**Secrets:** never write keys or tokens into files, code or chat. tIn's old Polymarket bot `.env` (shared in a March 2026 chat) exposed a Telegram bot token and a Polymarket wallet private key — advise revoking the token and moving funds/rotating the key if still used.

---

## 7. Documents and links

- Code and run history: GitHub `tinsekut/crypto-dashboard`, branch `claude/szones-handoff`, folder `szones/` (includes the vendored TradingView library).
- Google Drive copy of the latest files: folder "SZones latest — v3.3.4 (2 Nov 2026)" (the original morning bundle stays in "SZones handoff (2 Nov 2026)"; its obsolete relay script and AI setup checklist were moved to the Drive trash on 2 Nov).
- Dashboard "SZones Brief" (pre-open and evening editions plus the mid-day check card, refreshed by the three routines): https://claude.ai/artifact/Kt4NUYMt5fGiE2U3j1v1cj
- User manual (Claude Docs): https://claude.ai/artifact/SmpGy7mJL8PmKh7GqivdyE (id d0b653aa-7795-4f40-a063-8b8afb279e65) — updated 2 Nov to **v3.3.4** (dashboard layouts, "Timing row" subsection, Layout / Show timing estimates in the settings table, v3.3.3 defaults), with the sections "Test results (2 Nov 2026)" (timeframes, daily grade filter, gold) and "Daily swing routine" (grades A and B since 3 Nov; rules, the schedule 08:12 + 12:45 + 17:15 MYT, evening report, "The mid-day check" with its statuses (6 Nov), the SZones Brief dashboard parts incl. mid-day check and track record, Claude's review, TradingView settings to mirror it). Daily loss limit confirmed by tIn on 2 Nov: 2 consecutive losses or −3R, no weekly cap.

---

## 8. Open items / next steps

1. Weekday routines (section 6): 08:12 MYT pre-open check, 12:45 MYT mid-day check and 17:15 MYT evening run — daily swing paper record, each refreshing the SZones Brief dashboard. Data access working since 2 Nov (section 4). For EODHD, tIn adds `EODHD_API_KEY` as an environment secret.
2. Collect ≥50 scored trades before judging Claude's reviews.
3. (done 2 Nov) Manual updated to v3.3.4.
4. (done 2 Nov) 1H with overnight holds: least bad (−0.11R net, +0.03R gross) but still negative. Daily swing tested 2 Nov: break-even (+0.003R, t = 0.10), no edge (section 3).
5. Refresh the Shariah list after 27 Nov 2026.
6. (done 2 Nov) Gold tested: no significant edge at any timeframe; 4H/daily positive but far behind holding gold (section 3).

---

## 9. Files in this bundle

| Path | What |
|---|---|
| `SZONES_HANDOFF.md` | This file (save as `CLAUDE.md`) |
| `pine/SZones_v3_3_4.pine` | Current indicator (daily swing grade A defaults, Compact dashboard, Timing row) |
| `pine/SZones_v3_3_3.pine` | Previous version (same defaults, old dashboard) |
| `pine/SZones_v3_3_2.pine` | Previous version (same logic, old defaults) |
| `python/szones_v33.py` | Python port of the v3.3 rules (backtests, scans) |
| `python/szones_data.py` | Data fallback chain: Yahoo → EODHD → TradingView (+ `local` replay) |
| `python/vendor/tvDatafeed/` | Vendored, reviewed tvDatafeed 2.1.0 (MIT) for the TradingView source |
| `requirements.txt` | PyPI dependencies: numpy, websocket-client |
| `python/szones_daily.py` | Daily fetch + evaluate + watchlist; writes `snapshots.json` and `trades.json` |
| `python/szones_swing.py` | Daily swing routine (scheduled): signals for the next open, positions, closed trades, watch list |
| `python/szones_midday.py` | Mid-day check of the morning session against the pre-open plan (section 6) |
| `python/szones_brief.py` | Builds the SZones Brief dashboard's data files from a run (section 6, Brief refresh) |
| `python/szones_scorecard.py` | Scores Claude's reviews against the trades that follow (multi-day aware); `--json` gives counts for the dashboard |
| `dashboard/index.html` | Source of the SZones Brief dashboard page (stored in the artifact as `files/index.html`) |
| `runs/swing_D/`, `runs/premarket_T/`, `reviews/D.json` | Evening and pre-open daily swing outputs (with `brief/` files) and Claude's calls (section 6); `runs/run_D/` = old 15m runs |
| `data/shariah_sc_list_2026-05-29.json` | Shariah status per counter (SC list effective 29 May 2026) |
