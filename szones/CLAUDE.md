# SZones — project handoff for Claude Code

**As of:** Mon 2 Nov 2026, 07:36 MYT (pre-market). Latest data: Fri 30 Oct close.
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
| **v3.3.2 (current)** | AI bridge: JSON webhook snapshots (WATCH / SIGNAL / EXIT / POI) for a Claude relay |

Current script: `pine/SZones_v3_3_2.pine`.

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

- **Data:** Yahoo chart API, `https://query1.finance.yahoo.com/v8/finance/chart/{code}.KL?interval=15m&range=60d` and `interval=1d&range=1y`. About 15 minutes delayed. Lunch bars 12:30–14:30 are removed.
- **Universe:** `data/shariah_sc_list_2026-05-29.json` (SC list effective 29 May 2026, via a mirror site). 59 compliant in the file; MYEG (0138) and DNEX (7033) excluded for bad Yahoo data → 57. **Next SC list update: Fri 27 Nov 2026** — refresh the file then.
- **Non-compliant (excluded):** YTL Power, YTL, Yinson, Hap Seng, Astro, Maybank, Genting, Genting Malaysia, KLCC.
- **Watchlist filter:** grade A/B, valid trigger window (max entry ≥ trigger), net TP1 ≥ 0.25R, 20-day turnover ≥ RM10m, no squeeze.
- **Post-close report format used so far:** day verdict + KLCI context; signals table (time = signal-candle close, grade, entry, result, R); previous watchlist outcome; lesson; next watchlist table (close, trigger, SL, TP1/TP2, grade); grade-A deeper pullbacks; skip list; sources.

---

## 5. Today (Mon 2 Nov) — pre-market plan

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

## 6. AI bridge (built, not yet set up by tIn)

Pine cannot call Claude. Flow: SZones `alert()` JSON → TradingView webhook → Google Apps Script relay (`relay/SZones_AI_Relay.gs`) queues it → every-minute trigger asks Claude (Messages API, `claude-haiku-5-5` default, `anthropic-version: 2023-06-01`) → Telegram message → logged to a Sheet. `scorecard()` compares Claude TAKE vs all signals and Brier score vs base rate. Latency ~1–2 min after the 15m close.

Setup checklist: `docs/SZones_AI_Setup_Checklist.xlsx`. Script properties: `ANTHROPIC_API_KEY` (not created yet), `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`, `WEBHOOK_SECRET` (new; must equal the indicator's Shared secret), `MODEL` (optional). Requires a TradingView plan with webhooks + 2FA.

**Secrets:** never write keys or tokens into files, code or chat. A usable Telegram bot token + chat ID exist in tIn's old Polymarket bot `.env` (shared in a March 2026 chat); recommend `/revoke` and a fresh token. That same `.env` exposed a Polymarket wallet private key — advise moving funds/rotating if still used.

---

## 7. Documents and links

- User manual (Claude Docs): https://claude.ai/code/artifact/d0b653aa-7795-4f40-a063-8b8afb279e65 — covers up to **v3.3**. Needs a v3.3.1/v3.3.2 update: plot-limit fix (labels), AI bridge section, Friday's revised numbers. tIn left an open comment on the daily loss limit (2 losses or −3R) — unanswered.

---

## 8. Open items / next steps

1. Post-close evaluation today (`--closed` after 17:00), same report format; offer a weekday 17:15 scheduled task.
2. Update the manual to v3.3.2.
3. Help tIn set up the AI relay (Claude API key, Telegram, Apps Script deploy, TradingView alert); then run ≥50 signals before judging Claude via `scorecard()`.
4. Test the same rules on **1H charts with overnight holds** (the grading was tested on 1H; 15m intraday is negative after costs).
5. Refresh the Shariah list after 27 Nov 2026.
6. Gold (XAUUSD) has never been tested with these rules — validate before trusting grades there.

---

## 9. Files in this bundle

| Path | What |
|---|---|
| `SZONES_HANDOFF.md` | This file (save as `CLAUDE.md`) |
| `pine/SZones_v3_3_2.pine` | Current indicator (full script) |
| `relay/SZones_AI_Relay.gs` | Google Apps Script relay (webhook → Claude → Telegram → Sheet) |
| `python/szones_v33.py` | Python port of the v3.3 rules (backtests, scans) |
| `python/szones_daily.py` | Daily fetch + evaluate + watchlist |
| `data/shariah_sc_list_2026-05-29.json` | Shariah status per counter (SC list effective 29 May 2026) |
| `docs/SZones_AI_Setup_Checklist.xlsx` | Setup checklist for the AI bridge |
