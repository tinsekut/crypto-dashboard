/**
 * SZones AI relay · TradingView webhook → Claude → Telegram, logged to this Google Sheet.
 *
 * Setup (once):
 *  1. Create a Google Sheet → Extensions → Apps Script → paste this file.
 *  2. Project Settings → Script properties:
 *       ANTHROPIC_API_KEY   your Claude API key (console.anthropic.com)
 *       TELEGRAM_TOKEN      bot token from @BotFather
 *       TELEGRAM_CHAT_ID    your chat id
 *       WEBHOOK_SECRET      same text as "Shared secret" in SZones
 *       MODEL               optional: claude-haiku-5-5 (default, fast) or claude-sonnet-5-5
 *  3. Run setup() once and approve the permissions.
 *  4. Deploy → New deployment → Web app → Execute as: Me, Access: Anyone → copy the /exec URL.
 *  5. TradingView: one alert per chart, Condition = SZones → "Any alert() function call",
 *     Notifications → Webhook URL = the /exec URL. Turn on "Send JSON snapshots" in SZones.
 *
 * Flow: doPost only queues (TradingView expects a fast reply); processQueue runs every minute,
 * asks Claude, sends Telegram, logs every call. scorecard() measures whether Claude adds value.
 */

const Q_SHEET = 'Queue';
const LOG_SHEET = 'Calls';
const SCORE_SHEET = 'Scorecard';
const LOG_HEAD = ['logged', 'bar_time', 'id', 'event', 'ticker', 'tf', 'dir', 'grade', 'price',
  'call', 'p_tp1', 'next_move', 'horizon_bars', 'key_level', 'invalidation', 'reason', 'model',
  'r', 'tp1_hit', 'exit'];

const SYSTEM = [
  'You review one snapshot from the SZones TradingView indicator for a Malaysian retail trader (Bursa stocks or gold, intraday).',
  'Use only the numbers in the snapshot. Do not invent news, prices or fundamentals.',
  'The trader\'s rules: long only when long_only is true; enter only on a candle CLOSE between plan.trigger and plan.max_entry;',
  'skip when plan.net_tp1_r is below 0.25; stop and targets come from the actual entry; flatten by the session end.',
  'Base rates from a 60-session test on 57 Shariah Bursa counters: average about -0.2R per trade after costs, TP1 reached on roughly 45% of trades.',
  'Stay close to that base rate unless the snapshot gives concrete reasons (structure, discount/premium, FVG, barriers, squeeze, distance to liquidity, stats).',
  'Events: WATCH = price near or inside a buy/sell zone, no signal yet. SIGNAL = an entry triggered at this bar close. POI = price reached the order block. ',
  'For WATCH and POI use WAIT unless the setup is already invalid (then SKIP). TAKE only applies to SIGNAL.',
  'Reply with ONLY one JSON object, no other text:',
  '{"call":"TAKE|SKIP|WAIT","p_tp1":0.0-1.0,"next_move":"up|down|sideways","horizon_bars":1-8,',
  '"key_level":number,"invalidation":number,"reason":"max 160 characters, plain English"}'
].join(' ');

// ─── Webhook entry: queue and return fast ───────────────────────────────
function doPost(e) {
  const raw = (e && e.postData && e.postData.contents) || '';
  let p;
  try { p = JSON.parse(raw); } catch (err) { return out_('ignored: not JSON'); }   // plain-text SZones alerts
  if (!p || p.secret !== prop_('WEBHOOK_SECRET')) return out_('forbidden');
  delete p.secret;
  const lock = LockService.getScriptLock();
  lock.waitLock(10000);
  try {
    sheet_(Q_SHEET, ['received', 'payload']).appendRow([new Date(), JSON.stringify(p)]);
  } finally {
    lock.releaseLock();
  }
  return out_('queued');
}

// ─── Every-minute worker ─────────────────────────────────────────────────
function processQueue() {
  const lock = LockService.getScriptLock();
  if (!lock.tryLock(5000)) return;
  try {
    const q = sheet_(Q_SHEET, ['received', 'payload']);
    const n = q.getLastRow() - 1;
    if (n < 1) return;
    const rows = q.getRange(2, 1, n, 2).getValues();
    q.deleteRows(2, n);
    rows.forEach(function (r) {
      try { handle_(JSON.parse(r[1])); } catch (err) { console.error('handle failed: ' + err + ' · ' + r[1]); }
    });
  } finally {
    lock.releaseLock();
  }
}

function handle_(p) {
  if (p.event === 'EXIT') return recordExit_(p);
  const ai = askClaude_(p);
  const log = sheet_(LOG_SHEET, LOG_HEAD);
  log.appendRow([new Date(), p.time, p.id, p.event, p.ticker, p.tf, p.dir, p.grade, p.price,
    ai.call, ai.p_tp1, ai.next_move, ai.horizon_bars, ai.key_level, ai.invalidation, ai.reason, ai.model,
    '', '', '']);
  telegram_(format_(p, ai));
}

// ─── Claude call ─────────────────────────────────────────────────────────
function askClaude_(p) {
  const model = prop_('MODEL') || 'claude-haiku-5-5';
  const body = {
    model: model,
    max_tokens: 400,
    system: SYSTEM,
    messages: [{ role: 'user', content: 'SZones snapshot (JSON):\n' + JSON.stringify(p) }]
  };
  const res = UrlFetchApp.fetch('https://api.anthropic.com/v1/messages', {
    method: 'post',
    contentType: 'application/json',
    muteHttpExceptions: true,
    headers: { 'x-api-key': prop_('ANTHROPIC_API_KEY'), 'anthropic-version': '2023-06-01' },
    payload: JSON.stringify(body)
  });
  if (res.getResponseCode() !== 200) {
    return { call: 'ERROR', model: model, reason: ('HTTP ' + res.getResponseCode() + ' ' + res.getContentText()).slice(0, 180) };
  }
  const j = JSON.parse(res.getContentText());
  const text = (j.content || []).filter(function (c) { return c.type === 'text'; }).map(function (c) { return c.text; }).join('');
  const m = text.match(/\{[\s\S]*\}/);
  try {
    const a = JSON.parse(m ? m[0] : text);
    a.model = model;
    if (typeof a.p_tp1 === 'number') a.p_tp1 = Math.max(0, Math.min(1, a.p_tp1));
    return a;
  } catch (err) {
    return { call: 'ERROR', model: model, reason: 'Unreadable reply: ' + text.slice(0, 150) };
  }
}

// ─── Exit: attach the result to the matching SIGNAL row ──────────────────
function recordExit_(p) {
  const log = sheet_(LOG_SHEET, LOG_HEAD);
  const n = log.getLastRow() - 1;
  const t = p.trade || {};
  if (n >= 1) {
    const ids = log.getRange(2, 3, n, 2).getValues();   // id, event
    for (let i = ids.length - 1; i >= 0; i--) {
      if (ids[i][0] === p.id && ids[i][1] === 'SIGNAL') {
        log.getRange(i + 2, 18, 1, 3).setValues([[t.r, t.tp1_hit ? 1 : 0, t.exit]]);
        break;
      }
    }
  }
  telegram_('🏁 ' + p.ticker + ' (' + p.tf + ') exit: ' + t.exit + ' · ' + (t.r >= 0 ? '+' : '') + Number(t.r).toFixed(2) + 'R after costs');
}

// ─── Is Claude adding value? ─────────────────────────────────────────────
function scorecard() {
  const log = sheet_(LOG_SHEET, LOG_HEAD);
  const n = log.getLastRow() - 1;
  if (n < 1) return;
  const rows = log.getRange(2, 1, n, LOG_HEAD.length).getValues()
    .filter(function (r) { return r[3] === 'SIGNAL' && r[17] !== ''; });
  const grp = function (name, list) {
    if (!list.length) return [name, 0, '', '', '', ''];
    const R = list.map(function (r) { return Number(r[17]); });
    const hit = list.map(function (r) { return Number(r[18]); });
    const p = list.map(function (r) { return Number(r[10]); });
    const avg = R.reduce(function (a, b) { return a + b; }, 0) / R.length;
    const win = R.filter(function (x) { return x > 0; }).length / R.length;
    const base = hit.reduce(function (a, b) { return a + b; }, 0) / hit.length;
    const brier = p.reduce(function (s, x, i) { return s + Math.pow(x - hit[i], 2); }, 0) / p.length;
    const brierBase = hit.reduce(function (s, h) { return s + Math.pow(base - h, 2); }, 0) / hit.length;
    return [name, list.length, avg, win, brier, brierBase];
  };
  const out = [['Group', 'Trades', 'Avg R', 'Win %', 'Brier (Claude)', 'Brier (base rate)'],
    grp('All signals', rows),
    grp('Claude TAKE', rows.filter(function (r) { return r[9] === 'TAKE'; })),
    grp('Claude SKIP', rows.filter(function (r) { return r[9] === 'SKIP'; }))];
  const s = sheet_(SCORE_SHEET, out[0]);
  s.clearContents();
  s.getRange(1, 1, out.length, out[0].length).setValues(out);
  s.getRange(out.length + 2, 1).setValue('Claude helps only if TAKE beats All signals on Avg R over 50+ trades, and its Brier score is lower than the base-rate Brier.');
}

// ─── Helpers ─────────────────────────────────────────────────────────────
function format_(p, ai) {
  const side = p.dir === 1 ? 'LONG' : p.dir === -1 ? 'SHORT' : '';
  const pl = p.event === 'SIGNAL' ? p.trade : p.plan;
  const lines = [
    '🤖 SZones · ' + p.ticker + ' (' + p.tf + ') · ' + p.event + ' ' + side + ' ' + (p.grade || '') + ' @ ' + p.price,
    p.event === 'SIGNAL'
      ? 'Plan: SL ' + pl.sl + ' · TP1 ' + pl.tp1 + ' · TP2 ' + pl.tp2
      : 'Trigger close ' + pl.trigger + '–' + pl.max_entry + ' · SL ' + pl.sl + ' · TP1 ' + pl.tp1 + ' · net TP1 ' + (pl.net_tp1_r == null ? '–' : Number(pl.net_tp1_r).toFixed(2)) + 'R',
    'Claude: ' + ai.call + (ai.p_tp1 != null ? ' · P(TP1) ' + Math.round(ai.p_tp1 * 100) + '%' : '') +
      (ai.next_move ? ' · next: ' + ai.next_move + (ai.horizon_bars ? ' (≤' + ai.horizon_bars + ' bars)' : '') : ''),
    (ai.key_level != null ? 'Key ' + ai.key_level : '') + (ai.invalidation != null ? ' · invalid beyond ' + ai.invalidation : ''),
    'Why: ' + (ai.reason || '–'),
    'Second opinion, not advice. Check the live chart.'
  ];
  return lines.filter(function (l) { return l; }).join('\n');
}

function telegram_(text) {
  const token = prop_('TELEGRAM_TOKEN');
  const chat = prop_('TELEGRAM_CHAT_ID');
  if (!token || !chat) return;
  UrlFetchApp.fetch('https://api.telegram.org/bot' + token + '/sendMessage', {
    method: 'post',
    contentType: 'application/json',
    muteHttpExceptions: true,
    payload: JSON.stringify({ chat_id: chat, text: text, disable_web_page_preview: true })
  });
}

function sheet_(name, head) {
  const ss = SpreadsheetApp.getActive();
  let s = ss.getSheetByName(name);
  if (!s) {
    s = ss.insertSheet(name);
    s.appendRow(head);
    s.setFrozenRows(1);
  }
  return s;
}

function prop_(k) { return PropertiesService.getScriptProperties().getProperty(k); }
function out_(t) { return ContentService.createTextOutput(t); }

// ─── One-time setup and a test you can run from the editor ───────────────
function setup() {
  sheet_(Q_SHEET, ['received', 'payload']);
  sheet_(LOG_SHEET, LOG_HEAD);
  ScriptApp.getProjectTriggers().forEach(function (t) {
    if (t.getHandlerFunction() === 'processQueue') ScriptApp.deleteTrigger(t);
  });
  ScriptApp.newTrigger('processQueue').timeBased().everyMinutes(1).create();
}

function testClaude() {
  const sample = {
    v: '3.3.2', event: 'WATCH', id: 'DIALOG-0', ticker: 'DIALOG', tf: '15', time: '2026-11-02 09:45', price: 2.14, dir: 1,
    grade: 'B', reasons: 'BOS ', trend: { bias: 1, htf_on: true, htf_bull: true, struct: 1, choch: false, discount: false, squeeze: false },
    zones: { hit: 0, z1: [2.12, 2.13], dist_z1_atr: 0.6 }, plan: { trigger: 2.13, max_entry: 2.13, sl: 2.10, tp1: 2.15, net_tp1_r: 0.49 },
    smc: { maj: 1, msb: 2.09, poi: [2.08, 2.09], idm: 2.16, leg_ext: 2.18, ext: 2.24, bsl: 2.16, ssl: 2.09 },
    stats: { n: 3, avg_r: 0.33 }, cost_pct: 0.25, long_only: true
  };
  const ai = askClaude_(sample);
  console.log(JSON.stringify(ai));
  telegram_(format_(sample, ai));
}
