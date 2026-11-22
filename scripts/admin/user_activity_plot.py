"""Proofreaders' editing time as a self-contained HTML page (David 2026-11-12: "How are the 2.9 hr distributed
throughout the days? Could you find a nice way to plot this?"; 2026-11-13: the same page for several people).
Network-free: reads the newest figures/misc/*_user_activity_ops.csv (scripts/admin/user_activity_ops.py) and
*_cave_user_names.csv. One argument per person; a person with several CAVE accounts = ids joined by commas (1019,7220),
their edits pooled before sessions are cut.
Sessions as in user_activity_ops.py: edits less than --gap (30) minutes apart form one session, timed first to last edit
+ 2 minutes (a lower bound: looking without editing is not counted). The page computes everything in the browser so
its time-zone menu (UTC offset) re-splits sessions at local midnight:
  - with several people: a comparison table (click a row) and a person menu
  - stat tiles (hours per week, total, active days, sessions, median session)
  - every calendar day as a row: its sessions across the 24 h, its total, and each week's total
  - hours by time of day and by weekday; a weekly table
Writes figures/misc/<stamp>_user_activity_<first id of each person, joined by _>.html.
Usage: user_activity_plot.py USER_ID[,USER_ID] [USER_ID ...] [--ops CSV] [--gap 30] [--tz HOURS]
       user_activity_plot.py --all [--min-ops N] [--ops CSV] [--names CSV] [--out PAGE.html]   (faab-ol-bot's daily page)"""
import argparse
import glob
import html
import json
import os

import pandas as pd

from faab.config import FIGURES
from faab.partners.sheet import utc_stamp

MISC = os.path.join(FIGURES, "misc")


def newest(pattern):
    hits = sorted(glob.glob(os.path.join(MISC, pattern)))
    if not hits:
        raise SystemExit(f"no figures/misc/{pattern}")
    return hits[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("users", nargs="*", help="one per person; several accounts of one person joined by commas")
    ap.add_argument("--all", action="store_true", help="everyone with edits in the ops table (accounts with the same "
                    "name pooled as one person), most edits first")
    ap.add_argument("--min-ops", type=int, default=1, help="with --all: leave out people with fewer edits")
    ap.add_argument("--ops")
    ap.add_argument("--names", help="cave_user_names.csv (default: the newest figures/misc/*_cave_user_names.csv)")
    ap.add_argument("--out", help="the page's path (default: a stamped file in figures/misc)")
    ap.add_argument("--gap", type=float, default=30.0)
    ap.add_argument("--tz", type=float, default=0.0, help="initial UTC offset in hours (the page has a menu)")
    a = ap.parse_args()
    ops_csv = a.ops or newest("*_user_activity_ops.csv")
    d = pd.read_csv(ops_csv)
    names = pd.read_csv(a.names or newest("*_cave_user_names.csv"))
    ncol = [c for c in names.columns if "name" in c.lower()][0]
    ucol = [c for c in names.columns if c.lower() in ("user", "user_id", "id")][0]
    name_of = names.set_index(ucol)[ncol]
    if a.all:  # one spec per person: accounts with the same (non-empty) name together
        counts = d.user.value_counts()
        groups = {}
        for u in counts.index:
            nm = str(name_of.get(u, "")).strip()
            groups.setdefault(nm.lower() if nm and nm.lower() != "nan" else f"#{u}", []).append(int(u))
        a.users = [",".join(map(str, ids)) for ids in sorted(groups.values(), key=lambda ids: -sum(counts[i] for i in ids))
                   if sum(counts[i] for i in ids) >= a.min_ops]
    if not a.users:
        raise SystemExit("give user ids or --all")
    people = []
    for spec in a.users:
        ids = [int(x) for x in spec.split(",") if x.strip()]
        t = pd.to_datetime(d[d.user.isin(ids)].timestamp, format="ISO8601", utc=True).sort_values()
        if t.empty:
            raise SystemExit(f"user {spec} has no edits in {ops_csv}")
        name = str(name_of.get(ids[0], f"user {ids[0]}"))
        people.append(dict(users=ids, name=name, ops=[int(x.timestamp() * 1000) for x in t]))
        print(f"{name} ({', '.join(map(str, ids))}): {len(t)} edits {t.iloc[0]:%Y-%m-%d} .. {t.iloc[-1]:%Y-%m-%d}")
    data_end = os.path.basename(ops_csv)[:19]  # the run's stamp = end of the scanned window
    try:
        end_ms = int(pd.Timestamp(data_end.replace("d", "T", 1).replace("tUTC", ""), tz="UTC").timestamp() * 1000)
    except ValueError:  # a fixed file name (faab-ol-bot's daily run): the scan ended when the file was written
        end_ms = int(os.path.getmtime(ops_csv) * 1000)
    payload = dict(people=people, gap=a.gap, pad=2.0, tz=a.tz, end=end_ms, src=os.path.basename(ops_csv))
    tag = "_".join(str(p["users"][0]) for p in people)
    out = a.out or os.path.join(MISC, f"{utc_stamp()}_user_activity_{tag}.html")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    title = f"{people[0]['name']} editing time" if len(people) == 1 else "Proofreader editing time"
    page = PAGE.replace("__TITLE__", html.escape(title)).replace("__DATA__", json.dumps(payload))
    open(out, "w", encoding="utf-8").write(page)
    print(f"written {out}")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--axis:#c3c2b7;--ring:rgba(11,11,11,0.10);--s1:#2a78d6;--wkend:rgba(11,11,11,0.035);
--q1:#cde2fb;--q2:#9ec5f4;--q3:#6da7ec;--q4:#3987e5;--q5:#256abf;--q6:#184f95;--q7:#0d366b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,0.10);--s1:#3987e5;
--wkend:rgba(255,255,255,0.04);--q1:#104281;--q2:#184f95;--q3:#1c5cab;--q4:#256abf;--q5:#3987e5;--q6:#6da7ec;--q7:#9ec5f4}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,0.10);--s1:#3987e5;--wkend:rgba(255,255,255,0.04);
--q1:#104281;--q2:#184f95;--q3:#1c5cab;--q4:#256abf;--q5:#3987e5;--q6:#6da7ec;--q7:#9ec5f4}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 4px;font-weight:650}
h2{font-size:15px;margin:0 0 2px;font-weight:620}
.sub{color:var(--ink2);margin:0 0 16px;max-width:760px}
.note{color:var(--muted);font-size:12px;margin:2px 0 10px}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:0 0 16px}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:10px 14px;min-width:128px}
.tile b{display:block;font-size:24px;font-weight:650;line-height:1.15}
.tile span{color:var(--ink2);font-size:12px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:16px;margin:0 0 16px}
.pair{display:grid;grid-template-columns:2fr 1fr;gap:16px}
@media (max-width:720px){.pair{grid-template-columns:1fr}}
select{font:inherit;color:var(--ink);background:var(--surface);border:1px solid var(--ring);border-radius:8px;padding:4px 8px}
label{color:var(--ink2)}
svg{display:block;width:100%;height:auto;overflow:visible}
svg text{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
svg text.ink2{fill:var(--ink2)}
.tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--ring);
border-radius:8px;padding:6px 9px;font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.12);display:none;max-width:260px;z-index:5}
.tip strong{font-weight:650}
.hit{fill:transparent;cursor:default}
details{margin-top:8px}summary{cursor:pointer;color:var(--ink2)}
table{border-collapse:collapse;margin-top:8px;font-variant-numeric:tabular-nums;font-size:13px}
th,td{padding:3px 12px 3px 0;text-align:right;border-bottom:1px solid var(--grid)}th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600}
</style></head><body><main>
<div id="cmp" class="card" hidden><h2>Editing time, side by side</h2>
<p class="note">Same rules for everyone (session window and time zone below). Click a row to show that person.</p>
<table id="cmpTbl"></table></div>
<div class="row" id="whoRow" hidden><label for="who">Proofreader</label><select id="who"></select></div>
<h1 id="h1"></h1>
<p class="sub" id="sub"></p>
<div class="row" id="tiles"></div>
<div class="row"><label for="tz">Time zone</label><select id="tz"></select>
<span class="note" style="margin:0">Edit times are stored in UTC. Pick the proofreader's own offset to see their local day.</span></div>
<div class="row"><label for="gap">Session window</label><select id="gap"></select>
<span class="note" style="margin:0">Edits closer together than this count as one stretch of work; a longer pause ends the session and is not counted.</span></div>
<div class="card"><h2>Every day: when the sessions were</h2>
<p class="note">One row per calendar day (weekends shaded). Bars = editing sessions across the 24 hours. Right: that day's hours, and each week's total.</p>
<svg id="days"></svg></div>
<div class="pair">
<div class="card"><h2>Time of day</h2><p class="note">Average hours per week spent in each hour of the day.</p><svg id="hours"></svg></div>
<div class="card"><h2>Day of the week</h2><p class="note">Average hours per week on each weekday.</p><svg id="wdays"></svg></div>
</div>
<div class="card"><h2>Per week</h2>
<details open><summary>Table</summary><table id="tbl"></table></details></div>
<p class="note" id="src"></p>
</main><div class="tip" id="tip"></div>
<script>
const D = __DATA__;
const H = 3600e3, DAY = 24 * H, MIN = 60e3;
const WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const MON = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const NS = "http://www.w3.org/2000/svg";
const tip = document.getElementById("tip");

function el(tag, attrs, parent, text) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}
function fmtH(h) { return h >= 1 ? h.toFixed(1) + " h" : Math.round(h * 60) + " min"; }
function hm(ms) { const d = new Date(ms); return String(d.getUTCHours()).padStart(2, "0") + ":" + String(d.getUTCMinutes()).padStart(2, "0"); }
function dayLabel(ms) { const d = new Date(ms); return WD[(d.getUTCDay() + 6) % 7] + " " + MON[d.getUTCMonth()] + " " + d.getUTCDate(); }
function showTip(ev, lines) {
  tip.replaceChildren();
  lines.forEach((ln, i) => { const p = document.createElement(i ? "div" : "strong"); p.textContent = ln; tip.appendChild(p); if (!i) tip.appendChild(document.createElement("br")); });
  tip.style.display = "block";
  const x = Math.min(ev.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
  const y = Math.min(ev.clientY + 14, window.innerHeight - tip.offsetHeight - 8);
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
function hideTip() { tip.style.display = "none"; }
function hover(node, lines) {
  node.addEventListener("pointermove", ev => showTip(ev, lines));
  node.addEventListener("pointerleave", hideTip);
}

// sessions (absolute ms), as user_activity_ops.py: a gap > gap minutes starts a new session; + D.pad minutes
function sessions(ops, gap) {
  const S = [];
  let a = ops[0], b = ops[0], n = 1;
  for (let i = 1; i < ops.length; i++) {
    if (ops[i] - b > gap * MIN) { S.push({ a, b: b + D.pad * MIN, n }); a = b = ops[i]; n = 1; }
    else { b = ops[i]; n++; }
  }
  S.push({ a, b: b + D.pad * MIN, n });
  return S;
}
// one person's totals under the current rules (for the side-by-side table)
function summary(p, gap, tz) {
  const S = sessions(p.ops, gap);
  const hrs = s => (s.b - s.a) / H;
  const total = S.reduce((u, s) => u + hrs(s), 0);
  const day0 = ms => Math.floor((ms + tz) / DAY) * DAY;
  const first = day0(p.ops[0]), start = first - ((new Date(first).getUTCDay() + 6) % 7) * DAY;
  const endDay = day0(Math.max(D.end, p.ops[p.ops.length - 1]));
  const nWeeks = Math.ceil((Math.round((endDay - start) / DAY) + 1) / 7);
  const recent = S.filter(s => s.a >= D.end - 28 * DAY).reduce((u, s) => u + hrs(s), 0) / 4;
  return { total, nWeeks, perWeek: total / nWeeks, recent, days: new Set(p.ops.map(day0)).size, n: S.length };
}
let P = D.people[0];
const whoSel = document.getElementById("who");
D.people.forEach((p, i) => { const op = document.createElement("option"); op.value = i; op.textContent = p.name; whoSel.appendChild(op); });
whoSel.addEventListener("change", render);
if (D.people.length > 1) { document.getElementById("whoRow").hidden = false; document.getElementById("cmp").hidden = false; }
const gapSel = document.getElementById("gap");
[5, 10, 15, 20, 30, 45, 60, 90, 120].forEach(g => {
  const op = document.createElement("option");
  op.value = g; op.textContent = g + " min" + (g === 30 ? " (report default)" : "");
  if (g === D.gap) op.selected = true;
  gapSel.appendChild(op);
});
gapSel.addEventListener("change", render);

const tzSel = document.getElementById("tz");
for (let o = -12; o <= 14; o += 0.5) {
  if (o % 1 && ![-9.5, -3.5, 3.5, 4.5, 5.5, 6.5, 9.5, 10.5].includes(o)) continue;
  const op = document.createElement("option");
  const hh = Math.trunc(Math.abs(o)), mm = Math.abs(o) % 1 ? ":30" : "";
  op.value = o; op.textContent = o === 0 ? "UTC" : "UTC" + (o > 0 ? "+" : "−") + hh + mm;
  if (o === D.tz) op.selected = true;
  tzSel.appendChild(op);
}
tzSel.addEventListener("change", render);

function compare(gap, tz) {
  const tb = document.getElementById("cmpTbl"); tb.replaceChildren();
  const rows = D.people.map((p, i) => ({ p, i, ...summary(p, gap, tz) })).sort((x, y) => y.perWeek - x.perWeek);
  const mx = Math.max(...rows.map(r => Math.max(r.perWeek, r.recent)), 0.1);
  const hr = tb.insertRow();
  ["proofreader", "hours per week since the first edit", "", "last 28 days, per week", "", "total", "weeks", "days with edits", "first edit", "last edit"].forEach(h => {
    const th = document.createElement("th"); th.textContent = h; hr.appendChild(th); });
  const bar = (cell, v) => { const b = document.createElement("div");
    b.style.cssText = `height:8px;border-radius:2px;background:var(--s1);width:${Math.max(2, 90 * v / mx)}px`; cell.appendChild(b); };
  rows.forEach(r => {
    const tr = tb.insertRow(); tr.style.cursor = "pointer";
    if (r.i === +whoSel.value) tr.style.fontWeight = 650;
    tr.addEventListener("click", () => { whoSel.value = r.i; render(); });
    tr.insertCell().textContent = r.p.name;
    tr.insertCell().textContent = fmtH(r.perWeek); bar(tr.insertCell(), r.perWeek);
    tr.insertCell().textContent = fmtH(r.recent); bar(tr.insertCell(), r.recent);
    [fmtH(r.total), r.nWeeks, r.days, dayLabel(r.p.ops[0] + tz), dayLabel(r.p.ops[r.p.ops.length - 1] + tz)].forEach(v => { tr.insertCell().textContent = v; });
  });
}

function render() {
  const tz = parseFloat(tzSel.value) * H;
  const gap = parseFloat(gapSel.value);
  P = D.people[+whoSel.value];
  if (D.people.length > 1) compare(gap, tz);
  const S = sessions(P.ops, gap);
  const L = ms => ms + tz;                       // local clock as a UTC-read ms value
  const day0 = ms => Math.floor(L(ms) / DAY) * DAY; // local midnight (local ms)
  const firstDay = day0(P.ops[0]);
  const wd0 = (new Date(firstDay).getUTCDay() + 6) % 7;
  const start = firstDay - wd0 * DAY;            // the Monday of the first week
  const endDay = day0(Math.max(D.end, P.ops[P.ops.length - 1]));
  const nDays = Math.round((endDay - start) / DAY) + 1;
  const nWeeks = Math.ceil(nDays / 7);
  const days = Array.from({ length: nDays }, (_, i) => ({ t: start + i * DAY, h: 0, edits: 0, ses: [] }));
  const hourBins = new Array(24).fill(0), wdBins = new Array(7).fill(0);
  S.forEach((s, k) => {
    let a = L(s.a); const b = L(s.b);
    while (a < b) {
      const mid = Math.floor(a / DAY) * DAY + DAY, e = Math.min(b, mid);
      const di = Math.round((Math.floor(a / DAY) * DAY - start) / DAY);
      const piece = { a, b: e, s, k };
      days[di].ses.push(piece); days[di].h += (e - a) / H;
      wdBins[di % 7] += (e - a) / H;
      let x = a;
      while (x < e) { const nh = Math.floor(x / H) * H + H, y = Math.min(e, nh); hourBins[new Date(x).getUTCHours()] += (y - x) / H; x = y; }
      a = e;
    }
    const di0 = Math.round((day0(s.a) - start) / DAY);
    days[di0].edits += s.n;
  });
  const total = days.reduce((u, d) => u + d.h, 0);
  const active = days.filter(d => d.edits).length;
  // weeks from the first edit's week through the data end (partial first / last weeks included, as in the report)
  const weeks = Array.from({ length: nWeeks }, (_, w) => {
    const ds = days.slice(w * 7, w * 7 + 7);
    return { t: start + w * 7 * DAY, h: ds.reduce((u, d) => u + d.h, 0), edits: ds.reduce((u, d) => u + d.edits, 0),
             active: ds.filter(d => d.edits).length, ses: S.filter(s => { const i = Math.round((day0(s.a) - start) / DAY); return i >= w * 7 && i < w * 7 + 7; }).length };
  });
  const lens = S.map(s => (s.b - s.a) / MIN).sort((p, q) => p - q);
  const med = lens[Math.floor((lens.length - 1) / 2)];

  document.getElementById("h1").textContent = P.name + ": editing time";
  document.getElementById("sub").textContent = `${P.ops.length.toLocaleString()} CAVE edits from ${dayLabel(L(P.ops[0]))} to ${dayLabel(L(P.ops[P.ops.length - 1]))}, 2026. ` +
    `Time counts editing sessions only: edits less than ${gap} minutes apart form one session, timed from the first to the last edit plus ${D.pad} minutes. ` +
    `Looking at cells without editing is not counted, so these hours are a lower bound.`;
  const tiles = document.getElementById("tiles"); tiles.replaceChildren();
  [[fmtH(total / nWeeks), `per week (${nWeeks} weeks)`], [fmtH(total), "in total"], [String(active), "days with edits"],
   [String(S.length), "sessions"], [Math.round(med) + " min", "median session"]].forEach(([v, l]) => {
    const t = document.createElement("div"); t.className = "tile";
    const b = document.createElement("b"); b.textContent = v; const sp = document.createElement("span"); sp.textContent = l;
    t.append(b, sp); tiles.appendChild(t);
  });

  // ---- day rows
  const svg = document.getElementById("days"); svg.replaceChildren();
  const RH = 8, GAP = 2, TOP = 38, LEFT = 92, TW = 560, BX = LEFT + TW + 22, BW = 110, WX = BX + BW + 18, W = WX + 60;
  const maxDay = Math.max(1, Math.ceil(Math.max(...days.map(d => d.h))));
  const Hgt = TOP + nDays * (RH + GAP) + 6;
  svg.setAttribute("viewBox", `0 0 ${W} ${Hgt}`);
  el("text", { x: LEFT, y: TOP - 26, class: "ink2" }, svg, "time of day (" + tzSel.selectedOptions[0].textContent + ")");
  for (let h = 0; h <= 24; h += 3) {
    const x = LEFT + TW * h / 24;
    el("line", { x1: x, x2: x, y1: TOP - 4, y2: Hgt - 6, stroke: "var(--grid)", "stroke-width": 1 }, svg);
    el("text", { x, y: TOP - 9, "text-anchor": "middle" }, svg, String(h).padStart(2, "0") + ":00");
  }
  el("text", { x: BX, y: TOP - 26, class: "ink2" }, svg, "hours that day");
  const hstep = maxDay > 6 ? 2 : 1;
  for (let h = hstep; h <= maxDay; h += hstep) {
    const x = BX + BW * h / maxDay;
    el("line", { x1: x, x2: x, y1: TOP - 4, y2: Hgt - 6, stroke: "var(--grid)", "stroke-width": 1 }, svg);
    el("text", { x, y: TOP - 9, "text-anchor": "middle" }, svg, h + " h");
  }
  el("text", { x: WX, y: TOP - 26, class: "ink2" }, svg, "week total");
  days.forEach((d, i) => {
    const y = TOP + i * (RH + GAP), wd = i % 7;
    if (wd >= 5) el("rect", { x: LEFT - 2, y: y - GAP / 2, width: TW + 4, height: RH + GAP, fill: "var(--wkend)" }, svg);
    if (wd === 0) {
      if (i) el("line", { x1: 8, x2: W, y1: y - GAP / 2, y2: y - GAP / 2, stroke: "var(--axis)", "stroke-width": 1 }, svg);
      el("text", { x: LEFT - 8, y: y + RH, "text-anchor": "end", class: "ink2" }, svg, MON[new Date(d.t).getUTCMonth()] + " " + new Date(d.t).getUTCDate());
      const wk = weeks[i / 7];
      const wy = y + Math.min(7, nDays - i) * (RH + GAP) / 2 + 4;
      el("text", { x: WX, y: wy, class: "ink2" }, svg, fmtH(wk.h));
    }
    d.ses.forEach(p => {
      const x0 = LEFT + TW * ((p.a - d.t) / DAY), x1 = LEFT + TW * ((p.b - d.t) / DAY);
      const r = el("rect", { x: x0, y, width: Math.max(2, x1 - x0), height: RH, rx: 2, fill: "var(--s1)" }, svg);
      const hit = el("rect", { x: x0 - 3, y: y - 1, width: Math.max(2, x1 - x0) + 6, height: RH + 2, class: "hit" }, svg);
      const lines = [dayLabel(d.t), `${hm(L(p.s.a))}–${hm(L(p.s.b))} (${fmtH((p.s.b - p.s.a) / H)})`, `${p.s.n} edits in this session`];
      hover(hit, lines);
    });
    if (d.h > 0) {
      const w = Math.max(2, BW * d.h / maxDay);
      el("rect", { x: BX, y, width: w, height: RH, rx: 2, fill: "var(--s1)", opacity: 0.55 }, svg);
      const hit = el("rect", { x: BX, y: y - 1, width: BW, height: RH + 2, class: "hit" }, svg);
      hover(hit, [dayLabel(d.t), `${fmtH(d.h)} in ${d.ses.length} session${d.ses.length > 1 ? "s" : ""}`, `${d.edits} edits`]);
    }
  });
  el("line", { x1: BX, x2: BX, y1: TOP - 4, y2: Hgt - 6, stroke: "var(--axis)", "stroke-width": 1 }, svg);

  // ---- bar charts (average hours per week)
  function bars(id, vals, labels, tipLabel, every) {
    const s = document.getElementById(id); s.replaceChildren();
    const n = vals.length, Wd = n > 10 ? 560 : 260, Ht = 180, T = 10, B = 26, Lm = 34, bw = (Wd - Lm) / n;
    s.setAttribute("viewBox", `0 0 ${Wd} ${Ht}`);
    const mx = Math.max(...vals), step = mx > 1.5 ? 0.5 : mx > 0.6 ? 0.2 : 0.1, top = Math.max(step, Math.ceil(mx / step) * step);
    for (let v = 0; v <= top + 1e-9; v += step) {
      const y = Ht - B - (Ht - B - T) * v / top;
      el("line", { x1: Lm, x2: Wd, y1: y, y2: y, stroke: v ? "var(--grid)" : "var(--axis)", "stroke-width": 1 }, s);
      el("text", { x: Lm - 6, y: y + 4, "text-anchor": "end" }, s, v.toFixed(1));
    }
    vals.forEach((v, i) => {
      const x = Lm + i * bw + 1, w = bw - 2, h = (Ht - B - T) * v / top, y = Ht - B - h;
      if (h > 0) el("path", { d: `M${x},${Ht - B} V${y + Math.min(4, h)} q0,-${Math.min(4, h)} ${Math.min(4, w / 2)},-${Math.min(4, h)} H${x + w - Math.min(4, w / 2)} q${Math.min(4, w / 2)},0 ${Math.min(4, w / 2)},${Math.min(4, h)} V${Ht - B} Z`, fill: "var(--s1)" }, s);
      if (i % every === 0) el("text", { x: x + w / 2, y: Ht - 8, "text-anchor": "middle" }, s, labels[i]);
      const hit = el("rect", { x: x - 1, y: T, width: bw, height: Ht - B - T, class: "hit" }, s);
      hover(hit, [tipLabel(i), `${fmtH(v)} per week on average`, `${fmtH(v * nWeeks)} in total`]);
    });
  }
  bars("hours", hourBins.map(v => v / nWeeks), hourBins.map((_, i) => String(i).padStart(2, "0")),
       i => `${String(i).padStart(2, "0")}:00–${String(i + 1).padStart(2, "0")}:00`, 3);
  bars("wdays", wdBins.map(v => v / nWeeks), WD, i => WD[i], 1);

  // ---- weekly table
  const tb = document.getElementById("tbl"); tb.replaceChildren();
  const hr = tb.insertRow(); ["week of", "hours", "days with edits", "sessions", "edits"].forEach(h => { const th = document.createElement("th"); th.textContent = h; hr.appendChild(th); });
  weeks.forEach(w => { const r = tb.insertRow(); [dayLabel(w.t), w.h.toFixed(1), w.active, w.ses, w.edits].forEach(v => { r.insertCell().textContent = v; }); });
  const r = tb.insertRow(); ["mean", (total / nWeeks).toFixed(1), (active / nWeeks).toFixed(1), (S.length / nWeeks).toFixed(1), Math.round(P.ops.length / nWeeks)].forEach(v => { const c = r.insertCell(); c.textContent = v; c.style.fontWeight = 650; });
  document.getElementById("src").textContent = `CAVE ${P.users.length > 1 ? "accounts" : "user"} ${P.users.join(", ")}${P.users.length > 1 ? " (one person, edits pooled)" : ""}. Source: ${D.src} (edit log through ${new Date(D.end).toISOString().slice(0, 16).replace("T", " ")} UTC).`;
}
render();
</script></body></html>
"""

if __name__ == "__main__":
    main()
