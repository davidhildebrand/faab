"""Bring the tracking tabs in line with SeaTable (aedes_main), fix roots and dates, apply hand edits, and (optionally)
add the left-eye PRs to the PRs tab; writes one .xlsx per changed tab (import via File > Import > Insert new sheet).

Rules (David, 2026-10-30):
  - a row is 'in SeaTable' when the live root at its xyz (or, failing that, its root_id_curr) is the live root of a
    SeaTable row (figures/*_seatable_match.tsv from seatable_match.py);
  - status: SeaTable's, except that our 'checked' and 'proofread' are kept (both count as SeaTable 'adequate');
  - proofreader: the sheet's names, plus every SeaTable proofreader not already credited, as SeaTable writes them
    (initials matching a credited sheet name, e.g. KT for Kristo T., count as credited);
  - SeaTable column: '<mark>: <SeaTable label>' (label = superclass / class / subclass / subsubclass / type), mark
    'added' (or 'added (AK)') where the sheet said our team added it, else 'present' (also for NEW / NEW-DUP rows);
  - nothing from SeaTable goes into notes, and the annotation / class columns keep the sheet's values (David,
    2026-10-30: notes were too verbose and repeated the sheet; the label lives in the SeaTable column only). Replaced
    values are in the change log;
Always: root_id_curr -> the root at the xyz when root_id_curr is no longer current; rows whose root_id_curr is current
but elsewhere (MISMATCH) are highlighted and get the short note 'xyz off the cell (on <root at xyz>)', nothing moved
(notes are for the proofreaders: generated notes stay short, David 2026-10-30); notes our scripts wrote earlier are
shortened by faab.partners.short_notes (proofreaders' text kept verbatim; 'duplicate target of <id>' dropped, never a
row number - rows move, David 2026-11-12); statuses lowercase ('stoppped' -> 'stopped');
stale notes (faab.partners.stale_notes, 2026-11-07/08; notes are only ever cleared, never added or copied): an
earlier 'xyz off the cell' flag and the proofreader's 'resolved' reply go when the point is on the cell again
(highlight cleared), replies / reports of work done ('xyz coordinates and link updated') once the point is checked on
the cell, 'X ..., not merged' once X's rows are on the same cell, 'missing soma' notes when the cell carries one
nucleus now (nucleus table mapped to today's roots; --no-stale-notes skips it);
'added' typed into checker moves to SeaTable; dates typed day-month-year (2020-10-26 for 20-10-2026, text 27-10-26)
are corrected; text dates become dates and other text in date columns moves to notes. Hand edits (--edits JSON:
tab, row or root, column, value, optional append; column 'fill' with value clear / highlight) are applied next. --reconcile then brings the
rows of one live cell in line (faab.partners.reconcile, David 2026-10-31: same-block rows of a correctly merged cell
collapse into one, every row of a cell takes the best status and carries the same notes; --sizes = a
cell_consistency.py TSV, whose L2 chunk counts say which fragment-noted cells are whole now). --left-rows adds the
'>>> additional L side putative PRs' section (prs_left_rows.py) at the end of the PRs tab, styled like the R side.
CAVE read-only: one is_latest_roots call (+ with --reconcile, a synapse recount for collapsed rows from the local
snapshot, faab.synsnap, with its CAVE catch-up to now). Writes
figures/<stamp>_FAABopticlobetargets-<tab>.xlsx per changed tab, the full workbook in tmp/, and
figures/<stamp>_seatable_sync_changes.tsv (every changed cell with the reason; row numbers before collapsing).
Usage: seatable_sync.py BOOK.xlsx [--edits E.json] [--reconcile --sizes C.tsv] [--left-rows L.tsv --left-link URL]"""
import argparse
import copy
import datetime
import glob
import json
import os
import re
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import openpyxl
import pandas as pd
from openpyxl.styles import Font, PatternFill

from faab import synsnap
from faab.clients.cave import client as cave_client, throttle
from faab.config import FIGURES, TMP
from faab.partners import stale_notes as SN
from faab.partners.sheet import cell_text as s, scan_book, utc_stamp
from faab.partners.short_notes import shorten
from faab.pr import prs_tab as T
from faab.config import fig

HL = "FFFFF9C4"
KEEP = ("checked", "proofread")
STATUSES = ("checked", "proofread", "adequate", "review", "in progress", "stopped", "candidate")
SPELL = {"stoppped": "stopped"}
TABS = ("PRg1toT4T5", "PRs", "PRg1hop1", "PRg1hop2", "PRg2hop1", "Mi1s", "LMCs",  # LMCs: 2026-11-07, proofreaders work there
        "T4T5s")  # 2026-11-12, scripts/sheets/t4t5_tab.py


def initials(name):
    w = re.findall(r"[A-Za-z]+", name)
    return (w[0][0] + w[-1][0]).upper() if len(w) >= 2 else name.strip().upper()


def seatable_mark(cur):
    """'added' marks keep who added the cell ('added-AK' / 'added (AK)' -> 'added (AK)'; a SeaTable status in
    parentheses, as in 'added (proofread)', is dropped); anything else ('', 'in SeaTable', 'tba') -> 'present'"""
    c = cur.split(":")[0].strip()
    if c.lower().startswith("added"):
        who = re.fullmatch(r"added\s*(?:-|\()\s*([A-Za-z]{2,4})\)?", c, re.I)
        return f"added ({who.group(1).upper()})" if who else "added"
    return "present"


class Book:
    def __init__(self, wb):
        self.wb, self.log, self.touched = wb, [], set()
        self._hdr = {}  # tab -> (max_row when mapped, {row: header row at or above it})

    def put(self, tab, r, c, new, why):
        cell = self.wb[tab].cell(r, c)
        old = cell.value
        if isinstance(new, datetime.datetime) or isinstance(old, datetime.datetime):
            same = old == new
        else:
            same = s(old) == s(new)
        if same:
            return
        cell.value = new if new != "" else None
        if isinstance(new, datetime.datetime):
            cell.number_format = T.DATE_FMT
        self.log.append(dict(tab=tab, row=r, column=s(self.wb[tab].cell(self.header_row(tab, r), c).value) or c,
                             old=s(old), new=s(new), why=why))
        self.touched.add(tab)

    @staticmethod
    def _is_header(names):
        return "xyz" in names and ("root_id_curr" in names or "root_id_orig" in names)

    def header_row(self, tab, r):
        """the header row at or above row r (a row naming xyz and a root id column), else r. One pass per tab builds a
        map (2026-11-15: scanning upward cell by cell for every row took over an hour once the 3,500-row T4T5s tab sat
        under one header); the map is rebuilt when the tab's row count changed, the mapped header is no header any
        more (rows moved), or forget() was called"""
        ws = self.wb[tab]
        got = self._hdr.get(tab)
        if got is not None and got[0] == ws.max_row and r in got[1]:
            h = got[1][r]  # 0 = no header above r
            if h == 0 or self._is_header([s(ws.cell(h, c).value) for c in range(1, ws.max_column + 1)]):
                return h or r
        m, h = {}, 0
        for i, row in enumerate(ws.iter_rows(values_only=True), 1):
            if self._is_header([s(v) for v in row]):
                h = i
            m[i] = h
        self._hdr[tab] = (ws.max_row, m)
        return m.get(r, 0) or r

    def forget(self, tab=None):
        """drop the header map(s) after rows were deleted, inserted or moved"""
        if tab is None:
            self._hdr.clear()
        else:
            self._hdr.pop(tab, None)

    def fill(self, tab, r, cols, how, why):
        """highlight the named columns, or clear the highlight from every cell of the row (older highlights span it)"""
        ws = self.wb[tab]
        n = 0
        for c in (cols if how == "highlight" else range(1, ws.max_column + 1)):
            cell = ws.cell(r, c)
            rgb = cell.fill.fgColor.rgb if cell.fill is not None and cell.fill.fill_type == "solid" else None
            if how == "highlight" and rgb != HL:
                cell.fill, n = PatternFill("solid", fgColor=HL), n + 1
            elif how == "clear" and rgb == HL:
                cell.fill, n = PatternFill(fill_type=None), n + 1
        if n:
            self.log.append(dict(tab=tab, row=r, column="fill", old="", new=how, why=why))
            self.touched.add(tab)


def fix_date(v):
    """(new value, why) for a date-column cell, or (None, None) when it is fine"""
    now = datetime.datetime.now()
    if isinstance(v, datetime.datetime):
        if v.year < 2026 and v.day == 26:  # 'dd-mm-26' read as yy-mm-dd
            return datetime.datetime(2026, v.month, v.year - 2000), "date typed day-month-year"
        return None, None
    t = s(v)
    if not t:
        return None, None
    d = T.as_date(t)
    if d is None:
        return "", f"text in a date column: {t}"
    if d > now + datetime.timedelta(days=2):  # e.g. 27-10-26 -> 2027-10-26: typed day-month-year
        return datetime.datetime(2000 + d.day, d.month, d.year - 2000), "text date typed day-month-year"
    return d, "text date -> date"


def merge_people(sheet_val, st_val):
    """sheet names + SeaTable proofreaders not already credited, as SeaTable writes them (initials are not expanded to
    names: that would be a guess); 'KT' counts as crediting 'Kristo T.'"""
    have = [x.strip() for x in sheet_val.split(";") if x.strip()]
    known = {initials(x) for x in have} | {x.upper() for x in have}
    add = []
    for tok in (x.strip() for x in re.split(r"[,;]", st_val)):
        if tok and tok.upper() not in known and tok not in add:
            add.append(tok)
    return "; ".join(have + add)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("book")
    ap.add_argument("--match", default=None)
    ap.add_argument("--edits")
    ap.add_argument("--left-rows")
    ap.add_argument("--left-link", default="")
    ap.add_argument("--reconcile", action="store_true")
    ap.add_argument("--sizes")
    ap.add_argument("--no-stale-notes", action="store_true", help="skip faab.partners.stale_notes (no nucleus lookup)")
    ap.add_argument("--export", nargs="*", default=[], help="tabs exported even when this run did not change them")
    a = ap.parse_args()
    stamp = utc_stamp()
    match_path = a.match or sorted(glob.glob(fig("*_seatable_match.tsv")))[-1]
    m = pd.read_csv(match_path, sep="\t", dtype=str).fillna("")
    M = {(t, int(r)): x for t, r, x in zip(m.tab, m.row, m.to_dict("records"))}
    wb = openpyxl.load_workbook(a.book)
    B = Book(wb)
    rows = [x for x in scan_book(wb) if x["tab"] in TABS]

    # stale vs MISMATCH: one CAVE call
    curr_of = lambda x: s(x["values"].get("root_id_curr"))
    diff = sorted({curr_of(x) for x in rows if re.fullmatch(r"\d{15,}", curr_of(x))
                   and curr_of(x) != M.get((x["tab"], x["row"]), {}).get("root_at_xyz")})
    c = cave_client()
    throttle(0)
    latest = dict(zip(diff, np.asarray(c.chunkedgraph.is_latest_roots([int(i) for i in diff])).astype(bool))) if diff else {}
    # stale notes (2026-11-07): nuclei per live root, for 'missing soma' notes on cells that have one now
    copy_path = None if a.no_stale_notes else SN.default_table_copy(TMP)
    nuc = SN.nuclei_by_root(c, copy_path, throttle) if copy_path else None
    soma_status = []  # rows whose (SeaTable) status says 'missing soma' although the cell has a nucleus

    sea_col = {}  # PRs: blocks after the first have no 'SeaTable' header (the column holds the marks) -> name it
    for x in rows:
        if "SeaTable" in x["cols"]:
            sea_col.setdefault(x["tab"], x["cols"]["SeaTable"])
    for tab, sc in sea_col.items():
        ws = wb[tab]
        for hr in sorted({B.header_row(tab, x["row"]) for x in rows if x["tab"] == tab}):
            if not s(ws.cell(hr, sc).value):
                ws.cell(hr, sc).value = "SeaTable"
                ws.cell(hr, sc)._style = copy.copy(ws.cell(hr, sc - 1)._style)
                B.log.append(dict(tab=tab, row=hr, column="SeaTable", old="", new="SeaTable", why="header was blank"))
                B.touched.add(tab)
    by_orig = {}  # root_id_orig -> [(tab, row, nsyn)]: the record rows that 'duplicate target of' notes point to
    for x in rows:
        nsyn = re.match(r"\d+", s(x["values"].get("nsyn")))
        by_orig.setdefault(s(x["values"].get("root_id_orig")), []).append((x["tab"], x["row"], int(nsyn.group()) if nsyn else 0))

    orig_cells = {}  # root_id_orig -> live cells at its rows' points (stale 'not merged' flags)
    for x in rows:
        at_x = M.get((x["tab"], x["row"]), {}).get("root_at_xyz", "")
        if at_x and at_x != "0":
            orig_cells.setdefault(s(x["values"].get("root_id_orig")), set()).add(at_x)

    def row_of(orig, tab, row):
        """the record row of root_id_orig: same tab first, then any tab; largest nsyn; never the row itself"""
        cand = [c for c in by_orig.get(orig, []) if (c[0], c[1]) != (tab, row)]
        same = [c for c in cand if c[0] == tab]
        if not (same or cand):
            return None
        t, rr, _ = max(same or cand, key=lambda c: c[2])
        return f"r{rr}" if t == tab else f"{t} r{rr}"

    for x in rows:
        tab, r, cols, v = x["tab"], x["row"], x["cols"], {k: s(val) for k, val in x["values"].items()}
        mm = M.get((tab, r), {})
        col = lambda name: cols.get(name) or (sea_col.get(tab) if name == "SeaTable" else None)
        notes_add = []
        if v.get("notes") and "notes" in cols:
            short, _ = shorten(v["notes"], lambda orig: row_of(orig, tab, r))
            if short != v["notes"]:
                B.put(tab, r, cols["notes"], short, "generated notes shortened")
        # --- spelling / misplaced values
        st = v.get("status", "")
        if "status" in cols:
            low = SPELL.get(st.lower(), st.lower())
            if low in STATUSES and st != low:
                B.put(tab, r, cols["status"], low, "status lowercase / spelling")
                st = low
        if col("SeaTable"):
            sv = s(wb[tab].cell(r, col("SeaTable")).value)
            if sv == "ADDED" or (sv.startswith("=IFERROR") and '"added"' in sv):
                B.put(tab, r, col("SeaTable"), "added", "SeaTable mark spelling / formula leftover")
        if v.get("checker", "").lower() == "added" and col("SeaTable"):
            if not s(wb[tab].cell(r, col("SeaTable")).value):
                B.put(tab, r, col("SeaTable"), "added", "'added' was typed into checker")
            B.put(tab, r, cols["checker"], "", "'added' was typed into checker")
        for name, cc in cols.items():
            if name.endswith("date"):
                new, why = fix_date(wb[tab].cell(r, cc).value)
                if why:
                    if new == "" and why.startswith("text in"):
                        notes_add.append(s(wb[tab].cell(r, cc).value))  # the proofreader's text, moved as is
                    B.put(tab, r, cc, new, why)
        # --- roots
        curr, at = v.get("root_id_curr", ""), mm.get("root_at_xyz", "")
        if curr and not re.fullmatch(r"\d{15,}", curr) and re.fullmatch(r"\d{15,}", at or ""):
            # stray text typed over the id (2026-11-12: 'd'): the root at the row's xyz
            B.put(tab, r, cols["root_id_curr"], at, "root_id_curr was not an id: root at xyz")
        elif re.fullmatch(r"\d{15,}", curr) and at and at != "0" and at != curr:
            if not latest.get(curr, True):
                B.put(tab, r, cols["root_id_curr"], at, "root update (root at xyz)")
            else:
                B.fill(tab, r, cols.values(), "highlight", "MISMATCH")
                if f"xyz off the cell (on {at})" not in v.get("notes", ""):  # flagged by an earlier sync already
                    notes_add.append(f"xyz off the cell (on {at})")
        # --- SeaTable
        if mm.get("via"):
            sts = [t for t in dict.fromkeys(x.strip() for x in mm["st_status"].split(";")) if t]
            if "status" in cols and sts and st not in KEEP and st != "; ".join(sts):
                B.put(tab, r, cols["status"], "; ".join(sts), "SeaTable status")
            if "proofreader" in cols and mm["st_proofreader"]:
                B.put(tab, r, cols["proofreader"], merge_people(v.get("proofreader", ""), mm["st_proofreader"]),
                      "SeaTable proofreaders added")
            if col("SeaTable"):
                label = "; ".join(t for t in dict.fromkeys(x.strip() for x in mm["st_label"].split(";")) if t)
                mark = seatable_mark(s(wb[tab].cell(r, col("SeaTable")).value))
                B.put(tab, r, col("SeaTable"), mark + (f": {label}" if label else ""), "SeaTable mark + label")
        # --- stale notes (faab.partners.stale_notes): fixed off-cell points, somata found since
        if nuc is not None and "notes" in cols:
            ok_at = at if at and at != "0" else ""
            on_cell = bool(ok_at) and (ok_at == curr or not latest.get(curr, True))
            cell = ok_at or curr
            cur_notes = s(wb[tab].cell(r, cols["notes"]).value)
            # soma rule only where the point is on the row's cell (a MISMATCH point would count the wrong segment)
            n_nuc = nuc.get(cell, 0) if cell and (on_cell or not ok_at) else None
            new_notes, why = SN.clean(cur_notes, on_cell, n_nuc, ok_at, lambda i: orig_cells.get(i, set()))
            if new_notes != cur_notes:
                B.put(tab, r, cols["notes"], new_notes, "stale note: " + ", ".join(why))
                # the yellow goes with the flag, also when the proofreader replaced the flag by a reply
                # ('xyz is corrected -DR', 2026-11-08) and the point is now checked on the cell
                if "point is on the cell again" in why or (on_cell and any(w.startswith("reply") for w in why)):
                    B.fill(tab, r, cols.values(), "clear", "stale note: point is on the cell again")
            st_now = s(wb[tab].cell(r, cols["status"]).value) if "status" in cols else ""
            if "soma" in st_now.lower() and n_nuc == 1:
                soma_status.append((tab, v.get("root_id_orig", ""), cell, st_now))
        if notes_add and "notes" in cols:
            old = s(wb[tab].cell(r, cols["notes"]).value)
            B.put(tab, r, cols["notes"], "; ".join([old] * bool(old) + notes_add), "notes added")

    # --- hand edits (last); keyed by row, or by "root" (every row of the tab on that cell: root_id_curr or the root at
    # its xyz), which survives rows added or deleted in the sheet; "append": true adds the value to the notes
    for e in json.load(open(a.edits)) if a.edits else []:
        hit = [x for x in rows if x["tab"] == e["tab"] and (
            x["row"] == e["row"] if "row" in e else
            str(e["root"]) in (curr_of(x), M.get((x["tab"], x["row"]), {}).get("root_at_xyz")))]
        if not hit:
            print(f"  hand edit matched no row: {e}")
        for x in hit:
            if e["column"] == "fill":
                B.fill(e["tab"], x["row"], x["cols"].values(), e["value"], "hand edit")
                continue
            val = e["value"]
            if e.get("append"):
                old = s(wb[e["tab"]].cell(x["row"], x["cols"][e["column"]]).value)
                if val in old:
                    continue
                val = "; ".join([old] * bool(old) + [val])
            B.put(e["tab"], x["row"], x["cols"][e["column"]], val, "hand edit")

    # --- rows of one live cell (after the hand edits; collapsing renumbers rows, so it comes last among row edits)
    if a.reconcile:
        from faab.partners.reconcile import reconcile
        sizes = {}
        if a.sizes:
            z = pd.read_csv(a.sizes, sep="\t", dtype=str).fillna("")
            sizes = {r: int(n) for r, n in zip(z.cell, z.l2_chunks) if n.isdigit()}

        def cell_of(x):
            at = M.get((x["tab"], x["row"]), {}).get("root_at_xyz", "")
            cu = s(wb[x["tab"]].cell(x["row"], x["cols"]["root_id_curr"]).value) if "root_id_curr" in x["cols"] else ""
            return cu if latest.get(cu) else at

        def yellow(x):
            return any(wb[x["tab"]].cell(x["row"], cc).fill is not None and wb[x["tab"]].cell(x["row"], cc).fill.fill_type == "solid"
                       and wb[x["tab"]].cell(x["row"], cc).fill.fgColor.rgb == HL for cc in x["cols"].values())

        cur, last = {}, [time.time()]

        def wait():
            last[0] = throttle(last[0])

        def latest_of(ids):
            """current roots of upstream ids as written (an old root that split -> all its current pieces)"""
            need = sorted({u for u in ids if u not in cur})
            if need:
                wait()
                ok = np.asarray(c.chunkedgraph.is_latest_roots([int(u) for u in need])).astype(bool)
                for u, good in zip(need, ok):
                    if not good:
                        wait()
                    cur[u] = [int(u)] if good else [int(v) for v in np.atleast_1d(c.chunkedgraph.get_latest_roots(int(u)))]
            return {u: cur[u] for u in ids}

        def recount(pairs):
            """local synapse snapshot (faab.synsnap) caught up to now, self-contacts removed"""
            latest_of({u for u, _ in pairs})
            wait()
            try:
                e = synsnap.Snapshot(log=lambda *_: None).edges(
                    pre=sorted({v for u, _ in pairs for v in cur[u]}), post=sorted({int(cc) for _, cc in pairs}),
                    when="now", client=c)
            except Exception as ex:
                print(f"  recount failed ({str(ex)[:80]}); collapsed rows get their nsyn summed")
                return {}
            finally:
                last[0] = time.time()
            n = {(int(p), int(q)): int(k) for p, q, k in zip(e.pre, e.post, e.n)}
            return {(u, cc): sum(n.get((v, int(cc)), 0) for v in cur[u]) for u, cc in pairs}

        def recount_out(cell, ids):
            """synapses from a collapsed cell onto each downstream id as written (all current pieces of a stale id),
            local synapse snapshot caught up to now (T4T5-1 / T4T5-2 rows, 2026-11-11)"""
            latest_of(ids)
            wait()
            try:
                e = synsnap.Snapshot(log=lambda *_: None).edges(
                    pre=[int(cell)], post=sorted({v for u in ids for v in cur[u]}), when="now", client=c)
            except Exception as ex:
                print(f"  recount failed ({str(ex)[:80]}); collapsed rows get their per-partner nsyn summed")
                return {}
            finally:
                last[0] = time.time()
            n = {int(q): int(k) for q, k in zip(e.post, e.n)}
            return {u: sum(n.get(v, 0) for v in cur[u]) for u in ids}

        reconcile(wb, B, rows, cell_of, sizes, yellow, recount=recount, latest_of=latest_of, recount_out=recount_out)

    # --- left-eye PRs section
    if a.left_rows:
        add_left(wb, B, pd.read_csv(a.left_rows, sep="\t", dtype=str).fillna(""), a.left_link)
    ws = wb["PRs"]
    tid = T.tidy(ws)
    print(f"PRs tidy: {tid}; {T.lint_line(ws)}")

    # --- outputs
    full = os.path.join(TMP, f"{stamp}_synced_full.xlsx")
    wb.save(full)
    log = pd.DataFrame(B.log)
    log.to_csv(fig(f"{stamp}_seatable_sync_changes.tsv"), sep="\t", index=False)
    for tab in [t for t in TABS if t in B.touched or t in a.export]:
        w2 = openpyxl.load_workbook(full)
        for other in list(w2.worksheets):
            if other.title != tab:
                w2.remove(other)
        out = fig(f"{stamp}_FAABopticlobetargets-{tab}.xlsx")
        w2.save(out)
        print(f"wrote {out}")
    if soma_status:
        print(f"\nstatus says 'missing soma' but the cell has a nucleus now ({len(soma_status)} rows; tab, "
              f"root_id_orig, cell, status):")
        for x in soma_status:
            print("  " + "\t".join(x))
    print(f"\nchanges: {len(log)} cells; per tab and reason:")
    print(log.groupby(["tab", "why"]).size().to_string())
    print(f"full workbook {full}; log figures/{stamp}_seatable_sync_changes.tsv")


def add_left(wb, B, L, link):
    """append the L side section below the last row of the PRs tab, styled like the R side's"""
    ws = wb["PRs"]
    a_col = [s(ws.cell(r, 1).value) for r in range(1, ws.max_row + 1)]
    r_title = next(i + 1 for i, t in enumerate(a_col) if t.startswith(">>> additional R side"))
    r_group = next(i + 1 for i, t in enumerate(a_col) if i + 1 > r_title and re.fullmatch(r"group \d+", t))
    r_head = r_group + 1
    hdr = {s(ws.cell(r_head, c).value): c for c in range(1, ws.max_column + 1) if s(ws.cell(r_head, c).value)}
    r_data = next(r for r in range(r_head + 1, ws.max_row + 1) if s(ws.cell(r, hdr["status"]).value) == "candidate")
    width = ws.max_column
    last = max(r for r in range(1, ws.max_row + 1) if any(s(ws.cell(r, c).value) for c in range(1, width + 1)))

    def copy_row(src, dst):
        for c in range(1, width + 1):
            ws.cell(dst, c)._style = copy.copy(ws.cell(src, c)._style)
        ws.row_dimensions[dst].height = ws.row_dimensions[src].height

    r = last + 2
    copy_row(r_title, r)
    ws.cell(r, 1, ">>> additional L side putative PRs")
    if link:
        ws.cell(r, 4, "nglink")
        ws.cell(r, 5, link).hyperlink = link
    r += 2
    for g, sub in L.groupby(L.group.astype(int), sort=True):
        copy_row(r_group, r)
        ws.cell(r, 1, f"group {g}")
        r += 1
        copy_row(r_head, r)
        for name, c in hdr.items():
            ws.cell(r, c, name)
        r += 1
        for x in sub.to_dict("records"):  # not itertuples: 'class' is a keyword
            copy_row(r_data, r)
            for name in ("root_id_orig", "class", "root_id_curr", "xyz", "status", "proofreader", "notes", "SeaTable"):
                if name in hdr:
                    ws.cell(r, hdr[name], x[name] or None)
            r += 1
        r += 1
    B.log.append(dict(tab="PRs", row=last + 2, column="", old="", new=f"{len(L)} rows in {L.group.nunique()} groups",
                      why="L side section added"))
    B.touched.add("PRs")


if __name__ == "__main__":
    main()
