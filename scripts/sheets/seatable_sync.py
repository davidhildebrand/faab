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
shortened by faab.partners.short_notes (proofreaders' text kept verbatim; 'duplicate target of <id>' -> 'duplicate of
r<row>' pointing at the record row); statuses lowercase ('stoppped' -> 'stopped');
'added' typed into checker moves to SeaTable; dates typed day-month-year (2020-10-26 for 20-10-2026, text 27-10-26)
are corrected; text dates become dates and other text in date columns moves to notes. Hand edits (--edits JSON:
tab, row, column, value; column 'fill' with value clear / highlight) are applied last. --left-rows adds the
'>>> additional L side putative PRs' section (prs_left_rows.py) at the end of the PRs tab, styled like the R side.
CAVE read-only: one is_latest_roots call. Writes figures/<stamp>_FAABopticlobetargets-<tab>.xlsx per changed tab, the
full workbook in tmp/, and figures/<stamp>_seatable_sync_changes.tsv (every changed cell with the reason).
Usage: seatable_sync.py BOOK.xlsx [--edits E.json] [--left-rows L.tsv --left-link URL]"""
import argparse
import copy
import datetime
import glob
import json
import os
import re
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import openpyxl
import pandas as pd
from openpyxl.styles import Font, PatternFill

from faab.clients.cave import client as cave_client, throttle
from faab.config import FIGURES, TMP
from faab.partners.sheet import cell_text as s, scan_book, utc_stamp
from faab.partners.short_notes import shorten
from faab.pr import prs_tab as T
from faab.config import fig

HL = "FFFFF9C4"
KEEP = ("checked", "proofread")
STATUSES = ("checked", "proofread", "adequate", "review", "in progress", "stopped", "candidate")
SPELL = {"stoppped": "stopped"}
TABS = ("PRg1toT4T5", "PRs", "PRg1hop1", "PRg1hop2", "PRg2hop1", "Mi1s")


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

    def header_row(self, tab, r):
        ws = self.wb[tab]
        for rr in range(r, 0, -1):
            names = [s(ws.cell(rr, c).value) for c in range(1, ws.max_column + 1)]
            if "xyz" in names and ("root_id_curr" in names or "root_id_orig" in names):
                return rr
        return r

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
        if re.fullmatch(r"\d{15,}", curr) and at and at != "0" and at != curr:
            if not latest.get(curr, True):
                B.put(tab, r, cols["root_id_curr"], at, "root update (root at xyz)")
            else:
                B.fill(tab, r, cols.values(), "highlight", "MISMATCH")
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
        if notes_add and "notes" in cols:
            old = s(wb[tab].cell(r, cols["notes"]).value)
            B.put(tab, r, cols["notes"], "; ".join([old] * bool(old) + notes_add), "notes added")

    # --- hand edits (last)
    for e in json.load(open(a.edits)) if a.edits else []:
        x = next(x for x in rows if x["tab"] == e["tab"] and x["row"] == e["row"])
        if e["column"] == "fill":
            B.fill(e["tab"], e["row"], x["cols"].values(), e["value"], "hand edit")
        else:
            B.put(e["tab"], e["row"], x["cols"][e["column"]], e["value"], "hand edit")

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
    for tab in [t for t in TABS if t in B.touched]:
        w2 = openpyxl.load_workbook(full)
        for other in list(w2.worksheets):
            if other.title != tab:
                w2.remove(other)
        out = fig(f"{stamp}_FAABopticlobetargets-{tab}.xlsx")
        w2.save(out)
        print(f"wrote {out}")
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
