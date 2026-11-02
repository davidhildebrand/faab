"""Orphan fragments listed in the spreadsheets (David 2026-11-02: "pull out a list of all the orphan fragments ... that
should be reviewed"). Scans every tab of a workbook (optionally with newer single-tab files replacing their tab, e.g.
the imported PRs tab) for rows whose any cell mentions 'orphan', and lists them with their block, ids, xyz, status,
names, and notes. With --live: the root at each row's xyz now (cached supervoxels + one get_roots call), its size in L2
chunks (one call per new root, >= 2 s apart, read-only), and whether that root is a cell tracked on another row of the
book (then the fragment is merged into a tracked cell and is no longer an orphan).
Usage: orphan_fragments.py BOOK.xlsx [TAB.xlsx ...] [--live]"""
import datetime
import os
import re
import sys
import time

import openpyxl
import pandas as pd

from faab.clients.cave import client as cave_client, roots_at_xyz, throttle
from faab.config import fig

args = [a for a in sys.argv[1:] if not a.startswith("--")]
live = "--live" in sys.argv
sheets = {}
for path in args:
    wb = openpyxl.load_workbook(path, data_only=True)
    for ws in wb.worksheets:  # a later file's tab replaces the book's
        sheets[ws.title] = (ws, os.path.basename(path))


def text(v):
    return "" if v is None else str(v).strip()


rows, tracked = [], {}
for name, (ws, src) in sheets.items():
    cols, block, last_a = None, "", ""
    for r in range(1, ws.max_row + 1):
        vals = [text(ws.cell(r, c).value) for c in range(1, ws.max_column + 1)]
        if "root_id_orig" in vals:
            cols, block = {v: i for i, v in enumerate(vals) if v}, last_a
            continue
        a = vals[0] if vals else ""
        if cols is None or not text(vals[cols["root_id_orig"]]).isdigit():
            if a and not a.isdigit():
                last_a = a
            continue
        g = lambda k: vals[cols[k]] if k in cols and cols[k] < len(vals) else ""
        st = g("status").lower()
        for k in ("root_id_curr", "root_id_orig"):
            if g(k).isdigit():
                tracked.setdefault(g(k), set()).add((name, r, st))
        if not any("orphan" in v.lower() for v in vals):
            continue
        rows.append(dict(tab=name, row=r, block=block, root_id_orig=g("root_id_orig"), root_id_curr=g("root_id_curr"),
                         xyz=g("xyz"), status=g("status"), proofreader=g("proofreader"), pr_date=g("pr_date"),
                         checker=g("checker"), annotation=g("annotation"), notes=g("notes"), link=g("link"),
                         orphan_text=next((v for v in vals if "orphan" in v.lower()), ""), source=src))

D = pd.DataFrame(rows)
print("tabs: " + ", ".join(f"{k} ({s})" for k, (_, s) in sheets.items()))
print(f"{len(D)} rows mention 'orphan' in {D.tab.nunique() if len(D) else 0} tabs; "
      f"{D.root_id_orig.nunique() if len(D) else 0} distinct root_id_orig")
if len(D):
    print(D.groupby(["tab", "status"]).size().to_string())
    print(D.orphan_text.value_counts().head(20).to_string())

if live and len(D):
    client = cave_client()
    t = throttle(0.0)
    now = roots_at_xyz(D.xyz.tolist(), client_=client)
    t = time.time()
    D["root_now"] = [str(x) if x else "" for x in now]
    size = {}
    for r in sorted(set(D.root_now) - {""}):
        t = throttle(t)
        size[r] = len(client.chunkedgraph.get_leaves(int(r), stop_layer=2))
        t = time.time()
    D["l2_chunks_now"] = D.root_now.map(size)
    own = {(a, b) for a, b in zip(D.tab, D.row)}
    D["now_tracked_on"] = [", ".join(sorted({f"{tb} row {rw} ({st or 'no status'})" for tb, rw, st in tracked.get(r, set())
                                             if (tb, rw) not in own})) for r in D.root_now]
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dd%H%M%StUTC")
out = fig(f"sheets/{stamp}_orphan_fragments.tsv")
D.to_csv(out, sep="\t", index=False)
print(f"written {out}")
