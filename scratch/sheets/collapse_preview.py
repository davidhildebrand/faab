"""Offline preview of faab.partners.reconcile's same-block collapse (2026-10-31): rows of one live cell (cell column of a
cell_consistency.py TSV) under the same header row of the same tab, PRs excluded; which row stays (best status, then
most nsyn) and which go. No CAVE. Usage: collapse_preview.py CONSISTENCY.tsv BOOK.xlsx"""
import sys

import openpyxl
import pandas as pd

from faab.partners import reconcile as R

D = pd.read_csv(sys.argv[1], sep="\t", dtype=str).fillna("")
wb = openpyxl.load_workbook(sys.argv[2])
spec = __import__("importlib.util").util.spec_from_file_location("ss", "/Users/davidh/Data/Vibe/FAAB/faab/scripts/sheets/seatable_sync.py")
SS = __import__("importlib.util").util.module_from_spec(spec)
spec.loader.exec_module(SS)
B = SS.Book(wb)
M = D[(D.n_rows.astype(int) > 1) & ~D.tab.isin(R.NO_COLLAPSE + ("HomeR7column",))].copy()
M["hdr"] = [B.header_row(t, int(r)) for t, r in zip(M.tab, M.row)]
nsyn_col = {}
n_drop = 0
for (tab, h, c), g in M.groupby(["tab", "hdr", "cell"]):
    if len(g) < 2:
        continue
    ws = wb[tab]
    heads = {SS.s(ws.cell(h, k).value): k for k in range(1, ws.max_column + 1)}
    g = g.assign(nsyn=[R.first_n(ws.cell(int(r), heads["nsyn"]).value) if "nsyn" in heads else 0 for r in g.row],
                 rk=g.status.map(R.rank), dup=g.status.str.upper().eq("NEW-DUP"))
    g = g.sort_values(["rk", "dup", "nsyn", "row"], ascending=[True, True, False, True], key=lambda s: s.astype(int) if s.name == "row" else s)
    false = any(R.FALSE_MERGE.search(n) for n in g.notes)
    n_drop += 0 if false else len(g) - 1
    print(f"{tab} block@r{h} cell {c}{' (FALSE MERGE note: kept)' if false else ''}: keep r{g.row.iloc[0]} "
          f"[{g.status.iloc[0]}, nsyn {g.nsyn.iloc[0]}]; drop " + ", ".join(f"r{r} [{s}, {n}]" for r, s, n in zip(g.row[1:], g.status[1:], g.nsyn[1:])))
print(f"\nrows to delete: {n_drop}")
