"""Sheet rows annotated as LMCs (2026-11-01, LMC tab): every data row of every tab of a workbook (a header row starting
with root_id_orig / upstream_PR / PR_root_id / hop / root_id defines the columns below it) whose annotation, SeaTable,
class or notes column names LMC or an L1..L5 type. Library: sheet_lmc_rows(book) -> DataFrame (tab, row, the row's
columns by header name); as a script it prints counts per tab and column. Network-free.
Usage: sheet_lmc_rows.py BOOK.xlsx"""
import re
import sys

import openpyxl
import pandas as pd

PAT = re.compile(r"\bLMC\b|\bL[1-5]\b(?![-–]?L?\d)|\bL1\s*[-–]\s*L?5\b")
HEADS = ("root_id_orig", "upstream_PR", "PR_root_id", "hop", "root_id")
COLS = ("annotation", "SeaTable", "class", "notes")


def sheet_lmc_rows(book):
    wb = openpyxl.load_workbook(book, read_only=True)
    rows = []
    for ws in wb.worksheets:
        if ws.title == "PRs":
            continue  # photoreceptor groups; the LMC rows deleted there are not annotations
        hdr = None
        for k, row in enumerate(ws.iter_rows(values_only=True), 1):
            vals = ["" if v is None else v for v in row]
            first = str(vals[0]).strip() if vals else ""
            if first in HEADS and "xyz" in [str(v).strip() for v in vals]:
                hdr = [str(v).strip() for v in vals]
                continue
            if hdr is None or not first:
                continue
            cells = {h: vals[j] for j, h in enumerate(hdr) if h and j < len(vals)}
            hit = [c for c in COLS if c in cells and PAT.search(str(cells[c]))]
            if hit and str(cells.get("root_id_curr") or cells.get("root_id_orig") or "").strip().isdigit():
                rows.append(dict(tab=ws.title, row=k, hit=",".join(hit), **{h: cells.get(h, "") for h in
                                 ("root_id_orig", "root_id_curr", "xyz", "annotation", "status", "pr_date", "proofreader",
                                  "chk_date", "checker", "notes", "SeaTable", "class")}))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    R = sheet_lmc_rows(sys.argv[1])
    print(f"{len(R)} rows on {R.root_id_curr.nunique()} cells")
    print(R.groupby(["tab", "hit"]).size().to_string())
    print("\nstatus:", R.status.astype(str).replace("", "none").value_counts().to_dict())
    print(R.head(8)[["tab", "root_id_curr", "annotation", "status", "SeaTable", "class", "notes"]].to_string(index=False))
