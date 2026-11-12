"""Network-free: SeaTable aedes_main rows of optic-lobe neurons whose status is not 'adequate' (David 2026-11-11: "check
the SeaTable for any optic lobe neurons ... with status that is not 'adequate'. In particular those with status
'proofread'"). Optic lobe = superclass ol_intrinsic / ol_sensory / ol_bilateral / visual_projection / visual_centrifugal,
or the row's live root is a cell in the tracking workbook (any tab, by root_id_curr). 'duplicate' rows are reported
apart (they are bookkeeping, not proofreading states).
Writes figures/seatable/<stamp>_ol_status_check.tsv (one row per SeaTable row) and prints counts by status x
superclass, the 'proofread' rows in full, and the sheet status of rows that the workbook tracks.
Usage: seatable_status_check.py SEATABLE.tsv BOOK.xlsx"""
import os
import re
import sys

import openpyxl
import pandas as pd

from faab.config import FIGURES
from faab.partners.sheet import cell_text as s, scan_book, utc_stamp

OL = ("ol_intrinsic", "ol_sensory", "ol_bilateral", "visual_projection", "visual_centrifugal")


def main():
    st = pd.read_csv(sys.argv[1], sep="\t", dtype=str).fillna("")
    wb = openpyxl.load_workbook(sys.argv[2], read_only=False)
    tracked = {}
    for x in scan_book(wb):
        if re.search(r"_b\d{6}d$", x["tab"]):
            continue
        r = s(x["values"].get("root_id_curr")) or s(x["values"].get("root_id_orig"))
        if re.fullmatch(r"\d{15,}", r):
            tracked.setdefault(r, []).append((x["tab"], s(x["values"].get("status"))))
    st["status"] = st.status.str.strip()
    ol = st.superclass.isin(OL) | st.root_now.isin(tracked)
    sel = st[ol & (st.status.str.lower() != "adequate")].copy()
    sel["why_listed"] = ["optic-lobe superclass" if sc in OL else "tracked in the workbook" for sc in sel.superclass]
    sel["sheet_rows"] = ["; ".join(f"{t}: {v or '(no status)'}" for t, v in dict.fromkeys(tracked.get(r, []))) for r in sel.root_now]
    sel["label"] = [" / ".join(v for v in (a, b, c, d) if v) for a, b, c, d in zip(sel.superclass, sel["class"], sel.subclass, sel.type)]
    cols = ["root_now", "point_xyz", "side", "status", "label", "proofreader", "annotator", "notes", "sheet_rows", "why_listed"]
    sel = sel.sort_values(["status", "superclass", "side", "root_now"])
    out_dir = os.path.join(FIGURES, "seatable")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{utc_stamp()}_ol_status_check.tsv")
    sel[cols].to_csv(out, sep="\t", index=False)

    print(f"SeaTable {os.path.basename(sys.argv[1])}: {len(st)} rows; optic-lobe rows {int(ol.sum())} "
          f"(adequate {int((ol & (st.status.str.lower() == 'adequate')).sum())}); not adequate {len(sel)}")
    tab = pd.crosstab(sel.status.replace("", "(blank)"), sel.superclass.replace("", "(blank)"), margins=True)
    print(tab.to_string())
    pr = sel[sel.status.str.lower() == "proofread"]
    print(f"\nstatus 'proofread': {len(pr)} rows")
    for _, r in pr.iterrows():
        print(f"  {r.root_now}  {r.side}  {r.label or '(no label)'}  proofreader {r.proofreader or '-'}  annotator "
              f"{r.annotator or '-'}  sheet: {r.sheet_rows or '-'}")
    tr = sel[sel.sheet_rows != ""]
    print(f"\nnot adequate and tracked in the workbook: {len(tr)} rows; SeaTable status x sheet status:")
    pairs = [(a, re.sub(r"^[^:]+: ", "", p)) for a, ps in zip(tr.status, tr.sheet_rows) for p in ps.split("; ")]
    print(pd.Series([f"{a or '(blank)'} | sheet {b}" for a, b in pairs]).value_counts().to_string())
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
