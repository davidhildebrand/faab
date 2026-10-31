"""Which tracking-sheet rows are already in SeaTable (aedes_main)? Check only, writes nothing to the sheets.

Every tab row with an xyz is mapped to the live root at its xyz (one CAVE lookup) and matched to SeaTable rows by their
live root (newest figures/*_seatable_aedes_main.tsv from seatable_fetch.py). Rows whose xyz root has no SeaTable row
but whose root_id_curr does are reported too ('via root_id_curr'). Writes figures/<stamp>_seatable_match.tsv (one line
per sheet row, sheet vs SeaTable fields) and prints per tab: rows in SeaTable, by sheet status, and the fields that
differ. Usage: seatable_match.py BOOK.xlsx"""
import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import openpyxl
import pandas as pd

from faab.clients.cave import roots_at_xyz
from faab.config import FIGURES
from faab.partners.sheet import scan_book, cell_text as s, seatable_label, utc_stamp


def main():
    book = sys.argv[1]
    stamp = utc_stamp()
    st_path = sorted(glob.glob(os.path.join(FIGURES, "*_seatable_aedes_main.tsv")))[-1]
    st = pd.read_csv(st_path, sep="\t", dtype=str).fillna("")
    st["label"] = [seatable_label(r) for r in st.to_dict("records")]
    by_root = {r: g for r, g in st[st.root_now != ""].groupby("root_now")}
    rows = scan_book(openpyxl.load_workbook(book))
    at = roots_at_xyz([s(r["values"]["xyz"]) for r in rows])
    out = []
    for r, root in zip(rows, at):
        v = {k: s(x) for k, x in r["values"].items()}
        curr = v.get("root_id_curr", "")
        via, g = "", None
        if str(root) in by_root:
            via, g = "xyz", by_root[str(root)]
        elif curr in by_root:
            via, g = "root_id_curr", by_root[curr]
        d = dict(tab=r["tab"], row=r["row"], root_id_orig=v.get("root_id_orig", ""), root_id_curr=curr, xyz=v["xyz"],
                 root_at_xyz=str(root), via=via, sheet_status=v.get("status", ""),
                 sheet_annotation=v.get("annotation", v.get("class", "")), sheet_proofreader=v.get("proofreader", ""),
                 sheet_checker=v.get("checker", ""), sheet_notes=v.get("notes", ""), sheet_SeaTable=v.get("SeaTable", ""))
        if g is not None:
            d.update(st_n=len(g), st_serial="; ".join(g.serial_id), st_status="; ".join(dict.fromkeys(g.status)),
                     st_label="; ".join(dict.fromkeys(g.label)), st_proofreader="; ".join(dict.fromkeys(g.proofreader)),
                     st_annotator="; ".join(dict.fromkeys(g.annotator)), st_notes="; ".join(dict.fromkeys(g.notes)),
                     st_side="; ".join(dict.fromkeys(g.side)), st_point_xyz="; ".join(g.point_xyz),
                     st_root_id="; ".join(g.root_id))
        out.append(d)
    df = pd.DataFrame(out).fillna("")
    path = os.path.join(FIGURES, f"{stamp}_seatable_match.tsv")
    df.to_csv(path, sep="\t", index=False)
    print(f"{os.path.basename(st_path)}: {len(st)} SeaTable rows; {len(df)} sheet rows with an xyz in {df.tab.nunique()} tabs")
    for tab, g in df.groupby("tab", sort=False):
        m = g[g.via != ""]
        stat = m.sheet_status.replace("", "(blank)").value_counts().to_dict()
        print(f"\n{tab}: {len(m)} of {len(g)} rows in SeaTable ({(m.via == 'root_id_curr').sum()} only via root_id_curr); "
              f"{m.root_at_xyz.nunique()} cells; sheet status {stat}")
        if len(m):
            print("  SeaTable status: " + "; ".join(f"{k} {n}" for k, n in m.st_status.replace("", "(blank)").value_counts().items()))
            print(f"  sheet SeaTable column: {m.sheet_SeaTable.replace('', '(blank)').value_counts().to_dict()}; "
                  f"not in SeaTable but marked: {g[(g.via == '') & (g.sheet_SeaTable != '')].sheet_SeaTable.value_counts().to_dict()}")
            print(f"  proofreader differs {(m.sheet_proofreader != m.st_proofreader).sum()} (SeaTable blank "
                  f"{(m.st_proofreader == '').sum()}); multiple SeaTable rows {(pd.to_numeric(m.st_n) > 1).sum()}")
    print(f"\nwritten {path}")


if __name__ == "__main__":
    main()
