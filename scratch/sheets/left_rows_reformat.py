"""Network-free: rewrite a prs_left_rows.py TSV written before 2026-10-30's format change (David: no SeaTable text in
notes; the label goes into the SeaTable column) - notes lose the leading 'SeaTable R1-6; ' / 'SeaTable R7-8; ', and the
SeaTable column 'present' becomes 'present: <label>' from the newest SeaTable snapshot. Writes a new stamped TSV.
Usage: left_rows_reformat.py OLD.tsv"""
import glob
import os
import re
import sys

import pandas as pd

from faab.config import FIGURES
from faab.partners.sheet import seatable_label, utc_stamp

L = pd.read_csv(sys.argv[1], sep="\t", dtype=str).fillna("")
st = pd.read_csv(sorted(glob.glob(os.path.join(FIGURES, "*_seatable_aedes_main.tsv")))[-1], sep="\t", dtype=str).fillna("")
label = {r["root_now"]: seatable_label(r) for r in st.to_dict("records")}
L["notes"] = L.notes.str.replace(r"^SeaTable R(?:1-6|7-8); ", "", regex=True)
miss = [r for r, x in zip(L.root_id_curr, L.SeaTable) if x == "present" and r not in label]
L["SeaTable"] = [f"present: {label[r]}" if x == "present" and r in label else x for r, x in zip(L.root_id_curr, L.SeaTable)]
out = os.path.join(FIGURES, f"{utc_stamp()}_pr_left_groups.tsv")
L.to_csv(out, sep="\t", index=False)
print(f"{len(L)} rows; SeaTable labels {sum(L.SeaTable.str.startswith('present:'))}; not found {miss}; "
      f"labels {L[L.SeaTable != ''].SeaTable.value_counts().to_dict()}; written {out}")
