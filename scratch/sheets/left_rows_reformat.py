"""Network-free: rewrite a prs_left_rows.py TSV written before 2026-10-30's format changes (David: no SeaTable text in
notes, the label goes into the SeaTable column; notes stay short for the proofreaders):
  - SeaTable column 'present' -> 'present: <label>' (newest SeaTable snapshot);
  - notes -> 'LRS: cartridge N' / 'census bundle N' / 'axon not traced toward the retina'; the old long notes move to a
    'detail' column (not written to the sheet).
Writes a new stamped TSV. Usage: left_rows_reformat.py OLD.tsv"""
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
if "detail" not in L:
    L["detail"] = L.notes.str.replace(r"^SeaTable R(?:1-6|7-8); (LRS: cartridge \d+; )?", "", regex=True)


def short(old):
    out = []
    cart = re.search(r"LRS: cartridge \d+", old)
    bundle = re.search(r"census bundle (\d+)", old)
    if cart:
        out.append(cart.group(0))
    if bundle:
        out.append(f"census bundle {bundle.group(1)}")
    if "lamina-terminal piece" in old or old == "axon not traced toward the retina":
        out.append("axon not traced toward the retina")
    if "CHECK" in old:
        out.append("CHECK: no point on the axon")
    return "; ".join(out)


L["notes"] = [short(x) for x in L.notes]
miss = [r for r, x in zip(L.root_id_curr, L.SeaTable) if x == "present" and r not in label]
L["SeaTable"] = [f"present: {label[r]}" if x == "present" and r in label else x for r, x in zip(L.root_id_curr, L.SeaTable)]
out = os.path.join(FIGURES, f"{utc_stamp()}_pr_left_groups.tsv")
L.to_csv(out, sep="\t", index=False)
print(f"{len(L)} rows; notes {L.notes.value_counts().head(6).to_dict()}; max notes length {L.notes.str.len().max()}; "
      f"SeaTable labels {sum(L.SeaTable.str.startswith('present:'))}; not found {miss}; written {out}")
