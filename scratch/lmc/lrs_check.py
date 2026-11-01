"""LRS's cartridge LMCs in a fresh SeaTable fetch (David 2026-11-01: 'LRS's L1-L5 should all be updated in SeaTable
now'): every row whose notes say 'LRS: cartridge N', matched by serial id, with its current type, status and annotator,
against our call (lmc_cartridge_calls.tsv: the cartridge assignment, else the model consensus of lmc_typing.tsv); and
the two SeaTable L3s that look like L2 (40250, 40125). Network-free. Usage: lrs_check.py SEATABLE.tsv"""
import re
import sys

import pandas as pd

from faab.config import fig

S = pd.read_csv(sys.argv[1], sep="\t", dtype=str).fillna("")
S["label"] = S[["class", "subclass", "subsubclass", "type"]].agg(" / ".join, axis=1).str.replace(r"( / )+", " / ", regex=True).str.strip(" /")
S["t"] = S.label.map(lambda s: (re.findall(r"\bL[1-5](?:-L5)?\b", s) or [""])[-1])
Ty = pd.read_csv(fig("lmc_typing.tsv"), sep="\t", dtype={"root": str, "serial_id": str}).fillna("")
K = pd.read_csv(fig("lmc_cartridge_calls.tsv"), sep="\t", dtype={"root": str, "serial_id": str}).fillna("")
ours = dict(zip(Ty.serial_id, Ty.consensus)) | {s: a for s, a in zip(K.serial_id, K.assigned) if a}
L = S[S.notes.str.contains(r"LRS: cartridge", regex=True)].copy()
L["cartridge"] = L.notes.str.extract(r"cartridge\s*(\d+)", expand=False)
L["ours"] = L.serial_id.map(ours).fillna("")
L["agree"] = [("-" if not o else "yes" if t == o else "still L1-L5" if t == "L1-L5" else "NO") for t, o in zip(L.t, L.ours)]
print(f"LRS cartridge rows: {len(L)}; SeaTable types now {L.t.replace('', '?').value_counts().to_dict()}; vs our calls "
      f"{L.agree.value_counts().to_dict()}")
print(L.sort_values(["side", "cartridge", "t"])[["serial_id", "root_now", "side", "cartridge", "t", "ours", "agree", "status",
                                                  "annotator", "proofreader"]].to_string(index=False))
two = S[S.serial_id.isin(["40250", "40125"])]
print("\nthe two L3s that look like L2:")
print(two[["serial_id", "root_now", "t", "status", "annotator", "proofreader", "notes"]].to_string(index=False))
