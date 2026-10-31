"""SeaTable rows typed only to the generic LMC level (L1-5 / L1-L5) vs a specific L1..L5, from a seatable_fetch TSV.
Network-free. Usage: seatable_lmc_generic.py SEATABLE.tsv"""
import re
import sys

import pandas as pd

S = pd.read_csv(sys.argv[1], sep="\t", dtype=str).fillna("")
F = ["superclass", "class", "subclass", "subsubclass", "type", "group", "nblast_group", "notes"]
GEN = re.compile(r"\bL1\s*-\s*L?5\b", re.I)
SPEC = re.compile(r"\bL[1-5]\b(?!\s*-)", re.I)
lab = S[["class", "subclass", "subsubclass", "type"]].agg(" / ".join, axis=1)
gen = lab.str.contains(GEN)
spec = lab.map(lambda t: bool(SPEC.search(GEN.sub("", t))))
lmc = (S["class"] == "LMC") | gen | spec
print(f"{len(S)} SeaTable rows; LMC-labelled (class LMC or an L1..L5 / L1-5 label): {lmc.sum()}")
only = S[gen & ~spec].copy()
only["label"] = lab[gen & ~spec]
print(f"generic L1-5 only: {len(only)} rows, {only.root_now.nunique()} distinct live roots")
print("  by label:", only.label.value_counts().to_dict())
print("  by side:", only.side.value_counts().to_dict())
print("  by status:", only.status.value_counts().to_dict())
print("  by proofreader:", only.proofreader.value_counts().head(12).to_dict())
print("  notes mentioning an L type:", only.notes.str.contains(r"\bL[1-5]\b").sum())
sp = S[spec & lmc].copy()
sp["Ltype"] = lab[spec & lmc].map(lambda t: ",".join(sorted(set(re.findall(r"\bL[1-5]\b", GEN.sub("", t))))))
print(f"specific L type: {len(sp)} rows; {sp.Ltype.value_counts().to_dict()}")
bare = S[(S["class"] == "LMC") & ~gen & ~spec]
print("class LMC without any L label:", len(bare))
print(bare[["serial_id", "root_now", "point_xyz", "side", "superclass", "class", "subclass", "type", "status", "proofreader",
            "notes"]].to_string(index=False))
out = sys.argv[1].replace("_seatable_aedes_main.tsv", "_seatable_L1-5_only.tsv")
only[["serial_id", "root_now", "point_xyz", "side", "label", "status", "proofreader", "annotator", "notes"]].to_csv(out, sep="\t", index=False)
print("written", out)
