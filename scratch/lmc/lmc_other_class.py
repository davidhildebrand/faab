"""SeaTable rows typed L1..L5 whose class is not LMC (to decide whether they join the LMC typing set). Network-free.
Usage: lmc_other_class.py SEATABLE.tsv"""
import re
import sys

import pandas as pd

S = pd.read_csv(sys.argv[1], sep="\t", dtype=str).fillna("")
lab = S[["class", "subclass", "subsubclass", "type"]].agg(" / ".join, axis=1)
hit = lab.map(lambda t: bool(re.search(r"\bL[1-5]\b(?!\s*-)", t)) and not re.search(r"\bL1\s*-\s*L?5\b", t))
O = S[hit & (S["class"] != "LMC")]
print(f"{len(O)} rows typed L1..L5 outside class LMC")
print(O.groupby(["superclass", "class", "subclass", "subsubclass", "type", "side"]).size().to_string())
print(O[["serial_id", "root_now", "superclass", "class", "subclass", "type", "status", "proofreader"]].head(40).to_string(index=False))
