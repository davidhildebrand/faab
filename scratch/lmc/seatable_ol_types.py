"""SeaTable aedes_main types in the optic lobe that could be confused with LMCs (2026-11-01, LMC search): counts per
superclass / class / type and side for ol_intrinsic / visual_projection / visual_centrifugal rows and any row whose type
names C2, C3, T1, Lai, Lawf, L1-L5. Network-free. Usage: seatable_ol_types.py SEATABLE.tsv"""
import re
import sys

import pandas as pd

S = pd.read_csv(sys.argv[1], sep="\t", dtype=str).fillna("")
lab = S[["class", "subclass", "subsubclass", "type"]].agg(" ".join, axis=1)
pick = S.superclass.isin(["ol_intrinsic", "visual_projection", "visual_centrifugal"]) | \
    lab.map(lambda t: bool(re.search(r"\b(C2|C3|T1|Lai|Lawf\d?|L[1-5])\b", t)))
O = S[pick]
print(O.groupby(["superclass", "class", "type", "side"]).size().unstack(fill_value=0).to_string())
