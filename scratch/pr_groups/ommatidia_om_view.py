"""Members of some ommatidia of an ommatidia.py run with the PRs-tab group and class of each (2026-11-01 debugging).
Network-free. Usage: ommatidia_om_view.py RUN OM [OM ...]   (e.g. om017 om035)"""
import os
import sys

import pandas as pd

from faab.config import fig

run, oms = sys.argv[1], sys.argv[2:]
d = fig(f"ommatidia/{run}")
O = pd.read_csv(os.path.join(d, "ommatidia.tsv"), sep="\t", dtype={"root": str}).fillna("")
tab = pd.read_csv(os.path.join(d, "tab_roots.tsv"), sep="\t", dtype=str).fillna("")
where = tab.groupby("root_now").apply(lambda t: ", ".join(f"g{g} {c}" for g, c in zip(t.group, t.cls))).to_dict()
for om in oms:
    print(om)
    for x in O[O.ommatidium == om].itertuples(index=False):
        print(f"  {x.slot:6s} {x.root}  {x.verdict:12s} {x.dist_um:4.2f} um  tab: {where.get(x.root, '-')}")
