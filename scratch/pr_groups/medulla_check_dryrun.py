"""Dry run of the groups_refine medulla check (2026-10-31) on the stored tracking results (figures/refine/details_g*.tsv
and presyn.tsv): which additions it would block and which members it would flag. Network-free, writes nothing.
Usage: medulla_check_dryrun.py"""
import glob
import importlib.util
import os

import numpy as np
import pandas as pd

from faab.config import REPO, fig

spec = importlib.util.spec_from_file_location("groups_refine", os.path.join(REPO, "scripts", "pr_groups", "groups_refine.py"))
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
DAVID = {"648518347556420586": "LMC", "648518347544284097": "LMC", "648518347576277331": "PR + glia merge",
         "648518347582104647": "PR + glia merge", "648518347570497932": "PR + glia merge"}

D = pd.concat([pd.read_csv(p, sep="\t", dtype={"root": str}) for p in sorted(glob.glob(fig("refine/details_g*.tsv")))])
D = D.fillna("")
presyn = G.load_presyn()
fake = {"x": dict(status="ok", core=set(D.root), listed=set())}
me = G.medulla_shares(fake, presyn)
D["me"] = D.root.map(me)
added = D.role.str.startswith(("census", "PR-like"))
D["would"] = ""
D.loc[added & (D.me >= G.ME_NOT_PR) & (D.assigned != "R7"), "would"] = "not added"  # stored R7 choice as a proxy
cls = D.assigned.map({"R7": "R7", "R8": "R8", "outer": "R1-R6", "extra": "R1-R6"}).fillna("")
flag = [bool(c) and bool(G.medulla_flag(m if pd.notna(m) else None, c)) for c, m in zip(cls, D.me)]
D.loc[np.array(flag) & (D.would == ""), "would"] = "flagged"
D["David"] = D.root.map(DAVID).fillna("")
print(f"{D.root.nunique()} roots in {D.group.nunique()} groups; with output sites: {D.me.notna().sum()} rows")
print("would:", D.would.value_counts().to_dict())
show = D[(D.would != "") | (D.David != "")].sort_values(["would", "me"], ascending=[True, False])
print(show[["group", "root", "assigned", "n_pre", "me", "would", "David", "role"]].to_string(index=False, max_colwidth=50))
