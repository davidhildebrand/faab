"""Do version 2's disagreements fall on the lead groups version 1 itself was unsure about? (David 2026-11-01: is v2
worth keeping.) Version 1 clean = 6 outer + R7 + R8, no extras, ring with R7 (figures/refine_<V1RUN>/summary.tsv);
agreement with version 2 from figures/ommatidia/<RUN>/compare.tsv (exact / jaccard >= 0.75 / split or less).
Network-free. Usage: ommatidia_vs_v1_quality.py RUN V1RUN"""
import os
import sys

import pandas as pd

from faab.config import fig

run, v1 = sys.argv[1], sys.argv[2]
Cmp = pd.read_csv(os.path.join(fig(f"ommatidia/{run}"), "compare.tsv"), sep="\t").fillna("")
S = pd.read_csv(os.path.join(fig(f"refine_{v1}"), "summary.tsv"), sep="\t").fillna("")
S = S[S.status == "ok"].copy()
S["v1"] = ["clean" if float(x.outer) == 6 and float(x.extras) == 0 and x.R7 not in ("", "-") and x.R8 not in ("", "-")
           and str(x.ring).startswith("ring with R7") else "unsure" for x in S.itertuples(index=False)]
M = Cmp[Cmp.section == "leads"].merge(S[["group", "v1"]], on="group", how="left").fillna({"v1": "not tracked"})
M["v2"] = ["exact" if j == 1 else "mostly (>= 0.75)" if j >= 0.75 else "differs" for j in M.jaccard]
print(pd.crosstab(M.v1, M.v2, margins=True).to_string())
print("\nv1-clean groups that v2 splits or changes a lot:",
      ", ".join(f"g{x.group} ({x.jaccard})" for x in M[(M.v1 == "clean") & (M.v2 == "differs")].itertuples()) or "-")
