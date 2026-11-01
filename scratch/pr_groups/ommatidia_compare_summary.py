"""Summary of an ommatidia.py compare (2026-11-01): the lead groups by kind of difference to the PRs tab (exact; v2 keeps
all and adds; v2 drops members; split over ommatidia; shares an ommatidium with another group), with the reasons for
dropped members tallied, and the rows of some groups in full.
Network-free. Usage: ommatidia_compare_summary.py RUN [G ...]"""
import os
import re
import sys
from collections import Counter

import pandas as pd

from faab.config import fig

run, show = sys.argv[1], [int(g) for g in sys.argv[2:]]
R = pd.read_csv(os.path.join(fig(f"ommatidia/{run}"), "compare.tsv"), sep="\t").fillna("")
ld = R[R.section == "leads"].copy()
shared = ld.ommatidium.value_counts()
ld["kind"] = [("exact" if x.jaccard == 1 else
               "shares an ommatidium" if x.ommatidium and shared[x.ommatidium] > 1 else
               "split" if x.other_om else
               "v2 adds only" if x.kept == x.n_tab else
               "v2 drops" + (" and adds" if x.added else "")) for x in ld.itertuples()]
print(ld.kind.value_counts().to_string())
why = Counter(re.sub(r"om\d+", "omN", w) for x in ld.lost for w in re.findall(r"\(([^)]*)\)", x))
print("\nwhy tab members are not in the group's ommatidium:")
for w, n in why.most_common():
    print(f"  {n:4d}  {w}")
for k in ("shares an ommatidium", "split"):
    print(f"\n{k}: " + ", ".join(f"g{x.group} ({x.ommatidium}{'; ' + x.other_om if x.other_om else ''})"
                                 for x in ld[ld.kind == k].itertuples()))
for g in show:
    x = R[R.group == g].iloc[0]
    print(f"\ngroup {g}: tab {x.n_tab}, {x.ommatidium} {x.n_om}, kept {x.kept}, jaccard {x.jaccard}\n  added: {x.added or '-'}"
          f"\n  lost: {x.lost or '-'}\n  other ommatidia: {x.other_om or '-'}")
