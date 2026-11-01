"""Report what a groups_refine.py --run found besides the medulla notes (2026-11-01), for David to decide on; nothing
is written to the PRs tab. Per group, from figures/refine_<RUN>/details_g*.tsv: axons the run would add (PR-like or
census axons in the bundle core, not listed in the group), with whether the root is listed anywhere in the tab; listed
members outside the bundle core (OFF-BUNDLE) whose tab note does not say so yet; axons not added because most of their
output sites are in the medulla; and groups the run could not track.
Network-free. Writes tmp/refine_report_<RUN>.tsv. Usage: refine_report.py RUN BOOK"""
import glob
import os
import sys

import pandas as pd

from faab.config import TMP, fig
from faab.pr import prs_tab as T

run, book = sys.argv[1], sys.argv[2]
rdir = fig(f"refine_{run}")
ws = T.load(book)[1]
where, notes = {}, {}
for b in T.blocks(ws):
    for r in b.rows:
        for k in ("root_id_curr", "root_id_orig"):
            root = T.get(ws, b, r, k)
            if root:
                where.setdefault(root, set()).add(b.group)
                notes[(b.group, root)] = T.get(ws, b, r, "notes")

rows = []
for path in sorted(glob.glob(os.path.join(rdir, "details_g*.tsv")), key=lambda p: int(p.split("_g")[-1][:-4])):
    D = pd.read_csv(path, sep="\t", dtype={"root": str}).fillna("")
    for x in D.itertuples(index=False):
        g, me = int(x.group), float(x.me_share) if x.me_share != "" else None
        base = dict(group=g, root=x.root, assigned=x.assigned, n_pre=x.n_pre,
                    me_share="" if me is None else f"{100 * me:.0f}%")
        if x.role.startswith(("PR-like", "census")):
            other = sorted(where.get(x.root, set()) - {g})
            rows.append(dict(base, finding="would add" + (" (medulla output)" if me and me >= 0.1 and x.assigned != "R7" else ""),
                             detail=x.role + (f"; listed in group {', '.join(map(str, other))}" if other else "; not in the tab")))
        elif x.role == "listed, OFF-BUNDLE":
            if "OFF-BUNDLE" not in notes.get((g, x.root), "").upper():
                rows.append(dict(base, finding="off-bundle (new)", detail=notes.get((g, x.root), "")))
        elif "not this bundle's R7" in x.role:
            other = sorted(where.get(x.root, set()))
            rows.append(dict(base, finding="not added (medulla)",
                             detail=x.role.split(" (")[0] + (f"; listed in group {', '.join(map(str, other))}" if other else "")))
S = pd.read_csv(os.path.join(rdir, "summary.tsv"), sep="\t").fillna("")
for x in S[S.status != "ok"].itertuples(index=False):
    rows.append(dict(group=int(x.group), root="", assigned="", n_pre="", me_share="", finding="not tracked", detail=x.status))
R = pd.DataFrame(rows).sort_values(["group", "finding"], kind="stable")
out = os.path.join(TMP, f"refine_report_{run}.tsv")
R[["group", "finding", "root", "assigned", "n_pre", "me_share", "detail"]].to_csv(out, sep="\t", index=False)
print(R.finding.value_counts().to_string())
print("groups:", R.groupby("finding").group.apply(lambda s: ", ".join(map(str, sorted(set(s))))).to_string())
print("written", out)
