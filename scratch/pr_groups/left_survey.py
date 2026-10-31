"""Survey of the SeaTable photoreceptors on the LEFT side, before running the PR-group finder there (read-only).

The right-side finder (scripts/pr_groups/lattice.py -> src/faab/pr/bundles.py -> scripts/pr_groups/bundles_cluster.py)
starts from known cartridges: groups of R1-R6 with a pooled axis and a lamina point. This lists what SeaTable has on
side L: labels, notes (cartridge numbers?), current roots, and per root the presynaptic / postsynaptic counts and the
neuropils of the presynaptic sites (local synapse snapshot + one CAVE catch-up), so the seed set can be judged.
Output: figures/<stamp>_pr_left_survey.tsv; summary printed.
"""
import os
import re
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import caveclient

from faab import ol_meshes, synsnap
from faab.clients.cave import DATASTACK, throttle
from faab.config import FIGURES
from faab.partners.sheet import utc_stamp
from faab.config import fig

SERVER = "https://flytable.mrc-lmb.cam.ac.uk/"
CLS = ("superclass", "class", "subclass", "subsubclass", "type")
PR_RE = r"^R\d|photorecept"
VOX_UM = np.array([0.016, 0.016, 0.045])


def s(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v) else str(v).strip()


def main():
    import seaserpent as ss
    meta = ss.Table("aedes_main", server=SERVER).to_frame()
    meta.columns = [str(c) for c in meta.columns]
    lab = meta.apply(lambda row: " / ".join(x for x in (s(row.get(c)) for c in CLS) if x), axis=1)
    is_pr = meta["type"].map(s).str.contains(PR_RE, case=False) | meta["subclass"].map(s).str.contains(PR_RE, case=False) \
        | meta["class"].map(s).str.contains("photorecept", case=False)
    pr = meta[is_pr].copy()
    pr["label"] = lab[pr.index]
    pr["side_"] = pr["side"].map(s)
    print(f"SeaTable aedes_main rows: {len(meta)}; photoreceptor-labeled: {len(pr)}")
    print(pd.crosstab(pr["label"], pr["side_"]).to_string())

    L = pr[pr.side_ == "L"].copy()
    L["notes_"] = L["notes"].map(s) if "notes" in L else ""
    L["cart"] = L["notes_"].str.extract(r"cartridge\s*(\d+)", flags=re.I)[0]
    sv = L["supervoxel_id"].map(s)
    client = caveclient.CAVEclient(DATASTACK)
    throttle(time.time())
    ok = sv.str.isdigit()
    L["root"] = 0
    L.loc[ok, "root"] = np.asarray(client.chunkedgraph.get_roots([int(x) for x in sv[ok]])).astype(np.int64)
    roots = sorted(set(int(r) for r in L.root if r))
    print(f"\nside L: {len(L)} rows, {len(roots)} distinct current roots; rows without supervoxel {int((~ok).sum())}; "
          f"notes with a cartridge number {int(L.cart.notna().sum())}")

    synsnap.refresh()
    S = synsnap.Snapshot()
    pre = S.synapses(pre=roots, when="now", positions=True, client=client)
    post = S.synapses(post=roots, when="now", client=client)
    rows = []
    for r in L.itertuples():
        p = pre[pre.pre_root == r.root]
        pts = p[["pre_x", "pre_y", "pre_z"]].to_numpy(float) * VOX_UM
        fr = ol_meshes.summary(ol_meshes.neuropil_fractions(pts)) if len(pts) else ""
        med = np.median(pts, axis=0) / VOX_UM if len(pts) else [np.nan] * 3
        rows.append(dict(root=r.root, label=r.label, cart=s(r.cart), point_xyz=s(getattr(r, "point_xyz", "")),
                         n_pre=len(p), n_post=int((post.post_root == r.root).sum()), pre_neuropils=fr,
                         pre_median_vox=", ".join(str(int(round(v))) if np.isfinite(v) else "" for v in med),
                         notes=r.notes_))
    out = pd.DataFrame(rows).sort_values(["label", "cart", "n_pre"], ascending=[True, True, False])
    path = fig(f"{utc_stamp()}_pr_left_survey.tsv")
    out.to_csv(path, sep="\t", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(out.drop(columns=["notes"]).to_string(index=False))
    print("\nnotes (distinct):", sorted(set(out.notes)))
    print(f"\nwritten {path}")


if __name__ == "__main__":
    main()
