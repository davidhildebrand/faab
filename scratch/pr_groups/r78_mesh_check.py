"""Where do the right-side PR classes of data/pr_groups.tsv make their outputs? (snapshot + one CAVE catch-up)

Classes come from the group refinement (R8 = central axon of the bundle, R7 = the ring cell with medulla outputs or
the fewest lamina outputs; memory faab-groups-refine), not from synapse position. Per cell: outputs and inputs, and
the share of outputs inside LA_R, ME_R or neither mesh. Network-free: roots as listed, newest local checkpoint.
Question (David, 2026-10-29): do confirmed R8s put >= 20% of their outputs in the medulla, as bundles_verify.py's
'R7/R8-like' rule assumes?"""
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from faab import ol_meshes, synsnap
from faab.config import DATA, FIGURES
from faab.partners.sheet import utc_stamp
from faab.config import fig

VOX_UM = np.array([0.016, 0.016, 0.045])
g = pd.read_csv(os.path.join(DATA, "pr_groups.tsv"), sep="\t", dtype={"root_id_curr": str})
g["root"] = g.root_id_curr.astype("int64")
# offline: CAVE was down on 2026-10-29 (502 in the catch-up, then 503); roots as listed (a live check minutes before
# found 1 of 596 expired) against the newest local checkpoint, no catch-up
desc = {}
S = synsnap.Snapshot()
WHEN = "latest"
cand = sorted(set(g.root))
pre = S.synapses(pre=cand, when=WHEN, positions=True)
post = S.synapses(post=cand, when=WHEN)
print(f"synapses at checkpoint {S.at(WHEN)}; cells without any output there: "
      f"{len(set(cand) - set(pre.pre_root))} of {len(cand)} (roots edited after the checkpoint)")
npre = pre.pre_root.value_counts()
npost = post.post_root.value_counts()
g["root_now"] = [r if r not in desc else max(desc[r], key=lambda d: npre.get(d, 0)) for r in g.root]
pre = pre[pre.pre_root.isin(set(g.root_now))]
reg = pd.Series(ol_meshes.assign(pre[["pre_x", "pre_y", "pre_z"]].to_numpy(float) * VOX_UM, ["LA_R", "ME_R"]), index=pre.index)
share = pd.crosstab(pre.pre_root, reg, normalize="index")
for c in ("LA_R", "ME_R", "none"):
    if c not in share:
        share[c] = 0.0
g["n_pre"] = g.root_now.map(npre).fillna(0).astype(int)
g["n_post"] = g.root_now.map(npost).fillna(0).astype(int)
g["LA"] = g.root_now.map(share.LA_R).round(2)
g["ME"] = g.root_now.map(share.ME_R).round(2)
g["outside"] = g.root_now.map(share.none).round(2)
g["confirmed"] = g.status.fillna("").str.lower().isin(["proofread", "checked"])

pd.set_option("display.width", 250)
for label, sub_all in (("all", g), ("proofread/checked only", g[g.confirmed])):
    print(f"\n{label}:")
    for cls, sub in sub_all.groupby("class"):
        s = sub[sub.n_pre >= 10]
        q = lambda c: s[c].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(2).tolist()
        print(f"  {cls}: n {len(sub)} ({len(s)} with >= 10 outputs); outputs p50 {s.n_pre.median():.0f}, inputs p50 "
              f"{s.n_post.median():.0f}; LA share p10/25/50/75/90 {q('LA')}; ME share {q('ME')}; outside both {q('outside')}; "
              f">= 20% in ME: {int((s.ME >= 0.2).sum())}; >= 80% in LA: {int((s.LA >= 0.8).sum())}")
out = fig(f"{utc_stamp()}_pr_classes_mesh_shares.tsv")
g.drop(columns=["root"]).to_csv(out, sep="\t", index=False)
print(f"\nwritten {out}")
