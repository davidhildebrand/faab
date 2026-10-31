"""Neighboring photoreceptors of a left cartridge via shared lamina targets (snapshot + one CAVE catch-up; skeletons
one throttled sequential fetch, cached in figures/pr_left_skels.pkl).

Targets = cells with >= 3 synapses from the cartridge's R1-6; inputs of the targets with >= 3 synapses onto one
target; PR signature = >= 40 presynaptic sites, inputs <= 25% of outputs, >= 80% of the presynaptic sites in LA_L
(SeaTable R1-6 on the left: 151-325 outputs, 0-14 inputs, 97-100% LA_L). Each signature cell's L2 skeleton gives an
axis (PCA, oriented by its presynaptic median = lamina point) and a retina-ward length; cells reaching >= 45 um are
axis helpers. Usage: left_helpers.py L1 [L2 ...]
Result 2026-10-29 (L1 L2): 72 targets <- 1431 inputs; 587 PR-signature cells, length p50 10 um, only 48 >= 45 um and
those disagree in direction (p50 44 deg) -> no usable axis; the neighborhood is lamina-terminal pieces too."""
import os
import pickle
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import caveclient
from fafbseg import flywire

from faab import ol_meshes, synsnap
from faab.clients.cave import DATASTACK, throttle
from faab.config import FIGURES
from faab.partners.top10_xyz import VOX
from faab.config import fig

SCACHE = fig("pr_left_skels.pkl")
VOX_UM = np.array(VOX) / 1000.0
known = pd.concat([pd.read_csv(fig(f), sep="\t", dtype={"r16_ids": str, "r78_ids": str})
                   for f in ("pr_left_lattice_known.tsv", "pr_left_lattice_noaxis.tsv")]).sort_values("cartridge").reset_index(drop=True)
st = pd.read_csv(fig("pr_left_lattice_seatable_prs.tsv"), sep="\t", dtype=str)
seatable = set(int(x) for x in st.root)
want = sys.argv[1:] or ["L1", "L2"]

client = caveclient.CAVEclient(DATASTACK)
S = synsnap.Snapshot()
r16 = [int(x) for c in want for x in known.set_index("cartridge").loc[c, "r16_ids"].split(",")]
e = S.edges(pre=r16, when="now", client=client)
targets = sorted(set(e[e.n >= 3].post) - seatable)
e2 = S.edges(post=targets, when="now", client=client)
cands = sorted(set(e2[e2.n >= 3].pre) - seatable)
print(f"{want}: {len(r16)} R1-6 -> {len(targets)} targets (>= 3 syn) <- {len(cands)} inputs (>= 3 syn onto one target)")

pre = S.synapses(pre=cands, when="now", client=client)
post = S.synapses(post=cands, when="now", client=client)
npre = pre.pre_root.value_counts()
npost = post.post_root.value_counts()
c1 = [c for c in cands if npre.get(c, 0) >= 40 and npost.get(c, 0) <= 0.25 * npre.get(c, 0)]
pos = S.static(pre[pre.pre_root.isin(c1)].id.to_numpy(), columns=["pre_x", "pre_y", "pre_z"])
pp = pre[pre.pre_root.isin(c1)].merge(pos, on="id")
pp["region"] = ol_meshes.assign(pp[["pre_x", "pre_y", "pre_z"]].to_numpy(float) * VOX_UM, ["LA_L", "ME_L", "LO_L", "LOP_L"])
frac = pp.groupby("pre_root").region.apply(lambda s: float((s == "LA_L").mean()))
sig = [c for c in c1 if frac.get(c, 0) >= 0.8]
print(f"PR signature: {len(c1)} with >= 40 outputs and inputs <= 25% of outputs; {len(sig)} of them >= 80% in LA_L")

sk = pickle.load(open(SCACHE, "rb")) if os.path.exists(SCACHE) else {}
need = [c for c in sig if c not in sk]
print(f"skeletons to fetch: {len(need)}", flush=True)
t = time.time()
for k, r in enumerate(need):  # one root at a time, >= 5 s apart (a list call fires its requests back to back)
    t = throttle(t)
    for s in flywire.get_l2_skeleton([r], dataset=DATASTACK, max_threads=1, omit_failures=True, progress=False):
        sk[int(s.id)] = s
    if (k + 1) % 50 == 0:
        pickle.dump(sk, open(SCACHE, "wb"))
if need:
    pickle.dump(sk, open(SCACHE, "wb"))

rows = []
for c in sig:
    if c not in sk:
        continue
    X = sk[c].nodes[["x", "y", "z"]].values / 1000.0
    lam = np.median(pp[pp.pre_root == c][["pre_x", "pre_y", "pre_z"]].to_numpy(float) * VOX_UM, axis=0)
    m = X.mean(0)
    ax = np.linalg.svd(X - m, full_matrices=False)[2][0]
    if (lam - m) @ ax < 0:
        ax = -ax
    s_ = (X - m) @ ax
    ret = X[s_ < s_.min() + 35.0].mean(0)
    rows.append(dict(root=c, n_pre=int(npre[c]), n_post=int(npost.get(c, 0)), la_frac=round(frac[c], 2),
                     length_um=round(float(np.linalg.norm(lam - ret)), 1), lam_x=lam[0], lam_y=lam[1], lam_z=lam[2],
                     ax_x=ax[0], ax_y=ax[1], ax_z=ax[2]))
h = pd.DataFrame(rows).sort_values("length_um", ascending=False)
long_ = h[h.length_um >= 45]
print(f"skeletons {len(h)}; length p10/50/90 {h.length_um.quantile([0.1, 0.5, 0.9]).round(0).tolist()}; >= 45 um: {len(long_)}")
for cn in want:
    k = known.set_index("cartridge").loc[cn]
    lam = np.array([k.lam_x, k.lam_y, k.lam_z])
    if len(long_):
        P = long_[["lam_x", "lam_y", "lam_z"]].values
        A = long_[["ax_x", "ax_y", "ax_z"]].values
        d = np.linalg.norm(P - lam, axis=1)
        w = 1.0 / np.maximum(d, 1e-3) ** 2
        m = (A * w[:, None]).sum(0)
        m /= np.linalg.norm(m)
        angs = np.degrees(np.arccos(np.clip(A @ m, -1, 1)))
        print(f"{cn}: {len(long_)} long helpers at {np.sort(d)[:8].round(0).tolist()} um; IDW axis {m.round(3).tolist()}; "
              f"helper angles to it p50/p90 {np.percentile(angs, [50, 90]).round(0).tolist()} deg; within 30 um: "
              f"{int((d < 30).sum())}")
out = fig("pr_left_helpers.tsv")
h.to_csv(out, sep="\t", index=False)
print(f"written {out}")
