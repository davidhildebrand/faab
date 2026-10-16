"""(c) Side-by-side Dm2 morphology: Drosophila Dm2 vs top FAAB candidates,
rigidly aligned to a fly Dm2 reference. Local (cached dotprops)."""
from faab.config import REPO
import os
import pickle
import numpy as np
import pandas as pd
import navis
from navis.transforms.align import align_rigid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(REPO, "figures")
cal = pickle.load(open(os.path.join(OUT, "dm2_calib_neurons.pkl"), "rb"))
cand_dp, cand_ids = pickle.load(open(os.path.join(OUT, "candidate_dotprops.pkl"), "rb"))
cand = {int(i): d for i, d in zip(cand_ids, cand_dp)}
rank = pd.read_csv(os.path.join(OUT, "dm2_ranked_candidates.csv"))

ref = cal["Dm2.FW.720575940641270736"]            # fly FAFB Dm2 medoid (alignment reference)
jan = cal[sorted(l for l in cal if l.startswith("Dm2.OL."))[0]]


def al(n):
    a = align_rigid(navis.NeuronList(n), target=ref, progress=False)
    a = a[0] if isinstance(a, tuple) else a
    return a[0]


# the CSV root_ids are float64-corrupted; map each back to the exact cached id by nearest match
cand_arr = np.array(sorted(cand), dtype=np.int64)


def exact(rid):
    j = int(np.abs(cand_arr - int(rid)).argmin())
    return int(cand_arr[j]) if abs(int(cand_arr[j]) - int(rid)) < 1000 else None


# write a corrected worklist CSV with exact root_ids
rank_fixed = rank.copy()
rank_fixed["root_id"] = rank_fixed["root_id"].map(lambda v: exact(v) or v)
rank_fixed.to_csv(os.path.join(OUT, "dm2_ranked_candidates_fixed.csv"), index=False)

# top FAAB candidates: high dm2_score, Dm2-sized (<=100 um), present in cache
picks = []
for _, r in rank.sort_values("dm2_score", ascending=False).iterrows():
    rid = exact(r["root_id"])
    if rid is not None and rid in cand and r["span_um"] <= 100:
        picks.append((rid, r["span_um"], r["dm2_score"]))
    if len(picks) >= 5:
        break

cols = [("Dm2 FAFB\n(fly)", ref, "navy"), ("Dm2 Janelia\n(fly)", al(jan), "navy")]
for rid, sp, sc in picks:
    cols.append((f"FAAB {str(rid)[-6:]}\n{sp:.0f}um  s={sc:.2f}", al(cand[rid]), "firebrick"))

views = [(("x", "y"), "XY"), (("x", "z"), "XZ")]
fig, axes = plt.subplots(2, len(cols), figsize=(2.4 * len(cols), 5.4), squeeze=False)
for c, (title, n, color) in enumerate(cols):
    for r, (view, vn) in enumerate(views):
        ax = axes[r][c]
        navis.plot2d(n, ax=ax, method="2d", view=view, color=color, lw=1.0)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(title, fontsize=8)
        if c == 0:
            ax.set_ylabel(vn)
fig.suptitle("Dm2: Drosophila (navy) vs top FAAB candidates (red) — rigid-aligned to fly Dm2", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT, "dm2_sidebyside.png"), dpi=150)
print("picks:", [(str(p[0])[-6:], round(p[1]), round(p[2], 2)) for p in picks])
print("saved dm2_sidebyside.png")
