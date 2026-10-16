"""Inspect MeTu cluster structure: report top merge heights / gaps to find the
natural number of clusters, and redraw the dendrogram with explicit cut lines."""
from faab.config import REPO
import os
import pickle
import numpy as np
import pandas as pd
import navis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

OUT = os.path.join(REPO, "figures")
metu_dp, roots = pickle.load(open(os.path.join(OUT, "metu_dotprops.pkl"), "rb"))
keep = [(d, r) for d, r in zip(metu_dp, roots) if len(d.points) >= 10]
dps = navis.NeuronList([d for d, _ in keep])
labs = [str(r)[-6:] for _, r in keep]
dps.set_neuron_attributes(labs, "name")
dps.set_neuron_attributes(labs, "id")

S = navis.nblast_allbyall(dps, n_cores=1, progress=False)
D = 1 - (S.values + S.values.T) / 2
np.fill_diagonal(D, 0)
Z = linkage(squareform(D, checks=False), method="ward")

h = Z[:, 2]
top = h[::-1][:12]                      # 12 largest merge heights (descending)
print("top 12 merge heights:", np.round(top, 2))
print("gaps below each (height drop to next):", np.round(-np.diff(top), 2))
print("=> a big gap after the Nth merge suggests N clusters is natural\n")
for k in range(2, 9):
    sizes = sorted(pd.Series(fcluster(Z, k, "maxclust")).value_counts().tolist(), reverse=True)
    print(f"k={k}: cluster sizes {sizes}")

# redraw with explicit cut lines for k=2,3,4 (thresholds between successive top merges)
def thr(k):
    return (Z[-(k - 1), 2] + Z[-k, 2]) / 2

fig, ax = plt.subplots(figsize=(22, 7))
dendrogram(Z, labels=labs, leaf_rotation=90, leaf_font_size=3.5,
           color_threshold=thr(4), ax=ax)
for k, c in [(2, "red"), (3, "purple"), (4, "green")]:
    ax.axhline(thr(k), ls="--", lw=1, color=c, label=f"cut for k={k} (h={thr(k):.2f})")
ax.legend(loc="upper right")
ax.set_title("Aedes MeTu clustering — colored at k=4; dashed lines = k=2/3/4 cuts")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "metu_dendrogram_cuts.png"), dpi=150)
print("\nsaved metu_dendrogram_cuts.png")
