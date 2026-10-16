"""Run the SAME clustering pipeline used for aedes MeTu (dotprops -> straight
nblast_allbyall -> Ward) on Drosophila MeTu from FAFB and male-CNS, colored by
KNOWN MeTu1-4 labels. Quantifies subtype recovery (adjusted Rand index).

One hemisphere (right) per dataset to avoid L/R splitting (no alignment used,
matching the aedes pipeline). Read-only.
"""
from faab.config import REPO
import os
import re
import numpy as np
import pandas as pd
import navis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

from fafbseg import flywire
from neuprint import NeuronCriteria as NC, fetch_neurons
from navis.interfaces import neuprint as nvneu
from faab.sources.neuprint import client, MALE_CNS

OUT = os.path.join(REPO, "figures")
PER_TYPE = 25
RESAMPLE = 1.0


def to_um(n):
    n = n.copy()
    u = str(getattr(n, "units", "") or "").lower()
    m = re.match(r"([\d.]+)\s*nanomet", u)
    f = (float(m.group(1)) / 1000.0) if m else None
    co = n.nodes[["x", "y", "z"]].values.astype(float)
    if f is None:
        f = 1 / 1000.0 if (co.max(0) - co.min(0)).max() > 5000 else 1.0
    n.nodes[["x", "y", "z"]] = co * f
    n.units = "um"
    return n


def dp(n):
    d = navis.make_dotprops(to_um(n), k=5, resample=RESAMPLE)
    d.units = "um"
    return d


def major(lbl):
    m = re.match(r"(MeTu\d)", lbl)
    return m.group(1) if m else lbl


def cluster_report(dps, labels, name):
    nl = navis.NeuronList(dps)
    ids = [str(i) for i in range(len(dps))]
    nl.set_neuron_attributes(ids, "name")
    nl.set_neuron_attributes(ids, "id")
    S = navis.nblast_allbyall(nl, n_cores=1, progress=False)
    D = 1 - (S.values + S.values.T) / 2
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method="ward")

    sub = list(labels)
    maj = [major(l) for l in labels]
    k_sub, k_maj = len(set(sub)), len(set(maj))
    ari_sub = adjusted_rand_score(sub, fcluster(Z, k_sub, "maxclust"))
    ari_maj = adjusted_rand_score(maj, fcluster(Z, k_maj, "maxclust"))
    print(f"\n[{name}] n={len(labels)} | subtypes={k_sub} major={k_maj}")
    print(f"  ARI subtype (k={k_sub}): {ari_sub:.3f}   ARI major MeTu1-4 (k={k_maj}): {ari_maj:.3f}")
    print("  major-type vs cluster (k=major):")
    print(pd.crosstab(pd.Series(maj, name="true"),
                      pd.Series(fcluster(Z, k_maj, "maxclust"), name="clust")).to_string())

    palette = {t: c for t, c in zip(sorted(set(maj)), plt.cm.tab10.colors)}
    fig, ax = plt.subplots(figsize=(24, 7))
    dendrogram(Z, labels=maj, leaf_rotation=90, leaf_font_size=4, color_threshold=0, ax=ax)
    for lab in ax.get_xticklabels():
        lab.set_color(palette.get(lab.get_text(), "k"))
    handles = [plt.Line2D([0], [0], color=palette[t], lw=3, label=t) for t in sorted(palette)]
    ax.legend(handles=handles, loc="upper right", ncol=2)
    ax.set_title(f"{name} MeTu clustering (NBLAST, Ward) — leaves colored by true MeTu1-4 "
                 f"(ARI={ari_maj:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fly_metu_{name}.png"), dpi=150)
    print(f"  saved fly_metu_{name}.png")


def fafb():
    res = flywire.search_annotations("MeTu", dataset="public", verbose=False)
    res = res[res["cell_type"].astype(str).str.match(r"MeTu\d") & (res["side"] == "right")]
    ids, labs = [], []
    for t, grp in res.groupby("cell_type"):
        g = grp.head(PER_TYPE)
        ids += g["root_id"].astype("int64").tolist()
        labs += [t] * len(g)
    id2 = dict(zip(ids, labs))
    sk = flywire.get_skeletons(ids, dataset=783, max_threads=2, omit_failures=True, progress=False)
    dps, dl = [], []
    for n in sk:
        if int(n.id) in id2 and len(n.nodes) >= 10:
            dps.append(dp(n)); dl.append(id2[int(n.id)])
    print(f"FAFB: fetched {len(dps)} MeTu (right)")
    cluster_report(dps, dl, "FAFB")


def mcns():
    c = client(MALE_CNS)
    ndf, _ = fetch_neurons(NC(type="MeTu.*", regex=True), client=c)
    side = next((col for col in ["somaSide", "side"] if col in ndf.columns), None)
    if "instance" in ndf.columns:
        ndf = ndf[ndf["instance"].astype(str).str.endswith("_R")]
    elif side:
        ndf = ndf[ndf[side].astype(str).str.upper().str.startswith("R")]
    ids, labs = [], []
    for t, grp in ndf.groupby("type"):
        if not re.match(r"MeTu\d", str(t)):
            continue
        g = grp.head(PER_TYPE)
        ids += g["bodyId"].astype("int64").tolist()
        labs += [t] * len(g)
    id2 = dict(zip(ids, labs))
    sk = nvneu.fetch_skeletons(ids, client=c)
    dps, dl = [], []
    for n in sk:
        if int(n.id) in id2 and len(n.nodes) >= 10:
            dps.append(dp(n)); dl.append(id2[int(n.id)])
    print(f"male-CNS: fetched {len(dps)} MeTu (right)")
    cluster_report(dps, dl, "male-CNS")


if __name__ == "__main__":
    fafb()
    mcns()
