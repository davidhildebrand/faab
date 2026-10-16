"""(b) Cluster aedes MeTu into morphological subtypes.
(a) Rank Dm2-candidates by cross-species Dm2 similarity (vs FAFB+Janelia Dm2
    templates and Mi1/Tm1 negatives), relative ranking (no absolute threshold).

Local only (uses cached dotprops). Plain NBLAST for within-aedes clustering
(same brain/orientation); nblast_align (scale-on) for cross-species ranking.
"""
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

OUT = os.path.join(REPO, "figures")
CALIB = os.path.join(OUT, "dm2_calib_neurons.pkl")
MIN_PTS = 10        # NBLAST needs enough points (technical floor, not completeness)
SIZE_CAP = 150.0    # um upper bound only (drop brain-spanning; keep small + partial)


def load(name):
    return pickle.load(open(os.path.join(OUT, name), "rb"))


def span(d):
    return float((d.points.max(0) - d.points.min(0)).max())


def name_nl(nl, labels):
    nl.set_neuron_attributes(labels, "name")
    nl.set_neuron_attributes(labels, "id")
    return nl


def metu_subtypes():
    metu_dp, roots = load("metu_dotprops.pkl")
    keep = [(d, r) for d, r in zip(metu_dp, roots) if len(d.points) >= MIN_PTS]
    dps = navis.NeuronList([d for d, _ in keep])
    labs = [str(r)[-6:] for _, r in keep]
    name_nl(dps, labs)
    S = navis.nblast_allbyall(dps, n_cores=6, progress=False)
    D = 1 - (S.values + S.values.T) / 2
    np.fill_diagonal(D, 0)
    Z = linkage(D[np.triu_indices(len(D), 1)], method="ward")
    print("\n=== (b) MeTu subtype clustering (213 MeTu) ===")
    for k in (4, 6, 8, 10, 12):
        sizes = pd.Series(fcluster(Z, k, criterion="maxclust")).value_counts().sort_index().tolist()
        print(f"  k={k:2d} cluster sizes: {sizes}")
    fig, ax = plt.subplots(figsize=(20, 6))
    dendrogram(Z, labels=labs, leaf_rotation=90, leaf_font_size=4, color_threshold=0.7 * D.max(), ax=ax)
    ax.set_title("Aedes MeTu morphological clustering (NBLAST, Ward)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "metu_dendrogram.png"), dpi=150)
    pd.DataFrame({"root6": labs, "k8": fcluster(Z, 8, criterion="maxclust")}).to_csv(
        os.path.join(OUT, "metu_clusters.csv"), index=False)
    print("  saved metu_dendrogram.png + metu_clusters.csv")


def dm2_rank():
    cand_dp, cand_ids = load("candidate_dotprops.pkl")
    cal = load(CALIB)
    fafb = [l for l in cal if l.startswith("Dm2.FW.")][:3]
    jan = [l for l in cal if l.startswith("Dm2.OL.")][:3]
    negs = [l for l in cal if l.startswith("NEG.Mi1.") or l.startswith("NEG.Tm1.")]
    tmpl_labels = fafb + jan
    tmpl = name_nl(navis.NeuronList([cal[l] for l in tmpl_labels]), tmpl_labels)
    neg = name_nl(navis.NeuronList([cal[l] for l in negs]), negs)

    # candidates: size-cap (upper only) + min-points (technical)
    rows = [(d, i) for d, i in zip(cand_dp, cand_ids)
            if len(d.points) >= MIN_PTS and span(d) <= SIZE_CAP]
    cdps = navis.NeuronList([d for d, _ in rows])
    clabs = [str(i) for _, i in rows]
    name_nl(cdps, clabs)
    print(f"\n=== (a) Dm2 candidate ranking ===")
    print(f"candidates after size-cap(<= {SIZE_CAP:.0f}um)+minpts: {len(cdps)} / {len(cand_dp)}")

    dm2_s = navis.nblast_align(cdps, tmpl, align_method="rigid", two_way_align=False,
                               n_cores=6, progress=False, align_kwargs={"scale": True})
    neg_s = navis.nblast_align(cdps, neg, align_method="rigid", two_way_align=False,
                               n_cores=6, progress=False, align_kwargs={"scale": True})
    # nblast_align(query=cdps, target=tmpl): rows=candidates, cols=templates -> mean over axis=1
    df = pd.DataFrame({
        "root_id": [int(x) for x in clabs],
        "dm2_score": dm2_s.mean(axis=1).values,        # per-candidate mean over Dm2 templates
        "neg_score": neg_s.mean(axis=1).values,
        "span_um": [round(span(d), 1) for d in cdps],
        "n_pts": [len(d.points) for d in cdps],
    })
    df["dm2_minus_neg"] = (df["dm2_score"] - df["neg_score"]).round(3)
    df = df.sort_values("dm2_score", ascending=False).reset_index(drop=True)
    df.to_csv(os.path.join(OUT, "dm2_ranked_candidates.csv"), index=False)

    # reference: Dm2-Dm2 cross-dataset ~0.55, Mi1/Tm1 baseline ~0.42 (from calibration)
    print(df.head(20).round(3).to_string(index=False))
    print(f"\ncandidates with dm2_score > neg_score: {(df['dm2_minus_neg'] > 0).sum()} / {len(df)}")
    print(f"top-20 mean span: {df.head(20)['span_um'].median():.0f} um")
    print("saved dm2_ranked_candidates.csv")


if __name__ == "__main__":
    metu_subtypes()
    dm2_rank()
