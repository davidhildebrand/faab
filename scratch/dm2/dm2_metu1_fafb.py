"""FAFB Dm2 -> MeTu1 connection stats (right hemisphere) + Dm2 morphology
consistency check between FAFB and Janelia (mCNS/optic-lobe).

Read-only. Connection query via fafbseg get_adjacency on FAFB public; morphology
consistency reuses the cached Dm2 dotprops from dm2_calibration.py.
"""
from faab.config import REPO
import os
import pickle
import numpy as np
import pandas as pd
import navis
from fafbseg import flywire
from faab.config import fig

OUT = os.path.join(REPO, "figures")
CACHE = fig("dm2_calib_neurons.pkl")


def fafb_dm2_to_metu1():
    dm2 = pd.read_csv(os.path.join(REPO, "data", "flywire_dm2.csv"))
    dm2_r = dm2[dm2["side"] == "right"]["root_id"].astype("int64").tolist()
    res = flywire.search_annotations("MeTu1", exact=True, dataset="public", verbose=False)
    m1 = res[res["cell_type"].astype(str) == "MeTu1"]
    m1_r = m1[m1["side"] == "right"]["root_id"].astype("int64").tolist()
    print(f"FAFB right: Dm2={len(dm2_r)}  MeTu1={len(m1_r)} -> adjacency ...")

    cache = fig("fafb_dm2_metu1_edges.csv")
    if os.path.exists(cache):
        e = pd.read_csv(cache)
    else:
        e = flywire.get_adjacency(dm2_r, m1_r, square=False, dataset="public", progress=False)
        e.to_csv(cache, index=False)
    e = e[e["weight"] > 0]                       # edge list: source(Dm2), target(MeTu1), weight(synapses)
    w = e["weight"].astype(float)
    dm2_per = e.groupby("target")["source"].nunique()
    tot_per = e.groupby("target")["weight"].sum()
    print("\n=== FAFB Dm2 -> MeTu1 (right) ===")
    print(f"connections: {len(e)} | distinct Dm2: {e['source'].nunique()} | MeTu1 targeted: {e['target'].nunique()}")
    print(f"synapses/connection: mean {w.mean():.1f} median {w.median():.0f} range {int(w.min())}-{int(w.max())}")
    print(f"Dm2 inputs per MeTu1: mean {dm2_per.mean():.1f} range {dm2_per.min()}-{dm2_per.max()}")
    print(f"total Dm2 synapses per MeTu1: mean {tot_per.mean():.1f}")


def dm2_consistency():
    if not os.path.exists(CACHE):
        print("\n(no cached Dm2 dotprops -> skipping consistency check)")
        return
    neurons = pickle.load(open(CACHE, "rb"))
    fafb = sorted(l for l in neurons if l.startswith("Dm2.FW."))
    jan = sorted(l for l in neurons if l.startswith("Dm2.OL."))   # OL == mCNS reconstructions
    labels = fafb + jan
    nl = navis.NeuronList([neurons[l] for l in labels])
    nl.set_neuron_attributes(labels, "name")
    nl.set_neuron_attributes(labels, "id")
    m = navis.nblast_align(nl, nl, align_method="rigid", two_way_align=False,
                           n_cores=1, progress=False, align_kwargs={"scale": True})
    m.index = labels; m.columns = labels
    M = m.values
    idx = {l: i for i, l in enumerate(labels)}
    fi = [idx[l] for l in fafb]; ji = [idx[l] for l in jan]

    def block(rows, cols, drop_diag):
        sub = M[np.ix_(rows, cols)].copy()
        if drop_diag:
            np.fill_diagonal(sub, np.nan)
        return np.nanmean(sub)

    print("\n=== Dm2 morphology consistency (nblast_align, scale-on) ===")
    print(f"within FAFB  (n={len(fafb)}): mean {block(fi, fi, True):.3f}")
    print(f"within Janelia (n={len(jan)}): mean {block(ji, ji, True):.3f}")
    print(f"FAFB <-> Janelia (cross):      mean {block(fi, ji, False):.3f}")
    print("\n(cross ~ within => Dm2 morphology is consistent across datasets/pipelines/sex)")


if __name__ == "__main__":
    fafb_dm2_to_metu1()
    dm2_consistency()
