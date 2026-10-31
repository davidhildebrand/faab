"""Three-template cross-species Dm2 calibration.

Templates (one medoid Dm2 each): FlyWire FAFB, neuPrint optic-lobe, neuPrint male-CNS.
Targets: positive controls (other Dm2, every source), negatives (fly Mi1/Tm1, aedes
TuTuB), and the unknown strongest unannotated MeTu inputs (aedes).

Score = nblast_align (rigid), scale-ON (primary, cross-species size) and scale-OFF.
Read-only. fafbseg L2 fetches throttled; neuPrint is query-only. Neurons cached.

    .venv/bin/python faab/dm2_calibration.py
"""
from faab.config import REPO, fig
import os
import time
import pickle
import warnings

import numpy as np
import pandas as pd
import navis
from fafbseg import flywire
from neuprint import NeuronCriteria as NC, fetch_neurons
from navis.interfaces import neuprint as nvneu

from faab.clients.cave import throttle
from faab.clients.neuprint import client, OPTIC_LOBE, MALE_CNS

warnings.filterwarnings("ignore")
OUT = os.path.join(REPO, "figures")
CACHE = fig("dm2_calib_neurons.pkl")
RESAMPLE = 1.0   # micron spacing to standardize density across pipelines
K = 5


import re


def _um_per_unit(n):
    """Microns per coordinate unit, parsed from the neuron's .units (robust to
    navis convert_units not being in-place). fafbseg='1 nanometer', neuprint='8 nanometer'."""
    u = str(getattr(n, "units", "") or "").lower()
    m = re.match(r"([\d.]+)\s*nanomet", u)
    if m:
        return float(m.group(1)) / 1000.0
    m = re.match(r"([\d.]+)\s*(micromet|um)", u)
    if m:
        return float(m.group(1))
    return None


def to_microns(n):
    n = n.copy()
    f = _um_per_unit(n)
    co = n.nodes[["x", "y", "z"]].values.astype(float)
    if f is None:                              # unknown units -> infer from span
        span = (co.max(0) - co.min(0)).max()
        f = 1.0 if span < 500 else (8 / 1000.0 if span < 9000 else 1 / 1000.0)
    n.nodes[["x", "y", "z"]] = co * f
    n.units = "um"
    return n


def dp(n):
    m = to_microns(n)
    co = m.nodes[["x", "y", "z"]].values
    span = float((co.max(0) - co.min(0)).max())
    rs = max(0.5, span / 200.0)                # size-adaptive: ~200 points regardless of size
    d = navis.make_dotprops(m, k=K, resample=rs)
    d.units = "um"
    return d


def fetch_all():
    if os.path.exists(CACHE):
        print("loaded cached neurons")
        return pickle.load(open(CACHE, "rb"))

    # ---- Aedes ids from the saved MeTu partner table ----
    p = pd.read_csv(os.path.join(REPO, "data", "metu_upstream_partners.csv"))
    p["annotated"] = p[["superclass", "class", "subclass", "type"]].notna().any(axis=1)
    unknown = p[~p["annotated"]].sort_values("syn", ascending=False).head(8)
    unknown_ids = unknown["pre_pt_root_id"].astype("int64").tolist()
    unknown_syn = dict(zip(unknown["pre_pt_root_id"].astype("int64"), unknown["syn"]))
    tutub_ids = (p[p["subclass"] == "TuTuB"].sort_values("syn", ascending=False)
                 .head(2)["pre_pt_root_id"].astype("int64").tolist())

    # ---- FlyWire Dm2 ids ----
    fw = pd.read_csv(os.path.join(REPO, "data", "flywire_dm2.csv"))
    fw_ids = fw[fw["side"] == "right"]["root_id"].astype("int64").head(5).tolist()

    neurons = {}   # label -> dotprops ; meta dict alongside

    # ---- neuPrint: Dm2 (5 each) + negatives (Mi1, Tm1 from OL) ----
    def np_dm2(tag, src):
        c = client(tag)
        ndf, _ = fetch_neurons(NC(type="Dm2"), client=c)
        sort = "size" if "size" in ndf.columns else "bodyId"
        ids = ndf.sort_values(sort, ascending=False)["bodyId"].astype("int64").head(5).tolist()
        sk = nvneu.fetch_skeletons(ids, client=c)
        for n in sk:
            neurons[f"Dm2.{src}.{int(n.id)}"] = dp(n)
        return ids

    np_dm2(OPTIC_LOBE, "OL")
    np_dm2(MALE_CNS, "mCNS")
    c_ol = client(OPTIC_LOBE)
    for t in ["Mi1", "Tm1"]:
        ndf, _ = fetch_neurons(NC(type=t), client=c_ol)
        if len(ndf):
            bid = int(ndf.sort_values("size" if "size" in ndf.columns else "bodyId",
                                      ascending=False)["bodyId"].iloc[0])
            sk = nvneu.fetch_skeletons([bid], client=c_ol)
            neurons[f"NEG.{t}.{bid}"] = dp(sk[0])

    # ---- skeletons: FlyWire Dm2 (precomputed, release 783) + Aedes L2 (TuTuB, unknowns) ----
    t = time.time()
    fw_sk = flywire.get_skeletons(fw_ids, dataset=783, max_threads=1, progress=False)
    for n in fw_sk:
        neurons[f"Dm2.FW.{int(n.id)}"] = dp(n)

    t = throttle(t)
    tutub_set = set(tutub_ids)
    ae_sk = flywire.get_l2_skeleton(tutub_ids + unknown_ids, dataset="wclee_aedes_brain",
                                    max_threads=1, progress=False)
    for n in ae_sk:
        rid = int(n.id)
        if rid in tutub_set:
            neurons[f"NEG.TuTuB.{rid}"] = dp(n)
        else:
            neurons[f"UNK.{rid}.s{unknown_syn.get(rid, '?')}"] = dp(n)

    pickle.dump(neurons, open(CACHE, "wb"))
    print(f"fetched + cached {len(neurons)} neurons")
    return neurons


def category(label):
    if label.startswith("Dm2."):
        return "Dm2(fly)"
    if label.startswith("NEG.TuTuB"):
        return "TuTuB(aedes)"
    if label.startswith("NEG."):
        return "non-Dm2(fly)"
    return "UNKNOWN(aedes)"


def medoid(labels, neurons):
    nl = navis.NeuronList([neurons[l] for l in labels])
    m = navis.nblast_allbyall(nl, n_cores=1, progress=False).values
    return labels[int(np.nanmean(np.where(np.eye(len(m), dtype=bool), np.nan, m), axis=1).argmax())]


def main():
    neurons = fetch_all()
    for l in sorted(neurons):
        sp = (neurons[l].points.max(0) - neurons[l].points.min(0)).round(1)
        print(f"  {l:28s} pts={len(neurons[l].points):4d} span_um={sp}")

    # templates = medoid Dm2 per source
    templates = {}
    for src in ["FW", "OL", "mCNS"]:
        labs = [l for l in neurons if l.startswith(f"Dm2.{src}.")]
        templates[src] = medoid(labs, neurons)
    print("\ntemplates (medoids):", templates)

    tmpl_labels = list(templates.values())
    target_labels = [l for l in neurons if l not in tmpl_labels]
    tmpl_nl = navis.NeuronList([neurons[l] for l in tmpl_labels])
    tmpl_nl.set_neuron_attributes(tmpl_labels, "name")
    tmpl_nl.set_neuron_attributes(tmpl_labels, "id")
    tgt_nl = navis.NeuronList([neurons[l] for l in target_labels])
    tgt_nl.set_neuron_attributes(target_labels, "name")
    tgt_nl.set_neuron_attributes(target_labels, "id")

    for scale in (True, False):
        m = navis.nblast_align(tmpl_nl, tgt_nl, align_method="rigid",
                               two_way_align=False, n_cores=1, progress=False,
                               align_kwargs={"scale": scale})
        m.index = [f"T:{src}" for src in templates]
        m.columns = target_labels
        tag = "SCALE-ON" if scale else "SCALE-OFF"
        res = m.T.copy()
        res["cat"] = [category(l) for l in res.index]
        res["mean_tmpl"] = m.mean(axis=0).values
        res = res.sort_values(["cat", "mean_tmpl"], ascending=[True, False])
        print(f"\n================ nblast_align {tag} ================")
        print(res.round(3).to_string())
        print(f"\n--- {tag}: mean score by target category ---")
        print(res.groupby("cat")["mean_tmpl"].agg(["mean", "min", "max", "count"]).round(3).to_string())
        res.to_csv(fig(f"dm2_calib_{'on' if scale else 'off'}.csv"))


if __name__ == "__main__":
    main()
