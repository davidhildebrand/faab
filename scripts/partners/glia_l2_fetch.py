"""Stage 2 of the glia shape metrics: exact volumes and surface areas from the L2 cache.

For every comparison cell (figures/glia_metrics.tsv) and partner cell (figures/partner_glia_flags.tsv):
one throttled get_leaves(stop_layer=2) call per root (cached in figures/l2_leaves.pkl, resumable),
then l2cache.get_l2data in batches of 2000 (cached per chunk in figures/l2_chunkdata.pkl):
size_nm3, area_nm2, rep_coord_nm, max_dt_nm, mean_dt_nm. Per cell:
    vol_um3 = sum size, area_um2 = sum area, extents from the size-weighted chunk cloud,
    thick_um = size-weighted mean of mean_dt, cable_um from the skeleton cache.
Evaluation and partner application as in faab_glia_shape.py. Output figures/glia_shape_l2.tsv.
Read-only; one CAVE call per 5 s, so ~35 min for ~380 cells the first time.
"""
import datetime
import os
import pickle
import time

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import caveclient
from faab.clients.cave import DATASTACK, throttle
from faab.glia.shape import (OUT, SCACHE, MIN_CABLE, PRE_CUT, SHAPE, extent_metrics, derived, evaluate,
                             best_cut, logistic2, load_sets, skeleton_metrics)
from faab.config import fig

LEAVES = fig("l2_leaves.pkl")
CHUNKS = fig("l2_chunkdata.pkl")
ATTRS = ["size_nm3", "area_nm2", "rep_coord_nm", "max_dt_nm", "mean_dt_nm"]


def fetch(client, roots):
    leaves = pickle.load(open(LEAVES, "rb")) if os.path.exists(LEAVES) else {}
    chunks = pickle.load(open(CHUNKS, "rb")) if os.path.exists(CHUNKS) else {}
    t = time.time() - 10
    todo = [r for r in roots if r not in leaves]
    print(f"{len(roots)} roots, {len(todo)} need get_leaves", flush=True)
    for k, r in enumerate(todo):
        t = throttle(t)
        try:
            leaves[r] = [int(i) for i in client.chunkedgraph.get_leaves(r, stop_layer=2)]
        except Exception as e:
            print("get_leaves failed", r, str(e)[:80], flush=True)
            leaves[r] = None
        if (k + 1) % 25 == 0 or k + 1 == len(todo):
            pickle.dump(leaves, open(LEAVES, "wb"))
            print(f"  leaves {k + 1}/{len(todo)}", flush=True)
    need = sorted({i for r in roots for i in (leaves.get(r) or []) if i not in chunks})
    print(f"{len(need)} L2 chunks need attributes", flush=True)
    for i in range(0, len(need), 2000):
        batch = need[i:i + 2000]
        t = throttle(t)
        try:
            info = client.l2cache.get_l2data(batch, attributes=ATTRS)
        except Exception as e:
            print("get_l2data failed for batch at", i, str(e)[:80], flush=True)
            info = {}
        for b in batch:
            v = info.get(str(b), info.get(b))
            chunks[b] = v if v else {}
        pickle.dump(chunks, open(CHUNKS, "wb"))
        print(f"  l2data {min(i + 2000, len(need))}/{len(need)}", flush=True)
    return leaves, chunks


def cell_metrics(ids, chunks):
    size = np.array([chunks.get(i, {}).get("size_nm3", np.nan) for i in ids], float)
    area = np.array([chunks.get(i, {}).get("area_nm2", np.nan) for i in ids], float)
    mdt = np.array([chunks.get(i, {}).get("mean_dt_nm", np.nan) for i in ids], float)
    co = [chunks.get(i, {}).get("rep_coord_nm") for i in ids]
    has = np.array([c is not None for c in co])
    out = dict(n_chunks=len(ids), n_missing=int(np.sum(~np.isfinite(size))),
               vol_um3=float(np.nansum(size)) / 1e9, area_um2=float(np.nansum(area)) / 1e6)
    ok = np.isfinite(size) & np.isfinite(mdt)
    out["thick_um"] = float(np.sum(mdt[ok] * size[ok]) / np.sum(size[ok])) / 1000.0 if ok.any() else np.nan
    pts = np.array([c for c in co if c is not None], float) / 1000.0
    w = size[has]
    out.update(extent_metrics(pts, w=np.where(np.isfinite(w), w, 1.0)) if len(pts) else
               dict(L1=np.nan, L2=np.nan, L3=np.nan, rg=np.nan, hull=np.nan))
    return out


def main():
    t0 = time.time()
    gm, pf = load_sets()
    roots = sorted({int(r) for r in gm.root} | {int(r) for r in pf.root_id_curr})
    client = caveclient.CAVEclient(DATASTACK)
    leaves, chunks = fetch(client, roots)
    sk = pickle.load(open(SCACHE, "rb"))

    rows = []
    for _, r in gm.iterrows():
        d = dict(set="comparison", group=r.group, source=r.source, label=r.label, root=str(r.root),
                 n_pre=r.n_pre, n_post=r.n_post, pre_per_100um=r.pre_per_100um, notes="")
        rows.append(d)
    for _, r in pf.iterrows():
        rows.append(dict(set="partner", group="", source="", label=r.flag, root=str(r.root_id_curr), n_pre=r.n_pre,
                         n_post=r.n_post, pre_per_100um=r.pre_per_100um, notes=r.notes))
    for d in rows:
        ids = leaves.get(int(d["root"]))
        if ids:
            d.update(cell_metrics(ids, chunks))
        s = sk.get(int(d["root"]))
        d["cable_um"] = skeleton_metrics(s)["cable_um"] if s is not None else np.nan
    df = derived(pd.DataFrame(rows))
    df["area_per_vol"] = df.area_um2 / df.vol_um3
    df.to_csv(fig("glia_shape_l2.tsv"), sep="\t", index=False)

    comp = df[(df.set == "comparison") & df.vol_um3.notna()]
    judge = comp[comp.cable_um >= MIN_CABLE]
    print(f"\ncomparison cells with L2 data: {len(comp)} of {len(gm)}; with cable >= {MIN_CABLE:.0f} um: {len(judge)} "
          f"({(judge.group == 'glia').sum()} glia, {(judge.group == 'neuron').sum()} neurons)")
    metrics = ["pre_per_100um"] + SHAPE + ["area_per_vol", "thick_um"]
    summ = evaluate(judge, metrics, f"STAGE 2 (exact L2 volumes), cells >= {MIN_CABLE:.0f} um cable")
    shape_only = summ[summ.metric != "presynaptic sites / 100 um cable"]
    best_name = shape_only.sort_values("bal acc", ascending=False).iloc[0].metric
    best_m = [m for m in metrics if (m == best_name) or (__import__("faab_glia_shape").NAMES.get(m) == best_name)][0]

    dens_pred = judge.pre_per_100um.values.astype(float) < PRE_CUT
    err = judge[dens_pred != (judge.group.values == "glia")]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 40)
    cols = ["group", "label", "root", "pre_per_100um", "cable_um", "vol_um3", "L1_um", "vol_per_L1", "fill",
            "cable_per_L1", "area_per_vol", "thick_um"]
    print(f"\ndensity rule (< {PRE_CUT}) errors on the judgeable comparison set: {len(err)}")
    print(err[cols].to_string(index=False))
    b, mu, sd, acc, bal, ok, p = logistic2(judge, "pre_per_100um", best_m)
    print(f"\nlogistic (density + {best_m}): accuracy {acc * 100:.0f}%, balanced {bal * 100:.0f}% (in-sample, n={ok.sum()})")
    rule, _, _ = best_cut(judge[best_m], judge.group == "glia")
    print(f"best shape metric: {best_m}; rule: {rule}")

    part = df[df.set == "partner"]
    show = part[(part.label == "likely glia") | (part.root == "648518347577486275")]
    print("\nflagged partner cells and the reviewed cell 648518347577486275:")
    print(show[["root", "label", "pre_per_100um", "cable_um", "vol_um3", "L1_um", "vol_per_L1", "fill", "cable_per_L1",
                "area_per_vol", "thick_um", "notes"]].sort_values("vol_per_L1").to_string(index=False))
    print(f"\nlive at {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M} UTC; {time.time() - t0:.0f} s\nDONE")


if __name__ == "__main__":
    main()
