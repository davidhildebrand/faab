"""Can glia be told apart from neurons by connectivity / morphology metrics?

Sets: (a) FlyTable aedes_main rows whose class-like columns contain 'glia' (current roots via
supervoxel -> get_roots); (b) the cells your tracking sheet notes call glia or LMC; (c) a
random sample of FlyTable-annotated neurons (non-glia class), same size as (a)+(b) glia.
Per cell (live, one query each direction): presynaptic sites (cell is pre), postsynaptic
sites (cell is post), distinct partners each way; L2 skeleton cable length; nucleus present.
Reports per-cell metrics and, per metric, the best single threshold separating glia from
neurons (accuracy on this set). Read-only, one CAVE call per 5 s.
Outputs figures/glia_metrics.tsv and prints the summary.
"""
from faab.config import REPO
import datetime
import os
import pickle
import random
import re
import time

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import caveclient
import navis
from fafbseg import flywire
from faab.sources.cave import DATASTACK, throttle

HERE = REPO
OUT = os.path.join(HERE, "figures")
SCACHE = os.path.join(OUT, "pr_partner_skels.pkl")
SHEET = os.path.join(HERE, "..", "tmp", "sheet_partners.tsv")
SERVER = "https://flytable.mrc-lmb.cam.ac.uk/"
N_NEURON_SAMPLE = 40
random.seed(7)


def main():
    client = caveclient.CAVEclient(DATASTACK)
    now = datetime.datetime.now(datetime.timezone.utc)
    t = throttle(time.time())

    # ---- FlyTable: glia-labelled cells and a neuron sample ----
    import seaserpent as ss
    meta = ss.Table("aedes_main", server=SERVER).to_frame()
    meta.columns = [str(c) for c in meta.columns]
    cls_cols = [c for c in ("superclass", "class", "subclass", "type", "cell_class", "cell_type") if c in meta.columns]
    def _s(v):
        return "" if (v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v)) else str(v).strip()
    lab = pd.Series([" / ".join(x for x in (_s(row[c]) for c in cls_cols) if x)
                     for _, row in meta[cls_cols].astype(object).iterrows()], index=meta.index)
    is_glia = lab.str.contains("glia", case=False)
    has_label = lab.str.replace("/", "").str.strip() != ""
    glia_ft = meta[is_glia].copy()
    neu_ft = meta[has_label & ~is_glia].copy()
    print(f"FlyTable: {len(glia_ft)} glia-labelled rows, {len(neu_ft)} labelled non-glia rows "
          f"(class columns: {cls_cols})", flush=True)
    neu_ft = neu_ft.sample(n=min(N_NEURON_SAMPLE, len(neu_ft)), random_state=7)

    cells = []   # (label_group, source, supervoxel or None, root or None, flytable label)
    for _, r in glia_ft.iterrows():
        cells.append(("glia", "FlyTable", r.get("supervoxel_id", ""), r.get("root_id", ""), lab[r.name]))
    for _, r in neu_ft.iterrows():
        cells.append(("neuron", "FlyTable", r.get("supervoxel_id", ""), r.get("root_id", ""), lab[r.name]))

    # ---- sheet: glia / LMC by notes ----
    sheet = pd.read_csv(SHEET, sep="\t", dtype=str).fillna("")
    seen = set()
    for _, r in sheet.iterrows():
        txt = (r["notes"] + " " + r["annotation"]).lower()
        rid = r["root_id_curr"].strip()
        if not rid.isdigit() or rid in seen:
            continue
        if "glia" in txt:
            cells.append(("glia", "sheet", "", rid, r["notes"]))
            seen.add(rid)
        elif "lmc" in txt:
            cells.append(("neuron (LMC)", "sheet", "", rid, r["notes"]))
            seen.add(rid)

    # ---- current roots ----
    sv = [int(str(c[2])) for c in cells if str(c[2]).isdigit()]
    sv_root = {}
    if sv:
        t = throttle(t)
        roots = np.asarray(client.chunkedgraph.get_roots(sv))
        sv_root = dict(zip(sv, [int(x) for x in roots]))
    root_of = []
    for c in cells:
        if str(c[2]).isdigit():
            root_of.append(sv_root.get(int(c[2])))
        elif str(c[3]).isdigit():
            root_of.append(int(c[3]))
        else:
            root_of.append(None)
    # sheet roots may be stale: resolve to current (largest piece) once
    stale = [r for c, r in zip(cells, root_of) if c[1] == "sheet" and r]
    if stale:
        t = throttle(t)
        latest = dict(zip(stale, [bool(x) for x in np.asarray(client.chunkedgraph.is_latest_roots(stale))]))
        for i, (c, r) in enumerate(zip(cells, root_of)):
            if c[1] == "sheet" and r and not latest[r]:
                t = throttle(t)
                cand = [int(x) for x in np.atleast_1d(client.chunkedgraph.get_latest_roots(r))]
                root_of[i] = cand[0] if len(cand) == 1 else None
                if len(cand) > 1:
                    print(f"  sheet cell {r} split into {len(cand)} roots; skipped", flush=True)
    df = pd.DataFrame({"group": [c[0] for c in cells], "source": [c[1] for c in cells],
                       "label": [c[4] for c in cells], "root": root_of}).dropna(subset=["root"])
    df["root"] = df.root.astype("int64")
    df = df.drop_duplicates("root").reset_index(drop=True)
    roots = sorted(df.root.tolist())
    print(f"{len(df)} cells with current roots: {df.group.value_counts().to_dict()}", flush=True)

    # ---- live synapse counts, both directions ----
    t = throttle(t)
    pre = client.materialize.live_query("synapses_v2", timestamp=now, filter_in_dict={"pre_pt_root_id": roots})
    t = throttle(t)
    post = client.materialize.live_query("synapses_v2", timestamp=now, filter_in_dict={"post_pt_root_id": roots})
    pre = pre[pre.pre_pt_root_id != pre.post_pt_root_id]
    post = post[post.pre_pt_root_id != post.post_pt_root_id]
    n_pre = pre.groupby("pre_pt_root_id").size()
    n_post = post.groupby("post_pt_root_id").size()
    p_pre = pre.groupby("pre_pt_root_id").post_pt_root_id.nunique()
    p_post = post.groupby("post_pt_root_id").pre_pt_root_id.nunique()
    df["n_pre"] = df.root.map(n_pre).fillna(0).astype(int)
    df["n_post"] = df.root.map(n_post).fillna(0).astype(int)
    df["partners_out"] = df.root.map(p_pre).fillna(0).astype(int)
    df["partners_in"] = df.root.map(p_post).fillna(0).astype(int)
    df["pre_frac"] = df.n_pre / (df.n_pre + df.n_post).clip(lower=1)

    # ---- skeletons: cable length ----
    sk_by = pickle.load(open(SCACHE, "rb")) if os.path.exists(SCACHE) else {}
    need = [r for r in roots if r not in sk_by]
    if need:
        t = throttle(t)
        try:
            for s in flywire.get_l2_skeleton(need, dataset=DATASTACK, max_threads=1,
                                             omit_failures=True, progress=False):
                sk_by[int(s.id)] = s
        except Exception as e:
            print("skeleton fetch failed:", str(e)[:60], flush=True)
        pickle.dump(sk_by, open(SCACHE, "wb"))
    cable, nodes, branches = {}, {}, {}
    for r in roots:
        s = sk_by.get(r)
        if s is not None:
            cable[r] = float(s.cable_length) / 1000.0
            nodes[r] = int(s.n_nodes)
            branches[r] = int(s.n_branches)
    df["cable_um"] = df.root.map(cable)
    df["n_nodes"] = df.root.map(nodes)
    df["n_branches"] = df.root.map(branches)
    df["pre_per_100um"] = (df.n_pre / df.cable_um * 100).round(2)
    df["post_per_100um"] = (df.n_post / df.cable_um * 100).round(2)

    # ---- nuclei ----
    t = throttle(t)
    try:
        nuc = client.materialize.query_table("nuclei_v1_aedes", filter_in_dict={"pt_root_id": roots})
        has = set(int(x) for x in nuc.pt_root_id)
        vol = nuc.groupby("pt_root_id")["volume"].sum().to_dict() if "volume" in nuc.columns else {}
    except Exception as e:
        print("nuclei query failed:", str(e)[:60], flush=True)
        has, vol = set(), {}
    df["nucleus"] = df.root.map(lambda r: r in has)
    if vol:
        df["nucleus_vol"] = df.root.map(vol)

    df.to_csv(os.path.join(OUT, "glia_metrics.tsv"), sep="\t", index=False)

    # ---- report ----
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 40)
    cols = ["group", "source", "root", "n_pre", "n_post", "pre_frac", "partners_out", "partners_in",
            "cable_um", "n_branches", "pre_per_100um", "post_per_100um", "nucleus", "label"]
    print("\nGLIA CELLS")
    print(df[df.group == "glia"][cols].to_string(index=False))
    print("\nLMC CELLS FROM THE SHEET")
    print(df[df.group == "neuron (LMC)"][cols].to_string(index=False))
    print("\nMEDIAN (min-max) BY GROUP")
    g = df.copy()
    g["group2"] = g.group.replace({"neuron (LMC)": "neuron"})
    for m in ["n_pre", "n_post", "pre_frac", "partners_out", "partners_in", "cable_um", "n_branches",
              "pre_per_100um", "post_per_100um"]:
        line = f"  {m:>15}: "
        for grp in ["glia", "neuron"]:
            v = g[g.group2 == grp][m].dropna()
            if len(v):
                line += f"{grp} {v.median():.2f} ({v.min():.2f}-{v.max():.2f}) n={len(v)}   "
        print(line)
    print(f"  nucleus present: glia {g[g.group2 == 'glia'].nucleus.mean() * 100:.0f}%, "
          f"neuron {g[g.group2 == 'neuron'].nucleus.mean() * 100:.0f}%")

    print("\nBEST SINGLE THRESHOLD PER METRIC (accuracy glia vs neuron on this set)")
    for m in ["pre_frac", "pre_per_100um", "n_pre", "post_per_100um", "n_branches", "cable_um"]:
        v = g[[m, "group2"]].dropna()
        if v.empty:
            continue
        best = None
        for thr in np.unique(v[m]):
            for direction in ("<", ">="):
                pred = (v[m] < thr) if direction == "<" else (v[m] >= thr)
                acc = ((pred & (v.group2 == "glia")) | (~pred & (v.group2 == "neuron"))).mean()
                if best is None or acc > best[0]:
                    best = (acc, direction, thr)
        acc, direction, thr = best
        print(f"  glia if {m} {direction} {thr:.3g}: accuracy {acc * 100:.0f}% (n={len(v)})")
    print(f"\nlive at {now:%Y-%m-%d %H:%M} UTC\nDONE", flush=True)


if __name__ == "__main__":
    main()
