"""Cutouts to find R7 synapses the automatic prediction MISSES (David 2026-11-02). The earlier R7 boxes
(propose_targets.py) sit where the most R7 synapses are predicted, so they test whether predictions are right, not what
is missed. These boxes are placed by where the R7 terminal IS (its volume), independent of the predictions:
  - 'cold' boxes: much R7 terminal volume but few predicted R7 output synapses (where misses are most likely);
  - 'typical' boxes: drawn by R7 terminal volume alone (a fair sample for estimating how much is missed overall).
4 + 4, two of each per side. Annotators mark EVERY R7 presynaptic site in the box.

Steps (CAVE read-only, >= 2 s between calls):
  map BOOK_PRS.xlsx      R7 rows of the PRs tab (class R7; status proofread / checked / adequate first, on a side with
                         fewer than MIN_CELLS of those also the side's other R7 rows), their roots at the row's xyz now,
                         their L2 chunks (volume + position, cached in figures/l2_leaves.pkl / l2_chunkdata.pkl) and
                         predicted output synapses (one live query); medulla only (mesh + 5 um margin); depth along
                         each R7 from its most distal medulla chunk. -> figures/cutouts/r7_missed/{cells,chunks,syn}.tsv
  propose [--avoid=REG.tsv,...]   candidate boxes centered on R7 chunks in the terminal zone, R7 volume and predicted R7
                         outputs inside each; picks, live all-synapse counts, links, registry with a purpose column.
"""
import datetime
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from faab import nglink, ol_meshes
from faab.clients.cave import client as cave_client, roots_at_xyz, throttle
from faab.config import fig
from faab.partners.top10_xyz import VOX
from faab.pr import prs_tab as T

OUT = fig("cutouts/r7_missed")
GOOD = ("proofread", "checked", "adequate")
MIN_CELLS = 6
MARGIN_UM = 5.0
LEAVES, CHUNKS = fig("l2_leaves.pkl"), fig("l2_chunkdata.pkl")


def r7_rows(book):
    wb, ws = T.load(book)
    left = next((r for r in range(1, ws.max_row + 1) if T.text(ws, r, 1).startswith(">>> additional L side")),
                ws.max_row + 1)
    rows = []
    for b in T.blocks(ws):
        for r in b.rows:
            if T.get(ws, b, r, "class") == "R7":
                rows.append(dict(side="L" if b.title_row > left else "R", group=b.group, row=r,
                                 status=T.get(ws, b, r, "status").lower(), xyz=T.get(ws, b, r, "xyz"),
                                 root_id_curr=T.get(ws, b, r, "root_id_curr")))
    D = pd.DataFrame(rows)
    D["vetted"] = D.status.isin(GOOD)
    keep = []
    for side, S in D.groupby("side"):
        keep.append(S if S.vetted.sum() < MIN_CELLS else S[S.vetted])
    return pd.concat(keep, ignore_index=True)


def l2_fetch(client, roots, t):
    leaves = pickle.load(open(LEAVES, "rb")) if os.path.exists(LEAVES) else {}
    chunks = pickle.load(open(CHUNKS, "rb")) if os.path.exists(CHUNKS) else {}
    for r in [r for r in roots if r not in leaves]:
        t = throttle(t)
        leaves[r] = [int(i) for i in client.chunkedgraph.get_leaves(r, stop_layer=2)]
        t = time.time()
    pickle.dump(leaves, open(LEAVES, "wb"))
    need = sorted({i for r in roots for i in leaves[r] if i not in chunks})
    for i in range(0, len(need), 2000):
        batch = need[i:i + 2000]
        t = throttle(t)
        info = client.l2cache.get_l2data(batch, attributes=["size_nm3", "rep_coord_nm"])
        t = time.time()
        for b in batch:
            chunks[b] = info.get(str(b), info.get(b)) or {}
    pickle.dump(chunks, open(CHUNKS, "wb"))
    print(f"L2: {sum(len(leaves[r]) for r in roots)} chunks of {len(roots)} roots ({len(need)} fetched)", flush=True)
    return leaves, chunks, t


def depth_axis(P, w, la_center):
    """unit axis of a columnar cloud (weighted PCA), pointing away from the lamina; depth 0 at the most distal point"""
    c = np.average(P, axis=0, weights=w)
    u = np.linalg.svd((P - c) * np.sqrt(w)[:, None], full_matrices=False)[2][0]
    if np.dot(la_center - c, u) > 0:
        u = -u
    d = (P - c) @ u
    return c, u, d.min()


def step_map(book):
    os.makedirs(OUT, exist_ok=True)
    D = r7_rows(book)
    client, t = cave_client(), throttle(0.0)
    D["root"] = [str(r) if r else "" for r in roots_at_xyz(D.xyz.tolist(), client_=client)]
    t = time.time()
    D = D[D.root != ""].drop_duplicates("root")
    print(D.groupby(["side", "vetted"]).size().to_string(), flush=True)
    try:
        g = client.chunkedgraph.segmentation_info.get("graph", {})
        print(f"graph chunk size {g.get('chunk_size')} (voxels at the base resolution)", flush=True)
    except Exception as e:
        print(f"segmentation info: {str(e)[:80]}", flush=True)
    roots = [int(r) for r in D.root]
    leaves, chunks, t = l2_fetch(client, roots, t)
    t = throttle(t)
    now = datetime.datetime.now(datetime.timezone.utc)
    s = client.materialize.live_query("synapses_v2", timestamp=now, filter_in_dict={"pre_pt_root_id": roots})
    s = s[s.pre_pt_root_id != s.post_pt_root_id]
    print(f"predicted output synapses of {len(roots)} R7s: {len(s)}", flush=True)

    meshes = {sd: ol_meshes.mesh(f"ME_{sd}")[0] / 1000.0 for sd in ("R", "L")}
    trees = {sd: cKDTree(v) for sd, v in meshes.items()}
    la = {sd: ol_meshes.mesh(f"LA_{sd}")[0].mean(0) / 1000.0 for sd in ("R", "L")}

    def in_me(P, sd):
        inside = ol_meshes.assign(P, [f"ME_{sd}"]) == f"ME_{sd}"
        return inside | (trees[sd].query(P)[0] <= MARGIN_UM)

    crow, srow, cells = [], [], []
    for x in D.itertuples(index=False):
        ids = [i for i in leaves[int(x.root)] if chunks.get(i, {}).get("rep_coord_nm") is not None]
        P = np.array([chunks[i]["rep_coord_nm"] for i in ids], float) / 1000.0
        w = np.array([chunks[i].get("size_nm3", 0.0) for i in ids], float) / 1e9
        m = in_me(P, x.side) if len(P) else np.zeros(0, bool)
        S = s[s.pre_pt_root_id == int(x.root)]
        Q = np.stack(S.pre_pt_position.to_numpy()).astype(float) / 1000.0 if len(S) else np.zeros((0, 3))
        qm = in_me(Q, x.side) if len(Q) else np.zeros(0, bool)
        if m.sum() < 5:
            print(f"  {x.root}: only {m.sum()} medulla chunks, skipped", flush=True)
            continue
        c, u, d0 = depth_axis(P[m], np.maximum(w[m], 1e-6), la[x.side])
        dc, ds = (P[m] - c) @ u - d0, (Q[qm] - c) @ u - d0
        lab = f"R7 of PR group {x.group}" + ("" if x.vetted else " (not yet proofread)")
        for i, p, v, d in zip(np.array(ids)[m], P[m], w[m], dc):
            crow.append(dict(root=x.root, side=x.side, l2=i, x_um=p[0], y_um=p[1], z_um=p[2], vol_um3=v, depth_um=d))
        for p, d, post in zip(Q[qm], ds, S[qm].post_pt_root_id):
            srow.append(dict(root=x.root, side=x.side, post=post, x_um=p[0], y_um=p[1], z_um=p[2], depth_um=d))
        cells.append(dict(root=x.root, side=x.side, group=x.group, label=lab, vetted=x.vetted, n_chunks_me=int(m.sum()),
                          vol_me_um3=float(w[m].sum()), extent_um=float(dc.max()), n_out_me=int(qm.sum()),
                          n_out_all=len(S), out_per_um3=qm.sum() / max(w[m].sum(), 1e-9)))
    C, Ch, Sy = pd.DataFrame(cells), pd.DataFrame(crow), pd.DataFrame(srow)
    C.to_csv(os.path.join(OUT, "cells.tsv"), sep="\t", index=False)
    Ch.to_csv(os.path.join(OUT, "chunks.tsv"), sep="\t", index=False)
    Sy.to_csv(os.path.join(OUT, "syn.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 200)
    print(C.drop(columns=["label"]).round(2).to_string(index=False))
    print("L2 chunk volume um3 quantiles:", Ch.vol_um3.quantile([.1, .5, .9]).round(3).to_dict())
    for nm, X in [("chunk volume", Ch), ("predicted outputs", Sy)]:
        h = np.histogram(X.depth_um, bins=np.arange(0, 64, 4), weights=X.vol_um3 if "vol_um3" in X else None)[0]
        print(f"{nm} by depth from the distal end (4 um bins): {np.round(h / h.sum() * 100).astype(int).tolist()}")


def step_meshes(limit=None):
    """R7 meshes (the L2 chunks are 512 x 512 x 128 voxels = 8.2 x 8.2 x 5.8 um, far coarser than a 2 um box), cached as
    figures/cutouts/r7_missed/meshes/<root>.npz (vertices in um, faces); one neuron at a time, >= 2 s apart."""
    from fafbseg import flywire
    from faab.config import DATASTACK
    C = pd.read_csv(os.path.join(OUT, "cells.tsv"), sep="\t", dtype={"root": str})
    md = os.path.join(OUT, "meshes")
    os.makedirs(md, exist_ok=True)
    todo = [r for r in C.root if not os.path.exists(os.path.join(md, f"{r}.npz"))][:limit]
    t = throttle(0.0)
    for r in todo:
        t = throttle(t)
        t0 = time.time()
        m = flywire.get_mesh_neuron(int(r), dataset=DATASTACK, threads=1, progress=False)
        t = time.time()
        v, f = np.asarray(m.vertices, float) / 1000.0, np.asarray(m.faces, np.int64)
        np.savez_compressed(os.path.join(md, f"{r}.npz"), v=v, f=f)
        print(f"  {r}: {len(v)} vertices, {len(f)} faces, {t - t0:.1f} s", flush=True)


SIZE = np.array([125, 125, 50])  # voxels, as all cutouts
VOXU = np.asarray(VOX, float) / 1000.0  # um per voxel
N_PER, MIN_SEP_UM, DEPTH_BIN, SEED, N_CAND = 2, 10.0, 4.0, 20261102, 400
COLOR = {"cold": "#D55E00", "typical": "#FFFFFF"}  # vermillion / white; prediction checks magenta, existing light blue
R7_COLOR = "#0072B2"
PURPOSE = {
    "cold": "R7 missed synapses (cold): much R7 terminal membrane but few predicted R7 synapses; annotate EVERY R7 "
            "presynaptic site to find synapses the prediction missed",
    "typical": "R7 missed synapses (typical): centered on a random point of the R7 terminal membrane, ignoring the "
               "predictions; annotate EVERY R7 presynaptic site to estimate how many R7 synapses are missed overall",
}


def faces_in_me(root, side, axis_cache):
    """R7 mesh faces inside the medulla mesh (no margin: the margin would add axon in the chiasm): centroids (um), areas
    (um2), depth (um) along the R7 from its distal end in the medulla"""
    z = np.load(os.path.join(OUT, "meshes", f"{root}.npz"))
    v, f = z["v"], z["f"]
    tri = v[f]
    cen = tri.mean(1)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    me = ol_meshes.mesh(f"ME_{side}")[0] / 1000.0
    pre = np.all((cen >= me.min(0)) & (cen <= me.max(0)), axis=1)
    idx = np.flatnonzero(pre)
    inside = ol_meshes.assign(cen[idx], [f"ME_{side}"]) == f"ME_{side}"
    idx = idx[inside]
    cen, area = cen[idx], area[idx]
    la = ol_meshes.mesh(f"LA_{side}")[0].mean(0) / 1000.0
    c, u, _ = depth_axis(cen, np.maximum(area, 1e-9), la)
    d = (cen - c) @ u
    d0 = np.percentile(d, 1)
    axis_cache[root] = (c, u, d0)
    return cen, area, d - d0


def boxes_um(c):
    lo = np.round(c / VOXU) - np.floor(SIZE / 2.0)
    return lo, lo + SIZE, lo * VOXU, (lo + SIZE) * VOXU


def step_propose(avoid):
    C = pd.read_csv(os.path.join(OUT, "cells.tsv"), sep="\t", dtype={"root": str})
    S = pd.read_csv(os.path.join(OUT, "syn.tsv"), sep="\t", dtype={"root": str})
    inv = pd.read_csv(fig("cutouts/cutout_inventory.tsv") if os.path.exists(fig("cutouts/cutout_inventory.tsv"))
                      else fig("cutout_inventory.tsv"), sep="\t")
    taken = [(np.array(json.loads(a)), np.array(json.loads(b))) for a, b in zip(inv.lo, inv.hi)]
    for reg in avoid:
        P = pd.read_csv(reg, sep="\t")
        taken += [(np.array([float(v) for v in a.split(",")]), np.array([float(v) for v in b.split(",")]))
                  for a, b in zip(P.lo_xyz, P.hi_xyz)]
    overlap = lambda lo, hi: any(np.all(lo < b) and np.all(a < hi) for a, b in taken)
    rng = np.random.default_rng(SEED)
    half_um = SIZE / 2.0 * VOXU
    rows, used_cells, axes = [], set(), {}
    for side in ("R", "L"):
        cs = C[C.side == side]
        F = {r: faces_in_me(r, side, axes) for r in cs.root}
        cen = np.concatenate([F[r][0] for r in cs.root])
        area = np.concatenate([F[r][1] for r in cs.root])
        dep = np.concatenate([F[r][2] for r in cs.root])
        owner = np.concatenate([[r] * len(F[r][0]) for r in cs.root])
        Ss = S[S.side == side]
        Q = Ss[["x_um", "y_um", "z_um"]].to_numpy(float)
        qd = np.array([(q - axes[r][0]) @ axes[r][1] - axes[r][2] for q, r in zip(Q, Ss.root)])
        bins = np.arange(0, max(dep.max(), qd.max() if len(qd) else 0) + DEPTH_BIN, DEPTH_BIN)
        a_bin = np.histogram(dep, bins, weights=area)[0]
        s_bin = np.histogram(qd, bins)[0]
        dens = np.where(a_bin > 0, s_bin / np.maximum(a_bin, 1e-9), 0.0)  # predicted outputs per um2 of R7 membrane
        print(f"{side}: {len(cs)} R7s, membrane in the medulla {area.sum():.0f} um2, predicted outputs {len(Q)}; "
              f"outputs per 100 um2 by depth (4 um bins): {np.round(dens * 100, 1).tolist()}", flush=True)
        exp_face = area * dens[np.clip(np.digitize(dep, bins) - 1, 0, len(dens) - 1)]
        ftree, stree = cKDTree(cen / half_um), cKDTree(Q / half_um) if len(Q) else None

        def score(c):
            lo, hi, lo_um, hi_um = boxes_um(c)
            fi = [i for i in ftree.query_ball_point(c / half_um, 1.6, p=np.inf)
                  if np.all((cen[i] >= lo_um) & (cen[i] < hi_um))]
            si = [i for i in (stree.query_ball_point(c / half_um, 1.6, p=np.inf) if stree else [])
                  if np.all((Q[i] >= lo_um) & (Q[i] < hi_um))]
            by = pd.Series(area[fi], index=owner[fi]).groupby(level=0).sum().sort_values(ascending=False)
            return dict(center=np.round(c / VOXU), lo=lo, hi=hi, area=float(area[fi].sum()), exp=float(exp_face[fi].sum()),
                        obs=len(si), main=by.index[0] if len(by) else "", cells=by.round(1).to_dict(),
                        points=(Q[si] / VOXU).tolist(), depth=float(np.median(dep[fi])) if fi else np.nan)

        def ok(b, chosen):
            return (b["main"] and b["main"] not in used_cells and not overlap(b["lo"], b["hi"])
                    and all(np.linalg.norm((b["center"] - x["center"]) * VOXU) >= MIN_SEP_UM for x in chosen))

        p = area / area.sum()
        chosen = []
        # cold: candidates at area-weighted random points; keep the top quarter by expected outputs (= much R7
        # membrane at depths where R7s usually make synapses), lowest predicted / expected first
        cand = [score(cen[i]) for i in rng.choice(len(cen), N_CAND, replace=False, p=p)]
        e75 = np.percentile([b["exp"] for b in cand], 75)
        cold = sorted([b for b in cand if b["exp"] >= e75], key=lambda b: (b["obs"] / max(b["exp"], 1e-9), -b["exp"]))
        for b in cold:
            if sum(x["kind"] == "cold" for x in chosen) >= N_PER:
                break
            if ok(b, chosen):
                b["kind"] = "cold"
                chosen.append(b)
                used_cells.add(b["main"])
                taken.append((b["lo"], b["hi"]))
        # typical: successive area-weighted random points on the R7 membrane, the first that fit
        tries = 0
        while sum(x["kind"] == "typical" for x in chosen) < N_PER and tries < 2000:
            tries += 1
            b = score(cen[rng.choice(len(cen), p=p)])
            if ok(b, chosen):
                b["kind"] = "typical"
                chosen.append(b)
                used_cells.add(b["main"])
                taken.append((b["lo"], b["hi"]))
        for k, b in enumerate(chosen):
            b.update(side=side, label=C.set_index("root").label.get(b["main"], "R7"))
        rows += chosen
        print(f"{side}: cold {[(round(b['area']), b['obs'], round(b['exp'], 1)) for b in chosen if b['kind'] == 'cold']}; "
              f"typical {[(round(b['area']), b['obs'], round(b['exp'], 1)) for b in chosen if b['kind'] == 'typical']} "
              f"(R7 membrane um2, predicted R7 outputs, expected)", flush=True)

    # live: every synapse in each box, and the R7 outputs among them (read-only)
    client, t = cave_client(), throttle(0.0)
    now = datetime.datetime.now(datetime.timezone.utc)
    pool = {int(r) for r in C.root}
    for b in rows:
        t = throttle(t)
        a = client.materialize.live_query("synapses_v2", timestamp=now, filter_spatial_dict={
            "ctr_pt_position": [(b["lo"] * VOX).tolist(), (b["hi"] * VOX).tolist()]})
        t = time.time()
        a = a[a.pre_pt_root_id != a.post_pt_root_id]
        b.update(n_all=len(a), n_r7_live=int(a.pre_pt_root_id.isin(pool).sum()),
                 n_pre_cells=a.pre_pt_root_id.nunique(), n_post_cells=a.post_pt_root_id.nunique())
    ids = {}
    for b in sorted(rows, key=lambda b: (b["kind"], b["side"] != "R")):
        key = f"R7-{b['kind']}-{b['side']}"
        ids[key] = ids.get(key, 0) + 1
        b["box_id"] = f"{key}-{ids[key]}"
        name = f"cutout {b['box_id']} ({'few predicted R7 synapses' if b['kind'] == 'cold' else 'random R7 point'})"
        spec = {"neurons": [{"name": f"R7s in the box ({len(b['cells'])})", "color": R7_COLOR, "ids": list(b["cells"])}],
                "annotations": [{"name": name, "color": COLOR[b["kind"]], "width": 4, "boxes": [[b["lo"].tolist(), b["hi"].tolist()]]},
                                {"name": "predicted R7 output synapses (hidden; compare after annotating)", "color": "#56B4E9",
                                 "points": b["points"], "visible": False}],
                "layout": "xy-3d", "position": b["center"].tolist(), "scale": 2500, "cross_scale": 0.5, "image": True,
                "select": name}
        t = throttle(t)
        b["link"] = client.state.build_neuroglancer_url(client.state.upload_state_json(nglink.clean_state(nglink.build(spec))),
                                                        ngl_url="https://spelunker.cave-explorer.org")
        t = time.time()
    fmt = lambda v: ", ".join(str(int(x)) for x in v)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dd%H%M%StUTC")
    R = pd.DataFrame([dict(box_id=b["box_id"], purpose=PURPOSE[b["kind"]],
                           category=f"R7 missed synapses, {b['kind']}", side=b["side"], neuropil="ME",
                           center_xyz=fmt(b["center"]), lo_xyz=fmt(b["lo"]), hi_xyz=fmt(b["hi"]), size_voxels="125 x 125 x 50",
                           r7_membrane_um2=round(b["area"], 1), r7_depth_um=round(b["depth"], 1),
                           predicted_r7_outputs=b["n_r7_live"], expected_r7_outputs=round(b["exp"], 1),
                           n_all_synapses=b["n_all"], n_presynaptic_cells=b["n_pre_cells"],
                           n_postsynaptic_cells=b["n_post_cells"],
                           target_cells="; ".join(f"{c} ({a} um2 of membrane, {b['label'] if c == b['main'] else C.set_index('root').label.get(c, 'R7')})"
                                                  for c, a in b["cells"].items()),
                           link=b["link"]) for b in sorted(rows, key=lambda b: b["box_id"])])
    out = fig(f"cutouts/{stamp}_cutouts_r7_missed.tsv")
    R.to_csv(out, sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(R.drop(columns=["purpose", "target_cells", "link", "lo_xyz", "hi_xyz", "size_voxels"]).to_string(index=False))
    print(f"written {out}")


if __name__ == "__main__":
    if sys.argv[1] == "map":
        step_map(sys.argv[2])
    elif sys.argv[1] == "meshes":
        step_meshes(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif sys.argv[1] == "propose":
        step_propose([p for a in sys.argv[2:] if a.startswith("--avoid=") for p in a[8:].split(",")])
