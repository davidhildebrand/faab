"""Propose synapse-annotation cutouts on target cell types (David, 2026-11-01): R7 terminals (6), LMC terminals (4),
T4/T5 inputs (4), right and left where annotated cells exist. Box format = the existing 46 cutouts: 125 x 125 x 50
voxels (2.0 x 2.0 x 2.25 um), axis-aligned bounding boxes over the EM image. No chunk grid: the existing boxes are
hand-placed (corners at round numbers with click jitter, sizes 119-128 x 124-134 x 40-60 voxels; scratch/
cutouts_alignment.py), so corners here are whole voxels.

Box requests (REQUESTS): the target cells (current roots) and which of their synapses count - R7 and LMC: output
synapses in the medulla of their side; T4: input synapses in the medulla; T5: input synapses in the lobula (5 um mesh
margin). Target cells: R7 = class R7 rows of the PRs tab with status proofread / checked / adequate (right: groups
1-8; left: LRS's groups); LMC = the annotated LMCs of the LMCs tab with status adequate / proofread / checked; T4/T5 =
SeaTable rows typed T4 / T5 (all 'adequate'), plus on the right, where SeaTable has no T5, the T4/T5-like group-1
targets of figures/t4t5/*_pr_g1_to_t4t5.tsv (unproofread; flagged). Per request, greedily: the box (centered on a
target synapse) holding the most target synapses, then the next best that overlaps no existing cutout and no chosen
box and lies >= MIN_SEP_UM from the request's other boxes (R7: each box on a different R7 cell, i.e. columns).
Per box: every predicted synapse inside (live spatial query on synapses_v2, >= 2 s apart, read-only), and a Spelunker
link (README scene: EM, target cells, the box in yellow, the target synapses as points); one overview link with the
new boxes (yellow) and the existing cutouts (light blue).
Writes figures/cutouts/<stamp>_cutouts_targets.tsv: the registry, one row per box (id, category, side, target cells
by root_id with their synapses in the box, counts, corners, link).
Usage: propose_targets.py SEATABLE.tsv PRS_TAB.xlsx LMC_TAB.tsv"""
import datetime
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import caveclient
from scipy.spatial import cKDTree

from faab import nglink, ol_meshes, synsnap
from faab.clients.cave import DATASTACK, throttle
from faab.config import fig
from faab.partners.top10_xyz import VOX
from faab.pr import prs_tab as T

SIZE = np.array([125, 125, 50])  # voxels, as the existing cutouts
MIN_SEP_UM, MARGIN_UM = 10.0, 5.0
GOOD = ("proofread", "checked", "adequate")
REQUESTS = [  # (category, side, target set, synapse role, neuropil, boxes, one box per cell)
    ("R7 terminals", "R", "R7", "out", "ME", 4, True),
    ("R7 terminals", "L", "R7", "out", "ME", 2, True),
    ("L1 terminals", "R", "L1", "out", "ME", 1, False),
    ("L2 terminals", "R", "L2", "out", "ME", 1, False),
    ("L3 terminals", "R", "L3", "out", "ME", 1, False),
    ("LMC terminals", "L", "LMC", "out", "ME", 1, False),
    ("T4 inputs", "R", "T4T5", "in", "ME", 1, False),
    ("T5 inputs", "R", "T4T5", "in", "LO", 1, False),
    ("T4 inputs", "L", "T4T5", "in", "ME", 1, False),
    ("T5 inputs", "L", "T4T5", "in", "LO", 1, False),
]


def targets(seatable, prs_book, lmc_tsv):
    """{(set, side): {root: label}}"""
    out = {}
    wb, ws = T.load(prs_book)
    left = next((r for r in range(1, ws.max_row + 1) if T.text(ws, r, 1).startswith(">>> additional L side")), ws.max_row + 1)
    tab = pd.read_csv(os.path.join(fig("ommatidia/20261101"), "tab_roots.tsv"), sep="\t", dtype=str).fillna("")
    now = {(g, o): r for g, o, r in zip(tab.group.astype(int), tab.root_id_orig, tab.root_now)}
    for b in T.blocks(ws):
        side = "L" if b.title_row > left else "R"
        for r in b.rows:
            if T.get(ws, b, r, "class") == "R7" and T.get(ws, b, r, "status").lower() in GOOD:
                root = now.get((b.group, T.get(ws, b, r, "root_id_orig"))) or T.get(ws, b, r, "root_id_curr")
                out.setdefault(("R7", side), {})[root] = f"R7 of PR group {b.group}"
    lines = [l.split("\t") for l in open(lmc_tsv).read().split("\n")]
    hdr = next(l for l in lines if l and l[0] == "root_id_orig")
    L = pd.DataFrame([l for l in lines if l and l[0].isdigit()], columns=hdr)
    L = L[(L.confidence == "annotated") & L.status.str.lower().isin(GOOD)]
    for x in L.itertuples(index=False):
        out.setdefault((x.putative_type, x.side), {})[x.root_id_curr] = x.putative_type
        out.setdefault(("LMC", x.side), {})[x.root_id_curr] = x.putative_type
    S = pd.read_csv(seatable, sep="\t", dtype=str).fillna("")
    lab = S[["class", "subclass", "subsubclass", "type"]].agg(" ".join, axis=1).str.split().str.join(" ")
    t45 = S[lab.str.contains(r"\bT[45]", regex=True)]
    for r, sd, l in zip(t45.root_now, t45.side, lab[t45.index]):
        if sd in ("R", "L"):
            out.setdefault(("T4T5", sd), {})[r] = f"SeaTable {l}"
    g1 = sorted(glob.glob(fig("*_pr_g1_to_t4t5.tsv")))
    if g1:
        C = pd.read_csv(g1[-1], sep="\t", dtype={"cell": str})
        for r in C[C.morph_call == "T4/T5-like"].cell:
            out.setdefault(("T4T5", "R"), {}).setdefault(r, "T4/T5-like group-1 target (unproofread)")
    return out


def main():
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dd%H%M%StUTC")
    tg = targets(*sys.argv[1:4])
    for k, v in sorted(tg.items()):
        print(f"targets {k}: {len(v)}", flush=True)
    snap, client = synsnap.Snapshot(log=lambda *_: None), caveclient.CAVEclient(DATASTACK)
    now = datetime.datetime.now(datetime.timezone.utc)
    meshes = ["ME_R", "ME_L", "LO_R", "LO_L"]
    trees = {m: cKDTree(ol_meshes.mesh(m)[0] / 1000.0) for m in meshes}

    def region(P):
        reg = ol_meshes.assign(P, meshes).astype(object)
        none = np.flatnonzero(reg == "none")
        if len(none):
            D = np.column_stack([trees[m].query(P[none])[0] for m in meshes])
            b = D.argmin(1)
            reg[none] = np.where(D[np.arange(len(none)), b] <= MARGIN_UM, np.array(meshes)[b], "none")
        return reg

    syn = {}
    for role in ("out", "in"):
        roots = sorted({int(r) for (s, sd), v in tg.items() for r in v if r and (role == "out") == (s != "T4T5")})
        kw = {"pre": roots} if role == "out" else {"post": roots}
        d = snap.synapses(**kw, when="now", positions=True, client=client)
        d = d[d.pre_root != d.post_root]
        who = "pre" if role == "out" else "post"
        # snapshot positions are voxels; lamina_footprint.flat guesses the unit from the distance to the lamina and
        # took the medulla/lobula synapses of T4/T5 for nm (2026-11-01), so no guessing here
        V = d[[f"{who}_x", f"{who}_y", f"{who}_z"]].to_numpy(float)
        f = pd.DataFrame({"pre": d.pre_root.astype(str).to_numpy(), "post": d.post_root.astype(str).to_numpy(),
                          "vx": V[:, 0], "vy": V[:, 1], "vz": V[:, 2]})
        f["np"] = region(V * VOX / 1000.0)
        f["cell"] = f[who]
        syn[role] = f
        print(f"{role}put synapses of {len(roots)} target cells: {len(f)}", flush=True)

    inv = pd.read_csv(fig("cutout_inventory.tsv"), sep="\t")
    taken = [(np.array(json.loads(a)), np.array(json.loads(b))) for a, b in zip(inv.lo, inv.hi)]
    overlap = lambda lo, hi, boxes: any(np.all(lo < b) and np.all(a < hi) for a, b in boxes)
    half = SIZE / 2.0
    rows = []
    for cat, side, tset, role, npl, n_box, per_cell in REQUESTS:
        cells = tg.get((tset, side), {})
        f = syn[role]
        f = f[f.cell.isin(set(cells)) & (f.np == f"{npl}_{side}")]
        if not len(f):
            print(f"{cat} {side}: no target synapses", flush=True)
            continue
        V = f[["vx", "vy", "vz"]].to_numpy()
        tree = cKDTree(V / half)
        cnt = np.array([len(x) for x in tree.query_ball_point(V / half, 1.0, p=np.inf)])
        chosen, used_cells = [], set()
        for i in np.argsort(-cnt):
            if len(chosen) >= n_box:
                break
            c = np.round(V[i])
            lo, hi = c - np.floor(half), c - np.floor(half) + SIZE
            if overlap(lo, hi, taken) or any(np.linalg.norm((c - cc) * VOX / 1000.0) < MIN_SEP_UM for cc in chosen):
                continue
            inside = np.all((V >= lo) & (V < hi), axis=1)
            by_cell = f[inside].cell.value_counts()
            if per_cell and by_cell.index[0] in used_cells:
                continue
            used_cells.add(by_cell.index[0])
            chosen.append(c)
            taken.append((lo, hi))
            rows.append(dict(category=cat, side=side, neuropil=npl, center=c, lo=lo, hi=hi, n_target=int(inside.sum()),
                             cells=by_cell.to_dict(), labels={r: cells[r] for r in by_cell.index},
                             points=V[inside].tolist(), role=role, target_set=set(cells)))
        print(f"{cat} {side}: {len(chosen)} boxes, target synapses {[r['n_target'] for r in rows[-len(chosen):]] if chosen else []}",
              flush=True)

    # every predicted synapse in each box (live, read-only)
    t = throttle(0.0)
    for k, r in enumerate(rows):
        t = throttle(t)
        lo_nm, hi_nm = (r["lo"] * VOX).tolist(), (r["hi"] * VOX).tolist()
        try:
            a = client.materialize.live_query("synapses_v2", timestamp=now, filter_spatial_dict={"ctr_pt_position": [lo_nm, hi_nm]})
            a = a[a.pre_pt_root_id != a.post_pt_root_id]
            side_col = a.pre_pt_root_id if r["role"] == "out" else a.post_pt_root_id
            r.update(n_all=len(a), n_pre_cells=a.pre_pt_root_id.nunique(), n_post_cells=a.post_pt_root_id.nunique(),
                     n_target_live=int(side_col.astype(str).isin(r["target_set"]).sum()))
        except Exception as e:
            print(f"  box {k + 1}: spatial query failed: {str(e)[:100]}", flush=True)
            r.update(n_all=-1, n_pre_cells=-1, n_post_cells=-1, n_target_live=-1)
        t = time.time()

    # links: one per box, and an overview
    links = []
    ids = {}
    for r in rows:
        key = f"{r['category'].split()[0]}-{r['side']}"
        ids[key] = ids.get(key, 0) + 1
        r["box_id"] = f"{key}-{ids[key]}"
        spec = {"neurons": [{"name": f"{r['category']} cells in the box ({len(r['cells'])})", "color": "#E69F00",
                             "ids": list(r["cells"])}],
                "annotations": [{"name": f"cutout {r['box_id']}", "color": "#ffff00",
                                 "boxes": [[r["lo"].tolist(), r["hi"].tolist()]]},
                                {"name": f"{'output' if r['role'] == 'out' else 'input'} synapses of the target cells in the box",
                                 "color": "#56B4E9", "points": r["points"]}],
                "layout": "xy-3d", "position": r["center"].tolist(), "scale": 2500, "cross_scale": 0.5, "image": True,
                "select": f"cutout {r['box_id']}"}
        links.append((r, nglink.clean_state(nglink.build(spec))))
    over = {"meshes": [{"kind": "OL", "segments": ["11", "14", "21", "24"], "alpha": 0.1}],
            "annotations": [{"name": "new cutouts (2026-11-01)", "color": "#ffff00",
                             "boxes": [[r["lo"].tolist(), r["hi"].tolist()] for r in rows]},
                            {"name": "existing cutouts", "color": "#56B4E9",
                             "boxes": [[json.loads(a), json.loads(b)] for a, b in zip(inv.lo, inv.hi)]}],
            "layout": "3d", "select": "new cutouts (2026-11-01)"}
    t = throttle(0.0)
    for r, state in links:
        t = throttle(t)
        r["link"] = client.state.build_neuroglancer_url(client.state.upload_state_json(state),
                                                        ngl_url="https://spelunker.cave-explorer.org")
        t = time.time()
    t = throttle(t)
    overview = client.state.build_neuroglancer_url(client.state.upload_state_json(nglink.clean_state(nglink.build(over))),
                                                   ngl_url="https://spelunker.cave-explorer.org")
    fmt = lambda v: ", ".join(str(int(x)) for x in v)
    R = pd.DataFrame([dict(box_id=r["box_id"], category=r["category"], side=r["side"], neuropil=r["neuropil"],
                           center_xyz=fmt(r["center"]), lo_xyz=fmt(r["lo"]), hi_xyz=fmt(r["hi"]),
                           size_voxels="125 x 125 x 50", n_target_synapses=r["n_target_live"], n_all_synapses=r["n_all"],
                           n_presynaptic_cells=r["n_pre_cells"], n_postsynaptic_cells=r["n_post_cells"],
                           target_cells="; ".join(f"{c} ({n} syn, {r['labels'][c]})" for c, n in r["cells"].items()),
                           link=r["link"]) for r in rows])
    out = fig(f"{stamp}_cutouts_targets.tsv")
    R.to_csv(out, sep="\t", index=False)
    open(fig(f"{stamp}_cutouts_targets_overview_link.txt"), "w").write(overview + "\n")
    print(R[["box_id", "center_xyz", "n_target_synapses", "n_all_synapses"]].to_string(index=False))
    print(f"overview: {overview}\nwritten {out}")


if __name__ == "__main__":
    main()
