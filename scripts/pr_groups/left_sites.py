"""Step 1 of the photoreceptor-group finder for the LEFT eye: census sites around the SeaTable cartridges.

The right-side step 1 (lattice.py) fits a plane, a quadric and a hexagonal lattice to the lamina points of 12 known
cartridges. On the left SeaTable has three (LRS cartridges 1, 2 and 8, eight cells each; 1 and 2 are 14 um apart, 8
is ~115 um away), too few for a surface or lattice fit. The census (src/faab/pr/bundles.py) only needs sites whose
boxes cover the bundle zone, so each usable cartridge gets a hexagonal disc of RINGS rings at the right-side lattice
spacing, through its lamina point and PERPENDICULAR to its axis (the cross-section of the bundle zone, where
neighboring ommatidial bundles run in parallel). The LA_L mesh normal was tried for the patch plane and dropped: it
lies ~82 deg from the measured cartridge 8 axis (2026-10-29), which would put the patch along the axons.
Per cartridge as on the right: pooled PCA axis of the R1-R6 L2 skeletons, oriented retina -> lamina by the R1-R6
presynaptic sites, whose median is the lamina point (local synapse snapshot + one CAVE catch-up).
A cartridge is usable only if its R1-R6 reach the bundle zone (retina point >= OWN_AXIS_MIN_UM from the lamina
point). LRS 1 and 2 do not (9-13 um, terminal pieces; 2026-10-29 their axis cannot be recovered: the stubs are too
short, the mirrored right-side axes miss cartridge 8's axis by 138 deg, and 90% of the 587 PR-like cells feeding the
same lamina targets are terminal pieces too, ending in a ~17-um layer parallel to the lamina - scratch/pr_groups/
left_*.py); they are written to pr_left_lattice_noaxis.tsv and get no sites.
Outputs (read by bundles.py --left and bundles_cluster.py --left): figures/pr_left_lattice_known.tsv,
pr_left_lattice_predicted.tsv, pr_left_lattice_seatable_prs.tsv, pr_left_lattice_noaxis.tsv, pr_left_sites_ng.txt.
Read-only, throttled.
"""
import os
import pickle
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import caveclient
from fafbseg import flywire

from faab import synsnap
from faab.clients.cave import DATASTACK, throttle
from faab.config import FIGURES
from faab.partners.top10_xyz import VOX
import faab.nglink as nglink

SERVER = "https://flytable.mrc-lmb.cam.ac.uk/"
SCACHE = os.path.join(FIGURES, "pr_left_skels.pkl")
SPACING_UM = 5.9   # right-side hexagonal lattice spacing (lattice.py, 2026-10-01)
RINGS = 4
ZONE_UM = 35.0
OWN_AXIS_MIN_UM = 45.0  # the bundle zone starts 45 um above the lamina
DIMS = {"x": [1.6e-08, "m"], "y": [1.6e-08, "m"], "z": [4.5e-08, "m"]}


def _s(v):
    return "" if (v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v)) else str(v).strip()


def cartridge_geometry(skels_r16, syn_um):
    """pooled axis (unit, retina -> lamina), lamina point, retina point (um); same rule as lattice.py with synapses"""
    pts = np.vstack([sk.nodes[["x", "y", "z"]].values.astype(float) / 1000.0 for sk in skels_r16])
    c = pts.mean(0)
    axis = np.linalg.svd(pts - c, full_matrices=False)[2][0]
    s = (pts - c) @ axis
    if (np.median(syn_um, axis=0) - c) @ axis < 0:
        axis, s = -axis, -s
    return axis, np.median(syn_um, axis=0), pts[s < s.min() + ZONE_UM].mean(0)


def main():
    t0 = time.time()
    import seaserpent as ss
    meta = ss.Table("aedes_main", server=SERVER).to_frame()
    meta.columns = [str(c) for c in meta.columns]
    typ = meta["type"].astype(object).map(_s)
    pr = meta[typ.isin(["R1-6", "R7-8"]) & (meta["side"].astype(object).map(_s) == "L")].copy()
    pr["cart"] = pr["notes"].astype(object).map(_s).str.extract(r"cartridge\s*(\d+)")[0]
    pr["type"] = typ[pr.index]
    client = caveclient.CAVEclient(DATASTACK)
    t = throttle(time.time())
    pr["root"] = [int(x) for x in np.asarray(client.chunkedgraph.get_roots([int(_s(v)) for v in pr["supervoxel_id"]]))]
    print(f"SeaTable left-side PRs: {len(pr)}; cartridges {sorted(pr.cart.dropna().unique(), key=int)}", flush=True)
    pr[["root", "type", "cart", "notes"]].assign(notes=pr["notes"].astype(object).map(_s)).to_csv(
        os.path.join(FIGURES, "pr_left_lattice_seatable_prs.tsv"), sep="\t", index=False)
    carts = {f"L{c}": dict(r16=[int(r) for r in sub[sub.type == "R1-6"].root], r78=[int(r) for r in sub[sub.type == "R7-8"].root])
             for c, sub in pr.dropna(subset=["cart"]).groupby("cart")}

    sk = pickle.load(open(SCACHE, "rb")) if os.path.exists(SCACHE) else {}
    need = sorted({r for c in carts.values() for r in c["r16"] if r not in sk})
    print(f"skeletons to fetch: {len(need)}", flush=True)
    for r in need:  # one root at a time, >= 5 s apart
        t = throttle(t)
        for s in flywire.get_l2_skeleton([r], dataset=DATASTACK, max_threads=1, omit_failures=True, progress=False):
            sk[int(s.id)] = s
    if need:
        pickle.dump(sk, open(SCACHE, "wb"))

    synsnap.refresh()
    pre = synsnap.Snapshot().synapses(pre=[r for c in carts.values() for r in c["r16"]], when="now", positions=True, client=client)
    rows = []
    for name, c in carts.items():
        skels = [sk[r] for r in c["r16"] if r in sk]
        p = pre[pre.pre_root.isin(c["r16"])]
        syn = p[["pre_x", "pre_y", "pre_z"]].to_numpy(float) * np.array(VOX) / 1000.0
        axis, lam, ret = cartridge_geometry(skels, syn)
        rows.append(dict(cartridge=name, source="LRS", n_r16=len(skels), n_r78=len(c["r78"]),
                         lamina_from=f"{len(syn)} presynaptic sites", lam_x=lam[0], lam_y=lam[1], lam_z=lam[2],
                         ret_x=ret[0], ret_y=ret[1], ret_z=ret[2], ax_x=axis[0], ax_y=axis[1], ax_z=axis[2],
                         length_um=float(np.linalg.norm(lam - ret)),
                         r16_ids=",".join(map(str, c["r16"])), r78_ids=",".join(map(str, c["r78"]))))
    allk = pd.DataFrame(rows)
    for a, v in zip("xyz", VOX):
        allk[f"lam_vox_{a}"] = (allk[f"lam_{a}"] * 1000.0 / v).round(0).astype(int)
    print(allk[["cartridge", "n_r16", "n_r78", "lamina_from", "length_um", "lam_vox_x", "lam_vox_y", "lam_vox_z"]].to_string(index=False), flush=True)
    noaxis = allk[allk.length_um < OWN_AXIS_MIN_UM]
    known = allk[allk.length_um >= OWN_AXIS_MIN_UM].reset_index(drop=True)
    noaxis.to_csv(os.path.join(FIGURES, "pr_left_lattice_noaxis.tsv"), sep="\t", index=False)
    print(f"usable cartridges (R1-R6 reach >= {OWN_AXIS_MIN_UM:.0f} um): {known.cartridge.tolist()}; without an axis (no sites): "
          f"{noaxis.cartridge.tolist()}", flush=True)
    if known.empty:
        raise SystemExit("no usable cartridge")
    L = known[["lam_x", "lam_y", "lam_z"]].values
    A = known[["ax_x", "ax_y", "ax_z"]].values

    # ---- hexagonal discs perpendicular to each usable cartridge's axis ----
    placed = [x for x in L]
    pred = []
    for k, r in known.iterrows():
        e1 = np.cross(A[k], [0.0, 0.0, 1.0] if abs(A[k][2]) < 0.9 else [1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(A[k], e1)
        a1, a2 = SPACING_UM * e1, SPACING_UM * (0.5 * e1 + np.sqrt(3) / 2 * e2)
        for i in range(-RINGS, RINGS + 1):
            for j in range(-RINGS, RINGS + 1):
                ring = max(abs(i), abs(j), abs(i + j))
                if ring == 0 or ring > RINGS:
                    continue
                p3 = L[k] + i * a1 + j * a2
                if min(np.linalg.norm(q - p3) for q in placed) < 0.5 * SPACING_UM:
                    continue
                placed.append(p3)
                d = np.linalg.norm(L - p3, axis=1)
                wgt = 1.0 / np.maximum(d, 1e-3) ** 2
                ax = (A * wgt[:, None]).sum(0)
                ax /= np.linalg.norm(ax)
                pred.append(dict(site=f"{r.cartridge}({i},{j})", patch=r.cartridge, lat_i=i, lat_j=j, ring=ring,
                                 inside_hull=False, nearest_known_um=round(float(d.min()), 1),
                                 lam_x=p3[0], lam_y=p3[1], lam_z=p3[2], ax_x=ax[0], ax_y=ax[1], ax_z=ax[2]))
    pred = pd.DataFrame(pred)
    for a, v in zip("xyz", VOX):
        pred[f"lam_vox_{a}"] = (pred[f"lam_{a}"] * 1000.0 / v).round(0).astype(int)
    known.to_csv(os.path.join(FIGURES, "pr_left_lattice_known.tsv"), sep="\t", index=False)
    pred.to_csv(os.path.join(FIGURES, "pr_left_lattice_predicted.tsv"), sep="\t", index=False)
    print(f"sites: {len(known)} known + {len(pred)} predicted ({pred.patch.value_counts().to_dict()}; "
          f"{RINGS} rings at {SPACING_UM} um, perpendicular to the axis)", flush=True)

    # ---- link: EM, brain + LA_L meshes, SeaTable PRs, lamina points, sites ----
    spec = {"meshes": [{"kind": "brain", "alpha": 0.15}, {"kind": "OL", "segments": ["14"]}],
            "neurons": [{"name": f"SeaTable PRs, cartridge {c[1:]}", "color": col, "ids": [str(r) for r in carts[c]["r16"] + carts[c]["r78"]]}
                        for c, col in zip(sorted(carts), ["#ff9f1c", "#4363d8", "#3cb44b", "#f032e6"])],
            "view": "anterior", "layout": "3d", "image": True, "scale": 12000,
            "position": [int(v) for v in (L.mean(0) * 1000.0 / np.array(VOX))]}
    state = nglink.build(spec)
    state["layers"][0]["name"] = "EM"
    ann_known = [{"type": "point", "id": f"k_{r.cartridge}", "point": [int(r.lam_vox_x), int(r.lam_vox_y), int(r.lam_vox_z)],
                  "description": f"{r.cartridge} lamina point" + ("" if r.length_um >= OWN_AXIS_MIN_UM else " (no axon: no sites)")}
                 for r in allk.itertuples()]
    ann_pred = [{"type": "point", "id": f"p_{r.site}", "point": [int(r.lam_vox_x), int(r.lam_vox_y), int(r.lam_vox_z)],
                 "description": f"census site {r.site}, ring {r.ring}"} for r in pred.itertuples()]
    state["layers"] += [{"type": "annotation", "source": {"url": "local://annotations", "transform": {"outputDimensions": DIMS}},
                         "annotations": ann_known, "name": "cartridge lamina points", "annotationColor": "#2ecc71"},
                        {"type": "annotation", "source": {"url": "local://annotations", "transform": {"outputDimensions": DIMS}},
                         "annotations": ann_pred, "name": "census sites", "annotationColor": "#ffe119"}]
    open(os.path.join(FIGURES, "pr_left_sites_ng.txt"), "w").write(nglink.to_url(state) + "\n")
    throttle(t)
    sid = client.state.upload_state_json(state)
    print(f"short link: {client.state.build_neuroglancer_url(sid, ngl_url='https://spelunker.cave-explorer.org')}")
    print(f"DONE in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
