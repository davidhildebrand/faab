"""Step 3 of the photoreceptor-group finder: census of the axon-bundle zone above the lamina.

Why: the R1-R6 axons of an ommatidium fan out over the last ~40 um before the lamina (neural superposition) and are
a tight bundle (spread < 3.5 um) only 45-90 um retina-ward of the lamina point, 2-10 um off the line through the
lamina point (diagnostics tmp/pr_probe_diag.py, tmp/pr_r78_line_check.py). So instead of probing lattice sites,
the whole zone is sampled: at depths DEPTHS_UM along each site's axis (known + predicted sites from
faab_pr_lattice.py) a 12 x 12 x 1.4 um cutout (mip 1) is read; overlapping boxes are merged per supervoxel. Every
supervoxel >= MIN_SV_VOX voxels gets a count and centroid, supervoxels are mapped to roots (throttled batches), and
per root and depth the voxel count and centroid are known.
Root classes: PR-like = present at >= 2 depths with the centroid track within MAX_ANGLE_DEG of the local axis;
"big" (> BIG_VOX voxels in a slab) is flagged as possible glia. PR-like roots are clustered into bundles by single
linkage (LINK_UM) on their centroid at the reference depth (REF_DEPTH, falling back to the nearest depth present).
Calibration: the known groups' R1-R6 should each land in one cluster.
Outputs: figures/pr_bundle_boxes.pkl (cache), pr_bundles_roots.tsv, pr_bundles_clusters.tsv, pr_bundles_ng.txt /
pr_bundles_state.json (EM, brain mesh, known PRs grey, new bundles coloured per cluster, cluster centroids).
Read-only. Usage: python faab_pr_bundles.py [--refresh]
"""
from faab.config import REPO
import datetime
import json
import os
import pickle
import sys
import time

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import caveclient
from cloudvolume import CloudVolume
from scipy import ndimage
from scipy.cluster.hierarchy import fcluster, linkage
from faab.sources.cave import DATASTACK, throttle
from faab.partners.top10_xyz import VOX
import faab.nglink as nglink

HERE = REPO
OUT = os.path.join(HERE, "figures")
CACHE = os.path.join(OUT, "pr_bundle_boxes.pkl")
DEPTHS_UM = [45.0, 60.0, 75.0, 90.0]
REF_DEPTH = 75.0
BOX_HALF_UM = (6.0, 6.0, 0.7)
MIN_SV_VOX = 100
MIN_HIT_VOX = 2000      # ~0.09 um3 in a 1.4-um slab: a neurite >= ~0.3 um across
BIG_VOX = 150000        # > ~7 um3 per slab: not an axon (glia sheath, merged segment)
MAX_ANGLE_DEG = 25.0
LINK_UM = 3.5
DIMS = {"x": [1.6e-08, "m"], "y": [1.6e-08, "m"], "z": [4.5e-08, "m"]}
PALETTE = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
           "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9"]


def read_box(vol, centre_um, half_vox, res):
    c = np.round(np.asarray(centre_um) * 1000.0 / res).astype(int)
    lo = c - half_vox
    hi = c + half_vox
    for attempt in range(3):
        try:
            arr = np.asarray(vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).squeeze()
            break
        except Exception as e:
            print(f"  cutout at {c.tolist()} attempt {attempt + 1} failed: {str(e)[:80]}", flush=True)
            time.sleep(10)
    else:
        return {}
    sv, cnt = np.unique(arr, return_counts=True)
    keep = (sv != 0) & (cnt >= MIN_SV_VOX)
    sv, cnt = sv[keep], cnt[keep]
    if len(sv) == 0:
        return {}
    com = ndimage.center_of_mass(np.ones(arr.shape, np.uint8), arr, index=sv)
    out = {}
    for s, n, cm in zip(sv.astype(np.int64).tolist(), cnt.astype(int).tolist(), com):
        out[s] = (n, tuple(((lo + np.array(cm)) * res / 1000.0).tolist()))  # centroid in um
    return out


def main():
    refresh = "--refresh" in sys.argv
    t0 = time.time()
    known = pd.read_csv(os.path.join(OUT, "pr_lattice_known.tsv"), sep="\t", dtype={"r16_ids": str, "r78_ids": str})
    pred = pd.read_csv(os.path.join(OUT, "pr_lattice_predicted.tsv"), sep="\t")
    st_path = os.path.join(OUT, "pr_lattice_seatable_prs.tsv")
    st = pd.read_csv(st_path, sep="\t", dtype=str) if os.path.exists(st_path) else pd.DataFrame(columns=["root", "type", "cart"])
    st_type = {r.root: f"SeaTable {r.type}" + (f" cart {r.cart}" if isinstance(r.cart, str) and r.cart not in ("", "nan") else "")
               for r in st.itertuples()}
    root_group = {}
    for r in known.itertuples():
        for col, cls in (("r16_ids", "R1-R6"), ("r78_ids", "R7/R8")):
            for x in str(getattr(r, col)).split(","):
                if x and x != "nan":
                    root_group[x] = f"{r.cartridge} {cls}"
    sites = [dict(name=r.cartridge, lam=np.array([r.lam_x, r.lam_y, r.lam_z]), axis=np.array([r.ax_x, r.ax_y, r.ax_z]))
             for r in known.itertuples()]
    sites += [dict(name=f"p({r.lat_i},{r.lat_j})", lam=np.array([r.lam_x, r.lam_y, r.lam_z]), axis=np.array([r.ax_x, r.ax_y, r.ax_z]))
              for r in pred.itertuples()]
    mean_axis = np.mean([s["axis"] / np.linalg.norm(s["axis"]) for s in sites], axis=0)
    mean_axis /= np.linalg.norm(mean_axis)

    client = caveclient.CAVEclient(DATASTACK)
    vol = CloudVolume(client.info.segmentation_source(), use_https=True, agglomerate=False, progress=False,
                      fill_missing=True, bounded=False, mip=1)
    res = np.array(vol.resolution, float)
    half_vox = np.round(np.array(BOX_HALF_UM) * 1000.0 / res).astype(int)
    cache = {} if (refresh or not os.path.exists(CACHE)) else pickle.load(open(CACHE, "rb"))
    boxes = cache.setdefault("boxes", {})
    sv2root = cache.setdefault("sv2root", {})
    todo = [(s, d) for s in sites for d in DEPTHS_UM if (s["name"], d) not in boxes]
    print(f"{len(sites)} sites x {len(DEPTHS_UM)} depths; cutouts to read {len(todo)} (cached {len(boxes)}); box "
          f"{2 * BOX_HALF_UM[0]:.0f} x {2 * BOX_HALF_UM[1]:.0f} x {2 * BOX_HALF_UM[2]:.1f} um at {res.astype(int).tolist()} nm", flush=True)
    for k, (s, d) in enumerate(todo):
        a = s["axis"] / np.linalg.norm(s["axis"])
        boxes[(s["name"], d)] = read_box(vol, s["lam"] - a * d, half_vox, res)
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(todo)} cutouts, {time.time() - t0:.0f} s", flush=True)
            pickle.dump(cache, open(CACHE, "wb"))
    pickle.dump(cache, open(CACHE, "wb"))

    # ---- supervoxel -> root (throttled batches; cached) ----
    per_depth = {d: {} for d in DEPTHS_UM}  # depth -> sv -> (count, centroid); overlapping boxes merge here
    for (name, d), bx in boxes.items():
        per_depth[d].update(bx)
    need = sorted({sv for d in per_depth for sv in per_depth[d] if sv not in sv2root})
    print(f"supervoxels: {sum(len(v) for v in per_depth.values())} (distinct {len({sv for d in per_depth for sv in per_depth[d]})}), "
          f"to map {len(need)}", flush=True)
    t = time.time() - 10
    B = 50000
    for i in range(0, len(need), B):
        chunk = np.array(need[i:i + B], dtype=np.int64)
        for attempt in range(4):
            t = throttle(t)
            try:
                roots = np.asarray(client.chunkedgraph.get_roots(chunk)).astype(np.int64)
                break
            except Exception as e:
                print(f"  get_roots batch {i // B} attempt {attempt + 1} failed: {str(e)[:80]}", flush=True)
                time.sleep(15)
        else:
            raise RuntimeError("get_roots keeps failing")
        sv2root.update(zip(chunk.tolist(), roots.tolist()))
        if (i // B) % 10 == 9:
            print(f"  roots for {min(i + B, len(need))}/{len(need)} supervoxels, {time.time() - t0:.0f} s", flush=True)
            pickle.dump(cache, open(CACHE, "wb"))
    pickle.dump(cache, open(CACHE, "wb"))

    # ---- per root and depth: voxels and centroid ----
    acc = {}
    for d, svd in per_depth.items():
        for sv, (n, c) in svd.items():
            r = sv2root.get(sv, 0)
            if not r:
                continue
            e = acc.setdefault((r, d), [0, np.zeros(3)])
            e[0] += n
            e[1] += n * np.array(c)
    rows = {}
    for (r, d), (n, sc) in acc.items():
        if n >= MIN_HIT_VOX:
            rows.setdefault(r, {})[d] = (n, sc / n)
    recs = []
    for r, dd in rows.items():
        ds = sorted(dd)
        cents = np.array([dd[d][1] for d in ds])
        vox = [dd[d][0] for d in ds]
        if len(ds) >= 2:
            v = cents[0] - cents[-1]  # from the shallowest (nearest lamina) to the deepest (retina side) = -axis direction
            ang = np.degrees(np.arccos(np.clip(abs(v @ mean_axis) / np.linalg.norm(v), -1, 1)))
            # lateral wobble: max distance of any centroid from the straight line through the first and last
            u = v / np.linalg.norm(v)
            rel = cents - cents[0]
            wobble = float(np.linalg.norm(rel - np.outer(rel @ u, u), axis=1).max())
        else:
            ang, wobble = np.nan, np.nan
        big = max(vox) > BIG_VOX
        if len(ds) >= 2 and ang <= MAX_ANGLE_DEG and not big:
            cls = "PR-like"
        elif len(ds) >= 2 and big:
            cls = "big (glia/merge?)"
        elif len(ds) >= 2:
            cls = "oblique"
        else:
            cls = "single depth"
        ref = REF_DEPTH if REF_DEPTH in dd else min(ds, key=lambda x: abs(x - REF_DEPTH))
        rc = dd[ref][1]
        recs.append(dict(root=str(r), cls=cls, n_depths=len(ds), depths=",".join(f"+{int(d)}" for d in ds),
                         vox_per_depth=",".join(str(x) for x in vox), vox_med=int(np.median(vox)), angle_deg=round(ang, 1) if ds else np.nan,
                         wobble_um=round(wobble, 2) if len(ds) >= 2 else np.nan, ref_depth=int(ref),
                         ref_x=rc[0], ref_y=rc[1], ref_z=rc[2], known=root_group.get(str(r), ""), seatable=st_type.get(str(r), "")))
    df = pd.DataFrame(recs)
    for a_, v_ in zip("xyz", VOX):
        df[f"ref_vox_{a_}"] = (df[f"ref_{a_}"] * 1000.0 / v_).round(0).astype(int)
    df["xyz_ref"] = df.ref_vox_x.astype(str) + ", " + df.ref_vox_y.astype(str) + ", " + df.ref_vox_z.astype(str)
    print(f"\nroots present (>= {MIN_HIT_VOX} voxels in a slab): {len(df)}; classes {df.cls.value_counts().to_dict()}; "
          f"known PRs among them {int((df.known != '').sum())} of {len(root_group)}", flush=True)
    kn = df[df.known != ""]
    print(f"known PRs: classes {kn.cls.value_counts().to_dict()}; n_depths {kn.n_depths.value_counts().sort_index().to_dict()}; "
          f"vox_med p10/50/90 {kn.vox_med.quantile([0.1, 0.5, 0.9]).round(0).astype(int).tolist()}; angle p50/90 "
          f"{kn.angle_deg.quantile([0.5, 0.9]).round(1).tolist()}; wobble p50/90 {kn.wobble_um.quantile([0.5, 0.9]).round(2).tolist()}", flush=True)
    missing = sorted(set(root_group) - set(df.root))
    print(f"known PRs not seen at any depth: {len(missing)}: {[root_group[m] + ' ' + m for m in missing][:20]}", flush=True)

    # ---- bundles: single-linkage clusters of PR-like roots on their reference-depth centroid ----
    pr = df[df.cls == "PR-like"].copy()
    X = pr[["ref_x", "ref_y", "ref_z"]].values
    if len(pr) >= 2:
        pr["cluster"] = fcluster(linkage(X, method="single"), t=LINK_UM, criterion="distance")
    else:
        pr["cluster"] = 1
    cl = pr.groupby("cluster").agg(n_roots=("root", "size"), n_known=("known", lambda s: int((s != "").sum())),
                                   known_groups=("known", lambda s: ", ".join(sorted({x.split(" ")[0] for x in s if x}))),
                                   n_seatable=("seatable", lambda s: int((s != "").sum())),
                                   cx=("ref_x", "mean"), cy=("ref_y", "mean"), cz=("ref_z", "mean"),
                                   spread_um=("ref_x", lambda s: 0.0), ref_depths=("ref_depth", lambda s: ",".join(map(str, sorted(set(s))))))
    for c, sub in pr.groupby("cluster"):
        P = sub[["ref_x", "ref_y", "ref_z"]].values
        cl.loc[c, "spread_um"] = round(float(np.linalg.norm(P - P.mean(0), axis=1).max()), 1)
    cl["n_new"] = cl.n_roots - cl.n_known
    for a_, v_ in zip("xyz", VOX):
        cl[f"c_vox_{a_}"] = (cl[f"c{a_}"] * 1000.0 / v_).round(0).astype(int)
    cl["xyz_centroid"] = cl.c_vox_x.astype(str) + ", " + cl.c_vox_y.astype(str) + ", " + cl.c_vox_z.astype(str)
    cl = cl.sort_values(["n_new", "n_roots"], ascending=False)
    pr = pr.merge(cl[["n_roots", "n_known", "n_new"]], left_on="cluster", right_index=True)
    pr.to_csv(os.path.join(OUT, "pr_bundles_roots.tsv"), sep="\t", index=False)
    df.to_csv(os.path.join(OUT, "pr_bundles_all_roots.tsv"), sep="\t", index=False)
    cl.to_csv(os.path.join(OUT, "pr_bundles_clusters.tsv"), sep="\t")
    pd.set_option("display.width", 250)
    print(f"\nbundles (single linkage {LINK_UM} um on the +{int(REF_DEPTH)} um centroids): {len(cl)} clusters, "
          f"{int((cl.n_roots >= 4).sum())} with >= 4 roots, {int(((cl.n_roots >= 4) & (cl.n_known == 0)).sum())} of those entirely new")
    print("calibration: clusters containing known PRs:")
    print(cl[cl.n_known > 0][["n_roots", "n_known", "n_new", "known_groups", "spread_um", "xyz_centroid"]].to_string())
    print("\nnew bundles (no known PR), >= 3 roots:")
    print(cl[(cl.n_known == 0) & (cl.n_roots >= 3)][["n_roots", "n_seatable", "spread_um", "ref_depths", "xyz_centroid"]].to_string())

    # ---- link ----
    known_ids = sorted(root_group)
    new_cl = cl[(cl.n_known == 0) & (cl.n_roots >= 3)].index.tolist()
    neurons = [{"name": "known PRs (8 groups + LRS cartridges)", "color": "#808080", "ids": known_ids}]
    for k, c in enumerate(new_cl[:len(PALETTE)]):
        ids = sorted(pr[pr.cluster == c].root)
        neurons.append({"name": f"new bundle {k + 1} ({len(ids)} axons)", "color": PALETTE[k], "ids": ids})
    rest = sorted(pr[(pr.known == "") & ~pr.cluster.isin(new_cl[:len(PALETTE)])].root)
    if rest:
        neurons.append({"name": "other new PR-like axons", "color": "#ffffff", "ids": rest})
    spec = {"meshes": [{"kind": "brain", "alpha": 0.15}], "neurons": neurons, "view": "anterior", "layout": "3d", "image": True,
            "scale": 12000, "position": [int(v) for v in (np.mean([s["lam"] for s in sites], axis=0) * 1000.0 / VOX)]}
    state = nglink.build(spec)
    state["layers"][0]["name"] = "EM"
    anns = [{"type": "point", "id": f"b{c}", "point": [int(cl.loc[c, "c_vox_x"]), int(cl.loc[c, "c_vox_y"]), int(cl.loc[c, "c_vox_z"])],
             "description": f"bundle {c}: {int(cl.loc[c, 'n_roots'])} axons ({int(cl.loc[c, 'n_known'])} known: {cl.loc[c, 'known_groups']})"}
            for c in cl.index]
    state["layers"].append({"type": "annotation", "source": {"url": "local://annotations", "transform": {"outputDimensions": DIMS}},
                            "annotations": anns, "name": f"bundle centroids (+{int(REF_DEPTH)} um)", "annotationColor": "#ffe119"})
    url = nglink.to_url(state)
    open(os.path.join(OUT, "pr_bundles_ng.txt"), "w").write(url + "\n")
    json.dump(state, open(os.path.join(OUT, "pr_bundles_state.json"), "w"), indent=1)
    print(f"\nlink {len(url)} chars saved to figures/pr_bundles_ng.txt; live at {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M} UTC; "
          f"DONE in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
