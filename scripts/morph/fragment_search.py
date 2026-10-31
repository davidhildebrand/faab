"""Find the missing piece of a truncated cell, using complete copies of the same type as templates.

1. Predict the missing part. L2 skeletons (one CAVE call per cell, cached in figures/fragment_skeletons.pkl, um) are
   resampled to dense cable (0.5 um). Each template is moved onto the truncated cell in its native position by trimmed
   ICP (rotation <= 30 deg + translation, no scaling; every truncated-cell point is matched to its nearest template
   point, so the template's extra branch exerts no pull; several starting offsets, best trimmed RMS). Template cable
   farther than --gap-um from the truncated cell is 'missing'; cable missing in >= 2 templates (within --gap-um of each
   other) is the consensus missing region. Expected missing cable = median over templates.
2. Look for segments there: segmentation cutouts (CloudVolume, mip 1 = 32 x 32 x 45 nm, read-only) at up to --max-boxes
   points of the consensus region (nearest the truncation first, >= --spacing-um apart); supervoxels -> current roots
   in one get_roots call; roots ranked by the number of boxes they occupy.
3. Score the top --top roots. Size first (one get_leaves call each, L2 chunk count): roots far larger than the expected
   piece are 'large cell on the path' (a neighbor, or a merge worth checking) and are not skeletonized. The rest get an
   L2 skeleton (one call each) and:
     cable_um / size_ratio   candidate cable and its ratio to the expected missing cable (the length constraint)
     recall      fraction of the consensus missing cable within 2.5 um of the candidate
     precision   fraction of the candidate's cable within 2.5 um of the consensus missing region
     gap_um      closest approach to the truncated cell (a break should leave a small gap)
     inplace     NBLAST of the candidate against the predicted missing part, in place (no alignment)
     gain        nblast_align (rigid, both ways) of truncated cell + candidate vs the templates, minus the cell alone
Ranking only, nothing dropped. CAVE strictly read-only, >= 5 s between calls. Outputs
figures/<stamp>_fragment_search_<tag>.tsv and _ng.txt (truncated cell yellow, templates green, top candidates
magenta / orange / cyan, predicted missing cable and the truncation point as annotations). Example:
  venv/bin/python scripts/morph/fragment_search.py --tag tm546 --target 648518347546543998 \\
      --templates 648518347603175883 648518347534328191 648518347612400231"""
import argparse
import json
import os
import pickle
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np

if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
from scipy.spatial import cKDTree

from faab import nglink
from faab.clients.cave import DATASTACK, client as cave_client, throttle
from faab.config import FIGURES
from faab.partners.sheet import utc_stamp
from faab.config import fig

SKEL = fig("fragment_skeletons.pkl")
VOXNM = np.array([16.0, 16.0, 45.0])
MIN_SV_VOX = 30     # supervoxel slivers below this many mip-1 voxels are ignored
MIN_HIT_VOX = 100   # a root counts as present in a box with >= this many mip-1 voxels (~0.005 um3)
MID = 371.7         # midline x (um): SeaTable L cells have larger x (type_match.py, 2026-10-26)
DENSE, COARSE = 0.5, 2.0  # resampling (um): geometry / NBLAST
_t = [0.0]


def wait():
    _t[0] = throttle(_t[0])


def angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def kabsch(S, D):
    """rotation R and translation t minimising |S @ R.T + t - D|"""
    s0, d0 = S.mean(0), D.mean(0)
    U, _, Vt = np.linalg.svd((S - s0).T @ (D - d0))
    dd = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, dd]) @ U.T
    return R, d0 - s0 @ R.T


def register(template, fixed, trim=0.9, iters=60, n_starts=25, max_angle=30.0):
    """Trimmed ICP: the partial cell `fixed` (native frame) is matched point-by-point to its nearest `template` points.
    Starts: the fixed centroid placed at the template centroid and at n_starts template points (identity rotation).
    Returns the template moved into the native frame, the rotation applied to it, and the trimmed RMS.
    (pycpd rigid CPD collapsed sigma2 to its floor on sparse L2 clouds, 2026-10-27.)"""
    tree = cKDTree(template)
    rng = np.random.default_rng(0)
    starts = [template.mean(0)] + list(template[rng.choice(len(template), min(n_starts, len(template)), replace=False)])
    best = None
    for c in starts:
        R, t = np.eye(3), c - fixed.mean(0)
        for _ in range(iters):
            d, j = tree.query(fixed @ R.T + t)
            keep = d <= np.quantile(d, trim)
            R2, t2 = kabsch(fixed[keep], template[j[keep]])
            done = np.abs(R2 - R).max() < 1e-6 and np.abs(t2 - t).max() < 1e-4
            R, t = R2, t2
            if done:
                break
        d, _ = tree.query(fixed @ R.T + t)
        rms = float(np.sqrt(np.mean(np.sort(d)[:max(3, int(trim * len(d)))] ** 2)))
        if angle_deg(R) > max_angle:  # copies of a columnar type nearby are near-parallel
            continue
        if best is None or rms < best[0]:
            best = (rms, R, t)
    if best is None:
        raise RuntimeError(f"no registration within {max_angle} degrees")
    rms, R, t = best
    return (template - t) @ R, R, rms  # fixed -> template is x @ R.T + t, so template -> native is (y - t) @ R


def skeletons(ids, sk, flywire):
    for r in [r for r in ids if r not in sk]:
        wait()
        try:
            s = flywire.get_l2_skeleton(int(r), dataset=DATASTACK, progress=False, max_threads=1) / 1000.0
            s.units = "um"
            sk[r] = s
        except Exception as e:
            sk[r] = None
            print(f"  skeleton {r} failed: {str(e)[:80]}", flush=True)
        pickle.dump(sk, open(SKEL, "wb"))
    return sk


def cable_points(s, step, navis):
    try:
        s2 = navis.resample_skeleton(s, resample_to=step)
    except Exception:
        s2 = s
    return np.asarray(s2.nodes[["x", "y", "z"]].values, float)


def dotprops(points, navis):
    d = navis.make_dotprops(np.asarray(points, float), k=min(5, max(2, len(points) - 1)))
    d.units = "um"
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--target", required=True, help="the truncated cell")
    ap.add_argument("--templates", nargs="+", required=True, help="complete cells of the same type")
    ap.add_argument("--gap-um", type=float, default=4.0)
    ap.add_argument("--max-boxes", type=int, default=80)
    ap.add_argument("--box-um", type=float, nargs=3, default=[1.5, 1.5, 0.45], help="cutout half-widths x y z")
    ap.add_argument("--spacing-um", type=float, default=3.0, help="min spacing of cutout centres")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--max-ratio", type=float, default=3.0, help="skip skeletons of roots > this x the expected L2 count")
    a = ap.parse_args()
    import navis
    from cloudvolume import CloudVolume
    from fafbseg import flywire
    t0, stamp = time.time(), utc_stamp()
    client = cave_client()

    ids = [a.target] + list(a.templates)
    wait()
    ok = np.asarray(client.chunkedgraph.is_latest_roots([int(x) for x in ids])).astype(bool)
    now = []
    for x, good in zip(ids, ok):
        if not good:
            wait()
            new = [str(int(r)) for r in np.atleast_1d(client.chunkedgraph.get_latest_roots(int(x)))]
            print(f"{x} was edited; now {new} (using the first)", flush=True)
            x = new[0]
        now.append(x)
    target, templates = now[0], now[1:]
    sk = pickle.load(open(SKEL, "rb")) if os.path.exists(SKEL) else {}
    sk = skeletons(now, sk, flywire)
    T = cable_points(sk[target], DENSE, navis)
    tree_t = cKDTree(T)
    t_cable, t_l2 = float(sk[target].cable_length), len(sk[target].nodes)
    print(f"target {target}: cable {t_cable:.0f} um ({t_l2} L2 nodes); templates cable "
          f"{[round(float(sk[r].cable_length)) for r in templates]} um", flush=True)

    # ---- 1. predicted missing region in the target's native frame
    moved, missing = {}, {}
    tside = "L" if T[:, 0].mean() > MID else "R"
    for r in templates:
        P = cable_points(sk[r], DENSE, navis)
        if ("L" if P[:, 0].mean() > MID else "R") != tside:  # rigid registration cannot reflect
            P[:, 0] = 2 * MID - P[:, 0]
            print(f"  template {r} is on the other side: mirrored", flush=True)
        ty, R, rms = register(P, T)
        d_t, _ = tree_t.query(ty)
        d_back, _ = cKDTree(ty).query(T)
        moved[r] = ty
        missing[r] = d_t > a.gap_um
        print(f"  template {r}: RMS {rms:.2f} um, rotation {angle_deg(R):.0f} deg; target cable within 2 um of it "
              f"{np.mean(d_back <= 2):.0%}; template cable missing from the target ~{DENSE * missing[r].sum():.0f} um", flush=True)
    pts = np.vstack([moved[r][missing[r]] for r in templates])
    src = np.concatenate([[r] * int(missing[r].sum()) for r in templates])
    support = np.zeros(len(pts), bool)
    for r in templates:
        others = [moved[o][missing[o]] for o in templates if o != r and missing[o].any()]
        if others and (src == r).any():
            d, _ = cKDTree(np.vstack(others)).query(pts[src == r])
            support[src == r] = d <= a.gap_um
    C = pts[support] if len(templates) > 1 else pts
    if not len(C):
        raise SystemExit("no consensus missing region: the templates do not agree on a missing part")
    expected = float(np.median([DENSE * missing[r].sum() for r in templates]))
    exp_l2 = expected * t_l2 / max(t_cable, 1)
    dC, iT = tree_t.query(C)
    brk = T[iT[np.argmin(dC)]]
    brk_vox = np.round(brk * 1000 / VOXNM).astype(int).tolist()
    print(f"consensus missing region: {len(C)} points; expected missing cable ~{expected:.0f} um (~{exp_l2:.0f} L2 chunks); "
          f"extent {np.ptp(C, axis=0).round(1)} um; truncation near {brk_vox} (16x16x45 nm voxels)", flush=True)

    # ---- 2. cutouts along the predicted region, nearest the truncation first
    centres = []
    for i in np.argsort(np.linalg.norm(C - brk, axis=1)):
        if all(np.linalg.norm(C[i] - c) >= a.spacing_um for c in centres):
            centres.append(C[i])
        if len(centres) >= a.max_boxes:
            break
    vol = CloudVolume(client.info.segmentation_source(), use_https=True, agglomerate=False, progress=False,
                      fill_missing=True, bounded=False, mip=1)
    res = np.array(vol.resolution, float)
    half = np.maximum(1, np.round(np.array(a.box_um) * 1000.0 / res).astype(int))
    boxes = []
    for k, c in enumerate(centres):
        v = np.round(c * 1000.0 / res).astype(int)
        for attempt in range(3):
            try:
                cut = vol[v[0] - half[0]:v[0] + half[0], v[1] - half[1]:v[1] + half[1], v[2] - half[2]:v[2] + half[2]]
                break
            except Exception as e:
                print(f"  cutout {k} attempt {attempt + 1} failed: {str(e)[:80]}", flush=True)
                time.sleep(10)
        else:
            boxes.append({})
            continue
        sv, cnt = np.unique(np.asarray(cut).squeeze(), return_counts=True)
        keep = (sv != 0) & (cnt >= MIN_SV_VOX)
        boxes.append(dict(zip(sv[keep].astype(np.int64).tolist(), cnt[keep].astype(int).tolist())))
    svs = sorted({s for b in boxes for s in b})
    print(f"{len(centres)} cutouts ({(2 * half).tolist()} mip-1 voxels each), {len(svs)} supervoxels; {time.time() - t0:.0f} s", flush=True)
    wait()
    roots = dict(zip(svs, np.asarray(client.chunkedgraph.get_roots(np.array(svs, dtype=np.int64))).astype(np.int64).tolist()))
    hits = {}
    for k, b in enumerate(boxes):
        per = {}
        for s, n in b.items():
            per[roots[s]] = per.get(roots[s], 0) + n
        for r, n in per.items():
            if r and n >= MIN_HIT_VOX:
                hits.setdefault(str(r), []).append(k)
    tgt_boxes = len(hits.get(target, []))
    tmpl_boxes = {r: len(hits.get(r, [])) for r in templates}
    for r in [target] + templates:
        hits.pop(r, None)
    ranked = sorted(hits, key=lambda r: -len(hits[r]))
    print(f"roots in >= 1 box: {len(ranked)}; the target occupies {tgt_boxes} and the templates {tmpl_boxes} of {len(centres)} "
          f"boxes; top box counts {[len(hits[r]) for r in ranked[:12]]}", flush=True)

    # ---- 3. size screen, then score
    cand = ranked[:a.top]
    print(f"size screen (L2 chunk counts) for {len(cand)} candidates (~{len(cand) * 5.2 / 60:.0f} min)", flush=True)
    n_l2 = {}
    for r in cand:
        wait()
        try:
            n_l2[r] = len(client.chunkedgraph.get_leaves(int(r), stop_layer=2))
        except Exception as e:
            n_l2[r] = -1
            print(f"  get_leaves {r} failed: {str(e)[:80]}", flush=True)
    small = [r for r in cand if 0 <= n_l2[r] <= a.max_ratio * max(exp_l2, 1)]
    print(f"skeletons for {len([r for r in small if r not in sk])} candidates within {a.max_ratio:g} x the expected size", flush=True)
    sk = skeletons(small, sk, flywire)
    tree_c = cKDTree(C)
    miss_dp = dotprops(C[::4], navis)  # 2 um spacing
    tcoarse = cable_points(sk[target], COARSE, navis)
    tmpl = navis.NeuronList([dotprops(cable_points(sk[r], COARSE, navis), navis) for r in templates])
    kw = dict(align_method="rigid", two_way_align=True, n_cores=1, progress=False, align_kwargs={"scale": False})
    base = float(navis.nblast_align(navis.NeuronList([dotprops(tcoarse, navis)]), tmpl, **kw).values.mean())
    rows = []
    for r in cand:
        row = dict(root=r, boxes=len(hits[r]), l2_chunks=n_l2[r], l2_ratio=round(n_l2[r] / max(exp_l2, 1), 2))
        if r not in small or sk.get(r) is None:
            row["size_flag"] = "large cell on the path" if n_l2[r] > a.max_ratio * max(exp_l2, 1) else "no skeleton"
            rows.append(row)
            continue
        P = cable_points(sk[r], DENSE, navis)
        cable = float(sk[r].cable_length) if len(sk[r].nodes) > 1 else 0.0
        d_cp, _ = cKDTree(P).query(C)
        d_pc, _ = tree_c.query(P)
        d_gap, _ = tree_t.query(P)
        pc = cable_points(sk[r], COARSE, navis)
        inplace = float(navis.nblast(navis.NeuronList([dotprops(pc, navis)]), navis.NeuronList([miss_dp]),
                                     n_cores=1, progress=False).values[0, 0]) if len(pc) >= 3 else np.nan
        comb = float(navis.nblast_align(navis.NeuronList([dotprops(np.vstack([tcoarse, pc]), navis)]), tmpl, **kw).values.mean())
        ratio = cable / max(expected, 1)
        row.update(cable_um=round(cable, 1), size_ratio=round(ratio, 2), recall=float(np.mean(d_cp <= 2.5)),
                   precision=float(np.mean(d_pc <= 2.5)), gap_um=float(d_gap.min()), inplace=inplace, gain=comb - base,
                   size_flag="ok" if 0.25 <= ratio <= 2.0 else ("small" if ratio < 0.25 else "large"),
                   xyz_nearest_truncation=np.round(P[np.argmin(np.linalg.norm(P - brk, axis=1))] * 1000 / VOXNM).astype(int).tolist())
        rows.append(row)
    out = pd.DataFrame(rows)
    scored = out.recall.notna()
    out["score"] = np.nan
    s = out[scored]
    out.loc[scored, "score"] = (s.recall.rank(pct=True) + s.precision.rank(pct=True) + s.gain.rank(pct=True)
                                + (-s.gap_um).rank(pct=True) + (s.size_flag == "ok") * 1.0)
    out = out.sort_values(["score", "boxes"], ascending=[False, False], na_position="last")
    base_path = fig(f"{stamp}_fragment_search_{a.tag}")
    out.to_csv(base_path + ".tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"\ntruncated cell alone vs templates (nblast_align rigid, both ways): {base:.3f}")
    print(out.round(3).to_string(index=False))

    top = list(out[scored].root[:3]) if scored.any() else []
    cols = ["#ff00ff", "#ff8c00", "#00e5ff"]
    layers = [{"name": "truncated cell", "color": "#ffd400", "ids": [target]},
              {"name": f"complete copies ({len(templates)})", "color": "#2ecc40", "ids": templates}]
    layers += [{"name": f"candidate {k + 1}: {r}", "color": cols[k], "ids": [r]} for k, r in enumerate(top)]
    spec = {"neurons": layers, "layout": "xy-3d", "image": True, "position": brk_vox, "scale": 3000,
            "annotations": [{"name": "predicted missing cable", "color": "#ffffff",
                             "points": np.round(C[::4] * 1000 / VOXNM).tolist()},
                            {"name": "truncation point", "color": "#ff0000", "points": [brk_vox]}],
            "select": layers[2]["name"] if len(layers) > 2 else layers[0]["name"]}
    json.dump(spec, open(base_path + "_spec.json", "w"), indent=1)
    open(base_path + "_ng.txt", "w").write(nglink.to_url(nglink.build(spec)) + "\n")
    print(f"\nwrote {base_path}.tsv and _ng.txt; {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
