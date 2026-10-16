"""Step 2 of the photoreceptor-group finder: probe predicted cartridge sites for photoreceptor axon bundles.

For every cartridge in figures/pr_lattice_known.tsv (calibration) and pr_lattice_predicted.tsv (search) a small
segmentation cutout (BOX_HALF_UM half-widths, mip 1 = 32 x 32 x 45 nm) is read at depths DEPTHS_UM along the
cartridge axis relative to the lamina point (negative = beyond the lamina toward the medulla, positive = toward
the retina). Supervoxels with >= MIN_SV_VOX voxels are mapped to roots in throttled batches (one CAVE call per
5 s); per site, depth and root the voxel count is kept.
Classification per root (classify()): PR-like (straight retina->lamina bundle signature), PR-axon-like (truncated
fragment on the retina side), lamina-only, other. Calibration on the known cartridges: own PRs recovered, extra
PR-like roots. Outputs: figures/pr_probe_boxes.pkl (raw cutout counts, reused on rerun unless --refresh),
pr_probe_hits.tsv, pr_probe_candidates.tsv, pr_probe_ng.txt / pr_probe_state.json (link: EM, brain mesh, known PRs
grey, candidates by class, SeaTable-typed PRs blue, predicted sites yellow). Read-only.
Usage: python faab_pr_probe.py [known|predicted|both] [--refresh]
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
from faab.clients.cave import DATASTACK, throttle
from faab.partners.top10_xyz import VOX
import faab.nglink as nglink

HERE = REPO
OUT = os.path.join(HERE, "figures")
BOXES = os.path.join(OUT, "pr_probe_boxes.pkl")
DEPTHS_UM = [-15.0, 0.0, 15.0, 30.0, 45.0, 60.0]
BOX_HALF_UM = (3.0, 3.0, 0.7)  # 6 x 6 x 1.4 um box (cartridge spacing ~5.9 um; lattice residuals up to ~2.5 um)
MIN_SV_VOX = 100               # supervoxel slivers below this (mip-1 voxels, 4.6e-5 um3 each) are ignored
MIN_HIT_VOX = 500              # a root counts as present at a depth with >= this many voxels (~0.02 um3)
RETINA_MIN = 30.0
ROOT_BATCH = 50000
DIMS = {"x": [1.6e-08, "m"], "y": [1.6e-08, "m"], "z": [4.5e-08, "m"]}


def classify(depths):
    """PR-like: the straight retina->lamina axon-bundle signature (>= 3 depths hit, incl. one >= RETINA_MIN and the
    lamina at 0/+15). PR-axon-like: >= 2 retina-side depths incl. one >= RETINA_MIN but no lamina hit (truncated or
    cut fragment; kept and ranked, not dropped). lamina-only: hits only at -15/0/+15 (LMCs, lamina glia, R7/R8 of a
    neighbouring site). Note: the R1-R6 segments themselves extend ~20 um past their presynaptic zone, so a hit at
    -15 does not separate R7/R8 from R1-R6; it is kept as the `beyond` flag only."""
    ds = set(depths)
    retina_hits = [d for d in ds if d >= 15.0]
    retina = any(d >= RETINA_MIN for d in ds)
    lamina = any(d in (0.0, 15.0) for d in ds)
    if len(ds) >= 3 and retina and lamina:
        return "PR-like"
    if len(retina_hits) >= 2 and retina:
        return "PR-axon-like"
    if ds and max(ds) <= 15.0:
        return "lamina-only"
    return "other"


def read_boxes(vol, carts, cache):
    """supervoxel voxel counts per (cartridge name, depth); cached in `cache`"""
    res = np.array(vol.resolution, float)
    half = np.round(np.array(BOX_HALF_UM) * 1000.0 / res).astype(int)
    todo = [(c, d) for c in carts for d in DEPTHS_UM if (c["name"], d) not in cache]
    print(f"cutouts to read: {len(todo)} (cached {len(cache)})", flush=True)
    t0 = time.time()
    for k, (c, d) in enumerate(todo):
        a = c["axis"] / np.linalg.norm(c["axis"])
        centre = np.round((c["lam"] - a * d) * 1000.0 / res).astype(int)  # +axis = retina -> lamina
        for attempt in range(3):
            try:
                cut = vol[centre[0] - half[0]:centre[0] + half[0], centre[1] - half[1]:centre[1] + half[1],
                          centre[2] - half[2]:centre[2] + half[2]]
                break
            except Exception as e:  # transient fetch error
                print(f"  cutout {c['name']} d={d:+.0f} attempt {attempt + 1} failed: {str(e)[:80]}", flush=True)
                time.sleep(10)
        else:
            cache[(c["name"], d)] = {}
            continue
        sv, cnt = np.unique(np.asarray(cut).squeeze(), return_counts=True)
        keep = (sv != 0) & (cnt >= MIN_SV_VOX)
        cache[(c["name"], d)] = dict(zip(sv[keep].astype(np.int64).tolist(), cnt[keep].astype(int).tolist()))
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(todo)} cutouts, {time.time() - t0:.0f} s", flush=True)
            pickle.dump(cache, open(BOXES, "wb"))
    pickle.dump(cache, open(BOXES, "wb"))
    return cache


def map_roots(client, svs):
    """supervoxel -> current root, batched and throttled; transient CAVE errors retried"""
    svs = np.array(sorted(set(svs)), dtype=np.int64)
    out = {}
    t = time.time() - 10
    for i in range(0, len(svs), ROOT_BATCH):
        chunk = svs[i:i + ROOT_BATCH]
        for attempt in range(4):
            t = throttle(t)
            try:
                roots = np.asarray(client.chunkedgraph.get_roots(chunk)).astype(np.int64)
                break
            except Exception as e:
                print(f"  get_roots batch {i // ROOT_BATCH} attempt {attempt + 1} failed: {str(e)[:80]}", flush=True)
                time.sleep(15)
        else:
            raise RuntimeError("get_roots keeps failing")
        out.update(zip(chunk.tolist(), roots.tolist()))
        print(f"  roots for {min(i + ROOT_BATCH, len(svs))}/{len(svs)} supervoxels", flush=True)
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mode = args[0] if args else "both"
    refresh = "--refresh" in sys.argv
    known = pd.read_csv(os.path.join(OUT, "pr_lattice_known.tsv"), sep="\t", dtype={"r16_ids": str, "r78_ids": str})
    pred = pd.read_csv(os.path.join(OUT, "pr_lattice_predicted.tsv"), sep="\t")
    known_ids = set()
    for col in ("r16_ids", "r78_ids"):
        for s in known[col].fillna(""):
            known_ids |= {x for x in str(s).split(",") if x and x != "nan"}
    carts = []
    if mode in ("known", "both"):
        for r in known.itertuples():
            carts.append(dict(name=r.cartridge, kind="known", lam=np.array([r.lam_x, r.lam_y, r.lam_z]),
                              axis=np.array([r.ax_x, r.ax_y, r.ax_z]),
                              own={x for x in f"{r.r16_ids},{r.r78_ids}".split(",") if x and x != "nan"}))
    if mode in ("predicted", "both"):
        for r in pred.itertuples():
            carts.append(dict(name=f"p({r.lat_i},{r.lat_j})", kind="predicted", lam=np.array([r.lam_x, r.lam_y, r.lam_z]),
                              axis=np.array([r.ax_x, r.ax_y, r.ax_z]), own=set(), ring=r.ring, inside=r.inside_hull))
    now = datetime.datetime.now(datetime.timezone.utc)
    client = caveclient.CAVEclient(DATASTACK)
    vol = CloudVolume(client.info.segmentation_source(), use_https=True, agglomerate=False, progress=False,
                      fill_missing=True, bounded=False, mip=1)
    print(f"{len(carts)} cartridges x {len(DEPTHS_UM)} depths; box {2 * BOX_HALF_UM[0]} x {2 * BOX_HALF_UM[1]} x "
          f"{2 * BOX_HALF_UM[2]} um at {list(vol.resolution)} nm", flush=True)
    cache = {} if (refresh or not os.path.exists(BOXES)) else pickle.load(open(BOXES, "rb"))
    cache = read_boxes(vol, carts, cache)
    svs = {sv for c in carts for d in DEPTHS_UM for sv in cache[(c["name"], d)]}
    print(f"{len(svs)} supervoxels to map to roots", flush=True)
    sv2root = map_roots(client, svs)

    rows = []
    for c in carts:
        per_root = {}
        for d in DEPTHS_UM:
            acc = {}
            for sv, n in cache[(c["name"], d)].items():
                r = sv2root.get(sv, 0)
                if r:
                    acc[r] = acc.get(r, 0) + n
            for r, n in acc.items():
                if n >= MIN_HIT_VOX:
                    per_root.setdefault(r, {})[d] = n
        for r, dv in per_root.items():
            ds = sorted(dv)
            rows.append(dict(cart=c["name"], kind=c["kind"], root=str(r), n_depths=len(ds), vox_total=int(sum(dv.values())),
                             vox_min=int(min(dv.values())), vox_med=int(np.median(list(dv.values()))),
                             _depths=ds))
    agg = pd.DataFrame(rows)
    agg["class"] = agg._depths.map(classify)
    agg["beyond"] = agg._depths.map(lambda d: any(x <= -15.0 for x in d))
    agg["known_PR"] = agg.root.isin(known_ids)
    # PRs already typed in SeaTable (right side, R1-6 / R7-8; written by faab_pr_lattice.py) but not in a known cartridge
    st_path = os.path.join(OUT, "pr_lattice_seatable_prs.tsv")
    st = pd.read_csv(st_path, sep="\t", dtype=str) if os.path.exists(st_path) else pd.DataFrame(columns=["root", "type", "cart"])
    st_type = {r.root: f"SeaTable {r.type}" + (f" cartridge {r.cart}" if isinstance(r.cart, str) and r.cart not in ("", "nan") else "")
               for r in st.itertuples()}
    agg["seatable"] = agg.root.map(st_type).fillna("")
    agg["depths"] = agg._depths.map(lambda d: ",".join(f"{int(x):+d}" for x in d))
    agg = agg.drop(columns="_depths")
    agg.to_csv(os.path.join(OUT, "pr_probe_hits.tsv"), sep="\t", index=False)
    PRC = ["PR-like", "PR-axon-like"]

    # ---- calibration on the known cartridges ----
    if mode in ("known", "both"):
        print("\ncalibration on known cartridges (own PRs recovered; extra = PR-like/PR-axon-like roots that are not known PRs):")
        tot_rec, tot_own, tot_extra = 0, 0, 0
        for c in carts:
            if c["kind"] != "known":
                continue
            sub = agg[agg.cart == c["name"]]
            prlike = sub[sub["class"].isin(PRC)]
            rec = prlike[prlike.root.isin(c["own"])]
            anyhit = sub[sub.root.isin(c["own"])]
            extra = prlike[~prlike.root.isin(known_ids)]
            tot_rec += len(rec)
            tot_own += len(c["own"])
            tot_extra += len(extra)
            print(f"  {c['name']:4s}: own PRs {len(c['own'])}, touched {len(anyhit)}, recovered PR-like {len(rec)} "
                  f"(depths hit by own PRs: {sorted(set(sum([list(map(int, d.split(','))) for d in anyhit.depths], [])))}); "
                  f"other known PRs hit {int(sub.root.isin(known_ids).sum()) - len(anyhit)}; extra PR-like {len(extra)} "
                  f"(SeaTable-typed {int((extra.seatable != '').sum())}); lamina-only {int((sub['class'] == 'lamina-only').sum())}", flush=True)
        print(f"  overall: {tot_rec}/{tot_own} known PRs recovered as PR-like; {tot_extra} extra PR-like roots", flush=True)

    # ---- candidates at predicted sites ----
    cand = agg[(agg.kind == "predicted") & agg["class"].isin(PRC) & ~agg.known_PR].copy()
    if len(cand):
        meta = {c["name"]: c for c in carts if c["kind"] == "predicted"}
        cand["ring"] = cand.cart.map(lambda n: meta[n]["ring"])
        cand["inside_hull"] = cand.cart.map(lambda n: meta[n]["inside"])
        # a root touched from several neighbouring sites is attributed to the site where it has the most voxels
        cand["n_sites"] = cand.groupby("root").cart.transform("size")
        cand = cand.sort_values(["root", "vox_total", "ring"], ascending=[True, False, True]).drop_duplicates("root")
        cand = cand.sort_values(["cart", "class", "n_depths", "vox_total"], ascending=[True, True, False, False])
        cand.to_csv(os.path.join(OUT, "pr_probe_candidates.tsv"), sep="\t", index=False)
        known_hits = agg[(agg.kind == "predicted") & agg.known_PR].groupby("cart").root.nunique()
        summ = cand.groupby("cart").agg(n_cand=("root", "size"), full=("class", lambda s: int((s == "PR-like").sum())),
                                        frag=("class", lambda s: int((s == "PR-axon-like").sum())),
                                        seatable_typed=("seatable", lambda s: int((s != "").sum())),
                                        ring=("ring", "first"), inside=("inside_hull", "first"))
        summ["known_PRs_hit"] = summ.index.map(lambda n: int(known_hits.get(n, 0)))
        summ = summ.sort_values(["n_cand", "inside", "ring"], ascending=[False, False, True])
        pd.set_option("display.width", 220)
        print(f"\npredicted sites with PR-like candidates: {len(summ)} of {len(meta)}; candidate roots {cand.root.nunique()} "
              f"({int((cand.seatable != '').sum())} already typed as PRs in SeaTable)")
        print(summ.to_string())
        # link: EM first, brain mesh, known PRs grey, candidates by class, SeaTable-typed candidates blue, sites yellow
        new = cand[cand.seatable == ""]
        by_class = {k: sorted(set(v.root)) for k, v in new.groupby("class")}
        spec = {"meshes": [{"kind": "brain", "alpha": 0.15}],
                "neurons": [{"name": "known PRs (8 groups + LRS cartridges)", "color": "#808080", "ids": sorted(known_ids)},
                            {"name": "candidate PRs (axon + lamina)", "color": "#e6194b", "ids": by_class.get("PR-like", [])},
                            {"name": "candidate PR axon fragments", "color": "#ff9f1c", "ids": by_class.get("PR-axon-like", [])},
                            {"name": "SeaTable-typed PRs at predicted sites", "color": "#1f77d6", "ids": sorted(set(cand[cand.seatable != ""].root))}],
                "view": "anterior", "layout": "3d", "image": True, "scale": 12000,
                "position": [int(v) for v in (np.mean([c["lam"] for c in carts], axis=0) * 1000.0 / VOX)]}
        spec["neurons"] = [n for n in spec["neurons"] if n["ids"]]
        state = nglink.build(spec)
        state["layers"][0]["name"] = "EM"
        n_at = cand.groupby("cart").size()
        anns = [{"type": "point", "id": c["name"], "point": [int(v) for v in (c["lam"] * 1000.0 / VOX)],
                 "description": f"{c['name']} predicted site, ring {c['ring']}, {int(n_at.get(c['name'], 0))} candidate PRs"}
                for c in carts if c["kind"] == "predicted"]
        state["layers"].append({"type": "annotation", "source": {"url": "local://annotations", "transform": {"outputDimensions": DIMS}},
                                "annotations": anns, "name": "predicted cartridge sites", "annotationColor": "#ffe119"})
        url = nglink.to_url(state)
        open(os.path.join(OUT, "pr_probe_ng.txt"), "w").write(url + "\n")
        json.dump(state, open(os.path.join(OUT, "pr_probe_state.json"), "w"), indent=1)
        print(f"link {len(url)} chars saved to figures/pr_probe_ng.txt")
    print(f"live at {now:%Y-%m-%d %H:%M} UTC\nDONE")


if __name__ == "__main__":
    main()
