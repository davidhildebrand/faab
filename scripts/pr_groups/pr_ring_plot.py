"""Aligned cross-section of ONE photoreceptor bundle, in a single self-contained script.

Requirements: Python 3.10+, numpy, pandas, matplotlib, caveclient, fafbseg, cloud-volume, and a CAVE token with read
access to the FAAB datastack wclee_aedes_brain (caveclient's default token file). Nothing else.

Steps
 1. Members, one of:
      --roots  root ids (no files needed; stale roots are replaced by their latest root, with a warning);
      --xyz    points on the axons (16 x 16 x 45 nm voxels, as in the sheet); current roots are looked up at the points;
      --group  a group from ONE groups file (default data/pr_groups.tsv in the repo; --groups-file to point elsewhere),
               tidy TSV with at least the columns group, class, xyz (root_id_curr is used only when xyz is empty).
    The script imports nothing from the faab package and reads no other file, so it can be copied anywhere.
 2. Frame: L2 skeletons (one call) and output synapses (one live_query). Lamina point = median presynaptic site of the
    members. Bundle axis = PCA of the skeleton nodes inside a 15-um cylinder running from the lamina point toward the
    retina (toward the members' xyz points, or without points toward the median of the skeleton nodes more than 40 um
    from the lamina point); s = position along the axis, 0 at the lamina point, negative toward the retina. Box
    centers follow the skeletons (median of the members' mean node positions within 1.5 um of each s).
 3. Tracking: 6 x 6 x 6 um mip-1 cutouts every 3 um from s = -120 to -20 um; all supervoxels mapped to roots (one
    get_roots call per 50k); per slice and member the voxel-weighted centroid, projected onto the plane across the axis.
 4. Compact slices: all members present, median distance from the slice center <= 1.6 um, maximum <= 2.5 um. If fewer
    than 4 slices qualify, the member most often absent (then farthest out) is dropped and the step repeated.
 5. Roles and names: R8 = the member running closest to the center (median <= 0.7 um). R7 = --r7, else the row labeled
    R7 in the table, else the member whose output synapses lie farthest from the lamina point (> 25 um and > 10 um
    beyond any other). With R7 and six outer axons the fly pairs come from the ring (validated on groups 1 and 4):
    going around the ring from R7, the two cells flanking R7 are R1/R6, the next two R2/R5 (opposite each other), the
    far two R3/R4.
 6. Plot: every compact slice rotated so that R7 points up, overlaid as small dots; per-cell means as large markers;
    dotted trapezoid R1/R6-R3/R4-R3/R4-R1/R6; the symmetry axis through R7 and the center.

Outputs <out>/pr_ring_plot_<tag>.png, .tsv (per-slice centroids), .json (frame, members, query time) and _names.tsv
(per member: xyz, table class, ring role, distance from the center, angle vs R7, flank margin); <out> is the
repo's figures/ folder, or --out. --replot redraws from those files without any CAVE call. Read-only; CAVE calls are
spaced >= 5 s apart.

Usage
  python pr_ring_plot.py --roots 648518347590648323 648518347579284782 ... --r7 648518347500901740 --tag mybundle
  python pr_ring_plot.py --group 1
  python pr_ring_plot.py --group 23 --mirror           (mirror left-right, e.g. to compare eyes)
  python pr_ring_plot.py --xyz "8735, 18856, 6245" "8179, 19104, 6016" ... --r7 "8783, 18851, 6218" --tag mine
  python pr_ring_plot.py --group 1 --replot
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

DATASTACK = "wclee_aedes_brain"
VOX = np.array([16.0, 16.0, 45.0])  # nm per sheet voxel
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/pr_groups/ -> repo root
GROUPS_FILE = os.path.join(REPO, "data", "pr_groups.tsv")
IN_REPO = os.path.exists(GROUPS_FILE)  # run from the faab repo; a copy elsewhere writes to the current folder
OUT = os.path.join(REPO, "figures") if IN_REPO and os.path.isdir(os.path.join(REPO, "figures")) else os.getcwd()
BOX_UM, STEP_UM, S_FROM, S_TO = 6.0, 3.0, -120.0, -20.0
MED_MAX, RING_MAX, R8_MAX = 1.6, 2.5, 0.7
COL = {"R1/R6": "#D85A30", "R2/R5": "#7F77DD", "R3/R4": "#1D9E75", "R7": "#A32D2D", "R8": "#0F6E56", "outer": "#888780"}

_last = [0.0]


def throttle(gap=5.0):
    wait = _last[0] + gap - time.time()
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.time()


# ------------------------------------------------------------------------------------------------------------ members
def load_members(group, path):
    """members of one group from a tidy groups TSV: [dict(xyz, root, label)]"""
    d = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    sub = d[d.group.str.strip() == str(group)]
    if not len(sub):
        sys.exit(f"group {group} not in {path}")
    return [dict(xyz=r.get("xyz", ""), root=r.get("root_id_curr", ""), label=r.get("class", "").replace("–", "-"))
            for _, r in sub.iterrows()]


def xyz_um(s):
    return np.array([float(v) for v in s.split(",")]) * VOX / 1000.0


def positions(df, prefix):
    col = f"{prefix}_pt_position"
    if col in df.columns:
        return np.vstack([np.asarray(v, float) for v in df[col]]) if len(df) else np.zeros((0, 3))
    return df[[f"{col}_x", f"{col}_y", f"{col}_z"]].to_numpy(float)


# ---------------------------------------------------------------------------------------------- CAVE part (fetch)
def fetch(members, tag, r7_arg):
    import caveclient
    import navis
    from cloudvolume import CloudVolume
    from fafbseg import flywire
    from scipy import ndimage
    navis.set_pbars(hide=True)
    client = caveclient.CAVEclient(DATASTACK)
    if r7_arg:  # --r7 adds or marks the R7 (xyz point or root id)
        key = "xyz" if "," in r7_arg else "root"
        hit = [m for m in members if m[key].replace(" ", "") == r7_arg.replace(" ", "")]
        if hit:
            hit[0]["label"] = "R7"
        else:
            members = members + [dict(xyz=r7_arg if key == "xyz" else "", root=r7_arg if key == "root" else "", label="R7")]

    # 1. current roots: looked up at the xyz anchors; members given by root only are checked for staleness
    pts = [m for m in members if m["xyz"].strip()]
    if pts:
        throttle()
        found = np.asarray(flywire.locs_to_segments(np.array([xyz_um(m["xyz"]) * 1000.0 for m in pts]), coordinates="nm",
                                                    backend="cloudvolume", dataset=DATASTACK)).astype(np.int64)
        for m, r in zip(pts, found):
            if r == 0:
                print(f"  no segment at {m['xyz']}: falls back to its root id", flush=True)
            else:
                if m["root"] and m["root"] != str(int(r)):
                    print(f"  root at {m['xyz']} is now {r} (listed {m['root']})", flush=True)
                m["root"] = str(int(r))
    ids = [int(m["root"]) for m in members if m["root"].strip().isdigit()]
    throttle()
    latest = dict(zip(ids, client.chunkedgraph.is_latest_roots(ids)))
    for m in members:
        r = int(m["root"]) if m["root"].strip().isdigit() else 0
        if r and not latest.get(r, True):
            throttle()
            new = [int(x) for x in client.chunkedgraph.get_latest_roots(r)]
            print(f"  root {r} is stale; latest: {new}" + (" (several: using the first; check)" if len(new) > 1 else ""),
                  flush=True)
            m["root"] = str(new[0]) if new else ""
    mem, seen = [], set()
    for m in members:
        if not m["root"].strip().isdigit():
            print(f"  member {m} has no usable root: skipped", flush=True)
        elif m["root"] in seen:
            print(f"  {m['xyz'] or m['root']} resolves to {m['root']}, already a member (merged pieces?): skipped", flush=True)
        else:
            seen.add(m["root"])
            mem.append(dict(root=m["root"], xyz=m["xyz"].strip(), label=m["label"]))
    print(f"{len(mem)} members with current roots", flush=True)
    rlist = [int(m["root"]) for m in mem]

    # 2. skeletons and output synapses -> frame
    throttle()
    nodes = {}
    for sk in flywire.get_l2_skeleton(rlist, dataset=DATASTACK, max_threads=1, omit_failures=True, progress=False):
        try:
            sk = navis.resample_skeleton(sk, resample_to=1000, inplace=False)
        except Exception:
            pass
        nodes[str(int(sk.id))] = sk.nodes[["x", "y", "z"]].to_numpy(float) / 1000.0
    missing = [m["root"] for m in mem if m["root"] not in nodes]
    if missing:
        print(f"  no skeleton for {missing}", flush=True)
    now = datetime.datetime.now(datetime.timezone.utc)
    throttle()
    syn = client.materialize.live_query("synapses_v2", timestamp=now, filter_in_dict={"pre_pt_root_id": rlist})
    syn = syn[syn.pre_pt_root_id != syn.post_pt_root_id]
    P = positions(syn, "pre")
    ref = np.vstack(list(nodes.values())).mean(0)
    cand = {"nm": P / 1000.0, "voxels": P * VOX / 1000.0}
    unit = min(cand, key=lambda k: np.linalg.norm(np.median(cand[k], 0) - ref)) if len(P) else "nm"
    P = cand[unit]
    pre_root = syn.pre_pt_root_id.astype(np.int64).astype(str).to_numpy()
    if not len(P):
        sys.exit("no output synapses: the lamina side cannot be set")
    lam = np.median(P, 0)
    pts = np.vstack(list(nodes.values()))
    anchors = [xyz_um(m["xyz"]) for m in mem if m["xyz"]]
    if len(anchors) >= 3:  # the sheet points sit on the axons, retina side of the lamina
        d0 = np.median(anchors, 0) - lam
    else:  # no points: most skeleton length far from the lamina is the bundle running to the retina
        far = pts[np.linalg.norm(pts - lam, axis=1) > 40]
        d0 = np.median(far, 0) - lam
    d0 /= np.linalg.norm(d0)  # toward the retina
    q = pts - lam
    s0 = q @ d0
    lat = np.linalg.norm(q - np.outer(s0, d0), axis=1)
    sel = (s0 > 5) & (s0 < 150) & (lat < 15)
    axis = np.linalg.svd(pts[sel] - pts[sel].mean(0), full_matrices=False)[2][0]
    if axis @ d0 > 0:
        axis = -axis  # +s toward the lamina
    u = np.cross(axis, [0, 0, 1.0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    for m in mem:
        pr = P[pre_root == m["root"]]
        m["n_pre"] = int(len(pr))
        m["pre_dist_um"] = float(np.linalg.norm(np.median(pr, 0) - lam)) if len(pr) >= 3 else None
    print(f"lamina point from {len(P)} output synapses ({unit}); axis from {int(sel.sum())} skeleton nodes", flush=True)

    # box centers along the bundle
    centers = {}
    for s in np.arange(S_FROM, S_TO + 1e-6, STEP_UM):
        per = []
        for p in nodes.values():
            qq = p - lam
            sp = qq @ axis
            m = (np.abs(sp - s) <= 1.5) & (np.linalg.norm(qq - np.outer(sp, axis), axis=1) <= 20)
            if m.sum():
                per.append(p[m].mean(0))
        if len(per) >= 3:
            centers[float(s)] = np.median(per, 0)

    # 3. cutouts -> supervoxel centroids -> roots
    vol = CloudVolume(client.info.segmentation_source(), use_https=True, agglomerate=False, progress=False,
                      fill_missing=True, bounded=False, mip=1)
    res = np.array(vol.resolution, float)
    half = np.round(BOX_UM * 500.0 / res).astype(int)
    boxes = {}
    t0 = time.time()
    for k, (s, ctr) in enumerate(sorted(centers.items())):
        c = np.round(ctr * 1000.0 / res).astype(int)
        lo = c - half
        for attempt in range(3):
            try:
                arr = np.asarray(vol[lo[0]:c[0] + half[0], lo[1]:c[1] + half[1], lo[2]:c[2] + half[2]]).squeeze()
                break
            except Exception as e:
                print(f"  cutout at s = {s:.0f} failed ({str(e)[:60]}), retrying", flush=True)
                time.sleep(10)
        else:
            continue
        sv, cnt = np.unique(arr, return_counts=True)
        keep = (sv != 0) & (cnt >= 20)
        sv, cnt = sv[keep], cnt[keep]
        com = ndimage.center_of_mass(np.ones(arr.shape, np.uint8), arr, index=sv) if len(sv) else []
        boxes[s] = {int(a): (int(n), (lo + np.array(m)) * res / 1000.0) for a, n, m in zip(sv, cnt, com)}
        if k % 10 == 9:
            print(f"  {k + 1}/{len(centers)} cutouts, {time.time() - t0:.0f} s", flush=True)
    svs = sorted({a for b in boxes.values() for a in b})
    sv2root = {}
    for i in range(0, len(svs), 50000):
        throttle()
        rt = np.asarray(client.chunkedgraph.get_roots(svs[i:i + 50000])).astype(np.int64)
        sv2root.update(zip(svs[i:i + 50000], rt.astype(str).tolist()))
    want = {m["root"] for m in mem}
    rows = []
    for s, b in boxes.items():
        acc = {}
        for a, (n, p) in b.items():
            r = sv2root.get(a)
            if r in want:
                acc.setdefault(r, []).append((n, p))
        for r, lst in acc.items():
            w = np.array([n for n, _ in lst], float)
            p = np.sum([n * pp for n, pp in lst], axis=0) / w.sum()
            rows.append(dict(s=s, root=r, vox=int(w.sum()), pu=float((p - lam) @ u), pv=float((p - lam) @ v)))
    df = pd.DataFrame(rows)
    meta = dict(tag=tag, queried=f"{now:%Y-%m-%d %H:%M} UTC", synapse_unit=unit, lam=lam.tolist(), axis=axis.tolist(),
                u=u.tolist(), v=v.tolist(), members=mem, box_um=BOX_UM, step_um=STEP_UM)
    df.to_csv(os.path.join(OUT, f"pr_ring_plot_{tag}.tsv"), sep="\t", index=False)
    json.dump(meta, open(os.path.join(OUT, f"pr_ring_plot_{tag}.json"), "w"), indent=1)
    print(f"{len(boxes)} cutouts, {len(svs)} supervoxels, {len(df)} member-slices ({time.time() - t0:.0f} s)", flush=True)
    return df, meta


# ----------------------------------------------------------------------------------------- analysis (network-free)
def compact_slices(df, members):
    sub = df[df.root.isin(members)]
    full = sub.groupby("s").root.nunique()
    sub = sub[sub.s.isin(full[full == len(members)].index)]
    if not len(sub):
        return sub, []
    cen = sub.groupby("s")[["pu", "pv"]].median()
    sub = sub.join(cen, on="s", rsuffix="_c")
    sub = sub.assign(du=sub.pu - sub.pu_c, dv=sub.pv - sub.pv_c)
    sub["r"] = np.hypot(sub.du, sub.dv)
    st = sub.groupby("s").r.agg(["median", "max"])
    keep = list(st[(st["median"] <= MED_MAX) & (st["max"] <= RING_MAX)].index)
    return sub[sub.s.isin(keep)], keep


def circ_mean_deg(a):
    a = np.radians(np.asarray(a, float))
    return float(np.degrees(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))))


def analyse(df, meta, r7_root=None):
    mem = {m["root"]: m for m in meta["members"]}
    members = [r for r in mem if r in set(df.root)]
    dropped = [(r, "never found in the cutouts") for r in mem if r not in members]
    while True:
        sub, comp = compact_slices(df, members)
        if len(comp) >= 4 or len(members) <= 4:
            break
        d = df[df.root.isin(members)]
        busy = d.groupby("s").root.nunique()
        d = d[d.s.isin(busy[busy >= max(3, len(members) // 2)].index)]
        cen = d.groupby("s")[["pu", "pv"]].median()
        d = d.join(cen, on="s", rsuffix="_c")
        d["r"] = np.hypot(d.pu - d.pu_c, d.pv - d.pv_c)
        stat = d.groupby("root").agg(n=("s", "nunique"), r=("r", "median")).reindex(members).fillna({"n": 0, "r": 99})
        worst = stat.sort_values(["n", "r"], ascending=[True, False]).index[0]
        dropped.append((worst, f"present in {int(stat.loc[worst, 'n'])} slices, median {stat.loc[worst, 'r']:.1f} um out"))
        members.remove(worst)
    if len(comp) < 3:
        sys.exit(f"only {len(comp)} compact slices; dropped {dropped}")
    med_r = sub.groupby("root").r.median()
    # roles
    r8 = med_r.idxmin()
    r8 = r8 if med_r[r8] <= R8_MAX else None
    if r7_root and r7_root in members:
        r7, r7_how = r7_root, "given with --r7"
    else:
        lab7 = [r for r in members if mem[r]["label"] == "R7" and r != r8]
        if lab7:
            r7, r7_how = lab7[0], "labeled R7 in the table"
        else:
            dist = {r: mem[r]["pre_dist_um"] for r in members if r != r8 and mem[r].get("pre_dist_um") is not None}
            srt = sorted(dist, key=dist.get, reverse=True)
            ok = srt and dist[srt[0]] > 25 and (len(srt) < 2 or dist[srt[0]] > dist[srt[1]] + 10)
            r7, r7_how = (srt[0], f"outputs {dist[srt[0]]:.0f} um from the lamina point") if ok else (None, "not found")
    outer = [r for r in members if r not in (r7, r8)]
    # per-slice angles; rotation so that R7 points up (or the first outer axon, when there is no R7)
    sub = sub.assign(ang=np.degrees(np.arctan2(sub.dv, sub.du)))
    ref = r7 if r7 else outer[0]
    ref_ang = sub[sub.root == ref].set_index("s").ang
    sub["rel"] = ((sub.ang - sub.s.map(ref_ang)) + 180.0) % 360.0 - 180.0
    rot = np.radians(90.0 - sub.s.map(ref_ang))
    sub["x"] = sub.du * np.cos(rot) - sub.dv * np.sin(rot)
    sub["y"] = sub.du * np.sin(rot) + sub.dv * np.cos(rot)
    rel = {r: circ_mean_deg(sub[sub.root == r].rel) for r in outer}
    names, margin, ring = {}, None, []
    if r7 and len(outer) == 6:
        order = sorted(outer, key=lambda r: rel[r])
        k = next((j for j, r in enumerate(order) if rel[r] > 0), 0)
        ring = order[k:] + order[:k]  # counterclockwise from R7
        for i, nm in ((0, "R1/R6"), (5, "R1/R6"), (1, "R2/R5"), (4, "R2/R5"), (2, "R3/R4"), (3, "R3/R4")):
            names[ring[i]] = nm
        margin = min(rel[ring[0]] % 360.0, (-rel[ring[5]]) % 360.0)
    for r in members:
        names.setdefault(r, "R7" if r == r7 else "R8" if r == r8 else "outer")
    return dict(sub=sub, comp=comp, members=members, dropped=dropped, r7=r7, r7_how=r7_how, r8=r8, outer=outer,
                names=names, rel=rel, ring=ring, margin=margin, med_r=med_r)


# ---------------------------------------------------------------------------------------------------------- plot
def plot(A, meta, title, mirror=False):
    sub, names = A["sub"], A["names"]
    sx = -1.0 if mirror else 1.0
    means = {r: (sx * sub[sub.root == r].x.mean(), sub[sub.root == r].y.mean()) for r in A["members"]}
    fig, ax = plt.subplots(figsize=(6.6, 7.4))
    for r in A["members"]:
        s = sub[sub.root == r]
        ax.scatter(sx * s.x, s.y, s=10, color=COL[names[r]], alpha=0.35, zorder=2)
    if A["ring"]:
        c = A["ring"]
        ax.add_patch(Polygon([means[c[0]], means[c[2]], means[c[3]], means[c[5]]], closed=True, fill=False,
                             ls=(0, (1, 3)), lw=1.0, ec="#777", zorder=1))
    for r in A["members"]:
        mk = "D" if names[r] in ("R7", "R8") else "o"
        ax.scatter(*means[r], s=260, marker=mk, color=COL[names[r]], edgecolor="k", zorder=5)
        mx, my = means[r]
        d = np.hypot(mx, my)
        ox, oy = (mx / d, my / d) if d > 0.35 else (0.7, -0.7)  # label pushed radially outward (central R8: lower right)
        ax.annotate(f"{names[r]}\n...{r[-6:]}", means[r], xytext=(26 * ox, 26 * oy), textcoords="offset points",
                    fontsize=8, ha="center", va="center", zorder=6)
    if A["r7"]:
        ax.axvline(0, color="#378ADD", lw=1.2, ls="--", zorder=0)
    ax.axhline(0, color="#ccc", lw=0.6, zorder=0)
    lim = max(2.6, max(max(abs(x), abs(y)) for x, y in means.values()) + 0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("um (rotated so that R7 points up)" + (", mirrored" if mirror else ""))
    ax.set_ylabel("um")
    comp = A["comp"]
    ax.set_title(f"{title}: {len(comp)} compact slices overlaid\n(s = {min(comp):.0f} to {max(comp):.0f} um from the lamina point)",
                 fontsize=10)
    lines = [f"synapses and roots: {meta['queried']}",
             f"R7: {A['r7_how']}" + (f"; flanking cells {A['margin']:.0f} deg from R7" if A["margin"] is not None else ""),
             f"R8: {'central, ' + format(A['med_r'][A['r8']], '.2f') + ' um from the center' if A['r8'] else 'no axon within 0.7 um of the center'}"]
    if A["dropped"]:
        lines.append("left out: " + "; ".join(f"...{r[-6:]} ({why})" for r, why in A["dropped"]))
    if not A["ring"]:
        lines.append("fly names need R7 and six outer axons" + ("" if A["r7"] else "; orientation arbitrary (no R7)"))
    fig.text(0.02, 0.01, "\n".join(lines), fontsize=7.5, va="bottom", color="#444")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    png = os.path.join(OUT, f"pr_ring_plot_{meta['tag']}.png")
    fig.savefig(png, dpi=150)
    return png


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--roots", nargs="+", help="member root ids")
    ap.add_argument("--xyz", nargs="+", help="member xyz points (16 x 16 x 45 nm voxels), e.g. '8735, 18856, 6245'")
    ap.add_argument("--group", type=int, help="a group from the groups file")
    ap.add_argument("--groups-file", default=GROUPS_FILE, help=f"tidy TSV with group, class, xyz (default {GROUPS_FILE})")
    ap.add_argument("--r7", help="R7 as an xyz point or a root id (added to the members if it is not one)")
    ap.add_argument("--tag", help="output name (default g<group>, else 'bundle')")
    ap.add_argument("--out", help="output folder (default: the repo's figures/ folder)")
    ap.add_argument("--replot", action="store_true", help="redraw from the saved .tsv/.json (no CAVE calls)")
    ap.add_argument("--mirror", action="store_true", help="mirror left-right")
    a = ap.parse_args()
    if sum(x is not None for x in (a.roots, a.xyz, a.group)) != 1:
        ap.error("give exactly one of --roots, --xyz, --group")
    OUT = a.out or OUT
    os.makedirs(OUT, exist_ok=True)
    tag = a.tag or (f"g{a.group}" if a.group is not None else "bundle")
    title = f"group {a.group}" if a.group is not None else tag
    if a.replot:
        df = pd.read_csv(os.path.join(OUT, f"pr_ring_plot_{tag}.tsv"), sep="\t", dtype={"root": str})
        meta = json.load(open(os.path.join(OUT, f"pr_ring_plot_{tag}.json")))
    else:
        if a.group is not None:
            members = load_members(a.group, a.groups_file)
        elif a.roots:
            members = [dict(xyz="", root=str(r), label="") for r in a.roots]
        else:
            members = [dict(xyz=x, root="", label="") for x in a.xyz]
        print(f"{title}: {len(members)} members", flush=True)
        df, meta = fetch(members, tag, a.r7)
    r7_root = a.r7 if a.r7 and "," not in a.r7 else None
    if r7_root and r7_root not in {m["root"] for m in meta["members"]}:  # a stale --r7 root was replaced in fetch()
        r7_root = next((m["root"] for m in meta["members"] if m["label"] == "R7"), None)
    A = analyse(df, meta, r7_root)
    print(f"\n{len(A['comp'])} compact slices; R7 {A['r7_how']}; R8 {'central' if A['r8'] else 'none'}")
    print("root                 role    median r (um)  angle vs R7 (deg)")
    for r in sorted(A["members"], key=lambda r: A["rel"].get(r, -999 if r == A["r7"] else 999)):
        ang = A["rel"].get(r)
        print(f"{r}   {A['names'][r]:6s}  {A['med_r'][r]:5.2f}          {'' if ang is None else format(ang, '+.0f')}")
    for r, why in A["dropped"]:
        print(f"{r}   left out: {why}")
    # names table: one row per member (xyz as given, so results map back to sheet rows)
    rows = []
    for m in meta["members"]:
        r = m["root"]
        rows.append(dict(root=r, xyz=m.get("xyz", ""), label=m.get("label", ""),
                         role=A["names"].get(r, "left out"),
                         median_r_um=round(float(A["med_r"][r]), 2) if r in A["med_r"] else "",
                         angle_vs_r7=round(A["rel"][r]) if r in A["rel"] else "",
                         r7_how=A["r7_how"], flank_margin_deg=round(A["margin"]) if A["margin"] is not None else "",
                         left_out=dict(A["dropped"]).get(r, ""), queried=meta["queried"]))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"pr_ring_plot_{tag}_names.tsv"), sep="\t", index=False)
    print("saved", plot(A, meta, title, a.mirror))


if __name__ == "__main__":
    main()
