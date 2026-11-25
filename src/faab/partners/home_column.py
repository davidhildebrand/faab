"""The PR group 1 home column: one line per right optic-lobe neuropil, and distances from it in columns (built for
scripts/partners/t4t5_sort.py, David 2026-11-11/12: "proximity to a line/axis connecting the PRg1 terminals and the
T4/T5"; shared since 2026-11-25 with scripts/sheets/home_column_sort.py, which sorts more tabs by it, renumbers the
PR groups and draws the line in a link).

Lines (synapse positions, um):
  ME   from the median PR+1 -> PR+2 synapse (distal medulla) to the median input synapse of the PR+3 targets in ME
       (the T4 dendrites, proximal); ~23 deg from the medulla's thickness direction (2026-11-11 test).
  LO   through the median input synapse of the targets in LO (T5 dendrites), along the lobula's thickness.
  LOP  through the median output synapse of the targets in LOP (their axon terminals), along the lobula plate's thickness.
  LA   through the median PR group 1 -> PR+1 synapse, along the lamina's thickness.
Thickness = from the nearest point on the face toward the neighbor neuropil (LA: ME; ME: LA; LO: ME; LOP: LO) to the
nearest point on the opposite face. That mesh rule is only the FALLBACK since 2026-11-25: for the lobula it ran along
the T5 dendrite layer, 87 deg off the columns (T5 dendrite sites within 1 column of it spread over 28 um in depth
instead of ~10). refine() now sets the LA / LO / LOP directions from the layer itself: the direction through the anchor
along which the layer's synapses near the line (within 2 columns) sit at the most uniform depth (smallest p25-p75
spread; a column crosses a layer at one depth): LA = photoreceptor output sites, LO = T5 dendrite (input) sites,
LOP = T4/T5 terminal (output) sites. The ME line keeps its two path anchors (8 deg from its layer optimum).
Distance of a point = its perpendicular distance from its neuropil's line, in
column spacings: ME 5.5 um (nearest-neighbor spacing of the R7 terminals of 36 PR groups, 2026-11-11), the other
neuropils scaled by sqrt(face area / ME face area) (the same number of columns everywhere).
Network-free (the synapse tables come from the caller)."""
import glob
import json

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from faab import ol_meshes
from faab.config import fig

VOX = np.array([0.016, 0.016, 0.045])  # um per voxel
LABELS = ["LA_R", "ME_R", "LO_R", "LOP_R"]
TOWARD = {"LA_R": "ME_R", "ME_R": "LA_R", "LO_R": "ME_R", "LOP_R": "LO_R"}
ME_COL_UM = 5.5
X = ["x", "y", "z"]
_faces = {}


def faces(label):
    """(outer points, inner points, outer face area um^2): the face toward TOWARD[label] and the opposite one"""
    if label in _faces:
        return _faces[label]
    v, f = ol_meshes.mesh(label)
    m = trimesh.Trimesh(v / 1000.0, f, process=True)
    m.fix_normals()
    ov = ol_meshes.mesh(TOWARD[label])[0] / 1000.0
    to = ov[cKDTree(ov).query(m.triangles_center)[1]] - m.triangles_center
    sg = np.einsum("ij,ij->i", m.face_normals, to / np.linalg.norm(to, axis=1, keepdims=True))
    pts, fid = trimesh.sample.sample_surface(m, 200000, seed=0)
    _faces[label] = (pts[np.isin(fid, np.where(sg > 0.2)[0])], pts[np.isin(fid, np.where(sg < -0.2)[0])],
                     float(m.area_faces[sg > 0.2].sum()))
    return _faces[label]


def thickness_dir(outer, inner, p):
    o = outer[cKDTree(outer).query(p)[1]]
    i = inner[cKDTree(inner).query(p)[1]]
    return (i - o) / np.linalg.norm(i - o)


def build(ch_la, ch_me, tin, tout, min_n=10):
    """lines {label: (origin um, unit direction)} and col_um {label: column spacing um} from four synapse tables
    with x/y/z (um) and np (neuropil label): PR group 1 -> PR+1, PR+1 -> PR+2, inputs of the PR+3 targets (post
    sites), outputs of the targets (pre sites)"""
    med = lambda df, L: df[df.np == L][X].median().to_numpy() if int((df.np == L).sum()) >= min_n else None  # noqa: E731
    fc = {L: faces(L) for L in LABELS}
    lines = {}
    D, P = med(ch_me, "ME_R"), med(tin, "ME_R")
    if D is not None and P is not None:
        lines["ME_R"] = (D, (P - D) / np.linalg.norm(P - D))
    for L, anchor in (("LO_R", med(tin, "LO_R")), ("LOP_R", med(tout, "LOP_R")), ("LA_R", med(ch_la, "LA_R"))):
        if anchor is not None:
            lines[L] = (anchor, thickness_dir(fc[L][0], fc[L][1], anchor))
    col_um = {L: ME_COL_UM * np.sqrt(fc[L][2] / fc["ME_R"][2]) for L in LABELS}
    return lines, col_um


def _fib(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - i / n)  # one hemisphere
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


def depth_spread(P, o, d, col, k=2.0, min_n=50):
    """(p25-p75 of the depth t = (p - o).d of the points within k columns of the line o + t d, number of such points)"""
    q = P - o
    t = q @ d
    sel = t[np.linalg.norm(q - np.outer(t, d), axis=1) <= k * col]
    if len(sel) < min_n:
        return np.inf, len(sel)
    p25, p75 = np.percentile(sel, [25, 75])
    return float(p75 - p25), len(sel)


def layer_direction(P, o, n0, col, k=2.0, radius=80.0, max_pts=40000, seed=0):
    """the direction through o that crosses the layer of the points P (um) most squarely: the smallest depth spread
    (depth_spread) over a hemisphere grid, then a finer grid around the best; sign as n0. Returns (direction, spread,
    spread along n0, sites used) or None when too few points"""
    P = np.asarray(P, float)
    P = P[np.linalg.norm(P - o, axis=1) <= radius]
    if len(P) > max_pts:
        P = P[np.random.default_rng(seed).choice(len(P), max_pts, replace=False)]
    if len(P) < 200:
        return None
    best = min(((depth_spread(P, o, d, col, k)[0], i) for i, d in enumerate(_fib(4000))), key=lambda x: x[0])
    d0 = _fib(4000)[best[1]]
    fine = np.r_[_fib(800), -_fib(800)] * 0.08 + d0
    fine /= np.linalg.norm(fine, axis=1, keepdims=True)
    cand = [(best[0], d0)] + [(depth_spread(P, o, d, col, k)[0], d) for d in fine]
    sp, d = min(cand, key=lambda x: x[0])
    if not np.isfinite(sp):
        return None
    d = d if np.dot(d, n0) >= 0 else -d
    return d, sp, depth_spread(P, o, n0, col, k)[0], depth_spread(P, o, d, col, k)[1]


def refine(lines, col_um, layer_sites, labels=("LA_R", "LO_R", "LOP_R")):
    """lines with the LA / LO / LOP directions taken from their layers (layer_sites: {label: points um}); returns
    (lines, report {label: dict})"""
    out, rep = dict(lines), {}
    for L in labels:
        if L not in lines or L not in layer_sites:
            continue
        o, n = lines[L]
        res = layer_direction(layer_sites[L], o, n, col_um[L])
        if res is None:
            rep[L] = dict(method="mesh thickness (too few layer sites)")
            continue
        d, sp, sp0, used = res
        ang = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(d, n)))))))
        out[L] = (o, d)
        rep[L] = dict(method="layer (smallest depth spread)", spread_um=round(sp, 1), mesh_spread_um=round(sp0, 1),
                      angle_to_mesh_deg=round(ang), sites=int(used))
    return out, rep


def distance(points_um, labels, lines, col_um=None):
    """perpendicular distance of each point from the line of its neuropil (labels per point), in columns (in um
    when col_um is None); NaN outside the four neuropils / without a line"""
    Q = np.asarray(points_um, float).reshape(-1, 3)
    labels = np.asarray(labels, dtype=object)
    d = np.full(len(Q), np.nan)
    for L, (o, n) in lines.items():
        i = np.where(labels == L)[0]
        if len(i):
            q = Q[i] - o
            d[i] = np.linalg.norm(q - np.outer(q @ n, n), axis=1) / (col_um[L] if col_um else 1.0)
    return d


def dist_cols(df, lines, col_um):
    """per-synapse distance in columns for a table with x/y/z (um) and np"""
    if df is None or not len(df):
        return np.full(0 if df is None else len(df), np.nan)
    return distance(df[X].to_numpy(float), df.np.to_numpy(), lines, col_um)


def extent(label, o, n, reach=250.0, step=0.5):
    """end points (um) of the stretch of the line o + t n that lies inside the neuropil around o (the longest inside
    run, or the one nearest o)"""
    t = np.arange(-reach, reach + step, step)
    P = o + np.outer(t, n)
    inside = ol_meshes.assign(P, [label]) == label
    if not inside.any():
        return o, o
    runs, start = [], None
    for k, v in enumerate(inside):
        if v and start is None:
            start = k
        if (not v or k == len(inside) - 1) and start is not None:
            runs.append((start, k if v else k - 1))
            start = None
    i0 = int(np.argmin(np.abs(t)))
    a, b = min(runs, key=lambda r: (0 if r[0] <= i0 <= r[1] else 1, min(abs(r[0] - i0), abs(r[1] - i0))))
    return P[a], P[b]


def save(path, lines, col_um, **meta):
    out = {"lines": {L: {"origin_um": np.round(o, 3).tolist(), "direction": np.round(n, 6).tolist(),
                         "column_um": round(float(col_um[L]), 3)} for L, (o, n) in lines.items()},
           "column_um": {L: round(float(v), 3) for L, v in col_um.items()}, **meta}
    json.dump(out, open(path, "w"), indent=1)
    return path


def load(path=None):
    """(lines, col_um, meta) from a saved JSON (default: the newest figures/t4t5/*_t4t5_home_column_lines.json)"""
    path = path or newest()
    d = json.load(open(path))
    lines = {L: (np.array(v["origin_um"]), np.array(v["direction"])) for L, v in d["lines"].items()}
    col_um = {L: float(v) for L, v in d["column_um"].items()}
    meta = {k: v for k, v in d.items() if k not in ("lines", "column_um")}
    meta["path"] = path
    return lines, col_um, meta


def newest():
    hits = sorted(glob.glob(fig("*_t4t5_home_column_lines.json")))
    if not hits:
        raise FileNotFoundError("no figures/t4t5/*_t4t5_home_column_lines.json; run home_column_sort.py lines BOOK")
    return hits[-1]


def path_for(stamp):
    return fig(f"{stamp}_t4t5_home_column_lines.json")


__all__ = ["VOX", "LABELS", "ME_COL_UM", "faces", "thickness_dir", "build", "refine", "layer_direction",
           "depth_spread", "distance", "dist_cols", "extent", "save", "load", "newest", "path_for"]
