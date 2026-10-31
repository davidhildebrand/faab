"""Aedes optic-lobe and brain neuropil meshes (aedes_ol_meshes, Spelunker standard link of 2026-10-27) and where a
cell's cable lies in them.

Source: legacy precomputed meshes at SOURCE, segment ids LABELS (L = SeaTable left = larger x; R = smaller x). Meshes
are in the segmentation's nm space (no transform in the standard link). Each mesh is downloaded once (manifest
'<id>:0' + binary fragments) and cached in figures/ol_meshes/<id>.npz. HTTP only, no CAVE.

    from faab.ol_meshes import neuropil_fractions
    neuropil_fractions(points_um)            # {'LA_R': 0.0, 'ME_R': 0.61, 'LO_R': 0.37, ..., 'none': 0.02}
"""
import json
import os
import struct
import urllib.request

import numpy as np

from faab.config import FIGURES
from faab.config import fig

SOURCE = "https://flyemdev.mrc-lmb.cam.ac.uk/flyconnectome/rclone/aedes_ol_meshes/"
LABELS = {1: "CB", 2: "OL_L", 3: "OL_R", 11: "ME_L", 12: "LO_L", 13: "LOP_L", 14: "LA_L",
          21: "ME_R", 22: "LO_R", 23: "LOP_R", 24: "LA_R"}
IDS = {v: k for k, v in LABELS.items()}
# the four optic-lobe neuropils per side, and the coarse regions; a point is assigned to the first that contains it
FINE = ["LA_R", "ME_R", "LO_R", "LOP_R", "LA_L", "ME_L", "LO_L", "LOP_L"]
COARSE = ["OL_R", "OL_L", "CB"]
CACHE = fig("ol_meshes")
_vol = {}


def _get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def mesh(label):
    """(vertices nm, faces) of one neuropil, cached"""
    sid = IDS[label]
    path = os.path.join(CACHE, f"{sid}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["v"], z["f"]
    frags = json.loads(_get(f"{SOURCE}{sid}:0"))["fragments"]
    vs, fs, off = [], [], 0
    for name in frags:
        b = _get(SOURCE + name)
        n = struct.unpack("<I", b[:4])[0]
        v = np.frombuffer(b, dtype="<f4", count=3 * n, offset=4).reshape(n, 3)
        f = np.frombuffer(b, dtype="<u4", offset=4 + 12 * n).reshape(-1, 3)
        vs.append(v.astype(np.float64))
        fs.append(f.astype(np.int64) + off)
        off += n
    v, f = np.vstack(vs), np.vstack(fs)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, v=v, f=f)
    return v, f


def volume(label):
    """navis.Volume (um) of one neuropil"""
    import navis
    if label not in _vol:
        v, f = mesh(label)
        _vol[label] = navis.Volume(v / 1000.0, f, name=label, units="um")
    return _vol[label]


def assign(points_um, labels=None):
    """label of the region holding each point: the first of FINE then COARSE that contains it, else 'none'"""
    import navis
    pts = np.asarray(points_um, float)
    out = np.array(["none"] * len(pts), dtype=object)
    todo = np.ones(len(pts), bool)
    for lab in labels or FINE + COARSE:
        if not todo.any():
            break
        inside = np.asarray(navis.in_volume(pts[todo], volume(lab)), bool)
        idx = np.where(todo)[0][inside]
        out[idx] = lab
        todo[idx] = False
    return out


def neuropil_fractions(points_um, labels=None):
    """fraction of the points in each region (FINE, then COARSE for points outside the fine neuropils), plus 'none'"""
    a = assign(points_um, labels)
    n = max(len(a), 1)
    return {lab: float(np.sum(a == lab)) / n for lab in (labels or FINE + COARSE) + ["none"]}


def summary(fr, min_frac=0.05):
    """'ME_R 61% / LO_R 37%' from a fractions dict (regions >= min_frac, largest first)"""
    items = sorted(((k, v) for k, v in fr.items() if v >= min_frac), key=lambda kv: -kv[1])
    return " / ".join(f"{k} {100 * v:.0f}%" for k, v in items)
