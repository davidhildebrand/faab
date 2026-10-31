"""CALCULATE how many segmented cell objects the fitted medulla sheet intersects.
Fit the curved sheet through the known Mi1, sample it on a 2.5um grid over the
region the Mi1 cover, read the segment ID at each point, count unique roots.
Dataset: aedes. Read-only.
"""
from faab.config import REPO
import os
import time
import pickle
import numpy as np
import cloudvolume as cv
from scipy.spatial import cKDTree, ConvexHull, Delaunay
from caveclient import CAVEclient
from faab.clients.cave import DATASTACK, throttle
from faab.config import fig

OUT = os.path.join(REPO, "figures")
GRID_UM = 2.5


def main():
    dp = pickle.load(open(fig("mi1_dotprops.pkl"), "rb"))
    C = np.array([d.points.mean(0) for d in dp])          # Mi1 centroids (um)
    known = set(int(d.id) for d in dp)

    # PCA frame: u,v in-plane, w out-of-plane
    c0 = C.mean(0)
    _, _, V = np.linalg.svd(C - c0, full_matrices=False)
    uvw = (C - c0) @ V.T
    uv, w = uvw[:, :2], uvw[:, 2]

    # grid over (u,v) hull; w by k-NN interpolation (follows curved sheet)
    tree = cKDTree(uv)
    hull = Delaunay(uv[ConvexHull(uv).vertices])
    us = np.arange(uv[:, 0].min(), uv[:, 0].max(), GRID_UM)
    vs = np.arange(uv[:, 1].min(), uv[:, 1].max(), GRID_UM)
    gu, gv = np.meshgrid(us, vs)
    G = np.c_[gu.ravel(), gv.ravel()]
    G = G[hull.find_simplex(G) >= 0]                       # inside medulla region only
    dists, idx = tree.query(G, k=6)
    gw = w[idx].mean(1)
    pts_um = c0 + np.c_[G, gw] @ V                         # back to 3D (um)
    pts_nm = pts_um * 1000.0
    print(f"sheet area sampled: {len(pts_um)} points at {GRID_UM}um "
          f"(~{len(pts_um)*GRID_UM**2:.0f} um^2)")

    # read supervoxel at each point (cloudvolume, cached), then batch get_roots
    client = CAVEclient(DATASTACK)
    vol = cv.CloudVolume(client.info.segmentation_source(), use_https=True,
                         progress=False, fill_missing=True, cache=True)
    res = np.array(vol.resolution)
    vox = np.round(pts_nm / res).astype(int)
    t0 = time.time()
    svs = np.zeros(len(vox), dtype=np.uint64)
    for i, (x, y, z) in enumerate(vox):
        try:
            svs[i] = np.asarray(vol[x, y, z]).ravel()[0]
        except Exception:
            svs[i] = 0
        if i % 500 == 0:
            print(f"  read {i}/{len(vox)}  ({time.time()-t0:.0f}s)")
    uniq_sv = np.unique(svs[svs > 0])
    print(f"read done ({time.time()-t0:.0f}s): {len(uniq_sv)} unique supervoxels")

    # map supervoxels -> roots (batched)
    roots = np.array(client.chunkedgraph.get_roots(uniq_sv.tolist()), dtype=np.uint64)
    uniq_roots = set(int(r) for r in roots if r != 0)
    pickle.dump({"pts_nm": pts_nm, "svs": svs, "roots": sorted(uniq_roots)},
                open(fig("mi1_plane_cells.pkl"), "wb"))

    print(f"\n=== CELLS INTERSECTING THE Mi1 SHEET: {len(uniq_roots)} ===")
    print(f"    (from {len(pts_um)} surface samples; sheet area ~{len(pts_um)*GRID_UM**2/1e3:.0f}x10^3 um^2)")
    print(f"    known Mi1 recovered: {len(known & uniq_roots)}/{len(known)}")


if __name__ == "__main__":
    main()
