"""Feasibility of a surface-restricted Mi1 search: fit the medulla sheet through
the known Mi1 centroids, measure sheet thickness (candidate tolerance), and test
whether Mi1 principal axes align with the local surface normal (orientation prior).
Local (cached dotprops). Dataset: aedes.
"""
from faab.config import REPO
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from faab.config import fig

OUT = os.path.join(REPO, "figures")


def main():
    dp = pickle.load(open(fig("mi1_dotprops.pkl"), "rb"))
    C = np.array([d.points.mean(0) for d in dp])            # centroids (um)
    # principal axis (PC1) of each cell's points = columnar/depth axis
    A = []
    for d in dp:
        p = d.points - d.points.mean(0)
        A.append(np.linalg.svd(p, full_matrices=False)[2][0])
    A = np.array(A)
    n = len(C)

    # global plane fit
    c0 = C.mean(0)
    gnormal = np.linalg.svd(C - c0, full_matrices=False)[2][2]      # smallest-variance dir
    gres = np.abs((C - c0) @ gnormal)
    print(f"Mi1 centroids: {n}")
    print(f"global PLANE fit: RMS residual = {np.sqrt((gres**2).mean()):.1f} um (max {gres.max():.0f})")

    # local surface: per-cell plane fit to k nearest neighbors -> local normal + residual
    tree = cKDTree(C)
    k = 12
    local_res, orient_ang, normals = [], [], []
    for i in range(n):
        _, idx = tree.query(C[i], k=min(k, n))
        nb = C[idx]
        nrm = np.linalg.svd(nb - nb.mean(0), full_matrices=False)[2][2]
        local_res.append(abs((C[i] - nb.mean(0)) @ nrm))
        ang = np.degrees(np.arccos(min(1, abs(A[i] @ nrm))))       # PC1 vs local normal (0-90)
        orient_ang.append(ang)
        normals.append(nrm)
    local_res, orient_ang = np.array(local_res), np.array(orient_ang)
    print(f"local CURVED fit (k={k} nbrs): RMS residual = {np.sqrt((local_res**2).mean()):.1f} um "
          f"(median {np.median(local_res):.1f}) -> SHEET THICKNESS")
    print(f"curvature: global-plane RMS {np.sqrt((gres**2).mean()):.0f}um >> local RMS "
          f"{np.sqrt((local_res**2).mean()):.0f}um  => medulla is CURVED (need local/curved surface, not one plane)")
    print(f"orientation: Mi1 PC1 vs local surface normal = median {np.median(orient_ang):.0f} deg "
          f"(<{np.percentile(orient_ang,75):.0f} for 75%)")

    # figure: 3D centroids colored by local residual + short normal vectors
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(121, projection="3d")
    s = ax.scatter(C[:, 0], C[:, 1], C[:, 2], c=local_res, cmap="viridis", s=25)
    fig.colorbar(s, ax=ax, shrink=0.6, label="dist to local sheet (um)")
    ax.set_title("Mi1 centroids on the medulla sheet")
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(C[:, 0], C[:, 1], C[:, 2], c="gray", s=8)
    N = np.array(normals)
    for i in range(0, n, 3):
        seg = np.vstack([C[i], C[i] + A[i] * 12])
        ax2.plot(seg[:, 0], seg[:, 1], seg[:, 2], c="firebrick", lw=0.8)
    ax2.set_title("Mi1 principal axes (red) ~ local sheet normal")
    fig.tight_layout(); fig.savefig(fig("mi1_surface.png"), dpi=120)
    print("\nsaved mi1_surface.png")


if __name__ == "__main__":
    main()
