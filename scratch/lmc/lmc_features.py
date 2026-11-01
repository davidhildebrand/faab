"""Per-LMC position features in the lamina (LA) and medulla (ME) of the cell's own side, for the L1-L5 typing
(David, 2026-10-31: 'extension distance from mesh edge may be a useful predictor'). Network-free.

Depth in a neuropil: 0 at its outer face (ME: the face toward the lamina; LA: the face away from the medulla, toward the
retina), 1 at its inner face: d_outer / (d_outer + d_inner), with distances to the two face sets of the mesh (faces
sorted by whether their outward normal points toward the other neuropil). Points outside a neuropil get how far they
reach past its faces. Cable = L2 skeleton resampled to 1 um; synapses = local snapshot positions (lmc_data.py).
Writes figures/lmc/lmc_features.tsv. Usage: lmc_features.py"""
import pickle

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree

from faab import ol_meshes
from faab.config import fig

BINS = np.linspace(0, 1, 6)  # five depth bins


def faces_split(np_label, other_label, n_samples=200000):
    """(label, outer, inner) of a neuropil: KD-trees of points sampled densely (~1 um apart) on its outer and inner
    faces (outer = faces turned away from the other neuropil for LA, toward it for ME), in um. Sampled points stand in
    for exact closest-point queries, which need rtree (not installed)."""
    v, f = ol_meshes.mesh(np_label)
    m = trimesh.Trimesh(v / 1000.0, f, process=True)
    m.fix_normals()
    ov = ol_meshes.mesh(other_label)[0] / 1000.0  # nearest point of the other neuropil (the lamina is cup-shaped,
    to_other = ov[cKDTree(ov).query(m.triangles_center)[1]] - m.triangles_center  # so its centroid misleads)
    s = np.einsum("ij,ij->i", m.face_normals, to_other / np.linalg.norm(to_other, axis=1, keepdims=True))
    toward, away = np.where(s > 0.2)[0], np.where(s < -0.2)[0]
    outer, inner = (toward, away) if np_label.startswith("ME") else (away, toward)
    pts, fid = trimesh.sample.sample_surface(m, n_samples, seed=0)
    return (np_label, cKDTree(pts[np.isin(fid, outer)]), cKDTree(pts[np.isin(fid, inner)]))


def depth(P, M):
    """depth 0..1 between the outer and inner faces, the distances to each face, and whether inside"""
    label, outer, inner = M
    if not len(P):
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, bool)
    d_o, d_i = outer.query(P)[0], inner.query(P)[0]
    inside = ol_meshes.assign(P, [label]) == label
    return d_o / np.maximum(d_o + d_i, 1e-6), d_o, d_i, inside


def hist(x):
    h = np.histogram(np.clip(x, 0, 1), BINS)[0].astype(float)
    return h / max(h.sum(), 1)


def main():
    import navis
    navis.set_pbars(hide=True)
    cells = pd.read_csv(fig("lmc_cells.tsv"), sep="\t", dtype=str).fillna("")
    sk = pickle.load(open(fig("lmc_skeletons.pkl"), "rb"))
    syn = pd.read_parquet(fig("lmc_synapses.parquet"))
    mesh = {}
    for side in ("R", "L"):
        mesh[("LA", side)] = faces_split(f"LA_{side}", f"ME_{side}")
        mesh[("ME", side)] = faces_split(f"ME_{side}", f"LA_{side}")
    rows = []
    for c in cells.itertuples(index=False):
        if c.root not in sk:
            continue
        side = c.side if c.side in ("L", "R") else "R"
        s1 = navis.resample_skeleton(sk[c.root], resample_to=1)
        P = s1.nodes[["x", "y", "z"]].to_numpy(float)
        f = dict(root=c.root, serial_id=c.serial_id, side=c.side, label=c.label, cable_um=round(len(P)))
        for npl in ("LA", "ME"):
            dz, d_o, d_i, ins = depth(P, mesh[(npl, side)])
            f[f"cable_{npl}"] = int(ins.sum())
            z = dz[ins]
            f[f"{npl}_depth_min"] = float(z.min()) if len(z) else np.nan
            f[f"{npl}_depth_max"] = float(z.max()) if len(z) else np.nan
            f[f"{npl}_depth_med"] = float(np.median(z)) if len(z) else np.nan
            f[f"{npl}_max_um_from_outer"] = float(d_o[ins].max()) if ins.any() else np.nan
            out_side = ~ins & (d_o < d_i)
            f[f"{npl}_beyond_outer_um"] = float(d_o[out_side].max()) if out_side.any() else 0.0
            for k, v in enumerate(hist(z)):
                f[f"{npl}_cable_bin{k}"] = v
            if npl == "LA" and ins.sum() >= 5:  # lateral spread around the cell's lamina axis
                Q = P[ins] - P[ins].mean(0)
                ax = np.linalg.svd(Q, full_matrices=False)[2][0]
                rad = np.linalg.norm(Q - np.outer(Q @ ax, ax), axis=1)
                f["LA_spread_p90_um"], f["LA_spread_max_um"] = float(np.percentile(rad, 90)), float(rad.max())
            for role in ("in", "out"):
                q = syn[(syn.lmc == c.root) & (syn.role == role)][["x", "y", "z"]].to_numpy(float)
                zq, _, _, iq = depth(q, mesh[(npl, side)]) if len(q) else (np.zeros(0),) * 3 + (np.zeros(0, bool),)
                zz = zq[iq] if len(q) else np.zeros(0)
                f[f"n_{role}_{npl}"] = int(len(zz))
                f[f"{npl}_{role}_depth_med"] = float(np.median(zz)) if len(zz) else np.nan
                if npl == "ME" and role == "out":
                    for k, v in enumerate(hist(zz)):
                        f[f"ME_out_bin{k}"] = v
                    f["ME_out_depth_p10"] = float(np.percentile(zz, 10)) if len(zz) else np.nan
                    f["ME_out_depth_p90"] = float(np.percentile(zz, 90)) if len(zz) else np.nan
        f["n_in"], f["n_out"] = int((syn.lmc == c.root).sum() - (syn[(syn.lmc == c.root)].role == "out").sum()), \
            int(((syn.lmc == c.root) & (syn.role == "out")).sum())
        rows.append(f)
        if len(rows) % 20 == 0:
            print(f"  {len(rows)} cells", flush=True)
    F = pd.DataFrame(rows)
    F.to_csv(fig("lmc_features.tsv"), sep="\t", index=False)
    show = ["cable_LA", "cable_ME", "ME_depth_min", "ME_depth_max", "ME_max_um_from_outer", "LA_beyond_outer_um", "LA_spread_p90_um",
            "n_in_LA", "n_out_LA", "n_out_ME", "ME_out_depth_med", "ME_out_depth_p90", "LA_depth_min", "LA_depth_max"]
    print(F.groupby("label")[show].median().round(2).T.to_string())
    print("written", fig("lmc_features.tsv"))


if __name__ == "__main__":
    main()
