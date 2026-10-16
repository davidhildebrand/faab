"""Does a cell have a soma? Two read-only checks for one root id (argument, default below):
  1. live: nuclei_v1_aedes entries assigned to the root (position printed in nm and in 16x16x45 voxels);
  2. network-free: the thickest L2 chunks of the cell from figures/l2_chunkdata.pkl (max_dt_nm = radius of
     the largest inscribed sphere; a cell body gives a chunk with max_dt of a few um, neurites ~0.1-0.5 um),
     plus the same from the cached skeleton radii if the root is not in the chunk cache.
Usage: python faab_cell_soma_check.py <root_id>
"""
from faab.config import REPO
import os
import pickle
import sys
import time

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import caveclient
from faab.sources.cave import DATASTACK
from faab.partners.top10_xyz import VOX

HERE = REPO
OUT = os.path.join(HERE, "figures")
ROOT = int(sys.argv[1]) if len(sys.argv) > 1 else 648518347624865802


def vox(p_nm):
    return ", ".join(str(int(round(v))) for v in (np.asarray(p_nm, float) / VOX))


def main():
    leaves = pickle.load(open(os.path.join(OUT, "l2_leaves.pkl"), "rb")) if os.path.exists(os.path.join(OUT, "l2_leaves.pkl")) else {}
    chunks = pickle.load(open(os.path.join(OUT, "l2_chunkdata.pkl"), "rb")) if os.path.exists(os.path.join(OUT, "l2_chunkdata.pkl")) else {}
    ids = leaves.get(ROOT)
    if ids:
        rows = []
        for i in ids:
            c = chunks.get(i) or {}
            if c.get("max_dt_nm") is not None:
                rows.append((i, c["max_dt_nm"], c.get("mean_dt_nm"), c.get("size_nm3"), c.get("rep_coord_nm")))
        df = pd.DataFrame(rows, columns=["l2", "max_dt_nm", "mean_dt_nm", "size_nm3", "rep"]).sort_values("max_dt_nm", ascending=False)
        print(f"root {ROOT}: {len(ids)} L2 chunks (cached), max_dt_nm max {df.max_dt_nm.max():.0f}, "
              f"median {df.max_dt_nm.median():.0f}; total volume {df.size_nm3.sum() / 1e9:.1f} um3")
        print("thickest chunks (max_dt is cut by chunk borders, so a soma does NOT show as one thick chunk):")
        for _, r in df.head(4).iterrows():
            print(f"  max_dt {r.max_dt_nm:6.0f} nm  mean_dt {r.mean_dt_nm:5.0f} nm  size {r.size_nm3 / 1e9:6.2f} um3  "
                  f"at voxel {vox(r.rep) if r.rep is not None else '?'}")
        # local-volume test (2026-09-20, after a soma was missed): a cell body is a compact cluster of chunks,
        # ~50+ um3 within a 4 um radius; a 1 um-radius trunk holds only ~25 um3 in the same ball
        has = df[df.rep.notna()].copy()
        pts = np.vstack(has.rep.values).astype(float) / 1000.0
        sz = has.size_nm3.values.astype(float) / 1e9
        R_UM, SOMA_UM3 = 4.0, 50.0
        local = np.array([sz[np.linalg.norm(pts - p, axis=1) <= R_UM].sum() for p in pts])
        order = np.argsort(-local)
        print(f"local volume within {R_UM:.0f} um (soma candidate if >= {SOMA_UM3:.0f} um3): max {local.max():.1f} um3")
        used = np.zeros(len(pts), bool)
        n_cand = 0
        for k in order:
            if used[k]:
                continue
            near = np.linalg.norm(pts - pts[k], axis=1) <= R_UM
            used |= near
            if local[k] < SOMA_UM3 or n_cand >= 3:
                break
            n_cand += 1
            cen = np.average(pts[near], axis=0, weights=sz[near]) * 1000.0
            print(f"  SOMA CANDIDATE {n_cand}: {local[k]:.1f} um3 in {int(near.sum())} chunks, centre at voxel {vox(cen)}")
        if n_cand == 0:
            print("  no compact cluster of that size: no soma candidate from chunk volumes")
    else:
        sk = pickle.load(open(os.path.join(OUT, "pr_partner_skels.pkl"), "rb"))
        s = sk.get(ROOT)
        if s is None:
            print(f"root {ROOT}: not in the L2 chunk cache nor the skeleton cache")
        else:
            nodes = s.nodes.sort_values("radius", ascending=False)
            print(f"root {ROOT}: skeleton with {len(nodes)} nodes, radius max {nodes.radius.max():.0f} nm")
            for _, r in nodes.head(6).iterrows():
                print(f"  radius {r.radius:6.0f} nm at voxel {vox([r.x, r.y, r.z])}")

    client = caveclient.CAVEclient(DATASTACK)
    time.sleep(5)
    nuc = client.materialize.query_table("nuclei_v1_aedes", filter_in_dict={"pt_root_id": [ROOT]})
    print(f"\nnuclei_v1_aedes: {len(nuc)} nucleus/nuclei assigned to root {ROOT} (live)")
    for _, r in nuc.iterrows():
        p = np.array([r.pt_position_x, r.pt_position_y, r.pt_position_z], float) if "pt_position_x" in nuc.columns \
            else np.array(r.pt_position, float)
        extra = {k: r[k] for k in nuc.columns if k in ("volume", "id", "valid")}
        print(f"  nucleus at nm {p.astype(int).tolist()} = voxel {vox(p)}  {extra}")
    print("DONE")


if __name__ == "__main__":
    main()
