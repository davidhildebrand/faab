"""xyz anchor points for new partner rows: on the primary neurite ~10 um from the soma (nucleus from nuclei_v1_aedes),
else the longest-neurite midpoint, else the L2 rep coord of a single-chunk fragment. Read-only, throttled; skeletons
cached in figures/pr_partner_skels.pkl."""
import os
import pickle

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

from faab.config import DATASTACK, FIGURES, VOX
from faab.partners.sheet import fmt_vox
from faab.partners.top10_xyz import backbone_midpoint_nm, primary_neurite_point_nm
from faab.sources.cave import throttle

SCACHE = os.path.join(FIGURES, "pr_partner_skels.pkl")


def cell_points(roots, client, t, batch=40):
    """{root: (xyz voxel string, how)} for every root; returns (points, t)"""
    from fafbseg import flywire
    roots = sorted({int(r) for r in roots})
    if not roots:
        return {}, t
    sk = pickle.load(open(SCACHE, "rb")) if os.path.exists(SCACHE) else {}
    need = [r for r in roots if r not in sk]
    for b in range(0, len(need), batch):
        t = throttle(t)
        try:
            for s in flywire.get_l2_skeleton(need[b:b + batch], dataset=DATASTACK, max_threads=1, omit_failures=True,
                                             progress=False):
                sk[int(s.id)] = s
        except Exception as e:
            print(f"  skeleton batch {b} failed: {str(e)[:80]}", flush=True)
        pickle.dump(sk, open(SCACHE, "wb"))
        print(f"  skeletons {min(b + batch, len(need))}/{len(need)}", flush=True)
    soma = {}
    t = throttle(t)
    try:
        nuc = client.materialize.query_table("nuclei_v1_aedes", filter_in_dict={"pt_root_id": roots})
        for _, r in nuc.iterrows():
            soma[int(r.pt_root_id)] = (np.array([r.pt_position_x, r.pt_position_y, r.pt_position_z], float)
                                       if "pt_position_x" in nuc.columns else np.array(r.pt_position, float))
    except Exception as e:
        print("  nuclei query failed:", str(e)[:80], flush=True)
    out = {}
    for r in roots:
        s = sk.get(r)
        if s is None:
            try:
                t = throttle(t)
                l2 = [int(x) for x in np.atleast_1d(client.chunkedgraph.get_leaves(r, stop_layer=2))]
                d = client.l2cache.get_l2data(l2, attributes=["rep_coord_nm"])
                pts = np.array([np.array(v["rep_coord_nm"], float) for v in d.values() if "rep_coord_nm" in v])
                out[r] = (fmt_vox(pts.mean(0) / np.array(VOX)), "single L2 chunk fragment -> L2 rep coord")
            except Exception:
                out[r] = ("", "no skeleton, no L2 coord")
            continue
        pt, geo = (None, None)
        if r in soma and s.n_nodes >= 5:
            pt, geo = primary_neurite_point_nm(s, soma[r])
        if pt is not None:
            how = f"primary neurite ~{geo:.0f} um from soma"
        else:
            pt = backbone_midpoint_nm(s)
            how = "no nucleus -> backbone midpoint" if r not in soma else "trunk too short -> backbone midpoint"
        out[r] = (fmt_vox(np.asarray(pt, float) / np.array(VOX)), how)
    return out, t
