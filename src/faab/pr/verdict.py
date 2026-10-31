"""Synapse verdict of a photoreceptor-like axon: R1-R6-like / R8-like / R7-like / fragment / other.

Calibrated 2026-10-29 on the right-side PR classes of data/pr_groups.tsv (calibration, scores and history in
scripts/pr_groups/bundles_verify.py). Lamina share = fraction of output (presynaptic) sites inside the side's lamina
mesh; medulla share = inside the medulla mesh or outside both meshes within ME_MARGIN_UM of its surface (nearest mesh
vertex; the medulla mesh is tight)."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from faab import ol_meshes

PRE_MIN, POST_MAX, POST_MAX_R7 = 40, 100, 150
# right-side lamina share p10: R1-R6 0.86, R8 0.62; R7 p75 0.56. Medulla share (5 um margin) p90: R1-R6 and R8 0.0
LA_MIN, ME_R7, R8_IN_PER_OUT, ME_MARGIN_UM = 0.6, 0.1, 0.2, 5.0


def mesh_shares(roots, points_um, la, me):
    """per root: la_frac, me_frac of its output points (arrays aligned: one root id per point, points in um)"""
    if not len(points_um):
        return pd.DataFrame(columns=["la_frac", "me_frac"])
    P = np.asarray(points_um, float)
    region = ol_meshes.assign(P, [la, me])
    near_me = cKDTree(ol_meshes.mesh(me)[0] / 1000.0).query(P)[0] <= ME_MARGIN_UM
    region = np.where((region == "none") & near_me, me, region)
    t = pd.crosstab(np.asarray(roots), region, normalize="index")
    return pd.DataFrame({"la_frac": t[la] if la in t else 0.0, "me_frac": t[me] if me in t else 0.0}, index=t.index).round(2)


def verdict(n_pre, n_post, la_frac, me_frac):
    if n_pre < 10:
        return "fragment"
    if la_frac >= LA_MIN and me_frac < ME_R7 and n_post <= POST_MAX:
        if n_post / n_pre >= R8_IN_PER_OUT:
            return "R8-like"
        return "R1-R6-like" if n_pre >= PRE_MIN else "R1-R6-like (few sites)"
    if (la_frac < LA_MIN or me_frac >= ME_R7) and n_post <= POST_MAX_R7:
        return "R7-like"
    return "other"
