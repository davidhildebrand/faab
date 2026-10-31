"""Shape metrics for separating glia from neurons: volume relative to spatial extent.

Idea (user, 2026-09-20): glia fill relatively large volumes but stay local, so volume per unit
of reach (or per unit of convex-hull volume) should be high for glia and low for neurons.

Stage 1 (this file, network-free): metrics from the cached L2 skeletons
(figures/pr_partner_skels.pkl, nm; radius = max inscribed radius of the L2 chunk):
    cable_um     cable length
    vol_um3      tubular volume proxy  sum(pi * r^2 * edge length)
    L1/L2/L3     extent along the principal axes of the (volume-weighted) node cloud, um
    rg_um        radius of gyration, um
    hull         convex-hull volume of the node cloud, um^3
derived: vol_per_L1 (um^2), fill = vol/hull, cable_per_L1, xsec = vol/cable (um^2), cable_per_hull.
Evaluated on the glia-vs-neuron comparison set of figures/glia_metrics.tsv (median, min-max,
best single threshold, accuracy and balanced accuracy) and applied to the partner cells of
figures/partner_glia_flags.tsv. Output: figures/glia_shape_skel.tsv. Stage 2 with exact L2
volumes: faab_glia_l2_fetch.py (imports the helpers below).
"""
from faab.config import REPO
import os
import pickle

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from faab.config import fig

HERE = REPO
OUT = os.path.join(HERE, "figures")
SCACHE = fig("pr_partner_skels.pkl")
GM = fig("glia_metrics.tsv")
PF = fig("partner_glia_flags.tsv")
MIN_CABLE = 100.0
PRE_CUT = 10.6
SHAPE = ["vol_per_L1", "fill", "cable_per_L1", "xsec", "cable_per_hull", "L1_um", "rg_um", "vol_um3"]
NAMES = {"vol_per_L1": "volume / longest extent (um^2)", "fill": "volume / convex hull volume",
         "cable_per_L1": "cable / longest extent", "xsec": "volume / cable = mean cross-section (um^2)",
         "cable_per_hull": "cable / convex hull volume (um^-2)", "L1_um": "longest extent (um)",
         "rg_um": "radius of gyration (um)", "vol_um3": "volume (um^3)",
         "area_per_vol": "surface area / volume (um^-1)", "thick_um": "mean thickness, size-weighted mean_dt (um)",
         "pre_per_100um": "presynaptic sites / 100 um cable"}


def extent_metrics(pts, w=None):
    """pts (n,3) in um, w weights for centroid/PCA. Ranges along principal axes, rg, hull volume."""
    pts = np.asarray(pts, float)
    n = len(pts)
    if n < 2:
        return dict(L1=np.nan, L2=np.nan, L3=np.nan, rg=np.nan, hull=np.nan)
    w = np.ones(n) if w is None else np.asarray(w, float)
    w = np.where(np.isfinite(w) & (w > 0), w, 1e-9)
    c = np.average(pts, axis=0, weights=w)
    d = pts - c
    cov = (d * w[:, None]).T @ d / w.sum()
    vals, vecs = np.linalg.eigh(cov)
    proj = d @ vecs[:, ::-1]
    rng = proj.max(0) - proj.min(0)
    rg = float(np.sqrt(np.average((d ** 2).sum(1), weights=w)))
    hull = np.nan
    if n >= 4:
        try:
            hull = float(ConvexHull(pts).volume)
        except (QhullError, ValueError):
            hull = np.nan
    return dict(L1=float(rng[0]), L2=float(rng[1]), L3=float(rng[2]), rg=rg, hull=hull)


def skeleton_metrics(s):
    """metrics from a cached L2 TreeNeuron (coordinates and radius in nm)"""
    nodes = s.nodes
    xyz = nodes[["x", "y", "z"]].values.astype(float) / 1000.0
    r = nodes["radius"].values.astype(float) / 1000.0
    idx = {nid: i for i, nid in enumerate(nodes.node_id.values)}
    has_par = nodes.parent_id.values >= 0
    child = np.array([idx[n] for n in nodes.node_id.values[has_par]], dtype=int)
    par = np.array([idx[p] for p in nodes.parent_id.values[has_par]], dtype=int)
    seg = np.linalg.norm(xyz[child] - xyz[par], axis=1)
    rbar = 0.5 * (r[child] + r[par])
    ev = np.pi * rbar ** 2 * seg
    wnode = np.zeros(len(xyz))
    np.add.at(wnode, child, 0.5 * ev)
    np.add.at(wnode, par, 0.5 * ev)
    out = dict(n_nodes=len(xyz), cable_um=float(seg.sum()), vol_um3=float(ev.sum()))
    out.update(extent_metrics(xyz, w=wnode + 1e-9))
    return out


def derived(df):
    df = df.copy()
    df["vol_per_L1"] = df.vol_um3 / df.L1
    df["fill"] = df.vol_um3 / df.hull
    df["cable_per_L1"] = df.cable_um / df.L1
    df["xsec"] = df.vol_um3 / df.cable_um
    df["cable_per_hull"] = df.cable_um / df.hull
    df["L1_um"] = df.L1
    df["rg_um"] = df.rg
    return df


def best_cut(x, is_glia):
    """best single threshold in either direction; returns rule text, accuracy, balanced accuracy"""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, g = x[ok], np.asarray(is_glia, bool)[ok]
    if len(x) < 4 or g.all() or (~g).all():
        return "n/a", np.nan, np.nan
    u = np.unique(x)
    cuts = (u[:-1] + u[1:]) / 2
    best = ("n/a", -1, -1)
    for c in cuts:
        for name, pred in (("<", x < c), (">", x > c)):
            acc = np.mean(pred == g)
            bal = 0.5 * (np.mean(pred[g]) + np.mean(~pred[~g]))
            if bal > best[2] + 1e-12:
                best = (f"glia if {name} {c:.3g}", acc, bal)
    return best


def fmt(v):
    if not np.isfinite(v):
        return "nan"
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.3g}"


def evaluate(df, metrics, title):
    """df needs columns group ('glia'/'neuron') and the metrics; prints and returns a summary table"""
    g = df.group.values == "glia"
    rows = []
    for m in metrics:
        x = df[m].values.astype(float)
        gv, nv = x[g & np.isfinite(x)], x[~g & np.isfinite(x)]
        rule, acc, bal = best_cut(x, g)
        rows.append([NAMES.get(m, m), f"{fmt(np.median(gv))} ({fmt(gv.min())}-{fmt(gv.max())})",
                     f"{fmt(np.median(nv))} ({fmt(nv.min())}-{fmt(nv.max())})", rule,
                     f"{acc * 100:.0f}%", f"{bal * 100:.0f}%", len(gv), len(nv)])
    out = pd.DataFrame(rows, columns=["metric", "glia med (min-max)", "neurons med (min-max)", "best rule",
                                      "acc", "bal acc", "n glia", "n neur"])
    print(f"\n{title}")
    print(out.to_string(index=False))
    return out


def logistic2(df, m1, m2):
    """2-feature logistic regression (log10 features, standardised, tiny ridge) via scipy; in-sample scores"""
    from scipy.optimize import minimize
    ok = np.isfinite(df[m1].values.astype(float)) & np.isfinite(df[m2].values.astype(float))
    X = np.column_stack([np.log10(df[m1].values.astype(float)[ok] + 0.1), np.log10(df[m2].values.astype(float)[ok] + 1e-6)])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.column_stack([np.ones(ok.sum()), (X - mu) / sd])
    y = (df.group.values[ok] == "glia").astype(float)

    def nll(b):
        z = Z @ b
        return np.sum(np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y)) + 1e-3 * np.sum(b[1:] ** 2)

    b = minimize(nll, np.zeros(3), method="BFGS").x
    p = 1 / (1 + np.exp(-(Z @ b)))
    pred = p > 0.5
    g = y > 0.5
    acc = np.mean(pred == g)
    bal = 0.5 * (np.mean(pred[g]) + np.mean(~pred[~g]))
    return b, mu, sd, acc, bal, ok, p


def load_sets():
    gm = pd.read_csv(GM, sep="\t", dtype={"root": str})
    pf = pd.read_csv(PF, sep="\t", dtype={"root_id_curr": str})
    return gm, pf


def main():
    sk = pickle.load(open(SCACHE, "rb"))
    gm, pf = load_sets()
    rows = []
    for _, r in gm.iterrows():
        s = sk.get(int(r.root))
        d = dict(set="comparison", group=r.group, source=r.source, label=r.label, root=str(r.root),
                 n_pre=r.n_pre, n_post=r.n_post, pre_per_100um=r.pre_per_100um)
        if s is not None:
            d.update(skeleton_metrics(s))
        rows.append(d)
    for _, r in pf.iterrows():
        s = sk.get(int(r.root_id_curr))
        d = dict(set="partner", group="", source="", label=r.flag, root=str(r.root_id_curr), n_pre=r.n_pre,
                 n_post=r.n_post, pre_per_100um=r.pre_per_100um, notes=r.notes)
        if s is not None:
            d.update(skeleton_metrics(s))
        rows.append(d)
    df = derived(pd.DataFrame(rows))
    df.to_csv(fig("glia_shape_skel.tsv"), sep="\t", index=False)

    comp = df[(df.set == "comparison") & df.cable_um.notna()]
    print(f"comparison cells with a skeleton: {len(comp)} of {len(gm)} "
          f"({(comp.group == 'glia').sum()} glia, {(comp.group == 'neuron').sum()} neurons)")
    judge = comp[comp.cable_um >= MIN_CABLE]
    print(f"with cable >= {MIN_CABLE:.0f} um: {len(judge)} ({(judge.group == 'glia').sum()} glia, "
          f"{(judge.group == 'neuron').sum()} neurons)")
    evaluate(judge, ["pre_per_100um"] + SHAPE, f"STAGE 1 (skeleton radius volume proxy), cells >= {MIN_CABLE:.0f} um cable")

    # density-rule errors and what the shape metrics say about them
    dens_pred = judge.pre_per_100um.values.astype(float) < PRE_CUT
    err = judge[dens_pred != (judge.group.values == "glia")]
    print(f"\ndensity rule (< {PRE_CUT}) errors on the judgeable comparison set: {len(err)}")
    cols = ["group", "label", "root", "pre_per_100um", "cable_um", "vol_um3", "L1_um", "vol_per_L1", "fill", "cable_per_L1"]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 40)
    print(err[cols].to_string(index=False))

    # 2-feature combination: density + best shape metric
    best_m = "vol_per_L1"
    b, mu, sd, acc, bal, ok, p = logistic2(judge, "pre_per_100um", best_m)
    print(f"\nlogistic (density + {best_m}): accuracy {acc * 100:.0f}%, balanced {bal * 100:.0f}% (in-sample, n={ok.sum()})")

    # partner cells: the flagged ones and the reviewed cell
    part = df[df.set == "partner"].copy()
    part["shape_call"] = ""
    rule, _, _ = best_cut(judge[best_m], judge.group == "glia")
    print(f"\npartner cells: best shape rule on {best_m}: {rule}")
    show = part[(part.label == "likely glia") | (part.root == "648518347577486275")]
    print(show[["root", "label", "pre_per_100um", "cable_um", "vol_um3", "L1_um", "vol_per_L1", "fill", "cable_per_L1", "notes"]]
          .sort_values("vol_per_L1").to_string(index=False))
    print("DONE")


if __name__ == "__main__":
    main()
