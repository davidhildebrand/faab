"""Two-feature glia classifier: presynaptic density (sites per 100 um cable) + volume per longest
extent, logistic regression on log10 features, fitted on the judgeable comparison set (cable >= 100 um)
of a glia-shape table (default figures/glia_shape_skel.tsv; pass figures/glia_shape_l2.tsv for the
exact-volume version). Reports in-sample and leave-one-out accuracy, the boundary as an equation,
the cells it gets wrong, and applies it to every partner cell of the photoreceptor downstream table
(rows with set == partner). Output: figures/glia_logistic_partners.tsv (+ _l2 suffix for the L2 table).
Network-free."""
import datetime
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from faab.glia.shape import OUT, MIN_CABLE, PRE_CUT, fmt

PR_PAT = re.compile(r"\bR[1-8]\b|photoreceptor", re.I)  # rows already identified as a photoreceptor
TABLE = os.path.join(OUT, "faab_pr_top20_update.tsv")
MODEL = os.path.join(OUT, "glia_logistic_model.json")


def write_p_column(judge):
    """p_glia (recommended model) as a column of the partner table; blank for fragments, single chunks
    and photoreceptors. Run after faab_glia_l2_fetch.py and before faab_pr_top20_render.py."""
    if not os.path.exists(TABLE):
        return
    meta_path = os.path.join(OUT, "faab_pr_top20_update_meta.json")
    pr_roots = set(json.load(open(meta_path))["pr_roots"].values()) if os.path.exists(meta_path) else set()
    p_by = {str(r.root): float(r.p_glia) for r in judge.itertuples()}
    full = pd.read_csv(TABLE, sep="\t", dtype=str).fillna("")
    vals = []
    for _, r in full.iterrows():
        exempt = r["root_id_curr"] in pr_roots or bool(PR_PAT.search(f"{r['notes']} {r['annotation']}"))
        p = p_by.get(r["root_id_curr"])
        vals.append("" if (exempt or p is None or not np.isfinite(p)) else f"{p:.3f}")
    full["p_glia"] = vals
    full.to_csv(TABLE, sep="\t", index=False)
    print(f"\np_glia column written for {sum(1 for v in vals if v)} of {len(full)} table rows "
          f"(blank: fragments/single chunks and photoreceptor rows)")

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "glia_shape_skel.tsv")
M1, M2 = "pre_per_100um", (sys.argv[2] if len(sys.argv) > 2 else "vol_per_L1")  # 2nd feature: vol_per_L1 | fill | ...
TAG = ("_l2" if SRC.endswith("_l2.tsv") else "") + ("" if M2 == "vol_per_L1" else f"_{M2}")
EPS1, EPS2 = 0.1, 1e-6
RIDGE = 1e-3


def feats(df):
    return np.column_stack([np.log10(df[M1].astype(float).values + EPS1),
                            np.log10(df[M2].astype(float).values + EPS2)])


def fit(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])

    def nll(b):
        z = Z @ b
        return np.sum(np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y)) + RIDGE * np.sum(b[1:] ** 2)

    b = minimize(nll, np.zeros(3), method="BFGS").x
    return b, mu, sd


def predict(model, X):
    b, mu, sd = model
    return 1 / (1 + np.exp(-(np.column_stack([np.ones(len(X)), (X - mu) / sd]) @ b)))


def scores(pred, g):
    return np.mean(pred == g), 0.5 * (np.mean(pred[g]) + np.mean(~pred[~g]))


def main():
    df = pd.read_csv(SRC, sep="\t", dtype={"root": str}).fillna({"notes": "", "label": ""})
    ok = np.isfinite(df[M1].astype(float)) & np.isfinite(df[M2].astype(float)) & (df.cable_um >= MIN_CABLE)
    comp = df[(df.set == "comparison") & ok].reset_index(drop=True)
    X, y = feats(comp), (comp.group.values == "glia").astype(float)
    g = y > 0.5
    model = fit(X, y)
    p = predict(model, X)
    acc, bal = scores(p > 0.5, g)
    dacc, dbal = scores(comp[M1].astype(float).values < PRE_CUT, g)
    loo = np.zeros(len(X))
    for i in range(len(X)):
        m = np.ones(len(X), bool)
        m[i] = False
        loo[i] = predict(fit(X[m], y[m]), X[i:i + 1])[0]
    lacc, lbal = scores(loo > 0.5, g)
    b, mu, sd = model
    w = b[1:] / sd
    c = b[0] - np.sum(b[1:] * mu / sd)
    print(f"source {os.path.basename(SRC)}: fitted on {len(comp)} cells ({g.sum()} glia, {(~g).sum()} neurons), "
          f"cable >= {MIN_CABLE:.0f} um")
    print(f"density rule (< {PRE_CUT}) on the same cells: accuracy {dacc * 100:.0f}%, balanced {dbal * 100:.0f}%")
    print(f"logistic in-sample: accuracy {acc * 100:.0f}%, balanced {bal * 100:.0f}%; "
          f"leave-one-out: accuracy {lacc * 100:.0f}%, balanced {lbal * 100:.0f}%")
    print(f"boundary: {w[0]:.3f} * log10(density + {EPS1}) + {w[1]:.3f} * log10({M2} + {EPS2:g}) + {c:.3f} = 0 "
          f"(glia if the sum is > 0); p = 1 / (1 + exp(-sum))")
    print(f"p(glia): glia median {np.median(p[g]):.3f} ({p[g].min():.3f}-{p[g].max():.3f}); "
          f"neurons median {np.median(p[~g]):.3f} ({p[~g].min():.3f}-{p[~g].max():.3f})")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 42)
    comp["p_glia"], comp["p_loo"] = p.round(3), loo.round(3)
    err = comp[(loo > 0.5) != g]
    print(f"\nleave-one-out errors: {len(err)}")
    print(err[["group", "label", "root", M1, "cable_um", "vol_um3", "L1_um", M2, "p_glia", "p_loo"]].to_string(index=False))

    # ---- apply to the partner cells of the photoreceptor downstream table ----
    part = df[df.set == "partner"].copy()
    judge = part[ok.loc[part.index]].copy()
    judge["p_glia"] = predict(model, feats(judge)).round(3)
    judge["dens_flag"] = judge[M1].astype(float) < PRE_CUT
    judge["logit_flag"] = judge.p_glia > 0.5
    n_small = int((part.cable_um < MIN_CABLE).sum() + part.cable_um.isna().sum())
    print(f"\npartner cells: {len(part)} total, {len(judge)} judgeable (cable >= {MIN_CABLE:.0f} um), "
          f"{n_small} fragments or single chunks not judged")
    cols = ["root", M1, "cable_um", "vol_um3", "L1_um", M2, "p_glia", "notes"]
    for title, sel in (("BOTH: density flag and logistic agree (glia-like)", judge.dens_flag & judge.logit_flag),
                       ("ADDITIONAL: logistic says glia, density rule did not flag", ~judge.dens_flag & judge.logit_flag),
                       ("DISAGREE: density flagged, logistic says neuron", judge.dens_flag & ~judge.logit_flag)):
        sub = judge[sel].sort_values("p_glia", ascending=False)
        print(f"\n{title}: {len(sub)}")
        if len(sub):
            print(sub[cols].to_string(index=False))
    near = judge[~judge.logit_flag & (judge.p_glia > 0.2)].sort_values("p_glia", ascending=False)
    print(f"\nborderline (p 0.2-0.5, called neuron): {len(near)}")
    if len(near):
        print(near[cols].to_string(index=False))
    judge.sort_values("p_glia", ascending=False).to_csv(os.path.join(OUT, f"glia_logistic_partners{TAG}.tsv"),
                                                        sep="\t", index=False)
    if TAG == "_l2":  # recommended configuration: exact volumes + volume per extent
        json.dump({"features": [M1, M2], "eps": [EPS1, EPS2], "w": [float(w[0]), float(w[1])], "c": float(c),
                   "rule": "glia if w[0]*log10(density+eps[0]) + w[1]*log10(vol_per_L1+eps[1]) + c > 0",
                   "fitted_on": int(len(comp)), "loo_acc": float(lacc), "loo_bal": float(lbal),
                   "source": os.path.basename(SRC), "date": datetime.date.today().isoformat()},
                  open(MODEL, "w"), indent=1)
        write_p_column(judge)
    print("DONE")


if __name__ == "__main__":
    main()
