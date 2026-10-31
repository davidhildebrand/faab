"""MeTu4-vs-LC re-check by nblast_align (pose-invariant, all-by-all). For each
LC-annotated cell, compare its morphological similarity to the MeTu4 group vs the
rest of the LC group, and flag LC cells that belong in MeTu4. Read-only (local compute)."""
from faab.config import REPO
import os
import pickle
import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import navis
from faab.config import fig

OUT = os.path.join(REPO, "figures")
MIN_PTS = 20


def main():
    D = pickle.load(open(fig("metu_lc_dotprops_fresh.pkl"), "rb"))
    by = D["dotprops"]; lab = D["label_by_root"]; stat = D["status_by_root"]
    roots = [r for r in by if len(by[r].points) >= MIN_PTS]
    nls = []
    for r in roots:
        d = by[r]
        d.id = str(r); d.name = str(r); d.units = "um"
        nls.append(d)
    nls = navis.NeuronList(nls)
    labels = np.array([lab[r] for r in roots])
    print(f"cells in re-check: {len(roots)}  (MeTu4 {int((labels=='MeTu4').sum())}, "
          f"LC {int((labels=='LC').sum())})", flush=True)

    S = navis.nblast_align(nls, nls, align_method="rigid", two_way_align=False,
                           n_cores=6, progress=False, align_kwargs={"scale": True})
    M = (S.values + S.values.T) / 2.0
    np.fill_diagonal(M, np.nan)
    idx = {r: i for i, r in enumerate(roots)}
    is_m = labels == "MeTu4"
    is_l = labels == "LC"

    rows = []
    for r in roots:
        i = idx[r]
        row_m = M[i, is_m]; row_l = M[i, is_l]
        # exclude self is already handled by nan on diagonal (nanmean skips it)
        simM = np.nanmean(row_m)
        simL = np.nanmean(row_l)
        # nearest-neighbour label (max off-diagonal)
        order = np.argsort(-np.nan_to_num(M[i], nan=-1))
        knn = labels[order[:5]]
        rows.append({"root": r, "label": lab[r], "status": stat.get(r),
                     "sim_MeTu4": simM, "sim_LC": simL, "margin": simM - simL,
                     "knn5_MeTu4": int((knn == "MeTu4").sum()),
                     "npts": len(by[r].points)})
    df = pd.DataFrame(rows)

    # calibration
    mm = df[df.label == "MeTu4"]; ll = df[df.label == "LC"]
    print("\n=== calibration (mean +/- sd of margin = sim_MeTu4 - sim_LC) ===")
    print(f"  MeTu4 cells: margin {mm.margin.mean():+.3f} +/- {mm.margin.std():.3f}  "
          f"(sim_MeTu4 {mm.sim_MeTu4.mean():.3f}, sim_LC {mm.sim_LC.mean():.3f})")
    print(f"  LC cells   : margin {ll.margin.mean():+.3f} +/- {ll.margin.std():.3f}  "
          f"(sim_MeTu4 {ll.sim_MeTu4.mean():.3f}, sim_LC {ll.sim_LC.mean():.3f})")
    m4_lo = np.nanpercentile(mm.sim_MeTu4, 25)          # MeTu4 self-similarity band (25th pct)

    # LC cells that look like MeTu4: more similar to MeTu4 than to LC, and inside the MeTu4 band
    cand = ll[(ll.margin > 0) & (ll.sim_MeTu4 >= m4_lo)].sort_values("margin", ascending=False)
    strong = ll[(ll.margin > 0) & (ll.knn5_MeTu4 >= 3)].sort_values("margin", ascending=False)
    print(f"\nMeTu4 self-sim 25th-pct threshold: {m4_lo:.3f}")
    print(f"\n=== LC cells flagged as MeTu4 candidates ({len(cand)}) ===")
    cols = ["root", "status", "sim_MeTu4", "sim_LC", "margin", "knn5_MeTu4", "npts"]
    with pd.option_context("display.width", 160):
        print(cand[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n(subset also having >=3/5 nearest neighbours MeTu4: {len(strong)} — highest confidence)")

    df.to_csv(fig("metu_lc_scores.csv"), index=False)
    pickle.dump({"roots": roots, "labels": labels, "M": M, "df": df,
                 "candidates": cand["root"].tolist(), "strong": strong["root"].tolist()},
                open(fig("metu_lc_nblast.pkl"), "wb"))
    print("\nsaved metu_lc_scores.csv + metu_lc_nblast.pkl\nDONE", flush=True)


if __name__ == "__main__":
    main()
