"""Re-check #2 classifier: all-by-all nblast_align, then for each red LC cell decide
whether it is more similar to MeTu4 (reference) or to the other LC cells. Read-only."""
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
    D = pickle.load(open(fig("metu_lc2.pkl"), "rb"))
    by = D["dotprops"]; lab = D["label"]
    roots = [r for r in by if len(by[r].points) >= MIN_PTS]
    nl = []
    for r in roots:
        d = by[r]; d.id = str(r); d.name = str(r); d.units = "um"; nl.append(d)
    nl = navis.NeuronList(nl)
    labels = np.array([lab[r] for r in roots])
    is_m = labels == "MeTu4"; is_l = labels == "LC"
    print(f"cells: {len(roots)} (MeTu4 ref {int(is_m.sum())}, LC test {int(is_l.sum())})", flush=True)

    S = navis.nblast_align(nl, nl, align_method="rigid", two_way_align=False,
                           n_cores=6, progress=False, align_kwargs={"scale": True})
    M = (S.values + S.values.T) / 2.0
    np.fill_diagonal(M, np.nan)
    idx = {r: i for i, r in enumerate(roots)}

    rows = []
    for r in roots:
        i = idx[r]
        simM = np.nanmean(M[i, is_m])
        simL = np.nanmean(M[i, is_l])
        order = np.argsort(-np.nan_to_num(M[i], nan=-1))
        knn = labels[order[:5]]
        rows.append({"root": r, "label": lab[r], "sim_MeTu4": simM, "sim_LC": simL,
                     "margin": simM - simL, "knn5_MeTu4": int((knn == "MeTu4").sum()),
                     "npts": len(by[r].points)})
    df = pd.DataFrame(rows)
    mm = df[df.label == "MeTu4"]
    med = mm.sim_MeTu4.median()
    print(f"\nMeTu4 self-similarity median {med:.3f} (mean {mm.sim_MeTu4.mean():.3f})")
    lc = df[df.label == "LC"].copy().sort_values("margin", ascending=False)

    g1 = lc[(lc.sim_MeTu4 >= med) & (lc.margin > 0.05) & (lc.knn5_MeTu4 >= 3)]
    g2 = lc[(~lc.root.isin(g1.root)) & (lc.margin > 0.02) & (lc.knn5_MeTu4 >= 2)]
    stay = lc[~lc.root.isin(g1.root) & ~lc.root.isin(g2.root)]
    cols = ["root", "sim_MeTu4", "sim_LC", "margin", "knn5_MeTu4", "npts"]
    fmt = lambda x: f"{x:.3f}"
    with pd.option_context("display.width", 160):
        print(f"\n=== GROUP 1 — more like MeTu4 (recommend, {len(g1)}) ===")
        print(g1[cols].to_string(index=False, float_format=fmt))
        print(f"\n=== GROUP 2 — review ({len(g2)}) ===")
        print(g2[cols].to_string(index=False, float_format=fmt))
        print(f"\n=== stays LC ({len(stay)}) ===")
        print(stay[cols].to_string(index=False, float_format=fmt))

    open(fig("metu_lc2_group1.txt"), "w").write("\n".join(str(r) for r in g1.root) + "\n")
    open(fig("metu_lc2_group2.txt"), "w").write("\n".join(str(r) for r in g2.root) + "\n")
    open(fig("metu_lc2_stay.txt"), "w").write("\n".join(str(r) for r in stay.root) + "\n")
    open(fig("metu_lc2_metu4ref.txt"), "w").write("\n".join(str(r) for r in mm.root) + "\n")
    df.to_csv(fig("metu_lc2_scores.csv"), index=False)
    print("\nsaved metu_lc2_group1.txt / group2.txt / stay.txt / metu4ref.txt / scores.csv\nDONE", flush=True)


if __name__ == "__main__":
    main()
