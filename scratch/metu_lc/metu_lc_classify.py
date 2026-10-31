"""Tiered LC->MeTu4 reclassification from the saved nblast_align matrix.
Stricter, multi-signal rule. Read-only, local."""
from faab.config import REPO
import os
import pickle
import numpy as np
import pandas as pd
from faab.config import fig

OUT = os.path.join(REPO, "figures")


def main():
    R = pickle.load(open(fig("metu_lc_nblast.pkl"), "rb"))
    df = R["df"]
    mm = df[df.label == "MeTu4"]
    med = mm.sim_MeTu4.median()                 # a real MeTu4 is ~this MeTu4-like
    print(f"MeTu4 self-similarity: median {med:.3f}, mean {mm.sim_MeTu4.mean():.3f}")

    lc = df[df.label == "LC"].copy()
    # Tier 1 (recommend): as MeTu4-like as a typical MeTu4, clearly more MeTu4 than LC,
    #                     and a majority of nearest neighbours are MeTu4.
    t1 = lc[(lc.sim_MeTu4 >= med) & (lc.margin > 0.05) & (lc.knn5_MeTu4 >= 3)]
    t1 = t1.sort_values("margin", ascending=False)
    # Tier 2 (review): positive margin + some MeTu4 kNN support, but weaker.
    t2 = lc[(~lc.root.isin(t1.root)) & (lc.margin > 0.02) & (lc.knn5_MeTu4 >= 2)]
    t2 = t2.sort_values("margin", ascending=False)

    cols = ["root", "status", "sim_MeTu4", "sim_LC", "margin", "knn5_MeTu4", "npts"]
    fmt = lambda x: f"{x:.3f}"
    print(f"\n=== TIER 1 — recommend reclassify LC -> MeTu4 ({len(t1)}) ===")
    with pd.option_context("display.width", 160):
        print(t1[cols].to_string(index=False, float_format=fmt))
    print(f"\n=== TIER 2 — review ({len(t2)}) ===")
    with pd.option_context("display.width", 160):
        print(t2[cols].to_string(index=False, float_format=fmt))
    print(f"\nremaining LC (stay LC): {len(lc) - len(t1) - len(t2)}")

    open(fig("metu4_cand_tier1.txt"), "w").write(
        "\n".join(str(r) for r in t1.root) + "\n")
    open(fig("metu4_cand_tier2.txt"), "w").write(
        "\n".join(str(r) for r in t2.root) + "\n")
    # MeTu4 reference ids for the ng link
    open(fig("metu4_ref.txt"), "w").write(
        "\n".join(str(r) for r in mm.root) + "\n")
    t1[cols].to_csv(fig("metu4_tier1.csv"), index=False)
    print("\nsaved metu4_cand_tier1.txt / tier2.txt / metu4_ref.txt / metu4_tier1.csv\nDONE")


if __name__ == "__main__":
    main()
