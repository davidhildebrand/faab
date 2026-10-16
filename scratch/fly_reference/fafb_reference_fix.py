"""Rebuild the FAFB photoreceptor reference from the BEST-RECONSTRUCTED cells only.

FAFB's lamina lies at the edge of the imaged volume, so its photoreceptor population
is heavily truncated (R1-6 output synapses run 1..189, with 8/39 cells under 10).
An IQR filter straddles fragments and partial cells and measures truncation rather
than biology. The top quartile by output synapses represents the most completely
reconstructed cells and is the appropriate reference for "a complete photoreceptor".

Recomputed locally from the cached per-cell CSV - no new network calls.
"""
from faab.config import REPO
import os
import numpy as np
import pandas as pd

OUT = os.path.join(REPO, "figures")
COLS = ["out_syn", "out_partners", "in_syn", "in_partners",
        "top_partner", "top3_pct", "singletons"]


def main():
    df = pd.read_csv(os.path.join(OUT, "fafb_photoreceptor_percell.csv"))
    recs = []
    for cls, d in df.groupby("class"):
        q75 = d.out_syn.quantile(0.75)
        top = d[d.out_syn >= q75]
        recs.append({"source": "FAFB", "class": cls, "n": len(top),
                     "cutoff_out_syn": int(round(q75)),
                     **{c: int(round(top[c].median())) for c in COLS}})
        # for transparency also record what the contaminated IQR version gave
        q1, q3 = d.out_syn.quantile([0.25, 0.75])
        mid = d[(d.out_syn >= q1) & (d.out_syn <= q3)]
        print(f"{cls}: sampled {len(d)} cells | truncation-contaminated IQR "
              f"({int(q1)}-{int(q3)}) median out_syn {int(mid.out_syn.median())} "
              f"-> top-quartile (>={int(round(q75))}) median {int(top.out_syn.median())}")
    ref = pd.DataFrame(recs)
    pd.set_option("display.width", 170)
    print("\n=== FAFB reference, best-reconstructed quartile ===")
    print(ref.to_string(index=False))
    ref.to_csv(os.path.join(OUT, "fafb_photoreceptor_ref.csv"), index=False)
    print("\noverwrote fafb_photoreceptor_ref.csv\nDONE")


if __name__ == "__main__":
    main()
