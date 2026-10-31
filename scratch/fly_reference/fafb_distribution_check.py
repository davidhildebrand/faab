"""Is the low FAFB photoreceptor synapse count a detection-threshold effect, or are
many FAFB photoreceptors truncated fragments? Inspect the per-cell distribution."""
from faab.config import REPO
import os
import numpy as np
import pandas as pd
from faab.config import fig

OUT = os.path.join(REPO, "figures")


def main():
    df = pd.read_csv(fig("fafb_photoreceptor_percell.csv"))
    pd.set_option("display.width", 170)
    for cls, d in df.groupby("class"):
        v = d.out_syn.sort_values().values
        print(f"\n=== FAFB {cls}: n={len(v)} sampled cells with output ===")
        qs = np.percentile(v, [0, 10, 25, 50, 75, 90, 100])
        print("  out_syn percentiles  0/10/25/50/75/90/100: "
              + " / ".join(str(int(x)) for x in qs))
        print(f"  cells with <10 syn: {(v < 10).sum():2d}   <25: {(v < 25).sum():2d}   "
              f">=100: {(v >= 100).sum():2d}   >=200: {(v >= 200).sum():2d}")
        print(f"  full sorted values: {list(v)}")
        # a truncated-fragment population is bimodal / heavy at the bottom;
        # a uniform detection difference just rescales a unimodal distribution
        top = d.nlargest(5, "out_syn")[["out_syn", "out_partners", "top_partner", "top3_pct"]]
        print("  top-5 cells by output:")
        print(top.to_string(index=False))
    print("\nDONE")


if __name__ == "__main__":
    main()
