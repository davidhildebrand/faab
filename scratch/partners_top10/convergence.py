"""Convergent outputs of the FAAB photoreceptor set: which partners receive input
from MULTIPLE photoreceptors? In fly, R1-6 of one cartridge converge on the same
L1/L2/L3, so shared targets are the aedes equivalents (and the merge candidates).

Partner ids are joined against FlyTable aedes_main for any existing annotation.
Read-only; the convergence itself is computed from the cached partner CSV.
"""
from faab.config import REPO
import os
import numpy as np
import pandas as pd

OUT = os.path.join(REPO, "figures")
SERVER = "https://flytable.mrc-lmb.cam.ac.uk/"


def main():
    p = pd.read_csv(os.path.join(OUT, "photoreceptor_partners.csv"))
    io = pd.read_csv(os.path.join(OUT, "photoreceptor_io.csv"))
    cls_of = dict(zip(io.root.astype("int64"), io["class"]))

    out = p[p.direction == "output"].copy()
    r16 = out[out.cell.map(lambda c: cls_of.get(int(c)) == "R1-R6")]
    print(f"R1-R6 cells: {r16.cell.nunique()} | output partner rows: {len(r16)}")

    conv = (r16.groupby("partner")
            .agg(n_photoreceptors=("cell", "nunique"),
                 total_syn=("n_syn", "sum"),
                 max_syn_from_one=("n_syn", "max"))
            .reset_index()
            .sort_values(["n_photoreceptors", "total_syn"], ascending=False))

    n_cells = r16.cell.nunique()
    print(f"\ndistinct output partners of the {n_cells} R1-R6: {len(conv)}")
    print("convergence distribution (how many of the 6 R1-R6 hit the same partner):")
    print(conv.n_photoreceptors.value_counts().sort_index(ascending=False).to_string())

    # annotate partners where FlyTable knows them
    ann = {}
    try:
        import seaserpent as ss
        meta = ss.Table("aedes_main", server=SERVER).to_frame()
        meta.columns = [str(c) for c in meta.columns]
        for _, r in meta.iterrows():
            try:
                ann[int(str(r["root_id"]))] = " / ".join(
                    str(r[c]) for c in ("superclass", "class", "subclass")
                    if c in meta.columns and str(r[c]) not in ("nan", "<NA>", "None"))
            except Exception:
                pass
        print(f"\nFlyTable annotation map: {len(ann)} root_ids")
    except Exception as e:
        print("\n(FlyTable lookup unavailable:", str(e)[:70], ")")

    conv["annotation"] = conv.partner.map(lambda r: ann.get(int(r), ""))
    top = conv[conv.n_photoreceptors >= 2].copy()
    print(f"\n=== partners shared by >=2 of the {n_cells} R1-R6 ({len(top)}) ===")
    pd.set_option("display.width", 200)
    print(top.head(25).to_string(index=False))

    conv.to_csv(os.path.join(OUT, "faab_convergence.csv"), index=False)
    print("\nsaved faab_convergence.csv\nDONE")


if __name__ == "__main__":
    main()
