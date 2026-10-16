"""Emit fixed-width ASCII tables of the photoreceptor results for pasting into Slack.

No completeness filtering: every sampled cell is included, and each summary value is
reported as  median (min-max)  so the spread - including truncated cells - is visible.

Comparability: aedes counts come from CAVE `synapses_v2`, which counts EVERY synapse
regardless of partner size, so the Drosophila numbers use neuprint label='Segment'
(all partner bodies incl. fragments); the default label='Neuron' drops ~46% of R1-6
output synapses.
"""
from faab.config import REPO
import os
import numpy as np
import pandas as pd

OUT = os.path.join(REPO, "figures")
COLS = ["out_syn", "out_partners", "in_syn", "in_partners",
        "top_partner", "top3_pct", "singletons"]
HEAD = ["out_syn", "out_prtnrs", "in_syn", "in_prtnrs", "top_prtnr", "top3_%", "singletons"]


def table(headers, rows, aligns=None):
    cols = len(headers)
    aligns = aligns or ["<"] + [">"] * (cols - 1)
    data = [[str(h) for h in headers]] + [[str(c) for c in r] for r in rows]
    w = [max(len(row[i]) for row in data) for i in range(cols)]
    sep = "-+-".join("-" * x for x in w)
    out = [" | ".join(f"{headers[i]:{aligns[i]}{w[i]}}" for i in range(cols)), sep]
    for r in rows:
        out.append(" | ".join(f"{str(r[i]):{aligns[i]}{w[i]}}" for i in range(cols)))
    return "\n".join(out)


def mr(s):
    """median (min-max)"""
    s = pd.Series(s).dropna()
    return f"{int(round(s.median()))} ({int(s.min())}-{int(s.max())})"


def aedes_frame():
    df = pd.read_csv(os.path.join(OUT, "photoreceptor_io.csv"))
    df["root"] = df["root"].astype("int64").astype(str)
    p = pd.read_csv(os.path.join(OUT, "photoreceptor_partners.csv"))
    p = p[p.direction == "output"]
    top3 = {}
    for cell, d in p.groupby("cell"):
        tot = d.n_syn.sum()
        top3[str(int(cell))] = int(round(d.n_syn.nlargest(3).sum() / tot * 100)) if tot else 0
    df["top3_pct"] = df.root.map(top3)
    return df.rename(columns={"top_out_syn": "top_partner", "singleton_out": "singletons"})


def summary_rows(df, label_fmt):
    rows = []
    for cls, d in df.groupby("class"):
        rows.append([label_fmt.format(n=len(d)), cls] + [mr(d[c]) for c in COLS])
    return rows


def main():
    aed = aedes_frame()
    mcns = pd.read_csv(os.path.join(OUT, "mcns_photoreceptor_percell.csv"))
    fafb = pd.read_csv(os.path.join(OUT, "fafb_photoreceptor_percell.csv"))

    print("TABLE 1 - FAAB (aedes) photoreceptor connectivity, per cell")
    print("          dataset: wclee_aedes_brain; root_ids current as of this query")
    print()
    rows = [[r.root, r["class"]] + [r[c] for c in COLS] for _, r in aed.iterrows()]
    print(table(["root_id", "class"] + HEAD, rows))
    print()
    print("Counts from CAVE synapses_v2: EVERY synapse, regardless of partner size.")

    print("\n\nTABLE 2 - summary of all three datasets:  median (min-max)")
    print("          no filtering - every sampled cell is included")
    print()
    r16 = aed[aed["class"] == "R1-R6"].copy()
    r16["class"] = "R1-6"
    rows2 = summary_rows(r16, "FAAB aedes  (n={n})")
    rows2 += summary_rows(mcns, "Dmel mCNS  (n={n})")
    rows2 += summary_rows(fafb, "FAFB FlyWire  (n={n})")
    print(table(["source", "class"] + HEAD, rows2))
    print()
    print("FAAB    = all 6 aedes R1-R6 cells (the 2 R7-R8 cells are listed in Table 1;")
    print("          with only 2 cells a median is not meaningful).")
    print("Dmel    = male-CNS:v1.0, 40 cells per type sampled at random.")
    print("FAFB    = FlyWire public, materialization 783, cleft-score >= 50,")
    print("          150 cells per type sampled at random.")
    print()
    print("READ THE RANGES: FAFB photoreceptors are heavily truncated (its lamina sits")
    print("at the edge of the imaged volume) - R1-6 output runs from 1 to 189 synapses,")
    print("so its median sits well below what a COMPLETE cell reaches. mCNS is much more")
    print("uniform (72-252). The upper end of each range is the best guide to what a")
    print("fully reconstructed photoreceptor looks like.")
    print()
    print("NOTE ON SAMPLE SIZE: raising the FAFB sample from 40 to 150 cells did NOT")
    print("narrow these ranges (R1-6 stayed 1-189) - it moved the median down (47 -> 36)")
    print("and confirmed the spread is REAL population heterogeneity from truncation,")
    print("not sampling noise. More cells buy a more precise median, not a tighter range.")

    print("\n\nTABLE 3 - normalised comparison, R1-6 only  (per-cell ratios, then median)")
    print()

    def ratios(d, label):
        d = d[d.out_syn > 0]
        return [label, len(d),
                mr(d.out_partners / d.out_syn * 100),
                mr(d.top_partner / d.out_syn * 100),
                mr(d.top3_pct),
                mr(d.singletons / d.out_partners * 100)]

    rows3 = [ratios(r16, "FAAB aedes"),
             ratios(mcns[mcns["class"] == "R1-6"], "Dmel mCNS"),
             ratios(fafb[fafb["class"] == "R1-6"], "FAFB FlyWire")]
    print(table(["source", "n", "prtnrs/100syn", "top_prtnr_%", "top3_%", "singleton_%"],
                rows3))
    print()
    print("prtnrs/100syn = distinct output partners per 100 output synapses")
    print("top_prtnr_%   = share of output going to the single strongest partner")
    print("singleton_%   = fraction of partners receiving exactly 1 synapse")
    print()
    print("Ratios are computed PER CELL and then summarised, so they are insensitive to")
    print("how completely each cell is reconstructed - which is why they still separate")
    print("the datasets even though the raw counts above overlap. The two INDEPENDENT")
    print("fly datasets agree with each other and both differ from aedes.")

    print("\n\nTABLE 4 - top output partners per FAAB photoreceptor")
    print()
    p = pd.read_csv(os.path.join(OUT, "photoreceptor_partners.csv"))
    p = p[p.direction == "output"]
    rows4 = []
    for cell in aed.root:
        d = p[p.cell.astype("int64").astype(str) == cell].nlargest(3, "n_syn")
        top = "  ".join(f"{int(x.partner)} ({int(x.n_syn)})" for _, x in d.iterrows())
        rows4.append([cell, aed.loc[aed.root == cell, "class"].iloc[0], top])
    print(table(["root_id", "class", "top 3 output partners (n synapses)"], rows4,
                aligns=["<", "<", "<"]))
    print("\nDONE")


if __name__ == "__main__":
    main()
