"""ASCII convergence tables: which targets do photoreceptors converge onto, in
male-CNS, FAFB/FlyWire, and FAAB (aedes)."""
from faab.config import REPO
import os
import pandas as pd
from faab.config import fig

OUT = os.path.join(REPO, "figures")


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


def fly_block(csv, label, nsrc):
    cv = pd.read_csv(fig(csv))
    cv["post_type"] = cv["post_type"].astype(str)
    for cls in ("R1-6", "R7", "R8"):
        d = cv[cv.src_class == cls].sort_values("n_syn", ascending=False)
        d = d[~d.post_type.isin(["nan", "(untyped fragment)"])].head(6)
        frag = cv[(cv.src_class == cls) & (cv.post_type == "(untyped fragment)")]
        fragpct = float(frag.pct_of_output.iloc[0]) if len(frag) else 0.0
        rows = [[r.post_type, f"{r.pct_of_output:.1f}", int(r.n_sources), int(r.n_targets)]
                for _, r in d.iterrows()]
        print(f"\n  {label} {cls}  (sources sampled: {nsrc[cls]}; "
              f"{fragpct:.0f}% of output goes to untyped fragments)")
        print(table(["target type", "% of output", "n_sources", "n_targets"], rows))


def main():
    print("TABLE A - convergent OUTPUT TYPES, Drosophila male-CNS:v1.0")
    print("          n_sources = how many sampled photoreceptors hit that type")
    print("          n_targets = how many distinct cells of that type were hit")
    fly_block("mcns_convergence.csv", "mCNS", {"R1-6": 60, "R7": 60, "R8": 60})

    print("\n\nTABLE B - convergent OUTPUT TYPES, FAFB / FlyWire (independent specimen)")
    fly_block("fafb_convergence.csv", "FAFB", {"R1-6": 150, "R7": 150, "R8": 150})

    print("\n\nTABLE C - FAAB (aedes): partners shared by >1 of the 6 R1-R6 cells")
    print("          aedes has no cell-type annotations here, so targets are root_ids")
    print()
    cv = pd.read_csv(fig("faab_convergence.csv"))
    dist = cv.n_photoreceptors.value_counts().sort_index(ascending=False)
    rows = [[int(k), int(v)] for k, v in dist.items()]
    print(table(["shared by N of 6 R1-R6", "n partners"], rows))
    print()
    top = cv[cv.n_photoreceptors >= 2].sort_values(
        ["total_syn", "n_photoreceptors"], ascending=False).head(12)
    rows2 = [[str(int(r.partner)), int(r.n_photoreceptors), int(r.total_syn),
              int(r.max_syn_from_one),
              str(r.annotation) if str(r.annotation) != "nan" else ""]
             for _, r in top.iterrows()]
    print(table(["partner root_id", "n_photorec", "total_syn", "max_from_one",
                 "FlyTable annotation"], rows2,
                aligns=["<", ">", ">", ">", "<"]))
    print("\nDONE")


if __name__ == "__main__":
    main()
