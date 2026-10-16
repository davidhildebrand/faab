"""Top 10 output partners for each FAAB photoreceptor, as ASCII blocks.

Includes each partner's share of that cell's output, how many of the 8 photoreceptors
also contact it (convergence), and any FlyTable annotation. Read-only.
"""
from faab.config import REPO
import os
import pandas as pd

OUT = os.path.join(REPO, "figures")
SERVER = "https://flytable.mrc-lmb.cam.ac.uk/"
TOPN = 10


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


def main():
    p = pd.read_csv(os.path.join(OUT, "photoreceptor_partners.csv"))
    io = pd.read_csv(os.path.join(OUT, "photoreceptor_io.csv"))
    out = p[p.direction == "output"].copy()
    cls_of = dict(zip(io.root.astype("int64"), io["class"]))
    tot_of = dict(zip(io.root.astype("int64"), io.out_syn))

    # how many of the 8 photoreceptors contact each partner
    shared = out.groupby("partner")["cell"].nunique().to_dict()

    ann = {}
    try:
        import seaserpent as ss
        meta = ss.Table("aedes_main", server=SERVER).to_frame()
        meta.columns = [str(c) for c in meta.columns]
        for _, r in meta.iterrows():
            try:
                rid = int(str(r["root_id"]))
            except Exception:
                continue
            bits = [str(r[c]) for c in ("superclass", "class", "subclass", "type")
                    if c in meta.columns and str(r[c]) not in ("nan", "<NA>", "None", "")]
            if bits:
                ann[rid] = " / ".join(bits)
    except Exception as e:
        print("(FlyTable annotations unavailable:", str(e)[:60], ")\n")

    print("TOP 10 OUTPUT PARTNERS PER FAAB PHOTORECEPTOR")
    print("dataset: wclee_aedes_brain (CAVE synapses_v2); root_ids current as of query")
    print("shared_by = how many of the 8 photoreceptors in this set contact that partner")
    print()
    for cell in io.root.astype("int64"):
        d = out[out.cell == cell].nlargest(TOPN, "n_syn")
        tot = tot_of[cell]
        npart = int((out.cell == cell).sum())
        print(f"\n=== {cell}  [{cls_of[cell]}]  "
              f"{tot} output synapses across {npart} partners ===")
        rows = []
        for i, (_, r) in enumerate(d.iterrows(), 1):
            pid = int(r.partner)
            rows.append([i, str(pid), int(r.n_syn),
                         f"{r.n_syn / tot * 100:.1f}",
                         shared.get(pid, 1),
                         ann.get(pid, "")])
        print(table(["#", "partner root_id", "n_syn", "% of out", "shared_by",
                     "FlyTable annotation"], rows,
                    aligns=[">", "<", ">", ">", ">", "<"]))
    print("\nDONE")


if __name__ == "__main__":
    main()
