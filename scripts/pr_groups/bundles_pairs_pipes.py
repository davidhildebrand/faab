"""Pipe rendering of a pairing paste table (figures/pr_bundles_pairs_paste*.tsv): per group its 'group N' line, the
' | '-separated header and the cell rows, padded to equal column widths, NO dash row (David, 2026-10-02), blank
line between groups. Network-free. Usage: python faab_pr_bundles_pairs_pipes.py [suffix]  (e.g. _extra)"""
from faab.config import REPO
import os
import sys
from faab.config import fig

HERE = REPO
OUT = os.path.join(HERE, "figures")


def render(lines):
    out, header, rows, gname = [], None, [], None
    for ln in lines + [""]:
        cells = ln.rstrip("\n").split("\t")
        if ln.startswith("group "):
            gname = cells[0]
        elif cells[0] == "root_id_orig":
            header = cells
        elif ln.strip() == "":
            if header and rows:  # no padding: David only needs the ' | ' separators (2026-10-02)
                out += [gname, " | ".join(header)]
                out += [" | ".join(r[i] for i in range(len(header))) for r in rows] + [""]
            header, rows = None, []
        else:
            rows.append(cells + [""] * (len(header) - len(cells)) if header else cells)
    return "\n".join(out).rstrip("\n")


def main():
    suffix = sys.argv[1] if len(sys.argv) > 1 else ""
    lines = open(fig(f"pr_bundles_pairs_paste{suffix}.tsv")).read().split("\n")
    txt = render(lines)
    open(fig(f"pr_bundles_pairs_pipes{suffix}.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
