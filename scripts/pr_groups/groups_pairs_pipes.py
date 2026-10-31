"""Pipe/dash rendering of figures/faab_pr_groups_pairs_paste.tsv: per group a 'group N' line, a
' | '-separated header, a dash row and the 8 cell rows, blank line between groups (same layout as the
partner tables). Margins printed without a sign. Output figures/faab_pr_groups_pairs_pipes.txt."""
from faab.config import REPO
import os
from faab.config import fig

HERE = REPO
OUT = os.path.join(HERE, "figures")


def table(header, rows):
    data = [header] + rows
    w = [max(len(r[i]) for r in data) for i in range(len(header))]
    out = [" | ".join(f"{header[i]:<{w[i]}}" for i in range(len(header))), "-+-".join("-" * x for x in w)]
    for r in rows:
        out.append(" | ".join(f"{r[i]:<{w[i]}}" for i in range(len(header))))
    return out


def main():
    lines = open(fig("faab_pr_groups_pairs_paste.tsv")).read().split("\n")
    out, header, rows, gname = [], None, [], None
    for ln in lines + [""]:
        cells = ln.split("\t")
        if ln.startswith("group "):
            gname = cells[0]
        elif cells[0] == "root_id_orig":
            header = cells
        elif ln.strip() == "":
            if header and rows:
                out += [gname] + table(header, rows) + [""]
            header, rows = None, []
        else:
            cells = [c.lstrip("+") for c in cells]
            rows.append(cells)
    txt = "\n".join(out).rstrip("\n")
    open(fig("faab_pr_groups_pairs_pipes.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
