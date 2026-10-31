"""Network-free cell-by-cell diff of two TSV exports of one tab (xlsx_to_tsv.py), by sheet row and column header:
lists every changed cell (row, header of its block, old -> new) and added / removed lines. Works for any tab layout
(block tabs and plain tabs); rows are compared at the same row number, so an insertion shifts everything below it
(reported as such). Usage: tab_diff.py OLD.tsv NEW.tsv [MAX_LINES]"""
import sys

old = open(sys.argv[1], encoding="utf-8").read().rstrip("\n").split("\n")
new = open(sys.argv[2], encoding="utf-8").read().rstrip("\n").split("\n")
cap = int(sys.argv[3]) if len(sys.argv) > 3 else 200


def headers(lines):
    hdr, out = [], []
    for ln in lines:
        c = ln.split("\t")
        if "xyz" in c and ("root_id_curr" in c or "root_id_orig" in c):
            hdr = c
        out.append(hdr)
    return out


ho, hn = headers(old), headers(new)
changes = []
for i in range(max(len(old), len(new))):
    a = old[i].split("\t") if i < len(old) else []
    b = new[i].split("\t") if i < len(new) else []
    for j in range(max(len(a), len(b))):
        x, y = (a[j] if j < len(a) else ""), (b[j] if j < len(b) else "")
        if x != y:
            h = hn[i] if i < len(hn) else []
            changes.append((i + 1, h[j] if j < len(h) and h[j] else f"col{j + 1}", x, y))
print(f"{sys.argv[1].split('/')[-1]} ({len(old)} lines) -> {sys.argv[2].split('/')[-1]} ({len(new)} lines): "
      f"{len(changes)} changed cells in {len({c[0] for c in changes})} rows")
for r, h, x, y in changes[:cap]:
    print(f"  row {r} {h}: {x[:90]!r} -> {y[:90]!r}")
if len(changes) > cap:
    print(f"  ... {len(changes) - cap} more")
