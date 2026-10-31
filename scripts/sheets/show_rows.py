"""Network-free: print chosen rows of a block-TSV tab export as 'header=value' pairs (the header of the block each row
sits in), with the block title. Usage: show_rows.py TAB.tsv ROW [ROW ...]   (ROW = sheet row number, 1-based; a-b = range)"""
import sys

path, want = sys.argv[1], []
for a in sys.argv[2:]:
    lo, _, hi = a.partition("-")
    want += list(range(int(lo), int(hi or lo) + 1))
lines = open(path, encoding="utf-8").read().split("\n")
hdr, title = None, ""
for i, line in enumerate(lines, 1):
    cells = line.split("\t")
    if "xyz" in cells and ("root_id_curr" in cells or "root_id_orig" in cells):
        hdr = cells
        continue
    if cells and cells[0] and not any(cells[1:]):
        title = cells[0][:100]
    if i in want:
        if hdr is None:
            print(f"{i}: {line}")
            continue
        pairs = [f"{h}={v}" for h, v in zip(hdr, cells) if v and h]
        print(f"{i} [{title}]\n    " + " | ".join(pairs))
