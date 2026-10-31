"""Network-free diff of two workbook backups, tab by tab (backup tabs '*_bYYMMDDd' skipped): every changed cell value
and fill, by row and the column header of its block, plus tabs and row counts added or removed. Rows are compared at
the same row number (an inserted row shows as a run of changes below it). Usage: book_diff.py OLD.xlsx NEW.xlsx [MAX]"""
import re
import sys

import openpyxl

from faab.partners.sheet import cell_text as s

old, new = openpyxl.load_workbook(sys.argv[1]), openpyxl.load_workbook(sys.argv[2])
cap = int(sys.argv[3]) if len(sys.argv) > 3 else 300
skip = lambda t: re.search(r"_b\d{6}d$", t)
print(f"tabs only in old: {[t for t in old.sheetnames if t not in new.sheetnames and not skip(t)]}; "
      f"only in new: {[t for t in new.sheetnames if t not in old.sheetnames and not skip(t)]}")


def fill(c):
    return c.fill.fgColor.rgb if c.fill is not None and c.fill.fill_type == "solid" and isinstance(c.fill.fgColor.rgb, str) else ""


for t in [t for t in new.sheetnames if t in old.sheetnames and not skip(t)]:
    a, b = old[t], new[t]
    n_rows = max(a.max_row, b.max_row)
    n_cols = max(a.max_column, b.max_column)
    hdr, out = {}, []
    for r in range(1, n_rows + 1):
        names = [s(b.cell(r, c).value) for c in range(1, n_cols + 1)]
        if "xyz" in names:
            hdr = {c + 1: n for c, n in enumerate(names) if n}
        for c in range(1, n_cols + 1):
            x, y = a.cell(r, c), b.cell(r, c)
            if s(x.value) != s(y.value):
                out.append(f"  r{r} {hdr.get(c, c)}: '{s(x.value)[:70]}' -> '{s(y.value)[:70]}'")
            elif fill(x) != fill(y):
                out.append(f"  r{r} {hdr.get(c, c)}: fill {fill(x) or 'none'} -> {fill(y) or 'none'}")
    last = lambda ws: max((r for r in range(1, ws.max_row + 1) if any(s(ws.cell(r, c).value) for c in range(1, 6))), default=0)
    print(f"\n{t}: {len(out)} changed cells; last filled row {last(a)} -> {last(b)}")
    print("\n".join(out[:cap]) + (f"\n  ... {len(out) - cap} more" if len(out) > cap else ""))
