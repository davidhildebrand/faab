"""Read and write the PRdownstream tracking sheets (Google-sheet TSV exports in block layout).

Layout: an optional '>>> group N' line; per block an optional title line ('<PR root>  [R1–R6]  ...'), a header row
whose first cell is the block key column (upstream_PR, PR_root_id), data rows, a blank line. Lines are kept verbatim
so an updated copy differs only in the cells that were changed."""
import re

from faab.clients.cave import parse_xyz

KEYS = ("upstream_PR", "PR_root_id")
STAMP = "%Y%m%dd%H%M%St"   # + "UTC": David's datetime stamp, always in UTC (e.g. 20261016d031000tUTC)


def utc_stamp(when=None):
    import datetime
    when = when or datetime.datetime.now(datetime.timezone.utc)
    return when.astimezone(datetime.timezone.utc).strftime(STAMP) + "UTC"
TRACK = ["annotation", "status", "pr_date", "proofreader", "chk_date", "checker", "notes", "SeaTable", "link"]
SEATABLE_CLS = ("superclass", "class", "subclass", "subsubclass", "type")


def cell_text(v):
    """workbook / SeaTable cell -> text (dates yy-mm-dd, whole floats as ints, empty and NaN -> '')"""
    import datetime
    import math
    if v is None or (isinstance(v, float) and math.isnan(v)) or str(v) in ("<NA>", "nan"):
        return ""
    if isinstance(v, datetime.datetime):
        return f"{v:%y-%m-%d}"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def seatable_label(row):
    """'superclass / class / subclass / subsubclass / type' of a SeaTable row (dict or namedtuple), blanks skipped"""
    get = row.get if isinstance(row, dict) else (lambda c: getattr(row, c, ""))
    return " / ".join(x for x in (cell_text(get(c)) for c in SEATABLE_CLS) if x)


def scan_book(wb):
    """every tab row that has an xyz: dict(tab, row, cols {header: col}, values {header: value}); a row is read under the
    nearest header row above it (a row naming xyz and root_id_curr or root_id_orig)"""
    rows = []
    for ws in wb.worksheets:
        cols = None
        for r in range(1, ws.max_row + 1):
            vals = [ws.cell(r, c).value for c in range(1, ws.max_column + 1)]
            names = [cell_text(v) for v in vals]
            if "xyz" in names and ("root_id_curr" in names or "root_id_orig" in names):
                cols = {n: i + 1 for i, n in enumerate(names) if n}
                continue
            if cols is None:
                continue
            if len(re.findall(r"-?\d+(?:\.\d+)?", cell_text(vals[cols["xyz"] - 1]))) != 3:
                continue
            rows.append(dict(tab=ws.title, row=r, cols=cols, values={n: vals[c - 1] for n, c in cols.items()}))
    return rows


def parse_sheet(path):
    """(lines, rows): rows are dicts line (0-based), block_pr, cells (header -> value), header"""
    lines = open(path, encoding="utf-8").read().split("\n")
    rows, header = [], None
    for i, ln in enumerate(lines):
        c = ln.split("\t")
        if c[0] in KEYS:
            header = c
            continue
        if header and re.match(r"^\d{15,}(,\s*\d{15,})*$", c[0].strip()) and len(c) >= 5:
            d = dict(zip(header, c + [""] * (len(header) - len(c))))
            rows.append(dict(line=i, block_pr=d[header[0]], cells=d, header=header))
    return lines, rows


def as_int(s, n):
    """sheet nsyn cell -> int (blank / text -> n)"""
    s = str(s).strip()
    return int(float(s)) if re.fullmatch(r"\d+(\.\d+)?", s) else n


def fmt_vox(p_vox):
    return ", ".join(str(int(round(v))) for v in p_vox)


def norm_xyz(s):
    """'8270,  18350, 4638' -> '8270, 18350, 4638' (None if unreadable)"""
    v = parse_xyz(s)
    return fmt_vox(v) if v else None


def write_blocks(path, preamble, blocks, width, notes=()):
    """preamble: lines of text for column A at the top ('>>> group N'); blocks: list of (title or None, header list,
    list of row lists); notes: 'note: ...' lines written below the last block (David, 2026-10-17: notes go at the end).
    Pads every line to `width` cells like the export."""
    def line(cells):
        cells = [str(c) for c in cells]
        return "\t".join(cells + [""] * (width - len(cells)))
    out = [line([])] + [line([p]) for p in ([preamble] if isinstance(preamble, str) else preamble)]
    for title, header, rows in blocks:
        if title:
            out.append(line([title]))
        out.append(line(header))
        out += [line(r) for r in rows]
        out.append(line([]))
    out += [line([n]) for n in notes]
    open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")
