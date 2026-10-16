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


def write_blocks(path, preamble, blocks, width):
    """preamble: lines of text for column A ('>>> group N', notes); blocks: list of (title or None, header list, list of
    row lists). Pads every line to `width` cells like the export."""
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
    open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")
