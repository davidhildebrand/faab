"""The PRs tab of the 'FAAB optic lobe targets' Google sheet, downloaded as .xlsx: layout, member order, formatting.

Layout (column A): a title line; the proofreading targets (groups 1-8); a 'NOT PROOFREADING TARGETS (YET)' line; the
leads (groups 9+). Each group is a 'group N' line, a header row starting with 'root_id_orig', member rows (column A is
a root id) and a blank line. Columns are found by header name (the targets have an extra 'ring_check (...)' column).

David's conventions, applied by the writers in scripts/pr_groups/prs_tab.py and checked by lint():
  - members in order: R1-R6 by pairing (R1/R6, R2/R5, R3/R4, partners on adjacent rows; pairs without a fly name next;
    unpaired R1-R6 after), then R7, then R8, then everything else in its order (OFF-BUNDLE / SECOND BUNDLE rows last);
  - 'group N' lines and header rows bold;
  - no 'regroup YYYY-MM-DD: ...' notes;
  - dates are real dates shown as yy-mm-dd, never text (David, 2026-10-18); root ids stored as text, scores as numbers;
    rows move with their styles, links and heights;
  - outputs hold the PRs tab alone and are named with the UTC stamp (faab.partners.sheet.utc_stamp).
Network-free."""
import copy
import datetime
import os
import re
from dataclasses import dataclass, field

import openpyxl

from faab.config import FIGURES
from faab.partners.sheet import utc_stamp

TAB = "PRs"
LEADS = "NOT PROOFREADING"  # the line that starts the leads section
PAIR_RANK = {"R1/R6": 0, "R2/R5": 1, "R3/R4": 2, "R1/R6 or R3/R4": 3}
ASIDE = ("OFF-BUNDLE", "SECOND BUNDLE")  # rows with these notes are not members of the bundle
ID_COLS = ("root_id_orig", "root_id_curr", "partner_root_id_orig", "partner_root_id_curr")
DEFAULT_OUT = os.path.join(FIGURES, "{stamp}_FAABopticlobetargets-PRs.xlsx")
DATE_FMT = 'yy"-"mm"-"dd'  # how the sheet shows dates
DATE_TEXT = [  # text forms of a date -> (year, month, day)
    (re.compile(r"(\d\d)-(\d\d)-(\d\d)"), lambda m: (2000 + int(m[1]), int(m[2]), int(m[3]))),  # yy-mm-dd, as shown
    (re.compile(r"(\d{4})-(\d\d)-(\d\d)"), lambda m: (int(m[1]), int(m[2]), int(m[3]))),
    (re.compile(r"(\d{1,2})/(\d{1,2})/(\d\d|\d{4})"),
     lambda m: (int(m[3]) + (2000 if len(m[3]) == 2 else 0), int(m[1]), int(m[2]))),  # m/d/yy, m/d/yyyy
]


@dataclass
class Block:
    group: int
    section: str          # "targets" (above the LEADS line) or "leads"
    title_row: int        # the 'group N' line
    header_row: int
    cols: dict            # header name -> column number
    rows: list = field(default_factory=list)  # member rows (contiguous, below the header)


def load(path):
    wb = openpyxl.load_workbook(path)
    return wb, wb[TAB]


def text(ws, r, c):
    v = ws.cell(r, c).value if c else None
    return "" if v is None else str(v).strip()


def blocks(ws):
    res, group, title, section, cur = [], None, None, "targets", None
    for r in range(1, ws.max_row + 1):
        a = text(ws, r, 1)
        if a.startswith(LEADS):
            section = "leads"
        m = re.match(r"^group (\d+)$", a)
        if m:
            group, title, cur = int(m.group(1)), r, None
            continue
        if a == "root_id_orig":
            cur = Block(group, section, title, r, {text(ws, r, c): c for c in range(1, ws.max_column + 1) if text(ws, r, c)})
            res.append(cur)
            continue
        if cur is not None and a.isdigit() and (not cur.rows or r == cur.rows[-1] + 1):
            cur.rows.append(r)
        elif cur is not None and cur.rows:
            cur = None
    return res


def get(ws, b, r, name):
    return text(ws, r, b.cols.get(name))


def member_order(items, fields):
    """items in David's member order (module docstring); fields(item) -> (class, root, notes, fly name, partner root),
    the partner root in the same id space as root."""
    f = [fields(x) for x in items]
    member = [not any(t in f[i][2] for t in ASIDE) for i in range(len(f))]
    r16 = [i for i in range(len(f)) if f[i][0] == "R1-R6" and member[i]]
    by_root = {f[i][1]: i for i in r16}
    done, paired = set(), []
    for i in sorted(r16, key=lambda i: (PAIR_RANK.get(f[i][3], 9), f[i][1])):
        j = by_root.get(f[i][4]) if f[i][4] else None  # a pair without a fly name yet still sits together (rank 9)
        if j is not None and j != i and not {i, j} & done and f[j][4] == f[i][1]:
            paired += [i, j]
            done |= {i, j}
    rank, in16 = {"R7": 0, "R8": 1}, set(r16)
    rest = sorted((i for i in range(len(f)) if i not in in16), key=lambda i: rank.get(f[i][0], 2) if member[i] else 3)
    return [items[i] for i in paired + [i for i in r16 if i not in done] + rest]


def fly_name(ws, b, r):
    """the row's fly pair name: the 'fly_pair = X' of its manual fly_pair check when present, else fly_pair"""
    m = re.search(r"fly_pair\s*=\s*(R\d/R\d)", get(ws, b, r, "manual fly_pair check"))
    return m.group(1) if m else get(ws, b, r, "fly_pair")


def ordered_rows(ws, b):
    return member_order(b.rows, lambda r: (get(ws, b, r, "class"), get(ws, b, r, "root_id_orig"), get(ws, b, r, "notes"),
                                           fly_name(ws, b, r), get(ws, b, r, "partner_root_id_orig")))


def snapshot(ws, r):
    cells = [(x.value, copy.copy(x._style), x.hyperlink.target if x.hyperlink else None)
             for x in (ws.cell(r, c) for c in range(1, ws.max_column + 1))]
    return cells, ws.row_dimensions[r].height


def restore(ws, r, snap):
    cells, height = snap
    for c, (v, st, link) in enumerate(cells, 1):
        x = ws.cell(r, c)
        x.hyperlink = link  # before the value: setting a link on an empty cell would show the link as its value
        x.value, x._style = v, copy.copy(st)
    ws.row_dimensions[r].height = height


def order_blocks(ws):
    """put every block's members in David's order, in place (rows keep their styles, links, heights);
    returns (blocks changed, rows moved)"""
    nb = nr = 0
    for b in blocks(ws):
        new = ordered_rows(ws, b)
        if new == b.rows:
            continue
        snap = {r: snapshot(ws, r) for r in b.rows}
        for dst, src in zip(b.rows, new):
            restore(ws, dst, snap[src])
        nb, nr = nb + 1, nr + sum(d != s for d, s in zip(b.rows, new))
    return nb, nr


def title_rows(ws):
    return [r for b in blocks(ws) for r in (b.title_row, b.header_row) if r]


def bold_titles(ws):
    """make the non-empty cells of every 'group N' line and header row bold; returns the number of cells changed"""
    n = 0
    for r in title_rows(ws):
        for c in range(1, ws.max_column + 1):
            x = ws.cell(r, c)
            if x.value not in (None, "") and not x.font.b:
                f = copy.copy(x.font)
                f.b = True
                x.font, n = f, n + 1
    return n


def tag_pattern(tag):
    return re.compile(rf"(^|;\s*){re.escape(tag)} \d{{4}}-\d{{2}}-\d{{2}}: [^;]*")


def strip_tagged_notes(ws, tags=("regroup",)):
    """remove '<tag> YYYY-MM-DD: ...' segments from the notes cells; returns the number of cells changed"""
    pats, n = [tag_pattern(t) for t in tags], 0
    for b in blocks(ws):
        c = b.cols.get("notes")
        for r in b.rows if c else ():
            v = ws.cell(r, c).value
            if isinstance(v, str) and any(p.search(v) for p in pats):  # other cells stay exactly as they are
                new = v
                for p in pats:
                    new = p.sub("", new)
                ws.cell(r, c).value, n = new.strip().lstrip(";").strip() or None, n + 1
    return n


def as_date(s):
    """a date written as text -> datetime (None if s is not a valid date in one of the DATE_TEXT forms)"""
    for p, ymd in DATE_TEXT:
        m = p.fullmatch(str(s).strip())
        if m:
            try:
                return datetime.datetime(*ymd(m))
            except ValueError:
                return None
    return None


def date_cols(b):
    return [c for name, c in b.cols.items() if name.endswith("date")]


def convert_text_dates(ws):
    """text dates in the date columns (header ending in 'date') -> real dates shown as yy-mm-dd, cell style otherwise
    kept; returns (cells converted, [cells holding text that is not a date, left alone])"""
    n, left = 0, []
    for b in blocks(ws):
        for c in date_cols(b):
            for r in b.rows:
                x = ws.cell(r, c)
                if isinstance(x.value, str) and x.value.strip():
                    d = as_date(x.value)
                    if d is None:
                        left.append(x.coordinate)
                    else:
                        x.value, x.number_format, n = d, DATE_FMT, n + 1
    return n, left


def tidy(ws, tags=("regroup",)):
    """apply every convention that can be fixed mechanically: dates, tagged notes, member order, bold titles"""
    n_dates, left = convert_text_dates(ws)
    n_notes = strip_tagged_notes(ws, tags) if tags else 0
    nb, nr = order_blocks(ws)
    return dict(dates_converted=n_dates, date_text_left=left, notes_stripped=n_notes, groups_reordered=nb,
                rows_moved=nr, cells_bolded=bold_titles(ws))


def lint(ws, tags=("regroup",)):
    """{issue: [where, ...]} for every convention the tab breaks (empty dict = all good)"""
    out = {}
    add = lambda k, v: out.setdefault(k, []).append(v)
    for b in blocks(ws):
        for c in date_cols(b):
            for r in b.rows:
                v = ws.cell(r, c).value
                if isinstance(v, str) and v.strip():
                    add("date stored as text" if as_date(v) else "date column holds text", ws.cell(r, c).coordinate)
    for r in title_rows(ws):
        if any(ws.cell(r, c).value not in (None, "") and not ws.cell(r, c).font.b for c in range(1, ws.max_column + 1)):
            add("group line / header not bold", f"row {r}")
    pats = [tag_pattern(t) for t in tags]
    for b in blocks(ws):
        if ordered_rows(ws, b) != b.rows:
            add("members out of order", f"group {b.group}")
        seen = set()
        for r in b.rows:
            o = get(ws, b, r, "root_id_orig")
            if o in seen:
                add("root_id_orig twice in a group", f"group {b.group} row {r}")
            seen.add(o)
            if any(p.search(get(ws, b, r, "notes")) for p in pats):
                add(f"{'/'.join(tags)} notes", f"group {b.group} row {r}")
            for k in ID_COLS:
                if k in b.cols and isinstance(ws.cell(r, b.cols[k]).value, (int, float)):
                    add("root id stored as a number", f"{ws.cell(r, b.cols[k]).coordinate}")
            if len(re.findall(r"-?\d+(?:\.\d+)?", get(ws, b, r, "xyz"))) != 3:
                add("xyz unreadable", f"group {b.group} row {r}")
    return out


def lint_line(ws):
    issues = lint(ws)
    if not issues:
        return ("lint: ok (members in order, group lines and headers bold, no regroup notes, dates are dates, ids text, "
                "xyz readable)")
    return "lint: " + "; ".join(f"{k}: {len(v)} ({', '.join(v[:4])}{', ...' if len(v) > 4 else ''})"
                                for k, v in issues.items())


def save(wb, ws, out=None):
    """save the PRs tab alone ('{stamp}' in the path -> UTC stamp); returns the path"""
    path = (out or DEFAULT_OUT).replace("{stamp}", utc_stamp())
    for other in list(wb.worksheets):
        if other is not ws:
            wb.remove(other)
    wb.save(path)
    return path
