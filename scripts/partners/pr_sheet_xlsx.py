"""Write an updated PR downstream sheet (TSV from pr_sheet_check.py --write-sheet) back into the formatting of the
original Google-sheet tab, as a one-tab .xlsx to import into Google Sheets.

Inputs: the workbook downloaded from Google Sheets (.xlsx), the tab name, that tab's TSV export (same rows, used to
locate rows), and the updated TSV. Every output row takes the cell styles (fill, font, border, number format,
alignment), row height and hyperlinks of the source row it came from, so formatting travels with re-sorted rows:
  data rows      matched by (block PR, root_id_orig, xyz), else (block PR, xyz); added rows copy the row above them
  other lines    (block titles, headers, blank lines) matched in order
Columns are matched by header name; a dropped column (e.g. nsyn_old) disappears with its formatting and width.
Values keep the source cell's type: dates stay dates (yy-mm-dd), counts stay numbers, root ids stay text. Formulas in
the source tab are replaced by the updated values (rows move, so cell references would break). Only the one tab is
written; the workbook's other tabs are left out.

--like: for a sheet that never had its own tab (PRg2hop1, PRg1hop2), borrow the look of TAB instead: header, block
title, '>>>' / note and blank lines take the styles of their first counterpart in TAB; data cells take a plain data
row's styles by column (new columns borrow from the closest one: PR_root_id <- upstream_PR, hop1_root_id <-
root_id_orig, h1_nsyn <- nsyn, h1_conv <- PR_conv); rows for cells tracked in TAB (same root_id_orig and xyz) take
that row's styles, so e.g. a highlighted note stays highlighted.

Usage: python pr_sheet_xlsx.py BOOK.xlsx TAB TAB_EXPORT.tsv UPDATED.tsv --title PRg1hop1 [--out OUT.xlsx]
       python pr_sheet_xlsx.py BOOK.xlsx TAB TAB_EXPORT.tsv NEW.tsv --title PRg2hop1 --like
Network-free."""
import argparse
import copy
import datetime
import os
import re

import openpyxl
from openpyxl.utils import get_column_letter

from faab.config import FIGURES
from faab.partners.sheet import KEYS, norm_xyz, parse_sheet, utc_stamp

DATE = re.compile(r"^(\d\d)-(\d\d)-(\d\d)$")


def typed(text, src_value, col_kind):
    """the updated text as the type the source column holds"""
    if text == "":
        return None
    if col_kind == "date" or isinstance(src_value, datetime.datetime):
        m = DATE.match(text)
        if m:
            return datetime.datetime(2000 + int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return text
    if col_kind == "number":
        try:
            f = float(text)
            return int(f) if f.is_integer() else f
        except ValueError:
            return text
    return text


ALIAS = {"PR_root_id": "upstream_PR", "hop1_root_id": "root_id_orig", "h1_nsyn": "nsyn", "h1_conv": "PR_conv",
         "class": "PRtype"}


def column_kinds(src, s_rows):
    """which source header columns hold dates / numbers (dominant type of the data cells)"""
    s_header = s_rows[0]["header"]
    kind = {}
    for j, name in enumerate(s_header):
        vals = [src.cell(r["line"] + 1, j + 1).value for r in s_rows]
        vals = [v for v in vals if v not in (None, "")]
        if vals and sum(isinstance(v, datetime.datetime) for v in vals) >= len(vals) / 2:
            kind[name] = "date"
        elif vals and sum(isinstance(v, (int, float)) for v in vals) >= len(vals) / 2:
            kind[name] = "number"
        else:
            kind[name] = "text"
    return kind


def copy_sheet_setup(src, out):
    out.sheet_format = copy.copy(src.sheet_format)
    out.sheet_properties = copy.copy(src.sheet_properties)
    out.sheet_view.showGridLines = src.sheet_view.showGridLines
    out.sheet_view.zoomScale = src.sheet_view.zoomScale


def build_like(wb, src, s_lines, s_rows, n_lines, n_rows, title):
    """a new tab in the look of src (see --like in the module docstring)"""
    s_header = s_rows[0]["header"]
    kind = column_kinds(src, s_rows)
    s_data = {r["line"] for r in s_rows}

    def solid(r, c):
        f = src.cell(r, c).fill
        return f is not None and f.fill_type == "solid"

    plain = next(r["line"] + 1 for r in s_rows if not any(solid(r["line"] + 1, c) for c in range(1, len(s_header) + 1)))
    header_row = next(i + 1 for i, ln in enumerate(s_lines) if ln.split("\t")[0] in KEYS)
    title_row = next(i + 1 for i, ln in enumerate(s_lines) if re.match(r"^\d{15,}\s+\[", ln))
    first_row = next((i + 1 for i, ln in enumerate(s_lines) if ln.startswith(">>>")), title_row)
    blank_row = next(i + 1 for i, ln in enumerate(s_lines) if not ln.strip() and i not in s_data)
    tracked = {}
    for r in s_rows:
        c = r["cells"]
        tracked.setdefault((c["root_id_orig"].strip(), norm_xyz(c["xyz"])), r["line"] + 1)

    n_header = n_rows[0]["header"]
    src_col = [s_header.index(ALIAS.get(h, h)) + 1 if ALIAS.get(h, h) in s_header else None for h in n_header]
    out = wb.create_sheet(title)
    copy_sheet_setup(src, out)
    for j, sc in enumerate(src_col, 1):
        d = src.column_dimensions.get(get_column_letter(sc)) if sc else None
        if d is not None and d.width:
            out.column_dimensions[get_column_letter(j)].width = d.width
    if any("," in r["cells"][n_header[0]] for r in n_rows):  # PR_root_id lists several PRs
        out.column_dimensions["A"].width = 38
    data = {r["line"]: r for r in n_rows}
    n_tracked = 0
    for i, ln in enumerate(n_lines):
        cells = ln.split("\t")
        if i in data:
            c = data[i]["cells"]
            row = tracked.get((c["root_id_orig"].strip(), norm_xyz(c["xyz"])))
            n_tracked += row is not None
            style_row, kinds = row or plain, True
        else:
            first = cells[0].strip()
            style_row = (header_row if first in KEYS else first_row if first.startswith((">>>", "note:")) else
                         blank_row if not first else title_row)
            kinds = False
        for j, text in enumerate(cells):
            name = n_header[j] if j < len(n_header) else ""
            sc = src_col[j] if j < len(src_col) else None
            scell = src.cell(style_row, sc) if sc else None
            v = typed(text.strip(), scell.value if (scell is not None and kinds) else None,
                      kind.get(ALIAS.get(name, name), "text") if kinds else "text")
            oc = out.cell(i + 1, j + 1, v)
            if scell is not None:
                oc._style = copy.copy(scell._style)
            if isinstance(v, datetime.datetime):
                oc.number_format = 'yy"-"mm"-"dd'
            if kinds and name == "link" and isinstance(v, str) and v.startswith("http"):
                oc.hyperlink = v
    return out, f"{len(n_rows)} data rows, {n_tracked} styled from their tracked row in {src.title}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("book")
    ap.add_argument("tab")
    ap.add_argument("tab_export")
    ap.add_argument("updated")
    ap.add_argument("--title", required=True, help="name of the output tab, e.g. PRg1hop1")
    ap.add_argument("--like", action="store_true", help="new sheet: borrow the look of TAB instead of its rows")
    ap.add_argument("--out")
    a = ap.parse_args()

    wb = openpyxl.load_workbook(a.book)
    src = wb[a.tab]
    s_lines, s_rows = parse_sheet(a.tab_export)
    n_lines, n_rows = parse_sheet(a.updated)
    if a.like:
        out, msg = build_like(wb, src, s_lines, s_rows, n_lines, n_rows, a.title)
        for ws in list(wb.worksheets):
            if ws is not out:
                wb.remove(ws)
        path = a.out or os.path.join(FIGURES, f"{utc_stamp()}_FAABopticlobetargets-{a.title}.xlsx")
        wb.save(path)
        print(f"wrote {path}: tab {a.title}, {len(n_lines)} lines, {msg}")
        return

    s_header = s_rows[0]["header"]
    kind = column_kinds(src, s_rows)

    # data rows: source row for each updated row
    by_key, by_xyz = {}, {}
    for r in s_rows:
        c = r["cells"]
        by_key.setdefault((r["block_pr"], c["root_id_orig"].strip(), norm_xyz(c["xyz"])), r["line"])
        by_xyz.setdefault((r["block_pr"], norm_xyz(c["xyz"])), []).append(r["line"])
    src_line, added = {}, []
    prev = None
    for r in n_rows:
        c = r["cells"]
        k = (r["block_pr"], c["root_id_orig"].strip(), norm_xyz(c["xyz"]))
        line = by_key.get(k)
        if line is None and len(by_xyz.get(k[::2], [])) == 1:
            line = by_xyz[k[::2]][0]
        if line is None:
            line = prev
            added.append((r["line"] + 1, c["root_id_curr"]))
        src_line[r["line"]] = line
        prev = line
    # other lines, in order
    s_other = [i for i, ln in enumerate(s_lines) if i not in {r["line"] for r in s_rows}]
    n_other = [i for i, ln in enumerate(n_lines) if i not in {r["line"] for r in n_rows}]
    if len(s_other) != len(n_other):
        raise SystemExit(f"non-data lines differ in number ({len(s_other)} vs {len(n_other)}): layouts do not match")
    src_line.update(dict(zip(n_other, s_other)))

    # column map by header name (new column j -> source column)
    n_header = n_rows[0]["header"]
    col_src = [s_header.index(h) for h in n_header]
    width_extra = max(len(ln.split("\t")) for ln in n_lines) - len(n_header)
    col_src += list(range(len(s_header), len(s_header) + max(0, width_extra)))

    out = wb.create_sheet(a.title)
    copy_sheet_setup(src, out)
    for j, sj in enumerate(col_src):
        d = src.column_dimensions.get(get_column_letter(sj + 1))
        if d is not None and d.width:
            out.column_dimensions[get_column_letter(j + 1)].width = d.width
    data_lines = {r["line"]: r for r in n_rows}
    formulas = 0
    for i, ln in enumerate(n_lines):
        cells = ln.split("\t")
        si = src_line.get(i)
        if si is not None and src.row_dimensions[si + 1].height:
            out.row_dimensions[i + 1].height = src.row_dimensions[si + 1].height
        for j, text in enumerate(cells[:len(col_src)]):
            sc = src.cell(si + 1, col_src[j] + 1) if si is not None else None
            name = n_header[j] if j < len(n_header) else ""
            v = typed(text.strip(), sc.value if sc is not None else None,
                      kind.get(name, "text") if i in data_lines else "text")
            if sc is not None and isinstance(sc.value, str) and sc.value.startswith("="):
                formulas += 1
            oc = out.cell(i + 1, j + 1, v)
            if sc is not None:
                oc._style = copy.copy(sc._style)
                if sc.hyperlink is not None and v:
                    oc.hyperlink = sc.hyperlink.target or v
            if isinstance(v, datetime.datetime) and (sc is None or not isinstance(sc.value, datetime.datetime)):
                oc.number_format = 'yy"-"mm"-"dd'
    for ws in list(wb.worksheets):
        if ws is not out:
            wb.remove(ws)
    path = a.out or os.path.join(FIGURES, f"{utc_stamp()}_FAABopticlobetargets-{a.title}.xlsx")
    wb.save(path)
    print(f"wrote {path}: tab {a.title}, {len(n_lines)} lines, {len(n_rows)} data rows; "
          f"{len(added)} added rows styled like the row above: {added}; {formulas} source formulas replaced by values")


if __name__ == "__main__":
    main()
