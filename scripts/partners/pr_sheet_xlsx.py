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

Usage: python pr_sheet_xlsx.py BOOK.xlsx TAB TAB_EXPORT.tsv UPDATED.tsv --title PRg1hop1 [--out OUT.xlsx]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("book")
    ap.add_argument("tab")
    ap.add_argument("tab_export")
    ap.add_argument("updated")
    ap.add_argument("--title", required=True, help="name of the output tab, e.g. PRg1hop1")
    ap.add_argument("--out")
    a = ap.parse_args()

    wb = openpyxl.load_workbook(a.book)
    src = wb[a.tab]
    s_lines, s_rows = parse_sheet(a.tab_export)
    n_lines, n_rows = parse_sheet(a.updated)

    # which source header columns are dates / numbers (dominant type of the data cells)
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
    out.sheet_format = copy.copy(src.sheet_format)
    out.sheet_properties = copy.copy(src.sheet_properties)
    out.sheet_view.showGridLines = src.sheet_view.showGridLines
    out.sheet_view.zoomScale = src.sheet_view.zoomScale
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
