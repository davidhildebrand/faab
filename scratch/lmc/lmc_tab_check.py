"""Checks of a written LMCs tab (2026-11-01): one row per cell (root_id_curr unique), every row with a readable xyz,
block lines with a hyperlink, title / block / header lines bold, ids stored as text. Network-free.
Usage: lmc_tab_check.py LMCs.xlsx"""
import sys

import openpyxl

ws = openpyxl.load_workbook(sys.argv[1])["LMCs"]
roots, no_xyz, blocks, links, not_bold, num_ids = [], 0, 0, 0, 0, 0
for r in range(1, ws.max_row + 1):
    a = ws.cell(r, 1).value
    if a is None:
        continue
    s = str(a)
    if s.isdigit():
        roots.append(str(ws.cell(r, 2).value))
        no_xyz += str(ws.cell(r, 3).value or "").count(",") != 2
        num_ids += any(isinstance(ws.cell(r, c).value, (int, float)) for c in (1, 2))
    elif s.startswith(("putative", "LMC, type")):
        blocks += 1
        links += any(ws.cell(r, c).hyperlink is not None for c in range(1, ws.max_column + 1))
        not_bold += not ws.cell(r, 1).font.b
    elif s.startswith((">>>", "root_id_orig")):
        not_bold += not ws.cell(r, 1).font.b
print(f"rows {len(roots)}, unique cells {len(set(roots))}, without xyz {no_xyz}, ids stored as numbers {num_ids}; "
      f"blocks {blocks}, with a link {links}; title/header lines not bold {not_bold}")
