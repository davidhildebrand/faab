"""What changed in a new backup of the tracking workbook since the tabs were last delivered, keyed by root_id (David
2026-11-07: no row numbers, root_id lists). DELIVERED = the synced workbook of the last update (all tabs), LMCS = the
LMCs tab delivered with it (its own file when it was edited separately; else the same synced workbook), BEFORE = the
backup that update started from. Rows are keyed by (root_id_orig, n-th row of that cell in the tab); values by column
name, so moved columns or inserted rows don't show as changes. Reports whether the delivered tabs were imported (rows
only in old / only in new, rows matching the delivered tab vs the earlier backup), every changed value with the row's
proofreader / checker ('NO NAME' when both are blank), and the rows carrying an 'xyz off the cell' flag.
Writes OUT (full) and prints a summary.
Usage: backup_diff_by_root.py NEW.xlsx DELIVERED.xlsx LMCS.xlsx BEFORE.xlsx OUT.txt"""
import collections
import re
import sys

import openpyxl

from faab.partners.sheet import cell_text as s

if len(sys.argv) != 6:
    sys.exit(__doc__)
NEW, SYNCED, LMCS, BEFORE, OUT = sys.argv[1:6]
SKIP_COLS = {"link"}
WORK = ("root_id_curr", "xyz", "status", "notes", "pr_date", "annotation", "putative_type", "confirmed_type",
        "confidence", "reason", "chk_date")


def fill(c):
    return c.fill.fgColor.rgb if c.fill is not None and c.fill.fill_type == "solid" and isinstance(c.fill.fgColor.rgb, str) else ""


def rows(ws):
    """{(root_id_orig, n): {column: value, '_fill': fill of the root_id_orig cell}}"""
    out, hdr, seen = {}, None, collections.Counter()
    for r in range(1, ws.max_row + 1):
        vals = [s(ws.cell(r, c).value) for c in range(1, ws.max_column + 1)]
        if "root_id_orig" in vals:
            hdr = {v: i for i, v in enumerate(vals) if v}
            continue
        i0 = hdr.get("root_id_orig") if hdr else None
        if i0 is not None and i0 < len(vals) and vals[i0].isdigit():
            x = {k: vals[i] for k, i in hdr.items() if i < len(vals)}
            x["_fill"] = fill(ws.cell(r, i0 + 1))
            k = (x["root_id_orig"], seen[x["root_id_orig"]])
            seen[x["root_id_orig"]] += 1
            out[k] = x
    return out


new_wb = openpyxl.load_workbook(NEW)
syn_wb = openpyxl.load_workbook(SYNCED)
lmc_wb = openpyxl.load_workbook(LMCS)
bef_wb = openpyxl.load_workbook(BEFORE)
skip = lambda t: re.search(r"_b\d{6}d$", t)
lines, summary = [], []
lines.append(f"tabs in new: {[t for t in new_wb.sheetnames if not skip(t)]}")
lines.append(f"tabs only in the synced book: {[t for t in syn_wb.sheetnames if t not in new_wb.sheetnames and not skip(t)]}")
lines.append(f"tabs only in new: {[t for t in new_wb.sheetnames if t not in syn_wb.sheetnames and not skip(t)]}")
for t in [t for t in new_wb.sheetnames if not skip(t)]:
    src = lmc_wb if t == "LMCs" else syn_wb
    if t not in src.sheetnames:
        continue
    old, new = rows(src[t]), rows(new_wb[t])
    bef = rows(bef_wb[t]) if t in bef_wb.sheetnames else {}
    # imported? compare new against both the delivered tab and the morning backup
    same_del = sum(1 for k in new if k in old and all(new[k].get(c, "") == old[k].get(c, "") for c in WORK))
    same_bef = sum(1 for k in new if k in bef and all(new[k].get(c, "") == bef[k].get(c, "") for c in WORK))
    only_old, only_new = sorted(set(old) - set(new)), sorted(set(new) - set(old))
    ch = collections.defaultdict(list)
    for k in [k for k in new if k in old]:
        a, b = old[k], new[k]
        for c in sorted(set(a) | set(b)):
            if c in SKIP_COLS:
                continue
            if a.get(c, "") != b.get(c, ""):
                ch[k].append((c, a.get(c, ""), b.get(c, "")))
    summary.append(f"{t}: rows delivered {len(old)}, now {len(new)}; matching the delivered tab {same_del}, matching the "
                   f"morning backup {same_bef}; only delivered {len(only_old)}, only now {len(only_new)}; "
                   f"cells changed {len(ch)}")
    lines.append(f"\n== {t}: " + summary[-1].split(": ", 1)[1])
    for k in only_old:
        lines.append(f"  only in the delivered tab: {k[0]} (#{k[1] + 1}) status {old[k].get('status', '')!r}")
    for k in only_new:
        lines.append(f"  only now: {k[0]} (#{k[1] + 1}) status {new[k].get('status', '')!r} notes {new[k].get('notes', '')[:80]!r}")
    for k, cs in ch.items():
        b = new[k]
        who = b.get("proofreader", "") or b.get("checker", "")
        lines.append(f"  {k[0]} (#{k[1] + 1}) [proofreader {b.get('proofreader', '')!r}, checker {b.get('checker', '')!r}]"
                     + ("  NO NAME" if not who.strip() else ""))
        for c, x, y in cs:
            lines.append(f"      {c}: {x[:90]!r} -> {y[:90]!r}")
    # yellow 'xyz off the cell' rows now
    for k, x in new.items():
        m = re.search(r"xyz off the cell \(on (\d+)\)", x.get("notes", ""))
        if m:
            lines.append(f"  off-cell row: {k[0]} curr {x.get('root_id_curr')} xyz {x.get('xyz')} on {m.group(1)}; "
                         f"proofreader {x.get('proofreader', '')!r}")
open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(summary))
print(f"written {OUT} ({len(lines)} lines)")
