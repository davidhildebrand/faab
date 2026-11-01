"""Tools for the PRs tab (photoreceptor groups) of the 'FAAB optic lobe targets' sheet, downloaded as .xlsx.

Every writer reads IN.xlsx, changes only what it says, keeps each cell's formatting, saves the PRs tab alone to -o OUT
(default figures/<UTC stamp>_FAABopticlobetargets-PRs.xlsx; '{stamp}' in OUT -> UTC stamp) and prints a lint line
(the conventions are listed in src/faab/pr/prs_tab.py). Import the output with File > Import > "Insert new sheet(s)".

network-free
  tidy IN                   every mechanical convention fix at once: dates, strip-notes, order, bold
  dates IN                  text dates in pr_date / chk_date -> real dates shown as yy-mm-dd
  order IN                  members in David's order (R1-R6 by pairing, R7, R8, rest); rows move with their formatting
  bold IN                   'group N' lines and header rows bold
  strip-notes IN            remove '<tag> YYYY-MM-DD: ...' note segments (--tag, default regroup)
  add-note IN G:ROOT=TEXT   append TEXT to the notes of the row with root_id_orig ROOT in group G (repeatable)
  add-notes IN TSV          the same from a TSV (group, root_id_orig, note), e.g. groups_refine.py medulla_notes.tsv
  drop-rows IN ROOT ...     remove the member rows of these cells (root_id_orig or root_id_curr) from every group
  move-rows IN ROOT:G[=NOTE] ...  move member rows to the end of group G (NOTE appended); a block left without members
                            is removed
  replace-group IN PASTE G ...  rewrite only these groups' member rows from a groups_refine.py paste (rest untouched)
  apply-refine IN PASTE     rewrite the leads section (groups 9+) from a groups_refine.py paste
                            (figures/pr_groups9plus_refined_paste.tsv); a row keeps its own formatting and tracking
                            values; then tidy (regroup notes kept with --keep-regroup-notes)
  export-groups IN OUT.tsv  groups as a tidy TSV for pr_ring_plot.py --groups-file (the targets unless --all)
  ring IN                   ring-naming check of the targets (figures/pr_ring_plot_g<g>_names.tsv) into the
                            'ring_check (<date>)' column; --run first runs pr_ring_plot.py per group (CAVE, serial)
  pr-groups IN              rebuild data/pr_groups.tsv (R1-R6, R7, R8 members in tab order)
  lint IN                   list convention breaks
  diff A B                  what changed between two PRs tabs (member rows matched by group + root_id_orig)
read-only CAVE, calls >= 5 s apart
  update IN                 tidy + refresh-roots: the usual update of a freshly downloaded PRs tab
  refresh-roots IN          root_id_curr / partner_root_id_curr <- the root at xyz / partner_xyz now
  verify IN                 every root pair lies on its point, is current and descends from root_id_orig
  fix-points IN G:ROOT ...  move these rows' xyz onto their own axon (nearest L2 chunk); mirror into partner columns

Usual update: update, then verify. After a groups 9+ refinement: apply-refine, then verify.
Example: venv/bin/python scripts/pr_groups/prs_tab.py order figures/<stamp>_FAABopticlobetargets-PRs.xlsx"""
import argparse
import copy
import datetime
import difflib
import json
import os
import re
import subprocess
import sys
from collections import Counter

import numpy as np
import pandas as pd

from faab.config import DATA, FIGURES, TMP, VOX
from faab.partners import short_notes
from faab.partners.sheet import fmt_vox, norm_xyz
from faab.pr import prs_tab as T
from faab.config import fig

RING_PLOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pr_ring_plot.py")
LINEAGE_CACHE = os.path.join(TMP, "pr_sheet_check_lineage.json")
TRACK = ("status", "pr_date", "proofreader", "chk_date", "checker", "link")  # tab values win in apply-refine


def today():
    return datetime.datetime.now(datetime.timezone.utc).date().isoformat()


def finish(wb, ws, a):
    print("wrote", T.save(wb, ws, a.out))
    print(T.lint_line(ws))


def row_of(ws, spec):
    """'G:ROOT' -> (block, row) of root_id_orig ROOT in group G"""
    g, root = spec.split(":", 1)
    for b in T.blocks(ws):
        if b.group == int(g):
            for r in b.rows:
                if T.get(ws, b, r, "root_id_orig") == root.strip():
                    return b, r
    sys.exit(f"no row with root_id_orig {root} in group {g}")


# ---------------------------------------------------------------- network-free
def report_tidy(s):
    print(f"tidy: {s['dates_converted']} text dates converted"
          f"{' (left as text, not a date: ' + ', '.join(s['date_text_left']) + ')' if s['date_text_left'] else ''}; "
          f"tagged notes removed from {s['notes_stripped']} cells; members reordered in {s['groups_reordered']} groups "
          f"({s['rows_moved']} rows moved); {s['cells_bolded']} title/header cells made bold")


def cmd_tidy(a):
    wb, ws = T.load(a.book)
    report_tidy(T.tidy(ws))
    finish(wb, ws, a)


def cmd_dates(a):
    wb, ws = T.load(a.book)
    n, left = T.convert_text_dates(ws)
    print(f"{n} text dates converted to dates; text left alone (not a date): {left or 'none'}")
    finish(wb, ws, a)


def cmd_order(a):
    wb, ws = T.load(a.book)
    nb, nr = T.order_blocks(ws)
    print(f"{len(T.blocks(ws))} groups; reordered {nb} ({nr} rows moved)")
    finish(wb, ws, a)


def cmd_bold(a):
    wb, ws = T.load(a.book)
    print(f"{T.bold_titles(ws)} cells of group lines / header rows made bold")
    finish(wb, ws, a)


def cmd_strip_notes(a):
    wb, ws = T.load(a.book)
    print(f"{a.tag} note segments removed from {T.strip_tagged_notes(ws, a.tag)} cells")
    finish(wb, ws, a)


def cmd_add_note(a):
    wb, ws = T.load(a.book)
    for spec in a.notes:
        where, note = spec.split("=", 1)
        b, r = row_of(ws, where)
        c = ws.cell(r, b.cols["notes"])
        c.value = f"{c.value.strip()}; {note}" if isinstance(c.value, str) and c.value.strip() else note
        print(f"group {b.group} row {r}: note added")
    finish(wb, ws, a)


def cmd_add_notes(a):
    """notes from a TSV with columns group, root_id_orig, note (the id may also be the row's root_id_curr); a note
    already in the cell is not added again"""
    wb, ws = T.load(a.book)
    t = pd.read_csv(a.tsv, sep="\t", dtype=str).fillna("")
    at = {(b.group, T.get(ws, b, r, k)): (b, r) for b in T.blocks(ws) for r in b.rows for k in ("root_id_curr", "root_id_orig")}
    n = 0
    for x in t.itertuples(index=False):
        if (int(x.group), x.root_id_orig) not in at:
            sys.exit(f"no row of {x.root_id_orig} in group {x.group}")
        b, r = at[(int(x.group), x.root_id_orig)]
        c = ws.cell(r, b.cols["notes"])
        old = c.value.strip() if isinstance(c.value, str) else ""
        if x.note in old:
            continue
        c.value = f"{old}; {x.note}" if old else x.note
        n += 1
    print(f"{n} of {len(t)} notes added ({len(t) - n} already there)")
    finish(wb, ws, a)


def cmd_drop_rows(a):
    """remove the member rows of these cells (root_id_orig or root_id_curr) in every group; the rows below move up
    with their styles, links and heights"""
    wb, ws = T.load(a.book)
    want = set(a.roots)
    gone = sorted((r for b in T.blocks(ws) for r in b.rows
                   if {T.get(ws, b, r, "root_id_orig"), T.get(ws, b, r, "root_id_curr")} & want), reverse=True)
    last = ws.max_row
    for d in gone:
        print(f"row {d}: dropped ({T.text(ws, d, 1)})")
        for r in range(d, last):
            T.restore(ws, r, T.snapshot(ws, r + 1))
        for c in range(1, ws.max_column + 1):
            x = ws.cell(last, c)
            x.hyperlink, x.value = None, None
        last -= 1
    print(f"{len(gone)} rows dropped for {len(want)} cells")
    finish(wb, ws, a)


DEC = re.compile(r"^-?\d+\.\d+$")  # scores are numbers in the tab; ids and xyz stay text


def rewrite(ws, edits, inserts=None):
    """rebuild the tab from row 1: edits {row: [snapshots]} replaces a row by zero or more lines, inserts {row:
    [snapshots]} adds lines below a row; every other row moves with its style, links and height"""
    W = ws.max_column
    empty = lambda r: all(ws.cell(r, c).value in (None, "") for c in range(1, W + 1))
    end = max(r for r in range(1, ws.max_row + 1) if not empty(r))
    blank = T.snapshot(ws, next(r for r in range(1, end + 1) if empty(r)))
    lines = []
    for r in range(1, end + 1):
        lines += edits[r] if r in edits else [T.snapshot(ws, r)]
        lines += (inserts or {}).get(r, [])
    for r, s in enumerate(lines, 1):
        T.restore(ws, r, s)
    for r in range(len(lines) + 1, end + 1):
        T.restore(ws, r, blank)
    print(f"rows 1-{end} -> 1-{len(lines)}")


def make_row(template, cols, values, styles=None):
    """a row snapshot: values by header name (cols: name -> column) in template's styles, or the style of the same
    column name in `styles`; a link only where the value is a URL"""
    cells, height = template
    new = [(None, st, None) for _, st, _ in cells]
    for name, c in cols.items():
        v = values.get(name)
        st = (styles or {}).get(name, cells[c - 1][1])
        new[c - 1] = (v, st, v if isinstance(v, str) and v.startswith("http") else None)
    return new, height


def row_values(ws, b, r):
    return {n: ws.cell(r, c).value for n, c in b.cols.items()}, {n: ws.cell(r, c)._style for n, c in b.cols.items()}


def cmd_move_rows(a):
    """move member rows (root_id_orig or root_id_curr ROOT) below the last member of group G, with their values and
    styles by column name and NOTE appended; 'not refined ...' notes (about the old group's census bundle) are dropped.
    A block left without members loses its title line, header and the blank line below."""
    wb, ws = T.load(a.book)
    bl = T.blocks(ws)
    by_group = {b.group: b for b in bl}
    where = {T.get(ws, b, r, k): (b, r) for b in bl for r in b.rows for k in ("root_id_orig", "root_id_curr")}
    edits, inserts, gone = {}, {}, set()
    for spec in a.moves:
        head, _, note = spec.partition("=")
        root, g = head.rsplit(":", 1)
        if root not in where or int(g) not in by_group:
            sys.exit(f"{spec}: no row of {root} or no group {g}")
        (b, r), d = where[root], by_group[int(g)]
        vals, styles = row_values(ws, b, r)
        parts = [p for p in short_notes.split_parts(str(vals.get("notes") or "")) if not p.startswith("not refined")]
        vals["notes"] = "; ".join(parts + ([note.strip()] if note.strip() else [])) or None
        inserts.setdefault(d.rows[-1], []).append(make_row(T.snapshot(ws, d.rows[-1]), d.cols, vals, styles))
        edits[r] = []
        gone.add(r)
        print(f"row {r} (group {b.group}) -> group {d.group}: {root}")
    for b in bl:
        if set(b.rows) <= gone:
            below = b.rows[-1] + 1
            edits.update({b.title_row: [], b.header_row: []})
            if not any(T.text(ws, below, c) for c in range(1, ws.max_column + 1)):
                edits[below] = []
            print(f"group {b.group}: no members left, block removed")
    rewrite(ws, edits, inserts)
    report_tidy(T.tidy(ws))
    finish(wb, ws, a)


def paste_groups(path):
    """groups_refine.py paste -> {group: [row dicts by header name]}"""
    out, hdr, g = {}, None, None
    for line in open(path, encoding="utf-8").read().rstrip("\n").split("\n"):
        c = line.split("\t")
        x = c[0].strip()
        m = re.match(r"^group (\d+)", x)
        if m:
            g = int(m.group(1))
            out[g] = []
        elif x == "root_id_orig":
            hdr = c
        elif x.isdigit():
            out[g].append(dict(zip(hdr, c)))
    return out


def cmd_replace_group(a):
    """rewrite these groups' member rows from a groups_refine.py paste; nothing else in the tab changes. A row already
    in the group keeps its tracking values, SeaTable column, notes and style; its class and pairing columns come from
    the paste, its class change and the run's new note parts (short form, faab.partners.short_notes) are added to its
    notes. A new row takes the style of the group's first member and the run's note in short form."""
    wb, ws = T.load(a.book)
    paste = paste_groups(a.paste)
    edits = {}
    for b in T.blocks(ws):
        if b.group not in a.groups:
            continue
        own = {}
        for r in b.rows:
            for k in ("root_id_orig", "root_id_curr"):
                own.setdefault(T.get(ws, b, r, k), r)
        new, n_new = [], 0
        for p in paste[b.group]:
            r = own.get(p["root_id_orig"]) or own.get(p.get("root_id_curr", ""))
            old = row_values(ws, b, r)[0] if r else {}
            pn = p.get("notes", "")
            date = (re.findall(r"(?:refine|added) (\d{4}-\d\d-\d\d)", pn) or [today()])[0]
            vals = {}
            for n in b.cols:
                v = p.get(n)
                if r and (n not in p or (n in TRACK and old.get(n) not in (None, ""))):
                    v = old.get(n)
                if isinstance(v, str) and DEC.match(v.strip()):
                    v = float(v)
                vals[n] = None if v == "" else v
            mk = re.search(r";?\s*(?:refine|regroup) \d{4}-\d{2}-\d{2}:", pn)
            if r:
                parts = short_notes.split_parts(str(old.get("notes") or ""))
                if old.get("class") != vals.get("class"):
                    parts.append(f"class {old.get('class')} -> {vals.get('class')} (bundle tracking {date})")
                added = short_notes.shorten(pn[mk.start():].lstrip("; ") if mk else "")[0]
            else:
                parts, added = [], short_notes.shorten(pn)[0].replace("added by bundle tracking",
                                                                       f"added {date} by bundle tracking")
                n_new += 1
            parts += [s for s in short_notes.split_parts(added) if s not in parts]
            vals["notes"] = "; ".join(parts) or None
            new.append(make_row(T.snapshot(ws, r or b.rows[0]), b.cols, vals))
        edits[b.rows[0]] = new
        edits.update({r: [] for r in b.rows[1:]})
        print(f"group {b.group}: {len(b.rows)} rows -> {len(new)} ({n_new} new)")
    rewrite(ws, edits)
    report_tidy(T.tidy(ws))
    finish(wb, ws, a)


def cmd_apply_refine(a):
    wb, ws = T.load(a.book)
    W = ws.max_column
    leads = [b for b in T.blocks(ws) if b.section == "leads"]
    start = leads[0].title_row
    end = max(r for r in range(start, ws.max_row + 1) if any(ws.cell(r, c).value not in (None, "") for c in range(1, W + 1)))
    snap = {r: T.snapshot(ws, r) for r in range(start, end + 1)}
    val = lambda r, ci: snap[r][0][ci][0]  # ci 0-based
    tab_rows, by_orig, group, hdr = {}, {}, None, None
    t_group = t_header = t_blank = t_data = None  # template rows for each kind of line
    for r in range(start, end + 1):
        x = str(val(r, 0) or "").strip()
        m = re.match(r"^group (\d+)$", x)
        if m:
            group, t_group = int(m.group(1)), t_group or r
        elif x == "root_id_orig":
            hdr, t_header = {str(val(r, c) or "").strip(): c for c in range(W)}, t_header or r
        elif not x:
            t_blank = t_blank or r
        elif x.isdigit():
            tab_rows[(group, x)] = r
            by_orig.setdefault(x, r)
            t_data = t_data or r

    lines = [ln.split("\t") for ln in open(a.paste, encoding="utf-8").read().rstrip("\n").split("\n")]
    new, phdr, group, prev, kept = [], None, None, t_data, 0
    for c in lines:
        x = c[0].strip()
        m = re.match(r"^group (\d+)", x)
        if m:
            group = int(m.group(1))
            new.append(("group", {0: x}, t_group, False))
            continue
        if x == "root_id_orig":
            phdr = c
            new.append(("header", {}, t_header, False))
            continue
        if not x:
            new.append(("blank", {}, t_blank, False))
            continue
        p = dict(zip(phdr, c))
        own = tab_rows.get((group, x))  # this row's own tab row: its values may win; a moved row only borrows the style
        srow = own or by_orig.get(x)
        vals = {}
        for name, ci in hdr.items():
            if not name:
                continue
            v = p.get(name)
            if own is not None:
                tv = val(srow, ci)
                if name in TRACK and tv not in (None, ""):
                    v, kept = tv, kept + 1
                elif name == "notes":
                    pn = p.get("notes", "")
                    mk = re.search(r";?\s*(?:refine|regroup) \d{4}-\d{2}-\d{2}:", pn)  # the first dated addition
                    start_note = (pn[:mk.start()] if mk else pn).strip()
                    add = (pn[mk.start():] if mk else "").lstrip("; ").strip()
                    if tv not in (None, "") and str(tv).strip() != start_note:
                        v = f"{str(tv).strip()}; {add}" if add else str(tv).strip()
                elif name not in p:
                    v = tv
            elif srow is not None and name not in p:
                v = None
            if isinstance(v, str) and DEC.match(v.strip()):
                v = float(v)
            vals[ci] = v if v != "" else None
        new.append(("data", vals, srow or prev, own is not None))
        prev = srow or prev

    for i, (kind, vals, srow, own) in enumerate(new):
        r = start + i
        cells, height = snap[srow]
        if height:
            ws.row_dimensions[r].height = height
        for ci in range(W):
            cell = ws.cell(r, ci + 1)
            sv, sst, slink = cells[ci]
            cell._style, cell.hyperlink = copy.copy(sst), None
            v = sv if kind == "header" else vals.get(0) if kind == "group" and ci == 0 else \
                vals.get(ci) if kind == "data" else None
            cell.value = v
            if kind == "data" and slink and v and str(v).startswith("http"):
                cell.hyperlink = str(v)
    last = start + len(new) - 1
    for r in range(last + 1, end + 1):  # leftover rows below the new end
        for ci in range(W):
            cell = ws.cell(r, ci + 1)
            cell.hyperlink, cell.value, cell._style = None, None, copy.copy(snap[t_blank][0][ci][1])
    n_data = sum(k[0] == "data" for k in new)
    print(f"leads section rows {start}-{end} -> {start}-{last}; {n_data} member rows, "
          f"{sum(k[0] == 'data' and k[3] for k in new)} styled from their own tab row; tab tracking values kept: {kept}")
    report_tidy(T.tidy(ws, () if a.keep_regroup_notes else ("regroup",)))
    finish(wb, ws, a)


def group_table(ws, bl):
    rows = []
    for b in bl:
        for r in b.rows:
            g = lambda k: T.get(ws, b, r, k)
            rows.append(dict(group=b.group, **{"class": g("class")}, xyz=g("xyz"), root_id_curr=g("root_id_curr"),
                             root_id_orig=g("root_id_orig"), fly_pair=g("fly_pair"), pair_type=g("pair_type"), excel_row=r))
    return pd.DataFrame(rows)


def cmd_export_groups(a):
    ws = T.load(a.book)[1]
    t = group_table(ws, [b for b in T.blocks(ws) if a.all or b.section == "targets"])
    t.to_csv(a.tsv, sep="\t", index=False)
    print(f"{len(t)} rows in groups {sorted(int(g) for g in t.group.unique())}; wrote {a.tsv}")


def ring_text(ws, b, r, rec):
    role, cls = rec["role"], T.get(ws, b, r, "class")
    if cls == "R1-R6":
        manual = re.search(r"fly_pair\s*=\s*(R\d/R\d)", T.get(ws, b, r, "manual fly_pair check"))
        want = manual.group(1) if manual else T.get(ws, b, r, "fly_pair")
        if role == want:
            return "match" + (" (with the manual correction)" if manual else ""), True
        if role in ("R1/R6", "R2/R5", "R3/R4"):
            return f"ring {role}, sheet {want}", False
        return f"ring role {role}" + (f" ({rec['left_out']})" if rec["left_out"] else ""), False
    if cls == "R7":
        return (f"R7 ({rec['r7_how']}); flanking cells {rec['flank_margin_deg']} deg from R7" if role == "R7"
                else f"ring role {role}"), None
    if cls == "R8":
        return (f"central, {rec['median_r_um']} um from the center" if role == "R8"
                else f"not central ({rec['median_r_um']} um), ring role {role}"), None
    return f"ring role {role}", None


def cmd_ring(a):
    wb, ws = T.load(a.book)
    bl = [b for b in T.blocks(ws) if (b.group in a.groups if a.groups else b.section == "targets")]
    if a.run:
        tsv = fig("prs_tab_groups.tsv")
        group_table(ws, bl).to_csv(tsv, sep="\t", index=False)
        for b in bl:
            print(f"\n===== pr_ring_plot.py group {b.group}", flush=True)
            res = subprocess.run([sys.executable, RING_PLOT, "--group", str(b.group), "--groups-file", tsv])
            print(f"===== group {b.group} exit {res.returncode}", flush=True)
    have = sorted({c for b in bl for name, c in b.cols.items() if name.startswith("ring_check")})
    if len(have) > 1:
        sys.exit(f"ring_check sits in several columns {have}: not writing")
    fresh = not have  # an existing ring_check column is rewritten in place; else a new one after the last header
    col = have[0] if have else max(max(b.cols.values()) for b in bl) + 1
    if fresh and any(ws.cell(r, col).value not in (None, "") for b in bl for r in [b.header_row] + b.rows):
        sys.exit(f"column {col} is not empty: not writing")
    letter = ws.cell(1, col).column_letter
    if fresh:
        ws.column_dimensions[letter].width = 34
    label = a.label or f"ring_check ({today()})"
    summary, stale = [], []
    for b in bl:
        for r in [b.header_row] + b.rows:
            if fresh:
                ws.cell(r, col)._style = copy.copy(ws.cell(r, col - 1)._style)
        ws.cell(b.header_row, col).value = label
        p = fig(f"pr_ring_plot_g{b.group}_names.tsv")
        if not os.path.exists(p):
            summary.append(f"g{b.group}: no ring result")
            continue
        n = pd.read_csv(p, sep="\t", dtype=str).fillna("")
        by_xyz = {norm_xyz(x): rec for x, rec in zip(n.xyz, n.to_dict("records"))}
        agree = differ = 0
        for r in b.rows:
            rec = by_xyz.get(norm_xyz(T.get(ws, b, r, "xyz")))
            if rec is None:
                ws.cell(r, col).value = "not in the ring run"
                continue
            if rec["root"] != T.get(ws, b, r, "root_id_curr"):
                stale.append(f"group {b.group} row {r}: sheet {T.get(ws, b, r, 'root_id_curr')}, ring run {rec['root']}")
            ws.cell(r, col).value, ok = ring_text(ws, b, r, rec)
            agree, differ = agree + (ok is True), differ + (ok is False)
        summary.append(f"g{b.group}: {agree} match, {differ} differ")
    print(f"ring check in column {letter} ('{label}'):", "; ".join(summary))
    print(f"rows whose root_id_curr differs from the ring run's root (not changed): {stale or 'none'}")
    finish(wb, ws, a)


def cmd_pr_groups(a):
    ws = T.load(a.book)[1]
    rows = []
    for b in T.blocks(ws):
        for r in b.rows:
            g = lambda k: T.get(ws, b, r, k)
            if g("class") not in ("R1-R6", "R7", "R8") or any(t in g("notes") for t in T.ASIDE):
                continue
            rows.append(dict(group=b.group, **{"class": g("class")}, xyz=g("xyz"), root_id_curr=g("root_id_curr"),
                             root_id_orig=g("root_id_orig"), status=g("status"),
                             source=a.source_targets if b.section == "targets" else a.source_leads))
    out = pd.DataFrame(rows).sort_values(["group"], kind="stable")  # keeps the tab's member order
    out.to_csv(a.out, sep="\t", index=False)
    print(f"{len(out)} rows in {out.group.nunique()} groups; sizes {out.groupby('group').size().value_counts().sort_index().to_dict()}; "
          f"per source {out.source.value_counts().to_dict()}; wrote {a.out}")


def cmd_lint(a):
    ws = T.load(a.book)[1]
    issues = T.lint(ws)
    for k, v in issues.items():
        print(f"{k}: {len(v)}: {', '.join(v[:20])}{', ...' if len(v) > 20 else ''}")
    print(T.lint_line(ws) if not issues else f"{sum(map(len, issues.values()))} convention breaks")


def style_sig(x):
    col = lambda c: None if c is None else (c.rgb if isinstance(c.rgb, str) else f"theme{c.theme}/{c.tint}")
    f, fl, bd, al = x.font, x.fill, x.border, x.alignment
    return (f.b, f.i, f.sz, f.name, col(f.color), fl.fill_type, col(fl.fgColor),
            tuple(getattr(getattr(bd, s), "style", None) for s in ("left", "right", "top", "bottom")), x.number_format,
            al.horizontal, al.vertical, al.wrap_text, x.hyperlink.target if x.hyperlink else None)


def cmd_diff(a):
    A, B = T.load(a.a)[1], T.load(a.b)[1]
    norm = lambda v: None if v == "" else v

    def index(ws):
        d = {}
        for b in T.blocks(ws):
            for r in b.rows:
                d.setdefault((b.group, T.get(ws, b, r, "root_id_orig")), []).append((b, r))
        return d

    ia, ib = index(A), index(B)
    gone, added = sorted(ia.keys() - ib.keys()), sorted(ib.keys() - ia.keys())
    val, sty, ex, moved, heights = Counter(), Counter(), [], 0, 0
    for k in sorted(ia.keys() & ib.keys()):
        for (ba, ra), (bb, rb) in zip(ia[k], ib[k]):
            moved += ra != rb
            heights += A.row_dimensions[ra].height != B.row_dimensions[rb].height
            for name in sorted(set(ba.cols) | set(bb.cols)):
                xa = A.cell(ra, ba.cols[name]) if name in ba.cols else None
                xb = B.cell(rb, bb.cols[name]) if name in bb.cols else None
                va, vb = norm(xa.value) if xa else None, norm(xb.value) if xb else None
                if va != vb:
                    val[name] += 1
                    if len(ex) < a.examples:
                        sa, sb = str(va), str(vb)
                        p = next((i for i, (x, y) in enumerate(zip(sa, sb)) if x != y), min(len(sa), len(sb)))
                        cut = lambda s: ("..." if p > 20 else "") + s[max(0, p - 20):p + 50]  # from just before the change
                        ex.append(f"  group {k[0]} ...{k[1][-6:]} {name}: {cut(sa)!r} -> {cut(sb)!r}")
                if xa is not None and xb is not None and style_sig(xa) != style_sig(xb):
                    sty[name] += 1

    def others(ws):
        member = {r for b in T.blocks(ws) for r in b.rows}
        return [r for r in range(1, ws.max_row + 1)
                if r not in member and any(ws.cell(r, c).value not in (None, "") for c in range(1, ws.max_column + 1))]

    oa, ob = others(A), others(B)
    W = max(A.max_column, B.max_column)
    sig = lambda ws, r: [(norm(ws.cell(r, c).value), style_sig(ws.cell(r, c))) for c in range(1, W + 1)]
    key = lambda ws, r: tuple(norm(ws.cell(r, c).value) for c in range(1, W + 1))
    odiff, only_a, only_b = [], [], []  # lines aligned by content, so an inserted line does not shift the rest
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, [key(A, r) for r in oa], [key(B, r) for r in ob],
                                                       autojunk=False).get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            odiff += [(ra, rb) for ra, rb in zip(oa[i1:i2], ob[j1:j2]) if sig(A, ra) != sig(B, rb)]
        else:
            only_a, only_b = only_a + oa[i1:i2], only_b + ob[j1:j2]
    print(f"member rows: {sum(map(len, ia.values()))} -> {sum(map(len, ib.values()))}; matched rows at a new position: "
          f"{moved}; removed {len(gone)} {gone[:6]}; added {len(added)} {added[:6]}")
    print(f"value changes by column: {dict(val) or 'none'}")
    print(f"style/link changes by column: {dict(sty) or 'none'}; row heights changed: {heights}")
    if ex:
        print("\n".join(ex))
    print(f"other lines (group lines, headers, titles): {len(oa)} -> {len(ob)}; changed: "
          f"{len(odiff)} {[f'{x}->{y}' for x, y in odiff[:8]]}; only in A: rows {only_a[:8]}; only in B: rows {only_b[:8]}")


# ---------------------------------------------------------------- read-only CAVE
def cmd_refresh_roots(a):
    wb, ws = T.load(a.book)
    refresh_roots(ws)
    finish(wb, ws, a)


def cmd_update(a):
    wb, ws = T.load(a.book)
    report_tidy(T.tidy(ws))
    refresh_roots(ws)
    finish(wb, ws, a)


def refresh_roots(ws):
    from faab.clients.cave import parse_xyz, roots_at_xyz
    todo = []
    for b in T.blocks(ws):
        for r in b.rows:
            for pt, rc in (("xyz", "root_id_curr"), ("partner_xyz", "partner_root_id_curr")):
                x = T.get(ws, b, r, pt)
                if rc in b.cols and parse_xyz(x):
                    todo.append((b, r, b.cols[rc], x))
    print(f"{len(todo)} points to look up", flush=True)
    roots = roots_at_xyz([t[3] for t in todo])
    changed, zero, ex = Counter(), [], []
    for (b, r, c, x), root in zip(todo, roots):
        cell = ws.cell(r, c)
        old = str(cell.value or "").strip()
        if root == 0:
            zero.append(f"group {b.group} {cell.coordinate} ({x})")
        elif str(root) != old:
            cell.value = str(root)
            changed[(b.section, cell.column_letter)] += 1
            if len(ex) < 12:
                ex.append(f"  group {b.group} {cell.coordinate}: {old} -> {root}")
    print("root cells changed (section, column):", dict(changed) or "none")
    if ex:
        print("\n".join(ex))
    print("points with no segment (left alone):", zero or "none")


def cmd_verify(a):
    from faab.clients.cave import client as cave_client, parse_xyz, roots_at_xyz, throttle
    ws = T.load(a.book)[1]
    checks = []
    for b in T.blocks(ws):
        for r in b.rows:
            for o, cu, pt, what in (("root_id_orig", "root_id_curr", "xyz", "cell"),
                                    ("partner_root_id_orig", "partner_root_id_curr", "partner_xyz", "partner")):
                ov, cv, pv = (T.get(ws, b, r, k) for k in (o, cu, pt))
                if ov.isdigit() and cv.isdigit() and parse_xyz(pv):
                    checks.append(dict(section=b.section, group=b.group, row=r, what=what, orig=ov, curr=cv, xyz=pv))
    print(f"{len(checks)} root pairs to check ({dict(Counter(c['section'] for c in checks))})", flush=True)
    client, last = cave_client(), [0.0]

    def wait():
        last[0] = throttle(last[0])

    wait()
    at = roots_at_xyz([c["xyz"] for c in checks])
    ids = sorted({c["curr"] for c in checks} | {c["orig"] for c in checks})
    wait()
    latest = dict(zip(ids, map(bool, client.chunkedgraph.is_latest_roots([int(x) for x in ids]))))
    cache = json.load(open(LINEAGE_CACHE)) if os.path.exists(LINEAGE_CACHE) else {}
    refreshed = set()

    def fetch(x):
        wait()
        try:
            cache[x] = [str(int(y)) for y in client.chunkedgraph.get_latest_roots(int(x))]
        except Exception as e:
            cache[x] = ["ERROR " + str(e)[:80]]

    def descends(curr, orig):
        if orig == curr:
            return True
        if latest.get(orig, False):
            return False
        if orig not in cache:
            fetch(orig)
        if curr not in cache[orig] and orig not in refreshed:  # the cached list may predate an edit: refetch once
            refreshed.add(orig)
            fetch(orig)
        return curr in cache[orig]

    need = {c["orig"] for c in checks if c["orig"] != c["curr"] and not latest.get(c["orig"], False)} - cache.keys()
    print(f"{len(need)} lineage lookups needed (~{5 * len(need) / 60:.0f} min)", flush=True)
    problems, lines = Counter(), []
    for c, root in zip(checks, at):
        p = ([f"point lies on {root}"] if str(root) != c["curr"] else []) + \
            (["root_id_curr is not current"] if not latest.get(c["curr"], False) else []) + \
            (["does not descend from root_id_orig"] if not descends(c["curr"], c["orig"]) else [])
        for x in p:
            problems[(c["section"], c["what"], x.split(" on ")[0])] += 1
        if p:
            lines.append(f"  group {c['group']} row {c['row']} {c['what']}: orig {c['orig']} curr {c['curr']}: " + "; ".join(p))
    json.dump(cache, open(LINEAGE_CACHE, "w"))
    print(f"checked {dict(Counter(c['section'] for c in checks))}; problems: {dict(problems) or 'none'}")
    print("\n".join(lines[:40]) if lines else "all root pairs lie on their points, are current and descend from root_id_orig")


def cmd_fix_points(a):
    from faab.clients.cave import client as cave_client, roots_at_xyz, throttle
    wb, ws = T.load(a.book)
    client, t = cave_client(), throttle(0)
    for spec in a.cells:
        b, r = row_of(ws, spec)
        orig, curr = T.get(ws, b, r, "root_id_orig"), T.get(ws, b, r, "root_id_curr")
        old = np.array([float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", T.get(ws, b, r, "xyz"))]) * np.array(VOX)
        t = throttle(t)
        cands = [str(int(x)) for x in client.chunkedgraph.get_latest_roots(int(orig))]
        root = curr if curr in cands else cands[0] if len(cands) == 1 else None
        if root is None:
            print(f"{spec}: root_id_orig has {len(cands)} current roots and root_id_curr is none of them: skipped")
            continue
        t = throttle(t)
        l2 = [int(x) for x in client.chunkedgraph.get_leaves(int(root), stop_layer=2)]
        best, bd = None, np.inf
        for i in range(0, len(l2), 2000):
            t = throttle(t)
            for v in client.l2cache.get_l2data(l2[i:i + 2000], attributes=["rep_coord_nm"]).values():
                if v and "rep_coord_nm" in v:
                    p = np.array(v["rep_coord_nm"], float)
                    if np.linalg.norm(p - old) < bd:
                        best, bd = p, np.linalg.norm(p - old)
        if best is None:
            print(f"{spec}: no L2 chunk coordinates for root {root}: skipped")
            continue
        pt = fmt_vox(best / np.array(VOX))
        t = throttle(t)
        at = str(roots_at_xyz([pt])[0])
        same = norm_xyz(pt) == norm_xyz(T.get(ws, b, r, "xyz")) and curr == root
        print(f"{spec}: {T.get(ws, b, r, 'xyz')} -> {pt} ({bd / 1000:.1f} um away, {len(l2)} L2 chunks); root there {at} "
              f"({'MISMATCH: not written' if at != root else 'already on its axon: not written' if same else 'OK'})")
        if at != root or same:
            continue
        ws.cell(r, b.cols["xyz"]).value, ws.cell(r, b.cols["root_id_curr"]).value = pt, root
        c = ws.cell(r, b.cols["notes"])
        note = f"xyz moved onto the axon {today()} (the earlier point lay off it)"
        c.value = f"{c.value.strip()}; {note}" if isinstance(c.value, str) and c.value.strip() else note
        for r2 in b.rows:
            if T.get(ws, b, r2, "partner_root_id_orig") == orig:
                ws.cell(r2, b.cols["partner_xyz"]).value, ws.cell(r2, b.cols["partner_root_id_curr"]).value = pt, root
                print(f"  partner columns updated in row {r2}")
    finish(wb, ws, a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cmd(name, fn, writes=True, book=True):
        p = sub.add_parser(name)
        if book:
            p.add_argument("book", help="PRs tab .xlsx")
        if writes:
            p.add_argument("-o", "--out", help="output .xlsx ('{stamp}' -> UTC stamp; default figures/<stamp>_..-PRs.xlsx)")
        p.set_defaults(fn=fn)
        return p

    cmd("tidy", cmd_tidy)
    cmd("dates", cmd_dates)
    cmd("order", cmd_order)
    cmd("bold", cmd_bold)
    cmd("strip-notes", cmd_strip_notes).add_argument("--tag", action="append", default=None,
                                                     help="note tag to remove (repeatable; default regroup)")
    cmd("add-note", cmd_add_note).add_argument("notes", nargs="+", metavar="G:ROOT=TEXT")
    cmd("add-notes", cmd_add_notes).add_argument("tsv", help="TSV with columns group, root_id_orig, note")
    cmd("drop-rows", cmd_drop_rows).add_argument("roots", nargs="+", metavar="ROOT")
    cmd("move-rows", cmd_move_rows).add_argument("moves", nargs="+", metavar="ROOT:G[=NOTE]")
    p = cmd("replace-group", cmd_replace_group)
    p.add_argument("paste", help="groups_refine.py paste TSV (PRs-tab columns)")
    p.add_argument("groups", type=int, nargs="+", metavar="G")
    p = cmd("apply-refine", cmd_apply_refine)
    p.add_argument("paste", help="groups_refine.py paste TSV (PRs-tab columns)")
    p.add_argument("--keep-regroup-notes", action="store_true")
    p = cmd("export-groups", cmd_export_groups, writes=False)
    p.add_argument("tsv")
    p.add_argument("--all", action="store_true", help="all groups, not only the targets")
    p = cmd("ring", cmd_ring)
    p.add_argument("--groups", type=int, nargs="+", help="groups to check (default: the targets)")
    p.add_argument("--run", action="store_true", help="run pr_ring_plot.py for each group first (CAVE, serial)")
    p.add_argument("--label", help="header of the column (default 'ring_check (<UTC date>)')")
    p = cmd("pr-groups", cmd_pr_groups, writes=False)
    p.add_argument("--out", default=os.path.join(DATA, "pr_groups.tsv"))
    p.add_argument("--source-targets", default="PRs tab 2026-10-18 (proofread groups)")
    p.add_argument("--source-leads", default="refine + regroup 2026-10-17 (pending review)")
    cmd("lint", cmd_lint, writes=False)
    p = cmd("diff", cmd_diff, writes=False, book=False)
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--examples", type=int, default=12)
    cmd("update", cmd_update)
    cmd("refresh-roots", cmd_refresh_roots)
    cmd("verify", cmd_verify, writes=False)
    cmd("fix-points", cmd_fix_points).add_argument("cells", nargs="+", metavar="G:ROOT")
    a = ap.parse_args()
    if a.cmd == "strip-notes":
        a.tag = tuple(a.tag or ["regroup"])
    a.fn(a)


if __name__ == "__main__":
    main()
