"""Order the PRg1toT4T5 upstream blocks by a spatial penalty and mark the putative LMCs (David 2026-11-11/12: "as we
query further upstream the extents expand ... a spatial penalty for sorting the hops ... proximity to a line/axis
connecting the PRg1 terminals and the T4/T5"; "cross-reference the T4T5-2 list with the putative LMCs and add
'putative L#' notes. Sort these toward the top"; "(high confidence) etc."; a distance column "far off to the right,
not in the middle of existing columns"; rows are no identifiers).

Home column of PR group 1, one line per right optic-lobe neuropil (synapse positions, um):
  ME   from the median PR+1 -> PR+2 synapse (distal medulla) to the median input synapse of the PR+3 targets in ME
       (the T4 dendrites, proximal); this line runs ~23 deg from the medulla's thickness direction (2026-11-11 test).
  LO   through the median input synapse of the targets in LO (T5 dendrites), along the lobula's thickness.
  LOP  through the median output synapse of the targets in LOP (their axon terminals), along the lobula plate's thickness.
  LA   through the median PR group 1 -> PR+1 synapse, along the lamina's thickness.
  Thickness = from the nearest point on the face toward the neighbor neuropil (LA: ME; ME: LA; LO: ME; LOP: LO) to the
  nearest point on the opposite face. A synapse's distance = its perpendicular distance from its neuropil's line, in
  column spacings: ME 5.5 um (nearest-neighbor spacing of the R7 terminals of 36 PR groups, 2026-11-11), other
  neuropils scaled by sqrt(face area / ME face area) (same column count everywhere).
Per cell: the median distance of its path synapses inside those four neuropils - T4T5-1 cells: their synapses onto
the PR+3 targets; T4T5-2 cells: their synapses onto the T4T5-1 cells their row lists. Groups: up to 2, 2-3, 3-4, more
than 4 columns apart, none in the optic lobe.

Writes, in the T4T5-1 / T4T5-2 blocks and their LOW PRIORITY blocks:
  - row order: (T4T5-2 blocks) putative LMCs first, then by distance group, the old order inside each group; rows move
    with their values, styles, links and heights; any 'rN' reference left in notes follows (renumber_refs);
  - notes (T4T5-2 rows whose live cell is on the LMCs tab): 'putative L3 (high confidence)' / '(medium confidence)' /
    '(low confidence)' / 'putative L3 (annotated)' / 'L3 (confirmed)' / 'putative LMC, type unclear (low
    confidence)'; an earlier such part is replaced, other note text is kept as written; skipped (and printed) when the
    row's own annotation names a different type;
  - a distance column two columns right of the last used column (re-used on later runs), header
    'distance from PR group 1 column (columns apart)' on each of these blocks' header rows;
  - the block titles' order clause, one 'note: distance from PR group 1 column ...' row, and the tracked_in note.
Read-only toward CAVE (one is_latest_roots call, throttled get_latest_roots for stale ids; synapse snapshot caught up
to now). Writes tmp/<stamp>_t4t5_sort_full.xlsx (printed as 'full workbook ...' for update_book.py), the per-cell table
figures/t4t5/<stamp>_t4t5_sort.tsv and the change log figures/t4t5/<stamp>_t4t5_sort_changes.tsv.
Usage: t4t5_sort.py --book BOOK.xlsx [--tab PRg1toT4T5]"""
import argparse
import copy
import importlib.util
import os
import re
import time

import numpy as np
import openpyxl
import pandas as pd
import trimesh
from scipy.spatial import cKDTree

from faab import ol_meshes, synsnap
from faab.clients.cave import client as cave_client, throttle
from faab.config import DATA, TMP, fig
from faab.partners.reconcile import renumber_refs
from faab.partners.sheet import cell_text as s, utc_stamp

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("hop_in2_block", os.path.join(_here, "hop_in2_block.py"))
HIB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(HIB)
SS = HIB.SS

VOX = np.array([0.016, 0.016, 0.045])  # um per voxel
LABELS = ["LA_R", "ME_R", "LO_R", "LOP_R"]
TOWARD = {"LA_R": "ME_R", "ME_R": "LA_R", "LO_R": "ME_R", "LOP_R": "LO_R"}
ME_COL_UM = 5.5
GROUPS = [(2, "up to 2"), (3, "2-3"), (4, "3-4"), (np.inf, "more than 4")]
DIST = "distance from PR group 1 column (columns apart)"
IDS = re.compile(r"\d{15,}")
LMC_PART = re.compile(r"^(putative (L[1-5]|LMC, type unclear) \((annotated|(high|medium|low) confidence)\)|L[1-5] \(confirmed\))$")
ORDER_TXT = ("order: {lmc}by distance from the PR group 1 column (column '{dist}' far right; groups up to 2, 2-3, 3-4, "
             "more than 4 columns apart, none in the optic lobe); within each group {old}")
NOTE_DIST = ("note: distance from PR group 1 column (far-right column, 2026-11-12) = the median distance of the cell's "
             "synapses onto the next stage (T4T5-1 cells: onto the T4/T5 candidates; T4T5-2 cells: onto the T4T5-1 cells "
             "their row lists) from the line through the PR group 1 path - in the medulla from the PR+1 -> PR+2 synapses "
             "to the T4/T5 candidates' dendrites, in the lobula, lobula plate and lamina through the path's synapses "
             "there, across the layers - in columns (medulla 5.5 um, the spacing of neighboring R7 terminals; the other "
             "neuropils scaled to the same number of columns). Blank = no such synapses in the right optic lobe. "
             "'putative L# (... confidence)' notes on T4T5-2 rows = the cell's call on the LMCs tab.")
TRACKED_OLD = "tracked_in lists those rows."
TRACKED_NEW = "tracked_in names the cell's other tabs (no row numbers: rows move)."


def faces(label):
    """(outer points, inner points, outer face area um^2): the face toward TOWARD[label] and the opposite one"""
    v, f = ol_meshes.mesh(label)
    m = trimesh.Trimesh(v / 1000.0, f, process=True)
    m.fix_normals()
    ov = ol_meshes.mesh(TOWARD[label])[0] / 1000.0
    to = ov[cKDTree(ov).query(m.triangles_center)[1]] - m.triangles_center
    sg = np.einsum("ij,ij->i", m.face_normals, to / np.linalg.norm(to, axis=1, keepdims=True))
    pts, fid = trimesh.sample.sample_surface(m, 200000, seed=0)
    return (pts[np.isin(fid, np.where(sg > 0.2)[0])], pts[np.isin(fid, np.where(sg < -0.2)[0])],
            float(m.area_faces[sg > 0.2].sum()))


def thickness_dir(outer, inner, p):
    o = outer[cKDTree(outer).query(p)[1]]
    i = inner[cKDTree(inner).query(p)[1]]
    return (i - o) / np.linalg.norm(i - o)


def lmc_calls(wb):
    """LMCs tab: {root as written (curr, orig): note part}"""
    ws = wb["LMCs"]
    title, hdr, out = "", None, {}
    for row in ws.iter_rows(values_only=True):
        v = [s(x) for x in row]
        if not v or not v[0]:
            continue
        if v[0] == "root_id_orig":
            hdr = v
            continue
        if not IDS.fullmatch(v[0]):
            title = v[0]
            continue
        if hdr is None:
            continue
        d = dict(zip(hdr, v))
        m = re.match(r"putative (L[1-5])\b", title)
        typ = m.group(1) if m else ("LMC, type unclear" if title.startswith("LMC, type unclear") else "")
        if not typ:
            continue
        conf = d.get("confidence", "").strip().lower()
        ct = re.search(r"\bL[1-5]\b", d.get("confirmed_type", ""))
        part = (f"{ct.group()} (confirmed)" if ct else f"putative {typ} (annotated)" if conf == "annotated"
                else f"putative {typ} ({conf} confidence)" if conf in ("high", "medium", "low") else f"putative {typ}")
        for k in ("root_id_curr", "root_id_orig"):
            if IDS.fullmatch(d.get(k, "")):
                out.setdefault(d[k], part)
                break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", required=True)
    ap.add_argument("--tab", default="PRg1toT4T5")
    a = ap.parse_args()
    stamp = utc_stamp()
    wb = openpyxl.load_workbook(a.book)
    ws = wb[a.tab]
    B = SS.Book(wb)
    width0 = ws.max_column
    blocks, last_used = HIB.blocks_of(ws, width0)
    kinds = {"in1": "T4T5-1", "low_in1": "T4T5-1", "in2": "T4T5-2", "low_in2": "T4T5-2"}
    sort_blocks = [b for b in blocks if b["kind"] in kinds]
    if not any(b["kind"] == "in2" for b in sort_blocks):
        raise SystemExit(f"{a.tab}: no T4T5-2 block; nothing written")

    def block_rows(b):
        h = b["first"] + 1
        names = [s(ws.cell(h, j).value) for j in range(1, width0 + 1)]
        cols = {n: j for j, n in enumerate(names, 1) if n}
        data = [r for r in range(h + 1, b["last"] + 1) if s(ws.cell(r, 1).value) == kinds[b["kind"]]]
        return h, cols, data

    info = {id(b): block_rows(b) for b in sort_blocks}
    path_ids = {k: set() for k in ("hop1", "hop2", "hop3")}
    for b in blocks:
        if b["kind"] in path_ids:
            h = b["first"] + 1
            rc = next(j for j in range(1, width0 + 1) if s(ws.cell(h, j).value) == "root_id_curr")
            for r in range(h + 1, b["last"] + 1):
                v = s(ws.cell(r, rc).value)
                if IDS.fullmatch(v):
                    path_ids[b["kind"]].add(v)
    cell_rows = []  # (block, row, cols, root as written, fed list as written)
    for b in sort_blocks:
        h, cols, data = info[id(b)]
        for r in data:
            root = s(ws.cell(r, cols["root_id_curr"]).value) or s(ws.cell(r, cols["root_id_orig"]).value)
            fed = IDS.findall(s(ws.cell(r, cols["T4T5-1_root_id"]).value)) if "T4T5-1_root_id" in cols else []
            cell_rows.append(dict(b=b, row=r, cols=cols, root=root, fed=fed))
    lmc = lmc_calls(wb)
    prs = pd.read_csv(os.path.join(DATA, "pr_groups.tsv"), sep="\t", dtype=str)
    g1 = sorted(set(prs[prs.group == "1"].root_id_curr))
    print(f"{a.tab}: {len(cell_rows)} rows in {len(sort_blocks)} blocks to order; path cells "
          f"{ {k: len(v) for k, v in path_ids.items()} }; PR group 1 {len(g1)}; LMCs tab calls {len(lmc)}", flush=True)

    # ---- live roots of every id involved (one is_latest call, then throttled get_latest_roots)
    c = cave_client()
    every = sorted({x["root"] for x in cell_rows if IDS.fullmatch(x["root"])} | {f for x in cell_rows for f in x["fed"]}
                   | set().union(*path_ids.values()) | set(g1) | set(lmc))
    ok = np.concatenate([np.asarray(c.chunkedgraph.is_latest_roots([int(x) for x in every[i:i + 5000]])).astype(bool)
                         for i in range(0, len(every), 5000)])
    last = time.time()
    cur = {}
    for x, good in zip(every, ok):
        if good:
            cur[x] = [int(x)]
        else:
            last = throttle(last)
            cur[x] = [int(y) for y in np.atleast_1d(c.chunkedgraph.get_latest_roots(int(x)))]
    print(f"{len(every)} ids, {int((~ok).sum())} stale -> current pieces", flush=True)
    now = lambda ids: sorted({y for x in ids for y in cur.get(x, [])})
    lmc_now = {}
    for k, part in lmc.items():
        for y in cur.get(k, []):
            lmc_now.setdefault(y, part)

    # ---- synapses (post-synaptic positions, um)
    snap = synsnap.Snapshot(log=lambda *_: None)

    def syn(pre, post, side="post"):
        t0 = time.time()
        df = snap.synapses(pre=now(pre) if pre else None, post=now(post) if post else None, when="now",
                           positions=True, client=c)
        df = df[df.pre_root != df.post_root]
        P = df[[f"{side}_x", f"{side}_y", f"{side}_z"]].to_numpy(float) * VOX
        df = df.assign(x=P[:, 0], y=P[:, 1], z=P[:, 2])
        df["np"] = ol_meshes.assign(P, LABELS) if len(df) else []
        print(f"  {len(df)} synapses ({time.time() - t0:.0f} s)", flush=True)
        return df

    print("synapse snapshot (caught up to now):", flush=True)
    t1 = sorted({x["root"] for x in cell_rows if x["b"]["kind"] in ("in1", "low_in1") and IDS.fullmatch(x["root"])})
    t2 = sorted({x["root"] for x in cell_rows if x["b"]["kind"] in ("in2", "low_in2") and IDS.fullmatch(x["root"])})
    fed_all = sorted({f for x in cell_rows for f in x["fed"]})
    ch_la = syn(g1, path_ids["hop1"])
    ch_me = syn(path_ids["hop1"], path_ids["hop2"])
    tin = syn(None, path_ids["hop3"])
    tout = syn(path_ids["hop3"], None, side="pre")
    p1 = syn(t1, path_ids["hop3"])
    p2 = syn(t2, fed_all)

    # ---- the home-column lines
    X = ["x", "y", "z"]
    med = lambda df, L: df[df.np == L][X].median().to_numpy() if int((df.np == L).sum()) >= 10 else None
    lines, col_um, area = {}, {}, {}
    fc = {L: faces(L) for L in LABELS}
    for L in LABELS:
        area[L] = fc[L][2]
    D, P = med(ch_me, "ME_R"), med(tin, "ME_R")
    if D is not None and P is not None:
        lines["ME_R"] = (D, (P - D) / np.linalg.norm(P - D))
    for L, anchor in (("LO_R", med(tin, "LO_R")), ("LOP_R", med(tout, "LOP_R")), ("LA_R", med(ch_la, "LA_R"))):
        if anchor is not None:
            lines[L] = (anchor, thickness_dir(fc[L][0], fc[L][1], anchor))
    for L in LABELS:
        col_um[L] = ME_COL_UM * np.sqrt(area[L] / area["ME_R"])
    print("home-column lines: " + "; ".join(f"{L} through {np.round(o, 1).tolist()} (column {col_um[L]:.1f} um)"
                                            for L, (o, _) in lines.items()), flush=True)

    def dist_cols(df):
        d = np.full(len(df), np.nan)
        P = df[X].to_numpy(float)
        for L, (o, n) in lines.items():
            i = np.where(df.np.to_numpy() == L)[0]
            if len(i):
                q = P[i] - o
                d[i] = np.linalg.norm(q - np.outer(q @ n, n), axis=1) / col_um[L]
        return d

    p1 = p1.assign(d=dist_cols(p1))
    p2 = p2.assign(d=dist_cols(p2))

    # ---- per row: distance, group, LMC note (matched on live roots: two written ids can share one live cell)
    tab_rows = []
    for x in cell_rows:
        mine = cur.get(x["root"], [])
        if x["b"]["kind"] in ("in1", "low_in1"):
            g = p1[p1.pre_root.isin(mine)]
        else:
            g = p2[p2.pre_root.isin(mine) & p2.post_root.isin(now(x["fed"]))]
        dd = g.d.dropna()
        x["dist"] = float(np.median(dd)) if len(dd) else np.nan
        x["grp"] = next((i for i, (hi, _) in enumerate(GROUPS) if x["dist"] <= hi), len(GROUPS))
        x["lmc"] = ""
        if x["b"]["kind"] in ("in2", "low_in2"):
            parts = {lmc_now[y] for y in cur.get(x["root"], []) if y in lmc_now}
            x["lmc"] = sorted(parts)[0] if parts else ""
        tab_rows.append(dict(block=kinds[x["b"]["kind"]] + (" LOW PRIORITY" if x["b"]["kind"].startswith("low") else ""),
                             root_id=x["root"], n_path=len(g), n_path_optic_lobe=len(dd),
                             by_neuropil="; ".join(f"{k} {v}" for k, v in g.np.value_counts().items()),
                             distance_columns=round(x["dist"], 2) if len(dd) else "",
                             group=GROUPS[x["grp"]][1] if x["grp"] < len(GROUPS) else "none in the optic lobe",
                             lmc_note=x["lmc"]))

    # ---- notes: the LMC part (replaces an earlier one; skipped when the row's annotation names another type)
    conflicts, n_note = [], 0
    for x in cell_rows:
        cols, r = x["cols"], x["row"]
        if "notes" not in cols:
            continue
        old = s(ws.cell(r, cols["notes"]).value)
        keep = [p.strip() for p in re.split(r";\s*", old) if p.strip() and not LMC_PART.match(p.strip())]
        part = x["lmc"]
        ann = s(ws.cell(r, cols["annotation"]).value) if "annotation" in cols else ""
        if part and ann:
            typ = re.search(r"L[1-5]|LMC", part).group()
            named = set(re.findall(r"\bL[1-5]\b", ann))
            if (named and typ not in named) or (not named and "LMC" not in ann):
                conflicts.append((x["root"], part, ann))
                part = ""
        if part:
            keep.append(part)
            n_note += 1
        new = "; ".join(keep)
        if new != old and not (new == "" and old == ""):
            B.put(a.tab, r, cols["notes"], new, "putative LMC note (LMCs tab call)")

    # ---- distance column, far right (re-used when present)
    dc = None
    for b in sort_blocks:
        h = info[id(b)][0]
        for j in range(1, ws.max_column + 1):
            if s(ws.cell(h, j).value) == DIST:
                dc = j
    if dc is None:
        used = max(j for r in range(1, ws.max_row + 1) for j in range(1, ws.max_column + 1) if s(ws.cell(r, j).value))
        dc = used + 2
    width = max(ws.max_column, dc)
    for b in sort_blocks:
        h, cols, _ = info[id(b)]
        ref = cols.get("neuropils") or max(cols.values())
        if s(ws.cell(h, dc).value) != DIST:
            ws.cell(h, dc).value = DIST
            ws.cell(h, dc)._style = copy.copy(ws.cell(h, ref)._style)
            B.log.append(dict(tab=a.tab, row=h, column=DIST, old="", new=DIST, why="distance column added"))
    ws.column_dimensions[openpyxl.utils.get_column_letter(dc)].width = 18
    for x in cell_rows:
        v = round(x["dist"], 1) if not np.isnan(x["dist"]) else None
        cell = ws.cell(x["row"], dc)
        if s(cell.value) != s(v):
            cell.value = v
            cell._style = copy.copy(ws.cell(x["row"], x["cols"].get("cable_um") or 1)._style)
            B.log.append(dict(tab=a.tab, row=x["row"], column=DIST, old="", new=s(v), why="distance from PR group 1 column"))
    B.touched.add(a.tab)

    # ---- row order inside each block
    new_of, moved = {}, 0
    for b in sort_blocks:
        h, cols, data = info[id(b)]
        xs = [x for x in cell_rows if x["b"] is b]
        lmc_first = b["kind"] in ("in2", "low_in2")
        order = sorted(range(len(xs)), key=lambda i: ((0 if (lmc_first and xs[i]["lmc"]) else 1), xs[i]["grp"], i))
        snap_rows = {x["row"]: [(ws.cell(x["row"], j).value, copy.copy(ws.cell(x["row"], j)._style),
                                 copy.copy(ws.cell(x["row"], j).hyperlink) if ws.cell(x["row"], j).hyperlink else None)
                                for j in range(1, width + 1)] for x in xs}
        heights = {x["row"]: ws.row_dimensions[x["row"]].height for x in xs}
        slots = [x["row"] for x in xs]  # the block's data rows, in place
        for slot, i in zip(slots, order):
            o = xs[i]["row"]
            new_of[o] = slot
            moved += o != slot
            for j, (v, st, hl) in enumerate(snap_rows[o], 1):
                cell = ws.cell(slot, j)
                cell.hyperlink = None
                cell.value = v
                cell._style = st
                if hl is not None:
                    cell.hyperlink = hl
                    cell.hyperlink.ref = cell.coordinate
            ws.row_dimensions[slot].height = heights[o]
        # the title's order clause
        t = s(ws.cell(b["first"], 1).value)
        m = re.search(r"; order: .*?; within each group (.*)$", t)
        old_clause = m.group(1) if m else (re.search(r"; (most [^;]*)$", t).group(1) if re.search(r"; (most [^;]*)$", t) else "as before")
        base = t[:m.start()] if m else (t[:t.rfind("; most")] if re.search(r"; (most [^;]*)$", t) else t)
        clause = ORDER_TXT.format(lmc="putative LMCs (notes) first, then " if lmc_first else "", dist=DIST, old=old_clause)
        B.put(a.tab, b["first"], 1, f"{base}; {clause}", "block title: row order")
    if new_of:
        renumber_refs(wb, B, {a.tab: new_of}, why="rows moved (T4T5 order)")

    # ---- notes rows: the distance note (replaced or appended), the tracked_in wording
    notes = [b for b in blocks if b["kind"] == "notes"]
    if notes:
        nb = notes[-1]
        rows_n = range(nb["first"], nb["last"] + 1)
        hit = next((r for r in rows_n if s(ws.cell(r, 1).value).startswith("note: distance from PR group 1 column")), None)
        for r in rows_n:
            t = s(ws.cell(r, 1).value)
            if TRACKED_OLD in t:
                B.put(a.tab, r, 1, t.replace(TRACKED_OLD, TRACKED_NEW), "note: tracked_in wording")
        if hit:
            B.put(a.tab, hit, 1, NOTE_DIST, "note: distance column")
        else:
            r = nb["last"] + 1
            if any(s(ws.cell(r, j).value) for j in range(1, width + 1)):
                ws.insert_rows(r)
            ws.cell(r, 1).value = NOTE_DIST
            ws.cell(r, 1)._style = copy.copy(ws.cell(nb["last"], 1)._style)
            B.log.append(dict(tab=a.tab, row=r, column="notes row", old="", new=NOTE_DIST[:60], why="note: distance column"))

    # ---- outputs
    full = os.path.join(TMP, f"{stamp}_t4t5_sort_full.xlsx")
    wb.save(full)
    t = pd.DataFrame(tab_rows)
    out = fig(f"{stamp}_t4t5_sort.tsv")  # figures/t4t5/
    t.to_csv(out, sep="\t", index=False)
    pd.DataFrame(B.log).to_csv(fig(f"{stamp}_t4t5_sort_changes.tsv"), sep="\t", index=False)
    print(f"\ncells per block and distance group:\n{pd.crosstab(t.block, t.group).to_string()}")
    tt = t[t.distance_columns != ""]
    for blk, g in tt.groupby("block"):
        v = g.distance_columns.astype(float)
        print(f"  {blk}: median {v.median():.1f} columns apart (10-90% {v.quantile(0.1):.1f}-{v.quantile(0.9):.1f}); "
              f"no optic-lobe path synapses {int(((t.block == blk) & (t.distance_columns == '')).sum())}")
    print(f"putative-LMC notes on T4T5-2 rows: {n_note} ({t[t.lmc_note != ''].lmc_note.str.extract(r'(L[1-5]|unclear)')[0].value_counts().to_dict()}); "
          f"by confidence {t[t.lmc_note != ''].lmc_note.str.extract(r'\((.*)\)')[0].value_counts().to_dict()}")
    if conflicts:
        print(f"LMC call NOT noted - the row's annotation names another type ({len(conflicts)}; root, LMCs tab call, annotation):")
        for z in conflicts:
            print("  " + "\t".join(z))
    print(f"rows moved {moved}; distance column {openpyxl.utils.get_column_letter(dc)}; per-cell table {out}")
    print(f"full workbook {full}")


if __name__ == "__main__":
    main()
