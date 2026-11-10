"""Extend the PRg1toT4T5 tab two connections upstream of the T4/T5 candidates and put the input blocks first (David,
2026-11-10: "find 'hop-in-2': all the presynaptic cells connected by nsyn >= 3 to the 'hop-in-1' cells that in turn are
presynaptic with nsyn >= 3 to T4/T5 candidates ... 'hop-in-1' and 'hop-in-2' are the first two subtables in the sheet";
"Exclude the CT1 and LOME (and note this clearly in the exclusion criteria). Also consider excluding other wide-field
cells but make it clear which they are"; stage names "anchor + steps"; "Keep the 'not a T4/T5 target' subtable
separate and make it clear").

Stage names (hop column / titles): PR+1, PR+2, PR+3 = 1, 2, 3 connections downstream of the PR group 1 photoreceptors
(were 1 / 2 / 3, 'hop-1/2/3'; PR+3 = the T4/T5 candidates); T4T5-1, T4T5-2 = 1, 2 connections upstream of the T4/T5
candidates (T4T5-1 was 'more inputs', hop 'in'); notT4T5-1 = an input of the cell taken out of the targets (its block
stays separate, at the bottom); 'PR+3, not T4/T5' = that cell's row. Column names are unchanged (hop1_root_id etc.).

The tab is taken as it is in BOOK (proofreaders' edits, styles and links kept):
  T4T5-1  the 'more inputs' rows (scripts/partners/target_inputs_block.py, 2026-10-31) of its main and LOW PRIORITY
          blocks, renamed.
  seeds   the T4T5-1 cells whose inputs are followed: all but the wide-field ones - CT1 648518347494887381 and the LO ME
          (CT1-like) cell 648518347531730229 by name (EXCLUDE_SEEDS, --exclude-seed for more), and any T4T5-1 cell
          with >= --wide-factor x the median T4T5-1 cable (live L2 skeletons) or >= --large-outside of it outside the
          right optic lobe. Every excluded cell is named, with the reason, in the T4T5-2 title and note.
  T4T5-2  presynaptic cells with >= --thr synapses onto at least one seed (local synapse snapshot - new checkpoints
          pulled first - + CAVE catch-up to now; self-contacts removed), minus every cell the tab lists already (any
          block) and the targets. Columns: hop 'T4T5-2'; T4T5-1_root_id = the seeds it feeds with >= thr, strongest
          first; nsyn = its synapses onto each, same order; nsyn_out = its synapses onto all seeds (weaker contacts too);
          T4T5_root_id = the PR+3 targets those seeds feed with >= thr now; T4T5_reached = how many. Tracking,
          SeaTable, xyz, cable and neuropils as in target_inputs_block.py, but notes are never copied from the
          other rows (2026-11-08 rule; tracked_in names them); very large cells (>= --large-cable um or >=
          --large-outside outside the right optic lobe) go to a LOW PRIORITY block. Rows: most seeds fed first, then
          most targets reached, then most synapses.
  layout  title lines, T4T5-1, T4T5-2, PR+1, PR+2, PR+3, the LOW PRIORITY blocks (paths, T4T5-1, T4T5-2), the 'not a
          T4/T5 target' block(s), the notes (renamed; T4T5-2 and stage-name notes added). Rows move with their styles,
          links and heights; 'rN' / '<tab> rN' references in notes / tracked_in of every tab follow
          (faab.partners.reconcile.renumber_refs).
Refuses a tab that has a T4T5-2 block already. CAVE read-only and throttled. Writes the full workbook to
tmp/<stamp>_hop_in2_full.xlsx (the next steps of scripts/sheets/update_book.py --add-in2 run on it and export the tab),
the seed table to tmp/<stamp>_hop_in2_seeds.tsv, the new rows to figures/t4t5/<stamp>_t4t5_hop_in2.tsv and a
Spelunker link (targets, T4T5-1, T4T5-2, the excluded wide-field cells hidden).
Usage: hop_in2_block.py --book BACKUP.xlsx [--tab PRg1toT4T5] [--thr 3] [--exclude-seed ROOT[:WHY] ...] [--count-only]"""
import argparse
import copy
import datetime
import importlib.util
import os
import re
import time

import numpy as np
import openpyxl

from faab import nglink, synsnap
from faab.config import TMP, fig
from faab.partners.reconcile import renumber_refs
from faab.partners.sheet import cell_text as s, utc_stamp

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("target_inputs_block", os.path.join(_here, "target_inputs_block.py"))
TIB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TIB)
SS = TIB.SS

KINDS = [  # (kind, title pattern) in the order tried
    ("preamble", r">>>"),
    ("in2", r"(hop-in-2|T4T5-2)\b"), ("low_in2", r"LOW PRIORITY - very large (hop-in-2|T4T5-2)\b"),
    ("in1", r"(more inputs \(hop 'in'\)|hop-in-1\b|T4T5-1 \()"),
    ("low_in1", r"LOW PRIORITY - very large (input|hop-in-1|T4T5-1) cells"),
    ("hop1", r"(hop-1|PR\+1) \("), ("hop2", r"(hop-2|PR\+2) \("), ("hop3", r"(hop-3|PR\+3) \("),
    ("low_path", r"LOW PRIORITY - very large cells on these paths"),
    ("dropped", r"(revised, )?not a T4/T5 target"),
    ("notes", r"note:"),
]
ORDER = ["preamble", "in1", "in2", "hop1", "hop2", "hop3", "low_path", "low_in1", "low_in2", "dropped", "notes"]
HOP_VALUES = {"1", "2", "3", "in", "in-1", "in-2", "not T4/T5", "hop", "PR+1", "PR+2", "PR+3", "T4T5-1", "T4T5-2",
              "PR+3, not T4/T5", "notT4T5-1"}
EXCLUDE_SEEDS = {  # wide-field hop-in-1 cells whose inputs are not followed (David 2026-11-10)
    "648518347494887381": "wide-field CT1, by name",
    "648518347531730229": "wide-field LO ME (CT1-like), by name",
}
COLORS = {"targets": "#D55E00", "in1": "#0072B2", "in2": "#E69F00", "in2_low": "#CC79A7", "wide": "#009E73"}  # Okabe-Ito
IN1_COL = "T4T5-1_root_id"  # T4T5-2 rows: the T4T5-1 cells they feed (column D, hop1_root_id in the other blocks)
HOP_RENAME = {"1": "PR+1", "2": "PR+2", "3": "PR+3", "in": "T4T5-1", "in-1": "T4T5-1", "not T4/T5": "PR+3, not T4/T5"}


def blocks_of(ws, width):
    """runs of non-empty rows -> [{kind, first, last}]; a run that starts with a data / header row continues the
    block above (a blank row typed inside a block)"""
    filled = lambda r: any(s(ws.cell(r, c).value) for c in range(1, width + 1))
    runs, r = [], 1
    last_used = max((r for r in range(1, ws.max_row + 1) if filled(r)), default=0)
    while r <= last_used:
        if not filled(r):
            r += 1
            continue
        start = r
        while r <= last_used and filled(r):
            r += 1
        runs.append([start, r - 1])
    out = []
    for a, b in runs:
        t = s(ws.cell(a, 1).value)
        if t in HOP_VALUES and out:
            out[-1]["last"] = b
            continue
        kind = next((k for k, p in KINDS if re.match(p, t)), None)
        if kind is None:
            raise SystemExit(f"row {a}: unknown block title {t[:100]!r}; nothing written")
        if kind == "notes" and out and out[-1]["kind"] == "notes":
            out[-1]["last"] = b
            continue
        out.append(dict(kind=kind, first=a, last=b))
    return out, last_used


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", required=True)
    ap.add_argument("--tab", default="PRg1toT4T5")
    ap.add_argument("--thr", type=int, default=3)
    ap.add_argument("--large-cable", type=float, default=1500.0)
    ap.add_argument("--large-outside", type=float, default=0.15)
    ap.add_argument("--wide-factor", type=float, default=3.0,
                    help="T4T5-1 cells with >= this x the median T4T5-1 cable count as wide-field (inputs not followed)")
    ap.add_argument("--exclude-seed", nargs="+", action="extend", default=[], metavar="ROOT[:WHY]",
                    help="more hop-in-1 cells whose inputs are left out of hop-in-2")
    ap.add_argument("--count-only", action="store_true", help="stop after counting the hop-in-2 cells")
    a = ap.parse_args()
    t0, stamp = time.time(), utc_stamp()

    wb = openpyxl.load_workbook(a.book)
    ws = wb[a.tab]
    width = ws.max_column
    blocks, last_used = blocks_of(ws, width)
    kinds = [b["kind"] for b in blocks]
    if "in2" in kinds or "low_in2" in kinds:
        raise SystemExit(f"{a.tab} has a hop-in-2 block already; nothing written")
    for k in ("preamble", "in1", "hop1", "hop2", "hop3", "notes"):
        if kinds.count(k) != 1:
            raise SystemExit(f"{a.tab}: expected one '{k}' block, found {kinds.count(k)}; nothing written")
    kind_of = {r: b["kind"] for b in blocks for r in range(b["first"], b["last"] + 1)}
    print(a.tab + ": blocks " + ", ".join(f"{b['kind']} {b['first']}-{b['last']}" for b in blocks), flush=True)

    # ---- live cells of every row
    c = TIB.cave_client()
    rows = TIB.live_rows(wb, c)
    tab_rows = [x for x in rows if x["tab"] == a.tab]
    in_tab = {x["cell"] for x in tab_rows} - {"0"}
    hop3 = [x for x in tab_rows if kind_of.get(x["row"]) == "hop3" and x["cell"] != "0"]
    targets = sorted({x["cell"] for x in hop3})
    in1_rows = [x for x in tab_rows if kind_of.get(x["row"]) in ("in1", "low_in1") and x["cell"] != "0"]
    in1 = sorted({x["cell"] for x in in1_rows})
    off = [s(x["values"].get("root_id_orig")) for x in in1_rows + hop3 if x["at"] != x["cell"]]
    print(f"{len(targets)} targets (hop-3); {len(in1_rows)} hop-in-1 rows = {len(in1)} cells; rows whose xyz lies off "
          f"root_id_curr (root_id_curr used): {off}", flush=True)

    # ---- inputs of every hop-in-1 cell (the snapshot pulls new checkpoints first); hop-in-1 -> target edges now
    snap = synsnap.Snapshot(log=lambda m: print("  " + m, flush=True))
    TIB.wait()
    syn = snap.synapses(post=[int(x) for x in in1], when="now", client=c)
    when = datetime.datetime.now(datetime.timezone.utc)
    syn = syn[(syn.pre_root != syn.post_root) & (syn.pre_root != 0)]
    e = syn.groupby(["pre_root", "post_root"]).size().rename("n").reset_index()
    e["pre_root"], e["post_root"] = e.pre_root.astype(str), e.post_root.astype(str)
    e1 = snap.edges(pre=[int(x) for x in in1], post=[int(t) for t in targets], when="now", client=c, min_syn=a.thr)
    feeds = {}
    for p, q, n in zip(e1.pre.astype(str), e1.post.astype(str), e1.n):
        feeds.setdefault(p, []).append((int(n), q))
    feeds = {p: [q for _, q in sorted(v, key=lambda t: (-t[0], t[1]))] for p, v in feeds.items()}
    print(f"hop-in-1 cells feeding a target with >= {a.thr} now: {len(feeds)} of {len(in1)}; not any more: "
          f"{sorted(set(in1) - set(feeds))}", flush=True)
    path_kinds = {"hop1", "hop2", "hop3", "low_path"}
    where = {}
    for x in tab_rows:
        where.setdefault(x["cell"], set()).add(kind_of.get(x["row"]))
    skip = in_tab | set(targets)

    # ---- seeds: hop-in-1 cells whose inputs make hop-in-2. Wide-field cells are left out (David 2026-11-10: "Exclude
    # the CT1 and LOME ... consider excluding other wide-field cells but make it clear which they are"): the named
    # ones, and any hop-in-1 cell that is very large by the tab's LOW PRIORITY rule (live L2 skeleton)
    _, geo1, _ = TIB.geometry(in1, a.large_cable, a.large_outside, t0)
    ann1 = {r: "; ".join(dict.fromkeys(s(x["values"].get("annotation")) for x in in1_rows if x["cell"] == r
                                       and s(x["values"].get("annotation")))) for r in in1}
    excl = {}
    for spec_ in list(EXCLUDE_SEEDS) + a.exclude_seed:
        r, _, why = spec_.partition(":")
        if r in in1:
            excl[r] = why or EXCLUDE_SEEDS.get(spec_, "") or "named"
    med = float(np.nanmedian([geo1[r]["cable"] for r in in1]))
    for r in in1:
        g = geo1[r]
        out_fr = sum(v for kk, v in g["fr"].items() if kk in TIB.OUTSIDE)
        if g["cable"] >= a.wide_factor * med:
            excl.setdefault(r, f"wide-field by size: {g['cable'] / 1000:.2f} mm of cable, >= {a.wide_factor:g} x the "
                               f"median T4T5-1 cell ({med:.0f} um); {g['summ']}")
        elif out_fr >= a.large_outside:
            excl.setdefault(r, f"{100 * out_fr:.0f}% of the cable outside the right optic lobe; {g['summ']}")
    seeds = [r for r in in1 if r not in excl]
    st_e = e[e.n >= a.thr]
    new_by_seed = st_e[~st_e.pre_root.isin(skip)].groupby("post_root").pre_root.apply(set).to_dict()
    only_via = {r: len(v - set().union(*(w for k, w in new_by_seed.items() if k != r))) for r, v in new_by_seed.items()}
    seed_tsv = os.path.join(TMP, f"{stamp}_hop_in2_seeds.tsv")
    with open(seed_tsv, "w", encoding="utf-8") as fh:
        fh.write("hop_in1_root_id\tannotation\tcable_um\tneuropils\tinputs_ge_thr\tnew_inputs_ge_thr\tonly_via_this\t"
                 "targets_fed_now\texcluded\n")
        for r in sorted(in1, key=lambda r: -len(new_by_seed.get(r, ()))):
            g = geo1[r]
            fh.write(f"{r}\t{ann1[r]}\t{'' if np.isnan(g['cable']) else int(round(g['cable']))}\t{g['summ']}\t"
                     f"{int((st_e.post_root == r).sum())}\t{len(new_by_seed.get(r, ()))}\t{only_via.get(r, 0)}\t"
                     f"{len(feeds.get(r, []))}\t{excl.get(r, '')}\n")
    print(f"hop-in-1 cells left out as seeds ({len(excl)}; table {seed_tsv}):", flush=True)
    for r, why in excl.items():
        print(f"  {r}  {ann1[r] or '-'}  {why}; new inputs >= {a.thr}: {len(new_by_seed.get(r, ()))} "
              f"({only_via.get(r, 0)} only through it)", flush=True)
    e = e[e.post_root.isin(set(seeds))]
    info, n_thr, already = {}, 0, {}
    for pre, g in e.groupby("pre_root"):
        g = g.sort_values(["n", "post_root"], ascending=[False, True])
        strong = g[g.n >= a.thr]
        if not len(strong):
            continue
        n_thr += 1
        if pre in in_tab or pre in targets:
            k = where.get(pre, {"hop3"})
            lab = "PR paths" if k & path_kinds else ("hop-in-1" if k & {"in1", "low_in1"} else "other blocks")
            already[lab] = already.get(lab, 0) + 1
            continue
        fed = list(strong.post_root)
        reach = list(dict.fromkeys(t for i in fed for t in feeds.get(i, [])))
        info[pre] = dict(fed=fed, n=[int(v) for v in strong.n], total=int(g.n.sum()), reach=reach)
    cells = sorted(info)
    print(f"hop-in-2: {e.pre_root.nunique()} cells with any synapse onto a hop-in-1 cell; {n_thr} with >= {a.thr} onto "
          f"one; already in the tab {already}; {len(cells)} new; feeding 1 / 2 / 3+ hop-in-1 cells: "
          f"{sum(len(info[r]['fed']) == 1 for r in cells)} / {sum(len(info[r]['fed']) == 2 for r in cells)} / "
          f"{sum(len(info[r]['fed']) >= 3 for r in cells)}; {time.time() - t0:.0f} s", flush=True)
    if a.count_only:  # how the block size depends on the threshold and on how many hop-in-1 cells a cell feeds
        print("threshold | new cells | feeding >= 2 hop-in-1 cells")
        for t in (3, 4, 5, 6, 8, 10, 15):
            g = e[(e.n >= t) & ~e.pre_root.isin(skip)].groupby("pre_root").post_root.apply(set)
            print(f"  >= {t:2d} | {len(g):5d} | {int((g.apply(len) >= 2).sum()):5d}")
        TIB.wait()
        nuc = c.materialize.live_query("nuclei_v1_aedes", timestamp=when, filter_in_dict={"pt_root_id": [int(r) for r in cells]})
        has = set(nuc.pt_root_id.astype(str))
        print(f"with a nucleus (live): {sum(r in has for r in cells)} of {len(cells)} new cells at >= {a.thr}; feeding >= 2 "
              f"hop-in-1 cells: {sum(r in has for r in cells if len(info[r]['fed']) >= 2)} of "
              f"{sum(len(info[r]['fed']) >= 2 for r in cells)}")
        return

    # ---- tracking, skeletons, xyz
    rec = TIB.tracking(cells, rows, a.tab)
    sk, geo, large = TIB.geometry(cells, a.large_cable, a.large_outside, t0)
    xyz = TIB.find_xyz(cells, rec, sk, c)
    low = large

    def row(r):
        i, g, R = info[r], geo[r], rec[r]
        b = R["best"]["values"] if R["best"] else {}
        return {"hop": "T4T5-2", IN1_COL: ", ".join(i["fed"]), "T4T5_root_id": ", ".join(i["reach"]),
                "root_id_orig": r, "root_id_curr": r, "xyz": xyz.get(r, ""), "nsyn": ", ".join(str(n) for n in i["n"]),
                "nsyn_out": i["total"], "annotation": R["annotation"], "status": R["status"], "pr_date": b.get("pr_date"),
                "proofreader": R["proofreader"], "chk_date": b.get("chk_date"), "checker": R["checker"],
                # notes are never copied between rows (David 2026-11-08: "only clear notes ... less is more");
                # tracked_in points at the other rows
                "notes": "", "SeaTable": R["seatable"], "link": R["link"], "tracked_in": R["tracked_in"],
                "T4T5_reached": len(i["reach"]), "cable_um": "" if np.isnan(g["cable"]) else int(round(g["cable"])),
                "neuropils": g["summ"], "_src": R["best"] or (R["rows"][0] if R["rows"] else None),
                "_key": (-len(i["fed"]), -len(i["reach"]), -sum(i["n"]), -i["total"], r)}

    main_rows = sorted((row(r) for r in cells if r not in low), key=lambda d: d["_key"])
    low_rows = sorted((row(r) for r in cells if r in low), key=lambda d: d["_key"])

    # ---- the new layout
    B = SS.Book(wb)
    blk = {k: [b for b in blocks if b["kind"] == k] for k in ORDER}
    in1_b = blk["in1"][0]
    hdr = in1_b["first"] + 1
    names = [s(ws.cell(hdr, j).value) for j in range(1, width + 1)]
    names = names[:max(i for i, n in enumerate(names) if n) + 1]
    tab_cols = {n: i + 1 for i, n in enumerate(names) if n}
    in2_names = [IN1_COL if n == "hop1_root_id" else n for n in names]
    solid = lambda r: any(TIB.solid(ws.cell(r, j)) for j in range(1, len(names) + 1))
    plain = next((x["row"] for x in in1_rows if not solid(x["row"])), in1_rows[0]["row"])
    W = TIB.Writer(wb, ws, in1_b["first"], hdr, in2_names, plain, tab_cols)
    stamp_when = f"{when:%Y-%m-%d %H:%M} UTC"
    in1_when = re.search(r"live synapses of ([0-9-]+ [0-9:]+ UTC)", s(ws.cell(in1_b["first"], 1).value))
    in1_when = in1_when.group(1) if in1_when else "2026-10-31"
    thr_txt = f">= {a.thr} synapses"
    big = f"(>= {a.large_cable / 1000:g} mm of cable, or >= {100 * a.large_outside:.0f}% of it outside the right optic lobe)"
    n_in1_main = sum(1 for x in in1_rows if kind_of[x["row"]] == "in1")
    excl_txt = "; ".join(f"{r} ({why})" if r in EXCLUDE_SEEDS else f"{r} ({ann1[r] or 'no annotation'}; {why})"
                         for r, why in excl.items())

    # every old row's cells, before anything moves
    snap_rows = {}
    for r in range(1, last_used + 1):
        snap_rows[r] = [(ws.cell(r, j).value, copy.copy(ws.cell(r, j)._style),
                         copy.copy(ws.cell(r, j).hyperlink) if ws.cell(r, j).hyperlink else None) for j in range(1, width + 1)]
    heights = {r: ws.row_dimensions[r].height for r in range(1, last_used + 1) if ws.row_dimensions[r].height}
    blank_r = next(r for r in range(1, last_used + 1) if r not in kind_of)
    blank = [st for _, st, _ in snap_rows[blank_r]]

    plan = []  # ('old', r) | ('title', text) | ('header',) | ('row', d) | ('blank',)

    def add_old(b):
        if plan and plan[-1][0] != "blank":
            plan.append(("blank",))
        plan.extend(("old", r) for r in range(b["first"], b["last"] + 1))

    def add_new(title, ds):
        if plan and plan[-1][0] != "blank":
            plan.append(("blank",))
        plan.append(("title", title))
        plan.append(("header",))
        plan.extend(("row", d) for d in ds)

    for k in ORDER:
        if k == "in2":
            add_new(f"T4T5-2 (2 connections upstream of the T4/T5 candidates): {len(main_rows)} cells with {thr_txt} onto "
                    f"at least one T4T5-1 cell, not listed elsewhere in this tab (live synapses of {stamp_when}). EXCLUDED: "
                    f"the inputs of {len(excl)} wide-field T4T5-1 cells are not followed - {excl_txt}. {IN1_COL} = the "
                    f"T4T5-1 cells fed, T4T5_root_id = the T4/T5 candidates those feed; most T4T5-1 cells fed first, then "
                    f"most targets reached, then most synapses", main_rows)
        elif k == "low_in2":
            if low_rows:
                add_new(f"LOW PRIORITY - very large T4T5-2 cells, not Tm-like {big}: {len(low_rows)} cells", low_rows)
        else:
            for b in blk[k]:
                add_old(b)

    # write: clear rows 1..max, then every planned row at its new place
    n_rows = max(last_used, len(plan))
    for r in range(1, n_rows + 1):
        for j in range(1, width + 1):
            cell = ws.cell(r, j)
            cell.hyperlink = None
            cell.value = None
            cell._style = copy.copy(blank[j - 1])
    for r in list(ws.row_dimensions):
        ws.row_dimensions[r].height = None
    new_of, new_rows, r = {}, {"in2": [], "low_in2": []}, 0
    for item in plan:
        r += 1
        if item[0] == "old":
            o = item[1]
            new_of[o] = r
            for j, (v, st, h) in enumerate(snap_rows[o], 1):
                cell = ws.cell(r, j)
                cell.value = v
                cell._style = st
                if h is not None:
                    cell.hyperlink = h
            if o in heights:
                ws.row_dimensions[r].height = heights[o]
        elif item[0] == "title":
            W.title(r, item[1])
        elif item[0] == "header":
            W.header(r)
        elif item[0] == "row":
            W.row(r, item[1])
    moved = sum(1 for o, n in new_of.items() if o != n)
    renumber_refs(wb, B, {a.tab: new_of}, why="rows moved (input blocks first)")

    # renames (David 2026-11-10, "anchor + steps"): hop values, block titles, title lines, notes
    why_n = "stage names: PR+N / T4T5-N"
    for x in tab_rows:
        nr = new_of.get(x["row"])
        if not nr:
            continue
        old = s(ws.cell(nr, 1).value)
        new = HOP_RENAME.get(old, old)
        if kind_of.get(x["row"]) == "dropped" and old in ("in", "in-1"):
            new = "notT4T5-1"
        if new != old:
            B.put(a.tab, nr, 1, new, why_n)
    stage = {"hop-1 (1st stage)": "PR+1 (1 connection downstream of the PRs)",
             "hop-2 (2nd stage)": "PR+2 (2 connections downstream of the PRs)",
             "hop-3 (3rd stage = targets)": "PR+3 (3 connections downstream of the PRs = the targets)"}
    generic = lambda t: re.sub(r'\bhop-([123])\b', r"PR+\1", t)
    for o, nr in new_of.items():
        t = s(ws.cell(nr, 1).value)
        k = kind_of.get(o)
        new = t
        if not t or t in HOP_VALUES or t in HOP_RENAME.values() or t == "notT4T5-1":
            continue
        if k == "in1" and o == in1_b["first"]:
            new = re.sub(r"^more inputs \(hop 'in'\): \d+ further cells",
                         f"T4T5-1 (1 connection upstream of the T4/T5 candidates): {n_in1_main} cells", t)
            new = generic(new.replace("not listed above", "not on the PR paths below"))
        elif k == "low_in1":
            new = generic(t.replace("very large input cells", "very large T4T5-1 cells"))
        elif k == "preamble" and t.startswith(">>> extended to include"):
            new = (f'>>> extended upstream of the T4/T5 candidates: "T4T5-2" -> "T4T5-1" -> T4/T5 candidate (>= {a.thr} '
                   f"synapses per connection; T4T5-1 from live synapses of {in1_when}, T4T5-2 of {stamp_when})")
        elif k == "preamble":
            new = re.sub(r'"hop-([123])"', r'"PR+\1"', t)
        elif k in ("hop1", "hop2", "hop3") and o == blk[k][0]["first"]:
            for a_, b_ in stage.items():
                new = new.replace(a_, b_)
            new = generic(new)
        elif k in ("low_path", "notes"):
            new = generic(t)
            new = re.sub(r"^note: 'more inputs' block \(hop 'in', added ([0-9-]+)\)",
                         r"note: T4T5-1 block (hop 'T4T5-1'; added \1 as 'more inputs', hop 'in')", new)
        elif k == "dropped" and o == blk["dropped"][0]["first"] and "notT4T5-1" not in t:
            new = (t + ". Its input rows (hop 'notT4T5-1') are inputs of this cell, which is NOT a T4/T5 target: they "
                   "are not part of the T4T5-1 / T4T5-2 blocks above")
        if new != t:
            B.put(a.tab, nr, 1, new, why_n)
    pre_rows = [new_of[r] for b in blk["preamble"] for r in range(b["first"], b["last"] + 1)]
    if not any(s(ws.cell(r, 1).value).startswith(">>> extended upstream") for r in pre_rows):
        print("note: no '>>> extended to include' title line found; that title line was not rewritten", flush=True)
    note_last = max(new_of[r] for b in blk["notes"] for r in range(b["first"], b["last"] + 1))
    note_style = copy.copy(ws.cell(note_last, 1)._style)
    for i, text in enumerate((
            f"note: T4T5-2 block (hop 'T4T5-2', added {when:%Y-%m-%d}): {IN1_COL} = the T4T5-1 cells the cell feeds "
            f"with {thr_txt}, strongest first; nsyn = its synapses onto each of them, in that order; nsyn_out = its "
            f"synapses onto all {len(seeds)} followed T4T5-1 cells (weaker contacts included); T4T5_root_id = the PR+3 "
            f"targets those T4T5-1 cells feed with {thr_txt}; T4T5_reached = how many. EXCLUSIONS: (1) cells this tab "
            f"lists already (any block); (2) the inputs of wide-field T4T5-1 cells are not followed - the CT1 and LO ME "
            f"cells by name, and any T4T5-1 cell with >= {a.wide_factor:g} x the median T4T5-1 cable or >= "
            f"{100 * a.large_outside:.0f}% of it outside the right optic lobe: {excl_txt}. LOW PRIORITY: very large "
            f"T4T5-2 cells {big}. Generated by scripts/partners/hop_in2_block.py ({stamp}).",
            f"note: stage names (2026-11-10): PR+N = N connections downstream of the PR group 1 photoreceptors (was "
            f"hop-N; PR+3 = the T4/T5 candidates); T4T5-N = N connections upstream of the T4/T5 candidates (T4T5-1 was "
            f"'more inputs', hop 'in'); notT4T5-1 = an input of the cell taken out of the targets. The column names "
            f"hop1_root_id / hop2_root_id (the PR+1 / PR+2 cells on a path) are unchanged."), 1):
        ws.cell(note_last + i, 1).value = text
        ws.cell(note_last + i, 1)._style = copy.copy(note_style)
    B.touched.add(a.tab)

    # ---- outputs
    full = os.path.join(TMP, f"{stamp}_hop_in2_full.xlsx")
    wb.save(full)
    os.makedirs(fig("t4t5"), exist_ok=True)
    tsv = fig(os.path.join("t4t5", f"{stamp}_t4t5_hop_in2.tsv"))
    with open(tsv, "w", encoding="utf-8") as fh:
        fh.write("\t".join(in2_names) + "\tblock\n")
        for b_, ds in (("main", main_rows), ("low priority", low_rows)):
            for d in ds:
                fh.write("\t".join(s(d.get(n)) for n in in2_names) + f"\t{b_}\n")
    st = {}
    for d in main_rows + low_rows:
        st[d["status"]] = st.get(d["status"], 0) + 1
    print(f"layout: {', '.join(k for k in ORDER if blk.get(k) or k in ('in2', 'low_in2'))}; {moved} old rows moved; "
          f"T4T5-2 {len(main_rows)} + {len(low_rows)} LOW PRIORITY rows (statuses {st}); "
          f"{sum(1 for x in B.log if x['why'] == why_n)} renames; refs renumbered "
          f"{sum(1 for x in B.log if x['why'].startswith('rows moved'))}", flush=True)
    print(f"rows: T4T5-2 tsv {tsv}", flush=True)

    layers = [{"name": f"T4/T5 candidates ({len(targets)})", "color": COLORS["targets"], "ids": targets},
              {"name": f"T4T5-1 cells ({len(seeds)})", "color": COLORS["in1"], "ids": seeds},
              {"name": f"T4T5-2 cells ({len(main_rows)})", "color": COLORS["in2"], "ids": [d["root_id_curr"] for d in main_rows]},
              {"name": f"T4T5-1 wide-field, inputs not followed ({len(excl)})", "color": COLORS["wide"],
               "ids": list(excl), "visible": False}]
    if low_rows:
        layers.append({"name": f"T4T5-2 cells, low priority ({len(low_rows)})", "color": COLORS["in2_low"],
                       "ids": [d["root_id_curr"] for d in low_rows], "visible": False})
    spec = {"meshes": [{"kind": "OL", "segments": ["21", "22", "23", "24"], "alpha": 0.15}, {"kind": "neuropil"}],
            "neurons": layers, "view": "anterior", "layout": "3d", "image": True, "select": layers[2]["name"]}
    state = nglink.build(spec)
    open(tsv.replace(".tsv", "_ng.txt"), "w").write(nglink.to_url(state) + "\n")
    TIB.wait()
    sid = c.state.upload_state_json(state)
    print("short link:", c.state.build_neuroglancer_url(sid, ngl_url="https://spelunker.cave-explorer.org"), flush=True)
    print(f"full workbook {full}", flush=True)
    print(f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
