"""Tracking tab for every cell on the paths from a PR group to a set of target cells (first use 2026-10-27: PR group 1
-> David's T4/T5 candidates, paths from scratch/t4t5/pr_g1_to_t4t5.py), in the look of the PRg1hop2 tab.

Blocks (top to bottom): hop-1 cells, hop-2 cells, hop-3 cells (the targets), then LOW PRIORITY: very large cells on the
paths that are not lamina monopolar / Tm-like (>= --large-cable um of cable, or >= --large-outside of the cable
outside the right optic lobe, from the L2 skeleton and the aedes_ol_meshes). Each row:
  hop, PR_root_id / PRtype   group PRs at the start of the cell's paths
  hop1_root_id / hop2_root_id  upstream path cells (hop-2 rows: hop-1 cells; hop-3 rows: both)
  nsyn      synapses from the immediate upstream path cells, comma-separated in the order of PR_root_id (hop-1),
            hop1_root_id (hop-2) or hop2_root_id (hop-3); nsyn_out = synapses onto next-stage path cells
  T4T5_reached, cable_um, neuropils (faab.ol_meshes summary of the skeleton)
  annotation (SeaTable label; target info), status ... link carried from the tracking tabs (below), SeaTable, tracked_in
Tracking carried over: every tab row with an xyz (PRg1hop1, PRg1hop2, PRg2hop1, Mi1s, HomeR7column, ...) is mapped to
the live root at its xyz (one lookup); a path cell with any proofreading status takes the best one (checked > proofread
> adequate > review > in progress > stopped) with names / notes merged and the xyz, dates, link of that row; tracked
without a status -> NEW (--untracked-status; this tab is their primary record); untracked -> NEW with a new xyz on the
primary neurite near the soma (top10_xyz recipe,
nucleus from a live query), verified to lie on the cell. Rows carried from a tab take that row's cell styles.
CAVE read-only, >= 5 s between calls. Outputs figures/<stamp>_FAABopticlobetargets-<title>.xlsx (one tab, import
via File > Import > Insert new sheet) and .tsv, and a short Spelunker link (state server) with the optic-lobe meshes.
Example:
  venv/bin/python scripts/partners/path_cells_sheet.py --book ../tmp/20261027d161930tUTC_FAAB_OpticLobeTargets_Backup.xlsx \\
      --paths figures/20261027d152154tUTC_pr_g1_to_t4t5_paths.tsv --title PRg1toT4T5"""
import argparse
import copy
import datetime
import json
import os
import pickle
import re
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np

if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import openpyxl
import pandas as pd
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

from faab import nglink, ol_meshes
from faab.clients.cave import DATASTACK, client as cave_client, roots_at_xyz, throttle
from faab.config import FIGURES, TMP
from faab.partners.sheet import utc_stamp
from faab.pr.prs_tab import as_date

SERVER = "https://flytable.mrc-lmb.cam.ac.uk/"
CLS = ("superclass", "class", "subclass", "subsubclass", "type")
RANK = ["checked", "proofread", "adequate", "review", "in progress", "stopped"]
COLS = ["hop", "PR_root_id", "PRtype", "hop1_root_id", "hop2_root_id", "root_id_orig", "root_id_curr", "xyz", "nsyn",
        "nsyn_out", "annotation", "status", "pr_date", "proofreader", "chk_date", "checker", "notes", "SeaTable",
        "tracked_in", "link", "T4T5_reached", "cable_um", "neuropils"]  # David 2026-10-27: these three right of link
STYLE_ALIAS = {"upstream_PR": "PR_root_id", "hop": "PR_root_id", "hop2_root_id": "hop1_root_id", "nsyn_out": "nsyn",
               "T4T5_reached": "h1_conv", "cable_um": "syn_pct", "neuropils": "notes", "tracked_in": "notes"}
WIDTH = {"hop": 5, "nsyn_out": 9, "T4T5_reached": 12, "cable_um": 9, "neuropils": 30, "tracked_in": 30}
SKEL = os.path.join(FIGURES, "fragment_skeletons.pkl")
VOX = np.array([16.0, 16.0, 45.0])
_t = [0.0]


def wait():
    _t[0] = throttle(_t[0])


def s(v):
    if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):  # SeaTable empty fields come back as pd.NA
        return ""
    if isinstance(v, datetime.datetime):
        return f"{v:%y-%m-%d}"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def status_rank(st):
    st = s(st).lower()
    return RANK.index(st) if st in RANK else len(RANK)


def scan_book(wb):
    """every tab row that has an xyz: dict(tab, row, cols {header: col}, values {header: value})"""
    rows = []
    for ws in wb.worksheets:
        cols = None
        for r in range(1, ws.max_row + 1):
            vals = [ws.cell(r, c).value for c in range(1, ws.max_column + 1)]
            names = [s(v) for v in vals]
            if "xyz" in names and ("root_id_curr" in names or "root_id_orig" in names):
                cols = {n: i + 1 for i, n in enumerate(names) if n}
                continue
            if cols is None:
                continue
            xyz = s(vals[cols["xyz"] - 1])
            if len(re.findall(r"-?\d+(?:\.\d+)?", xyz)) != 3:
                continue
            rows.append(dict(tab=ws.title, row=r, cols=cols, values={n: vals[c - 1] for n, c in cols.items()}))
    return rows


def flytable(client):
    import seaserpent as ss
    meta = ss.Table("aedes_main", server=SERVER).to_frame()
    meta.columns = [str(c) for c in meta.columns]
    sv = meta["supervoxel_id"].map(s)
    meta = meta[sv.str.isdigit()].copy()
    wait()
    meta["root"] = [str(int(r)) for r in np.asarray(client.chunkedgraph.get_roots([int(x) for x in meta["supervoxel_id"].map(s)]))]
    meta["label"] = [" / ".join(x for x in (s(r.get(c)) for c in CLS) if x) for _, r in meta.iterrows()]
    return {r.root: (r.label, s(r.get("status")), s(r.get("side"))) for _, r in meta.iterrows()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", required=True)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--edges", nargs=3, default=["g1pr_F1", "g1pr_F2", "t4t5cand_B1"],
                    help="tmp/paths_cache tags of the PR->hop-1, hop-1->hop-2 and hop-2->target edge tables")
    ap.add_argument("--thr", type=int, default=3)
    ap.add_argument("--title", default="PRg1toT4T5")
    ap.add_argument("--style-tab", default="PRg1hop2")
    ap.add_argument("--group", default="1")
    ap.add_argument("--large-cable", type=float, default=1500.0)
    ap.add_argument("--large-outside", type=float, default=0.15)
    ap.add_argument("--untracked-status", default="NEW",
                    help="status of cells listed in other tabs without a proofreading status (NEW: this tab is their "
                         "primary record, David 2026-10-27; NEW-DUP: the other tab is)")
    a = ap.parse_args()
    import navis
    from fafbseg import flywire
    from faab.partners.top10_xyz import backbone_midpoint_nm, primary_neurite_point_nm
    t0, stamp = time.time(), utc_stamp()

    # ---- paths and edges
    P = pd.read_csv(a.paths, sep="\t", dtype={"pr": str, "x": str, "y": str, "t": str}).fillna("")
    P = P[(P.thr == a.thr) & (P.k == 3)]
    E = [pd.read_csv(os.path.join(TMP, "paths_cache", f"{tag}.tsv"), sep="\t", dtype={"pre": str, "post": str})
         for tag in a.edges]
    n_e = [dict(zip(zip(e.pre, e.post), e.n)) for e in E]
    pr_class = {}
    prs = pd.read_csv(os.path.join(os.path.dirname(FIGURES), "data", "pr_groups.tsv"), sep="\t", dtype=str)
    for c, r in zip(prs["class"], prs.root_id_curr):
        pr_class[r] = c
    pr_class.update(dict(zip(P.pr, P.pr_class)))
    stage = {1: sorted(set(P.x)), 2: sorted(set(P.y)), 3: sorted(set(P.t))}
    info = {}
    for hop, col, up_col, ed in ((1, "x", "pr", 0), (2, "y", "x", 1), (3, "t", "y", 2)):
        for cid, g in P.groupby(col):
            ups = sorted(set(g[up_col]), key=lambda u: -n_e[ed].get((u, cid), 0))
            out = 0
            if hop < 3:
                nxt = {1: "y", 2: "t"}[hop]
                out = sum(n_e[ed + 1].get((cid, v), 0) for v in set(g[nxt]))
            info[(hop, cid)] = dict(ups=ups, nsyn=[n_e[ed].get((u, cid), 0) for u in ups], nsyn_out=out,
                                    reached=g.t.nunique(), prs=sorted(set(g.pr)), hop1=sorted(set(g.x)),
                                    hop2=sorted(set(g.y)), best=int(g.weakest.max()), total=int(g.total.max()))
    cells = sorted({c for v in stage.values() for c in v})
    print(f"path cells: hop-1 {len(stage[1])}, hop-2 {len(stage[2])}, hop-3 {len(stage[3])}; PRs {P.pr.nunique()}", flush=True)

    # ---- CAVE: current roots, SeaTable, tracked rows
    client = cave_client()
    wait()
    ok = np.asarray(client.chunkedgraph.is_latest_roots([int(c) for c in cells])).astype(bool)
    now_of = {}
    for c, good in zip(cells, ok):
        if good:
            now_of[c] = c
            continue
        wait()
        new = [int(x) for x in np.atleast_1d(client.chunkedgraph.get_latest_roots(int(c)))]
        if len(new) > 1:  # never take [0]: the largest piece is the cell
            sizes = []
            for n in new:
                wait()
                sizes.append(len(client.chunkedgraph.get_leaves(n, stop_layer=2)))
            new = [new[int(np.argmax(sizes))]]
        now_of[c] = str(new[0])
        print(f"  {c} was edited; now {now_of[c]}", flush=True)
    ft = flytable(client)
    wb = openpyxl.load_workbook(a.book)
    tracked = scan_book(wb)
    wait()
    at_xyz = roots_at_xyz([s(t["values"]["xyz"]) for t in tracked])
    by_root = {}
    for t, r in zip(tracked, at_xyz):
        by_root.setdefault(str(r), []).append(t)
    print(f"tracking tabs: {len(tracked)} rows with an xyz; SeaTable {len(ft)} roots; {time.time() - t0:.0f} s", flush=True)

    # ---- skeletons, neuropils, size
    sk = pickle.load(open(SKEL, "rb")) if os.path.exists(SKEL) else {}
    for c in cells:
        r = now_of[c]
        if r not in sk:
            wait()
            try:
                x = flywire.get_l2_skeleton(int(r), dataset=DATASTACK, progress=False, max_threads=1) / 1000.0
                x.units = "um"
                sk[r] = x
            except Exception as e:
                sk[r] = None
                print(f"  skeleton {r} failed: {str(e)[:80]}", flush=True)
            pickle.dump(sk, open(SKEL, "wb"))
    geo = {}
    for c in cells:
        x = sk.get(now_of[c])
        if x is None:
            geo[c] = dict(cable=np.nan, fr={}, summ="")
            continue
        p = navis.resample_skeleton(x, resample_to=1.0).nodes[["x", "y", "z"]].values
        fr = ol_meshes.neuropil_fractions(p)
        geo[c] = dict(cable=float(x.cable_length), fr=fr, summ=ol_meshes.summary(fr))
    large = {c for c in cells if geo[c]["cable"] >= a.large_cable
             or sum(v for k, v in geo[c]["fr"].items() if k in ("CB", "OL_L", "LA_L", "ME_L", "LO_L", "LOP_L")) >= a.large_outside}
    print(f"very large (low priority): {sorted(large)}", flush=True)

    # ---- tracking carried over
    rec = {}
    for c in cells:
        rows = by_root.get(now_of[c], [])
        real = [t for t in rows if status_rank(t["values"].get("status")) < len(RANK)]
        best = min(real, key=lambda t: status_rank(t["values"].get("status"))) if real else None
        merged = lambda key: "; ".join(dict.fromkeys(x for t in (real or rows) for x in re.split(r";\s*", s(t["values"].get(key))) if x))
        notes = merged("notes")
        if not best:  # NEW here (this tab is the primary record): drop other tabs' 'duplicate target of ...' notes
            notes = "; ".join(x for x in notes.split("; ") if x and not x.startswith("duplicate target of"))
        rec[c] = dict(rows=rows, best=best,
                      # David 2026-10-27: cells tracked elsewhere without a status are NEW in this tab, not NEW-DUP
                      status=s(best["values"]["status"]).lower() if best else (a.untracked_status if rows else "NEW"),
                      proofreader=merged("proofreader"), checker=merged("checker"),
                      notes=notes, annotation=merged("annotation") or merged("class"),
                      seatable=("added" + (f" ({ft[now_of[c]][1]})" if now_of[c] in ft and ft[now_of[c]][1] else ""))
                      if (now_of[c] in ft or any(s(t["values"].get("SeaTable")).lower() == "added" for t in rows)) else "",
                      link=next((s(t["values"].get("link")) for t in ([best] if best else []) + rows if s(t["values"].get("link"))), ""),
                      tracked_in="; ".join(f"{t['tab']} r{t['row']}" for t in rows))

    # ---- xyz: tracked rows keep theirs, others get a primary-neurite point, verified on the cell
    xyz = {}
    for c in cells:
        src = rec[c]["best"] or (rec[c]["rows"][0] if rec[c]["rows"] else None)
        if src is not None:
            xyz[c] = s(src["values"]["xyz"])
    need = [c for c in cells if c not in xyz and sk.get(now_of[c]) is not None]
    if need:
        wait()
        nuc = client.materialize.live_query("nuclei_v1_aedes", timestamp=datetime.datetime.now(datetime.timezone.utc),
                                            filter_in_dict={"pt_root_id": [int(now_of[c]) for c in need]})
        soma = {str(r): np.asarray(p, float) for r, p in zip(nuc.pt_root_id, nuc.pt_position)}
        cand = {}
        for c in need:
            x = sk[now_of[c]] * 1000.0
            pt = None
            if now_of[c] in soma:
                pt, _ = primary_neurite_point_nm(x, soma[now_of[c]])
            cand[c] = [p for p in (pt, backbone_midpoint_nm(x)) if p is not None]
        for attempt in range(2):
            todo = [c for c in need if c not in xyz and len(cand[c]) > attempt]
            if not todo:
                break
            pts = [np.round(cand[c][attempt] / VOX).astype(int) for c in todo]
            wait()
            got = roots_at_xyz([f"{p[0]}, {p[1]}, {p[2]}" for p in pts])
            for c, p, g in zip(todo, pts, got):
                if str(g) == now_of[c]:
                    xyz[c] = f"{p[0]}, {p[1]}, {p[2]}"
        print(f"new xyz points: {sum(c in xyz for c in need)} of {len(need)} verified on their cell", flush=True)

    # ---- target annotations (morphology call, line group)
    knn = pd.read_csv(os.path.join(TMP, "t4t5_knn.tsv"), sep="\t", dtype={"cell": str})
    sv = pd.read_csv(os.path.join(TMP, "t4t5_query_survey.tsv"), sep="\t", dtype=str)
    line = dict(zip(sv.now, sv.line))
    call = dict(zip(knn.cell, knn.call))
    t5 = dict(zip(knn.cell, knn.t5_minus_t4))

    def target_note(c):
        if c not in line:
            return ""
        lean = t5.get(c, np.nan)
        lean = "" if np.isnan(lean) else (", leans T5" if lean > 0.1 else ", leans T4" if lean < -0.1 else "")
        return f"T4/T5 candidate (line {line[c]}; morphology {call.get(c, 'SeaTable T4')}{lean})"

    # ---- rows
    def row(hop, c):
        i, g, r = info[(hop, c)], geo[c], rec[c]
        best = r["best"]["values"] if r["best"] else {}
        lab = ft.get(now_of[c], ("", "", ""))[0]
        ann = "; ".join(x for x in (lab or r["annotation"], target_note(c) if hop == 3 else "") if x)
        return {"hop": str(hop), "PR_root_id": ", ".join(i["prs"]), "PRtype": ", ".join(dict.fromkeys(pr_class.get(p, "") for p in i["prs"])),
                "hop1_root_id": ", ".join(i["ups"]) if hop == 2 else (", ".join(i["hop1"]) if hop == 3 else ""),
                "hop2_root_id": ", ".join(i["ups"]) if hop == 3 else "",
                "root_id_orig": c, "root_id_curr": now_of[c], "xyz": xyz.get(c, ""),
                "nsyn": ", ".join(str(int(n)) for n in i["nsyn"]), "nsyn_out": str(i["nsyn_out"]) if hop < 3 else "",
                "T4T5_reached": str(i["reached"]), "cable_um": "" if np.isnan(g["cable"]) else str(int(round(g["cable"]))),
                "neuropils": g["summ"], "annotation": ann, "status": r["status"],
                "pr_date": best.get("pr_date"), "proofreader": r["proofreader"], "chk_date": best.get("chk_date"),
                "checker": r["checker"], "notes": r["notes"], "SeaTable": r["seatable"], "tracked_in": r["tracked_in"],
                "link": r["link"], "_src": r["best"] or (r["rows"][0] if r["rows"] else None), "_key": (-i["reached"], -i["nsyn_out"], -sum(i["nsyn"]), -i["best"])}

    blocks = []
    for hop in (1, 2, 3):
        rs = [row(hop, c) for c in stage[hop] if c not in large]
        rs.sort(key=lambda d: d["_key"] if hop < 3 else (-info[(3, d["root_id_orig"])]["best"], -info[(3, d["root_id_orig"])]["total"]))
        blocks.append((hop, rs))
    low = sorted([row(h, c) for h in (1, 2, 3) for c in stage[h] if c in large], key=lambda d: (d["hop"], -float(d["cable_um"] or 0)))
    m = re.match(r"(\d{4})(\d{2})(\d{2})d(\d{2})(\d{2})\d{2}tUTC", os.path.basename(a.paths))  # synapses = the path run's
    when = (f"{m[1]}-{m[2]}-{m[3]} {m[4]}:{m[5]} UTC" if m else "unknown time")
    n = {h: len(rs) for h, rs in blocks}
    titles = {
        1: f"hop-1 (1st stage): {n[1]} cells postsynaptic to the PR group {a.group} photoreceptors (>= {a.thr} synapses) that lead on to a T4/T5 candidate",
        2: f"hop-2 (2nd stage): {n[2]} cells postsynaptic to a hop-1 cell and presynaptic to a T4/T5 candidate (>= {a.thr} synapses each); Tm-sized, inside the right optic lobe",
        3: f"hop-3 (3rd stage = targets): {n[3]} T4/T5 candidates reached from PR group {a.group} through a hop-1 and a hop-2 cell",
    }
    low_title = (f"LOW PRIORITY - very large cells on these paths, not lamina monopolar or Tm-like (>= {a.large_cable / 1000:g} mm "
                 f"of cable, or >= {100 * a.large_outside:.0f}% of it outside the right optic lobe): {len(low)} cells; hop level in column 'hop'")
    top = (f">>> PR group {a.group} -> T4/T5 candidates: every cell on a three-connection path (PR -> hop-1 -> hop-2 -> T4/T5 "
           f"candidate, >= {a.thr} synapses per connection; live synapses of {when})")
    notes = [
        "note: nsyn = synapses the cell receives from its upstream path cells, comma-separated in the order of PR_root_id "
        "(hop-1 rows), hop1_root_id (hop-2 rows) or hop2_root_id (hop-3 rows); nsyn_out = synapses onto next-stage path cells; "
        "T4T5_reached = T4/T5 candidates reached through the cell.",
        "note: cable_um and neuropils come from the cell's L2 skeleton and the aedes_ol_meshes (LA / ME / LO / LOP, _R = right; "
        "OL_R = elsewhere in the right optic lobe, CB = central brain, none = outside the meshes).",
        "note: status, proofreader, dates, notes and link are carried over from the cell's rows in the other tabs (matched by the "
        "live root at each row's xyz; best status checked > proofread > adequate > review > in progress > stopped; names merged); "
        f"tracked_in lists those rows. Cells with no proofreading status are {a.untracked_status} here even when another tab also "
        "lists them (this tab is their primary record); untracked cells get an xyz on the primary neurite near the soma.",
        "note: SeaTable = 'added' when the cell is in SeaTable aedes_main (annotation then starts with its SeaTable label). "
        f"Generated by scripts/partners/path_cells_sheet.py ({stamp}).",
    ]
    lines = [("top", top), ("blank", "")]
    for hop, rs in blocks:
        lines += [("block", titles[hop]), ("header", None)] + [("data", d) for d in rs] + [("blank", "")]
    lines += [("block", low_title), ("header", None)] + [("data", d) for d in low] + [("blank", "")]
    lines += [("note", t) for t in notes]

    # ---- TSV
    base = os.path.join(FIGURES, f"{stamp}_FAABopticlobetargets-{a.title}")
    with open(base + ".tsv", "w", encoding="utf-8") as fh:
        for kind, v in lines:
            if kind == "data":
                fh.write("\t".join(s(v[c]) for c in COLS) + "\n")
            elif kind == "header":
                fh.write("\t".join(COLS) + "\n")
            else:
                fh.write(v + "\n")

    # ---- xlsx in the look of the style tab
    st = wb[a.style_tab]
    s_hdr_row = next(r for r in range(1, st.max_row + 1) if s(st.cell(r, 1).value) in ("PR_root_id", "upstream_PR"))
    s_cols = {s(st.cell(s_hdr_row, c).value): c for c in range(1, st.max_column + 1) if s(st.cell(s_hdr_row, c).value)}
    s_title = next(r for r in range(1, st.max_row + 1) if s(st.cell(r, 1).value).startswith(">>>"))
    s_block = s_hdr_row - 1
    s_blank = next(r for r in range(1, st.max_row + 1) if not any(s(st.cell(r, c).value) for c in range(1, 6)))
    data_rows = [r for r in range(s_hdr_row + 1, st.max_row + 1) if s(st.cell(r, s_cols["root_id_curr"]).value).isdigit()]
    s_plain = next(r for r in data_rows if not any(st.cell(r, c).fill is not None and st.cell(r, c).fill.fill_type == "solid"
                                                  for c in s_cols.values()))
    s_note = next((r for r in range(1, st.max_row + 1) if s(st.cell(r, 1).value).startswith("note:")), s_title)

    def src_col(ws_cols, name):
        for nm in (name, STYLE_ALIAS.get(name, name), {"PR_root_id": "upstream_PR"}.get(name, name)):
            if nm in ws_cols:
                return ws_cols[nm]
        return None

    out = wb.create_sheet(a.title + "_new")
    out.sheet_format = copy.copy(st.sheet_format)
    out.sheet_view.showGridLines = st.sheet_view.showGridLines
    for j, name in enumerate(COLS, 1):
        sc = src_col(s_cols, name)
        w = WIDTH.get(name) or (st.column_dimensions[get_column_letter(sc)].width if sc else None)
        if w:
            out.column_dimensions[get_column_letter(j)].width = w
    for i, (kind, v) in enumerate(lines, 1):
        if kind in ("top", "block", "note", "blank"):
            sr = {"top": s_title, "block": s_block, "note": s_note, "blank": s_blank}[kind]
            cell = out.cell(i, 1, v if v else None)
            cell._style = copy.copy(st.cell(sr, 1)._style)
            if kind in ("top", "block"):
                f = cell.font
                cell.font = Font(name=f.name, size=f.size, bold=True, italic=f.italic, color=f.color)
            continue
        if kind == "header":
            for j, name in enumerate(COLS, 1):
                c = out.cell(i, j, name)
                c._style = copy.copy(st.cell(s_hdr_row, src_col(s_cols, name) or 1)._style)
                f = c.font
                c.font = Font(name=f.name, size=f.size, bold=True, italic=f.italic, color=f.color)
            continue
        src = v["_src"]
        sws = wb[src["tab"]] if src else st
        srow = src["row"] if src else s_plain
        scols = src["cols"] if src else s_cols
        for j, name in enumerate(COLS, 1):
            val = v[name]
            if name in ("pr_date", "chk_date"):
                val = val if isinstance(val, datetime.datetime) else (as_date(s(val)) or (s(val) or None))
            elif name in ("nsyn", "nsyn_out", "T4T5_reached", "cable_um") and re.fullmatch(r"\d+", s(val)):
                val = int(val)
            else:
                val = s(val) or None
            c = out.cell(i, j, val)
            sc = src_col(scols, name)
            c._style = copy.copy(sws.cell(srow, sc or 1)._style)
            if isinstance(val, datetime.datetime):
                c.number_format = 'yy"-"mm"-"dd'
            if name == "link" and isinstance(val, str) and val.startswith("http"):
                c.hyperlink = val
    for ws in list(wb.worksheets):
        if ws is not out:
            wb.remove(ws)
    out.title = a.title
    wb.save(base + ".xlsx")
    print(f"wrote {base}.xlsx and .tsv: {len(lines)} lines; blocks hop-1 {n[1]}, hop-2 {n[2]}, hop-3 {n[3]}, low priority "
          f"{len(low)}; statuses carried for {sum(1 for c in cells if rec[c]['best'])} cells; without a status but listed "
          f"elsewhere {sum(1 for c in cells if not rec[c]['best'] and rec[c]['rows'])} ({a.untracked_status}); untracked "
          f"{sum(1 for c in cells if not rec[c]['rows'])} (NEW)", flush=True)

    # ---- Spelunker link: stages + 'hop-2 large' etc. + optic-lobe meshes
    prs_ids = sorted(set(P.pr))
    layers = [{"name": f"PR group {a.group} ({len(prs_ids)})", "color": "#f7c873", "ids": prs_ids}]
    for hop, color in ((1, "#e9953a"), (2, "#c4552a"), (3, "#7a1010")):
        ids = [now_of[c] for c in stage[hop] if c not in large]
        nm = {1: "hop-1", 2: "hop-2", 3: "hop-3 T4/T5 candidates"}[hop]
        layers.append({"name": f"{nm} ({len(ids)})", "color": color, "ids": ids})
    for hop, color in ((1, "#1f77d6"), (2, "#9467bd"), (3, "#2ecc40")):
        ids = [now_of[c] for c in stage[hop] if c in large]
        if ids:
            layers.append({"name": f"hop-{hop} large ({len(ids)})", "color": color, "ids": ids})
    spec = {"meshes": [{"kind": "OL", "segments": ["21", "22", "23", "24"], "alpha": 0.15}, {"kind": "neuropil"}],
            "neurons": layers, "view": "anterior", "layout": "3d", "image": True, "select": layers[2]["name"]}
    state = nglink.build(spec)
    open(base + "_ng.txt", "w").write(nglink.to_url(state) + "\n")
    wait()
    sid = client.state.upload_state_json(state)
    print("short link:", client.state.build_neuroglancer_url(sid, ngl_url="https://spelunker.cave-explorer.org"))
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
