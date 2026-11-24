"""Order the PRg1toT4T5 upstream blocks by photoreceptor paths and a spatial penalty, mark the putative LMCs, and mark
the PR groups that feed the targets on the PRs tab (David 2026-11-11/12: "a spatial penalty for sorting the hops ...
proximity to a line/axis connecting the PRg1 terminals and the T4/T5"; "cross-reference the T4T5-2 list with the
putative LMCs and add 'putative L#' notes. Sort these toward the top"; a distance column "far off to the right";
rows are no identifiers; after the 'PRgroupAny' test: "sorting these targets to the top of their respective blocks
... moving the LOW PRIORITY ones out into their respective blocks ... for the PRs we can highlight those PR groups
as being higher priority on the PRs tab").

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
the PR+3 targets; T4T5-2 cells: their synapses onto the cells their row lists. Groups: up to 2, 2-3, 3-4, more than 4
columns apart, none in the optic lobe.

Photoreceptor paths (any PR on the PRs tab, any group or status; >= PATH_THR synapses per connection, as for PR group
1): PR -> X -> Y -> target. T4T5-2 rows whose cell is an X, T4T5-1 rows whose cell is a Y are 'on a path'.
X / Y cells that are on NO row of the tab (they feed a PR+2 cell, or a wide-field cell whose inputs the T4T5-2 block
does not follow, or they feed a target now but did not when T4T5-1 was built) are ADDED: X to T4T5-2, Y to T4T5-1
(tracking from the other tabs, a point on the cell, cable from the L2 skeleton). LOW PRIORITY rows on a path move into
their main block (T4T5-1 / T4T5-2; the very large PR+1 / PR+2 cells into PR+1 / PR+2); a moved row whose cell already
has a row there is dropped when its notes add nothing (its cable is kept if larger), else moved (the sync's collapse
merges it); emptied LOW PRIORITY blocks go.

Writes:
  - row order in T4T5-1 / T4T5-2 (and LOW PRIORITY T4T5): on a path first, then (T4T5-2) putative LMCs, then the
    rest; inside each by distance group, then the old order; column A light blue on rows on a path;
  - notes (T4T5-2 rows whose live cell is on the LMCs tab): 'putative L3 (high confidence)' etc. (an earlier such part
    is replaced, other text kept; skipped when the row's own annotation names another type);
  - the distance column two columns right of the last used column (re-used later), header DIST;
  - block titles: cell counts, the order clause; the note rows (distance, paths, tracked_in wording);
  - PRs tab: every group with a PR on a path gets a line in column B of its 'group N' line (PRs, distance); groups
    whose path PRs sit within PRS_NEAR columns (median) get column A light blue.
Rows move with their values, styles, links and heights (any 'rN' left in notes follows, renumber_refs).
Read-only toward CAVE (is_latest_roots, throttled get_latest_roots, point lookups, nucleus query, L2 skeletons for
added cells; synapse snapshot caught up to now). Writes tmp/<stamp>_t4t5_sort_full.xlsx (printed as 'full workbook
...' for update_book.py), figures/t4t5/<stamp>_t4t5_sort.tsv and figures/t4t5/<stamp>_t4t5_sort_changes.tsv.
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
from openpyxl.styles import Font, PatternFill
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
TIB = HIB.TIB

VOX = np.array([0.016, 0.016, 0.045])  # um per voxel
LABELS = ["LA_R", "ME_R", "LO_R", "LOP_R"]
TOWARD = {"LA_R": "ME_R", "ME_R": "LA_R", "LO_R": "ME_R", "LOP_R": "LO_R"}
ME_COL_UM = 5.5
GROUPS = [(2, "up to 2"), (3, "2-3"), (4, "3-4"), (np.inf, "more than 4")]
DIST = "distance from PR group 1 column (columns apart)"
IDS = re.compile(r"\d{15,}")
KIND_HOP = {"in1": "T4T5-1", "low_in1": "T4T5-1", "in2": "T4T5-2", "low_in2": "T4T5-2"}
MAIN_OF = {"low_in1": "in1", "low_in2": "in2"}
COUNTED = {"in1": "T4T5-1", "low_in1": "T4T5-1", "in2": "T4T5-2", "low_in2": "T4T5-2", "hop1": "PR+1", "hop2": "PR+2",
           "low_path": ("PR+1", "PR+2")}
LMC_PART = re.compile(r"^(putative (L[1-5]|LMC, type unclear) \((annotated|(high|medium|low) confidence)\)|L[1-5] \(confirmed\))$")
ORDER_TXT = ("order: {lmc}by distance from the PR group 1 column (column '{dist}' far right; groups up to 2, 2-3, 3-4, "
             "more than 4 columns apart, none in the optic lobe); within each group {old}")
NOTE_DIST = ("note: distance from PR group 1 column (far-right column, 2026-11-12) = the median distance of the cell's "
             "synapses onto the next stage (T4T5-1 cells: onto the T4/T5 candidates; T4T5-2 cells: onto the cells "
             "their row lists) from the line through the PR group 1 path - in the medulla from the PR+1 -> PR+2 synapses "
             "to the T4/T5 candidates' dendrites, in the lobula, lobula plate and lamina through the path's synapses "
             "there, across the layers - in columns (medulla 5.5 um, the spacing of neighboring R7 terminals; the other "
             "neuropils scaled to the same number of columns). Blank = no such synapses in the right optic lobe. "
             "'putative L# (... confidence)' notes on T4T5-2 rows = the cell's call on the LMCs tab.")
PATH_THR = 3  # synapses per connection, as for PR group 1's paths
PATH_FILL = "FFCCE8F8"  # light tint of Okabe-Ito sky blue (#56B4E9)
PATH_TXT1 = ("rows on a photoreceptor path first (column A light blue: any PR -> a cell -> this cell -> T4/T5 candidate, "
             ">= 3 synapses each), ")
PATH_TXT2 = ("rows on a photoreceptor path first (column A light blue: any PR -> this cell -> a cell -> T4/T5 "
             "candidate, >= 3 synapses each), ")
NOTE_PATH = (" Column A light blue = the cell is on a 3-connection path from a photoreceptor of ANY group on the PRs tab "
             "(proofread or not) to the T4/T5 candidates, >= 3 synapses per connection (live synapses of each update). "
             "Such cells that no block's own rule lists (they feed a PR+2 cell, a wide-field cell whose inputs are not "
             "followed, or a T4/T5 candidate they did not feed when T4T5-1 was built) are added to T4T5-2 / T4T5-1; "
             "LOW PRIORITY cells on such paths are moved into their main block. The PRs tab marks the PR groups "
             "whose photoreceptors are on these paths (column B of the group line; light blue = within 4 columns).")
TRACKED_OLD = "tracked_in lists those rows."
TRACKED_NEW = "tracked_in names the cell's other tabs (no row numbers: rows move)."
PRS_NEAR = 4.0  # columns: PR groups this close (median of their path PRs) are highlighted on the PRs tab
PRS_TXT = "on photoreceptor paths to the {n_t} PRg1toT4T5 T4/T5 candidates: {n} PRs, {d} columns from PR group 1"
PRS_RX = re.compile(r"^on photoreceptor paths to the \d+ PRg1toT4T5")


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


def pr_tab(wb):
    """PRs tab: ({root as written: group or ''}, {group: row of its 'group N' line}) for every row with an id"""
    grp_of, line_of, g, cols = {}, {}, "", None
    for r, row in enumerate(wb["PRs"].iter_rows(values_only=True), 1):
        v = [s(x) for x in row]
        a0 = v[0] if v else ""
        m = re.match(r"^group (\d+)$", a0)
        if m:
            g = m.group(1)
            line_of[g] = r
            continue
        if "xyz" in v and ("root_id_curr" in v or "root_id_orig" in v):
            cols = {n: i for i, n in enumerate(v) if n}
            continue
        if cols is None:
            continue
        get = lambda n: v[cols[n]] if n in cols and cols[n] < len(v) else ""
        root = get("root_id_curr") if IDS.fullmatch(get("root_id_curr")) else get("root_id_orig")
        if IDS.fullmatch(root):
            grp_of.setdefault(root, g)
    return grp_of, line_of


def lmc_ok(part, ann):
    """the LMC note part, or '' when the row's own annotation names another type"""
    if not part or not ann:
        return part
    typ = re.search(r"L[1-5]|LMC", part).group()
    named = set(re.findall(r"\bL[1-5]\b", ann))
    return "" if (named and typ not in named) or (not named and "LMC" not in ann) else part


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
    if not any(b["kind"] == "in2" for b in blocks):
        raise SystemExit(f"{a.tab}: no T4T5-2 block; nothing written")
    hdr_of = {}
    for b in blocks:
        if b["kind"] in ("preamble", "notes"):
            continue
        h = b["first"] + 1
        names = [s(ws.cell(h, j).value) for j in range(1, width0 + 1)]
        hdr_of[id(b)] = (h, {n: j for j, n in enumerate(names, 1) if n}, names)

    def root_at(b, r):
        cols = hdr_of[id(b)][1]
        cu = s(ws.cell(r, cols["root_id_curr"]).value) if "root_id_curr" in cols else ""
        return cu if IDS.fullmatch(cu) else (s(ws.cell(r, cols["root_id_orig"]).value) if "root_id_orig" in cols else "")

    sort_blocks = [b for b in blocks if b["kind"] in KIND_HOP]
    path_ids = {k: set() for k in ("hop1", "hop2", "hop3")}
    tab_all, cell_rows, lp_rows = set(), [], []
    for b in blocks:
        if id(b) not in hdr_of:
            continue
        cols = hdr_of[id(b)][1]
        for r in range(b["first"] + 2, b["last"] + 1):
            root, hop = root_at(b, r), s(ws.cell(r, 1).value)
            if not IDS.fullmatch(root):
                continue
            tab_all.add(root)
            if b["kind"] in path_ids:
                path_ids[b["kind"]].add(root)
            if b["kind"] in KIND_HOP and hop == KIND_HOP[b["kind"]]:
                fed = IDS.findall(s(ws.cell(r, cols["T4T5-1_root_id"]).value)) if "T4T5-1_root_id" in cols else []
                cell_rows.append(dict(b=b, row=r, cols=cols, root=root, fed=fed))
            if b["kind"] == "low_path" and hop in ("PR+1", "PR+2"):
                lp_rows.append(dict(b=b, row=r, cols=cols, root=root, hop=hop))
    lmc = lmc_calls(wb)
    prs = pd.read_csv(os.path.join(DATA, "pr_groups.tsv"), sep="\t", dtype=str)
    g1 = sorted(set(prs[prs.group == "1"].root_id_curr))
    pr_grp, pr_line = pr_tab(wb)
    print(f"{a.tab}: {len(cell_rows)} T4T5 rows in {len(sort_blocks)} blocks; path cells "
          f"{ {k: len(v) for k, v in path_ids.items()} }; PR group 1 {len(g1)}; PRs tab {len(pr_grp)} ids in "
          f"{len(pr_line)} groups; LMCs tab calls {len(lmc)}", flush=True)

    # ---- live roots of every id involved (is_latest in batches, then throttled get_latest_roots)
    c = cave_client()
    every = sorted({f for x in cell_rows for f in x["fed"]} | tab_all | set(g1) | set(lmc) | set(pr_grp))
    ok = np.concatenate([np.asarray(c.chunkedgraph.is_latest_roots([int(x) for x in every[i:i + 5000]])).astype(bool)
                         for i in range(0, len(every), 5000)])
    last = time.time()
    cur, bad = {}, []
    for x, good in zip(every, ok):
        if good:
            cur[x] = [int(x)]
        else:
            last = throttle(last)
            try:
                cur[x] = [int(y) for y in np.atleast_1d(c.chunkedgraph.get_latest_roots(int(x)))]
            except ValueError:  # no creation time: not a root id of this datastack (a typo in the sheet?)
                cur[x] = []
                bad.append(x)
    print(f"{len(every)} ids, {int((~ok).sum())} stale -> current pieces", flush=True)
    if bad:
        print(f"  not a root id (no creation time; typo?), left out of the sort: {', '.join(bad)}", flush=True)
    now = lambda ids: sorted({y for x in ids for y in cur.get(str(x), [])})
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
    t1 = sorted({x["root"] for x in cell_rows if KIND_HOP[x["b"]["kind"]] == "T4T5-1"})
    t2 = sorted({x["root"] for x in cell_rows if KIND_HOP[x["b"]["kind"]] == "T4T5-2"})
    fed_all = sorted({f for x in cell_rows for f in x["fed"]})
    ch_la = syn(g1, path_ids["hop1"])
    ch_me = syn(path_ids["hop1"], path_ids["hop2"])
    tin = syn(None, path_ids["hop3"])
    tout = syn(path_ids["hop3"], None, side="pre")
    p1 = syn(t1, path_ids["hop3"])
    p2 = syn(t2, fed_all)

    # ---- photoreceptor paths PR -> X -> Y -> target (>= PATH_THR each), from every PR on the PRs tab
    PR = set(now(pr_grp))
    T_now = set(now(path_ids["hop3"]))
    e_pr = snap.edges(pre=sorted(PR), when="now", client=c, min_syn=PATH_THR)
    e_pr = e_pr[~e_pr.post.isin(list(PR))]
    pr_fed = set(e_pr.post)
    te_all = tin.groupby(["pre_root", "post_root"]).size().rename("n").reset_index()
    te = te_all[te_all.n >= PATH_THR]
    Y = set(te.pre_root) - T_now
    e_xy = snap.edges(pre=sorted(pr_fed), post=sorted(Y), when="now", client=c, min_syn=PATH_THR)
    e_xy = e_xy[e_xy.pre != e_xy.post]
    role1, role2 = set(e_xy.pre), set(e_xy.post)
    tab_now = set(now(tab_all))
    new_t2 = sorted(role1 - tab_now - PR - T_now)
    new_t1 = sorted(role2 - tab_now - PR - T_now - set(new_t2))
    for k in set(new_t2) | set(new_t1) | role1 | role2:
        cur.setdefault(str(k), [int(k)])
    print(f"photoreceptor paths: {len(PR)} PRs -> {len(role1)} cells -> {len(role2)} cells -> targets; not on any row of "
          f"the tab: {len(new_t2)} (-> T4T5-2) and {len(new_t1)} (-> T4T5-1)", flush=True)
    pn2 = syn([str(k) for k in new_t2], [str(k) for k in role2]) if new_t2 else None
    pn1 = syn([str(k) for k in new_t1], path_ids["hop3"]) if new_t1 else None
    on_prs = set(e_pr[e_pr.post.isin(list(role1))].pre)
    pr_written = [k for k in pr_grp if set(cur.get(k, [])) & on_prs]
    ppr = syn(pr_written, [str(k) for k in role1]) if pr_written else None

    # ---- the home-column lines
    X = ["x", "y", "z"]
    med = lambda df, L: df[df.np == L][X].median().to_numpy() if int((df.np == L).sum()) >= 10 else None
    lines, col_um = {}, {}
    fc = {L: faces(L) for L in LABELS}
    D, P = med(ch_me, "ME_R"), med(tin, "ME_R")
    if D is not None and P is not None:
        lines["ME_R"] = (D, (P - D) / np.linalg.norm(P - D))
    for L, anchor in (("LO_R", med(tin, "LO_R")), ("LOP_R", med(tout, "LOP_R")), ("LA_R", med(ch_la, "LA_R"))):
        if anchor is not None:
            lines[L] = (anchor, thickness_dir(fc[L][0], fc[L][1], anchor))
    for L in LABELS:
        col_um[L] = ME_COL_UM * np.sqrt(fc[L][2] / fc["ME_R"][2])
    print("home-column lines: " + "; ".join(f"{L} through {np.round(o, 1).tolist()} (column {col_um[L]:.1f} um)"
                                            for L, (o, _) in lines.items()), flush=True)

    def dist_cols(df):
        d = np.full(len(df), np.nan)
        if df is None or not len(df):
            return d
        Q = df[X].to_numpy(float)
        for L, (o, n) in lines.items():
            i = np.where(df.np.to_numpy() == L)[0]
            if len(i):
                q = Q[i] - o
                d[i] = np.linalg.norm(q - np.outer(q @ n, n), axis=1) / col_um[L]
        return d

    p1 = p1.assign(d=dist_cols(p1))
    p2 = p2.assign(d=dist_cols(p2))
    med_d = lambda v: float(np.median(v.dropna())) if len(v.dropna()) else np.nan
    grp_i = lambda d: next((i for i, (hi, _) in enumerate(GROUPS) if d <= hi), len(GROUPS))
    grp_name = lambda i: GROUPS[i][1] if i < len(GROUPS) else "none in the optic lobe"

    # ---- per existing row: distance, group, path, LMC note
    tab_rows = []
    for x in cell_rows:
        mine = cur.get(x["root"], [])
        k1 = KIND_HOP[x["b"]["kind"]] == "T4T5-1"
        g = p1[p1.pre_root.isin(mine)] if k1 else p2[p2.pre_root.isin(mine) & p2.post_root.isin(now(x["fed"]))]
        x["dist"] = med_d(g.d)
        x["grp"] = grp_i(x["dist"])
        x["on_path"] = bool(set(mine) & (role2 if k1 else role1))
        parts = {lmc_now[y] for y in mine if y in lmc_now} if not k1 else set()
        x["lmc"] = sorted(parts)[0] if parts else ""
        tab_rows.append(dict(block=KIND_HOP[x["b"]["kind"]] + (" LOW PRIORITY" if x["b"]["kind"].startswith("low") else ""),
                             root_id=x["root"], added="", n_path=len(g), n_path_optic_lobe=int(g.d.notna().sum()),
                             distance_columns=round(x["dist"], 2) if not np.isnan(x["dist"]) else "",
                             group=grp_name(x["grp"]), on_photoreceptor_path=x["on_path"], lmc_note=x["lmc"]))

    # ---- added rows: X / Y cells on no row of the tab
    new_rows = {"in1": [], "in2": []}
    new_cells = [str(k) for k in new_t2 + new_t1]
    if new_cells:
        print(f"tracking / points / cable for {len(new_cells)} added cells", flush=True)
        rows_all = TIB.live_rows(wb, c)
        rec = TIB.tracking(new_cells, rows_all, a.tab)
        sk, geo, _ = TIB.geometry(new_cells, 1500.0, 0.15, time.time())
        xyz = TIB.find_xyz(new_cells, rec, sk, c)
        tg_of = {int(k): g.sort_values("n", ascending=False) for k, g in te.groupby("pre_root")}
        tot_of = te_all.groupby("pre_root").n.sum().to_dict()
        for i, k in enumerate(new_t2 + new_t1):
            ks, R = str(k), rec[str(k)]
            bv = R["best"]["values"] if R["best"] else {}
            gm = geo.get(ks, {})
            d = {"root_id_orig": ks, "root_id_curr": ks, "xyz": xyz.get(ks, ""), "annotation": R["annotation"],
                 "status": R["status"], "pr_date": bv.get("pr_date"), "proofreader": R["proofreader"],
                 "chk_date": bv.get("chk_date"), "checker": R["checker"], "notes": "", "SeaTable": R["seatable"],
                 "link": R["link"], "tracked_in": R["tracked_in"],
                 "cable_um": "" if np.isnan(gm.get("cable", np.nan)) else int(round(gm["cable"])),
                 "neuropils": gm.get("summ", ""), "_src": R["best"] or (R["rows"][0] if R["rows"] else None)}
            if k in new_t2:
                f = e_xy[e_xy.pre == k].sort_values("n", ascending=False)
                tg = sorted({int(t) for q in f.post for t in tg_of.get(int(q), pd.DataFrame(columns=["post_root"])).post_root})
                g = pn2[(pn2.pre_root == k) & pn2.post_root.isin(list(f.post))]
                part = lmc_ok(lmc_now.get(k, ""), d["annotation"])
                d.update({"hop": "T4T5-2", "T4T5-1_root_id": ", ".join(str(q) for q in f.post),
                          "nsyn": ", ".join(str(n) for n in f.n), "nsyn_out": int(f.n.sum()),
                          "T4T5_root_id": ", ".join(map(str, tg)), "T4T5_reached": len(tg), "notes": part})
                kind = "in2"
            else:
                f = tg_of.get(k, pd.DataFrame(columns=["post_root", "n"]))
                g = pn1[pn1.pre_root == k]
                part = ""
                d.update({"hop": "T4T5-1", "T4T5_root_id": ", ".join(str(q) for q in f.post_root),
                          "nsyn": ", ".join(str(n) for n in f.n), "nsyn_out": int(tot_of.get(k, 0)),
                          "T4T5_reached": len(f)})
                kind = "in1"
            g = g.assign(d=dist_cols(g))
            dist = med_d(g.d)
            d.update(_dist=dist, _key=(0, grp_i(dist), 20000 + i), _kind=kind)
            new_rows[kind].append(d)
            tab_rows.append(dict(block=KIND_HOP[kind], root_id=ks, added="added", n_path=len(g),
                                 n_path_optic_lobe=int(g.d.notna().sum()),
                                 distance_columns=round(dist, 2) if not np.isnan(dist) else "", group=grp_name(grp_i(dist)),
                                 on_photoreceptor_path=True, lmc_note=part))

    # ---- moves out of LOW PRIORITY (rows on a path), duplicates of a main-block row dropped when they add nothing
    blk = {b["kind"]: b for b in blocks if b["kind"] in ("in1", "in2", "hop1", "hop2")}
    cells_in = {k: {y for r in range(b["first"] + 2, b["last"] + 1) for y in cur.get(root_at(b, r), [])} for k, b in blk.items()}
    target_of, drop = {}, set()
    for x in cell_rows:
        if x["b"]["kind"] in MAIN_OF and x["on_path"]:
            target_of[x["row"]] = MAIN_OF[x["b"]["kind"]]
    for y in lp_rows:
        live = set(cur.get(y["root"], []))
        if (y["hop"] == "PR+1" and live & role1) or (y["hop"] == "PR+2" and live & role2):
            target_of[y["row"]] = "hop1" if y["hop"] == "PR+1" else "hop2"
    for r, k in list(target_of.items()):
        if k not in blk:
            del target_of[r]
            continue
        b = blk[k]
        src = next((z for z in cell_rows + lp_rows if z["row"] == r), None)
        live = set(cur.get(src["root"], []))
        if not live & cells_in[k]:
            continue
        twin = next(q for q in range(b["first"] + 2, b["last"] + 1) if set(cur.get(root_at(b, q), [])) & live)
        tcols, scols = hdr_of[id(b)][1], src["cols"]
        mine_n = {p.strip().lower() for p in re.split(r";\s*", s(ws.cell(r, scols["notes"]).value)) if p.strip()} if "notes" in scols else set()
        twin_n = {p.strip().lower() for p in re.split(r";\s*", s(ws.cell(twin, tcols["notes"]).value)) if p.strip()} if "notes" in tcols else set()
        if mine_n <= twin_n:
            drop.add(r)
            del target_of[r]
            cab = lambda q, cc: float(s(ws.cell(q, cc["cable_um"]).value) or 0) if "cable_um" in cc else 0.0
            if cab(r, scols) > cab(twin, tcols):
                for n in ("cable_um", "neuropils"):
                    if n in tcols and n in scols:
                        B.put(a.tab, twin, tcols[n], ws.cell(r, scols[n]).value, "twin from LOW PRIORITY: its cable")
            print(f"  LOW PRIORITY row of {src['root']} dropped: its cell is on a {KIND_HOP.get(k, k)} row already", flush=True)

    # ---- column A light blue on rows on a path (an earlier such fill goes when a row is not any more)
    for x in cell_rows:
        cell = ws.cell(x["row"], 1)
        has = cell.fill is not None and cell.fill.fill_type == "solid" and cell.fill.fgColor.rgb == PATH_FILL
        if x["on_path"] and not has:
            cell.fill = PatternFill("solid", fgColor=PATH_FILL)
            B.log.append(dict(tab=a.tab, row=x["row"], column="hop", old="", new="light blue", why="on a photoreceptor path"))
        elif has and not x["on_path"]:
            cell.fill = PatternFill(fill_type=None)
            B.log.append(dict(tab=a.tab, row=x["row"], column="hop", old="light blue", new="", why="not on a photoreceptor path"))

    # ---- notes: the LMC part (replaces an earlier one)
    conflicts, n_note = [], 0
    for x in cell_rows:
        cols, r = x["cols"], x["row"]
        if "notes" not in cols:
            continue
        old = s(ws.cell(r, cols["notes"]).value)
        keep = [p.strip() for p in re.split(r";\s*", old) if p.strip() and not LMC_PART.match(p.strip())]
        ann = s(ws.cell(r, cols["annotation"]).value) if "annotation" in cols else ""
        part = lmc_ok(x["lmc"], ann)
        if x["lmc"] and not part:
            conflicts.append((x["root"], x["lmc"], ann))
        if part:
            keep.append(part)
            n_note += 1
        new = "; ".join(keep)
        if new != old:
            B.put(a.tab, r, cols["notes"], new, "putative LMC note (LMCs tab call)")
    n_note += sum(1 for d in new_rows["in2"] if d["notes"])

    # ---- distance column, far right (re-used when present)
    dc = None
    for b in sort_blocks:
        h = hdr_of[id(b)][0]
        for j in range(1, ws.max_column + 1):
            if s(ws.cell(h, j).value) == DIST:
                dc = j
    if dc is None:
        used = max(j for r in range(1, ws.max_row + 1) for j in range(1, ws.max_column + 1) if s(ws.cell(r, j).value))
        dc = used + 2
    width = max(ws.max_column, dc)
    for b in sort_blocks:
        h, cols, _ = hdr_of[id(b)]
        ref = cols.get("neuropils") or max(cols.values())
        if s(ws.cell(h, dc).value) != DIST:
            ws.cell(h, dc).value = DIST
            ws.cell(h, dc)._style = copy.copy(ws.cell(h, ref)._style)
    ws.column_dimensions[openpyxl.utils.get_column_letter(dc)].width = 18
    for x in cell_rows:
        v = round(x["dist"], 1) if not np.isnan(x["dist"]) else None
        cell = ws.cell(x["row"], dc)
        if s(cell.value) != s(v):
            cell.value = v
            cell._style = copy.copy(ws.cell(x["row"], x["cols"].get("cable_um") or 1)._style)
    B.touched.add(a.tab)

    # ---- the new layout: every block in its place; T4T5 blocks ordered; moved rows in their main block; added rows
    def below(x):  # rows the membership step moved down ('now below block criteria' note) stay at the bottom
        return "notes" in x["cols"] and "now below block criteria" in s(ws.cell(x["row"], x["cols"]["notes"]).value)
    tier = lambda x, lmc_first: 9 if below(x) else (0 if x["on_path"] else (1 if (lmc_first and x["lmc"]) else 2))  # noqa: E731
    key_of = {}
    for b in sort_blocks:
        xs = [x for x in cell_rows if x["b"] is b]
        for i, x in enumerate(xs):
            dest = target_of.get(x["row"], b["kind"])
            key_of[x["row"]] = (tier(x, KIND_HOP[dest] == "T4T5-2"), x["grp"], i if dest == b["kind"] else 10000 + i)
    plan, kept_blocks = [], []
    for b in blocks:
        kind = b["kind"]
        if kind in ("preamble", "notes") or id(b) not in hdr_of:
            body = [("old", r) for r in range(b["first"], b["last"] + 1)]
        else:
            own = [r for r in range(b["first"] + 2, b["last"] + 1) if r not in target_of and r not in drop]
            incoming = sorted(r for r, k in target_of.items() if k == kind)
            if kind in ("in1", "in2", "low_in1", "low_in2"):
                data = [r for r in own if s(ws.cell(r, 1).value) == KIND_HOP[kind]]
                other = [r for r in own if r not in data]
                ent = [(key_of[r], ("old", r)) for r in data + incoming] + [(d["_key"], ("new", d)) for d in new_rows.get(kind, [])]
                ent.sort(key=lambda e: e[0])
                rows_ = [e[1] for e in ent] + [("old", r) for r in other]
            else:
                rows_ = [("old", r) for r in own + incoming]
            if kind in ("low_in1", "low_in2", "low_path") and not any(
                    it[0] == "old" and s(ws.cell(it[1], 1).value) in ("T4T5-1", "T4T5-2", "PR+1", "PR+2") for it in rows_):
                print(f"  block '{s(ws.cell(b['first'], 1).value)[:60]}' is empty now: removed", flush=True)
                continue
            body = [("old", b["first"]), ("old", b["first"] + 1)] + rows_
        if plan and plan[-1][0] != "blank":
            plan.append(("blank",))
        kept_blocks.append(b)
        plan.extend(body)
    writers = {}
    for kind in ("in1", "in2"):
        if new_rows[kind]:
            b = blk[kind]
            h, cols, names = hdr_of[id(b)]
            names = names[:max(i for i, n in enumerate(names) if n) + 1]
            plain = next((r for r in range(h + 1, b["last"] + 1)
                          if not any(TIB.solid(ws.cell(r, j)) for j in range(2, len(names) + 1))), h + 1)
            writers[kind] = (TIB.Writer(wb, ws, b["first"], h, names, plain, cols), names)
    snap_rows = {r: [(ws.cell(r, j).value, copy.copy(ws.cell(r, j)._style),
                      copy.copy(ws.cell(r, j).hyperlink) if ws.cell(r, j).hyperlink else None) for j in range(1, width + 1)]
                 for r in range(1, last_used + 1)}
    heights = {r: ws.row_dimensions[r].height for r in range(1, last_used + 1) if ws.row_dimensions[r].height}
    blank_r = next(r for r in range(1, last_used + 1) if not any(s(v) for v, _, _ in snap_rows[r]))
    blank = [st for _, st, _ in snap_rows[blank_r]]
    for r in range(1, max(last_used, len(plan)) + 1):
        for j in range(1, width + 1):
            cell = ws.cell(r, j)
            cell.hyperlink = None
            cell.value = None
            cell._style = copy.copy(blank[j - 1])
    for r in list(ws.row_dimensions):
        ws.row_dimensions[r].height = None
    new_of, r, n_moved = {}, 0, 0
    for item in plan:
        r += 1
        if item[0] == "old":
            o = item[1]
            new_of[o] = r
            n_moved += o != r
            for j, (v, st, hl) in enumerate(snap_rows[o], 1):
                cell = ws.cell(r, j)
                cell.value = v
                cell._style = st
                if hl is not None:
                    cell.hyperlink = hl
                    cell.hyperlink.ref = cell.coordinate
            if o in heights:
                ws.row_dimensions[r].height = heights[o]
        elif item[0] == "new":
            d = item[1]
            W, names = writers[d["_kind"]]
            W.row(r, d, names=names)
            ws.cell(r, 1).fill = PatternFill("solid", fgColor=PATH_FILL)
            cell = ws.cell(r, dc)
            cell.value = round(d["_dist"], 1) if not np.isnan(d["_dist"]) else None
            B.log.append(dict(tab=a.tab, row=r, column="row", old="", new=d["root_id_curr"], why=f"added to {d['hop']}: on a photoreceptor path"))
    for o in drop:
        new_of[o] = 0
    renumber_refs(wb, B, {a.tab: new_of}, why="rows moved (T4T5 order)")

    # ---- titles: cell counts and the order clause
    blocks2, _ = HIB.blocks_of(ws, width0)
    for b in blocks2:
        if b["kind"] not in COUNTED:
            continue
        want = COUNTED[b["kind"]]
        rows_b = [q for q in range(b["first"] + 2, b["last"] + 1) if s(ws.cell(q, 1).value) in (want if isinstance(want, tuple) else (want,))]
        # rows the membership step noted 'now below block criteria' are the title's '(+ k now below the criteria ...)',
        # not part of its 'N cells' (2026-11-24: counting them too turned '99 cells (+ 4 ...)' back into '103 cells (+ 4 ...)')
        nc = next((j for j in range(1, width0 + 1) if s(ws.cell(b["first"] + 1, j).value) == "notes"), None)
        n_dn = sum(1 for q in rows_b if nc and "now below block criteria" in s(ws.cell(q, nc).value))
        n = len(rows_b) - n_dn
        t0 = t = s(ws.cell(b["first"], 1).value)
        m0 = re.search(r"(?<![\w+-])(\d+) cells\b", t)  # not the '2' of 'T4T5-2 cells'
        if m0 and int(m0.group(1)) != n:
            t = t[:m0.start()] + f"{n} cells" + t[m0.end():]
        if m0:
            t = re.sub(r"\s*\(\+ \d+ now below the criteria[^)]*\)", "", t)
            if n_dn:
                t = re.sub(r"(?<![\w+-])(\d+ cells\b)", rf"\1 (+ {n_dn} now below the criteria, noted, at the bottom)", t, count=1)
        if t != t0:
            B.put(a.tab, b["first"], 1, t, "block title: cell count")
        if b["kind"] in KIND_HOP:
            m = re.search(r"; order: .*?; within each group (.*)$", t)
            tail = re.search(r"; (most [^;]*)$", t)
            old_clause = m.group(1) if m else (tail.group(1) if tail else "as before")
            base = t[:m.start()] if m else (t[:tail.start()] if tail else t)
            first = (PATH_TXT2 + "then putative LMCs (notes), then ") if KIND_HOP[b["kind"]] == "T4T5-2" else (PATH_TXT1 + "then ")
            B.put(a.tab, b["first"], 1, f"{base}; {ORDER_TXT.format(lmc=first, dist=DIST, old=old_clause)}", "block title: row order")

    # ---- notes rows: the distance / path note (replaced or appended), the tracked_in wording
    notes = [b for b in blocks2 if b["kind"] == "notes"]
    if notes:
        nb = notes[-1]
        rows_n = range(nb["first"], nb["last"] + 1)
        hit = next((q for q in rows_n if s(ws.cell(q, 1).value).startswith("note: distance from PR group 1 column")), None)
        for q in rows_n:
            t = s(ws.cell(q, 1).value)
            if TRACKED_OLD in t:
                B.put(a.tab, q, 1, t.replace(TRACKED_OLD, TRACKED_NEW), "note: tracked_in wording")
        if hit:
            B.put(a.tab, hit, 1, NOTE_DIST + NOTE_PATH, "note: distance column")
        else:
            q = nb["last"] + 1
            if any(s(ws.cell(q, j).value) for j in range(1, width + 1)):
                ws.insert_rows(q)
            ws.cell(q, 1).value = NOTE_DIST + NOTE_PATH
            ws.cell(q, 1)._style = copy.copy(ws.cell(nb["last"], 1)._style)

    # ---- PRs tab: the groups whose photoreceptors are on these paths
    wp = wb["PRs"]
    pd_of = {}
    if ppr is not None and len(ppr):
        ppr = ppr[ppr.post_root.isin(list(role1))]
        ppr = ppr.assign(d=dist_cols(ppr))
        live_grp = {y: pr_grp[k] for k in pr_written for y in cur.get(k, [])}
        per_pr = ppr.groupby("pre_root").d.median()
        for pr_, dd in per_pr.items():
            g = live_grp.get(int(pr_), "")
            if g:
                pd_of.setdefault(g, []).append(dd)
    n_t = len(path_ids["hop3"])
    near, marked = [], 0
    for g, row in pr_line.items():
        cB, cA = wp.cell(row, 2), wp.cell(row, 1)
        if g in pd_of:
            v = [x for x in pd_of[g] if not np.isnan(x)]
            dmed = float(np.median(v)) if v else np.nan
            txt = PRS_TXT.format(n_t=n_t, n=len(pd_of[g]), d=f"{dmed:.1f}" if v else "?")
            B.put("PRs", row, 2, txt, "PR group on photoreceptor paths")
            cB.font = Font(name=cA.font.name, size=cA.font.size, bold=True)
            marked += 1
            is_near = bool(v) and dmed <= PRS_NEAR
        else:
            if PRS_RX.match(s(cB.value)):
                B.put("PRs", row, 2, "", "PR group no longer on photoreceptor paths")
            is_near = False
        has = cA.fill is not None and cA.fill.fill_type == "solid" and cA.fill.fgColor.rgb == PATH_FILL
        if is_near and not has:
            cA.fill = PatternFill("solid", fgColor=PATH_FILL)
            B.log.append(dict(tab="PRs", row=row, column="group line", old="", new="light blue", why="PR group near"))
        elif has and not is_near:
            cA.fill = PatternFill(fill_type=None)
        if is_near:
            near.append(g)
    B.touched.add("PRs")

    # ---- outputs
    full = os.path.join(TMP, f"{stamp}_t4t5_sort_full.xlsx")
    wb.save(full)
    t = pd.DataFrame(tab_rows)
    out = fig(f"{stamp}_t4t5_sort.tsv")  # figures/t4t5/
    t.to_csv(out, sep="\t", index=False)
    pd.DataFrame(B.log).to_csv(fig(f"{stamp}_t4t5_sort_changes.tsv"), sep="\t", index=False)
    print(f"\ncells per block and distance group (before moves):\n{pd.crosstab(t.block, t.group).to_string()}")
    print(f"on a photoreceptor path (first, column A light blue): {t[t.on_photoreceptor_path].block.value_counts().to_dict()}; "
          f"of them putative LMCs {int((t.on_photoreceptor_path & (t.lmc_note != '')).sum())}")
    print(f"added: T4T5-2 {len(new_rows['in2'])} {[d['root_id_curr'] for d in new_rows['in2']]}; "
          f"T4T5-1 {len(new_rows['in1'])} {[d['root_id_curr'] for d in new_rows['in1']]}")
    print(f"moved out of LOW PRIORITY: {len(target_of)}; LOW PRIORITY twins dropped: {len(drop)}")
    print(f"putative-LMC notes on T4T5-2 rows: {n_note}")
    if conflicts:
        print(f"LMC call NOT noted - the row's annotation names another type ({len(conflicts)}; root, LMCs tab call, annotation):")
        for z in conflicts:
            print("  " + "\t".join(z))
    print(f"PRs tab: {marked} groups with photoreceptors on these paths marked; {len(near)} within {PRS_NEAR:g} columns "
          f"(light blue): {sorted(near, key=int)}")
    print(f"rows moved {n_moved}; distance column {openpyxl.utils.get_column_letter(dc)}; per-cell table {out}")
    print(f"full workbook {full}")


if __name__ == "__main__":
    main()
