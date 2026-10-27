"""Shortest synaptic paths from one class of neurons to another in FAAB (read-only, live CAVE queries, <= 4 connections).

Sources and targets come from FlyTable aedes_main labels (superclass / class / subclass / subsubclass / type, matched
by regex) resolved fresh via supervoxel_id -> current root; --pr-groups adds the photoreceptors of data/pr_groups.tsv
(root at each cell's xyz). An edge is a cell -> cell connection with at least --thr synapses (several thresholds may be
given; the queries are run once, at the lowest, and every threshold is reported from the same synapses).

Search (bidirectional, 4 query stages, one UTC timestamp for all):
  F1  synapses out of the sources                 -> D1 = cells >= thr synapses from a source
  B1  synapses into the targets                   -> U1 = cells >= thr synapses onto a target
  B2  synapses into U1                            -> U2 = cells >= thr onto a U1 cell (not in U1, not a target)
  M   synapses from D1 onto U2 (both filters)     -> the middle edge of 4-connection paths
Paths: 1 = source -> target; 2 = source -> X -> target (X in D1 & U1); 3 = source -> X -> Y -> target (X in D1 & U2,
Y in U1); 4 = source -> X -> Z -> Y -> target (M edge X -> Z). Self synapses are dropped; FlyTable glia never serve
as intermediates (reported). Paths are ranked by their weakest edge, then by total synapses.

Outputs (figures/<UTC stamp>_paths_<tag>_*): summary.txt, paths.tsv (one row per path, shortest length per threshold,
capped), cells.tsv (every intermediate cell with its role, label and strongest path), ng.txt + spec.json (Spelunker
link of the strongest paths: sources, intermediates, targets). Raw edge tables are cached in ../tmp/paths_cache/.
Synapse source (--synapses, default auto, 2026-10-27): stages that filter more than --snapshot-min-ids (300) cells read
the local synapse snapshot (faab.synsnap: newest daily checkpoint, refreshed over HTTP if > 6 h old, plus ONE CAVE
catch-up to the run's timestamp shared by all stages - validated identical to live_query); small stages stay one
live_query. --synapses live restores the old behaviour, --synapses snapshot uses the snapshot for every stage.
CAVE calls >= 5 s apart. Example:
    venv/bin/python scripts/paths/class_paths.py --target "DNp01" --pr-groups --thr 5 3 --tag pr_to_DNp01"""
import argparse
import datetime
import json
import os
import re
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from faab import nglink
from faab.clients.cave import client as cave_client, roots_at_xyz, throttle
from faab.config import DATA, FIGURES, TMP
from faab.partners.sheet import utc_stamp

SERVER = "https://flytable.mrc-lmb.cam.ac.uk/"
CLS = ("superclass", "class", "subclass", "subsubclass", "type")
PR_RE = r"R1-6|R7-8|photoreceptor"
CACHE = os.path.join(TMP, "paths_cache")
LIMIT = 500000  # the materialization server caps a live query at this many rows (warning "201 - Limited query")
STAGE_SHADES = {  # Spelunker layer colors by stage along a path (sources light -> targets dark), by path length
    1: ["#f7c873", "#7a1010"],
    2: ["#f7c873", "#d9742b", "#7a1010"],
    3: ["#f7c873", "#e9953a", "#c4552a", "#7a1010"],
    4: ["#f7c873", "#eca240", "#d9742b", "#b2412a", "#7a1010"],
}
_t = [0.0]


def wait():
    _t[0] = throttle(_t[0])


def s(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v) else str(v).strip()


def flytable(client):
    """root -> (label, side, is_glia, point_xyz voxels or None) for every aedes_main row, resolved fresh (one get_roots
    call)"""
    import seaserpent as ss
    meta = ss.Table("aedes_main", server=SERVER).to_frame()
    meta.columns = [str(c) for c in meta.columns]
    sv = meta["supervoxel_id"].map(s)
    ok = sv.str.isdigit()
    wait()
    roots = np.asarray(client.chunkedgraph.get_roots([int(x) for x in sv[ok]]))
    info = {}
    for (_, row), r in zip(meta[ok].iterrows(), roots):
        label = " / ".join(x for x in (s(row.get(c)) for c in CLS) if x)
        p = re.findall(r"-?\d+(?:\.\d+)?", s(row.get("point_xyz")))
        info[int(r)] = (label, s(row.get("side")), "glia" in s(row.get("superclass")).lower(),
                        np.array([float(v) for v in p]) if len(p) == 3 else None)
    print(f"FlyTable aedes_main: {len(info)} labeled current roots", flush=True)
    return info


def side_axis(info, source_re):
    """(midpoint, unit vector toward side R) from the FlyTable sources with a side and a point: left and right eyes
    are far apart, so a point's side is the sign of its projection onto the R-L axis"""
    pts = {sd: np.array([p for lab, s_, _, p in info.values() if re.search(source_re, lab, re.I) and s_ == sd
                         and p is not None]) for sd in ("R", "L")}
    if min(len(v) for v in pts.values()) < 3:
        raise SystemExit("not enough sided FlyTable sources with points to define the R-L axis")
    cr, cl = pts["R"].mean(0), pts["L"].mean(0)
    sep = [float(np.dot(p - (cr + cl) / 2, (cr - cl) / np.linalg.norm(cr - cl))) for sd in ("R", "L") for p in pts[sd]]
    print(f"R-L axis from {len(pts['R'])} R and {len(pts['L'])} L FlyTable sources; centroid separation "
          f"{np.linalg.norm(cr - cl) / 62.5:.0f} um (x16 nm voxels), smallest |offset| {min(abs(v) for v in sep):.0f} vox",
          flush=True)
    return (cr + cl) / 2, (cr - cl) / np.linalg.norm(cr - cl)


def refresh_roots(client, ids, what):
    """{id: [current root ids]}: an expired id (edited since it was listed) is replaced by its current descendants, since
    a live query rejects expired ids. One is_latest_roots call, then one call per expired id."""
    ids = sorted(set(int(i) for i in ids))
    wait()
    ok = np.asarray(client.chunkedgraph.is_latest_roots(ids)).astype(bool)
    out = {r: [r] for r, good in zip(ids, ok) if good}
    stale = [r for r, good in zip(ids, ok) if not good]
    for r in stale:
        wait()
        out[r] = [int(x) for x in np.atleast_1d(client.chunkedgraph.get_latest_roots(r))]
    if stale:
        print(f"  {what}: {len(stale)} expired ids -> {sum(len(out[r]) for r in stale)} current roots "
              f"({', '.join(str(r) for r in stale[:5])}{', ...' if len(stale) > 5 else ''})", flush=True)
    return out


def query(client, now, tag, snap=None, **filters):
    """edge table (pre, post, n) of the synapses matching the filters at `now`.
    snap given (a faab.synsnap.Snapshot): from the local synapse snapshot - newest checkpoint plus one CAVE catch-up to
    `now`, shared by every stage of the run - with no row cap and no per-chunk calls (not cached; it is fast).
    Otherwise live: chunked live_query (<= 300 pre / 2000 post ids per call), cached by tag in ../tmp/paths_cache. A
    cached table is reused when the ids it was queried with (sidecar <tag>_ids.json) cover the request, filtered to the
    requested ids; a cache without a sidecar is assumed to cover it."""
    path, side = os.path.join(CACHE, f"{tag}.tsv"), os.path.join(CACHE, f"{tag}_ids.json")
    pre, post = filters.get("pre_pt_root_id"), filters.get("post_pt_root_id")
    if snap is not None:
        t1 = time.time()
        e = snap.edges(pre=pre, post=post, when=now, client=client)
        print(f"  {tag}: {int(e.n.sum())} synapses -> {len(e)} edges from the synapse snapshot "
              f"({snap.at(now)} + catch-up to {now:%Y-%m-%d %H:%M} UTC) in {time.time() - t1:.0f} s", flush=True)
        return e[["pre", "post", "n"]].astype({"pre": np.int64, "post": np.int64, "n": int})
    if os.path.exists(path):
        e = pd.read_csv(path, sep="\t", dtype={"pre": np.int64, "post": np.int64, "n": int})
        had = json.load(open(side)) if os.path.exists(side) else None
        covers = had is None or all(not ids or set(ids) <= set(had.get(k, []))
                                    for k, ids in (("pre", pre), ("post", post)))
        if covers:
            if had is None:  # record what the cache covers: the request plus every id that appears in it
                had = {k: sorted(set(ids or []) | set(e[k].tolist())) for k, ids in (("pre", pre), ("post", post))}
                json.dump(had, open(side, "w"))
            for k, ids in (("pre", pre), ("post", post)):
                if ids:
                    e = e[e[k].isin(set(ids))]
            print(f"  {tag}: {len(e)} edges (cached; filtered to the requested ids)", flush=True)
            return e
        print(f"  {tag}: cache does not cover the request; requerying", flush=True)
    pre_chunks = [pre[i:i + 300] for i in range(0, len(pre), 300)] if pre else [None]
    post_chunks = [post[i:i + 2000] for i in range(0, len(post), 2000)] if post else [None]
    parts, stats = [], dict(rows=0, calls=0, splits=0)

    def fetch(pc, qc):
        """one call; a result at the server's row cap (500000) is incomplete, so the chunk is split and refetched"""
        f = {k: v for k, v in (("pre_pt_root_id", pc), ("post_pt_root_id", qc)) if v}
        wait()
        syn = client.materialize.live_query("synapses_v2", timestamp=now, filter_in_dict=f,
                                            select_columns=["id", "pre_pt_root_id", "post_pt_root_id"])
        stats["calls"] += 1
        if len(syn) >= LIMIT:
            big = qc if qc and len(qc) > 1 else pc  # split the longer side that still has > 1 id
            if big is None or len(big) < 2:
                raise SystemExit(f"{tag}: a single cell has >= {LIMIT} synapses in one direction; cannot split")
            stats["splits"] += 1
            h = len(big) // 2
            for part in (big[:h], big[h:]):
                fetch(part, qc) if big is pc else fetch(pc, part)
            return
        stats["rows"] += len(syn)
        syn = syn[syn.pre_pt_root_id != syn.post_pt_root_id]
        parts.append(syn.groupby(["pre_pt_root_id", "post_pt_root_id"]).size())

    for pc in pre_chunks:
        for qc in post_chunks:
            fetch(pc, qc)
    e = pd.concat(parts).groupby(level=[0, 1]).sum().rename("n").reset_index()
    e.columns = ["pre", "post", "n"]
    e.to_csv(path, sep="\t", index=False)
    json.dump({"pre": sorted(pre or []), "post": sorted(post or [])}, open(side, "w"))
    print(f"  {tag}: {stats['rows']} synapses in {stats['calls']} calls ({stats['splits']} chunks split at the row cap) "
          f"-> {len(e)} edges", flush=True)
    return e


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=PR_RE, help=f"regex on the FlyTable label of the sources (default {PR_RE!r})")
    ap.add_argument("--target", help="regex on the FlyTable label of the targets, e.g. DNp01")
    ap.add_argument("--source-ids", metavar="FILE", help="root ids (one per line) used as sources instead of --source")
    ap.add_argument("--target-ids", metavar="FILE", help="root ids (one per line) used as targets instead of --target")
    ap.add_argument("--max-connections", type=int, default=4, choices=[1, 2, 3, 4],
                    help="longest path searched; stages beyond it are not queried (2 = F1 + B1 only)")
    ap.add_argument("--pr-groups", action="store_true", help="add the photoreceptors of data/pr_groups.tsv as sources")
    ap.add_argument("--side", choices=["R", "L"], help="only sources on this side (FlyTable side column; pr_groups.tsv "
                    "cells by their point's side of the R-L axis through the FlyTable sources)")
    ap.add_argument("--thr", type=int, nargs="+", default=[5, 3], help="synapses per edge; each is reported")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-paths", type=int, default=3000, help="paths kept per threshold in paths.tsv")
    ap.add_argument("--paths-connections", type=int, nargs="+", metavar="K",
                    help="path lengths written to paths.tsv / cells.tsv (default: the shortest found)")
    ap.add_argument("--link-connections", type=int, metavar="K", help="path length shown in the Spelunker link (default shortest)")
    ap.add_argument("--link-min", type=int, default=0, help="only paths whose weakest edge has >= this many synapses in the link")
    ap.add_argument("--link-paths", type=int, default=10, help="strongest paths shown in the Spelunker link (0 = all that qualify)")
    ap.add_argument("--fresh", action="store_true", help="ignore cached edge tables")
    ap.add_argument("--redo", nargs="+", default=[], metavar="STAGE", help="requery these stages (F1 B1 B2 M)")
    ap.add_argument("--synapses", choices=["auto", "snapshot", "live"], default="auto",
                    help="where stage synapses come from: auto = the local synapse snapshot (faab.synsnap, updated to now) "
                         "for stages filtering more than --snapshot-min-ids cells, live_query for the small ones")
    ap.add_argument("--snapshot-min-ids", type=int, default=300)
    a = ap.parse_args()
    t0, stamp = time.time(), utc_stamp()
    os.makedirs(CACHE, exist_ok=True)
    for f in os.listdir(CACHE):
        stage = f[len(a.tag) + 1:-4] if f.startswith(a.tag + "_") else None
        if stage and (a.fresh or stage in a.redo):
            os.remove(os.path.join(CACHE, f))
            print(f"cache dropped: {f}", flush=True)
    client = cave_client()
    now = datetime.datetime.now(datetime.timezone.utc)
    info = flytable(client)
    label = lambda r: info.get(r, ("", "", False, None))[0]
    glia = {r for r, (_, _, g, _) in info.items() if g}
    axis = side_axis(info, a.source) if a.side else None
    side_of = lambda p: ("R" if np.dot(p - axis[0], axis[1]) > 0 else "L") if p is not None else ""

    read_ids = lambda path: [int(x) for x in open(path).read().split() if x.strip().isdigit()]
    if a.source_ids:
        src_class = {r: (label(r) or "listed") for r in read_ids(a.source_ids)
                     if not a.side or info.get(r, ("", a.side, 0, 0))[1] in (a.side, "")}
        print(f"sources from {a.source_ids}: {len(src_class)} cells", flush=True)
    else:
        src_class = {r: lab for r, (lab, sd, _, _) in info.items() if re.search(a.source, lab, re.I)
                     and (not a.side or sd == a.side)}
    n_ft = len(src_class)
    if a.pr_groups:
        g = pd.read_csv(os.path.join(DATA, "pr_groups.tsv"), sep="\t", dtype=str)
        wait()
        at = roots_at_xyz(list(g.xyz))
        n_new, n_side, groups_side = 0, defaultdict(int), defaultdict(set)
        for row, r in zip(g.itertuples(), at):
            r = int(r) if int(r) > 0 else int(row.root_id_curr)
            sd = side_of(np.array([float(v) for v in row.xyz.split(",")])) if a.side else ""
            n_side[sd] += 1
            groups_side[sd].add(int(row.group))
            if r not in src_class and (not a.side or sd == a.side):
                src_class[r] = f"pr_groups.tsv group {row.group} {row._2}"
                n_new += 1
        if a.side:
            both = sorted(groups_side["R"] & groups_side["L"])
            print(f"\npr_groups.tsv cells by side: {dict(n_side)}; groups with cells on both sides: {both or 'none'}", flush=True)
        print(f"sources: {n_ft} FlyTable + {n_new} from pr_groups.tsv" + (f" (side {a.side} only)" if a.side else ""), flush=True)
    if a.target_ids:
        targets = {r: (label(r) or "listed") for r in read_ids(a.target_ids)}
    elif a.target:
        targets = {r: lab for r, (lab, _, _, _) in info.items() if re.search(a.target, lab, re.I)}
    else:
        raise SystemExit("give --target or --target-ids")
    print(f"sources {len(src_class)}; targets {len(targets)}" +
          (": " + "; ".join(f"{r} ({lab}, side {info.get(r, ('', '', 0, 0))[1]})" for r, lab in targets.items())
           if len(targets) <= 6 else ""), flush=True)
    target_re = a.target or os.path.basename(a.target_ids)
    if not targets or not src_class:
        raise SystemExit("no sources or no targets")
    cur = refresh_roots(client, list(src_class) + list(targets), "sources + targets")
    src_class = {n: v for r, v in src_class.items() for n in cur[r]}
    targets = {n: v for r, v in targets.items() for n in cur[r]}
    sources, tgt = sorted(src_class), sorted(targets)
    lo = min(a.thr)
    out_tag = a.tag + (f"_sources{a.side}" if a.side else "")

    # ---- query stages, at the lowest threshold; big stages from the local synapse snapshot (one shared catch-up)
    _snap = []

    def src(*id_lists):
        """the Snapshot for a stage filtering these id lists, or None for a live query"""
        n = max((len(x) for x in id_lists if x), default=0)
        if a.synapses == "live" or (a.synapses == "auto" and n <= a.snapshot_min_ids):
            return None
        if not _snap:
            from faab import synsnap
            tag = synsnap.refresh(log=lambda m: print("  " + m, flush=True))
            _snap.append(synsnap.Snapshot(log=lambda m: print("  " + m, flush=True)))
            print(f"  synapse snapshot: newest local checkpoint {tag}", flush=True)
        return _snap[0]

    print("F1: synapses out of the sources", flush=True)
    F1 = query(client, now, f"{a.tag}_F1", src(sources), pre_pt_root_id=sources)
    empty = F1.iloc[:0]
    B1 = B2 = M = empty
    D1 = sorted(set(F1[F1.n >= lo].post) - set(sources) - glia)
    U1 = U2 = []
    if a.max_connections >= 2:
        print("B1: synapses into the targets", flush=True)
        B1 = query(client, now, f"{a.tag}_B1", src(tgt), post_pt_root_id=tgt)
        U1 = sorted(set(B1[B1.n >= lo].pre) - set(tgt) - glia)
        print(f"  D1 (>= {lo} from a source): {len(D1)} cells (glia dropped: {len(set(F1[F1.n >= lo].post) & glia)}); "
              f"U1 (>= {lo} onto a target): {len(U1)} cells (glia dropped: {len(set(B1[B1.n >= lo].pre) & glia)})", flush=True)
    if a.max_connections >= 3:
        print("B2: synapses into U1", flush=True)
        B2 = query(client, now, f"{a.tag}_B2", src(U1), post_pt_root_id=U1)
        U2 = sorted(set(B2[B2.n >= lo].pre) - set(U1) - set(tgt) - glia)
        print(f"  U2 (>= {lo} onto a U1 cell, new): {len(U2)} cells", flush=True)
    if a.max_connections >= 4:
        print("M: synapses from D1 onto U2", flush=True)
        M = query(client, now, f"{a.tag}_M", src(D1, U2), pre_pt_root_id=D1, post_pt_root_id=U2) if D1 and U2 else empty

    # ---- paths per threshold
    def adj(e, thr):
        d = defaultdict(dict)
        for pre, post, n in e[e.n >= thr].itertuples(index=False):
            d[int(pre)][int(post)] = int(n)
        return d

    out_rows, cell_rows, summary, best_link = [], [], [], None
    for thr in sorted(a.thr, reverse=True):
        f1, b1, b2, m = adj(F1, thr), adj(B1, thr), adj(B2, thr), adj(M, thr)
        d1 = {x for v in f1.values() for x in v} - set(sources) - glia
        u1 = {y for y in b1} - set(tgt) - glia
        u2 = {z for z in b2 if any(y in u1 for y in b2[z])} - u1 - set(tgt) - glia
        paths = {1: [], 2: [], 3: [], 4: []}
        for src in sources:
            for x, n1 in f1.get(src, {}).items():
                if x in targets:
                    paths[1].append(([src, x], [n1]))
                if x not in d1:
                    continue
                for t_, n2 in b1.get(x, {}).items():  # 2: src -> x -> target
                    paths[2].append(([src, x, t_], [n1, n2]))
                if x in u2:  # 3: src -> x -> y -> target
                    for y, n2 in b2[x].items():
                        if y in u1:
                            for t_, n3 in b1[y].items():
                                paths[3].append(([src, x, y, t_], [n1, n2, n3]))
                for z, n2 in m.get(x, {}).items():  # 4: src -> x -> z -> y -> target
                    if z in u2:
                        for y, n3 in b2[z].items():
                            if y in u1:
                                for t_, n4 in b1[y].items():
                                    paths[4].append(([src, x, z, y, t_], [n1, n2, n3, n4]))
        counts = {k: len(v) for k, v in paths.items()}
        shortest = next((k for k in (1, 2, 3, 4) if paths[k]), None)
        line = f"thr >= {thr}: paths by connections {counts}; shortest = {shortest or 'none within 4'}"
        for k in (1, 2, 3, 4):  # strength profile per length: paths whose weakest edge reaches each level
            if paths[k]:
                mins = np.array([min(ns) for _, ns in paths[k]])
                line += (f"\n    {k} connections: weakest edge >= " +
                         ", ".join(f"{lv}: {int((mins >= lv).sum())}" for lv in (5, 10, 15, 20, 30) if lv >= thr) +
                         f"; distinct cells per stage {[len({c[i] for c, _ in paths[k]}) for i in range(1, k)]}")
        want_k = a.link_connections or shortest
        for k in [k for k in (a.paths_connections or [shortest]) if k and paths[k]]:
            ranked = sorted(paths[k], key=lambda p: (-min(p[1]), -sum(p[1])))
            cells_mid = {c for p in ranked for c in p[0][1:-1]}
            srcs_hit = {p[0][0] for p in ranked}
            tgts_hit = {p[0][-1] for p in ranked}
            line += (f"\n    written: {len(ranked)} paths of {k} connections through {len(cells_mid)} intermediate cells, "
                     f"from {len(srcs_hit)} sources onto targets {sorted(tgts_hit)}; strongest weakest-edge "
                     f"{min(ranked[0][1])}")
            for cells, ns in ranked[:a.max_paths]:
                row = dict(thr=thr, connections=k, min_nsyn=min(ns), sum_nsyn=sum(ns),
                           source=cells[0], source_class=src_class.get(cells[0], ""))
                for j in range(1, 5):  # fixed columns for up to 4 connections (k is the path length)
                    c = cells[j] if j < len(cells) - 1 else ""
                    row[f"cell{j}"] = c
                    row[f"label{j}"] = label(c) if c else ""
                    row[f"n{j - 1}{j}"] = ns[j - 1] if j - 1 < len(ns) else ""
                row.update(target=cells[-1], target_label=targets[cells[-1]], n_to_target=ns[-1])
                out_rows.append(row)
            best = defaultdict(lambda: dict(paths=0, best_min=0, roles=set(), srcs=set(), tgts=set(), syn_in={}, syn_out={}))
            for cells, ns in ranked:
                for h, c in enumerate(cells[1:-1], 1):  # h = hop index (k is the path length)
                    b = best[c]
                    b["paths"] += 1
                    b["best_min"] = max(b["best_min"], min(ns))
                    b["roles"].add(f"hop{h}")
                    b["srcs"].add(cells[0])
                    b["tgts"].add(cells[-1])
                    b["syn_in"][cells[h - 1]] = ns[h - 1]   # synapses from the cell before it on the path
                    b["syn_out"][cells[h + 1]] = ns[h]      # synapses onto the cell after it
            for c, b in sorted(best.items(), key=lambda kv: (-len(kv[1]["srcs"]), -kv[1]["best_min"], -kv[1]["paths"])):
                cell_rows.append(dict(thr=thr, connections=k, cell=c, label=label(c), side=info.get(c, ("", "", 0, 0))[1],
                                      roles="+".join(sorted(b["roles"])), paths=b["paths"], best_min_nsyn=b["best_min"],
                                      n_sources=len(b["srcs"]), n_targets=len(b["tgts"]),
                                      syn_from_prev=sum(b["syn_in"].values()), syn_to_next=sum(b["syn_out"].values()),
                                      targets=";".join(f"{t}:{n}" for t, n in sorted(b["syn_out"].items(), key=lambda kv: -kv[1])[:6])
                                      if k == 2 else "",
                                      in_D1=c in d1, in_U1=c in u1, in_U2=c in u2))
            if best_link is None and k == want_k:  # the link: highest threshold with paths of the wanted length
                strong = [p for p in ranked if min(p[1]) >= a.link_min]
                if strong:
                    best_link = (thr, k, strong[:a.link_paths] if a.link_paths else strong)
        print(line, flush=True)
        summary.append(line)

    # ---- outputs
    base = os.path.join(FIGURES, f"{stamp}_paths_{out_tag}")
    pd.DataFrame(out_rows).to_csv(base + "_paths.tsv", sep="\t", index=False)
    pd.DataFrame(cell_rows).to_csv(base + "_cells.tsv", sep="\t", index=False)
    head = [f"paths {out_tag}: sources {a.source_ids or '/' + a.source + '/'} ({len(sources)} cells"
            f"{' incl. pr_groups.tsv' if a.pr_groups else ''}{', side ' + a.side + ' only' if a.side else ''}) -> "
            f"targets {a.target_ids or '/' + a.target + '/'} ({len(tgt)}); synapses at {now:%Y-%m-%d %H:%M} UTC ("
            f"{'live_query' if not _snap else 'snapshot ' + _snap[0].at(now) + ' + catch-up for stages > ' + str(a.snapshot_min_ids) + ' cells, live_query for the rest'}); "
            f"edges = cell pairs with >= thr "
            f"synapses; glia (FlyTable) never intermediates; self synapses dropped",
            f"set sizes at >= {lo}: D1 {len(D1)}, U1 {len(U1)}, U2 {len(U2)}; M edges {len(M)}"]
    if best_link:
        thr, k, ranked = best_link
        # one layer per stage of the paths, shaded light (sources) to dark (targets); a cell that sits at two stages in
        # different paths takes its earliest
        stage_of = {}
        for cells, _ in ranked:
            for i, c in enumerate(cells):
                stage_of[c] = min(i, stage_of.get(c, i))
        names = ["sources"] + [f"hop {i}" for i in range(1, k)] + [f"targets {target_re}"]
        layers = []
        for i, name in enumerate(names):
            ids = sorted(str(c) for c, st in stage_of.items() if st == i)
            layers.append({"name": f"{i}: {name} ({len(ids)})", "color": STAGE_SHADES[k][i], "ids": ids})
        spec = {"meshes": [{"kind": "brain", "alpha": 1.0}], "neurons": layers, "view": "anterior", "layout": "3d",
                "select": layers[1]["name"], "image": True}
        json.dump(spec, open(base + "_spec.json", "w"), indent=1)
        url = nglink.to_url(nglink.build(spec))
        open(base + "_ng.txt", "w").write(url + "\n")
        head.append(f"Spelunker link: {len(ranked)} paths of {k} connections at >= {thr}"
                    f"{f', weakest edge >= {a.link_min}' if a.link_min else ''}, one layer per stage "
                    f"({', '.join(l['name'] for l in layers)}) ({base}_ng.txt)")
    open(base + "_summary.txt", "w").write("\n".join(head + summary) + "\n")
    print("\n".join(head), flush=True)
    print(f"wrote {base}_summary.txt / _paths.tsv / _cells.tsv; {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
