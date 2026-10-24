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
_t = [0.0]


def wait():
    _t[0] = throttle(_t[0])


def s(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v) else str(v).strip()


def flytable(client):
    """root -> (label, side, is_glia) for every aedes_main row, resolved fresh (one get_roots call)"""
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
        info[int(r)] = (label, s(row.get("side")), "glia" in s(row.get("superclass")).lower())
    print(f"FlyTable aedes_main: {len(info)} labeled current roots", flush=True)
    return info


def query(client, now, tag, **filters):
    """edge table (pre, post, n) of the live synapses matching the filters, chunked (<= 300 pre / 2000 post ids per
    call) and cached by tag in ../tmp/paths_cache"""
    path = os.path.join(CACHE, f"{tag}.tsv")
    if os.path.exists(path):
        e = pd.read_csv(path, sep="\t", dtype={"pre": np.int64, "post": np.int64, "n": int})
        print(f"  {tag}: {len(e)} edges (cached)", flush=True)
        return e
    pre, post = filters.get("pre_pt_root_id"), filters.get("post_pt_root_id")
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
    print(f"  {tag}: {stats['rows']} synapses in {stats['calls']} calls ({stats['splits']} chunks split at the row cap) "
          f"-> {len(e)} edges", flush=True)
    return e


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=PR_RE, help=f"regex on the FlyTable label of the sources (default {PR_RE!r})")
    ap.add_argument("--target", required=True, help="regex on the FlyTable label of the targets, e.g. DNp01")
    ap.add_argument("--pr-groups", action="store_true", help="add the photoreceptors of data/pr_groups.tsv as sources")
    ap.add_argument("--thr", type=int, nargs="+", default=[5, 3], help="synapses per edge; each is reported")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-paths", type=int, default=3000, help="paths kept per threshold in paths.tsv")
    ap.add_argument("--link-paths", type=int, default=10, help="strongest paths shown in the Spelunker link")
    ap.add_argument("--fresh", action="store_true", help="ignore cached edge tables")
    ap.add_argument("--redo", nargs="+", default=[], metavar="STAGE", help="requery these stages (F1 B1 B2 M)")
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
    label = lambda r: info.get(r, ("", "", False))[0]
    glia = {r for r, (_, _, g) in info.items() if g}

    src_class = {r: lab for r, (lab, _, _) in info.items() if re.search(a.source, lab, re.I)}
    if a.pr_groups:
        g = pd.read_csv(os.path.join(DATA, "pr_groups.tsv"), sep="\t", dtype=str)
        wait()
        at = roots_at_xyz(list(g.xyz))
        n_new = 0
        for row, r in zip(g.itertuples(), at):
            r = int(r) if int(r) > 0 else int(row.root_id_curr)
            if r not in src_class:
                src_class[r] = f"pr_groups.tsv group {row.group} {row._2}"
                n_new += 1
        print(f"sources: {len(src_class) - n_new} FlyTable + {n_new} from pr_groups.tsv", flush=True)
    targets = {r: lab for r, (lab, _, _) in info.items() if re.search(a.target, lab, re.I)}
    print(f"sources {len(src_class)}; targets {len(targets)}: " +
          "; ".join(f"{r} ({lab}, side {info[r][1]})" for r, lab in targets.items()), flush=True)
    if not targets or not src_class:
        raise SystemExit("no sources or no targets")
    sources, tgt = sorted(src_class), sorted(targets)
    lo = min(a.thr)

    # ---- query stages, at the lowest threshold
    print("F1: synapses out of the sources", flush=True)
    F1 = query(client, now, f"{a.tag}_F1", pre_pt_root_id=sources)
    print("B1: synapses into the targets", flush=True)
    B1 = query(client, now, f"{a.tag}_B1", post_pt_root_id=tgt)
    D1 = sorted(set(F1[F1.n >= lo].post) - set(sources) - glia)
    U1 = sorted(set(B1[B1.n >= lo].pre) - set(tgt) - glia)
    print(f"  D1 (>= {lo} from a source): {len(D1)} cells (glia dropped: {len(set(F1[F1.n >= lo].post) & glia)}); "
          f"U1 (>= {lo} onto a target): {len(U1)} cells (glia dropped: {len(set(B1[B1.n >= lo].pre) & glia)})", flush=True)
    print("B2: synapses into U1", flush=True)
    B2 = query(client, now, f"{a.tag}_B2", post_pt_root_id=U1)
    U2 = sorted(set(B2[B2.n >= lo].pre) - set(U1) - set(tgt) - glia)
    print(f"  U2 (>= {lo} onto a U1 cell, new): {len(U2)} cells", flush=True)
    print("M: synapses from D1 onto U2", flush=True)
    M = query(client, now, f"{a.tag}_M", pre_pt_root_id=D1, post_pt_root_id=U2) if D1 and U2 else F1.iloc[:0]

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
        if shortest:
            ranked = sorted(paths[shortest], key=lambda p: (-min(p[1]), -sum(p[1])))
            cells_mid = {c for p in ranked for c in p[0][1:-1]}
            srcs_hit = {p[0][0] for p in ranked}
            tgts_hit = {p[0][-1] for p in ranked}
            line += (f"; {len(ranked)} paths of {shortest} connections through {len(cells_mid)} intermediate cells, "
                     f"from {len(srcs_hit)} sources onto targets {sorted(tgts_hit)}; strongest weakest-edge "
                     f"{min(ranked[0][1])}")
            for cells, ns in ranked[:a.max_paths]:
                row = dict(thr=thr, connections=shortest, min_nsyn=min(ns), sum_nsyn=sum(ns),
                           source=cells[0], source_class=src_class.get(cells[0], ""))
                for k in range(1, 5):
                    c = cells[k] if k < len(cells) - 1 else ""
                    row[f"cell{k}"] = c
                    row[f"label{k}"] = label(c) if c else ""
                    row[f"n{k - 1}{k}"] = ns[k - 1] if k - 1 < len(ns) else ""
                row.update(target=cells[-1], target_label=targets[cells[-1]], n_to_target=ns[-1])
                out_rows.append(row)
            best = defaultdict(lambda: dict(paths=0, best_min=0, roles=set()))
            for cells, ns in ranked:
                for k, c in enumerate(cells[1:-1], 1):
                    b = best[c]
                    b["paths"] += 1
                    b["best_min"] = max(b["best_min"], min(ns))
                    b["roles"].add(f"hop{k}")
            for c, b in sorted(best.items(), key=lambda kv: (-kv[1]["best_min"], -kv[1]["paths"])):
                cell_rows.append(dict(thr=thr, connections=shortest, cell=c, label=label(c), side=info.get(c, ("", "", 0))[1],
                                      roles="+".join(sorted(b["roles"])), paths=b["paths"], best_min_nsyn=b["best_min"],
                                      in_D1=c in d1, in_U1=c in u1, in_U2=c in u2))
            if best_link is None:
                best_link = (thr, shortest, ranked[:a.link_paths])
        print(line, flush=True)
        summary.append(line)

    # ---- outputs
    base = os.path.join(FIGURES, f"{stamp}_paths_{a.tag}")
    pd.DataFrame(out_rows).to_csv(base + "_paths.tsv", sep="\t", index=False)
    pd.DataFrame(cell_rows).to_csv(base + "_cells.tsv", sep="\t", index=False)
    head = [f"paths {a.tag}: sources /{a.source}/ ({len(sources)} cells{' incl. pr_groups.tsv' if a.pr_groups else ''}) -> "
            f"targets /{a.target}/ ({len(tgt)}); live synapses at {now:%Y-%m-%d %H:%M} UTC; edges = cell pairs with >= thr "
            f"synapses; glia (FlyTable) never intermediates; self synapses dropped",
            f"set sizes at >= {lo}: D1 {len(D1)}, U1 {len(U1)}, U2 {len(U2)}; M edges {len(M)}"]
    if best_link:
        thr, k, ranked = best_link
        mids = sorted({c for p in ranked for c in p[0][1:-1]})
        spec = {"meshes": [{"kind": "brain", "alpha": 1.0}],
                "neurons": [{"name": f"sources ({len({p[0][0] for p in ranked})})", "color": "#f0a030",
                             "ids": sorted({str(p[0][0]) for p in ranked})},
                            {"name": f"intermediates ({len(mids)}, {k - 1} per path)", "color": "#3b6fb0", "ids": [str(c) for c in mids]},
                            {"name": f"targets {a.target}", "color": "#e6194b", "ids": sorted({str(p[0][-1]) for p in ranked})}],
                "view": "anterior", "layout": "3d", "select": f"intermediates ({len(mids)}, {k - 1} per path)", "image": True}
        json.dump(spec, open(base + "_spec.json", "w"), indent=1)
        url = nglink.to_url(nglink.build(spec))
        open(base + "_ng.txt", "w").write(url + "\n")
        head.append(f"Spelunker link: the {len(ranked)} strongest paths of {k} connections at >= {thr} ({base}_ng.txt)")
    open(base + "_summary.txt", "w").write("\n".join(head + summary) + "\n")
    print("\n".join(head), flush=True)
    print(f"wrote {base}_summary.txt / _paths.tsv / _cells.tsv; {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
