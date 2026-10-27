"""Does a list of cells have the morphology of a SeaTable-labeled type? Pose-invariant NBLAST (navis.nblast_align,
rigid + scale) of every query cell against the labeled reference cells, calibrated by two controls scored the same way:
  positive  each reference cell against the other references (leave-one-out)
  negative  cells of other types (--neg-seatable REGEX and/or --neg-pickle dotprops), ideally similar in size and shape
CAUTION (2026-10-27): for small columnar cells (~80 L2 points, e.g. T4/T5) rigid + scale alignment makes every
columnar cell fit (T4/T5 leave-one-out 0.70 vs L1-L5/LMC/Mi1 0.65; a quarter of labeled T4/T5 failed their own type).
For those, centroid-aligned NBLAST with a nearest-neighbour vote over references + negatives separated them
(tmp/t4t5_knn.py: 97% of references, 3% of negatives); this script still supplies the fetch/cache and the controls.
Query ids: --query-file, ids separated by commas/spaces, one group per line (groups are reported separately).
Expired query roots are replaced by their current roots (get_latest_roots). References are resolved fresh from SeaTable
(supervoxel_id -> current root, one get_roots call).

Morphology: L2 dotprops (nm -> um), one CAVE call per cell >= 5 s apart, thinned to <= 1500 points, cached in
figures/type_copies_dotprops.pkl (shared with type_copies.py). Side: every cell is put in the left-side frame (cells on
the right are mirrored x -> 2*mid - x; rigid alignment cannot reflect). The side is SeaTable's when the cell has one,
otherwise its centroid against the midline (midpoint of the median centroid x of SeaTable L and R cells).

Scores per cell vs each reference: fwd = cell as query, rev = reference as query, sym = mean. Per cell: top5 = mean of
its 5 best sym scores (a reference is never scored against itself), best reference and its label, and the best label
group (mean of the top 3 within each group). Calls from the negative distribution of top5: match >= its max,
likely >= its 95th percentile, ambiguous >= its 80th percentile, else no match; cells with fewer than --min-ratio x the
references' median points are 'too small to call'. Pairwise scores are cached per cell in
figures/type_match_<tag>_scores.pkl (resumable; recomputed if the reference set changes).
Outputs figures/<stamp>_type_match_<tag>.tsv (+ _groups.tsv, _ng.txt, _spec.json). Example:
  venv/bin/python scripts/morph/type_match.py --tag t4t5 --query-file ../tmp/t4t5_query_ids_20261026.txt \
      --ref-seatable "\\bT4|\\bT5" --neg-seatable "L1-L5|LMC" --neg-pickle figures/mi1_dotprops.pkl:Mi1"""
import argparse
import json
import os
import pickle
import time

import numpy as np

if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd

from faab import nglink
from faab.clients.cave import DATASTACK, client as cave_client, throttle
from faab.config import FIGURES
from faab.partners.sheet import utc_stamp

SERVER = "https://flytable.mrc-lmb.cam.ac.uk/"
CLS = ("superclass", "class", "subclass", "subsubclass", "type")
POINTS = 1500
CACHE = os.path.join(FIGURES, "type_copies_dotprops.pkl")
_t = [0.0]


def wait():
    _t[0] = throttle(_t[0])


def s(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v) else str(v).strip()


def thin(dp, navis):
    if len(dp.points) > POINTS:
        idx = np.random.default_rng(0).choice(len(dp.points), POINTS, replace=False)
        dp = navis.Dotprops(dp.points[idx], k=5)
    dp.units = "um"
    return dp


def mirrored(dp, mid, navis):
    """x -> 2*mid - x; the tangent vectors are flipped too (kept from the L2 data, not recomputed)"""
    pts = np.asarray(dp.points, dtype=float).copy()
    pts[:, 0] = 2 * mid - pts[:, 0]
    if getattr(dp, "vect", None) is not None:
        v = np.asarray(dp.vect, dtype=float).copy()
        v[:, 0] *= -1
        try:  # L2 dotprops are not k-NN based: their .alpha raises instead of returning None
            alpha = dp.alpha
        except ValueError:
            alpha = None
        out = navis.Dotprops(pts, k=None, vect=v, alpha=alpha)
    else:
        out = navis.Dotprops(pts, k=5)
    out.units = "um"
    return out


def topk(row, k):
    v = np.sort(row[~np.isnan(row)])[::-1]
    return float(np.mean(v[:k])) if len(v) else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--query-file", required=True)
    ap.add_argument("--ref-seatable", required=True, help="regex on the SeaTable label for the reference type")
    ap.add_argument("--neg-seatable", help="regex on the SeaTable label for negative-control cells")
    ap.add_argument("--neg-pickle", nargs="*", default=[], help="PATH:NAME dotprops NeuronList/dict (um) as extra negatives")
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=40, help="cells per nblast_align call (progress / resume unit)")
    ap.add_argument("--min-points", type=int, default=10)
    ap.add_argument("--min-ratio", type=float, default=0.2)
    a = ap.parse_args()
    import navis
    import seaserpent as ss
    from fafbseg import flywire
    t0, stamp = time.time(), utc_stamp()
    client = cave_client()

    # ---- SeaTable: references and SeaTable negatives
    meta = ss.Table("aedes_main", server=SERVER).to_frame()
    meta.columns = [str(c) for c in meta.columns]
    meta = meta[meta["supervoxel_id"].map(s).str.isdigit()].copy()
    wait()
    meta["root"] = [str(int(r)) for r in np.asarray(client.chunkedgraph.get_roots([int(s(x)) for x in meta["supervoxel_id"]]))]
    meta["label"] = [" / ".join(x for x in (s(r.get(c)) for c in CLS) if x) for _, r in meta.iterrows()]
    meta = meta.drop_duplicates("root")
    lab, side_st = dict(zip(meta.root, meta.label)), dict(zip(meta.root, meta["side"].map(s)))
    refs = sorted(meta[meta.label.str.contains(a.ref_seatable, regex=True)].root)
    negs_st = sorted(meta[meta.label.str.contains(a.neg_seatable, regex=True)].root) if a.neg_seatable else []
    negs_st = [r for r in negs_st if r not in refs]
    print(f"references /{a.ref_seatable}/: {len(refs)}; SeaTable negatives /{a.neg_seatable}/: {len(negs_st)}", flush=True)

    # ---- queries (grouped by line); expired roots -> current roots
    rows = []
    for gi, line in enumerate(open(a.query_file).read().strip().splitlines(), 1):
        rows += [(x.strip(), gi) for x in line.replace(",", " ").split() if x.strip()]
    q = pd.DataFrame(rows, columns=["given", "group"]).drop_duplicates("given")
    wait()
    ok = np.asarray(client.chunkedgraph.is_latest_roots([int(x) for x in q.given])).astype(bool)
    qrows = []
    for (g, grp), good in zip(q.itertuples(index=False), ok):
        if good:
            qrows.append((g, g, grp))
        else:
            wait()
            for n in np.atleast_1d(client.chunkedgraph.get_latest_roots(int(g))):
                qrows.append((g, str(int(n)), grp))
            print(f"query {g} was edited; now {[r[1] for r in qrows if r[0] == g]}", flush=True)
    q = pd.DataFrame(qrows, columns=["given", "cell", "group"])
    print(f"queries: {q.given.nunique()} given, {q.cell.nunique()} current roots, groups {q.groupby('group').size().to_dict()}", flush=True)

    # ---- dotprops (cached, resumable)
    cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
    need = list(dict.fromkeys(refs + list(q.cell) + negs_st))
    todo = [r for r in need if r not in cache]
    print(f"dotprops cached {len(need) - len(todo)} of {len(need)}; to fetch {len(todo)} (~{len(todo) * 5.5 / 60:.0f} min)", flush=True)
    failed = []
    for k, r in enumerate(todo, 1):
        wait()
        try:
            dp = flywire.get_l2_dotprops([int(r)], dataset=DATASTACK, max_threads=1, progress=False)[0] / 1000.0
            dp = thin(dp, navis)
            dp.id = dp.name = r
            cache[r] = dp
        except Exception as e:
            failed.append((r, str(e)[:80]))
        if k % 25 == 0 or k == len(todo):
            pickle.dump(cache, open(CACHE, "wb"))
            print(f"  fetched {k}/{len(todo)} ({len(failed)} failed) at {time.time() - t0:.0f} s", flush=True)
    if failed:
        print(f"failed: {failed[:10]}", flush=True)
    neuron = {r: cache[r] for r in need if r in cache}
    role, nlabel = {}, {}
    for r in negs_st:
        role[r], nlabel[r] = "negative", lab.get(r, "")
    for spec in a.neg_pickle:
        path, name = (spec.rsplit(":", 1) + [os.path.basename(spec)])[:2] if ":" in spec else (spec, os.path.basename(spec))
        d = pickle.load(open(path, "rb"))
        items = d.items() if isinstance(d, dict) else [(str(n.id), n) for n in d]
        n0 = 0
        for k, dp in items:
            key = f"{name}:{k}"
            if len(dp.points) >= a.min_points:
                neuron[key] = thin(dp, navis)
                role[key], nlabel[key] = "negative", name
                n0 += 1
        print(f"negatives from {path}: {n0} ({name})", flush=True)
    for r in refs:
        role[r] = "reference"
    for r in q.cell:
        role.setdefault(r, "query")  # a query that is also a reference stays a reference (scored leave-one-out)

    # ---- side and mirroring into the left frame
    cx = {k: float(np.mean(v.points[:, 0])) for k, v in neuron.items()}
    xs = pd.DataFrame([(k, cx[k], side_st.get(k, "")) for k in neuron if side_st.get(k, "") in ("L", "R")], columns=["k", "x", "side"])
    ml, mr = xs[xs.side == "L"].x.median(), xs[xs.side == "R"].x.median()
    mid = (ml + mr) / 2
    sgn = 1 if ml < mr else -1  # +1: left side has smaller x
    side = {k: (side_st.get(k) if side_st.get(k) in ("L", "R") else ("L" if sgn * (cx[k] - mid) < 0 else "R")) for k in neuron}
    agree = np.mean([("L" if sgn * (x - mid) < 0 else "R") == sd for x, sd in zip(xs.x, xs.side)])
    print(f"midline x {mid:.1f} um (median L {ml:.1f}, R {mr:.1f}); SeaTable side agrees with centroid for {agree:.0%} of "
          f"{len(xs)} sided cells", flush=True)
    frame = {k: (mirrored(v, mid, navis) if side[k] == "R" else v) for k, v in neuron.items()}

    refs = [r for r in refs if r in frame and len(frame[r].points) >= a.min_points]
    cells = [k for k in frame if len(frame[k].points) >= a.min_points]
    small = [k for k in frame if len(frame[k].points) < a.min_points]
    ref_pts = float(np.median([len(frame[r].points) for r in refs]))
    print(f"scoring {len(cells)} cells ({sum(role[c] == 'query' for c in cells)} queries, {sum(role[c] == 'reference' for c in cells)} "
          f"references, {sum(role[c] == 'negative' for c in cells)} negatives) x {len(refs)} references; reference points median "
          f"{ref_pts:.0f}; too small to align: {len(small)}", flush=True)

    # ---- pose-invariant NBLAST, both directions, chunked + resumable
    spath = os.path.join(FIGURES, f"type_match_{a.tag}_scores.pkl")
    store = pickle.load(open(spath, "rb")) if os.path.exists(spath) else {}
    if store.get("refs") != refs:
        store = {"refs": refs, "fwd": {}, "rev": {}}
    T = navis.NeuronList([frame[r] for r in refs])
    for d, r in zip(T, refs):
        d.id = d.name = r
    kw = dict(align_method="rigid", two_way_align=False, n_cores=a.cores, progress=False, align_kwargs={"scale": True})
    todo = [c for c in cells if c not in store["fwd"]]
    print(f"pairwise scores cached for {len(cells) - len(todo)} cells; {len(todo)} to score", flush=True)
    for i in range(0, len(todo), a.chunk):
        part = todo[i:i + a.chunk]
        Q = navis.NeuronList([frame[c] for c in part])
        for d, c in zip(Q, part):
            d.id = d.name = c
        f = navis.nblast_align(Q, T, **kw).values
        r_ = navis.nblast_align(T, Q, **kw).values.T
        for j, c in enumerate(part):
            store["fwd"][c], store["rev"][c] = f[j], r_[j]
        pickle.dump(store, open(spath, "wb"))
        print(f"  scored {min(i + a.chunk, len(todo))}/{len(todo)} at {time.time() - t0:.0f} s", flush=True)

    # ---- per-cell summaries
    rlab = np.array([lab.get(r, "") for r in refs])
    groups = sorted(set(rlab))
    out = []
    for c in cells:
        f, r_ = store["fwd"][c].astype(float).copy(), store["rev"][c].astype(float).copy()
        if c in refs:
            j = refs.index(c)
            f[j] = r_[j] = np.nan
        sym = (f + r_) / 2
        b = int(np.nanargmax(np.where(np.isnan(sym), -9, sym)))
        gs = {g: topk(sym[rlab == g], 3) for g in groups}
        bg = max(gs, key=lambda g: -9 if np.isnan(gs[g]) else gs[g])
        out.append(dict(cell=c, role=role[c], label=lab.get(c, nlabel.get(c, "")), side=side[c], points=len(frame[c].points),
                        top5=topk(sym, 5), sym_max=float(np.nanmax(sym)), fwd_max=float(np.nanmax(f)), rev_max=float(np.nanmax(r_)),
                        best_ref=refs[b], best_ref_label=rlab[b], best_group=bg, best_group_score=gs[bg]))
    out = pd.DataFrame(out)
    neg, pos = out[out.role == "negative"].top5, out[out.role == "reference"].top5
    th = {"match": neg.max(), "likely": neg.quantile(0.95), "ambiguous": neg.quantile(0.80)}
    tiny = out.points < a.min_ratio * ref_pts
    out["call"] = np.select([tiny, out.top5 >= th["match"], out.top5 >= th["likely"], out.top5 >= th["ambiguous"]],
                            ["too small to call", "match", "likely", "ambiguous"], "no match")
    out["pct_of_refs_below"] = [float((pos < v).mean()) for v in out.top5]
    out["pct_of_negs_below"] = [float((neg < v).mean()) for v in out.top5]
    qmap = q.drop_duplicates("cell").set_index("cell")
    out["group"] = out.cell.map(qmap.group)
    out["given"] = out.cell.map(qmap.given)
    if small:
        out = pd.concat([out, pd.DataFrame(dict(cell=small, role=[role[k] for k in small], points=[len(frame[k].points) for k in small],
                                                call=f"too small to align (< {a.min_points} points)",
                                                group=[qmap.group.get(k, np.nan) for k in small]))])
    out = out.sort_values(["role", "top5"], ascending=[True, False])
    base = os.path.join(FIGURES, f"{stamp}_type_match_{a.tag}")
    out.to_csv(base + ".tsv", sep="\t", index=False)

    print(f"\ncontrols (top5 = mean of 5 best scores vs references):")
    for nm, v in (("references, leave-one-out", pos), ("negatives", neg)):
        print(f"  {nm:27s} n {len(v):4d}  median {v.median():.3f}  5% {v.quantile(0.05):.3f}  95% {v.quantile(0.95):.3f}  max {v.max():.3f}")
    print("  negatives by label:", out[out.role == "negative"].groupby("label").top5.agg(["size", "median", "max"]).round(3).to_dict("index"))
    print(f"  thresholds: match >= {th['match']:.3f}, likely >= {th['likely']:.3f}, ambiguous >= {th['ambiguous']:.3f}")
    print(f"  references called: {out[out.role == 'reference'].call.value_counts().to_dict()}")
    rr = out[out.role == "reference"]
    print(f"  references whose best label group is their own label: {(rr.best_group == rr.label).mean():.0%}")
    print(pd.crosstab(rr.label, rr.best_group).to_string())
    qq = out[out.role == "query"]
    print(f"\nqueries (excluding the ones that are references): {len(qq)}")
    summ = qq.groupby("group").apply(lambda g: pd.Series(dict(
        n=len(g), top5_median=round(g.top5.median(), 3), match=int((g.call == "match").sum()), likely=int((g.call == "likely").sum()),
        ambiguous=int((g.call == "ambiguous").sum()), no_match=int((g.call == "no match").sum()),
        too_small=int(g.call.str.startswith("too small").sum()), side=g.side.value_counts().to_dict(),
        best_group=g[g.call.isin(["match", "likely"])].best_group.value_counts().to_dict())))
    print(summ.to_string())
    summ.to_csv(base + "_groups.tsv", sep="\t")

    colors = {"match": "#ffd400", "likely": "#ff9f1a", "ambiguous": "#3b6fb0", "no match": "#e6194b"}
    layers = [{"name": f"SeaTable {a.ref_seatable.replace(chr(92) + 'b', '')} ({len(refs)})", "color": "#2ecc40", "ids": refs}]
    for c, col in colors.items():
        ids = list(qq[qq.call == c].cell)
        if ids:
            layers.append({"name": f"query {c} ({len(ids)})", "color": col, "ids": ids})
    spec = {"meshes": [{"kind": "brain", "alpha": 1.0}], "neurons": layers, "view": "anterior", "layout": "3d",
            "select": layers[1]["name"] if len(layers) > 1 else layers[0]["name"], "image": True}
    json.dump(spec, open(base + "_spec.json", "w"), indent=1)
    open(base + "_ng.txt", "w").write(nglink.to_url(nglink.build(spec)) + "\n")
    print(f"\nwrote {base}.tsv, _groups.tsv, _ng.txt; {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
