"""Find every copy of a cell type by morphology: score a pool of cells against template cells with pose-invariant NBLAST
(navis.nblast_align, rigid + scale), so copies at other positions of the visual field still match.

Pool (union, all statuses - no completeness filtering):
  --seatable REGEX --side R   SeaTable (FlyTable aedes_main) cells whose label matches, resolved fresh via
                              supervoxel_id -> current root (one get_roots call); --exclude REGEX drops labels
  --near-nuclei R_UM          cells whose nucleus (nuclei_v1_aedes dump, positions in nm) lies within R_UM of the
                              templates' nucleus centroid (one get_roots call on their supervoxels)
Cells with fewer than --min-points (10) L2 dotprops points are listed as too small to score, not aligned.
Templates: --templates ROOT ... (their own dotprops come from the cache or are fetched like the pool).
Morphology: L2 dotprops (nm -> um), one CAVE call per cell, >= 5 s apart, thinned to <= POINTS points; cached in
figures/type_copies_dotprops.pkl (resumable; an id whose root changed is refetched).
Scores: fwd = cell as query vs template (fragment-tolerant), rev = template as query vs cell (penalizes fragments),
sym = their mean; per cell the best template (a template is never scored against itself).
Outputs figures/<stamp>_type_copies_<tag>.tsv (ranked) and _ng.txt / _spec.json (Spelunker: templates, strong / possible
copies, fragment-like hits). Example:
  venv/bin/python scripts/morph/type_copies.py --tag cl15 --templates 648518347672514529 648518347579200814 \
      648518347598366865 648518347559939795 --seatable "visual_projection" --side R --near-nuclei 30"""
import argparse
import json
import os
import pickle
import re
import time

import numpy as np

if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd

from faab import nglink
from faab.clients.cave import DATASTACK, client as cave_client, throttle
from faab.config import FIGURES, TMP
from faab.partners.sheet import utc_stamp
from faab.config import fig

SERVER = "https://flytable.mrc-lmb.cam.ac.uk/"
CLS = ("superclass", "class", "subclass", "subsubclass", "type")
POINTS = 1500
CACHE = fig("type_copies_dotprops.pkl")
OLD_CACHE = fig("vpn_pop_dotprops.pkl")  # dotprops fetched for the 2026-10-25 VPN clustering
NUCLEI = os.path.join(TMP, "nuclei_v1_aedes_v542.tsv")
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


def call_rows(out, templates, strong, possible, min_ratio, tmpl_points):
    """copy calls. A cell much smaller than the templates (< min_ratio x their median L2 points) is not called: with
    scaling allowed, the alignment shrinks the template onto its few points and the template-as-query score is spurious
    (seen 2026-10-25: 10-16 point pieces scoring 0.75-0.85 one way, ~0 the other)"""
    tiny = out.points < min_ratio * tmpl_points
    return np.select([out.cell.isin(templates), tiny, out.sym_max >= strong, out.sym_max >= possible,
                      (out.fwd_max >= strong) & (out.rev_max < possible)],
                     ["template", f"too small vs templates (< {min_ratio:g} x {tmpl_points:.0f} points)", "strong copy",
                      "possible copy", "fragment-like"], "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--templates", nargs="+", required=True)
    ap.add_argument("--seatable", help="regex on the SeaTable label for the pool")
    ap.add_argument("--side", help="SeaTable side of the pool (R / L)")
    ap.add_argument("--exclude", help="regex on the SeaTable label to leave out of the pool")
    ap.add_argument("--near-nuclei", type=float, default=0, metavar="UM", help="add cells with a nucleus this close to the templates'")
    ap.add_argument("--cores", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--min-points", type=int, default=10, help="cells with fewer L2 dotprops points are listed, not aligned")
    ap.add_argument("--min-ratio", type=float, default=0.2,
                    help="cells smaller than this fraction of the templates' median points are scored but not called")
    ap.add_argument("--strong", type=float, default=0.45)
    ap.add_argument("--possible", type=float, default=0.35)
    a = ap.parse_args()
    import navis
    import seaserpent as ss
    from fafbseg import flywire
    t0, stamp = time.time(), utc_stamp()
    client = cave_client()
    given = [str(x) for x in a.templates]
    wait()
    ok = np.asarray(client.chunkedgraph.is_latest_roots([int(x) for x in given])).astype(bool)
    templates = []
    for g, good in zip(given, ok):  # an edited template is replaced by its current root (refetched), never kept twice
        if good:
            templates.append(g)
        else:
            wait()
            new = [str(int(x)) for x in np.atleast_1d(client.chunkedgraph.get_latest_roots(int(g)))]
            print(f"template {g} was edited; now {new}", flush=True)
            templates += new

    # ---- pool
    meta = ss.Table("aedes_main", server=SERVER).to_frame()
    meta.columns = [str(c) for c in meta.columns]
    sv = meta["supervoxel_id"].map(s)
    meta = meta[sv.str.isdigit()].copy()
    wait()
    meta["root"] = [str(int(r)) for r in np.asarray(client.chunkedgraph.get_roots([int(x) for x in meta["supervoxel_id"].map(s)]))]
    meta["label"] = [" / ".join(x for x in (s(r.get(c)) for c in CLS) if x) for _, r in meta.iterrows()]
    lab = dict(zip(meta.root, meta.label))
    side = dict(zip(meta.root, meta["side"].map(s)))
    status = dict(zip(meta.root, meta["status"].map(s)))
    pool = {}
    if a.seatable:
        m = meta.label.str.contains(a.seatable, regex=True) & (meta["side"].map(s) == a.side if a.side else True)
        if a.exclude:
            m &= ~meta.label.str.contains(a.exclude, regex=True)
        for r in meta[m].root:
            pool[r] = "SeaTable"
    print(f"SeaTable pool /{a.seatable}/ side {a.side}: {len(pool)} cells", flush=True)
    if a.near_nuclei:
        nuc = pd.read_csv(NUCLEI, sep="\t", dtype={"pt_root_id": str, "pt_supervoxel_id": str})
        nuc = nuc[nuc.valid.astype(str).str.lower().isin(["true", "t", "1"])]
        pos = np.array([[float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", str(p))] for p in nuc.pt_position]) / 1000.0
        tmpl = nuc.pt_root_id.isin(templates + given).to_numpy()
        if not tmpl.any():
            raise SystemExit("no template has a nucleus in the nuclei dump: --near-nuclei needs one")
        cen = pos[tmpl].mean(0)
        near = nuc[np.linalg.norm(pos - cen, axis=1) <= a.near_nuclei]
        wait()
        roots = [str(int(r)) for r in np.asarray(client.chunkedgraph.get_roots([int(x) for x in near.pt_supervoxel_id]))]
        n_new = 0
        for r in roots:
            if r != "0" and r not in pool and "glia" not in lab.get(r, "").lower():
                pool[r] = f"nucleus <= {a.near_nuclei:g} um"
                n_new += 1
        print(f"nuclei within {a.near_nuclei:g} um of the templates' centroid: {len(near)}; added {n_new} cells not yet in the pool", flush=True)
    for t in templates:
        pool.setdefault(t, "template")
    print(f"pool {len(pool)} cells incl. {len(templates)} templates", flush=True)

    # ---- dotprops (cached, resumable)
    cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
    if os.path.exists(OLD_CACHE):
        for k, v in pickle.load(open(OLD_CACHE, "rb")).items():
            cache.setdefault(k, v)
    todo = [r for r in pool if r not in cache]
    print(f"dotprops cached {len(pool) - len(todo)}; to fetch {len(todo)} (~{len(todo) * 5.5 / 60:.0f} min)", flush=True)
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
    have = [r for r in pool if r in cache and len(cache[r].points) >= a.min_points]
    small = [r for r in pool if r in cache and len(cache[r].points) < a.min_points]  # listed, not scored
    print(f"dotprops for {len(have) + len(small)} of {len(pool)} cells; failed {len(failed)}: {failed[:5]}; "
          f"too small to align (< {a.min_points} points): {len(small)}", flush=True)

    # ---- pose-invariant NBLAST, both directions
    Q = navis.NeuronList([cache[r] for r in have])
    for d, r in zip(Q, have):
        d.id = d.name = r
    Tm = navis.NeuronList([cache[r] for r in templates if r in cache])
    tids = [r for r in templates if r in cache]
    print(f"nblast_align {len(have)} x {len(tids)} templates, both directions, {a.cores} cores ...", flush=True)
    kw = dict(align_method="rigid", two_way_align=False, n_cores=a.cores, progress=False, align_kwargs={"scale": True})
    fwd = navis.nblast_align(Q, Tm, **kw).values  # cell as query
    print(f"  forward done at {time.time() - t0:.0f} s", flush=True)
    rev = navis.nblast_align(Tm, Q, **kw).values.T  # template as query
    sym = (fwd + rev) / 2
    for j, t in enumerate(tids):  # never score a template against itself
        if t in have:
            i = have.index(t)
            fwd[i, j] = rev[i, j] = sym[i, j] = np.nan
    best = np.nanargmax(np.where(np.isnan(sym), -9, sym), axis=1)
    out = pd.DataFrame(dict(cell=have, source=[pool[r] for r in have], label=[lab.get(r, "") for r in have],
                            side=[side.get(r, "") for r in have], status=[status.get(r, "") for r in have],
                            points=[len(cache[r].points) for r in have], sym_max=np.nanmax(sym, axis=1),
                            sym_mean=np.nanmean(sym, axis=1), fwd_max=np.nanmax(fwd, axis=1), rev_max=np.nanmax(rev, axis=1),
                            best_template=[tids[b] for b in best]))
    out["call"] = call_rows(out, templates, a.strong, a.possible, a.min_ratio, np.median([len(cache[t].points) for t in tids]))
    if small:
        out = pd.concat([out, pd.DataFrame(dict(cell=small, source=[pool[r] for r in small], label=[lab.get(r, "") for r in small],
                                                side=[side.get(r, "") for r in small], status=[status.get(r, "") for r in small],
                                                points=[len(cache[r].points) for r in small],
                                                call=f"too small to score (< {a.min_points} L2 points)"))])
    out = out.sort_values("sym_max", ascending=False, na_position="last")
    base = fig(f"{stamp}_type_copies_{a.tag}")
    out.to_csv(base + ".tsv", sep="\t", index=False)
    q = out[out.call != "template"].sym_max.dropna()
    print(f"\nsym_max over the pool: median {q.median():.2f}, 90% {q.quantile(0.9):.2f}, 99% {q.quantile(0.99):.2f}; "
          f"calls {out.call.replace('', '-').value_counts().to_dict()}")
    pd.set_option("display.width", 230)
    pd.set_option("display.max_colwidth", 42)
    print(out.head(40).to_string(index=False))

    colors = {"template": "#3b6fb0", "strong copy": "#e6194b", "possible copy": "#ffd400", "fragment-like": "#2ecc40"}
    layers = [{"name": f"{c} ({int((out.call == c).sum())})", "color": col, "ids": list(out[out.call == c].cell)}
              for c, col in colors.items() if (out.call == c).any()]
    spec = {"meshes": [{"kind": "brain", "alpha": 1.0}], "neurons": layers, "view": "anterior", "layout": "3d",
            "select": layers[0]["name"], "image": True}
    json.dump(spec, open(base + "_spec.json", "w"), indent=1)
    open(base + "_ng.txt", "w").write(nglink.to_url(nglink.build(spec)) + "\n")
    print(f"\nwrote {base}.tsv and _ng.txt; {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
