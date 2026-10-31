"""Search a cached FAAB pool for homologs of Drosophila cell types, using fly cells (neuPrint male CNS) as templates in
pose-invariant NBLAST (navis.nblast_align, rigid + scale).

Templates: --n-templates cells per --types type (right side, chosen spread along the eye: quantiles of their soma
position on the long axis), each used as-is and mirrored (x -> -x), because the two datasets' handedness is unknown and a
rigid alignment cannot reflect. Fly skeletons (8 nm voxels -> um) become dotprops at ~--spacing um, capped at 1500 points.
Fly controls, scored exactly like pool cells: --n-controls more cells of each template type (same type: should score
high) and of each --controls type (similar but different types).
Pool: the cells of a type_copies TSV (FAAB, L2 dotprops cached in figures/type_copies_dotprops.pkl) with >= --min-points.
Stage A: every pool cell and control as query vs all templates. Stage B: the top --top-n pool cells by stage A and all
controls also with the templates as query; sym = mean of both. Per cell: the best type, its handedness and template.
Outputs figures/<stamp>_fly_templates_<tag>.tsv (+ _controls.tsv, _ng.txt). Network: neuPrint only (no CAVE).
Example: venv/bin/python scripts/morph/fly_template_search.py --tag lplc2_lc4 --types LPLC2 LC4 --controls LPLC1 LC6 \
             --pool figures/20261025d235027tUTC_type_copies_cl15_recalled.tsv"""
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
from faab.config import FIGURES, REPO
from faab.partners.sheet import utc_stamp
from faab.config import fig

CACHE = fig("type_copies_dotprops.pkl")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--types", nargs="+", required=True)
    ap.add_argument("--controls", nargs="*", default=[])
    ap.add_argument("--n-templates", type=int, default=3)
    ap.add_argument("--n-controls", type=int, default=3)
    ap.add_argument("--pool", required=True, help="type_copies TSV whose cells form the pool")
    ap.add_argument("--min-points", type=int, default=100)
    ap.add_argument("--spacing", type=float, default=3.0, help="um between fly dotprops points (~FAAB L2 spacing)")
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--cores", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    a = ap.parse_args()
    import navis
    from navis.interfaces import neuprint as npn
    from faab.clients.neuprint import client as npclient, MALE_CNS
    t0, stamp = time.time(), utc_stamp()

    # ---- fly cells: templates spread along the eye, controls
    c = npclient(MALE_CNS)
    rng = np.random.default_rng(0)
    pick = {}
    for ty in a.types + a.controls:
        meta, _ = npn.fetch_neurons(npn.NeuronCriteria(type=ty), client=c)
        meta = meta[meta.somaSide.astype(str) == "R"] if "somaSide" in meta.columns else meta
        meta = meta[meta.somaLocation.notna()] if "somaLocation" in meta.columns else meta
        print(f"fly {ty}: {len(meta)} right-side cells with a soma", flush=True)
        if len(meta) == 0:
            continue
        loc = np.array([list(v) for v in meta.somaLocation])
        ax = int(np.argmax(loc.max(0) - loc.min(0)))
        order = meta.bodyId.to_numpy()[np.argsort(loc[:, ax])]
        if ty in a.types:
            q = np.linspace(0.1, 0.9, a.n_templates)
            tmpl = [int(order[int(round(x * (len(order) - 1)))]) for x in q]
            rest = [int(b) for b in order if b not in tmpl]
            ctrl = [int(b) for b in rng.choice(rest, size=min(a.n_controls, len(rest)), replace=False)]
            pick[ty] = dict(templates=tmpl, controls=ctrl)
        else:
            pick[ty] = dict(templates=[], controls=[int(b) for b in rng.choice(order, size=min(a.n_controls, len(order)), replace=False)])
    ids = sorted({b for v in pick.values() for b in v["templates"] + v["controls"]})
    sk = npn.fetch_skeletons(ids, client=c, heal=True)
    fly = {}
    for n in sk:
        n.units = "8 nm"
        u = n.convert_units("um")
        fly[int(n.id)] = u
    print(f"fetched {len(fly)} fly skeletons; cable (um) median {np.median([n.cable_length for n in fly.values()]):.0f}", flush=True)

    def fly_dp(bid, name, mirror=False):
        n = fly[bid].copy()
        if mirror:
            n.nodes["x"] = -n.nodes["x"]
        dp = navis.make_dotprops(n, k=5, resample=max(a.spacing, n.cable_length / 1500))
        dp.units, dp.id, dp.name = "um", name, name
        return dp

    T, tinfo = [], []
    for ty, v in pick.items():
        for b in v["templates"]:
            for mir in (False, True):
                nm = f"{ty}_{b}{'_mirror' if mir else ''}"
                T.append(fly_dp(b, nm, mir))
                tinfo.append(dict(name=nm, type=ty, mirror=mir))
    ctrl, cinfo = [], []
    for ty, v in pick.items():
        for b in v["controls"]:
            nm = f"ctrl_{ty}_{b}"
            ctrl.append(fly_dp(b, nm))
            cinfo.append(dict(cell=nm, type=ty, same_as_template=ty in a.types))
    print(f"{len(T)} templates (points median {np.median([len(d.points) for d in T]):.0f}); {len(ctrl)} fly controls", flush=True)

    # ---- pool
    P = pd.read_csv(a.pool if os.path.isabs(a.pool) else os.path.join(REPO, a.pool), sep="\t", dtype={"cell": str})
    cache = pickle.load(open(CACHE, "rb"))
    P = P[P.cell.isin(cache.keys()) & (P.points >= a.min_points)]
    Q = [cache[r] for r in P.cell]
    for d, r in zip(Q, P.cell):
        d.id = d.name = r
    print(f"pool {len(Q)} FAAB cells with >= {a.min_points} points", flush=True)
    kw = dict(align_method="rigid", two_way_align=False, n_cores=a.cores, progress=False, align_kwargs={"scale": True})
    TL = navis.NeuronList(T)
    names = [t["name"] for t in tinfo]

    # ---- stage A: cell / control as query
    A = navis.nblast_align(navis.NeuronList(Q + ctrl), TL, **kw).values
    print(f"stage A done at {time.time() - t0:.0f} s", flush=True)
    nQ = len(Q)
    fwd_best = np.nanmax(A[:nQ], axis=1)
    top = np.argsort(-fwd_best)[:a.top_n]
    # ---- stage B: templates as query for the top pool cells and all controls
    tgt = [Q[i] for i in top] + ctrl
    B = navis.nblast_align(TL, navis.NeuronList(tgt), **kw).values.T
    print(f"stage B done at {time.time() - t0:.0f} s", flush=True)

    def summarize(fwd, rev):
        sym = (fwd + rev) / 2 if rev is not None else fwd
        rows = []
        for i in range(len(fwd)):
            j = int(np.nanargmax(sym[i]))
            by_type = {}
            for ty in a.types:
                cols = [k for k, t in enumerate(tinfo) if t["type"] == ty]
                by_type[ty] = float(np.nanmax(sym[i, cols]))
            rows.append(dict(best=float(sym[i, j]), best_template=names[j], best_type=tinfo[j]["type"],
                             mirror=tinfo[j]["mirror"], fwd=float(fwd[i, j]), rev=float(rev[i, j]) if rev is not None else np.nan,
                             **{f"sym_{ty}": by_type[ty] for ty in a.types}))
        return pd.DataFrame(rows)

    nT = len(top)
    res = P.iloc[top][["cell", "label", "status", "points"]].reset_index(drop=True)
    res = pd.concat([res, summarize(A[:nQ][top], B[:nT])], axis=1).sort_values("best", ascending=False)
    allA = P[["cell", "label", "status", "points"]].reset_index(drop=True)
    allA["fwd_best"] = fwd_best
    con = pd.concat([pd.DataFrame(cinfo), summarize(A[nQ:], B[nT:])], axis=1).sort_values("best", ascending=False)
    base = fig(f"{stamp}_fly_templates_{a.tag}")
    res.to_csv(base + ".tsv", sep="\t", index=False)
    allA.to_csv(base + "_stageA.tsv", sep="\t", index=False)
    con.to_csv(base + "_controls.tsv", sep="\t", index=False)
    pd.set_option("display.width", 230)
    print("\nfly controls (same type should beat other types):")
    print(con.to_string(index=False))
    print(f"\nstage A over the pool (cell as query, best of {len(T)} templates): median {np.median(fwd_best):.2f}, "
          f"90% {np.quantile(fwd_best, 0.9):.2f}, 99% {np.quantile(fwd_best, 0.99):.2f}, max {fwd_best.max():.2f}")
    print(f"handedness of the best template among the top {nT}: {res.mirror.value_counts().to_dict()}")
    print("\ntop 30 FAAB cells (both directions):")
    print(res.head(30).to_string(index=False))
    colors = ["#e6194b", "#3b6fb0", "#2ecc40"]
    layers = []
    for k, ty in enumerate(a.types):
        sub = res[(res.best_type == ty)].head(15)
        layers.append({"name": f"best match {ty} (top {len(sub)})", "color": colors[k % 3], "ids": list(sub.cell)})
    spec = {"meshes": [{"kind": "brain", "alpha": 1.0}], "neurons": layers, "view": "anterior", "layout": "3d",
            "select": layers[0]["name"], "image": True}
    json.dump(spec, open(base + "_spec.json", "w"), indent=1)
    open(base + "_ng.txt", "w").write(nglink.to_url(nglink.build(spec)) + "\n")
    pickle.dump(dict(A=A, B=B, names=names, tinfo=tinfo, cinfo=cinfo, pool=list(P.cell), top=top), open(base + ".pkl", "wb"))
    print(f"\nwrote {base}.tsv / _stageA.tsv / _controls.tsv / _ng.txt; {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
