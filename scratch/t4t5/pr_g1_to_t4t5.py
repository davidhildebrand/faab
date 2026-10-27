"""Read-only forward path search: which of David's 411 T4/T5 candidates (tmp/t4t5_query_survey.tsv, current roots)
receive input arising from PR group 1 (data/pr_groups.tsv group 1: six R1-R6, R7, R8)? (2026-10-27)
One UTC timestamp; live synapses via class_paths.query (chunked, cached in tmp/paths_cache, >= 5 s between calls):
  F1  outputs of the group-1 PRs                 -> X = cells with >= 3 synapses from one PR (sources excluded)
  F2  outputs of X                                -> X -> Y edges
  B1  inputs of the 411 candidates                -> Y -> T edges
Paths: 1 = PR -> T; 2 = PR -> X -> T; 3 = PR -> X -> Y -> T. SeaTable glia never serve as intermediates. Reported at
>= 3 and >= 5 synapses per edge, ranked by the weakest edge. Writes figures/<stamp>_pr_g1_to_t4t5.tsv (per candidate)
and _paths.tsv, and prints a short Spelunker link (state server) with stage-shaded layers."""
import datetime
import importlib.util
import json
import os

import numpy as np
import pandas as pd

from faab import nglink
from faab.config import DATA, FIGURES, TMP
from faab.partners.sheet import utc_stamp

spec = importlib.util.spec_from_file_location("cp", "/Users/davidh/Data/Vibe/FAAB/faab/scripts/paths/class_paths.py")
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)


def main():
    stamp = utc_stamp()
    client = cp.cave_client()
    prs = pd.read_csv(os.path.join(DATA, "pr_groups.tsv"), sep="\t", dtype=str)
    g1 = prs[prs.group == "1"]
    src_map = cp.refresh_roots(client, g1.root_id_curr.astype(np.int64), "PR group 1")
    src = sorted({r for v in src_map.values() for r in v})
    pr_class = {}
    for rid, cls in zip(g1.root_id_curr.astype(np.int64), g1["class"]):
        for r in src_map[rid]:
            pr_class[r] = cls
    sv = pd.read_csv(os.path.join(TMP, "t4t5_query_survey.tsv"), sep="\t", dtype=str)
    knn = pd.read_csv(os.path.join(TMP, "t4t5_knn.tsv"), sep="\t", dtype={"cell": str})
    call = dict(zip(knn.cell, knn.call))
    call.update({c: "SeaTable T4" for c in knn[knn.role == "reference"].cell})
    t5 = dict(zip(knn.cell, knn.t5_minus_t4))
    grp = dict(zip(sv.now, sv.line))
    tmap = cp.refresh_roots(client, sv.now.astype(np.int64), "T4/T5 candidates")
    tgt_of = {r: str(k) for k, v in tmap.items() for r in v}
    tgt = sorted(int(v) for v in tgt_of)
    info = cp.flytable(client)
    glia = {r for r, v in info.items() if v[2]}
    lab = {r: v[0] for r, v in info.items()}
    now = datetime.datetime.now(datetime.timezone.utc)
    print(f"sources {len(src)} PRs ({pd.Series(pr_class).value_counts().to_dict()}); targets {len(tgt)}; timestamp {now:%Y-%m-%d %H:%M} UTC", flush=True)

    f1 = cp.query(client, now, "g1pr_F1", pre_pt_root_id=src)
    f1 = f1[~f1.post.isin(src)]
    X = sorted(int(v) for v in set(f1[f1.n >= 3].post) - glia)  # plain ints: the cache sidecar is JSON
    print(f"X (>= 3 synapses from a group-1 PR, not glia): {len(X)}; labeled: "
          f"{pd.Series([lab[x] for x in X if x in lab]).value_counts().head(8).to_dict()}", flush=True)
    side = os.path.join(cp.CACHE, "g1pr_F2_ids.json")  # 2026-10-27 run 1 wrote the edge table, then died writing this
    if os.path.exists(side):
        try:
            json.load(open(side))
        except ValueError:
            json.dump({"pre": X, "post": []}, open(side, "w"))
            print("  repaired the truncated g1pr_F2 cache sidecar", flush=True)
    f2 = cp.query(client, now, "g1pr_F2", pre_pt_root_id=X)
    b1 = cp.query(client, now, "t4t5cand_B1", post_pt_root_id=tgt)
    tset = set(tgt)
    rows = []
    for thr in (3, 5):
        e1 = f1[f1.n >= thr]
        e2 = f2[(f2.n >= thr) & ~f2.post.isin(src)]
        e3 = b1[b1.n >= thr]
        xs = set(e1.post) - glia
        # 1 connection
        # x / y kept as strings ("" = none): a numeric column with gaps becomes float and corrupts 18-digit ids
        for r in e1[e1.post.isin(tset)].itertuples():
            rows.append(dict(thr=thr, k=1, pr=str(r.pre), x="", y="", t=str(r.post), weakest=r.n, total=r.n))
        # 2 connections
        xt = e3[e3.pre.isin(xs)]
        a = e1[e1.post.isin(set(xt.pre))]
        m = a.merge(xt, left_on="post", right_on="pre", suffixes=("_1", "_2"))
        for r in m.itertuples():
            rows.append(dict(thr=thr, k=2, pr=str(r.pre_1), x=str(r.post_1), y="", t=str(r.post_2), weakest=min(r.n_1, r.n_2),
                             total=r.n_1 + r.n_2))
        # 3 connections
        ys = set(e3.pre) - glia - set(src)
        xy = e2[e2.pre.isin(xs) & e2.post.isin(ys)]
        m = e1[e1.post.isin(set(xy.pre))].merge(xy, left_on="post", right_on="pre", suffixes=("_1", "_2"))
        m = m.merge(e3, left_on="post_2", right_on="pre")
        for r in m.itertuples():
            rows.append(dict(thr=thr, k=3, pr=str(r.pre_1), x=str(r.post_1), y=str(r.post_2), t=str(r.post),
                             weakest=min(r.n_1, r.n_2, r.n), total=r.n_1 + r.n_2 + r.n))
    P = pd.DataFrame(rows)
    base = os.path.join(FIGURES, f"{stamp}_pr_g1_to_t4t5")
    if P.empty:
        print("no paths at >= 3 synapses per edge")
        return
    P["pr_class"] = P.pr.map(lambda v: pr_class.get(int(v), ""))
    P["x_label"] = P.x.map(lambda v: lab.get(int(v), "") if v else "")
    P["y_label"] = P.y.map(lambda v: lab.get(int(v), "") if v else "")
    P = P.sort_values(["thr", "k", "weakest", "total"], ascending=[True, True, False, False])
    P.to_csv(base + "_paths.tsv", sep="\t", index=False)
    per = []
    for t, g in P.groupby("t"):
        given = tgt_of[int(t)]
        row = dict(cell=t, given=given, line=grp.get(given, ""), morph_call=call.get(given, call.get(t, "")),
                   t5_minus_t4=t5.get(given, np.nan))
        for thr in (3, 5):
            h = g[g.thr == thr]
            if len(h):
                k = int(h.k.min())
                best = h[h.k == k].iloc[0]
                row.update({f"min_connections_thr{thr}": k, f"best_weakest_thr{thr}": int(best.weakest),
                            f"n_paths_thr{thr}": int((h.k == k).sum()),
                            f"via_thr{thr}": " -> ".join(v for v in (best.x, best.y) if v)})
        b = g[g.thr == 3]
        row["pr_classes"] = ",".join(sorted(set(v for v in b.pr_class if v)))
        row["n_X"] = int(b.x[b.x != ""].nunique())
        row["n_Y"] = int(b.y[b.y != ""].nunique())
        per.append(row)
    S = pd.DataFrame(per).sort_values(["min_connections_thr3", "best_weakest_thr3"], ascending=[True, False])
    S.to_csv(base + ".tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"\ncandidates reached: {len(S)} of {len(tgt)}")
    for thr in (3, 5):
        c = f"min_connections_thr{thr}"
        if c in S:
            print(f"  >= {thr} synapses per edge: {S[c].value_counts().sort_index().to_dict()} (connections -> candidates)")
    print(S.to_string(index=False))
    print("\nX cells on the paths (thr 3):", P[P.thr == 3].groupby(["x", "x_label"]).size().sort_values(ascending=False).head(15).to_dict())

    # Spelunker: strongest shortest path per reached candidate at thr 3
    best = P[P.thr == 3].sort_values(["k", "weakest"], ascending=[True, False]).groupby("t").head(1)
    k = int(best.k.max())
    shades = cp.STAGE_SHADES[k]
    stages = [("PR group 1", sorted(set(best.pr)))]
    if k >= 2:
        stages.append(("X: PR targets", sorted(set(v for v in best.x if v))))
    if k >= 3:
        stages.append(("Y: inputs of the candidates", sorted(set(v for v in best.y if v))))
    stages.append(("T4/T5 candidates reached", sorted(set(best.t))))
    layers = [{"name": f"{n} ({len(ids)})", "color": shades[i], "ids": [str(v) for v in ids]} for i, (n, ids) in enumerate(stages)]
    spec = {"meshes": [{"kind": "brain", "alpha": 1.0}], "neurons": layers, "view": "anterior", "layout": "3d",
            "select": layers[-1]["name"], "image": True}
    state = nglink.build(spec)
    open(base + "_ng.txt", "w").write(nglink.to_url(state) + "\n")
    cp.wait()
    sid = client.state.upload_state_json(state)
    print(f"\nshort link: {client.state.build_neuroglancer_url(sid, ngl_url='https://spelunker.cave-explorer.org')}")
    print(f"wrote {base}.tsv and _paths.tsv")


if __name__ == "__main__":
    main()
