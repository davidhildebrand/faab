"""Network-free: nearest-neighbour test 'does the cell look more like a SeaTable T4/T5 or like a columnar negative
(L1-L5 / LMC / Mi1)?' with centroid-aligned NBLAST (translation only, left frame, as in t4t5_discrim.py).
Panel = references + negatives. Per cell: top5 mean sym score vs references and vs negatives, margin = difference,
and the vote among its 7 nearest panel cells (a panel cell never votes for itself). Validated leave-one-out on the
panel, separately for right-side references (the queries are all right-side). Writes tmp/t4t5_knn.tsv."""
import pickle

import navis
import numpy as np
import pandas as pd

from faab.config import fig

TMP = "/Users/davidh/Data/Vibe/FAAB/tmp/"
K = 7


def main():
    cache = pickle.load(open(fig("type_copies_dotprops.pkl"), "rb"))
    st = pd.read_csv(TMP + "t4t5_seatable_20261026.tsv", sep="\t", dtype=str).fillna("").drop_duplicates("root_now").set_index("root_now")
    sv = pd.read_csv(TMP + "t4t5_query_survey.tsv", sep="\t", dtype=str)
    refs = sorted(st.index[st.label.str.contains(r"\bT4|\bT5", regex=True)])
    negs = sorted(r for r in st.index[st.label.str.contains("L1-L5|LMC", regex=True)] if r not in refs)
    mi1 = {f"Mi1:{n.id}": n for n in pickle.load(open(fig("mi1_dotprops.pkl"), "rb")) if len(n.points) >= 10}
    queries = [r for r in sv.now if r not in refs]
    group = dict(zip(sv.now, sv.line))
    neuron = {**{r: cache[r] for r in refs + negs + queries}, **mi1}
    role = {**{r: "reference" for r in refs}, **{r: "negative" for r in negs}, **{k: "negative" for k in mi1},
            **{r: "query" for r in queries}}
    label = {k: (st.label.get(k, "") if k in st.index else "Mi1" if k.startswith("Mi1:") else "") for k in neuron}
    MID = 371.7
    side = {k: (st.side.get(k) if k in st.index and st.side.get(k) in ("L", "R")
                else ("L" if float(v.points[:, 0].mean()) > MID else "R")) for k, v in neuron.items()}

    def centered(k):
        p = np.asarray(neuron[k].points, float).copy()
        v = np.asarray(neuron[k].vect, float).copy()
        if side[k] == "R":
            p[:, 0] = 2 * MID - p[:, 0]
            v[:, 0] *= -1
        d = navis.Dotprops(p - p.mean(0), k=None, vect=v)
        d.units = "um"
        d.id = d.name = k
        return d

    cells = list(neuron)
    panel = refs + negs + list(mi1)
    C = navis.NeuronList([centered(k) for k in cells])
    Pn = navis.NeuronList([centered(k) for k in panel])
    fwd = navis.nblast(C, Pn, n_cores=1, progress=False).values
    rev = navis.nblast(Pn, C, n_cores=1, progress=False).values.T
    sym = (fwd + rev) / 2
    pidx = {k: j for j, k in enumerate(panel)}
    isref = np.array([role[k] == "reference" for k in panel])
    rows = []
    for i, k in enumerate(cells):
        s = sym[i].copy()
        if k in pidx:
            s[pidx[k]] = np.nan
        ok = ~np.isnan(s)
        r5 = np.sort(s[ok & isref])[::-1][:5].mean()
        n5 = np.sort(s[ok & ~isref])[::-1][:5].mean()
        order = np.argsort(np.where(ok, -s, np.inf))[:K]
        vote = int(isref[order].sum())
        rows.append(dict(cell=k, role=role[k], label=label[k], side=side[k], group=group.get(k, ""), ref_top5=r5, neg_top5=n5,
                         margin=r5 - n5, ref_votes=vote, nearest=label[panel[order[0]]] if label[panel[order[0]]] else panel[order[0]]))
    d = pd.DataFrame(rows)
    d["call"] = np.where(d.ref_votes >= 4, "T4/T5-like", "negative-like")
    d.to_csv(TMP + "t4t5_knn.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220)
    ref, neg, q = d[d.role == "reference"], d[d.role == "negative"], d[d.role == "query"]
    print(f"leave-one-out on the panel ({K}-NN majority):")
    print(f"  references called T4/T5-like: {(ref.call == 'T4/T5-like').mean():.0%} of {len(ref)}; right-side references: "
          f"{(ref[ref.side == 'R'].call == 'T4/T5-like').sum()} of {(ref.side == 'R').sum()}")
    print(f"  negatives called T4/T5-like: {(neg.call == 'T4/T5-like').mean():.0%} of {len(neg)}; by label "
          f"{neg.groupby('label').apply(lambda g: f'{(g.call == chr(84) + chr(52) + chr(47) + chr(84) + chr(53) + chr(45) + chr(108) + chr(105) + chr(107) + chr(101)).sum()}/{len(g)}').to_dict()}")
    print(f"  right-side negatives called T4/T5-like: {(neg[neg.side == 'R'].call == 'T4/T5-like').sum()} of {(neg.side == 'R').sum()}")
    print(f"  margin (ref_top5 - neg_top5) medians: references {ref.margin.median():.3f} (right {ref[ref.side == 'R'].margin.median():.3f}), "
          f"negatives {neg.margin.median():.3f}")
    print("\nright-side references:")
    print(ref[ref.side == "R"][["cell", "label", "ref_top5", "neg_top5", "margin", "ref_votes", "nearest"]].round(3).to_string(index=False))
    print("\nqueries per group:")
    print(q.groupby("group").apply(lambda g: pd.Series(dict(n=len(g), t4t5_like=int((g.call == "T4/T5-like").sum()),
                                                           margin_median=round(g.margin.median(), 3),
                                                           votes_median=g.ref_votes.median(),
                                                           nearest_top=g.nearest.value_counts().head(3).to_dict()))).to_string())
    print(f"\nall queries: T4/T5-like {(q.call == 'T4/T5-like').sum()} of {len(q)}; vote distribution {q.ref_votes.value_counts().sort_index().to_dict()}")
    print("\nnegative-like queries:")
    print(q[q.call != "T4/T5-like"][["cell", "group", "ref_top5", "neg_top5", "margin", "ref_votes", "nearest"]].round(3).to_string(index=False))

    # ---- T4 vs T5: top-3 mean vs T4 references and vs T5 references (leave-one-out for references)
    is_t5 = np.array([("T5" in label[k]) for k in panel])
    t4m, t5m = isref & ~is_t5, isref & is_t5
    pref = []
    for i, k in enumerate(cells):
        s = sym[i].copy()
        if k in pidx:
            s[pidx[k]] = np.nan
        a4 = np.sort(s[t4m & ~np.isnan(s)])[::-1][:3].mean()
        a5 = np.sort(s[t5m & ~np.isnan(s)])[::-1][:3].mean()
        pref.append(a5 - a4)
    d["t5_minus_t4"] = pref
    d.to_csv(TMP + "t4t5_knn.tsv", sep="\t", index=False)
    ref, q = d[d.role == "reference"], d[d.role == "query"]
    r5, r4 = ref[ref.label.str.contains("T5")], ref[~ref.label.str.contains("T5")]
    print(f"\nT5 minus T4 (top-3 means), leave-one-out: T5 references median {r5.t5_minus_t4.median():.3f} "
          f"({(r5.t5_minus_t4 > 0).sum()}/{len(r5)} > 0); T4 references median {r4.t5_minus_t4.median():.3f} "
          f"({(r4.t5_minus_t4 > 0).sum()}/{len(r4)} > 0)")
    print("queries T5-leaning (> 0) per group:", q.groupby("group").apply(
        lambda g: f"{(g.t5_minus_t4 > 0).sum()}/{len(g)} (median {g.t5_minus_t4.median():.3f})").to_dict())

    # ---- do the line groups differ in shape? query x query centered NBLAST, within vs between groups
    qs = list(q.cell)
    Q = navis.NeuronList([centered(k) for k in qs])
    m = navis.nblast_allbyall(Q, n_cores=1, progress=False).values
    m = (m + m.T) / 2
    np.fill_diagonal(m, np.nan)
    g = np.array([group[k] for k in qs])
    print("\nmean query-query similarity, rows = group, columns = group:")
    tab = pd.DataFrame({b: [np.nanmean(m[np.ix_(g == a, g == b)]) for a in sorted(set(g))] for b in sorted(set(g))},
                       index=sorted(set(g))).round(3)
    print(tab.to_string())

    # ---- Spelunker link
    import json
    import os

    from faab import nglink
    from faab.partners.sheet import utc_stamp
    layers = [{"name": f"SeaTable T4/T5 ({len(refs)})", "color": "#2ecc40", "ids": refs},
              {"name": f"query T4/T5-like ({int((q.call == 'T4/T5-like').sum())})", "color": "#ffd400",
               "ids": list(q[q.call == "T4/T5-like"].cell)},
              {"name": f"query not T4/T5-like ({int((q.call != 'T4/T5-like').sum())})", "color": "#e6194b",
               "ids": list(q[q.call != "T4/T5-like"].cell)}]
    spec = {"meshes": [{"kind": "brain", "alpha": 1.0}], "neurons": layers, "view": "anterior", "layout": "3d",
            "select": layers[2]["name"], "image": True}
    base = fig(f"{utc_stamp()}_t4t5_query_match")
    json.dump(spec, open(base + "_spec.json", "w"), indent=1)
    open(base + "_ng.txt", "w").write(nglink.to_url(nglink.build(spec)) + "\n")
    d.to_csv(base + ".tsv", sep="\t", index=False)
    print(f"\nwrote {base}.tsv and _ng.txt")


if __name__ == "__main__":
    main()
