"""L1-L5 typing of the SeaTable LMCs (David, 2026-10-31): (1) are the typed groups consistent? (2) putative types for
LRS's 'L1-L5' cells. Leave-one-out on the typed cells with
  features  lamina / medulla position features (lmc_features.py: depth from the neuropil faces, extension past the
            faces, lamina spread, synapse counts and depths), three models: LDA (shrinkage), kNN (k=5), random forest
  nblast    7-nearest-neighbour vote on the NBLAST matrices of lmc_nblast.py (align = nblast_align rigid, centroid =
            translation only), when present
The call is the consensus of the three feature models; NBLAST votes are reported beside it (NB_agree). A typed cell is
'consistent' when that consensus is its own label. Network-free. Writes
figures/lmc/lmc_typing.tsv and the report tmp/lmc_typing_report.txt. Usage: lmc_classify.py"""
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from faab.config import TMP, fig

warnings.filterwarnings("ignore")
TYPES = ["L1", "L2", "L3", "L4", "L5"]
REPORT = os.path.join(TMP, "lmc_typing_report.txt")
out_lines = []


def say(s=""):
    print(s, flush=True)
    out_lines.append(s)


def feature_table(F):
    X = pd.DataFrame(index=F.root)
    for c in ("n_in_LA", "n_out_LA", "n_in_ME", "n_out_ME", "cable_LA", "cable_ME"):
        X["log_" + c] = np.log1p(F[c].to_numpy(float))
    X["out_LA_frac"] = (F.n_out_LA / (F.n_out_LA + F.n_out_ME).clip(lower=1)).to_numpy()
    X["in_LA_frac"] = (F.n_in_LA / (F.n_in_LA + F.n_in_ME).clip(lower=1)).to_numpy()
    X["in_per_out"] = np.log1p((F.n_in / (F.n_out + 1)).to_numpy(float))
    keep = ["ME_depth_max", "ME_max_um_from_outer", "ME_out_depth_med", "ME_out_depth_p10", "ME_out_depth_p90",
            "ME_in_depth_med", "LA_in_depth_med", "LA_out_depth_med", "LA_spread_p90_um", "LA_spread_max_um",
            "LA_beyond_outer_um", "LA_depth_min", "LA_depth_max"] + \
           [f"ME_out_bin{k}" for k in range(5)] + [f"ME_cable_bin{k}" for k in range(5)] + [f"LA_cable_bin{k}" for k in range(5)]
    for c in keep:
        X[c] = F[c].to_numpy(float)
    return X.fillna(X.median())


def knn_vote(S, ids, lab, k=7):
    """leave-one-out k-NN vote on a similarity matrix: for each cell, the most frequent label among its k most similar
    typed cells (itself excluded), and the share of that label"""
    idx = {r: i for i, r in enumerate(ids)}
    typed = [r for r in ids if lab[r] in TYPES]
    pred, share = {}, {}
    for r in ids:
        cand = [t for t in typed if t != r]
        sims = np.array([S[idx[r], idx[t]] for t in cand])
        top = [cand[j] for j in np.argsort(-sims)[:k]]
        v = pd.Series([lab[t] for t in top]).value_counts()
        pred[r], share[r] = v.index[0], v.iloc[0] / k
    return pred, share


def main():
    F = pd.read_csv(fig("lmc_features.tsv"), sep="\t", dtype={"root": str, "serial_id": str}).fillna({"label": ""})
    cells = pd.read_csv(fig("lmc_cells.tsv"), sep="\t", dtype=str).fillna("")
    F = F.merge(cells[["root", "status", "proofreader", "point_xyz", "notes"]], on="root", how="left")
    X = feature_table(F)
    lab = dict(zip(F.root, F.label))
    typed = F[F.label.isin(TYPES)].root.tolist()
    rest = F[~F.label.isin(TYPES)].root.tolist()
    y = np.array([lab[r] for r in typed])
    say(f"{len(typed)} typed cells {pd.Series(y).value_counts().sort_index().to_dict()}; {len(rest)} to call; "
        f"{X.shape[1]} features")
    models = {"LDA": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
              "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance")),
              "RF": RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=0)}
    P = pd.DataFrame(index=F.root)
    for name, m in models.items():
        loo = cross_val_predict(m, X.loc[typed].to_numpy(), y, cv=LeaveOneOut())
        P.loc[typed, name] = loo
        m.fit(X.loc[typed].to_numpy(), y)
        if rest:
            P.loc[rest, name] = m.predict(X.loc[rest].to_numpy())
        acc = np.mean(loo == y)
        say(f"\n{name} leave-one-out accuracy {acc:.0%}")
        say(pd.crosstab(pd.Series(y, name="label"), pd.Series(loo, name="predicted")).to_string())
    rf = models["RF"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    say("\nrandom-forest feature importance (top 12): " + ", ".join(f"{k} {v:.2f}" for k, v in imp.head(12).items()))
    proba = pd.DataFrame(rf.predict_proba(X.to_numpy()), index=F.root, columns=rf.classes_)
    for r in typed:  # leave-one-out probabilities for the typed cells
        i = typed.index(r)
        m = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0)
        m.fit(np.delete(X.loc[typed].to_numpy(), i, axis=0), np.delete(y, i))
        proba.loc[r] = pd.Series(m.predict_proba(X.loc[[r]].to_numpy())[0], index=m.classes_).reindex(rf.classes_).fillna(0).to_numpy()
    nb_path = fig("lmc_nblast.pkl")
    nb = pickle.load(open(nb_path, "rb")) if os.path.exists(nb_path) else {}
    for key in ("align", "centroid"):
        if key in nb:
            pred, share = knn_vote(nb[key], nb["ids"], lab)
            col = f"NB_{key}"
            P[col] = pd.Series(pred)
            P[col + "_share"] = pd.Series(share)
            acc = np.mean([pred[r] == lab[r] for r in typed])
            say(f"\nNBLAST {key} 7-NN leave-one-out accuracy {acc:.0%}")
            say(pd.crosstab(pd.Series([lab[r] for r in typed], name="label"),
                            pd.Series([pred[r] for r in typed], name="predicted")).to_string())
    # the call comes from the feature models (97% leave-one-out); the NBLAST votes (62% align, 51% centroid on
    # 2026-10-31: LMC types differ mostly in lamina / medulla depth, which shape alignment largely ignores) are shown
    vote_cols = ["LDA", "kNN", "RF"]
    nb_cols = [c for c in ("NB_align", "NB_centroid") if c in P.columns]
    V = P[vote_cols]
    top_rf = proba.idxmax(axis=1)  # three different votes: the random forest's most likely type
    P["consensus"] = [v.value_counts().index[0] if v.value_counts().iloc[0] >= 2 else top_rf[r] for r, v in V.iterrows()]
    P["votes"] = V.apply(lambda r: f"{r.value_counts().iloc[0]}/{len(r)}", axis=1)
    if nb_cols:
        P["NB_agree"] = P.apply(lambda r: f"{sum(r[c] == r['consensus'] for c in nb_cols)}/{len(nb_cols)}", axis=1)
    for t in rf.classes_:
        P[f"P_{t}"] = proba[t].round(3)
    P["RF_proba"] = proba.apply(lambda r: " ".join(f"{k} {v:.2f}" for k, v in r.sort_values(ascending=False).head(3).items() if v >= 0.1), axis=1)
    T = F[["root", "serial_id", "side", "label", "status", "proofreader", "point_xyz"]].set_index("root").join(P)
    T["consistent"] = np.where(T.label.isin(TYPES), np.where(T.consensus == T.label, "yes", "NO"), "")
    T = T.reset_index()
    T.to_csv(fig("lmc_typing.tsv"), sep="\t", index=False)
    show = vote_cols + nb_cols + ["consensus", "votes"] + (["NB_agree"] if nb_cols else []) + ["RF_proba"]
    say(f"\nconsensus over {vote_cols}; NBLAST shown: {nb_cols}")
    say("typed cells whose consensus differs from their label:")
    bad = T[T.consistent == "NO"]
    say(bad[["root", "serial_id", "label", "status", "proofreader"] + show].to_string(index=False) if len(bad) else "  none")
    say("\ncalls for the unresolved cells:")
    un = T[~T.label.isin(TYPES)].sort_values(["consensus", "side"])
    say(un[["root", "serial_id", "side", "label", "status"] + show].to_string(index=False))
    say(f"\nwritten {fig('lmc_typing.tsv')}")
    open(REPORT, "w").write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
