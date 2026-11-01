"""L1-L5 calls for the LMC search candidates (2026-11-01). Trained on the SeaTable-typed LMCs of the typing set
(lmc_features.tsv; 97% leave-one-out on 2026-10-31, all right side), applied to lmc_cand_features.tsv (lmc_features.py
--set cand): the consensus of LDA, kNN and random forest as in lmc_classify.py, with the forest's probability.
Two checks per candidate:
  typical   how far its features lie from the typed cells (mean distance to its 5 nearest typed cells in the
            standardized feature space, as a percentile of the typed cells' own leave-one-out distances); beyond the
            95th percentile = 'unlike any typed LMC' (another cell type, or an incomplete cell)
  complete  medulla outputs and medulla cable compared with the typed cells of its called type (below their 10th
            percentile = 'incomplete?')
  soma      the nucleus on the root (soma_side.py): every known LMC with one has it by the lamina (90/90, median
            4.7 um from the lamina, 51 um from the medulla), so a soma by the medulla means 'not an LMC'
Confidence: high = 3/3 models, probability >= 0.6, typical, complete, soma by the lamina; low = probability < 0.4, a
1/3 vote, not typical or soma by the medulla; else medium (e.g. no nucleus on the root). Left-side calls are marked:
the model has only seen right-side cells.
Network-free. Writes figures/lmc/lmc_cand_typing.tsv. Usage: lmc_cand_classify.py"""
import warnings

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from faab.config import fig
from lmc_classify import TYPES, feature_table
from soma_side import soma_table

warnings.filterwarnings("ignore")
F0 = pd.read_csv(fig("lmc_features.tsv"), sep="\t", dtype={"root": str, "serial_id": str}).fillna({"label": ""})
Fc = pd.read_csv(fig("lmc_cand_features.tsv"), sep="\t", dtype={"root": str, "serial_id": str}).fillna({"label": ""})
tr = F0[F0.label.isin(TYPES)].reset_index(drop=True)
X0, Xc = feature_table(tr), feature_table(Fc)
Xc = Xc[X0.columns].fillna(X0.median())
y = tr.label.to_numpy()
models = {"LDA": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
          "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance")),
          "RF": RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=0)}
P = pd.DataFrame(index=Fc.root)
for name, m in models.items():
    m.fit(X0.to_numpy(), y)
    P[name] = m.predict(Xc.to_numpy())
proba = pd.DataFrame(models["RF"].predict_proba(Xc.to_numpy()), index=Fc.root, columns=models["RF"].classes_)
V = P[["LDA", "kNN", "RF"]]
P["call"] = [v.value_counts().index[0] if v.value_counts().iloc[0] >= 2 else proba.loc[r].idxmax() for r, v in V.iterrows()]
P["votes"] = V.apply(lambda r: f"{r.value_counts().iloc[0]}/3", axis=1)
P["p"] = [float(proba.loc[r, c]) for r, c in zip(P.index, P.call)]
for t in TYPES:
    P[f"P_{t}"] = proba[t].round(3)

# typical: distance to the 5 nearest typed cells, against the typed cells' own leave-one-out distances
sc = StandardScaler().fit(X0.to_numpy())
Z0, Zc = sc.transform(X0.to_numpy()), sc.transform(Xc.to_numpy())
nn = NearestNeighbors(n_neighbors=6).fit(Z0)
d0 = nn.kneighbors(Z0)[0][:, 1:].mean(1)  # itself excluded
dc = nn.kneighbors(Zc, n_neighbors=5)[0].mean(1)
P["typical_pct"] = np.round([100 * (d0 <= x).mean() for x in dc], 0)
P["typical"] = P.typical_pct <= 95
# complete: medulla outputs and cable against the typed cells of the called type
ref = tr.groupby("label")[["n_out_ME", "cable_ME"]].quantile(0.1)
P["complete"] = [bool(f.n_out_ME >= ref.loc[c, "n_out_ME"] and f.cable_ME >= ref.loc[c, "cable_ME"])
                 for f, c in zip(Fc.itertuples(index=False), P.call)]
So = soma_table(Fc.root, Fc.side).set_index("root")
P = P.join(So)
P["confidence"] = ["high" if (v == "3/3" and p >= 0.6 and t and c and s == "LA") else
                   "low" if (p < 0.4 or v == "1/3" or not t or s == "ME") else "medium"
                   for v, p, t, c, s in zip(P.votes, P.p, P.typical, P.complete, P.soma_by)]
T = Fc[["root", "serial_id", "side", "label", "n_in_LA", "n_out_LA", "n_in_ME", "n_out_ME", "cable_LA", "cable_ME"]] \
    .set_index("root").join(P).reset_index()
T["note"] = [", ".join(x for x in (
    "soma by the medulla: probably not an LMC" if s == "ME" else "",
    "no nucleus on this root (soma cut off or fragment)" if s == "" else "",
    "several nuclei on this root: merge?" if s == "merge?" else "",
    "left side: model trained on right-side cells only" if sd == "L" else "") if x)
    for s, sd in zip(T.soma_by, T.side)]
T.to_csv(fig("lmc_cand_typing.tsv"), sep="\t", index=False)
print(f"{len(T)} candidates; calls by side:\n{pd.crosstab(T.call, T.side).to_string()}")
print(f"\nconfidence by side:\n{pd.crosstab(T.confidence, T.side).to_string()}")
print(f"\nunlike any typed LMC: {int((~T.typical).sum())}; incomplete?: {int((~T.complete).sum())}")
print(f"soma: {T.soma_by.replace('', 'no nucleus').value_counts().to_dict()}")
print(f"\nSeaTable-labeled candidates (stale roots of known cells): "
      f"{T[T.label != ''][['root', 'label', 'call', 'p']].to_string(index=False) if (T.label != '').any() else 'none'}")
print("written", fig("lmc_cand_typing.tsv"))
