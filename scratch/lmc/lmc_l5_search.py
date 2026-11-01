"""L5 search (2026-11-01). L5 gets almost no photoreceptor input, and its inputs/outputs are spread over many
partners; the one consistent LMC partner is L4 (both directions: 12 L4 give 430 synapses to 12 of 39 L5s, L5s give
244 to 11 L4s). So: partners (>= MIN_SYN synapses either way) of the L4s - SeaTable L4s (roots at their points,
lmc_known_roots.tsv) and the candidates called L4 with high/medium confidence (lmc_cand_typing.tsv) - with the typed
L5 profile (lmc_type_profiles.py: lamina inputs 11-23, lamina outputs 55-138, medulla outputs 676-1179, p10-p90),
loosened for incomplete cells: <= MAX_IN_LA lamina inputs, >= MIN_OUT_LA lamina outputs, >= MIN_OUT_ME medulla outputs,
and the soma by the lamina when there is a nucleus (soma_side.py). Recall check: the SeaTable L5s.
Synapse positions from the local snapshot (caught up to now). Writes figures/lmc/lmc_l5_candidates.tsv.
Usage: lmc_l5_search.py SEATABLE.tsv"""
import datetime
import sys

import numpy as np
import pandas as pd
import caveclient
from scipy.spatial import cKDTree

from faab import ol_meshes, synsnap
from faab.clients.cave import DATASTACK
from faab.config import fig
from faab.partners.top10_xyz import VOX
from faab.pr.lamina_footprint import flat
from soma_side import soma_table

MIN_SYN, MAX_IN_LA, MIN_OUT_LA, MIN_OUT_ME, MARGIN_UM = 3, 60, 20, 100, 5.0
S = pd.read_csv(sys.argv[1], sep="\t", dtype=str).fillna("")
K = pd.read_csv(fig("lmc_known_roots.tsv"), sep="\t", dtype={"root": str}).fillna("")
Tc = pd.read_csv(fig("lmc_cand_typing.tsv"), sep="\t", dtype={"root": str}).fillna("")
ktype = K.label.str.extract(r"(?:^|/ )(L[1-5])$", expand=False)  # exact types; LRS's 'L1-L5' is not an L5
l4 = set(K[ktype == "L4"].root) | set(Tc[(Tc.call == "L4") & Tc.confidence.isin(["high", "medium"])].root)
side_of = dict(zip(K.root, K.side)) | dict(zip(Tc.root, Tc.side))
known_l5 = set(K[ktype == "L5"].root)
taken = (set(K.root) | set(Tc[Tc.call != "L5"].root)) - known_l5
print(f"L4s: {len(l4)}; known L5s: {len(known_l5)}", flush=True)

snap, client = synsnap.Snapshot(log=lambda *_: None), caveclient.CAVEclient(DATASTACK)
ids = [int(r) for r in l4]
e = pd.concat([snap.edges(pre=ids, when="now", client=client).assign(l4=lambda d: d.pre, partner=lambda d: d.post),
               snap.edges(post=ids, when="now", client=client).assign(l4=lambda d: d.post, partner=lambda d: d.pre)])
e = e.astype({"l4": str, "partner": str})
e = e[~e.partner.isin(l4)]
w = e.groupby("partner").agg(n_l4=("n", "sum"), l4s=("l4", "nunique"),
                             side=("l4", lambda x: pd.Series([side_of.get(r, "R") for r in x]).value_counts().index[0]))
P = w[(w.n_l4 >= MIN_SYN) & ~w.index.isin(taken)].reset_index().rename(columns={"partner": "root"})
print(f"partners of the L4s with >= {MIN_SYN} synapses (not already LMCs): {len(P)}; known L5s among them "
      f"{len(known_l5 & set(P.root))}/{len(known_l5)}", flush=True)

now = datetime.datetime.now(datetime.timezone.utc)
ref = pd.read_csv(fig("pr_lattice_known.tsv"), sep="\t")[["lam_x", "lam_y", "lam_z"]].to_numpy(float).mean(0)
MESHES = ["LA_R", "LA_L", "ME_R", "ME_L"]
trees = {m: cKDTree(ol_meshes.mesh(m)[0] / 1000.0) for m in MESHES}


def region(Q):
    reg = ol_meshes.assign(Q, MESHES).astype(object) if len(Q) else np.array([], dtype=object)
    none = np.flatnonzero(reg == "none")
    if len(none):
        D = np.column_stack([trees[m].query(Q[none])[0] for m in MESHES])
        b = D.argmin(1)
        reg[none] = np.where(D[np.arange(len(none)), b] <= MARGIN_UM, np.array(MESHES)[b], "none")
    return reg


def counts(kw, who):
    d = snap.synapses(**kw, when="now", positions=True, client=client)
    d = d.rename(columns={"pre_root": "pre_pt_root_id", "post_root": "post_pt_root_id",
                          **{f"{p}_{a}": f"{p}_pt_position_{a}" for p in ("pre", "post") for a in "xyz"}})
    f = flat(d, list(VOX), f"{now:%Y-%m-%d %H:%M} UTC (synapse snapshot)", ref)
    f = f[f.pre != f.post].astype({"pre": str, "post": str})
    f["np"] = region(f[[f"{who}_x", f"{who}_y", f"{who}_z"]].to_numpy(float))
    return f.groupby([who, "np"]).size().unstack(fill_value=0)


roots = [int(r) for r in P.root]
co, ci = counts({"pre": roots}, "pre"), counts({"post": roots}, "post")
for name, c in (("out", co), ("in", ci)):
    for npl in ("LA", "ME"):
        P[f"n_{name}_{npl}"] = [int(c.loc[r, f"{npl}_{sd}"]) if r in c.index and f"{npl}_{sd}" in c.columns else 0
                                for r, sd in zip(P.root, P.side)]
P = P.merge(soma_table(P.root, P.side), on="root")
P["known_l5"] = P.root.isin(known_l5)
P["l5_profile"] = (P.n_in_LA <= MAX_IN_LA) & (P.n_out_LA >= MIN_OUT_LA) & (P.n_out_ME >= MIN_OUT_ME) & (P.soma_by != "ME")
lab = S[["superclass", "class", "subclass", "subsubclass", "type"]].agg(" / ".join, axis=1).str.strip(" /")
P["seatable"] = P.root.map(dict(zip(S.root_now, lab)) | dict(zip(K.root, K.label))).fillna("")
P = P.sort_values(["l5_profile", "n_out_LA"], ascending=False)
P.to_csv(fig("lmc_l5_candidates.tsv"), sep="\t", index=False)
k = P[P.known_l5]
print(f"known L5s with the L5 profile: {int(k.l5_profile.sum())}/{len(k)} found (of {len(known_l5)})")
if (~k.l5_profile).any():
    print(k[~k.l5_profile][["root", "n_l4", "n_in_LA", "n_out_LA", "n_out_ME", "soma_by"]].to_string(index=False))
new = P[P.l5_profile & ~P.known_l5]
print(f"\nnew L5 candidates with the L5 profile: {len(new)} ({new.side.value_counts().to_dict()}); soma "
      f"{new.soma_by.replace('', 'no nucleus').value_counts().to_dict()}; SeaTable labels "
      f"{new.seatable.replace('', 'none').value_counts().head(6).to_dict()}")
q = lambda s: f"{int(s.quantile(0.1))}/{int(s.median())}/{int(s.quantile(0.9))}"
print("profile p10/p50/p90 (known L5 found | new):")
for c in ("n_l4", "n_in_LA", "n_out_LA", "n_in_ME", "n_out_ME"):
    print(f"  {c}: {q(k[k.l5_profile][c]) if k.l5_profile.any() else '-'} | {q(new[c]) if len(new) else '-'}")
print("written", fig("lmc_l5_candidates.tsv"))
