"""Find LMC candidates on both sides (David 2026-11-01: find as many LMCs as possible on the R and L sides, subtype
them, others confirm). Step 1: every cell that receives synapses from photoreceptors inside a lamina.

Photoreceptors (today's roots where available): the PRs-tab rows of PR classes (figures/ommatidia/<RUN>/tab_roots.tsv,
roots looked up at their xyz), the right-side column-zone axons with an R1-R6 / R7 / R8 synapse verdict
(figures/ommatidia/<RUN>/features.tsv), the left-side census axons with such a verdict
(figures/pr_groups/pr_left_bundles_roots_verified.tsv, roots of 2026-10-29) and SeaTable's PRs / OLSN rows.
Only R1-R6 and R8 (both end in the lamina), all their output synapses (local snapshot caught up to now; CAVE catch-up
only); per postsynaptic cell the count and its side (the side of its photoreceptors). Then every cell with >= MIN_PR
such synapses gets its own input and output counts per neuropil (LA, ME of its side; within MARGIN_UM of a mesh
counts) and its SeaTable label. Known LMCs (SeaTable class LMC, roots looked up at their SeaTable points now, written
to figures/lmc/lmc_known_roots.tsv) show how many this net catches.
Writes figures/lmc/lmc_candidates.tsv. Usage: lmc_candidates.py SEATABLE.tsv RUN"""
import datetime
import os
import sys

import numpy as np
import pandas as pd
import caveclient
from fafbseg import flywire
from scipy.spatial import cKDTree

from faab import ol_meshes, synsnap
from faab.clients.cave import DATASTACK
from faab.config import fig
from faab.partners.top10_xyz import VOX
from faab.pr.lamina_footprint import flat

MIN_PR = 3
# R1-R6 and R8 only: in Aedes both end in the lamina (0% of outputs in the medulla), so all their synapses count and no
# lamina mesh is needed (2026-11-01: the mesh had dropped two known L1/L2 whose 52/89 photoreceptor synapses lie just
# outside it); R7 (and unresolved R7/R8) would add medulla targets
PR_CLASSES = ("R1-R6", "R8", "R1-R6?")
PR_VERDICTS = ("R1-R6-like", "R1-R6-like (few sites)", "R8-like")
MARGIN_UM = 5.0  # a synapse outside both meshes but within this of one counts for that neuropil

st_path, run = sys.argv[1], sys.argv[2]
S = pd.read_csv(st_path, sep="\t", dtype=str).fillna("")
od = fig(f"ommatidia/{run}")
tab = pd.read_csv(os.path.join(od, "tab_roots.tsv"), sep="\t", dtype=str).fillna("")
F = pd.read_csv(os.path.join(od, "features.tsv"), sep="\t", dtype={"root": str})
Lv = pd.read_csv(fig("pr_left_bundles_roots_verified.tsv"), sep="\t", dtype={"root": str}).fillna("")
side = {}
for r, s in zip(tab[tab.cls.isin(PR_CLASSES)].root_now, tab[tab.cls.isin(PR_CLASSES)].side):
    side[r] = "R" if s == "right" else "L"
side.update({r: "R" for r in F[F.verdict.isin(PR_VERDICTS)].root})
side.update({r: "L" for r in Lv[Lv.verdict.isin(PR_VERDICTS)].root})
sp = S[(S.superclass == "ol_sensory") & S[["subclass", "subsubclass", "type"]].agg(" ".join, axis=1).str.contains("R1-6")]
side.update({r: s for r, s in zip(sp.root_now, sp.side) if r})
prs = sorted(int(r) for r in side if r not in ("", "0"))
print(f"photoreceptors: {len(prs)} roots ({pd.Series(list(side.values())).value_counts().to_dict()})", flush=True)

snap, client = synsnap.Snapshot(log=lambda *_: None), caveclient.CAVEclient(DATASTACK)
now = datetime.datetime.now(datetime.timezone.utc)
ref = pd.read_csv(fig("pr_lattice_known.tsv"), sep="\t")[["lam_x", "lam_y", "lam_z"]].to_numpy(float).mean(0)


def positions(df):
    df = df.rename(columns={"pre_root": "pre_pt_root_id", "post_root": "post_pt_root_id",
                            **{f"{p}_{a}": f"{p}_pt_position_{a}" for p in ("pre", "post") for a in "xyz"}})
    out = flat(df, list(VOX), f"{now:%Y-%m-%d %H:%M} UTC (synapse snapshot)", ref)
    return out[out.pre != out.post].astype({"pre": str, "post": str}).reset_index(drop=True)


MESHES = ["LA_R", "LA_L", "ME_R", "ME_L"]
trees = {m: cKDTree(ol_meshes.mesh(m)[0] / 1000.0) for m in MESHES}


def region(P):
    """neuropil of each point (um): inside a mesh, else the nearest mesh within MARGIN_UM, else 'none'"""
    reg = ol_meshes.assign(P, MESHES) if len(P) else np.array([], dtype=object)
    out = reg.astype(object)
    none = np.flatnonzero(reg == "none")
    if len(none):
        D = np.column_stack([trees[m].query(P[none])[0] for m in MESHES])
        best = D.argmin(1)
        out[none] = np.where(D[np.arange(len(none)), best] <= MARGIN_UM, np.array(MESHES)[best], "none")
    return out


s = positions(snap.synapses(pre=prs, when="now", positions=True, client=client))
# no photoreceptor is an LMC candidate: R7s get R8 / R1-R6 synapses and have lamina input and medulla output, so 56 R7s
# (and R7/R8, R1-R6? rows of the PRs tab) reached the LMCs tab on 2026-11-01 (removed 2026-11-07)
all_prs = set(map(str, prs)) | set(tab[tab.cls.isin(["R1-R6", "R1-R6?", "R7", "R8", "R7/R8"])].root_now) | \
    set(F[F.verdict.str.contains("R[1-8]", regex=True, na=False)].root) | \
    set(Lv[Lv.verdict.str.contains("R[1-8]", regex=True, na=False)].root)
s = s[~s.post.isin(all_prs)]
s["side"] = s.pre.map({str(r): sd for r, sd in side.items()})
T = s.groupby("post").agg(n_pr=("pre", "size"), n_pr_cells=("pre", "nunique"),
                          side=("side", lambda x: x.value_counts().index[0])).reset_index().rename(columns={"post": "root"})
T = T[T.n_pr >= MIN_PR].copy()
print(f"cells with >= {MIN_PR} synapses from R1-R6 / R8: {len(T)} ({T.side.value_counts().to_dict()})", flush=True)

# their own synapses, per neuropil of their side
roots = [int(r) for r in T.root]
out = positions(snap.synapses(pre=roots, when="now", positions=True, client=client))
inn = positions(snap.synapses(post=roots, when="now", positions=True, client=client))
for name, df, who, col in (("out", out, "pre", "pre"), ("in", inn, "post", "post")):
    P = df[[f"{col}_x", f"{col}_y", f"{col}_z"]].to_numpy(float)
    df["np"] = region(P)
    c = df.groupby([who, "np"]).size().unstack(fill_value=0)
    T[f"n_{name}"] = T.root.map(df.groupby(who).size()).fillna(0).astype(int)
    for npl in ("LA", "ME"):
        T[f"n_{name}_{npl}"] = [int(c.loc[r, f"{npl}_{sd}"]) if r in c.index and f"{npl}_{sd}" in c.columns else 0
                                for r, sd in zip(T.root, T.side)]
lab = S[["superclass", "class", "subclass", "subsubclass", "type"]].agg(" / ".join, axis=1).str.strip(" /")
known = S[S["class"] == "LMC"].copy()
# SeaTable's roots can be stale (several LRS cells showed 0 synapses): the root at each known LMC's point now
ok = known.point_xyz.str.count(",") == 2
locs = np.array([[float(v) for v in x.split(",")] for x in known.point_xyz[ok]]) * np.array(VOX)
known.loc[ok, "root_now"] = [str(int(x)) for x in flywire.locs_to_segments(locs, coordinates="nm", backend="cloudvolume",
                                                                           dataset=DATASTACK)]
pd.DataFrame({"serial_id": known.serial_id, "root": known.root_now, "side": known.side,
              "label": lab[known.index]}).to_csv(fig("lmc_known_roots.tsv"), sep="\t", index=False)
st = dict(zip(S.root_now, lab)) | dict(zip(known.root_now, lab[known.index]))
sid = dict(zip(S.root_now, S.serial_id)) | dict(zip(known.root_now, known.serial_id))
T["seatable"] = T.root.map(st).fillna("")
T["serial_id"] = T.root.map(sid).fillna("")
T = T.sort_values("n_pr", ascending=False)
T.to_csv(fig("lmc_candidates.tsv"), sep="\t", index=False)
caught = known.root_now.isin(set(T.root))
print(f"\nknown LMCs (SeaTable class LMC): {len(known)}; caught {int(caught.sum())} "
      f"({known[caught].side.value_counts().to_dict()}); missed {int((~caught).sum())} ({known[~caught].side.value_counts().to_dict()})")
T["kind"] = np.where(T.seatable.str.contains("LMC"), "SeaTable LMC", np.where(T.seatable != "", "SeaTable other", "not in SeaTable"))
cols = ["n_pr", "n_in_LA", "n_out_LA", "n_in_ME", "n_out_ME", "n_in", "n_out"]
print(T.groupby("kind")[cols].median().round(0).to_string())
print(f"\nSeaTable labels among the candidates:\n{T[T.seatable != ''].seatable.value_counts().head(25).to_string()}")
print("written", fig("lmc_candidates.tsv"))
