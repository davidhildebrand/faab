"""Every input of the 20 T4/T5 targets of the PRg1toT4T5 tab, with or without LMC input, and how deep each reaches
into the lobula (2026-10-27). A collaborator saw that the tab's hop-2 cells all seem to end at the T5 dendrite layer
(Lo1) and asked whether T4/T5 inputs that project deeper (like fly Tm3 / Tm4) were left out: the tab keeps only the
inputs that also receive >= 3 synapses from a group-1 hop-1 cell. Read-only: local synapse snapshot (faab.synsnap,
newest checkpoint + one CAVE catch-up to now), SeaTable labels, aedes_ol_meshes.

Depth: the input synapses of all T4/T5-like candidates (tmp/t4t5_knn.tsv) inside LO_R mark the T5 dendrite layer.
Local planes are fitted at anchors (a subsample of those points, each from the points within RADIUS um - the layer is
~15 um thick, so smaller patches are not flat) and oriented toward the lobula interior (the side with more LO_R
volume, probed 5-40 um out). A synapse's depth is its signed distance to the plane of the nearest anchor (NaN if more
than RADIUS um sideways from it). Per input cell (>= THR synapses onto one target): median and 95th-percentile depth
of its own synapses inside LO_R (outputs at the presynaptic, inputs at the postsynaptic position). Classes, with
edge = the T5 layer's own 95th percentile + MARGIN um: 'medulla only' (< MIN_LO synapses in LO_R), 'deep lobula'
(median beyond edge), 'slightly below T5 layer' (95th percentile beyond edge), else 'at T5 layer'.
Writes figures/<stamp>_t4t5_inputs_depth.tsv + _ng.txt, prints a short link."""
import datetime
import importlib.util
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from faab import nglink, ol_meshes, synsnap
from faab.config import FIGURES, TMP, VOX
from faab.partners.sheet import utc_stamp
from faab.config import fig

warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location("cp", "/Users/davidh/Data/Vibe/FAAB/faab/scripts/paths/class_paths.py")
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)

TAB = fig("20261027d195202tUTC_FAABopticlobetargets-PRg1toT4T5.tsv")
THR = 2          # synapses onto one target to list an input (the paths used 3; fly Tm3 -> T4 is weak)
RADIUS = 30.0    # um: local plane fit and sideways reach of an anchor
MIN_LO = 5       # synapses inside LO_R needed to call a depth
MARGIN = 3.0     # um beyond the T5 layer's own 95th percentile
LMC_RE = r"\bLMC\b|\bL[1-5]\b"
POS = ["pre_x", "pre_y", "pre_z", "post_x", "post_y", "post_z"]


def tab_rows(path):
    """rows of the path tab as dicts of text (every block: hop-1, hop-2, hop-3, low priority)"""
    rows, cols = [], None
    for ln in open(path):
        f = ln.rstrip("\n").split("\t")
        if f[0] == "hop":
            cols = f
        elif cols and f[0] in ("1", "2", "3"):
            rows.append(dict(zip(cols, f)))
    return rows


def um(df, side):
    return df[[f"{side}_x", f"{side}_y", f"{side}_z"]].to_numpy(float) * np.array(VOX) / 1000.0


def lo_side(points, normals, lo):
    """+1 / -1 / 0 per plane: the side of the normal with more LO_R volume (probes at 5-40 um)"""
    import navis
    d = np.arange(5.0, 40.5, 5.0)
    plus = (points[:, None, :] + d[None, :, None] * normals[:, None, :]).reshape(-1, 3)
    minus = (points[:, None, :] - d[None, :, None] * normals[:, None, :]).reshape(-1, 3)
    p = np.asarray(navis.in_volume(plus, lo), int).reshape(len(points), -1).sum(1)
    m = np.asarray(navis.in_volume(minus, lo), int).reshape(len(points), -1).sum(1)
    return np.sign(p - m)


def main():
    stamp = utc_stamp()
    rows = tab_rows(TAB)
    client = cp.cave_client()
    prs = sorted({int(p) for r in rows if r["hop"] == "1" for p in r["PR_root_id"].split(", ") if p})
    tab = {h: [int(r["root_id_curr"] or r["root_id_orig"]) for r in rows if r["hop"] == h] for h in ("1", "2", "3")}
    m = cp.refresh_roots(client, prs + [i for v in tab.values() for i in v], "tab cells")
    now_of = lambda ids: sorted({r for i in ids for r in m[int(i)]})
    pr_now, targets = now_of(prs), now_of(tab["3"])
    in_tab = {**{r: "hop-1" for r in now_of(tab["1"])}, **{r: "hop-2" for r in now_of(tab["2"])},
              **{r: "target" for r in targets}}
    info = cp.flytable(client)
    lab = {r: v[0] for r, v in info.items()}
    glia = {r for r, v in info.items() if v[2]}
    synsnap.refresh()
    snap = synsnap.Snapshot()
    T = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    print(f"{len(pr_now)} PRs, {len(targets)} targets; synapses at {T:%Y-%m-%d %H:%M} UTC "
          f"({snap.at('latest')} + catch-up)", flush=True)

    # hop-1 cells as on the paths: >= THR synapses from a group-1 PR
    f1 = snap.edges(pre=pr_now, when=T, client=client)
    X = set(int(v) for v in f1[(f1.n >= 3) & ~f1.post.isin(pr_now)].post) - glia
    s_in = snap.synapses(post=targets, when=T, client=client)
    s_in = s_in[s_in.pre_root != s_in.post_root]
    e_in = s_in.groupby(["pre_root", "post_root"]).size().rename("n").reset_index()
    strong = e_in[e_in.n >= THR]
    U = sorted(int(v) for v in set(strong.pre_root))
    print(f"inputs of the targets: {e_in.pre_root.nunique()} cells (any synapse), {len(U)} with >= {THR} onto one "
          f"target ({strong[strong.n >= 3].pre_root.nunique()} with >= 3); hop-1 cells (>= 3 from a group-1 PR): "
          f"{len(X)}", flush=True)
    s_inU = snap.synapses(post=U, when=T, client=client)
    s_outU = snap.synapses(pre=U, when=T, client=client)
    s_inU = s_inU[s_inU.pre_root != s_inU.post_root]
    s_outU = s_outU[s_outU.pre_root != s_outU.post_root]
    knn = pd.read_csv(os.path.join(TMP, "t4t5_knn.tsv"), sep="\t", dtype={"cell": str})
    cand = sorted(int(c) for c in knn[knn.call == "T4/T5-like"].cell if c.isdigit())  # skip 'Mi1:<id>' references
    s_cand = snap.synapses(post=cand, when="latest")  # positions are static: ids from the survey, no catch-up
    ids = np.unique(np.r_[s_in.id, s_inU.id, s_outU.id, s_cand.id].astype(np.int64))
    print(f"positions of {len(ids)} synapses from static.parquet ...", flush=True)
    st = snap.static(ids, columns=POS)
    s_in, s_inU, s_outU, s_cand = (d.merge(st, on="id", how="left") for d in (s_in, s_inU, s_outU, s_cand))

    # T4 or T5: where each target receives its input
    s_in["region"] = ol_meshes.assign(um(s_in, "post"))
    kind = {int(t): ("T5" if (g.region == "LO_R").mean() > (g.region == "ME_R").mean() else "T4")
            for t, g in s_in.groupby("post_root")}
    print(f"targets: {pd.Series(kind).value_counts().to_dict()}", flush=True)

    # the T5 dendrite layer: input synapses of all T4/T5-like candidates inside LO_R
    cp_ = um(s_cand, "post")
    sheet = cp_[ol_meshes.assign(cp_, ["LO_R"]) == "LO_R"]
    lo = ol_meshes.volume("LO_R")
    tree = cKDTree(sheet)
    rng = np.random.default_rng(0)
    anc = sheet[rng.choice(len(sheet), min(2000, len(sheet)), replace=False)]
    C, N = [], []
    for a in anc:
        q = sheet[tree.query_ball_point(a, RADIUS)]
        if len(q) < 30:
            continue
        c = q.mean(0)
        w, v = np.linalg.eigh(np.cov((q - c).T))
        if w[0] > 0.25 * w[1]:  # not sheet-like here
            continue
        C.append(c)
        N.append(v[:, 0])
    C, N = np.array(C), np.array(N)
    sgn = lo_side(C, N, lo)
    C, N = C[sgn != 0], N[sgn != 0] * sgn[sgn != 0][:, None]
    atree = cKDTree(C)
    print(f"T5 layer: {len(sheet)} input synapses of {len(cand)} T4/T5-like candidates in LO_R; {len(C)} oriented "
          f"local planes", flush=True)

    def depth(x):
        _, j = atree.query(x)
        r = x - C[j]
        d = np.einsum("ij,ij->i", r, N[j])
        side = np.sqrt(np.maximum((r ** 2).sum(1) - d ** 2, 0))
        d[side > RADIUS] = np.nan
        return d

    sd = depth(sheet)
    s5, s50, s95 = np.nanpercentile(sd, [5, 50, 95])
    # lobula thickness along the normal at the targets' T5 patch
    t5pts = um(s_in, "post")[(s_in.region == "LO_R").to_numpy() & s_in.post_root.map(kind).eq("T5").to_numpy()]
    _, j0 = atree.query(t5pts.mean(0))
    import navis
    steps = np.arange(-60, 150.5, 0.5)
    inside = np.asarray(navis.in_volume(C[j0] + steps[:, None] * N[j0][None, :], lo), bool)
    k0 = np.where(steps == 0)[0][0]
    hi_ = k0
    while hi_ + 1 < len(steps) and inside[hi_ + 1]:
        hi_ += 1
    lo_ = k0
    while lo_ - 1 >= 0 and inside[lo_ - 1]:
        lo_ -= 1
    print(f"T5 layer depth spread (5/50/95%): {s5:.1f} / {s50:.1f} / {s95:.1f} um; at the targets' patch LO_R runs "
          f"from {steps[lo_]:.1f} um (outer surface) to {steps[hi_]:.1f} um (deepest) along the normal "
          f"(inside at the plane: {bool(inside[k0])})", flush=True)

    # per input cell: synapses inside the neuropils and their lobula depth
    P_cell = np.r_[s_outU.pre_root.to_numpy(np.int64), s_inU.post_root.to_numpy(np.int64)]
    P_xyz = np.vstack([um(s_outU, "pre"), um(s_inU, "post")])
    P_reg = ol_meshes.assign(P_xyz)
    P = pd.DataFrame({"cell": P_cell, "reg": P_reg})
    isLO = P.reg.eq("LO_R").to_numpy()
    P["depth"] = np.nan
    P.loc[isLO, "depth"] = depth(P_xyz[isLO])
    fromX = s_inU[s_inU.pre_root.isin(X)].groupby(["post_root", "pre_root"]).size()
    is_lmc = s_inU.pre_root.map(lambda r: bool(re.search(LMC_RE, lab.get(int(r), ""))))
    fromLMC = s_inU[is_lmc].groupby("post_root").size()
    out = []
    for u in U:
        g = strong[strong.pre_root == u]
        k = g.post_root.map(kind)
        p = P[P.cell == u]
        fr = p.reg.value_counts(normalize=True).to_dict()
        dl = p.depth.dropna()
        nlo = int((p.reg == "LO_R").sum())
        d95 = float(np.percentile(dl, 95)) if len(dl) >= MIN_LO else np.nan
        d50 = float(dl.median()) if len(dl) >= MIN_LO else np.nan
        cls = ("medulla only" if nlo < MIN_LO else "outside patch" if np.isnan(d95)
               else "deep lobula" if d50 > s95 + MARGIN else "slightly below T5 layer" if d95 > s95 + MARGIN
               else "at T5 layer")
        fx = fromX.loc[u] if u in fromX.index.get_level_values(0) else pd.Series(dtype=int)
        out.append(dict(cell=str(u), seatable=lab.get(u, ""), in_tab=in_tab.get(u, ""),
                        max_to_target=int(g.n.max()), n_T4=int((k == "T4").sum()), n_T5=int((k == "T5").sum()),
                        syn_to_T4=int(g.n[k == "T4"].sum()), syn_to_T5=int(g.n[k == "T5"].sum()),
                        hop1_max=int(fx.max()) if len(fx) else 0, hop1_sum=int(fx.sum()) if len(fx) else 0,
                        from_LMC=int(fromLMC.get(u, 0)), n_syn=len(p), neuropils=ol_meshes.summary(fr),
                        lo_syn=nlo, lo_depth_med=round(float(dl.median()), 1) if len(dl) else np.nan,
                        lo_depth_p95=round(d95, 1) if not np.isnan(d95) else np.nan, depth_class=cls,
                        glia=u in glia))
    D = pd.DataFrame(out)
    D["syn_to_targets"] = D.syn_to_T4 + D.syn_to_T5
    D = D.sort_values(["in_tab", "depth_class", "syn_to_targets"], ascending=[False, True, False])
    base = fig(f"{stamp}_t4t5_inputs_depth")
    D.to_csv(base + ".tsv", sep="\t", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 60)
    D["grp"] = np.where(D.in_tab.eq(""), "not in tab", D.in_tab)
    D["onto"] = np.where(D.n_T4 > 0, np.where(D.n_T5 > 0, "T4+T5", "T4"), "T5")
    for t in (3, THR):
        d = D[D.max_to_target >= t]
        print(f"\ninputs (>= {t} onto a target) by tab role and depth class:")
        print(pd.crosstab(d.grp, d.depth_class, margins=True).to_string())
        print(f"... by targets contacted (T4 / T5) and depth class:")
        print(pd.crosstab(d.onto, d.depth_class, margins=True).to_string())
    print("\nsynapses onto the targets by group:", D.groupby("grp").syn_to_targets.sum().to_dict())
    show = ["cell", "seatable", "in_tab", "max_to_target", "n_T4", "n_T5", "syn_to_T4", "syn_to_T5", "hop1_max",
            "from_LMC", "n_syn", "neuropils", "lo_syn", "lo_depth_med", "lo_depth_p95", "depth_class"]
    print("\ntab hop-2 cells:")
    print(D[D.in_tab == "hop-2"][show].to_string(index=False))
    print("\nnot in the tab, deep lobula or slightly below the T5 layer:")
    print(D[(D.grp == "not in tab") & D.depth_class.isin(["deep lobula", "slightly below T5 layer"])][show]
          .to_string(index=False))
    print("\nnot in the tab, other classes (top 30 by synapses onto targets):")
    print(D[(D.grp == "not in tab") & ~D.depth_class.isin(["deep lobula", "slightly below T5 layer"])]
          .sort_values("syn_to_targets", ascending=False)[show].head(30).to_string(index=False))

    # Spelunker: targets, tab hop-2, inputs not in the tab by depth class
    sel = lambda q: [c for c in D[q].cell]
    layers = [
        ("T4 targets", [str(t) for t in targets if kind.get(t) == "T4"], "#7a1010", True),
        ("T5 targets", [str(t) for t in targets if kind.get(t) == "T5"], "#c0392b", True),
        ("hop-2 in tab", sel(D.in_tab == "hop-2"), "#e9953a", True),
        ("not in tab - deep lobula", sel(D.grp.eq("not in tab") & D.depth_class.eq("deep lobula")), "#1f77d6", True),
        ("not in tab - slightly below T5 layer",
         sel(D.grp.eq("not in tab") & D.depth_class.eq("slightly below T5 layer")), "#17becf", False),
        ("not in tab - at T5 layer", sel(D.grp.eq("not in tab") & D.depth_class.eq("at T5 layer")), "#2ecc71", False),
        ("not in tab - medulla only", sel(D.grp.eq("not in tab") & D.depth_class.eq("medulla only")), "#9467bd", False),
    ]
    neurons = [{"name": f"{n} ({len(ids_)})", "ids": ids_, "color": col} for n, ids_, col, _ in layers if ids_]
    centre = t5pts.mean(0) * 1000.0 / np.array(VOX)
    spec_ = {"meshes": [{"kind": "OL", "segments": ["21", "22", "23"]}], "neurons": neurons, "view": "anterior",
             "layout": "3d", "select": neurons[3]["name"] if len(neurons) > 3 else neurons[-1]["name"],
             "position": [float(v) for v in centre], "scale": 9000, "image": True}
    state = nglink.build(spec_)
    hidden = {f"{n} ({len(ids_)})" for n, ids_, _, vis in layers if not vis}
    for L in state["layers"]:
        if L["name"] in hidden:
            L["visible"] = False
    open(base + "_ng.txt", "w").write(nglink.to_url(state) + "\n")
    cp.wait()
    sid = client.state.upload_state_json(state)
    print(f"\nshort link: {client.state.build_neuroglancer_url(sid, ngl_url='https://spelunker.cave-explorer.org')}")
    print(f"wrote {base}.tsv and _ng.txt")


if __name__ == "__main__":
    main()
