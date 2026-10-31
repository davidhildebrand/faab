"""Side-by-side comparison of Drosophila R1-R6 lamina terminals (male CNS, neuPrint) with a FAAB PR group.

Drosophila: one lamina cartridge = the six R1-R6 cells presynaptic to one L1 (neuPrint male-cns:v1.0; types printed and
checked at run time). Their skeletons (8 nm voxels -> um) and presynaptic sites are put through the SAME frame as the
FAAB pairing analysis: bundle axis = PCA of the pooled nodes, lamina side from the presynaptic sites, terminal zone =
last 35 um, mirror plane search by reflection + NBLAST, t/d of the terminal-zone nodes. (In a Drosophila cartridge the
six axons come from six different ommatidia, so there is no bundle: the frame is still well defined by the six parallel
axons entering one cartridge.) FAAB: group 1 from faab_pr_pairing_demo.analyse_group (cached skeletons).
Figure: figures/pr_fly_vs_faab_group<g>.png, three columns at the same scale (face-on, edge-on, terminal cross-section)
for both; per-cell terminal extent (p90 radius of terminal-zone nodes about the cell's own centroid) and the length of
the terminal zone are printed and saved to figures/pr_fly_vs_faab_group<g>.tsv.
Usage: python faab_pr_fly_compare.py [faab group] [--l1 bodyId]"""
import os
import sys
import warnings

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import navis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from faab.clients.neuprint import client as np_client, MALE_CNS
import neuprint
from faab.pr.groups_mirror import reflect, THETAS
from faab.pr.pairing_demo import analyse_group, PAIR_COL, OUT
from faab.config import fig

warnings.filterwarnings("ignore")
navis.set_pbars(hide=True)
ZONE = 35.0


def fly_cartridge(c, l1_body=None):
    """six R1-R6 inputs of one L1 (the L1 with the most R-type inputs unless given)"""
    rtypes = neuprint.fetch_neurons(neuprint.NeuronCriteria(type="^R[1-8].*", regex=True), client=c)[0]
    print("photoreceptor types in male-cns:", rtypes.type.value_counts().head(10).to_dict(), flush=True)
    R16 = neuprint.NeuronCriteria(type="R1-R6")
    l1s = neuprint.fetch_neurons(neuprint.NeuronCriteria(type="L1"), client=c)[0]
    if l1_body is None:
        cand = l1s.sort_values("post", ascending=False).head(40)
        conn = neuprint.fetch_simple_connections(upstream_criteria=R16,
                                                 downstream_criteria=neuprint.NeuronCriteria(bodyId=list(cand.bodyId)), client=c)
        # the cartridge whose six strongest R1-R6 inputs are all strong: maximize the 6th-largest weight
        sixth = conn.groupby("bodyId_post").weight.apply(lambda w: sorted(w, reverse=True)[5] if len(w) >= 6 else 0)
        l1_body = int(sixth.idxmax())
        print(f"  best L1 among {len(cand)} probed: 6th-largest R1-R6 weight {int(sixth.max())}", flush=True)
    conn = neuprint.fetch_simple_connections(upstream_criteria=R16, downstream_criteria=neuprint.NeuronCriteria(bodyId=l1_body),
                                             client=c)
    conn = conn.sort_values("weight", ascending=False)
    print(f"L1 {l1_body}: R1-R6 inputs", list(zip(conn.bodyId_pre, conn.weight))[:8], flush=True)
    bodies = [int(b) for b in conn.bodyId_pre.head(6)]
    neu, presyn = {}, {}
    for b in bodies:
        swc = c.fetch_skeleton(b, format="pandas")  # rowId, x, y, z, radius, link (8-nm voxels)
        nodes = swc.rename(columns={"rowId": "node_id", "link": "parent_id"})[["node_id", "parent_id", "x", "y", "z", "radius"]]
        sk = navis.TreeNeuron(nodes, id=b)
        sk.soma = None
        sk = navis.resample_skeleton(sk, resample_to=1000 / 8, inplace=False)  # 1 um in 8-nm voxels
        sk.soma = None
        sk.nodes[["x", "y", "z"]] = sk.nodes[["x", "y", "z"]].values * 8 / 1000.0
        sk.units = "um"
        neu[b] = sk
        syn = neuprint.fetch_synapses(neuprint.NeuronCriteria(bodyId=b), neuprint.SynapseCriteria(type="pre"), client=c)
        presyn[b] = syn[["x", "y", "z"]].to_numpy(float) * 8 / 1000.0
    return l1_body, bodies, neu, presyn


def fly_cartridge_flywire(l1_root=None, mat=783):
    """FlyWire public (FAFB, release 783): six R1-6 inputs of one L1, precomputed skeletons (nm -> um), presynaptic sites"""
    from fafbseg import flywire
    ann = flywire.get_hierarchical_annotations(materialization=mat, dataset="public")
    ann = ann[ann.root_id.notna()]
    ct = ann.cell_type.astype(str)
    r16 = set(ann.root_id[ct.str.fullmatch(r"R1-6|R1-R6|R[1-6]")].astype(np.int64))
    l1 = ann[ct == "L1"].root_id.astype(np.int64).tolist()
    print(f"FlyWire {mat}: {len(r16)} R1-6 cells, {len(l1)} L1 cells", flush=True)
    if l1_root is None:
        probe = l1[:90]
        up = flywire.get_connectivity(probe, upstream=True, downstream=False, materialization=mat, dataset="public",
                                      min_score=50, progress=False)
        up = up[up.pre.astype(np.int64).isin(r16)]
        # the cartridge whose six strongest R1-6 inputs are all strong: maximize the 6th-largest weight
        sixth = up.groupby("post").weight.apply(lambda w: sorted(w, reverse=True)[5] if len(w) >= 6 else 0)
        l1_root = int(sixth.idxmax())
        print(f"  best L1 among {len(probe)} probed: 6th-largest R1-6 weight {int(sixth.max())}", flush=True)
    up = flywire.get_connectivity([l1_root], upstream=True, downstream=False, materialization=mat, dataset="public",
                                  min_score=50, progress=False)
    up = up[up.pre.astype(np.int64).isin(r16)].sort_values("weight", ascending=False)
    print(f"L1 {l1_root}: R1-6 inputs", list(zip(up.pre.astype(np.int64), up.weight))[:8], flush=True)
    bodies = [int(b) for b in up.pre.head(6)]
    sks = flywire.get_skeletons(bodies, dataset=mat, progress=False)
    neu, presyn = {}, {}
    syn = flywire.get_synapses(bodies, pre=True, post=False, materialization=mat, dataset="public", progress=False)
    for sk in sks:
        sk.soma = None  # precomputed skeletons carry an array-valued soma that breaks navis.subset_neuron
        sk = navis.resample_skeleton(sk, resample_to=1000, inplace=False)
        sk.soma = None
        sk.nodes[["x", "y", "z"]] = sk.nodes[["x", "y", "z"]].values / 1000.0
        sk.units = "um"
        b = int(sk.id)
        neu[b] = sk
        s = syn[syn.pre.astype(np.int64) == b][["pre_x", "pre_y", "pre_z"]].to_numpy(float)
        # positions may come in 4 x 4 x 40 nm voxels or in nm: pick the scaling that lands on the skeleton
        box = sk.nodes[["x", "y", "z"]].values
        cand = {"nm": s / 1000.0, "vox": s * np.array([4, 4, 40]) / 1000.0}
        unit = min(cand, key=lambda k: np.linalg.norm(np.median(cand[k], 0) - box.mean(0)))
        presyn[b] = cand[unit]
    print(f"  presynaptic positions interpreted as {unit}", flush=True)
    return l1_root, [b for b in bodies if b in neu], neu, presyn


def frame_and_pairs(neu, presyn, outer):
    pts = np.vstack([neu[r].nodes[["x", "y", "z"]].values for r in outer])
    c = pts.mean(0)
    axis = np.linalg.svd(pts - c, full_matrices=False)[2][0]
    P = np.vstack([presyn[r] for r in outer])
    if np.median((P - c) @ axis) < 0:
        axis = -axis
    s_end = max(((neu[r].nodes[["x", "y", "z"]].values - c) @ axis).max() for r in outer)
    u = np.cross(axis, [0, 0, 1.0]); u /= np.linalg.norm(u); v = np.cross(axis, u)

    def zone(n):
        s = (n.nodes[["x", "y", "z"]].values - c) @ axis
        keep = n.nodes.node_id.values[s > s_end - ZONE]
        return navis.subset_neuron(n, keep, inplace=False) if len(keep) >= 10 else n
    term = {r: zone(neu[r]) for r in outer}
    ct = np.vstack([term[r].nodes[["x", "y", "z"]].values for r in outer]).mean(0)
    dps = navis.NeuronList([navis.make_dotprops(term[r], k=5) for r in outer])
    best = None
    for th in THETAS:
        nrm = np.cos(np.radians(th)) * u + np.sin(np.radians(th)) * v
        refl = navis.NeuronList([navis.make_dotprops(reflect(term[r], ct, nrm), k=5) for r in outer])
        S = navis.nblast(refl, dps, scores="mean", normalized=True, n_cores=1, progress=False).values
        np.fill_diagonal(S, np.nan)
        sc = np.nanmax(S, axis=1).mean()
        if best is None or sc > best[0]:
            best = (sc, th, S, nrm)
    sc, th, S, nrm = best
    that = np.cross(axis, nrm); that /= np.linalg.norm(that)
    bm = {i: int(np.nanargmax(S[i])) for i in range(len(outer))}
    pairs = sorted({tuple(sorted((i, j))) for i, j in bm.items() if bm.get(j) == i})

    def frame(n, origin):
        p = n.nodes[["x", "y", "z"]].values - origin
        return p @ axis, p @ that, p @ nrm
    return dict(c=c, ct=ct, axis=axis, that=that, nrm=nrm, s_end=s_end, term=term, S=S, th=th, score=sc, pairs=pairs,
                frame=frame, P=P)


def extent(term, outer, F):
    rows = []
    for r in outer:
        s, t, d = F["frame"](term[r], F["ct"])
        q = np.c_[t, d]
        rad = np.linalg.norm(q - q.mean(0), axis=1)
        rows.append(dict(cell=r, nodes_um=len(s), p90_radius_um=float(np.percentile(rad, 90)),
                         t_span_um=float(np.percentile(t, 95) - np.percentile(t, 5)),
                         d_span_um=float(np.percentile(d, 95) - np.percentile(d, 5)),
                         s_span_um=float(np.percentile(s, 95) - np.percentile(s, 5))))
    return pd.DataFrame(rows)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    g = int(args[0]) if args else 1
    l1 = int(sys.argv[sys.argv.index("--l1") + 1]) if "--l1" in sys.argv else None
    if "--flywire" in sys.argv:
        src = "FlyWire 783"
        l1_body, bodies, fneu, fpre = fly_cartridge_flywire(l1)
    else:
        src = "male CNS"
        c = np_client(MALE_CNS)
        l1_body, bodies, fneu, fpre = fly_cartridge(c, l1)
    FF = frame_and_pairs(fneu, fpre, bodies)
    R = analyse_group(g, verbose=False)
    outer = R["outer"]
    FA = dict(c=R["c"], ct=R["ct"], axis=R["axis"], that=R["that"], nrm=R["nrm"], s_end=R["s_end"], term=R["term"],
              pairs=R["pairs"], frame=R["frame"], P=R["P"], th=R["th"], score=R["score"])
    ef, ea = extent(FF["term"], bodies, FF), extent(FA["term"], outer, FA)
    ef["dataset"], ea["dataset"] = f"Drosophila {src} (one L1 cartridge)", f"FAAB group {g}"
    tab = pd.concat([ef, ea])
    tab.to_csv(fig(f"pr_fly_vs_faab_group{g}.tsv"), sep="\t", index=False)
    print(tab.groupby("dataset")[["p90_radius_um", "t_span_um", "d_span_um", "s_span_um"]].median().round(1))

    fig, ax = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    cols = ["#7F77DD", "#D85A30", "#1D9E75", "#378ADD", "#BA7517", "#D4537E"]
    for row, (title, F, neu, cells, names) in enumerate((
            (f"Drosophila {src}: six R1-R6 of one lamina cartridge (L1 {l1_body}); plane {FF['th']} deg, mirror {FF['score']:.2f}",
             FF, fneu, bodies, {}),
            (f"FAAB group {g}: six R1-R6 of one ommatidium; plane {FA['th']} deg, mirror {FA['score']:.2f}",
             FA, R["neu"], outer, R["names"]))):
        A, B, C = ax[row]
        for i, r in enumerate(cells):
            s, t, d = F["frame"](neu[r], F["c"])
            lab = names.get(i, "") + f" ...{str(r)[-6:]}"
            A.scatter(s, t, s=2, color=cols[i], label=lab)
            B.scatter(s, d, s=2, color=cols[i])
            st, tt, dt = F["frame"](F["term"][r], F["ct"])
            C.scatter(tt, dt, s=3, color=cols[i], alpha=0.5)
            C.scatter(tt.mean(), dt.mean(), s=140, color=cols[i], edgecolor="k", zorder=5)
        pre = F["P"] - F["c"]
        A.scatter(pre @ F["axis"], pre @ F["that"], s=4, color="k", marker="x", alpha=0.3, label="presynaptic sites")
        for axx in (A, B):
            axx.axvspan(F["s_end"] - ZONE, F["s_end"], color="#EF9F27", alpha=0.15)
            axx.set_xlim(F["s_end"] - 110, F["s_end"] + 10)
            axx.set_ylim(-45, 45)
            axx.set_xlabel("s (um) along the axis, lamina to the right")
        A.set_ylabel("t (um)"); B.set_ylabel("d (um)")
        B.axhline(0, color="#378ADD", lw=1.5)
        C.axhline(0, color="#378ADD", lw=1.5); C.axvline(0, color="#aaa", lw=0.6)
        C.set_xlim(-30, 30); C.set_ylim(-30, 30); C.set_aspect("equal")
        C.set_xlabel("t (um)"); C.set_ylabel("d (um)")
        for i, j in F["pairs"]:
            a = F["frame"](F["term"][cells[i]], F["ct"]); b = F["frame"](F["term"][cells[j]], F["ct"])
            C.plot([a[1].mean(), b[1].mean()], [a[2].mean(), b[2].mean()], "k--", lw=1)
        A.set_title(title, fontsize=10, loc="left")
        B.set_title("edge-on to the mirror plane (same scale)", fontsize=10)
        C.set_title("terminal-zone cross-section, same scale; dashed = mutual mirror pairs", fontsize=10)
        A.legend(fontsize=7, loc="upper left", markerscale=4)
    fig.suptitle("Same frame, same scale: Drosophila lamina cartridge terminals vs FAAB ommatidial group")
    png = fig(f"pr_fly_vs_faab_group{g}.png")
    fig.savefig(png, dpi=120)
    print("saved", png)


if __name__ == "__main__":
    main()
