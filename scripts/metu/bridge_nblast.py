"""Do the PR -> MeTu bridge cells share one morphology? Align-then-NBLAST on the strong bridges.

Cells: bridges with >= MIN_SYN synapses on both legs (figures/pr_metu_bridge.tsv) plus, as a reference type,
the SeaTable-labelled L1-L5 cells among the weak bridges. One read-only CAVE fetch of L2 dotprops (um), then
local: plain NBLAST all-by-all and rigid-aligned NBLAST (navis.nblast_align, pycpd), both normalised mean
scores; average-linkage clustering of 1 - aligned score; clusters cut at CUT. Outputs:
figures/bridge_nblast_cells.tsv (per cell: PR input class, size, cluster, mean score to cluster mates),
figures/bridge_nblast_aligned.tsv / _plain.tsv (matrices), figures/bridge_nblast.png (clustered heatmap),
figures/bridge_dotprops.pkl (cache). Rate limit respected (single fetch, max_threads=1).
"""
from faab.config import REPO
import os
import pickle
import sys
import time
import warnings

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import navis
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform

from faab.morph import fetch_dotprops_um

warnings.filterwarnings("ignore")
navis.set_pbars(hide=True)
HERE = REPO
OUT = os.path.join(HERE, "figures")
MIN_SYN = int(sys.argv[1]) if len(sys.argv) > 1 else 3
CUT = 0.5  # cluster cut on aligned similarity (distance 1 - score)


def main():
    t0 = time.time()
    bridge = pd.read_csv(os.path.join(OUT, "pr_metu_bridge.tsv"), sep="\t", dtype={"root": str})
    strong = bridge[bridge.path_strength >= MIN_SYN].copy()
    lmc = bridge[bridge.flytable.astype(str).str.contains("L1-L5") & (bridge.path_strength < MIN_SYN)].copy()
    cells = pd.concat([strong.assign(role="bridge"), lmc.assign(role="LMC reference")]).drop_duplicates("root")
    cells["pr_input"] = np.where(cells.syn_from_R7R8 >= cells.syn_from_R1R6, "R7/R8", "R1-R6")
    cells.loc[(cells.syn_from_R7R8 > 0) & (cells.syn_from_R1R6 > 0), "pr_input"] = "mixed"
    print(f"{len(strong)} bridges (>= {MIN_SYN} syn both legs) + {len(lmc)} LMC references", flush=True)

    cache = os.path.join(OUT, "bridge_dotprops.pkl")
    dps = pickle.load(open(cache, "rb")) if os.path.exists(cache) else {}
    need = [r for r in cells.root if r not in dps]
    if need:
        dp, roots = fetch_dotprops_um(need, update_roots=False)
        for n in dp:
            n.units = "um"
            dps[str(n.id)] = n
        pickle.dump(dps, open(cache, "wb"))
    nl = navis.NeuronList([dps[r] for r in cells.root if r in dps])
    ids = [str(n.id) for n in nl]
    cells = cells.set_index("root").loc[ids].reset_index()
    cells["n_points"] = [len(n.points) for n in nl]
    cells["extent_um"] = [round(float(np.linalg.norm(n.points.max(0) - n.points.min(0))), 1) for n in nl]
    print(f"dotprops: {len(nl)} cells, points {min(cells.n_points)}-{max(cells.n_points)}; fetch+prep {time.time() - t0:.0f} s", flush=True)

    plain = navis.nblast_allbyall(nl, normalized=True, n_cores=1, progress=False)
    plain = (plain + plain.T) / 2
    print(f"plain NBLAST done {time.time() - t0:.0f} s", flush=True)
    al = navis.nblast_align(nl, nl, align_method="rigid", normalized=True, scores="mean", n_cores=1, progress=False)
    al = pd.DataFrame(np.asarray(al, float), index=ids, columns=ids)
    al = (al + al.T) / 2
    np.fill_diagonal(al.values, 1.0)
    print(f"aligned NBLAST done {time.time() - t0:.0f} s", flush=True)
    plain.index = plain.columns = ids
    plain.round(3).to_csv(os.path.join(OUT, "bridge_nblast_plain.tsv"), sep="\t")
    al.round(3).to_csv(os.path.join(OUT, "bridge_nblast_aligned.tsv"), sep="\t")

    D = 1 - al.values
    D = np.clip((D + D.T) / 2, 0, None)
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method="average")
    cl = fcluster(Z, t=1 - CUT, criterion="distance")
    cells["cluster"] = cl
    off = ~np.eye(len(ids), dtype=bool)
    cells["mean_aligned_to_all"] = [round(float(al.values[i][off[i]].mean()), 3) for i in range(len(ids))]
    cells["mean_plain_to_all"] = [round(float(plain.values[i][off[i]].mean()), 3) for i in range(len(ids))]
    mates = []
    for i, c in enumerate(cl):
        idx = [j for j in range(len(ids)) if cl[j] == c and j != i]
        mates.append(round(float(al.values[i, idx].mean()), 3) if idx else np.nan)
    cells["mean_aligned_to_cluster"] = mates
    cells = cells.sort_values(["cluster", "mean_aligned_to_cluster"], ascending=[True, False])
    keep = ["root", "role", "pr_input", "syn_from_R7R8", "syn_from_R1R6", "syn_to_MeTu", "n_MeTu", "n_points", "extent_um",
            "cluster", "mean_aligned_to_cluster", "mean_aligned_to_all", "mean_plain_to_all", "flytable"]
    cells[keep].to_csv(os.path.join(OUT, "bridge_nblast_cells.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    print("\nper cell:")
    print(cells[keep].to_string(index=False))
    print(f"\nclusters at aligned similarity >= {CUT}: {len(set(cl))}; sizes {pd.Series(cl).value_counts().sort_index().to_dict()}")
    print(f"aligned score, all off-diagonal pairs: median {np.median(al.values[off]):.2f}, "
          f"IQR {np.percentile(al.values[off], 25):.2f}-{np.percentile(al.values[off], 75):.2f}; "
          f"plain: median {np.median(plain.values[off]):.2f}")
    for c in sorted(set(cl)):
        idx = [i for i in range(len(ids)) if cl[i] == c]
        if len(idx) > 1:
            sub = al.values[np.ix_(idx, idx)]
            print(f"  cluster {c}: n={len(idx)}, within-cluster aligned median {np.median(sub[~np.eye(len(idx), dtype=bool)]):.2f}, "
                  f"members {[ids[i] for i in idx]}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = leaves_list(Z)
        lab = [f"{ids[i][-6:]} {cells.set_index('root').loc[ids[i], 'pr_input']} c{cl[i]}" for i in order]
        fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
        for ax, M, title in ((axes[0], plain.values, "plain NBLAST (in situ)"), (axes[1], al.values, "rigid-aligned NBLAST")):
            im = ax.imshow(M[np.ix_(order, order)], vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(range(len(order)))
            ax.set_yticks(range(len(order)))
            ax.set_xticklabels(lab, rotation=90, fontsize=7)
            ax.set_yticklabels(lab, fontsize=7)
            ax.set_title(title)
        fig.colorbar(im, ax=axes, fraction=0.02, label="normalised mean NBLAST")
        fig.suptitle(f"PR->MeTu bridge cells (>= {MIN_SYN} syn both legs) + LMC references; clusters cut at {CUT}")
        fig.savefig(os.path.join(OUT, "bridge_nblast.png"), dpi=130, bbox_inches="tight")
        print("saved figures/bridge_nblast.png")
    except Exception as e:
        print("figure failed:", str(e)[:100])
    print(f"DONE in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
