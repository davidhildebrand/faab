"""Within-ommatidium NBLAST for the 8 photoreceptor groups, mirroring allowed.

For every pair of cells in a group:
  direct = NBLAST after rigid (rotation + translation) alignment of cell b onto cell a
  mirror = the same after first reflecting b (x -> -x); reflection + free rotation covers
           every mirror image, so this asks "is b the mirror image of a?"
Scores are mean-of-both-directions, normalised NBLAST on 1-um-resampled L2 skeletons.
Mutual best mirror partners among R1-R6 are reported per group. Local only (cached
skeletons in figures/pr_partner_skels.pkl; any missing root is fetched, throttled).
Outputs: figures/faab_pr_groups_nblast.tsv, figures/faab_pr_groups_nblast_pairs.txt,
         figures/faab_pr_groups_nblast.png
"""
from faab.config import REPO
import os
import pickle
import time
import warnings

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import navis

warnings.filterwarnings("ignore")
navis.set_pbars(hide=True)
HERE = REPO
OUT = os.path.join(HERE, "figures")
SCACHE = os.path.join(OUT, "pr_partner_skels.pkl")
G1 = os.path.join(OUT, "faab_pr_current_roots.tsv")       # user's group 1 (original 8)
G28 = os.path.join(OUT, "faab_pr_groups_xyz.tsv")          # user's groups 2-8
RESAMPLE_NM = 1000


def load_groups():
    groups = {}
    g1 = pd.read_csv(G1, sep="\t", dtype=str)
    groups[1] = [(r["root_id_curr"], r["class"]) for _, r in g1.iterrows()]
    g = pd.read_csv(G28, sep="\t", dtype=str)
    for k, sub in g.groupby("group", sort=False):
        groups[int(k) + 1] = [(r["root_id_curr"], r["class"]) for _, r in sub.iterrows()]
    return groups


def prep(sk):
    n = sk.copy()
    n = navis.resample_skeleton(n, resample_to=RESAMPLE_NM, inplace=False)
    n = n / 1000.0                       # nm -> um
    n.units = "um"
    return n


def mirrored(n):
    """Reflect through the plane x = centroid_x, so the copy stays in place and the rigid
    registration only has to find the rotation (reflection + any rotation = any mirror)."""
    m = n.copy()
    cx = float(m.nodes["x"].mean())
    m.nodes["x"] = 2 * cx - m.nodes["x"]
    return m


def aligned_score(a, b):
    """NBLAST(a, b) after rigid alignment of b onto a (mean of both directions)."""
    try:
        b_al = navis.align_rigid(b, a, scale=False, progress=False)[0][0]
    except Exception:
        b_al = b
    da = navis.make_dotprops(a, k=5)
    db = navis.make_dotprops(b_al, k=5)
    s = navis.nblast(da, db, scores="mean", normalized=True, n_cores=1, progress=False)
    return float(s.values[0, 0])


def main():
    t0 = time.time()
    groups = load_groups()
    sk_by = pickle.load(open(SCACHE, "rb"))
    need = sorted({int(r) for cells in groups.values() for r, _ in cells if int(r) not in sk_by})
    if need:
        from fafbseg import flywire
        from faab.clients.cave import DATASTACK
        time.sleep(5)
        for s in flywire.get_l2_skeleton(need, dataset=DATASTACK, max_threads=1,
                                         omit_failures=True, progress=False):
            sk_by[int(s.id)] = s
        pickle.dump(sk_by, open(SCACHE, "wb"))
    print(f"{sum(len(v) for v in groups.values())} cells in {len(groups)} groups; "
          f"skeletons missing: {len(need)}", flush=True)

    rows, mats, pair_lines = [], {}, []
    for g, cells in groups.items():
        ids = [r for r, _ in cells]
        cls = {r: c for r, c in cells}
        neu = {}
        for r in ids:
            sk = sk_by.get(int(r))
            if sk is None or sk.n_nodes < 5:
                continue
            neu[r] = prep(sk)
        n = len(ids)
        D = np.full((n, n), np.nan)
        M = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = neu.get(ids[i]), neu.get(ids[j])
                if a is None or b is None:
                    continue
                d = aligned_score(a, b)
                m = aligned_score(a, mirrored(b))
                D[i, j] = D[j, i] = d
                M[i, j] = M[j, i] = m
                rows.append([g, ids[i], cls[ids[i]], ids[j], cls[ids[j]], round(d, 3), round(m, 3)])
        mats[g] = (ids, cls, D, M)

        # mutual best mirror partners among R1-R6
        outer = [k for k, r in enumerate(ids) if cls[r].startswith("R1")]
        best = {}
        for k in outer:
            cand = [(M[k, l], l) for l in outer if l != k and not np.isnan(M[k, l])]
            if cand:
                best[k] = max(cand)[1]
        mutual = sorted({tuple(sorted((k, l))) for k, l in best.items() if best.get(l) == k})
        pair_lines.append(f"group {g}: mutual best mirror pairs among R1-R6: "
                          f"{len(mutual)} of 3 possible")
        for k, l in mutual:
            others_m = [M[k, o] for o in outer if o not in (k, l)]
            pair_lines.append(
                f"   {ids[k]} <-> {ids[l]}   mirror {M[k, l]:.3f}  direct {D[k, l]:.3f}  "
                f"next-best mirror for these cells {max(others_m):.3f}")
        print(pair_lines[-1 - len(mutual)], flush=True)
        for ln in pair_lines[-len(mutual):] if mutual else []:
            print(ln, flush=True)
        print(f"   ({time.time() - t0:.0f} s elapsed)", flush=True)

    df = pd.DataFrame(rows, columns=["group", "cell_a", "class_a", "cell_b", "class_b",
                                     "nblast_direct", "nblast_mirror"])
    df.to_csv(os.path.join(OUT, "faab_pr_groups_nblast.tsv"), sep="\t", index=False)
    with open(os.path.join(OUT, "faab_pr_groups_nblast_pairs.txt"), "w") as f:
        f.write("\n".join(pair_lines) + "\n")

    # summary across groups
    print("\nSUMMARY (R1-R6 pairs only)")
    r16 = df[df.class_a.str.startswith("R1") & df.class_b.str.startswith("R1")]
    print(f"  pairs: {len(r16)};  mirror > direct in {(r16.nblast_mirror > r16.nblast_direct).mean()*100:.0f}% of pairs")
    print(f"  median direct {r16.nblast_direct.median():.3f}, median mirror {r16.nblast_mirror.median():.3f}")
    print("\n" + "\n".join(pair_lines))

    # heatmaps of mirror scores, R1-R6 only
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for ax, (g, (ids, cls, D, M)) in zip(axes.ravel(), mats.items()):
            outer = [k for k, r in enumerate(ids) if cls[r].startswith("R1")]
            sub = M[np.ix_(outer, outer)]
            im = ax.imshow(sub, vmin=0, vmax=max(0.5, np.nanmax(sub)), cmap="viridis")
            ax.set_xticks(range(len(outer)))
            ax.set_yticks(range(len(outer)))
            ax.set_xticklabels([ids[k][-5:] for k in outer], rotation=90, fontsize=8)
            ax.set_yticklabels([ids[k][-5:] for k in outer], fontsize=8)
            ax.set_title(f"group {g}: mirror NBLAST (R1-R6)", fontsize=10)
            for i in range(len(outer)):
                for j in range(len(outer)):
                    if i != j and not np.isnan(sub[i, j]):
                        ax.text(j, i, f"{sub[i, j]:.2f}", ha="center", va="center", fontsize=7,
                                color="w" if sub[i, j] < 0.35 else "k")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="NBLAST after reflection + rigid alignment")
        fig.savefig(os.path.join(OUT, "faab_pr_groups_nblast.png"), dpi=130, bbox_inches="tight")
        print("\nsaved heatmaps")
    except Exception as e:
        print("heatmap failed:", str(e)[:80])
    print(f"DONE in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
