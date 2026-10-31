"""Analyze a putative Mi1 set (aedes): (1) do they all match (nblast consistency),
(2) overlapping territories (fragment/duplicate/wrong-type flags), (3) rank by
proofreading completeness. Read-only. Caches dotprops.
"""
from faab.config import REPO
import os
import re
import time
import pickle
import numpy as np
import pandas as pd
import navis
import caveclient
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from faab.clients.cave import throttle, DATASTACK
from faab.config import fig

OUT = os.path.join(REPO, "figures")
CACHE = fig("mi1_dotprops.pkl")

RAW = """
648518347611904256, 648518347550912741, 648518347638273763, 648518347638461063 648518347630364807 648518347759173922,
648518347656410834, 648518347506988091,648518347611903488, 648518347622100158,648518347719671074,648518347668959074, 648518347603345439,648518347642142087, 648518347642139783, 648518347571341087, 48518347626902919,648518347534003537,
648518347642254471, 648518347582537619, 648518347648028039, 648518347642246791,648518347645162887, 648518347582241529,
64851834764224474, 648518347642292103, 648518347677491713, 648518347571764511, 648518347533409858, 648518347533240130, 648518347668744224, 648518347560168770, 648518347677498369, 648518347642142343, 648518347585303667, 648518347497757552,
648518347670743809, 648518347677578753, 648518347464257993,
648518347677579265, 648518347677533441, 648518347677661441,
648518347642244743, 648518347677718273, 648518347641968007, 648518347677535233,648518347642241671, 648518347642252935, 648518347707746338, 648518347571178271, 648518347509498721, 648518347698730391, 648518347578223288, 648518347548084034, 648518347595386080, 648518347495945176, 648518347720246818,
648518347491429742, 648518347582670961, 648518347614848923, 648518347581964103, 648518347509997429, 648518347631253895, 648518347503873779, 648518347543590222, 648518347505032434, 648518347705484834, 648518347520466472, 648518347668425985, 648518347536688531, 648518347694484701, 648518347672633083, 648518347657936129, 648518347604177439, 648518347667454496, 648518347587051807, 648518347553800770, 648518347553815618, 648518347537960105, 648518347530639868, 648518347581964103, 648518347503880435, 648518347694474717, 648518347812465146, 648518347548578370, 648518347520467752
648518347641989255, 648518347812465146, 648518347548578370, 648518347520467752, 648518347488589589, 648518347548702786, 648518347548456770, 648518347576273886, 648518347616831674, 648518347677637633, 648518347495670416, 648518347632575367, 648518347677589505, 648518347632571015, 648518347677775873, 648518347595192485,
648518347542595138, 648518347642060679, 648518347530609660, 648518347492866285, 648518347554054210, 648518347548273986, 648518347642151047
"""


def main():
    tokens = re.findall(r"\d+", RAW)
    ids = sorted({int(t) for t in tokens if len(t) == 18})
    malformed = sorted({t for t in tokens if len(t) != 18})
    print(f"parsed {len(tokens)} tokens -> {len(ids)} unique valid 18-digit ids")
    print(f"MALFORMED (flagged, excluded): {malformed}")

    if os.path.exists(CACHE):
        dp = pickle.load(open(CACHE, "rb"))
        print("loaded cached dotprops")
    else:
        from fafbseg import flywire
        client = caveclient.CAVEclient(DATASTACK)
        t = time.time()
        is_latest = client.chunkedgraph.is_latest_roots(ids)
        print(f"current roots: {int(sum(is_latest))}/{len(ids)} "
              f"(non-current: {[ids[i] for i in range(len(ids)) if not is_latest[i]][:8]}...)")
        t = throttle(t)
        dp = (flywire.get_l2_dotprops(ids, dataset=DATASTACK, max_threads=1,
                                      omit_failures=True, progress=False)) / 1000.0
        for d in dp:
            d.units = "um"
        pickle.dump(dp, open(CACHE, "wb"))
    fetched = {int(d.id): d for d in dp}
    failed = [i for i in ids if i not in fetched]
    print(f"fetched {len(fetched)} dotprops | failed/empty: {len(failed)} {failed[:8]}")

    rids = list(fetched)
    nls = navis.NeuronList([fetched[r] for r in rids])

    # ---- (3) completeness proxies ----
    rows = []
    for r in rids:
        d = fetched[r]
        co = d.points
        span = co.max(0) - co.min(0)
        rows.append({"root_id": r, "n_pts": len(co),
                     "bbox_diag_um": round(float(np.linalg.norm(span)), 1),
                     "span_um": tuple(span.round(0))})
    comp = pd.DataFrame(rows)

    # ---- (1) consistency: center each (remove retinotopic position), then NBLAST all-by-all ----
    cen = navis.NeuronList([navis.make_dotprops(fetched[r].points - fetched[r].points.mean(0), k=5)
                            for r in rids])
    for d in cen:
        d.units = "um"
    cen.set_neuron_attributes([str(r) for r in rids], "name")
    cen.set_neuron_attributes([str(r) for r in rids], "id")
    S = navis.nblast_allbyall(cen, n_cores=4, progress=False)
    Sm = (S.values + S.values.T) / 2
    np.fill_diagonal(Sm, np.nan)
    comp["mean_consistency"] = [np.nanmean(Sm[i]) for i in range(len(rids))]
    comp = comp.sort_values(["bbox_diag_um"], ascending=False).reset_index(drop=True)
    comp.to_csv(fig("mi1_completeness.csv"), index=False)

    print("\n=== completeness + consistency (sorted by size) ===")
    print(comp[["root_id", "n_pts", "bbox_diag_um", "mean_consistency"]].head(30).to_string(index=False))
    print(f"\nmedian bbox_diag={comp['bbox_diag_um'].median():.0f}um  "
          f"median consistency={comp['mean_consistency'].median():.2f}")
    print("low-consistency outliers (mean<0.3, possible wrong-type/fragment):",
          comp[comp["mean_consistency"] < 0.3]["root_id"].tolist()[:15])

    # ---- (2) overlapping territories ----
    trees = {r: cKDTree(fetched[r].points) for r in rids}
    npts = {r: len(fetched[r].points) for r in rids}
    overlaps = []
    for a in range(len(rids)):
        for b in range(a + 1, len(rids)):
            ra, rb = rids[a], rids[b]
            small, big = (ra, rb) if npts[ra] <= npts[rb] else (rb, ra)
            d, _ = trees[big].query(fetched[small].points, distance_upper_bound=3.0)
            frac = np.mean(np.isfinite(d))            # fraction of small within 3um of big
            if frac > 0.5:
                overlaps.append({"small": small, "big": big, "frac_small_in_big": round(float(frac), 2),
                                 "n_small": npts[small], "n_big": npts[big]})
    ov = pd.DataFrame(overlaps).sort_values("frac_small_in_big", ascending=False) if overlaps else pd.DataFrame()
    ov.to_csv(fig("mi1_overlaps.csv"), index=False)
    print(f"\n=== overlapping pairs (small >50% inside big -> fragment/duplicate/wrong-type) : {len(ov)} ===")
    if len(ov):
        print(ov.head(25).to_string(index=False))

    # consistency heatmap
    order = comp["root_id"].tolist()
    idx = [rids.index(r) for r in order]
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(np.nan_to_num(Sm[np.ix_(idx, idx)]), cmap="viridis", vmin=0, vmax=1)
    ax.set_title("Putative aedes Mi1 — centered NBLAST consistency (sorted by size)")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout(); fig.savefig(fig("mi1_consistency.png"), dpi=130)
    print("\nsaved mi1_completeness.csv, mi1_overlaps.csv, mi1_consistency.png")


if __name__ == "__main__":
    main()
