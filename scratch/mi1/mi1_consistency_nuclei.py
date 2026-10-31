"""Mi1 consistency among the soma-bearing Mi1: nblast_align all-by-all, rank the
most-similar cells, gallery of their morphologies, and a neuroglancer link
(top cells + brain mesh). Dataset: aedes. Read-only."""
from faab.config import REPO
import os
import time
import json
import pickle
import urllib.parse
import numpy as np
import pandas as pd
import caveclient
import navis
from navis.transforms.align import align_rigid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from faab.clients.cave import DATASTACK
from faab.config import fig

OUT = os.path.join(REPO, "figures")


def ng_link(segids):
    brain = {"type": "segmentation",
             "source": {"url": "precomputed://gs://aedes_whole-brain/meshes",
                        "transform": {"matrix": [[0.12187, 0.99254, 0, 0], [-0.99254, 0.12187, 0, 30600], [0, 0, 1, 0]],
                                      "outputDimensions": {"x": [1.6e-08, "m"], "y": [1.6e-08, "m"], "z": [4.5e-08, "m"]}}},
             "segments": ["1"], "objectAlpha": 0.12, "segmentDefaultColor": "#ffffff", "name": "brain mesh"}
    seg = "graphene://middleauth+https://cave.fanc-fly.com/segmentation/table/wclee_aedes_brain"
    state = {"layers": [
        {"type": "image", "source": "precomputed://gs://zetta_lee_mosquito_001_alignment_central/img/v2_sharded", "name": "img"},
        brain,
        {"type": "segmentation", "source": seg, "segments": [str(s) for s in segids], "name": "Mi1 top"}],
        "showDefaultAnnotations": False, "layout": "3d"}
    return "https://spelunker.cave-explorer.org/#!" + urllib.parse.quote(json.dumps(state, separators=(",", ":")))


def main():
    by = {int(d.id): d for d in pickle.load(open(fig("mi1_dotprops.pkl"), "rb"))}
    mi1 = list(by)
    client = caveclient.CAVEclient(DATASTACK)
    ncache = fig("nuclei_cache.pkl")
    if os.path.exists(ncache):
        nuc = pickle.load(open(ncache, "rb"))
    else:
        for a in range(15):                       # pre-warm the flaky metadata endpoint
            try:
                client.materialize.get_table_metadata("nuclei_v1_aedes"); break
            except Exception:
                time.sleep(8)
        for a in range(8):
            try:
                nuc = client.materialize.query_table("nuclei_v1_aedes"); break
            except Exception:
                if a == 7:
                    raise
                print(f"  (retry {a+1}/8)"); time.sleep(8)
        pickle.dump(nuc, open(ncache, "wb"))
    soma = set(int(x) for x in nuc["pt_root_id"] if pd.notna(x))
    mi1n = [r for r in mi1 if r in soma]
    print(f"soma-bearing Mi1: {len(mi1n)}")

    mcache = fig("mi1_align_matrix.pkl")
    if os.path.exists(mcache):
        M = pickle.load(open(mcache, "rb"))
    else:
        nls = navis.NeuronList([by[r] for r in mi1n])
        nls.set_neuron_attributes([str(r) for r in mi1n], "name")
        nls.set_neuron_attributes([str(r) for r in mi1n], "id")
        S = navis.nblast_align(nls, nls, align_method="rigid", two_way_align=False,
                               n_cores=6, progress=False, align_kwargs={"scale": True})
        M = (S.values + S.values.T) / 2
        np.fill_diagonal(M, np.nan)
        pickle.dump(M, open(mcache, "wb"))
    mean_sim = np.nanmean(M, axis=1)
    order = list(np.argsort(-mean_sim))
    print(f"\nnblast_align consistency (soma-bearing Mi1): median pairwise "
          f"{np.nanmedian(M):.3f}, range {np.nanmin(M):.2f}-{np.nanmax(M):.2f}")
    print("\n=== top 10 most self-consistent (canonical) Mi1 ===")
    print(f"{'root_id':>18}  mean_sim")
    for i in order[:10]:
        print(f"{mi1n[i]:>18}  {mean_sim[i]:.3f}")
    print("\nleast consistent (possible different-type / worst-traced):")
    for i in order[-3:]:
        print(f"{mi1n[i]:>18}  {mean_sim[i]:.3f}")

    # gallery: align all to the medoid, sorted by consistency
    ref = by[mi1n[order[0]]]
    fig, axes = plt.subplots(5, 8, figsize=(20, 13))
    for ax in axes.flat:
        ax.axis("off")
    for k, i in enumerate(order):
        if k >= 40:
            break
        r = mi1n[i]
        al = align_rigid(navis.NeuronList(by[r]), target=ref, progress=False)
        n = (al[0] if isinstance(al, tuple) else al)[0]
        ax = axes.flat[k]
        norm = (mean_sim[i] - np.nanmin(mean_sim)) / (np.nanmax(mean_sim) - np.nanmin(mean_sim))
        col = plt.cm.plasma(0.10 + 0.62 * norm)     # dark purple(low) -> orange(high); both clear on white
        navis.plot2d(n, ax=ax, method="2d", view=("x", "y"), color=col, lw=1.2)
        ax.set_title(f"{str(r)[-6:]}  {mean_sim[i]:.2f}", fontsize=10)
        ax.set_aspect("equal")
    fig.suptitle("Soma-bearing Mi1 — nblast_align-aligned to medoid, sorted by consistency "
                 "(color = mean similarity: dark purple = low, orange = high)", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(fig("mi1_gallery.png"), dpi=200)

    top10 = [mi1n[order[i]] for i in range(min(10, len(mi1n)))]
    url = ng_link(top10)
    open(fig("mi1_top_ng.txt"), "w").write(url + "\n")
    print(f"\ntop-10 root_ids: {top10}")
    print("saved mi1_gallery.png + mi1_top_ng.txt")


if __name__ == "__main__":
    main()
