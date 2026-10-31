"""Diagnose why MeTu4/LC dotprops are near-empty: stale roots vs genuine stubs.
Read-only, one chunkedgraph call."""
from faab.config import REPO
import os
import pickle
import numpy as np
import caveclient
from faab.clients.cave import DATASTACK
from faab.config import fig

OUT = os.path.join(REPO, "figures")


def main():
    D = pickle.load(open(fig("metu_lc_dotprops.pkl"), "rb"))
    by = D["dotprops"]; lab = D["label_by_root"]
    ann_ids = [int(i) for i in lab]                       # all 165 annotated roots
    print(f"annotated ids: {len(ann_ids)} | dotprops fetched: {len(by)}")

    npts = {r: len(by[r].points) if r in by else 0 for r in ann_ids}
    arr = np.array(list(npts.values()))
    for thr in (1, 5, 20, 50, 100):
        print(f"  cells with >= {thr:3d} points: {(arr >= thr).sum():3d} / {len(arr)}")

    client = caveclient.CAVEclient(DATASTACK)
    isl = np.asarray(client.chunkedgraph.is_latest_roots(ann_ids))
    print(f"\nroot currency: latest {int(isl.sum())} / {len(ann_ids)}  |  stale {int((~isl).sum())}")

    latest_ids = [r for r, ok in zip(ann_ids, isl) if ok]
    stale_ids = [r for r, ok in zip(ann_ids, isl) if not ok]
    lp = np.array([npts[r] for r in latest_ids]) if latest_ids else np.array([0])
    sp = np.array([npts[r] for r in stale_ids]) if stale_ids else np.array([0])
    print(f"  LATEST roots: n={len(latest_ids)}, points median {int(np.median(lp))}, "
          f">=50pts: {(lp>=50).sum()}")
    print(f"  STALE  roots: n={len(stale_ids)}, points median {int(np.median(sp))}, "
          f">=50pts: {(sp>=50).sum()}")
    print("\ninterpretation: if LATEST roots also have ~1 pt -> genuine stubs (not proofread);")
    print("if STALE dominate -> need robust root update then refetch.")
    pickle.dump({"latest_ids": latest_ids, "stale_ids": stale_ids, "npts": npts},
                open(fig("metu_lc_diag.pkl"), "wb"))
    print("DONE")


if __name__ == "__main__":
    main()
