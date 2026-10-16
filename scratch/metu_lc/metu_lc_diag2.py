"""Are the MeTu4/LC cells genuine stubs or real cells missing from the L2 cache?
Check L2 leaf counts (segmentation extent) for a sample. Read-only, throttled."""
from faab.config import REPO
import os
import time
import pickle
import numpy as np
import caveclient
from faab.sources.cave import DATASTACK, throttle

OUT = os.path.join(REPO, "figures")


def main():
    diag = pickle.load(open(os.path.join(OUT, "metu_lc_diag.pkl"), "rb"))
    npts = diag["npts"]; latest = diag["latest_ids"]; stale = diag["stale_ids"]
    D = pickle.load(open(os.path.join(OUT, "metu_lc_dotprops.pkl"), "rb"))
    lab = D["label_by_root"]

    good = [r for r in npts if npts[r] >= 50]                 # the 2 with morphology
    latest_empty = [r for r in latest if npts.get(r, 0) < 5][:8]
    stale_empty = [r for r in stale if npts.get(r, 0) < 5][:5]
    sample = good + latest_empty + stale_empty
    print(f"sampling {len(sample)} cells (good {len(good)}, latest-empty {len(latest_empty)}, "
          f"stale {len(stale_empty)})\n")

    client = caveclient.CAVEclient(DATASTACK)
    t = time.time()
    print(f"{'root_id':>19} {'label':>6} {'latest':>6} {'dp_pts':>7} {'L2_nodes':>9}")
    for r in sample:
        t = throttle(t)
        try:
            l2 = client.chunkedgraph.get_leaves(int(r), stop_layer=2)
            n2 = len(l2)
        except Exception as e:
            n2 = f"ERR:{str(e)[:20]}"
        isl = r in set(latest)
        print(f"{r:>19} {lab.get(r,'?'):>6} {str(isl):>6} {npts.get(r,0):>7} {str(n2):>9}", flush=True)
    print("\nkey: many L2 nodes + 0 dp_pts -> real cell, L2 CACHE missing (use meshes);")
    print("     1-2 L2 nodes -> genuine stub (needs tracing before morphology works).")
    print("DONE")


if __name__ == "__main__":
    main()
