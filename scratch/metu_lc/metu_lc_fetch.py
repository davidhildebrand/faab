"""Fetch L2 dotprops (um) for the MeTu4 + LC cells and cache them for the
MeTu4-vs-LC nblast_align re-check. Read-only, throttled CAVE."""
from faab.config import REPO
import os
import time
import pickle
import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
from fafbseg import flywire
from faab.clients.cave import DATASTACK, latest_roots, throttle

OUT = os.path.join(REPO, "figures")


def main():
    metu4 = pickle.load(open(os.path.join(OUT, "ids_MeTu4.pkl"), "rb"))
    lc = pickle.load(open(os.path.join(OUT, "ids_LC.pkl"), "rb"))
    ids = [(r, "MeTu4") for r in metu4] + [(r, "LC") for r in lc]
    print(f"MeTu4 {len(metu4)}  LC {len(lc)}  total {len(ids)}", flush=True)

    raw = [int(r) for r, _ in ids]
    label_by_root = {int(r): lab for r, lab in ids}   # label by the annotated root id

    # Skip the atomic get_latest_roots update (it 500s if ANY id is stale/invalid);
    # get_l2_dotprops(omit_failures=True) just skips bad/stale ids instead.
    t = throttle(time.time())
    dp = flywire.get_l2_dotprops(raw, dataset=DATASTACK, max_threads=1,
                                 omit_failures=True, progress=True)
    by = {}
    for d in dp:
        d2 = d / 1000.0
        d2.units = "um"
        by[int(d.id)] = d2
    got = {"MeTu4": 0, "LC": 0}
    for r in by:
        got[label_by_root.get(r, "?")] = got.get(label_by_root.get(r, "?"), 0) + 1
    npts = [len(d.points) for d in by.values()]
    print(f"fetched {len(by)} / {len(ids)}  (MeTu4 {got['MeTu4']}, LC {got['LC']})", flush=True)
    print(f"dotprops point counts: min {min(npts)}, median {int(np.median(npts))}, max {max(npts)}", flush=True)

    pickle.dump({"dotprops": by, "label_by_root": label_by_root},
                open(os.path.join(OUT, "metu_lc_dotprops.pkl"), "wb"))
    print("saved metu_lc_dotprops.pkl\nDONE", flush=True)


if __name__ == "__main__":
    main()
