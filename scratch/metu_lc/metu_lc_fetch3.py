"""Re-fetch MeTu4/LC morphology FRESH: resolve each annotation via its immutable
supervoxel_id -> CURRENT root (not the cached, possibly-stale root_id column),
then fetch L2 dotprops. Read-only, throttled."""
from faab.config import REPO
import os
import time
import pickle
import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import caveclient
import seaserpent as ss
from fafbseg import flywire
from faab.sources.cave import DATASTACK, throttle

SERVER = "https://flytable.mrc-lmb.cam.ac.uk/"
OUT = os.path.join(REPO, "figures")


def main():
    meta = ss.Table("aedes_main", server=SERVER).to_frame()
    meta.columns = [str(c) for c in meta.columns]
    sub = meta[meta["subclass"].astype(str).isin(["MeTu4", "LC"])].copy()
    sub = sub[sub["supervoxel_id"].notna() & (sub["supervoxel_id"].astype(str) != "")]
    svid = sub["supervoxel_id"].astype("int64").tolist()
    old_root = sub["root_id"].astype(str).tolist()
    label = sub["subclass"].astype(str).tolist()
    status = sub["status"].astype(str).tolist()
    print(f"MeTu4/LC rows with supervoxel_id: {len(sub)}", flush=True)

    client = caveclient.CAVEclient(DATASTACK)
    t = throttle(time.time())
    cur = client.chunkedgraph.get_roots(svid)               # immutable svid -> CURRENT root
    cur = [int(x) for x in cur]
    nstale = sum(1 for o, c in zip(old_root, cur) if str(c) != o.strip())
    print(f"current root differs from table root_id (stale): {nstale}/{len(cur)}", flush=True)

    lab_by_root = {c: l for c, l in zip(cur, label)}
    stat_by_root = {c: s for c, s in zip(cur, status)}

    t = throttle(t)
    dp = flywire.get_l2_dotprops(cur, dataset=DATASTACK, max_threads=1,
                                 omit_failures=True, progress=True)
    by = {}
    for d in dp:
        d2 = d / 1000.0
        d2.units = "um"
        by[int(d.id)] = d2

    npts = {r: len(by[r].points) if r in by else 0 for r in cur}
    arr = np.array([npts[r] for r in cur])
    print(f"\nfetched dotprops: {len(by)} / {len(cur)}")
    for thr in (5, 20, 50, 100, 300):
        print(f"  cells with >= {thr:3d} points: {(arr >= thr).sum():3d}")
    print("\npoints by status:")
    for st in ("adequate", "needs_extending", "missing soma"):
        v = np.array([npts[r] for r in cur if stat_by_root.get(r) == st])
        if len(v):
            print(f"  {st:16}: n={len(v):3d}  median {int(np.median(v)):4d}  >=50pts {int((v>=50).sum()):3d}")
    print("\npoints by subclass:")
    for lc in ("MeTu4", "LC"):
        v = np.array([npts[r] for r in cur if lab_by_root.get(r) == lc])
        print(f"  {lc:6}: n={len(v):3d}  median {int(np.median(v)):4d}  >=50pts {int((v>=50).sum()):3d}")

    pickle.dump({"dotprops": by, "label_by_root": lab_by_root, "status_by_root": stat_by_root,
                 "svid_by_root": {c: s for c, s in zip(cur, svid)}},
                open(os.path.join(OUT, "metu_lc_dotprops_fresh.pkl"), "wb"))
    print("\nsaved metu_lc_dotprops_fresh.pkl\nDONE", flush=True)


if __name__ == "__main__":
    main()
