"""Read-only, throttled access to FAAB on CAVE.

CAVE access is strictly read-only (default token = view-only account) and throttled (>= 2 s between top-level calls,
David 2026-10-31, was 5 s; no fan-out). Never call edit methods. Sheet / Spelunker xyz points are 16 x 16 x 45 nm voxels
(faab.config.VOX).

    from faab.clients.cave import DATASTACK, throttle, client, roots_at_xyz, live_outputs
    t = throttle(0)
    roots = roots_at_xyz(["8735, 18856, 6245"]); t = throttle(t)
"""
import datetime
import json
import os
import re
import time
import warnings

import numpy as np

from faab.config import DATASTACK, TMP, VOX

warnings.filterwarnings("ignore")


def throttle(last, gap=2.0):
    """Sleep so >=`gap` seconds elapse since `last`; return a fresh timestamp."""
    dt = time.time() - last
    if dt < gap:
        time.sleep(gap - dt)
    return time.time()


def client(datastack=DATASTACK):
    import caveclient
    return caveclient.CAVEclient(datastack)


def latest_roots(ids, client=None, t=None):
    """Return current root IDs for `ids` (read-only). Throttled."""
    import caveclient
    ids = [int(i) for i in ids]
    client = client or caveclient.CAVEclient(DATASTACK)
    t = t if t is not None else time.time()
    is_latest = client.chunkedgraph.is_latest_roots(ids)
    if all(is_latest):
        return ids, t
    t = throttle(t)
    latest = client.chunkedgraph.get_latest_roots(ids)
    return [int(x) for x in latest], t


def parse_xyz(s):
    """'8735, 18856, 6245' (any separators) -> [x, y, z] voxels, or None"""
    v = re.findall(r"-?\d+(?:\.\d+)?", str(s or ""))
    return [float(x) for x in v] if len(v) == 3 else None


SV_CACHE = os.path.join(TMP, "sv_at_xyz.json")


def supervoxels_at_xyz(points, datastack=DATASTACK):
    """Supervoxel at each sheet point (voxel xyz strings or triples; 0 = none / unreadable point). Supervoxels never
    change with proofreading, so every lookup is kept in tmp/sv_at_xyz.json and only new points are read from the
    segmentation volume (cloudvolume, ~0.5 s per point; not a CAVE server call)."""
    from fafbseg import flywire
    vox = [p if isinstance(p, (list, tuple, np.ndarray)) else parse_xyz(p) for p in points]
    keys = [",".join(f"{float(x):g}" for x in v) if v is not None else None for v in vox]
    cache = json.load(open(SV_CACHE)) if os.path.exists(SV_CACHE) else {}
    new = sorted({k for k in keys if k is not None and k not in cache})
    if new:
        found = flywire.locs_to_supervoxels(np.array([[float(x) for x in k.split(",")] for k in new]) * np.array(VOX),
                                            coordinates="nm", backend="cloudvolume", dataset=datastack)
        cache.update({k: int(sv) for k, sv in zip(new, np.asarray(found).astype(np.int64))})
        tmp = SV_CACHE + ".tmp"
        json.dump(cache, open(tmp, "w"))
        os.replace(tmp, SV_CACHE)
    return [cache[k] if k is not None else 0 for k in keys]


def roots_at_xyz(points, datastack=DATASTACK, client_=None):
    """Current root at each sheet point (voxel xyz strings or triples); 0 where there is no segment. Cached supervoxels
    (supervoxels_at_xyz) + one chunkedgraph get_roots call (the caller throttles)."""
    sv = supervoxels_at_xyz(points, datastack)
    out = [0] * len(sv)
    ok = [i for i, v in enumerate(sv) if v]
    if ok:
        c = client_ or client(datastack)
        roots = np.asarray(c.chunkedgraph.get_roots([sv[i] for i in ok])).astype(np.int64)
        for i, r in zip(ok, roots):
            out[i] = int(r)
    return out


def live_outputs(pre_roots, client, timestamp=None):
    """All synapses whose presynaptic root is in `pre_roots`, live at `timestamp` (default now), self-contacts removed.
    One query. Returns (DataFrame, timestamp); positions are in nm, as the live query returns them."""
    timestamp = timestamp or datetime.datetime.now(datetime.timezone.utc)
    syn = client.materialize.live_query("synapses_v2", timestamp=timestamp,
                                        filter_in_dict={"pre_pt_root_id": [int(r) for r in pre_roots]})
    return syn[syn.pre_pt_root_id != syn.post_pt_root_id], timestamp
