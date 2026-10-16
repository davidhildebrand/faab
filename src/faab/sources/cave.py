"""Read-only, throttled access to FAAB on CAVE.

CAVE access is strictly read-only (default token = view-only account) and throttled (>= 5 s between top-level calls,
no fan-out). Never call edit methods. Sheet / Spelunker xyz points are 16 x 16 x 45 nm voxels (faab.config.VOX).

    from faab.sources.cave import DATASTACK, throttle, client, roots_at_xyz, live_outputs
    t = throttle(0)
    roots = roots_at_xyz(["8735, 18856, 6245"]); t = throttle(t)
"""
import datetime
import re
import time
import warnings

import numpy as np

from faab.config import DATASTACK, VOX

warnings.filterwarnings("ignore")


def throttle(last, gap=5.0):
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


def roots_at_xyz(points, datastack=DATASTACK):
    """Current root at each sheet point (voxel xyz strings or triples), in one lookup; 0 where there is no segment."""
    from fafbseg import flywire
    vox = [p if isinstance(p, (list, tuple, np.ndarray)) else parse_xyz(p) for p in points]
    ok = [i for i, v in enumerate(vox) if v is not None]
    out = [0] * len(vox)
    if ok:
        found = flywire.locs_to_segments(np.array([vox[i] for i in ok], float) * np.array(VOX), coordinates="nm",
                                         backend="cloudvolume", dataset=datastack)
        for i, r in zip(ok, np.asarray(found).astype(np.int64)):
            out[i] = int(r)
    return out


def live_outputs(pre_roots, client, timestamp=None):
    """All synapses whose presynaptic root is in `pre_roots`, live at `timestamp` (default now), self-contacts removed.
    One query. Returns (DataFrame, timestamp); positions are in nm, as the live query returns them."""
    timestamp = timestamp or datetime.datetime.now(datetime.timezone.utc)
    syn = client.materialize.live_query("synapses_v2", timestamp=timestamp,
                                        filter_in_dict={"pre_pt_root_id": [int(r) for r in pre_roots]})
    return syn[syn.pre_pt_root_id != syn.post_pt_root_id], timestamp
