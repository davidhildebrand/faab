"""Read-only test of nblast_align on a user-supplied set of aedes root IDs.

Only the data fetch touches CAVE; everything after is local CPU.
CAVE access: read-only default token; max_threads=1 (no fan-out); top-level
CAVE calls spaced >=5 s to respect the rate-limit rule.
"""
import time
import warnings
import numpy as np
import caveclient
import navis
from fafbseg import flywire

warnings.filterwarnings("ignore")

DATASTACK = "wclee_aedes_brain"
IDS = [648518347611904256, 648518347550912741, 648518347638273763]


def throttle(last, gap=5.0):
    """Sleep so >=gap seconds elapse since `last`; return new timestamp."""
    dt = time.time() - last
    if dt < gap:
        time.sleep(gap - dt)
    return time.time()


def main():
    client = caveclient.CAVEclient(DATASTACK)  # read-only default token
    print("datastack:", DATASTACK, "| mat version:", client.materialize.most_recent_version())

    t = time.time()
    # 1) are the supplied IDs current?
    is_latest = client.chunkedgraph.is_latest_roots(IDS)
    print("\nsupplied IDs latest?", dict(zip(IDS, [bool(x) for x in is_latest])))

    roots = list(IDS)
    if not all(is_latest):
        t = throttle(t)
        latest = client.chunkedgraph.get_latest_roots(IDS)
        print("updated to latest roots:", [int(x) for x in latest])
        roots = [int(x) for x in latest]

    # 2) fetch L2 dotprops from the aedes datastack (the only heavy CAVE op)
    t = throttle(t)
    print("\nfetching L2 dotprops (max_threads=1) ...")
    dp = flywire.get_l2_dotprops(roots, dataset=DATASTACK, max_threads=1, progress=False)
    print("got", len(dp), "dotprops")

    # 3) units / scale diagnostics -> convert to microns
    for d in dp:
        co = d.points
        span = co.max(0) - co.min(0)
        print(f"  id {int(d.id)}: pts={len(co):5d}  units={getattr(d,'units',None)}  raw span(={span.round(0)})")
    dp_um = dp / 1000.0  # L2 cache rep_coords are in nm -> microns
    spans_um = [ (d.points.max(0)-d.points.min(0)) for d in dp_um ]
    print("micron spans:", [s.round(1).tolist() for s in spans_um])

    # 4) plain NBLAST (pose-dependent) vs align-then-NBLAST (pose-invariant)
    print("\n--- plain NBLAST (no alignment) ---")
    nb = navis.nblast_allbyall(dp_um, n_cores=1, progress=False)
    print(nb.round(3))

    print("\n--- nblast_align (rigid, two-way) ---")
    al = navis.nblast_align(dp_um, dp_um, align_method="rigid",
                            two_way_align=True, n_cores=1, progress=False)
    print(al.round(3))

    # mean off-diagonal as a quick "how similar overall" readout
    def offdiag_mean(m):
        a = m.values.copy(); np.fill_diagonal(a, np.nan)
        return np.nanmean(a)
    print(f"\nmean off-diagonal similarity:  plain={offdiag_mean(nb):.3f}   aligned={offdiag_mean(al):.3f}")


if __name__ == "__main__":
    main()
