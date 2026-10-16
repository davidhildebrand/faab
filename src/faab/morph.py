"""Morphology fetches for NBLAST / alignment work (read-only; CAVE helpers live in faab.clients.cave)."""
import time

import caveclient
from fafbseg import flywire

from faab.clients.cave import DATASTACK, latest_roots, throttle


def fetch_dotprops_um(ids, datastack=DATASTACK, update_roots=True, max_threads=1):
    """Read-only fetch of L2 dotprops, returned in MICRONS.

    Only CAVE access here; everything downstream (NBLAST, alignment) is local.
    `max_threads=1` to avoid fanning out. L2 cache coords are nm -> /1000 = um.
    """
    client = caveclient.CAVEclient(datastack)
    t = time.time()
    roots = [int(i) for i in ids]
    if update_roots:
        roots, t = latest_roots(roots, client=client, t=t)
    t = throttle(t)
    dp = flywire.get_l2_dotprops(roots, dataset=datastack,
                                 max_threads=max_threads, progress=False)
    return (dp / 1000.0), roots
