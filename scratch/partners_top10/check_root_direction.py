"""Verify the lineage direction of the `root_id_new` column in faab_pr_top10_xyz.tsv.

Read-only. For every (cached, resolved) pair recorded in the notes column:
  * creation timestamp of both roots (get_root_timestamps)
  * is_latest_roots at NOW and at the client's default timestamp
  * for a few pairs, the forward lineage explicitly: get_latest_roots(cached, timestamp=now)
CAVE calls are spaced >= 5 s apart.
"""
from faab.config import REPO
import datetime
import os
import re
import time

import numpy as np
import pandas as pd
import caveclient

DATASTACK = "wclee_aedes_brain"
OUT = os.path.join(REPO, "figures")
SRC = os.path.join(OUT, "faab_pr_top10_xyz.tsv")
PAUSE = 5.0


def main():
    df = pd.read_csv(SRC, sep="\t", dtype=str)
    pairs = []
    for _, r in df.iterrows():
        m = re.match(r"root changed; xyz from current root (\d+)", str(r["notes"]))
        if m:
            pairs.append((int(r["downstream"]), int(m.group(1))))
    pairs = sorted(set(pairs))
    print(f"{len(pairs)} distinct (cached, resolved) pairs\n")

    client = caveclient.CAVEclient(DATASTACK)
    now = datetime.datetime.now(datetime.timezone.utc)
    print("client.version      :", getattr(client, "version", None))
    print("client.timestamp    :", getattr(client, "timestamp", None))
    print("chunkedgraph.timestamp:", getattr(client.chunkedgraph, "timestamp", None))
    print("materialize.version :", client.materialize.version)
    try:
        meta = client.materialize.get_version_metadata()
        print("mat version time    :", meta.get("time_stamp"))
    except Exception as e:
        print("version metadata failed:", str(e)[:80])
    print("now (utc)           :", now, "\n")

    ids = sorted({x for p in pairs for x in p})
    time.sleep(PAUSE)
    ts = client.chunkedgraph.get_root_timestamps(ids)
    created = dict(zip(ids, ts))
    time.sleep(PAUSE)
    lat_now = dict(zip(ids, np.asarray(client.chunkedgraph.is_latest_roots(ids, timestamp=now))))
    time.sleep(PAUSE)
    lat_def = dict(zip(ids, np.asarray(client.chunkedgraph.is_latest_roots(ids))))

    def fmt(t):
        return pd.Timestamp(t).strftime("%Y-%m-%d %H:%M")

    print(f"{'cached (downstream)':>20} {'created':>17} {'lat@now':>7} {'lat@def':>7}   "
          f"{'resolved (root_id_new)':>22} {'created':>17} {'lat@now':>7} {'lat@def':>7}  newer?")
    n_newer = 0
    for old, new in pairs:
        newer = created[new] > created[old]
        n_newer += newer
        print(f"{old:>20} {fmt(created[old]):>17} {str(lat_now[old]):>7} {str(lat_def[old]):>7}   "
              f"{new:>22} {fmt(created[new]):>17} {str(lat_now[new]):>7} {str(lat_def[new]):>7}  "
              f"{'resolved is NEWER' if newer else 'resolved is OLDER'}")
    print(f"\nresolved root created AFTER cached root: {n_newer}/{len(pairs)}")
    print(f"cached roots latest at now: {sum(bool(lat_now[o]) for o, _ in pairs)}/{len(pairs)}")
    print(f"resolved roots latest at now: {sum(bool(lat_now[n]) for _, n in pairs)}/{len(pairs)}")

    print("\nExplicit forward lineage, get_latest_roots(cached, timestamp=now):")
    for old, new in pairs[:4]:
        time.sleep(PAUSE)
        fwd = [int(x) for x in np.atleast_1d(client.chunkedgraph.get_latest_roots(old, timestamp=now))]
        print(f"  {old} -> {fwd}   (table had {new})")
    print("DONE")


if __name__ == "__main__":
    main()
