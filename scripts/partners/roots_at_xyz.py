"""Current root at each xyz of a pasted table (root_id_orig, class, root_id_curr, xyz in 16x16x45 voxels).
Reads tmp/roots_at_xyz_in.tsv (tab-separated, no header), looks up the segment at every point live
(fafbseg cloudvolume backend, nm), writes tmp/roots_at_xyz_out.tsv and prints the updated table with
a note where the segment differs from the pasted root_id_curr. Read-only."""
from faab.config import REPO
import datetime
import os

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
from fafbseg import flywire
from faab.sources.cave import DATASTACK
from faab.partners.top10_xyz import VOX

HERE = REPO
TMP = os.path.join(HERE, "..", "tmp")


def main():
    df = pd.read_csv(os.path.join(TMP, "roots_at_xyz_in.tsv"), sep="\t", header=None, dtype=str,
                     names=["root_id_orig", "class", "root_id_curr", "xyz"]).fillna("")
    locs = np.array([[float(v) for v in x.split(",")] for x in df.xyz]) * VOX
    now = datetime.datetime.now(datetime.timezone.utc)
    roots = flywire.locs_to_segments(locs, coordinates="nm", backend="cloudvolume", dataset=DATASTACK)
    df["segment_at_xyz"] = [str(int(r)) for r in roots]
    df["note"] = ["" if s == c else ("no segment at xyz" if s == "0" else f"changed from {c}")
                  for s, c in zip(df.segment_at_xyz, df.root_id_curr)]
    df.to_csv(os.path.join(TMP, "roots_at_xyz_out.tsv"), sep="\t", index=False)
    print(f"looked up at {now:%Y-%m-%d %H:%M} UTC")
    print("root_id_orig\tclass\troot_id_curr\txyz\tnote")
    for r in df.itertuples():
        print(f"{r.root_id_orig}\t{r._2}\t{r.segment_at_xyz}\t{r.xyz}\t{r.note}")
    print("DONE")


if __name__ == "__main__":
    main()
