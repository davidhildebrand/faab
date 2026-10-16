"""Patch the one split case in figures/faab_pr_groups_xyz.tsv (group 2 R8,
root_id_orig 648518347571683763): use the R8-like piece 648518347700929678 (51 output
synapses, in the photoreceptor axon region) instead of the synapse-richest piece
648518347504455355 (264 output synapses, looks like a merged non-photoreceptor cell), flag
it, and re-print the seven blocks with short notes. Local; fetches a skeleton only if the
cached one is missing.
"""
from faab.config import REPO
import os
import pickle
import time

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
from fafbseg import flywire
from faab.sources.cave import DATASTACK
from faab.partners.top10_xyz import backbone_midpoint_nm, VOX

HERE = REPO
OUT = os.path.join(HERE, "figures")
TSV = os.path.join(OUT, "faab_pr_groups_xyz.tsv")
SCACHE = os.path.join(OUT, "pr_partner_skels.pkl")
ORIG, USE, OTHER = 648518347571683763, 648518347700929678, 648518347504455355
HEADERS = ["root_id_orig", "class", "root_id_curr", "xyz", "status", "pr_date",
           "proofreader", "chk_date", "checker", "notes"]


def table(headers, rows):
    data = [[str(h) for h in headers]] + [[str(c) for c in r] for r in rows]
    w = [max(len(row[i]) for row in data) for i in range(len(headers))]
    out = [" | ".join(f"{headers[i]:<{w[i]}}" for i in range(len(headers))),
           "-+-".join("-" * x for x in w)]
    for r in rows:
        out.append(" | ".join(f"{str(r[i]):<{w[i]}}" for i in range(len(headers))))
    return "\n".join(out)


def main():
    df = pd.read_csv(TSV, sep="\t", dtype=str).fillna("")
    sk_by = pickle.load(open(SCACHE, "rb"))
    if USE not in sk_by:
        time.sleep(5)
        for s in flywire.get_l2_skeleton([USE], dataset=DATASTACK, max_threads=1,
                                         omit_failures=True, progress=False):
            sk_by[int(s.id)] = s
        pickle.dump(sk_by, open(SCACHE, "wb"))
    s = sk_by[USE]
    pt = backbone_midpoint_nm(s)
    xyz = ", ".join(str(int(round(v))) for v in (pt / VOX))
    print(f"{USE}: {s.n_nodes} skeleton nodes, xyz {xyz}")

    m = df.root_id_orig == str(ORIG)
    df.loc[m, "root_id_curr"] = str(USE)
    df.loc[m, "xyz"] = xyz
    df.loc[m, "notes"] = (f"CHECK: root split in 3; chose R8-like piece {USE} (51 out syn, "
                          f"114 chunks, in PR axon region); other piece {OTHER} has 264 out syn "
                          f"(218 chunks, z~4950) = likely merged non-PR cell; plus 1-chunk scrap")
    # short notes for the unchanged rows
    df.loc[~m, "notes"] = df.loc[~m, "notes"].str.replace(
        "longest-neurite midpoint; root unchanged; ", "", regex=False)
    df.to_csv(TSV, sep="\t", index=False)

    for g, sub in df.groupby("group", sort=False):
        print(f"\n--- group {g} ---")
        print(table(HEADERS, sub[HEADERS].values.tolist()))
    print("\nsaved", TSV)


if __name__ == "__main__":
    main()
