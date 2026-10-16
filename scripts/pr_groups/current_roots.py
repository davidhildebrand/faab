"""The 8 photoreceptors themselves: root_id_orig (20 Aug id) -> current root(s), which
piece is the photoreceptor (size AND output-synapse count per piece), and a backbone xyz
on that piece (no nucleus expected: midpoint of the longest neurite of the L2 skeleton).
Read-only, one CAVE call per 5 s. Writes figures/faab_pr_current_roots.tsv + prints ASCII.
"""
from faab.config import REPO
import datetime
import os
import pickle
import time

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import caveclient
from fafbseg import flywire
from faab.clients.cave import DATASTACK, throttle
from faab.partners.top10_xyz import backbone_midpoint_nm, VOX

OUT = os.path.join(REPO, "figures")
DST = os.path.join(OUT, "faab_pr_current_roots.tsv")
SCACHE = os.path.join(OUT, "pr_partner_skels.pkl")
PRS = {648518347649903570: "R1-R6", 648518347649905106: "R1-R6", 648518347642147360: "R1-R6",
       648518347626884562: "R1-R6", 648518347643620818: "R1-R6", 648518347567256005: "R1-R6",
       648518347777360890: "R7-R8", 648518347626873298: "R7-R8"}


def table(headers, rows, aligns):
    data = [[str(h) for h in headers]] + [[str(c) for c in r] for r in rows]
    w = [max(len(row[i]) for row in data) for i in range(len(headers))]
    out = [" | ".join(f"{headers[i]:{aligns[i]}{w[i]}}" for i in range(len(headers))),
           "-+-".join("-" * x for x in w)]
    for r in rows:
        out.append(" | ".join(f"{str(r[i]):{aligns[i]}{w[i]}}" for i in range(len(headers))))
    return "\n".join(out)


def main():
    client = caveclient.CAVEclient(DATASTACK)
    now = datetime.datetime.now(datetime.timezone.utc)
    t = throttle(time.time())

    cands = {}
    for pr in PRS:
        t = throttle(t)
        cands[pr] = [int(x) for x in np.atleast_1d(client.chunkedgraph.get_latest_roots(pr))]
        print(f"  {pr}: {cands[pr]}", flush=True)
    allc = sorted({x for v in cands.values() for x in v})

    n_l2 = {}
    for rid in allc:
        t = throttle(t)
        n_l2[rid] = int(len(np.atleast_1d(client.chunkedgraph.get_leaves(rid, stop_layer=2))))
    print("sizes done", flush=True)

    # output synapses per candidate piece (live, at now)
    t = throttle(t)
    syn = client.materialize.live_query("synapses_v2", timestamp=now,
                                        filter_in_dict={"pre_pt_root_id": allc})
    out_syn = syn.groupby("pre_pt_root_id").size().to_dict()
    print(f"live output synapses over all pieces: {len(syn)}", flush=True)

    chosen = {}
    for pr, roots in cands.items():
        ranked = sorted(roots, key=lambda x: (-int(out_syn.get(x, 0)), -n_l2[x]))
        chosen[pr] = ranked[0]

    sk_by = pickle.load(open(SCACHE, "rb")) if os.path.exists(SCACHE) else {}
    need = sorted({r for r in chosen.values() if r not in sk_by})
    if need:
        t = throttle(t)
        sk = flywire.get_l2_skeleton(need, dataset=DATASTACK, max_threads=1,
                                     omit_failures=True, progress=True)
        for s in sk:
            sk_by[int(s.id)] = s
        pickle.dump(sk_by, open(SCACHE, "wb"))

    rows = []
    for pr, cls in PRS.items():
        roots, cur = cands[pr], chosen[pr]
        tot = sum(n_l2[x] for x in roots)
        s = sk_by.get(cur)
        if s is not None:
            pt = backbone_midpoint_nm(s)
            xyz = ", ".join(str(int(round(v))) for v in (pt / VOX))
            how = f"longest-neurite midpoint ({s.n_nodes} skeleton nodes)"
        else:
            xyz, how = "", "no skeleton"
        others = [x for x in roots if x != cur]
        if cur == pr:
            note = "root unchanged"
        elif not others:
            note = "root changed, single current root"
        else:
            note = ("split into %d pieces; this one has %d of %d output synapses; other: %s"
                    % (len(roots), out_syn.get(cur, 0), sum(out_syn.get(x, 0) for x in roots),
                       ", ".join(f"{x} ({out_syn.get(x, 0)} syn, {n_l2[x]} chunks)"
                                 for x in others)))
        rows.append([pr, cls, cur, xyz, len(roots), n_l2[cur],
                     f"{n_l2[cur] / tot * 100:.0f}" if tot else "", int(out_syn.get(cur, 0)),
                     how + "; " + note])
    hdr = ["root_id_orig", "class", "root_id_curr", "xyz", "pieces", "curr_chunks",
           "pct_chunks", "out_syn_curr", "notes"]
    df = pd.DataFrame(rows, columns=hdr)
    df.to_csv(DST, sep="\t", index=False)
    print()
    print(table(hdr, rows, ["<", "<", "<", "<", ">", ">", ">", ">", "<"]))
    print(f"\nchecked at {now}\nsaved {DST}\nDONE", flush=True)


if __name__ == "__main__":
    main()
