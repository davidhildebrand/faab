"""For the cached partners that now map to SEVERAL current roots: how big is each piece,
and which piece actually carries the photoreceptor's synapses?

Per case: candidate roots (get_latest_roots), size of each (L2-chunk count), and the
number of synapses from the photoreceptor's current root(s) onto each candidate
(live synapse query at 'now', so roots made after the last materialization count).
Read-only, one CAVE call per 5 s. Saves figures/pr_partner_split_check.json.
"""
from faab.config import REPO
import datetime
import json
import os
import re
import time

import numpy as np
import pandas as pd
import caveclient
from faab.clients.cave import DATASTACK, throttle

OUT = os.path.join(REPO, "figures")
TSV = os.path.join(OUT, "faab_pr_top10_xyz.tsv")
DST = os.path.join(OUT, "pr_partner_split_check.json")
PAT = re.compile(r"root changed; xyz from current root (\d+); largest of (\d+) current roots")


def main():
    df = pd.read_csv(TSV, sep="\t", dtype=str)
    cases = {}                                   # cached id -> {pr, major, n_syn_cached}
    for _, r in df.iterrows():
        m = PAT.match(str(r["notes"]))
        if m:
            cases.setdefault(int(r["downstream"]), []).append(
                {"pr": int(r["upstream_PR"]), "major": int(m.group(1)),
                 "n_syn_cached": int(r["n_syn"])})
    print(f"{len(cases)} multi-root cases", flush=True)

    client = caveclient.CAVEclient(DATASTACK)
    now = datetime.datetime.now(datetime.timezone.utc)
    t = throttle(time.time())

    cands, n_l2 = {}, {}
    for cid in cases:
        t = throttle(t)
        cands[cid] = [int(x) for x in np.atleast_1d(client.chunkedgraph.get_latest_roots(cid))]
    for rid in sorted({x for v in cands.values() for x in v}):
        t = throttle(t)
        try:
            n_l2[rid] = int(len(np.atleast_1d(client.chunkedgraph.get_leaves(rid, stop_layer=2))))
        except Exception as e:
            n_l2[rid] = -1
            print(f"  get_leaves failed {rid}: {str(e)[:40]}", flush=True)
    print("sizes done", flush=True)

    prs = sorted({c["pr"] for v in cases.values() for c in v})
    pr_roots = {}
    for pr in prs:
        t = throttle(t)
        pr_roots[pr] = [int(x) for x in np.atleast_1d(client.chunkedgraph.get_latest_roots(pr))]
        print(f"  PR {pr} -> current roots {pr_roots[pr]}", flush=True)

    # synapses from each PR's current roots, counted per postsynaptic root (live, at now)
    post_counts = {}
    for pr in prs:
        t = throttle(t)
        try:
            syn = client.materialize.live_query(
                "synapses_v2", timestamp=now,
                filter_in_dict={"pre_pt_root_id": pr_roots[pr]})
            post_counts[pr] = syn.groupby("post_pt_root_id").size().to_dict()
            print(f"  PR {pr}: {len(syn)} output synapses live", flush=True)
        except Exception as e:
            post_counts[pr] = {}
            print(f"  PR {pr}: live synapse query failed: {str(e)[:60]}", flush=True)

    rows, out = [], {}
    for cid, uses in cases.items():
        for u in uses:
            pr = u["pr"]
            tot = sum(max(0, n_l2[x]) for x in cands[cid])
            detail = []
            for x in sorted(cands[cid], key=lambda x: -n_l2[x]):
                s = int(post_counts.get(pr, {}).get(x, 0))
                detail.append({"root": x, "l2_chunks": n_l2[x],
                               "pct_chunks": round(n_l2[x] / tot * 100, 1) if tot else None,
                               "syn_from_pr": s})
            syn_on_major = next(d["syn_from_pr"] for d in detail if d["root"] == u["major"])
            syn_total = sum(d["syn_from_pr"] for d in detail)
            best = max(detail, key=lambda d: d["syn_from_pr"])
            out[f"{pr}->{cid}"] = {"pr": pr, "cached": cid, "major": u["major"],
                                   "n_syn_cached": u["n_syn_cached"], "pieces": detail}
            rows.append([pr, cid, u["n_syn_cached"], len(detail), u["major"],
                         f"{detail[0]['l2_chunks']} ({detail[0]['pct_chunks']}%)",
                         f"{detail[1]['l2_chunks']}" if len(detail) > 1 else "-",
                         syn_on_major, syn_total,
                         "yes" if best["root"] == u["major"] or syn_total == 0 else
                         f"NO -> {best['root']} ({best['syn_from_pr']} syn)"])
    json.dump(out, open(DST, "w"), indent=1)

    hdr = ["upstream_PR", "root_id_orig", "n_syn_cached", "pieces", "major_root",
           "major_chunks", "2nd_chunks", "syn_on_major", "syn_all_pieces", "major_has_syn"]
    w = [max(len(str(x)) for x in [h] + [r[i] for r in rows]) for i, h in enumerate(hdr)]
    print()
    print(" | ".join(f"{h:<{w[i]}}" for i, h in enumerate(hdr)))
    print("-+-".join("-" * x for x in w))
    for r in rows:
        print(" | ".join(f"{str(c):<{w[i]}}" for i, c in enumerate(r)))
    print(f"\nsaved {DST}\nDONE", flush=True)


if __name__ == "__main__":
    main()
