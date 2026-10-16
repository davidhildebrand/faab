"""Current root + backbone xyz for a list of photoreceptor root ids (grouped by ommatidium).

Input : TSV with columns  group  root_id_orig  class   (default tmp/pr_groups.tsv)
Per id: is the root current? if not -> all current roots; for split roots the size of
        each piece (L2 chunks); ONE live synapse query over every candidate piece gives
        output synapses per piece; root_id_curr = piece with the output synapses (ties ->
        larger); xyz = longest-neurite midpoint of that piece's L2 skeleton (no nucleus).
Output: figures/faab_pr_groups_xyz.tsv (sheet headers + group) and ASCII blocks per group.
Read-only, one CAVE call per 5 s; skeletons cached in pr_partner_skels.pkl.
"""
from faab.config import REPO
import datetime
import os
import pickle
import sys
import time

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import caveclient
from fafbseg import flywire
from faab.sources.cave import DATASTACK, throttle
from faab.partners.top10_xyz import backbone_midpoint_nm, VOX

HERE = REPO
OUT = os.path.join(HERE, "figures")
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "tmp", "pr_groups.tsv")
DST = os.path.join(OUT, "faab_pr_groups_xyz.tsv")
SCACHE = os.path.join(OUT, "pr_partner_skels.pkl")
HEADERS = ["root_id_orig", "class", "root_id_curr", "xyz", "status", "pr_date",
           "proofreader", "chk_date", "checker", "notes"]
ALIGNS = ["<"] * len(HEADERS)


def table(headers, rows, aligns):
    data = [[str(h) for h in headers]] + [[str(c) for c in r] for r in rows]
    w = [max(len(row[i]) for row in data) for i in range(len(headers))]
    out = [" | ".join(f"{headers[i]:{aligns[i]}{w[i]}}" for i in range(len(headers))),
           "-+-".join("-" * x for x in w)]
    for r in rows:
        out.append(" | ".join(f"{str(r[i]):{aligns[i]}{w[i]}}" for i in range(len(headers))))
    return "\n".join(out)


def main():
    src = pd.read_csv(SRC, sep="\t", dtype=str)
    ids = [int(x) for x in src.root_id_orig]
    print(f"{len(ids)} ids in {src.group.nunique()} groups", flush=True)

    client = caveclient.CAVEclient(DATASTACK)
    now = datetime.datetime.now(datetime.timezone.utc)
    t = throttle(time.time())

    # ---- currency ----
    latest = {}
    try:
        lat = np.asarray(client.chunkedgraph.is_latest_roots(ids, timestamp=now))
        latest = dict(zip(ids, [bool(x) for x in lat]))
    except Exception as e:
        print("is_latest_roots failed:", str(e)[:80], flush=True)
    stale = [i for i in ids if not latest.get(i, False)]
    print(f"current: {len(ids) - len(stale)}, changed/unknown: {len(stale)}", flush=True)

    cands, err = {}, {}
    for i in ids:
        if latest.get(i, False):
            cands[i] = [i]
            continue
        t = throttle(t)
        try:
            cands[i] = [int(x) for x in np.atleast_1d(client.chunkedgraph.get_latest_roots(i))]
        except Exception as e:
            err[i] = "could not resolve current root (" + str(e)[:40] + ")"
            cands[i] = []
        print(f"  {i} -> {cands[i]}", flush=True)

    # ---- sizes, only where a root split ----
    n_l2 = {}
    for i, roots in cands.items():
        if len(roots) < 2:
            continue
        for r in roots:
            if r in n_l2:
                continue
            t = throttle(t)
            try:
                n_l2[r] = int(len(np.atleast_1d(client.chunkedgraph.get_leaves(r, stop_layer=2))))
            except Exception:
                n_l2[r] = -1
    print("sizes done", flush=True)

    # ---- output synapses per candidate piece (live, one query; per-group fallback) ----
    allc = sorted({r for v in cands.values() for r in v})
    out_syn = {}
    t = throttle(t)
    try:
        syn = client.materialize.live_query("synapses_v2", timestamp=now,
                                            filter_in_dict={"pre_pt_root_id": allc})
        out_syn = syn.groupby("pre_pt_root_id").size().to_dict()
        print(f"live output synapses over all pieces: {len(syn)}", flush=True)
    except Exception as e:
        print("single live query failed:", str(e)[:80], "-> per-group queries", flush=True)
        for g, sub in src.groupby("group"):
            roots = sorted({r for i in sub.root_id_orig.astype("int64") for r in cands[i]})
            t = throttle(t)
            try:
                syn = client.materialize.live_query("synapses_v2", timestamp=now,
                                                    filter_in_dict={"pre_pt_root_id": roots})
                out_syn.update(syn.groupby("pre_pt_root_id").size().to_dict())
            except Exception as e2:
                print(f"  group {g} synapse query failed: {str(e2)[:60]}", flush=True)

    chosen = {}
    for i, roots in cands.items():
        if roots:
            chosen[i] = sorted(roots, key=lambda r: (-int(out_syn.get(r, 0)), -n_l2.get(r, 0)))[0]

    # ---- skeletons ----
    sk_by = pickle.load(open(SCACHE, "rb")) if os.path.exists(SCACHE) else {}
    need = sorted({r for r in chosen.values() if r not in sk_by})
    if need:
        t = throttle(t)
        sk = flywire.get_l2_skeleton(need, dataset=DATASTACK, max_threads=1,
                                     omit_failures=True, progress=True)
        for s in sk:
            sk_by[int(s.id)] = s
        pickle.dump(sk_by, open(SCACHE, "wb"))

    # ---- rows ----
    rows = []
    for _, r in src.iterrows():
        i, cls = int(r.root_id_orig), r["class"]
        if i in err:
            rows.append([r.group, i, cls, "", "", err[i]])
            continue
        cur, roots = chosen[i], cands[i]
        s = sk_by.get(cur)
        if s is not None:
            pt = backbone_midpoint_nm(s)
            xyz = ", ".join(str(int(round(v))) for v in (pt / VOX))
            how = "longest-neurite midpoint"
        else:
            xyz, how = "", "no skeleton (single L2 chunk?)"
        if cur == i:
            note = "root unchanged"
        elif len(roots) == 1:
            note = "root changed, single current root"
        else:
            others = [x for x in roots if x != cur]
            tot_syn = sum(int(out_syn.get(x, 0)) for x in roots)
            tot_chk = sum(max(0, n_l2.get(x, 0)) for x in roots)
            pct = f"{n_l2.get(cur, 0) / tot_chk * 100:.0f}%" if tot_chk else "?"
            note = (f"split into {len(roots)} pieces; this one has {int(out_syn.get(cur, 0))}/"
                    f"{tot_syn} output syn and {pct} of chunks; other: "
                    + ", ".join(f"{x} ({int(out_syn.get(x, 0))} syn, {n_l2.get(x, '?')} chunks)"
                                for x in others))
        rows.append([r.group, i, cls, cur, xyz, f"{how}; {note}; out_syn {int(out_syn.get(cur, 0))}"])

    df = pd.DataFrame(rows, columns=["group", "root_id_orig", "class", "root_id_curr", "xyz", "notes"])
    for c in ("status", "pr_date", "proofreader", "chk_date", "checker"):
        df[c] = ""
    df = df[["group"] + HEADERS]
    df.to_csv(DST, sep="\t", index=False)

    for g, sub in df.groupby("group", sort=False):
        print(f"\n--- group {g} ---")
        print(table(HEADERS, sub[HEADERS].values.tolist(), ALIGNS))
    print(f"\nchecked at {now}\nsaved {DST}\nDONE", flush=True)


if __name__ == "__main__":
    main()
