"""For cached partner roots that now map to SEVERAL current roots, pick the MAJOR current
root (largest by L2-chunk count = the new main structure), put the xyz point on it, and
drop the "(+N more)" wording. Rows that map to a single current root are left untouched.

Updates figures/faab_pr_top10_xyz.tsv in place (row order preserved). Read-only CAVE
access, throttled to one call per 5 s. Skeletons are cached in pr_partner_skels.pkl.
"""
from faab.config import REPO
import os
import re
import time
import pickle
import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import caveclient
from fafbseg import flywire
from faab.clients.cave import DATASTACK, throttle
from faab.partners.top10_xyz import primary_neurite_point_nm, backbone_midpoint_nm, VOX
from faab.config import fig

OUT = os.path.join(REPO, "figures")
TSV = fig("faab_pr_top10_xyz.tsv")
SCACHE = fig("pr_partner_skels.pkl")
PAT = re.compile(r"root changed; xyz from current root (\d+) \(\+(\d+) more\); (.*)")


def main():
    df = pd.read_csv(TSV, sep="\t", dtype=str)
    multi = {}                       # cached id -> previously used root
    for _, r in df.iterrows():
        m = PAT.match(str(r["notes"]))
        if m:
            multi[int(r["downstream"])] = int(m.group(1))
    print(f"{len(multi)} cached roots map to several current roots", flush=True)

    client = caveclient.CAVEclient(DATASTACK)
    t = throttle(time.time())

    # ---- all current roots per cached id ----
    cands = {}
    for cid in multi:
        t = throttle(t)
        cands[cid] = [int(x) for x in np.atleast_1d(client.chunkedgraph.get_latest_roots(cid))]
        print(f"  {cid}: {len(cands[cid])} current roots", flush=True)

    # ---- size (L2 chunk count) of every candidate ----
    n_l2 = {}
    for rid in sorted({x for v in cands.values() for x in v}):
        t = throttle(t)
        try:
            n_l2[rid] = len(np.atleast_1d(client.chunkedgraph.get_leaves(rid, stop_layer=2)))
        except Exception as e:
            print(f"  get_leaves failed for {rid}: {str(e)[:50]}", flush=True)
            n_l2[rid] = -1
    major = {}
    for cid, roots in cands.items():
        ranked = sorted(roots, key=lambda x: (-n_l2[x], x != multi[cid]))
        major[cid] = ranked[0]
        share = n_l2[ranked[0]] / max(1, sum(max(0, n_l2[x]) for x in roots)) * 100
        print(f"  {cid}: major {ranked[0]} ({n_l2[ranked[0]]} L2 chunks, {share:.0f}% of all "
              f"pieces); previously used {multi[cid]} ({n_l2.get(multi[cid], '?')} chunks)"
              f"{'  <- CHANGED' if ranked[0] != multi[cid] else ''}", flush=True)

    # ---- skeletons for the major roots ----
    sk_by = pickle.load(open(SCACHE, "rb")) if os.path.exists(SCACHE) else {}
    need = sorted({m for m in major.values() if m not in sk_by})
    if need:
        t = throttle(t)
        sk = flywire.get_l2_skeleton(need, dataset=DATASTACK, max_threads=1,
                                     omit_failures=True, progress=True)
        for s in sk:
            sk_by[int(s.id)] = s
        pickle.dump(sk_by, open(SCACHE, "wb"))
    print(f"skeletons for major roots: {sum(m in sk_by for m in major.values())}/{len(major)}",
          flush=True)

    # ---- nuclei for the major roots ----
    soma = {}
    t = throttle(t)
    ids = sorted(set(major.values()))
    for a in range(4):
        try:
            nuc = client.materialize.query_table("nuclei_v1_aedes",
                                                 filter_in_dict={"pt_root_id": ids})
            break
        except Exception as e:
            nuc = None
            print(f"  (nuclei retry {a+1}/4: {str(e)[:40]})", flush=True)
            time.sleep(8)
    if nuc is not None and len(nuc):
        for _, r in nuc.iterrows():
            if "pt_position_x" in nuc.columns:
                soma[int(r.pt_root_id)] = np.array([r.pt_position_x, r.pt_position_y,
                                                    r.pt_position_z], float)
            else:
                soma[int(r.pt_root_id)] = np.array(r.pt_position, float)
    print(f"nuclei for major roots: {len(soma)}/{len(ids)}", flush=True)

    # ---- xyz on the major root ----
    l2rep = {}
    new_xyz, new_note = {}, {}
    for cid, m in major.items():
        n = len(cands[cid])
        s = sk_by.get(m)
        if s is None:
            t = throttle(t)
            l2ids = [int(x) for x in np.atleast_1d(client.chunkedgraph.get_leaves(m, stop_layer=2))]
            d = client.l2cache.get_l2data(l2ids, attributes=["rep_coord_nm"])
            pts = np.array([np.array(v["rep_coord_nm"], float) for v in d.values()
                            if "rep_coord_nm" in v])
            pt = pts.mean(0)
            tag = "single L2 chunk fragment -> L2 rep coord (no soma/backbone)"
        else:
            pt, geo = (None, None)
            if m in soma and s.n_nodes >= 5:
                pt, geo = primary_neurite_point_nm(s, soma[m])
            if pt is not None:
                tag = f"primary neurite ~{geo:.0f}um from soma"
            else:
                pt = backbone_midpoint_nm(s)
                tag = ("no nucleus -> backbone midpoint" if m not in soma
                       else "trunk too short -> backbone midpoint")
        new_xyz[cid] = ",".join(str(int(round(v))) for v in (pt / VOX))
        new_note[cid] = (f"root changed; xyz from current root {m}; "
                         f"largest of {n} current roots; {tag}")

    # ---- write back, order preserved ----
    changed = 0
    for i, r in df.iterrows():
        cid = int(r["downstream"])
        if cid in new_xyz:
            if r["xyz"] != new_xyz[cid]:
                changed += 1
            df.at[i, "xyz"] = new_xyz[cid]
            df.at[i, "notes"] = new_note[cid]
    df.to_csv(TSV, sep="\t", index=False)
    print(f"\nrows updated: {sum(int(r) in new_xyz for r in df.downstream)}; "
          f"xyz actually moved: {changed}\nsaved {TSV}\nDONE", flush=True)


if __name__ == "__main__":
    main()
