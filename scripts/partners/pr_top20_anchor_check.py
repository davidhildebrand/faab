"""xyz anchor check for the refreshed photoreceptor -> partner table.

The sheet's xyz is the anchor of a row and is never recomputed. This looks up which segment
(root) occupies each row's xyz today and flags rows where that segment is not root_id_curr:
    "CHECK: segment at xyz is <root>, not root_id_curr"
Same check for the 8 PR points from tmp/sheet_prs.tsv (printed only). Rewrites the `update`
column of figures/faab_pr_top20_update.tsv in place (old CHECK parts are replaced) and writes
figures/faab_pr_top20_anchor_check.tsv with one line per looked-up point. Read-only.
"""
from faab.config import REPO
import json
import os
import time

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import caveclient
from fafbseg import flywire
from faab.clients.cave import DATASTACK
from faab.partners.top10_xyz import VOX
from faab.config import fig

HERE = REPO
OUT = os.path.join(HERE, "figures")
TMP = os.path.join(HERE, "..", "tmp")
TSV = fig("faab_pr_top20_update.tsv")
META = fig("faab_pr_top20_update_meta.json")
PRS = os.path.join(TMP, "sheet_prs.tsv")
CHK = fig("faab_pr_top20_anchor_check.tsv")


def valid(s):
    s = str(s).strip()
    return s if s.isdigit() and len(s) == 18 else None


def parse_xyz(s):
    try:
        v = [float(x) for x in str(s).split(",")]
        return v if len(v) == 3 else None
    except Exception:
        return None


def lookup(locs_vox):
    """root id at each point (sheet voxel coords, 16x16x45 nm) at the time of the call"""
    locs_nm = np.asarray(locs_vox, float) * VOX
    try:
        roots = flywire.locs_to_segments(locs_nm, coordinates="nm", backend="cloudvolume",
                                         dataset=DATASTACK)
        return np.asarray(roots).astype(np.int64), "fafbseg/cloudvolume"
    except Exception as e:
        print("fafbseg lookup failed, falling back to cloudvolume directly:", str(e)[:150], flush=True)
    import cloudvolume
    client = caveclient.CAVEclient(DATASTACK)
    vol = cloudvolume.CloudVolume(client.info.segmentation_source(), use_https=True, progress=False,
                                  fill_missing=True)
    res = np.asarray(vol.resolution, float)
    sv = []
    for p in locs_nm:
        v = np.floor(p / res).astype(int)
        cut = vol[v[0]:v[0] + 1, v[1]:v[1] + 1, v[2]:v[2] + 1]
        sv.append(int(np.asarray(cut).ravel()[0]))
    sv = np.array(sv, dtype=np.uint64)
    roots = np.zeros(len(sv), dtype=np.int64)
    nz = sv != 0
    if nz.any():
        time.sleep(5)
        roots[nz] = np.asarray(client.chunkedgraph.get_roots(sv[nz])).astype(np.int64)
    return roots, "cloudvolume+get_roots"


def main():
    t0 = time.time()
    now = pd.Timestamp.utcnow()
    df = pd.read_csv(TSV, sep="\t", dtype=str).fillna("")
    prs = pd.read_csv(PRS, sep="\t", dtype=str).fillna("")
    meta = json.load(open(META))
    pr_curr = meta["pr_roots"]

    locs, keys = [], []
    for i, r in df.iterrows():
        v = parse_xyz(r["xyz"])
        if v:
            locs.append(v)
            keys.append(("row", i))
    for _, r in prs.iterrows():
        v = parse_xyz(r["xyz"])
        if v:
            locs.append(v)
            keys.append(("pr", r["root_id_orig"]))
    print(f"{len(df)} table rows, {sum(k[0] == 'row' for k in keys)} with xyz; "
          f"{sum(k[0] == 'pr' for k in keys)} PR points; lookup at {now:%Y-%m-%d %H:%M} UTC", flush=True)
    roots, how = lookup(locs)
    print(f"lookup via {how}: {len(roots)} points, {int((roots == 0).sum())} returned 0", flush=True)
    seg = dict(zip(keys, [int(r) for r in roots]))

    # which table rows carry each current root (to say where a mismatching segment lives)
    where = {}
    for _, r in df.iterrows():
        if valid(r["root_id_curr"]):
            where.setdefault(r["root_id_curr"], []).append(f"{r['upstream_PR'][-6:]}/{r['root_id_orig']}")

    lines, n_mis, n_zero, n_ok = [], 0, 0, 0
    for i, r in df.iterrows():
        parts = [p.strip() for p in str(r["update"]).split(";")
                 if p.strip() and not p.strip().startswith("CHECK:")]
        s = seg.get(("row", i))
        verdict = "no xyz"
        if s is not None and valid(r["root_id_curr"]):
            if s == 0:
                parts.append("CHECK: no segment at xyz (id 0)")
                n_zero += 1
                verdict = "zero"
            elif s != int(r["root_id_curr"]):
                note = f"CHECK: segment at xyz is {s}, not root_id_curr"
                if str(s) in where:
                    note += f" (= root_id_curr of {', '.join(where[str(s)])})"
                parts.append(note)
                n_mis += 1
                verdict = "MISMATCH"
            else:
                n_ok += 1
                verdict = "ok"
        df.at[i, "update"] = "; ".join(parts)
        lines.append((r["upstream_PR"], r["root_id_orig"], r["root_id_curr"], r["xyz"],
                      "" if s is None else str(s), verdict))
    df.to_csv(TSV, sep="\t", index=False)
    pd.DataFrame(lines, columns=["upstream_PR", "root_id_orig", "root_id_curr", "xyz", "segment_at_xyz",
                                 "verdict"]).to_csv(CHK, sep="\t", index=False)
    print(f"\nxyz anchor check: {n_ok} rows where the segment at xyz == root_id_curr, "
          f"{n_mis} MISMATCH, {n_zero} with no segment at xyz", flush=True)
    for l in lines:
        if l[5] in ("MISMATCH", "zero"):
            print("  ", "\t".join(l), flush=True)
    print("\nPR points (sheet xyz):", flush=True)
    for _, r in prs.iterrows():
        s = seg.get(("pr", r["root_id_orig"]))
        used = pr_curr.get(r["root_id_orig"], "?")
        if s is None:
            print(f"  PR {r['root_id_orig']}: no xyz", flush=True)
        else:
            print(f"  PR {r['root_id_orig']} [{r.get('class', '')}]: segment at xyz = {s}; root used = {used} "
                  f"{'OK' if str(s) == used else 'MISMATCH'}", flush=True)
    print(f"DONE in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
