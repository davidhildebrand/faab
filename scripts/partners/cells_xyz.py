"""xyz points for an arbitrary list of aedes root ids (one per line, order kept).

Per id: validity check; is the root current? if not -> current roots, largest piece by L2
chunk count = root_id_curr (flagged when it holds < 70% of the chunks); L2 skeleton (cached
in figures/pr_partner_skels.pkl) and nucleus (nuclei_v1_aedes); xyz = point on the primary
neurite ~10 um from the nucleus, else longest-neurite midpoint, else L2 rep coord for
single-chunk fragments. FlyTable class/type added if present. Read-only, one CAVE call
per 5 s. Output: <prefix>.tsv and <prefix>_ascii.txt with the tracking-sheet headers.

usage: python faab_cells_xyz.py ids.txt figures/<prefix>
"""
from faab.config import REPO
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
from faab.clients.cave import DATASTACK, throttle
from faab.partners.top10_xyz import primary_neurite_point_nm, backbone_midpoint_nm, VOX

HERE = REPO
OUT = os.path.join(HERE, "figures")
SCACHE = os.path.join(OUT, "pr_partner_skels.pkl")
SERVER = "https://flytable.mrc-lmb.cam.ac.uk/"
HEADERS = ["root_id_orig", "root_id_curr", "xyz", "annotation", "status", "pr_date",
           "proofreader", "chk_date", "checker", "notes"]


def table(headers, rows):
    data = [[str(h) for h in headers]] + [[str(c) for c in r] for r in rows]
    w = [max(len(row[i]) for row in data) for i in range(len(headers))]
    out = [" | ".join(f"{headers[i]:<{w[i]}}" for i in range(len(headers))),
           "-+-".join("-" * x for x in w)]
    for r in rows:
        out.append(" | ".join(f"{str(r[i]):<{w[i]}}" for i in range(len(headers))))
    return "\n".join(out)


def fmt(pt_nm):
    return ", ".join(str(int(round(v))) for v in (np.asarray(pt_nm, float) / VOX))


def main():
    src, prefix = sys.argv[1], sys.argv[2]
    raw = [ln.strip() for ln in open(src) if ln.strip()]
    valid = [r for r in raw if r.isdigit() and len(r) == 18]
    ids = sorted({int(r) for r in valid})
    print(f"{len(raw)} lines, {len(valid)} valid ids, {len(ids)} distinct", flush=True)

    client = caveclient.CAVEclient(DATASTACK)
    t = throttle(time.time())
    note, cur = {}, {}
    try:
        latest = dict(zip(ids, [bool(x) for x in np.asarray(client.chunkedgraph.is_latest_roots(ids))]))
    except Exception as e:
        print("is_latest_roots failed:", str(e)[:60], flush=True)
        latest = {i: True for i in ids}
    stale = [i for i in ids if not latest[i]]
    print(f"current: {len(ids) - len(stale)}, changed: {len(stale)}", flush=True)
    cands = {}
    for i in stale:
        t = throttle(t)
        try:
            cands[i] = [int(x) for x in np.atleast_1d(client.chunkedgraph.get_latest_roots(i))]
        except Exception as e:
            cands[i] = []
            note[i] = "root changed; could not resolve (" + str(e)[:30] + ")"
    n_l2 = {}
    for i, roots in cands.items():
        if len(roots) > 1:
            for r in roots:
                if r not in n_l2:
                    t = throttle(t)
                    try:
                        n_l2[r] = int(len(np.atleast_1d(client.chunkedgraph.get_leaves(r, stop_layer=2))))
                    except Exception:
                        n_l2[r] = -1
    for i in ids:
        if latest[i]:
            cur[i] = i
        elif cands.get(i):
            roots = cands[i]
            if len(roots) == 1:
                cur[i] = roots[0]
                note[i] = "root changed, single current root"
            else:
                tot = sum(max(0, n_l2[r]) for r in roots)
                big = max(roots, key=lambda r: n_l2[r])
                cur[i] = big
                share = n_l2[big] / tot * 100 if tot else 0
                note[i] = (f"root split into {len(roots)}; largest piece taken ({n_l2[big]} chunks, "
                           f"{share:.0f}%)" + ("; CHECK: split into comparable pieces" if share < 70 else ""))
    print(f"resolved: {len(cur)}/{len(ids)}", flush=True)

    # ---- skeletons ----
    sk_by = pickle.load(open(SCACHE, "rb")) if os.path.exists(SCACHE) else {}
    fetch = sorted({c for c in cur.values() if c not in sk_by})
    if fetch:
        t = throttle(t)
        try:
            for s in flywire.get_l2_skeleton(fetch, dataset=DATASTACK, max_threads=1,
                                             omit_failures=True, progress=False):
                sk_by[int(s.id)] = s
        except Exception as e:
            print("skeleton fetch failed:", str(e)[:60], flush=True)
        pickle.dump(sk_by, open(SCACHE, "wb"))
    print(f"skeletons: {sum(c in sk_by for c in cur.values())}/{len(cur)}", flush=True)

    # ---- L2 rep coord for single-chunk fragments ----
    l2rep = {}
    for c in sorted({c for c in cur.values() if c not in sk_by}):
        t = throttle(t)
        try:
            l2ids = [int(x) for x in np.atleast_1d(client.chunkedgraph.get_leaves(c, stop_layer=2))]
            d = client.l2cache.get_l2data(l2ids, attributes=["rep_coord_nm"])
            pts = np.array([np.array(v["rep_coord_nm"], float) for v in d.values() if "rep_coord_nm" in v])
            if len(pts):
                l2rep[c] = pts.mean(0)
        except Exception as e:
            print(f"l2 rep failed for {c}: {str(e)[:40]}", flush=True)

    # ---- nuclei ----
    soma = {}
    t = throttle(t)
    for a in range(4):
        try:
            nuc = client.materialize.query_table("nuclei_v1_aedes",
                                                 filter_in_dict={"pt_root_id": sorted(set(cur.values()))})
            break
        except Exception as e:
            nuc = None
            print(f"nuclei retry {a + 1}/4: {str(e)[:40]}", flush=True)
            time.sleep(8)
    if nuc is not None and len(nuc):
        for _, r in nuc.iterrows():
            soma[int(r.pt_root_id)] = (np.array([r.pt_position_x, r.pt_position_y, r.pt_position_z], float)
                                       if "pt_position_x" in nuc.columns else np.array(r.pt_position, float))
    print(f"nuclei: {len(soma)}/{len(set(cur.values()))}", flush=True)

    # ---- annotations ----
    ann = {}
    try:
        import seaserpent as ss
        meta = ss.Table("aedes_main", server=SERVER).to_frame()
        meta.columns = [str(c) for c in meta.columns]
        for _, r in meta.iterrows():
            try:
                rid = int(str(r["root_id"]))
            except Exception:
                continue
            bits = [str(r[c]) for c in ("superclass", "class", "subclass", "type")
                    if c in meta.columns and str(r[c]) not in ("nan", "<NA>", "None", "")]
            if bits:
                ann[rid] = " / ".join(bits)
    except Exception as e:
        print("(FlyTable unavailable:", str(e)[:50], ")", flush=True)

    # ---- xyz ----
    xyz, how = {}, {}
    for i, c in cur.items():
        s = sk_by.get(c)
        if s is None:
            if c in l2rep:
                xyz[i], how[i] = fmt(l2rep[c]), "single L2 chunk fragment -> L2 rep coord"
            else:
                xyz[i], how[i] = "", "no skeleton, no L2 coord"
            continue
        pt, geo = (None, None)
        if c in soma and s.n_nodes >= 5:
            pt, geo = primary_neurite_point_nm(s, soma[c])
        if pt is not None:
            how[i] = f"primary neurite ~{geo:.0f}um from soma"
        else:
            pt = backbone_midpoint_nm(s)
            how[i] = "no nucleus -> backbone midpoint" if c not in soma else "trunk too short -> backbone midpoint"
        xyz[i] = fmt(pt)

    # ---- rows in the user's order ----
    rows, seen = [], {}
    for k, r in enumerate(raw, 1):
        if not (r.isdigit() and len(r) == 18):
            rows.append([r, "", "", "", "", "", "", "", "", f"INVALID id ({len(r)} digits)"])
            continue
        i = int(r)
        n = [note.get(i, ""), how.get(i, "")]
        if i in seen:
            n.append(f"duplicate of row {seen[i]}")
        else:
            seen[i] = k
        rows.append([r, str(cur.get(i, "")) if i in cur else "", xyz.get(i, ""),
                     ann.get(cur.get(i, i), "") or ann.get(i, ""), "", "", "", "", "",
                     "; ".join(x for x in n if x)])
    df = pd.DataFrame(rows, columns=HEADERS)
    df.to_csv(prefix + ".tsv", sep="\t", index=False)
    txt = table(HEADERS, rows)
    open(prefix + "_ascii.txt", "w").write(txt + "\n")
    print("\n" + txt)
    print(f"\nsaved {prefix}.tsv\nDONE", flush=True)


if __name__ == "__main__":
    main()
