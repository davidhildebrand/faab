"""Top-10 output partners per FAAB photoreceptor, from CACHED connectivity, plus an
xyz point ON THE PRIMARY NEURITE NEAR THE SOMA of each downstream partner.

xyz choice (per partner):
  1. soma centre = nucleus position (nuclei_v1_aedes) for the cell;
  2. reroot the L2 skeleton at the node nearest that centre;
  3. take the main trunk (path root -> geodesically farthest node) and pick the node
     ~TARGET_UM along it, requiring >= MIN_EUCLID_UM from the nucleus centre so the
     point is on the primary neurite, not on the soma body;
  4. no nucleus / tiny fragment -> fall back to the longest-neurite midpoint (flagged).

- Connectivity is NOT re-queried: rows come from figures/photoreceptor_partners.csv.
- Cached root_ids are reported in `downstream`; only for the xyz/nucleus lookup, a
  root that changed with proofreading is resolved to its current root (flagged).
- xyz in raw voxels (16,16,45 nm) = the Spelunker/FlyTable frame.
- Skeletons are cached to figures/pr_partner_skels.pkl so re-runs don't refetch.
Read-only, throttled.
"""
from faab.config import REPO
import os
import time
import pickle
import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import networkx as nx
import navis
import caveclient
from fafbseg import flywire
from faab.sources.cave import DATASTACK, throttle

OUT = os.path.join(REPO, "figures")
SERVER = "https://flytable.mrc-lmb.cam.ac.uk/"
TOPN = 10
VOX = np.array([16.0, 16.0, 45.0])
TARGET_UM, MIN_EUCLID_UM = 10.0, 5.0
HEADERS = ["upstream_PR", "class", "downstream", "xyz", "n_syn", "pct_of_outp",
           "PR_conv", "annotation", "status", "pr_date", "proofreader", "checked", "notes"]


def backbone_midpoint_nm(sk):
    try:
        if sk.n_nodes >= 5:
            ln = navis.longest_neurite(sk, n=1, reroot_soma=False)
            g = ln.get_graph_nx().to_undirected()
            ends = [n for n, d in g.degree() if d == 1]
            if len(ends) >= 2:
                path = nx.shortest_path(g, ends[0], ends[-1])
                row = ln.nodes.set_index("node_id").loc[path[len(path) // 2]]
                return np.array([row.x, row.y, row.z], float)
    except Exception:
        pass
    xyz = sk.nodes[["x", "y", "z"]].values.astype(float)
    return xyz[np.argmin(((xyz - xyz.mean(0)) ** 2).sum(1))]


def primary_neurite_point_nm(sk, soma_nm):
    """Point ~TARGET_UM along the main trunk from the soma, off the soma body."""
    nodes = sk.nodes.set_index("node_id")
    xyz = nodes[["x", "y", "z"]].values.astype(float)
    root = nodes.index.values[np.argmin(np.linalg.norm(xyz - soma_nm, axis=1))]
    g = sk.get_graph_nx().to_undirected()
    for u, v in g.edges():
        g[u][v]["w"] = float(np.linalg.norm(xyz[nodes.index.get_loc(u)]
                                            - xyz[nodes.index.get_loc(v)]))
    dist = nx.single_source_dijkstra_path_length(g, root, weight="w")
    far = max(dist, key=dist.get)
    path = nx.shortest_path(g, root, far, weight="w")
    best, best_err = None, None
    for n in path:
        geo_um = dist[n] / 1000.0
        euc_um = np.linalg.norm(xyz[nodes.index.get_loc(n)] - soma_nm) / 1000.0
        if euc_um < MIN_EUCLID_UM:
            continue
        err = abs(geo_um - TARGET_UM)
        if best is None or err < best_err:
            best, best_err = n, err
    if best is None:            # trunk never leaves the soma exclusion zone
        return None, None
    return xyz[nodes.index.get_loc(best)], dist[best] / 1000.0


def main():
    p = pd.read_csv(os.path.join(OUT, "photoreceptor_partners.csv"))
    io = pd.read_csv(os.path.join(OUT, "photoreceptor_io.csv"))
    out = p[p.direction == "output"].copy()
    cls_of = dict(zip(io.root.astype("int64"), io["class"]))
    tot_of = dict(zip(io.root.astype("int64"), io.out_syn))
    shared = out.groupby("partner")["cell"].nunique().to_dict()

    rows = []
    for cell in io.root.astype("int64"):
        d = out[out.cell == cell].nlargest(TOPN, "n_syn")
        for _, r in d.iterrows():
            rows.append({"upstream_PR": int(cell), "class": cls_of[cell],
                         "downstream": int(r.partner), "n_syn": int(r.n_syn),
                         "pct_of_outp": round(r.n_syn / tot_of[cell] * 100, 1),
                         "PR_conv": int(shared.get(int(r.partner), 1))})
    df = pd.DataFrame(rows)
    partners = sorted(set(df.downstream))
    print(f"{len(df)} rows from cache; {len(partners)} distinct downstream partners", flush=True)

    # ---- annotations (FlyTable; not the connectivity query) ----
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
        print("(FlyTable annotations unavailable:", str(e)[:60], ")", flush=True)

    # ---- root currency: resolve changed roots ONLY for the xyz/nucleus lookup ----
    client = caveclient.CAVEclient(DATASTACK)
    t = throttle(time.time())
    resolved, note = {}, {}
    try:
        latest = np.asarray(client.chunkedgraph.is_latest_roots(partners))
    except Exception as e:
        print("is_latest_roots failed:", str(e)[:60], flush=True)
        latest = np.ones(len(partners), bool)
    stale = [r for r, ok in zip(partners, latest) if not ok]
    print(f"cached roots still current: {int(latest.sum())}/{len(partners)}; "
          f"changed: {len(stale)}", flush=True)
    for r in stale:
        t = throttle(t)
        try:
            new = [int(x) for x in np.atleast_1d(client.chunkedgraph.get_latest_roots(r))]
            resolved[r] = new[0]
            note[r] = (f"root changed; xyz from current root {new[0]}"
                       + (f" (+{len(new)-1} more)" if len(new) > 1 else ""))
        except Exception as e:
            note[r] = "root changed; could not resolve (" + str(e)[:30] + ")"
    fetch_ids = [resolved.get(r, r) for r in partners]

    # ---- skeletons (cached) ----
    scache = os.path.join(OUT, "pr_partner_skels.pkl")
    sk_by = pickle.load(open(scache, "rb")) if os.path.exists(scache) else {}
    need = [i for i in fetch_ids if i not in sk_by]
    if need:
        t = throttle(t)
        sk = flywire.get_l2_skeleton(need, dataset=DATASTACK, max_threads=1,
                                     omit_failures=True, progress=True)
        for s in sk:
            sk_by[int(s.id)] = s
        pickle.dump(sk_by, open(scache, "wb"))
    print(f"skeletons available: {sum(1 for i in fetch_ids if i in sk_by)}/{len(fetch_ids)}",
          flush=True)

    # ---- fallback for single-L2-chunk fragments (no skeleton possible): L2 rep coord ----
    l2rep = {}
    no_sk = [i for i in fetch_ids if i not in sk_by]
    if no_sk:
        # route 1: fafbseg get_l2_info, with flexible column detection
        try:
            t = throttle(t)
            info = flywire.get_l2_info(no_sk, dataset=DATASTACK, progress=False)
            print("  get_l2_info columns:", list(info.columns)[:14], flush=True)
            rc = next((c for c in info.columns if "rep_coord" in c.lower()), None)
            rid_col = next((c for c in info.columns if c.lower() in ("root_id", "root")), None)
            if rc and rid_col:
                for rid, g in info.groupby(rid_col):
                    pts = np.array([np.array(v, float).ravel()[:3] for v in g[rc]])
                    if len(pts):
                        l2rep[int(rid)] = pts.mean(0)
        except Exception as e:
            print("  (get_l2_info route failed:", str(e)[:60], ")", flush=True)
        # route 2: canonical caveclient path for anything still missing
        for rid in [i for i in no_sk if i not in l2rep]:
            try:
                t = throttle(t)
                l2ids = [int(x) for x in np.atleast_1d(
                    client.chunkedgraph.get_leaves(rid, stop_layer=2))]
                if not l2ids:
                    continue
                d = client.l2cache.get_l2data(l2ids, attributes=["rep_coord_nm"])
                pts = np.array([np.array(v["rep_coord_nm"], float)
                                for v in d.values() if "rep_coord_nm" in v])
                if len(pts):
                    l2rep[int(rid)] = pts.mean(0)
            except Exception as e:
                print(f"  (l2cache route failed for {rid}: {str(e)[:40]})", flush=True)
        print(f"single-chunk fragments given an L2 rep coord: {len(l2rep)}/{len(no_sk)}",
              flush=True)

    # ---- nucleus (soma centre) per resolved root ----
    soma = {}
    t = throttle(t)
    for a in range(4):
        try:
            nuc = client.materialize.query_table("nuclei_v1_aedes",
                                                 filter_in_dict={"pt_root_id": fetch_ids})
            break
        except Exception as e:
            nuc = None
            print(f"  (nuclei retry {a+1}/4: {str(e)[:40]})", flush=True)
            time.sleep(8)
    if nuc is not None and len(nuc):
        if "pt_position_x" in nuc.columns:
            for _, r in nuc.iterrows():
                soma[int(r.pt_root_id)] = np.array([r.pt_position_x, r.pt_position_y,
                                                    r.pt_position_z], float)
        else:
            for _, r in nuc.iterrows():
                soma[int(r.pt_root_id)] = np.array(r.pt_position, float)
    print(f"nuclei found for {len(soma)}/{len(fetch_ids)} partners", flush=True)

    xyz = {}
    for r in partners:
        fid = resolved.get(r, r)
        s = sk_by.get(fid)
        if s is None:
            if fid in l2rep:
                pt = l2rep[fid]
                tag = "single L2 chunk fragment -> L2 rep coord (no soma/backbone)"
                note[r] = (note.get(r, "") + "; " if note.get(r) else "") + tag
                xyz[r] = ",".join(str(int(round(v))) for v in (pt / VOX))
            else:
                note[r] = (note.get(r, "") + "; " if note.get(r) else "") + "no skeleton, no L2 info"
            continue
        pt, geo = None, None
        if fid in soma and s.n_nodes >= 5:
            pt, geo = primary_neurite_point_nm(s, soma[fid])
        if pt is not None:
            tag = f"primary neurite ~{geo:.0f}um from soma"
        else:
            pt = backbone_midpoint_nm(s)
            tag = ("no nucleus -> backbone midpoint" if fid not in soma
                   else "trunk too short -> backbone midpoint")
        note[r] = (note.get(r, "") + "; " if note.get(r) else "") + tag
        xyz[r] = ",".join(str(int(round(v))) for v in (pt / VOX))

    df["xyz"] = df.downstream.map(lambda r: xyz.get(int(r), ""))
    df["annotation"] = df.downstream.map(lambda r: ann.get(int(r), ""))
    for c in ("status", "pr_date", "proofreader", "checked"):
        df[c] = ""
    df["notes"] = df.downstream.map(lambda r: note.get(int(r), ""))
    df = df[HEADERS]

    tsv = os.path.join(OUT, "faab_pr_top10_xyz.tsv")
    df.to_csv(tsv, sep="\t", index=False)
    n_prim = sum("primary neurite" in v for v in note.values())
    print(f"\nxyz on primary neurite: {n_prim}/{len(partners)} partners; "
          f"rest fell back to backbone midpoint", flush=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 48)
    print("\n" + df.to_string(index=False))
    print(f"\nsaved {tsv}\nDONE", flush=True)


if __name__ == "__main__":
    main()
