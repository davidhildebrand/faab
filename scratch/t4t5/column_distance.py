"""Prototype spatial penalty for the PRg1toT4T5 upstream blocks (David 2026-11-11: "as we query further upstream the
extents expand ... a kind of spatial penalty for sorting the hops? Perhaps something like proximity to a line/axis
connecting the PRg1 terminals and the T4/T5").

Idea: in each optic-lobe neuropil (LA / ME / LO / LOP, right side) columns run roughly normal to the neuropil's outer
face, so a point's COLUMN POSITION is its nearest point ("foot") on that face. The HOME COLUMN of a neuropil is the
median foot of the synapses the chain itself makes there: PR group 1 -> PR+1 (LA), PR+1 -> PR+2 (ME), PR+2 -> PR+3
(ME / LO), and the PR+3 targets' own input synapses (the T4 dendrites in ME, the T5 dendrites in LO). A synapse's
column distance = |foot - home foot| (um, and in medulla column spacings from the R7 terminal lattice). Each T4T5-1
cell is scored on its synapses onto the targets, each T4T5-2 cell on its synapses onto the T4T5-1 cells it feeds:
n_path, n_near (column distance <= NEAR columns), near share, median distance. Read-only (snapshot caught up to now;
one is_latest_roots call + throttled get_latest_roots for stale ids). Also: T4T5-2 cells that are on the LMCs tab
(putative type and confidence).
Writes figures/t4t5/<stamp>_column_distance.tsv (one row per T4T5-1 / T4T5-2 cell) and prints summaries.
Usage: column_distance.py BOOK.xlsx [--near 2]"""
import argparse
import os
import re
import time

import numpy as np
import openpyxl
import pandas as pd
import trimesh
from scipy.spatial import cKDTree

from faab import ol_meshes, synsnap
from faab.clients.cave import client as cave_client, throttle
from faab.config import DATA, FIGURES, TMP
from faab.partners.sheet import cell_text as s, utc_stamp

VOX = np.array([0.016, 0.016, 0.045])  # um per voxel (16 x 16 x 45 nm)
NEIGHBOR = {"LA_R": "ME_R", "ME_R": "LA_R", "LO_R": "ME_R", "LOP_R": "ME_R"}  # the face toward this one = outer face
IDS = re.compile(r"\d{15,}")


def face_tree(label, n=200000):
    """KD-tree of points sampled on the face of `label` that looks toward its NEIGHBOR (um)"""
    v, f = ol_meshes.mesh(label)
    m = trimesh.Trimesh(v / 1000.0, f, process=True)
    m.fix_normals()
    ov = ol_meshes.mesh(NEIGHBOR[label])[0] / 1000.0
    to = ov[cKDTree(ov).query(m.triangles_center)[1]] - m.triangles_center
    sgn = np.einsum("ij,ij->i", m.face_normals, to / np.linalg.norm(to, axis=1, keepdims=True))
    pts, fid = trimesh.sample.sample_surface(m, n, seed=0)
    keep = np.isin(fid, np.where(sgn > 0.2)[0])
    return cKDTree(pts[keep]), pts[keep]


def tab_rows(book):
    ws = openpyxl.load_workbook(book, read_only=True)["PRg1toT4T5"]
    hdr, out = None, []
    for row in ws.iter_rows(values_only=True):
        v = [s(x) for x in row]
        if v and v[0] == "hop":
            hdr = v
            continue
        if hdr and v and v[0] in ("PR+1", "PR+2", "PR+3", "T4T5-1", "T4T5-2"):
            out.append(dict(zip(hdr, v)))
    return out


def lmc_calls(book):
    ws = openpyxl.load_workbook(book, read_only=True)["LMCs"]
    title, hdr, out = "", None, {}
    for row in ws.iter_rows(values_only=True):
        v = [s(x) for x in row]
        if not v or not v[0]:
            continue
        if v[0] == "root_id_orig":
            hdr = v
            continue
        if not IDS.fullmatch(v[0]):
            title = v[0]
            continue
        if hdr:
            d = dict(zip(hdr, v))
            m = re.match(r"putative (L\d)", title)
            out[d.get("root_id_curr") or d["root_id_orig"]] = (m.group(1) if m else "type unclear", d.get("confidence", ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("book")
    ap.add_argument("--near", type=float, default=2.0, help="near = within this many medulla column spacings")
    a = ap.parse_args()
    stamp = utc_stamp()
    rows = tab_rows(a.book)
    by = {h: sorted({r["root_id_curr"] for r in rows if r["hop"] == h and IDS.fullmatch(r.get("root_id_curr", ""))})
          for h in ("PR+1", "PR+2", "PR+3", "T4T5-1", "T4T5-2")}
    feeds = {r["root_id_curr"]: IDS.findall(r.get("T4T5-1_root_id", "")) for r in rows if r["hop"] == "T4T5-2"}
    prs = pd.read_csv(os.path.join(DATA, "pr_groups.tsv"), sep="\t", dtype=str)
    g1 = sorted(set(prs[prs.group == "1"].root_id_curr))
    r7 = prs[prs["class"].str.contains("R7", na=False)].groupby("group").root_id_curr.apply(list).to_dict()

    # current roots (sheet ids can be hours old)
    c = cave_client()
    every = sorted({x for v in by.values() for x in v} | set(g1) | {x for v in feeds.values() for x in v}
                   | {x for v in r7.values() for x in v})
    last = throttle(0)
    ok = np.asarray(c.chunkedgraph.is_latest_roots([int(x) for x in every])).astype(bool)
    cur = {}
    for x, good in zip(every, ok):
        if good:
            cur[x] = [int(x)]
        else:
            last = throttle(last)
            cur[x] = [int(y) for y in np.atleast_1d(c.chunkedgraph.get_latest_roots(int(x)))]
    print(f"{len(every)} ids, {int((~ok).sum())} stale -> current pieces", flush=True)
    now = lambda ids: sorted({y for x in ids for y in cur[x]})
    back = {y: x for x, v in cur.items() for y in v}

    snap = synsnap.Snapshot(log=lambda *_: None)

    def syn(pre, post):
        last_t = time.time()
        df = snap.synapses(pre=now(pre) if pre else None, post=now(post) if post else None, when="now",
                           positions=True, client=c)
        df = df[df.pre_root != df.post_root]
        P = df[["post_x", "post_y", "post_z"]].to_numpy(float) * VOX
        df = df.assign(x=P[:, 0], y=P[:, 1], z=P[:, 2], pre=df.pre_root.map(back), post=df.post_root.map(back))
        print(f"  {len(df)} synapses ({time.time() - last_t:.0f} s)", flush=True)
        return df

    print("querying the snapshot (caught up to now)", flush=True)
    chain = pd.concat([syn(g1, by["PR+1"]).assign(stage="PR>PR+1"), syn(by["PR+1"], by["PR+2"]).assign(stage="PR+1>PR+2"),
                       syn(by["PR+2"], by["PR+3"]).assign(stage="PR+2>PR+3")])
    tin = syn(None, by["PR+3"])  # every input of the targets: their dendrite territory
    p1 = syn(by["T4T5-1"], by["PR+3"])
    p2 = syn(by["T4T5-2"], by["T4T5-1"])
    r7s = syn(sorted({x for v in r7.values() for x in v}), None)

    labels = ["LA_R", "ME_R", "LO_R", "LOP_R"]
    trees = {lab: face_tree(lab)[0] for lab in labels}

    def place(df):
        P = df[["x", "y", "z"]].to_numpy(float)
        lab = ol_meshes.assign(P, labels)
        foot = np.full((len(P), 3), np.nan)
        for L in labels:
            i = np.where(lab == L)[0]
            if len(i):
                foot[i] = trees[L].data[trees[L].query(P[i])[1]]
        return df.assign(np=lab, fx=foot[:, 0], fy=foot[:, 1], fz=foot[:, 2])

    chain, tin, p1, p2, r7s = (place(d) for d in (chain, tin, p1, p2, r7s))
    keep = os.path.join(TMP, f"{stamp}_coldist_synapses.parquet")  # for diagnostics
    pd.concat([d.assign(set=k) for k, d in (("chain", chain), ("tin", tin), ("p1", p1), ("p2", p2), ("r7", r7s))]
              ).drop(columns=["pre_root", "post_root"]).to_parquet(keep)
    print(f"synapse table {keep}")
    home, home_chain = {}, {}
    for L in labels:
        t = tin[tin.np == L][["fx", "fy", "fz"]].to_numpy()
        ch = chain[chain.np == L][["fx", "fy", "fz"]].to_numpy()
        if len(t):
            home[L] = np.median(t, axis=0)
        if len(ch):
            home_chain[L] = np.median(ch, axis=0)
    # medulla column spacing: nearest-neighbor distance between the R7 groups' medulla output feet
    feet = []
    for g, ids in r7.items():
        f = r7s[(r7s.pre.isin(ids)) & (r7s.np == "ME_R")][["fx", "fy", "fz"]].to_numpy()
        if len(f) >= 10:
            feet.append(np.median(f, axis=0))
    feet = np.array(feet)
    nn = cKDTree(feet).query(feet, k=2)[0][:, 1] if len(feet) > 2 else np.array([np.nan])
    col = float(np.median(nn))
    print(f"\nmedulla column spacing (R7 feet of {len(feet)} groups, nearest neighbor): median {col:.1f} um "
          f"(IQR {np.percentile(nn, 25):.1f}-{np.percentile(nn, 75):.1f})")
    for L in labels:
        n_t = int((tin.np == L).sum())
        gap = (np.linalg.norm(home[L] - home_chain[L]) if L in home and L in home_chain else np.nan)
        print(f"  {L}: target-input synapses {n_t}; home = their median foot; PR-chain home {gap:.1f} um from it")
    # spread of the targets' own input around home (how wide the T4/T5 subset's territory is)
    for L in ("ME_R", "LO_R"):
        t = tin[tin.np == L]
        d = np.linalg.norm(t[["fx", "fy", "fz"]].to_numpy() - home[L], axis=1)
        print(f"  target dendrite inputs in {L}: column distance median {np.median(d):.1f} um, 90% {np.percentile(d, 90):.1f} um")

    def score(df, key):
        d = np.full(len(df), np.nan)
        for L in labels:
            i = np.where(df.np.to_numpy() == L)[0]
            if L in home and len(i):
                d[i] = np.linalg.norm(df[["fx", "fy", "fz"]].to_numpy()[i] - home[L], axis=1)
        df = df.assign(dist=d)
        g = df.groupby(key)
        near = a.near * col
        return pd.DataFrame({
            "n_path": g.size(),
            "n_placed": g.dist.count(),
            "n_near": g.dist.apply(lambda v: int((v <= near).sum())),
            "median_um": g.dist.median().round(1),
            "neuropil": g.np.agg(lambda v: v.value_counts().index[0]),
        }).assign(near_share=lambda t: (t.n_near / t.n_path.clip(lower=1)).round(2),
                  median_cols=lambda t: (t.median_um / col).round(2))

    s1 = score(p1, "pre").assign(block="T4T5-1")
    s2 = score(p2, "pre").assign(block="T4T5-2")
    lmc = lmc_calls(a.book)
    out = pd.concat([s1, s2]).reset_index().rename(columns={"pre": "root_id"})
    out["putative_lmc"] = [" ".join(v for v in lmc.get(r, ("", ""))).strip() for r in out.root_id]
    path = os.path.join(FIGURES, "t4t5", f"{stamp}_column_distance.tsv")
    out.to_csv(path, sep="\t", index=False)
    print(f"\nnear = within {a.near:g} columns ({a.near * col:.0f} um) of home, in the synapse's own neuropil")
    for b, t in out.groupby("block"):
        q = t.median_cols.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(1).tolist()
        print(f"{b}: {len(t)} cells; median column distance (cols) deciles 10/25/50/75/90%: {q}; near share >= 0.5: "
              f"{int((t.near_share >= 0.5).sum())}; no near synapse: {int((t.n_near == 0).sum())}")
        print(f"  main neuropil of the path synapses: {t.neuropil.value_counts().to_dict()}")
    l2 = out[(out.block == "T4T5-2") & (out.putative_lmc != "")]
    print(f"\nT4T5-2 cells on the LMCs tab: {len(l2)}; by call: {l2.putative_lmc.value_counts().to_dict()}")
    print(f"  their median column distance (cols): {l2.median_cols.describe()[['25%', '50%', '75%']].round(2).to_dict()}")
    print(f"\nwritten {path}")


if __name__ == "__main__":
    main()
