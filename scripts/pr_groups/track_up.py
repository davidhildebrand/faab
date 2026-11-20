"""Follow a PRs-tab group's axons from the lamina up through the column zone (David 2026-11-20: "Track group 19 up
through 90 um first, then try to split"), keeping EVERY axon in the slabs, so that neighbor bundles and the unlisted
continuations of truncated members show up too.

  track   read-only CAVE. Bundle axis = first principal axis of the members' cached L2 skeletons within 40 um of
          their xyz, pointing away from the lamina (the members' presynaptic sites, local synapse snapshot). Slabs of
          2*HALF_XY x 2*HALF_XY x 2*HALF_Z um (mip-1 segmentation) every --step um from --from to --to um above the
          lamina level (median of those sites along the axis). Each slab's supervoxels are mapped to roots (one
          get_roots call per slab, >= 2 s apart). Each next center = this one + step along the axis, moved sideways
          to the mean of the axons seen within FOLLOW_UM of it (so the slabs follow a bending bundle after the
          members end). -> figures/pr_groups/track_up_g<G>/track.tsv (slab, height, root, voxels, centroid um).
  report  network-free. Per slab, the axon cross-sections (centroids projected across the axis; pieces of
          >= MIN_VOX voxels) clustered by single linkage at LINK_UM; members labeled with their class. Axons seen in
          >= MIN_SLABS slabs of the column zone (45-90 um) are clustered by their mean pairwise distance there
          (average linkage at CUT_UM), and each member is assigned to the column-zone bundle its last slabs lie
          nearest. -> report.txt (printed) and bundles.tsv.
  follow  network-free: each member followed through its breaks (the nearest axon-sized piece continuing where it
          ends), its chain's position 45-60 um up, and those positions clustered -> follow.tsv (cmd_follow).
Usage: track_up.py track BOOK.xlsx G [--from -5] [--to 90] [--step 3]
       track_up.py follow BOOK.xlsx G [--jump 0.8] [--gap 2]
       track_up.py report BOOK.xlsx G"""
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from faab.config import VOX, fig
from faab.pr import prs_tab as T

HALF_XY, HALF_Z = 6.0, 0.7
RES1 = np.array([32.0, 32.0, 45.0])
MIN_VOX, FOLLOW_UM, LINK_UM, CUT_UM, MIN_SLABS = 150, 4.0, 1.8, 2.4, 4
# report: a piece counts as an axon cross-section from AXON_VOX mip-1 voxels (a 0.5 um axon in a 1.4 um slab ~ 6,000;
# 2026-11-20 group 19: ~1,600 pieces per slab above 150 voxels, nearly all specks)
AXON_VOX = 6000  # members' pieces: median 6,000-50,000 voxels
ZONE = (45.0, 90.0)


def group_rows(book, G):
    wb, ws = T.load(book)
    b = next((b for b in T.blocks(ws) if b.group == G), None)
    if b is None:
        sys.exit(f"no group {G}")
    return [dict(root=T.get(ws, b, r, "root_id_curr"), cls=T.get(ws, b, r, "class"), xyz=T.get(ws, b, r, "xyz"),
                 notes=T.get(ws, b, r, "notes")) for r in b.rows]


def outdir(G):
    d = fig(os.path.join("pr_groups", f"track_up_g{G}"))
    os.makedirs(d, exist_ok=True)
    return d


def frame(rows):
    from faab import synsnap
    from faab.pr.bundles_pairs import SCACHE
    sk = pickle.load(open(SCACHE, "rb"))
    xyz = np.array([[float(c) for c in r["xyz"].split(",")] for r in rows]) * np.array(VOX) / 1000.0
    p0 = np.median(xyz, axis=0)
    pts = []
    for r in rows:
        n = sk.get(int(r["root"]))
        if n is not None:
            q = n.nodes[["x", "y", "z"]].to_numpy(float) / 1000.0
            pts.append(q[np.linalg.norm(q - p0, axis=1) < 40])
    if not pts:
        sys.exit("no cached skeletons for the members")
    P = np.vstack(pts)
    u = np.linalg.svd(P - P.mean(0), full_matrices=False)[2][0]
    syn = synsnap.Snapshot(log=lambda *_: None).synapses(pre=[int(r["root"]) for r in rows], when="latest", positions=True)
    pre = syn[syn.pre_root != syn.post_root][["pre_x", "pre_y", "pre_z"]].to_numpy(float) * np.array(VOX) / 1000.0
    lam = float(np.median((pre - p0) @ u))
    if lam > 0:
        u, lam = -u, -lam
    return p0, u, lam, len(pre)


def cmd_track(book, G, lo, hi, step):
    from cloudvolume import CloudVolume
    from faab.clients.cave import client as cave_client, throttle
    rows = [r for r in group_rows(book, G) if r["root"].isdigit()]
    p0, u, lam, npre = frame(rows)
    print(f"group {G}: {len(rows)} members; axis {np.round(u, 3)}; lamina level {lam:.1f} um from the members' median "
          f"xyz ({npre} presynaptic sites)", flush=True)
    client = cave_client()
    vol = CloudVolume(client.info.segmentation_source(), use_https=True, agglomerate=False, progress=False,
                      fill_missing=True, bounded=False, mip=1)
    half = np.round(np.array([HALF_XY, HALF_XY, HALF_Z]) * 1000.0 / RES1).astype(int)
    ctr = p0 + u * (lam + lo)
    rec, t = [], throttle(0)
    from faab.pr.bundles import read_box
    for k, h in enumerate(np.arange(lo, hi + 1e-6, step)):
        t0 = time.time()
        box = read_box(vol, ctr, half, RES1)
        svs = sorted(box)
        t = throttle(t)
        roots = np.asarray(client.chunkedgraph.get_roots(svs)).astype(np.int64) if svs else []
        per = defaultdict(list)
        for sv, rt in zip(svs, roots):
            per[int(rt)].append(box[sv])
        seen = []
        for rt, lst in per.items():
            w = np.array([n for n, _ in lst], float)
            if w.sum() < MIN_VOX:
                continue
            cm = np.sum([n * np.array(p) for n, p in lst], axis=0) / w.sum()
            rec.append(dict(slab=k, height=float(h), root=str(rt), vox=int(w.sum()), x=cm[0], y=cm[1], z=cm[2],
                            cx=ctr[0], cy=ctr[1], cz=ctr[2]))
            d = cm - ctr
            if np.linalg.norm(d - u * (d @ u)) < FOLLOW_UM:
                seen.append(cm)
        mem = sum(1 for x in rec if x["slab"] == k and x["root"] in {r["root"] for r in rows})
        print(f"  slab {k:2d} at {h:5.1f} um: {len(svs)} supervoxels, {len(per)} roots, {mem} members; {time.time() - t0:.0f} s",
              flush=True)
        nxt = ctr + u * step
        if seen:
            m = np.mean(seen, axis=0)
            d = m - ctr
            nxt = nxt + (d - u * (d @ u))
        ctr = nxt
    out = os.path.join(outdir(G), "track.tsv")
    pd.DataFrame(rec).to_csv(out, sep="\t", index=False)
    pickle.dump(dict(p0=p0, u=u, lam=lam, book=book), open(os.path.join(outdir(G), "frame.pkl"), "wb"))
    print(f"written {out}")


def cmd_report(book, G):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    rows = group_rows(book, G)
    cls = {r["root"]: r["cls"] for r in rows}
    d = outdir(G)
    F = pickle.load(open(os.path.join(d, "frame.pkl"), "rb"))
    u = F["u"]
    tr = pd.read_csv(os.path.join(d, "track.tsv"), sep="\t", dtype={"root": str})
    n0 = len(tr)
    tr = tr[(tr.vox >= AXON_VOX) | tr.root.isin(set(cls))]  # axon-sized pieces only (members always kept)
    e1 = np.cross(u, [1.0, 0, 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    q = tr[["x", "y", "z"]].to_numpy() - F["p0"]
    tr["a"], tr["b"] = q @ e1, q @ e2
    lines = []
    say = lambda x="": (print(x), lines.append(x))  # noqa: E731
    short = lambda r: f"{r[-6:]}{'(' + cls[r] + ')' if r in cls else ''}"  # noqa: E731
    say(f"pieces kept: {len(tr)} of {n0} (>= {AXON_VOX} voxels, or a member)")
    say(f"group {G}: {len(rows)} rows; slabs {tr.slab.nunique()} from {tr.height.min():.0f} to {tr.height.max():.0f} um "
        f"above the lamina level; members are marked with their class, other axons by their last 6 digits")
    for k, g in tr.groupby("slab"):
        if len(g) < 2:
            continue
        P = g[["a", "b"]].to_numpy()
        lab = fcluster(linkage(P, "single"), LINK_UM, "distance")
        cl = sorted(set(lab), key=lambda c: -sum(lab == c))
        mem = [r for r in g.root if r in cls]
        say(f"\n{g.height.iloc[0]:5.1f} um: {len(g)} axons, {len(mem)} members, {len(cl)} clusters")
        for c in cl:
            sel = g[lab == c]
            if len(sel) < 2 and not any(r in cls for r in sel.root):
                continue
            cen = sel[["a", "b"]].mean().to_numpy()
            say(f"   [{len(sel):2d} at ({cen[0]:6.1f}, {cen[1]:6.1f})] " + " ".join(short(r) for r in sel.root))
    # column zone: axons seen in >= MIN_SLABS zone slabs, average-linkage clusters of their mean pairwise distance
    z = tr[(tr.height >= ZONE[0]) & (tr.height <= ZONE[1])]
    cnt = z.root.value_counts()
    ax = [r for r in cnt.index if cnt[r] >= MIN_SLABS]
    pos = {r: z[z.root == r].set_index("slab")[["a", "b"]] for r in ax}
    n = len(ax)
    bundles = {}
    if n >= 2:
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                sh = pos[ax[i]].index.intersection(pos[ax[j]].index)
                D[i, j] = D[j, i] = (np.linalg.norm(pos[ax[i]].loc[sh].to_numpy() - pos[ax[j]].loc[sh].to_numpy(), axis=1).mean()
                                     if len(sh) >= 2 else 20.0)
        lab = fcluster(linkage(squareform(D), "average"), CUT_UM, "distance")
        bundles = {ax[i]: int(lab[i]) for i in range(n)}
        say(f"\ncolumn zone {ZONE[0]:.0f}-{ZONE[1]:.0f} um: {n} axons in >= {MIN_SLABS} slabs, {len(set(lab))} bundles "
            f"(average linkage at {CUT_UM} um)")
        for c in sorted(set(lab), key=lambda c: -sum(lab == c)):
            idx = [i for i in range(n) if lab[i] == c]
            cen = np.mean([pos[ax[i]].mean().to_numpy() for i in idx], axis=0)
            say(f"   bundle {c}: {len(idx)} axons at ({cen[0]:6.1f}, {cen[1]:6.1f}): " + " ".join(short(ax[i]) for i in idx))
    # each member: its last slabs vs the zone bundles' centers at the nearest heights
    say("\nmembers: highest slab reached, and the column-zone bundle nearest to where they end")
    out = []
    cen_of = defaultdict(dict)
    for r, c in bundles.items():
        for k, p in pos[r].iterrows():
            cen_of[c].setdefault(k, []).append(p.to_numpy())
    for r in cls:
        m = tr[tr.root == r]
        if not len(m):
            say(f"   {short(r):26s} not seen in any slab")
            out.append(dict(root=r, cls=cls[r], top_um="", bundle="", dist_um=""))
            continue
        top = m.height.max()
        last = m[m.height >= top - 6][["a", "b"]].mean().to_numpy()
        best, bd = "", np.inf
        for c, by in cen_of.items():
            ks = sorted(by)
            k0 = min(ks, key=lambda k: abs(tr[tr.slab == k].height.iloc[0] - top))
            dd = np.linalg.norm(np.mean(by[k0], axis=0) - last)
            if dd < bd:
                best, bd = c, dd
        inz = bundles.get(r, "")
        say(f"   {short(r):26s} up to {top:5.1f} um; " + (f"IN bundle {inz}" if inz else
                                                         f"nearest bundle {best} ({bd:.1f} um)" if best != "" else "no bundle"))
        out.append(dict(root=r, cls=cls[r], top_um=top, bundle=inz or best, dist_um=0 if inz else round(bd, 2)))
    pd.DataFrame(out).to_csv(os.path.join(d, "bundles.tsv"), sep="\t", index=False)
    open(os.path.join(d, "report.txt"), "w").write("\n".join(lines) + "\n")


def cmd_follow(book, G, jump=0.8, gap=2):
    """network-free: follow each member through its breaks. When the member's root ends, the axon-sized piece in the
    next `gap` slabs nearest (<= `jump` um across the axis) to where it would continue (last position + the drift of
    its last slabs) is taken as its continuation, and followed in turn. Prints each chain (root ids, heights) and the
    chains' positions in the column zone, then clusters those positions (single linkage at LINK_UM) and lists pieces
    claimed by two chains. -> follow.tsv"""
    from scipy.cluster.hierarchy import fcluster, linkage
    rows = group_rows(book, G)
    cls = {r["root"]: r["cls"] for r in rows}
    d = outdir(G)
    F = pickle.load(open(os.path.join(d, "frame.pkl"), "rb"))
    u = F["u"]
    tr = pd.read_csv(os.path.join(d, "track.tsv"), sep="\t", dtype={"root": str})
    tr = tr[(tr.vox >= AXON_VOX) | tr.root.isin(set(cls))].copy()
    e1 = np.cross(u, [1.0, 0, 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    q = tr[["x", "y", "z"]].to_numpy() - F["p0"]
    tr["a"], tr["b"] = q @ e1, q @ e2
    H = tr.groupby("slab").height.first().to_dict()
    at = {k: g.set_index("root")[["a", "b"]] for k, g in tr.groupby("slab")}
    nslab = max(H) + 1
    chains, claim = {}, defaultdict(list)
    for m in cls:
        path, r, used = [], m, {m}
        while True:
            ks = sorted(k for k in at if r in at[k].index and (not path or k > path[-1][0]))
            if not ks:
                break
            for k in ks:
                path.append((k, r, *at[k].loc[r].to_numpy()[:2]))
            last = np.array(path[-1][2:])
            drift = (last - np.array(path[-3][2:])) / max(1, path[-1][0] - path[-3][0]) if len(path) >= 3 else np.zeros(2)
            nxt = None
            for k in range(path[-1][0] + 1, min(nslab, path[-1][0] + 1 + gap)):
                if k not in at:
                    continue
                pred = last + drift * (k - path[-1][0])
                c = at[k][~at[k].index.isin(used)]
                if not len(c):
                    continue
                dist = np.linalg.norm(c.to_numpy() - pred, axis=1)
                i = int(np.argmin(dist))
                if dist[i] <= jump:
                    nxt = c.index[i]
                    break
            if nxt is None:
                break
            used.add(nxt)
            r = nxt
        chains[m] = path
        for _, rr, _, _ in path:
            if rr != m:
                claim[rr].append(m)
    out = []
    zk = [k for k in H if ZONE[0] <= H[k] <= 60.0]
    print(f"group {G}: following {len(cls)} members through breaks (jump <= {jump} um within {gap} slabs; pieces >= "
          f"{AXON_VOX} voxels)")
    zpos = {}
    for m, path in chains.items():
        roots = list(dict.fromkeys(r for _, r, _, _ in path))
        top = H[path[-1][0]] if path else None
        z = [(a, b) for k, _, a, b in path if k in zk]
        if z:
            zpos[m] = np.mean(z, axis=0)
        print(f"  {m} ({cls[m]}): {len(roots)} pieces, up to {top if top is not None else '-'} um"
              + (f"; at 45-60 um ({zpos[m][0]:.1f}, {zpos[m][1]:.1f})" if m in zpos else "")
              + ("; via " + " -> ".join(r[-6:] for r in roots[1:]) if len(roots) > 1 else ""))
        for k, r, a, b in path:
            out.append(dict(member=m, cls=cls[m], slab=k, height=H[k], root=r, a=a, b=b))
    shared = {r: ms for r, ms in claim.items() if len(set(ms)) > 1}
    if shared:
        print("pieces claimed by two chains: " + "; ".join(f"{r} by {', '.join(sorted(set(ms)))}" for r, ms in shared.items()))
    if len(zpos) >= 2:
        ms = list(zpos)
        lab = fcluster(linkage(np.array([zpos[m] for m in ms]), "single"), LINK_UM, "distance")
        print(f"\ncolumn-zone positions of the {len(ms)} chains that reach it, single linkage at {LINK_UM} um:")
        for c in sorted(set(lab)):
            sel = [m for m, l in zip(ms, lab) if l == c]
            cen = np.mean([zpos[m] for m in sel], axis=0)
            print(f"  cluster {c} at ({cen[0]:.1f}, {cen[1]:.1f}): " + ", ".join(f"{m} ({cls[m]})" for m in sel))
    pd.DataFrame(out).to_csv(os.path.join(d, "follow.tsv"), sep="\t", index=False)


def main():
    a = [x for x in sys.argv[1:]]
    opt = lambda k, dflt: float(a[a.index(k) + 1]) if k in a else dflt  # noqa: E731
    if a[0] == "track":
        cmd_track(a[1], int(a[2]), opt("--from", -5.0), opt("--to", 90.0), opt("--step", 3.0))
    elif a[0] == "follow":
        cmd_follow(a[1], int(a[2]), opt("--jump", 0.8), int(opt("--gap", 2)))
    elif a[0] == "report":
        cmd_report(a[1], int(a[2]))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
