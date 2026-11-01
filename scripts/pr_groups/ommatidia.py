"""Ommatidium assignment of photoreceptor axons, version 2 (David, 2026-11-01). The fallback is version 1,
groups_refine.py (git commit 311f1c2), with its run figures/refine_20261101 and the PRs tab
figures/sheets/20261101d044156tUTC_FAABopticlobetargets-PRs.xlsx.

Why: an ommatidium's eight axons (six R1-R6, R7, R8) run as one tight column 45-90 um above the lamina, ~1-2 um
across, with neighboring ommatidia ~6 um apart; nearer the lamina they fan out and mix, and their lamina terminals
overlap (a terminal is nearest its own group's terminals for only 31% of R1-R6; scratch/pr_groups/terminal_discrim.py).
Version 1 decided membership per group in whatever stretch it tracked, with a 1.7 um core radius, so neighbors got in
(group 19: 17 rows). Version 2 decides membership in the column zone, for all axons at once, with typed capacity (6
R1-R6 + 1 R7 + 1 R8 per ommatidium); terminals only veto gross outliers; R8 -> R7 synapses help pick the R7.

Steps (files in figures/ommatidia/<run>/; --side right|left, default right):
  roots     CAVE (read-only): the census cutouts (figures/pr_groups/<pr_|pr_left_>bundle_boxes.pkl: 12 x 12 x 1.4 um
            slabs at 45/60/75/90 um above every lattice site, per supervoxel its voxels and centroid) mapped to the
            roots of one timestamp (get_roots, batches of 50k, >= 2 s apart; cached, so a rerun is network-free). Every
            slab on its own: the slabs of neighboring sites overlap but are z-slabs at different heights, so pooling
            them by nominal depth put one axon's pieces from several heights together (2026-11-01: group 4's two-depth
            lines came out 28-89 deg off). Per root and slab the largest axon-sized piece (single linkage BLOB_UM) is
            one point; the points of the ZONE slabs (45, 60 um; at 75/90 um the axons fan out toward their somata)
            give a robust line (points > OUTLIER_UM off dropped) and the local axis (the slabs' site axes).
            -> tracks.tsv, lines.tsv. With --book, also the PRs tab rows' roots at their xyz -> tab_roots.tsv.
  synapses  local synapse snapshot caught up to now (faab.synsnap; its CAVE catch-up keeps the 2 s gap): per line
            root and tab root its output and input counts, lamina / medulla shares and verdict (faab.pr.verdict),
            terminal (median of its lamina output sites) -> features.tsv; synapses among them -> edges.tsv.
  assign    network-free. Candidates: straight lines (<= MAX_ANGLE from the local axis) with an R1-R6 / R7 / R8 verdict.
            Distance of two lines = their mean lateral distance (across the local axis) over the stretch both cover.
            In the column zone bundles nearly touch (centers ~3.3 um apart, members ~1.3 um from their center), so
            distance alone lets the outer axons of two neighbors form a spurious center; every ommatidium is therefore
            started on an R8-like axon (the central axon; R8-like axons within MERGE_UM of another start none), which
            stays fixed in its R8 slot. Then repeat until nothing changes: one assignment of all other candidates to
            typed slots at once (scipy linear_sum_assignment; cost = distance + type penalty, R7 slot cheaper for an
            axon the center's R8 synapses onto; an axon farther than D_MAX from every center stays unassigned); two
            ommatidia merge when duplicates or when their R1-R6 together fit one (never two R8-anchored ones), a second
            R7/R8 is released; a center follows its R1-R6 ring once it has >= 4; leftover R1-R6 that cluster
            (LINK_UM) start centers without an R8. Finally an R1-R6 whose terminal lies more than TERM_MAX from the
            other terminals of an ommatidium with >= TERM_MIN R1-R6 is vetoed and the assignment rerun.
            -> ommatidia.tsv (members by slot, unassigned candidates), centers.tsv.
  compare   network-free: every PRs-tab group against the ommatidia (best match, members kept / added / lost and why),
            and the recovery of the proofread groups 1-8 -> compare.tsv, printed summary.
Read-only; CAVE <= 1 call / 2 s, no fan-out.
Usage: ommatidia.py <roots|synapses|assign|compare> [--run NAME] [--side right|left] [--book PRs.xlsx]"""
import argparse
import datetime
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import caveclient
from fafbseg import flywire
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from faab import ol_meshes, synsnap
from faab.clients.cave import DATASTACK, throttle
from faab.config import fig
from faab.partners.top10_xyz import VOX
from faab.pr import prs_tab as T
from faab.pr.bundles import BIG_VOX, DEPTHS_UM, MIN_HIT_VOX
from faab.pr.lamina_footprint import flat
from faab.pr.side import NEUROPILS, PREFIX
from faab.pr.verdict import mesh_shares, verdict

BLOB_UM = 1.5      # supervoxel centroids of one root in one slab closer than this belong to one piece
# the column zone: at 75/90 um the axons already fan out toward their somata (group 7: members within 2 um of each
# other at 45/60 um, 5-10 um apart at 75/90; group 1's somata are visible at ~71 um), so lines use these depths only
ZONE = (45.0, 60.0)
MIN_SPAN = 5.0     # um of height a line's points must span (else: a point on the local axis, for PRs-tab axons only)
OUTLIER_UM = 2.0   # a point this far off its root's line is dropped (merged piece)
MAX_ANGLE = 25.0   # deg: a candidate's line runs within this angle of the local axis (census rule)
# calibration on groups 1-8 + v1's clean groups (scratch/pr_groups/ommatidia_calibrate.py, 2026-11-01): a member lies a
# median 1.24 um (p95 2.5) from the center line of the others; neighboring centers are a median 3.2 um apart (p5 2.3)
# in the column zone, so bundles nearly touch there
LINK_UM = 2.0      # leftover R1-R6 cluster (average linkage on the line distance) into centers without an R8
D_MAX = 2.5        # um: an axon farther than this from every center stays unassigned
MERGE_UM = 1.5     # centers closer than this are one ommatidium (duplicates: groups 8/58 at 1.1 um)
MERGE_MAX = 4.0    # centers closer than this merge when their members together still fit one ommatidium
NEW_MIN_UM = 2.0   # leftover axons start a new center only this far from every existing center
# R1-R6 terminal -> mean of its group's other terminals, 301 known axons: median 8.4 um, p99 16.9; beyond 20 um only
# the merge errors 648518347570497932 (54 um) and 648518347589178508 (43 um). With fewer than TERM_MIN R1-R6 the
# comparison is too noisy (a proofread group 2 axon was vetoed in a three-axon partial ommatidium).
TERM_MAX, TERM_MIN = 25.0, 5
MIN_OVERLAP = 10.0  # um of shared stretch for a line distance (shorter: the lines are extended up to EXTEND)
EXTEND = 10.0
SLOTS = [("R1-R6", 6), ("R7", 1), ("R8", 1)]
# cost added to the distance when an axon of this verdict takes a slot of this type (None: never)
PENALTY = {"R1-R6": {"R1-R6-like": 0.0, "R1-R6-like (few sites)": 0.0, "R8-like": 0.5, "R7-like": 1.0},
           "R8": {"R8-like": 0.0, "R1-R6-like": 0.5, "R1-R6-like (few sites)": 0.5, "R7-like": None},
           "R7": {"R7-like": 0.0, "R1-R6-like (few sites)": 0.5, "R1-R6-like": 1.0, "R8-like": 1.0}}
R8_TO_R7_BONUS, R8_TO_R7_MIN = 0.7, 2
PR_CLASSES = ("R1-R6", "R7", "R8", "R7/R8", "R1-R6?")


def rdir(a):
    d = fig(f"ommatidia/{a.run}" + ("" if a.side == "right" else "_left"))
    os.makedirs(d, exist_ok=True)
    return d


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# ------------------------------------------------------------------------------------------------------------ roots
def site_axes(side):
    pfx = PREFIX[side]
    known = pd.read_csv(fig(f"{pfx}lattice_known.tsv"), sep="\t")
    pred = pd.read_csv(fig(f"{pfx}lattice_predicted.tsv"), sep="\t")
    ax = {r.cartridge: unit([r.ax_x, r.ax_y, r.ax_z]) for r in known.itertuples()}  # names as in faab.pr.bundles
    ax.update({getattr(r, "site", f"p({r.lat_i},{r.lat_j})"): unit([r.ax_x, r.ax_y, r.ax_z]) for r in pred.itertuples()})
    return ax


def robust_line(P, ax):
    """PCA line through points (oriented along ax), dropping points more than OUTLIER_UM off it (a merged piece) as
    long as >= 2 remain; returns (point, direction, kept mask, residuals)"""
    keep = np.ones(len(P), bool)
    for _ in range(3):
        Q = P[keep]
        p0 = Q.mean(0)
        u = unit(np.linalg.svd(Q - p0)[2][0])
        u = u if u @ ax > 0 else -u
        rel = P - p0
        res = np.linalg.norm(rel - np.outer(rel @ u, u), axis=1)
        new = res <= OUTLIER_UM
        if new.sum() < 2 or (new == keep).all():
            break
        keep = new
    return p0, u, keep, res


def cmd_roots(a):
    d = rdir(a)
    state_path = os.path.join(d, "state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}
    cache = pickle.load(open(fig(f"{PREFIX[a.side]}bundle_boxes.pkl"), "rb"))
    axes = site_axes(a.side)
    mean_ax = unit(np.mean(list(axes.values()), axis=0))
    svs = np.array(sorted({sv for bx in cache["boxes"].values() for sv in bx}), dtype=np.int64)
    m_path = os.path.join(d, "sv2root.npy")
    if "when" not in state:
        state["when"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        json.dump(state, open(state_path, "w"), indent=1)
    when = datetime.datetime.fromisoformat(state["when"])
    roots = np.load(m_path) if os.path.exists(m_path) else np.zeros(len(svs), np.int64)
    todo = np.flatnonzero(roots == 0)
    print(f"{len(svs)} supervoxels in {len(cache['boxes'])} census slabs; {len(todo)} to map to roots at {when:%Y-%m-%d %H:%M} UTC",
          flush=True)
    client, t, B = caveclient.CAVEclient(DATASTACK), time.time() - 10, 50000
    for k in range(0, len(todo), B):
        idx = todo[k:k + B]
        for attempt in range(4):
            t = throttle(t)
            try:
                roots[idx] = np.asarray(client.chunkedgraph.get_roots(svs[idx], timestamp=when)).astype(np.int64)
                break
            except Exception as e:
                print(f"  get_roots batch {k // B} attempt {attempt + 1} failed: {str(e)[:80]}", flush=True)
                time.sleep(15)
        else:
            raise RuntimeError("get_roots keeps failing")
        np.save(m_path, roots)
        if (k // B) % 10 == 9 or k + B >= len(todo):
            print(f"  {min(k + B, len(todo))}/{len(todo)} mapped", flush=True)
    # every slab on its own (slabs of neighboring sites overlap but sit at different heights): per root its pieces
    # (single linkage BLOB_UM), the largest axon-sized piece = one point of the root at that slab's height
    tracks = []
    for (name, dp), bx in cache["boxes"].items():
        if not bx:
            continue
        sv = np.fromiter(bx.keys(), np.int64, len(bx))
        vals = list(bx.values())
        n = np.fromiter((v[0] for v in vals), float, len(vals))
        c = np.array([v[1] for v in vals], float)
        r = roots[np.searchsorted(svs, sv)]
        o = np.argsort(r, kind="stable")
        r, n, c = r[o], n[o], c[o]
        ax = axes.get(name, mean_ax)
        starts = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
        for s0, s1 in zip(starts, np.r_[starts[1:], len(r)]):
            N, Cc = n[s0:s1], c[s0:s1]
            if r[s0] == 0 or N.sum() < MIN_HIT_VOX:
                continue
            if len(Cc) > 3000:  # glia-sized: one piece
                pieces = [(N.sum(), (N[:, None] * Cc).sum(0) / N.sum())]
            else:
                lab = fcluster(linkage(Cc, "single"), BLOB_UM, "distance") if len(Cc) > 1 else np.ones(1, int)
                pieces = [(N[lab == k].sum(), (N[lab == k, None] * Cc[lab == k]).sum(0) / N[lab == k].sum())
                          for k in np.unique(lab)]
            big = [p for p in pieces if p[0] >= MIN_HIT_VOX]
            axon = [p for p in big if p[0] <= BIG_VOX]
            if not axon:
                continue
            v, cc = max(axon, key=lambda p: p[0])
            tracks.append(dict(root=str(r[s0]), site=name, depth=dp, vox=int(v), x=cc[0], y=cc[1], z=cc[2],
                               ax_x=ax[0], ax_y=ax[1], ax_z=ax[2], pieces=len(big), big=int(len(big) > len(axon))))
    tr = pd.DataFrame(tracks)
    tp = os.path.join(d, "tab_roots.tsv")
    listed = set(pd.read_csv(tp, sep="\t", dtype=str).root_now) if os.path.exists(tp) else set()
    lines = []
    for r, g in tr[tr.depth.isin(ZONE)].groupby("root"):
        P = g[["x", "y", "z"]].to_numpy(float)
        A = g[["ax_x", "ax_y", "ax_z"]].to_numpy(float)
        ax = unit((A * g.vox.to_numpy(float)[:, None]).sum(0))
        base = dict(root=r, n_sites=g.site.nunique(), depths=",".join(str(int(x)) for x in sorted(set(g.depth))),
                    ax_x=ax[0], ax_y=ax[1], ax_z=ax[2], vox_med=int(g.vox.median()), multi=int((g.pieces > 1).any()),
                    big=int(g.big.any()))
        if len(P) >= 2 and np.ptp(P @ ax) >= MIN_SPAN:
            p0, u, keep, res = robust_line(P, ax)
            if keep.sum() >= 2 and np.ptp(P[keep] @ ax) >= MIN_SPAN:
                s = (P[keep] - p0) @ u
                lines.append(base | dict(n_points=int(keep.sum()), dropped=int((~keep).sum()), px=p0[0], py=p0[1],
                                         pz=p0[2], ux=u[0], uy=u[1], uz=u[2], smin=float(s.min()), smax=float(s.max()),
                                         angle=float(np.degrees(np.arccos(np.clip(u @ ax, -1, 1)))),
                                         wobble=float(res[keep].max())))
                continue
        if r in listed:  # a PRs-tab axon without a usable stretch: its largest piece as a point on the local axis
            k = int(np.argmax(g.vox.to_numpy()))
            lines.append(base | dict(n_points=1, dropped=0, px=P[k, 0], py=P[k, 1], pz=P[k, 2], ux=ax[0], uy=ax[1],
                                     uz=ax[2], smin=0.0, smax=0.0, angle=0.0, wobble=0.0))
    L = pd.DataFrame(lines)
    tr.to_csv(os.path.join(d, "tracks.tsv"), sep="\t", index=False)
    L.to_csv(os.path.join(d, "lines.tsv"), sep="\t", index=False)
    print(f"roots with an axon-sized piece: {tr.root.nunique()}; with a line in the column zone: "
          f"{int((L.n_points >= 2).sum())} (straight, <= {MAX_ANGLE:.0f} deg: {int(((L.n_points >= 2) & (L.angle <= MAX_ANGLE)).sum())}); "
          f"PRs-tab axons as points: {int((L.n_points == 1).sum())}; points per line p50 "
          f"{L[L.n_points >= 2].n_points.median():.0f}", flush=True)
    if a.book:
        tab_roots(a, d)


def tab_roots(a, d):
    ws = T.load(a.book)[1]
    left = next((r for r in range(1, ws.max_row + 1) if T.text(ws, r, 1).startswith(">>> additional L side")), ws.max_row + 1)
    rows = []
    for b in T.blocks(ws):
        side = "left" if b.title_row > left else "right"
        for r in b.rows:
            rows.append(dict(group=b.group, section=b.section, side=side, row=r, cls=T.get(ws, b, r, "class"),
                             root_id_orig=T.get(ws, b, r, "root_id_orig"), root_id_curr=T.get(ws, b, r, "root_id_curr"),
                             xyz=T.get(ws, b, r, "xyz"), notes=T.get(ws, b, r, "notes")))
    R = pd.DataFrame(rows)
    ok = R.xyz.str.count(",") == 2
    locs = np.array([[float(v) for v in x.split(",")] for x in R.xyz[ok]]) * VOX
    now = np.asarray(flywire.locs_to_segments(locs, coordinates="nm", backend="cloudvolume", dataset=DATASTACK)).astype(np.int64)
    R["root_now"] = ""
    R.loc[ok, "root_now"] = [str(x) for x in now]
    R.loc[R.root_now.isin(["", "0"]), "root_now"] = R.root_id_curr
    R.to_csv(os.path.join(d, "tab_roots.tsv"), sep="\t", index=False)
    print(f"PRs tab: {len(R)} rows, {int(ok.sum())} roots looked up at their xyz; {int((R.root_now != R.root_id_curr).sum())} "
          f"differ from root_id_curr", flush=True)


# --------------------------------------------------------------------------------------------------------- synapses
def ref_um():
    lat = pd.read_csv(fig("pr_lattice_known.tsv"), sep="\t")
    return lat[["lam_x", "lam_y", "lam_z"]].to_numpy(float).mean(0)


def cmd_synapses(a):
    d = rdir(a)
    L = pd.read_csv(os.path.join(d, "lines.tsv"), sep="\t", dtype={"root": str})
    roots = set(L[L.angle <= MAX_ANGLE].root)
    tp = os.path.join(d, "tab_roots.tsv")
    if os.path.exists(tp):
        tab = pd.read_csv(tp, sep="\t", dtype=str).fillna("")
        roots |= set(tab[tab.side == a.side].root_now) - {"", "0"}
    roots = sorted(int(r) for r in roots)
    snap, client = synsnap.Snapshot(log=lambda *_: None), caveclient.CAVEclient(DATASTACK)
    print(f"synapses of {len(roots)} roots (local snapshot, caught up to now)", flush=True)
    now = datetime.datetime.now(datetime.timezone.utc)
    s = snap.synapses(pre=roots, when="now", positions=True, client=client)
    s = s.rename(columns={"pre_root": "pre_pt_root_id", "post_root": "post_pt_root_id",
                          **{f"{p}_{x}": f"{p}_pt_position_{x}" for p in ("pre", "post") for x in "xyz"}})
    pre = flat(s, list(VOX), f"{now:%Y-%m-%d %H:%M} UTC (synapse snapshot)", ref_um())
    pre = pre[pre.pre != pre.post].astype({"pre": str, "post": str}).reset_index(drop=True)
    post = snap.synapses(post=roots, when="now", client=client)
    post = post[post.pre_root != post.post_root]
    la, me = NEUROPILS[a.side]
    P = pre[["pre_x", "pre_y", "pre_z"]].to_numpy(float)
    sh = mesh_shares(pre.pre.to_numpy(), P, la, me)
    region = ol_meshes.assign(P, [la, me]) if len(P) else np.array([])
    idx = pre.groupby("pre").indices
    n_post = post.groupby(post.post_root.astype(str)).size()
    rows = []
    for r in map(str, roots):
        m = idx.get(r, np.empty(0, int))
        npr, npo = len(m), int(n_post.get(r, 0))
        laf = float(sh.la_frac.get(r, 0.0)) if r in sh.index else 0.0
        mef = float(sh.me_frac.get(r, 0.0)) if r in sh.index else 0.0
        lam = m[region[m] == la] if len(m) else m
        Q = P[lam] if len(lam) >= 10 else P[m]
        tx = np.median(Q, axis=0) if len(Q) else [np.nan] * 3
        rows.append(dict(root=r, n_pre=npr, n_post=npo, la_frac=laf, me_frac=mef, verdict=verdict(npr, npo, laf, mef),
                         term_x=tx[0], term_y=tx[1], term_z=tx[2], n_term=int(len(Q))))
    F = pd.DataFrame(rows)
    F.to_csv(os.path.join(d, "features.tsv"), sep="\t", index=False)
    rs = set(map(str, roots))
    E = pre[pre.post.astype(str).isin(rs)].groupby(["pre", "post"]).size().rename("n").reset_index()
    E.to_csv(os.path.join(d, "edges.tsv"), sep="\t", index=False)
    print(f"verdicts: {F.verdict.value_counts().to_dict()}; {len(E)} connected pairs among them", flush=True)


# ----------------------------------------------------------------------------------------------------------- assign
class Lines:
    """candidate axons as lines: point p, direction u, local axis ax, covered stretch [smin, smax] along u"""

    def __init__(self, L):
        self.roots = L.root.tolist()
        self.p = L[["px", "py", "pz"]].to_numpy(float)
        self.u = L[["ux", "uy", "uz"]].to_numpy(float)
        self.ax = L[["ax_x", "ax_y", "ax_z"]].to_numpy(float)
        self.s = L[["smin", "smax"]].to_numpy(float)
        self.i = {r: k for k, r in enumerate(self.roots)}

    def at(self, k, t, ax):
        """position of line k where its axial coordinate (along ax) is t"""
        s = (t - self.p[k] @ ax) / (self.u[k] @ ax)
        return self.p[k] + np.outer(s, self.u[k])

    def trange(self, k, ax):
        e = self.p[k] + np.outer(self.s[k], self.u[k])
        return np.sort(e @ ax)


def line_dist(A, i, B, j):
    """mean lateral distance of line i of A and line j of B over their shared stretch (extended to MIN_OVERLAP)"""
    ax = unit(A.ax[i] + B.ax[j])
    a0, a1 = A.trange(i, ax)
    b0, b1 = B.trange(j, ax)
    lo, hi = max(a0, b0), min(a1, b1)
    if a1 - a0 < 1.0 or b1 - b0 < 1.0:  # a point (axon seen at one depth): compare at its height only
        if lo - hi > EXTEND:
            return np.inf
        t = np.array([a0 if a1 - a0 < 1.0 else b0])
    elif hi - lo < MIN_OVERLAP:
        if lo - hi > EXTEND:  # the stretches are more than EXTEND apart
            return np.inf
        mid = 0.5 * (lo + hi)
        t = np.linspace(mid - MIN_OVERLAP / 2, mid + MIN_OVERLAP / 2, 4)
    else:
        t = np.linspace(lo, hi, 4)
    dv = A.at(i, t, ax) - B.at(j, t, ax)
    lat = dv - np.outer(dv @ ax, ax)
    return float(np.linalg.norm(lat, axis=1).mean())


def fit_center(C, members):
    """a center line through the members' lines (sampled along the shared axis)"""
    ax = unit(C.ax[members].sum(0))
    pts = []
    for k in members:
        t0, t1 = C.trange(k, ax)
        pts.append(C.at(k, np.linspace(t0, t1, 5), ax))
    P = np.vstack(pts)
    p0 = P.mean(0)
    u = unit(np.linalg.svd(P - p0)[2][0]) if len(P) > 2 else ax
    u = u if u @ ax > 0 else -u
    s = (P - p0) @ u
    return dict(p=p0, u=u, ax=ax, s=(float(s.min()), float(s.max())))


def centers_as_lines(cents):
    df = pd.DataFrame([dict(root=f"c{k}", px=c["p"][0], py=c["p"][1], pz=c["p"][2], ux=c["u"][0], uy=c["u"][1],
                            uz=c["u"][2], ax_x=c["ax"][0], ax_y=c["ax"][1], ax_z=c["ax"][2], smin=c["s"][0], smax=c["s"][1])
                       for k, c in enumerate(cents)])
    return Lines(df) if len(df) else None


def slot_kind(v):
    return "R1-R6" if v.startswith("R1-R6") else "R8" if v == "R8-like" else "R7" if v == "R7-like" else ""


def anchor_center(C, k):
    """a center on an R8 axon's own line"""
    return dict(p=C.p[k], u=C.u[k], ax=C.ax[k], s=(float(C.s[k][0]), float(C.s[k][1])), anchor=k, r8=k)


def assign_once(C, cents, verd, r8r7, forbid):
    """one typed-capacity assignment of all candidates; returns {cand index: (center, slot type, distance)}. A center
    anchored on an R8 has that R8 fixed in its R8 slot."""
    CL = centers_as_lines(cents)
    if CL is None:
        return {}
    mid = np.array([c["p"] for c in cents])
    tree = cKDTree(mid)
    fixed = {ce["anchor"]: c for c, ce in enumerate(cents) if ce.get("anchor") is not None}
    cols = [(c, ty) for c, ce in enumerate(cents) for ty, n in SLOTS for _ in range(n)
            if not (ty == "R8" and ce.get("anchor") is not None)]
    col_of = {}
    for j, key in enumerate(cols):
        col_of.setdefault(key, []).append(j)
    N = len(C.roots)
    BIG = 1e6
    M = np.full((N, len(cols) + N), BIG)
    M[np.arange(N), len(cols) + np.arange(N)] = D_MAX  # stay unassigned
    dist = {}
    for i in range(N):
        if i in fixed:
            continue
        for c in tree.query_ball_point(C.p[i], r=25.0):
            if any(np.linalg.norm(cents[c]["p"] - q) < 3.0 for q in forbid.get(i, [])):
                continue
            dd = line_dist(C, i, CL, c)
            if dd > D_MAX:
                continue
            dist[(i, c)] = dd
            r8 = cents[c].get("r8")
            for ty, _ in SLOTS:
                pen = PENALTY[ty].get(verd[i])
                if pen is None or (c, ty) not in col_of:
                    continue
                cost = dd + pen
                if ty == "R7" and r8 is not None and r8r7.get((C.roots[r8], C.roots[i]), 0) >= R8_TO_R7_MIN:
                    cost -= R8_TO_R7_BONUS
                M[i, col_of[(c, ty)]] = cost
    ri, ci = linear_sum_assignment(M)
    out = {i: (c, "R8", 0.0) for i, c in fixed.items()}
    for i, j in zip(ri, ci):
        if i not in fixed and j < len(cols) and M[i, j] < BIG:
            c, ty = cols[j]
            out[i] = (c, ty, dist[(i, c)])
    return out


def core_of(g):
    """members that define a center: the R1-R6 when there are >= 3 (an extra R7/R8 candidate must not pull the center
    off, 2026-11-01 group 6), else the R1-R6 and R8, else all"""
    r16 = [i for i, ty in g if ty == "R1-R6"]
    if len(r16) >= 3:
        return r16
    return [i for i, ty in g if ty in ("R1-R6", "R8")] or [i for i, _ in g]


def fits(C, g):
    ce = centers_as_lines([fit_center(C, core_of(g))])
    return all(line_dist(C, i, ce, 0) <= D_MAX for i, _ in g)


def merge_compatible(C, groups, anchors):
    """merge two ommatidia when their centers are closer than MERGE_UM (duplicates), or closer than MERGE_MAX and
    their R1-R6 together still fit one ommatidium (<= 6, every member within D_MAX of the merged center): a bundle split
    in two halves. Full neighbors (6 + 6) never merge, nor do two ommatidia anchored on an R8 unless duplicates. A
    second R7 or R8 is released (the one nearer the merged center stays; the released axon is placed again in the next
    assignment, a released anchor as an ordinary candidate) - 2026-11-01: groups 2 and 6 had been split by an unlisted
    R7-like / R8-like axon that made the merge look over capacity. Returns (groups, merges); `anchors` is updated."""
    n = 0
    while True:
        cs = [fit_center(C, core_of(g)) for g in groups]
        CL = centers_as_lines(cs)
        pairs = sorted((line_dist(CL, x, CL, y), x, y) for x in range(len(groups)) for y in range(x + 1, len(groups))
                       if np.linalg.norm(cs[x]["p"] - cs[y]["p"]) < 25)
        done = False
        for dd, x, y in pairs:
            if dd >= MERGE_MAX:
                break
            g = groups[x] + groups[y]
            both_anchored = all(any(i in anchors for i, ty in h if ty == "R8") for h in (groups[x], groups[y]))
            if dd >= MERGE_UM and (both_anchored or sum(1 for _, ty in g if ty == "R1-R6") > 6):
                continue
            ce = centers_as_lines([fit_center(C, core_of(g))])
            dist = {i: line_dist(C, i, ce, 0) for i, _ in g}
            if dd >= MERGE_UM and any(dist[i] > D_MAX for i, ty in g if ty == "R1-R6"):
                continue
            # keep the nearest R7 and R8 within reach; the rest are placed again in the next assignment
            for ty in ("R7", "R8"):
                drop = {i for _, i in sorted((dist[i], i) for i, t2 in g if t2 == ty)[1:]}
                drop |= {i for i, t2 in g if t2 == ty and dist[i] > D_MAX}
                g = [m for m in g if m[0] not in drop]
                anchors -= drop
            groups = [h for k, h in enumerate(groups) if k not in (x, y)] + [g]
            n, done = n + 1, True
            break
        if not done:
            return groups, n


def cmd_assign(a):
    d = rdir(a)
    L = pd.read_csv(os.path.join(d, "lines.tsv"), sep="\t", dtype={"root": str})
    F = pd.read_csv(os.path.join(d, "features.tsv"), sep="\t", dtype={"root": str}).set_index("root")
    E = pd.read_csv(os.path.join(d, "edges.tsv"), sep="\t", dtype={"pre": str, "post": str})
    r8r7 = {(x.pre, x.post): int(x.n) for x in E.itertuples(index=False)}
    L = L[(L.angle <= MAX_ANGLE) & L.root.isin(F.index)]
    L = L[L.root.map(lambda r: slot_kind(F.loc[r, "verdict"]) != "")].reset_index(drop=True)
    C = Lines(L)
    verd = [F.loc[r, "verdict"] for r in C.roots]
    kind = [slot_kind(v) for v in verd]
    N = len(C.roots)
    print(f"candidates: {N} straight axons with a photoreceptor verdict ({pd.Series(verd).value_counts().to_dict()})", flush=True)

    tree = cKDTree(C.p)
    D = np.full((N, N), 99.0)
    np.fill_diagonal(D, 0.0)
    for i, j in tree.query_pairs(r=25.0):
        D[i, j] = D[j, i] = min(line_dist(C, i, C, j), 99.0)
    # start: one center on every R8-like axon (the R8 runs at the bundle center; 2026-11-01: distance alone let the
    # outer axons of two touching bundles form a spurious center, e.g. three of group 10's with two of group 22's);
    # R8-like axons closer than MERGE_UM to an earlier one start no center of their own
    anchors = set()
    for k in sorted((k for k in range(N) if kind[k] == "R8"), key=lambda k: -(C.s[k][1] - C.s[k][0])):
        if all(D[k, q] >= MERGE_UM for q in anchors):
            anchors.add(k)
    cents = [anchor_center(C, k) for k in sorted(anchors)]
    print(f"start: {len(cents)} centers on R8-like axons", flush=True)

    forbid, vetoed, prev, seen = {}, {}, None, set()  # forbid: candidate -> centers (midpoints) it was vetoed from
    for it in range(30):
        asg = assign_once(C, cents, verd, r8r7, forbid)
        key = sorted((C.roots[i], c, ty) for i, (c, ty, _) in asg.items())
        cycle = str(key) in seen and key != prev  # back to an earlier state: two states alternate
        seen.add(str(key))
        mem = {}
        for i, (c, ty, _) in asg.items():
            mem.setdefault(c, []).append((i, ty))
        groups, n_merged = merge_compatible(C, [mem[c] for c in sorted(mem)], anchors)
        new = []
        for g in groups:
            r8 = [i for i, ty in g if ty == "R8"]
            if r8 and r8[0] in anchors:
                # the R8 identifies the ommatidium; its center follows the R1-R6 ring once there are >= 4 (an R8 can
                # run up to ~1.5 um off the ring center, group 2)
                n16 = sum(1 for _, ty in g if ty == "R1-R6")
                ce = (fit_center(C, core_of(g)) | {"anchor": r8[0], "r8": r8[0]}) if n16 >= 4 else anchor_center(C, r8[0])
            else:
                ce = fit_center(C, core_of(g)) | {"anchor": None, "r8": r8[0] if r8 else None}
            new.append(ce)
        # leftover R1-R6 / R8 that cluster: new centers
        left = [i for i in range(N) if i not in asg and kind[i] in ("R1-R6", "R8")]
        if len(left) >= 3:
            sub = D[np.ix_(left, left)]
            lab2 = fcluster(linkage(sub[np.triu_indices(len(left), 1)], "average"), LINK_UM, "distance")
            old = centers_as_lines(new)
            for q in np.unique(lab2):
                m = [left[k] for k in np.flatnonzero(lab2 == q)]
                if len(m) < 3:
                    continue
                ce = fit_center(C, m) | {"anchor": None, "r8": None}
                cl = centers_as_lines([ce])
                if old is None or all(line_dist(cl, 0, old, k) >= NEW_MIN_UM for k in range(len(groups))
                                      if np.linalg.norm(new[k]["p"] - ce["p"]) < 25):
                    new.append(ce)
        n_new = len(new) - len(groups)
        changed = key != prev and not cycle
        print(f"  iteration {it + 1}: {len(asg)} assigned to {len(mem)} centers; {len(left)} R1-R6/R8 left over; "
              f"{n_new} new centers, {n_merged} merged", flush=True)
        if not changed and not n_new and not n_merged:
            # terminal veto, then go on until stable again
            veto = terminal_veto(C, F, asg)
            if not veto:
                break
            for i, (c, why) in veto.items():
                forbid.setdefault(i, []).append(cents[c]["p"])
                vetoed[C.roots[i]] = why
            print(f"  terminal veto: {len(veto)} R1-R6 ({', '.join(C.roots[i] for i in veto)})", flush=True)
        prev, cents = key, new
    write_assignment(a, d, C, F, verd, cents, asg, vetoed, D)


def terminal_veto(C, F, asg):
    mem = {}
    for i, (c, ty, _) in asg.items():
        if ty == "R1-R6":
            mem.setdefault(c, []).append(i)
    out = {}
    for c, m in mem.items():
        pts = {i: F.loc[C.roots[i], ["term_x", "term_y", "term_z"]].to_numpy(float) for i in m}
        pts = {i: p for i, p in pts.items() if np.isfinite(p).all()}
        if len(pts) < TERM_MIN:
            continue
        for i, p in pts.items():
            loo = np.mean([q for k, q in pts.items() if k != i], axis=0)
            dd = float(np.linalg.norm(p - loo))
            if dd > TERM_MAX:
                out[i] = (c, f"terminal {dd:.0f} um from the other terminals")
    return out


def write_assignment(a, d, C, F, verd, cents, asg, vetoed, D):
    # number the ommatidia by position (along the main lateral axis) so the ids are stable between runs
    mem = {}
    for i, (c, ty, dd) in asg.items():
        mem.setdefault(c, []).append((i, ty, dd))
    order = sorted(mem, key=lambda c: tuple(np.round(np.mean([C.p[i] for i, _, _ in mem[c]], axis=0), 0)))
    oid = {c: f"om{k + 1:03d}" for k, c in enumerate(order)}
    rows, crow = [], []
    vox = np.array(VOX, float)
    for c in order:
        m = sorted(mem[c], key=lambda x: ({"R1-R6": 0, "R7": 1, "R8": 2}[x[1]], x[2]))
        P = np.array([C.p[i] for i, _, _ in m])
        cen = P.mean(0)
        for i, ty, dd in m:
            f = F.loc[C.roots[i]]
            rows.append(dict(ommatidium=oid[c], slot=ty, root=C.roots[i], verdict=verd[i], dist_um=round(dd, 2),
                             n_pre=int(f.n_pre), n_post=int(f.n_post), me_frac=f.me_frac,
                             xyz=", ".join(str(int(round(v))) for v in C.p[i] * 1000.0 / vox), flag=""))
        n = {ty: sum(1 for _, t2, _ in m if t2 == ty) for ty, _ in SLOTS}
        crow.append(dict(ommatidium=oid[c], n_R16=n["R1-R6"], R7=n["R7"], R8=n["R8"],
                         xyz=", ".join(str(int(round(v))) for v in cen * 1000.0 / vox)))
    # unassigned candidates: nearest ommatidium and why
    cpos = np.array([np.mean([C.p[i] for i, _, _ in mem[c]], axis=0) for c in order]) if order else np.zeros((0, 3))
    ctree = cKDTree(cpos) if len(cpos) else None
    for i in range(len(C.roots)):
        if i in asg:
            continue
        f = F.loc[C.roots[i]]
        near, dn = "", np.nan
        if ctree is not None:
            dn, k = ctree.query(C.p[i])
            near = oid[order[k]]
        why = vetoed.get(C.roots[i]) or "no free slot within reach"
        rows.append(dict(ommatidium="", slot="unassigned", root=C.roots[i], verdict=verd[i], dist_um=np.nan,
                         n_pre=int(f.n_pre), n_post=int(f.n_post), me_frac=f.me_frac,
                         xyz=", ".join(str(int(round(v))) for v in C.p[i] * 1000.0 / vox),
                         flag=f"{why}; nearest {near} ({dn:.1f} um, midpoint)"))
    R, Ce = pd.DataFrame(rows), pd.DataFrame(crow)
    if len(Ce):
        nn = cKDTree(cpos).query(cpos, k=2)[0][:, 1] if len(cpos) > 1 else [np.nan]
        Ce["nearest_um"] = np.round(nn, 1)
    R.to_csv(os.path.join(d, "ommatidia.tsv"), sep="\t", index=False)
    Ce.to_csv(os.path.join(d, "centers.tsv"), sep="\t", index=False)
    full = Ce[(Ce.n_R16 == 6) & (Ce.R7 == 1) & (Ce.R8 == 1)]
    print(f"\n{len(Ce)} ommatidia: {len(full)} complete (6 + R7 + R8); R1-R6 per ommatidium "
          f"{Ce.n_R16.value_counts().sort_index().to_dict()}; with R7 {int(Ce.R7.sum())}, with R8 {int(Ce.R8.sum())}; "
          f"neighbor centers p10/p50/p90 {np.nanpercentile(Ce.nearest_um, [10, 50, 90]).round(1).tolist()} um", flush=True)
    print(f"unassigned candidates: {int((R.slot == 'unassigned').sum())} ({len(vetoed)} by the terminal veto)", flush=True)


# ---------------------------------------------------------------------------------------------------------- compare
def cmd_compare(a):
    d = rdir(a)
    tab = pd.read_csv(os.path.join(d, "tab_roots.tsv"), sep="\t", dtype=str).fillna("")
    tab = tab[(tab.side == a.side) & tab.cls.isin(PR_CLASSES) & ~tab.notes.str.contains("OFF-BUNDLE|SECOND BUNDLE")]
    O = pd.read_csv(os.path.join(d, "ommatidia.tsv"), sep="\t", dtype={"root": str}).fillna("")
    L = pd.read_csv(os.path.join(d, "lines.tsv"), sep="\t", dtype={"root": str})
    F = pd.read_csv(os.path.join(d, "features.tsv"), sep="\t", dtype={"root": str}).set_index("root")
    om_of = {x.root: x.ommatidium for x in O.itertuples() if x.ommatidium}
    slot_of = {x.root: x.slot for x in O.itertuples()}
    members = O[O.ommatidium != ""].groupby("ommatidium").root.apply(set).to_dict()
    lines = set(L.root)
    straight = set(L[L.angle <= MAX_ANGLE].root)
    rows = []
    for g, t in tab.groupby(tab.group.astype(int)):
        mine = set(t.root_now)
        votes = pd.Series([om_of[r] for r in mine if r in om_of]).value_counts()
        best = votes.index[0] if len(votes) else ""
        om = members.get(best, set())
        why = {}
        for r in mine - om:
            if r not in lines:
                why[r] = "not seen 45-90 um above the lamina"
            elif r not in straight:
                why[r] = "not straight there"
            elif r not in F.index or slot_kind(F.loc[r, "verdict"]) == "":
                why[r] = f"verdict {F.loc[r, 'verdict'] if r in F.index else '?'}"
            elif r in om_of:
                why[r] = f"in {om_of[r]} as {slot_of[r]}"
            else:
                why[r] = "unassigned"
        rows.append(dict(group=g, section=t.section.iloc[0], n_tab=len(mine), ommatidium=best, n_om=len(om),
                         kept=len(mine & om), jaccard=round(len(mine & om) / len(mine | om), 2) if mine | om else 0,
                         added=", ".join(sorted(om - mine)), lost="; ".join(f"{r} ({w})" for r, w in sorted(why.items())),
                         other_om=", ".join(sorted(set(votes.index[1:])))))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(d, "compare.tsv"), sep="\t", index=False)
    tg = R[R.section == "targets"]
    print(f"proofread groups 1-8: exact {int((tg.jaccard == 1).sum())}/{len(tg)}; members kept {int(tg.kept.sum())}/"
          f"{int(tg.n_tab.sum())}")
    print(tg[["group", "n_tab", "ommatidium", "n_om", "kept", "jaccard", "added", "lost"]].to_string(index=False))
    ld = R[R.section == "leads"]
    print(f"\nleads: {len(ld)} groups; exact {int((ld.jaccard == 1).sum())}; jaccard >= 0.75 {int((ld.jaccard >= 0.75).sum())}; "
          f"split over >= 2 ommatidia {int((ld.other_om != '').sum())}; ommatidia matched by >= 2 groups "
          f"{int((ld.ommatidium.value_counts() >= 2).sum())}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["roots", "synapses", "assign", "compare"])
    ap.add_argument("--run", default=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d"))
    ap.add_argument("--side", default="right", choices=["right", "left"])
    ap.add_argument("--book", help="PRs tab .xlsx (roots step: look up its rows' roots at their xyz)")
    a = ap.parse_args()
    {"roots": cmd_roots, "synapses": cmd_synapses, "assign": cmd_assign, "compare": cmd_compare}[a.step](a)


if __name__ == "__main__":
    main()
