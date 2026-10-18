"""Refine the photoreceptor group leads 9+ (David, 2026-10-14) with the two-criterion pairing test.

Groups: figures/pr_bundles_pairs_paste.tsv (groups 9-29) and pr_bundles_pairs_paste_extra.tsv (groups 30-95), as handed
over on 2026-10-01/02. xyz is the anchor: every member is re-resolved at its sheet xyz.

Steps (run in this order; every step is resumable and saves into figures/refine/):
  prep      CAVE: roots at all xyz (one lookup), L2 skeletons of changed roots (batches of 40, cache saved per batch),
            presynaptic sites of all members (live_query in chunks of 100 roots).
  terminal  offline: per group the bundle frame (axis = PCA of the pooled R1-R6 nodes, lamina side from presynaptic
            sites, skeleton-median centers along the axis) and the existing terminal-zone analysis
            (faab_pr_bundles_pairs.analyse) on the listed members.
  track     CAVE: 6 x 6 x 6 um mip-1 cutouts every 3 um along each bundle (s_end-150 .. s_end-30, boxes on the
            skeleton centers), all supervoxels mapped to roots (throttled get_roots); EVERY root in a box is kept, so
            axons missing from the lists are found too. One TSV per group.
  refine    bundle core = roots present in >= 50% of the compact slices within 2.2 um of the slice center; listed
            members outside it are flagged off-bundle, unlisted roots inside it are classified (member of another
            group / census PR-like / PR-like by fresh presynaptic sites / non-PR). R8 = the central axon (median radius
            <= 0.7 um); R7 = the long visual fiber (presynaptic sites beyond the lamina zone), else the inner R7/R8-class
            cell on the ring. Ring test: with R7, pairs/names from the angle relative to R7; without R7, pairs from the
            best mirror axis of the six outer axons, R2/R5 = the pair farthest from the axis, end pairs unresolved.
            Terminal test rerun on the refined membership (R7 only as reference). Per cell: confirmed / conflict
            (larger normalized margin wins) / terminal only / ring only. Writes the sheet-format tables
            figures/pr_groups9plus_refined_paste.tsv and _pipes.txt (unpadded ' | ', no dash rows) and
            figures/refine/summary.tsv (printed as markdown).
Read-only; CAVE <= 1 call / 5 s, no fan-out.
Usage: python faab_pr_groups_refine.py <prep|terminal|track|refine> [--groups 9-29]"""
import argparse
import datetime
import itertools
import json
import os
import pickle
import sys
import time
import warnings

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import navis
import caveclient
from fafbseg import flywire

from faab.clients.cave import DATASTACK, throttle
from faab.partners.top10_xyz import VOX
from faab.pr.bundles_pairs import analyse, NAMES, COLS, SCACHE, OUT
from faab.pr.groups_mirror import prep as skprep
from faab.pr.bundles import read_box
from faab.pr.lamina_footprint import flat, retry

warnings.filterwarnings("ignore")
navis.set_pbars(hide=True)
RDIR = os.path.join(OUT, "refine")
os.makedirs(RDIR, exist_ok=True)
PASTES = ["pr_bundles_pairs_paste.tsv", "pr_bundles_pairs_paste_extra.tsv", "refine/new_groups.tsv",  # split-off groups
          "refine/regroup.tsv"]  # 2026-10-17: one group per bundle (groups_regroup.py); 'group N removed' drops a group
SECOND = os.path.join(OUT, "refine", "second_bundles.json")
STATE = os.path.join(RDIR, "state.json")
PRESYN = os.path.join(RDIR, "presyn.tsv")
FRAMES = os.path.join(RDIR, "frames.pkl")
RES1 = np.array([32.0, 32.0, 45.0])
HALF_UM, STEP_UM, ZONE = 3.0, 3.0, 35.0
CORE_R, CORE_FRAC, MIN_VOX = 1.7, 0.8, 300  # pilot 2026-10-15: neighbor-bundle axons reach 1-2.2 um at 50-80% presence
TODAY = datetime.date.today().isoformat()


# ----------------------------------------------------------------------------------------------------------- inputs
def load_groups():
    groups = {}
    for fn in PASTES:
        if not os.path.exists(os.path.join(OUT, fn)):
            continue
        g, header = None, None
        for ln in open(os.path.join(OUT, fn)).read().split("\n"):
            cells = ln.split("\t")
            if ln.startswith("group ") and " removed" in cells[0]:
                groups.pop(int(cells[0].split()[1]), None)
                g = None
                continue
            if ln.startswith("group "):
                g = int(cells[0].split()[1])
                groups[g] = []
                continue
            if cells[0] == "root_id_orig":
                header = cells
                continue
            if not ln.strip() or g is None or header is None:
                continue
            d = dict(zip(header, cells + [""] * (len(header) - len(cells))))
            note = d.get("notes", "")
            kind = ""
            if d["class"] == "R7/R8":
                kind = "R7" if "R7-like" in note else ("R8" if "R8-like" in note else "R?")
            groups[g].append(dict(group=g, root_orig=d["root_id_orig"], cls=d["class"], root_curr=d["root_id_curr"],
                                  xyz=d["xyz"], status=d.get("status", ""), notes=note, kind=kind))
    return groups


def select(groups, spec):
    if not spec:
        return sorted(groups)
    if "," in spec:
        return [g for g in sorted(groups) if g in {int(x) for x in spec.split(",")}]
    lo, hi = (int(x) for x in spec.split("-")) if "-" in spec else (int(spec), int(spec))
    return [g for g in sorted(groups) if lo <= g <= hi]


def load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def save_state(state):
    json.dump(state, open(STATE, "w"), indent=0)


def root_now(r, state):
    v = state.get("roots", {}).get(r["xyz"], "0")
    return v if v != "0" else r["root_curr"]


def load_presyn():
    if not os.path.exists(PRESYN):
        return {}
    d = pd.read_csv(PRESYN, sep="\t", dtype={"pre": str})
    return {k: v[["pre_x", "pre_y", "pre_z"]].to_numpy(float) for k, v in d.groupby("pre")}


def ref_um():
    lat = pd.read_csv(os.path.join(OUT, "pr_lattice_known.tsv"), sep="\t")
    return lat[["lam_x", "lam_y", "lam_z"]].to_numpy(float).mean(0)


def query_presyn(client, roots, state, t, label="members"):
    done = set(state.get("presyn_done", []))
    todo = sorted({str(r) for r in roots} - done)
    print(f"presynaptic sites: {len(todo)} {label} to query", flush=True)
    now = datetime.datetime.now(datetime.timezone.utc)
    for i in range(0, len(todo), 100):
        chunk = [int(x) for x in todo[i:i + 100]]
        t = throttle(t)
        raw = retry(lambda: client.materialize.live_query("synapses_v2", timestamp=now,
                                                          filter_in_dict={"pre_pt_root_id": chunk}))
        if len(raw):
            f = flat(raw, list(VOX), f"{now:%Y-%m-%d %H:%M} UTC", ref_um())
            f = f[f.pre != f.post][["pre", "pre_x", "pre_y", "pre_z", "queried"]]
            f.to_csv(PRESYN, sep="\t", index=False, mode="a", header=not os.path.exists(PRESYN))
        else:
            f = []
        done.update(str(x) for x in chunk)
        state["presyn_done"] = sorted(done)
        save_state(state)
        print(f"  {min(i + 100, len(todo))}/{len(todo)} roots queried, {len(f)} synapses", flush=True)
    return t


def fetch_skeletons(roots, t, label=""):
    sk = pickle.load(open(SCACHE, "rb"))
    need = sorted({int(r) for r in roots if int(r) and int(r) not in sk})
    print(f"skeletons to fetch{label}: {len(need)}", flush=True)
    for b in range(0, len(need), 40):
        t = throttle(t)
        for s in flywire.get_l2_skeleton(need[b:b + 40], dataset=DATASTACK, max_threads=1, omit_failures=True, progress=False):
            sk[int(s.id)] = s
        pickle.dump(sk, open(SCACHE, "wb"))
        print(f"  {min(b + 40, len(need))}/{len(need)} fetched and saved", flush=True)
    return sk, t


# ------------------------------------------------------------------------------------------------------------ prep
def cmd_prep(groups, sel):
    client = caveclient.CAVEclient(DATASTACK)
    t = throttle(time.time())
    state = load_state()
    rows = [r for g in sel for r in groups[g] if r["xyz"]]
    xyzs = sorted({r["xyz"] for r in rows} - set(state.get("roots", {})))
    if xyzs:
        locs = np.array([[float(v) for v in x.split(",")] for x in xyzs]) * VOX
        roots = np.asarray(flywire.locs_to_segments(locs, coordinates="nm", backend="cloudvolume", dataset=DATASTACK)).astype(np.int64)
        state.setdefault("roots", {}).update({x: str(int(r)) for x, r in zip(xyzs, roots)})
        state["roots_when"] = f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M} UTC"
        save_state(state)
    changed = sum(1 for r in rows if root_now(r, state) != r["root_curr"])
    print(f"roots at xyz ({state.get('roots_when', '?')}): {len(rows)} rows, {changed} changed since the tables", flush=True)
    pr = [root_now(r, state) for r in rows if r["cls"] in ("R1-R6", "R7/R8")]
    _, t = fetch_skeletons(pr, t)
    t = query_presyn(client, [r for r in pr if r != "0"], state, t)
    print("prep DONE", flush=True)


# -------------------------------------------------------------------------------------------------------- terminal
def frame_of(outer, inner, sk, presyn):
    """axis/side/s_end from `outer`; slice centers from the skeletons of outer + inner"""
    neu = {r: skprep(sk[int(r)]) for r in outer + inner if int(r) in sk}
    outer = [r for r in outer if r in neu]
    pts = np.vstack([neu[r].nodes[["x", "y", "z"]].values for r in outer])
    c = pts.mean(0)
    axis = np.linalg.svd(pts - c, full_matrices=False)[2][0]
    P = [presyn[r] for r in outer if r in presyn and len(presyn[r])]
    if P:
        sign, how = (1.0 if np.median((np.vstack(P) - c) @ axis) > 0 else -1.0), f"{sum(len(p) for p in P)} presynaptic sites"
    else:
        s = (pts - c) @ axis
        sign, how = (1.0 if pts[s > np.percentile(s, 80), 0].mean() < pts[s < np.percentile(s, 20), 0].mean() else -1.0), "low-x rule"
    axis = axis * sign
    s_end = max(((neu[r].nodes[["x", "y", "z"]].values - c) @ axis).max() for r in outer)
    u = np.cross(axis, [0, 0, 1.0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    centers = {}
    for s in np.arange(np.floor(s_end - 150), s_end, 1.0):
        per = []
        for r in neu:
            p = neu[r].nodes[["x", "y", "z"]].values
            m = np.abs((p - c) @ axis - s) <= 1.5
            if m.sum():
                per.append(p[m].mean(0))
        if len(per) >= 3:
            centers[float(s)] = np.median(per, axis=0)
    return dict(c=c, axis=axis, u=u, v=v, s_end=float(s_end), how=how, centers=centers)


def safe_analyse(outer, inner, sk, presyn):
    for r in outer + inner:  # some L2 skeletons carry an array-valued soma that breaks navis.subset_neuron
        if int(r) in sk:
            sk[int(r)].soma = None
    try:
        return analyse(outer, inner, sk, presyn)
    except Exception as e:  # one odd group must not stop the run
        return {}, f"terminal analysis failed: {str(e)[:80]}"


def members(rows, state, classes=("R1-R6", "R7/R8")):
    seen, out = set(), []
    for r in rows:
        rn = root_now(r, state)
        if r["cls"] in classes and rn not in seen and rn != "0":
            seen.add(rn)
            out.append(rn)
    return out


def frame_fallback(rows, state, sk, presyn, frames):
    """frame for a lead without 3 lamina-ending skeletons: axis of the nearest framed bundle, origin at the median of
    the rows' xyz (census points ~75 um above the lamina), centers from whatever member skeletons pass nearby"""
    pts = np.array([[float(v) for v in r["xyz"].split(",")] for r in rows if r["xyz"]]) * VOX / 1000.0
    c = np.median(pts, axis=0)
    ok = [h for h in frames if "centers" in frames[h] and not frames[h].get("fallback")]
    h = min(ok, key=lambda h: np.linalg.norm(frames[h]["c"] - c))
    axis = frames[h]["axis"]
    u = np.cross(axis, [0, 0, 1.0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    roots = [x for x in members(rows, state, classes=("R1-R6", "R7/R8", "fragment", "other")) if int(x) in sk]
    P = [presyn[r] for r in roots if r in presyn and len(presyn[r]) >= 5]
    lam = float(np.median((np.vstack(P) - c) @ axis)) if P and sum(len(p) for p in P) >= 20 else 75.0
    s_end = lam + 25.0
    neu = {r: skprep(sk[int(r)]) for r in roots}
    centers = {}
    for s in np.arange(np.floor(s_end - 150), s_end, 1.0):
        per = []
        for r in neu:
            p = neu[r].nodes[["x", "y", "z"]].values
            q = p - c
            sp = q @ axis
            lat = np.linalg.norm(q - np.outer(sp, axis), axis=1)
            m = (np.abs(sp - s) <= 1.5) & (lat <= 8.0)
            if m.sum():
                per.append(p[m].mean(0))
        if per:
            centers[float(s)] = np.median(per, axis=0)
    return dict(c=c, axis=axis, u=u, v=v, s_end=float(s_end), centers=centers, fallback=True,
                how=f"axis of group {h}; lamina {'from member synapses' if P else 'assumed 75 um below the xyz points'}")


def cmd_terminal(groups, sel):
    """bundle frames from the lamina-ending members (R1-R6 and R8-kind R7/R8 rows: synapse depth cannot separate R8
    from R1-R6), fallback frames for the rest; the listed-membership terminal analysis is kept for reference only"""
    state = load_state()
    sk = pickle.load(open(SCACHE, "rb"))
    presyn = load_presyn()
    frames = pickle.load(open(FRAMES, "rb")) if os.path.exists(FRAMES) else {}
    deferred = []
    for g in sel:
        rows = groups[g]
        lam = [r for r in members([x for x in rows if x["cls"] in ("R1-R6", "R7/R8") and x["kind"] != "R7"], state) if int(r) in sk]
        allm = [r for r in members(rows, state, classes=("R1-R6", "R7/R8", "fragment", "other")) if int(r) in sk]
        if len(lam) < 3:
            deferred.append(g)
            continue
        F = frame_of(lam, [r for r in allm if r not in lam], sk, presyn)
        F.update(frame_from=f"{len(lam)} lamina-ending members")
        frames[g] = F
        pickle.dump(frames, open(FRAMES, "wb"))
        print(f"group {g}: frame from {len(lam)} lamina-ending members; lamina side from {F['how']}; "
              f"{len(F['centers'])} centers", flush=True)
    for g in deferred:
        F = frame_fallback(groups[g], state, sk, presyn, frames)
        if len(F["centers"]) < 5:
            frames[g] = dict(error=f"fallback frame has {len(F['centers'])} centers")
        else:
            frames[g] = F
        pickle.dump(frames, open(FRAMES, "wb"))
        print(f"group {g}: FALLBACK frame ({F['how']}), {len(F['centers'])} centers", flush=True)
    print("terminal DONE", flush=True)


# ----------------------------------------------------------------------------------------------------------- track
def cmd_track(groups, sel):
    from cloudvolume import CloudVolume
    frames = pickle.load(open(FRAMES, "rb"))
    client = caveclient.CAVEclient(DATASTACK)
    vol = CloudVolume(client.info.segmentation_source(), use_https=True, agglomerate=False, progress=False,
                      fill_missing=True, bounded=False, mip=1)
    half = np.round(HALF_UM * 1000.0 / RES1).astype(int)
    t = throttle(time.time())
    for g in sel:
        out = os.path.join(RDIR, f"track_g{g}.tsv")
        F = frames.get(g, {})
        if os.path.exists(out) or "centers" not in F:
            continue
        t0 = time.time()
        keys = np.array(sorted(F["centers"]))
        S = [s for s in np.arange(F["s_end"] - 150, F["s_end"] - 30, STEP_UM) if len(keys) and np.min(np.abs(keys - s)) <= 1.0]
        boxes = {}
        for s in S:
            ctr = F["centers"][float(keys[np.argmin(np.abs(keys - s))])]
            boxes[float(s)] = (ctr, read_box(vol, ctr, half, RES1))
        svs = sorted({sv for _, b in boxes.values() for sv in b})
        sv2root = {}
        for i in range(0, len(svs), 50000):
            t = throttle(t)
            rt = np.asarray(client.chunkedgraph.get_roots(svs[i:i + 50000])).astype(np.int64)
            sv2root.update(zip(svs[i:i + 50000], rt.tolist()))
        rec = []
        for s, (ctr, b) in boxes.items():
            per = {}
            for sv, (n, cm) in b.items():
                per.setdefault(sv2root[sv], []).append((n, np.array(cm)))
            for rt, lst in per.items():
                w = np.array([n for n, _ in lst], float)
                if w.sum() < MIN_VOX:
                    continue
                cm = np.sum([n * p for n, p in lst], axis=0) / w.sum()
                big = max(lst, key=lambda x: x[0])[1]
                rec.append(dict(group=g, s_box=s, root=rt, vox=int(w.sum()), x=cm[0], y=cm[1], z=cm[2],
                                bx=big[0], by=big[1], bz=big[2], cx=ctr[0], cy=ctr[1], cz=ctr[2]))
        pd.DataFrame(rec).to_csv(out, sep="\t", index=False)
        print(f"group {g}: {len(S)} cutouts, {len(svs)} supervoxels, {len(rec)} root-slices; {time.time() - t0:.0f} s", flush=True)
    print("track DONE", flush=True)


# ---------------------------------------------------------------------------------------------------------- refine
def circ_mean(a_deg):
    a = np.radians(np.asarray(a_deg, float))
    return float(np.degrees(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))))


def wrap(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def matchings(items):
    if not items:
        yield []
        return
    a = items[0]
    for k in range(1, len(items)):
        rest = items[1:k] + items[k + 1:]
        for m in matchings(rest):
            yield [(a, items[k])] + m


def mirror_fit(rel):
    """best mirror axis of six points on a ring (angles in deg): over all 15 pairings and axis angles phi (0-180),
    the cost of a pairing is the mean deviation of its pairs from mirror images about phi. Returns sorted
    [(cost, phi, pairing)]."""
    out = []
    for m in matchings(sorted(rel)):
        costs = [np.mean([abs(wrap(rel[a] + rel[b] - 2 * phi)) for a, b in m]) / 2.0 for phi in np.arange(0, 180, 1.0)]
        k = int(np.argmin(costs))
        out.append((float(costs[k]), float(k), m))
    out.sort(key=lambda x: x[0])
    return out


def ring_test(sl, outer, r7):
    """Ring test on the compact slices (s_box, root, ang). Pairs = the best mirror pairing of the six outer axons
    (angles averaged over the slices relative to R7, or to one outer axon without R7); margin = cost of the second-best
    pairing minus the best (deg). Names: lateral pair (farthest from the axis) = R2/R5; with R7 the end pair on the
    R7 side of the axis = R1/R6, the other R3/R4; without R7 the ends stay unresolved. (The earlier rank-by-angle-
    from-R7 rule mispairs when R7 sits off the axis, as in group 41.)"""
    need = set(outer) | ({r7} if r7 else set())
    comp = [s for s, sub in sl.groupby("s_box") if need <= set(sub.root)]
    if len(comp) < 4:
        return dict(status=f"ring: {len(comp)} complete slices (need >= 4)", pairs={})
    sub = sl[sl.s_box.isin(comp)]
    ang = {s: dict(zip(x.root, x.ang)) for s, x in sub.groupby("s_box")}
    ref = r7 if r7 else outer[0]
    rel = {r: circ_mean([wrap(ang[s][r] - ang[s][ref]) for s in comp]) for r in outer}
    order = sorted(outer, key=lambda r: rel[r])  # cyclic order around the ring
    # the three pairings with trapezoid topology: going around R1..R6, pairs are (R1,R6) and (R3,R4) neighbors and
    # (R2,R5) opposite; rotation i: c = order[i:] + order[:i] -> (c0,c5) neighbors, (c1,c4) opposite, (c2,c3) neighbors
    topo = []
    for i in range(3):
        c = order[i:] + order[:i]
        m = [(c[0], c[5]), (c[1], c[4]), (c[2], c[3])]
        costs = [np.mean([abs(wrap(rel[a] + rel[b] - 2 * phi)) for a, b in m]) / 2.0 for phi in np.arange(0, 180, 1.0)]
        topo.append((float(min(costs)), i, c))
    topo.sort()
    info = {"mirror_fit_err": round(topo[0][0], 1), "mirror_margin": round(topo[1][0] - topo[0][0], 1)}
    if r7:
        # R7 sits at angle 0: the cells flanking its gap are R1/R6 -> rotation starting at the first cell past R7
        k = next((j for j, r in enumerate(order) if rel[r] > 0), 0)
        c = order[k:] + order[:k]
        margin = float(min(rel[c[0]] % 360.0, (-rel[c[5]]) % 360.0))  # how far R7 is from either flanking cell
        names = {(c[0], c[5]): "R1/R6", (c[1], c[4]): "R2/R5", (c[2], c[3]): "R3/R4"}
        info["valid_pairs"] = {frozenset(p) for i in range(3) for cc in [order[i:] + order[:i]]
                               for p in [(cc[0], cc[5]), (cc[1], cc[4]), (cc[2], cc[3])]}
        fit_c = topo[0][2]
        info["mirror_agrees"] = {frozenset(p) for p in [(fit_c[0], fit_c[5]), (fit_c[1], fit_c[4]), (fit_c[2], fit_c[3])]} == \
            {frozenset(p) for p in names}
        how = "R7 gap"
    else:
        # without R7 the angles cannot pick the rotation (validated on groups 1 and 4: the mirror fit chose a wrong
        # rotation in both), so the ring only VALIDATES a pairing: the three trapezoid-topology pairings are returned
        valid = []
        for i in range(3):
            c = order[i:] + order[:i]
            valid.append({frozenset((c[0], c[5])): "R1/R6 or R3/R4", frozenset((c[1], c[4])): "R2/R5",
                          frozenset((c[2], c[3])): "R1/R6 or R3/R4"})
        return dict(status=f"ring without R7: ring-order check ({len(comp)} slices)", pairs={}, valid=valid, margin=np.nan,
                    unit="deg", angles={r: round(rel[r], 1) for r in outer}, order=order, n_slices=len(comp), **info)
    pairs = {}
    for p, nm in names.items():
        pairs[p[0]], pairs[p[1]] = (p[1], nm), (p[0], nm)
    gaps = {nm: round(abs(wrap(rel[p[0]] - rel[p[1]])), 0) for p, nm in names.items()}
    return dict(status=f"ring {'with' if r7 else 'without'} R7, {how} ({len(comp)} slices)", pairs=pairs, margin=margin,
                unit="deg", sides_ok=True, angles={r: round(rel[r], 1) for r in outer}, gaps=gaps,
                fit_err=info["mirror_fit_err"], n_slices=len(comp), **info)


def cmd_refine(groups, sel):
    state = load_state()
    frames = pickle.load(open(FRAMES, "rb"))
    presyn = load_presyn()
    rv = pd.read_csv(os.path.join(OUT, "pr_bundles_roots_verified.tsv"), sep="\t", dtype={"root": str})
    census = dict(zip(rv.root, rv.verdict))
    owner = {}
    for g in groups:
        for r in groups[g]:
            owner.setdefault(root_now(r, state), set()).add(g)
    client = caveclient.CAVEclient(DATASTACK)
    t = throttle(time.time())
    results = {}
    # ---- pass 1: bundle cores
    for g in sel:
        F, path = frames.get(g, {}), os.path.join(RDIR, f"track_g{g}.tsv")
        if "centers" not in F or not os.path.exists(path):
            results[g] = dict(status=F.get("error", "no track"))
            continue
        tr = pd.read_csv(path, sep="\t")
        P = tr[["x", "y", "z"]].to_numpy(float)
        tr["s"] = (P - F["c"]) @ F["axis"]
        tr["pu"], tr["pv"] = P @ F["u"], P @ F["v"]
        listed = set(members(groups[g], state))  # listed photoreceptor rows (R1-R6, R7/R8)
        listed_all = set(members(groups[g], state, classes=("R1-R6", "R7/R8", "fragment", "other")))
        core_set, done_iter = set(listed_all), 0
        for it in range(3):  # slice center from the current core, then recompute the core
            cen = tr[tr.root.astype(str).isin(core_set)].groupby("s_box")[["pu", "pv"]].median()
            tr2 = tr.join(cen, on="s_box", rsuffix="_c").dropna(subset=["pu_c"])
            tr2["r"] = np.hypot(tr2.pu - tr2.pu_c, tr2.pv - tr2.pv_c)
            lst = tr2[tr2.root.astype(str).isin(listed_all)]
            st = lst.groupby("s_box").agg(n=("root", "size"), med=("r", "median"))
            compact = st[(st.n >= 3) & (st.med <= 1.6)].index
            if len(compact) < 3:
                break
            c2 = tr2[tr2.s_box.isin(compact)]
            distal = set(sorted(compact)[: max(1, len(compact) // 2)])  # retina-side half of the compact slices
            stats = c2.groupby("root").agg(frac=("s_box", lambda x: x.nunique() / len(compact)), med_r=("r", "median"),
                                           vox=("vox", "median"),
                                           frac_distal=("s_box", lambda x: len(set(x) & distal) / len(distal)))
            core_set = {str(r) for r, x in stats.iterrows() if x.frac >= CORE_FRAC and x.med_r <= CORE_R}
            done_iter += 1
        if len(compact) < 3 or not done_iter:
            results[g] = dict(status=f"bundle not compact in the tracked stretch ({len(compact)} slices)")
            continue
        c2 = c2.assign(ang=np.degrees(np.arctan2(c2.pv - c2.pv_c, c2.pu - c2.pu_c)), root=c2.root.astype(str))
        # second bundle: >= 3 listed axons outside the core that form their own compact bundle beside it
        offl = [r for r in listed_all - core_set if int(r) in stats.index and stats.loc[int(r), "frac"] >= 0.6]
        second, sep = [], np.nan
        if len(offl) >= 3:
            sub = c2[c2.root.isin(offl)]
            cen2 = sub.groupby("s_box")[["pu", "pv"]].median()
            s2 = sub.join(cen2, on="s_box", rsuffix="_2")
            s2["r2"] = np.hypot(s2.pu - s2.pu_2, s2.pv - s2.pv_2)
            st2 = s2.groupby("root").agg(n=("s_box", "nunique"), med2=("r2", "median"))
            second = [r for r, x in st2.iterrows() if x.med2 <= 1.7 and x.n >= 0.6 * len(compact)]
            both = cen2.join(c2.groupby("s_box")[["pu_c", "pv_c"]].first())
            sep = float(np.median(np.hypot(both.pu - both.pu_c, both.pv - both.pv_c)))
            if len(second) < 3 or sep < 1.5:
                second = []
        results[g] = dict(status="ok", stats=stats, core=core_set, listed=listed, slices=c2, n_compact=len(compact), track=tr2,
                          second=second, second_sep=sep)
    # ---- every core root gets a fresh synapse check (unlisted axons, listed fragment/other rows)
    allcore = sorted({r for g, R in results.items() if R.get("status") == "ok" for r in R["core"]})
    unknown = [r for r in allcore if r not in set(state.get("presyn_done", []))]
    print(f"{len(allcore)} axons in bundle cores; {len(unknown)} need a synapse check", flush=True)
    if unknown:
        t = query_presyn(client, unknown, state, t, label="core axons")
        presyn = load_presyn()
    # ---- pass 2: roles from geometry and synapses (the listed R7/R8 class is not trusted: synapse depth cannot
    # separate R8 from R1-R6, and several leads list 5-8 'R8-like' cells in one bundle)
    acc_all = set()
    for g, R in results.items():
        if R.get("status") != "ok":
            continue
        F, st = frames[g], R["stats"]
        rowcls = {root_now(x, state): x for x in groups[g]}
        npre = {r: len(presyn.get(r, [])) for r in R["core"] | R["listed"]}
        sp = {r: float(np.median((presyn[r] - F["c"]) @ F["axis"])) for r in npre if npre[r] >= 3}
        role = {}
        for r in sorted(R["core"]):
            if r in rowcls:
                x = rowcls[r]
                role[r] = f"listed {x['cls']}" + ("" if x["cls"] in ("R1-R6", "R7/R8") else f" ({npre[r]} presynaptic sites)")
                continue
            other = sorted(owner.get(r, set()) - {g})
            if other:
                role[r] = f"in group {', '.join(map(str, other))}"
            elif float(st.loc[int(r), "frac_distal"]) < 0.6:
                role[r] = (f"joins the bundle only near the lamina ({npre[r]} presynaptic sites; lamina-cell neurite?); not added")
            elif census.get(r, "") in ("R1-R6-like", "R1-R6-like (few sites)", "R7/R8-like"):
                role[r] = f"census {census[r]}"
            elif npre[r] >= 10 and sp.get(r, -1e9) >= F["s_end"] - 50:
                role[r] = f"PR-like by synapses ({npre[r]} presynaptic sites)"
            elif npre[r] < 10:
                role[r] = f"axon with {npre[r]} presynaptic sites (fragment or glia; not added)"
            else:
                role[r] = "non-PR axon (no lamina output; not added)"
        prset = [r for r in R["core"] if role[r].startswith(("listed", "census", "PR-like"))]
        add = [r for r in prset if r not in rowcls]
        acc_all.update(prset)
        lamP = [presyn[r] for r in prset if npre[r] >= 10]
        lam_s = float(np.median((np.vstack(lamP) - F["c"]) @ F["axis"])) if lamP else None
        med = {r: float(st.loc[int(r), "med_r"]) for r in prset}
        lim = (lam_s + 15.0) if lam_s is not None else (F["s_end"] - 10.0)
        r7_how = ""
        r7c = [r for r in prset if npre[r] >= 3 and sp.get(r, -1e9) > lim and float(st.loc[int(r), "frac_distal"]) >= 0.6]
        if not r7c:
            r7c = [r for r in prset if r in rowcls and rowcls[r]["kind"] == "R7"]
            r7_how = "listed R7-like (table)"
        # several candidates: a listed R7/R8-class cell first, then the one reaching farthest past the lamina
        r7 = max(r7c, key=lambda r: (r in rowcls and rowcls[r]["cls"] == "R7/R8", sp.get(r, 0))) if r7c else None
        if r7 and r7 in sp and sp[r7] > lim:
            r7_how = f"output synapses beyond the lamina (median {sp[r7] - (lam_s or 0):.0f} um past the R1-R6 sites)"
        rest = [r for r in prset if r != r7]
        r8 = min(rest, key=lambda r: med[r]) if rest else None
        r8_how = "central axon of the bundle"
        if not (r8 and med[r8] <= 0.7):
            inner_listed = [r for r in rest if r in rowcls and rowcls[r]["cls"] == "R7/R8"]
            r8 = min(inner_listed, key=lambda r: med[r]) if inner_listed else None
            r8_how = f"most central listed R7/R8-class cell (table; {med[r8]:.1f} um from the center, not central)" if r8 else ""
        outer = [r for r in prset if r not in (r7, r8)]
        is_listed_r16 = lambda r: r in rowcls and rowcls[r]["cls"] == "R1-R6"
        outer.sort(key=lambda r: (is_listed_r16(r), npre[r] >= 10, float(st.loc[int(r), "frac"]), -med[r]), reverse=True)
        extras = outer[6:]
        outer = outer[:6]
        # 2026-10-17 (David: groups should come out as 6 outer + R7 + R8). The ring fit cannot choose among candidates
        # (best and second-best options differ by 0-2 deg), so:
        # R7 without medulla synapses = the ring cell with clearly the fewest lamina outputs (R7 passes the lamina with
        # few synapses; 9 of 10 seven-cell groups show one cell at <= half the next-lowest count)
        cands = [r for r in prset if r not in (r7, r8)]
        select_note = ""
        closer = {r: [h for h, Q in results.items() if h != g and Q.get("status") == "ok" and r in Q["core"]
                      and int(r) in Q["stats"].index and float(Q["stats"].loc[int(r), "med_r"]) < med[r]] for r in cands}
        if not r7 and len(cands) >= 6:
            low = sorted([r for r in cands if not closer[r]], key=lambda r: npre[r])
            if len(low) >= 2 and npre[low[0]] <= 70 and npre[low[0]] <= 0.5 * npre[low[1]]:
                r7 = low[0]
                r7_how = (f"few lamina outputs: {npre[r7]} vs >= {npre[low[1]]} for the other ring cells (R7 passes the "
                          f"lamina with few synapses; its medulla part was not found - check that it reaches the medulla)")
                cands.remove(r7)
        # more than six outer candidates: drop those that run closer to another group's center, then cells not listed
        # in this group, then the most peripheral
        if len(cands) > 6:
            rank = sorted(cands, key=lambda r: (bool(closer[r]), r not in rowcls, med[r]))
            outer, extras = rank[:6], rank[6:]
            select_note = (f"six outer axons kept from {len(cands)} candidates (left out first: axons closer to another "
                           f"group's center, then unlisted axons, then the most peripheral)")
        else:
            outer, extras = cands, []
        # a group without a central R8: the most central axon left over by the ring fit, if it runs near the center
        if not r8 and extras:
            e = min(extras, key=lambda r: med[r])
            if med[e] <= 0.9:
                r8, r8_how = e, f"most central axon left over by the ring fit ({med[e]:.1f} um from the center)"
                extras.remove(e)
        extra_why = {}
        for r in extras:
            others = closer.get(r, [])
            if role[r].startswith("in group"):
                extra_why[r] = f"{role[r][3:]} (listed there); left out of this group's six"
            elif others:
                extra_why[r] = f"runs closer to the center of group {', '.join(map(str, others))}; left out of this group's six"
            elif r not in rowcls:
                extra_why[r] = "not listed in this group; seventh lamina-ending axon in the core (proofreading lead)"
            else:
                extra_why[r] = (f"most peripheral of seven lamina-ending axons ({med[r]:.1f} um from the center); "
                                "neighbor-bundle axon or misassigned (proofreading lead)")
        # pieces of missing members: axon-sized partial axons in the core region, in no group's core (top 5)
        piece_leads = []
        if len(outer) < 6 or not r7 or not r8:
            in_cores = {x for h, Q in results.items() if Q.get("status") == "ok" for x in Q["core"]}
            core_vox = float(np.median([st.loc[int(r), "vox"] for r in prset if int(r) in st.index])) if prset else 0.0
            pl = st[(st.frac >= 0.3) & (st.frac < CORE_FRAC) & (st.med_r <= 1.5) & (st.vox >= 0.3 * core_vox)]
            piece_leads = [str(r) for r in pl.sort_values("frac", ascending=False).index
                           if str(r) not in in_cores and str(r) not in R["listed"]][:5]
        # shell: listed lamina-ending axons that co-travel just outside the core (>= 90% of slices, 1.7-2.4 um);
        # admitted only when they exactly fill the missing outer places (otherwise they may be a neighbor bundle)
        shell = [r for r in sorted(R["listed"] - R["core"]) if int(r) in st.index and r not in R["second"]
                 and float(st.loc[int(r), "frac"]) >= 0.9 and CORE_R < float(st.loc[int(r), "med_r"]) <= 2.4
                 and npre.get(r, 0) >= 10 and sp.get(r, 1e9) <= lim]
        shell_added = shell if (len(outer) < 6 and 0 < len(shell) <= 6 - len(outer)) else []
        outer = outer + shell_added
        for r in shell_added:
            role[r] = f"listed, loosely packed ({float(st.loc[int(r), 'med_r']):.1f} um from the center); admitted to complete the ring"
        acc_all.update(shell_added)
        R.update(role=role, add=add, prset=prset, r7=r7, r7_how=r7_how if r7 else "", r8=r8, r8_how=r8_how if r8 else "",
                 outer=outer, extras=extras, shell=shell, shell_added=shell_added,
                 npre=npre, lam_s=lam_s, off=sorted(R["listed"] - R["core"]),
                 select_note=select_note, extra_why=extra_why, piece_leads=piece_leads)
    # ---- skeletons of all core photoreceptors, then the two tests on the refined membership
    sk, t = fetch_skeletons(sorted(acc_all), t, label=" (core photoreceptors)")
    for g, R in results.items():
        if R.get("status") != "ok":
            continue
        outer = [r for r in R["outer"] if int(r) in sk]
        R["ring"] = ring_test(R["slices"], outer, R["r7"]) if len(outer) == 6 else \
            dict(status=f"ring: {len(outer)} outer axons in the core (need 6)", pairs={})
        term_outer = [r for r in outer if R["npre"][r] >= 10]
        inner = [R["r7"]] if R["r7"] and int(R["r7"]) in sk else []
        cells, gnote = safe_analyse(term_outer, inner, sk, presyn) if len(term_outer) >= 3 else ({}, f"{len(term_outer)} outer axons with terminals")
        R["cells"], R["gnote"] = cells, gnote
        R["final"] = group_consensus(R)
        det = []
        st = R["stats"]
        for r in sorted(R["core"] | R["listed"], key=lambda r: float(st.loc[int(r), "med_r"]) if int(r) in st.index else 99):
            cl = cells.get(r, {})
            rp, rn = R["ring"]["pairs"].get(r, ("", ""))
            det.append(dict(group=g, root=r, role=R["role"].get(r, "listed, OFF-BUNDLE"),
                            med_r=round(float(st.loc[int(r), "med_r"]), 2) if int(r) in st.index else None,
                            frac=round(float(st.loc[int(r), "frac"]), 2) if int(r) in st.index else None,
                            n_pre=R["npre"].get(r, len(presyn.get(r, []))),
                            pre_vs_lamina=round(float(np.median((presyn[r] - frames[g]["c"]) @ frames[g]["axis"])) - R["lam_s"], 1)
                            if R["lam_s"] is not None and r in presyn and len(presyn[r]) else None,
                            assigned=("R7" if r == R["r7"] else "R8" if r == R["r8"] else "outer" if r in R["outer"] else
                                      "extra" if r in R["extras"] else "second bundle" if r in R["second"] else ""),
                            ring_angle=R["ring"].get("angles", {}).get(r), ring_partner=rp, ring_name=rn,
                            term_partner=cl.get("terminal_partner", ""), term_name=cl.get("fly_pair", ""),
                            term_margin=round(cl["terminal_score"] - cl["terminal_next"], 2) if cl.get("terminal_partner") else None,
                            verdict=R["final"][r][2] if r in R["outer"] else ""))
        pd.DataFrame(det).to_csv(os.path.join(RDIR, f"details_g{g}.tsv"), sep="\t", index=False)
        print(f"group {g}: core {len(R['core'])} ({len(R['listed'] & R['core'])} listed PR), photoreceptors {len(R['prset'])}, "
              f"off-bundle {len(R['off'])}, added {len(R['add'])}, extras {len(R['extras'])}; R7 {'yes' if R['r7'] else 'no'}, "
              f"R8 {'yes' if R['r8'] else 'no'}; {R['ring']['status']}; terminal: {gnote[:70]}", flush=True)
    sec = json.load(open(SECOND)) if os.path.exists(SECOND) else {}
    for g, R in results.items():
        if R.get("status") == "ok" and R["second"]:
            sec[str(g)] = sorted(R["second"])
        elif str(g) in sec and R.get("status") == "ok":
            sec.pop(str(g))
    json.dump(sec, open(SECOND, "w"), indent=1)
    write_outputs(groups, sel, results, state)


def cmd_split(groups, sel):
    """second bundles -> new groups (numbered after the last existing group) in figures/refine/new_groups.tsv, rows
    copied from the source group with a note; existing new groups are kept (idempotent per source group)"""
    state = load_state()
    sec = json.load(open(SECOND))
    path = os.path.join(OUT, "refine", "new_groups.tsv")
    have = {}
    if os.path.exists(path):
        for g, rows in groups.items():
            for r in rows:
                if "split from group" in r["notes"]:
                    have[int(r["notes"].split("split from group ")[1].split()[0].rstrip(";,"))] = g
    nxt = max(groups) + 1
    lines = open(path).read().rstrip("\n").split("\n") + [""] if os.path.exists(path) else []
    header = "\t".join(["root_id_orig", "class", "root_id_curr", "xyz", "status", "pr_date", "proofreader", "chk_date", "checker",
                        "notes"])
    for src in sorted(int(k) for k in sec):
        if src in have:
            print(f"group {src}: already split into group {have[src]}", flush=True)
            continue
        roots = set(sec[str(src)])
        rows = [r for r in groups[src] if root_now(r, state) in roots]
        lines += [f"group {nxt}", header]
        for r in rows:
            lines.append("\t".join([r["root_orig"], r["cls"], root_now(r, state), r["xyz"], r["status"] or "candidate", "", "", "", "",
                                    r["notes"] + f"; split from group {src} {TODAY} (second bundle beside group {src}'s bundle)"]))
        lines.append("")
        print(f"group {src}: second bundle of {len(rows)} -> new group {nxt}", flush=True)
        nxt += 1
    open(path, "w").write("\n".join(lines) + "\n")
    print("split DONE", flush=True)


def group_consensus(R):
    """per outer cell: (partner, name, verdict, terminal margin, ring margin)"""
    ring = R["ring"]
    if "valid" not in ring:
        return {r: consensus(r, R) for r in R["outer"]}
    out = {}
    term = {r: R["cells"].get(r, {}) for r in R["outer"]}
    T = {frozenset((r, cl["terminal_partner"])) for r, cl in term.items() if cl.get("terminal_partner") in R["outer"]}
    best = max(ring["valid"], key=lambda v: len(set(v) & T))
    for r in R["outer"]:
        cl = term[r]
        tp = cl.get("terminal_partner", "")
        tm = cl.get("terminal_score", np.nan) - cl.get("terminal_next", np.nan) if tp else np.nan
        mine = next(p for p in best if r in p)
        rp = next(iter(mine - {r}))
        if tp and frozenset((r, tp)) in best:
            name = best[mine]
            v = "confirmed (terminal pair fits the ring order" + ("; R2/R5 = the opposite pair)" if name == "R2/R5" else ")")
            out[r] = (tp, name, v, tm, np.nan)
        elif tp:
            out[r] = (tp, cl.get("fly_pair", ""), f"conflict: terminal pair breaks the ring order (ring-consistent partner {rp})", tm, np.nan)
        elif len(set(best) & T) >= 2:  # two terminal pairs fit the ring order: the third pair is fixed by it
            out[r] = (rp, best[mine], "ring completes the pairing (the other two terminal pairs fit the ring order)", tm, np.nan)
        else:
            out[r] = ("", "", "", tm, np.nan)
    return out


def consensus(r, R):
    cl = R["cells"].get(r, {})
    tp, tn = cl.get("terminal_partner", ""), cl.get("fly_pair", "")
    tm = cl.get("terminal_score", np.nan) - cl.get("terminal_next", np.nan) if cl else np.nan
    rp, rn = R["ring"]["pairs"].get(r, ("", ""))
    rm = R["ring"].get("margin", np.nan)
    unit = 20.0 if R["ring"].get("unit") == "deg" else 10.0
    if tp and rp:
        if tp != rp and frozenset((r, tp)) not in R["ring"].get("valid_pairs", {frozenset((r, tp))}):
            # the terminal pair is not a ring-valid pair at all (validated groups: every terminal pair fits)
            strength = "ok" if rm >= 15 else "weak R7 margin"
            return rp, rn, f"conflict, ring wins: terminal pair ({tp}) breaks the ring order; R7 margin {rm:.0f} deg ({strength})", tm, rm
        if tp == rp:
            if tn == rn:
                return tp, tn, "confirmed", tm, rm
            resolved = [x for x in (tn, rn) if x and "or" not in x]
            name = resolved[0] if len(set(resolved)) == 1 else (tn if not np.isnan(tm) and tm / 0.10 >= rm / unit else rn)
            return tp, name, "pair confirmed, names differ" if len(set(resolved)) > 1 else "pair confirmed", tm, rm
        tnorm = (tm / 0.10) if not np.isnan(tm) else 0.0
        rnorm = (rm / unit) * (1.0 if R["ring"].get("sides_ok", True) else 0.5)
        if rnorm > tnorm:
            return rp, rn, f"conflict, ring wins ({rnorm:.1f} vs {tnorm:.1f}; terminal said {tp})", tm, rm
        return tp, tn, f"conflict, terminal wins ({tnorm:.1f} vs {rnorm:.1f}; ring said {rp})", tm, rm
    if tp:
        return tp, tn, "terminal only", tm, rm
    if rp:
        return rp, rn, "ring only", tm, rm
    return "", "", "", tm, rm


def confidence(verdict, tm, rm, unit):
    good_t = not np.isnan(tm) and tm >= 0.10
    good_r = not np.isnan(rm) and rm >= unit
    if verdict.startswith("confirmed") or verdict.startswith("pair confirmed"):
        return "good" if (good_t or good_r) else "ok"
    if verdict in ("terminal only", "ring only"):
        return "ok" if (good_t or good_r) else "weak"
    if verdict.startswith("conflict, ring wins: terminal pair") and "(ok)" in verdict:
        return "ok"
    if verdict.startswith("ring completes"):
        return "ok"
    return "weak" if verdict else ""


PAIR_RANK = {"R1/R6": 0, "R2/R5": 1, "R3/R4": 2, "R1/R6 or R3/R4": 3}


def order_members(rows):
    """David 2026-10-18: a group's rows as R1-R6 sorted by pairing (R1/R6, R2/R5, R3/R4, partners on adjacent rows;
    unpaired R1-R6 after), then R7, then R8, then everything else in its original order (extras, fragments; OFF-BUNDLE
    and SECOND BUNDLE rows last). Rows are sheet rows: class at [1], root at [2], notes at [9], fly_pair at [13],
    partner root at [15]."""
    member = lambda x: "OFF-BUNDLE" not in x[9] and "SECOND BUNDLE" not in x[9]  # notes at [9]
    r16 = [x for x in rows if x[1] == "R1-R6" and member(x)]
    by_root = {x[2]: x for x in r16}
    done, paired = set(), []
    for x in sorted(r16, key=lambda x: (PAIR_RANK.get(x[13], 9), x[2])):
        mate = by_root.get(x[15]) if x[15] else None  # a pair without a fly name yet still sits together (rank 9)
        if x[2] not in done and mate is not None and mate[2] not in done and mate[15] == x[2]:
            paired += [x, mate]
            done |= {x[2], mate[2]}
    rank = {"R7": 0, "R8": 1}
    rest = sorted((x for x in rows if x not in r16), key=lambda x: rank.get(x[1], 2) if member(x) else 3)  # stable
    return paired + [x for x in r16 if x[2] not in done] + rest


def write_outputs(groups, sel, results, state):
    lines, summ = [], []
    for g in sel:
        R = results.get(g, {})
        rows = groups[g]
        orig_of = {root_now(x, state): x["root_orig"] for x in rows}
        xyz_of = {root_now(x, state): x["xyz"] for x in rows}
        out = []
        if R.get("status") != "ok":
            for x in rows:
                out.append([x["root_orig"], x["cls"], root_now(x, state), x["xyz"], x["status"], "", "", "", "", x["notes"] +
                            f"; refine {TODAY}: not refined ({R.get('status', 'not run')})", "", ""] + [""] * 10)
            summ.append(dict(group=g, status=R.get("status", "not run")))
        else:
            st = R["stats"]
            unit = 20.0 if R["ring"].get("unit") == "deg" else 10.0
            fin = R["final"]
            # added axons get an xyz from their largest supervoxel in the most central compact slice
            tr = R["track"]
            for r in R["add"]:
                sub = tr[(tr.root.astype(str) == r) & tr.s_box.isin(R["slices"].s_box.unique())].sort_values("r")
                if len(sub):
                    p = sub.iloc[0][["bx", "by", "bz"]].to_numpy(float)
                    xyz_of[r] = ", ".join(str(int(round(v))) for v in p * 1000.0 / VOX)
            seen = set()
            entries = [(x, root_now(x, state)) for x in rows] + [(None, r) for r in R["add"]]
            for x, r in entries:
                if r in seen:
                    continue
                seen.add(r)
                note = (x["notes"] if x else f"added {TODAY} by bundle tracking ({R['role'][r]})")
                cls = x["cls"] if x else "R1-R6"
                bits = []
                if x and r in R["outer"] and x["cls"] != "R1-R6":
                    cls = "R1-R6"
                    was = x["cls"] + (f", {x['kind']}-like" if x["kind"] else "")
                    bits.append(f"reclassified from {was}: lamina-ending axon on the ring of the bundle")
                if r in R["extras"]:
                    cls = "R1-R6?"
                    bits.append("extra lamina-ending axon in the core beyond six: " +
                                R.get("extra_why", {}).get(r, "not paired (neighbor bundle or second piece?)"))
                if r in R["outer"] and R.get("select_note"):
                    bits.append(R["select_note"])
                if x and r == R["r8"] and x["cls"] != "R7/R8":
                    bits.append(f"listed as {x['cls']} but runs at the center of the bundle")
                if r in R["core"]:
                    bits.append(f"in the bundle core (median {st.loc[int(r), 'med_r']:.1f} um from the center, "
                                f"{100 * st.loc[int(r), 'frac']:.0f}% of {R['n_compact']} compact slices)")
                elif x and r in R["shell_added"]:
                    bits.append(R["role"][r])
                elif x and r in R["second"]:
                    moved = [h for h in groups if h != g and any(root_now(y, state) == r and "split from group" in y["notes"]
                                                                    for y in groups[h])]
                    bits.append(f"SECOND BUNDLE: runs with {len(R['second']) - 1} other listed axons in a compact bundle "
                                f"{R['second_sep']:.1f} um beside this one" +
                                (f"; moved to group {moved[0]}" if moved else " (candidate separate ommatidium)"))
                elif x and x["cls"] in ("R1-R6", "R7/R8"):
                    if int(r) in st.index:
                        bits.append(f"OFF-BUNDLE (median {st.loc[int(r), 'med_r']:.1f} um from the center, "
                                    f"{100 * st.loc[int(r), 'frac']:.0f}% of slices)")
                    else:
                        bits.append("OFF-BUNDLE (not in the tracked stretch)")
                    other = [h for h, Q in results.items() if h != g and Q.get("status") == "ok" and r in Q["core"]]
                    if other:
                        bits.append(f"runs in the core of group {', '.join(map(str, other))}")
                if r == R["r8"]:
                    cls = "R8"
                    bits.append(f"R8 ({R['r8_how']})")
                if r == R["r7"]:
                    cls = "R7"
                    bits.append(f"R7 ({R['r7_how']})")
                partner, name, verdict, tm, rm = fin.get(r, ("", "", "", np.nan, np.nan))
                if verdict:
                    ang = R["ring"].get("angles", {}).get(r)
                    bits.append(f"pairing: {verdict}" + (f"; ring angle {ang:+.0f} deg" + (" vs R7" if R["r7"] else "") if ang is not None else ""))
                note += f"; refine {TODAY}: " + "; ".join(bits)
                pairing = [""] * 10
                if partner:
                    cl = R["cells"].get(r, {})
                    ms, mn, ts = cl.get("mirror_score", np.nan), cl.get("mirror_next", np.nan), cl.get("terminal_score", np.nan)
                    pairing = [NAMES.get(name, ""), name, orig_of.get(partner, partner), partner, xyz_of.get(partner, ""),
                               confidence(verdict, tm, rm, unit), "" if np.isnan(ms) else f"{ms:.2f}",
                               "" if np.isnan(ms) or np.isnan(mn) else f"{ms - mn:.2f}", "" if np.isnan(ts) else f"{ts:.2f}", ""]
                out.append([x["root_orig"] if x else r, cls, r, xyz_of.get(r, ""), x["status"] if x else "candidate",
                            "", "", "", "", note, "", ""] + pairing)
            out = order_members(out)
            v = [R["final"][r][2] for r in R["outer"]]
            summ.append(dict(group=g, status="ok", listed=len(R["listed"]), core_PR=len(R["prset"]),
                             off_bundle=len(set(R["off"]) - set(R["shell_added"]) - set(R["second"])), shell=len(R["shell_added"]),
                             second_bundle=len(R["second"]), added=len(R["add"]), outer=len(R["outer"]), extras=len(R["extras"]),
                             R7=("medulla synapses" if R["r7_how"].startswith("output") else
                                 "few lamina outputs" if R["r7_how"].startswith("few lamina") else "table") if R["r7"] else "-",
                             R8=("central" if R["r8_how"].startswith("central") else
                                 "ring leftover" if R["r8_how"].startswith("most central axon left") else "table") if R["r8"] else "-",
                             members=len(R["outer"]) + bool(R["r7"]) + bool(R["r8"]),
                             selected_by_ring=bool(R.get("select_note")),
                             piece_leads=", ".join(R.get("piece_leads", [])),
                             ring=R["ring"]["status"],
                             confirmed=sum(1 for x in v if x.startswith("confirmed") or x.startswith("pair confirmed")),
                             conflict=sum(1 for x in v if x.startswith("conflict")),
                             single=sum(1 for x in v if x in ("terminal only", "ring only")),
                             names=("full" if R["r7"] else "ends unresolved")))
        lines += [f"group {g}", " | ".join(COLS)] + [" | ".join(str(c) for c in row) for row in out] + [""]
    open(os.path.join(OUT, "pr_groups9plus_refined_pipes.txt"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(OUT, "pr_groups9plus_refined_paste.tsv"), "w").write(
        "\n".join(l.replace(" | ", "\t") for l in lines) + "\n")
    S = pd.DataFrame(summ)
    for col in S.columns:
        if col != "group" and pd.api.types.is_numeric_dtype(S[col]):
            S[col] = S[col].astype("Int64")
    S.to_csv(os.path.join(RDIR, "summary.tsv"), sep="\t", index=False)
    print("\n| " + " | ".join(S.columns) + " |\n|" + "---|" * len(S.columns))
    for r in S.astype(object).where(S.notna(), "").itertuples(index=False):
        print("| " + " | ".join(str(x) for x in r) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "terminal", "track", "refine", "split"])
    ap.add_argument("--groups", default="")
    a = ap.parse_args()
    groups = load_groups()
    sel = select(groups, a.groups)
    print(f"{len(groups)} groups loaded ({min(groups)}-{max(groups)}); this run: {len(sel)}", flush=True)
    {"prep": cmd_prep, "terminal": cmd_terminal, "track": cmd_track, "refine": cmd_refine, "split": cmd_split}[a.step](groups, sel)


if __name__ == "__main__":
    main()
