"""R1-R6 pairing table for the found photoreceptor bundles (the 16 tier-1 bundles of the census link, in link
order) and for LRS's SeaTable cartridges 2-7, in the sheet's row format.

Groups: bundle members from figures/pr_bundles_roots_verified.tsv (class from the synapse verdict: R1-R6 /
R7/R8 / fragment / other; xyz = the axon point 75 um above the lamina); SeaTable right-side cells typed R1-6 /
R7-8 with "cartridge N" in their notes (plus 648518347642151200, the R7-8 cell that the census put in the c3
bundle), xyz = the SeaTable position when the table has one, else a skeleton point. One segment lookup at every
xyz gives root_id_curr (mismatches with the census/SeaTable root are noted); missing L2 skeletons are fetched
once (throttled). Per group the analyses of faab_pr_groups_mirror.py (mirror plane + NBLAST, full skeletons)
and faab_pr_groups_flynames2.py (terminal-zone pairs, fly names) are rerun on the R1-R6 members; the terminal
side is the side carrying the cells' presynaptic sites (fallback: low-x rule). Fly names need three pairs and an
R7/R8 reference; without R7/R8 the two end pairs are left as "R1/R6 or R3/R4".
Outputs: figures/pr_bundles_pairs_paste.tsv (tab) and pr_bundles_pairs_pipes.txt (pipe/dash layout, printed).
Columns (sheet order): root_id_orig class root_id_curr xyz status pr_date proofreader chk_date checker notes link
(blank) pair_type fly_pair partner_root_id_orig partner_root_id_curr partner_xyz confidence mirror_score
mirror_margin terminal_score "manual fly_pair check". Read-only.
"""
from faab.config import REPO
import datetime
import json
import os
import pickle
import re
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
from faab.sources.cave import DATASTACK, throttle
from faab.partners.top10_xyz import VOX
from faab.pr.groups_mirror import prep, reflect, THETAS

warnings.filterwarnings("ignore")
navis.set_pbars(hide=True)
HERE = REPO
OUT = os.path.join(HERE, "figures")
SCACHE = os.path.join(OUT, "pr_partner_skels.pkl")
SERVER = "https://flytable.mrc-lmb.cam.ac.uk/"
LINK_ORDER = [34, 108, 15, 31, 41, 85, 109, 134, 158, 165, 172, 39, 66, 75, 7, 100]  # ranks 1-16 in the census link
EXTRA_C3 = "648518347642151200"  # SeaTable R7-8 cell without a cartridge label that travels in the c3 bundle
ZONE_UM = 35.0
NAMES = {"R2/R5": "lateral", "R1/R6": "near7/8", "R3/R4": "opp7/8", "R1/R6 or R3/R4": "end (no R7/R8 ref)"}
CLASS_OF = {"R1-R6-like": "R1-R6", "R1-R6-like (few sites)": "R1-R6", "R7/R8-like": "R7/R8", "fragment": "fragment", "other": "other"}
COLS = ["root_id_orig", "class", "root_id_curr", "xyz", "status", "pr_date", "proofreader", "chk_date", "checker", "notes", "link",
        "", "pair_type", "fly_pair", "partner_root_id_orig", "partner_root_id_curr", "partner_xyz", "confidence", "mirror_score",
        "mirror_margin", "terminal_score", "manual fly_pair check"]


def _s(v):
    return "" if (v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v)) else str(v).strip()


def vox_str(p_um):
    return ", ".join(str(int(round(v))) for v in (np.asarray(p_um) * 1000.0 / VOX))


def plane_search(neu_dp, neu_nodes, outer, c, u, v):
    """best mirror plane (containing the axis) for the cells in `outer`; returns S (diag nan), theta, normal"""
    orig = navis.NeuronList([neu_dp[r] for r in outer])
    best = None
    for th in THETAS:
        nrm = np.cos(np.radians(th)) * u + np.sin(np.radians(th)) * v
        refl = navis.NeuronList([navis.make_dotprops(reflect(neu_nodes[r], c, nrm), k=5) for r in outer])
        S = navis.nblast(refl, orig, scores="mean", normalized=True, n_cores=1, progress=False).values
        Sd = S.copy()
        np.fill_diagonal(Sd, np.nan)
        score = np.nanmax(Sd, axis=1).mean()
        if best is None or score > best[0]:
            best = (score, th, Sd, nrm)
    return best


def mutual_pairs(S):
    bm = {i: int(np.nanargmax(S[i])) for i in range(len(S))}
    return sorted({tuple(sorted((i, j))) for i, j in bm.items() if bm.get(j) == i})


def analyse(outer, inner, sk, presyn):
    """pairing of the R1-R6 cells `outer` (roots with skeletons); returns per-cell dict and a group note"""
    neu = {r: prep(sk[int(r)]) for r in outer + inner}
    pts = np.vstack([neu[r].nodes[["x", "y", "z"]].values for r in outer])
    c = pts.mean(0)
    axis = np.linalg.svd(pts - c, full_matrices=False)[2][0]
    u = np.cross(axis, [0, 0, 1.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(axis, [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    notes = []
    # ---- mirror analysis on full skeletons
    dps = {r: navis.make_dotprops(neu[r], k=5) for r in outer}
    m_score, m_th, S, m_nrm = plane_search(dps, neu, outer, c, u, v)
    m_pairs = mutual_pairs(S)
    m_part, m_sc, m_next = {}, {}, {}
    for i, j in m_pairs:
        oth = [S[i, k] for k in range(len(outer)) if k not in (i, j)] + [S[j, k] for k in range(len(outer)) if k not in (i, j)]
        m_part[outer[i]], m_part[outer[j]] = outer[j], outer[i]
        m_sc[outer[i]] = m_sc[outer[j]] = float(S[i, j])
        m_next[outer[i]] = m_next[outer[j]] = float(max(oth)) if oth else np.nan
    # ---- terminal zone: orientation from the presynaptic sites (lamina), fallback low-x rule
    s_pool = (pts - c) @ axis
    P = [presyn[r] for r in outer if r in presyn]
    if P:
        s_pre = float(np.median((np.vstack(P) - c) @ axis))
        sign = 1.0 if s_pre > 0 else -1.0
        notes.append(f"terminal side from {sum(len(p) for p in P)} presynaptic sites")
    else:
        x_hi = pts[s_pool > np.percentile(s_pool, 80), 0].mean()
        x_lo = pts[s_pool < np.percentile(s_pool, 20), 0].mean()
        sign = 1.0 if x_hi < x_lo else -1.0
        notes.append("terminal side from the low-x rule (no presynaptic sites)")
    s_end = max((sign * ((neu[r].nodes[["x", "y", "z"]].values - c) @ axis)).max() for r in outer)

    def terminal_subset(n, lamina_only=False):
        """R1-R6: the terminal-most ZONE_UM along the axis. R7/R8 (lamina_only): the same s-range only, so that
        their medulla portion (which folds sideways beyond the lamina) cannot drag the reference position."""
        p = n.nodes[["x", "y", "z"]].values - c
        s = sign * (p @ axis)
        m = (s > s_end - ZONE_UM) & ((s <= s_end + 5.0) if lamina_only else True)
        keep = n.nodes.node_id.values[m]
        if lamina_only:
            return navis.subset_neuron(n, keep, inplace=False) if len(keep) >= 3 else None
        return n if len(keep) < 10 else navis.subset_neuron(n, keep, inplace=False)
    term = {r: terminal_subset(neu[r]) for r in outer}
    for r in inner:
        sub78 = terminal_subset(neu[r], lamina_only=True)
        if sub78 is not None:
            term[r] = sub78
        else:
            notes.append(f"R7/R8 {r} has no nodes in the lamina zone")
    inner = [r for r in inner if r in term]
    tpts = np.vstack([term[r].nodes[["x", "y", "z"]].values for r in outer])
    ct = tpts.mean(0)
    tdps = {r: navis.make_dotprops(term[r], k=5) for r in outer}
    t_score, t_th, St, t_nrm = plane_search(tdps, term, outer, ct, u, v)
    that = np.cross(axis, t_nrm)
    that /= np.linalg.norm(that)

    def pos(r):
        p = term[r].nodes[["x", "y", "z"]].values - ct
        return float((p @ that).mean()), float((p @ t_nrm).mean())
    Pos = {r: pos(r) for r in outer}
    # R7 vs R8 among the inner cells from where they make their output synapses: R7 (long visual fibre) beyond the
    # R1-R6 terminal zone, R8 (short visual fibre) inside it. The near7/8 reference is R7 alone: in the 8 known
    # groups R7 always sits on the R1/R6 side of the terminal cross-section, R8 on the R3/R4 side in 5 of 8, and
    # averaging the two swapped group 4 (David's manual check, 2026-10-01).
    kind = {}
    for r in inner:
        if r in presyn:
            s_pre = float(np.median(sign * ((presyn[r] - c) @ axis)))
            kind[r] = "R7" if s_pre > s_end - 10.0 else "R8"
        else:
            kind[r] = "R?"
    ref = [r for r in inner if kind[r] == "R7"] or [r for r in inner if kind[r] == "R?"] or [r for r in inner if kind[r] == "R8"]
    t78 = float(np.mean([pos(r)[0] for r in ref])) if ref else np.nan
    if ref:
        notes.append("near7/8 reference: " + ", ".join(f"{kind[r]} {r}" for r in ref))
        if kind[ref[0]] != "R7":
            notes.append("no R7 (medulla-depth presynaptic sites) among the R7/R8 cells; reference is " + kind[ref[0]])
    t_pairs = mutual_pairs(St)
    left = [i for i in range(len(outer)) if not any(i in p for p in t_pairs)]
    if len(left) == 2 and len(outer) == 6:
        t_pairs.append((left[0], left[1]))
        notes.append(f"third terminal pair = leftover {outer[left[0]]}/{outer[left[1]]}")
    names = {}
    if len(t_pairs) == 3:  # three mutual pairs: name them; extra cells of a 7-8 cell bundle stay unnamed
        if len(outer) != 6:
            notes.append(f"{len(outer)} R1-R6 cells: fly names from the 3 mutual terminal pairs, {len(outer) - 6} cell(s) unnamed")
        dmean = {p: (abs(Pos[outer[p[0]]][1]) + abs(Pos[outer[p[1]]][1])) / 2 for p in t_pairs}
        lat = max(t_pairs, key=lambda p: dmean[p])
        ends = [p for p in t_pairs if p != lat]
        for k in lat:
            names[k] = "R2/R5"
        if not np.isnan(t78):
            tmean = {p: (Pos[outer[p[0]]][0] + Pos[outer[p[1]]][0]) / 2 for p in ends}
            near = min(ends, key=lambda p: abs(tmean[p] - t78))
            far = [p for p in ends if p != near][0]
            for k in near:
                names[k] = "R1/R6"
            for k in far:
                names[k] = "R3/R4"
            r8s = [r for r in inner if kind[r] == "R8"]
            if r8s and kind[ref[0]] == "R7":
                t8 = float(np.mean([pos(r)[0] for r in r8s]))
                near8 = min(ends, key=lambda p: abs(tmean[p] - t8))
                notes.append("R8 sits on the " + ("same" if near8 == near else "opposite (R3/R4)") + " side as R7")
        else:
            for p in ends:
                for k in p:
                    names[k] = "R1/R6 or R3/R4"
            notes.append("no R7/R8 reference: end pairs not resolved")
    else:
        notes.append(f"{len(t_pairs)} mutual terminal pairs among {len(outer)} R1-R6 cells (fly names need 3)")
    t_part, t_sc, t_next = {}, {}, {}
    for i, r in enumerate(outer):
        pr = next((p for p in t_pairs if i in p), None)
        if pr is not None:
            j = pr[1] if pr[0] == i else pr[0]
            t_part[r] = outer[j]
            t_sc[r] = float(St[i, j])
            oth = [St[i, k] for k in range(len(outer)) if k not in (i, j)]
            t_next[r] = float(np.nanmax(oth)) if oth else np.nan
    cells = {}
    for i, r in enumerate(outer):
        cells[r] = dict(mirror_partner=m_part.get(r, ""), mirror_score=m_sc.get(r, np.nan), mirror_next=m_next.get(r, np.nan),
                        terminal_partner=t_part.get(r, ""), terminal_score=t_sc.get(r, np.nan), terminal_next=t_next.get(r, np.nan),
                        fly_pair=names.get(i, ""))
    for r in inner:
        cells[r] = dict(kind=kind[r])
    agree = sum(1 for r in outer if m_part.get(r) and m_part.get(r) == t_part.get(r))
    notes.insert(0, f"mirror plane {m_th} deg (mean best match {m_score:.2f}), terminal plane {t_th} deg ({t_score:.2f}); "
                    f"mirror pairs {len(m_pairs)}, mirror=terminal partner for {agree}/{len(outer)} cells")
    return cells, "; ".join(notes)


def conf(cell):
    if not cell or not cell["mirror_partner"] or np.isnan(cell["mirror_score"]) or np.isnan(cell.get("terminal_score", np.nan)):
        return ""
    m = cell["mirror_score"] - cell["mirror_next"]
    tm = cell["terminal_score"] - cell["terminal_next"]
    if m >= 0.10 and tm >= 0.10:
        return "good"
    if m < 0.05 or tm < 0:
        return "weak"
    return "ok"


def track_centroids(roots_wanted):
    """per root (int) and depth: voxel-weighted centroid (um) of its supervoxels in the census boxes, >= MIN_HIT voxels"""
    cache = pickle.load(open(os.path.join(OUT, "pr_bundle_boxes.pkl"), "rb"))
    sv2root = cache["sv2root"]
    acc = {}
    for (site, depth), bx in cache["boxes"].items():
        for sv, (n, cen) in bx.items():
            rt = sv2root.get(sv, 0)
            if rt in roots_wanted:
                e = acc.setdefault((rt, depth), [0, np.zeros(3)])
                e[0] += n
                e[1] += n * np.array(cen)
    del cache
    tracks = {}
    for (rt, d), (n, sc) in acc.items():
        if n >= 2000:
            tracks.setdefault(rt, {})[d] = sc / n
    return tracks


def track_distance(members, tracks, far=1000.0):
    """mean centroid distance over shared depths (>= 2 shared), else `far`"""
    n = len(members)
    D = np.full((n, n), far)
    for i in range(n):
        D[i, i] = 0.0
        for j in range(i + 1, n):
            ti, tj = tracks.get(members[i], {}), tracks.get(members[j], {})
            shared = sorted(set(ti) & set(tj))
            if len(shared) >= 2:
                D[i, j] = D[j, i] = float(np.mean([np.linalg.norm(ti[d] - tj[d]) for d in shared]))
    return D


def split_cluster(members, tracks, per_ommatidium=8, min_piece=4):
    """cut a merged cluster into round(n / 8) pieces by average linkage on the track distance; pieces smaller than
    `min_piece` are merged into their nearest sibling. Returns a list of member lists, largest first."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    n = len(members)
    k = max(2, int(round(n / per_ommatidium)))
    D = track_distance(members, tracks)
    lab = fcluster(linkage(squareform(D, checks=False), method="average"), k, criterion="maxclust")
    pieces = {l: [i for i in range(n) if lab[i] == l] for l in set(lab)}
    while len(pieces) > 1:
        small = [l for l, idx in pieces.items() if len(idx) < min_piece]
        if not small:
            break
        l = min(small, key=lambda x: len(pieces[x]))
        others = [m for m in pieces if m != l]
        best = min(others, key=lambda m: np.mean([D[i, j] for i in pieces[l] for j in pieces[m]]))
        pieces[best] += pieces.pop(l)
    return sorted(([members[i] for i in idx] for idx in pieces.values()), key=len, reverse=True)


def extra_groups(rv, cl, census_date):
    """groups 30+: census clusters that are not tier-1 bundles, with >= 4 axons and no known PR; clusters with more
    than 9 axons are split into ommatidium-sized pieces first"""
    new = cl[(cl.n_known == 0) & (cl.n_roots >= 4) & ~cl.index.isin(LINK_ORDER)].copy()
    new["tier"] = np.where(new.spread_um <= 6.0, 2, 3)
    new = new.sort_values(["tier", "n_roots"], ascending=[True, False], kind="stable")
    members = {cid: rv[rv.cluster == cid].sort_values(["verdict", "n_pre"], ascending=[True, False]) for cid in new.index}
    wanted = {int(r) for cid in new.index for r in members[cid].root}
    tracks = track_centroids(wanted)
    groups = []
    for cid in new.index:
        sub = members[cid]
        roots = [int(r) for r in sub.root]
        pieces = split_cluster(roots, tracks) if len(roots) > 9 else [roots]
        for pi, piece in enumerate(pieces, 1):
            part = f", part {pi}/{len(pieces)} after splitting by track distance" if len(pieces) > 1 else ""
            rows = []
            for r in sub[sub.root.isin([str(x) for x in piece])].itertuples():
                note = (f"census cluster {cid} ({int(new.loc[cid, 'n_roots'])} axons, tier {int(new.loc[cid, 'tier'])}{part}), "
                        f"{census_date}: {r.verdict}, {int(r.n_pre)} pre / {int(r.n_post)} post")
                if r.verdict == "other":
                    note += "; probable merge or non-PR" if r.n_post > 100 else "; synapses off the lamina line"
                rows.append(dict(root_src=r.root, cls=CLASS_OF[r.verdict], xyz=r.xyz_ref, status="candidate", notes=note, src="census"))
            groups.append(dict(name=f"cluster {cid} part {pi}", rows=rows))
    for k, g in enumerate(groups, 30):
        g["name"] = f"group {k}"
    print(f"extra groups: {len(new)} clusters -> {len(groups)} groups (30-{29 + len(groups)}); split: "
          f"{int((new.n_roots > 9).sum())} clusters with > 9 axons", flush=True)
    return groups


def seatable_groups(client, t):
    """LRS's SeaTable cartridges 2-7 (right side, typed R1-6 / R7-8, 'cartridge N' in the notes) as groups"""
    import seaserpent as ss
    meta = ss.Table("aedes_main", server=SERVER).to_frame()
    meta.columns = [str(c) for c in meta.columns]
    typ = meta["type"].astype(object).map(_s)
    side = meta["side"].astype(object).map(_s)
    notes_col = meta["notes"].astype(object).map(_s)
    pos_col = next((c for c in meta.columns if "position" in c.lower()), None)
    print(f"SeaTable: {len(meta)} rows; position column: {pos_col}", flush=True)
    pr = meta[typ.isin(["R1-6", "R7-8"]) & (side == "R")].copy()
    pr["cart"] = notes_col[pr.index].str.extract(r"cartridge\s*(\d+)")[0]
    pr["typ"] = typ[pr.index]
    pr["note"] = notes_col[pr.index]
    pr["sv"] = pr["supervoxel_id"].astype(object).map(_s)
    sv_roots = {}
    svs = [int(s) for s in pr.sv if s.isdigit()]
    for attempt in range(3):
        try:
            t = throttle(t)
            rr = np.asarray(client.chunkedgraph.get_roots(svs)).astype(np.int64)
            sv_roots = dict(zip(svs, rr.tolist()))
            break
        except Exception as e:
            print(f"  get_roots attempt {attempt + 1} failed: {str(e)[:100]}", flush=True)
            time.sleep(15)
    pr["root"] = [str(sv_roots.get(int(s), "")) if s.isdigit() else "" for s in pr.sv]

    def pos_of(row):
        if pos_col is None:
            return ""
        v = row[pos_col]
        if isinstance(v, (list, tuple)) and len(v) == 3:
            return ", ".join(str(int(round(float(x)))) for x in v)
        s = _s(v)
        nums = re.findall(r"-?\d+\.?\d*", s)
        return ", ".join(str(int(round(float(x)))) for x in nums[:3]) if len(nums) >= 3 else ""
    groups = []
    for cnum in ["2", "3", "4", "5", "6", "7"]:
        sub = pr[(pr.cart == cnum) & (pr["root"] != EXTRA_C3)]
        rows = []
        for _, r in sub.iterrows():
            rows.append(dict(root_src=r["root"], cls="R1-R6" if r["typ"] == "R1-6" else "R7/R8", xyz=pos_of(r), status="",
                             notes=f"SeaTable {r['typ']}; {r['note']}", src="seatable"))
        if cnum == "3":  # the R7-8 cell whose SeaTable note says cartridge 2 travels in the cartridge-3 bundle (census)
            x = pr[pr["root"] == EXTRA_C3]
            rows.append(dict(root_src=EXTRA_C3, cls="R7/R8", xyz=pos_of(x.iloc[0]) if len(x) else "", status="",
                             notes=f"SeaTable R7-8; {x.iloc[0]['note'] if len(x) else ''}; its axon runs in the cartridge-3 bundle "
                                   f"(census), listed here", src="seatable"))
        if rows:
            groups.append(dict(name=f"SeaTable cartridge {cnum}", rows=rows))
    return groups, t


def main():
    t0 = time.time()
    extra = "--extra" in sys.argv
    suffix = "_extra" if extra else ""
    now = datetime.datetime.now(datetime.timezone.utc)
    rv = pd.read_csv(os.path.join(OUT, "pr_bundles_roots_verified.tsv"), sep="\t", dtype={"root": str, "known": str, "seatable": str})
    census_date = str(rv.queried.iloc[0])[:10] if "queried" in rv.columns else f"{now:%Y-%m-%d}"  # date of the synapse counts
    client = caveclient.CAVEclient(DATASTACK)
    t = throttle(time.time() - 10)
    groups = []  # numbering continues David's sheet: groups 1-8 exist; SeaTable cartridges come next, then the census bundles
    bundle_groups = []
    if extra:
        cl = pd.read_csv(os.path.join(OUT, "pr_bundles_clusters.tsv"), sep="\t", index_col=0)
        groups = extra_groups(rv, cl, census_date)
    for rank, cid in enumerate([] if extra else LINK_ORDER, 1):
        sub = rv[rv.cluster == cid].sort_values(["verdict", "n_pre"], ascending=[True, False])
        rows = []
        for r in sub.itertuples():
            cls = CLASS_OF[r.verdict]
            note = f"census bundle {cid} (link rank {rank}), {census_date}: {r.verdict}, {int(r.n_pre)} pre / {int(r.n_post)} post"
            if r.verdict == "other":
                note += "; probable merge or non-PR" if r.n_post > 100 else "; synapses off the lamina line"
            rows.append(dict(root_src=r.root, cls=cls, xyz=r.xyz_ref, status="candidate", notes=note, src="census"))
        bundle_groups.append(dict(name=f"bundle {cid}", rows=rows))

    if not extra:
        seatable, t = seatable_groups(client, t)
        groups = seatable + bundle_groups
        for k, g in enumerate(groups, 9):  # group 9 = first SeaTable cartridge, then the census bundles in link order
            g["name"] = f"group {k}"

    # ---- skeletons first (needed for xyz fallback); fetched in batches with the cache saved after each batch, so a run
    # that hits the 10-minute background limit keeps its progress (rerun with --fetch-only until none are missing)
    sk = pickle.load(open(SCACHE, "rb"))
    need = sorted({int(r["root_src"]) for g in groups for r in g["rows"] if r["root_src"] and int(r["root_src"]) not in sk})
    print(f"skeletons to fetch: {len(need)}", flush=True)
    for b in range(0, len(need), 40):
        t = throttle(t)
        for s in flywire.get_l2_skeleton(need[b:b + 40], dataset=DATASTACK, max_threads=1, omit_failures=True, progress=False):
            sk[int(s.id)] = s
        pickle.dump(sk, open(SCACHE, "wb"))
        print(f"  skeletons {min(b + 40, len(need))}/{len(need)} fetched and saved, {time.time() - t0:.0f} s", flush=True)
    if "--fetch-only" in sys.argv:
        print("fetch-only run: DONE", flush=True)
        return

    # ---- xyz anchors: candidate points per root (census centroid at 75 um, then the largest supervoxels' centroids
    # of that root at 75/60/90/45 um from the census cache, then a skeleton node); a point is accepted when the
    # segment at it IS the root (one lookup per round, at most 4 rounds)
    want = {r["root_src"] for g in groups for r in g["rows"] if r["root_src"]}
    cands = {w: [] for w in want}
    for g in groups:
        for r in g["rows"]:
            if r["xyz"]:
                cands[r["root_src"]].append(r["xyz"])
    xyz_census = {r.root: r.xyz_ref for r in rv.itertuples()}
    for w in want:
        if w in xyz_census and xyz_census[w] not in cands[w]:
            cands[w].append(xyz_census[w])
    cache = pickle.load(open(os.path.join(OUT, "pr_bundle_boxes.pkl"), "rb"))
    sv2root = cache["sv2root"]
    want_int = {int(w) for w in want}
    per_root = {}
    for (site, depth), bx in cache["boxes"].items():
        for sv, (n, cen) in bx.items():
            rt = sv2root.get(sv, 0)
            if rt in want_int:
                per_root.setdefault(rt, {})[sv] = (depth, n, cen)
    del cache
    for rt, d in per_root.items():
        order = sorted(d.values(), key=lambda x: ({75.0: 0, 60.0: 1, 90.0: 2, 45.0: 3}.get(x[0], 4), -x[1]))
        for depth, n, cen in order[:6]:
            p = vox_str(cen)
            if p not in cands[str(rt)]:
                cands[str(rt)].append(p)
    for w in want:
        if int(w) in sk:
            nodes = sk[int(w)].nodes[["x", "y", "z"]].values.astype(float)
            cands[w].append(vox_str(nodes[len(nodes) // 2] / 1000.0))
    # anchors already resolved by an earlier run of this mode are reused (phases must each fit the 10-min limit)
    anchors_path = os.path.join(OUT, f"pr_bundles_anchors{suffix}.json")
    saved = json.load(open(anchors_path)) if os.path.exists(anchors_path) else {}
    resolved = {w: saved["resolved"][w] for w in want if w in saved.get("resolved", {})}
    first_hit = {w: saved["first_hit"][w] for w in want if w in saved.get("first_hit", {})}
    tried = {w: (len(cands[w]) if w in first_hit else 0) for w in want}  # a root that failed every point before is not retried
    print(f"anchors reused from {anchors_path}: {len(resolved)} resolved, {len(first_hit)} unresolved", flush=True)
    for rnd in range(4):
        todo = [w for w in want if w not in resolved and tried[w] < len(cands[w])]
        if not todo:
            break
        locs = np.array([[float(v) for v in cands[w][tried[w]].split(",")] for w in todo]) * VOX
        t = throttle(t)
        roots_at = np.asarray(flywire.locs_to_segments(locs, coordinates="nm", backend="cloudvolume", dataset=DATASTACK)).astype(np.int64)
        for w, ra in zip(todo, roots_at):
            if str(int(ra)) == w:
                resolved[w] = cands[w][tried[w]]
            else:
                first_hit.setdefault(w, str(int(ra)))  # what the first point hit instead
                tried[w] += 1
        print(f"  lookup round {rnd + 1}: {len(todo)} points, {sum(1 for w in todo if w in resolved)} resolved", flush=True)
    json.dump({"resolved": resolved, "first_hit": first_hit, "when": f"{now:%Y-%m-%d %H:%M} UTC"}, open(anchors_path, "w"), indent=1)
    if "--anchors-only" in sys.argv:
        print(f"anchors-only run: {len(resolved)} resolved, {len(want) - len(resolved)} unresolved; DONE", flush=True)
        return
    changed = 0
    for g in groups:
        for r in g["rows"]:
            w = r["root_src"]
            if not w:
                r["root_curr"] = ""
                continue
            if w in resolved:
                r["xyz"] = resolved[w]
                r["root_curr"] = w
            else:  # no point resolves to this root any more: it was edited; keep the first point and the root found there
                r["xyz"] = cands[w][0] if cands[w] else ""
                r["root_curr"] = first_hit.get(w) or w
                changed += 1
                r["notes"] += f"; root changed since the census: segment at xyz is now {r['root_curr']}"
    print(f"segment lookup at {now:%Y-%m-%d %H:%M} UTC: {len(want)} cells, {len(resolved)} anchored on their own root, {changed} changed", flush=True)
    need2 = sorted({int(r["root_curr"]) for g in groups for r in g["rows"] if r["root_curr"] and int(r["root_curr"]) not in sk})
    if need2:
        t = throttle(t)
        for s in flywire.get_l2_skeleton(need2, dataset=DATASTACK, max_threads=1, omit_failures=True, progress=False):
            sk[int(s.id)] = s
        pickle.dump(sk, open(SCACHE, "wb"))

    # ---- presynaptic positions (um) per current root
    presyn = {}
    for path in ("pr_bundles_synapses_raw_pre.tsv", "pr_metu_down_synapses.tsv"):
        p = os.path.join(OUT, path)
        if os.path.exists(p):
            d = pd.read_csv(p, sep="\t", dtype={"pre_pt_root_id": str})
            for rid, sub in d.groupby("pre_pt_root_id"):
                presyn.setdefault(rid, np.array([[float(v) for v in str(s).strip("[]").split()] for s in sub.pre_pt_position]) / 1000.0)

    # ---- analyses
    lines_tsv, lines_pipe = [], []
    acache_path = os.path.join(OUT, f"pr_bundles_pairs_cache{suffix}.pkl")  # per-group analysis results, saved as they finish
    acache = pickle.load(open(acache_path, "rb")) if os.path.exists(acache_path) else {}
    for g in groups:
        seen = set()
        for r in g["rows"]:  # two rows resolving to one root (fragments merged since the census): analyse once
            if r["root_curr"] in seen and r["root_curr"]:
                r["dup"] = True
                r["notes"] += "; same root as another row of this bundle"
            seen.add(r["root_curr"])
        outer = [r["root_curr"] for r in g["rows"] if r["cls"] == "R1-R6" and r["root_curr"] and int(r["root_curr"]) in sk and not r.get("dup")]
        inner = [r["root_curr"] for r in g["rows"] if r["cls"] == "R7/R8" and r["root_curr"] and int(r["root_curr"]) in sk and not r.get("dup")]
        key = (tuple(sorted(outer)), tuple(sorted(inner)))
        if key in acache:
            cells, gnote = acache[key]
        else:
            cells, gnote = ({}, "fewer than 3 R1-R6 cells with skeletons") if len(outer) < 3 else analyse(outer, inner, sk, presyn)
            acache[key] = (cells, gnote)
            pickle.dump(acache, open(acache_path, "wb"))
        xyz_of = {r["root_curr"]: r["xyz"] for r in g["rows"]}
        src_of = {r["root_curr"]: r["root_src"] for r in g["rows"]}
        print(f"{g['name']}: {gnote}", flush=True)
        out_rows = []
        for r in g["rows"]:
            cl = cells.get(r["root_curr"]) if not r.get("dup") else None
            note = r["notes"]
            pairing = [""] * 10
            if cl and "kind" in cl:
                note += {"R7": "; output synapses beyond the R1-R6 terminal zone (R7-like, long visual fibre)",
                         "R8": "; output synapses within the R1-R6 terminal zone (R8-like, or an R7 without its medulla part)",
                         "R?": "; no presynaptic sites to tell R7 from R8"}[cl["kind"]]
                cl = None
            if cl:
                partner = cl["mirror_partner"]
                pc = conf(cl)
                if partner:
                    q = cells.get(partner)
                    rank_ = {"": -1, "weak": 0, "ok": 1, "good": 2}
                    pc = min([pc, conf(q)], key=lambda x: rank_[x]) if q else pc
                    if cl["terminal_partner"] and cl["terminal_partner"] != partner:
                        note += f"; terminal-zone partner differs ({cl['terminal_partner']})"
                else:
                    note += "; no mutual mirror partner"
                pairing = [NAMES.get(cl["fly_pair"], ""), cl["fly_pair"], src_of.get(partner, partner), partner, xyz_of.get(partner, ""), pc,
                           "" if np.isnan(cl["mirror_score"]) else f"{cl['mirror_score']:.2f}",
                           "" if np.isnan(cl["mirror_score"]) or np.isnan(cl["mirror_next"]) else f"{cl['mirror_score'] - cl['mirror_next']:.2f}",
                           "" if np.isnan(cl["terminal_score"]) else f"{cl['terminal_score']:.2f}", ""]
            out_rows.append([r["root_src"], r["cls"], r["root_curr"], r["xyz"], r["status"], "", "", "", "", note, "", ""] + pairing)
        lines_tsv += [g["name"] + "\t" * (len(COLS) - 1), "\t".join(COLS)] + ["\t".join(x) for x in out_rows] + [""]
        data = [COLS] + out_rows
        w = [max(len(row[i]) for row in data) for i in range(len(COLS))]
        lines_pipe += [g["name"], " | ".join(f"{COLS[i]:<{w[i]}}" for i in range(len(COLS)))]  # no dash row (David, 2026-10-02)
        lines_pipe += [" | ".join(f"{row[i]:<{w[i]}}" for i in range(len(COLS))) for row in out_rows] + [""]
    open(os.path.join(OUT, f"pr_bundles_pairs_paste{suffix}.tsv"), "w").write("\n".join(lines_tsv) + "\n")
    open(os.path.join(OUT, f"pr_bundles_pairs_pipes{suffix}.txt"), "w").write("\n".join(lines_pipe) + "\n")
    print("\n".join(lines_pipe))
    print(f"DONE in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
