"""One group per bundle (David, 2026-10-17): find sets of refined groups 9+ whose bundle cores share axons (duplicate
or overlapping leads), cluster each set's axons into compact bundles from the saved tracks, and write a membership
override (figures/refine/regroup.tsv) that groups_refine.py reads after its other group tables.

Sets: groups linked when their cores (details_g*.tsv rows with med_r <= 1.7 um and frac >= 0.8) share >= 4 axons, or
>= 2 that are at least half of the smaller core. Sets with a SeaTable/FlyTable group, or with a member point within
4 um of a group 1-8 member, are only reported (David resolves those) - see --flag-only.
Bundles: positions come from the track points of all the set's groups, projected into the frame of the set's lowest
group (3-um depth bins). Each group's center = the median of its core axons per depth bin. Groups whose centers stay
within --same um (median over shared depths) are one bundle: the lowest number keeps it, the others are removed
('group N removed' lines). (Pairwise axon clustering was tried first and rejected: it split a wide single bundle and
joined two adjacent ones.) Every axon of the set goes to the bundle whose center it runs closest to, if within the
1.7-um core radius; listed axons beyond every core stay with their own group if it survives, else go to the bundle
that absorbed their group (refine marks them OFF-BUNDLE). A root listed in two groups of the set is written once.

Usage: python groups_regroup.py [--same 1.0] [--dry-run]     Network-free; run groups_refine.py terminal / track /
refine for the printed groups afterwards (move their old track files aside first)."""
import argparse
import datetime
import glob
import itertools
import os
import pickle
import re
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd

from faab.pr.bundles_pairs import OUT

RDIR = os.path.join(OUT, "refine")
TODAY = datetime.date.today().isoformat()
VOX_UM = np.array([16.0, 16.0, 45.0]) / 1000.0
PASTES = ["pr_bundles_pairs_paste.tsv", "pr_bundles_pairs_paste_extra.tsv", "refine/new_groups.tsv"]


def load_rows():
    """group -> list of row dicts: the same tables groups_refine.py reads, including earlier regroup overrides"""
    groups = {}
    for fn in PASTES + ["refine/regroup.tsv"]:
        p = os.path.join(OUT, fn)
        if not os.path.exists(p):
            continue
        g, header = None, None
        for ln in open(p).read().split("\n"):
            c = ln.split("\t")
            if ln.startswith("group ") and " removed" in c[0]:
                groups.pop(int(c[0].split()[1]), None)
                g = None
                continue
            if ln.startswith("group "):
                g = int(c[0].split()[1])
                groups[g] = []
                continue
            if c[0] == "root_id_orig":
                header = c
                continue
            if ln.strip() and g is not None and header is not None:
                groups[g].append(dict(zip(header, c + [""] * (len(header) - len(c)))))
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--same", type=float, default=1.0, help="groups whose centers stay within this distance (um) are "
                                                             "the same bundle")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--flag-only", default="", help="extra comma-separated groups whose sets are only reported")
    ap.add_argument("--sets", default="", help="test: semicolon-separated sets to cluster instead, e.g. '9;10;11,12' "
                                                 "(implies --dry-run, no flags)")
    ap.add_argument("--allow", default="", help="comma-separated groups David approved despite a flag")
    ap.add_argument("--only", default="", help="comma-separated groups: handle only the sets containing them")
    a = ap.parse_args()
    if a.sets:
        a.dry_run = True
    state = __import__("json").load(open(os.path.join(RDIR, "state.json")))
    roots_at = state.get("roots", {})
    rows = load_rows()
    now = lambda r: (roots_at.get(r["xyz"], "0") if roots_at.get(r["xyz"], "0") != "0" else r["root_id_curr"])
    frames = pickle.load(open(os.path.join(RDIR, "frames.pkl"), "rb"))

    core, det = {}, {}
    for f in glob.glob(os.path.join(RDIR, "details_g*.tsv")):
        g = int(os.path.basename(f)[9:-4])
        if g not in rows:  # details left over from a group that was merged away
            continue
        d = pd.read_csv(f, sep="\t", dtype={"root": str}).fillna("")
        det[g] = d
        core[g] = set(d[(d.med_r != "") & (pd.to_numeric(d.med_r, errors="coerce") <= 1.7)
                        & (pd.to_numeric(d.frac, errors="coerce") >= 0.8)].root)
    G = nx.Graph()
    for x, y in itertools.combinations(sorted(core), 2):
        sh = len(core[x] & core[y])
        if sh >= 4 or (sh >= 2 and sh >= 0.5 * min(len(core[x]), len(core[y]))):
            G.add_edge(x, y)
    sets = sorted((sorted(c) for c in nx.connected_components(G)), key=lambda c: c[0])
    if a.sets:
        sets = [sorted(int(x) for x in s.split(",")) for s in a.sets.split(";")]
    if a.only:
        only = {int(x) for x in a.only.split(",")}
        sets = [s for s in sets if only & set(s)]
    allow = {int(x) for x in a.allow.split(",") if x.strip()}

    # flags: SeaTable groups, groups near groups 1-8
    seatable = {g for g, rs in rows.items() if any("SeaTable" in r.get("notes", "") for r in rs)}
    g18 = pd.read_csv(os.path.join(os.path.dirname(OUT), "data", "pr_groups.tsv"), sep="\t", dtype=str).fillna("")
    P18 = np.array([[float(v) for v in re.findall(r"-?\d+", s)] for s in g18[g18.group.astype(int) <= 8].xyz]) * VOX_UM
    near18 = set()
    for g, rs in rows.items():
        pts = [[float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", r["xyz"])] for r in rs if r["xyz"]]
        if pts and np.linalg.norm(np.array(pts) * VOX_UM - P18[:, None, :], axis=2).min() < 4.0:
            near18.add(g)
    extra_flag = {int(x) for x in a.flag_only.split(",") if x.strip()}

    out_rows, removed, changed, report = {}, {}, set(), []
    next_free = max(rows) + 1
    for S in sets:
        why = [f"SeaTable group {g}" for g in S if g in seatable and g not in allow] + \
              [f"group {g} near groups 1-8" for g in S if g in near18 and g not in allow] + \
              [f"group {g} flagged" for g in S if g in extra_flag]
        if why and not a.sets:
            report.append(f"{S}: FLAGGED, not changed ({'; '.join(why)})")
            continue
        F = frames[min(S)]
        # candidate axons: listed rows of the set + unlisted core axons
        listed = {}
        for g in S:
            for r in rows[g]:
                rn = now(r)
                if rn != "0":
                    listed.setdefault(rn, []).append((g, r))
        unl = {r for g in S for r in core[g]} - set(listed)
        # per-depth positions across the bundle
        tr = pd.concat([pd.read_csv(os.path.join(RDIR, f"track_g{g}.tsv"), sep="\t", dtype={"root": str})
                        for g in S if os.path.exists(os.path.join(RDIR, f"track_g{g}.tsv"))])
        Pm = tr[["x", "y", "z"]].to_numpy(float)
        tr = tr.assign(s=np.round(((Pm - F["c"]) @ F["axis"]) / 3.0) * 3.0, pu=Pm @ F["u"], pv=Pm @ F["v"])
        cand = sorted((set(listed) | unl) & set(tr.root))
        pos = {r: x.groupby("s")[["pu", "pv"]].mean() for r, x in tr[tr.root.isin(cand)].groupby("root")}

        def dist(A, B):
            sh = A.index.intersection(B.index)
            return float(np.median(np.hypot(A.loc[sh, "pu"] - B.loc[sh, "pu"], A.loc[sh, "pv"] - B.loc[sh, "pv"]))) \
                if len(sh) >= 3 else 99.0

        # each group's center along the bundle (median of its core axons per depth bin)
        cen = {}
        for g in S:
            c = tr[tr.root.isin(core[g])].groupby("s")[["pu", "pv"]].median()
            if len(c) >= 3:
                cen[g] = c
        if len(cen) < 1:
            report.append(f"{S}: no group center could be traced: NOT changed")
            continue
        # same bundle = centers within --same um along the shared depths
        U = nx.Graph()
        U.add_nodes_from(cen)
        cdist = {}
        for x, y in itertools.combinations(sorted(cen), 2):
            cdist[(x, y)] = dist(cen[x], cen[y])
            if cdist[(x, y)] <= a.same:
                U.add_edge(x, y)
        comps = sorted((sorted(c) for c in nx.connected_components(U)), key=lambda c: c[0])
        bcen = {c[0]: pd.concat([cen[g] for g in c]).groupby(level=0).mean() for c in comps}  # bundle number -> center
        assign = {c[0]: set() for c in comps}
        for c in comps:
            for g in c[1:]:
                removed[g] = c[0]
        free = [g for g in S if g not in assign and g not in removed]  # groups without a traceable center
        # every axon to the nearest bundle center within the core radius; others stay with their own group
        d_of = {}
        for r in cand:
            d = {b: dist(pos[r], bc) for b, bc in bcen.items()}
            b = min(d, key=d.get)
            d_of[r] = (b, d[b])
            if d[b] <= 1.7:
                assign[b].add(r)
        ident = {r for g in S for r in det[g][det[g].assigned.isin(["outer", "R7", "R8", "extra"])].root}
        pr = lambda r: r in ident or (r in listed and any(x["class"] in ("R1-R6", "R7/R8") for _, x in listed[r]))
        members_of = {g: set(b) for g, b in assign.items()}
        placed = set().union(*members_of.values())
        for r, src in listed.items():
            if r in placed:
                continue
            gs = [g for g, _ in src if g in members_of]
            tgt = gs[0] if gs else removed.get(src[0][0], min(members_of))
            members_of[tgt].add(r)
        bundles = [assign[b] for b in sorted(assign)]
        for g, ms in members_of.items():
            out = []
            for r in sorted(ms, key=lambda r: (r not in listed, r)):
                if r not in listed:
                    continue  # unlisted core axons are picked up by groups_refine.py itself
                src = listed[r]
                own = next((x for gg, x in src if gg == g), src[0][1])
                fr = sorted({gg for gg, _ in src})
                x = dict(own)
                note = f"regroup {TODAY}: " + ("same bundle as " if len(S) > 1 else "") + \
                       f"groups {', '.join(map(str, S))} -> group {g}" + (f" (listed in group {', '.join(map(str, fr))})" if fr != [g] else "")
                x["notes"] = (x.get("notes", "") + "; " if x.get("notes") else "") + note
                out.append(x)
            out_rows[g] = out
            changed.add(g)
        report.append(f"{S}: centers {', '.join(f'{x}-{y} {v:.1f} um' for (x, y), v in cdist.items())} -> "
                      f"{len(bundles)} bundle(s): " +
                      "; ".join(f"group {g}: {len(b)} axons in the core ({sum(pr(r) for r in b)} PR; from groups "
                                f"{dict(Counter(gg for r in b if r in listed for gg, _ in listed[r]))})"
                                for g, b in sorted(assign.items())) +
                      (f"; merged away: {sorted(g for g in S if g in removed)}" if any(g in removed for g in S) else "") +
                      (f"; no center: {free}" if free else ""))
    print("\n".join(report))
    print(f"\ngroups to re-run (terminal, track, refine): {sorted(changed)}; removed: {removed}")
    if a.dry_run:
        return
    hdr = ["root_id_orig", "class", "root_id_curr", "xyz", "status", "pr_date", "proofreader", "chk_date", "checker",
           "notes", "link"]
    p = os.path.join(RDIR, "regroup.tsv")
    # keep earlier overrides for groups this run did not touch
    old_blocks, old_removed = {}, {}
    if os.path.exists(p):
        g = None
        for ln in open(p).read().split("\n"):
            c = ln.split("\t")[0]
            if ln.startswith("group ") and " removed" in c:
                old_removed[int(c.split()[1])] = ln
                g = None
            elif ln.startswith("group "):
                g = int(c.split()[1])
                old_blocks[g] = [ln]
            elif g is not None and ln.strip():
                old_blocks[g].append(ln)
    blocks = {g: v for g, v in old_blocks.items() if g not in out_rows and g not in removed}
    for g in out_rows:
        blocks[g] = [f"group {g}", "\t".join(hdr)] + ["\t".join(str(x.get(h, "")) for h in hdr) for x in out_rows[g]]
    rem = {g: v for g, v in old_removed.items() if g not in out_rows}
    for g, keep in removed.items():
        rem[g] = f"group {g} removed (same bundle as group {keep}; regroup {TODAY})"
    lines = []
    for g in sorted(blocks):
        lines += blocks[g] + [""]
    for g in sorted(rem):
        lines += [rem[g], ""]
    open(p, "w").write("\n".join(lines) + "\n")
    print(f"wrote {p}: {len(blocks)} regrouped groups, {len(rem)} removed")


if __name__ == "__main__":
    main()
