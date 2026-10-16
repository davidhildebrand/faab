"""Refresh the photoreceptor -> downstream partner tables (top 20 per PR) after proofreading,
preserving the tracking-sheet rows.

Inputs : tmp/sheet_prs.tsv       (8 PRs: root_id_orig, class, root_id_curr, ...)
         tmp/sheet_partners.tsv  (existing partner rows with hand-edited columns)
Steps  : 0. ONE segment lookup at every sheet xyz (PRs and partner rows): the point is the anchor
         1. PR roots = the root at the PR's xyz (lineage only when a PR has no point)
         2. ONE live synapse query (pre_pt_root_id in PR roots) -> counts per partner
         3. match each current partner root to sheet rows by the root at the row's xyz; rows
            without a point use the lineage of their ids (cached in figures/pr_partner_lineage.json)
         4. top 20 per PR + any sheet row that dropped out (kept, flagged)
         5. xyz: keep the sheet value when the row's root is unchanged, else the usual recipe
Output : figures/faab_pr_top20_update.tsv (sheet headers + n_syn_prev + update) and the
         same as per-PR blocks in figures/faab_pr_top20_update_blocks.tsv. Read-only, throttled.
"""
from faab.config import REPO
import datetime
import json
import os
import pickle
import time

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import caveclient
from fafbseg import flywire
from faab.sources.cave import DATASTACK, throttle
from faab.partners.top10_xyz import primary_neurite_point_nm, backbone_midpoint_nm, VOX

HERE = REPO
OUT = os.path.join(HERE, "figures")
TMP = os.path.join(HERE, "..", "tmp")
SCACHE = os.path.join(OUT, "pr_partner_skels.pkl")
LIN = os.path.join(OUT, "pr_partner_lineage.json")
TOPN = 20
SHEET_COLS = ["upstream_PR", "class", "root_id_orig", "root_id_curr", "xyz", "n_syn", "pct_of_outp",
              "PR_conv", "annotation", "status", "pr_date", "proofreader", "chk_date", "checker",
              "notes", "followups", "SeaTable"]
KEEP = ["annotation", "status", "pr_date", "proofreader", "chk_date", "checker", "notes", "followups", "SeaTable"]


def valid(s):
    s = str(s).strip()
    return s if s.isdigit() and len(s) == 18 else None


def fmt(pt_nm):
    return ", ".join(str(int(round(v))) for v in (np.asarray(pt_nm, float) / VOX))


def main():
    t0 = time.time()
    prs = pd.read_csv(os.path.join(TMP, "sheet_prs.tsv"), sep="\t", dtype=str).fillna("")
    sheet = pd.read_csv(os.path.join(TMP, "sheet_partners.tsv"), sep="\t", dtype=str).fillna("")
    for c in SHEET_COLS:
        if c not in sheet.columns:
            sheet[c] = ""
    print(f"{len(prs)} PRs, {len(sheet)} existing partner rows", flush=True)

    client = caveclient.CAVEclient(DATASTACK)
    now = datetime.datetime.now(datetime.timezone.utc)
    t = throttle(time.time())

    # ---- 0. the xyz is the anchor (user, 2026-09-25): the current root at every sheet point,
    #         PRs and partner rows, in ONE lookup; lineage is only the fallback for rows without a point ----
    def parse_xyz(s):
        try:
            v = [float(x) for x in str(s).split(",")]
            return v if len(v) == 3 else None
        except Exception:
            return None

    pts, keys = [], []
    for o, x in zip(prs.root_id_orig.map(valid), prs.xyz):
        v = parse_xyz(x)
        if o and v:
            pts.append(v)
            keys.append(("pr", o))
    for i, x in enumerate(sheet.xyz):
        v = parse_xyz(x)
        if v:
            pts.append(v)
            keys.append(("row", i))
    seg_at = {}
    if pts:
        roots_at = flywire.locs_to_segments(np.array(pts, float) * VOX, dataset=DATASTACK,
                                            coordinates="nm", backend="cloudvolume")
        seg_at = {k: int(r) for k, r in zip(keys, roots_at) if int(r) > 0}
    row_xyz_root = {i: seg_at.get(("row", i)) for i in range(len(sheet))}
    print(f"roots at sheet xyz: {sum(1 for k in seg_at if k[0] == 'pr')}/{len(prs)} PRs, "
          f"{sum(1 for k in seg_at if k[0] == 'row')}/{len(sheet)} partner rows", flush=True)

    # ---- 1. PR roots: the root at the sheet xyz; lineage only when there is no point ----
    pr_orig = [valid(x) for x in prs.root_id_orig]
    pr_curr = {o: valid(c) for o, c in zip(pr_orig, prs.root_id_curr)}
    pr_class = dict(zip(pr_orig, prs["class"]))
    latest = dict(zip(pr_curr.values(), [bool(x) for x in np.asarray(
        client.chunkedgraph.is_latest_roots([int(c) for c in pr_curr.values()]))]))
    pr_note, cands = {}, {}
    for o, c in pr_curr.items():
        if ("pr", o) in seg_at:
            cands[o] = [seg_at[("pr", o)]]
            if str(seg_at[("pr", o)]) != c:
                pr_note[o] = f"root at sheet xyz is {seg_at[('pr', o)]} (sheet root_id_curr {c} is stale)"
            continue
        if latest[c]:
            cands[o] = [int(c)]
            continue
        t = throttle(t)
        cands[o] = [int(x) for x in np.atleast_1d(client.chunkedgraph.get_latest_roots(int(c)))]

    # ---- 1b. lineage of every sheet id, refreshed for ids that are no longer current roots
    #          (their descendants keep changing while proofreading runs) ----
    LIN_T = LIN.replace(".json", "_times.json")
    lin = json.load(open(LIN)) if os.path.exists(LIN) else {}
    lin_t = json.load(open(LIN_T)) if os.path.exists(LIN_T) else {}
    sheet_ids = sorted({int(x) for col in ("root_id_orig", "root_id_curr") for x in sheet[col].map(valid) if x})
    sid_latest = dict(zip(sheet_ids, [bool(x) for x in np.asarray(client.chunkedgraph.is_latest_roots(sheet_ids))]))
    fresh_cut = time.time() - 15 * 60
    need = [i for i in sheet_ids if str(i) not in lin
            or (not sid_latest[i] and lin_t.get(str(i), 0) < fresh_cut)]
    print(f"lineage fetches needed: {len(need)} (of {len(sheet_ids)} sheet ids)", flush=True)
    for i in need:
        t = throttle(t)
        try:
            g = client.chunkedgraph.get_lineage_graph(i, as_nx_graph=True)
            lin[str(i)] = sorted(int(n) for n in g.nodes)
            lin_t[str(i)] = time.time()
        except Exception as e:
            lin.setdefault(str(i), [i])
            print(f"  lineage failed {i}: {str(e)[:50]}", flush=True)
    json.dump(lin, open(LIN, "w"))
    json.dump(lin_t, open(LIN_T, "w"))

    # ---- 2. ONE live synapse query over every candidate piece; the PR is the piece that
    #         carries the output synapses (size alone picks retina scraps) ----
    allc = sorted({r for v in cands.values() for r in v})
    t = throttle(t)
    syn = client.materialize.live_query("synapses_v2", timestamp=now,
                                        filter_in_dict={"pre_pt_root_id": allc})
    out_by_root = syn.groupby("pre_pt_root_id").size().to_dict()
    pair = syn[syn.pre_pt_root_id != syn.post_pt_root_id].groupby(
        ["pre_pt_root_id", "post_pt_root_id"]).size().reset_index(name="n")
    # known partners per PR = the sheet rows of that PR plus their lineages (cached file)
    lin0 = lin
    nk0 = {}
    for k, nodes in lin0.items():
        for n in nodes:
            nk0.setdefault(int(n), set()).add(int(k))
    known = {o: set() for o in pr_orig}
    for _, r in sheet.iterrows():
        if r.upstream_PR not in known:
            continue
        for col in ("root_id_orig", "root_id_curr"):
            v = valid(r[col])
            if v:
                known[r.upstream_PR].add(int(v))
                for k in nk0.get(int(v), set()):
                    known[r.upstream_PR].update(lin0[str(k)])

    def overlap(o, r):
        sub = pair[pair.pre_pt_root_id == r]
        return int(sub[sub.post_pt_root_id.isin(known[o])].n.sum())

    # candidate roots can be SHARED between PRs after merges: assign jointly, one root per
    # PR, best overlap with that PR's tracked partners first
    assigned = {}
    for o in sorted(pr_orig, key=lambda o: -max(overlap(o, r) for r in cands[o])):
        free = [r for r in cands[o] if r not in assigned.values()] or cands[o]
        best = max(free, key=lambda r: (overlap(o, r), out_by_root.get(r, 0)))
        assigned[o] = best
        if str(best) != pr_curr[o] or len(cands[o]) > 1:
            others = ", ".join(f"{r} ({out_by_root.get(r, 0)} syn, {overlap(o, r)} onto tracked partners)"
                               for r in cands[o] if r != best)
            pr_note[o] = (f"PR root {pr_curr[o]} changed since the sheet; now {len(cands[o])} root(s); using "
                          f"{best} ({out_by_root.get(best, 0)} output syn, {overlap(o, best)} onto tracked partners)"
                          + (f"; other pieces: {others}" if others else ""))
        pr_curr[o] = str(best)
    print("PR roots used:", pr_curr, flush=True)
    for o, n in pr_note.items():
        print("  ", o, n, flush=True)
    syn = syn[syn.pre_pt_root_id.isin([int(c) for c in pr_curr.values()])]
    syn = syn[syn.pre_pt_root_id != syn.post_pt_root_id]
    counts = syn.groupby(["pre_pt_root_id", "post_pt_root_id"]).size().reset_index(name="n")
    curr_to_orig = {int(c): o for o, c in pr_curr.items()}
    counts["pr"] = counts.pre_pt_root_id.map(curr_to_orig)
    tot = counts.groupby("pr").n.sum().to_dict()
    npart = counts.groupby("pr").post_pt_root_id.nunique().to_dict()
    conv = counts.groupby("post_pt_root_id").pr.nunique().to_dict()
    print(f"live synapses: {len(syn)} at {now:%Y-%m-%d %H:%M} UTC", flush=True)

    # ---- 3. node -> set of lineage keys; sheet row -> keys (lineages refreshed in 1b) ----
    node_keys = {}
    for k, nodes in lin.items():
        for n in nodes:
            node_keys.setdefault(int(n), set()).add(int(k))
    row_keys = []
    for _, r in sheet.iterrows():
        ks = set()
        for col in ("root_id_orig", "root_id_curr"):
            v = valid(r[col])
            if v:
                ks.add(int(v))
                ks |= node_keys.get(int(v), set())
        row_keys.append(ks)

    def rows_for(partner, pr):
        """sheet rows anchored to this current partner root: the root at the row's xyz decides;
        rows without a point fall back to lineage. Same-PR rows first."""
        p = int(partner)
        hits = [i for i, ks in enumerate(row_keys)
                if (row_xyz_root[i] == p) or (row_xyz_root[i] is None and
                                              (p in ks or any(p in lin.get(str(k), []) for k in ks)))]
        same = [i for i in hits if sheet.upstream_PR[i] == pr]
        return same, [i for i in hits if i not in same]

    # ---- 4. build rows ----
    new_rows = []
    fetch_xyz = set()
    for o in pr_orig:
        sub = counts[counts.pr == o].copy()
        # ties at the cut-off favour partners already tracked in the sheet
        sub["tracked"] = [1 if rows_for(p, o)[0] else 0 for p in sub.post_pt_root_id]
        sub = sub.sort_values(["n", "tracked", "post_pt_root_id"], ascending=[False, False, True])
        top = list(sub.post_pt_root_id.head(TOPN))
        used = set()
        for rank, p in enumerate(top, 1):
            n = int(sub[sub.post_pt_root_id == p].n.iloc[0])
            same, other = rows_for(p, o)
            src = same[0] if same else (other[0] if other else None)
            upd = []
            if len(same) > 1:
                upd.append("merged: also matches rows " + ", ".join(sheet.root_id_orig[i] for i in same[1:]))
            row = {c: "" for c in SHEET_COLS}
            row.update({"upstream_PR": o, "class": pr_class[o], "root_id_curr": str(p), "n_syn": n,
                        "pct_of_outp": round(n / tot[o] * 100, 1), "PR_conv": int(conv.get(p, 1))})
            if src is not None:
                used.update(same)
                s = sheet.iloc[src]
                for c in KEEP:
                    row[c] = s[c]
                row["root_id_orig"] = s.root_id_orig
                row["n_syn_prev"] = s.n_syn if same else ""
                # the sheet's xyz is the anchor and is never recomputed; roots move, points do not
                row["xyz"] = s.xyz
                if str(p) != valid(s.root_id_curr):
                    upd.append(f"root changed from sheet value {s.root_id_curr}")
                if not str(s.xyz).strip():
                    fetch_xyz.add(int(p))
                if not same:
                    upd.append(f"info carried from PR {s.upstream_PR}'s row for this cell")
            else:
                row["root_id_orig"] = str(p)
                row["n_syn_prev"] = ""
                fetch_xyz.add(int(p))
            row["update"] = "; ".join(upd)
            row["rank"] = rank
            new_rows.append(row)
        # sheet rows of this PR not in the top 20: keep, flagged
        for i in sheet.index[sheet.upstream_PR == o]:
            if i in used:
                continue
            s = sheet.iloc[i]
            row = {c: s[c] for c in SHEET_COLS}
            now_ids = set()
            if row_xyz_root[i] is not None:  # the point decides which current root this row is
                now_ids = {row_xyz_root[i]} & set(sub.post_pt_root_id)
            else:
                cands = {int(v) for v in (valid(s.root_id_orig), valid(s.root_id_curr)) if v}
                for k in cands:
                    now_ids |= {n for n in lin.get(str(k), []) if n in set(sub.post_pt_root_id)}
                    if k in set(sub.post_pt_root_id):
                        now_ids.add(k)
            n_now = int(sub[sub.post_pt_root_id.isin(now_ids)].n.sum()) if now_ids else 0
            row.update({"n_syn": n_now, "pct_of_outp": round(n_now / tot[o] * 100, 1) if tot.get(o) else "",
                        "PR_conv": int(max([conv.get(x, 0) for x in now_ids] or [0])),
                        "n_syn_prev": s.n_syn, "rank": "",
                        "update": ("dropped out of top 20 (now " + str(n_now) + " syn)") if n_now
                        else "no synapses from this PR at present (partner may have been split/merged elsewhere)"})
            new_rows.append(row)

    # ---- 5. xyz for new / changed roots ----
    sk_by = pickle.load(open(SCACHE, "rb")) if os.path.exists(SCACHE) else {}
    need_sk = sorted(i for i in fetch_xyz if i not in sk_by)
    if need_sk:
        t = throttle(t)
        try:
            for s in flywire.get_l2_skeleton(need_sk, dataset=DATASTACK, max_threads=1,
                                             omit_failures=True, progress=False):
                sk_by[int(s.id)] = s
        except Exception as e:
            print("skeleton fetch failed:", str(e)[:60], flush=True)
        pickle.dump(sk_by, open(SCACHE, "wb"))
    soma = {}
    if fetch_xyz:
        t = throttle(t)
        try:
            nuc = client.materialize.query_table("nuclei_v1_aedes",
                                                 filter_in_dict={"pt_root_id": sorted(fetch_xyz)})
            for _, r in nuc.iterrows():
                soma[int(r.pt_root_id)] = (np.array([r.pt_position_x, r.pt_position_y, r.pt_position_z], float)
                                           if "pt_position_x" in nuc.columns else np.array(r.pt_position, float))
        except Exception as e:
            print("nuclei query failed:", str(e)[:60], flush=True)
    xyz, how = {}, {}
    for i in fetch_xyz:
        s = sk_by.get(i)
        if s is None:
            try:
                t = throttle(t)
                l2ids = [int(x) for x in np.atleast_1d(client.chunkedgraph.get_leaves(i, stop_layer=2))]
                d = client.l2cache.get_l2data(l2ids, attributes=["rep_coord_nm"])
                pts = np.array([np.array(v["rep_coord_nm"], float) for v in d.values() if "rep_coord_nm" in v])
                xyz[i], how[i] = fmt(pts.mean(0)), "single L2 chunk fragment -> L2 rep coord"
            except Exception as e:
                xyz[i], how[i] = "", "no skeleton, no L2 coord"
            continue
        pt, geo = (None, None)
        if i in soma and s.n_nodes >= 5:
            pt, geo = primary_neurite_point_nm(s, soma[i])
        if pt is not None:
            how[i] = f"primary neurite ~{geo:.0f}um from soma"
        else:
            pt = backbone_midpoint_nm(s)
            how[i] = "no nucleus -> backbone midpoint" if i not in soma else "trunk too short -> backbone midpoint"
        xyz[i] = fmt(pt)
    for row in new_rows:
        p = int(row["root_id_curr"]) if valid(row["root_id_curr"]) else None
        if p in xyz and not row["xyz"]:
            row["xyz"] = xyz[p]
            row["update"] = (row["update"] + "; " if row["update"] else "") + how[p]

    # ---- xyz anchor check: which root occupies each row's point now? (one lookup; new partners'
    #      points were just computed, so this re-checks every row, not only the sheet rows) ----
    locs, keys = [], []
    for i, row in enumerate(new_rows):
        v = parse_xyz(row["xyz"])
        if v:
            locs.append(v)
            keys.append(("row", i))
    for o, x in zip(pr_orig, prs.xyz):
        v = parse_xyz(x)
        if v:
            locs.append(v)
            keys.append(("pr", o))
    seg_at = {}
    if locs:
        t = throttle(t)
        try:
            # sheet xyz are 16x16x45 nm voxels -> pass nm so fafbseg does not guess the resolution
            roots_at = flywire.locs_to_segments(np.array(locs, float) * VOX, dataset=DATASTACK,
                                                coordinates="nm", backend="cloudvolume")
            seg_at = dict(zip(keys, [int(r) for r in roots_at]))
        except Exception as e:
            print("locs_to_segments failed:", str(e)[:80], flush=True)
    n_mis = 0
    for i, row in enumerate(new_rows):
        r = seg_at.get(("row", i))
        if r is not None and valid(row["root_id_curr"]) and r != int(row["root_id_curr"]):
            n_mis += 1
            row["update"] = (row["update"] + "; " if row["update"] else "") + \
                f"CHECK: segment at xyz is {r}, not root_id_curr"
    print(f"xyz anchor check: {sum(1 for k in seg_at if k[0] == 'row')} partner points looked up, "
          f"{n_mis} where the segment at xyz is not root_id_curr", flush=True)
    for o in pr_orig:
        r = seg_at.get(("pr", o))
        if r is not None:
            print(f"  PR {o}: segment at sheet xyz = {r}; root used = {pr_curr[o]} "
                  f"{'OK' if str(r) == pr_curr[o] else 'MISMATCH'}", flush=True)

    cols = SHEET_COLS + ["n_syn_prev", "update"]
    df = pd.DataFrame(new_rows)[["rank"] + cols]
    df.to_csv(os.path.join(OUT, "faab_pr_top20_update.tsv"), sep="\t", index=False)
    with open(os.path.join(OUT, "faab_pr_top20_update_blocks.tsv"), "w") as f:
        for o in pr_orig:
            f.write(f"{o}  [{pr_class[o]}]  {tot.get(o, 0)} output synapses across {npart.get(o, 0)} partners  "
                    f"(live {now:%Y-%m-%d}, root {pr_curr[o]})\n")
            f.write("\t".join(cols) + "\n")
            for _, r in df[df.upstream_PR == o].iterrows():
                f.write("\t".join(str(r[c]) for c in cols) + "\n")
            f.write("\n")
    json.dump({"live_utc": f"{now:%Y-%m-%d %H:%M}", "pr_roots": pr_curr, "totals": {o: [tot.get(o, 0), npart.get(o, 0)] for o in pr_orig}},
              open(os.path.join(OUT, "faab_pr_top20_update_meta.json"), "w"), indent=1)
    print(open(os.path.join(OUT, "faab_pr_top20_update_blocks.tsv")).read())
    print(f"DONE in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
