"""Build PR downstream tracking sheets in the layout of the Google sheet (block TSV, ready to paste/upload).
Sheet names: PRg<N>hop1 = top-20 downstream partners of each PR of group N; PRg<N>hop2 = one hop further.

  --group N      PRg<N>hop1: the top-20 downstream partners of each photoreceptor of group N (PRs located at their xyz
                 in data/pr_groups.tsv), same columns as PRg1hop1 (no nsyn_old)
  --hop2 SHEET   PRg<N>hop2 from the reviewed PRg<N>hop1 SHEET:
                 hop-1 cells (David, 2026-10-15) = per PR block the cells with nsyn >= --strong (3), at most --cap
                 (5); a block with fewer than --min-take (3) such cells takes its top 3 instead, filling down to
                 --floor (2), never nsyn 1; ties at a cut go to the better proofreading status, then higher PR_conv,
                 then lower root id;
                 skipped: the group's own photoreceptors, glia (annotation 'glia' on any row of the cell), cells whose
                 strongest row is 'stopped', and --skip ROOT=reason. Each hop-1 cell is listed once (PR_root_id and
                 h1_nsyn list every PR it qualifies under), followed by its top-20 downstream partners without
                 partners of nsyn < --row-min (2). --dry-run prints the counts for both rules and writes nothing.

The cell of a sheet row is the root at its xyz. Partners already tracked in SHEET or a --known sheet keep that row's
root_id_orig, xyz and tracking columns; untracked partners get root_id_orig = root_id_curr = the current root, a fresh
xyz anchor (faab.partners.points) and status NEW. Ties at the 20th place go to tracked cells, then the lower root id.
Every output xyz is looked up again at the end; disagreements are printed.

Output: figures/<YYYYMMDDdHHMMSStUTC>_FAABopticlobetargets-PRg<N>hop1.tsv / -PRg<N>hop2.tsv (UTC stamp, David's
format) and a _log.tsv with how each xyz was chosen. Read-only; CAVE calls >= 5 s apart."""
import argparse
import datetime
import os
import re
import time

import pandas as pd

from faab.clients.cave import client as cave_client, live_outputs, roots_at_xyz, throttle
from faab.config import DATA, FIGURES
from faab.partners.points import cell_points
from faab.partners.sheet import TRACK, as_int, norm_xyz, parse_sheet, utc_stamp, write_blocks
from faab.config import fig

G_HEADER = ["upstream_PR", "PRtype", "root_id_orig", "root_id_curr", "xyz", "nsyn", "syn_pct", "PR_conv"] + TRACK
H_HEADER = ["PR_root_id", "PRtype", "hop1_root_id", "h1_nsyn", "root_id_orig", "root_id_curr", "xyz", "nsyn", "syn_pct",
            "h1_conv"] + TRACK
TITLE_CLASS = {"R1-R6": "R1–R6", "R7": "R7", "R8": "R8"}
STATUS = TRACK.index("status")
STATUS_RANK = {"checked": 0, "proofread": 1, "adequate": 2, "in progress": 3, "new": 4}  # tie-break for hop-1 cells


def lookup(sheet_paths, extra_points, t):
    """one segment lookup for every row of the sheets plus extra points -> (rows per sheet with 'cell', extra roots, t)"""
    per = [[r["cells"] for r in parse_sheet(p)[1]] for p in sheet_paths]
    pts = [r.get("xyz", "") for rows in per for r in rows] + list(extra_points)
    t = throttle(t)
    at = roots_at_xyz(pts)
    k = 0
    for rows in per:
        for r in rows:
            r["cell"] = at[k] if norm_xyz(r.get("xyz", "")) else 0
            k += 1
    return per, at[k:], t


def known_from(per):
    """current root -> tracking info of its strongest sheet row"""
    known = {}
    for r in sorted((r for rows in per for r in rows if r["cell"]), key=lambda r: -as_int(r.get("nsyn", ""), 0)):
        known.setdefault(r["cell"], dict(root_id_orig=r["root_id_orig"].strip(), xyz=norm_xyz(r["xyz"]),
                                         **{c: r.get(c, "").strip() for c in TRACK}))
    return known


def top_n(counts, n, known):
    """counts: DataFrame post_pt_root_id, n -> top n (ties: tracked first, then lower root id)"""
    c = counts.assign(tracked=[int(r) in known for r in counts.post_pt_root_id])
    return c.sort_values(["n", "tracked", "post_pt_root_id"], ascending=[False, False, True]).head(n)


def partner_cells(root, known, pts):
    k = known.get(int(root))
    if k:
        return k["root_id_orig"], k["xyz"], [k[c] for c in TRACK], "tracked: sheet row's xyz and tracking columns"
    xyz, how = pts.get(int(root), ("", "no point"))
    track = [""] * len(TRACK)
    track[STATUS] = "NEW"
    return str(root), xyz, track, "new: " + how


def anchor_check(out_rows, i_curr, i_xyz, t):
    t = throttle(t)
    at = roots_at_xyz([r[i_xyz] for r in out_rows])
    bad = [(r[i_curr], r[i_xyz], a) for r, a in zip(out_rows, at) if r[i_xyz] and str(a) != str(r[i_curr])]
    print(f"xyz anchor check: {len(out_rows)} rows, {len(bad)} where the point is not on root_id_curr", flush=True)
    for b in bad:
        print("  ", b, flush=True)
    return t


def group_prs(g):
    grp = pd.read_csv(os.path.join(DATA, "pr_groups.tsv"), sep="\t", dtype=str).fillna("")
    return grp[grp.group == str(g)].reset_index(drop=True)


def build_group(g, known_paths, topn, ts_gen):
    prs = group_prs(g)
    client = cave_client()
    per, pr_root, t = lookup(known_paths, prs.xyz.tolist(), 0)
    known = known_from(per)
    print("PR roots at xyz:", dict(zip(prs.root_id_orig, pr_root)), f"; known cells: {len(known)}", flush=True)
    if 0 in pr_root or len(set(pr_root)) < len(pr_root):
        raise SystemExit("a PR point has no segment, or two PR points share a root: fix data/pr_groups.tsv first")
    t = throttle(t)
    syn, ts = live_outputs(pr_root, client)
    counts = syn.groupby(["pre_pt_root_id", "post_pt_root_id"]).size().reset_index(name="n")
    conv = counts.groupby("post_pt_root_id").pre_pt_root_id.nunique().to_dict()
    tops = {r: top_n(counts[counts.pre_pt_root_id == r], topn, known) for r in pr_root}
    new = {int(p) for tp in tops.values() for p in tp.post_pt_root_id if int(p) not in known}
    print(f"live {ts:%Y-%m-%d %H:%M} UTC: {len(syn)} output synapses; {len(new)} partners need a new xyz", flush=True)
    pts, t = cell_points(new, client, t)
    blocks, log, flat = [], [], []
    for pr, root in zip(prs.to_dict("records"), pr_root):
        sub = counts[counts.pre_pt_root_id == root]
        tot = int(sub.n.sum())
        title = (f"{pr['root_id_orig']}  [{TITLE_CLASS.get(pr['class'], pr['class'])}]  {tot} output synapses across "
                 f"{len(sub)} partners (updated {ts:%Y-%m-%d %H:%M} UTC)")
        rows = []
        for p in tops[root].itertuples():
            orig, xyz, track, how = partner_cells(p.post_pt_root_id, known, pts)
            rows.append([pr["root_id_orig"], pr["class"], orig, str(p.post_pt_root_id), xyz, int(p.n),
                         f"{round(100 * p.n / tot, 1):g}" if tot else "", int(conv.get(p.post_pt_root_id, 1))] + track)
            log.append(dict(upstream_PR=pr["root_id_orig"], PR_root_now=root, root_id_curr=p.post_pt_root_id, nsyn=int(p.n),
                            how=how))
        blocks.append((title, G_HEADER, rows))
        flat += rows
    t = anchor_check(flat, 3, 4, t)
    pre = [f">>> group {g}",
           f"note: generated {ts_gen} by faab/scripts/partners/pr_downstream_build.py --group {g}; live synapses "
           f"{ts:%Y-%m-%d %H:%M} UTC; PRs located at their xyz in faab/data/pr_groups.tsv",
           "note: per PR the top 20 downstream partners by synapse count; ties at 20th go to already tracked cells, then "
           "the lower root id; PR_conv = how many of the group's PRs synapse onto the cell",
           "note: cells already tracked in " + ", ".join(os.path.basename(p) for p in known_paths)
           + " keep that row's root_id_orig, xyz and tracking columns; status NEW = not tracked yet, needs proofreading"]
    return pre, blocks, log, f"PRg{g}hop1"


def build_hop2(sheet, known_paths, cap, min_take, strong, floor, row_min, topn, skip_extra, ts_gen, dry_run=False):
    lines = open(sheet, encoding="utf-8").read().split("\n")
    first = next((ln.split("\t")[0] for ln in lines if ln.startswith(">>>")), ">>> group 1")
    g = int(re.search(r"group\s+(\d+)", first).group(1))
    prs = group_prs(g)
    client = cave_client()
    per, pr_root, t = lookup([sheet] + known_paths, prs.xyz.tolist(), 0)
    known = known_from(per)
    own = {r for r in pr_root if r}
    rows = [r for r in per[0] if r["cell"]]
    glia = {r["cell"] for r in rows if "glia" in r.get("annotation", "").lower()}

    # strongest row per (PR block, cell), with its skip reason
    best = {}
    for r in rows:
        k = (r["upstream_PR"], r["cell"])
        if k not in best or as_int(r["nsyn"], 0) > as_int(best[k]["nsyn"], 0):
            best[k] = r
    tracked_live = {r["cell"] for r in rows if r.get("status", "").strip().lower() != "stopped"}
    cand = []
    for (b, cell), r in best.items():
        # 'stopped' skips only orphans: a stopped piece merged into a cell tracked elsewhere is that cell
        stopped = r.get("status", "").strip().lower() == "stopped" and cell not in tracked_live
        why = ("photoreceptor of this group" if cell in own else "glia (annotation)" if cell in glia else
               "status 'stopped' (orphan)" if stopped else skip_extra.get(cell, ""))
        st = r.get("status", "").strip().lower()
        cand.append(dict(block=b, cell=cell, n=as_int(r["nsyn"], 0), orig=r["root_id_orig"].strip(),
                         prtype=r.get("PRtype", r.get("class", "")).strip(), why=why,
                         st_rank=STATUS_RANK.get(st, len(STATUS_RANK)), conv=as_int(r.get("PR_conv", ""), 0)))
    cand = pd.DataFrame(cand)
    blocks_order = list(dict.fromkeys(r["upstream_PR"] for r in rows))

    def select():
        """per PR block (skips applied, nsyn >= floor), strongest first; ties at a cut go to the better proofreading
        status, then higher PR_conv, then lower root id. Cells with nsyn >= strong, at most `cap`; a block with fewer
        than `min_take` such cells takes its top `min_take` cells instead (filling down to floor)."""
        out = []
        for b in blocks_order:
            d = cand[(cand.block == b) & (cand.why == "") & (cand.n >= floor)].sort_values(
                ["n", "st_rank", "conv", "cell"], ascending=[False, True, False, True])
            s = d[d.n >= strong]
            out.append(s.head(cap) if len(s) >= min_take else d.head(min_take))
        return pd.concat(out)

    rule_used = f"cap{cap}min{min_take}"
    sel = {rule_used: select()}
    superset = sorted(set(sel[rule_used].cell))
    t = throttle(t)
    syn, ts = live_outputs(superset, client)
    counts = syn.groupby(["pre_pt_root_id", "post_pt_root_id"]).size().reset_index(name="n")
    tops = {}
    for c in superset:
        tp = top_n(counts[counts.pre_pt_root_id == c], topn, known)
        tops[c] = tp[tp.n >= row_min]
    for r, s_ in sel.items():
        cs = set(s_.cell)
        targets = {p for c in cs for p in tops[c].post_pt_root_id}
        print(f"\nrule {r}: {len(s_)} PR -> hop-1 pairs, {len(cs)} distinct hop-1 cells, "
              f"{sum(len(tops[c]) for c in cs)} hop-2 rows (nsyn >= {row_min}), {len(targets)} distinct hop-2 targets, "
              f"{len([p for p in targets if int(p) not in known])} not tracked yet", flush=True)
        for b in blocks_order:
            v = s_[s_.block == b].n.tolist()
            print(f"  PR {b}: {len(v)} hop-1 cells, nsyn {v}", flush=True)
    if dry_run:
        return None
    s = sel[rule_used]
    cells = list(dict.fromkeys(s.cell))
    conv = counts[counts.pre_pt_root_id.isin(cells)].groupby("post_pt_root_id").pre_pt_root_id.nunique().to_dict()
    new = {int(p) for c in cells for p in tops[c].post_pt_root_id if int(p) not in known}
    print(f"live {ts:%Y-%m-%d %H:%M} UTC; {len(new)} hop-2 partners need a new xyz", flush=True)
    pts, t = cell_points(new, client, t)
    blocks, log, flat = [], [], []
    for c in cells:
        d = s[s.cell == c].sort_values("n", ascending=False)
        orig = cand[cand.cell == c].sort_values("n", ascending=False).orig.iloc[0]
        sub = counts[counts.pre_pt_root_id == c]
        tot = int(sub.n.sum())
        prs_txt = "; ".join(f"PR {b} [{TITLE_CLASS.get(p, p)}] {n} syn" for b, p, n in zip(d.block, d.prtype, d.n))
        title = (f"hop-1 {orig} (now {c}) <- {prs_txt}: {tot} output synapses across {len(sub)} partners "
                 f"(updated {ts:%Y-%m-%d %H:%M} UTC)")
        rws = []
        for p in tops[c].itertuples():
            porig, xyz, track, how = partner_cells(p.post_pt_root_id, known, pts)
            rws.append([", ".join(d.block), ", ".join(dict.fromkeys(d.prtype)), orig, ", ".join(map(str, d.n)), porig,
                        str(p.post_pt_root_id), xyz, int(p.n), f"{round(100 * p.n / tot, 1):g}" if tot else "",
                        int(conv.get(p.post_pt_root_id, 1))] + track)
            log.append(dict(hop1_root_id=orig, hop1_root_now=c, root_id_curr=p.post_pt_root_id, nsyn=int(p.n), how=how))
        blocks.append((title, H_HEADER, rws))
        flat += rws
    t = anchor_check(flat, 5, 6, t)
    skipped = cand[(cand.why != "") & (cand.n >= floor)].sort_values(["block", "n"], ascending=[True, False])
    pre = [f">>> group {g}, two hops (PR -> hop-1 -> hop-2)",
           f"note: generated {ts_gen} by faab/scripts/partners/pr_downstream_build.py --hop2 from {os.path.basename(sheet)}; "
           f"live synapses {ts:%Y-%m-%d %H:%M} UTC; the cell of each PRg{g}hop1 row is the root at its xyz",
           f"note: hop-1 cells = per PR block of PRg{g}hop1, the cells with nsyn >= {strong}, at most {cap}; a block with "
           f"fewer than {min_take} such cells takes its top {min_take} instead, filling with nsyn {floor} cells (never "
           f"nsyn < {floor}); ties at a cut go to the better proofreading status (checked, proofread, adequate, in "
           "progress, NEW), then to the cell more group PRs synapse onto, then to the lower root id",
           "note: skipped as hop-1: the group's own photoreceptors (their targets are in PRg1hop1 already), glia, cells "
           "whose strongest row is 'stopped'; each hop-1 cell is listed once, PR_root_id / h1_nsyn list every PR it "
           "qualifies under",
           f"note: rows = the top {topn} downstream partners of each hop-1 cell (live; ties at {topn}th go to tracked cells, "
           f"then the lower root id), without partners of nsyn < {row_min}; h1_conv = how many of the hop-1 cells "
           "synapse onto the row's cell",
           "note: cells tracked in " + ", ".join(os.path.basename(p) for p in [sheet] + known_paths)
           + " keep that row's root_id_orig, xyz and tracking columns; status NEW = not tracked yet, needs proofreading"]
    pre += [f"note: skipped {r.cell} (hop-1 of PR {r.block}, {r.n} syn): {r.why}" for r in skipped.itertuples()]
    return pre, blocks, log, f"PRg{g}hop2"


def main():
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--group", type=int)
    m.add_argument("--hop2", metavar="SHEET")
    ap.add_argument("--known", nargs="*", default=[], help="sheets whose rows carry tracking info for cells they contain")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--cap", type=int, default=5, help="hop2: at most this many hop-1 cells per PR block")
    ap.add_argument("--min-take", type=int, default=3,
                    help="hop2: a block with fewer strong cells than this takes its top min-take cells instead")
    ap.add_argument("--strong", type=int, default=3, help="hop2: nsyn of a strong hop-1 cell")
    ap.add_argument("--floor", type=int, default=2, help="hop2: never take hop-1 cells below this nsyn")
    ap.add_argument("--row-min", type=int, default=2, help="hop2: drop hop-2 rows with nsyn below this")
    ap.add_argument("--dry-run", action="store_true", help="hop2: print the selection counts and stop")
    ap.add_argument("--skip", action="append", default=[], metavar="ROOT=REASON", help="hop2: skip this hop-1 cell")
    ap.add_argument("--stamp", help="file-name stamp (default: now, YYYYMMDDdHHMMSStUTC)")
    a = ap.parse_args()
    t0 = time.time()
    now = datetime.datetime.now(datetime.timezone.utc)
    ts_gen = f"{now:%Y-%m-%d %H:%M:%S} UTC"
    if a.group:
        pre, blocks, log, name = build_group(a.group, a.known, a.top, ts_gen)
    else:
        skip = {int(k): v for k, v in (s.split("=", 1) for s in a.skip)}
        res = build_hop2(a.hop2, a.known, a.cap, a.min_take, a.strong, a.floor, a.row_min, a.top, skip, ts_gen,
                         a.dry_run)
        if res is None:
            return
        pre, blocks, log, name = res
    width = max(len(b[1]) for b in blocks)
    out = fig(f"{a.stamp or utc_stamp(now)}_FAABopticlobetargets-{name}.tsv")
    write_blocks(out, [p for p in pre if not p.startswith("note:")], blocks, width,
                 notes=[p for p in pre if p.startswith("note:")])
    pd.DataFrame(log).to_csv(out.replace(".tsv", "_log.tsv"), sep="\t", index=False)
    print(f"wrote {out} ({sum(len(b[2]) for b in blocks)} rows in {len(blocks)} blocks) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
