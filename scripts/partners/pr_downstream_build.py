"""Build PRdownstream tracking sheets in the layout of the Google sheet (block TSV, ready to paste/upload).

  --group N            top-20 downstream partners of each photoreceptor of group N (PRs and their xyz from
                       data/pr_groups.tsv); same columns as the group 1 sheet
  --hop2 SHEET         one hop further: for each PR block of SHEET (the reviewed group sheet), the top --top1 partners
                       with nsyn >= --min-nsyn (PRd1), and the top-20 downstream partners of each PRd1 (PRd2)

Partners that already have a row in a --known sheet (matched by the root at that row's xyz) keep that row's root_id_orig,
xyz and tracking columns (status, proofreader, notes, ...); new partners get root_id_orig = root_id_curr = the current
root and a fresh xyz anchor (faab.partners.points). Ties at a cut-off favor partners that are already tracked, then the
lower root id. Every output row's xyz is looked up again at the end; disagreements are printed, not hidden.

Output: figures/<YYYYMMDD_HHMM>_PRdownstream_group<N>.tsv or ..._group<N>_hop2.tsv, plus a _log.tsv with how each
xyz was chosen. Read-only; CAVE calls >= 5 s apart."""
import argparse
import datetime
import os
import time

import pandas as pd

from faab.config import DATA, FIGURES
from faab.partners.points import cell_points
from faab.partners.sheet import TRACK, as_int, norm_xyz, parse_sheet, write_blocks
from faab.sources.cave import client as cave_client, live_outputs, roots_at_xyz, throttle

G_HEADER = ["upstream_PR", "PRtype", "root_id_orig", "root_id_curr", "xyz", "nsyn", "nsyn_old", "syn_pct", "PR_conv"] + TRACK
H_HEADER = ["PR_root_id", "PRtype", "PRd1_root_id", "PRd1_nsyn", "root_id_orig", "root_id_curr", "xyz", "nsyn", "syn_pct",
            "PRd1_conv"] + TRACK
TITLE_CLASS = {"R1-R6": "R1–R6", "R7": "R7", "R8": "R8"}


def known_cells(paths, t):
    """current root -> tracking info of the first sheet row whose xyz lies on it (one lookup for all sheets)"""
    rows = []
    for p in paths:
        rows += [r["cells"] for r in parse_sheet(p)[1] if norm_xyz(r["cells"].get("xyz", ""))]
    if not rows:
        return {}, t
    t = throttle(t)
    at = roots_at_xyz([r["xyz"] for r in rows])
    known = {}
    for r, root in zip(rows, at):
        if root and root not in known:
            known[root] = dict(root_id_orig=r["root_id_orig"], xyz=norm_xyz(r["xyz"]), **{k: r.get(k, "").strip() for k in TRACK})
    print(f"known cells: {len(known)} from {len(rows)} rows of {len(paths)} sheet(s)", flush=True)
    return known, t


def top_n(counts, n, known):
    """counts: DataFrame post_pt_root_id, n -> top n (ties: tracked first, then lower root id)"""
    c = counts.assign(tracked=[int(r) in known for r in counts.post_pt_root_id])
    return c.sort_values(["n", "tracked", "post_pt_root_id"], ascending=[False, False, True]).head(n)


def partner_cells(root, known, pts):
    k = known.get(int(root))
    if k:
        return k["root_id_orig"], k["xyz"], [k[c] for c in TRACK], "tracked: sheet row's xyz and tracking columns"
    xyz, how = pts.get(int(root), ("", "no point"))
    return str(root), xyz, [""] * len(TRACK), "new: " + how


def anchor_check(out_rows, i_curr, i_xyz, t):
    t = throttle(t)
    at = roots_at_xyz([r[i_xyz] for r in out_rows])
    bad = [(r[i_curr], r[i_xyz], a) for r, a in zip(out_rows, at) if r[i_xyz] and str(a) != str(r[i_curr])]
    print(f"xyz anchor check: {len(out_rows)} rows, {len(bad)} where the point is not on root_id_curr", flush=True)
    for b in bad:
        print("  ", b, flush=True)
    return t


def build_group(g, known_paths, topn):
    grp = pd.read_csv(os.path.join(DATA, "pr_groups.tsv"), sep="\t", dtype=str).fillna("")
    prs = grp[grp.group == str(g)].reset_index(drop=True)
    client = cave_client()
    t = throttle(0)
    known, t = known_cells(known_paths, t)
    t = throttle(t)
    pr_root = [int(r) for r in roots_at_xyz(prs.xyz.tolist())]
    print("PR roots at xyz:", dict(zip(prs.root_id_orig, pr_root)), flush=True)
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
            rows.append([pr["root_id_orig"], pr["class"], orig, str(p.post_pt_root_id), xyz, int(p.n), "",
                         round(100 * p.n / tot, 1) if tot else "", int(conv.get(p.post_pt_root_id, 1))] + track)
            log.append(dict(upstream_PR=pr["root_id_orig"], PR_root_now=root, root_id_curr=p.post_pt_root_id, nsyn=int(p.n),
                            how=how))
        blocks.append((title, G_HEADER, rows))
        flat += rows
    t = anchor_check(flat, 3, 4, t)
    return blocks, log, f"group{g}", f">>> group {g}"


def build_hop2(sheet, known_paths, top1, min_nsyn, ties, topn):
    lines, rows = parse_sheet(sheet)
    client = cave_client()
    t = throttle(0)
    known, t = known_cells([sheet] + known_paths, t)
    # PRd1 selection per PR block, from the sheet's nsyn (what the sheet shows)
    sel = []
    for pr in dict.fromkeys(r["block_pr"] for r in rows):
        b = sorted([r for r in rows if r["block_pr"] == pr and as_int(r["cells"]["nsyn"], 0) >= min_nsyn],
                   key=lambda r: -as_int(r["cells"]["nsyn"], 0))
        if len(b) > top1 and ties == "include":
            cut = as_int(b[top1 - 1]["cells"]["nsyn"], 0)
            b = [r for r in b if as_int(r["cells"]["nsyn"], 0) >= cut]
        else:
            b = b[:top1]
        sel += b
    t = throttle(t)
    d1_root = [int(r) for r in roots_at_xyz([r["cells"]["xyz"] for r in sel])]
    if 0 in d1_root:
        raise SystemExit("a selected PRd1 row has no segment at its xyz: fix the sheet first")
    print(f"{len(sel)} PRd1 rows ({len(set(d1_root))} distinct cells) from {len({r['block_pr'] for r in sel})} PR blocks",
          flush=True)
    t = throttle(t)
    syn, ts = live_outputs(sorted(set(d1_root)), client)
    counts = syn.groupby(["pre_pt_root_id", "post_pt_root_id"]).size().reset_index(name="n")
    conv = counts.groupby("post_pt_root_id").pre_pt_root_id.nunique().to_dict()
    tops = {r: top_n(counts[counts.pre_pt_root_id == r], topn, known) for r in set(d1_root)}
    new = {int(p) for tp in tops.values() for p in tp.post_pt_root_id if int(p) not in known}
    print(f"live {ts:%Y-%m-%d %H:%M} UTC: {len(syn)} PRd1 output synapses; {len(new)} partners need a new xyz", flush=True)
    pts, t = cell_points(new, client, t)
    blocks, log, flat = [], [], []
    for r, d1 in zip(sel, d1_root):
        c = r["cells"]
        sub = counts[counts.pre_pt_root_id == d1]
        tot = int(sub.n.sum())
        prtype = c.get("PRtype", c.get("class", ""))
        title = (f"{r['block_pr']}  [{TITLE_CLASS.get(prtype, prtype)}]  ->  PRd1 {c['root_id_orig']} ({c['nsyn']} syn from the PR; "
                 f"now {d1}): {tot} output synapses across {len(sub)} partners (updated {ts:%Y-%m-%d %H:%M} UTC)")
        rws = []
        for p in tops[d1].itertuples():
            orig, xyz, track, how = partner_cells(p.post_pt_root_id, known, pts)
            rws.append([r["block_pr"], prtype, c["root_id_orig"], c["nsyn"], orig, str(p.post_pt_root_id), xyz, int(p.n),
                        round(100 * p.n / tot, 1) if tot else "", int(conv.get(p.post_pt_root_id, 1))] + track)
            log.append(dict(PR_root_id=r["block_pr"], PRd1_root_id=c["root_id_orig"], PRd1_root_now=d1,
                            root_id_curr=p.post_pt_root_id, nsyn=int(p.n), how=how))
        blocks.append((title, H_HEADER, rws))
        flat += rws
    t = anchor_check(flat, 5, 6, t)
    first = next((ln.split("\t")[0] for ln in lines if ln.startswith(">>>")), ">>> group")
    return blocks, log, None, first + ", two hops (PR -> PRd1 -> PRd2)"


def main():
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--group", type=int)
    m.add_argument("--hop2", metavar="SHEET")
    ap.add_argument("--known", nargs="*", default=[], help="sheets whose rows carry tracking info for cells they contain")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--top1", type=int, default=5, help="hop2: PRd1 cells per PR block")
    ap.add_argument("--min-nsyn", type=int, default=3, help="hop2: PRd1 cells need at least this many synapses from the PR")
    ap.add_argument("--ties", choices=["include", "cut"], default="include", help="hop2: ties at the PRd1 cut-off")
    a = ap.parse_args()
    t0 = time.time()
    stamp = f"{datetime.datetime.now(datetime.timezone.utc):%Y%m%d_%H%M}"
    if a.group:
        blocks, log, tag, first = build_group(a.group, a.known, a.top)
    else:
        blocks, log, tag, first = build_hop2(a.hop2, a.known, a.top1, a.min_nsyn, a.ties, a.top)
        tag = first.split(",")[0].replace(">>>", "").replace(" ", "") + "_hop2"
    width = max(len(b[1]) for b in blocks)
    out = os.path.join(FIGURES, f"{stamp}_PRdownstream_{tag}.tsv")
    write_blocks(out, first, blocks, width)
    pd.DataFrame(log).to_csv(out.replace(".tsv", "_log.tsv"), sep="\t", index=False)
    print(f"wrote {out} ({sum(len(b[2]) for b in blocks)} rows in {len(blocks)} blocks) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
