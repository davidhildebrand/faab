"""Check a PR downstream sheet (TSV export, block layout) against the live segmentation and synapse table.

For every row with an xyz point (16 x 16 x 45 nm voxels; the anchor):
  root_at_xyz   the segment at the point now (one batched lookup for all points)
  curr_latest   whether root_id_curr is a current root (batched is_latest_roots)
  lineage       whether root_at_xyz descends from root_id_orig (get_latest_roots per stale id, throttled, cached in
                ../tmp/pr_sheet_check_lineage.json; a cached entry that does not explain a point is fetched again)
and a verdict:
  ok             root_id_curr is current, is the root at xyz, and descends from root_id_orig
  update         root_id_curr is stale; the root at xyz descends from it -> proposed root_id_curr = root at xyz
  MISMATCH       root_id_curr is current but the point lies on another current root
  STALE-NO-LINK  root_id_curr is stale and the root at xyz does not descend from it
  LINEAGE        the root at xyz does not descend from root_id_orig
  NO-SEGMENT     no segment at the point
The block PRs (upstream_PR = the PR's original root) are located at their xyz in data/pr_groups.tsv.

nsyn recount: ONE live query of the block PRs' outputs. Each row's cell = the root at its point. When several rows of a
block now sit on the same cell (merged pieces), the live count goes on the main row (the larger sheet nsyn, ties ->
verdict ok first) and the other rows keep their sheet values, as the sheet already does for merges. syn_pct and PR_conv
(how many of the group's PRs synapse onto the cell) are recomputed for main rows; nsyn_old is left as it is.
Changes of more than 2 synapses are flagged. Also reported: cells in a PR's live top 20 that the block lacks (top 20 by the
builder's rule - exactly 20, ties to cells already in the sheet then the lower root id - so ties at the cutoff do not count).

--write-sheet OUT writes the updated copy: root_id_curr ('update' rows), nsyn / syn_pct / PR_conv (main rows), block
title totals, --set fixes ('LINE:column=value', LINE = sheet line number), every cell stripped of leading/trailing
spaces and xyz as 'x, y, z', --add-missing N rows (a cell tracked in another block keeps that row's root_id_orig,
xyz and tracking columns incl. its status - NEW only when it has none; an untracked cell gets a fresh xyz and status
NEW), --orphan-notes (see below), and each block's rows sorted by nsyn (stable). Everything else verbatim.
Report: figures/pr_sheet_check_<YYYYMMDDdHHMMSStUTC>.tsv.

--orphan-notes: a 'stopped' row whose cell is still an orphan (no non-stopped row of the sheet sits on the same cell,
and the cell is not a whole glia cell) gets notes "orphan fragment", keeping only note clauses that are not about
being a fragment / orphan / lacking a soma; --note 'ROOT_ORIG[@BLOCK_PR]=TEXT' sets a row's notes outright.
Usage: python pr_sheet_check.py <sheet.tsv> [--write-sheet OUT.tsv] [--set 152:root_id_orig=6485...] [--add-missing 2]
Read-only; CAVE calls >= 5 s apart."""
import argparse
import datetime
import json
import os
import re
import time

import numpy as np
import pandas as pd

from faab.clients.cave import live_outputs, parse_xyz
from faab.config import DATASTACK, FIGURES, DATA, TMP, VOX
from faab.partners.sheet import as_int, fmt_vox, parse_sheet, utc_stamp
from faab.config import fig

CACHE = os.path.join(TMP, "pr_sheet_check_lineage.json")
FRAG = re.compile(r"fragment|orphan|soma|difficult to extend", re.I)   # note clauses replaced by "orphan fragment"
TITLE = re.compile(r"^(\d{15,})(\s+\[[^\]]*\]\s+)\d+ output synapses across \d+ partners \(updated [^)]*\)")
TITLE2 = re.compile(r"^(hop-1 (\d{15,}) \(now )(\d{15,})(\).*?: )\d+ output synapses across \d+ partners \(updated [^)]*\)")
_last = [0.0]


def throttle(gap=5.0):
    wait = _last[0] + gap - time.time()
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet")
    ap.add_argument("--write-sheet", help="write the updated sheet here (only after reviewing the report)")
    ap.add_argument("--set", action="append", default=[], help="LINE:column=value fix applied when writing")
    ap.add_argument("--add-missing", type=int, default=0, metavar="N",
                    help="when writing, add rows (status NEW) for cells with >= N synapses in a PR's live top 20 that "
                         "its block lacks")
    ap.add_argument("--main", type=int, action="append", default=[], metavar="LINE",
                    help="treat this sheet line as a main row (live nsyn / syn_pct / PR_conv) even if it shares a cell")
    ap.add_argument("--drop-col", action="append", default=[], help="column to leave out of the written sheet")
    ap.add_argument("--orphan-notes", action="store_true", help="note stopped orphan fragments as 'orphan fragment'")
    ap.add_argument("--note", action="append", default=[], metavar="ROOT_ORIG[@BLOCK_PR]=TEXT",
                    help="with --orphan-notes: set this stopped row's notes outright")
    ap.add_argument("--pre-points", metavar="SHEET.tsv",
                    help="two-hop sheet (blocks per hop1_root_id): the one-hop sheet whose rows give each hop-1 cell's xyz")
    a = ap.parse_args()
    import caveclient
    from fafbseg import flywire
    lines, rows = parse_sheet(a.sheet)
    hop2 = "hop1_root_id" in rows[0]["header"]
    if hop2:  # a two-hop sheet: the block's presynaptic cell is the hop-1 cell, not the PR
        for r in rows:
            r["block_pr"] = r["cells"]["hop1_root_id"].strip()
    prs = list(dict.fromkeys(r["block_pr"] for r in rows))
    if hop2:
        if not a.pre_points:
            raise SystemExit("a two-hop sheet needs --pre-points (the one-hop sheet with the hop-1 cells' xyz)")
        _, pre_rows = parse_sheet(a.pre_points)
        pr_xyz = {}
        for pr in pre_rows:
            for k in ("root_id_orig", "root_id_curr"):
                v = pr["cells"][k].strip()
                if v in prs and parse_xyz(pr["cells"]["xyz"]):
                    pr_xyz.setdefault(v, pr["cells"]["xyz"])
    else:
        groups = pd.read_csv(os.path.join(DATA, "pr_groups.tsv"), sep="\t", dtype=str).fillna("")
        pr_xyz = {r.root_id_orig: r.xyz for r in groups.itertuples() if r.root_id_orig in prs}
    CONV = "PR_conv" if "PR_conv" in rows[0]["header"] else "h1_conv"
    missing_pre = [p for p in prs if p not in pr_xyz]
    print(f"{len(rows)} partner rows in {len(prs)} blocks; {'hop-1' if hop2 else 'PR'} points found for {len(pr_xyz)}/{len(prs)}"
          + (f"; without a point: {missing_pre}" if missing_pre else ""), flush=True)
    if missing_pre:
        raise SystemExit("every block's presynaptic cell needs an xyz point")
    client = caveclient.CAVEclient(DATASTACK)
    now = datetime.datetime.now(datetime.timezone.utc)

    # 1. roots at all points (partner rows + PRs)
    pts = sorted({r["cells"]["xyz"] for r in rows if parse_xyz(r["cells"]["xyz"])} | set(pr_xyz.values()))
    throttle()
    found = np.asarray(flywire.locs_to_segments(np.array([parse_xyz(p) for p in pts]) * np.array(VOX), coordinates="nm",
                                                backend="cloudvolume", dataset=DATASTACK)).astype(np.int64)
    at = {p: str(int(r)) for p, r in zip(pts, found)}

    # 2. which listed roots are current
    ids = sorted({x.strip() for r in rows for x in (r["cells"]["root_id_curr"], r["cells"]["root_id_orig"])
                  if x.strip().isdigit()} | set(prs))
    throttle()
    latest_flag = dict(zip(ids, map(bool, client.chunkedgraph.is_latest_roots([int(x) for x in ids]))))

    # 3. forward lineage of every stale listed root (cached, resumable)
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

    def fetch(x):
        throttle()
        try:
            cache[x] = [str(int(y)) for y in client.chunkedgraph.get_latest_roots(int(x))]
        except Exception as e:
            cache[x] = ["ERROR " + str(e)[:80]]

    stale = [x for x in ids if not latest_flag[x] and x not in cache]
    print(f"{len(ids)} listed roots, {sum(not v for v in latest_flag.values())} stale; {len(stale)} new lineage lookups",
          flush=True)
    for k, x in enumerate(stale):
        fetch(x)
        if k % 10 == 9 or k == len(stale) - 1:
            json.dump(cache, open(CACHE, "w"))
            print(f"  {k + 1}/{len(stale)} lineage lookups", flush=True)
    refreshed = set()

    def descends(ra, x):
        """does current root ra descend from listed root x? (refetches a cached lineage once if it says no)"""
        if latest_flag.get(x, False):
            return ra == x
        if ra in cache.get(x, []):
            return True
        if x not in refreshed and x.isdigit():
            refreshed.add(x)
            fetch(x)
            json.dump(cache, open(CACHE, "w"))
        return ra in cache.get(x, [])

    # 4. verdicts
    out = []
    for r in rows:
        c = r["cells"]
        orig, curr, xyz = c["root_id_orig"].strip(), c["root_id_curr"].strip(), c["xyz"]
        ra = at.get(xyz, "") if parse_xyz(xyz) else ""
        v, note, proposed = "ok", "", curr
        if not parse_xyz(xyz):
            v, note = "NO-POINT", "xyz missing or unreadable"
        elif ra == "0":
            v, note = "NO-SEGMENT", "no segment at the point"
        elif latest_flag.get(curr, False):
            if ra != curr:
                v, note = "MISMATCH", f"root_id_curr is current, but the point lies on {ra}"
        elif descends(ra, curr):
            v, proposed, note = "update", ra, f"root_id_curr stale; the point's root {ra} descends from it"
        else:
            v, note = "STALE-NO-LINK", f"root_id_curr stale; the point lies on {ra}, which does not descend from it"
        if v in ("ok", "update") and orig.isdigit() and not descends(ra, orig):
            v, note = "LINEAGE", (note + "; " if note else "") + f"the point's root {ra} does not descend from root_id_orig"
        elif v in ("ok", "update") and not orig.isdigit():
            note = (note + "; " if note else "") + f"root_id_orig '{orig}' is not a root id: lineage not checked"
        out.append(dict(line=r["line"] + 1, block_pr=r["block_pr"], root_id_orig=orig, root_id_curr=curr, xyz=xyz,
                        nsyn=c["nsyn"].strip(), status=c["status"].strip(), annotation=c.get("annotation", "").strip(),
                        root_at_xyz=ra, curr_latest=latest_flag.get(curr, ""), verdict=v,
                        proposed_root_id_curr=proposed if v in ("ok", "update") else "", note=note))
    df = pd.DataFrame(out)
    df["cell"] = df.root_at_xyz

    # 5. live recount
    pr_root = {p: int(at[pr_xyz[p]]) for p in prs}
    throttle()
    syn, ts = live_outputs(sorted(pr_root.values()), client)
    counts = syn.groupby(["pre_pt_root_id", "post_pt_root_id"]).size()
    tot = syn.groupby("pre_pt_root_id").size()
    npart = syn.groupby("pre_pt_root_id").post_pt_root_id.nunique()
    conv = syn.groupby("post_pt_root_id").pre_pt_root_id.nunique()
    df["nsyn_sheet"] = [as_int(x, 0) for x in df.nsyn]
    df["nsyn_live"] = [int(counts.get((pr_root[b], int(c)), 0)) if c.isdigit() else 0 for b, c in zip(df.block_pr, df.cell)]
    pr_cells = {str(v) for v in pr_root.values()}
    df["role"] = "main"
    for (b, c), g in df[df.cell.str.isdigit() & (df.cell != "0")].groupby(["block_pr", "cell"]):
        if c in pr_cells:
            df.loc[g.index, "role"] = "is a PR of this group"
            continue
        if len(g) > 1:  # the record of a cell is the row with the largest sheet nsyn (David, 2026-10-25); ties -> verdict ok first
            main = g.assign(o=(g.verdict != "ok").astype(int), n=-g.nsyn_sheet).sort_values(["n", "o", "line"]).index[0]
            for i in g.index.drop(main):
                df.at[i, "role"] = f"merged with line {df.at[main, 'line']}"
    df.loc[df.line.isin(a.main), "role"] = "main"
    m = df.role == "main"
    df["nsyn_new"] = np.where(m, df.nsyn_live, df.nsyn_sheet)
    df["syn_pct_new"] = [round(100 * n / tot.get(pr_root[b], 1), 1) for n, b in zip(df.nsyn_new, df.block_pr)]
    df["PR_conv_new"] = [int(conv.get(int(c), 0)) if c.isdigit() else "" for c in df.cell]  # also h1_conv for two-hop sheets
    df["diff"] = df.nsyn_new - df.nsyn_sheet
    df["flag"] = np.where(m & (df["diff"].abs() > 2), "CHANGED BY >2", "")

    # 6. live top 20 per block vs the block's rows, with the builder's rule (pr_downstream_build.top_n): exactly 20 rows,
    #    ties broken in favor of cells already in the sheet, then the lower root id - so a tie at the cutoff is not "missing"
    missing = []
    in_sheet = set(df.cell)
    for p in prs:
        r = pr_root[p]
        sub = counts.loc[r] if r in counts.index.get_level_values(0) else pd.Series(dtype=int)
        order = sorted(sub.items(), key=lambda kv: (-int(kv[1]), -int(str(kv[0]) in in_sheet), int(kv[0])))[:20]
        have = set(df[df.block_pr == p].cell)
        for rank, (post, n) in enumerate(order, 1):
            if str(post) not in have:
                missing.append(dict(block_pr=p, rank=rank, root=post, nsyn=int(n), PR_conv=int(conv.get(post, 0))))

    stamp = utc_stamp(now)
    rep = fig(f"pr_sheet_check_{stamp}.tsv")
    df.to_csv(rep, sep="\t", index=False)
    pd.DataFrame(missing).to_csv(rep.replace(".tsv", "_missing_top20.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 110)
    print(f"\nlive check at {ts:%Y-%m-%d %H:%M} UTC; verdicts: {df.verdict.value_counts().to_dict()}")
    print("PR roots at their points:", pr_root)
    print("\nrows needing attention (verdict not ok):")
    print(df[df.verdict != "ok"][["line", "block_pr", "root_id_orig", "root_id_curr", "cell", "verdict", "note"]].to_string(index=False))
    print("\nrows sharing a cell within a block (role):")
    print(df[df.role != "main"][["line", "block_pr", "root_id_orig", "root_id_curr", "cell", "nsyn_sheet", "nsyn_live", "role"]]
          .to_string(index=False))
    ch = df[m & (df["diff"] != 0)]
    print(f"\nnsyn changes on main rows: {len(ch)} of {int(m.sum())} ({int((df.flag != '').sum())} by more than 2):")
    print(ch[["line", "block_pr", "root_id_orig", "cell", "status", "nsyn_sheet", "nsyn_live", "diff", "flag"]].to_string(index=False))
    print(f"\ncells in a PR's live top 20 (builder's rule, exactly 20) that the block lacks: {len(missing)}")
    if missing:
        print(pd.DataFrame(missing).to_string(index=False))
    print("\nblock totals now:", {p: (int(tot.get(pr_root[p], 0)), int(npart.get(pr_root[p], 0))) for p in prs})
    print(f"report: {os.path.relpath(rep, os.path.dirname(FIGURES))}")

    if a.write_sheet:
        fixes = {}
        for s in a.set:
            ln, rest = s.split(":", 1)
            col, val = rest.split("=", 1)
            fixes.setdefault(int(ln) - 1, {})[col] = val
        by_line = {int(r.line) - 1: r for r in df.itertuples()}
        not_stopped = set(df[df.status.str.lower() != "stopped"].cell)
        glia_cells = set(df[df.annotation.str.lower().str.contains("glia")].cell)
        overrides = dict(s.split("=", 1) for s in a.note)
        hdr_of = {r["line"]: r["header"] for r in rows}
        width = max(len(ln.split("\t")) for ln in lines)

        def pad(c):
            return c + [""] * (width - len(c))

        # updated cells of every existing row
        body = {}
        for i in by_line:
            r, h = by_line[i], hdr_of[i]
            c = pad([x.strip() for x in lines[i].split("\t")])
            if r.verdict == "update":
                c[h.index("root_id_curr")] = r.proposed_root_id_curr
            if r.role == "main" and r.cell.isdigit() and r.cell != "0":
                c[h.index("nsyn")] = str(int(r.nsyn_new))
                c[h.index("syn_pct")] = f"{r.syn_pct_new:g}"
                c[h.index(CONV)] = str(r.PR_conv_new)
            v = parse_xyz(c[h.index("xyz")])
            if v:
                c[h.index("xyz")] = fmt_vox(v)
            for col, val in fixes.get(i, {}).items():
                c[h.index(col)] = val
            if a.orphan_notes and c[h.index("status")].lower() == "stopped":
                old = c[h.index("notes")]
                o = c[h.index("root_id_orig")]
                forced = overrides.get(f"{o}@{c[0]}", overrides.get(o))
                if forced is not None:
                    c[h.index("notes")] = forced
                elif r.cell in not_stopped or (r.cell in glia_cells and "fragment" not in old.lower()):
                    print(f"  not an orphan: {o} (block {c[0]}, cell {r.cell}); notes kept: {old!r}")
                else:
                    keep = [x.strip().rstrip(";").strip() for x in re.split(r"(?<=[.?;])\s+", old.strip()) if x.strip()]
                    c[h.index("notes")] = "; ".join(["orphan fragment"] + [x for x in keep if not FRAG.search(x)])
                if c[h.index("notes")] != old:
                    print(f"  orphan note {o} (block {c[0]}): {old!r} -> {c[h.index('notes')]!r}")
            body[i] = c

        # rows for cells in a PR's live top 20 that its block lacks (status NEW)
        added = {p: [] for p in prs}
        add = [m for m in missing if m["nsyn"] >= a.add_missing] if a.add_missing else []
        if add:
            from faab.partners.points import cell_points
            tracked = {}
            for r in df[df.role == "main"].itertuples():
                tracked.setdefault(r.cell, r)
            untracked = {int(m["root"]) for m in add if str(m["root"]) not in tracked}
            pts, _ = cell_points(untracked, client, time.time())
            for m in add:
                p, root = m["block_pr"], str(m["root"])
                h = rows[[x["block_pr"] for x in rows].index(p)]["header"]
                blk_rows = [body[x["line"]] for x in rows if x["block_pr"] == p]
                first = blk_rows[0]
                c = pad([""] * len(h))
                for k in range(len(h)):  # block-constant columns (PR ids, PRtype, hop1_root_id, h1_nsyn) come along
                    if h[k] and h[k] not in ("root_id_orig", "root_id_curr", "xyz", "nsyn", "syn_pct", CONV) + tuple(
                            x for x in ("annotation", "status", "pr_date", "proofreader", "chk_date", "checker", "notes",
                                        "SeaTable", "link")) and len({x[k] for x in blk_rows}) == 1:
                        c[k] = first[k]
                c[h.index("root_id_curr")], c[h.index("nsyn")] = root, str(m["nsyn"])
                c[h.index("syn_pct")] = f"{round(100 * m['nsyn'] / tot.get(pr_root[p], 1), 1):g}"
                c[h.index(CONV)], c[h.index("status")] = str(m["PR_conv"]), "NEW"
                if root in tracked:
                    # the cell is tracked in another block: keep its tracking columns; NEW only if it has no status
                    t_ = tracked[root]
                    src, sh = body[int(t_.line) - 1], hdr_of[int(t_.line) - 1]
                    c[h.index("root_id_orig")], c[h.index("xyz")] = t_.root_id_orig, fmt_vox(parse_xyz(t_.xyz))
                    for col in ("annotation", "status", "pr_date", "proofreader", "chk_date", "checker"):
                        c[h.index(col)] = src[sh.index(col)]
                    c[h.index("status")] = c[h.index("status")] or "NEW"
                    c[h.index("notes")] = f"same cell as root_id_orig {t_.root_id_orig} in block {t_.block_pr}"
                else:
                    c[h.index("root_id_orig")] = root
                    c[h.index("xyz")], how = pts.get(int(root), ("", "no point"))
                    print(f"  new row {root}: xyz {c[h.index('xyz')]} ({how})")
                added[p].append(c)

        # blocks re-sorted by nsyn (stable), title totals refreshed, every cell stripped
        block_lines = {}
        for x in rows:
            block_lines.setdefault(x["block_pr"], []).append(x["line"])
        first_line = {v[0]: k for k, v in block_lines.items()}
        skip = {i for v in block_lines.values() for i in v[1:]}
        new, keep = [], None
        for i, ln in enumerate(lines):
            if i in skip:
                continue
            if i in first_line:
                p = first_line[i]
                h = hdr_of[i]
                blk = [body[j] for j in block_lines[p]] + added[p]
                blk.sort(key=lambda c: -as_int(c[h.index("nsyn")], 0))
                new += ["\t".join(c[k] for k in keep) for c in blk]
                continue
            c = [x.strip() for x in ln.split("\t")]
            if c[0] in ("upstream_PR", "PR_root_id"):
                keep = [k for k in range(width) if k >= len(c) or c[k] not in a.drop_col]
                c = pad(c)
                new.append("\t".join(c[k] for k in keep))
                continue
            mt = TITLE.match(c[0])
            mt2 = TITLE2.match(c[0])
            if mt and mt.group(1) in pr_root:
                r = pr_root[mt.group(1)]
                c[0] = (f"{mt.group(1)}{mt.group(2)}{int(tot.get(r, 0))} output synapses across {int(npart.get(r, 0))} "
                        f"partners (updated {ts:%Y-%m-%d %H:%M} UTC)")
            elif mt2 and mt2.group(2) in pr_root:  # two-hop block title: refresh the hop-1 cell's current root and totals
                r = pr_root[mt2.group(2)]
                c[0] = (f"{mt2.group(1)}{r}{mt2.group(4)}{int(tot.get(r, 0))} output synapses across {int(npart.get(r, 0))} "
                        f"partners (updated {ts:%Y-%m-%d %H:%M} UTC)")
            n_out = width - len(a.drop_col)
            c = pad(c)
            new.append("\t".join(c[:n_out] if not any(c[n_out:]) else c))
        open(a.write_sheet, "w", encoding="utf-8").write("\n".join(new))
        print(f"wrote {a.write_sheet}: {sum(len(v) for v in added.values())} NEW rows, blocks sorted by nsyn")


if __name__ == "__main__":
    main()
