"""Block membership by connectivity - DRY RUN (David 2026-11-21: "If a large branch is merged in or split off during
proofreading, that could change the partners"; rules in memory faab-block-membership-refresh). Nothing is written to
the workbook: every proposed change goes to a report.

Every row's cell is the root at its xyz when the check runs (as in the sheet update; cells edited since the
update have new roots). For every row of the connectivity tabs, the row's connection is recounted from the local synapse snapshot, caught up
to one fixed time (one CAVE catch-up for all queries), and the block's own rule is applied again:
  PRg1hop1 / PRg2hop1  block = one photoreceptor (upstream_PR, today's root from the PRs tab): its top 20 partners
  PRg1hop2             block = one hop-1 cell (hop1_root_id, today's root from PRg1hop1): its top 20 partners with
                       nsyn >= 2; which hop-1 cells get a block (rule B: per PR the cells with nsyn >= 3, at most 5;
                       fewer than 3 -> the top 3 down to nsyn 2) is recomputed and reported per block
  PRg1toT4T5           PR+1: >= 3 from a PR group 1 photoreceptor and on to a T4/T5 target through a PR+2 cell;
                       PR+2: >= 3 from a PR+1 cell and >= 3 onto a target; PR+3: the targets (only checked for a path);
                       T4T5-1: >= 3 onto a target; T4T5-2 (also its LOW PRIORITY rows): >= 3 onto a seed = a T4T5-1
                       cell other than the wide-field ones its title names (and any T4T5-1 row with >= 3 x the median
                       cable); notT4T5-1: recounted only
Margins (so cells do not flip in and out with small edits): a row is 'below' its rule when it fails it, and 'clearly
below' only when it is short by 2 or more (top-N rules: nsyn <= the cutoff count - 2, or 0; '>= 3' rules: <= 1). A new
cell is added only when it passes strictly (top-N: nsyn above the cutoff count, so no ties at the cut).
Actions proposed (David's rules): clearly below + untracked (status blank / NEW / NEW-DUP, no proofreader) -> remove;
clearly below + tracked -> re-rank in the block and note '<date>: now below block criteria (nsyn=<n>)'; new
qualifying cells -> insert by nsyn; nsyn recounted on every row.
Not yet covered: T4T5s (lobula-plate output rule) and LMCs (photoreceptor input rule).
Writes figures/sheets/<stamp>_membership_dryrun.tsv (one line per proposed change) and prints a summary per tab and
block. Read-only. Usage: membership_check.py BOOK.xlsx"""
import collections
import datetime
import re
import statistics
import sys

import openpyxl
import pandas as pd

from faab import synsnap
from faab.clients.cave import client as cave_client
from faab.config import fig
from faab.partners.sheet import cell_text as s, utc_stamp

ID = re.compile(r"\d{18}")
UNTRACKED = {"", "new", "new-dup", "new?"}
TOP, HOP2_MIN, THR = 20, 2, 3


def read_tab(wb, tab):
    """data rows of a tab: dict(row, title, v (values by header)); titles = text-only lines above the rows"""
    ws, out, cols, title = wb[tab], [], None, ""
    for r, row in enumerate(ws.iter_rows(values_only=True), 1):
        v = [s(x) for x in row]
        if "xyz" in v and ("root_id_curr" in v or "root_id_orig" in v):
            cols = {h: i for i, h in enumerate(v) if h}
            continue
        if cols is None:
            if v and v[0]:
                title = v[0]
            continue
        vals = {h: (v[i] if i < len(v) else "") for h, i in cols.items()}
        if ID.fullmatch(vals.get("root_id_curr", "")) or ID.fullmatch(vals.get("root_id_orig", "")):
            out.append(dict(row=r, title=title, v=vals))
        elif v and v[0] and not any(v[1:4]):
            title = v[0]
    return out


def cell(x):
    """the row's cell NOW: the root at its xyz (looked up when the check runs - cells edited since the sheet update
    have new roots), else root_id_curr"""
    if x.get("now"):
        return x["now"]
    v = x["v"]
    return v.get("root_id_curr") if ID.fullmatch(v.get("root_id_curr", "")) else v.get("root_id_orig", "")


def tracked(x):
    v = x["v"]
    return v.get("status", "").strip().lower() not in UNTRACKED or bool(v.get("proofreader", "").strip())


def first_int(t):
    m = re.search(r"\d+", t or "")
    return int(m.group()) if m else 0


def main():
    book = sys.argv[1]
    wb = openpyxl.load_workbook(book, read_only=True)
    tabs = {t: read_tab(wb, t) for t in ("PRs", "PRg1hop1", "PRg2hop1", "PRg1hop2", "PRg1toT4T5") if t in wb.sheetnames}
    allrows = [x for rows in tabs.values() for x in rows if len(re.findall(r"-?\d+(?:\.\d+)?", x["v"].get("xyz", ""))) == 3]
    from faab.clients.cave import roots_at_xyz
    at = roots_at_xyz([x["v"]["xyz"] for x in allrows])
    for x, r in zip(allrows, at):
        if r:
            x["now"] = str(int(r))
    moved = sum(1 for x in allrows if x.get("now") and x["now"] != x["v"].get("root_id_curr"))
    print(f"{len(allrows)} rows looked up at their xyz: {moved} on a root other than root_id_curr now", flush=True)
    curr_of = collections.defaultdict(collections.Counter)  # root_id_orig -> today's roots of its rows
    for rows in tabs.values():
        for x in rows:
            o, c = x["v"].get("root_id_orig", ""), cell(x)
            if ID.fullmatch(o) and ID.fullmatch(c):
                curr_of[o][c] += 1
    now_of = lambda i: curr_of[i].most_common(1)[0][0] if curr_of.get(i) else i  # noqa: E731
    prs = collections.defaultdict(set)  # PR groups 1 and 2 from the PRs tab ('group N' lines)
    ws = wb["PRs"]
    g = None
    for row in ws.iter_rows(values_only=True):
        a = s(row[0]) if row else ""
        m = re.fullmatch(r"group (\d+)", a)
        if m:
            g = int(m.group(1))
        elif g in (1, 2) and ID.fullmatch(a):
            prs[g].add(a)
    pr_now = {g: {now_of(o) for o in ids} for g, ids in prs.items()}
    t5 = tabs["PRg1toT4T5"]
    hop = lambda x: x["v"].get("hop", "")  # noqa: E731
    targets = {cell(x) for x in t5 if hop(x) == "PR+3"}
    in1 = [x for x in t5 if hop(x) == "T4T5-1"]
    t2title = next((x["title"] for x in t5 if hop(x) == "T4T5-2" and x["title"].startswith("T4T5-2")), "")
    excl = {now_of(i) for i in ID.findall(t2title.split("EXCLUDED", 1)[-1].split("T4T5-1_root_id")[0])}
    cab = [first_int(x["v"].get("cable_um")) for x in in1 if first_int(x["v"].get("cable_um"))]
    wide = {cell(x) for x in in1 if cab and first_int(x["v"].get("cable_um")) >= 3 * statistics.median(cab)}
    seeds = {cell(x) for x in in1} - excl - wide
    not_t = {now_of(i) for x in t5 if hop(x).startswith("PR+3, not") for i in [cell(x)]}
    hop1_U = {}
    for x in tabs["PRg1hop2"]:
        h = x["v"].get("hop1_root_id", "")
        if ID.fullmatch(h):
            hop1_U[h] = now_of(h)

    when = datetime.datetime.now(datetime.timezone.utc)
    snap, client = synsnap.Snapshot(), cave_client()
    pre1 = set().union(*pr_now.values()) | set(hop1_U.values())
    E = snap.edges(pre=[int(i) for i in pre1], when=when, client=client)
    X = set(E[E.pre.isin([int(i) for i in pr_now.get(1, ())]) & (E.n >= THR)].post.astype(str))
    xrows = {cell(x) for x in t5 if hop(x) == "PR+1"}
    E2 = snap.edges(pre=[int(i) for i in (X | xrows) - pre1], when=when, client=client)
    Ein = snap.edges(post=[int(i) for i in targets | seeds | not_t], when=when, client=client)
    out_n = collections.defaultdict(dict)
    for e in pd.concat([E, E2]).itertuples():
        out_n[str(e.pre)][str(e.post)] = int(e.n)
    in_n = collections.defaultdict(dict)
    for e in Ein.itertuples():
        in_n[str(e.post)][str(e.pre)] = int(e.n)
    n_onto = lambda c, S: max([in_n[t].get(c, 0) for t in S] or [0])  # noqa: E731
    print(f"snapshot caught up to {when:%Y-%m-%d %H:%M} UTC: {len(E) + len(E2)} output edges, {len(Ein)} input edges; "
          f"{len(targets)} targets, {len(seeds)} T4T5-2 seeds ({len(excl | wide)} wide-field left out)", flush=True)

    rep = []
    day = f"{when:%Y-%m-%d}"

    def judge(tab, block, x, n_now, rule_ok, clearly_below, rule):
        n_old = first_int(x["v"].get("nsyn"))
        tr = tracked(x)
        act = ""
        if clearly_below:
            act = ("flag + re-rank: '" + f"{day}: now below block criteria (nsyn={n_now})'") if tr else "remove (untracked)"
        rep.append(dict(tab=tab, block=block[:80], row=x["row"], root=cell(x), tracked=tr, nsyn_old=n_old, nsyn_now=n_now,
                        nsyn_changed=n_old != n_now, rule=rule, passes=rule_ok, clearly_below=clearly_below, action=act))

    def add(tab, block, c, n, rule):
        rep.append(dict(tab=tab, block=block[:80], row="", root=c, tracked=False, nsyn_old="", nsyn_now=n,
                        nsyn_changed=False, rule=rule, passes=True, clearly_below=False, action="add (NEW, by nsyn)"))

    # PRg1hop1 / PRg2hop1: top 20 partners of each PR
    for tab in ("PRg1hop1", "PRg2hop1"):
        by_block = collections.defaultdict(list)
        for x in tabs.get(tab, []):
            by_block[x["v"].get("upstream_PR", "")].append(x)
        for u_orig, rows in by_block.items():
            U = now_of(u_orig)
            part = sorted(out_n.get(U, {}).items(), key=lambda kv: (-kv[1], kv[0]))
            top = part[:TOP]
            cut = top[-1][1] if len(top) == TOP else 1
            for x in rows:
                n = out_n.get(U, {}).get(cell(x), 0)
                judge(tab, f"PR {u_orig}", x, n, n >= cut and n > 0, n == 0 or n <= cut - 2, f"top {TOP} (cutoff {cut})")
            have = {cell(x) for x in rows}
            for c, n in top:
                if n > cut and c not in have:
                    add(tab, f"PR {u_orig}", c, n, f"top {TOP} (cutoff {cut})")

    # PRg1hop2: top 20 (>= 2) per hop-1 cell, and which hop-1 cells get a block
    by_block = collections.defaultdict(list)
    for x in tabs.get("PRg1hop2", []):
        by_block[x["v"].get("hop1_root_id", "")].append(x)
    for h_orig, rows in by_block.items():
        U = hop1_U.get(h_orig, h_orig)
        part = sorted(((c, n) for c, n in out_n.get(U, {}).items() if n >= HOP2_MIN), key=lambda kv: (-kv[1], kv[0]))
        top = part[:TOP]
        cut = max(HOP2_MIN, top[-1][1]) if len(top) == TOP else HOP2_MIN
        for x in rows:
            n = out_n.get(U, {}).get(cell(x), 0)
            judge("PRg1hop2", f"hop-1 {h_orig}", x, n, n >= cut, n == 0 or n <= cut - 2, f"top {TOP}, nsyn >= {HOP2_MIN} (cutoff {cut})")
        have = {cell(x) for x in rows}
        for c, n in top:
            if n > cut and c not in have:
                add("PRg1hop2", f"hop-1 {h_orig}", c, n, f"top {TOP}, nsyn >= {HOP2_MIN} (cutoff {cut})")
    own = pr_now.get(1, set())
    skip = {cell(x) for x in tabs["PRg1hop1"] if "glia" in x["v"].get("annotation", "").lower()
            or x["v"].get("status", "").strip().lower() == "stopped"}
    sel = set()
    for U in own:
        d = sorted(((c, n) for c, n in out_n.get(U, {}).items() if n >= 2 and c not in own and c not in skip),
                   key=lambda kv: (-kv[1], kv[0]))
        strong = [c for c, n in d if n >= 3]
        sel |= set(strong[:5]) if len(strong) >= 3 else {c for c, _ in d[:3]}
    have_blocks = set(hop1_U.values())
    hop2_blocks = dict(new=sorted(sel - have_blocks), gone=sorted(have_blocks - sel))

    # PRg1toT4T5
    Y = {c for c in set().union(*[set(out_n.get(x, {})) for x in X]) if n_onto(c, targets) >= THR
         and any(out_n.get(x, {}).get(c, 0) >= THR for x in X)} if X else set()
    Xp = {x for x in X if any(out_n.get(x, {}).get(y, 0) >= THR for y in Y)}
    listed = {cell(x) for x in t5}
    for x in t5:
        c, h, blk = cell(x), hop(x), x["title"].split(":")[0][:60]
        if h == "PR+1":
            n = max([out_n.get(U, {}).get(c, 0) for U in own] or [0])
            path = any(out_n.get(c, {}).get(y, 0) >= THR for y in Y)
            judge("PRg1toT4T5", "PR+1", x, n, n >= THR and path, n <= 1, ">= 3 from a PR group 1 PR, on to a target")
        elif h == "PR+2":
            n = max([out_n.get(p, {}).get(c, 0) for p in Xp | {cell(z) for z in t5 if hop(z) == "PR+1"}] or [0])
            n2 = n_onto(c, targets)
            judge("PRg1toT4T5", "PR+2", x, n, n >= THR and n2 >= THR, n <= 1 or n2 <= 1, ">= 3 from a PR+1 cell, >= 3 onto a target")
        elif h == "T4T5-1":
            n = n_onto(c, targets)
            judge("PRg1toT4T5", "T4T5-1", x, n, n >= THR, n <= 1, ">= 3 onto a target (max over targets)")
        elif h == "T4T5-2":
            n = n_onto(c, seeds)
            judge("PRg1toT4T5", blk if blk.startswith("LOW") else "T4T5-2", x, n, n >= THR, n <= 1,
                  ">= 3 onto a T4T5-1 seed (max over seeds)")
        elif h == "notT4T5-1":
            n = n_onto(c, not_t)
            judge("PRg1toT4T5", "notT4T5-1", x, n, True, False, "recount only")
    for t in targets:
        if not any(n_onto(y, {t}) >= THR for y in Y):
            rep.append(dict(tab="PRg1toT4T5", block="PR+3", row="", root=t, tracked=True, nsyn_old="", nsyn_now="",
                            nsyn_changed=False, rule="reached from PR group 1 at >= 3 per connection", passes=False,
                            clearly_below=False, action="report: no PR group 1 path at >= 3 now"))
    skip5 = listed | targets | set().union(*pr_now.values())
    for c in sorted(Xp - skip5):
        add("PRg1toT4T5", "PR+1", c, max(out_n.get(U, {}).get(c, 0) for U in own), "on a PR group 1 path")
    for c in sorted(Y - skip5):
        add("PRg1toT4T5", "PR+2", c, n_onto(c, targets), "on a PR group 1 path")
    cand1 = collections.Counter()
    for t in targets:
        for c, n in in_n[t].items():
            cand1[c] = max(cand1[c], n)
    for c, n in sorted(cand1.items(), key=lambda kv: -kv[1]):
        if n >= THR and c not in skip5 | Y:
            add("PRg1toT4T5", "T4T5-1", c, n, ">= 3 onto a target")
    cand2 = collections.Counter()
    for t in seeds:
        for c, n in in_n[t].items():
            cand2[c] = max(cand2[c], n)
    for c, n in sorted(cand2.items(), key=lambda kv: -kv[1]):
        if n >= THR and c not in skip5 | Y and cand1.get(c, 0) < THR:  # a T4T5-1 qualifier is added there instead
            add("PRg1toT4T5", "T4T5-2", c, n, ">= 3 onto a T4T5-1 seed")

    R = pd.DataFrame(rep)
    path = fig(f"{utc_stamp()}_membership_dryrun.tsv")
    R.to_csv(path, sep="\t", index=False)
    print(f"\nDRY RUN on {book.rsplit('/', 1)[-1]} (nothing written to the sheet)")
    for tab, g in R.groupby("tab", sort=False):
        rows = g[g.row != ""]
        print(f"\n{tab}: {len(rows)} rows checked; nsyn changed on {int(rows.nsyn_changed.sum())}; below their rule "
              f"{int((~rows.passes.astype(bool)).sum())}, clearly below {int(rows.clearly_below.astype(bool).sum())} "
              f"-> remove {int((rows.action == 'remove (untracked)').sum())} untracked, flag + re-rank "
              f"{int(rows.action.str.startswith('flag').sum())} tracked; add {int((g.action.str.startswith('add')).sum())} new cells")
        for b, gb in g.groupby("block", sort=False):
            r2 = gb[gb.row != ""]
            k = dict(changed=int(r2.nsyn_changed.sum()), remove=int((r2.action == "remove (untracked)").sum()),
                     flag=int(r2.action.str.startswith("flag").sum()), add=int(gb.action.str.startswith("add").sum()),
                     report=int(gb.action.str.startswith("report").sum()))
            if any(v for kk, v in k.items() if kk != "changed") or k["changed"] >= 5:
                print(f"   {b[:60]:60s} rows {len(r2):4d}  nsyn changed {k['changed']:3d}  remove {k['remove']:3d}  "
                      f"flag {k['flag']:3d}  add {k['add']:3d}" + (f"  report {k['report']}" if k["report"] else ""))
    print(f"\nPRg1hop2 blocks (rule B on today's PR group 1 -> hop-1 counts): {len(hop2_blocks['new'])} hop-1 cells would "
          f"get a new block {hop2_blocks['new'][:10]}; {len(hop2_blocks['gone'])} blocks' hop-1 cells no longer qualify "
          f"{hop2_blocks['gone'][:10]}")
    print(f"written {path}")


if __name__ == "__main__":
    main()
