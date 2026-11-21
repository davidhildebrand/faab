"""Block membership by connectivity (David 2026-11-21: "If a large branch is merged in or split off during
proofreading, that could change the partners"; rules in memory faab-block-membership-refresh). Dry run by default
(a report only); with --apply, the changes are written into a copy of the workbook (a step of every update_book run,
before the T4/T5 sort, so the new rows go through the SeaTable match and sync like any other).

Every row's cell is the root at its xyz when the check runs (as in the sheet update; cells edited since the update have
new roots). Each row's connection is recounted from the local synapse snapshot, caught up once to a fixed time, and the
block's own rule is applied again, AS WRITTEN, for rows that stay and for cells that join (David 2026-11-21):
  PRg1hop1 / PRg2hop1  block = one photoreceptor (upstream_PR): its top 20 partners (ties at the cut: cells the block
                       lists already first, then tracked cells, then the lower root id)
  PRg1hop2             block = one hop-1 cell (hop1_root_id): its top 20 partners with nsyn >= 2; the hop-1 cells
                       themselves by rule B on today's PR group 1 counts (per PR the cells with nsyn >= 3, at most 5;
                       fewer than 3 such -> its top 3 down to nsyn 2; own PRs, glia and stopped orphans skipped)
  PRg1toT4T5           PR+1: >= 3 from a PR group 1 photoreceptor and on to a target through a PR+2 cell; PR+2: >= 3
                       from a PR+1 cell and >= 3 onto a target; T4T5-1: >= 3 onto a target; T4T5-2 (and its LOW
                       PRIORITY rows): >= 3 onto a seed = a T4T5-1 cell other than the wide-field ones its title names
                       (and any T4T5-1 row with >= 3 x the median cable); a T4T5-1 / T4T5-2 row on a photoreceptor
                       path (column A light blue, the T4/T5 sort's rule) passes too; PR+3 targets only checked for a path;
                       notT4T5-1 recounted only
What happens (David 2026-11-21):
  - every row: nsyn recounted (hop tabs: nsyn, syn_pct, PR_conv / h1_conv; T4T5-1 / T4T5-2: the partner lists, nsyn,
    nsyn_out, T4T5_reached as the builders define them; PR+1 / PR+2 / notT4T5-1: nsyn per listed id); block titles'
    counts and 'updated' times follow;
  - a row whose cell no longer passes: WORKED ON (a status other than blank / NEW / NEW-DUP, or a proofreader) -> moved
    down in its block, below every row that passes, with the note '<date>: now below block criteria (nsyn=<n>)' (added
    once; its nsyn kept current; cleared when the cell passes again); UNTRACKED -> removed;
  - a cell that now passes and has no row: added as NEW in its block by nsyn (tracking carried from the other tabs,
    a point on the cell - never a synapse - from its L2 skeleton, cable / neuropils); new T4T5-2 cells that are very
    large (>= 1.5 mm of cable or >= 15% outside the right optic lobe) go to the LOW PRIORITY block, new PR+2 cells that
    large are not added (reported);
  - PRg1hop2 blocks: a hop-1 cell that now qualifies gets a new block (before the tab's notes); a block whose hop-1
    cell no longer qualifies stays, its rows handled as rows below the criteria, its title noted.
Not covered yet: T4T5s (lobula-plate rule) and LMCs (photoreceptor input rule).
Writes figures/<topic>/<stamp>_membership_<dryrun|changes>.tsv (one line per row decision / added cell); with --apply
also tmp/<stamp>_membership_full.xlsx and prints 'full workbook <path>' and 'changed tabs: ...'. Read-only CAVE.
Usage: membership_check.py BOOK.xlsx [--apply]"""
import collections
import copy
import datetime
import importlib.util
import os
import re
import statistics
import sys
import time

import openpyxl
import pandas as pd

from faab import synsnap
from faab.clients.cave import client as cave_client, roots_at_xyz
from faab.config import TMP, fig
from faab.partners.sheet import cell_text as s, utc_stamp

ID = re.compile(r"\d{18}")
UNTRACKED = {"", "new", "new-dup", "new?"}
TOP, HOP2_MIN, THR = 20, 2, 3
BELOW = re.compile(r"(\d{4}-\d{2}-\d{2}): now below block criteria \(([^)]*)\)")
LARGE_CABLE, LARGE_OUTSIDE = 1500.0, 0.15
PATH_FILL = "FFCCE8F8"  # column A of PRg1toT4T5 rows on a photoreceptor path (scripts/partners/t4t5_sort.py)
TABS = ("PRs", "PRg1hop1", "PRg2hop1", "PRg1hop2", "PRg1toT4T5")


def read_tab(ws, tab):
    """data rows of a tab: dict(tab, row, title, hdr (header row), cols, v (values by header))"""
    out, cols, title, hdr = [], None, "", None
    for r, row in enumerate(ws.iter_rows(values_only=True), 1):
        v = [s(x) for x in row]
        if "xyz" in v and ("root_id_curr" in v or "root_id_orig" in v):
            cols, hdr = {h: i for i, h in enumerate(v) if h}, r
            continue
        if cols is None:
            if v and v[0]:
                title = v[0]
            continue
        vals = {h: (v[i] if i < len(v) else "") for h, i in cols.items()}
        if ID.fullmatch(vals.get("root_id_curr", "")) or ID.fullmatch(vals.get("root_id_orig", "")):
            out.append(dict(tab=tab, row=r, title=title, hdr=hdr, cols=cols, v=vals))
        elif v and v[0] and not any(v[1:4]):
            title = v[0]
    return out


def cell(x):
    """the row's cell NOW: the root at its xyz (looked up when the check runs), else root_id_curr"""
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


def ranked(counts, tracked_cells, min_n=1, listed=()):
    """partners by count; ties: cells the block lists already first (a tie carries no information, so a listed cell
    keeps its place - otherwise edits that only change root ids would swap tied rows in and out), then tracked cells,
    then the lower root id (the builders' rule)"""
    listed = set(listed)
    return sorted(((c, n) for c, n in counts.items() if n >= min_n),
                  key=lambda kv: (-kv[1], kv[0] not in listed, kv[0] not in tracked_cells, int(kv[0])))


def analyze(book):
    wb = openpyxl.load_workbook(book, read_only=True)
    tabs = {t: read_tab(wb[t], t) for t in TABS if t in wb.sheetnames}
    allrows = [x for rows in tabs.values() for x in rows if len(re.findall(r"-?\d+(?:\.\d+)?", x["v"].get("xyz", ""))) == 3]
    for x, r in zip(allrows, roots_at_xyz([x["v"]["xyz"] for x in allrows])):
        if r:
            x["now"] = str(int(r))
    moved = sum(1 for x in allrows if x.get("now") and x["now"] != x["v"].get("root_id_curr"))
    print(f"{len(allrows)} rows looked up at their xyz: {moved} on a root other than root_id_curr now", flush=True)
    curr_of = collections.defaultdict(collections.Counter)
    for rows in tabs.values():
        for x in rows:
            o, c = x["v"].get("root_id_orig", ""), cell(x)
            if ID.fullmatch(o) and ID.fullmatch(c):
                curr_of[o][c] += 1
    now_of = lambda i: curr_of[i].most_common(1)[0][0] if curr_of.get(i) else i  # noqa: E731
    tracked_cells = {cell(x) for rows in tabs.values() for x in rows if tracked(x)}
    prs, pr_orig_of, g = collections.defaultdict(set), {}, None
    for row in wb["PRs"].iter_rows(values_only=True):
        a = s(row[0]) if row else ""
        m = re.fullmatch(r"group (\d+)", a)
        if m:
            g = int(m.group(1))
        elif g in (1, 2) and ID.fullmatch(a):
            prs[g].add(a)
    pr_now = {g: {now_of(o) for o in ids} for g, ids in prs.items()}
    pr_class = {x["v"].get("root_id_orig", ""): x["v"].get("class", "") for x in tabs["PRs"]}
    for g, ids in prs.items():
        for o in ids:
            pr_orig_of[now_of(o)] = o
    t5 = tabs["PRg1toT4T5"]
    hop = lambda x: x["v"].get("hop", "")  # noqa: E731
    # rows on a photoreceptor path (column A light blue, set by the T4/T5 sort, David 2026-11-12) pass the T4T5-1 /
    # T4T5-2 rule too: that sort adds such cells whatever their synapses onto the seeds, so removing them here would
    # only have it add them back
    wsf = openpyxl.load_workbook(book)["PRg1toT4T5"]
    for x in t5:
        f = wsf.cell(x["row"], 1).fill
        x["on_path"] = bool(f is not None and f.fill_type == "solid" and f.fgColor.rgb == PATH_FILL)
    targets = {cell(x) for x in t5 if hop(x) == "PR+3"}
    in1 = [x for x in t5 if hop(x) == "T4T5-1"]
    t2title = next((x["title"] for x in t5 if hop(x) == "T4T5-2" and x["title"].startswith("T4T5-2")), "")
    excl = {now_of(i) for i in ID.findall(t2title.split("EXCLUDED", 1)[-1].split("T4T5-1_root_id")[0])}
    cab = [first_int(x["v"].get("cable_um")) for x in in1 if first_int(x["v"].get("cable_um"))]
    wide = {cell(x) for x in in1 if cab and first_int(x["v"].get("cable_um")) >= 3 * statistics.median(cab)}
    seeds = {cell(x) for x in in1} - excl - wide
    not_t = {cell(x) for x in t5 if hop(x).startswith("PR+3, not")}
    hop1_U = {}
    for x in tabs["PRg1hop2"]:
        h = x["v"].get("hop1_root_id", "")
        if ID.fullmatch(h):
            hop1_U[h] = now_of(h)
    own = pr_now.get(1, set())
    skip_h1 = {cell(x) for x in tabs["PRg1hop1"] if "glia" in x["v"].get("annotation", "").lower()
               or x["v"].get("status", "").strip().lower() == "stopped"}

    when = datetime.datetime.now(datetime.timezone.utc)
    snap, client = synsnap.Snapshot(), cave_client()
    pre1 = set().union(*pr_now.values()) | set(hop1_U.values())
    E = snap.edges(pre=[int(i) for i in pre1], when=when, client=client)
    out_n = collections.defaultdict(dict)
    for e in E.itertuples():
        out_n[str(e.pre)][str(e.post)] = int(e.n)
    X = {c for U in own for c, n in out_n.get(U, {}).items() if n >= THR}
    # rule B on today's counts -> the hop-1 cells that should have a PRg1hop2 block
    sel = collections.defaultdict(list)  # hop-1 cell -> [(PR orig, n)]
    for U in own:
        d = ranked({c: n for c, n in out_n.get(U, {}).items() if c not in own and c not in skip_h1}, tracked_cells, 2,
                   listed=set(hop1_U.values()))
        strong = [(c, n) for c, n in d if n >= 3]
        for c, n in (strong[:5] if len(strong) >= 3 else d[:3]):
            sel[c].append((pr_orig_of.get(U, U), n))
    new_h1 = sorted(set(sel) - set(hop1_U.values()))
    xrows = {cell(x) for x in t5 if hop(x) == "PR+1"}
    E2 = snap.edges(pre=[int(i) for i in (X | xrows | set(new_h1)) - pre1], when=when, client=client)
    for e in E2.itertuples():
        out_n[str(e.pre)][str(e.post)] = int(e.n)
    Ein = snap.edges(post=[int(i) for i in targets | seeds | not_t], when=when, client=client)
    in_n = collections.defaultdict(dict)
    for e in Ein.itertuples():
        in_n[str(e.post)][str(e.pre)] = int(e.n)
    n_onto = lambda c, S: max([in_n[t].get(c, 0) for t in S] or [0])  # noqa: E731
    print(f"snapshot caught up to {when:%Y-%m-%d %H:%M} UTC: {len(E) + len(E2)} output edges, {len(Ein)} input edges; "
          f"{len(targets)} targets, {len(seeds)} T4T5-2 seeds ({len(excl | wide)} wide-field left out)", flush=True)

    A = dict(book=book, when=when, out_n=out_n, in_n=in_n, targets=targets, seeds=seeds, own=own, pr_now=pr_now,
             pr_orig_of=pr_orig_of, pr_class=pr_class, now_of=now_of, hop1_U=hop1_U, sel=sel, rows=[], adds=[],
             blocks=[], titles={})
    day = f"{when:%Y-%m-%d}"

    def judge(x, block, n_now, ok, rule, extra=None):
        tr = tracked(x)
        act = "" if ok else ("move down + note" if tr else "remove (untracked)")
        A["rows"].append(dict(tab=x["tab"], block=block, row=x["row"], root=cell(x), tracked=tr,
                              nsyn_old=first_int(x["v"].get("nsyn")), nsyn_now=n_now, passes=ok, rule=rule, action=act,
                              x=x, extra=extra or {}))

    def add(tab, block, c, n, rule, extra=None):
        A["adds"].append(dict(tab=tab, block=block, root=c, nsyn_now=n, rule=rule, extra=extra or {}))

    for tab, g in (("PRg1hop1", 1), ("PRg2hop1", 2)):
        by_block = collections.defaultdict(list)
        for x in tabs.get(tab, []):
            by_block[x["v"].get("upstream_PR", "")].append(x)
        for u_orig, rows in by_block.items():
            U = now_of(u_orig)
            top = ranked(out_n.get(U, {}), tracked_cells, listed={cell(x) for x in rows})[:TOP]
            in_top = {c for c, _ in top}
            rule = f"top {TOP} partners of the PR"
            for x in rows:
                judge(x, f"PR {u_orig}", out_n.get(U, {}).get(cell(x), 0), cell(x) in in_top, rule, dict(U=U))
            have = {cell(x) for x in rows}
            for c, n in top:
                if c not in have:
                    add(tab, f"PR {u_orig}", c, n, rule, dict(U=U, u_orig=u_orig, ptype=rows[0]["v"].get("PRtype", rows[0]["v"].get("class", ""))))
    by_block = collections.defaultdict(list)
    for x in tabs.get("PRg1hop2", []):
        by_block[x["v"].get("hop1_root_id", "")].append(x)
    gone = {h for h, U in hop1_U.items() if U not in sel}
    for h_orig, rows in by_block.items():
        U = hop1_U.get(h_orig, h_orig)
        top = ranked(out_n.get(U, {}), tracked_cells, HOP2_MIN, listed={cell(x) for x in rows})[:TOP]
        in_top = {c for c, _ in top}
        rule = f"top {TOP} partners of the hop-1 cell, nsyn >= {HOP2_MIN}"
        for x in rows:
            ok = cell(x) in in_top and h_orig not in gone
            judge(x, f"hop-1 {h_orig}", out_n.get(U, {}).get(cell(x), 0), ok,
                  rule if h_orig not in gone else "its hop-1 cell no longer qualifies (rule B)", dict(U=U))
        if h_orig in gone:
            A["blocks"].append(dict(tab="PRg1hop2", kind="gone", hop1=h_orig, U=U))
            continue
        have = {cell(x) for x in rows}
        for c, n in top:
            if c not in have:
                add("PRg1hop2", f"hop-1 {h_orig}", c, n, rule, dict(U=U, h_orig=h_orig, h1_rows=rows))
    for c in new_h1:
        top = ranked(out_n.get(c, {}), tracked_cells, HOP2_MIN)[:TOP]
        A["blocks"].append(dict(tab="PRg1hop2", kind="new", hop1=c, U=c, prs=sel[c], top=top))

    Y = {c for x in X for c, n in out_n.get(x, {}).items() if n >= THR and n_onto(c, targets) >= THR}
    Xp = {x for x in X if any(out_n.get(x, {}).get(y, 0) >= THR for y in Y)}
    listed = {cell(x) for x in t5}
    for x in t5:
        c, h = cell(x), hop(x)
        lowp = x["title"].startswith("LOW PRIORITY")
        if h == "PR+1":
            n = max([out_n.get(U, {}).get(c, 0) for U in own] or [0])
            judge(x, "PR+1", n, n >= THR and c in Xp, ">= 3 from a PR group 1 PR, on to a target")
        elif h == "PR+2":
            n = max([out_n.get(p, {}).get(c, 0) for p in Xp | xrows] or [0])
            judge(x, "PR+2", n, n >= THR and n_onto(c, targets) >= THR, ">= 3 from a PR+1 cell, >= 3 onto a target")
        elif h == "T4T5-1":
            n = n_onto(c, targets)
            judge(x, "T4T5-1", n, n >= THR or x["on_path"], ">= 3 onto a target, or on a photoreceptor path")
        elif h == "T4T5-2":
            n = n_onto(c, seeds)
            judge(x, "LOW PRIORITY T4T5-2" if lowp else "T4T5-2", n, n >= THR or x["on_path"],
                  ">= 3 onto a T4T5-1 seed, or on a photoreceptor path")
        elif h == "notT4T5-1":
            judge(x, "notT4T5-1", n_onto(c, not_t), True, "recount only")
    for t in sorted(targets):
        if not any(in_n[t].get(y, 0) >= THR for y in Y):
            A.setdefault("reports", []).append(f"PR+3 target {t}: no PR group 1 path at >= 3 synapses per connection now")
    skip5 = listed | targets | set().union(*pr_now.values())
    for c in sorted(Xp - skip5):
        add("PRg1toT4T5", "PR+1", c, max(out_n.get(U, {}).get(c, 0) for U in own), "on a PR group 1 path")
    for c in sorted(Y - skip5):
        add("PRg1toT4T5", "PR+2", c, max([out_n.get(p, {}).get(c, 0) for p in Xp] or [0]), "on a PR group 1 path")
    best1 = collections.Counter()
    for t in targets:
        for c, n in in_n[t].items():
            best1[c] = max(best1[c], n)
    for c, n in sorted(best1.items(), key=lambda kv: (-kv[1], int(kv[0]))):
        if n >= THR and c not in skip5 | Y:
            add("PRg1toT4T5", "T4T5-1", c, n, ">= 3 onto a target")
    best2 = collections.Counter()
    for t in seeds:
        for c, n in in_n[t].items():
            best2[c] = max(best2[c], n)
    for c, n in sorted(best2.items(), key=lambda kv: (-kv[1], int(kv[0]))):
        if n >= THR and c not in skip5 | Y and best1.get(c, 0) < THR:
            add("PRg1toT4T5", "T4T5-2", c, n, ">= 3 onto a T4T5-1 seed")
    A["day"] = day
    return A


def report(A, applied=False):
    rows = pd.DataFrame([{k: v for k, v in r.items() if k not in ("x", "extra")} for r in A["rows"]])
    adds = pd.DataFrame([{k: v for k, v in r.items() if k != "extra"} for r in A["adds"]])
    if len(adds):
        adds = adds.assign(row="", tracked=False, nsyn_old="", passes=True, action="add (NEW, by nsyn)")
    R = pd.concat([rows, adds], ignore_index=True)
    path = fig(f"{utc_stamp()}_membership_{'changes' if applied else 'dryrun'}.tsv")
    R.to_csv(path, sep="\t", index=False)
    print(f"\n{'APPLIED' if applied else 'DRY RUN (nothing written to the sheet)'} on {A['book'].rsplit('/', 1)[-1]}")
    for tab, g in R.groupby("tab", sort=False):
        r = g[g.row != ""]
        ch = (r.nsyn_old.astype(int) != r.nsyn_now.astype(int)).sum() if len(r) else 0
        print(f"\n{tab}: {len(r)} rows; nsyn changed {int(ch)}; below the criteria {int((r.passes == False).sum())} -> "  # noqa: E712
              f"removed (untracked) {int((r.action == 'remove (untracked)').sum())}, moved down + noted "
              f"{int((r.action == 'move down + note').sum())}; added {int((g.action == 'add (NEW, by nsyn)').sum())}")
        for b, gb in g.groupby("block", sort=False):
            rb = gb[gb.row != ""]
            k = dict(remove=int((rb.action == "remove (untracked)").sum()), move=int((rb.action == "move down + note").sum()),
                     add=int((gb.action == "add (NEW, by nsyn)").sum()))
            if any(k.values()):
                print(f"   {str(b)[:60]:60s} rows {len(rb):4d}  removed {k['remove']:3d}  moved {k['move']:3d}  added {k['add']:3d}")
    for b in A["blocks"]:
        if b["kind"] == "new":
            print(f"PRg1hop2: new block for hop-1 cell {b['hop1']} ({', '.join(f'PR {p} {n} syn' for p, n in b['prs'])}), "
                  f"{len(b['top'])} rows")
        else:
            print(f"PRg1hop2: block of hop-1 cell {b['hop1']} kept, its cell no longer qualifies")
    for line in A.get("reports", []):
        print(line)
    print(f"written {path}")
    return path


# --------------------------------------------------------------------------------------------------------- apply
def _tib():
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "partners", "target_inputs_block.py")
    spec = importlib.util.spec_from_file_location("target_inputs_block", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def note_with_below(old, day, txt, below):
    """the notes with the 'now below block criteria' part added / updated (date kept) / removed"""
    parts = [p.strip() for p in re.split(r";\s*", old or "") if p.strip()]
    prev = next((BELOW.match(p) for p in parts if BELOW.match(p)), None)
    parts = [p for p in parts if not BELOW.match(p)]
    if below:
        parts.append(f"{prev.group(1) if prev else day}: now below block criteria ({txt})")
    return "; ".join(parts)


def lists_t4t5(c, S, in_n, top_below=3):
    """(ids strongest first, counts, total): the cells of S that c feeds with >= THR; when none, its strongest
    contacts (>= 1, at most top_below) so a row below the criteria still shows what it has"""
    n = sorted(((t, in_n[t].get(c, 0)) for t in S if in_n[t].get(c, 0)), key=lambda kv: (-kv[1], kv[0]))
    keep = [kv for kv in n if kv[1] >= THR] or n[:top_below]
    return [k for k, _ in keep], [v for _, v in keep], sum(v for _, v in n)


def apply(A, out_path):
    TIB = _tib()
    wb = openpyxl.load_workbook(A["book"])
    c = cave_client()
    out_n, in_n, day = A["out_n"], A["in_n"], A["day"]
    when_txt = f"{A['when']:%Y-%m-%d %H:%M} UTC"
    rows_by_tab = collections.defaultdict(list)
    for r in A["rows"]:
        rows_by_tab[r["tab"]].append(r)
    adds_by_tab = collections.defaultdict(list)
    for a in A["adds"]:
        adds_by_tab[a["tab"]].append(a)
    new_blocks = [b for b in A["blocks"] if b["kind"] == "new"]
    gone = {b["hop1"] for b in A["blocks"] if b["kind"] == "gone"}

    # new cells: tracking from the other tabs, skeleton geometry, a point on each
    need = sorted({a["root"] for a in A["adds"]} | {cc for b in new_blocks for cc, _ in b["top"]})
    rec, geo, large, xyz = {}, {}, set(), {}
    if need:
        live = TIB.live_rows(wb, c)
        for tab in adds_by_tab.keys() | ({"PRg1hop2"} if new_blocks else set()):
            cells_t = sorted({a["root"] for a in adds_by_tab.get(tab, [])} |
                             ({cc for b in new_blocks for cc, _ in b["top"]} if tab == "PRg1hop2" else set()))
            rec.update({(tab, k): v for k, v in TIB.tracking(cells_t, live, tab).items()})
        sk, geo, large = TIB.geometry(need, LARGE_CABLE, LARGE_OUTSIDE, time.time())
        anyrec = {}
        for (tab, k), v in rec.items():
            anyrec.setdefault(k, v)
        xyz = TIB.find_xyz(need, anyrec, sk, c)
    changed_tabs = []
    log = []
    for tab in ("PRg1hop1", "PRg2hop1", "PRg1hop2", "PRg1toT4T5"):
        if tab not in wb.sheetnames:
            continue
        ws = wb[tab]
        width, last = ws.max_column, ws.max_row
        snap = {r: [(ws.cell(r, j).value, copy.copy(ws.cell(r, j)._style),
                     copy.copy(ws.cell(r, j).hyperlink) if ws.cell(r, j).hyperlink else None) for j in range(1, width + 1)]
                for r in range(1, last + 1)}
        heights = {r: ws.row_dimensions[r].height for r in range(1, last + 1) if ws.row_dimensions[r].height}
        dec = {r["row"]: r for r in rows_by_tab.get(tab, [])}
        upd = collections.defaultdict(dict)  # row -> {column name: value}

        def setv(r, name, val):
            if name in dec[r]["x"]["cols"]:
                upd[r][name] = val

        # value updates of existing rows
        totals = {}
        for r, d in dec.items():
            x, U = d["x"], d["extra"].get("U")
            cc = d["root"]
            if tab in ("PRg1hop1", "PRg2hop1", "PRg1hop2") and U:
                tot = totals.setdefault(U, sum(out_n.get(U, {}).values()))
                n = out_n.get(U, {}).get(cc, 0)
                setv(r, "nsyn", n)
                setv(r, "syn_pct", f"{round(100 * n / tot, 1):g}" if tot else "")
                if tab == "PRg1hop2":
                    hs = set(A["hop1_U"].values()) | {b["hop1"] for b in new_blocks}
                    setv(r, "h1_conv", sum(1 for h in hs if out_n.get(h, {}).get(cc, 0) > 0))
                else:
                    g = 1 if tab == "PRg1hop1" else 2
                    setv(r, "PR_conv", sum(1 for p in A["pr_now"].get(g, ()) if out_n.get(p, {}).get(cc, 0) > 0))
            elif tab == "PRg1toT4T5":
                h = x["v"].get("hop", "")
                if h == "T4T5-1":
                    ids, ns, tot = lists_t4t5(cc, A["targets"], in_n)
                    setv(r, "T4T5_root_id", ", ".join(ids))
                    setv(r, "nsyn", ", ".join(map(str, ns)))
                    setv(r, "nsyn_out", tot)
                    setv(r, "T4T5_reached", len([v for v in ns if v >= THR]))
                elif h == "T4T5-2":
                    ids, ns, tot = lists_t4t5(cc, A["seeds"], in_n)
                    reach = sorted({t for s_ in ids for t in A["targets"] if in_n[t].get(s_, 0) >= THR})
                    setv(r, "T4T5-1_root_id", ", ".join(ids))
                    setv(r, "nsyn", ", ".join(map(str, ns)))
                    setv(r, "nsyn_out", tot)
                    setv(r, "T4T5_root_id", ", ".join(reach))
                    setv(r, "T4T5_reached", len(reach))
                elif h in ("PR+1", "PR+2", "notT4T5-1"):
                    col = {"PR+1": "PR_root_id", "PR+2": "hop1_root_id", "notT4T5-1": None}[h]
                    if col:
                        ids = ID.findall(x["v"].get(col, ""))
                        ns = [out_n.get(A["now_of"](i), {}).get(cc, 0) for i in ids]
                    else:
                        ns = [d["nsyn_now"]]
                    if ns:
                        setv(r, "nsyn", ", ".join(map(str, ns)) if len(ns) > 1 else ns[0])
            if "notes" in x["cols"]:
                txt = (f"nsyn={d['nsyn_now']}" if "no longer qualifies" not in d["rule"]
                       else f"its hop-1 cell no longer qualifies, nsyn={d['nsyn_now']}")
                new = note_with_below(x["v"].get("notes", ""), day, txt, d["action"] == "move down + note")
                if new != x["v"].get("notes", ""):
                    setv(r, "notes", new)
            for k2, v2 in upd[r].items():
                if s(v2) != s(x["v"].get(k2, "")):
                    log.append(dict(tab=tab, row=r, root=cc, column=k2, old=x["v"].get(k2, ""), new=v2,
                                    why="membership: recount / criteria"))

        # runs of data rows (one block's rows); their new order
        runs, cur = [], []
        for r in range(1, last + 1):
            if r in dec:
                cur.append(r)
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        names_of = {}
        for rr in runs:
            x0 = dec[rr[0]]["x"]
            names_of[rr[0]] = [n for n, _ in sorted(x0["cols"].items(), key=lambda kv: kv[1])]
        plan, n_removed, n_added = [], 0, 0
        run_at = {rr[0]: rr for rr in runs}
        writer_cache = {}

        def writer_for(rr):
            if rr[0] not in writer_cache:
                x0 = dec[rr[0]]["x"]
                hdr = x0["hdr"]
                names = [s(ws.cell(hdr, j).value) for j in range(1, width + 1)]
                k = max(i for i, nn in enumerate(names) if nn) + 1
                names = names[:k]
                plain = next((q for q in rr if not any(TIB.solid(ws.cell(q, j)) for j in range(2, k + 1))), rr[0])
                title_r = max(1, hdr - 1)
                writer_cache[rr[0]] = (TIB.Writer(wb, ws, title_r, hdr, names, plain, x0["cols"]), names)
            return writer_cache[rr[0]]

        def new_row(tab_, a, rr):
            k, ex = a["root"], a["extra"]
            R_ = rec.get((tab_, k)) or {}
            bv = R_["best"]["values"] if R_.get("best") else {}
            d = {"root_id_orig": k, "root_id_curr": k, "xyz": xyz.get(k, ""), "annotation": R_.get("annotation", ""),
                 "status": R_.get("status", "NEW"), "pr_date": bv.get("pr_date"), "proofreader": R_.get("proofreader", ""),
                 "chk_date": bv.get("chk_date"), "checker": R_.get("checker", ""), "notes": "",
                 "SeaTable": R_.get("seatable", ""), "link": R_.get("link", ""), "tracked_in": R_.get("tracked_in", ""),
                 "_src": R_.get("best") or (R_["rows"][0] if R_.get("rows") else None)}
            if tab_ in ("PRg1hop1", "PRg2hop1"):
                U = ex["U"]
                tot = sum(out_n.get(U, {}).values())
                g = 1 if tab_ == "PRg1hop1" else 2
                d.update(upstream_PR=ex["u_orig"], PRtype=ex.get("ptype", ""), **{"class": ex.get("ptype", "")},
                         nsyn=a["nsyn_now"], syn_pct=f"{round(100 * a['nsyn_now'] / tot, 1):g}" if tot else "",
                         PR_conv=sum(1 for p in A["pr_now"].get(g, ()) if out_n.get(p, {}).get(k, 0) > 0))
            elif tab_ == "PRg1hop2":
                U = ex["U"]
                tot = sum(out_n.get(U, {}).values())
                v0 = ex["h1_rows"][0]["v"] if ex.get("h1_rows") else {}
                hs = set(A["hop1_U"].values()) | {b["hop1"] for b in new_blocks}
                d.update(PR_root_id=ex.get("prs_txt", v0.get("PR_root_id", "")), PRtype=ex.get("ptype", v0.get("PRtype", "")),
                         hop1_root_id=ex.get("h_orig", ""), h1_nsyn=ex.get("h1_nsyn", v0.get("h1_nsyn", "")),
                         nsyn=a["nsyn_now"], syn_pct=f"{round(100 * a['nsyn_now'] / tot, 1):g}" if tot else "",
                         h1_conv=sum(1 for h in hs if out_n.get(h, {}).get(k, 0) > 0))
            else:
                gm = geo.get(k, {})
                d.update(hop=a["block"].replace("LOW PRIORITY ", ""),
                         cable_um="" if not gm or gm.get("cable") != gm.get("cable") else int(round(gm["cable"])),
                         neuropils=gm.get("summ", ""))
                if a["block"].endswith("T4T5-1"):
                    ids, ns, tot = lists_t4t5(k, A["targets"], in_n)
                    d.update(T4T5_root_id=", ".join(ids), nsyn=", ".join(map(str, ns)), nsyn_out=tot, T4T5_reached=len(ids))
                elif a["block"].endswith("T4T5-2"):
                    ids, ns, tot = lists_t4t5(k, A["seeds"], in_n)
                    reach = sorted({t for s_ in ids for t in A["targets"] if in_n[t].get(s_, 0) >= THR})
                    d.update(**{"T4T5-1_root_id": ", ".join(ids)}, nsyn=", ".join(map(str, ns)), nsyn_out=tot,
                             T4T5_root_id=", ".join(reach), T4T5_reached=len(reach))
                elif a["block"] == "PR+1":
                    pp = sorted(((A["pr_orig_of"].get(U, U), out_n.get(U, {}).get(k, 0)) for U in A["own"]
                                 if out_n.get(U, {}).get(k, 0) >= THR), key=lambda kv: -kv[1])
                    d.update(PR_root_id=", ".join(p for p, _ in pp), nsyn=", ".join(str(n) for _, n in pp))
            return ("new", d, writer_for(rr))

        # which run gets which added cells
        label_run = {}
        for rr in runs:
            label_run.setdefault(dec[rr[0]]["block"], rr)
        adds_run = collections.defaultdict(list)
        for a in adds_by_tab.get(tab, []):
            b = a["block"]
            if tab == "PRg1toT4T5" and b == "T4T5-2" and a["root"] in large:
                b = "LOW PRIORITY T4T5-2" if "LOW PRIORITY T4T5-2" in label_run else b
                a["block"] = b
            if tab == "PRg1toT4T5" and b == "PR+2" and a["root"] in large:
                A.setdefault("reports", []).append(f"PR+2 candidate {a['root']} not added: very large, not Tm-sized")
                continue
            if b in label_run:
                adds_run[label_run[b][0]].append(a)
        first_note = next((r for r in range(1, last + 1) if s(snap[r][0][0]).startswith("note:")), None)
        r = 1
        while r <= last:
            if tab == "PRg1hop2" and r == first_note and new_blocks:
                hdr_src = dec[runs[-1][0]]["x"]["hdr"]
                title_src = hdr_src - 1
                for b in new_blocks:
                    cls = [A["pr_class"].get(p, "").replace("R1-R6", "R1–R6") for p, _ in b["prs"]]
                    prs_txt = "; ".join(f"PR {p} [{k}] {n} syn" for (p, n), k in zip(b["prs"], cls))
                    tot = sum(out_n.get(b["U"], {}).values())
                    title = (f"hop-1 {b['hop1']} (now {b['hop1']}) <- {prs_txt}: {tot} output synapses across "
                             f"{len(out_n.get(b['U'], {}))} partners (updated {when_txt}; block added {day}: its cell now "
                             f"qualifies)")
                    plan.append(("title", title, title_src))
                    plan.append(("old", hdr_src))
                    fake_rr = runs[-1]
                    for cc, n in b["top"]:
                        a = dict(root=cc, nsyn_now=n, block=f"hop-1 {b['hop1']}",
                                 extra=dict(U=b["U"], h_orig=b["hop1"], prs_txt=", ".join(p for p, _ in b["prs"]),
                                            h1_nsyn=", ".join(str(n2) for _, n2 in b["prs"]),
                                            ptype=", ".join(dict.fromkeys(cls))))
                        plan.append(new_row(tab, a, fake_rr))
                        n_added += 1
                    plan.append(("blank",))
            if r in run_at:
                rr = run_at[r]
                ok = [q for q in rr if dec[q]["passes"]]
                down = [q for q in rr if not dec[q]["passes"] and dec[q]["action"] == "move down + note"]
                n_removed += sum(1 for q in rr if dec[q]["action"] == "remove (untracked)")
                new = [new_row(tab, a, rr) for a in adds_run.get(rr[0], [])]
                n_added += len(new)
                if tab in ("PRg1hop1", "PRg2hop1", "PRg1hop2"):
                    key = lambda q: (-dec[q]["nsyn_now"], not dec[q]["tracked"], int(dec[q]["root"]))  # noqa: E731
                    items = [(-dec[q]["nsyn_now"], 0, int(dec[q]["root"]), ("old", q)) for q in ok]
                    items += [(-it[1]["nsyn"], 1, int(it[1]["root_id_curr"]), it) for it in new]
                    items.sort(key=lambda t: t[:3])
                    plan += [t[3] for t in items] + [("old", q) for q in sorted(down, key=key)]
                else:  # PRg1toT4T5: the T4/T5 sort orders the T4T5 blocks afterwards (rows below the criteria last)
                    plan += [("old", q) for q in ok] + new + [("old", q) for q in down]
                r = rr[-1] + 1
                continue
            plan.append(("old", r))
            r += 1
        # block titles: counts and update times
        title_upd = {}
        for rr in runs:
            t_row = next((q for q in range(rr[0] - 1, 0, -1) if s(snap[q][0][0]) and not s(snap[q][1][0]) and q not in dec
                          and s(snap[q][0][0]) not in ("hop", "upstream_PR", "PR_root_id")), None)
            if t_row is None:
                continue
            t = s(snap[t_row][0][0])
            n_ok = sum(1 for q in rr if dec[q]["passes"]) + len(adds_run.get(rr[0], []))
            n_dn = sum(1 for q in rr if dec[q]["action"] == "move down + note")
            if tab in ("PRg1hop1", "PRg2hop1", "PRg1hop2"):
                U = dec[rr[0]]["extra"].get("U")
                if U:
                    t2 = re.sub(r"\d+ output synapses across \d+ partners \(updated [^)]*\)",
                                f"{sum(out_n.get(U, {}).values())} output synapses across {len(out_n.get(U, {}))} partners "
                                f"(updated {when_txt})", t)
                    h1 = dec[rr[0]]["x"]["v"].get("hop1_root_id", "")
                    if tab == "PRg1hop2" and h1 in gone and "no longer qualifies" not in t2:
                        t2 += f"; {day}: its hop-1 cell no longer qualifies for a block (rule B)"
                    title_upd[t_row] = t2
            elif re.match(r"(T4T5-1|T4T5-2|PR\+1|PR\+2|LOW PRIORITY)", t):
                t2 = re.sub(r"\(\+ \d+ now below the criteria[^)]*\)\s*", "", t)
                t2 = re.sub(r"(: )(\d+)( cells)", lambda m: f"{m.group(1)}{n_ok}{m.group(3)}", t2, count=1)
                if n_dn:
                    t2 = re.sub(r"(: \d+ cells)", rf"\1 (+ {n_dn} now below the criteria, noted, at the bottom)", t2, count=1)
                title_upd[t_row] = t2
        # write the plan
        blank_r = next((q for q in range(1, last + 1) if not any(s(v) for v, _, _ in snap[q])), None)
        for i, it in enumerate(plan, 1):
            ws.row_dimensions[i].height = None
            if it[0] == "old":
                q = it[1]
                for j, (v, st, hl) in enumerate(snap[q], 1):
                    cc = ws.cell(i, j)
                    cc.hyperlink = hl
                    cc.value, cc._style = v, copy.copy(st)
                if q in title_upd:
                    ws.cell(i, 1).value = title_upd[q]
                if q in upd:
                    cols = dec[q]["x"]["cols"]
                    for nme, val in upd[q].items():
                        ws.cell(i, cols[nme] + 1).value = val
                if q in heights:
                    ws.row_dimensions[i].height = heights[q]
            elif it[0] == "new":
                d, (w, names) = it[1], it[2]
                for j in range(1, width + 1):
                    ws.cell(i, j).hyperlink = None
                    ws.cell(i, j).value = None
                w.row(i, d, names)
                log.append(dict(tab=tab, row="", root=d["root_id_curr"], column="(row)", old="", new="added",
                                why="membership: now passes its block's rule"))
            elif it[0] == "title":
                src = snap[it[2]]
                for j, (v, st, hl) in enumerate(src, 1):
                    cc = ws.cell(i, j)
                    cc.hyperlink = None
                    cc.value, cc._style = (it[1] if j == 1 else None), copy.copy(st)
            else:  # blank
                for j in range(1, width + 1):
                    cc = ws.cell(i, j)
                    cc.hyperlink = None
                    cc.value = None
                    if blank_r:
                        cc._style = copy.copy(snap[blank_r][j - 1][1])
        for i in range(len(plan) + 1, last + 1):
            ws.row_dimensions[i].height = None
            for j in range(1, width + 1):
                cc = ws.cell(i, j)
                cc.hyperlink = None
                cc.value = None
                if blank_r:
                    cc._style = copy.copy(snap[blank_r][j - 1][1])
        if upd or n_removed or n_added or title_upd:
            changed_tabs.append(tab)
        print(f"{tab}: {sum(1 for q in upd if upd[q])} rows updated, {n_removed} removed, {n_added} added", flush=True)
    wb.save(out_path)
    pd.DataFrame(log).to_csv(fig(f"{utc_stamp()}_membership_changes_cells.tsv"), sep="\t", index=False)
    print(f"changed tabs: {' '.join(changed_tabs)}")
    print(f"full workbook {out_path}")


def main():
    book = sys.argv[1]
    do = "--apply" in sys.argv
    A = analyze(book)
    if do:
        out = os.path.join(TMP, f"{utc_stamp()}_membership_full.xlsx")
        apply(A, out)
    report(A, applied=do)


if __name__ == "__main__":
    main()
