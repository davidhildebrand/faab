"""Sort tabs and renumber PR groups by their distance from the PR group 1 home column, and draw that column in a
link (David 2026-11-25: "renumber any PR groups on the R side that have not been worked on yet (no status change or
proofreader assignment) based on their proximity to PR group 1"; "sort the Mi1 and T4T5 sheet tabs based on their
proximity to the axis/line between the main target PR g1 through the main target T4T5s"; "give me a ng link showing
the PR g1 through T4T5 axis/line").

The column = the lines of faab.partners.home_column, saved by scripts/partners/t4t5_sort.py on every update
(figures/t4t5/<stamp>_t4t5_home_column_lines.json; the newest one unless --lines): one line per right optic-lobe
neuropil through the PR group 1 -> PRg1toT4T5 T4/T5 candidate path, distances in columns (medulla 5.5 um, the R7
terminal spacing; the other neuropils scaled to the same number of columns).

--tabs T4T5s [Mi1s ...]  per cell: the median distance of its synapses (inputs at their post site, outputs at their
      pre site, at most --cap of each per cell) inside the right LA / ME / LO / LOP. Inside each block (a line
      'putative T4: ...' / 'putative T5: ...', or the whole tab when it has one header row) rows go by whole columns
      (0-1, 1-2, ... of the value as shown), inside each by confidence tier then distance (T4T5s) or in the old order;
      cells with no synapses there last. --export writes each changed tab alone (figures/sheets). The distance
      goes into a far-right column (header DIST, two columns right of the last used one, re-used later); a note row
      explains it; the T4T5s title says the new order. A tab whose cells have no synapses in the right optic lobe is
      left as it is (the Mi1s tab: its cells are in the LEFT optic lobe).
--renumber-prs  PRs tab, right side (above '>>> additional L side'): a group counts as worked on when any member row
      has a status other than 'candidate' (or blank), a proofreader, a pr_date, a checker or a chk_date; those keep
      their numbers. The others take the remaining numbers (1, 2, ... not used by a worked-on group) in order of the
      distance of their lamina cartridge center from the PR group 1 column: center = the median of the members'
      median output-site positions in the right lamina (members with >= 3 sites there; else R7 / R8 sites in the
      medulla, measured from the medulla line); no such sites: last, in their old order. Blocks move with their rows'
      values, styles, links and heights; the section is in number order again; 'group N' / 'groups N, M' in the PRs
      notes and 'PRs group N' / 'PR group N' elsewhere follow the new numbers. Every right group line gets
      'N.N columns from PR group 1 (lamina cartridge center)' in a far-right cell; the right section's title line
      says what was done. Writes figures/pr_groups/<stamp>_pr_group_renumber.tsv (old, new, distance, worked on).
--link  (no BOOK needed) the column as a Spelunker link: one yellow line per neuropil where the line runs inside it,
      joined lamina -> medulla -> lobula -> lobula plate, PR group 1 and the T4/T5 candidates (ids saved with the
      lines), the optic-lobe meshes; built on the README scene (faab.nglink). Writes tmp/<stamp>_home_column_ng.txt.

Read-only toward CAVE (is_latest_roots, throttled get_latest_roots; synapse snapshot caught up to now). Writes
tmp/<stamp>_home_column_sort_full.xlsx (printed as 'full workbook ...' for update_book.py, with 'changed tabs: ...')
and figures/sheets/<stamp>_home_column_sort_changes.tsv.
Usage: home_column_sort.py BOOK.xlsx [--tabs T4T5s Mi1s] [--renumber-prs] [--lines JSON] [--cap 400] [--export]
       home_column_sort.py --link [--lines JSON]"""
import argparse
import copy
import json
import os
import re
import time

import numpy as np
import openpyxl
import pandas as pd

from faab import nglink, ol_meshes, synsnap
from faab.clients.cave import client as cave_client, throttle
from faab.config import TMP, fig
from faab.partners import home_column as HC
from faab.partners.sheet import cell_text as s, utc_stamp
from faab.pr.prs_tab import restore, snapshot

IDS = re.compile(r"\d{15,}")
DIST = "distance from PR group 1 column (columns apart)"
NOTE_PREFIX = "note: order and distance from PR group 1 column"
NOTE = (NOTE_PREFIX + " (far-right column, {date}) = the median distance of the cell's synapses (inputs and outputs) "
        "in the right lamina, medulla, lobula and lobula plate from the line through the PR group 1 path (the same line "
        "as on the PRg1toT4T5 tab: in the medulla from the PR+1 -> PR+2 synapses to the T4/T5 candidates' dendrites, "
        "in the lobula and lobula plate through the candidates' synapses there, across the layers), in columns "
        "(medulla 5.5 um, the spacing of neighboring R7 terminals; the other neuropils scaled to the same number of "
        "columns). Rows go by whole columns (0-1, 1-2, ...), {inside} inside each; blank = no synapses in the right "
        "optic lobe (at the end).")
ORDER_OLD = "by confidence of the T4/T5 match"
ORDER_NEW = ("by distance from the PR group 1 column (whole columns, closest first; inside each by confidence, then "
             f"distance; column '{DIST}' far right)")
PR_DIST = re.compile(r"^\d+(?:\.\d+)? columns from PR group 1 \((lamina cartridge center|medulla R7/R8 sites)\)$|"
                     r"^no lamina or medulla sites")
PR_SECTION = ("groups not worked on yet (status 'candidate', no proofreader or dates) renumbered {date} by the distance "
              "of their lamina cartridge center from the PR group 1 column (far-right value on each group line); "
              "worked-on groups kept their numbers")
GROUP_REF = re.compile(r"\b(groups?)(\s+)(\d+(?:\s*(?:,|and|&|/)\s*\d+)*)", re.I)
OTHER_REF = re.compile(r"\b(PRs?\s+groups?)(\s+)(\d+(?:\s*(?:,|and|&|/)\s*\d+)*)", re.I)
AXIS_COLOR = "#F0E442"  # Okabe-Ito yellow
CONF = {"annotated": 0, "high": 1, "medium": 2, "low": 3}  # T4T5s confidence tiers
LOG = []


def note(tab, row, column, old, new, why):
    LOG.append(dict(tab=tab, row=row, column=column, old=s(old), new=s(new), why=why))


def live_roots(c, ids):
    """{id: [current pieces]} (is_latest_roots in batches, get_latest_roots for stale ids; <= 1 call per 2 s)"""
    ids = sorted({str(x) for x in ids if IDS.fullmatch(str(x))})
    cur, t, bad = {}, 0.0, []
    for i in range(0, len(ids), 5000):
        part = ids[i:i + 5000]
        t = throttle(t)
        ok = np.asarray(c.chunkedgraph.is_latest_roots([int(x) for x in part])).astype(bool)
        for x, good in zip(part, ok):
            if good:
                cur[x] = [int(x)]
    stale = [x for x in ids if x not in cur]
    for x in stale:
        t = throttle(t)
        try:
            cur[x] = [int(y) for y in np.atleast_1d(c.chunkedgraph.get_latest_roots(int(x)))]
        except ValueError:  # no creation time: not a root id of this datastack
            cur[x] = []
            bad.append(x)
    print(f"  {len(ids)} ids, {len(stale)} stale -> current pieces" + (f"; not root ids: {', '.join(bad)}" if bad else ""),
          flush=True)
    return cur


def synapse_sites(snap, c, pieces, side, cap=None, labels=HC.LABELS):
    """one row per synapse of the pieces: cell (the piece), x / y / z um at the `side` site, np; side 'pre' = the
    pieces' outputs, 'post' = their inputs; at most cap per piece (seeded sample)"""
    t0 = time.time()
    kw = {"pre": pieces} if side == "pre" else {"post": pieces}
    df = snap.synapses(**kw, when="now", positions=True, client=c)
    df = df[df.pre_root != df.post_root]
    df = df.assign(cell=df[f"{side}_root"].astype(np.int64))
    if cap:
        df = df.sample(frac=1.0, random_state=0).groupby("cell", sort=False).head(cap)
    P = df[[f"{side}_x", f"{side}_y", f"{side}_z"]].to_numpy(float) * HC.VOX
    out = pd.DataFrame({"cell": df.cell.to_numpy(), "x": P[:, 0], "y": P[:, 1], "z": P[:, 2]})
    out["np"] = ol_meshes.assign(P, labels) if len(out) else []
    print(f"  {len(out)} {'output' if side == 'pre' else 'input'} sites ({time.time() - t0:.0f} s)", flush=True)
    return out


def last_used_col(ws):
    return max((j for row in ws.iter_rows() for j, x in enumerate(row, 1) if s(x.value)), default=1)


# ---------------------------------------------------------------- tabs sorted by distance
def tab_blocks(ws):
    """[(title row or None, header row, [data rows])]: a block = an optional title line, a header row starting with
    root_id_orig, then rows with an id in column A or B, up to a blank row"""
    out, r, n = [], 1, ws.max_row
    while r <= n:
        if s(ws.cell(r, 1).value) == "root_id_orig":
            title = r - 1 if r > 1 and s(ws.cell(r - 1, 1).value) and s(ws.cell(r - 1, 1).value) != "root_id_orig" else None
            rows, q = [], r + 1
            while q <= n and (IDS.fullmatch(s(ws.cell(q, 1).value)) or IDS.fullmatch(s(ws.cell(q, 2).value))):
                rows.append(q)
                q += 1
            out.append((title, r, rows))
            r = q
        else:
            r += 1
    return out


def sort_tab(wb, tab, lines, col_um, snap, c, cap, date):
    ws = wb[tab]
    blocks = tab_blocks(ws)
    if not blocks:
        print(f"{tab}: no header row; left as it is", flush=True)
        return None
    hdr = {b[1]: {s(ws.cell(b[1], j).value): j for j in range(1, ws.max_column + 1) if s(ws.cell(b[1], j).value)}
           for b in blocks}
    root_of = {}
    for _, h, rows in blocks:
        cols = hdr[h]
        for r in rows:
            cu = s(ws.cell(r, cols["root_id_curr"]).value) if "root_id_curr" in cols else ""
            root_of[r] = cu if IDS.fullmatch(cu) else s(ws.cell(r, cols.get("root_id_orig", 1)).value)
    print(f"{tab}: {sum(len(b[2]) for b in blocks)} rows in {len(blocks)} blocks; live roots:", flush=True)
    cur = live_roots(c, root_of.values())
    pieces = sorted({p for v in cur.values() for p in v})
    S = pd.concat([synapse_sites(snap, c, pieces, "pre", cap), synapse_sites(snap, c, pieces, "post", cap)])
    S = S[S.np.isin(HC.LABELS)]
    S = S.assign(d=HC.dist_cols(S, lines, col_um))
    med = S.groupby("cell").d.median().to_dict()
    n_ol = S.groupby("cell").size().to_dict()
    dist, rows_t = {}, []
    for r, root in root_of.items():
        v = [med[p] for p in cur.get(root, []) if p in med]
        k = int(sum(n_ol.get(p, 0) for p in cur.get(root, [])))
        dist[r] = float(np.median(v)) if v else np.nan
        rows_t.append(dict(tab=tab, row=r, root_id=root, n_optic_lobe_sites=k,
                           distance_columns=round(dist[r], 2) if v else ""))
    have = [d for d in dist.values() if not np.isnan(d)]
    print(f"{tab}: {len(have)} of {len(dist)} cells with synapses in the right optic lobe; distance median "
          f"{np.median(have) if have else float('nan'):.1f} columns", flush=True)
    if not have:
        print(f"{tab}: no cell has synapses in the right optic lobe (left optic lobe?): left as it is", flush=True)
        return pd.DataFrame(rows_t)

    # ---- distance column, far right (re-used when present)
    dc = next((j for h in hdr for j in range(1, ws.max_column + 1) if s(ws.cell(h, j).value) == DIST), None)
    if dc is None:
        dc = last_used_col(ws) + 2
    ws.column_dimensions[openpyxl.utils.get_column_letter(dc)].width = 18
    for _, h, _ in blocks:
        if s(ws.cell(h, dc).value) != DIST:
            ref = max(j for j in hdr[h].values() if j < dc)
            ws.cell(h, dc).value = DIST
            ws.cell(h, dc)._style = copy.copy(ws.cell(h, ref)._style)
            note(tab, h, dc, "", DIST, "distance column header")

    # ---- order inside each block: whole columns, old order inside; no sites last
    moved = 0
    for _, h, rows in blocks:
        conf = hdr[h].get("confidence")  # T4T5s: inside a whole column, by confidence tier, then by distance
        tier = lambda r: CONF.get(s(ws.cell(r, conf).value).lower(), len(CONF)) if conf else 0  # noqa: E731
        key = {r: (int(np.floor(round(dist[r], 1))) if not np.isnan(dist[r]) else 10 ** 6, tier(r),
                   dist[r] if conf and not np.isnan(dist[r]) else 0.0, i) for i, r in enumerate(rows)}
        new = sorted(rows, key=lambda r: key[r])
        for r in rows:
            v = round(dist[r], 1) if not np.isnan(dist[r]) else None
            cell = ws.cell(r, dc)
            if s(cell.value) != s(v):
                cell.value = v
            ref = hdr[h].get("confidence") or hdr[h].get("notes") or 1
            cell._style = copy.copy(ws.cell(r, ref)._style)
        if new != rows:
            snap_rows = {r: snapshot(ws, r) for r in rows}
            for dst, src in zip(rows, new):
                restore(ws, dst, snap_rows[src])
            moved += sum(a != b for a, b in zip(rows, new))
    note(tab, blocks[0][1], dc, "", f"{moved} rows moved", "rows ordered by distance from the PR group 1 column")

    # ---- title and note
    for r in range(1, blocks[0][1]):
        t = s(ws.cell(r, 1).value)
        if t.startswith(">>>"):
            u = re.sub(r"by distance from the PR group 1 column \(.*? far right\)", lambda m: ORDER_NEW,
                       t.replace(ORDER_OLD, ORDER_NEW))
            if u != t:
                ws.cell(r, 1).value = u
                note(tab, r, 1, t, u, "title: new order")
    inside = ("by confidence (annotated, high, medium, low), then by distance" if "confidence" in hdr[blocks[0][1]]
              else "the old order")
    txt = NOTE.format(date=date, inside=inside)
    end = max(b[2][-1] if b[2] else b[1] for b in blocks)
    note_rows = [r for r in range(end + 1, ws.max_row + 1) if s(ws.cell(r, 1).value).startswith("note:")]
    hit = next((r for r in note_rows if s(ws.cell(r, 1).value).startswith(NOTE_PREFIX)), None)
    if hit is None:
        last = note_rows[-1] if note_rows else end + 1
        hit = last + 1
        while any(s(ws.cell(hit, j).value) for j in range(1, ws.max_column + 1)):
            hit += 1
        if note_rows:
            ws.cell(hit, 1)._style = copy.copy(ws.cell(note_rows[-1], 1)._style)
    if s(ws.cell(hit, 1).value) != txt:
        note(tab, hit, 1, ws.cell(hit, 1).value, txt, "note: distance column")
        ws.cell(hit, 1).value = txt
    print(f"{tab}: {moved} rows moved; distance column {openpyxl.utils.get_column_letter(dc)}; note row {hit}",
          flush=True)
    return pd.DataFrame(rows_t)


# ---------------------------------------------------------------- PR groups renumbered
def pr_groups(ws):
    """right-side groups: {number: dict(title, end, header cols, member rows)} and the right section's title row"""
    L_row = next((r for r in range(1, ws.max_row + 1) if s(ws.cell(r, 1).value).startswith(">>>")
                  and " L side" in s(ws.cell(r, 1).value)), ws.max_row + 1)
    sec = next((r for r in range(1, L_row) if s(ws.cell(r, 1).value).startswith(">>> additional R side")), None)
    titles = [(r, int(m.group(1))) for r in range(1, L_row)
              if (m := re.fullmatch(r"group (\d+)", s(ws.cell(r, 1).value)))]
    G = {}
    for k, (r, g) in enumerate(titles):
        end = (titles[k + 1][0] - 1) if k + 1 < len(titles) else L_row - 1
        if sec and r < sec < end:  # the last group of the first section ends before the section line
            end = sec - 1
        h = r + 1
        cols = {s(ws.cell(h, j).value): j for j in range(1, ws.max_column + 1) if s(ws.cell(h, j).value)}
        rows = [q for q in range(h + 1, end + 1) if IDS.fullmatch(s(ws.cell(q, 1).value))]
        if g in G:
            raise SystemExit(f"PRs: group {g} twice on the right side; nothing written")
        G[g] = dict(title=r, end=end, cols=cols, rows=rows)
    return G, sec, L_row


def worked_on(ws, g):
    get = lambda r, k: s(ws.cell(r, g["cols"][k]).value) if k in g["cols"] else ""  # noqa: E731
    return any(get(r, "status").lower() not in ("", "candidate") or any(get(r, k) for k in
               ("proofreader", "pr_date", "checker", "chk_date")) for r in g["rows"])


def renumber_text(t, mapping, rx):
    def sub(m):
        nums = re.sub(r"\d+", lambda q: str(mapping.get(int(q.group()), int(q.group()))), m.group(3))
        return m.group(1) + m.group(2) + nums
    return rx.sub(sub, t)


def renumber_prs(wb, lines, col_um, snap, c, date, stamp):
    ws = wb["PRs"]
    G, sec, L_row = pr_groups(ws)
    if sec is None:
        raise SystemExit("PRs: no '>>> additional R side' line; nothing written")
    work = {g: worked_on(ws, x) for g, x in G.items()}
    print(f"PRs: {len(G)} right-side groups, {sum(work.values())} worked on "
          f"({sorted(g for g, w in work.items() if w)}); live roots:", flush=True)
    root_of = {}
    for g, x in G.items():
        cols = x["cols"]
        for r in x["rows"]:
            cu = s(ws.cell(r, cols["root_id_curr"]).value) if "root_id_curr" in cols else ""
            root_of[(g, r)] = cu if IDS.fullmatch(cu) else s(ws.cell(r, cols["root_id_orig"]).value)
    cur = live_roots(c, root_of.values())
    pieces = sorted({p for v in cur.values() for p in v})
    S = synapse_sites(snap, c, pieces, "pre", None, labels=["LA_R", "ME_R"])
    cen = {L: S[S.np == L].groupby("cell").agg(n=("x", "size"), x=("x", "median"), y=("y", "median"),
                                                z=("z", "median")) for L in ("LA_R", "ME_R")}
    rows_t = []
    for g, x in G.items():
        rec = dict(old=g, worked_on=work[g], members=len(x["rows"]))
        for L in ("LA_R", "ME_R"):
            c_ = cen[L]
            P = [c_.loc[p, ["x", "y", "z"]].to_numpy(float) for r in x["rows"] for p in cur.get(root_of[(g, r)], [])
                 if p in c_.index and c_.loc[p, "n"] >= 3]
            rec[f"members_{L}"] = len(P)
            if P and "dist" not in rec:
                ctr = np.median(np.array(P), axis=0)
                rec.update(dist=float(HC.distance(ctr[None], [L], lines, col_um)[0]), measured_in=L,
                           center_um=np.round(ctr, 1).tolist())
        rec.setdefault("dist", np.nan)
        rec.setdefault("measured_in", "")
        rows_t.append(rec)
    T = pd.DataFrame(rows_t).set_index("old")
    fixed = {g for g, w in work.items() if w}
    todo = sorted((g for g in G if not work[g]), key=lambda g: (np.isnan(T.loc[g, "dist"]), T.loc[g, "dist"], g))
    free = [n for n in range(1, len(G) + len(fixed) + 1) if n not in fixed][:len(todo)]
    new_of = {g: g for g in fixed}
    new_of.update(dict(zip(todo, free)))
    mapping = {g: n for g, n in new_of.items() if n != g}
    T["new"] = pd.Series(new_of)
    print(f"PRs: {len(todo)} groups renumbered ({len(mapping)} change number); no lamina or medulla sites: "
          f"{sorted(int(g) for g in T.index[T.dist.isna()])}", flush=True)

    # ---- blocks of the right additional section in the new number order (rows keep values, styles, links, heights)
    region = sorted((g for g, x in G.items() if x["title"] > sec), key=lambda g: G[g]["title"])
    first, last = G[region[0]]["title"], G[region[-1]]["end"]
    spans = {g: list(range(G[g]["title"], G[g]["end"] + 1)) for g in region}
    if sum(len(v) for v in spans.values()) != last - first + 1:
        raise SystemExit("PRs: the right section has rows outside the group blocks; nothing written")
    order = sorted(region, key=lambda g: new_of[g])
    snaps = {r: snapshot(ws, r) for r in range(first, last + 1)}
    r = first
    title_now = {}
    for g in order:
        for k, src in enumerate(spans[g]):
            restore(ws, r, snaps[src])
            if k == 0:
                title_now[g] = r
            r += 1
    for g in G:
        title_now.setdefault(g, G[g]["title"])
    for g, n in mapping.items():
        ws.cell(title_now[g], 1).value = f"group {n}"
        note("PRs", title_now[g], 1, f"group {g}", f"group {n}", "renumbered by distance from PR group 1")
    moved = sum(1 for g in region if title_now[g] != G[g]["title"])
    print(f"PRs: {moved} group blocks moved", flush=True)

    # ---- references to group numbers
    n_ref = 0
    for name in wb.sheetnames:
        w = wb[name]
        rx = GROUP_REF if name == "PRs" else OTHER_REF
        for row in w.iter_rows():
            for x in row:
                t = x.value
                if not isinstance(t, str) or re.fullmatch(r"group \d+", t.strip()):
                    continue
                if name == "PRs" and x.row >= L_row:
                    continue
                u = renumber_text(t, mapping, rx)
                if u != t:
                    x.value = u
                    n_ref += 1
                    note(name, x.row, x.column, t, u, "group numbers follow the renumbering")
    print(f"group-number references renumbered: {n_ref}", flush=True)

    # ---- far-right distance on every right group line; the section line says what was done
    dc = next((j for g in G for j in range(1, ws.max_column + 1) if PR_DIST.match(s(ws.cell(title_now[g], j).value))),
              None) or last_used_col(ws) + 2
    for g in G:
        rr = title_now[g]
        d = T.loc[g, "dist"]
        where = "lamina cartridge center" if T.loc[g, "measured_in"] == "LA_R" else "medulla R7/R8 sites"
        txt = f"{d:.1f} columns from PR group 1 ({where})" if not np.isnan(d) else "no lamina or medulla sites yet"
        cell = ws.cell(rr, dc)
        if s(cell.value) != txt:
            note("PRs", rr, dc, cell.value, txt, "distance from PR group 1")
            cell.value = txt
        cell._style = copy.copy(ws.cell(rr, 1)._style)
    ws.column_dimensions[openpyxl.utils.get_column_letter(dc)].width = 40
    sec_txt = PR_SECTION.format(date=date)
    cell = ws.cell(sec, dc)
    if s(cell.value) != sec_txt:
        note("PRs", sec, dc, cell.value, sec_txt, "renumbering note")
        cell.value = sec_txt
        cell._style = copy.copy(ws.cell(sec, 1)._style)

    T = T.reset_index().sort_values("new")
    T["distance_columns"] = T.dist.round(2)
    out = fig(f"{stamp}_pr_group_renumber.tsv")
    T[["old", "new", "worked_on", "members", "distance_columns", "measured_in", "members_LA_R", "members_ME_R",
       "center_um"]].to_csv(out, sep="\t", index=False)
    print(f"renumbering table {out}", flush=True)
    return T


# ---------------------------------------------------------------- the link
def link(lines, meta, stamp):
    order = [L for L in ("LA_R", "ME_R", "LO_R", "LOP_R") if L in lines]
    seg = {L: HC.extent(L, *lines[L]) for L in order}
    vox = lambda p: (np.asarray(p, float) / HC.VOX).round(1).tolist()  # noqa: E731
    parts = [[vox(a), vox(b)] for a, b in seg.values()]
    for A, B in zip(order, order[1:]):  # join each neuropil's line to the next at their nearest ends
        ends = [(np.linalg.norm(p - q), p, q) for p in seg[A] for q in seg[B]]
        _, p, q = min(ends, key=lambda e: e[0])
        parts.append([vox(p), vox(q)])
    for L in order:
        a, b = seg[L]
        print(f"  {L}: {np.linalg.norm(b - a):.0f} um inside, from {np.round(a, 1).tolist()} to {np.round(b, 1).tolist()} um",
              flush=True)
    centre = np.mean([np.mean(v, axis=0) for v in seg.values()], axis=0)
    def count(ids, cells):  # cells on the sheet vs their current pieces
        n = len(ids)
        return f"{cells} cells, {n} pieces now" if cells and cells != n else f"{n} cells"
    neurons = []
    if meta.get("pr_group1"):
        neurons.append({"name": f"PR group 1 ({count(meta['pr_group1'], meta.get('pr_group1_cells'))})",
                        "color": "#CC79A7", "ids": meta["pr_group1"]})
    if meta.get("targets"):
        neurons.append({"name": f"T4/T5 candidates of PRg1toT4T5 ({count(meta['targets'], meta.get('targets_cells'))})",
                        "color": "#009E73", "ids": meta["targets"]})
    spec = {"meshes": [{"kind": "OL", "segments": ["21", "22", "23", "24"]}], "neurons": neurons,
            "annotations": [{"name": "PR group 1 column (line)", "color": AXIS_COLOR, "lines": parts, "width": 4},
                            {"name": "line anchors", "color": AXIS_COLOR, "points": [vox(o) for o, _ in lines.values()],
                             "visible": False}],
            "layout": "3d", "compact": True, "position": vox(centre), "scale": 12000, "view": "anterior"}
    url = nglink.to_url(nglink.build(spec))
    out = os.path.join(TMP, f"{stamp}_home_column_ng.txt")
    open(out, "w").write(url + "\n")
    json.dump(spec, open(os.path.join(TMP, f"{stamp}_home_column_spec.json"), "w"), indent=1)
    print(f"link ({len(url)} characters) {out}\n{url}", flush=True)
    return url


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("book", nargs="?")
    ap.add_argument("--tabs", nargs="*", default=[])
    ap.add_argument("--renumber-prs", action="store_true")
    ap.add_argument("--link", action="store_true")
    ap.add_argument("--lines")
    ap.add_argument("--cap", type=int, default=400, help="synapses per cell and direction (seeded sample)")
    ap.add_argument("--export", action="store_true", help="also write each changed tab alone (as the sync does)")
    a = ap.parse_args()
    stamp = utc_stamp()
    date = stamp[:4] + "-" + stamp[4:6] + "-" + stamp[6:8]
    lines, col_um, meta = HC.load(a.lines)
    print(f"home-column lines {meta['path']} (built from {os.path.basename(meta.get('book', '?'))}): "
          + "; ".join(f"{L} through {np.round(o, 1).tolist()} um" for L, (o, _) in lines.items()), flush=True)
    if a.link:
        link(lines, meta, stamp)
    if not a.book:
        if a.tabs or a.renumber_prs:
            raise SystemExit("--tabs / --renumber-prs need BOOK")
        return
    wb = openpyxl.load_workbook(a.book)
    c = cave_client()
    snap = synsnap.Snapshot(log=lambda *_: None)
    changed = []
    for tab in a.tabs:
        n0 = len(LOG)
        res = sort_tab(wb, tab, lines, col_um, snap, c, a.cap, date)
        if res is not None and len(res):
            res.to_csv(fig(f"{stamp}_t4t5_home_column_{tab}.tsv"), sep="\t", index=False)
        if len(LOG) > n0:
            changed.append(tab)
    if a.renumber_prs:
        n0 = len(LOG)
        renumber_prs(wb, lines, col_um, snap, c, date, stamp)
        changed += sorted({x["tab"] for x in LOG[n0:]} - set(changed))
    full = os.path.join(TMP, f"{stamp}_home_column_sort_full.xlsx")
    wb.save(full)
    pd.DataFrame(LOG).to_csv(fig(f"sheets/{stamp}_home_column_sort_changes.tsv"), sep="\t", index=False)
    if a.export:
        for tab in changed:
            w2 = openpyxl.load_workbook(full)
            for other in list(w2.worksheets):
                if other.title != tab:
                    w2.remove(other)
            out = fig(f"{stamp}_FAABopticlobetargets-{tab}.xlsx")
            w2.save(out)
            print(f"wrote {out}")
    print(f"changes {len(LOG)}")
    print(f"changed tabs: {' '.join(changed)}")
    print(f"full workbook {full}")


if __name__ == "__main__":
    main()
