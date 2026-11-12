"""Prototype 'PRgroupAny -> T4/T5' (David 2026-11-12: "run the same PR to T4/T5 search but keep the T4/T5 targets the
same (identical) but allow the PRs to expand to the whole list (even those not yet fully proofread), then penalize by
distance"). Read-only (snapshot caught up to now; one is_latest_roots call + throttled get_latest_roots).

Targets = the PR+3 cells of PRg1toT4T5 (the 19 T4/T5 candidates). PRs = every right-side photoreceptor we know of:
the PRs tab (all groups, any status) plus SeaTable right-side photoreceptor rows. Paths PR -> PR+1 -> PR+2 -> target
with >= THR synapses per connection (as for PR group 1). Each PR+1 / PR+2 cell is scored like t4t5_sort.py: the
median distance of its path synapses (PR+1: its synapses onto path PR+2 cells; PR+2: onto the targets) from the PR
group 1 home-column lines, in columns; each PR by the distance of its synapses onto path PR+1 cells (lamina).
Prints how many cells per stage and per distance group, how many are new (not in PRg1toT4T5), and which PR groups
the reaching PRs belong to. Writes figures/t4t5/<stamp>_prg_any_paths.tsv (one row per path cell).
Usage: prg_any_paths.py BOOK.xlsx SEATABLE.tsv [--thr 3]"""
import argparse
import importlib.util
import os
import re
import time

import numpy as np
import openpyxl
import pandas as pd

from faab import ol_meshes, synsnap
from faab.clients.cave import client as cave_client, throttle
from faab.config import DATA, fig
from faab.partners.sheet import cell_text as s, utc_stamp

_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "partners", "t4t5_sort.py")
_spec = importlib.util.spec_from_file_location("t4t5_sort", _p)
TS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TS)
IDS = re.compile(r"\d{15,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("book")
    ap.add_argument("seatable")
    ap.add_argument("--thr", type=int, default=3)
    a = ap.parse_args()
    stamp = utc_stamp()
    wb = openpyxl.load_workbook(a.book, read_only=True)
    # PRg1toT4T5 cells by hop
    tab = {}
    hdr = None
    for row in wb["PRg1toT4T5"].iter_rows(values_only=True):
        v = [s(x) for x in row]
        if v and v[0] == "hop":
            hdr = v
            continue
        if hdr and v and v[0] in ("PR+1", "PR+2", "PR+3", "T4T5-1", "T4T5-2"):
            d = dict(zip(hdr, v))
            if IDS.fullmatch(d.get("root_id_curr", "")):
                tab.setdefault(v[0], set()).add(d["root_id_curr"])
    in_tab = set().union(*tab.values())
    # PRs: PRs tab (any group / status) + SeaTable right photoreceptors
    prs_tab = {}
    hdr, grp = None, ""
    for row in wb["PRs"].iter_rows(values_only=True):
        v = [s(x) for x in row]
        if not v or not v[0]:
            continue
        if "root_id_curr" in v and "xyz" in v:
            hdr = v
            continue
        m = re.match(r"(?:group|PR group)\s*(\d+)", v[0], re.I)
        if m and not IDS.fullmatch(v[0]):
            grp = m.group(1)
            continue
        if hdr:
            d = dict(zip(hdr, v))
            r = d.get("root_id_curr") or d.get("root_id_orig", "")
            if IDS.fullmatch(r):
                prs_tab.setdefault(r, grp)
    st = pd.read_csv(a.seatable, sep="\t", dtype=str).fillna("")
    pr_rx = r"\bR[1-8]\b|R1/R6|R2/R5|R3/R4|photoreceptor"
    st_pr = st[(st.side == "R") & st[["class", "subclass", "type"]].agg(" ".join, axis=1).str.contains(pr_rx, regex=True)]
    sea = set(st_pr.root_now[st_pr.root_now.str.fullmatch(r"\d{15,}")])
    targets = sorted(tab["PR+3"])
    print(f"targets {len(targets)}; PRs tab {len(prs_tab)} roots, SeaTable right photoreceptors {len(sea)}, "
          f"together {len(set(prs_tab) | sea)}", flush=True)

    c = cave_client()
    every = sorted(set(prs_tab) | sea | in_tab)
    ok = np.concatenate([np.asarray(c.chunkedgraph.is_latest_roots([int(x) for x in every[i:i + 5000]])).astype(bool)
                         for i in range(0, len(every), 5000)])
    last, cur = time.time(), {}
    for x, good in zip(every, ok):
        if good:
            cur[x] = [int(x)]
        else:
            last = throttle(last)
            cur[x] = [int(y) for y in np.atleast_1d(c.chunkedgraph.get_latest_roots(int(x)))]
    print(f"{len(every)} ids, {int((~ok).sum())} stale", flush=True)
    now = lambda ids: sorted({y for x in ids for y in cur.get(x, [])})
    group_of = {y: prs_tab.get(x, "SeaTable only") for x in set(prs_tab) | sea for y in cur.get(x, [])}
    PR, T = set(now(set(prs_tab) | sea)), set(now(targets))
    g1 = set(now(set(pd.read_csv(os.path.join(DATA, "pr_groups.tsv"), sep="\t", dtype=str).query("group == '1'").root_id_curr)))

    snap = synsnap.Snapshot(log=lambda *_: None)
    print("PR outputs ...", flush=True)
    e1 = snap.edges(pre=sorted(PR), when="now", client=c)
    e1 = e1[(e1.n >= a.thr) & ~e1.post.isin(list(PR))]
    print("target inputs ...", flush=True)
    e3 = snap.edges(post=sorted(T), when="now", client=c)
    e3 = e3[(e3.n >= a.thr) & ~e3.pre.isin(list(T))]
    x1, x2 = set(e1.post), set(e3.pre)
    print("PR+1 -> PR+2 ...", flush=True)
    e2 = snap.edges(pre=sorted(x1), post=sorted(x2), when="now", client=c)
    e2 = e2[(e2.n >= a.thr) & (e2.pre != e2.post)]
    p1, p2 = set(e2.pre), set(e2.post)
    e1, e3 = e1[e1.post.isin(p1)], e3[e3.pre.isin(p2)]
    prs_used = set(e1.pre)
    print(f"paths with >= {a.thr} per connection: PRs {len(prs_used)} (PR group 1: {len(prs_used & g1)}), PR+1 {len(p1)}, "
          f"PR+2 {len(p2)}, targets reached {e3.post.nunique()}", flush=True)

    # distances: synapse positions of the path edges, scored on the PR group 1 home lines
    def pos(pre, post):
        df = snap.synapses(pre=sorted(pre), post=sorted(post), when="now", positions=True, client=c)
        P = df[["post_x", "post_y", "post_z"]].to_numpy(float) * TS.VOX
        return df.assign(x=P[:, 0], y=P[:, 1], z=P[:, 2], np=ol_meshes.assign(P, TS.LABELS))

    s1, s2, s3 = pos(prs_used, p1), pos(p1, p2), pos(p2, set(e3.post))
    s2 = s2.merge(e2[["pre", "post"]], left_on=["pre_root", "post_root"], right_on=["pre", "post"])
    s3 = s3[s3.pre_root.isin(list(p2))]
    # home lines from PR group 1's own path (the same construction as t4t5_sort.py)
    g1_p1 = set(now(tab["PR+1"]))
    g1_p2 = set(now(tab["PR+2"]))
    me_chain = pos(g1_p1, g1_p2)
    tin = snap.synapses(post=sorted(T), when="now", positions=True, client=c)
    P = tin[["post_x", "post_y", "post_z"]].to_numpy(float) * TS.VOX
    tin = tin.assign(x=P[:, 0], y=P[:, 1], z=P[:, 2], np=ol_meshes.assign(P, TS.LABELS))
    tout = snap.synapses(pre=sorted(T), when="now", positions=True, client=c)
    P = tout[["pre_x", "pre_y", "pre_z"]].to_numpy(float) * TS.VOX
    tout = tout.assign(x=P[:, 0], y=P[:, 1], z=P[:, 2], np=ol_meshes.assign(P, TS.LABELS))
    la_chain = pos(g1, g1_p1)
    X = ["x", "y", "z"]
    med = lambda df, L: df[df.np == L][X].median().to_numpy() if int((df.np == L).sum()) >= 10 else None
    fc = {L: TS.faces(L) for L in TS.LABELS}
    lines = {}
    D, Pm = med(me_chain, "ME_R"), med(tin, "ME_R")
    lines["ME_R"] = (D, (Pm - D) / np.linalg.norm(Pm - D))
    for L, anchor in (("LO_R", med(tin, "LO_R")), ("LOP_R", med(tout, "LOP_R")), ("LA_R", med(la_chain, "LA_R"))):
        if anchor is not None:
            lines[L] = (anchor, TS.thickness_dir(fc[L][0], fc[L][1], anchor))
    col = {L: TS.ME_COL_UM * np.sqrt(fc[L][2] / fc["ME_R"][2]) for L in TS.LABELS}

    def dist(df):
        d = np.full(len(df), np.nan)
        Q = df[X].to_numpy(float)
        for L, (o, n) in lines.items():
            i = np.where(df.np.to_numpy() == L)[0]
            if len(i):
                q = Q[i] - o
                d[i] = np.linalg.norm(q - np.outer(q @ n, n), axis=1) / col[L]
        return df.assign(d=d)

    s1, s2, s3 = dist(s1), dist(s2), dist(s3)
    rows = []
    for stage, df, key in (("PR", s1, "pre_root"), ("PR+1", s2, "pre_root"), ("PR+2", s3, "pre_root")):
        g = df.groupby(key)
        for cell, h in g:
            dd = h.d.dropna()
            rows.append(dict(stage=stage, root_id=str(cell), n_path=len(h), distance_columns=round(float(dd.median()), 2) if len(dd) else np.nan,
                             in_PRg1toT4T5=str(cell) in {str(y) for y in now(in_tab)},
                             pr_group=group_of.get(int(cell), "") if stage == "PR" else ""))
    t = pd.DataFrame(rows)
    t["group"] = pd.cut(t.distance_columns, [-1, 2, 3, 4, 6, np.inf], labels=["up to 2", "2-3", "3-4", "4-6", "more than 6"])
    out = fig(f"{stamp}_prg_any_paths.tsv")
    t.to_csv(out, sep="\t", index=False)
    print(f"\ncells per stage and distance group (columns from the PR group 1 home column):")
    print(pd.crosstab(t.stage, t.group, dropna=False).to_string())
    for st_ in ("PR+1", "PR+2"):
        u = t[t.stage == st_]
        print(f"  {st_}: {len(u)} cells, in PRg1toT4T5 already {int(u.in_PRg1toT4T5.sum())}; new within 2 / 3 / 4 columns: "
              f"{int(((u.distance_columns <= 2) & ~u.in_PRg1toT4T5).sum())} / {int(((u.distance_columns <= 3) & ~u.in_PRg1toT4T5).sum())} / "
              f"{int(((u.distance_columns <= 4) & ~u.in_PRg1toT4T5).sum())}")
    u = t[t.stage == "PR"]
    print(f"  PRs on paths: {len(u)}; by PR group (top): {u.pr_group.value_counts().head(15).to_dict()}")
    print(f"written {out}")


if __name__ == "__main__":
    main()
