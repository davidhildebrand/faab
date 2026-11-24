"""Proofread counts and the proofread cells around the PR group 1 column (David 2026-11-24: "count the number of cells
proofread to status adequate or proofread ... the subset of those cells clearly annotated with a type"; "a list of all
root_ids for adequate or proofread status PR+1 +2 +3 T4/T5 and T4T5-1 -2 -3 within a 25um column radius of the
PR-to-T4/T5 axis").

  cells   SeaTable (a fresh *_seatable_aedes_main.tsv) and the optic-lobe workbook (every tab): rows with status
          adequate / proofread, distinct live cells, and those clearly typed. Workbook cells get their live roots
          (is_latest_roots, throttled get_latest_roots for stale ids; cached in tmp/) and SeaTable's status when one of
          their live pieces is a SeaTable row's root_now (status follows SeaTable), else the sheet's status.
          Writes figures/t4t5/<stamp>_proofread_cells.tsv.
  column  the PRg1toT4T5 cells (PR+1, PR+2, PR+3, T4T5-1, T4T5-2) and the T4T5s tab cells with status adequate /
          proofread, each scored by its synapses (inputs at their post site, outputs at their pre site) inside the
          right LA / ME / LO / LOP: perpendicular distance from that neuropil's PR group 1 home-column line, built
          exactly as in scripts/partners/t4t5_sort.py (ME: median PR+1 -> PR+2 synapse to the median PR+3 input
          synapse in ME; LO: median PR+3 input in LO along the lobula thickness; LOP: median PR+3 output in LOP along
          the lobula plate thickness; LA: median PR group 1 -> PR+1 synapse along the lamina thickness). A cell is
          WITHIN when the median distance of its optic-lobe synapses is <= RADIUS um (half of them inside the
          cylinder). Writes figures/t4t5/<stamp>_column<R>_proofread.tsv, <stamp>_column<R>_ids.txt and an ng spec.
Read-only toward CAVE (is_latest_roots, throttled get_latest_roots, snapshot catch-up).
Usage: proofread_column.py cells BOOK.xlsx SEATABLE.tsv
       proofread_column.py column CELLS.tsv [--radius 25]"""
import argparse
import importlib.util
import json
import os
import re
import time

import numpy as np
import openpyxl
import pandas as pd

from faab import ol_meshes, synsnap
from faab.clients.cave import client as cave_client, throttle
from faab.config import DATA, TMP, fig
from faab.partners import annotation as ANN
from faab.partners.sheet import cell_text as s, utc_stamp

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "t4t5_sort", os.path.join(HERE, "..", "..", "scripts", "partners", "t4t5_sort.py"))
TS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TS)

ID18 = re.compile(r"^\d{18}$")
GOOD = ("proofread", "adequate")
HOPS = ("PR+1", "PR+2", "PR+3", "T4T5-1", "T4T5-2", "T4T5-3")
LIVE_CACHE = os.path.join(TMP, "proofread_column_live_roots.json")
VOX = TS.VOX
LABELS = TS.LABELS
# label words naming a family or class, not a cell type
FAMILY = {"lmc", "tm", "mi", "tmy", "dm", "pm", "lawf", "lc", "lplc", "llpc", "lpc", "metu", "lt", "lpt", "vpn", "olsn",
          "photoreceptor", "pr", "t4/t5", "t4t5", "c", "t", "y", "ct", "tlp", "lpi", "mt", "tml", "tmy-like", "ol",
          "neuron", "cell", "visual", "intrinsic", "projection", "centrifugal", "wide-field", "widefield", "large",
          "mevc", "sm", "dn", "lome"}
NONTYPE = set(FAMILY)  # + SeaTable's superclass and class vocabularies (cmd_cells)
UNSURE = re.compile(r"\?|-like|\blike\b|putative|unclear|unknown|uncertain|cand|maybe|possib|probab|\bor\b|fragment|"
                    r"glia|not a neuron|tbd|check|unsure|\bnot\b", re.I)


def typed(text):
    """True when the text names a specific cell type with no doubt marker. Levels are split at ' / ' (spaces), ';',
    ',' or '|'; a level holding '/' without spaces ('C2/C3', 'T4/T5') names alternatives, not a type"""
    t = s(text).strip()
    if not t or UNSURE.search(t):
        return False
    words = [w.strip().lower() for w in re.split(r"\s+/\s+|[;,|]", t) if w.strip()]
    return any(w not in ANN.GENERIC and w not in NONTYPE and ("/" not in w or re.fullmatch(r"r\d/r\d", w))
               for w in words)  # photoreceptor pair types R1/R6, R2/R5, R3/R4 are types


def sheet_rows(book):
    wb = openpyxl.load_workbook(book, read_only=True)
    out = []
    for ws in wb.worksheets:
        hdr, title = None, ""
        for row in ws.iter_rows(values_only=True):
            v = [s(x).strip() for x in row]
            if not any(v):
                continue
            if "root_id_orig" in v or "root_id_curr" in v:
                hdr = {n: j for j, n in enumerate(v) if n}
                continue
            if hdr is None:
                title = v[0]
                continue
            get = lambda n: v[hdr[n]] if n in hdr and hdr[n] < len(v) else ""  # noqa: E731
            cu, orig = get("root_id_curr"), get("root_id_orig")
            root = cu if ID18.match(cu) else (orig if ID18.match(orig) else "")
            if not root:
                title = v[0]
                continue
            hop = v[0] if ws.title == "PRg1toT4T5" else ("T4/T5 (T4T5s tab)" if ws.title == "T4T5s" else ws.title)
            ann = get("annotation") or (" / ".join(x for x in (get("class"), get("type")) if x) if ws.title == "PRs" else "")
            out.append(dict(tab=ws.title, hop=hop, block=title[:80], root_written=root, status=get("status").lower(),
                            annotation=ann, confirmed_type=get("confirmed_type"), seatable_mark=get("SeaTable")))
    return pd.DataFrame(out)


def live_roots(ids, c):
    cache = json.load(open(LIVE_CACHE)) if os.path.exists(LIVE_CACHE) else {}
    todo = [x for x in ids if x not in cache]
    if todo:
        last = throttle(0)
        ok = []
        for i in range(0, len(todo), 5000):
            last = throttle(last)
            ok.append(np.asarray(c.chunkedgraph.is_latest_roots([int(x) for x in todo[i:i + 5000]])).astype(bool))
        ok = np.concatenate(ok)
        stale = [x for x, g in zip(todo, ok) if not g]
        print(f"{len(todo)} ids checked, {len(stale)} stale -> get_latest_roots one by one (2 s apart)", flush=True)
        res = {x: [x] for x, g in zip(todo, ok) if g}
        for k, x in enumerate(stale):
            last = throttle(last)
            try:
                res[x] = [str(int(y)) for y in np.atleast_1d(c.chunkedgraph.get_latest_roots(int(x)))]
            except ValueError:
                res[x] = []
            if k % 25 == 24:
                print(f"  {k + 1}/{len(stale)}", flush=True)
        # cache only the stale answers (a latest id can go stale later; re-check those every run)
        cache.update({x: res[x] for x in stale})
        json.dump(cache, open(LIVE_CACHE, "w"))
        cache = {**cache, **res}
    return {x: cache[x] for x in ids}


def cmd_cells(a):
    stamp = utc_stamp()
    st = pd.read_csv(a.seatable, sep="\t", dtype=str).fillna("")
    st["status"] = st.status.str.strip()
    lab_cols = ["superclass", "class", "subclass", "subsubclass", "type"]
    st["label"] = st[lab_cols].apply(lambda r: " / ".join(x.strip() for x in r if x.strip()), axis=1)
    NONTYPE.update(v.strip().lower() for c in ("superclass", "class") for v in st[c] if v.strip())
    st["type_ok"] = st["type"].map(lambda x: typed(x) and not x.strip().endswith("_cand"))

    # ---- SeaTable
    g = st[st.status.isin(GOOD)]
    roots = g[g.root_now.str.match(r"^\d{18}$")]
    per = roots.groupby("root_now").agg(type_ok=("type_ok", "any"), superclass=("superclass", "first"),
                                        status=("status", lambda v: "proofread" if "proofread" in set(v) else "adequate"))
    ol = per.superclass.isin(["ol_intrinsic", "ol_sensory", "visual_projection", "visual_centrifugal", "ol_bilateral"])
    print(f"SEATABLE ({os.path.basename(a.seatable)}): rows adequate {int((st.status == 'adequate').sum())}, "
          f"proofread {int((st.status == 'proofread').sum())}; distinct live cells {len(per)} "
          f"(proofread {int((per.status == 'proofread').sum())}); clearly typed ('type' filled, no ?, -like, _cand) "
          f"{int(per.type_ok.sum())}")
    print(f"  optic-lobe superclasses: {int(ol.sum())} cells, typed {int((ol & per.type_ok).sum())}")
    print("  by superclass (cells / typed): " + "; ".join(
        f"{k} {int(n)}/{int(t)}" for k, (n, t) in per.groupby("superclass").type_ok.agg(["size", "sum"])
        .sort_values("size", ascending=False).iterrows()))
    unsure = g[g["type"].str.strip().ne("") & ~g.type_ok]["type"].value_counts()
    print(f"  'type' filled but not counted: {unsure.to_dict()}")

    # ---- workbook
    R = sheet_rows(a.book)
    print(f"\nWORKBOOK ({os.path.basename(a.book)}): {len(R)} rows with an id in {R.tab.nunique()} tabs")
    c = cave_client()
    ids = sorted(set(R.root_written))
    live = live_roots(ids, c)
    st_by = {}
    for r in st.itertuples():
        if ID18.match(r.root_now):
            st_by.setdefault(r.root_now, []).append(r)
    rank = {"proofread": 0, "adequate": 1}
    cells = []
    for root, rows in R.groupby("root_written"):
        pieces = live.get(root, [])
        hit = [(p, x) for p in pieces for x in st_by.get(p, [])]
        primary = hit[0][0] if hit else (pieces[0] if pieces else root)
        if hit:
            stats = sorted({x.status for _, x in hit}, key=lambda z: rank.get(z, 9))
            status, src = stats[0], "SeaTable"
        else:
            stats = sorted(set(rows.status) - {""}, key=lambda z: rank.get(z, 9))
            status, src = (stats[0] if stats else ""), "sheet"
        anns = sorted(set(x for x in rows.annotation if x) | set(x for x in rows.confirmed_type if x))
        cells.append(dict(root=primary, root_written=root, pieces=";".join(pieces), status=status, status_from=src,
                          tabs="; ".join(sorted(set(rows.tab))), hops="; ".join(sorted(set(rows.hop))),
                          annotation=" | ".join(anns), seatable_label=" | ".join(sorted({x.label for _, x in hit})),
                          seatable_type_ok=any(x.type_ok for _, x in hit),
                          typed_sheet=any(typed(x) for x in anns)))
    C = pd.DataFrame(cells)
    C["typed"] = C.typed_sheet | C.seatable_type_ok
    path = fig(f"t4t5/{stamp}_proofread_cells.tsv")
    C.to_csv(path, sep="\t", index=False)
    good = C[C.status.isin(GOOD)].drop_duplicates("root")
    print(f"distinct cells {C.root.nunique()}; adequate/proofread {len(good)} (proofread "
          f"{int((good.status == 'proofread').sum())}; status from SeaTable {int((good.status_from == 'SeaTable').sum())}, "
          f"from the sheet {int((good.status_from == 'sheet').sum())}); clearly typed {int(good.typed.sum())} "
          f"(sheet annotation {int(good.typed_sheet.sum())}, SeaTable type {int(good.seatable_type_ok.sum())})")
    tab_of = R.groupby("root_written").tab.agg(lambda v: sorted(set(v)))
    for t in R.tab.unique():
        q = good[good.root_written.map(lambda x: t in tab_of.get(x, []))]
        if len(q):
            print(f"  {t}: {len(q)} adequate/proofread, typed {int(q.typed.sum())}")
    nt = good[~good.typed].annotation.replace("", "(blank)").value_counts().head(25)
    print(f"  not counted as typed (top annotations): {nt.to_dict()}")
    tt = good[good.typed].annotation.replace("", "(blank, SeaTable type)").value_counts().head(25)
    print(f"  counted as typed (top annotations): {tt.to_dict()}")
    print(f"written {path}")


def cmd_column(a):
    stamp = utc_stamp()
    C = pd.read_csv(a.cells, sep="\t", dtype=str).fillna("")
    hop_sets = {h: C[C.hops.str.split("; ").map(lambda v: h in v)] for h in (*HOPS, "T4/T5 (T4T5s tab)")}
    print("rows per stage (all statuses): " + "; ".join(f"{h} {len(q)}" for h, q in hop_sets.items()), flush=True)
    piece = lambda q: sorted({p for v in q.pieces for p in v.split(";") if p})  # noqa: E731

    prs = pd.read_csv(os.path.join(DATA, "pr_groups.tsv"), sep="\t", dtype=str)
    c = cave_client()
    g1_live = live_roots(sorted(set(prs[prs.group == "1"].root_id_curr)), c)
    g1 = sorted({p for v in g1_live.values() for p in v})
    snap = synsnap.Snapshot(log=lambda *_: None)

    def syn(pre, post, side="post"):
        t0 = time.time()
        df = snap.synapses(pre=[int(x) for x in pre] if pre is not None else None,
                           post=[int(x) for x in post] if post is not None else None, when="now", positions=True,
                           client=c)
        df = df[df.pre_root != df.post_root]
        P = df[[f"{side}_x", f"{side}_y", f"{side}_z"]].to_numpy(float) * VOX
        df = df.assign(x=P[:, 0], y=P[:, 1], z=P[:, 2])
        df["np"] = ol_meshes.assign(P, LABELS) if len(df) else []
        print(f"  {len(df)} synapses ({time.time() - t0:.0f} s)", flush=True)
        return df

    print("home-column lines (as t4t5_sort):", flush=True)
    h1, h2, h3 = piece(hop_sets["PR+1"]), piece(hop_sets["PR+2"]), piece(hop_sets["PR+3"])
    ch_la, ch_me, tin, tout = syn(g1, h1), syn(h1, h2), syn(None, h3), syn(h3, None, side="pre")
    X = ["x", "y", "z"]
    med = lambda df, L: df[df.np == L][X].median().to_numpy() if int((df.np == L).sum()) >= 10 else None  # noqa: E731
    fc = {L: TS.faces(L) for L in LABELS}
    lines = {}
    D, P = med(ch_me, "ME_R"), med(tin, "ME_R")
    lines["ME_R"] = (D, (P - D) / np.linalg.norm(P - D))
    for L, anchor in (("LO_R", med(tin, "LO_R")), ("LOP_R", med(tout, "LOP_R")), ("LA_R", med(ch_la, "LA_R"))):
        lines[L] = (anchor, TS.thickness_dir(fc[L][0], fc[L][1], anchor))
    for L, (o, n) in lines.items():
        print(f"  {L}: through {np.round(o, 1).tolist()} um, direction {np.round(n, 2).tolist()}", flush=True)

    cand = C[C.status.isin(GOOD) & C.hops.str.split("; ").map(lambda v: bool(set(v) & set(hop_sets)))]
    cand = cand.drop_duplicates("root")
    ids = piece(cand)
    print(f"\ncandidates (adequate / proofread): {len(cand)} cells, {len(ids)} live pieces; their synapses:", flush=True)
    outs = syn(ids, None, side="pre").assign(cell=lambda d: d.pre_root.astype(str))
    ins = syn(None, ids, side="post").assign(cell=lambda d: d.post_root.astype(str))
    S = pd.concat([outs, ins])
    S = S[S.np.isin(LABELS)]
    d = np.full(len(S), np.nan)
    Q = S[X].to_numpy(float)
    for L, (o, n) in lines.items():
        i = np.where(S.np.to_numpy() == L)[0]
        q = Q[i] - o
        d[i] = np.linalg.norm(q - np.outer(q @ n, n), axis=1)
    S = S.assign(d=d)
    p2c = {p: r.root for r in cand.itertuples() for p in r.pieces.split(";") if p}
    S["root"] = S.cell.map(p2c)
    agg = S.groupby("root").agg(n_ol=("d", "size"), median_um=("d", "median"),
                                within_share=("d", lambda v: float((v <= a.radius).mean())),
                                main_np=("np", lambda v: v.value_counts().index[0]))
    out = cand.set_index("root").join(agg, how="left").reset_index()
    out["n_ol"] = out.n_ol.fillna(0).astype(int)
    out["within"] = out.median_um <= a.radius
    rows = []
    for h in (*HOPS, "T4/T5 (T4T5s tab)"):
        q = out[out.hops.str.split("; ").map(lambda v: h in v)]
        for r in q.itertuples():
            rows.append(dict(stage=h, root_id=r.root, root_id_written=r.root_written, status=r.status,
                             annotation=r.annotation or r.seatable_label, n_ol_synapses=r.n_ol,
                             median_um=round(r.median_um, 1) if pd.notna(r.median_um) else "",
                             share_within=round(r.within_share, 2) if pd.notna(r.within_share) else "",
                             main_neuropil=r.main_np if isinstance(r.main_np, str) else "", within=bool(r.within)))
    T = pd.DataFrame(rows).sort_values(["stage", "within", "median_um"], ascending=[True, False, True])
    R = int(a.radius)
    path = fig(f"t4t5/{stamp}_column{R}_proofread.tsv")
    T.to_csv(path, sep="\t", index=False)
    print(f"\nwithin {a.radius:g} um (median synapse distance from the PR group 1 column line):")
    lines_txt = []
    for h in (*HOPS, "T4/T5 (T4T5s tab)"):
        q = T[T.stage == h]
        w = q[q.within]
        print(f"  {h}: {len(w)} of {len(q)} adequate/proofread cells", flush=True)
        if len(w):
            lines_txt.append(f"# {h} ({len(w)})")
            lines_txt += w.root_id.tolist()
    ids_path = fig(f"t4t5/{stamp}_column{R}_ids.txt")
    open(ids_path, "w").write("\n".join(lines_txt) + "\n")
    print(f"written {path}\nwritten {ids_path}")

    # ng spec: one layer per stage (Okabe-Ito), PR group 1 for context, the right optic lobe in view
    col = {"PR+1": "#E69F00", "PR+2": "#56B4E9", "PR+3": "#009E73", "T4T5-1": "#0072B2", "T4T5-2": "#D55E00",
           "T4T5-3": "#000000", "T4/T5 (T4T5s tab)": "#F0E442"}
    neurons = [{"name": f"PR group 1 ({len(g1)})", "color": "#CC79A7", "ids": g1}]
    for h in (*HOPS, "T4/T5 (T4T5s tab)"):
        w = T[(T.stage == h) & T.within]
        w = w[~w.root_id.isin([x for n in neurons for x in n["ids"]])]
        if len(w):
            neurons.append({"name": f"{h} proofread, within {R} um ({len(w)})", "color": col[h],
                            "ids": w.root_id.tolist()})
    centre = np.mean([o for o, _ in lines.values()], axis=0) / VOX
    spec = {"meshes": [{"kind": "brain"}, {"kind": "OL"}], "neurons": neurons, "layout": "3d", "image": True,
            "compact": True, "position": np.round(centre, 0).tolist(), "scale": 9000,
            "out": os.path.join(TMP, f"{stamp}_column{R}_ng.txt")}
    sp = os.path.join(TMP, f"{stamp}_column{R}_spec.json")
    json.dump(spec, open(sp, "w"), indent=1)
    print(f"ng spec {sp}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cells")
    p.add_argument("book")
    p.add_argument("seatable")
    p = sub.add_parser("column")
    p.add_argument("cells")
    p.add_argument("--radius", type=float, default=25.0)
    a = ap.parse_args()
    {"cells": cmd_cells, "column": cmd_column}[a.cmd](a)


if __name__ == "__main__":
    main()
