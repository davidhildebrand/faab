"""The LMCs tab of the 'FAAB optic lobe targets' sheet (David, 2026-11-01): every known and putative lamina monopolar
cell, both sides, in six blocks - putative L1, L2, L3, L4, L5 and 'LMC, type unclear' - for others to confirm.

Sources, merged per current root (the first one listed wins the type):
  SeaTable   rows of class LMC, or typed L1..L5 in ol_intrinsic (fresh seatable_fetch.py TSV; roots there are live):
             type, status, proofreader, annotator - 'annotated'
  sheet      rows of the workbook's other tabs whose annotation / SeaTable / class / notes name an LMC type
             (scratch/lmc/sheet_lmc_rows.py; roots looked up at their xyz now): best status, names, dates - 'annotated'
  model      the L1-L5 typing (figures/lmc/lmc_typing.tsv + lmc_cartridge_calls.tsv, by SeaTable serial id), the
             LMC search (lmc_cand_typing.tsv: candidates with a soma by the medulla are left out as probably not
             LMCs; its 'L5' calls do not fit the L5 profile and go to 'type unclear') and the L5 search
             (lmc_l5_candidates.tsv) - 'high' / 'medium' / 'low'
Within a block: annotated rows first (checked, proofread, adequate, other), then high, medium, low; right side before
left; then by probability. An annotated row whose model call differs says so in its notes.
Points (xyz) - SeaTable's point, else the sheet's, else computed (step 'points'): on the primary neurite ~10 um from the
nucleus (faab.partners.top10_xyz), else the middle of the longest neurite; every point is checked to lie on its root
(roots at the points, one cloudvolume lookup), else the next skeleton nodes, else the nucleus.

Steps (files in figures/lmc/):
  points SEATABLE BOOK   CAVE read-only: skeleton points of the model-only cells -> lmc_tab_points.tsv
  build SEATABLE BOOK    CAVE read-only (sheet roots at xyz; state uploads for the block links) -> <stamp>_lmc_tab.tsv
  xlsx BOOK TSV          network-free: one-tab .xlsx in the look of BOOK's PRg1hop1 tab -> figures/sheets/
                         <stamp>_FAABopticlobetargets-LMCs.xlsx (import with File > Import > Insert new sheet)
Plain labels for the proofreaders, colorblind-safe link colors (memory: plain-labels-colorblind-colors)."""
import argparse
import copy
import datetime
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd

from faab.config import fig

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scratch", "lmc"))
TYPES = ["L1", "L2", "L3", "L4", "L5"]
VOX = np.array([16.0, 16.0, 45.0])
STATUS_RANK = {"checked": 0, "proofread": 1, "adequate": 2, "review": 3, "in progress": 4, "incomplete": 5,
               "missing soma": 6, "stopped": 7}
CONF_RANK = {"annotated": 0, "high": 1, "medium": 2, "low": 3}
COLOR = {"annotated": "#0072B2", "high": "#56B4E9", "medium": "#E69F00", "low": "#CC79A7"}  # Okabe-Ito, no red/green
# David 2026-11-07: the standard columns in the order of the other sheets; the LMC-specific ones right of link
COLS = ["root_id_orig", "root_id_curr", "xyz", "side", "putative_type", "annotation", "status", "pr_date", "proofreader",
        "chk_date", "checker", "notes", "SeaTable", "link", "confidence", "confirmed_type", "reason"]
# typed L5 profile (lmc_type_profiles.py, p10-p90 of the 11 SeaTable L5s): lamina inputs <= 26, lamina outputs >= 76,
# medulla outputs >= 683
L5_TIGHT = dict(max_in_LA=26, min_out_LA=76, min_out_ME=683)


def xyz_text(nm):
    v = np.round(np.asarray(nm, float) / VOX).astype(int)
    return f"{v[0]}, {v[1]}, {v[2]}"


def seatable_lmcs(path):
    S = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    lab = S[["superclass", "class", "subclass", "subsubclass", "type"]].agg(" / ".join, axis=1) \
        .str.replace(r"( / )+", " / ", regex=True).str.strip(" /")
    typed = S[["subclass", "subsubclass", "type"]].agg(" ".join, axis=1)
    pick = (S["class"] == "LMC") | ((S.superclass == "ol_intrinsic") & typed.str.contains(r"\bL[1-5]\b(?!\s*-)"))
    L = S[pick].copy()
    L["label"] = lab[L.index]
    L["t"] = L.label.map(lambda s: next((t for t in re.findall(r"\bL[1-5]\b(?![-–])", s)[::-1] if t), ""))
    L.loc[L.label.str.contains(r"L1\s*-\s*L?5"), "t"] = ""
    return L


def model_calls():
    """root -> (type, probability, confidence, reason, source) and serial -> the same, from the three model tables"""
    by_root, by_serial = {}, {}
    Ty = pd.read_csv(fig("lmc_typing.tsv"), sep="\t", dtype={"root": str, "serial_id": str}).fillna("")
    K = pd.read_csv(fig("lmc_cartridge_calls.tsv"), sep="\t", dtype={"root": str, "serial_id": str}).fillna("")
    assigned = {r: a for r, a in zip(K.root, K.assigned) if a}
    for x in Ty.itertuples(index=False):
        t = assigned.get(x.root, x.consensus)
        p = float(getattr(x, f"P_{t}"))
        conf = "high" if (x.votes == "3/3" and p >= 0.6) else "low" if (p < 0.4 or x.votes == "1/3") else "medium"
        why = (f"automatic typing: {x.votes[0]} of 3 methods agree (probability {p:.2f})" +
               (", one of each type within its LRS cartridge" if x.root in assigned else ""))
        v = (t, p, conf, why, "typing")
        by_root[x.root] = v
        if x.serial_id:
            by_serial[x.serial_id] = v
    C = pd.read_csv(fig("lmc_cand_typing.tsv"), sep="\t", dtype={"root": str}).fillna({"soma_by": "", "note": ""})
    for x in C.itertuples(index=False):
        if x.soma_by == "ME":
            continue  # soma by the medulla: probably not an LMC
        t = x.call if x.call != "L5" else ""  # the photoreceptor net's L5 calls do not fit the L5 profile
        why = (f"automatic typing: {x.votes[0]} of 3 methods agree (probability {x.p:.2f})" if t else
               "LMC-like (lamina input, medulla output) but its features fit no type well")
        by_root.setdefault(x.root, (t, float(x.p), x.confidence if t else "low", why, "search"))
    L5 = pd.read_csv(fig("lmc_l5_candidates.tsv"), sep="\t", dtype={"root": str}).fillna({"soma_by": ""})
    for x in L5[L5.l5_profile & ~L5.known_l5].itertuples(index=False):
        tight = x.n_in_LA <= L5_TIGHT["max_in_LA"] and x.n_out_LA >= L5_TIGHT["min_out_LA"] and \
            x.n_out_ME >= L5_TIGHT["min_out_ME"]
        conf = ("high" if tight else "medium") if x.soma_by == "LA" else "low"
        by_root.setdefault(x.root, ("L5", np.nan, conf, "L5 profile: few lamina inputs, many lamina and medulla outputs; "
                                                      "connected to L4s", "l5"))
    return by_root, by_serial


def side_by_xyz(xyz):
    """R / L by the nearer lamina mesh to a sheet point (voxels 'x, y, z'), '' if unreadable"""
    from soma_side import tree
    try:
        p = np.array([float(v) for v in str(xyz).split(",")]) * VOX / 1000.0
    except ValueError:
        return ""
    return "R" if tree("LA_R").query(p)[0] <= tree("LA_L").query(p)[0] else "L"


def soma_info():
    N = pd.read_csv(fig("nuclei_roots_now.tsv"), sep="\t", dtype={"root": str})
    return {r: g[["x", "y", "z"]].to_numpy(float)[0] * 1000.0 for r, g in N.groupby("root") if len(g) == 1}


# ------------------------------------------------------------------------------------------------------------ points
def cmd_points(a):
    import navis
    from fafbseg import flywire
    from faab.clients.cave import DATASTACK
    from faab.partners.top10_xyz import backbone_midpoint_nm, primary_neurite_point_nm
    navis.set_pbars(hide=True)
    by_root, _ = model_calls()
    S = seatable_lmcs(a.seatable)
    need = sorted(set(by_root) - set(S.root_now))
    Ty = pd.read_csv(fig("lmc_cells.tsv"), sep="\t", dtype=str).fillna("")
    have = dict(zip(Ty.root, Ty.point_xyz))
    need = [r for r in need if not have.get(r)]
    sk = pickle.load(open(fig("lmc_skeletons.pkl"), "rb"))
    soma = soma_info()
    rows, cands = [], {}
    for r in need:
        opts = []
        if r in sk:
            s = sk[r] * 1000.0  # um -> nm
            if r in soma:
                p, _ = primary_neurite_point_nm(s, soma[r])
                if p is not None:
                    opts.append(("primary neurite near the soma", p))
            opts.append(("middle of the longest neurite", backbone_midpoint_nm(s)))
            xyz = s.nodes[["x", "y", "z"]].to_numpy(float)
            ref = opts[0][1]
            for q in xyz[np.argsort(np.linalg.norm(xyz - ref, axis=1))[1:6]]:
                opts.append(("skeleton node", q))
        if r in soma:
            opts.append(("nucleus", soma[r]))
        cands[r] = opts
    print(f"{len(need)} cells need a point", flush=True)
    ok, k = {}, 0
    while True:  # round k: the k-th option of every cell still without a point on it
        todo = [(r, cands[r][k]) for r in need if r not in ok and k < len(cands[r])]
        if not todo:
            break
        locs = np.array([p for _, (_, p) in todo])
        seg = np.asarray(flywire.locs_to_segments(locs, coordinates="nm", backend="cloudvolume", dataset=DATASTACK)).astype(np.int64)
        for (r, (how, p)), sg in zip(todo, seg):
            if str(sg) == r:
                ok[r] = (how, p)
        print(f"  round {k + 1}: {len(todo)} checked, {len(ok)} cells with a point so far", flush=True)
        k += 1
    for r in need:
        how, p = ok.get(r, ("", None))
        rows.append(dict(root=r, xyz=xyz_text(p) if p is not None else "", point=how or "none on the cell"))
    P = pd.DataFrame(rows)
    P.to_csv(fig("lmc_tab_points.tsv"), sep="\t", index=False)
    print(f"points: {P.point.value_counts().to_dict()}; written {fig('lmc_tab_points.tsv')}")


# ------------------------------------------------------------------------------------------------------------- build
def merge_sheet(R):
    """sheet rows of one cell -> one record: best status, names and notes merged, latest dates"""
    R = R.copy()
    R["rank"] = R.status.astype(str).str.lower().map(STATUS_RANK).fillna(9)
    best = R.sort_values("rank").iloc[0]
    join = lambda col: "; ".join(dict.fromkeys(x.strip() for v in R[col].astype(str) for x in v.split(";")
                                                if x.strip() and x.strip() != "nan"))
    latest = lambda col: max((v for v in R[col] if isinstance(v, datetime.datetime)), default="")
    return dict(root_id_orig=str(best.root_id_orig), status=best.status, proofreader=join("proofreader"),
                checker=join("checker"), pr_date=latest("pr_date"), chk_date=latest("chk_date"),
                annotation=next((v for v in R.annotation.astype(str) if v.strip() and v != "nan"), ""),
                notes=join("notes"), SeaTable=next((v for v in R.SeaTable.astype(str) if v.strip() and v != "nan"), ""),
                cls=next((v for v in R["class"].astype(str) if v.strip() and v != "nan"), ""),
                xyz=str(best.xyz), tabs=", ".join(sorted(set(R.tab))))


def sheet_type(rec):
    for v in (rec["SeaTable"], rec["annotation"], rec["cls"]):
        m = re.findall(r"\bL[1-5]\b(?![-–])", v)
        if m and not re.search(r"L1\s*[-–]\s*L?5\b(?!\s*\()", v):
            return m[-1] + ("?" if "?" in v else "")
    return ""


def cmd_build(a):
    from fafbseg import flywire
    from faab.clients.cave import DATASTACK
    from sheet_lmc_rows import sheet_lmc_rows
    by_root, by_serial = model_calls()
    S = seatable_lmcs(a.seatable)
    pts = pd.read_csv(fig("lmc_tab_points.tsv"), sep="\t", dtype={"root": str}).fillna("")
    pt = dict(zip(pts.root, pts.xyz))
    Ty = pd.read_csv(fig("lmc_cells.tsv"), sep="\t", dtype=str).fillna("")
    extra_note = {r: n.replace("extra: ", "").replace("-", " ") for r, n in zip(Ty.root, Ty.notes) if n.startswith("extra:")}
    side_of = dict(zip(Ty.root, Ty.side))
    for f in ("lmc_cand_typing.tsv", "lmc_l5_candidates.tsv"):
        D = pd.read_csv(fig(f), sep="\t", dtype={"root": str})
        side_of.update(dict(zip(D.root, D.side)))
    # sheet rows -> current roots at their points
    SR = sheet_lmc_rows(a.book)
    SR["xyz"] = SR.xyz.astype(str)
    ok = SR.xyz.str.count(",") == 2
    u = sorted(set(SR.xyz[ok]))
    locs = np.array([[float(v) for v in x.split(",")] for x in u]) * VOX
    now = dict(zip(u, [str(int(x)) for x in flywire.locs_to_segments(locs, coordinates="nm", backend="cloudvolume",
                                                                      dataset=DATASTACK)]))
    SR["root_now"] = [now.get(x, "") or str(c) for x, c in zip(SR.xyz, SR.root_id_curr)]
    sheet = {r: merge_sheet(g) for r, g in SR.groupby("root_now") if r and r != "0"}
    print(f"SeaTable LMC rows {len(S)}; sheet rows {len(SR)} on {len(sheet)} cells; model calls {len(by_root)}", flush=True)

    rec = {}
    for x in S.itertuples(index=False):
        r = x.root_now
        m = by_serial.get(x.serial_id) or by_root.get(r)
        st_typed = bool(x.t)
        t = x.t or (m[0] if m else "")
        who = ", ".join(re.findall(r"[A-Za-z][\w .-]*", x.annotator))
        rec[r] = dict(root_id_orig=r, root_id_curr=r, xyz=x.point_xyz, side=x.side, putative_type=t,
                      confidence="annotated" if st_typed else (m[2] if m else "low"),
                      reason=((f"SeaTable type, annotated by {who}" if who else "SeaTable type") if st_typed else
                              (m[3] if m else "SeaTable: LMC, type not set")),
                      annotation=x.label, status=x.status, pr_date="", proofreader=x.proofreader, chk_date="", checker="",
                      notes=[], SeaTable=f"present: {x.label}", model=m, st_rank=STATUS_RANK.get(x.status.lower(), 8),
                      p=m[1] if m else np.nan, annotator=x.annotator)
        if st_typed and m and m[0] and m[0] != x.t:
            rec[r]["notes"].append(f"SeaTable says {x.t}; the automatic typing says {m[0]} (probability {m[1]:.2f})")
    for r, s in sheet.items():
        t = sheet_type(s)
        if r in rec:
            e = rec[r]
            if STATUS_RANK.get(str(s["status"]).lower(), 9) <= 1:  # our checked / proofread is kept (sheet rule)
                e["status"] = s["status"]
            e.update(root_id_orig=s["root_id_orig"], pr_date=s["pr_date"], chk_date=s["chk_date"], checker=s["checker"])
            if s["proofreader"]:  # the sheet's names; SeaTable's initials only when the sheet has none (no duplicates)
                e["proofreader"] = s["proofreader"]
            if s["notes"]:
                e["notes"].append(s["notes"])
            continue
        m = by_root.get(r)
        rec[r] = dict(root_id_orig=s["root_id_orig"], root_id_curr=r, xyz=s["xyz"], side=side_of.get(r) or side_by_xyz(s["xyz"]),
                      putative_type=t.rstrip("?") or (m[0] if m else ""),
                      confidence="annotated" if t and not t.endswith("?") else (m[2] if m else "low"),
                      reason=f"sheet ({s['tabs']}): {s['SeaTable'] or s['annotation'] or s['cls']}",
                      annotation=s["annotation"] or s["cls"], status=s["status"], pr_date=s["pr_date"],
                      proofreader=s["proofreader"], chk_date=s["chk_date"], checker=s["checker"],
                      notes=[s["notes"]] if s["notes"] else [], SeaTable=s["SeaTable"], model=m,
                      st_rank=STATUS_RANK.get(str(s["status"]).lower(), 8), p=m[1] if m else np.nan, annotator="")
    for r, m in by_root.items():
        if r in rec or m[4] == "typing" and r not in extra_note:
            continue  # typing-set cells come in through SeaTable by serial id; only David's flagged extras here
        sd = side_of.get(r, "")
        xyz = pt.get(r, "") or dict(zip(Ty.root, Ty.point_xyz)).get(r, "")
        rec[r] = dict(root_id_orig=r, root_id_curr=r, xyz=xyz, side=sd, putative_type=m[0], confidence=m[2], reason=m[3],
                      annotation="", status="", pr_date="", proofreader="", chk_date="", checker="",
                      notes=([f"flagged as an LMC by David ({extra_note[r]})"] if r in extra_note else []),
                      SeaTable="", model=m, st_rank=9, p=m[1], annotator="")
    # notes in plain words
    C = pd.read_csv(fig("lmc_cand_typing.tsv"), sep="\t", dtype={"root": str}).fillna({"soma_by": ""}).set_index("root")
    L5 = pd.read_csv(fig("lmc_l5_candidates.tsv"), sep="\t", dtype={"root": str}).fillna({"soma_by": ""}).set_index("root")
    for r, e in rec.items():
        src = C if r in C.index else L5 if r in L5.index else None
        if src is not None and e["confidence"] != "annotated":
            x = src.loc[r]
            if x.soma_by == "":
                e["notes"].append("no nucleus on this root (cell body cut off, or a fragment)")
            elif x.soma_by == "merge?":
                e["notes"].append("several nuclei on this root: merge?")
            if src is C and not bool(x.complete):
                e["notes"].append("medulla part short or missing: proofread before typing")
            if src is C and not bool(x.typical) and e["putative_type"]:
                e["notes"].append("features unlike the typed LMCs")
        if e["side"] == "L" and e["confidence"] != "annotated":
            e["notes"].append("left side: the automatic typing was trained on right-side cells")
        if not e["xyz"]:
            e["notes"].append("no point found on the cell")
    T = pd.DataFrame(rec.values())
    T["block"] = T.putative_type.where(T.putative_type.isin(TYPES), "unclear")
    T["rank"] = T.confidence.map(CONF_RANK) * 10 + T.st_rank.where(T.confidence == "annotated", 0)
    T["side_rank"] = T.side.map({"R": 0, "L": 1}).fillna(2)
    T = T.sort_values(["block", "rank", "side_rank", "p"], ascending=[True, True, True, False])
    T["notes"] = T.notes.map(lambda v: "; ".join(dict.fromkeys(x for x in v if x)))
    for c in ("pr_date", "chk_date"):
        T[c] = T[c].map(lambda v: v.strftime("%Y-%m-%d") if isinstance(v, datetime.datetime) else (v or ""))
    T["link"], T["confirmed_type"] = "", ""
    T["putative_type"] = T.putative_type.replace("", "?")
    # one view per block: its cells by confidence
    links = block_links(T, a.stamp)
    stamp = a.stamp
    lines = [f">>> LMCs (lamina monopolar cells), both sides: known and putative types for confirmation "
             f"(updated {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M} UTC)"]
    names = {t: f"putative {t}" for t in TYPES} | {"unclear": "LMC, type unclear"}
    for b in TYPES + ["unclear"]:
        g = T[T.block == b]
        if not len(g):
            continue
        n = g.confidence.value_counts()
        title = [""] * len(COLS)  # the block's link sits in the link column
        title[0] = f"{names[b]}: {len(g)} cells (" + ", ".join(f"{k} {int(n.get(k, 0))}" for k in CONF_RANK) + ")"
        title[COLS.index("link")] = links.get(b, "")
        lines.append("\t".join(title).rstrip("\t"))
        lines.append("\t".join(COLS))
        for x in g[COLS].itertuples(index=False):
            lines.append("\t".join("" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v).replace("\t", " ")
                                   for v in x))
        lines.append("")
    lines += [
        "note: putative_type / confidence - 'annotated' = typed in SeaTable or in this sheet; 'high' / 'medium' / 'low' = "
        "automatic typing (lamina and medulla depth of each cell's branches and synapses, trained on the SeaTable-typed "
        "right-side LMCs) or, for L5, the L5 synapse profile. Low = incomplete or unusual cells: proofread first.",
        "note: to confirm, write the type in confirmed_type and your name in checker (date in chk_date). Cells were found "
        "as targets of photoreceptors R1-R6 / R8 with lamina input and medulla output, and L5s as partners of L4s; cells "
        "with a cell body by the medulla were left out (probably not LMCs).",
        "note: the link on each block line shows its cells colored by confidence: annotated blue, high sky blue, "
        "medium orange, low purple."]
    out = fig(f"{stamp}_lmc_tab.tsv")
    open(out, "w").write("\n".join(lines) + "\n")
    print(pd.crosstab(T.block, T.confidence).reindex(columns=list(CONF_RANK), fill_value=0).to_string())
    print(f"sides {T.side.value_counts().to_dict()}; written {out}")


def block_links(T, stamp):
    from faab import nglink
    from faab.clients.cave import client as cave_client, throttle
    client, t, out = cave_client(), 0.0, {}
    for b in TYPES + ["unclear"]:
        g = T[T.block == b]
        if not len(g):
            continue
        neurons = [{"name": f"{'putative ' + b if b in TYPES else 'type unclear'}: {k} ({(g.confidence == k).sum()})",
                    "color": COLOR[k], "ids": g[g.confidence == k].root_id_curr.tolist()} for k in CONF_RANK
                   if (g.confidence == k).any()]
        spec = {"meshes": [{"kind": "OL", "segments": ["11", "14", "21", "24"], "alpha": 0.12}], "neurons": neurons,
                "layout": "3d", "select": neurons[0]["name"]}
        state = nglink.clean_state(nglink.build(spec))
        open(fig(f"{stamp}_lmc_tab_{b}_ng.txt"), "w").write(nglink.to_url(state) + "\n")
        t = throttle(t)
        out[b] = client.state.build_neuroglancer_url(client.state.upload_state_json(state),
                                                     ngl_url="https://spelunker.cave-explorer.org")
        print(f"  {b}: {out[b]}", flush=True)
    return out


# -------------------------------------------------------------------------------------------------------------- xlsx
def cmd_xlsx(a):
    import openpyxl
    from openpyxl.utils import get_column_letter
    from faab.partners.sheet import utc_stamp
    from faab.pr.prs_tab import as_date
    wb = openpyxl.load_workbook(a.book)
    src = wb["PRg1hop1"]
    s_hdr = {str(src.cell(2, c).value).strip(): c for c in range(1, src.max_column + 1) if src.cell(2, c).value}
    plain = next(r for r in range(3, src.max_row + 1) if str(src.cell(r, 1).value or "").strip().isdigit()
                 and not any(src.cell(r, c).fill is not None and src.cell(r, c).fill.fill_type == "solid"
                             for c in range(1, src.max_column + 1)))
    out = wb.create_sheet("LMCs")
    out.sheet_view.showGridLines = src.sheet_view.showGridLines
    near = {"side": "PRtype", "putative_type": "PRtype", "confidence": "status", "confirmed_type": "annotation",
            "reason": "notes", "root_id_orig": "root_id_orig"}
    widths = {"side": 6, "putative_type": 12, "confidence": 11, "confirmed_type": 14, "reason": 48}
    for j, name in enumerate(COLS, 1):
        sc = s_hdr.get(name) or s_hdr.get(near.get(name, ""), None)
        d = src.column_dimensions.get(get_column_letter(sc)) if sc else None
        out.column_dimensions[get_column_letter(j)].width = widths.get(name) or (d.width if d is not None and d.width else 14)
    lines = open(a.tsv).read().rstrip("\n").split("\n")
    for i, ln in enumerate(lines, 1):
        cells = ln.split("\t")
        first = cells[0]
        kind = ("title" if first.startswith(">>>") or first.startswith("note:") else "header" if first == COLS[0] else
                "data" if first.isdigit() else "block" if first else "blank")
        style_row = {"title": 1, "header": 2, "data": plain, "block": 1, "blank": None}[kind]
        for j, text in enumerate(cells, 1):
            name = COLS[j - 1] if j <= len(COLS) else ""
            sc = s_hdr.get(name) or s_hdr.get(near.get(name, ""), None) or 1
            v = text if text != "" else None
            if kind == "data" and name in ("pr_date", "chk_date") and v:
                v = as_date(v) or v
            c = out.cell(i, j)
            if style_row:
                c._style = copy.copy(src.cell(style_row, sc if kind in ("data", "header") else 1)._style)
            if kind in ("block", "header", "title") and v:
                f = copy.copy(c.font)
                f.b = True
                c.font = f
            if isinstance(v, str) and v.startswith("http"):
                c.hyperlink = v  # before the value (faab.pr.prs_tab.restore)
            c.value = v
            if isinstance(v, datetime.datetime):
                c.number_format = 'yy"-"mm"-"dd'
    for ws in list(wb.worksheets):
        if ws is not out:
            wb.remove(ws)
    path = a.out or fig(f"{utc_stamp()}_FAABopticlobetargets-LMCs.xlsx")
    wb.save(path)
    print(f"wrote {path}: {len(lines)} lines")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("points", "build"):
        p = sub.add_parser(name)
        p.add_argument("seatable")
        p.add_argument("book")
        p.add_argument("--stamp", default=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dd%H%M%StUTC"))
    p = sub.add_parser("xlsx")
    p.add_argument("book")
    p.add_argument("tsv")
    p.add_argument("--out")
    a = ap.parse_args()
    {"points": cmd_points, "build": cmd_build, "xlsx": cmd_xlsx}[a.cmd](a)


if __name__ == "__main__":
    main()
