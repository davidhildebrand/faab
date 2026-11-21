"""Refresh the synapse-annotation cutouts (David 2026-11-20: "provide me the prioritized list of cutouts with links
including the (refreshed) PR or other terminal intersecting with the box"). Read-only CAVE, >= 2 s between calls.

Per box of a registry (scripts/cutouts/registry_all.py, e.g. figures/cutouts/20261102d123131tUTC_cutouts_all.tsv):
  1. every synapse inside the box now (one live spatial query on synapses_v2);
  2. the registry's target cells mapped to today's roots (is_latest_roots; get_latest_roots for edited ones, keeping
     the descendants with synapses in the box, else all of them);
  3. the terminals in the box: every cell with output synapses there that is a photoreceptor (PRs tab class / group at
     today's roots, or a SeaTable photoreceptor label) or an LMC (SeaTable LMC / L1-L5, LMCs tab), and for 'T4/T5 inputs'
     boxes every T4 / T5 (SeaTable, T4T5s tab) with input synapses there - plus the refreshed target cells;
  4. a short Spelunker link (README scene via faab.nglink; state uploaded to the state server, which only stores a view):
     EM, 4 panels zoomed on the box, the box drawn, the terminals in one layer (Okabe-Ito colors, the box's own target
     type first), and the predicted synapses in the box as a HIDDEN points layer (compare after annotating).
Priority: PRIORITY below (David's main open question first: R7 synapses the prediction misses, on proofread R7s; then
the lamina photoreceptor boxes, which no existing cutout covers; then R7, LMC and T4/T5 prediction checks; the R7
missed-synapse boxes on unproofread left R7s last, until those R7s are proofread).
Writes figures/cutouts/<stamp>_cutouts_refreshed.tsv and .md. Usage: refresh_boxes.py REGISTRY.tsv BOOK.xlsx SEATABLE.tsv"""
import datetime
import os
import re
import sys
import time

import numpy as np
import openpyxl
import pandas as pd

from faab import nglink
from faab.clients.cave import client as cave_client, throttle
from faab.config import VOX, fig
from faab.partners.sheet import cell_text as s, scan_book, utc_stamp

PRIORITY = [  # (box, why)
    ("R7-typical-R-2", "R7 misses, random R7 point, on a vetted R7 (LRS cartridge 7)"),
    ("R7-cold-R-1", "R7 misses where the prediction is coldest"),
    ("R7-cold-R-2", "R7 misses where the prediction is coldest (second box)"),
    ("R7-typical-R-1", "R7 misses, random R7 point; check first: SeaTable calls this R7 (group 7) an R1-R6"),
    ("R1-R6-R-1", "lamina R1-R6 outputs, proofread groups 1, 2, 4, 5"),
    ("R8-R-1", "lamina R8 outputs"),
    ("R1-R6-L-2", "lamina R1-R6 outputs, left side, vetted group 104"),
    ("R8-L-1", "lamina R8 outputs, left side, vetted group 105"),
    ("R7-R-3", "R7 prediction check on a vetted R7"),
    ("R7-R-1", "R7 prediction check on a vetted R7"),
    ("R7-L-2", "R7 prediction check, left side, vetted group 105"),
    ("L3-R-1", "LMC (L3) terminal prediction check, most target synapses"),
    ("L2-R-1", "LMC (L2) terminal prediction check"),
    ("L1-R-1", "LMC (L1) terminal prediction check"),
    ("LMC-L-1", "LMC terminal prediction check, left side"),
    ("T4-R-1", "T4 input prediction check"),
    ("T5-R-1", "T5 input prediction check"),
    ("T4-L-1", "T4 input prediction check, left side"),
    ("T5-L-1", "T5 input prediction check, left side"),
    ("R1-R6-R-2", "second lamina R1-R6 box (right)"),
    ("R1-R6-L-1", "second lamina R1-R6 box (left; group 103 partly unproofread)"),
    ("R7-R-2", "further R7 prediction check"),
    ("R7-R-4", "further R7 prediction check"),
    ("R7-L-1", "further R7 prediction check (left; group 103)"),
    ("R7-cold-L-1", "R7 misses on an unproofread left R7: proofread the R7 first"),
    ("R7-typical-L-1", "R7 misses on an unproofread left R7: proofread the R7 first"),
    ("R7-cold-L-2", "R7 misses on an unproofread left R7: proofread the R7 first"),
    ("R7-typical-L-2", "R7 misses on an unproofread left R7: proofread the R7 first"),
]
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#FFFFFF"]
BOX_COLOR = {"prediction check": "#F0E442", "R7 missed, cold": "#D55E00", "R7 missed, typical": "#FFFFFF"}
PR_LABEL = re.compile(r"\b(R1-6|R1-R6|R1/R6|R2/R5|R3/R4|R7|R8|R7-8|R7/R8)\b")
LMC_LABEL = re.compile(r"\b(LMC|L[1-5])\b")
T45_LABEL = re.compile(r"\b(T4[a-d]?|T5[a-d]?)\b")


def labels(book, seatable):
    """today's root -> plain label: PRs tab ('R7 of group 13', status), LMCs tab, T4T5s tab, SeaTable label"""
    lab = {}
    st = pd.read_csv(seatable, sep="\t", dtype=str).fillna("")
    for r in st.to_dict("records"):  # dict rows: itertuples renames the 'class' column (memory pitfall)
        path = " / ".join(r[k] for k in ("superclass", "class", "subclass", "subsubclass", "type") if r.get(k))
        if r.get("root_now"):
            lab.setdefault(r["root_now"], {})["seatable"] = path
    wb = openpyxl.load_workbook(book)
    from faab.pr import prs_tab as T
    ws = wb["PRs"]
    for b in T.blocks(ws):
        for row in b.rows:
            root = T.get(ws, b, row, "root_id_curr")
            if root:
                d = lab.setdefault(root, {})
                d.setdefault("pr", f"{T.get(ws, b, row, 'class')} of group {b.group}")
                d.setdefault("status", T.get(ws, b, row, "status"))
    for x in scan_book(wb):
        if x["tab"] in ("LMCs", "T4T5s"):
            v = {k: s(y) for k, y in x["values"].items()}
            root = v.get("root_id_curr", "")
            if root:
                d = lab.setdefault(root, {})
                d.setdefault("tab", x["tab"])
                if v.get("status"):
                    d.setdefault("status", v["status"])
    return lab


def describe(root, lab):
    d = lab.get(root, {})
    bits = [d["pr"]] if "pr" in d else []
    if d.get("seatable"):
        bits.append(f"SeaTable {d['seatable'].split(' / ')[-1]}")
    elif d.get("tab"):
        bits.append(f"{d['tab']} tab")
    if d.get("status"):
        bits.append(d["status"])
    return ", ".join(bits) or "no label"


def kind_of(root, lab):
    d = lab.get(root, {})
    text = " ".join([d.get("pr", ""), d.get("seatable", "")])
    if d.get("pr") or (PR_LABEL.search(d.get("seatable", "")) and "ol_sensory" in d.get("seatable", "")):
        return "PR"
    if LMC_LABEL.search(text) or d.get("tab") == "LMCs":
        return "LMC"
    if T45_LABEL.search(text) or d.get("tab") == "T4T5s":
        return "T4T5"
    return ""


def main():
    reg, book, seatable = sys.argv[1:4]
    R = pd.read_csv(reg, sep="\t", dtype=str).fillna("")
    lab = labels(book, seatable)
    client, t = cave_client(), 0.0
    now = datetime.datetime.now(datetime.timezone.utc)
    old = sorted({m for c in R.target_cells for m in re.findall(r"\d{18}", c)})
    t = throttle(t)
    latest = dict(zip(old, np.asarray(client.chunkedgraph.is_latest_roots([int(x) for x in old])).astype(bool)))
    desc = {}
    for o in old:
        if latest[o]:
            desc[o] = [o]
        else:
            t = throttle(t)
            desc[o] = [str(int(x)) for x in client.chunkedgraph.get_latest_roots(int(o))]
    print(f"{len(old)} target cells: {sum(latest.values())} unchanged, {len(old) - sum(latest.values())} edited since "
          f"2026-11-02", flush=True)
    order = {b: i for i, (b, _) in enumerate(PRIORITY)}
    why = dict(PRIORITY)
    out = []
    for x in sorted(R.itertuples(index=False), key=lambda x: order.get(x.box_id, 99)):
        lo = np.array([float(v) for v in x.lo_xyz.split(",")])
        hi = np.array([float(v) for v in x.hi_xyz.split(",")])
        t = throttle(t)
        a = client.materialize.live_query("synapses_v2", timestamp=now,
                                          filter_spatial_dict={"ctr_pt_position": [(lo * VOX).tolist(), (hi * VOX).tolist()]})
        t = time.time()
        a = a[a.pre_pt_root_id != a.post_pt_root_id]
        a["pre"], a["post"] = a.pre_pt_root_id.astype("int64").astype(str), a.post_pt_root_id.astype("int64").astype(str)
        n_out = a.pre.value_counts()
        n_in = a.post.value_counts()
        inputs_box = x.category.endswith("inputs")
        cells = {}  # root -> (role, count)
        for o in re.findall(r"\d{18}", x.target_cells):
            ds = desc[o]
            keep = [d for d in ds if n_out.get(d, 0) or n_in.get(d, 0)] or ds
            for d in keep:
                cells[d] = ("target" + ("" if d == o else f" (was {o})"), int((n_in if inputs_box else n_out).get(d, 0)))
                if d != o and d not in lab and o in lab:  # edited after the sheet's last update: label of the old id
                    lab[d] = dict(lab[o], status=lab[o].get("status", "") + "; root updated since the sheet update")
        for root, n in n_out.items():
            if root not in cells and kind_of(root, lab) in ("PR", "LMC"):
                cells[root] = ("terminal", int(n))
        if inputs_box:
            for root, n in n_in.items():
                if root not in cells and kind_of(root, lab) == "T4T5":
                    cells[root] = ("terminal", int(n))
        ranked = sorted(cells.items(), key=lambda kv: (not kv[1][0].startswith("target"), -kv[1][1]))
        colors = {r: OKABE[i % len(OKABE)] for i, (r, _) in enumerate(ranked)}
        pts = np.vstack(a.ctr_pt_position.values) / np.array(VOX) if len(a) else np.zeros((0, 3))
        box_name = f"cutout {x.box_id} ({x.category})"
        spec = {"neurons": [{"name": f"terminals in the box ({len(ranked)})", "colors": colors, "ids": list(colors)}],
                "annotations": [{"name": box_name, "color": BOX_COLOR.get(x.kind, "#F0E442"), "width": 4,
                                 "boxes": [[lo.tolist(), hi.tolist()]]},
                                {"name": f"predicted synapses in the box ({len(a)}; hidden: compare after annotating)",
                                 "color": "#56B4E9", "points": pts.round(1).tolist(), "visible": False}],
                "layout": "4panel", "position": [float(v) for v in x.center_xyz.split(",")], "scale": 2500,
                "cross_scale": 0.5, "image": True, "select": box_name}
        t = throttle(t)
        link = client.state.build_neuroglancer_url(client.state.upload_state_json(nglink.clean_state(nglink.build(spec))),
                                                   ngl_url="https://spelunker.cave-explorer.org")
        t = time.time()
        txt = "; ".join(f"{r} ({n} {'in' if inputs_box and kind_of(r, lab) == 'T4T5' else 'out'}, {describe(r, lab)}"
                        + (f", {role}" if role != 'target' and role != 'terminal' else "") + ")"
                        for r, (role, n) in ranked)
        out.append(dict(priority=order.get(x.box_id, 99) + 1, box_id=x.box_id, why=why.get(x.box_id, ""), kind=x.kind,
                        side=x.side, neuropil=x.neuropil, center_xyz=x.center_xyz, lo_xyz=x.lo_xyz, hi_xyz=x.hi_xyz,
                        n_synapses_now=len(a), n_synapses_20261102=x.n_all_synapses,
                        target_synapses_now=sum(n for r, (role, n) in ranked if role.startswith("target")),
                        target_synapses_20261102=x.target_synapses, terminals=txt, link=link, purpose=x.purpose))
        print(f"{order.get(x.box_id, 99) + 1:2d} {x.box_id:15s} {len(a):4d} synapses (was {x.n_all_synapses}); "
              f"{len(ranked)} terminals; {link}", flush=True)
    O = pd.DataFrame(out)
    stamp = utc_stamp()
    path = fig(f"{stamp}_cutouts_refreshed.tsv")
    O.to_csv(path, sep="\t", index=False)
    md = ["| # | box | why | synapses now (Nov 2) | terminals in the box (today's roots) | link |", "|---|---|---|---|---|---|"]
    md += [f"| {r.priority} | {r.box_id} | {r.why} | {r.n_synapses_now} ({r.n_synapses_20261102}) | {r.terminals} | "
           f"[open]({r.link}) |" for r in O.itertuples()]
    open(path.replace(".tsv", ".md"), "w").write("\n".join(md) + "\n")
    print(f"written {path} and .md")


if __name__ == "__main__":
    main()
