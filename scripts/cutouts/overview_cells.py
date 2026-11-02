"""Overview of proposed target cutouts with their cells in 3D (David 2026-11-01): the new boxes (magenta, thick) and the
existing cutouts (light blue), plus one layer per cell type of interest - the R7s, LMCs and T4/T5 cells that have
target synapses in a box (registry 'target_cells', propose_targets.py), each in its own colorblind-safe color, all
visible (toggle a layer to see one type). Also one overview per type (only that type's cells and boxes shown).
2026-11-02: several registries (the lamina R1-R6 / R8 boxes came in a second one) and an R1-R6/R8 layer; the new
boxes in one layer per purpose: prediction checks magenta, R7 missed-synapse boxes vermillion (cold) / white (typical).
State uploads only (a view; >= 2 s apart). Usage: overview_cells.py REGISTRY.tsv [REGISTRY2.tsv ...]"""
import json
import re
import sys
import time

import pandas as pd

from faab import nglink
from faab.clients.cave import client as cave_client, throttle
from faab.config import fig

NEW_COLOR = "#ff00ff"
TYPES = [("R1-R6/R8", "R1-R6/R8 terminals", "#F0E442"), ("R7", "R7 terminals", "#0072B2"),
         ("LMC", "LMC terminals", "#E69F00"), ("T4/T5", "T4/T5 inputs", "#009E73")]

R = pd.concat([pd.read_csv(p, sep="\t") for p in sys.argv[1:]], ignore_index=True)
inv = pd.read_csv(fig("cutout_inventory.tsv"), sep="\t")
existing = [[json.loads(a), json.loads(b)] for a, b in zip(inv.lo, inv.hi)]
kind = lambda c: ("R1-R6/R8" if c.startswith(("R1-R6", "R8")) else "R7" if c.startswith("R7")
                  else "T4/T5" if c.startswith(("T4", "T5")) else "LMC")
R["type"] = R.category.map(kind)
box = lambda x: [[float(v) for v in x.lo_xyz.split(",")], [float(v) for v in x.hi_xyz.split(",")]]
cells = {t: sorted({c for s in R[R.type == t].target_cells for c in re.findall(r"\b(\d{18})\b", s)}) for t, _, _ in TYPES}


PURPOSES = [("check", "prediction checks (placed where the most synapses are predicted)", NEW_COLOR),
            ("cold", "R7 missed synapses: cold boxes (much R7, few predicted synapses)", "#D55E00"),
            ("typical", "R7 missed synapses: typical boxes (random R7 points)", "#FFFFFF")]
R["purpose_kind"] = R.category.map(lambda c: "cold" if "cold" in c else "typical" if "typical" in c else "check")


def spec(types, title):
    neurons = [{"name": f"{name}: cells in the new boxes ({len(cells[t])})", "color": col, "ids": cells[t]}
               for t, name, col in TYPES if t in types and cells[t]]
    sel = R[R.type.isin(types)]
    ann = [{"name": f"new cutouts, {label} - {title}", "color": col, "width": 4,
            "boxes": [box(x) for x in sel[sel.purpose_kind == k].itertuples(index=False)]}
           for k, label, col in PURPOSES if (sel.purpose_kind == k).any()]
    ann.append({"name": "existing cutouts", "color": "#56B4E9", "boxes": existing})
    return {"meshes": [{"kind": "OL", "segments": ["11", "14", "21", "24"], "alpha": 0.08}], "neurons": neurons,
            "annotations": ann, "layout": "3d", "select": neurons[0]["name"]}


client, t, out = cave_client(), 0.0, {}
for key, types in [("all", [t for t, _, _ in TYPES])] + [(t, [t]) for t, _, _ in TYPES]:
    t = throttle(t)
    state = nglink.clean_state(nglink.build(spec(types, "all target types" if key == "all" else key)))
    out[key] = client.state.build_neuroglancer_url(client.state.upload_state_json(state),
                                                   ngl_url="https://spelunker.cave-explorer.org")
    t = time.time()
    print(f"{key}: {out[key]}", flush=True)
for t_, _, _ in TYPES:
    print(f"{t_} cells: {' '.join(cells[t_])}")
open(sys.argv[-1].replace(".tsv", "_overview_cells_links.txt"), "w").write(
    "\n".join(f"{k}\t{v}" for k, v in out.items()) + "\n")
