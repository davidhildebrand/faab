"""Showcase Spelunker links for talks and meetings (David, 2026-11-10: "Generate several other 'fun' and/or pretty links
... e.g. LMCs, the bridge between PRs and MeTus, etc. I want to have some on hand to show tomorrow in a meeting").

Each link is built with faab.nglink (README scene, dark 3D background, fainter neuropils, EM layer present), plain layer
names and Okabe-Ito colors, from tables the project already has; ids are brought up to date first (one batched
is_latest_roots call, then one suggest_latest_roots call per edited id, throttled; points -> roots where a table has
points). Links:
  eye_to_motion   PR group 1 -> PR+1 -> PR+2 -> PR+3 (T4/T5 candidates), from a PRg1toT4T5 full workbook
  lmc_types       the typed lamina monopolar cells of the right eye, L1 / L2 / L3 / L4 / L5
  cartridge       one lamina cartridge (right cartridge 7: L1-L5) with the photoreceptors that feed it (synapse snapshot)
  pr_metu_bridge  photoreceptors -> bridge cells -> MeTu cells (figures/metu/pr_metu_*.tsv, strongest bridges)
  metu_clusters   MeTu cells by morphology cluster (figures/metu/faab_metu_tubu_clusters.csv)
  pr_to_dnp01     shortest photoreceptor -> DNp01 (giant fiber) paths (figures/paths/*_paths_pr_to_DNp01_sourcesR_spec.json)
  vpn_clusters    visual projection neuron clusters converging on two descending neurons (figures/paths/*_vpn_clusters_spec.json)
  mi1_array       the Mi1 cells tiling the right medulla (figures/mi1/mi1_ids.txt)
  ct1             the CT1 and the CT1-like LO ME cell, the two wide-field optic-lobe cells
Writes figures/showcase/<stamp>_showcase_links.md (title, what it shows, short link) and one _ng.txt (full URL) per
link. CAVE read-only and throttled. Usage: showcase_links.py --t4t5-book FULL.xlsx [--only NAME ...]"""
import argparse
import glob
import json
import os
import re
import time

import numpy as np
import openpyxl
import pandas as pd

from faab import nglink, synsnap
from faab.clients.cave import client as cave_client, roots_at_xyz, throttle
from faab.config import FIGURES, fig
from faab.partners.sheet import cell_text as s, scan_book, utc_stamp

OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
      "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#BBBBBB"}
OL = [{"kind": "OL", "segments": ["21", "22", "23", "24"]}, {"kind": "neuropil"}]
OL_CENTER = [7182.0, 17639.0, 4180.0]  # right optic lobe (median of the T4T5-2 points, 2026-11-10), voxels
LEFT_ME = [38300.0, 19000.0, 2900.0]  # left medulla (median output synapse of the Mi1 cells, 2026-11-11)
_t = [0.0]


def wait():
    _t[0] = throttle(_t[0])


class Ids:
    """current root for any id (batched is_latest_roots, then suggest_latest_roots per edited id); cached per run"""

    def __init__(self, c):
        self.c, self.map = c, {}

    def __call__(self, ids):
        ids = [str(i) for i in ids if re.fullmatch(r"\d{15,}", str(i))]
        todo = sorted(set(ids) - set(self.map))
        for k in range(0, len(todo), 500):
            part = todo[k:k + 500]
            wait()
            ok = np.asarray(self.c.chunkedgraph.is_latest_roots([int(i) for i in part])).astype(bool)
            for i, g in zip(part, ok):
                if g:
                    self.map[i] = i
            for i in [i for i, g in zip(part, ok) if not g]:
                wait()
                try:
                    self.map[i] = str(int(self.c.chunkedgraph.suggest_latest_roots(int(i))))
                except Exception as e:
                    print(f"  {i}: no current root ({str(e)[:60]})", flush=True)
                    self.map[i] = None
        out = list(dict.fromkeys(self.map[i] for i in ids if self.map.get(i)))
        return out

    def stats(self):
        return sum(k != v for k, v in self.map.items()), len(self.map)


def center(points):
    p = [[float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", s(x))] for x in points]
    p = np.array([q for q in p if len(q) == 3])
    return [float(v) for v in np.median(p, axis=0)] if len(p) else None


def spec_of(layers, meshes=OL, position=None, scale=None, select=None, layout="3d"):
    sp = {"meshes": meshes, "neurons": [L for L in layers if L["ids"]], "view": "anterior", "layout": layout,
          "image": True, "select": select or layers[0]["name"]}
    if position:
        sp.update(position=position, scale=scale or 14000)
    return sp


# ------------------------------------------------------------------ the links
def eye_to_motion(a, cur):
    import importlib.util
    sp = importlib.util.spec_from_file_location("hop_in2_block", os.path.join(os.path.dirname(__file__), "..",
                                                                              "partners", "hop_in2_block.py"))
    H = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(H)
    wb = openpyxl.load_workbook(a.t4t5_book)
    ws = wb["PRg1toT4T5"]
    blocks, _ = H.blocks_of(ws, ws.max_column)
    kind = {r: b["kind"] for b in blocks for r in range(b["first"], b["last"] + 1)}
    rows = [x for x in scan_book(wb) if x["tab"] == "PRg1toT4T5"]
    by = lambda k: [x for x in rows if kind.get(x["row"]) == k]
    prs = cur([i.strip() for x in by("hop1") for i in s(x["values"].get("PR_root_id")).split(",")])
    st = {k: cur([s(x["values"].get("root_id_curr")) for x in by(k)]) for k in ("hop1", "hop2", "hop3")}
    layers = [{"name": f"photoreceptors, PR group 1 ({len(prs)})", "color": OI["yellow"], "ids": prs},
              {"name": f"PR+1: lamina cells ({len(st['hop1'])})", "color": OI["sky"], "ids": st["hop1"]},
              {"name": f"PR+2: medulla cells ({len(st['hop2'])})", "color": OI["green"], "ids": st["hop2"]},
              {"name": f"PR+3: T4/T5 candidates ({len(st['hop3'])})", "color": OI["vermillion"], "ids": st["hop3"]}]
    pos = center([s(x["values"].get("xyz")) for k in ("hop1", "hop2", "hop3") for x in by(k)])
    return ("From the eye to motion detectors",
            "Every cell on a three-connection path from one ommatidium's photoreceptors (PR group 1) to T4/T5 motion-"
            "detector candidates, >= 3 synapses per connection.", spec_of(layers, position=pos, scale=12000))


def lmc_types(a, cur):
    t = pd.read_csv(fig(os.path.join("lmc", "lmc_typing.tsv")), sep="\t", dtype=str).fillna("")
    t = t[t.side == "R"]
    wait()
    roots = [str(r) for r in roots_at_xyz(list(t.point_xyz))]
    t = t.assign(now=roots)
    cols = {"L1": OI["orange"], "L2": OI["sky"], "L3": OI["green"], "L4": OI["yellow"], "L5": OI["purple"]}
    layers = [{"name": k, "color": c, "ids": list(dict.fromkeys(r for r in t.now[t.consensus == k] if r != "0"))}
              for k, c in cols.items()]
    for L in layers:
        L["name"] += f" ({len(L['ids'])})"
    return ("Lamina monopolar cells, L1 to L5",
            f"The {len(t)} typed LMCs of the right eye, colored by type (automatic typing from lamina / medulla depth "
            f"and synapse features).", spec_of(layers, position=center(t.point_xyz), scale=11000))


def cartridge(a, cur):
    t = pd.read_csv(fig(os.path.join("lmc", "lmc_cartridge_calls.tsv")), sep="\t", dtype=str)
    pts = pd.read_csv(fig(os.path.join("lmc", "lmc_typing.tsv")), sep="\t", dtype=str).set_index("root").point_xyz
    g = t[(t.side == "R") & (t.cartridge == "7")]
    wait()
    roots = dict(zip(g.assigned, (str(r) for r in roots_at_xyz([pts[r] for r in g.root]))))
    # photoreceptors feeding these LMCs: the PRs tab's cells and SeaTable's ol_sensory cells, >= 5 synapses
    wb = openpyxl.load_workbook(a.t4t5_book)
    wait()
    known = {str(r) for r in roots_at_xyz([s(x["values"]["xyz"]) for x in scan_book(wb) if x["tab"] == "PRs"])}
    st = pd.read_csv(sorted(glob.glob(fig("*_seatable_aedes_main.tsv")))[-1], sep="\t", dtype=str).fillna("")
    known |= set(st.root_now[st.superclass == "ol_sensory"])
    wait()
    e = synsnap.Snapshot().edges(post=[int(r) for r in roots.values()], when="now", client=cave_client(), min_syn=5)
    e = e[e.pre.astype(str).isin(known)]
    prs = sorted(set(e.pre.astype(str)))
    cols = {"L1": OI["orange"], "L2": OI["sky"], "L3": OI["green"], "L4": OI["yellow"], "L5": OI["purple"]}
    layers = [{"name": f"photoreceptors feeding it ({len(prs)})", "color": OI["grey"], "ids": prs}]
    layers += [{"name": k, "color": c, "ids": [roots[k]]} for k, c in cols.items() if k in roots]
    return ("One lamina cartridge",
            f"Right-eye cartridge 7: one each of L1, L2, L3, L4, and L5 (typing confidence 0.67-0.97) and the "
            f"{len(prs)} photoreceptors that synapse onto them (>= 5 synapses).",
            spec_of(layers, position=center([pts[r] for r in g.root]), scale=6000, select="L1"))


def pr_metu_bridge(a, cur):
    import importlib.util
    sp = importlib.util.spec_from_file_location("bl", os.path.join(os.path.dirname(__file__), "..", "metu",
                                                                   "pr_metu_bridge_link.py"))
    BL = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(BL)
    d = os.path.join(FIGURES, "metu")
    bridge = pd.read_csv(os.path.join(d, "pr_metu_bridge.tsv"), sep="\t", dtype={"root": str})
    down = pd.read_csv(os.path.join(d, "pr_metu_down_synapses.tsv"), sep="\t", dtype={"pre_pt_root_id": str, "post_pt_root_id": str})
    up = pd.read_csv(os.path.join(d, "pr_metu_up_synapses.tsv"), sep="\t", dtype={"pre_pt_root_id": str, "post_pt_root_id": str})
    cells = pd.read_csv(os.path.join(d, "pr_metu_link_cells.tsv"), sep="\t", dtype={"root": str})
    strong = bridge[bridge.path_strength >= BL.MIN_STRENGTH]
    prs = set(down[down.post_pt_root_id.isin(set(strong.root))].pre_pt_root_id)
    metus = set(up[up.pre_pt_root_id.isin(set(strong.root))].post_pt_root_id)
    cls = dict(zip(cells.root, cells.cls.fillna("")))
    layers = [{"name": "photoreceptors R7/R8", "color": OI["purple"], "ids": cur(sorted(p for p in prs if cls.get(p) == "R7/R8"))},
              {"name": "photoreceptors R1-R6", "color": OI["yellow"], "ids": cur(sorted(p for p in prs if cls.get(p) != "R7/R8"))},
              {"name": f"bridge cells ({len(strong)})", "color": OI["orange"], "ids": cur(sorted(strong.root))},
              {"name": f"MeTu cells ({len(metus)})", "color": OI["sky"], "ids": cur(sorted(metus))}]
    for L in layers[:2]:
        L["name"] += f" ({len(L['ids'])})"
    return ("Photoreceptors to MeTu cells",
            f"The strongest two-connection bridges from photoreceptors to MeTu cells (medulla to anterior optic "
            f"tubercle), bridge strength >= {BL.MIN_STRENGTH} synapses.",
            spec_of(layers, select=layers[2]["name"], position=OL_CENTER, scale=16000))


def metu_clusters(a, cur):
    t = pd.read_csv(fig(os.path.join("metu", "faab_metu_tubu_clusters.csv")), dtype=str)
    cols = [OI["orange"], OI["sky"], OI["green"], OI["purple"]]
    ks = sorted(t.metu_tubu_cluster_k4.unique(), key=int)
    layers = [{"name": f"MeTu cluster {k} ({(t.metu_tubu_cluster_k4 == k).sum()})", "color": cols[i % 4],
               "ids": cur(list(t.root_id[t.metu_tubu_cluster_k4 == k]))} for i, k in enumerate(ks)]
    return ("MeTu cells by cluster",
            f"{len(t)} MeTu cells (medulla to anterior optic tubercle) in four morphology clusters.",
            spec_of(layers, position=OL_CENTER, scale=16000, select=layers[1]["name"]))


def from_spec(path_glob, names, colors, title, text, cur):
    path = sorted(glob.glob(path_glob))[-1]
    sp = json.load(open(path))
    layers = []
    for n, nm, col in zip(sp["neurons"], names, colors):
        ids = cur(n.get("ids") or [ln.strip() for ln in open(n["ids_file"]) if ln.strip()])
        layers.append({"name": f"{nm} ({len(ids)})", "color": col, "ids": ids})
    return title, text, spec_of(layers, meshes=[{"kind": "neuropil"}], select=layers[-1]["name"])


def pr_to_dnp01(a, cur):
    return from_spec(fig(os.path.join("paths", "*_paths_pr_to_DNp01_sourcesR_spec.json")),
                     ["photoreceptors", "PR+1", "PR+2", "PR+3", "DNp01 (giant fiber)"],
                     [OI["yellow"], OI["sky"], OI["green"], OI["orange"], OI["vermillion"]],
                     "Photoreceptors to the giant fiber",
                     "All shortest paths (three connections, >= 3 synapses each) from right-eye photoreceptors to the "
                     "giant-fiber descending neurons DNp01.", cur)


def vpn_clusters(a, cur):
    return from_spec(fig(os.path.join("paths", "*_vpn_clusters_spec.json")),
                     ["VPN cluster 13", "VPN cluster 15", "VPN cluster 20", "VPN cluster 24", "descending neurons"],
                     [OI["orange"], OI["sky"], OI["green"], OI["purple"], OI["vermillion"]],
                     "Visual projection neurons onto descending neurons",
                     "Four small clusters of visual projection neurons that converge on two large descending neurons.", cur)


def mi1_array(a, cur):
    ids = cur([ln.strip() for ln in open(fig(os.path.join("mi1", "mi1_ids.txt"))) if ln.strip()])
    layers = [{"name": f"Mi1 cells ({len(ids)})", "color": OI["sky"], "ids": ids}]
    return ("The Mi1 array",
            f"{len(ids)} Mi1 cells tiling the LEFT medulla, one per column.",
            spec_of(layers, meshes=[{"kind": "OL", "segments": ["11", "12", "13", "14"]}], position=LEFT_ME,
                    scale=12000))


def ct1(a, cur):
    layers = [{"name": "CT1", "color": OI["orange"], "ids": cur(["648518347494887381"])},
              {"name": "LO ME cell (CT1-like)", "color": OI["sky"], "ids": cur(["648518347531730229"])}]
    return ("Two wide-field optic-lobe cells",
            "The CT1 (648518347494887381) and the CT1-like LO ME cell (648518347531730229): each spreads over the "
            "whole medulla and lobula, with about 35 mm of cable.", spec_of(layers, scale=14000, position=OL_CENTER))


LINKS = {"eye_to_motion": eye_to_motion, "lmc_types": lmc_types, "cartridge": cartridge,
         "pr_metu_bridge": pr_metu_bridge, "metu_clusters": metu_clusters, "pr_to_dnp01": pr_to_dnp01,
         "vpn_clusters": vpn_clusters, "mi1_array": mi1_array, "ct1": ct1}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t4t5-book", required=True, help="a workbook with the PRg1toT4T5 (PR+N names) and PRs tabs")
    ap.add_argument("--only", nargs="+", choices=list(LINKS))
    a = ap.parse_args()
    t0, stamp = time.time(), utc_stamp()
    c = cave_client()
    cur = Ids(c)
    out_dir = os.path.join(FIGURES, "showcase")
    os.makedirs(out_dir, exist_ok=True)
    md = [f"# Showcase links ({stamp})", "",
          "Spelunker links for talks: dark background, plain layer names, colorblind-safe colors. Hidden or extra "
          "layers can be switched on in the layer bar.", ""]
    for name in a.only or LINKS:
        try:
            title, text, spec = LINKS[name](a, cur)
        except Exception as e:
            print(f"{name}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        state = nglink.build(spec)
        url_full = nglink.to_url(state)
        open(os.path.join(out_dir, f"{stamp}_{name}_ng.txt"), "w").write(url_full + "\n")
        wait()
        sid = c.state.upload_state_json(state)
        url = c.state.build_neuroglancer_url(sid, ngl_url="https://spelunker.cave-explorer.org")
        n = {L["name"]: len(L["ids"]) for L in spec["neurons"]}
        print(f"{name}: {title}; layers {n}; {url}", flush=True)
        md += [f"## {title}", "", text, "", f"Layers: {', '.join(n)}.", "", url, ""]
    edited, total = cur.stats()
    md_path = os.path.join(out_dir, f"{stamp}_showcase_links.md")
    open(md_path, "w").write("\n".join(md))
    print(f"ids: {total} checked, {edited} brought up to date; wrote {md_path}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
