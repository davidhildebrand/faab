"""Spelunker review links (README scene base) for PRs-tab groups in an ommatidia.py run (2026-11-01): per group one
layer per version 2 ommatidium holding any of the group's axons ("automatic bundle N", all its members, slots in the
layer name), a layer of the group's axons in none of them, and the group as listed in the tab (hidden, to toggle
against). Colors colorblind-safe; texts and layer names in plain words for the proofreaders (no v1/v2, no om ids).
With --upload every state goes to the CAVE state server (a view only; >= 2 s apart) and figures/ommatidia/<RUN>/
review_links.tsv gets group, text (the agreement, for the tab) and the short link - the input of
prs_tab.py group-links. Without it the long links are only written to figures/ommatidia/<RUN>/links/; with 'all' and
an existing review_links.tsv its texts are rewritten and its short links kept (the view itself is unchanged).
Usage: ommatidia_link.py RUN STAMP G [G ...] | all  [--upload]"""
import os
import sys

import numpy as np
import pandas as pd

from faab import nglink
from faab.config import fig

args = [x for x in sys.argv[1:] if x != "--upload"]
upload = "--upload" in sys.argv
run, stamp, which = args[0], args[1], args[2:]
d = fig(f"ommatidia/{run}")
os.makedirs(os.path.join(d, "links"), exist_ok=True)
Om = pd.read_csv(os.path.join(d, "ommatidia.tsv"), sep="\t", dtype={"root": str}).fillna("")
Cmp = pd.read_csv(os.path.join(d, "compare.tsv"), sep="\t").fillna("")
tab = pd.read_csv(os.path.join(d, "tab_roots.tsv"), sep="\t", dtype=str).fillna("")
groups = Cmp.group.tolist() if which == ["all"] else [int(g) for g in which]
# colorblind-safe (Okabe-Ito; David 2026-11-01: no red vs green); yellow is kept for the listed group
COL = ["#0072B2", "#E69F00", "#CC79A7", "#56B4E9", "#D55E00", "#009E73"]
if upload:
    from faab.clients.cave import client as cave_client, throttle
    client, t = cave_client(), 0.0
lp = os.path.join(d, "review_links.tsv")
old_url = {}
if os.path.exists(lp):
    old = pd.read_csv(lp, sep="\t").fillna("")
    old_url = dict(zip(old.group.astype(int), old.url))
rows = []
for g in groups:
    mine = tab[tab.group.astype(int) == g]
    x = Cmp[Cmp.group == g].iloc[0]
    oms = [o for o in [x.ommatidium] + str(x.other_om).split(", ") if o]
    neurons, placed = [], set()
    for k, om in enumerate(oms):
        m = Om[Om.ommatidium == om]
        placed |= set(m.root)
        slots = ", ".join(f"{(m.slot == s).sum()} {s}" for s in ("R1-R6", "R7", "R8"))
        neurons.append({"name": f"automatic bundle {k + 1} ({slots})", "color": COL[k % len(COL)], "ids": m.root.tolist()})
    ids = sorted(set(mine.root_now) - {"", "0"})
    rest = sorted(set(ids) - placed)
    if rest:
        neurons.append({"name": f"group {g} axons in no automatic bundle ({len(rest)})", "color": "#ffffff", "ids": rest})
    neurons.append({"name": f"group {g} as listed in the PRs tab ({len(ids)})", "color": "#F0E442", "ids": ids,
                    "visible": False})
    xyz = np.array([[float(v) for v in s.split(",")] for s in mine.xyz if s.count(",") == 2])
    spec = {"meshes": [{"kind": "OL", "segments": ["11", "14", "21", "24"], "alpha": 0.12}], "neurons": neurons,
            "layout": "xy-3d", "select": neurons[0]["name"], "scale": 9000}
    if len(xyz):
        spec["position"] = np.median(xyz, axis=0).tolist()
    state = nglink.clean_state(nglink.build(spec))
    path = os.path.join(d, "links", f"{stamp}_g{g}_ng.txt")
    open(path, "w").write(nglink.to_url(state) + "\n")
    # plain text for the proofreaders (David 2026-11-01: no 'v1 vs v2', no internal ids)
    more = int(x.n_om) - int(x.kept) if x.ommatidium else 0
    lost = [w for w in str(x.lost).split("; ") if w]
    elsewhere = sum(1 for w in lost if "(in om" in w)
    unplaced = len(lost) - elsewhere
    plus = f"{more} unlisted axon{'s' * (more != 1)}"
    if not x.ommatidium:
        text = "Automatic grouping check: not available here"
    elif x.jaccard == 1:
        text = "Automatic grouping check: agrees"
    elif x.kept == x.n_tab:
        text = f"Automatic grouping check: agrees, plus {plus}"
    else:
        other = "1 in another bundle" if elsewhere == 1 else f"{elsewhere} in other bundles"
        bits = [f"keeps {x.kept} of {x.n_tab} together"] + ([other] if elsewhere else []) + \
               ([f"{unplaced} unplaced"] if unplaced else []) + ([f"adds {plus}"] if more else [])
        text = "Automatic grouping check: " + "; ".join(bits)
    url = old_url.get(g, "")
    if upload:
        t = throttle(t)
        url = client.state.build_neuroglancer_url(client.state.upload_state_json(state),
                                                  ngl_url="https://spelunker.cave-explorer.org")
    rows.append(dict(group=g, text=text, url=url))
    print(f"group {g}: {text}" + (f" -> {url}" if url else ""), flush=True)
if upload or (which == ["all"] and old_url):  # without --upload: relabel, keeping the uploaded links
    pd.DataFrame(rows).to_csv(lp, sep="\t", index=False)
    print("written", lp)
