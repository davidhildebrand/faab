"""One Spelunker link per tab row for David to look at (2026-10-31): the rows whose notes call the cell a fragment while
the cell is still small (< MAX_L2 L2 chunks) and the status is not 'stopped', plus every other row of those cells, plus
fragment rows now part of a cell other rows track (PRg1hop2 r162). Each link opens on the row's xyz (EM + the cell,
xy + 3d panels, zoomed in). Short links via the state server (one upload per row, >= 5 s apart). Reads the
cell_consistency.py TSV; writes figures/sheets/<stamp>_fragment_row_links.tsv and prints a markdown table.
Usage: fragment_row_links.py CONSISTENCY.tsv"""
import re
import sys

import pandas as pd

from faab import nglink
from faab.clients.cave import client as cave_client, parse_xyz, throttle
from faab.config import fig
from faab.partners.sheet import utc_stamp

MAX_L2 = 50
EXTRA = [("PRg1hop2", "162")]  # merged into a tracked cell
D = pd.read_csv(sys.argv[1], sep="\t", dtype=str).fillna("")
D["l2"] = pd.to_numeric(D.l2_chunks, errors="coerce")
small = {c for c, g in D.groupby("cell") if (g.frag_note == "True").any() and (g.l2 < MAX_L2).all()
         and (g.status.str.lower() != "stopped").any()}  # by cell: a stopped orphan's NEW twin row counts too
sel = D[D.cell.isin(small) | D.apply(lambda x: (x.tab, x.row) in EXTRA, axis=1)].copy()
sel = sel.sort_values(["tab", "row"], key=lambda c: c.astype(int) if c.name == "row" else c)
c = cave_client()
t = 0.0
links = []
for x in sel.to_dict("records"):
    name = f"{x['tab']} r{x['row']}: {x['cell']}"
    st = nglink.build({"neurons": [{"name": name, "color": "#ff7f0e", "ids": [x["cell"]]}], "layout": "xy-3d",
                       "position": parse_xyz(x["xyz"]), "scale": 2500, "image": True, "select": name})
    st["crossSectionScale"] = 1.0
    t = throttle(t)
    links.append(c.state.build_neuroglancer_url(c.state.upload_state_json(st), ngl_url="https://spelunker.cave-explorer.org"))
sel["link"] = links
out = fig(f"{utc_stamp()}_cell_consistency_fragment_links.tsv")
sel[["tab", "row", "cell", "l2_chunks", "status", "proofreader", "notes", "xyz", "link"]].to_csv(out, sep="\t", index=False)
print("| tab | row | status | L2 chunks | notes | link |\n|---|---|---|---|---|---|")
for x in sel.to_dict("records"):
    note = re.sub(r"\s+", " ", x["notes"])[:60].replace("|", "/")
    print(f"| {x['tab']} | {x['row']} | {x['status']} | {x['l2_chunks']} | {note} | {x['link']} |")
print(f"\nwritten {out}")
