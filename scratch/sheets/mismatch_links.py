"""Spelunker links for the 2026-10-30 off-cell sheet points (one per issue, for messages to the proofreaders): the
intended cell (green), the speck the sheet point lies on (red), extra pieces (blue), the sheet point (red dot) and a
verified point on the cell (green dot); opens on the sheet point with the EM slice. CAVE read-only: one state upload
per link, >= 5 s apart. Usage: mismatch_links.py"""
from faab import nglink
from faab.clients.cave import client as cave_client, throttle

CASES = [
    ("Kristo: PRg1hop2 rows 35/160/286 (463102525)", "648518347463102525", "648518347187180486",
     ["648518347463231049"], [7418, 18627, 4397], [7461, 18632, 4385]),
    ("Kristo: PRg1hop2 rows 20/33, PRg1toT4T5 row 59 (605671615)", "648518347605671615", "648518347411010187",
     [], [9562, 17822, 4958], [9524, 17785, 4952]),
    ("Kristo: PRg1toT4T5 row 58 (546546302)", "648518347546546302", "648518346766051589",
     [], [9048, 17693, 5083], [8915, 17654, 5072]),
    ("Zohar: PRg1hop2 row 210 (509939307)", "648518347509939307", "648518347528338916",
     [], [10697, 20234, 5090], [10645, 20292, 5057]),
]
c = cave_client()
t = 0.0
for title, cell, speck, extra, bad, good in CASES:
    neurons = [{"name": f"cell {cell[-9:]}", "color": "#2ecc40", "ids": [cell]},
               {"name": f"speck at the sheet point {speck[-9:]}", "color": "#e8202a", "ids": [speck]}]
    if extra:
        neurons.append({"name": "extension (separate root)", "color": "#2a7fd4", "ids": extra})
    spec = {"neurons": neurons, "image": True, "layout": "xy-3d", "position": bad, "scale": 3000,
            "annotations": [{"name": "sheet xyz (off the cell)", "color": "#e8202a", "points": [bad]},
                            {"name": "point on the cell", "color": "#2ecc40", "points": [good]}],
            "select": neurons[0]["name"]}
    t = throttle(t)
    sid = c.state.upload_state_json(nglink.build(spec))
    print(f"{title}: {c.state.build_neuroglancer_url(sid, ngl_url='https://spelunker.cave-explorer.org')}")
