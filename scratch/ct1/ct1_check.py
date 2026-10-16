"""For the putative aedes CT1-like net cell 648518347700703374:
 - build a Spelunker URL
 - (male-CNS) what fly CT1 connects to (top partner types)
 - (aedes) is the cell a huge wide-field cell? (extent) + what 'LOME' means
Read-only.
"""
from faab.config import REPO
import os
import json
import time
import urllib.parse
import numpy as np
import pandas as pd
import caveclient
import seaserpent as ss
from neuprint import NeuronCriteria as NC, fetch_neurons, fetch_simple_connections
from faab.sources.neuprint import client as npclient, MALE_CNS
from faab.sources.cave import throttle, DATASTACK

OUT = os.path.join(REPO, "figures")
SERVER = "https://flytable.mrc-lmb.cam.ac.uk/"
CELL = 648518347700703374


def to_int64(s):
    return s.map(lambda x: int(str(x)) if str(x).strip().lstrip("-").isdigit() else pd.NA).astype("Int64")


def ng_link(segids):
    img = "precomputed://gs://zetta_lee_mosquito_001_alignment_central/img/v2_sharded"
    seg = "graphene://middleauth+https://cave.fanc-fly.com/segmentation/table/wclee_aedes_brain"
    state = {"layers": [{"type": "image", "source": img, "name": "img"},
                        {"type": "segmentation", "source": seg,
                         "segments": [str(s) for s in segids], "name": "CT1-like"}],
             "layout": "xy-3d"}
    return "https://spelunker.cave-explorer.org/#!" + urllib.parse.quote(json.dumps(state, separators=(",", ":")))


def main():
    print("[1] Spelunker URL:")
    print(ng_link([CELL]))

    # [3] fly CT1 partners (male-CNS)
    c = npclient(MALE_CNS)
    n = fetch_neurons(NC(type="CT1"), client=c)[0]
    print(f"\n[3] (male-CNS) CT1 neurons: {len(n)}")
    if len(n):
        dn = fetch_simple_connections(NC(type="CT1"), None, min_weight=5, client=c)
        upc = fetch_simple_connections(None, NC(type="CT1"), min_weight=5, client=c)
        print("  CT1 -> top downstream partner types:")
        print(dn.groupby("type_post")["weight"].sum().sort_values(ascending=False).head(10).to_string())
        print("  top upstream partner types -> CT1:")
        print(upc.groupby("type_pre")["weight"].sum().sort_values(ascending=False).head(10).to_string())

    # [4] aedes LOME annotation
    meta = ss.Table("aedes_main", server=SERVER).to_frame()
    meta.columns = [str(c2) for c2 in meta.columns]
    lome = meta[meta["class"].astype(str) == "LOME"]
    print(f"\n[4] (aedes) 'LOME' class: {len(lome)} cells | annotators: {lome['annotator'].value_counts().to_dict()}")
    print("    subclass/type/neuropils:",
          lome[["subclass", "type", "neuropils"]].astype(str).agg(" | ".join, axis=1).tolist())

    # [2] is the net cell a huge wide-field cell? (extent vs a typical Mi1 ~100um)
    from fafbseg import flywire
    t = time.time()
    dp = flywire.get_l2_dotprops([CELL], dataset=DATASTACK, max_threads=1, omit_failures=True, progress=False) / 1000.0
    if len(dp):
        co = dp[0].points
        span = (co.max(0) - co.min(0))
        print(f"\n[2] (aedes) net cell {CELL}: {len(co)} L2 points, bbox span um = {span.round(0)}, "
              f"diag = {np.linalg.norm(span):.0f}um  (Mi1 ~100um)")
        print("    -> large wide-field extent is the CT1 signature (connectivity already diagnostic)")


if __name__ == "__main__":
    main()
