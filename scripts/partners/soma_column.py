"""soma_check column for the partner table (every row, not just "no soma?" rows).

Per distinct root_id_curr: local-volume test on the cached L2 chunks (figures/l2_leaves.pkl +
l2_chunkdata.pkl; run faab_glia_l2_fetch.py first so new roots are cached), ONE live nuclei lookup for
the non-expired roots, then manual verdicts from figures/soma_overrides.tsv (root_id_curr, verdict, note,
by, date) take precedence. Photoreceptor rows get "n/a (photoreceptor)": no soma expected in the volume.
Values: "confirmed at x, y, z (by, date)" | "not a soma at x, y, z (by, date)" | "nucleus at x, y, z" |
"candidate at x, y, z (V um3)" | "none found" | "fragment (n chunks)" | "expired root" | "not cached".
Writes the column into figures/faab_pr_top20_update.tsv; run before faab_pr_top20_render.py. Read-only.
"""
from faab.config import REPO
import datetime
import json
import os
import pickle
import re
import time

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import caveclient
from faab.clients.cave import DATASTACK
from faab.partners.top10_xyz import VOX
from faab.partners.no_soma_batch import local_volume, retry, vox, SOMA_UM3
from faab.config import fig

HERE = REPO
OUT = os.path.join(HERE, "figures")
TSV = fig("faab_pr_top20_update.tsv")
META = fig("faab_pr_top20_update_meta.json")
OVR = fig("soma_overrides.tsv")
PR_PAT = re.compile(r"\bR[1-8]\b|photoreceptor", re.I)


def main():
    df = pd.read_csv(TSV, sep="\t", dtype=str).fillna("")
    pr_roots = set(json.load(open(META))["pr_roots"].values())
    ovr = pd.read_csv(OVR, sep="\t", dtype=str).fillna("") if os.path.exists(OVR) else pd.DataFrame()
    ovr_by = {r.root_id_curr: r for r in ovr.itertuples()} if len(ovr) else {}
    leaves = pickle.load(open(fig("l2_leaves.pkl"), "rb"))
    chunks = pickle.load(open(fig("l2_chunkdata.pkl"), "rb"))

    roots = sorted({int(r) for r in df.root_id_curr if r.isdigit()} - {int(r) for r in pr_roots})
    client = caveclient.CAVEclient(DATASTACK)
    now = datetime.datetime.now(datetime.timezone.utc)
    latest = retry(lambda: client.chunkedgraph.is_latest_roots(roots, timestamp=now), "is_latest_roots")
    expired = set() if latest is None else {r for r, ok in zip(roots, latest) if not ok}
    live_roots = [r for r in roots if r not in expired]
    nuc = retry(lambda: client.materialize.live_query("nuclei_v1_aedes", timestamp=now,
                                                      filter_in_dict={"pt_root_id": live_roots}), "nuclei live_query")
    nuc_by = {}
    if nuc is not None:
        for _, r in nuc.iterrows():
            p = np.array([r.pt_position_x, r.pt_position_y, r.pt_position_z], float) if "pt_position_x" in nuc.columns \
                else np.array(r.pt_position, float)
            nuc_by.setdefault(int(r.pt_root_id), []).append(p)
    print(f"{len(roots)} partner roots, {len(expired)} expired, nuclei lookup "
          f"{'failed' if nuc is None else f'{len(nuc)} rows'} at {now:%Y-%m-%d %H:%M} UTC", flush=True)

    vals, counts = [], {}
    for _, r in df.iterrows():
        rid = r.root_id_curr
        if rid in pr_roots or PR_PAT.search(f"{r.notes} {r.annotation}"):
            v = "n/a (photoreceptor)"
        elif rid in ovr_by:
            o = ovr_by[rid]
            if o.verdict.startswith("n/a"):
                v = o.verdict
            else:
                m = re.search(r"\d+, \d+, \d+", o.note)
                v = f"{'confirmed' if o.verdict.startswith('soma') else 'not a soma'} at {m.group(0) if m else '?'} ({o.by}, {o.date})"
        elif not rid.isdigit():
            v = ""
        else:
            ids = leaves.get(int(rid))
            nucs = nuc_by.get(int(rid), [])
            lv = local_volume(ids, chunks) if ids else None
            if lv is not None and lv["cand"] is not None:
                v = f"candidate at {vox(lv['cand'])} ({lv['local_um3']:.0f} um3)"
            elif nucs:
                v = f"nucleus at {vox(nucs[0])}"
            elif int(rid) in expired:
                v = "expired root"
            elif lv is None:
                v = "not cached"
            elif lv["total_um3"] < 10:
                v = f"fragment ({lv['n_chunks']} chunks)"
            else:
                v = "none found"
        vals.append(v)
        counts[v.split(" at ")[0].split(" (")[0]] = counts.get(v.split(" at ")[0].split(" (")[0], 0) + 1
    df["soma_check"] = vals
    df.to_csv(TSV, sep="\t", index=False)
    print("soma_check written:", counts)
    print("DONE")


if __name__ == "__main__":
    main()
