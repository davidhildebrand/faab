"""Apply the synapse-placement check (figures/pr_partner_split_check.json) to the table:
where the largest piece of a split partner carries NONE of the photoreceptor's synapses,
switch root_id_curr to the piece that does, recompute its xyz from the cached skeleton
(+ nucleus lookup), and re-render. Local except one nuclei query (read-only).
"""
from faab.config import REPO
import json
import os
import pickle
import runpy
import time

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import caveclient
from faab.clients.cave import DATASTACK
from faab.partners.top10_xyz import primary_neurite_point_nm, backbone_midpoint_nm, VOX
from faab.config import fig

HERE = REPO
OUT = os.path.join(HERE, "figures")
TSV = fig("faab_pr_top10_xyz.tsv")
CHK = fig("pr_partner_split_check.json")
SCACHE = fig("pr_partner_skels.pkl")


def main():
    chk = json.load(open(CHK))
    df = pd.read_csv(TSV, sep="\t", dtype=str)
    fixes = {}
    for key, c in chk.items():
        on_major = next(p["syn_from_pr"] for p in c["pieces"] if p["root"] == c["major"])
        best = max(c["pieces"], key=lambda p: p["syn_from_pr"])
        if on_major == 0 and best["syn_from_pr"] > 0:
            fixes[(str(c["pr"]), str(c["cached"]))] = (best, c)
    print(f"{len(fixes)} row(s) where the largest piece carries no PR synapses", flush=True)
    if not fixes:
        return

    sk_by = pickle.load(open(SCACHE, "rb"))
    client = caveclient.CAVEclient(DATASTACK)
    roots = sorted({f[0]["root"] for f in fixes.values()})
    time.sleep(5)
    nuc = client.materialize.query_table("nuclei_v1_aedes", filter_in_dict={"pt_root_id": roots})
    soma = {}
    for _, r in nuc.iterrows():
        soma[int(r.pt_root_id)] = (np.array([r.pt_position_x, r.pt_position_y, r.pt_position_z], float)
                                   if "pt_position_x" in nuc.columns else np.array(r.pt_position, float))

    for (pr, cached), (best, c) in fixes.items():
        rid = best["root"]
        s = sk_by.get(rid)
        if s is None:
            print(f"  no cached skeleton for {rid}; leaving row unchanged", flush=True)
            continue
        pt, geo = (None, None)
        if rid in soma and s.n_nodes >= 5:
            pt, geo = primary_neurite_point_nm(s, soma[rid])
        tag = (f"primary neurite ~{geo:.0f}um from soma" if pt is not None
               else "no nucleus -> backbone midpoint")
        if pt is None:
            pt = backbone_midpoint_nm(s)
        xyz = ", ".join(str(int(round(v))) for v in (pt / VOX))
        other = [p for p in c["pieces"] if p["root"] != rid]
        other_txt = ", ".join(f"{p['root']} ({p['syn_from_pr']} syn, {p['l2_chunks']} chunks)"
                              for p in other)
        note = (f"root changed; xyz from current root {rid}; split into {len(c['pieces'])} "
                f"comparable pieces - this one carries all {best['syn_from_pr']} PR synapses, "
                f"other: {other_txt}; {tag}")
        m = (df.upstream_PR == pr) & (df.downstream == cached)
        print(f"  PR {pr} / cached {cached}: {c['major']} -> {rid}; xyz {xyz}", flush=True)
        df.loc[m, "xyz"] = xyz
        df.loc[m, "notes"] = note
    df.to_csv(TSV, sep="\t", index=False)
    print("patched", TSV, "\n", flush=True)
    runpy.run_path(os.path.join(REPO, "scratch", "partners_top10", "top10_ascii.py"), run_name="__main__")


if __name__ == "__main__":
    main()
