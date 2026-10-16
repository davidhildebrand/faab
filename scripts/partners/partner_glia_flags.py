"""Apply the glia rule (presynaptic density < 10.6 per 100 um cable) to every partner cell
in figures/faab_pr_top20_update.tsv. Live pre/post counts, cached L2 skeletons for cable
length. Cells under 100 um of cable are called fragments (too small to judge).
Output: figures/partner_glia_flags.tsv and a printed table, and the flag "possible glia?" is
written into the followups column of figures/faab_pr_top20_update.tsv (idempotent; the eight
photoreceptor roots and rows labelled R7/R8/photoreceptor are exempt, because a long
synapse-free retinal axon dilutes the density). Run after faab_pr_top20_anchor_check.py and
before faab_pr_top20_render.py. Read-only, throttled."""
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
from fafbseg import flywire
from faab.clients.cave import DATASTACK, throttle

HERE = REPO
OUT = os.path.join(HERE, "figures")
SCACHE = os.path.join(OUT, "pr_partner_skels.pkl")
TSV = os.path.join(OUT, "faab_pr_top20_update.tsv")
PRE_DENSITY_CUT = 10.6
MIN_CABLE = 100.0
FLAG = "possible glia?"
PR_PAT = re.compile(r"\bR[1-8]\b|photoreceptor", re.I)  # rows already identified as a photoreceptor


def main():
    df = pd.read_csv(TSV, sep="\t", dtype=str).fillna("")
    df = df[df.root_id_curr.str.isdigit()]
    notes = df.groupby("root_id_curr").agg(notes=("notes", lambda s: "; ".join(sorted(set(x for x in s if x)))),
                                           annotation=("annotation", lambda s: "; ".join(sorted(set(x for x in s if x)))),
                                           prs=("upstream_PR", "nunique"),
                                           max_syn=("n_syn", lambda s: max(int(x) for x in s)))
    roots = sorted(int(r) for r in notes.index)
    client = caveclient.CAVEclient(DATASTACK)
    now = datetime.datetime.now(datetime.timezone.utc)
    t = throttle(time.time())
    pre = client.materialize.live_query("synapses_v2", timestamp=now, filter_in_dict={"pre_pt_root_id": roots})
    t = throttle(t)
    post = client.materialize.live_query("synapses_v2", timestamp=now, filter_in_dict={"post_pt_root_id": roots})
    pre = pre[pre.pre_pt_root_id != pre.post_pt_root_id]
    post = post[post.pre_pt_root_id != post.post_pt_root_id]
    n_pre = pre.groupby("pre_pt_root_id").size()
    n_post = post.groupby("post_pt_root_id").size()

    sk_by = pickle.load(open(SCACHE, "rb")) if os.path.exists(SCACHE) else {}
    need = [r for r in roots if r not in sk_by]
    if need:
        t = throttle(t)
        try:
            for s in flywire.get_l2_skeleton(need, dataset=DATASTACK, max_threads=1,
                                             omit_failures=True, progress=False):
                sk_by[int(s.id)] = s
        except Exception as e:
            print("skeleton fetch failed:", str(e)[:60], flush=True)
        pickle.dump(sk_by, open(SCACHE, "wb"))

    rows = []
    for r in roots:
        s = sk_by.get(r)
        cable = float(s.cable_length) / 1000.0 if s is not None else np.nan
        npre, npost = int(n_pre.get(r, 0)), int(n_post.get(r, 0))
        dens = npre / cable * 100 if cable and cable > 0 else np.nan
        if s is None or cable < MIN_CABLE:
            flag = "fragment (too small to judge)"
        elif dens < PRE_DENSITY_CUT:
            flag = "likely glia"
        else:
            flag = "neuron-like"
        info = notes.loc[str(r)]
        rows.append([r, npre, npost, round(npre / max(1, npre + npost), 2), round(cable, 0) if cable == cable else "",
                     round(dens, 1) if dens == dens else "", flag, int(info.prs), int(info.max_syn),
                     info.annotation, info.notes])
    out = pd.DataFrame(rows, columns=["root_id_curr", "n_pre", "n_post", "pre_frac", "cable_um", "pre_per_100um",
                                      "flag", "n_PRs", "max_syn_from_PR", "annotation", "notes"])
    out = out.sort_values(["flag", "pre_per_100um"])
    out.to_csv(os.path.join(OUT, "partner_glia_flags.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 45)
    print(out.flag.value_counts().to_string())
    print()
    print(out[out.flag != "fragment (too small to judge)"].to_string(index=False))
    print(f"\nfragments: {int((out.flag == 'fragment (too small to judge)').sum())} cells under {MIN_CABLE:.0f} um")

    # ---- write "possible glia?" into the table's followups column (photoreceptors exempt) ----
    meta_path = os.path.join(OUT, "faab_pr_top20_update_meta.json")
    pr_roots = set(json.load(open(meta_path))["pr_roots"].values()) if os.path.exists(meta_path) else set()
    by_root = {str(r.root_id_curr): r for r in out.itertuples()}
    full = pd.read_csv(TSV, sep="\t", dtype=str).fillna("")
    flagged = []
    for i, r in full.iterrows():
        fu = [p.strip() for p in str(r["followups"]).split(";") if p.strip() and p.strip() != FLAG]
        m = by_root.get(r["root_id_curr"])
        is_pr = r["root_id_curr"] in pr_roots or bool(PR_PAT.search(f"{r['notes']} {r['annotation']}"))
        if m is not None and m.flag == "likely glia" and not is_pr:
            fu.append(FLAG)
            flagged.append((r["upstream_PR"], r["root_id_orig"], r["root_id_curr"], m.n_pre, m.n_post,
                            m.cable_um, m.pre_per_100um, r["notes"]))
        full.at[i, "followups"] = "; ".join(fu)
    full.to_csv(TSV, sep="\t", index=False)
    print(f"\nfollowups flag '{FLAG}' written to {len(flagged)} table rows "
          f"({len(set(f[2] for f in flagged))} distinct cells); photoreceptor rows exempt")
    for f in flagged:
        print("  ", "\t".join(str(x) for x in f))
    print(f"live at {now:%Y-%m-%d %H:%M} UTC\nDONE", flush=True)


if __name__ == "__main__":
    main()
