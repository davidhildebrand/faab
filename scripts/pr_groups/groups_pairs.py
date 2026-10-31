"""R1-R6 pairing of the eight photoreceptor groups on TODAY's roots, resolved at the sheet xyz.

1. tmp/pr_groups_pasted.tsv (group, root_id_orig, class, root_id_curr, xyz) -> one segment lookup at
   every xyz; the root at the point is used (user rule); rows whose pasted root_id_curr differs are
   reported as MISMATCH.
2. L2 skeletons fetched for roots not in figures/pr_partner_skels.pkl (one throttled call).
3. The two group files the analyses read are rewritten (old ones backed up with a date suffix):
   figures/faab_pr_current_roots.tsv (group 1) and figures/faab_pr_groups_xyz.tsv (groups 2-8, 'group' 1..7).
4. faab_pr_groups_mirror.main() (in-situ mirror-plane NBLAST, mutual mirror pairs) and
   faab_pr_groups_flynames2.main() (terminal-zone pairs + fly-name recipe) are rerun.
5. Output figures/faab_pr_groups_pairs.tsv: per cell -> mirror partner/score, terminal partner/score,
   position in the terminal cross-section (t, d), proposed fly pair, and whether the two analyses agree.
"""
from faab.config import REPO
import datetime
import os
import pickle
import shutil
import time

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import caveclient
from fafbseg import flywire
from faab.clients.cave import DATASTACK, throttle
from faab.partners.top10_xyz import VOX
from faab.config import fig

HERE = REPO
OUT = os.path.join(HERE, "figures")
TMP = os.path.join(HERE, "..", "tmp")
SCACHE = fig("pr_partner_skels.pkl")
G1 = fig("faab_pr_current_roots.tsv")
G28 = fig("faab_pr_groups_xyz.tsv")


def main():
    t0 = time.time()
    df = pd.read_csv(os.path.join(TMP, "pr_groups_pasted.tsv"), sep="\t", dtype=str)
    df["class"] = df["class"].str.replace("–", "-")
    now = datetime.datetime.now(datetime.timezone.utc)
    locs = np.array([[float(v) for v in x.split(",")] for x in df.xyz]) * VOX
    roots = flywire.locs_to_segments(locs, coordinates="nm", backend="cloudvolume", dataset=DATASTACK)
    df["root_at_xyz"] = [str(int(r)) for r in roots]
    df["root_used"] = [a if a != "0" else c for a, c in zip(df.root_at_xyz, df.root_id_curr)]
    df["match"] = ["ok" if a == c else ("no segment at xyz" if a == "0" else "MISMATCH") for a, c in zip(df.root_at_xyz, df.root_id_curr)]
    print(f"root lookup at {now:%Y-%m-%d %H:%M} UTC: {(df.match == 'ok').sum()} ok, "
          f"{(df.match == 'MISMATCH').sum()} mismatch, {(df.match == 'no segment at xyz').sum()} empty", flush=True)
    for r in df[df.match != "ok"].itertuples():
        print(f"  group {r.group} {r._3} {r.root_id_orig}: pasted root_id_curr {r.root_id_curr}, at xyz {r.root_at_xyz} ({r.match})", flush=True)
    df.to_csv(fig("faab_pr_groups_roots_at_xyz.tsv"), sep="\t", index=False)

    # skeletons for roots not cached
    sk_by = pickle.load(open(SCACHE, "rb"))
    need = sorted({int(r) for r in df.root_used if int(r) not in sk_by})
    print(f"skeletons to fetch: {len(need)}", flush=True)
    if need:
        client = caveclient.CAVEclient(DATASTACK)
        t = throttle(time.time())
        got = 0
        for s in flywire.get_l2_skeleton(need, dataset=DATASTACK, max_threads=1, omit_failures=True, progress=False):
            sk_by[int(s.id)] = s
            got += 1
        pickle.dump(sk_by, open(SCACHE, "wb"))
        print(f"  fetched {got}", flush=True)
    missing = [r for r in df.root_used if int(r) not in sk_by]
    if missing:
        print("  no skeleton for:", missing, flush=True)

    # group files for the analyses (backup the old ones)
    stamp = "2026-09-11"
    for p in (G1, G28):
        if os.path.exists(p) and not os.path.exists(p.replace(".tsv", f"_{stamp}.tsv")):
            shutil.copy(p, p.replace(".tsv", f"_{stamp}.tsv"))
    g1 = df[df.group == "1"][["root_id_orig", "class", "root_used", "xyz"]].rename(columns={"root_used": "root_id_curr"})
    g1.to_csv(G1, sep="\t", index=False)
    g28 = df[df.group != "1"].copy()
    g28["group"] = (g28.group.astype(int) - 1).astype(str)
    g28[["group", "root_id_orig", "class", "root_used", "xyz"]].rename(columns={"root_used": "root_id_curr"}).to_csv(G28, sep="\t", index=False)

    import faab.pr.groups_mirror as M
    import faab.pr.groups_flynames2 as F
    print("\n=== mirror-pair analysis (in situ) ===", flush=True)
    M.main()
    print("\n=== terminal-zone pairs + fly names ===", flush=True)
    F.main()

    mir = pd.read_csv(fig("faab_pr_groups_mirror.tsv"), sep="\t", dtype={"cell_a": str, "cell_b": str})
    fly = pd.read_csv(fig("faab_pr_groups_flynames_terminal.tsv"), sep="\t", dtype={"cell": str, "partner": str})
    mpart, mscore, mnext = {}, {}, {}
    for r in mir.itertuples():
        mpart[r.cell_a], mpart[r.cell_b] = r.cell_b, r.cell_a
        mscore[r.cell_a] = mscore[r.cell_b] = r.mirror_ab
        mnext[r.cell_a] = mnext[r.cell_b] = r.next_best_mirror
    fpart = {r.cell: r for r in fly.itertuples()}
    rows = []
    for r in df[df["class"] == "R1-R6"].itertuples():
        c = r.root_used
        f = fpart.get(c)
        rows.append(dict(group=r.group, root_id_orig=r.root_id_orig, root_used=c, xyz=r.xyz, match=r.match,
                         mirror_partner=mpart.get(c, ""), mirror_score=mscore.get(c, ""), mirror_next_best=mnext.get(c, ""),
                         terminal_partner=f.partner if f is not None else "", terminal_score=f.pair_score if f is not None else "",
                         terminal_next=f.next_best if f is not None else "",
                         t_um=f.t_um if f is not None else "", d_um=f.d_um if f is not None else "",
                         fly_pair=f.proposed_fly_pair if f is not None else "",
                         agree=("yes" if (f is not None and mpart.get(c, "") == f.partner) else ("no" if f is not None else ""))))
    out = pd.DataFrame(rows)
    out.to_csv(fig("faab_pr_groups_pairs.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    print("\n=== per-cell summary ===")
    print(out.to_string(index=False))
    print(f"\nagreement mirror vs terminal partner: {(out.agree == 'yes').sum()}/{len(out)} cells")
    print(f"DONE in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
