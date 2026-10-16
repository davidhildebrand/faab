"""Step 4 of the photoreceptor-group finder: synapse-based verification of the PR-like axons found by the census.

For every PR-like root in figures/pr_bundles_roots.tsv (known and new) two live queries (timestamp = now) count
presynaptic and postsynaptic sites; the median presynaptic position is expressed as a depth d along the mean
cartridge axis relative to the root's bundle centroid 75 um above the lamina (d ~ 45-90 = lamina, d > 90 =
medulla side, d < 45 = inside the bundle zone where photoreceptors make no synapses). Rules, calibrated on the
known PRs (their distributions are printed):
  R1-R6-like : n_pre >= PRE_MIN and 45 <= d <= 90 and n_post <= POST_MAX
  R7/R8-like : n_pre >= 10 and d > 90 and n_post <= POST_MAX_R78
  fragment   : n_pre < 10 (axon piece without its terminal; keep for proofreading)
  other      : anything else (synapses in the wrong place or far too many inputs) - inspect
Counts are cached in figures/pr_bundles_synapses.tsv (rerun with --refresh to re-query); the classification is
network-free and written to figures/pr_bundles_roots_verified.tsv and pr_bundles_clusters_verified.tsv; the
paste-ready summary figures/pr_bundles_summary.md is regenerated with the verified counts.
Usage: python faab_pr_bundles_verify.py [--refresh]
"""
from faab.config import REPO
import datetime
import os
import sys
import time

import numpy as np
import pandas as pd
import caveclient
from faab.clients.cave import DATASTACK, throttle

HERE = REPO
OUT = os.path.join(HERE, "figures")
VOX = np.array([16.0, 16.0, 45.0])
SYN = os.path.join(OUT, "pr_bundles_synapses.tsv")
PRE_MIN, POST_MAX, POST_MAX_R78 = 40, 100, 150


def fetch(ids):
    client = caveclient.CAVEclient(DATASTACK)
    now = datetime.datetime.now(datetime.timezone.utc)
    t = throttle(time.time() - 10)

    def query(side):  # transient CAVE 500s (sqlalchemy connection drops) are retried
        nonlocal t
        for attempt in range(4):
            t = throttle(t)
            try:
                return client.materialize.live_query("synapses_v2", timestamp=now, filter_in_dict={side: ids})
            except Exception as e:
                print(f"  {side} query attempt {attempt + 1} failed: {str(e)[:120]}", flush=True)
                time.sleep(20)
        raise RuntimeError("synapse query keeps failing")
    pre = query("pre_pt_root_id")
    post = query("post_pt_root_id")
    print(f"synapse rows: {len(pre)} pre, {len(post)} post (live at {now:%Y-%m-%d %H:%M} UTC)", flush=True)
    keep = ["id", "pre_pt_root_id", "post_pt_root_id", "pre_pt_position", "post_pt_position", "size"]
    pre[[c for c in keep if c in pre.columns]].to_csv(os.path.join(OUT, "pr_bundles_synapses_raw_pre.tsv"), sep="\t", index=False)
    post[[c for c in keep if c in post.columns]].to_csv(os.path.join(OUT, "pr_bundles_synapses_raw_post.tsv"), sep="\t", index=False)
    n_pre = pre.groupby("pre_pt_root_id").size()
    n_post = post.groupby("post_pt_root_id").size()
    med = {}
    for rid, sub in pre.groupby("pre_pt_root_id"):
        P = np.array([[float(v) for v in str(s).strip("[]").split()] for s in sub.pre_pt_position]) / 1000.0
        med[int(rid)] = np.median(P, axis=0)
    rows = []
    for r in ids:
        m = med.get(r)
        rows.append(dict(root=str(r), n_pre=int(n_pre.get(r, 0)), n_post=int(n_post.get(r, 0)),
                         pre_x=m[0] if m is not None else np.nan, pre_y=m[1] if m is not None else np.nan,
                         pre_z=m[2] if m is not None else np.nan, queried=f"{now:%Y-%m-%d %H:%M} UTC"))
    return pd.DataFrame(rows)


def main():
    roots = pd.read_csv(os.path.join(OUT, "pr_bundles_roots.tsv"), sep="\t", dtype={"root": str, "known": str, "seatable": str})
    roots["known"] = roots.known.fillna("")
    cl = pd.read_csv(os.path.join(OUT, "pr_bundles_clusters.tsv"), sep="\t", index_col=0)
    cl["known_groups"] = cl.known_groups.fillna("")
    known = pd.read_csv(os.path.join(OUT, "pr_lattice_known.tsv"), sep="\t")
    ax = known[["ax_x", "ax_y", "ax_z"]].values.mean(0)
    ax /= np.linalg.norm(ax)
    ids = [int(x) for x in roots.root]
    if "--refresh" in sys.argv or not os.path.exists(SYN):
        syn = fetch(ids)
        syn.to_csv(SYN, sep="\t", index=False)
    else:
        syn = pd.read_csv(SYN, sep="\t", dtype={"root": str})
        missing = set(roots.root) - set(syn.root)
        if missing:
            syn = pd.concat([syn, fetch([int(x) for x in missing])], ignore_index=True)
            syn.to_csv(SYN, sep="\t", index=False)
    df = roots.merge(syn, on="root", how="left")
    cen = cl[["cx", "cy", "cz"]]
    d = []
    for r in df.itertuples():
        if np.isnan(r.pre_x):
            d.append(np.nan)
        else:
            c = cen.loc[r.cluster].values
            d.append(float((np.array([r.pre_x, r.pre_y, r.pre_z]) - c) @ ax))
    df["presyn_depth_um"] = np.round(d, 1)
    df["presyn_xyz"] = ["" if np.isnan(r.pre_x) else f"{int(round(r.pre_x * 1000 / VOX[0]))}, {int(round(r.pre_y * 1000 / VOX[1]))}, {int(round(r.pre_z * 1000 / VOX[2]))}"
                        for r in df.itertuples()]

    def rule(r):
        if r.n_pre < 10:
            return "fragment"
        if r.n_pre >= PRE_MIN and 45 <= r.presyn_depth_um <= 90 and r.n_post <= POST_MAX:
            return "R1-R6-like"
        if r.presyn_depth_um > 90 and r.n_post <= POST_MAX_R78:
            return "R7/R8-like"
        if 45 <= r.presyn_depth_um <= 90 and r.n_post <= POST_MAX:
            return "R1-R6-like (few sites)"
        return "other"
    df["verdict"] = [rule(r) for r in df.itertuples()]
    df["note"] = np.where(df.n_post > 30, "high n_post", "")

    # calibration on the known PRs
    kn = df[df.known != ""].copy()
    kn["cls"] = kn.known.str.split(" ").str[1]
    pd.set_option("display.width", 250)
    print("known PRs (calibration): per class n_pre / n_post / presyn depth quantiles (p10, p50, p90) and rule verdicts")
    for cls, sub in kn.groupby("cls"):
        q = lambda s: sub[s].quantile([0.1, 0.5, 0.9]).round(0).astype(int).tolist()
        print(f"  {cls}: n {len(sub)}; n_pre {q('n_pre')}; n_post {q('n_post')}; depth {sub.presyn_depth_um.quantile([0.1, 0.5, 0.9]).round(0).tolist()}; "
              f"verdicts {sub.verdict.value_counts().to_dict()}")
    new = df[df.known == ""]
    print(f"\nnew PR-like axons: {len(new)}; verdicts {new.verdict.value_counts().to_dict()}")
    df.to_csv(os.path.join(OUT, "pr_bundles_roots_verified.tsv"), sep="\t", index=False)

    # per-cluster verified counts
    g = df.groupby("cluster")
    v = pd.DataFrame({"n_R16_like": g.verdict.apply(lambda s: int(s.isin(["R1-R6-like", "R1-R6-like (few sites)"]).sum())),
                      "n_R78_like": g.verdict.apply(lambda s: int((s == "R7/R8-like").sum())),
                      "n_fragment": g.verdict.apply(lambda s: int((s == "fragment").sum())),
                      "n_other": g.verdict.apply(lambda s: int((s == "other").sum()))})
    clv = cl.join(v)
    clv["n_PR_like"] = clv.n_R16_like + clv.n_R78_like
    clv.to_csv(os.path.join(OUT, "pr_bundles_clusters_verified.tsv"), sep="\t")
    newc = clv[(clv.n_known == 0) & (clv.n_roots >= 4)].copy()
    newc["tier"] = 3
    newc.loc[newc.spread_um <= 6.0, "tier"] = 2
    newc.loc[(newc.spread_um <= 6.0) & newc.n_roots.between(6, 9), "tier"] = 1
    newc = newc.sort_values(["tier", "n_PR_like", "n_roots"], ascending=[True, False, False])
    newc["rank"] = range(1, len(newc) + 1)
    print("\nnew bundles (no known PR, >= 4 axons) with verified member counts:")
    print(newc[["rank", "tier", "n_roots", "n_R16_like", "n_R78_like", "n_fragment", "n_other", "spread_um", "xyz_centroid"]].to_string())
    print("\nmembers judged 'other' (inspect):")
    oth = df[(df.verdict == "other") & (df.known == "")].sort_values("n_post", ascending=False)
    print(oth[["cluster", "root", "xyz_ref", "n_pre", "n_post", "presyn_depth_um", "presyn_xyz", "vox_med", "depths"]].to_string(index=False))

    # paste-ready summary with verdicts
    lines = ["| rank | tier | bundle | n_axons | R1-R6-like | R7/R8-like | fragments | other | spread_um | xyz_centroid_+75um | member_roots (verdict) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    short = {"R1-R6-like": "R1-6", "R1-R6-like (few sites)": "R1-6?", "R7/R8-like": "R7/8", "fragment": "frag", "other": "OTHER"}
    for r in newc.itertuples():
        mem = df[df.cluster == r.Index].sort_values(["verdict", "n_pre"], ascending=[True, False])
        lines.append(f"| {r.rank} | {r.tier} | {r.Index} | {int(r.n_roots)} | {int(r.n_R16_like)} | {int(r.n_R78_like)} | {int(r.n_fragment)} | "
                     f"{int(r.n_other)} | {r.spread_um} | {r.xyz_centroid} | " + ", ".join(f"{m.root} ({short[m.verdict]})" for m in mem.itertuples()) + " |")
    klines = ["| bundle | known_groups | n_known | new members (verdict) |", "|---|---|---|---|"]
    for r in clv[clv.n_known > 0].sort_values("n_new", ascending=False).itertuples():
        mem = df[(df.cluster == r.Index) & (df.known == "")]
        klines.append(f"| {r.Index} | {r.known_groups} | {int(r.n_known)} | " + ", ".join(f"{m.root} ({short[m.verdict]})" for m in mem.itertuples()) + " |")
    text = (f"# New photoreceptor bundle candidates with synapse verification ({syn.queried.iloc[0]})\n\n"
            f"Verdict per axon from its synapses: R1-6 = >= {PRE_MIN} presynaptic sites at lamina depth and <= {POST_MAX} postsynaptic; "
            f"R7/8 = presynaptic sites on the medulla side; frag = < 10 presynaptic sites (axon piece without terminal); "
            f"OTHER = synapses in the wrong place or far too many inputs (inspect). Tiers: 1 = compact (spread <= 6 um) with 6-9 axons, "
            f"2 = compact other sizes, 3 = loose. Per-axon xyz 75 um above the lamina: figures/pr_bundles_roots_verified.tsv.\n\n"
            + "\n".join(lines) + "\n\n## Clusters containing known PRs: new members\n\n" + "\n".join(klines) + "\n")
    open(os.path.join(OUT, "pr_bundles_summary.md"), "w").write(text)
    print("\nsummary written to figures/pr_bundles_summary.md")


if __name__ == "__main__":
    main()
