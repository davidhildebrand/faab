"""Step 4 of the photoreceptor-group finder: synapse-based verification of the PR-like axons found by the census.

For every PR-like root in figures/pr_bundles_roots.tsv (known and new) presynaptic and postsynaptic sites are counted
(local synapse snapshot + one CAVE catch-up to now; validated identical to live_query, faab.synsnap), and where the
outputs are (network-free, from the raw presynaptic table): the lamina share = fraction of presynaptic sites inside the
side's lamina mesh, the medulla share = inside the medulla mesh or outside both meshes within ME_MARGIN_UM of its
surface (nearest mesh vertex; faab.ol_meshes). The margin is there because the medulla mesh is tight: R7 outputs
outside both meshes lie a median 2.4 um from it (R1-R6 and R8 ~56 um), and with 5 um the median R7 medulla share goes
from 0.25 to 0.66 (scratch/pr_groups/r7_misses.py). Rules (calibrated 2026-10-29 on the right-side classes of
data/pr_groups.tsv, which come from bundle geometry: R8 = central axon; scratch/pr_groups/r78_mesh_check.py +
r78_rule_eval.py + r7_misses.py):
  fragment   : n_pre < 10 (axon piece without its terminal; keep for proofreading)
  lamina-ending (lamina share >= LA_MIN, medulla share < ME_R7, n_post <= POST_MAX), split by inputs per output:
    R8-like    : n_post / n_pre >= R8_IN_PER_OUT (R8 p10-p90 0.31-1.03; R1-R6 0.01-0.09)
    R1-R6-like : below that and n_pre >= PRE_MIN; 'R1-R6-like (few sites)' if fewer
  R7-like    : (lamina share < LA_MIN or medulla share >= ME_R7) and n_post <= POST_MAX_R7 (R7 outputs: median 26%
               lamina, 66% in or near the medulla; some R7s have no medulla outputs, noted)
  other      : anything else (e.g. medulla neurons with hundreds of inputs) - inspect
In FAAB R8 ends in the lamina: of 76 right-side R8s with >= 10 outputs one has >= 20% of its outputs in ME_R (none of
the 8 proofread), so the 'R7/R8 = medulla outputs' rule of the first mesh version (2026-10-29, same day) only found R7s.
Scored on the right (all cells / proofread-checked): R1-R6 406/443 and 47/47, R8 67/76 and 8/8, R7 55/68 and 7/8
(without the margin R7 52/68). The proofread miss (648518347606577544, group 3) has no outputs in or near the medulla;
only its 1.88 inputs per output (above every R8) sets it apart, which is not used (one cell; David 2026-10-29). Six of
the other 'R7' misses look like R1-R6 by every measure and are probably mislabeled in the table.
Until 2026-10-29 the position test was a depth d of the median presynaptic site along the mean cartridge axis,
relative to the bundle centroid 75 um above the lamina (45-90 = lamina, > 90 = medulla side). On the left it fails:
along cartridge 8's axis the medulla lies at lamina depth, so 20 R7s (one per bundle, 69% of outputs in ME_L) were
called R1-R6 and 23 lamina axons just short of 45 um (all of cluster 81) 'other' (scratch/pr_groups/verify_mesh_rule.py).
d is still reported (presyn_depth_um) and a disagreement noted.
Counts are cached in figures/pr_bundles_synapses.tsv and the synapses in the raw tables (rerun with --refresh to
re-query); the mesh shares and the classification are network-free and written to
figures/pr_bundles_roots_verified.tsv and pr_bundles_clusters_verified.tsv; the paste-ready summary
figures/pr_bundles_summary.md is regenerated with the verified counts. The raw synapse tables keep the live_query
columns (root ids, '[x y z]' positions in nm), which src/faab/pr/bundles_pairs.py reads.
--side left reads/writes the pr_left_* files instead; default right.
Usage: python bundles_verify.py [--refresh] [--side right|left]
"""
from faab.config import REPO
import datetime
import os

import numpy as np
import pandas as pd
import caveclient
from scipy.spatial import cKDTree
from faab import ol_meshes, synsnap
from faab.clients.cave import DATASTACK
from faab.pr.side import NEUROPILS, PREFIX, parse as parse_side

HERE = REPO
OUT = os.path.join(HERE, "figures")
VOX = np.array([16.0, 16.0, 45.0])
SIDE, ARGS = parse_side()
PFX = PREFIX[SIDE]
LA, ME = NEUROPILS[SIDE]
SYN = os.path.join(OUT, f"{PFX}bundles_synapses.tsv")
RAW = os.path.join(OUT, f"{PFX}bundles_synapses_raw_{{}}.tsv")
PRE_MIN, POST_MAX, POST_MAX_R7 = 40, 100, 150
# right-side lamina share p10: R1-R6 0.86, R8 0.62; R7 p75 0.56. Medulla share (5 um margin) p90: R1-R6 and R8 0.0
LA_MIN, ME_R7, R8_IN_PER_OUT, ME_MARGIN_UM = 0.6, 0.1, 0.2, 5.0


def mesh_shares(pre):
    """lamina and medulla shares of each root's outputs, from a raw presynaptic table (positions in nm)"""
    if not len(pre):
        return pd.DataFrame(columns=["la_frac", "me_frac"])
    P = pre.pre_pt_position.str.strip("[]").str.split(expand=True).to_numpy(float) / 1000.0
    region = ol_meshes.assign(P, [LA, ME])
    near_me = cKDTree(ol_meshes.mesh(ME)[0] / 1000.0).query(P)[0] <= ME_MARGIN_UM
    region = np.where((region == "none") & near_me, ME, region)
    t = pd.crosstab(pre.pre_pt_root_id.to_numpy(), region, normalize="index")
    return pd.DataFrame({"la_frac": t[LA] if LA in t else 0.0, "me_frac": t[ME] if ME in t else 0.0}, index=t.index).round(2)


def raw(df):
    """snapshot rows -> the live_query layout of the raw tables (root ids, '[x y z]' positions in nm)"""
    pos = lambda p: [f"[{int(x * VOX[0])} {int(y * VOX[1])} {int(z * VOX[2])}]"
                     for x, y, z in df[[f"{p}_x", f"{p}_y", f"{p}_z"]].to_numpy()]
    return pd.DataFrame({"id": df.id, "pre_pt_root_id": df.pre_root, "post_pt_root_id": df.post_root,
                         "pre_pt_position": pos("pre"), "post_pt_position": pos("post"), "size": df["size"]})


def fetch(ids, append=False):
    """counts per root; the raw tables are rewritten (append=True: extended by these roots' rows)"""
    client = caveclient.CAVEclient(DATASTACK)
    now = datetime.datetime.now(datetime.timezone.utc)
    synsnap.refresh()
    S = synsnap.Snapshot()
    pre = S.synapses(pre=ids, when="now", positions=True, client=client)
    post = S.synapses(post=ids, when="now", positions=True, client=client)
    print(f"synapse rows: {len(pre)} pre, {len(post)} post (snapshot + catch-up to {now:%Y-%m-%d %H:%M} UTC)", flush=True)
    for kind, df in (("pre", pre), ("post", post)):
        r = raw(df).astype(str)
        if append and os.path.exists(RAW.format(kind)):
            r = pd.concat([pd.read_csv(RAW.format(kind), sep="\t", dtype=str), r]).drop_duplicates("id")
        r.to_csv(RAW.format(kind), sep="\t", index=False)
    n_pre = pre.groupby("pre_root").size()
    n_post = post.groupby("post_root").size()
    P_all = pre[["pre_x", "pre_y", "pre_z"]].to_numpy(float) * VOX / 1000.0
    rows = []
    for r in ids:
        sel = (pre.pre_root == r).to_numpy()
        m = np.median(P_all[sel], axis=0) if sel.any() else None
        rows.append(dict(root=str(r), n_pre=int(n_pre.get(r, 0)), n_post=int(n_post.get(r, 0)),
                         pre_x=m[0] if m is not None else np.nan, pre_y=m[1] if m is not None else np.nan,
                         pre_z=m[2] if m is not None else np.nan, queried=f"{now:%Y-%m-%d %H:%M} UTC"))
    return pd.DataFrame(rows)


def main():
    roots = pd.read_csv(os.path.join(OUT, f"{PFX}bundles_roots.tsv"), sep="\t", dtype={"root": str, "known": str, "seatable": str})
    roots["known"] = roots.known.fillna("")
    cl = pd.read_csv(os.path.join(OUT, f"{PFX}bundles_clusters.tsv"), sep="\t", index_col=0)
    cl["known_groups"] = cl.known_groups.fillna("")
    known = pd.read_csv(os.path.join(OUT, f"{PFX}lattice_known.tsv"), sep="\t")
    ax = known[["ax_x", "ax_y", "ax_z"]].values.mean(0)
    ax /= np.linalg.norm(ax)
    ids = [int(x) for x in roots.root]
    if "--refresh" in ARGS or not os.path.exists(SYN):
        syn = fetch(ids)
        syn.to_csv(SYN, sep="\t", index=False)
    else:
        syn = pd.read_csv(SYN, sep="\t", dtype={"root": str})
        missing = set(roots.root) - set(syn.root)
        if missing:
            syn = pd.concat([syn, fetch([int(x) for x in missing], append=True)], ignore_index=True)
            syn.to_csv(SYN, sep="\t", index=False)
    pre = pd.read_csv(RAW.format("pre"), sep="\t", dtype={"pre_pt_root_id": str})
    lost = set(syn.root[syn.n_pre > 0]) - set(pre.pre_pt_root_id)
    if lost:
        raise SystemExit(f"{len(lost)} roots with outputs are missing from {RAW.format('pre')}: rerun with --refresh")
    shares = mesh_shares(pre[pre.pre_pt_root_id.isin(set(syn.root))])
    syn = syn.drop(columns=["la_frac", "me_frac"], errors="ignore").join(shares, on="root")
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
        if r.la_frac >= LA_MIN and r.me_frac < ME_R7 and r.n_post <= POST_MAX:
            if r.n_post / r.n_pre >= R8_IN_PER_OUT:
                return "R8-like"
            return "R1-R6-like" if r.n_pre >= PRE_MIN else "R1-R6-like (few sites)"
        if (r.la_frac < LA_MIN or r.me_frac >= ME_R7) and r.n_post <= POST_MAX_R7:
            return "R7-like"
        return "other"
    df["verdict"] = [rule(r) for r in df.itertuples()]

    def note(r):
        out = ["high n_post"] if r.n_post > 30 else []
        if r.verdict.startswith("R1-R6") or r.verdict == "R8-like":
            if not 45 <= r.presyn_depth_um <= 90:
                out.append(f"outputs at depth {r.presyn_depth_um:.0f} um, outside the 45-90 lamina window")
        if r.verdict == "R7-like" and r.me_frac == 0:
            out.append(f"no outputs in or within {ME_MARGIN_UM:.0f} um of {ME}")
        return "; ".join(out)
    df["note"] = [note(r) for r in df.itertuples()]

    # calibration on the known PRs
    kn = df[df.known != ""].copy()
    kn["cls"] = kn.known.str.split(" ").str[1]
    pd.set_option("display.width", 250)
    print(f"side {SIDE}; known PRs (calibration): per class n_pre / n_post / presyn depth / {LA} and {ME} fractions "
          f"(p10, p50, p90) and rule verdicts")
    for cls, sub in kn.groupby("cls"):
        q = lambda s: sub[s].quantile([0.1, 0.5, 0.9]).round(0).astype(int).tolist()
        qf = lambda s: sub[s].quantile([0.1, 0.5, 0.9]).round(2).tolist()
        print(f"  {cls}: n {len(sub)}; n_pre {q('n_pre')}; n_post {q('n_post')}; depth {sub.presyn_depth_um.quantile([0.1, 0.5, 0.9]).round(0).tolist()}; "
              f"{LA} {qf('la_frac')}; {ME} {qf('me_frac')}; verdicts {sub.verdict.value_counts().to_dict()}")
    new = df[df.known == ""]
    print(f"\nnew PR-like axons: {len(new)}; verdicts {new.verdict.value_counts().to_dict()}; depth-window disagreements "
          f"{int(new.note.str.contains('depth').sum())}")
    df.to_csv(os.path.join(OUT, f"{PFX}bundles_roots_verified.tsv"), sep="\t", index=False)

    # per-cluster verified counts
    g = df.groupby("cluster")
    v = pd.DataFrame({"n_R16_like": g.verdict.apply(lambda s: int(s.isin(["R1-R6-like", "R1-R6-like (few sites)"]).sum())),
                      "n_R8_like": g.verdict.apply(lambda s: int((s == "R8-like").sum())),
                      "n_R7_like": g.verdict.apply(lambda s: int((s == "R7-like").sum())),
                      "n_fragment": g.verdict.apply(lambda s: int((s == "fragment").sum())),
                      "n_other": g.verdict.apply(lambda s: int((s == "other").sum()))})
    clv = cl.join(v)
    clv["n_PR_like"] = clv.n_R16_like + clv.n_R8_like + clv.n_R7_like
    clv.to_csv(os.path.join(OUT, f"{PFX}bundles_clusters_verified.tsv"), sep="\t")
    newc = clv[(clv.n_known == 0) & (clv.n_roots >= 4)].copy()
    newc["tier"] = 3
    newc.loc[newc.spread_um <= 6.0, "tier"] = 2
    newc.loc[(newc.spread_um <= 6.0) & newc.n_roots.between(6, 9), "tier"] = 1
    newc = newc.sort_values(["tier", "n_PR_like", "n_roots"], ascending=[True, False, False])
    newc["rank"] = range(1, len(newc) + 1)
    print("\nnew bundles (no known PR, >= 4 axons) with verified member counts:")
    print(newc[["rank", "tier", "n_roots", "n_R16_like", "n_R8_like", "n_R7_like", "n_fragment", "n_other", "spread_um", "xyz_centroid"]].to_string())
    print("\nmembers judged 'other' (inspect):")
    oth = df[(df.verdict == "other") & (df.known == "")].sort_values("n_post", ascending=False)
    print(oth[["cluster", "root", "xyz_ref", "n_pre", "n_post", "presyn_depth_um", "la_frac", "me_frac", "presyn_xyz", "vox_med", "depths"]].to_string(index=False))

    # paste-ready summary with verdicts
    lines = ["| rank | tier | bundle | n_axons | R1-R6-like | R8-like | R7-like | fragments | other | spread_um | xyz_centroid_+75um | member_roots (verdict) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    short = {"R1-R6-like": "R1-6", "R1-R6-like (few sites)": "R1-6?", "R8-like": "R8", "R7-like": "R7", "fragment": "frag", "other": "OTHER"}
    for r in newc.itertuples():
        mem = df[df.cluster == r.Index].sort_values(["verdict", "n_pre"], ascending=[True, False])
        lines.append(f"| {r.rank} | {r.tier} | {r.Index} | {int(r.n_roots)} | {int(r.n_R16_like)} | {int(r.n_R8_like)} | {int(r.n_R7_like)} | {int(r.n_fragment)} | "
                     f"{int(r.n_other)} | {r.spread_um} | {r.xyz_centroid} | " + ", ".join(f"{m.root} ({short[m.verdict]})" for m in mem.itertuples()) + " |")
    klines = ["| bundle | known_groups | n_known | new members (verdict) |", "|---|---|---|---|"]
    for r in clv[clv.n_known > 0].sort_values("n_new", ascending=False).itertuples():
        mem = df[(df.cluster == r.Index) & (df.known == "")]
        klines.append(f"| {r.Index} | {r.known_groups} | {int(r.n_known)} | " + ", ".join(f"{m.root} ({short[m.verdict]})" for m in mem.itertuples()) + " |")
    text = (f"# New photoreceptor bundle candidates with synapse verification, {SIDE} eye ({syn.queried.iloc[0]})\n\n"
            f"Verdict per axon from its synapses: lamina-ending = >= {LA_MIN:.0%} of the presynaptic sites in {LA}, < {ME_R7:.0%} in or within "
            f"{ME_MARGIN_UM:.0f} um of {ME}, <= {POST_MAX} postsynaptic, split into R8 (>= {R8_IN_PER_OUT} inputs per output) and R1-6 (fewer; "
            f">= {PRE_MIN} outputs, else R1-6?); R7 = < {LA_MIN:.0%} of the presynaptic sites in {LA} or >= {ME_R7:.0%} in or near {ME}, "
            f"<= {POST_MAX_R7} postsynaptic; frag = < 10 presynaptic sites (axon piece without terminal); "
            f"OTHER = synapses in the wrong place or far too many inputs (inspect). Tiers: 1 = compact (spread <= 6 um) with 6-9 axons, "
            f"2 = compact other sizes, 3 = loose. Per-axon xyz 75 um above the lamina: figures/{PFX}bundles_roots_verified.tsv.\n\n"
            + "\n".join(lines) + "\n\n## Clusters containing known PRs: new members\n\n" + "\n".join(klines) + "\n")
    open(os.path.join(OUT, f"{PFX}bundles_summary.md"), "w").write(text)
    print(f"\nsummary written to figures/{PFX}bundles_summary.md")


if __name__ == "__main__":
    main()
