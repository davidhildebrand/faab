"""Summation footprint of the photoreceptor groups in the lamina, and R7 <-> R8 synapses within groups.

Question 1 (David, 2026-10-10): how many ommatidia (PR groups) converge on each lamina target cell? Aedes has fused
rhabdoms, so there is no neural superposition; strict retinotopy predicts one group per lamina monopolar cell,
neural pooling (as in nocturnal nematocerans) predicts input from several neighbouring groups.
Question 2 (David, 2026-10-10): do R7 cells receive input from their own group's R8 but not vice versa?

Steps (read-only, <= 1 CAVE call / 5 s, no fan-out):
 1. tmp/pr_groups_pasted.tsv (groups 1-8; xyz is the anchor) -> root at every xyz (one locs_to_segments call);
    the root at the point is used, pasted root_id_curr only as fallback for an empty lookup.
 2. live_query synapses_v2 with pre_pt_root_id in the 64 PR roots -> every PR output synapse (saved raw with the
    query time: figures/pr_groups_out_raw.tsv).
 3. targets = postsynaptic roots with >= MIN_TOTAL synapses from the groups (PRs themselves excluded); FlyTable
    labels (seaserpent aedes_main: class / subclass / type, supervoxel -> root, one get_roots call).
 4. live_query synapses_v2 with post_pt_root_id in targets + the 16 R7/R8 cells -> their complete input (saved raw:
    figures/pr_targets_in_raw.tsv). Presynaptic roots classed as group PR (class) / FlyTable PR (type) / census
    PR-like (figures/pr_bundles_roots_verified.tsv verdicts, roots of 2026-10-01) / other labelled / unlabelled.
 5. Per target: total and PR input, number of PR cells, distinct groups (>= 1 and >= 3 synapses), per-group counts,
    top-group share, footprint = p90 radius of all R1-R6-type presynaptic sites onto it (um and lattice spacings).
 6. Group pairs: targets shared (>= 3 synapses from each) against the lattice distance of the two lamina points
    (figures/pr_lattice_known.tsv, hexagonal spacing SPACING_UM).
 7. R7/R8: PR -> PR synapses within and between groups by class; position of every R8->R7 and R7->R8 synapse along
    the group's lamina (R1-R6 presynaptic centroid) -> R7 terminal (R7 presynaptic centroid) axis; and the complete
    input composition of each R7 and R8 cell.
Outputs (figures/): pr_lamina_footprint.tsv, pr_lamina_footprint_pairs.tsv, pr_r7r8_synapses.tsv,
pr_r7r8_inputs.tsv, pr_lamina_footprint_summary.md (markdown tables, also printed).
Usage: python faab_pr_lamina_footprint.py [--cached]   (--cached reuses the saved roots and raw synapse tables)
"""
from faab.config import REPO
import argparse
import datetime
import json
import os
import time

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import caveclient
from fafbseg import flywire
from faab.sources.cave import DATASTACK, throttle
from faab.partners.top10_xyz import VOX

HERE = REPO
OUT = os.path.join(HERE, "figures")
TMP = os.path.join(HERE, "..", "tmp")
SERVER = "https://flytable.mrc-lmb.cam.ac.uk/"
SPACING_UM = 5.9      # hexagonal spacing of the lamina points (faab_pr_lattice.py, 2026-10-01)
MIN_TOTAL = 3         # a target must receive >= this many synapses from the groups in total (selection only)
ROOTS_CACHE = os.path.join(OUT, "pr_footprint_roots.json")
RAW_OUT = os.path.join(OUT, "pr_groups_out_raw.tsv")
RAW_IN = os.path.join(OUT, "pr_targets_in_raw.tsv")
PRCLASS = ["R1-R6", "R7", "R8"]


def _s(v):
    return "" if (v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v)) else str(v).strip()


def retry(fn, tries=3):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # transient CAVE 500s
            if k == tries - 1:
                raise
            print(f"  query failed ({str(e)[:90]}), retrying in 20 s", flush=True)
            time.sleep(20)


def positions(df, prefix, res_nm, ref_um):
    """N x 3 array in um from a live_query frame, whatever the position column layout.
    caveclient may hand positions back in nm or in table voxels (2026-10-10: nm, although the table metadata says
    16 x 16 x 45 nm voxels), so the unit is chosen by which interpretation puts the median position nearest the
    lamina region (ref_um); the two candidates differ by >= 16x, so the choice is unambiguous."""
    col = f"{prefix}_pt_position"
    if col in df.columns:
        p = np.vstack([np.asarray(v, float) for v in df[col]])
    else:
        p = df[[f"{col}_x", f"{col}_y", f"{col}_z"]].to_numpy(float)
    cand = {"nm": p / 1000.0, "voxels": p * np.asarray(res_nm, float) / 1000.0}
    unit = min(cand, key=lambda k: np.linalg.norm(np.median(cand[k], axis=0) - ref_um))
    positions.unit = unit
    return cand[unit]


def flat(df, res_nm, when, ref_um):
    f = pd.DataFrame({"pre": df.pre_pt_root_id.astype(np.int64).to_numpy(),
                      "post": df.post_pt_root_id.astype(np.int64).to_numpy()})
    f[["pre_x", "pre_y", "pre_z"]] = positions(df, "pre", res_nm, ref_um)
    f[["post_x", "post_y", "post_z"]] = positions(df, "post", res_nm, ref_um)
    f["queried"] = when
    print(f"  positions interpreted as {positions.unit}; median presynaptic site "
          f"{np.round(np.median(f[['pre_x', 'pre_y', 'pre_z']].to_numpy(), axis=0), 1)} um", flush=True)
    return f


def md_table(df, cols=None):
    cols = cols or list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in df[cols].itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", action="store_true", help="reuse saved roots and raw synapse tables")
    args = ap.parse_args()
    t0 = time.time()
    client = caveclient.CAVEclient(DATASTACK)
    t = throttle(time.time())

    # ---- 1. groups and roots at xyz -------------------------------------------------------------------------
    df = pd.read_csv(os.path.join(TMP, "pr_groups_pasted.tsv"), sep="\t", dtype=str)
    df["class"] = df["class"].str.replace("–", "-")
    df["group"] = df.group.astype(int)
    if args.cached and os.path.exists(ROOTS_CACHE):
        cache = json.load(open(ROOTS_CACHE))
        df["root_used"] = [str(x) for x in cache["root_used"]]
        print(f"roots from cache ({cache['date']})", flush=True)
    else:
        locs = np.array([[float(v) for v in x.split(",")] for x in df.xyz]) * VOX
        roots = np.asarray(flywire.locs_to_segments(locs, coordinates="nm", backend="cloudvolume",
                                                    dataset=DATASTACK)).astype(np.int64)
        df["root_at_xyz"] = [str(int(r)) for r in roots]
        df["root_used"] = [a if a != "0" else c for a, c in zip(df.root_at_xyz, df.root_id_curr)]
        diff = df[df.root_used != df.root_id_curr]
        print(f"roots at xyz: {len(df)} cells, {len(diff)} differ from the pasted root_id_curr "
              f"({(df.root_at_xyz == '0').sum()} empty lookups)", flush=True)
        for r in diff.itertuples():
            print(f"  group {r.group} {getattr(r, '_3')} xyz {r.xyz}: pasted {r.root_id_curr} -> at xyz {r.root_used}",
                  flush=True)
        json.dump({"date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                   "root_used": list(df.root_used)}, open(ROOTS_CACHE, "w"))
    df["root_used"] = df.root_used.astype(np.int64)
    dup = df[df.root_used.duplicated(keep=False)]
    if len(dup):
        print("WARNING: cells sharing one root (merge?):", dup[["group", "class", "root_used"]].to_dict("records"), flush=True)
    pr_group = dict(zip(df.root_used, df.group))
    pr_class = dict(zip(df.root_used, df["class"]))
    pr_roots = sorted(set(pr_group))
    lat = pd.read_csv(os.path.join(OUT, "pr_lattice_known.tsv"), sep="\t")
    lam = {int(c[1:]): np.array([r.lam_x, r.lam_y, r.lam_z]) for c, r in zip(lat.cartridge, lat.itertuples()) if c.startswith("g")}
    ref_um = np.mean(list(lam.values()), axis=0)  # lamina region, for the position-unit check

    # synapse table voxel resolution (positions come back in table voxels)
    t = throttle(t)
    try:
        meta = client.materialize.get_table_metadata("synapses_v2")
        res = [meta.get("voxel_resolution_x"), meta.get("voxel_resolution_y"), meta.get("voxel_resolution_z")]
        if any(v is None for v in res):
            raise ValueError("no voxel_resolution in metadata")
    except Exception as e:
        print(f"  table metadata unavailable ({str(e)[:60]}); assuming {list(VOX)} nm voxels", flush=True)
        res = list(VOX)
    print(f"synapse positions in voxels of {res} nm", flush=True)

    # ---- 2. every output synapse of the 64 PRs ---------------------------------------------------------------
    now = datetime.datetime.now(datetime.timezone.utc)
    when = f"{now:%Y-%m-%d %H:%M} UTC"
    if args.cached and os.path.exists(RAW_OUT):
        out = pd.read_csv(RAW_OUT, sep="\t")
        when_out = str(out.queried.iloc[0])
        print(f"PR outputs from cache ({when_out}): {len(out)} synapses", flush=True)
    else:
        t = throttle(t)
        raw = retry(lambda: client.materialize.live_query("synapses_v2", timestamp=now,
                                                          filter_in_dict={"pre_pt_root_id": pr_roots}))
        out = flat(raw, res, when, ref_um)
        out.to_csv(RAW_OUT, sep="\t", index=False)
        when_out = when
        print(f"PR outputs: {len(out)} synapses from {out.pre.nunique()} of {len(pr_roots)} PRs ({when})", flush=True)
    out = out[out.pre != out.post].copy()
    out["pre_group"] = out.pre.map(pr_group)
    out["pre_class"] = out.pre.map(pr_class)
    is_pr_post = out.post.isin(pr_roots)
    pp = out[is_pr_post].copy()
    pp["post_group"] = pp.post.map(pr_group)
    pp["post_class"] = pp.post.map(pr_class)
    ext = out[~is_pr_post].copy()
    cnt = ext.groupby("post").size()
    targets = sorted(int(x) for x in cnt[cnt >= MIN_TOTAL].index)
    print(f"{len(cnt)} postsynaptic cells, {len(targets)} with >= {MIN_TOTAL} synapses from the groups "
          f"(these receive {int(cnt[cnt >= MIN_TOTAL].sum())} of {int(cnt.sum())} external PR synapses)", flush=True)

    # ---- 3. FlyTable labels -----------------------------------------------------------------------------------
    import seaserpent as ss
    ftab = ss.Table("aedes_main", server=SERVER).to_frame()
    ftab.columns = [str(c) for c in ftab.columns]
    sv = ftab["supervoxel_id"].astype(object).map(_s)
    ok = sv.str.isdigit()
    t = throttle(t)
    ft_roots = np.asarray(client.chunkedgraph.get_roots([int(x) for x in sv[ok]])).astype(np.int64)
    ft_label, ft_pr = {}, {}
    for (_, row), rt in zip(ftab[ok].iterrows(), ft_roots):
        typ, cl, sub, side = _s(row.get("type")), _s(row.get("class")), _s(row.get("subclass")), _s(row.get("side"))
        if typ in ("R1-6", "R7-8"):
            ft_pr[int(rt)] = "R1-R6" if typ == "R1-6" else "R7/R8"
        lab = " / ".join(x for x in (cl, sub, typ) if x)
        if lab:
            ft_label[int(rt)] = lab + (f" ({side})" if side else "")
    print(f"FlyTable: {len(ft_label)} labelled roots, {len(ft_pr)} photoreceptors", flush=True)
    census = {}
    vpath = os.path.join(OUT, "pr_bundles_roots_verified.tsv")
    if os.path.exists(vpath):
        v = pd.read_csv(vpath, sep="\t", dtype=str)
        vcol = [c for c in v.columns if "verdict" in c.lower()]
        rcol = [c for c in v.columns if c.lower() in ("root", "root_id")]
        if vcol and rcol:
            census = {int(r): _s(x) for r, x in zip(v[rcol[0]], v[vcol[0]]) if _s(r).isdigit()}
            print(f"census verdicts: {len(census)} roots, {pd.Series(list(census.values())).value_counts().to_dict()}", flush=True)

    def kind(root):
        root = int(root)
        if root in pr_group:
            return f"group PR {pr_class[root]}"
        if root in ft_pr:
            return f"FlyTable PR {ft_pr[root]}"
        if root in census:
            return f"census {census[root]}"
        return "other labelled" if root in ft_label else "unlabelled"

    def is_pr(k):
        return k.startswith(("group PR", "FlyTable PR")) or (k.startswith("census") and "like" in k)

    def is_r16(k):
        return is_pr(k) and "R1-R6" in k

    # ---- 4. complete input of the targets and of the R7/R8 cells ----------------------------------------------
    r78 = sorted(int(r) for r in df[df["class"].isin(["R7", "R8"])].root_used)
    post_set = sorted(set(targets) | set(r78))
    if args.cached and os.path.exists(RAW_IN):
        inp = pd.read_csv(RAW_IN, sep="\t")
        print(f"target inputs from cache ({inp.queried.iloc[0]}): {len(inp)} synapses", flush=True)
    else:
        t = throttle(t)
        raw = retry(lambda: client.materialize.live_query("synapses_v2", timestamp=now,
                                                          filter_in_dict={"post_pt_root_id": post_set}))
        inp = flat(raw, res, when, ref_um)
        inp.to_csv(RAW_IN, sep="\t", index=False)
        print(f"target inputs: {len(inp)} synapses onto {inp.post.nunique()} of {len(post_set)} cells", flush=True)
    inp = inp[inp.pre != inp.post].copy()
    inp["kind"] = inp.pre.map(kind)
    inp["pr"] = inp.kind.map(is_pr)
    inp["r16"] = inp.kind.map(is_r16)

    # ---- 5. per-target footprint -------------------------------------------------------------------------------
    lat = pd.read_csv(os.path.join(OUT, "pr_lattice_known.tsv"), sep="\t")
    lam = {int(c[1:]): np.array([r.lam_x, r.lam_y, r.lam_z]) for c, r in zip(lat.cartridge, lat.itertuples()) if c.startswith("g")}
    piv = ext[ext.post.isin(targets)].groupby(["post", "pre_group"]).size().unstack(fill_value=0)
    rows = []
    for tg in targets:
        s = inp[inp.post == tg]
        pg = piv.loc[tg] if tg in piv.index else pd.Series(dtype=int)
        pg = pg[pg > 0].sort_values(ascending=False)
        strong = [int(g) for g, n in pg.items() if n >= 3]
        span = max((np.linalg.norm(lam[a] - lam[b]) for a in strong for b in strong), default=0.0) / SPACING_UM
        P = s[s.r16][["pre_x", "pre_y", "pre_z"]].to_numpy(float)
        if len(P) >= 3:
            d = np.linalg.norm(P - P.mean(0), axis=1)
            p90, dmax = float(np.percentile(d, 90)), float(d.max())
        else:
            p90 = dmax = float("nan")
        rows.append(dict(target=tg, label=ft_label.get(tg, "unlabelled"), total_in=len(s), PR_in=int(s.pr.sum()),
                         pct_PR=round(100 * s.pr.mean(), 1) if len(s) else np.nan, n_PR_cells=int(s[s.pr].pre.nunique()),
                         n_R16_cells=int(s[s.r16].pre.nunique()), group_syn=int(pg.sum()), n_groups=int(len(pg)),
                         n_groups_ge3=len(strong), top_group=int(pg.index[0]) if len(pg) else 0,
                         top_share=round(float(pg.iloc[0] / pg.sum()), 2) if len(pg) else np.nan,
                         per_group=" ".join(f"g{int(g)}:{int(n)}" for g, n in pg.items()),
                         lattice_span_cart=round(span, 2), footprint_p90_um=round(p90, 1),
                         footprint_p90_cart=round(p90 / SPACING_UM, 2), footprint_max_um=round(dmax, 1)))
    foot = pd.DataFrame(rows).sort_values(["group_syn", "total_in"], ascending=False)
    foot.to_csv(os.path.join(OUT, "pr_lamina_footprint.tsv"), sep="\t", index=False)

    # ---- 6. group pairs: shared targets vs lattice distance --------------------------------------------------------
    groups = sorted(lam)
    prow = []
    for i, a in enumerate(groups):
        for b in groups[i + 1:]:
            A = set(piv.index[piv.get(a, pd.Series(0, index=piv.index)) >= 3]) if a in piv.columns else set()
            B = set(piv.index[piv.get(b, pd.Series(0, index=piv.index)) >= 3]) if b in piv.columns else set()
            sh = A & B
            prow.append(dict(group_a=a, group_b=b, lattice_dist_cart=round(np.linalg.norm(lam[a] - lam[b]) / SPACING_UM, 2),
                             targets_a=len(A), targets_b=len(B), shared=len(sh),
                             jaccard=round(len(sh) / max(len(A | B), 1), 2),
                             shared_labels="; ".join(sorted({ft_label.get(x, "unlabelled") for x in sh}))[:120]))
    pairs = pd.DataFrame(prow).sort_values("lattice_dist_cart")
    pairs.to_csv(os.path.join(OUT, "pr_lamina_footprint_pairs.tsv"), sep="\t", index=False)

    # ---- 7. R7 / R8 --------------------------------------------------------------------------------------------
    def n_pp(g, c1, c2, same=True):
        m = (pp.pre_class == c1) & (pp.post_class == c2)
        m &= (pp.pre_group == g) & ((pp.post_group == g) if same else (pp.post_group != g))
        return int(m.sum())

    grow, srow = [], []
    for g in sorted(df.group.unique()):
        lamc = out[(out.pre_group == g) & (out.pre_class == "R1-R6")][["pre_x", "pre_y", "pre_z"]].to_numpy(float).mean(0)
        r7c = out[(out.pre_group == g) & (out.pre_class == "R7")][["pre_x", "pre_y", "pre_z"]].to_numpy(float)
        r7c = r7c.mean(0) if len(r7c) else lamc
        L = np.linalg.norm(r7c - lamc)
        u = (r7c - lamc) / L if L > 0 else np.zeros(3)
        grow.append(dict(group=g, R8_to_R7=n_pp(g, "R8", "R7"), R7_to_R8=n_pp(g, "R7", "R8"),
                         R16_to_R7=n_pp(g, "R1-R6", "R7"), R16_to_R8=n_pp(g, "R1-R6", "R8"),
                         R7_to_R16=n_pp(g, "R7", "R1-R6"), R8_to_R16=n_pp(g, "R8", "R1-R6"),
                         R16_to_R16=n_pp(g, "R1-R6", "R1-R6"),
                         R8_to_R7_other_groups=n_pp(g, "R8", "R7", same=False),
                         R7_to_R8_other_groups=n_pp(g, "R7", "R8", same=False),
                         lamina_to_R7_terminal_um=round(float(L), 1)))
        for r in pp[(pp.pre_group == g) & (pp.post_group == g) & (pp.pre_class.isin(["R7", "R8"])) &
                    (pp.post_class.isin(["R7", "R8"]))].itertuples():
            p = np.array([r.pre_x, r.pre_y, r.pre_z])
            srow.append(dict(group=g, direction=f"{r.pre_class}->{r.post_class}", pre=r.pre, post=r.post,
                             dist_from_lamina_um=round(float(np.linalg.norm(p - lamc)), 1),
                             frac_lamina_to_R7term=round(float(np.dot(p - lamc, u) / L), 2) if L > 0 else np.nan,
                             x_vox=int(round(r.pre_x * 1000 / VOX[0])), y_vox=int(round(r.pre_y * 1000 / VOX[1])),
                             z_vox=int(round(r.pre_z * 1000 / VOX[2]))))
    r78tab = pd.DataFrame(grow)
    syn78 = pd.DataFrame(srow)
    r78tab.to_csv(os.path.join(OUT, "pr_r7r8_synapses.tsv"), sep="\t", index=False)
    syn78.to_csv(os.path.join(OUT, "pr_r7r8_synapse_sites.tsv"), sep="\t", index=False)
    cross = pp[pp.pre_group != pp.post_group].groupby(["pre_class", "post_class"]).size()

    irow = []
    for r in df[df["class"].isin(["R7", "R8"])].sort_values(["group", "class"]).itertuples():
        s = inp[inp.post == r.root_used]
        own = s[s.pre.map(lambda x: pr_group.get(int(x)) == r.group)]
        own_cls = own.pre.map(lambda x: pr_class[int(x)]).value_counts()
        other_pr = s[s.pr & ~s.pre.isin(own.pre)]
        oth = s[s.kind == "other labelled"].groupby(s.pre.map(ft_label)).size().sort_values(ascending=False)
        irow.append(dict(group=r.group, cell=getattr(r, "_3"), root=r.root_used, total_in=len(s),
                         own_R8=int(own_cls.get("R8", 0)), own_R7=int(own_cls.get("R7", 0)),
                         own_R16=int(own_cls.get("R1-R6", 0)), other_PR=len(other_pr),
                         other_PR_kinds=", ".join(f"{k} {n}" for k, n in other_pr.kind.value_counts().items()),
                         other_labelled=int((s.kind == "other labelled").sum()), unlabelled=int((s.kind == "unlabelled").sum()),
                         top_labelled_inputs="; ".join(f"{k} {n}" for k, n in oth.head(3).items())))
    r78in = pd.DataFrame(irow)
    r78in.to_csv(os.path.join(OUT, "pr_r7r8_inputs.tsv"), sep="\t", index=False)

    # ---- summary --------------------------------------------------------------------------------------------------
    lmc = foot[foot.label.str.contains("L1-L5|lamina|monopolar|LMC", case=False, regex=True)]
    S = []
    S.append(f"# PR-group summation footprint and R7/R8 synapses\n\nSynapses queried {when_out}; roots resolved at the "
             f"sheet xyz; groups 1-8 ({len(pr_roots)} PR cells); dataset {DATASTACK}.\n")
    S.append(f"## Targets\n\n{len(cnt)} cells receive output from the groups' PRs; {len(targets)} receive >= {MIN_TOTAL} "
             f"synapses in total and are analysed. Label counts among them: "
             f"{foot.label.map(lambda s: s.split(' (')[0]).value_counts().head(12).to_dict()}\n")
    S.append("Distribution of distinct groups per target (>= 1 synapse / >= 3 synapses):\n")
    d1 = foot.n_groups.value_counts().sort_index()
    d3 = foot.n_groups_ge3.value_counts().sort_index()
    S.append(md_table(pd.DataFrame({"n_groups": sorted(set(d1.index) | set(d3.index))}).assign(
        targets_ge1=lambda x: x.n_groups.map(d1).fillna(0).astype(int), targets_ge3=lambda x: x.n_groups.map(d3).fillna(0).astype(int))))
    S.append("\n### Top targets (by synapses from the groups)\n")
    top = foot.head(40).copy()
    S.append(md_table(top, ["target", "label", "total_in", "PR_in", "pct_PR", "n_PR_cells", "n_R16_cells", "group_syn",
                            "n_groups", "n_groups_ge3", "top_share", "per_group", "lattice_span_cart",
                            "footprint_p90_um", "footprint_p90_cart"]))
    if len(lmc):
        S.append(f"\n### Lamina-monopolar-labelled targets ({len(lmc)})\n")
        S.append(md_table(lmc, ["target", "label", "total_in", "PR_in", "pct_PR", "n_PR_cells", "n_R16_cells", "group_syn",
                                "n_groups", "n_groups_ge3", "top_share", "per_group", "footprint_p90_um", "footprint_p90_cart"]))
    S.append("\n## Group pairs: targets shared (>= 3 synapses from each) vs lattice distance\n")
    S.append(md_table(pairs))
    S.append("\n## R7 / R8 synapses within each group (pre class -> post class)\n")
    S.append(md_table(r78tab))
    S.append(f"\nPR -> PR synapses BETWEEN groups, by class pair: {cross.to_dict() if len(cross) else 'none'}\n")
    if len(syn78):
        S.append("\n### Every R7<->R8 synapse within a group (position along lamina -> R7 terminal; 0 = R1-R6 presynaptic centroid)\n")
        S.append(md_table(syn78))
    S.append("\n## Complete input of the R7 and R8 cells\n")
    S.append(md_table(r78in))
    txt = "\n".join(S)
    open(os.path.join(OUT, "pr_lamina_footprint_summary.md"), "w").write(txt + "\n")
    print(txt)
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
