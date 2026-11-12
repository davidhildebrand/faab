"""The T4T5s tab of the 'FAAB optic lobe targets' sheet (David 2026-11-12: a new tab of T4/T5 candidates, sorted by
confidence of the T4/T5 match; expected ~2,700 right-side T4/T5 = fly 7.7 per column x ~350 Aedes ommatidia).

Pool (right optic lobe), from the synapse snapshot (scratch/t4t5/t4t5_lop_scan.py):
  rule   >= 50% of a cell's output synapses in the lobula plate (LOP_R) AND more inputs in the medulla + lobula than in
         the lobula plate. Fly calibration (male-CNS v1.0, tmp/fly_lop_criterion.py): passes all 6,793 right T4/T5 and
         3 other cells - no CT1 contact needed (CT1 contact alone is 70% precise).
  near   35-50% of outputs in the lobula plate, inputs as above, >= 20 outputs: T4/T5 whose axon is partly outside
         the lobula-plate mesh, or TmY-like cells - low confidence.
  + SeaTable cells labeled T4 / T5 on the right side (annotated), whatever the rule says.
T4 or T5: more medulla than lobula inputs = T4 (dendrites in the medulla), else T5.

Steps:
  pool SCAN.parquet SEATABLE.tsv      network-free -> figures/t4t5/<stamp>_t4t5_tab_pool.tsv
  fetch POOL.tsv                      CAVE read-only, resumable: L2 dotprops of the pool (batches of BATCH roots, one
                                      throttled call each; cache figures/t4t5/t4t5_tab_dotprops.pkl, which also takes
                                      the entries of type_copies_dotprops.pkl) + nuclei (nuclei_v1_aedes, filtered)
  knn POOL.tsv                        network-free shape check -> figures/t4t5/<stamp>_t4t5_tab_knn.tsv
  build POOL KNN SEATABLE BOOK        points on the cells = one of their own synapse sites (the snapshot stores the
                                      supervoxel there; nearest the nucleus when there is one, else the dendrite
                                      middle), seeded into the shared tmp/sv_at_xyz.json cache, roots now by batched
                                      get_roots (no segmentation reads: a 3,000-chunk cloudvolume lookup hung for an
                                      hour on 2026-11-12), SeaTable / sheet status, confidence, one Spelunker link per
                                      block (two state uploads) -> figures/t4t5/<stamp>_t4t5_tab.tsv
  xlsx BOOK TSV                       one-tab .xlsx in the look of BOOK's PRg1hop1 tab ->
                                      figures/sheets/<stamp>_FAABopticlobetargets-T4T5s.xlsx (Insert new sheet)
Blocks 'putative T4' and 'putative T5' (columns of the LMCs tab, scripts/sheets/lmc_tab.py, + tracked_in far right);
within a block annotated rows first (by status), then high, medium, low, each by shape vote, margin and lobula-plate
output share. The tab's own remarks are listed in faab.partners.reconcile OWN so they stay on this tab's rows."""
import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd

from faab.config import fig
from faab.partners.sheet import utc_stamp

CACHE = fig("t4t5_tab_dotprops.pkl")
OLD_CACHE = fig("type_copies_dotprops.pkl")
NUCLEI = fig("t4t5_tab_nuclei.tsv")
POINTS = 1500  # thin like type_copies.py so shapes compare with the kNN panel
BATCH = 20


def thin(dp):
    """as scripts/morph/type_copies.py (same shapes as the cached panel)"""
    import navis
    if len(dp.points) > POINTS:
        idx = np.random.default_rng(0).choice(len(dp.points), POINTS, replace=False)
        dp = navis.Dotprops(dp.points[idx], k=5)
    dp.units = "um"
    return dp


def cmd_pool(a):
    d = pd.read_parquet(a.scan).set_index("root")
    st = pd.read_csv(a.seatable, sep="\t", dtype=str).fillna("")
    st = st[st.root_now.str.fullmatch(r"\d{15,}")]
    st["label"] = st[["superclass", "class", "subclass", "subsubclass", "type"]].agg(" / ".join, axis=1) \
        .str.replace(r"( / )+$", "", regex=True).str.replace(r"( / )+", " / ", regex=True)
    lab = st[st.label.str.contains(r"\bT4|\bT5", regex=True) & (st.side == "R")]
    io_ok = d.in_ME_R + d.in_LO_R > d.in_LOP_R
    tier = pd.Series("", index=d.index)
    tier[d.rule] = "rule"
    tier[~d.rule & io_ok & (d.share_out_LOP >= 0.35) & (d.n_out >= 20)] = "near"
    P = d[tier != ""].assign(tier=tier[tier != ""])
    extra = [int(r) for r in lab.root_now if int(r) not in P.index]
    if extra:
        E = d.reindex([r for r in extra if r in d.index]).assign(tier="SeaTable only")
        missing = [r for r in extra if r not in d.index]
        E = pd.concat([E, pd.DataFrame(index=pd.Index(missing, name="root")).assign(tier="SeaTable only")])
        P = pd.concat([P, E])
    P["kind"] = np.where(P.in_ME_R.fillna(0) >= P.in_LO_R.fillna(0), "T4", "T5")
    P["seatable_label"] = P.index.map(dict(zip(lab.root_now.astype("int64"), lab.label))).fillna("")
    out = fig(f"{utc_stamp()}_t4t5_tab_pool.tsv")
    P.reset_index().to_csv(out, sep="\t", index=False)
    print(pd.crosstab(P.tier, P.kind, margins=True).to_string())
    print(f"SeaTable right T4/T5 labels {len(lab)} (outside the rule: {len(extra)}); written {out}")


def cmd_fetch(a):
    from fafbseg import flywire
    from faab.clients.cave import DATASTACK, client as cave_client, throttle
    P = pd.read_csv(a.pool, sep="\t", dtype={"root": str})
    roots = list(dict.fromkeys(P.root))
    cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
    if os.path.exists(OLD_CACHE):
        old = pickle.load(open(OLD_CACHE, "rb"))
        for r in roots:
            if r in old and r not in cache:
                cache[r] = old[r]
    todo = [r for r in roots if r not in cache]
    print(f"pool {len(roots)}; dotprops cached {len(roots) - len(todo)}; to fetch {len(todo)} in batches of {BATCH}", flush=True)
    t0, t, failed = time.time(), 0.0, []
    for b in range(0, len(todo), BATCH):
        ids = todo[b:b + BATCH]
        t = throttle(t)
        try:
            dps = flywire.get_l2_dotprops([int(r) for r in ids], dataset=DATASTACK, max_threads=1, progress=False,
                                          omit_failures=True)
            got = set()
            for dp in dps:
                dp = thin(dp / 1000.0)
                dp.id = dp.name = str(dp.id)
                cache[str(dp.id)] = dp
                got.add(str(dp.id))
            failed += [r for r in ids if r not in got]
        except Exception as e:
            print(f"  batch {b} failed: {str(e)[:100]}", flush=True)
            failed += ids
        t = time.time()
        pickle.dump(cache, open(CACHE, "wb"))
        if (b // BATCH) % 10 == 0 or b + BATCH >= len(todo):
            print(f"  {min(b + BATCH, len(todo))}/{len(todo)} fetched ({len(failed)} failed) at {time.time() - t0:.0f} s", flush=True)
    # nuclei of the pool (soma check and the xyz near the soma), filtered queries of 500 roots
    c = cave_client()
    rows = []
    for b in range(0, len(roots), 500):
        t = throttle(t)
        n = c.materialize.query_table("nuclei_v1_aedes", filter_in_dict={"pt_root_id": [int(r) for r in roots[b:b + 500]]})
        t = time.time()
        rows.append(n)
    N = pd.concat(rows)
    pos = [c for c in ("pt_position_x", "pt_position_y", "pt_position_z") if c in N.columns]
    if pos:
        N["xyz_nm"] = N[pos].astype(float).values.tolist()
    else:
        N["xyz_nm"] = N.pt_position.map(lambda v: list(map(float, v)))
    keep = [k for k in ("id", "pt_root_id", "xyz_nm", "volume") if k in N.columns]
    N[keep].to_csv(NUCLEI, sep="\t", index=False)
    print(f"nuclei on pool roots: {N.pt_root_id.nunique()} roots ({len(N)} nuclei); failed dotprops {len(failed)}: "
          f"{failed[:10]}; written {CACHE} and {NUCLEI}", flush=True)


PANEL_TSV = "/Users/davidh/Data/Vibe/FAAB/tmp/t4t5_seatable_20261026.tsv"  # SeaTable labels of the 10-26 panel
MID = 371.7  # x of the midline (um); right-side cells are mirrored into the left frame
K = 7
MIN_POINTS = 10


def cmd_knn(a):
    """shape check, network-free (faab/scratch/t4t5/t4t5_knn.py on the pool): centroid-aligned NBLAST (translation
    only, keeps size and orientation) against SeaTable T4/T5 (references) and L1-L5 / LMC / Mi1 (negatives); per
    cell the vote of its 7 nearest panel cells, the top-5 margin, and T5-minus-T4 (top-3 means)"""
    import navis
    P = pd.read_csv(a.pool, sep="\t", dtype={"root": str})
    cache = pickle.load(open(CACHE, "rb"))
    old = pickle.load(open(OLD_CACHE, "rb"))
    st = pd.read_csv(PANEL_TSV, sep="\t", dtype=str).fillna("").drop_duplicates("root_now").set_index("root_now")
    refs = sorted(r for r in st.index[st.label.str.contains(r"\bT4|\bT5", regex=True)] if r in old)
    negs = sorted(r for r in st.index[st.label.str.contains("L1-L5|LMC", regex=True)] if r in old and r not in refs)
    mi1 = {f"Mi1:{n.id}": n for n in pickle.load(open(fig("mi1_dotprops.pkl"), "rb")) if len(n.points) >= MIN_POINTS}
    panel_dp = {**{r: old[r] for r in refs + negs}, **mi1}
    label = {k: (st.label.get(k, "") if k in st.index else "Mi1") for k in panel_dp}
    side = {k: (st.side.get(k) if k in st.index and st.side.get(k) in ("L", "R")
                else ("L" if float(v.points[:, 0].mean()) > MID else "R")) for k, v in panel_dp.items()}

    def centered(dp, sd, name):
        p = np.asarray(dp.points, float).copy()
        v = np.asarray(dp.vect, float).copy()
        if sd == "R":
            p[:, 0] = 2 * MID - p[:, 0]
            v[:, 0] *= -1
        d = navis.Dotprops(p - p.mean(0), k=None, vect=v)
        d.units = "um"
        d.id = d.name = name
        return d

    panel = list(panel_dp)
    Pn = navis.NeuronList([centered(panel_dp[k], side[k], k) for k in panel])
    q = [r for r in dict.fromkeys(P.root) if r in cache and len(cache[r].points) >= MIN_POINTS]
    small = [r for r in dict.fromkeys(P.root) if r not in q]
    Q = navis.NeuronList([centered(cache[r], "R", r) for r in q])
    print(f"panel: {len(refs)} T4/T5, {len(negs)} LMC, {len(mi1)} Mi1; queries {len(q)} (too small / missing {len(small)})", flush=True)
    t0 = time.time()
    fwd = navis.nblast(Q, Pn, n_cores=a.cores, progress=False).values
    rev = navis.nblast(Pn, Q, n_cores=a.cores, progress=False).values.T
    sym = (fwd + rev) / 2
    print(f"  nblast {time.time() - t0:.0f} s", flush=True)
    isref = np.array([k in refs for k in panel])
    is_t5 = np.array(["T5" in label[k] for k in panel])
    pidx = {k: j for j, k in enumerate(panel)}

    def score(srow, self_key=None):
        s = srow.copy()
        if self_key in pidx:
            s[pidx[self_key]] = np.nan
        ok = ~np.isnan(s)
        r5 = np.sort(s[ok & isref])[::-1][:5].mean()
        n5 = np.sort(s[ok & ~isref])[::-1][:5].mean()
        order = np.argsort(np.where(ok, -s, np.inf))[:K]
        a4 = np.sort(s[ok & isref & ~is_t5])[::-1][:3].mean()
        a5 = np.sort(s[ok & isref & is_t5])[::-1][:3].mean()
        return r5, n5, int(isref[order].sum()), label[panel[order[0]]], a5 - a4

    rows = []
    for i, r in enumerate(q):
        r5, n5, v, near, t5 = score(sym[i], r)
        rows.append(dict(root=r, points=len(cache[r].points), ref_top5=r5, neg_top5=n5, margin=r5 - n5, ref_votes=v,
                         nearest=near, t5_minus_t4=t5))
    for r in small:
        rows.append(dict(root=r, points=len(cache[r].points) if r in cache else 0))
    K_ = pd.DataFrame(rows)
    K_["shape"] = np.where(K_.ref_votes.isna(), "too small to compare",
                           np.where(K_.ref_votes >= 4, "T4/T5-like", "not T4/T5-like"))
    # leave-one-out on the panel itself (same scoring), as a check of this run
    pp = navis.nblast(Pn, Pn, n_cores=a.cores, progress=False).values
    pp = (pp + pp.T) / 2
    loo = pd.DataFrame([dict(k=k, ref=isref[j], t5=is_t5[j], vote=score(pp[j], k)[2], t5m=score(pp[j], k)[4])
                        for j, k in enumerate(panel)])
    print(f"leave-one-out: T4/T5 called T4/T5-like {(loo[loo.ref].vote >= 4).mean():.0%} of {int(loo.ref.sum())}; "
          f"negatives {(loo[~loo.ref].vote >= 4).mean():.0%} of {int((~loo.ref).sum())}; T5 lean T5 "
          f"{(loo[loo.ref & loo.t5].t5m > 0).mean():.0%}, T4 lean T5 {(loo[loo.ref & ~loo.t5].t5m > 0).mean():.0%}")
    out = fig(f"{utc_stamp()}_t4t5_tab_knn.tsv")
    K_.to_csv(out, sep="\t", index=False)
    M = P.merge(K_, on="root", how="left")
    print(pd.crosstab([M.tier, M.kind], M["shape"]).to_string())
    print("votes:", M.ref_votes.value_counts().sort_index().to_dict())
    m = M.ref_votes.notna()
    print(f"kind (inputs) vs shape lean (t5_minus_t4 > 0 = T5): agree {((M.kind == 'T5') == (M.t5_minus_t4 > 0))[m].mean():.0%}")
    print(f"written {out}")


VOXNM = np.array([16.0, 16.0, 45.0])
COLS = ["root_id_orig", "root_id_curr", "xyz", "side", "putative_type", "annotation", "status", "pr_date", "proofreader",
        "chk_date", "checker", "notes", "SeaTable", "link", "confidence", "confirmed_type", "reason", "tracked_in"]
CONF_RANK = {"annotated": 0, "high": 1, "medium": 2, "low": 3}
COLOR = {"annotated": "#0072B2", "high": "#56B4E9", "medium": "#E69F00", "low": "#CC79A7"}  # Okabe-Ito, as the LMCs tab
STATUS_RANK = {"checked": 0, "proofread": 1, "complete": 2, "adequate": 2, "review": 3, "to_review": 3, "in progress": 4,
               "needs_extending": 5, "incomplete": 5, "missing soma": 6, "stopped": 7}
SHEET_TABS = ("PRg1toT4T5", "PRs", "PRg1hop1", "PRg1hop2", "PRg2hop1", "Mi1s", "LMCs")
TAB = "T4T5s"


def synapse_points(roots, tag, nuc):
    """{root: (xyz text, supervoxel)}: a point on each cell that needs no segmentation lookup - one of its own synapse
    sites, whose supervoxel the snapshot stores (the synapse table's pre/post supervoxel is the one AT that point).
    With a nucleus: the site nearest it (toward the cell body); else the input site nearest the median of the inputs
    (dendrite), else the output site nearest the median of the outputs."""
    from faab import synsnap
    snap = synsnap.Snapshot(log=lambda *_: None)
    ids = [int(r) for r in roots]
    ins = snap.synapses(post=ids, when=tag, positions=True)
    outs = snap.synapses(pre=ids, when=tag, positions=True)
    ins = ins[ins.pre_root != ins.post_root]
    outs = outs[outs.pre_root != outs.post_root]
    sites = pd.concat([
        pd.DataFrame({"root": ins.post_root, "x": ins.post_x, "y": ins.post_y, "z": ins.post_z, "sv": ins.post_sv, "io": "in"}),
        pd.DataFrame({"root": outs.pre_root, "x": outs.pre_x, "y": outs.pre_y, "z": outs.pre_z, "sv": outs.pre_sv, "io": "out"})])
    sites = sites[sites.sv > 0]
    out = {}
    for r, g in sites.groupby("root"):
        c = nuc.get(str(r))
        if c is None and (g.io == "in").any():
            g = g[g.io == "in"]
        P = g[["x", "y", "z"]].to_numpy(float)
        target = c if c is not None else np.median(P * VOXNM, axis=0)
        i = int(np.argmin(np.linalg.norm(P * VOXNM - target, axis=1)))
        x, y, z = (int(v) for v in P[i])
        out[str(r)] = (f"{x}, {y}, {z}", int(g.sv.iloc[i]))
    return out


def seed_sv_cache(points):
    """add (xyz text -> supervoxel) to the shared lookup cache of faab.clients.cave.supervoxels_at_xyz (supervoxels
    never change), so later root-at-xyz lookups of these rows need no segmentation read"""
    from faab.clients.cave import SV_CACHE, parse_xyz
    cache = json.load(open(SV_CACHE)) if os.path.exists(SV_CACHE) else {}
    for xyz, sv in points.values():
        cache[",".join(f"{float(v):g}" for v in parse_xyz(xyz))] = int(sv)
    tmp = SV_CACHE + ".tmp"
    json.dump(cache, open(tmp, "w"))
    os.replace(tmp, SV_CACHE)


def sheet_rows(book):
    """{root_id_curr: [(tab, status, proofreader, pr_date, checker, chk_date)]} of the workbook's tracking tabs"""
    import openpyxl
    from faab.partners.sheet import cell_text as s, scan_book
    out = {}
    for x in scan_book(openpyxl.load_workbook(book)):
        if x["tab"] not in SHEET_TABS:
            continue
        v = x["values"]
        r = s(v.get("root_id_curr")) or s(v.get("root_id_orig"))
        if r.isdigit():
            out.setdefault(r, []).append((x["tab"], s(v.get("status")), s(v.get("proofreader")), v.get("pr_date"),
                                          s(v.get("checker")), v.get("chk_date")))
    return out


def cmd_build(a):
    import datetime
    import re
    from faab.clients.cave import client as cave_client, throttle
    stamp = utc_stamp()
    P = pd.read_csv(a.pool, sep="\t", dtype={"root": str})
    K_ = pd.read_csv(a.knn, sep="\t", dtype={"root": str})
    T = P.merge(K_, on="root", how="left")
    N = pd.read_csv(NUCLEI, sep="\t", dtype={"pt_root_id": str})
    nuc = N.groupby("pt_root_id").xyz_nm.agg(list).to_dict()
    st = pd.read_csv(a.seatable, sep="\t", dtype=str).fillna("")
    st = st[st.root_now.str.fullmatch(r"\d{15,}")].drop_duplicates("root_now").set_index("root_now")
    from faab.partners.sheet import seatable_label
    # typical T4/T5 size: the rule-passing cells with a T4/T5-like shape (median and spread of outputs / inputs)
    core = T[(T.tier == "rule") & (T.ref_votes >= 6)]
    lo_out, hi_out = core.n_out.quantile(0.05), core.n_out.quantile(0.95)
    lo_in, hi_in = core.n_in.quantile(0.05), core.n_in.quantile(0.95)
    print(f"typical T4/T5 (rule + shape 6-7 of 7, {len(core)} cells): outputs {lo_out:.0f}-{hi_out:.0f}, inputs "
          f"{lo_in:.0f}-{hi_in:.0f} (5-95%)", flush=True)
    # points on the cells' own synapse sites (supervoxel known) and their roots now: get_roots in batches, throttled
    nuc1 = {r: np.array(json.loads(v[0]) if isinstance(v[0], str) else v[0], float) for r, v in nuc.items()}
    tag = str(T.checkpoint.dropna().iloc[0])
    pts = synapse_points(list(T.root), tag, nuc1)
    seed_sv_cache(pts)
    xyz = {r: p[0] for r, p in pts.items()}
    c, t = cave_client(), 0.0
    keys = list(pts)
    root_now = {}
    for b in range(0, len(keys), 1000):
        t = throttle(t)
        rr = c.chunkedgraph.get_roots([pts[k][1] for k in keys[b:b + 1000]])
        t = time.time()
        root_now.update({k: str(int(v)) for k, v in zip(keys[b:b + 1000], rr)})
    print(f"points on {len(pts)} of {len(T)} cells (synapse sites, cache seeded); roots changed since {tag}: "
          f"{sum(root_now[k] != k for k in root_now)}", flush=True)
    SR = sheet_rows(a.book)
    recs = []
    for x in T.itertuples(index=False):
        r = x.root
        cur = root_now.get(r) or r
        cur = r if cur == "0" else cur
        s_ = st.loc[cur] if cur in st.index else (st.loc[r] if r in st.index else None)
        label = seatable_label(s_) if s_ is not None else ""
        typed = bool(re.search(r"\bT4|\bT5", label))
        notes, why = [], []
        nn = len(nuc.get(r, []))
        if nn == 0:
            notes.append("no nucleus on this root (cell body cut off, or a fragment)")
        elif nn > 1:
            notes.append("several nuclei on this root: merge?")
        big = max((x.n_out or 0) / hi_out, (x.n_in or 0) / hi_in) if pd.notna(x.n_out) else 0
        small = pd.notna(x.n_out) and (x.n_out < lo_out / 2 or x.n_in < lo_in / 2)
        if big >= 2:
            notes.append(f"about {big:.0f} times the usual T4/T5 size: merge?")
        elif small:
            notes.append("smaller than the usual T4/T5: part missing?")
        if x.tier == "near":
            notes.append(f"only {100 * x.share_out_LOP:.0f}% of its outputs in the lobula plate")
        if isinstance(x.shape, str) and x.shape == "not T4/T5-like":
            notes.append(f"shape unlike the typed T4/T5 (nearest typed cell: {x.nearest})")
        if pd.notna(x.n_out):
            ins = x.in_ME_R if x.kind == "T4" else x.in_LO_R
            tot = max(x.in_LA_R + x.in_ME_R + x.in_LO_R + x.in_LOP_R, 1)
            why.append(f"{100 * x.share_out_LOP:.0f}% of {int(x.n_out)} outputs in the lobula plate; inputs "
                       f"{100 * ins / tot:.0f}% in the {'medulla' if x.kind == 'T4' else 'lobula'}")
        if pd.notna(x.ref_votes):
            why.append(f"shape: {int(x.ref_votes)} of the 7 most similar typed cells are T4/T5")
        elif isinstance(x.shape, str):
            why.append("shape: too small to compare")
        if typed:
            conf = "annotated"
            ptype = "T5" if re.search(r"\bT5", label) else "T4"
            why.insert(0, "SeaTable type")
        else:
            ptype = x.kind
            ok_size = pd.notna(x.n_out) and not small and big < 2
            v = x.ref_votes if pd.notna(x.ref_votes) else -1
            if x.tier == "rule" and v >= 6 and ok_size:
                conf = "high"
            elif x.tier == "rule" and v >= 4:
                conf = "medium"
            else:
                conf = "low"
        if typed and ptype != x.kind and pd.notna(x.n_out):
            notes.append(f"SeaTable says {ptype}; its inputs say {x.kind}")
        rows = SR.get(cur, [])
        best = min(rows, key=lambda y: STATUS_RANK.get(y[1].lower(), 9)) if rows else None
        status = best[1] if best else (s_["status"] if s_ is not None else "")
        recs.append(dict(root_id_orig=r, root_id_curr=cur, xyz=xyz.get(r, ""), side="R", putative_type=ptype,
                         annotation=label, status=status, pr_date=best[3] if best else "",
                         proofreader=(best[2] if best and best[2] else (s_["proofreader"] if s_ is not None else "")),
                         chk_date=best[5] if best else "", checker=best[4] if best else "",
                         notes="; ".join(notes), SeaTable=f"present: {label}" if label else "", link="",
                         confidence=conf, confirmed_type="", reason="; ".join(why),
                         tracked_in="; ".join(dict.fromkeys(y[0] for y in rows)),
                         _votes=x.ref_votes if pd.notna(x.ref_votes) else -1, _margin=x.margin if pd.notna(x.margin) else -9,
                         _share=x.share_out_LOP if pd.notna(x.share_out_LOP) else 0,
                         _st=STATUS_RANK.get(str(status).lower(), 9)))
    R = pd.DataFrame(recs)
    if not xyz:
        raise SystemExit("no points")
    R = R.drop_duplicates("root_id_curr")  # two checkpoint roots merged since: one row
    R["rank"] = R.confidence.map(CONF_RANK) * 10 + R._st.where(R.confidence == "annotated", 0)
    R = R.sort_values(["putative_type", "rank", "_votes", "_margin", "_share"], ascending=[True, True, False, False, False])
    for c in ("pr_date", "chk_date"):
        R[c] = R[c].map(lambda v: v.strftime("%Y-%m-%d") if isinstance(v, datetime.datetime) else ("" if pd.isna(v) else v))
    links = block_links(R, stamp)
    lines = [f">>> T4/T5 candidates, right optic lobe: {len(R)} cells by confidence of the T4/T5 match (expected about "
             f"2,700 from the fly) (updated {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M} UTC)"]
    for b in ("T4", "T5"):
        g = R[R.putative_type == b]
        n = g.confidence.value_counts()
        title = [""] * len(COLS)
        title[0] = f"putative {b}: {len(g)} cells (" + ", ".join(f"{k} {int(n.get(k, 0))}" for k in CONF_RANK) + ")"
        title[COLS.index("link")] = links.get(b, "")
        lines.append("\t".join(title).rstrip("\t"))
        lines.append("\t".join(COLS))
        for y in g[COLS].itertuples(index=False):
            lines.append("\t".join("" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v).replace("\t", " ")
                                   for v in y))
        lines.append("")
    lines += [
        "note: how the cells were found - a T4/T5 sends most of its output to the lobula plate and gets its input in "
        "the medulla (T4) or the lobula (T5). In the fly this rule (at least half of the outputs in the lobula plate, more "
        "inputs in the medulla and lobula than in the lobula plate) picks out all 6,793 right-side T4/T5 and only 3 other "
        "cells. putative_type: T4 when most inputs are in the medulla, T5 when in the lobula.",
        "note: confidence - 'annotated' = typed T4 / T5 in SeaTable; 'high' = passes the rule, T4/T5 size, and 6 or 7 of "
        "the 7 most similar typed cells (by shape) are T4/T5; 'medium' = passes the rule, shape or size less typical; "
        "'low' = 35-50% of outputs in the lobula plate, a shape unlike T4/T5, or too small to compare: proofread first.",
        "note: to confirm, write the type in confirmed_type (T4 or T5, with the subtype a-d if you can tell) and your name "
        "in checker (date in chk_date).",
        "note: the link on each block line shows its cells colored by confidence: annotated blue, high sky blue, "
        "medium orange, low purple."]
    out = fig(f"{stamp}_t4t5_tab.tsv")
    open(out, "w").write("\n".join(lines) + "\n")
    print(pd.crosstab(R.putative_type, R.confidence).reindex(columns=list(CONF_RANK), fill_value=0).to_string())
    print(f"tracked in other tabs: {int((R.tracked_in != '').sum())}; in SeaTable: {int((R.SeaTable != '').sum())}; "
          f"roots changed since the checkpoint: {int((R.root_id_orig != R.root_id_curr).sum())}; written {out}")


def block_links(R, stamp):
    from faab import nglink
    from faab.clients.cave import client as cave_client, throttle
    client, t, out = cave_client(), 0.0, {}
    for b in ("T4", "T5"):
        g = R[R.putative_type == b]
        neurons = [{"name": f"putative {b}: {k} ({(g.confidence == k).sum()})", "color": COLOR[k],
                    "ids": g[g.confidence == k].root_id_curr.tolist()} for k in CONF_RANK if (g.confidence == k).any()]
        spec = {"meshes": [{"kind": "OL", "segments": ["11", "14", "21", "24"], "alpha": 0.12}], "neurons": neurons,
                "layout": "3d", "select": neurons[0]["name"]}
        state = nglink.clean_state(nglink.build(spec))
        open(fig(f"{stamp}_t4t5_tab_{b}_ng.txt"), "w").write(nglink.to_url(state) + "\n")
        t = throttle(t)
        out[b] = client.state.build_neuroglancer_url(client.state.upload_state_json(state),
                                                     ngl_url="https://spelunker.cave-explorer.org")
        print(f"  {b}: {out[b]}", flush=True)
    return out


def cmd_xlsx(a):
    """one-tab .xlsx in the look of BOOK's PRg1hop1 tab (as scripts/sheets/lmc_tab.py xlsx)"""
    import copy
    import datetime
    import openpyxl
    from openpyxl.utils import get_column_letter
    from faab.pr.prs_tab import as_date
    wb = openpyxl.load_workbook(a.book)
    src = wb["PRg1hop1"]
    s_hdr = {str(src.cell(2, c).value).strip(): c for c in range(1, src.max_column + 1) if src.cell(2, c).value}
    plain = next(r for r in range(3, src.max_row + 1) if str(src.cell(r, 1).value or "").strip().isdigit()
                 and not any(src.cell(r, c).fill is not None and src.cell(r, c).fill.fill_type == "solid"
                             for c in range(1, src.max_column + 1)))
    out = wb.create_sheet(TAB)
    out.sheet_view.showGridLines = src.sheet_view.showGridLines
    near = {"side": "PRtype", "putative_type": "PRtype", "confidence": "status", "confirmed_type": "annotation",
            "reason": "notes", "tracked_in": "notes"}
    widths = {"side": 6, "putative_type": 12, "confidence": 11, "confirmed_type": 14, "reason": 60, "tracked_in": 22}
    for j, name in enumerate(COLS, 1):
        sc = s_hdr.get(name) or s_hdr.get(near.get(name, ""), None)
        d = src.column_dimensions.get(get_column_letter(sc)) if sc else None
        out.column_dimensions[get_column_letter(j)].width = widths.get(name) or (d.width if d is not None and d.width else 14)
    lines = open(a.tsv).read().rstrip("\n").split("\n")
    for i, ln in enumerate(lines, 1):
        cells = ln.split("\t")
        first = cells[0]
        kind = ("title" if first.startswith(">>>") or first.startswith("note:") else "header" if first == COLS[0] else
                "data" if first.isdigit() else "block" if first else "blank")
        style_row = {"title": 1, "header": 2, "data": plain, "block": 1, "blank": None}[kind]
        for j, text in enumerate(cells, 1):
            name = COLS[j - 1] if j <= len(COLS) else ""
            sc = s_hdr.get(name) or s_hdr.get(near.get(name, ""), None) or 1
            v = text if text != "" else None
            if kind == "data" and name in ("pr_date", "chk_date") and v:
                v = as_date(v) or v
            c = out.cell(i, j)
            if style_row:
                c._style = copy.copy(src.cell(style_row, sc if kind in ("data", "header") else 1)._style)
            if kind in ("block", "header", "title") and v:
                f = copy.copy(c.font)
                f.b = True
                c.font = f
            if isinstance(v, str) and v.startswith("http"):
                c.hyperlink = v
            c.value = v
            if isinstance(v, datetime.datetime):
                c.number_format = 'yy"-"mm"-"dd'
    for ws in list(wb.worksheets):
        if ws is not out:
            wb.remove(ws)
    path = fig(f"{utc_stamp()}_FAABopticlobetargets-{TAB}.xlsx")
    wb.save(path)
    print(f"wrote {path}: {len(lines)} lines")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool")
    p.add_argument("scan")
    p.add_argument("seatable")
    p = sub.add_parser("fetch")
    p.add_argument("pool")
    p = sub.add_parser("knn")
    p.add_argument("pool")
    p.add_argument("--cores", type=int, default=6)
    p = sub.add_parser("build")
    p.add_argument("pool")
    p.add_argument("knn")
    p.add_argument("seatable")
    p.add_argument("book")
    p = sub.add_parser("xlsx")
    p.add_argument("book")
    p.add_argument("tsv")
    a = ap.parse_args()
    {"pool": cmd_pool, "fetch": cmd_fetch, "knn": cmd_knn, "build": cmd_build, "xlsx": cmd_xlsx}[a.cmd](a)


if __name__ == "__main__":
    main()
