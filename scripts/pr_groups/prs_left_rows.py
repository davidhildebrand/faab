"""Rows for the PRs tab section '>>> additional L side putative PRs' (David, 2026-10-30): the left-eye PRs as groups
numbered after the right side's (103 = SeaTable LRS cartridge 8 with its census bundle, 104/105 = LRS cartridges 1 and
2, then the census bundles with >= 4 axons in the rank order of figures/pr_left_bundles_summary.md).

Members: every census axon of the bundle except 'other' (medulla neurons), with the class from its synapse verdict
(faab.pr.verdict: R1-R6-like -> R1-R6, few sites -> R1-R6?, R7-like -> R7, R8-like -> R8, fragment), ordered R1-R6,
R7, R8, R7/R8, R1-R6?, fragment. SeaTable cells (side L, type R1-6 / R7-8) keep SeaTable's type (R1-6 -> R1-R6; R7-8 ->
R7 or R8 by the synapse verdict, else R7/R8), status and proofreader, SeaTable column 'present: <SeaTable label>',
notes 'LRS: cartridge N' (no other SeaTable text in notes, David 2026-10-30). Notes stay short ('census bundle N';
the synapse verdict, counts and mesh shares are in the TSV's 'detail' column, not in the sheet). Census axons: status
'candidate'. xyz: the census point 75 um above the lamina (SeaTable cells: SeaTable's point), accepted only if the live
root there is the axon's current root, else the nearest L2-chunk point verified on it. root_id_curr = the root at the
xyz. CAVE read-only, >= 5 s between calls.
Writes figures/<stamp>_pr_left_groups.tsv (PRs-tab columns + group) and a short Spelunker link (classes as layers,
LA_L / ME_L meshes). Usage: prs_left_rows.py"""
import glob
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from faab import nglink, synsnap
from faab.clients.cave import client as cave_client, parse_xyz, roots_at_xyz, throttle
from faab.config import FIGURES, VOX
from faab.partners.sheet import seatable_label, utc_stamp
from faab.pr.verdict import mesh_shares, verdict
from faab.config import fig

FIRST_GROUP = 103
CLASS = {"R1-R6-like": "R1-R6", "R1-R6-like (few sites)": "R1-R6?", "R7-like": "R7", "R8-like": "R8", "fragment": "fragment"}
ORDER = ["R1-R6", "R7", "R8", "R7/R8", "R1-R6?", "fragment"]
_t = [0.0]


def wait():
    _t[0] = throttle(_t[0])


def pct(x):
    return "n/a" if pd.isna(x) else f"{100 * float(x):.0f}%"


def main():
    stamp = utc_stamp()
    roots = pd.read_csv(fig("pr_left_bundles_roots_verified.tsv"), sep="\t", dtype={"root": str})
    cl = pd.read_csv(fig("pr_left_bundles_clusters_verified.tsv"), sep="\t", index_col=0)
    st = pd.read_csv(sorted(glob.glob(fig("*_seatable_aedes_main.tsv")))[-1], sep="\t", dtype=str).fillna("")
    st = st[(st.side == "L") & st["type"].isin(["R1-6", "R7-8"])].copy()
    st["cart"] = st.notes.str.extract(r"cartridge (\d+)")[0]
    st["label"] = [seatable_label(r) for r in st.to_dict("records")]
    print(f"SeaTable side-L PRs: {len(st)}; cartridges {st.cart.value_counts().to_dict()}")

    # ---- groups: cartridge 8 bundle (census, with the SeaTable cells), cartridges 1 and 2, new bundles by rank
    roots = roots[roots.verdict != "other"]
    cl["tier"] = 3
    cl.loc[cl.spread_um <= 6.0, "tier"] = 2
    cl.loc[(cl.spread_um <= 6.0) & cl.n_roots.between(6, 9), "tier"] = 1
    known_cl = cl[cl.n_known > 0].index.tolist()
    new = cl[(cl.n_known == 0) & (cl.n_roots >= 4)].sort_values(["tier", "n_PR_like", "n_roots"], ascending=[True, False, False])
    print(f"census clusters with SeaTable cells: {known_cl}; new bundles >= 4 axons: {len(new)}; census PR-like axons not "
          f"in a listed bundle: {int((~roots.cluster.isin(known_cl + new.index.tolist()) & (roots.verdict != 'fragment')).sum())}")
    rows = []
    for c in known_cl:
        for r in roots[roots.cluster == c].itertuples():
            rows.append(dict(key="cart8", root=r.root, census=r, st=None))
    for cart in ("8", "1", "2"):
        for r in st[st.cart == cart].itertuples():
            hit = [x for x in rows if x["key"] == "cart8" and x["root"] == r.root_now]
            if hit:
                hit[0]["st"] = r
            else:
                rows.append(dict(key="cart8" if cart == "8" else f"cart{cart}", root=r.root_now, census=None, st=r))
    for c in new.index:
        for r in roots[roots.cluster == c].itertuples():
            rows.append(dict(key=f"cl{c}", root=r.root, census=r, st=None))
    keys = ["cart8", "cart1", "cart2"] + [f"cl{c}" for c in new.index]
    group = {k: FIRST_GROUP + i for i, k in enumerate(keys)}

    # ---- current roots, xyz verified on them
    c = cave_client()
    ids = sorted({x["root"] for x in rows})
    wait()
    ok = dict(zip(ids, np.asarray(c.chunkedgraph.is_latest_roots([int(i) for i in ids])).astype(bool)))
    now = {}
    for i in ids:
        if ok[i]:
            now[i] = i
            continue
        wait()
        new_ids = [int(x) for x in np.atleast_1d(c.chunkedgraph.get_latest_roots(int(i)))]
        if len(new_ids) > 1:
            sizes = []
            for n in new_ids:
                wait()
                sizes.append(len(c.chunkedgraph.get_leaves(n, stop_layer=2)))
            new_ids = [new_ids[int(np.argmax(sizes))]]
        now[i] = str(new_ids[0])
        print(f"  {i} was edited; now {now[i]}")
    for x in rows:
        x["now"] = now[x["root"]]
        x["xyz"] = x["census"].xyz_ref if x["census"] is not None else x["st"].point_xyz
    wait()
    at = roots_at_xyz([x["xyz"] for x in rows])
    bad = [x for x, a in zip(rows, at) if str(a) != x["now"]]
    print(f"xyz on the axon: {len(rows) - len(bad)} of {len(rows)}; fixing {len(bad)}")
    for x in bad:
        p = np.array(parse_xyz(x["xyz"])) * np.array(VOX)
        wait()
        l2 = c.chunkedgraph.get_leaves(int(x["now"]), stop_layer=2)
        wait()
        data = c.l2cache.get_l2data([int(v) for v in l2], attributes=["rep_coord_nm"])
        pts = np.array([v["rep_coord_nm"] for v in data.values() if "rep_coord_nm" in v], float)
        order = np.argsort(np.linalg.norm(pts - p, axis=1))[:5]
        cand = [np.round(pts[i] / np.array(VOX)).astype(int) for i in order]
        wait()
        got = roots_at_xyz([list(v) for v in cand])
        hit = next((v for v, g in zip(cand, got) if str(g) == x["now"]), None)
        x["xyz_note"] = "" if hit is None else "xyz = nearest point on the axon (census point lay off it)"
        if hit is None:
            x["xyz_note"] = "CHECK: no verified point on the axon"
        else:
            x["xyz"] = f"{hit[0]}, {hit[1]}, {hit[2]}"

    # ---- synapse verdicts for the SeaTable cells outside the census (snapshot + catch-up)
    need = [x for x in rows if x["census"] is None]
    S = synsnap.Snapshot()
    pre = S.synapses(pre=[int(x["now"]) for x in need], when="now", positions=True, client=c)
    post = S.synapses(post=[int(x["now"]) for x in need], when="now", client=c)
    sh = mesh_shares(pre.pre_root.astype(str).to_numpy(), pre[["pre_x", "pre_y", "pre_z"]].to_numpy(float) * np.array(VOX) / 1000.0,
                     "LA_L", "ME_L")
    npre, npost = pre.pre_root.astype(str).value_counts(), post.post_root.astype(str).value_counts()
    for x in need:
        r = x["now"]
        x["syn"] = dict(n_pre=int(npre.get(r, 0)), n_post=int(npost.get(r, 0)),
                        la=sh.la_frac.get(r, np.nan), me=sh.me_frac.get(r, np.nan))
        x["syn"]["verdict"] = verdict(x["syn"]["n_pre"], x["syn"]["n_post"], x["syn"]["la"], x["syn"]["me"])

    # ---- rows
    out = []
    for x in rows:
        k, cen, s = x["key"], x["census"], x["st"]
        if cen is not None:
            v = dict(n_pre=int(cen.n_pre), n_post=int(cen.n_post), la=cen.la_frac, me=cen.me_frac, verdict=cen.verdict)
        else:
            v = x["syn"]
        syn = (f"synapses: {v['verdict']}, {v['n_pre']} pre / {v['n_post']} post, {pct(v['la'])} of outputs in LA_L, "
               f"{pct(v['me'])} in or near ME_L")
        # notes stay short (they are for the proofreaders, David 2026-10-30); the details go to the 'detail' column of
        # the TSV, which is not written to the sheet
        notes, detail = [], []
        if s is not None:
            cls = "R1-R6" if s.type == "R1-6" else {"R7-like": "R7", "R8-like": "R8"}.get(v["verdict"], "R7/R8")
            notes.append(s.notes)  # 'LRS: cartridge N'; the SeaTable label goes into the SeaTable column
        else:
            cls = CLASS[v["verdict"]]
        if cen is not None:
            b = cl.loc[cen.cluster]
            notes.append(f"census bundle {cen.cluster}")
            detail.append(f"census bundle {cen.cluster} ({int(b.n_roots)} axons, tier {int(b.tier)}, spread {b.spread_um} um), 2026-10-29")
        elif k in ("cart1", "cart2"):
            notes.append("axon not traced toward the retina")
            detail.append("lamina-terminal piece: the axon is not traced toward the retina, so it was not in the census")
        else:
            detail.append("not in the census bundle (its axon stops before the bundle zone)")
        detail.append(syn)
        if cen is not None and isinstance(cen.note, str) and cen.note:
            detail.append(cen.note)
        if x.get("xyz_note"):
            (notes if x["xyz_note"].startswith("CHECK") else detail).append(x["xyz_note"])
        out.append(dict(group=group[k], root_id_orig=x["root"], **{"class": cls}, root_id_curr=x["now"], xyz=x["xyz"],
                        status=(s.status if s is not None and s.status else "candidate"),
                        proofreader=(s.proofreader if s is not None else ""), notes="; ".join(notes),
                        SeaTable=(f"present: {s.label}" if s is not None else ""), n_pre=v["n_pre"],
                        detail="; ".join(detail)))
    df = pd.DataFrame(out)
    df["o"] = df["class"].map({c: i for i, c in enumerate(ORDER)})
    df = df.sort_values(["group", "o", "n_pre"], ascending=[True, True, False]).drop(columns=["o"])
    path = fig(f"{stamp}_pr_left_groups.tsv")
    df.to_csv(path, sep="\t", index=False)
    print(f"{len(df)} rows in {df.group.nunique()} groups ({df.group.min()}-{df.group.max()}); classes "
          f"{df['class'].value_counts().to_dict()}; written {path}")

    # ---- link
    layers = []
    for cls, color in (("R1-R6", "#e9953a"), ("R7", "#2a7fd4"), ("R8", "#2ecc40"), ("R7/R8", "#17becf"),
                       ("R1-R6?", "#f7c873"), ("fragment", "#999999")):
        idx = sorted(set(df[df["class"] == cls].root_id_curr))
        if idx:
            layers.append({"name": f"L {cls} ({len(idx)})", "color": color, "ids": idx})
    spec = {"meshes": [{"kind": "OL", "segments": ["11", "14"], "alpha": 0.15}, {"kind": "neuropil"}],
            "neurons": layers, "view": "anterior", "layout": "3d", "image": True, "select": layers[0]["name"]}
    state = nglink.build(spec)
    open(path.replace(".tsv", "_ng.txt"), "w").write(nglink.to_url(state) + "\n")
    wait()
    sid = c.state.upload_state_json(state)
    url = c.state.build_neuroglancer_url(sid, ngl_url="https://spelunker.cave-explorer.org")
    open(path.replace(".tsv", "_short_link.txt"), "w").write(url + "\n")
    print("short link:", url)


if __name__ == "__main__":
    main()
