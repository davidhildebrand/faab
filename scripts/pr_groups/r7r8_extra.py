"""R7 <-> R8 synapses in the SeaTable cartridges (groups 9-13 of the pairing table), replicating the groups 1-8 test
(faab_pr_lamina_footprint.py) on cartridges whose R7/R8 identity comes from presynaptic depth: 'R7-like' = output
synapses beyond the R1-R6 terminal zone (long visual fibre), 'R8-like' = outputs within it (figures/pr_bundles_pairs_paste.tsv
notes, 2026-10-01). Cartridge 7 has two R8-like cells and no R7-like one; cartridge 3's R7-like cell is the SeaTable
'cartridge 2' cell whose axon runs in the cartridge-3 bundle.
Read-only, <= 1 CAVE call / 5 s: one root lookup at the sheet xyz, one live_query of the PR outputs, one of the
complete inputs of the R7/R8-like cells. Output figures/pr_r7r8_extra.tsv + printed markdown tables."""
from faab.config import REPO
import datetime
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
from faab.pr.lamina_footprint import flat, md_table, retry

HERE = REPO
OUT = os.path.join(HERE, "figures")
GROUPS = range(9, 14)


def load_paste():
    rows, g, header = [], None, None
    for ln in open(os.path.join(OUT, "pr_bundles_pairs_paste.tsv")).read().split("\n"):
        cells = ln.rstrip("\n").split("\t")
        if ln.startswith("group "):
            g = int(cells[0].split()[1])
        elif cells[0] == "root_id_orig":
            header = cells
        elif ln.strip() and header and g in GROUPS:
            d = dict(zip(header, cells + [""] * (len(header) - len(cells))))
            note = d.get("notes", "")
            cls = d["class"]
            if cls == "R7/R8":
                cls = "R7" if "R7-like" in note else "R8" if "R8-like" in note else "R7/R8?"
            rows.append(dict(group=g, root_id_curr=d["root_id_curr"], cls=cls, xyz=d["xyz"], note=note[:60]))
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    df = load_paste()
    print(f"{len(df)} cells in groups {list(GROUPS)}: {df.groupby(['group', 'cls']).size().to_dict()}", flush=True)
    client = caveclient.CAVEclient(DATASTACK)
    locs = np.array([[float(v) for v in x.split(",")] for x in df.xyz]) * VOX
    roots = np.asarray(flywire.locs_to_segments(locs, coordinates="nm", backend="cloudvolume", dataset=DATASTACK)).astype(np.int64)
    df["root"] = [int(r) if r != 0 else int(c) for r, c in zip(roots, df.root_id_curr)]
    changed = int((df.root.astype(str) != df.root_id_curr).sum())
    print(f"roots at xyz: {changed} of {len(df)} differ from the pairing table (2026-10-01)", flush=True)
    pr_group = dict(zip(df.root, df.group))
    pr_class = dict(zip(df.root, df.cls))
    lat = pd.read_csv(os.path.join(OUT, "pr_lattice_known.tsv"), sep="\t")
    ref_um = lat[["lam_x", "lam_y", "lam_z"]].to_numpy(float).mean(0)
    now = datetime.datetime.now(datetime.timezone.utc)
    when = f"{now:%Y-%m-%d %H:%M} UTC"
    t = throttle(time.time())
    raw = retry(lambda: client.materialize.live_query("synapses_v2", timestamp=now,
                                                      filter_in_dict={"pre_pt_root_id": sorted(set(df.root))}))
    out = flat(raw, list(VOX), when, ref_um)
    out = out[out.pre != out.post]
    out["pre_group"], out["pre_class"] = out.pre.map(pr_group), out.pre.map(pr_class)
    pp = out[out.post.isin(df.root)].copy()
    pp["post_group"], pp["post_class"] = pp.post.map(pr_group), pp.post.map(pr_class)
    print(f"{len(out)} output synapses, {len(pp)} onto cells of these groups ({when})", flush=True)

    def n(g, c1, c2, same=True):
        m = (pp.pre_class == c1) & (pp.post_class == c2) & (pp.pre_group == g)
        m &= (pp.post_group == g) if same else (pp.post_group != g)
        return int(m.sum())

    rows = []
    for g in GROUPS:
        lamc = out[(out.pre_group == g) & (out.pre_class == "R1-R6")][["pre_x", "pre_y", "pre_z"]].to_numpy(float).mean(0)
        r7 = out[(out.pre_group == g) & (out.pre_class == "R7")][["pre_x", "pre_y", "pre_z"]].to_numpy(float)
        L = float(np.linalg.norm(r7.mean(0) - lamc)) if len(r7) else float("nan")
        rows.append(dict(group=g, cartridge=f"LRS c{g - 6}", R7like_cells=int(((df.group == g) & (df.cls == "R7")).sum()),
                         R8like_cells=int(((df.group == g) & (df.cls == "R8")).sum()),
                         R8_to_R7=n(g, "R8", "R7"), R7_to_R8=n(g, "R7", "R8"), R8_to_R8=n(g, "R8", "R8"),
                         R16_to_R7=n(g, "R1-R6", "R7"), R16_to_R8=n(g, "R1-R6", "R8"), R7_to_R16=n(g, "R7", "R1-R6"),
                         R8_to_R16=n(g, "R8", "R1-R6"), R16_to_R16=n(g, "R1-R6", "R1-R6"),
                         R8_to_R7_other=n(g, "R8", "R7", False), R7_to_R8_other=n(g, "R7", "R8", False),
                         lamina_to_R7_terminal_um=round(L, 1)))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT, "pr_r7r8_extra.tsv"), sep="\t", index=False)
    sites = []
    for r in pp[pp.pre_class.isin(["R7", "R8"]) & pp.post_class.isin(["R7", "R8"])].itertuples():
        sites.append(dict(pre_group=r.pre_group, post_group=r.post_group, direction=f"{r.pre_class}->{r.post_class}",
                          pre=r.pre, post=r.post, x_vox=int(round(r.pre_x * 1000 / VOX[0])),
                          y_vox=int(round(r.pre_y * 1000 / VOX[1])), z_vox=int(round(r.pre_z * 1000 / VOX[2]))))
    sites = pd.DataFrame(sites)

    # complete input of the R7/R8-like cells
    r78 = sorted(int(r) for r in df[df.cls.isin(["R7", "R8", "R7/R8?"])].root)
    t = throttle(t)
    raw = retry(lambda: client.materialize.live_query("synapses_v2", timestamp=now, filter_in_dict={"post_pt_root_id": r78}))
    inp = flat(raw, list(VOX), when, ref_um)
    inp = inp[inp.pre != inp.post]
    irow = []
    for r in df[df.cls.isin(["R7", "R8", "R7/R8?"])].sort_values(["group", "cls"]).itertuples():
        s = inp[inp.post == r.root]
        own = s[s.pre.map(lambda x: pr_group.get(int(x)) == r.group)]
        oc = own.pre.map(lambda x: pr_class[int(x)]).value_counts()
        irow.append(dict(group=r.group, cell=r.cls, root=r.root, total_in=len(s), own_R8=int(oc.get("R8", 0)),
                         own_R7=int(oc.get("R7", 0)), own_R16=int(oc.get("R1-R6", 0)),
                         other_group_PR=int(s.pre.isin(df.root).sum() - len(own)), rest=int(len(s) - s.pre.isin(df.root).sum())))
    r78in = pd.DataFrame(irow)
    r78in.to_csv(os.path.join(OUT, "pr_r7r8_extra_inputs.tsv"), sep="\t", index=False)
    print("\n## R7-like / R8-like synapses within the SeaTable cartridges (groups 9-13)\n")
    print(md_table(tab))
    print(f"\nPR -> PR synapses between these groups by class: "
          f"{pp[pp.pre_group != pp.post_group].groupby(['pre_class', 'post_class']).size().to_dict()}")
    if len(sites):
        print("\n### R7<->R8 synapse sites\n")
        print(md_table(sites))
    print("\n## Complete input of the R7-like / R8-like cells\n")
    print(md_table(r78in))
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
