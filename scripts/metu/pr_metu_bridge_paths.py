"""Long-format path table for the PR -> bridge -> MeTu search (network-free, from the saved tables of
faab_pr_metu_bridge.py): one row per (PR_input, bridge_cell, MeTu) with the synapse count of each leg.
Columns: PR_input | PR_label | bridge_cell | nsyn_PR_to_bridge | MeTu | nsyn_bridge_to_MeTu | bridge_in_table
Rows for bridges whose weaker leg carries >= MIN_SYN synapses are printed as a pipe table; every bridge
goes to figures/pr_metu_bridge_paths.tsv. Usage: python faab_pr_metu_bridge_paths.py [MIN_SYN]"""
from faab.config import REPO
import os
import sys

import pandas as pd

HERE = REPO
OUT = os.path.join(HERE, "figures")
MIN_SYN = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def table(headers, rows, right=()):
    data = [headers] + rows
    w = [max(len(str(r[i])) for r in data) for i in range(len(headers))]
    fmt = lambda r: " | ".join(f"{str(r[i]):{'>' if headers[i] in right else '<'}{w[i]}}" for i in range(len(headers)))
    return [fmt(headers), "-+-".join("-" * x for x in w)] + [fmt(r) for r in rows]


def main():
    bridge = pd.read_csv(os.path.join(OUT, "pr_metu_bridge.tsv"), sep="\t", dtype={"root": str})
    down = pd.read_csv(os.path.join(OUT, "pr_metu_down_synapses.tsv"), sep="\t", dtype={"pre_pt_root_id": str, "post_pt_root_id": str})
    up = pd.read_csv(os.path.join(OUT, "pr_metu_up_synapses.tsv"), sep="\t", dtype={"pre_pt_root_id": str, "post_pt_root_id": str})
    cells = pd.read_csv(os.path.join(OUT, "pr_metu_link_cells.tsv"), sep="\t", dtype={"root": str, "link_id": str})
    gtab = pd.read_csv(os.path.join(HERE, "..", "tmp", "pr_groups_pasted.tsv"), sep="\t", dtype=str)
    label = {r.root_id_curr: f"g{r.group} {r._3}" for r in gtab.itertuples()}
    for r in cells[cells.kind == "PR"].itertuples():  # SeaTable (aedes_main) photoreceptors: cartridge group + type
        ft = str(r.flytable)
        label.setdefault(r.root, "SeaTable " + " ".join(ft.split(" / ")[-2:]) if " / " in ft else "SeaTable PR")
    # in_PRdownstream: the bridge cell is a row of the PRdownstream partner table -> which PR block, status, proofreader
    part = pd.read_csv(os.path.join(OUT, "faab_pr_top20_update.tsv"), sep="\t", dtype=str).fillna("")
    in_tab = {r.root_id_curr: (f"{r['class']} {r.upstream_PR[-9:]} block" + (f", {r.status}" if r.status else "") +
                               (f", {r.proofreader}" if r.proofreader else "")) for _, r in part.iterrows()}

    d_cnt = down.groupby(["post_pt_root_id", "pre_pt_root_id"]).size()
    u_cnt = up.groupby(["pre_pt_root_id", "post_pt_root_id"]).size()
    rows = []
    for b in bridge.itertuples():
        for (pr, n1) in d_cnt.loc[b.root].items():
            for (mt, n2) in u_cnt.loc[b.root].items():
                rows.append(dict(PR_input=pr, PR_label=label.get(pr, "PR"), bridge_cell=b.root, nsyn_PR_to_bridge=int(n1),
                                 MeTu=mt, nsyn_bridge_to_MeTu=int(n2), in_PRdownstream=in_tab.get(b.root, ""),
                                 bridge_min_leg=int(b.path_strength)))
    paths = pd.DataFrame(rows).sort_values(["bridge_min_leg", "bridge_cell", "nsyn_PR_to_bridge", "nsyn_bridge_to_MeTu"],
                                           ascending=[False, True, False, False])
    paths.to_csv(os.path.join(OUT, "pr_metu_bridge_paths.tsv"), sep="\t", index=False)
    show = paths[paths.bridge_min_leg >= MIN_SYN]
    cols = ["PR_input", "PR_label", "bridge_cell", "nsyn_PR_to_bridge", "MeTu", "nsyn_bridge_to_MeTu", "in_PRdownstream"]
    txt = "\n".join(table(cols, show[cols].values.tolist(), right=("nsyn_PR_to_bridge", "nsyn_bridge_to_MeTu")))
    open(os.path.join(OUT, "pr_metu_bridge_paths_pipes.txt"), "w").write(txt + "\n")
    print(txt)
    print(f"\n{len(show)} rows for {show.bridge_cell.nunique()} bridges with >= {MIN_SYN} synapses on both legs; "
          f"{len(paths)} rows for all {paths.bridge_cell.nunique()} bridges in the tsv")


if __name__ == "__main__":
    main()
