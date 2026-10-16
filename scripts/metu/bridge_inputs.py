"""Complete input of the strong PR -> MeTu bridge cells: who feeds them besides our photoreceptors?

ONE live synapse query (post_pt_root_id in the bridges). Each presynaptic root is classified: R7/R8 or R1-R6
from the 104 photoreceptor sources of the bridge search (figures/pr_metu_link_cells.tsv), any other SeaTable
photoreceptor (type R1-6 / R7-8, either side), MeTu (link cells), other SeaTable-labelled cell (class shown), or
unlabelled. Output figures/bridge_inputs.tsv (per bridge: totals and fractions) and a per-bridge list of the
largest non-PR inputs. Read-only, throttled."""
from faab.config import REPO
import datetime
import os
import sys
import time

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import caveclient
from faab.sources.cave import DATASTACK, throttle

HERE = REPO
OUT = os.path.join(HERE, "figures")
SERVER = "https://flytable.mrc-lmb.cam.ac.uk/"
MIN_SYN = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def _s(v):
    return "" if (v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v)) else str(v).strip()


def main():
    bridge = pd.read_csv(os.path.join(OUT, "pr_metu_bridge.tsv"), sep="\t", dtype={"root": str})
    strong = bridge[bridge.path_strength >= MIN_SYN]
    roots = [int(r) for r in strong.root]
    cells = pd.read_csv(os.path.join(OUT, "pr_metu_link_cells.tsv"), sep="\t", dtype={"root": str})
    cls = {int(r.root): ("PR " + r.cls if r.kind == "PR" else "MeTu") for r in cells.itertuples() if r.kind in ("PR", "MeTu")}

    import seaserpent as ss
    meta = ss.Table("aedes_main", server=SERVER).to_frame()
    meta.columns = [str(c) for c in meta.columns]
    client = caveclient.CAVEclient(DATASTACK)
    t = throttle(time.time())
    sv = meta["supervoxel_id"].astype(object).map(_s)
    ok = sv.str.isdigit()
    ft_roots = np.asarray(client.chunkedgraph.get_roots([int(x) for x in sv[ok]]))
    ft = {}
    for (i, row), rt in zip(meta[ok].iterrows(), ft_roots):
        typ, cl, sub, side = _s(row.get("type")), _s(row.get("class")), _s(row.get("subclass")), _s(row.get("side"))
        if typ in ("R1-6", "R7-8"):
            ft[int(rt)] = f"PR {'R7/R8' if typ == 'R7-8' else 'R1-R6'} (SeaTable {side})"
        else:
            lab = " / ".join(x for x in (cl, sub, typ) if x)
            if lab:
                ft[int(rt)] = lab
    now = datetime.datetime.now(datetime.timezone.utc)
    t = throttle(t)
    syn = client.materialize.live_query("synapses_v2", timestamp=now, filter_in_dict={"post_pt_root_id": roots})
    syn = syn[syn.pre_pt_root_id != syn.post_pt_root_id].copy()

    def label(pre):
        pre = int(pre)
        if pre in cls:
            return cls[pre] + (" (source)" if cls[pre].startswith("PR") else "")
        return ft.get(pre, "unlabelled")

    syn["pre_label"] = syn.pre_pt_root_id.map(label)
    syn["pre_kind"] = syn.pre_label.map(lambda s: "R7/R8" if s.startswith("PR R7/R8") else "R1-R6" if s.startswith("PR R1-R6")
                                        else "MeTu" if s == "MeTu" else "unlabelled" if s == "unlabelled" else "other labelled")
    rows = []
    pd.set_option("display.width", 250)
    for r in strong.itertuples():
        s = syn[syn.post_pt_root_id == int(r.root)]
        n = len(s)
        k = s.pre_kind.value_counts()
        others = s[s.pre_kind.isin(["other labelled", "unlabelled"])].groupby(["pre_pt_root_id", "pre_label"]).size().sort_values(ascending=False)
        top = "; ".join(f"{lab} {int(c)}" for (_, lab), c in others.head(3).items() if lab != "unlabelled")
        rows.append(dict(bridge=r.root, total_in=n, from_R7R8=int(k.get("R7/R8", 0)), from_R1R6=int(k.get("R1-R6", 0)),
                         n_PR_cells=int(s[s.pre_kind.isin(["R7/R8", "R1-R6"])].pre_pt_root_id.nunique()),
                         pct_PR=round(100 * (k.get("R7/R8", 0) + k.get("R1-R6", 0)) / max(n, 1), 1),
                         from_MeTu=int(k.get("MeTu", 0)), other_labelled=int(k.get("other labelled", 0)),
                         unlabelled=int(k.get("unlabelled", 0)), n_input_cells=int(s.pre_pt_root_id.nunique()),
                         top_other_inputs=top, syn_to_MeTu=int(r.syn_to_MeTu)))
    out = pd.DataFrame(rows).sort_values("from_R7R8", ascending=False)
    out.to_csv(os.path.join(OUT, "bridge_inputs.tsv"), sep="\t", index=False)
    pd.set_option("display.max_colwidth", 60)
    print(out.to_string(index=False))
    pr_in = syn[syn.pre_kind.isin(["R7/R8", "R1-R6"])]
    extra = pr_in[~pr_in.pre_label.str.contains("source")]
    print(f"\nPR input from cells OUTSIDE our 104 sources (SeaTable-labelled PRs): {len(extra)} synapses from "
          f"{extra.pre_pt_root_id.nunique()} cells: {extra.pre_label.value_counts().to_dict()}")
    print(f"live at {now:%Y-%m-%d %H:%M} UTC\nDONE")


if __name__ == "__main__":
    main()
