"""Output synapse numbers for every photoreceptor whose class is R7.

Sources: FlyTable aedes_main rows whose class-like columns equal 'R7' (current roots via supervoxel ->
get_roots) plus the tracked R7 of the partner table (figures/faab_pr_top20_update_meta.json). ONE live
synapse query (pre_pt_root_id in roots) -> per cell: output synapses, distinct postsynaptic partners,
top partner and its count; ONE query for inputs. Output figures/r7_outputs.tsv. Read-only, throttled."""
from faab.config import REPO
import datetime
import json
import os
import time

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import caveclient
from faab.clients.cave import DATASTACK, throttle

HERE = REPO
OUT = os.path.join(HERE, "figures")
SERVER = "https://flytable.mrc-lmb.cam.ac.uk/"
CLS = ("superclass", "class", "subclass", "subsubclass", "type")  # FlyTable: photoreceptors are R1..R8 in subsubclass, R1-6/R7-8 in type


def _s(v):
    return "" if (v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v)) else str(v).strip()


def main():
    import seaserpent as ss
    meta = ss.Table("aedes_main", server=SERVER).to_frame()
    meta.columns = [str(c) for c in meta.columns]
    cols = [c for c in CLS if c in meta.columns]
    # photoreceptors: type 'R7-8' (R7 and R8 are not separated in FlyTable); subsubclass 'R7' is an
    # ellipsoid-body tile label on central-complex cells (EPG/PEN/PEG), so exclude cb_intrinsic rows
    typ = meta["type"].astype(object).map(_s).str.upper() if "type" in meta.columns else pd.Series("", index=meta.index)
    ssc = meta["subsubclass"].astype(object).map(_s).str.upper() if "subsubclass" in meta.columns else pd.Series("", index=meta.index)
    sup = meta["superclass"].astype(object).map(_s).str.lower() if "superclass" in meta.columns else pd.Series("", index=meta.index)
    hit = (typ == "R7-8") | ((ssc == "R7") & ~sup.str.contains("intrinsic"))
    ft = meta[hit].copy()
    print(f"FlyTable aedes_main: {len(ft)} photoreceptor rows typed R7-8 (or subsubclass R7 outside cb_intrinsic)", flush=True)
    label = {i: " / ".join(x for x in (_s(ft.loc[i, c]) for c in cols) if x) for i in ft.index}

    client = caveclient.CAVEclient(DATASTACK)
    t = throttle(time.time())
    rows = []
    sv = [int(_s(v)) for v in ft.get("supervoxel_id", pd.Series(dtype=object)) if _s(v).isdigit()]
    sv_root = {}
    if sv:
        t = throttle(t)
        sv_root = dict(zip(sv, [int(x) for x in np.asarray(client.chunkedgraph.get_roots(sv))]))
    for i, r in ft.iterrows():
        s = _s(r.get("supervoxel_id", ""))
        root = sv_root.get(int(s)) if s.isdigit() else (int(_s(r.get("root_id", ""))) if _s(r.get("root_id", "")).isdigit() else None)
        rows.append(dict(source="FlyTable", flytable_root=_s(r.get("root_id", "")), root=root, label=label[i],
                         side=_s(r.get("side", "")), status=_s(r.get("status", "")), name=_s(r.get("cell_name", r.get("name", "")))))
    meta_pr = json.load(open(os.path.join(OUT, "faab_pr_top20_update_meta.json")))
    prs = pd.read_csv(os.path.join(HERE, "..", "tmp", "sheet_prs.tsv"), sep="\t", dtype=str).fillna("")
    for _, p in prs[prs["class"].str.strip().str.upper() == "R7"].iterrows():
        cur = int(meta_pr["pr_roots"].get(p.root_id_orig, p.root_id_curr or p.root_id_orig))
        if cur not in {r["root"] for r in rows}:
            rows.append(dict(source="tracked sheet", flytable_root="", root=cur, label="R7 (sheet)", side="", status="", name=p.root_id_orig))
    df = pd.DataFrame(rows).dropna(subset=["root"])
    df["root"] = df.root.astype("int64")
    df = df.drop_duplicates("root")
    roots = sorted(df.root.tolist())
    now = datetime.datetime.now(datetime.timezone.utc)
    t = throttle(t)
    pre = client.materialize.live_query("synapses_v2", timestamp=now, filter_in_dict={"pre_pt_root_id": roots})
    pre = pre[pre.pre_pt_root_id != pre.post_pt_root_id]
    t = throttle(t)
    post = client.materialize.live_query("synapses_v2", timestamp=now, filter_in_dict={"post_pt_root_id": roots})
    post = post[post.pre_pt_root_id != post.post_pt_root_id]
    g = pre.groupby("pre_pt_root_id")
    n_out = g.size()
    n_part = g.post_pt_root_id.nunique()
    top = {r: sub.post_pt_root_id.value_counts().iloc[:1] for r, sub in g}
    n_in = post.groupby("post_pt_root_id").size()
    df["out_syn"] = [int(n_out.get(r, 0)) for r in df.root]
    df["out_partners"] = [int(n_part.get(r, 0)) for r in df.root]
    df["top_partner"] = [str(top[r].index[0]) if r in top and len(top[r]) else "" for r in df.root]
    df["top_partner_syn"] = [int(top[r].iloc[0]) if r in top and len(top[r]) else 0 for r in df.root]
    df["in_syn"] = [int(n_in.get(r, 0)) for r in df.root]
    df = df.sort_values("out_syn", ascending=False)
    df.to_csv(os.path.join(OUT, "r7_outputs.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(df[["source", "root", "label", "side", "status", "out_syn", "out_partners", "top_partner", "top_partner_syn", "in_syn"]]
          .to_string(index=False))
    print(f"\n{len(df)} R7 cells; live at {now:%Y-%m-%d %H:%M} UTC\nDONE")


if __name__ == "__main__":
    main()
