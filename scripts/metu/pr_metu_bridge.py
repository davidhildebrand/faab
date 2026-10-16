"""Cells downstream of the photoreceptors AND upstream of the MeTu neurons of a shared ng state.

1. Parse the pasted Spelunker URL (tmp/pr_metu_link.txt): visible segments of the wclee_aedes_brain
   layer ('!'-prefixed ids are hidden in the view and are listed separately).
2. Resolve to current roots (stale ids -> all current descendants).
3. Label via FlyTable aedes_main (supervoxel -> current root, one get_roots call): photoreceptor if
   type is R1-6 / R7-8, MeTu if any class column contains 'MeTu'; others reported.
4. Live synapses: pre in PR roots (everything downstream of the PRs), post in MeTu roots (everything
   upstream of the MeTus). Bridge cells = intersection. Direct PR->MeTu synapses counted too.
Output figures/pr_metu_bridge.tsv (bridge cells with counts, FlyTable label, membership in the partner
table) and figures/pr_metu_link_cells.tsv (every link cell with its label). Read-only, throttled."""
from faab.config import REPO
import datetime
import json
import os
import re
import time
import urllib.parse

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
CLS = ("superclass", "class", "subclass", "subsubclass", "type")
INCLUDE_HIDDEN_PRS = False  # the hidden ('!') photoreceptors are all side L (checked 2026-10-01): no MeTu on that side
EXTRA_PR_TABLE = os.path.join(TMP, "pr_groups_pasted.tsv")  # the 64 grouped PRs (group, root_id_orig, class, root_id_curr, xyz)


def _s(v):
    return "" if (v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v)) else str(v).strip()


def main():
    t0 = time.time()
    url = open(os.path.join(TMP, "pr_metu_link.txt")).read().strip()
    st = json.loads(urllib.parse.unquote(url.split("#!", 1)[1]))
    seg_layer = next(L for L in st["layers"] if L.get("name") == "wclee_aedes_brain")
    raw = seg_layer["segments"]
    visible = [int(s) for s in raw if not s.startswith("!")]
    hidden = [int(s[1:]) for s in raw if s.startswith("!")]
    print(f"link: {len(visible)} visible segments, {len(hidden)} hidden", flush=True)

    client = caveclient.CAVEclient(DATASTACK)
    now = datetime.datetime.now(datetime.timezone.utc)
    t = throttle(time.time())
    latest = dict(zip(visible, [bool(x) for x in np.asarray(client.chunkedgraph.is_latest_roots(visible, timestamp=now))]))
    cur_of = {}
    for r in visible:
        if latest[r]:
            cur_of[r] = [r]
        else:
            t = throttle(t)
            cur_of[r] = [int(x) for x in np.atleast_1d(client.chunkedgraph.get_latest_roots(r))]
    n_stale = sum(1 for r in visible if not latest[r])
    print(f"{n_stale} link ids were stale; resolved to {sum(len(v) for v in cur_of.values())} current roots", flush=True)

    import seaserpent as ss
    meta = ss.Table("aedes_main", server=SERVER).to_frame()
    meta.columns = [str(c) for c in meta.columns]
    sv = meta["supervoxel_id"].astype(object).map(_s)
    ok = sv.str.isdigit()
    t = throttle(t)
    roots_ft = np.asarray(client.chunkedgraph.get_roots([int(x) for x in sv[ok]]))
    lab = {}
    for (i, row), rt in zip(meta[ok].iterrows(), roots_ft):
        lab[int(rt)] = " / ".join(x for x in (_s(row[c]) for c in CLS if c in meta.columns) if x)
    print(f"FlyTable: {len(lab)} labelled current roots", flush=True)

    def kind(label):
        u = label.upper()
        if "R1-6" in u or "R7-8" in u or "PHOTORECEPTOR" in u:
            return "PR"
        if "METU" in u:
            return "MeTu"
        return "other" if label else "unlabelled"

    rows = []
    for r in visible:
        for c in cur_of[r]:
            rows.append(dict(link_id=r, root=c, stale=not latest[r], flytable=lab.get(c, ""), kind=kind(lab.get(c, "")),
                             hidden=False))
    # hidden ('!') segments: resolve and label them too; photoreceptors among them join the PR set when
    # INCLUDE_HIDDEN_PRS is set (user request 2026-10-01), other hidden cells stay out
    if hidden:
        t = throttle(t)
        h_latest = dict(zip(hidden, [bool(x) for x in np.asarray(client.chunkedgraph.is_latest_roots(hidden, timestamp=now))]))
        for h in hidden:
            cur = [h]
            if not h_latest[h]:
                t = throttle(t)
                cur = [int(x) for x in np.atleast_1d(client.chunkedgraph.get_latest_roots(h))]
            for c in cur:
                k = kind(lab.get(c, ""))
                if INCLUDE_HIDDEN_PRS and k == "PR":
                    rows.append(dict(link_id=h, root=c, stale=not h_latest[h], flytable=lab.get(c, ""), kind=k, hidden=True))
                else:
                    print(f"  hidden {h} -> {c}: {k} {lab.get(c, '')} (not used)", flush=True)
    for d in rows:  # class from the FlyTable label for link cells
        d["cls"] = "R7/R8" if "R7-8" in d["flytable"].upper() else ("R1-R6" if d["kind"] == "PR" else "")
    # the grouped photoreceptors (user's 8 groups): root at each cell's xyz (point rule), class from the table
    if EXTRA_PR_TABLE and os.path.exists(EXTRA_PR_TABLE):
        g = pd.read_csv(EXTRA_PR_TABLE, sep="\t", dtype=str)
        g["class"] = g["class"].str.replace("–", "-")
        locs = np.array([[float(v) for v in x.split(",")] for x in g.xyz]) * VOX
        at = flywire.locs_to_segments(locs, coordinates="nm", backend="cloudvolume", dataset=DATASTACK)
        n_new = 0
        for r, root in zip(g.itertuples(), at):
            root = int(root) if int(root) > 0 else int(r.root_id_curr)
            if str(root) != r.root_id_curr:
                print(f"  grouped PR {r.root_id_orig}: root at xyz {root} differs from table {r.root_id_curr}", flush=True)
            if any(d["root"] == root for d in rows):
                continue
            rows.append(dict(link_id=int(r.root_id_orig), root=root, stale=False, flytable=lab.get(root, ""), kind="PR",
                             hidden=False, cls=("R7/R8" if r._3 in ("R7", "R8") else "R1-R6")))
            n_new += 1
        print(f"grouped PRs added as sources: {n_new} (of {len(g)})", flush=True)
    cells = pd.DataFrame(rows).drop_duplicates("root")
    print(f"hidden PRs included: {int(cells.hidden.sum())}", flush=True)
    cells.to_csv(os.path.join(OUT, "pr_metu_link_cells.tsv"), sep="\t", index=False)
    print("link cells by kind:", cells.kind.value_counts().to_dict(), flush=True)
    prs = sorted(cells[cells.kind == "PR"].root)
    metus = sorted(cells[cells.kind == "MeTu"].root)
    if not prs or not metus:
        print("missing a class; stopping")
        return

    t = throttle(t)
    down = client.materialize.live_query("synapses_v2", timestamp=now, filter_in_dict={"pre_pt_root_id": prs})
    down = down[down.pre_pt_root_id != down.post_pt_root_id]
    t = throttle(t)
    up = client.materialize.live_query("synapses_v2", timestamp=now, filter_in_dict={"post_pt_root_id": metus})
    up = up[up.pre_pt_root_id != up.post_pt_root_id]
    print(f"{len(down)} synapses out of {len(prs)} PRs onto {down.post_pt_root_id.nunique()} cells; "
          f"{len(up)} synapses into {len(metus)} MeTus from {up.pre_pt_root_id.nunique()} cells", flush=True)
    direct = down[down.post_pt_root_id.isin(metus)]
    print(f"DIRECT PR -> MeTu synapses: {len(direct)}", flush=True)
    if len(direct):
        print(direct.groupby(["pre_pt_root_id", "post_pt_root_id"]).size().to_string(), flush=True)

    down.to_csv(os.path.join(OUT, "pr_metu_down_synapses.tsv"), sep="\t", index=False)
    up.to_csv(os.path.join(OUT, "pr_metu_up_synapses.tsv"), sep="\t", index=False)
    pr_class = {int(r.root): r.cls for r in cells[cells.kind == "PR"].itertuples()}
    down["pr_class"] = down.pre_pt_root_id.map(pr_class)
    d_cnt = down.groupby("post_pt_root_id").agg(syn_from_PR=("id", "size"), n_PR=("pre_pt_root_id", "nunique"),
                                                syn_from_R7R8=("pr_class", lambda s: int((s == "R7/R8").sum())),
                                                syn_from_R1R6=("pr_class", lambda s: int((s == "R1-R6").sum())))
    u_cnt = up.groupby("pre_pt_root_id").agg(syn_to_MeTu=("id", "size"), n_MeTu=("post_pt_root_id", "nunique"))
    bridge = d_cnt.join(u_cnt, how="inner")
    bridge.index.name = "root"
    bridge = bridge.reset_index()
    bridge["path_strength"] = np.minimum(bridge.syn_from_PR, bridge.syn_to_MeTu)  # weaker leg of the two-hop path
    bridge["flytable"] = [lab.get(int(r), "") for r in bridge.root]
    print(f"PR classes in the link: {pd.Series(pr_class).value_counts().to_dict()}; hidden R7-8 ids not used: "
          f"{[h for h in hidden if 'R7-8' in lab.get(h, '').upper()]}", flush=True)
    part = pd.read_csv(os.path.join(OUT, "faab_pr_top20_update.tsv"), sep="\t", dtype=str).fillna("")
    in_tab = set(part.root_id_curr)
    pinfo = part.groupby("root_id_curr").agg(notes=("notes", lambda s: "; ".join(sorted(set(x for x in s if x)))),
                                            status=("status", lambda s: "; ".join(sorted(set(x for x in s if x)))))
    bridge["in_partner_table"] = [str(r) in in_tab for r in bridge.root]
    bridge["table_status"] = [pinfo.status.get(str(r), "") for r in bridge.root]
    bridge["table_notes"] = [pinfo.notes.get(str(r), "") for r in bridge.root]
    bridge["in_link"] = [int(r) in set(cells.root) for r in bridge.root]
    bridge = bridge.sort_values(["path_strength", "syn_from_PR", "syn_to_MeTu"], ascending=False)
    bridge.to_csv(os.path.join(OUT, "pr_metu_bridge.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 40)
    print(f"\nBRIDGE cells (downstream of a PR and upstream of a MeTu): {len(bridge)}")
    print(bridge.to_string(index=False))
    print(f"\nlive at {now:%Y-%m-%d %H:%M} UTC; {time.time() - t0:.0f} s\nDONE")


if __name__ == "__main__":
    main()
