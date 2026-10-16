"""Soma check for every row of the partner table whose followups say "no soma?".

Per distinct root_id_curr: (1) local-volume test from the cached L2 chunks (figures/l2_leaves.pkl,
l2_chunkdata.pkl): a compact cluster >= 50 um3 within 4 um is a soma candidate; (2) ONE live nuclei
lookup (live_query, current roots) for all roots; (3) a Spelunker link (faab/nglink.py) centred on the
soma candidate if found, else on the sheet xyz, with the block's photoreceptor in blue.
Outputs: figures/no_soma_check.tsv, .md (table with links), .html (self-contained). Read-only.
"""
from faab.config import REPO
import datetime
import json
import os
import pickle
import sys
import time

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import caveclient
from faab.sources.cave import DATASTACK
from faab.partners.top10_xyz import VOX
import faab.nglink as nglink

HERE = REPO
OUT = os.path.join(HERE, "figures")
TSV = os.path.join(OUT, "faab_pr_top20_update.tsv")
META = os.path.join(OUT, "faab_pr_top20_update_meta.json")
R_UM, SOMA_UM3 = 4.0, 50.0


def vox(p_nm):
    return ", ".join(str(int(round(v))) for v in (np.asarray(p_nm, float) / VOX))


def local_volume(ids, chunks):
    rows = [(np.array(c["rep_coord_nm"], float) / 1000.0, c.get("size_nm3", 0) / 1e9)
            for i in ids for c in [chunks.get(i) or {}] if c.get("rep_coord_nm") is not None]
    if not rows:
        return dict(n_chunks=len(ids), total_um3=np.nan, local_um3=np.nan, cand=None)
    pts = np.vstack([r[0] for r in rows])
    sz = np.array([r[1] for r in rows])
    local = np.array([sz[np.linalg.norm(pts - p, axis=1) <= R_UM].sum() for p in pts])
    k = int(np.argmax(local))
    near = np.linalg.norm(pts - pts[k], axis=1) <= R_UM
    cen = np.average(pts[near], axis=0, weights=sz[near]) * 1000.0
    return dict(n_chunks=len(ids), total_um3=float(sz.sum()), local_um3=float(local[k]),
                cand=cen if local[k] >= SOMA_UM3 else None)


def retry(fn, label):
    for a in range(5):
        time.sleep(5)
        try:
            return fn()
        except Exception as e:
            print(f"  ({label}: attempt {a + 1} failed: {str(e)[:70]})", flush=True)
            time.sleep(10)
    return None


def main():
    df = pd.read_csv(TSV, sep="\t", dtype=str).fillna("")
    sel = df[df.followups.str.contains("no soma", case=False)]
    meta = json.load(open(META))
    pr_curr = meta["pr_roots"]
    groups = sel.groupby("root_id_curr")
    print(f"{len(sel)} rows with 'no soma?' -> {groups.ngroups} distinct cells", flush=True)

    leaves = pickle.load(open(os.path.join(OUT, "l2_leaves.pkl"), "rb"))
    chunks = pickle.load(open(os.path.join(OUT, "l2_chunkdata.pkl"), "rb"))
    roots = [int(r) for r in groups.groups]
    client = caveclient.CAVEclient(DATASTACK)
    now = datetime.datetime.now(datetime.timezone.utc)
    latest = retry(lambda: client.chunkedgraph.is_latest_roots(roots, timestamp=now), "is_latest_roots")
    expired = set() if latest is None else {r for r, ok in zip(roots, latest) if not ok}
    if expired:
        print(f"{len(expired)} roots edited since the table refresh (expired, no live nucleus lookup): "
              f"{sorted(expired)}", flush=True)
    live_roots = [r for r in roots if r not in expired]
    nuc = retry(lambda: client.materialize.live_query("nuclei_v1_aedes", timestamp=now,
                                                      filter_in_dict={"pt_root_id": live_roots}), "nuclei live_query")
    nuc_by = {}
    if nuc is not None:
        for _, r in nuc.iterrows():
            p = np.array([r.pt_position_x, r.pt_position_y, r.pt_position_z], float) if "pt_position_x" in nuc.columns \
                else np.array(r.pt_position, float)
            nuc_by.setdefault(int(r.pt_root_id), []).append(p)
    print(f"live nuclei lookup at {now:%Y-%m-%d %H:%M} UTC: {'failed' if nuc is None else f'{len(nuc)} rows'}", flush=True)

    rows = []
    for rid, g in groups:
        r0 = g.iloc[0]
        blocks = ", ".join(b[-9:] for b in g.upstream_PR)
        origs = ", ".join(sorted(set(g.root_id_orig)))
        who = "; ".join(sorted(set(f"{s} {p}".strip() for s, p in zip(g.status, g.proofreader) if (s or p))))
        notes = "; ".join(sorted(set(n for n in g.notes if n)))
        ids = leaves.get(int(rid))
        lv = local_volume(ids, chunks) if ids else dict(n_chunks=0, total_um3=np.nan, local_um3=np.nan, cand=None)
        nucs = nuc_by.get(int(rid), [])
        if lv["cand"] is not None:
            verdict = "soma candidate (chunk volume)"
        elif nucs:
            verdict = "nucleus in table, no volume cluster"
        elif not ids:
            verdict = "not in chunk cache"
        elif lv["total_um3"] < 10:
            verdict = f"tiny fragment ({lv['n_chunks']} chunks, {lv['total_um3']:.1f} um3)"
        else:
            verdict = "no soma found"
        centre = lv["cand"] if lv["cand"] is not None else (nucs[0] if nucs else None)
        sheet_xyz = [float(v) for v in r0.xyz.split(",")] if r0.xyz.strip() else None
        pos = [int(round(v)) for v in (np.asarray(centre) / VOX)] if centre is not None else sheet_xyz
        pr = pr_curr.get(r0.upstream_PR, "")
        spec = {"meshes": [{"kind": "brain", "alpha": 0.15}],
                "neurons": [{"name": f"{rid} ({verdict})", "color": "#e6194b", "ids": [rid]},
                            {"name": f"upstream PR {pr}", "color": "#1f77d6", "ids": [pr]}],
                "view": "anterior", "layout": "3d", "select": f"{rid} ({verdict})", "scale": 6000}
        if pos is not None:
            spec["position"] = pos
        url = nglink.to_url(nglink.build(spec))
        rows.append(dict(root_id_curr=rid, blocks=blocks, root_id_orig=origs, sheet_xyz=r0.xyz, status=who, notes=notes,
                         n_chunks=lv["n_chunks"], total_um3=round(lv["total_um3"], 1) if lv["total_um3"] == lv["total_um3"] else "",
                         local_um3=round(lv["local_um3"], 1) if lv["local_um3"] == lv["local_um3"] else "",
                         soma_candidate_xyz=vox(lv["cand"]) if lv["cand"] is not None else "",
                         nucleus_live=("; ".join(vox(p) for p in nucs)) if nucs else
                         ("root expired, refresh table" if int(rid) in expired else "none"), verdict=verdict, url=url))
    out = pd.DataFrame(rows)
    order = {"soma candidate (chunk volume)": 0, "nucleus in table, no volume cluster": 1, "no soma found": 2}
    out["o"] = [order.get(v, 3) for v in out.verdict]
    out = out.sort_values(["o", "local_um3"], ascending=[True, False]).drop(columns="o")
    out.to_csv(os.path.join(OUT, "no_soma_check.tsv"), sep="\t", index=False)

    show = ["root_id_curr", "blocks", "status", "n_chunks", "total_um3", "local_um3", "soma_candidate_xyz", "nucleus_live", "verdict"]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 34)
    print(out[show].to_string(index=False))
    print("\nverdicts:", out.verdict.str.split(" \\(").str[0].value_counts().to_dict())

    md = ["| root_id_curr | block(s) | status | chunks | total µm³ | local µm³ | soma candidate xyz | nucleus (live) | verdict | link |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in out.itertuples():
        md.append(f"| {r.root_id_curr} | {r.blocks} | {r.status} | {r.n_chunks} | {r.total_um3} | {r.local_um3} | "
                  f"{r.soma_candidate_xyz} | {r.nucleus_live} | {r.verdict} | [open]({r.url}) |")
    open(os.path.join(OUT, "no_soma_check.md"), "w").write("\n".join(md) + "\n")
    html = ["<!doctype html><meta charset='utf-8'><title>no soma? re-check</title>",
            "<style>body{font:14px sans-serif;margin:20px}table{border-collapse:collapse}td,th{border:1px solid #ccc;"
            "padding:4px 8px;font-size:13px}th{background:#eee}</style>",
            f"<h2>'no soma?' rows re-checked ({now:%Y-%m-%d %H:%M} UTC)</h2><table><tr><th>root_id_curr</th><th>block(s)</th>"
            "<th>status</th><th>notes</th><th>chunks</th><th>total µm³</th><th>local µm³</th><th>soma candidate xyz</th>"
            "<th>nucleus (live)</th><th>verdict</th><th>link</th></tr>"]
    for r in out.itertuples():
        html.append(f"<tr><td>{r.root_id_curr}</td><td>{r.blocks}</td><td>{r.status}</td><td>{r.notes}</td><td>{r.n_chunks}</td>"
                    f"<td>{r.total_um3}</td><td>{r.local_um3}</td><td>{r.soma_candidate_xyz}</td><td>{r.nucleus_live}</td>"
                    f"<td>{r.verdict}</td><td><a href='{r.url}' target='_blank'>open</a></td></tr>")
    html.append("</table>")
    open(os.path.join(OUT, "no_soma_check.html"), "w").write("\n".join(html))
    print("DONE")


if __name__ == "__main__":
    main()
