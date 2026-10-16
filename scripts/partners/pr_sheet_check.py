"""Check a PRdownstream sheet (TSV export, block layout) against the live segmentation, without changing anything.

For every row with an xyz point (16 x 16 x 45 nm voxels; the anchor):
  root_at_xyz   the segment at the point now (one batched lookup for all points)
  curr_latest   whether root_id_curr is a current root (batched is_latest_roots)
  lineage       whether root_at_xyz descends from root_id_orig (get_latest_roots per stale root_id_orig, throttled,
                cached in ../tmp/pr_sheet_check_lineage.json so an interrupted run resumes)
and a verdict:
  ok             root_id_curr is current, is the root at xyz, and descends from root_id_orig
  update         root_id_curr is stale; the root at xyz descends from it -> proposed root_id_curr = root at xyz
  MISMATCH       root_id_curr is current but the point lies on another current root
  STALE-NO-LINK  root_id_curr is stale and the root at xyz does not descend from it
  LINEAGE        the root at xyz does not descend from root_id_orig
  NO-SEGMENT     no segment at the point
  DUPLICATE      another row of the same block resolves to the same root (merged pieces)
The block PRs (upstream_PR = the PR's original root) are checked the same way at their xyz from data/pr_groups.tsv
(matched by root_id_orig). Nothing is written to the sheet; --write-sheet produces the updated copy (only
root_id_curr changes, only for 'update' rows) after the report has been reviewed.

Outputs: figures/pr_sheet_check_<stamp>.tsv (one row per sheet row) and a printed summary.
Usage: python pr_sheet_check.py <sheet.tsv> [--write-sheet OUT.tsv]
Read-only; CAVE calls >= 5 s apart."""
import argparse
import datetime
import json
import os
import re
import time

import numpy as np
import pandas as pd

from faab.config import DATASTACK, FIGURES, DATA, TMP, VOX
from faab.partners.sheet import parse_sheet

CACHE = os.path.join(TMP, "pr_sheet_check_lineage.json")
_last = [0.0]


def throttle(gap=5.0):
    wait = _last[0] + gap - time.time()
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.time()


def xyz_vals(s):
    v = re.findall(r"-?\d+(?:\.\d+)?", s or "")
    return [float(x) for x in v] if len(v) == 3 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet")
    ap.add_argument("--write-sheet", help="write the updated sheet here (only after reviewing the report)")
    a = ap.parse_args()
    import caveclient
    from fafbseg import flywire
    lines, rows = parse_sheet(a.sheet)
    prs = sorted({r["block_pr"] for r in rows})
    groups = pd.read_csv(os.path.join(DATA, "pr_groups.tsv"), sep="\t", dtype=str).fillna("")
    pr_xyz = {r.root_id_orig: r.xyz for r in groups.itertuples() if r.root_id_orig in prs}
    print(f"{len(rows)} partner rows in {len(prs)} blocks; PR points found for {len(pr_xyz)}/{len(prs)} PRs", flush=True)
    client = caveclient.CAVEclient(DATASTACK)
    now = datetime.datetime.now(datetime.timezone.utc)

    # 1. roots at all points (partner rows + PRs)
    pts = sorted({r["cells"]["xyz"] for r in rows if xyz_vals(r["cells"]["xyz"])} | set(pr_xyz.values()))
    throttle()
    found = np.asarray(flywire.locs_to_segments(np.array([xyz_vals(p) for p in pts]) * np.array(VOX), coordinates="nm",
                                                backend="cloudvolume", dataset=DATASTACK)).astype(np.int64)
    at = {p: str(int(r)) for p, r in zip(pts, found)}

    # 2. which listed roots are current
    ids = sorted({x for r in rows for x in (r["cells"]["root_id_curr"], r["cells"]["root_id_orig"]) if x.strip().isdigit()}
                 | set(prs))
    throttle()
    latest_flag = dict(zip(ids, map(bool, client.chunkedgraph.is_latest_roots([int(x) for x in ids]))))

    # 3. forward lineage of every stale listed root (cached, resumable)
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    stale = [x for x in ids if not latest_flag[x] and x not in cache]
    print(f"{len(ids)} listed roots, {sum(not v for v in latest_flag.values())} stale; {len(stale)} lineage lookups "
          f"(~{5 * len(stale) / 60:.0f} min)", flush=True)
    for k, x in enumerate(stale):
        throttle()
        try:
            cache[x] = [str(int(y)) for y in client.chunkedgraph.get_latest_roots(int(x))]
        except Exception as e:
            cache[x] = ["ERROR " + str(e)[:80]]
        if k % 10 == 9 or k == len(stale) - 1:
            json.dump(cache, open(CACHE, "w"))
            print(f"  {k + 1}/{len(stale)} lineage lookups", flush=True)

    def descendants(x):
        return {x} if latest_flag.get(x, False) else set(cache.get(x, []))

    # 4. verdicts
    out = []
    for r in rows:
        c = r["cells"]
        orig, curr, xyz = c["root_id_orig"].strip(), c["root_id_curr"].strip(), c["xyz"]
        ra = at.get(xyz, "") if xyz_vals(xyz) else ""
        v, note, proposed = "ok", "", curr
        if not xyz_vals(xyz):
            v, note = "NO-POINT", "xyz missing or unreadable"
        elif ra == "0":
            v, note = "NO-SEGMENT", "no segment at the point"
        elif latest_flag.get(curr, False):
            if ra != curr:
                v, note = "MISMATCH", f"root_id_curr is current, but the point lies on {ra}"
        else:
            if ra in descendants(curr):
                v, proposed, note = "update", ra, f"root_id_curr stale; the point's root {ra} descends from it"
            else:
                v, note = "STALE-NO-LINK", (f"root_id_curr stale (now {sorted(descendants(curr))[:3]}); the point lies on {ra}, "
                                            "which does not descend from it")
        if v in ("ok", "update") and orig.isdigit() and ra not in descendants(orig):
            v, note = "LINEAGE", (note + "; " if note else "") + f"the point's root {ra} does not descend from root_id_orig"
        elif v in ("ok", "update") and not orig.isdigit():
            note = (note + "; " if note else "") + f"root_id_orig '{orig}' is not a root id: lineage not checked"
        out.append(dict(line=r["line"] + 1, block_pr=r["block_pr"], root_id_orig=orig, root_id_curr=curr, xyz=xyz, nsyn=c["nsyn"],
                        status=c["status"].strip(), root_at_xyz=ra, curr_latest=latest_flag.get(curr, ""), verdict=v,
                        proposed_root_id_curr=proposed if v in ("ok", "update") else "", note=note))
    df = pd.DataFrame(out)
    key = df.proposed_root_id_curr.where(df.proposed_root_id_curr != "", df.root_at_xyz)
    dup = df.assign(k=key)[key != ""].groupby(["block_pr", "k"]).line.transform("count") > 1
    for i in df.index[dup.reindex(df.index, fill_value=False)]:
        others = df[(df.block_pr == df.at[i, "block_pr"]) & (key == key[i]) & (df.index != i)].line.tolist()
        df.at[i, "note"] = (df.at[i, "note"] + "; " if df.at[i, "note"] else "") + f"DUPLICATE: same root as sheet line(s) {others}"
    # PRs
    pr_rows = []
    for p in prs:
        xyz = pr_xyz.get(p, "")
        ra = at.get(xyz, "")
        ok = ra in descendants(p) if ra else False
        pr_rows.append(dict(upstream_PR=p, pr_xyz=xyz, root_at_xyz=ra, orig_latest=latest_flag.get(p),
                            descends_from_orig=ok))
    stamp = f"{now:%Y%m%d_%H%M}"
    df.to_csv(os.path.join(FIGURES, f"pr_sheet_check_{stamp}.tsv"), sep="\t", index=False)
    pd.DataFrame(pr_rows).to_csv(os.path.join(FIGURES, f"pr_sheet_check_{stamp}_prs.tsv"), sep="\t", index=False)
    print(f"\nlive check at {now:%Y-%m-%d %H:%M} UTC; verdicts: {df.verdict.value_counts().to_dict()}; "
          f"rows with a DUPLICATE note: {int(df.note.str.contains('DUPLICATE').sum())}")
    print("\nPR roots:")
    print(pd.DataFrame(pr_rows).to_string(index=False))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print("\nrows needing attention:")
    att = df[(df.verdict != "ok") | df.note.str.contains("DUPLICATE|not a root id")]
    print(att[["line", "block_pr", "root_id_orig", "root_id_curr", "xyz", "nsyn", "status", "root_at_xyz", "verdict",
               "note"]].to_string(index=False))
    print(f"\nreport: figures/pr_sheet_check_{stamp}.tsv")

    if a.write_sheet:
        upd = {int(r.line) - 1: r.proposed_root_id_curr for r in df.itertuples() if r.verdict == "update"}
        new = list(lines)
        for i, root in upd.items():
            c = new[i].split("\t")
            hdr = rows[[x["line"] for x in rows].index(i)]["header"]
            c[hdr.index("root_id_curr")] = root
            new[i] = "\t".join(c)
        open(a.write_sheet, "w", encoding="utf-8").write("\n".join(new))
        print(f"wrote {a.write_sheet}: root_id_curr updated on {len(upd)} rows, everything else verbatim")


if __name__ == "__main__":
    main()
