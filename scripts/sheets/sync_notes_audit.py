"""Every notes change of a sync run, by tab and root_id_orig (David 2026-11-08: only clear notes; check replies before
removing). For each: old -> new, the reason, and whether anything was ADDED (a new part not in the old notes, other
than the sync's own 'xyz off the cell (on X)' flag or a 'missing soma, ...' remainder). Also the MISMATCH rows.
Usage: sync_notes_audit.py SYNC_STAMP MATCH_STAMP; writes tmp/notes_changes_<SYNC_STAMP>.txt"""
import sys

import pandas as pd

from faab.config import fig

sync_stamp, match_stamp = sys.argv[1], sys.argv[2]
L = pd.read_csv(fig(f"{sync_stamp}_seatable_sync_changes.tsv"), sep="\t", dtype=str).fillna("")
M = pd.read_csv(fig(f"{match_stamp}_seatable_match.tsv"), sep="\t", dtype=str).fillna("")
info = {(t, r): x for t, r, x in zip(M.tab, M.row, M.to_dict("records"))}
parts = lambda t: [p.strip() for p in t.split(";") if p.strip()]
lines, added = [], 0
n = L[L.column == "notes"]
for x in n.itertuples():
    m = info.get((x.tab, x.row), {})
    old, new = parts(x.old), parts(x.new)
    extra = [p for p in new if p not in old and not p.startswith("xyz off the cell")
             and not any(o.lower().endswith(p.lower()) for o in old)]
    added += bool(extra)
    lines.append(f"{x.tab} {m.get('root_id_orig', '?')} [{x.why}]\n    old: {x.old}\n    new: {x.new}"
                 + (f"\n    ADDED: {extra}" if extra else ""))
mm = L[(L.column == "fill") & (L.why == "MISMATCH")]
lines.append(f"\nMISMATCH rows ({len(mm)}):")
for x in mm.itertuples():
    m = info.get((x.tab, x.row), {})
    lines.append(f"  {x.tab} {m.get('root_id_orig')}: curr {m.get('root_id_curr')}, point on {m.get('root_at_xyz')}, "
                 f"proofreader {m.get('sheet_proofreader')!r}")
cl = L[(L.column == "fill") & L.why.str.startswith("stale note")]
lines.insert(0, f"notes changed: {len(n)}; with anything added: {added}; highlights cleared: {len(cl)}")
out = f"/Users/davidh/Data/Vibe/FAAB/tmp/notes_changes_{sync_stamp}.txt"
open(out, "w").write("\n".join(lines) + "\n")
print(lines[0])
print("written", out)
