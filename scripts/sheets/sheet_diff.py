"""Network-free: what changed between two block-TSV versions of a partner sheet (rows matched by block key +
root_id_orig; the block key is hop1_root_id for two-hop sheets, else the first column). Reports added / removed rows and
per-column value changes, listing tracking-column edits in full. Usage: sheet_diff.py OLD.tsv NEW.tsv"""
import sys
from collections import Counter

from faab.partners.sheet import TRACK, parse_sheet

old_path, new_path = sys.argv[1], sys.argv[2]


def load(path):
    lines, rows = parse_sheet(path)
    hdr = rows[0]["header"]
    key_col = "hop1_root_id" if "hop1_root_id" in hdr else hdr[0]
    d = {}
    for r in rows:
        c = r["cells"]
        d.setdefault((c[key_col].strip(), c["root_id_orig"].strip()), []).append((r["line"] + 1, c))
    return hdr, d, len(rows), len(lines)


ho, old, n_old, l_old = load(old_path)
hn, new, n_new, l_new = load(new_path)
print(f"old: {n_old} rows / {l_old} lines, {len(ho)} columns; new: {n_new} rows / {l_new} lines, {len(hn)} columns")
if [h for h in ho if h] != [h for h in hn if h]:
    print(f"header changed: old {[h for h in ho if h]}\n                new {[h for h in hn if h]}")
added, removed = sorted(new.keys() - old.keys()), sorted(old.keys() - new.keys())
print(f"rows added: {len(added)}; removed: {len(removed)}; blocks old {len({k[0] for k in old})} new {len({k[0] for k in new})}")
for k in added[:15]:
    c = new[k][0][1]
    print(f"  + line {new[k][0][0]} block ..{k[0][-6:]} orig {k[1]}: status {c.get('status', '')!r} nsyn {c.get('nsyn', '')} notes {c.get('notes', '')[:60]!r}")
for k in removed[:15]:
    c = old[k][0][1]
    print(f"  - block ..{k[0][-6:]} orig {k[1]}: status {c.get('status', '')!r} nsyn {c.get('nsyn', '')} notes {c.get('notes', '')[:60]!r}")
changes, examples = Counter(), []
for k in sorted(old.keys() & new.keys()):
    for (lo, co), (ln, cn) in zip(old[k], new[k]):
        for col in hn:
            if not col or col not in co:
                continue
            a, b = co.get(col, "").strip(), cn.get(col, "").strip()
            if a != b:
                changes[col] += 1
                if col in TRACK or col in ("root_id_curr", "xyz"):
                    examples.append(f"  line {ln} block ..{k[0][-6:]} orig ..{k[1][-6:]} {col}: {a[:50]!r} -> {b[:50]!r}")
print(f"value changes by column (matched rows): {dict(changes)}")
print("\n".join(examples[:80]))
if len(examples) > 80:
    print(f"  ... {len(examples) - 80} more")
