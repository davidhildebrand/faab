"""Network-free: in a PRs-tab TSV export, the rows whose notes held pairing text ('pairing: ...', ring angle, R7 margin,
mirror / terminal partner remarks) and what the pair columns say for them - does the dropped text hold anything the
columns do not? Usage: pairing_notes_check.py PRs.tsv"""
import re
import sys
from collections import Counter

from faab.partners.short_notes import split_parts

lines = open(sys.argv[1], encoding="utf-8").read().split("\n")
hdr, rows = None, []
for i, ln in enumerate(lines, 1):
    c = ln.split("\t")
    if c and c[0] == "root_id_orig":
        hdr = c
        continue
    if hdr and c and c[0].isdigit():
        rows.append((i, dict(zip(hdr, c))))
PAIR = re.compile(r"(pairing: |ring angle|R\d margin|no mutual mirror partner|terminal-zone partner differs)")
kinds, empty_cols, ex = Counter(), Counter(), {}
for i, d in rows:
    parts = [p for p in split_parts(d.get("notes", "")) if PAIR.search(p)]
    for p in parts:
        k = re.sub(r"\d{15,}", "<id>", re.sub(r"[-+]?\d+(\.\d+)?", "#", p))[:90]
        kinds[k] += 1
        ex.setdefault(k, (i, d.get("pair_type", ""), d.get("fly_pair", ""), d.get("confidence", ""), p[:140]))
    if parts and not (d.get("pair_type") or d.get("fly_pair")):
        empty_cols[d.get("class", "")] += 1
print(f"{len(rows)} member rows; rows with pairing text whose pair_type and fly_pair are both blank: {dict(empty_cols)}")
for k, n in kinds.most_common():
    i, pt, fp, conf, p = ex[k]
    print(f"{n:4d}  {k}\n        e.g. row {i}: pair_type={pt!r} fly_pair={fp!r} confidence={conf!r}")
