"""Follow-up to prg_any_paths.py (David 2026-11-12): where do the 'any PR -> T4/T5' path cells sit in PRg1toT4T5 (which
blocks), and what are the path cells that are not in the tab - who feeds them (PR groups), whom they feed (which
PR+2 / target cells, and whether those are wide-field T4T5-1 cells whose inputs the T4T5-2 block does not follow),
SeaTable label, other tabs. Read-only (snapshot caught up to now). Usage: prg_any_membership.py PATHS.tsv BOOK SEATABLE.tsv"""
import re
import sys

import numpy as np
import openpyxl
import pandas as pd

from faab import synsnap
from faab.clients.cave import client as cave_client
from faab.partners.sheet import cell_text as s, scan_book

paths = pd.read_csv(sys.argv[1], sep="\t", dtype={"root_id": str, "pr_group": str})
wb = openpyxl.load_workbook(sys.argv[2])
st = pd.read_csv(sys.argv[3], sep="\t", dtype=str).fillna("")
lab = dict(zip(st.root_now, st[["superclass", "class", "subclass", "type"]].agg(lambda r: " / ".join(x for x in r if x), axis=1)))
where, title = {}, {}
for x in scan_book(wb):
    if re.search(r"_b\d{6}d$", x["tab"]):
        continue
    r = s(x["values"].get("root_id_curr")) or s(x["values"].get("root_id_orig"))
    tag = x["tab"] + (f" {s(x['values'].get('hop'))}" if x["tab"] == "PRg1toT4T5" else "")
    if x["tab"] == "PRg1toT4T5":
        h = x["row"]
        ws = wb[x["tab"]]
        while h > 1 and s(ws.cell(h, 1).value) != "hop":
            h -= 1
        if s(ws.cell(h - 1, 1).value).startswith("LOW PRIORITY"):
            tag += " (LOW PRIORITY)"
    where.setdefault(r, []).append(tag)
pa = paths[paths.stage.isin(["PR+1", "PR+2"])].copy()
pa["blocks"] = pa.root_id.map(lambda r: "; ".join(dict.fromkeys(t for t in where.get(r, []) if t.startswith("PRg1toT4T5"))))
print("path cells by stage and PRg1toT4T5 block:")
print(pa.assign(b=pa.blocks.replace("", "not in the tab")).groupby(["stage", "b"]).size().to_string())

new = pa[~pa.in_PRg1toT4T5]
prs = sorted(int(r) for r in paths[paths.stage == "PR"].root_id)
grp = dict(zip(paths[paths.stage == "PR"].root_id.astype(int), paths[paths.stage == "PR"].pr_group))
p2 = sorted(int(r) for r in pa[pa.stage == "PR+2"].root_id)
tgt = []
for x in scan_book(wb):
    if x["tab"] == "PRg1toT4T5" and s(x["values"].get("hop")) == "PR+3":
        tgt.append(int(s(x["values"].get("root_id_curr"))))
c = cave_client()
snap = synsnap.Snapshot(log=lambda *_: None)
ids = sorted(int(r) for r in new.root_id)
e_in = snap.edges(pre=prs, post=ids, when="now", client=c)
e_out = snap.edges(pre=ids, post=sorted(set(p2) | set(tgt)), when="now", client=c)
WIDE = {"648518347494887381": "CT1", "648518347531730229": "LO ME (CT1-like)", "648518347570703976": "wide-field by size",
        "648518347643564937": "wide-field by size"}
print(f"\npath cells NOT in PRg1toT4T5: {len(new)}")
for _, r in new.sort_values(["stage", "distance_columns"]).iterrows():
    k = int(r.root_id)
    fin = e_in[(e_in.post == k) & (e_in.n >= 3)]
    fout = e_out[(e_out.pre == k) & (e_out.n >= 3)]
    feeds = "; ".join(f"{q} ({n}{', ' + WIDE[str(q)] if str(q) in WIDE else ''}{', target' if q in tgt else ''})"
                      for q, n in zip(fout.post, fout.n))
    groups = sorted({grp.get(int(q), "?") for q in fin.pre}, key=lambda g: (len(g), g))
    others = "; ".join(dict.fromkeys(t for t in where.get(r.root_id, []) if not t.startswith("PRg1toT4T5"))) or "-"
    print(f"{r.stage} {r.root_id}: {r.distance_columns} columns; from PRs of groups {', '.join(groups)} "
          f"({len(fin)} PRs, {int(fin.n.sum())} synapses); feeds {feeds or '-'}; SeaTable {lab.get(r.root_id, '-')}; other tabs {others}")
