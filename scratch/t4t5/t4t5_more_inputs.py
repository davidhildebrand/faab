"""How many more inputs do the T4/T5 targets of the PRg1toT4T5 tab have (David, 2026-10-31: a new subtable of their
inputs with nsyn >= 3)? Read-only: the tab's rows (an exported one-tab .xlsx), one root lookup at all its xyz points,
the local synapse snapshot + one CAVE catch-up to now (faab.synsnap), SeaTable labels from the newest snapshot.

Targets = hop-3 rows (the T4/T5 candidates), root at their xyz. Every presynaptic cell with >= THR synapses onto at
least one target is an input (self-contacts and target-to-target inputs are reported separately). Each input is
checked against the cells already in the tab (live root at every row's xyz). Writes figures/t4t5/<stamp>_t4t5_more_inputs.tsv
(one row per input: per-target synapse counts, total, best target, in-tab rows, SeaTable label). Usage:
t4t5_more_inputs.py PRg1toT4T5.xlsx [THR]"""
import glob
import sys

import numpy as np
import openpyxl
import pandas as pd

from faab import synsnap
from faab.clients.cave import client as cave_client, roots_at_xyz, throttle
from faab.config import fig
from faab.partners.sheet import cell_text as s, scan_book, seatable_label, utc_stamp

book = sys.argv[1]
THR = int(sys.argv[2]) if len(sys.argv) > 2 else 3
rows = [x for x in scan_book(openpyxl.load_workbook(book)) if s(x["values"].get("hop")) in ("1", "2", "3")]
at = [str(r) for r in roots_at_xyz([s(x["values"]["xyz"]) for x in rows])]
tab_of = {}
for x, r in zip(rows, at):
    tab_of.setdefault(r, []).append(f"hop-{s(x['values']['hop'])} r{x['row']}")
targets = sorted({r for x, r in zip(rows, at) if s(x["values"]["hop"]) == "3" and r != "0"})
print(f"{len(rows)} tab rows; {len(targets)} T4/T5 targets (hop-3) at their xyz")

c = cave_client()
throttle(0)
S = synsnap.Snapshot()
syn = S.synapses(post=[int(t) for t in targets], when="now", client=c)
syn = syn[syn.pre_root != syn.post_root]
e = syn.groupby(["pre_root", "post_root"]).size().rename("n").reset_index()
e["pre_root"], e["post_root"] = e.pre_root.astype(str), e.post_root.astype(str)
wide = e.pivot_table(index="pre_root", columns="post_root", values="n", fill_value=0)
best = wide.max(axis=1)
inp = pd.DataFrame({"max_onto_one": best, "total": wide.sum(axis=1), "n_targets": (wide >= THR).sum(axis=1),
                    "best_target": wide.idxmax(axis=1)})
q = inp[inp.max_onto_one >= THR].copy()
q["in_tab"] = ["; ".join(tab_of.get(r, [])) for r in q.index]
q["is_target"] = q.index.isin(targets)
st = pd.read_csv(sorted(glob.glob(fig("*_seatable_aedes_main.tsv")))[-1], sep="\t", dtype=str).fillna("")
lab = {r["root_now"]: seatable_label(r) for r in st.to_dict("records") if r["root_now"]}
q["seatable"] = [lab.get(r, "") for r in q.index]
q = q.sort_values(["max_onto_one", "total"], ascending=False)

print(f"inputs (presynaptic cells, self-contacts removed): {len(inp)} with any synapse onto a target")
for t in (2, 3, 5, 10):
    print(f"  >= {t} synapses onto at least one target: {int((inp.max_onto_one >= t).sum())}; "
          f">= {t} summed over targets: {int((inp.total >= t).sum())}")
new = q[(q.in_tab == "") & ~q.is_target]
print(f"\nat >= {THR} onto one target: {len(q)} cells = {int((q.in_tab != '').sum())} already in the tab "
      f"({q[q.in_tab != ''].in_tab.str.extract(r'(hop-\d)')[0].value_counts().to_dict()}), "
      f"{int(q.is_target.sum())} are T4/T5 targets themselves, {len(new)} new")
print(f"new inputs: synapses onto one target p50 {new.max_onto_one.median():.0f}, max {new.max_onto_one.max()}; "
      f"feeding >= 2 targets: {int((new.n_targets >= 2).sum())}; with a SeaTable label: {int((new.seatable != '').sum())}")
print("  SeaTable labels:", new[new.seatable != ""].seatable.value_counts().head(10).to_dict())
per_t = e[e.n >= THR].groupby("post_root").size().reindex(targets, fill_value=0)
print(f"inputs at >= {THR} per target: p10/50/90 {per_t.quantile([0.1, 0.5, 0.9]).round(0).astype(int).tolist()}, "
      f"min {per_t.min()}, max {per_t.max()}")
out = fig(f"{utc_stamp()}_t4t5_more_inputs.tsv")
q.reset_index().rename(columns={"pre_root": "input_root"}).to_csv(out, sep="\t", index=False)
print(f"written {out}")
