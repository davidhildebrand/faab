"""One registry of all proposed cutouts with a plain purpose note per box (David 2026-11-02: "clearly differentiate so
that I can keep track"), from the registries of propose_targets.py and r7_missed.py. Writes
figures/cutouts/<stamp>_cutouts_all.tsv and a markdown table (<stamp>_cutouts_all.md) to paste.
Usage: registry_all.py REGISTRY.tsv [REGISTRY2.tsv ...]"""
import datetime
import sys

import pandas as pd

from faab.config import fig

CHECK = {"R1-R6": "R1-R6", "R8": "R8", "R7": "R7", "L1": "L1", "L2": "L2", "L3": "L3", "LMC": "LMC", "T4": "T4 input",
         "T5": "T5 input"}


def purpose(x):
    if isinstance(x.get("purpose"), str) and x["purpose"]:
        return x["purpose"]
    what = CHECK.get(x["category"].split()[0], x["category"])
    return (f"prediction check: placed where the most {what} synapses are predicted; annotate every synapse in the box "
            f"(mainly checks that the predicted synapses are real)")


R = pd.concat([pd.read_csv(p, sep="\t", dtype=str) for p in sys.argv[1:]], ignore_index=True).fillna("")
R["purpose"] = R.apply(purpose, axis=1)
R["kind"] = R.category.map(lambda c: "R7 missed, cold" if "cold" in c else "R7 missed, typical" if "typical" in c
                           else "prediction check")
R["target_synapses"] = R.n_target_synapses.where(R.n_target_synapses != "", R.get("predicted_r7_outputs", ""))
cols = ["box_id", "kind", "purpose", "category", "side", "neuropil", "center_xyz", "lo_xyz", "hi_xyz", "size_voxels",
        "target_synapses", "n_all_synapses", "target_cells", "link"]
extra = [c for c in ("r7_membrane_um2", "r7_depth_um", "expected_r7_outputs") if c in R]
order = {"prediction check": 0, "R7 missed, cold": 1, "R7 missed, typical": 2}
R = R.sort_values(["kind", "box_id"], key=lambda s: s.map(order) if s.name == "kind" else s)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dd%H%M%StUTC")
out = fig(f"cutouts/{stamp}_cutouts_all.tsv")
R[cols + extra].to_csv(out, sep="\t", index=False)
md = ["| box | kind | side | neuropil | center (x, y, z voxels) | target synapses / all | purpose | link |",
      "|---|---|---|---|---|---|---|---|"]
for x in R.itertuples(index=False):
    md.append(f"| {x.box_id} | {x.kind} | {x.side} | {x.neuropil} | {x.center_xyz} | {x.target_synapses} / "
              f"{x.n_all_synapses} | {x.purpose} | {x.link} |")
open(out.replace(".tsv", ".md"), "w").write("\n".join(md) + "\n")
print(R.groupby(["kind", "side"]).size().to_string())
print(f"written {out} and {out.replace('.tsv', '.md')}")
