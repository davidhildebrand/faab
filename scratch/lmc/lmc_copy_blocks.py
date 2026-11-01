"""Copy blocks of the LMC typing for David (2026-11-01): one row per called cell (the 28 LRS 'L1-L5' cells by cartridge,
the SeaTable-typed cells whose call differs from their label, the untyped extras), with root ids, and per putative type
the root ids on one line. Confidence: strong = 3/3 feature models, kept by the cartridge assignment, probability >= 0.5;
weak = probability < 0.3, a 1/3 vote, or a cartridge assignment that overrides the models; else ok.
Network-free. Writes tmp/lmc_copy_blocks.txt. Usage: lmc_copy_blocks.py"""
import os

import pandas as pd

from faab.config import TMP, fig

TYPES = ["L1", "L2", "L3", "L4", "L5"]
T = pd.read_csv(fig("lmc_typing.tsv"), sep="\t", dtype={"root": str, "serial_id": str}).fillna("")
K = pd.read_csv(fig("lmc_cartridge_calls.tsv"), sep="\t", dtype={"root": str, "cartridge": str}).fillna("")
T = T.merge(K[["root", "cartridge", "types_free", "assigned", "assigned_p"]], on="root", how="left").fillna("")


def confidence(x):
    call = x.assigned or x.consensus
    p = float(x.assigned_p) if x.assigned else float(getattr(x, f"P_{call}"))
    if p < 0.3 or x.votes == "1/3" or (x.assigned and x.assigned != x.consensus):
        return call, p, "weak"
    return call, p, "strong" if x.votes == "3/3" and p >= 0.5 else "ok"


rows = []
for x in T.itertuples(index=False):
    if x.label == "L1-L5":
        group = f"cartridge {x.cartridge}"
    elif x.label in TYPES and x.consistent == "NO":
        group = f"SeaTable {x.label}, looks {x.consensus}"
    elif x.label not in TYPES:
        group = "not typed in SeaTable" if x.serial_id else "not in SeaTable"
    else:
        continue
    call, p, conf = confidence(x)
    note = ""
    if x.assigned and x.assigned != x.consensus:
        note = f"by elimination in the cartridge (free {x.types_free.replace(',', ' ')}); the models say {x.consensus}"
    rows.append(dict(root_id=x.root, side=x.side, group=group, seatable=x.label or ("LMC" if x.serial_id else "-"),
                     putative=call, p=f"{p:.2f}", confidence=conf, note=note))
R = pd.DataFrame(rows)
R["k"] = R.group.map(lambda g: (0 if g.startswith("cartridge") else 1, g))
R = R.sort_values(["k", "putative"]).drop(columns="k")
cols = ["root_id", "side", "group", "seatable", "putative", "p", "confidence", "note"]
out = ["\t".join(cols)] + ["\t".join(str(v) for v in r) for r in R[cols].itertuples(index=False)]
out += ["", "putative type: root ids (all called cells above)"]
for t in TYPES:
    out.append(f"{t}: " + ", ".join(R[R.putative == t].root_id))
path = os.path.join(TMP, "lmc_copy_blocks.txt")
open(path, "w").write("\n".join(out) + "\n")
print(f"{len(R)} cells; written {path}")
