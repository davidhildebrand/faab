"""Per LRS cartridge: the putative types of its 'L1-L5' cells (lmc_typing.tsv), and a one-type-per-cell assignment
within the cartridge (a cartridge holds one each of L1..L5): maximum summed log random-forest probability over distinct
types (scipy linear_sum_assignment), over the types not already taken by a SeaTable-typed cell of the same cartridge
(cartridge 6: LRS typed its L2 and L4; 2026-11-01). LRS's cartridge = 'LRS: cartridge N' in the SeaTable notes.
Network-free. Writes figures/lmc/lmc_cartridge_calls.tsv. Usage: lmc_cartridges.py"""
import re

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from faab.config import fig

TYPES = ["L1", "L2", "L3", "L4", "L5"]
T = pd.read_csv(fig("lmc_typing.tsv"), sep="\t", dtype={"root": str, "serial_id": str}).fillna("")
C = pd.read_csv(fig("lmc_cells.tsv"), sep="\t", dtype=str).fillna("")
T = T.merge(C[["root", "notes"]], on="root", how="left")
T["cartridge"] = T.notes.map(lambda s: (re.findall(r"cartridge\s*(\d+)", s) or [""])[0])
U = T[T.label == "L1-L5"].copy()
U["assigned"], U["assigned_p"], U["types_free"] = "", np.nan, ""
taken = T[T.label.isin(TYPES) & (T.cartridge != "")].groupby(["side", "cartridge"]).label.apply(set).to_dict()
print(f"{len(U)} L1-L5 cells; cartridges {U.cartridge.value_counts().sort_index().to_dict()}; "
      f"typed cells in those cartridges: {({k: sorted(v) for k, v in taken.items()})}")
for (side, cart), g in U.groupby(["side", "cartridge"]):
    free = [t for t in TYPES if t not in taken.get((side, cart), set())]
    Pm = g[[f"P_{t}" for t in free]].to_numpy(float)
    rows, cols = linear_sum_assignment(-np.log(Pm + 1e-3))
    U.loc[g.index[rows], "assigned"] = [free[c] for c in cols]
    U.loc[g.index[rows], "assigned_p"] = Pm[rows, cols]
    U.loc[g.index, "types_free"] = ",".join(free)
    g = U.loc[g.index]
    calls = sorted(g.consensus)
    dup = [t for t in set(calls) if calls.count(t) > 1]
    print(f"\nside {side} cartridge {cart or '?'}: {len(g)} cells, free types {', '.join(free)}; consensus {', '.join(calls)}"
          + (f"; REPEATED {', '.join(dup)}" if dup else "") + "; one-per-type assignment changes "
          + str(int((g.assigned != g.consensus).sum())))
    print(g[["root", "serial_id", "status", "consensus", "votes", "assigned", "assigned_p", "RF_proba"]].to_string(index=False))
U[["root", "serial_id", "side", "cartridge", "status", "consensus", "votes", "types_free", "assigned", "assigned_p",
   "RF_proba"]].to_csv(
    fig("lmc_cartridge_calls.tsv"), sep="\t", index=False)
print("\nwritten", fig("lmc_cartridge_calls.tsv"))
