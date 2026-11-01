"""Per LRS cartridge: the putative types of its 'L1-L5' cells (lmc_typing.tsv), and a one-type-per-cell assignment
within the cartridge (a cartridge holds one each of L1..L5): maximum summed log random-forest probability over distinct
types (scipy linear_sum_assignment). Network-free. Writes figures/lmc/lmc_cartridge_calls.tsv.
Usage: lmc_cartridges.py"""
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
U["assigned"], U["assigned_p"] = "", np.nan
print(f"{len(U)} L1-L5 cells; cartridges {U.cartridge.value_counts().sort_index().to_dict()}")
for (side, cart), g in U.groupby(["side", "cartridge"]):
    Pm = g[[f"P_{t}" for t in TYPES]].to_numpy(float)
    rows, cols = linear_sum_assignment(-np.log(Pm + 1e-3))
    U.loc[g.index[rows], "assigned"] = [TYPES[c] for c in cols]
    U.loc[g.index[rows], "assigned_p"] = Pm[rows, cols]
    g = U.loc[g.index]
    calls = sorted(g.consensus)
    dup = [t for t in set(calls) if calls.count(t) > 1]
    print(f"\nside {side} cartridge {cart or '?'}: {len(g)} cells; consensus {', '.join(calls)}"
          + (f"; REPEATED {', '.join(dup)}" if dup else "") + "; one-per-type assignment changes "
          + str(int((g.assigned != g.consensus).sum())))
    print(g[["root", "serial_id", "status", "consensus", "votes", "assigned", "assigned_p", "RF_proba"]].to_string(index=False))
U[["root", "serial_id", "side", "cartridge", "status", "consensus", "votes", "assigned", "assigned_p", "RF_proba"]].to_csv(
    fig("lmc_cartridge_calls.tsv"), sep="\t", index=False)
print("\nwritten", fig("lmc_cartridge_calls.tsv"))
