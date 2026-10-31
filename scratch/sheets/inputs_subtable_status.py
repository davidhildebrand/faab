"""How far along are the PRg1toT4T5 'more inputs' cells: sheet status, SeaTable (newest fetch, by live root at the
xyz), cable, main neuropil, synapses onto the targets. Network-free.
Usage: inputs_subtable_status.py INPUTS.tsv SEATABLE_MATCH.tsv"""
import sys

import pandas as pd

I = pd.read_csv(sys.argv[1], sep="\t", dtype=str).fillna("")
M = pd.read_csv(sys.argv[2], sep="\t", dtype=str).fillna("")
M = M[M.tab == "PRg1toT4T5"]
st = M[M.st_n != ""].set_index("root_at_xyz")
I["in_st"] = I.root_id_curr.isin(st.index)
I["st_label"] = I.root_id_curr.map(lambda r: " / ".join(x for x in st.loc[r, ["st_label"]].tolist()) if r in st.index else "")
I["st_status"] = I.root_id_curr.map(lambda r: st.loc[r, "st_status"] if r in st.index else "")
I["cable"] = pd.to_numeric(I.cable_um, errors="coerce")
I["out"] = pd.to_numeric(I.nsyn_out, errors="coerce")
I["main_np"] = I.neuropils.str.split(" ").str[0]
print(f"{len(I)} subtable cells; sheet status {I.status.value_counts().to_dict()}; blocks {I.block.value_counts().to_dict()}")
print(f"tracked in other tabs: {(I.tracked_in != '').sum()}; annotation filled: {(I.annotation != '').sum()}")
print(f"in SeaTable (newest fetch, root at xyz): {I.in_st.sum()}")
print(I[I.in_st][["root_id_curr", "status", "st_status", "st_label", "nsyn_out", "cable_um"]].to_string(index=False))
bins = [0, 50, 150, 300, 600, 1e9]
print("\ncable (um):", pd.cut(I.cable, bins, labels=["<50", "50-150", "150-300", "300-600", ">=600"]).value_counts(sort=False).to_dict())
print("synapses onto all targets:", pd.cut(I.out, [0, 3, 5, 10, 20, 1e9], labels=["3", "4-5", "6-10", "11-20", ">20"]).value_counts(sort=False).to_dict())
print("targets reached:", I.T4T5_reached.value_counts().sort_index().to_dict())
print("main neuropil:", I.main_np.value_counts().to_dict())
new = I[(I.status == "NEW") & ~I.in_st]
print(f"\nNEW and not in SeaTable: {len(new)}; of these cable >= 300 um: {(new.cable >= 300).sum()}, "
      f">= 10 synapses onto targets: {(new.out >= 10).sum()}")
