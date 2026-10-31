"""Rows whose root_id_curr is not the live root at their xyz (from the newest figures/*_seatable_match.tsv), split into
stale ids (root_id_curr no longer current: normal root update) and MISMATCH (root_id_curr current but elsewhere: hand-set
to another cell, or the xyz moved). One CAVE call (is_latest_roots). Also prints the rows of the 2026-10-25 flagged pairs
in PRg1hop2. Usage: root_check.py"""
import glob
import os

import numpy as np
import pandas as pd

from faab.clients.cave import client as cave_client, throttle
from faab.config import FIGURES

path = sorted(glob.glob(os.path.join(FIGURES, "*_seatable_match.tsv")))[-1]
d = pd.read_csv(path, sep="\t", dtype=str).fillna("")
cur = d.root_id_curr.str.fullmatch(r"\d{15,}")
diff = d[cur & (d.root_id_curr != d.root_at_xyz)].copy()
ids = sorted(set(diff.root_id_curr))
c = cave_client()
throttle(0)
latest = dict(zip(ids, np.asarray(c.chunkedgraph.is_latest_roots([int(i) for i in ids])).astype(bool)))
diff["curr_is_current"] = diff.root_id_curr.map(latest)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
cols = ["tab", "row", "root_id_orig", "root_id_curr", "root_at_xyz", "xyz", "sheet_status", "sheet_proofreader"]
print(f"{os.path.basename(path)}: {len(d)} rows; root_id_curr differs from the root at xyz in {len(diff)} "
      f"(stale {int((~diff.curr_is_current).sum())}, MISMATCH {int(diff.curr_is_current.sum())}); no segment at xyz "
      f"{int((d.root_at_xyz == '0').sum())}; no root_id_curr {int((~cur).sum())}")
print("\nstale per tab:", diff[~diff.curr_is_current].tab.value_counts().to_dict())
print("\nMISMATCH rows:")
print(diff[diff.curr_is_current][cols].to_string(index=False))
print("\nno segment at xyz:")
print(d[d.root_at_xyz == "0"][cols].to_string(index=False))
flag = d[(d.tab == "PRg1hop2") & d.row.astype(int).isin([35, 36, 71, 72, 160, 161, 286, 287, 520, 521])]
print("\n2026-10-25 flagged pairs in PRg1hop2:")
print(flag[cols + ["via", "st_status", "st_label", "st_proofreader"]].to_string(index=False))
