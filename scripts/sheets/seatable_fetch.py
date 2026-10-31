"""Snapshot of the SeaTable (FlyTable) aedes_main table with each row's live root (CAVE read-only: one get_roots call on
the supervoxel ids). Writes figures/<stamp>_seatable_aedes_main.tsv (all columns + root_now) and prints the columns
and the value counts of the short-vocabulary ones. Usage: seatable_fetch.py"""
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from faab.clients.cave import client as cave_client, throttle
from faab.config import FIGURES
from faab.partners.sheet import utc_stamp

SERVER = "https://flytable.mrc-lmb.cam.ac.uk/"


def main():
    import seaserpent as ss
    stamp = utc_stamp()
    t = ss.Table("aedes_main", server=SERVER).to_frame()
    t.columns = [str(c) for c in t.columns]
    print(f"aedes_main: {len(t)} rows, {len(t.columns)} columns:\n  {list(t.columns)}")
    sv = pd.to_numeric(t["supervoxel_id"], errors="coerce")
    ok = sv.notna() & (sv > 0)
    c = cave_client()
    throttle(0)
    roots = np.asarray(c.chunkedgraph.get_roots(sv[ok].astype("int64").to_numpy()))
    t["root_now"] = ""
    t.loc[ok, "root_now"] = [str(int(r)) for r in roots]
    print(f"rows with a supervoxel id: {int(ok.sum())}; distinct live roots {t.loc[ok, 'root_now'].nunique()}")
    for col in t.columns:
        v = t[col].dropna().astype(str).str.strip()
        v = v[v != ""]
        if 0 < v.nunique() <= 60:
            print(f"\n{col} ({len(v)} filled): " + "; ".join(f"{k} {n}" for k, n in v.value_counts().head(25).items()))
        elif v.nunique() > 60:
            print(f"\n{col} ({len(v)} filled, {v.nunique()} distinct), e.g. {v.iloc[:3].tolist()}")
    out = os.path.join(FIGURES, f"{stamp}_seatable_aedes_main.tsv")
    t.to_csv(out, sep="\t", index=False)
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
