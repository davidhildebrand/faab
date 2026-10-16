"""Probe FAFB/FlyWire for photoreceptor cells: which datastack, which annotation
table/cell types (R1-6 / R7 / R8), and how to count synapses. Read-only, throttled."""
import time
import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import caveclient
from faab.sources.cave import throttle


def main():
    print("=== available datastacks (public) ===")
    try:
        c0 = caveclient.CAVEclient()
        ds = [d for d in c0.info.get_datastacks() if "flywire" in d.lower() or "fafb" in d.lower()]
        print(" ", ds)
    except Exception as e:
        print("  datastack list failed:", str(e)[:100])
        ds = []

    name = "flywire_fafb_public"
    if ds and name not in ds:
        name = ds[0]
    print(f"\nusing datastack: {name}")
    t = throttle(time.time())
    client = caveclient.CAVEclient(name)
    tables = client.materialize.get_tables()
    print(f"\n=== materialized tables ({len(tables)}) ===")
    for tb in tables:
        print("  ", tb)

    # find the cell-type / annotation table and look for photoreceptor types
    cand = [tb for tb in tables if any(k in tb.lower()
            for k in ("cell_type", "celltype", "info", "annotation", "hierarch", "visual"))]
    print(f"\ncandidate annotation tables: {cand}")
    for tb in cand[:4]:
        t = throttle(t)
        try:
            df = client.materialize.query_table(tb, limit=5)
            print(f"\n--- {tb}: columns {list(df.columns)}")
        except Exception as e:
            print(f"\n--- {tb}: query failed {str(e)[:80]}")
    print("\nDONE")


if __name__ == "__main__":
    main()
