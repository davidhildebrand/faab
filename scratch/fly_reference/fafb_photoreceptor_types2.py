"""Find photoreceptor cell types in FAFB/FlyWire via fafbseg's annotation search
(filters server-side; the full-table pull 503s). Read-only, throttled."""
import time
import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
from fafbseg import flywire
from faab.sources.cave import throttle


def search(term, **kw):
    for a in range(4):
        try:
            return flywire.search_annotations(term, dataset="public", **kw)
        except Exception as e:
            if a == 3:
                print(f"  '{term}' failed: {str(e)[:90]}")
                return pd.DataFrame()
            time.sleep(8)


def main():
    t = throttle(time.time())
    for term in ["R1-6", "R7", "R8", "photoreceptor"]:
        t = throttle(t)
        df = search(term)
        if not len(df):
            print(f"\n'{term}': no rows")
            continue
        cols = [c for c in ("cell_type", "hemibrain_type", "cell_class", "super_class",
                            "side", "root_id") if c in df.columns]
        print(f"\n=== search '{term}': {len(df)} rows | columns {list(df.columns)[:12]}")
        for c in ("cell_type", "cell_class", "super_class"):
            if c in df.columns:
                vc = df[c].astype(str).value_counts().head(12)
                print(f"  {c}: {vc.to_dict()}")
    print("\nDONE")


if __name__ == "__main__":
    main()
