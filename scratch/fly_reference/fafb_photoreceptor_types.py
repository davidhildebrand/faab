"""Find photoreceptor cell types in FAFB/FlyWire public annotations and count cells.
Caches the annotation table. Read-only, throttled."""
from faab.config import REPO
import os
import time
import pickle
import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import caveclient
from faab.sources.cave import throttle

OUT = os.path.join(REPO, "figures")
DS = "flywire_fafb_public"
CACHE = os.path.join(OUT, "fafb_annotations.pkl")


def main():
    if os.path.exists(CACHE):
        ann = pickle.load(open(CACHE, "rb"))
        print(f"loaded cached annotations: {len(ann)} rows")
    else:
        client = caveclient.CAVEclient(DS)
        throttle(time.time())
        ann = client.materialize.query_table("hierarchical_neuron_annotations")
        pickle.dump(ann, open(CACHE, "wb"))
        print(f"fetched annotations: {len(ann)} rows")

    ann["cell_type"] = ann["cell_type"].astype(str)
    ct = ann["cell_type"]

    print("\nclassification_system values:",
          ann["classification_system"].astype(str).value_counts().head(10).to_dict())

    print("\n=== cell_type values matching photoreceptor patterns ===")
    pat = ct.str.match(r"^R[1-8]", na=False) | ct.str.contains("photorecep", case=False, na=False)
    hits = ann[pat]["cell_type"].value_counts()
    if len(hits):
        for k, v in hits.items():
            print(f"  {k:20} {v}")
    else:
        print("  none found")

    print("\n=== any cell_type containing 'R7'/'R8'/'R1-6' loosely ===")
    loose = ct[ct.str.contains(r"\bR[1-8]\b|R1-6|R7|R8", regex=True, na=False)].value_counts()
    for k, v in loose.head(20).items():
        print(f"  {k:20} {v}")

    print("\n=== optic/lamina-related classification systems ===")
    for cs, d in ann.groupby(ann["classification_system"].astype(str)):
        types = d["cell_type"].nunique()
        if any(k in cs.lower() for k in ("optic", "visual", "lamina", "primary")):
            print(f"  {cs}: {len(d)} rows, {types} types, e.g. "
                  f"{sorted(set(d['cell_type']))[:8]}")
    print("\nDONE")


if __name__ == "__main__":
    main()
