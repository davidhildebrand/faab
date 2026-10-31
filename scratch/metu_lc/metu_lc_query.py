"""Read current MeTu / MeTu4 / LC annotations from FlyTable (aedes_main) and cache
their root_ids for the MeTu4-vs-LC re-check. Read-only."""
from faab.config import REPO
import os
import pickle
import pandas as pd
import seaserpent as ss
from faab.config import fig

SERVER = "https://flytable.mrc-lmb.cam.ac.uk/"
OUT = os.path.join(REPO, "figures")


def to_int(s):
    try:
        return int(str(s))
    except Exception:
        return None


def main():
    meta = ss.Table("aedes_main", server=SERVER).to_frame()
    meta.columns = [str(c) for c in meta.columns]
    meta["rid"] = meta["root_id"].map(to_int)

    for col in ["superclass", "class", "subclass", "type"]:
        if col in meta.columns:
            vc = meta[col].astype(str)
            hits = vc[vc.str.contains("MeTu|LC", case=False, na=False)].value_counts()
            if len(hits):
                print(f"[{col}] MeTu/LC values:\n{hits.to_string()}\n")

    def grp(label):
        m = meta[meta["subclass"].astype(str) == label]
        return m.dropna(subset=["rid"]).drop_duplicates("rid")

    for lab in ["MeTu", "MeTu4", "LC"]:
        g = grp(lab)
        sides = g["side"].astype(str).value_counts().to_dict() if "side" in g else "?"
        anns = g["annotator"].astype(str).value_counts().to_dict() if "annotator" in g else "?"
        types = g["type"].astype(str).value_counts().to_dict() if "type" in g else "?"
        print(f"subclass=={lab!r}: {len(g)} cells | sides {sides} | annotators {anns}")
        print(f"    type breakdown: {types}")
        pickle.dump([int(x) for x in g["rid"]], open(fig(f"ids_{lab}.pkl"), "wb"))
    print("DONE")


if __name__ == "__main__":
    main()
