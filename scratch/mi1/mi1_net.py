"""#4: Find a shared wide-field partner of the putative Mi1 that could serve as an
annotation-free 'net' for finding more Mi1. Rank every partner by how many of the
Mi1 it touches (up + downstream). Dataset: aedes. Read-only, throttled.
"""
from faab.config import REPO
import os
import time
import pickle
import pandas as pd
import caveclient
import seaserpent as ss

from faab.clients.cave import throttle, DATASTACK
from faab.config import fig

OUT = os.path.join(REPO, "figures")
SERVER = "https://flytable.mrc-lmb.cam.ac.uk/"


def to_int64(s):
    return s.map(lambda x: int(str(x)) if str(x).strip().lstrip("-").isdigit() else pd.NA).astype("Int64")


def main():
    dp = pickle.load(open(fig("mi1_dotprops.pkl"), "rb"))
    mi1 = [int(d.id) for d in dp]
    mi1set = set(mi1)
    print(f"putative Mi1: {len(mi1)}")

    meta = ss.Table("aedes_main", server=SERVER).to_frame()
    meta.columns = [str(c) for c in meta.columns]
    meta["rid"] = to_int64(meta["root_id"])
    ann = (meta.dropna(subset=["rid"]).drop_duplicates("rid")
           .set_index(meta.dropna(subset=["rid"]).drop_duplicates("rid")["rid"].astype("int64"))
           [["superclass", "class", "subclass", "type"]])

    client = caveclient.CAVEclient(DATASTACK)

    def q(colname):
        for attempt in range(5):
            try:
                return client.materialize.query_table("synapses_v2", filter_in_dict={colname: mi1})
            except Exception as e:
                if attempt == 4:
                    raise
                print(f"  (retry {attempt+1}/5 on {colname}: {str(e)[:60]})")
                time.sleep(6)

    t = time.time()
    down = q("pre_pt_root_id")
    down = down[(down["post_pt_root_id"] != down["pre_pt_root_id"])
                & (~down["post_pt_root_id"].isin(mi1set))]
    t = throttle(t)
    up = q("post_pt_root_id")
    up = up[(up["pre_pt_root_id"] != up["post_pt_root_id"])
            & (~up["pre_pt_root_id"].isin(mi1set))]

    e1 = down.rename(columns={"post_pt_root_id": "partner", "pre_pt_root_id": "mi1"})[["partner", "mi1"]].assign(d="down")
    e2 = up.rename(columns={"pre_pt_root_id": "partner", "post_pt_root_id": "mi1"})[["partner", "mi1"]].assign(d="up")
    edges = pd.concat([e1, e2], ignore_index=True)

    cov = edges.groupby("partner").agg(n_mi1=("mi1", "nunique"), n_syn=("mi1", "size")).reset_index()
    # direction split
    cov = cov.merge(e1.groupby("partner")["mi1"].nunique().rename("n_mi1_down"), on="partner", how="left")
    cov = cov.merge(e2.groupby("partner")["mi1"].nunique().rename("n_mi1_up"), on="partner", how="left")
    cov = cov.join(ann, on="partner").fillna({"n_mi1_down": 0, "n_mi1_up": 0})
    cov = cov.sort_values("n_mi1", ascending=False).reset_index(drop=True)
    cov.to_csv(fig("mi1_net_partners.csv"), index=False)

    pd.set_option("display.width", 160)
    print(f"\ntotal distinct partners of the {len(mi1)} Mi1: {len(cov)}")
    print(f"\n=== top partners by # of Mi1 touched (the 'net' candidates) ===")
    show = cov.head(20).copy()
    show["partner"] = show["partner"].astype(str)
    print(show[["partner", "n_mi1", "n_mi1_up", "n_mi1_down", "n_syn",
                "superclass", "class", "subclass", "type"]].to_string(index=False))
    top = cov.iloc[0]
    print(f"\nbest net: partner {int(top['partner'])} touches {int(top['n_mi1'])}/{len(mi1)} Mi1 "
          f"({int(top['n_mi1'])/len(mi1)*100:.0f}%)")


if __name__ == "__main__":
    main()
