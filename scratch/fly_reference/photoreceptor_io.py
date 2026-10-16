"""Per-cell output (and input) connectivity for 8 putative aedes photoreceptors,
with fresh Drosophila reference numbers for R1-R6 / R7 / R8.

Read-only. CAVE calls batched (2 synapse queries total) + throttled.
root_ids change with proofreading -> currency is checked and reported.
"""
from faab.config import REPO
import os
import time
import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import caveclient
from faab.sources.cave import DATASTACK, throttle

OUT = os.path.join(REPO, "figures")

R16 = [648518347649903570, 648518347649905106, 648518347642147360,
       648518347626884562, 648518347643620818, 648518347567256005]
R78 = [648518347777360890, 648518347626873298]
LABEL = {r: "R1-R6" for r in R16}
LABEL.update({r: "R7-R8" for r in R78})
ALL = R16 + R78


def q(client, col, ids):
    for a in range(6):
        try:
            return client.materialize.query_table("synapses_v2", filter_in_dict={col: ids})
        except Exception as e:
            if a == 5:
                raise
            print(f"  (retry {a+1}/6 on {col}: {str(e)[:45]})", flush=True)
            time.sleep(6)


def main():
    client = caveclient.CAVEclient(DATASTACK)
    t = throttle(time.time())
    try:
        latest = np.asarray(client.chunkedgraph.is_latest_roots(ALL))
        stale = [r for r, ok in zip(ALL, latest) if not ok]
        print(f"root currency: {int(latest.sum())}/{len(ALL)} current"
              + (f" | STALE (results unreliable): {stale}" if stale else " | all current"))
    except Exception as e:
        print("currency check failed:", str(e)[:80])

    # ---- two batched synapse queries ----
    t = throttle(t)
    out = q(client, "pre_pt_root_id", ALL)      # these cells -> partners
    t = throttle(t)
    inp = q(client, "post_pt_root_id", ALL)     # partners -> these cells
    out = out[(out.post_pt_root_id != 0) & (out.post_pt_root_id != out.pre_pt_root_id)]
    inp = inp[(inp.pre_pt_root_id != 0) & (inp.pre_pt_root_id != inp.post_pt_root_id)]

    rows = []
    per_cell_partners = {}
    for r in ALL:
        o = out[out.pre_pt_root_id == r]
        i = inp[inp.post_pt_root_id == r]
        op = o.groupby("post_pt_root_id").size().sort_values(ascending=False)
        ip = i.groupby("pre_pt_root_id").size().sort_values(ascending=False)
        per_cell_partners[r] = (op, ip)
        rows.append({"root": r, "class": LABEL[r],
                     "out_syn": len(o), "out_partners": len(op),
                     "in_syn": len(i), "in_partners": len(ip),
                     "top_out_syn": int(op.iloc[0]) if len(op) else 0,
                     "singleton_out": int((op == 1).sum())})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 170)
    print("\n=== FAAB (aedes) per-cell connectivity ===")
    print(df.to_string(index=False))
    print("\nclass totals / medians:")
    for cls in ("R1-R6", "R7-R8"):
        d = df[df["class"] == cls]
        print(f"  {cls}: n={len(d)}  out_syn median {d.out_syn.median():.0f} "
              f"(range {d.out_syn.min()}-{d.out_syn.max()})  "
              f"in_syn median {d.in_syn.median():.0f} "
              f"(range {d.in_syn.min()}-{d.in_syn.max()})")

    print("\n=== per-cell OUTPUT partner breakdown (root : #synapses) ===")
    for r in ALL:
        op, ip = per_cell_partners[r]
        print(f"\n--- {r}  [{LABEL[r]}]  {int(op.sum())} out-syn -> {len(op)} partners "
              f"| {int(ip.sum())} in-syn <- {len(ip)} partners")
        if len(op):
            for pid, n in op.head(10).items():
                print(f"      -> {int(pid)} : {n}")
            if len(op) > 10:
                print(f"      ... +{len(op)-10} more partners (all with "
                      f"{int(op.iloc[10])} or fewer synapses)")
        else:
            print("      (no output synapses)")

    df.to_csv(os.path.join(OUT, "photoreceptor_io.csv"), index=False)
    allp = []
    for r in ALL:
        op, ip = per_cell_partners[r]
        for pid, n in op.items():
            allp.append({"cell": r, "class": LABEL[r], "direction": "output",
                         "partner": int(pid), "n_syn": int(n)})
        for pid, n in ip.items():
            allp.append({"cell": r, "class": LABEL[r], "direction": "input",
                         "partner": int(pid), "n_syn": int(n)})
    pd.DataFrame(allp).to_csv(os.path.join(OUT, "photoreceptor_partners.csv"), index=False)
    print("\nsaved photoreceptor_io.csv + photoreceptor_partners.csv")
    print("DONE")


if __name__ == "__main__":
    main()
