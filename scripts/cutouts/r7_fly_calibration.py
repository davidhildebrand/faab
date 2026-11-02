"""Fly calibration for missed R7 synapses (David 2026-11-02): do Aedes R7s have fewer predicted output synapses than fly
R7s, and is the shortfall specific to R7? Output synapses per cell (pre-post pairs) for R1-R6, R7, and R8 in
  - Aedes (FAAB): PRs tab rows with status proofread / checked / adequate, roots at the row's xyz now, live synapses_v2;
  - fly, two independent specimens: male-CNS v1.0 (neuPrint 'downstream' / 'upstream' counts; both eyes) and FlyWire
    FAFB release 783 (public; a seeded random sample of up to N_FW cells per type, cleft score >= 50).
Species and detectors differ, so the comparison that matters is the RATIO within each dataset: R7 outputs relative to
R1-R6 and to R8. If Aedes R7 / R1-R6 is much lower than the fly's, Aedes R7 outputs are specifically short (missed by
the prediction, or a real species difference - the cutouts tell which).
Read-only; CAVE calls >= 2 s apart, one at a time. Usage: r7_fly_calibration.py PRS_TAB.xlsx"""
import datetime
import sys
import time

import numpy as np
import pandas as pd
import neuprint

from faab.clients.cave import client as cave_client, roots_at_xyz, throttle
from faab.clients.neuprint import client as np_client, MALE_CNS
from faab.config import fig
from faab.pr import prs_tab as T

GOOD = ("proofread", "checked", "adequate")
TYPES = {"R1-R6": "R1-R6", "R7": "R7", "R8": "R8"}
N_FW, SEED = 80, 20261102


def fly_type(t):
    t = str(t)
    return "R1-R6" if t in ("R1-R6", "R1-6") or t[:2] in ("R1", "R2", "R3", "R4", "R5", "R6") else \
        "R7" if t.startswith("R7") else "R8" if t.startswith("R8") else ""


def aedes(book, client, t):
    wb, ws = T.load(book)
    left = next((r for r in range(1, ws.max_row + 1) if T.text(ws, r, 1).startswith(">>> additional L side")),
                ws.max_row + 1)
    rows = []
    for b in T.blocks(ws):
        for r in b.rows:
            ty = TYPES.get(T.get(ws, b, r, "class").replace("–", "-"))
            if ty and T.get(ws, b, r, "status").lower() in GOOD:
                rows.append(dict(type=ty, side="L" if b.title_row > left else "R", group=b.group, xyz=T.get(ws, b, r, "xyz")))
    D = pd.DataFrame(rows)
    t = throttle(t)
    D["cell"] = roots_at_xyz(D.xyz.tolist(), client_=client)
    t = time.time()
    D = D[D.cell != 0].drop_duplicates("cell")
    roots = [int(r) for r in D.cell]
    now = datetime.datetime.now(datetime.timezone.utc)
    n = {}
    for col in ("pre_pt_root_id", "post_pt_root_id"):
        t = throttle(t)
        s = client.materialize.live_query("synapses_v2", timestamp=now, filter_in_dict={col: roots},
                                          select_columns=["pre_pt_root_id", "post_pt_root_id"])
        t = time.time()
        s = s[s.pre_pt_root_id != s.post_pt_root_id]
        n[col] = s[col].value_counts()
    D["outputs"] = D.cell.map(n["pre_pt_root_id"]).fillna(0).astype(int)
    D["inputs"] = D.cell.map(n["post_pt_root_id"]).fillna(0).astype(int)
    D["dataset"] = "Aedes (FAAB, predicted)"
    return D, t


def male_cns():
    c = np_client(MALE_CNS)
    N = neuprint.fetch_neurons(neuprint.NeuronCriteria(type="^R[1-8].*", regex=True), client=c)[0]
    N["type"] = N.type.map(fly_type)
    N = N[N.type != ""]
    return pd.DataFrame(dict(dataset="fly male-CNS v1.0", type=N.type, side="", cell=N.bodyId,
                             outputs=N.downstream, inputs=N.upstream, tbars=N.pre))


def flywire(t):
    from fafbseg import flywire as fw
    t = throttle(t)
    ann = fw.get_hierarchical_annotations(materialization=783, dataset="public")
    t = time.time()
    ann = ann[ann.root_id.notna()].copy()
    ann["type"] = ann.cell_type.map(fly_type)
    rng = np.random.default_rng(SEED)
    out = []
    for ty in ("R1-R6", "R7", "R8"):
        ids = ann[ann.type == ty].root_id.astype(np.int64).unique()
        pick = [int(i) for i in rng.choice(ids, min(N_FW, len(ids)), replace=False)]
        cnt = {}
        for up in (False, True):
            t = throttle(t)
            cn = fw.get_connectivity(pick, upstream=up, downstream=not up, materialization=783, dataset="public",
                                     min_score=50, progress=False)
            t = time.time()
            key = "post" if up else "pre"
            cnt[up] = cn[cn[key].astype(np.int64).isin(pick)].groupby(cn[key].astype(np.int64)).weight.sum()
        print(f"FlyWire {ty}: {len(ids)} cells annotated, {len(pick)} sampled", flush=True)
        out.append(pd.DataFrame(dict(dataset="fly FlyWire 783", type=ty, side="", cell=pick,
                                     outputs=[int(cnt[False].get(i, 0)) for i in pick],
                                     inputs=[int(cnt[True].get(i, 0)) for i in pick])))
    return pd.concat(out, ignore_index=True), t


def main():
    client, t = cave_client(), throttle(0.0)
    A, t = aedes(sys.argv[1], client, t)
    M = male_cns()
    F, t = flywire(t)
    D = pd.concat([A, M, F], ignore_index=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dd%H%M%StUTC")
    D.to_csv(fig(f"cutouts/{stamp}_r7_fly_calibration_cells.tsv"), sep="\t", index=False)
    S = D.groupby(["dataset", "type"]).agg(cells=("cell", "size"), outputs_median=("outputs", "median"),
                                           outputs_q25=("outputs", lambda x: x.quantile(.25)),
                                           outputs_q75=("outputs", lambda x: x.quantile(.75)),
                                           inputs_median=("inputs", "median")).reset_index()
    med = S.set_index(["dataset", "type"]).outputs_median
    for ds in S.dataset.unique():
        S.loc[S.dataset == ds, "outputs_vs_R1-R6"] = (S[S.dataset == ds].outputs_median / med.get((ds, "R1-R6"), np.nan)).round(2)
        S.loc[S.dataset == ds, "outputs_vs_R8"] = (S[S.dataset == ds].outputs_median / med.get((ds, "R8"), np.nan)).round(2)
    S.to_csv(fig(f"cutouts/{stamp}_r7_fly_calibration.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 220)
    print(S.to_string(index=False))
    a = A[A.type == "R7"].groupby("side").outputs.median().to_dict()
    print(f"Aedes R7 median outputs by side: {a}")
    print(f"written {fig(f'cutouts/{stamp}_r7_fly_calibration.tsv')}")


if __name__ == "__main__":
    main()
