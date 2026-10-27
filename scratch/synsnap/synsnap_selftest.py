"""Offline test of faab.synsnap on a tiny synthetic snapshot (same layout as the real one, R-style meta.json with
one-element arrays): chain replay (latest t wins per side), head tables, queries by pre / post / both, edges,
static lookup, at() by time, and the catch-up patch merge. Writes to tmp/synsnap_selftest/."""
import json
import os
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from faab import synsnap

R = "/Users/davidh/Data/Vibe/FAAB/tmp/synsnap_selftest"
if os.path.exists(R):
    shutil.rmtree(R)
os.makedirs(R)
# real-size root ids: a float64 detour rounds these (the 2026-10-27 bug), small ids would hide it
A, B, C, D, E, F, G = (648518347699279442, 648518347699279443, 648518347590648323, 648518347495897232,
                       648518347579284782, 648518347478062063, 648518347557109359)
ids = np.arange(1, 7)
pq.write_table(pa.table({"id": ids.astype(np.int32), "pre_sv": ids * 10, "post_sv": ids * 10 + 1,
                         "pre_x": ids, "pre_y": ids, "pre_z": ids, "post_x": ids, "post_y": ids, "post_z": ids,
                         "size": ids}), os.path.join(R, "static.parquet"))
# base v1: (pre, post) per synapse id
base = {1: (A, B), 2: (A, B), 3: (A, C), 4: (B, C), 5: (C, A), 6: (A, A)}


def meta(tag, ts, base_=None, parent=None, kind="version"):
    os.makedirs(os.path.join(R, tag), exist_ok=True)
    json.dump({"tag": [tag], "timestamp": [ts], "base": [base_] if base_ else [], "parent": [parent] if parent else [],
               "kind": [kind]}, open(os.path.join(R, tag, "meta.json"), "w"))


os.makedirs(os.path.join(R, "v1"))
order = sorted(base, key=lambda i: (base[i][0], i))
pq.write_table(pa.table({"pre_root": [base[i][0] for i in order], "post_root": [base[i][1] for i in order],
                         "id": np.array(order, np.int32)}), os.path.join(R, "v1", "by_pre.parquet"))
meta("v1", "2026-10-01 14:00:00 UTC")


def log(tag, rows):
    """exact int64 columns straight from Python ints (a pandas frame would round the nullable 18-digit ids)"""
    cols = list(zip(*rows))
    tb = pa.table({"id": pa.array(cols[0], pa.int32()),
                   "t": pa.array([pd.Timestamp(x).to_pydatetime() for x in cols[1]], pa.timestamp("us")),
                   **{n: pa.array(c, pa.int64()) for n, c in zip(["pre_root", "post_root", "old_pre", "old_post"], cols[2:])}})
    pq.write_table(tb, os.path.join(R, tag, "log.parquet"))


meta("c1", "2026-10-02 14:00:00 UTC", "v1", "v1")
log("c1", [(3, "2026-10-02 10:00", D, None, A, None), (4, "2026-10-02 11:00", None, D, None, C)])
meta("c2", "2026-10-03 14:00:00 UTC", "v1", "c1")
log("c2", [(3, "2026-10-03 09:00", E, None, D, None), (5, "2026-10-03 09:30", F, F, C, A)])

snap = synsnap.Snapshot(R, log=lambda m: None)
ok = True


def check(name, got, want):
    global ok
    good = got == want
    ok &= good
    print(("PASS " if good else "FAIL ") + name + ("" if good else f": got {got}, want {want}"))


check("tags", list(snap.tags().tag), ["v1", "c1", "c2"])
check("chain c2", snap.chain("c2"), ["c2", "c1", "v1"])
check("at time between c1 and c2", snap.at("2026-10-02 20:00"), "c1")
s = snap.synapses(pre=[A], when="v1")
check("v1 pre=A ids", sorted(s.id), [1, 2, 3, 6])
s = snap.synapses(pre=[A], when="c2")
check("c2 pre=A ids (3 moved to D then E)", sorted(s.id), [1, 2, 6])
check("c2 pre=E ids", sorted(snap.synapses(pre=[E], when="c2").id), [3])
check("c1 pre=D ids", sorted(snap.synapses(pre=[D], when="c1").id), [3])
check("c2 post=D ids (4 post changed, pre kept)", sorted(snap.synapses(post=[D], when="c2").id), [4])
check("c2 synapse 4 pre stays B", int(snap.synapses(post=[D], when="c2").pre_root.iloc[0]), B)
check("c2 pre=F & post=F (5 both sides)", sorted(snap.synapses(pre=[F], post=[F], when="c2").id), [5])
e = snap.edges(pre=[A], when="c2")
check("c2 edges from A (self 6 dropped)", sorted(map(tuple, e[["pre", "post", "n"]].to_numpy().tolist())), [(A, B, 2)])
st = snap.static([2, 5])
check("static", st.pre_sv.tolist(), [20, 50])
patch = pd.DataFrame({"id": [1, 4], "pre_root": [G, B], "post_root": [B, G]})
df = snap.synapses(pre=[A, B], when="c2")
m = synsnap.Snapshot._apply_patch(df, patch, [A, B], None)
check("patch: 1 leaves A, 4 kept (pre B)", sorted(m.id), [2, 4, 6])
print("ALL PASS" if ok else "SOME FAILED")
