"""Validate faab.synsnap against live CAVE (read-only, slow: every CAVE call >= 15 s apart):
1. build / load the root table at the newest checkpoint (timed);
2. PR group 1 outputs from the snapshot at the newest checkpoint and at 'now' (catch-up) vs ONE live_query of the same
   8 PRs: synapse ids and partner roots must agree;
3. position units: static pre_x/y/z vs the live query's pre_pt_position for the same synapse ids;
4. speed of a large query: all inputs of the 411 T4/T5 candidates (post side) from the snapshot."""
import datetime
import os
import time

import numpy as np
import pandas as pd

from faab import synsnap
from faab.clients.cave import client as cave_client, throttle
from faab.config import DATA, TMP

t0 = time.time()
snap = synsnap.Snapshot(log=lambda m: print(m, flush=True))
tags = snap.tags()
tag = tags.tag.iloc[-1]
print(f"{len(tags)} local checkpoints, newest {tag} at {tags.timestamp.iloc[-1]}", flush=True)
snap.load(tag)
print(f"root table at {tag} ready in {time.time() - t0:.0f} s", flush=True)

client = cave_client()
last = [0.0]


def wait():
    last[0] = throttle(last[0], gap=15.0)


prs = pd.read_csv(os.path.join(DATA, "pr_groups.tsv"), sep="\t", dtype=str)
g1 = [int(x) for x in prs[prs.group == "1"].root_id_curr]
wait()
ok = np.asarray(client.chunkedgraph.is_latest_roots(g1)).astype(bool)
cur = []
for r, good in zip(g1, ok):
    if good:
        cur.append(r)
    else:
        wait()
        cur += [int(x) for x in np.atleast_1d(client.chunkedgraph.get_latest_roots(r))]
print(f"PR group 1: {len(cur)} current roots", flush=True)

t1 = time.time()
s_tag = snap.synapses(pre=cur, when=tag)
print(f"snapshot at {tag}: {len(s_tag)} output synapses of PR group 1 ({time.time() - t1:.2f} s)", flush=True)
t1 = time.time()
s_now = snap.synapses(pre=cur, when="now", client=client, pause_s=10.0)
print(f"snapshot at now (catch-up): {len(s_now)} synapses ({time.time() - t1:.0f} s incl. CAVE)", flush=True)
now = datetime.datetime.now(datetime.timezone.utc)
wait()
live = client.materialize.live_query("synapses_v2", timestamp=now, filter_in_dict={"pre_pt_root_id": cur})
print("live columns:", list(live.columns), flush=True)
live = live[["id", "pre_pt_root_id", "post_pt_root_id"] + [c for c in live.columns if c.startswith("pre_pt_position")]]
print(f"live: {len(live)} synapses", flush=True)
a = s_now.set_index("id")
b = live.set_index("id")
common = a.index.intersection(b.index)
print(f"ids: snapshot-now {len(a)}, live {len(b)}, common {len(common)}, only snapshot {len(a.index.difference(b.index))}, "
      f"only live {len(b.index.difference(a.index))}")
mis_post = int((a.loc[common, "post_root"].to_numpy() != b.loc[common, "post_pt_root_id"].to_numpy()).sum())
print(f"post root mismatches on common ids: {mis_post}")
x = s_now[s_now.pre_root != s_now.post_root]  # reuse the catch-up above (no second round of CAVE calls)
e_snap = x.groupby(["pre_root", "post_root"]).size().rename("n").reset_index().rename(columns={"pre_root": "pre", "post_root": "post"})
e_live = live[live.pre_pt_root_id != live.post_pt_root_id].groupby(["pre_pt_root_id", "post_pt_root_id"]).size()
m = e_snap.set_index(["pre", "post"]).n.reindex(e_live.index).fillna(0)
print(f"edges: snapshot {len(e_snap)}, live {len(e_live)}, identical counts {(m.to_numpy() == e_live.to_numpy()).mean():.1%}")

st = snap.static(common[:50].to_numpy())
pos_col = [c for c in live.columns if c.startswith("pre_pt_position")]
lv = b.loc[st.id]
if len(pos_col) == 1:
    lp = np.vstack(lv[pos_col[0]].to_numpy())
else:
    lp = lv[sorted(pos_col)].to_numpy()
sp = st[["pre_x", "pre_y", "pre_z"]].to_numpy().astype(float)
print("position ratio live / static (x, y, z), median over 50 synapses:", np.round(np.median(lp / sp, axis=0), 3))

t1 = time.time()
sv = pd.read_csv(os.path.join(TMP, "t4t5_query_survey.tsv"), sep="\t", dtype=str)
e_in = snap.edges(post=[int(x) for x in sv.now], when=tag)
print(f"inputs of the 411 T4/T5 candidates at {tag}: {int(e_in.n.sum())} synapses, {len(e_in)} edges in {time.time() - t1:.2f} s")
print(f"total {time.time() - t0:.0f} s")
