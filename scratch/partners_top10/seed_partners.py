"""Query all synaptic partners of a seed cell (both directions) and build a
seed+partners neuroglancer link. Resolves the seed to its CURRENT root first
(root_ids change). Read-only, throttled."""
from faab.config import REPO
import os
import time
import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import caveclient
from faab.clients.cave import DATASTACK, latest_roots, throttle
import faab.nglink as nglink
from faab.config import fig

SEED = 648518347777360890
OUT = os.path.join(REPO, "figures")


def qtable(client, col, val):
    for a in range(6):
        try:
            return client.materialize.query_table("synapses_v2", filter_in_dict={col: [val]})
        except Exception as e:
            if a == 5:
                raise
            print(f"  (retry {a+1}/6 {col}: {str(e)[:45]})", flush=True)
            time.sleep(6)


def main():
    client = caveclient.CAVEclient(DATASTACK)
    try:
        roots, _ = latest_roots([SEED])
        cur = roots[0]
    except Exception as e:
        cur = SEED
        print(f"  (root resolve failed, using seed as-is: {str(e)[:50]})")
    print(f"seed {SEED} -> current {cur}  (changed: {cur != SEED})", flush=True)

    t = throttle(time.time())
    out = qtable(client, "pre_pt_root_id", cur)     # seed -> X (downstream)
    t = throttle(t)
    inp = qtable(client, "post_pt_root_id", cur)    # X -> seed (upstream)
    out = out[(out.post_pt_root_id != 0) & (out.post_pt_root_id != cur)]
    inp = inp[(inp.pre_pt_root_id != 0) & (inp.pre_pt_root_id != cur)]
    outp = out.groupby("post_pt_root_id").size().sort_values(ascending=False)
    inpp = inp.groupby("pre_pt_root_id").size().sort_values(ascending=False)

    print(f"\nDOWNSTREAM (seed -> partner): {len(out):4d} synapses, {len(outp):3d} partners")
    print(f"UPSTREAM   (partner -> seed): {len(inp):4d} synapses, {len(inpp):3d} partners")
    print(f"combined distinct partners: {len(set(outp.index) | set(inpp.index))}")
    print("\ntop downstream partners (root : #syn):")
    for r, n in outp.head(12).items():
        print(f"  {int(r)} : {n}")
    print("top upstream partners (root : #syn):")
    for r, n in inpp.head(12).items():
        print(f"  {int(r)} : {n}")

    out_ids = [str(int(r)) for r in outp.index]
    in_ids = [str(int(r)) for r in inpp.index]
    open(fig("seed_out_ids.txt"), "w").write("\n".join(out_ids) + "\n")
    open(fig("seed_in_ids.txt"), "w").write("\n".join(in_ids) + "\n")

    spec = {"meshes": [{"kind": "brain", "alpha": 1.0}, {"kind": "AL", "alpha": 0.45}],
            "neurons": [
                {"name": f"upstream -> seed ({len(in_ids)})", "color": "#ff9800", "ids": in_ids},
                {"name": f"seed -> downstream ({len(out_ids)})", "color": "#00bcd4", "ids": out_ids},
                {"name": f"seed {str(cur)[-6:]}", "color": "#00d65d", "ids": [str(cur)]}],
            "view": "anterior", "layout": "3d", "select": f"seed {str(cur)[-6:]}"}
    url = nglink.to_url(nglink.build(spec))
    open(fig("seed_partners_ng.txt"), "w").write(url + "\n")
    print("\n" + url)
    print("\nDONE")


if __name__ == "__main__":
    main()
