"""Re-check #2: LC cells from the user's link (red layer) vs MeTu4 (green layer +
ALL other annotated MeTu4 from the live table). Fetch morphology FRESH; resolve
the table MeTu4 via immutable supervoxel_id -> current root. Read-only, throttled."""
from faab.config import REPO
import os
import time
import pickle
import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import seaserpent as ss
import caveclient
from fafbseg import flywire
from faab.clients.cave import DATASTACK, throttle

OUT = os.path.join(REPO, "figures")
SERVER = "https://flytable.mrc-lmb.cam.ac.uk/"

# from the user's link: green "wclee_aedes_brain" layer (segmentQuery "/MeTu4")
GREEN_METU4 = [648518347579426472, 648518347481594443, 648518347646665307, 648518347520445481,
    648518347508923343, 648518347566091556, 648518347581688207, 648518347524979786,
    648518347482009629, 648518347544441754, 648518347575757029, 648518347594762021,
    648518347534048345, 648518347423730256, 648518347489627565, 648518347505179378,
    648518347522594452, 648518347543773262, 648518347573233607, 648518347583796422,
    648518347584020145, 648518347594532197, 648518347567657002, 648518347610881112,
    648518347618659468, 648518347651544455, 648518347399925553, 648518347474283276,
    648518347510182261, 648518347518071079, 648518347522600852, 648518347526800512,
    648518347531422046, 648518347534256767, 648518347542074235, 648518347564167260,
    648518347556049634, 648518347564131164, 648518347564169820, 648518347567749674,
    648518347577293251, 648518347577320387, 648518347577354947, 648518347579443624,
    648518347598309265, 648518347605567167, 648518347606512008, 648518347611377524,
    648518347622318270, 648518347759365666, 648518347623021846, 648518347636981878,
    648518347659255938, 648518347618681996, 648518347598310289, 648518347488774421,
    648518347622956793, 648518347564233564, 648518347564238172, 648518347527931318]

# from the user's link: red "LC" layer
RED_LC = [648518347399880241, 648518347463566622, 648518347475132830, 648518347482175517,
    648518347485707273, 648518347493033197, 648518347498185039, 648518347503924240,
    648518347513990684, 648518347527900342, 648518347542506264, 648518347543603534,
    648518347545551861, 648518347555914090, 648518347557070447, 648518347564132188,
    648518347564133212, 648518347567696938, 648518347581682575, 648518347581683343,
    648518347583809222, 648518347586260383, 648518347598306193, 648518347599353155,
    648518347605610431, 648518347606524296, 648518347623019286, 648518347660289039,
    648518347687133185, 648518347689315303, 648518347699048018, 648518347700868494,
    648518347704420670]


def main():
    print(f"link: green MeTu4 {len(GREEN_METU4)}, red LC {len(RED_LC)}", flush=True)
    # all other annotated MeTu4 from the live table -> supervoxel -> current root
    meta = ss.Table("aedes_main", server=SERVER).to_frame()
    meta.columns = [str(c) for c in meta.columns]
    m4 = meta[meta["subclass"].astype(str) == "MeTu4"]
    m4 = m4[m4["supervoxel_id"].notna() & (m4["supervoxel_id"].astype(str) != "")]
    svid = m4["supervoxel_id"].astype("int64").tolist()
    client = caveclient.CAVEclient(DATASTACK)
    t = throttle(time.time())
    table_m4 = [int(x) for x in client.chunkedgraph.get_roots(svid)]
    print(f"table MeTu4 (fresh supervoxel->root): {len(table_m4)}", flush=True)

    metu_ref = sorted(set(GREEN_METU4) | set(table_m4))
    lc = list(dict.fromkeys(RED_LC))
    overlap = set(lc) & set(metu_ref)
    if overlap:
        print(f"WARNING: {len(overlap)} ids in BOTH LC and MeTu4: {sorted(overlap)}")
        lc = [i for i in lc if i not in overlap]
    print(f"MeTu4 reference (green union table): {len(metu_ref)} | LC to test: {len(lc)}", flush=True)

    allids = list(dict.fromkeys(lc + metu_ref))
    lab = {}
    for i in lc:
        lab[i] = "LC"
    for i in metu_ref:
        lab.setdefault(i, "MeTu4")

    t = throttle(t)
    dp = flywire.get_l2_dotprops(allids, dataset=DATASTACK, max_threads=1,
                                 omit_failures=True, progress=True)
    by = {}
    for d in dp:
        d2 = d / 1000.0; d2.units = "um"; by[int(d.id)] = d2
    npts = {i: len(by[i].points) if i in by else 0 for i in allids}
    for grp in ("LC", "MeTu4"):
        v = np.array([npts[i] for i in allids if lab[i] == grp])
        print(f"  {grp:6}: n={len(v)}, fetched {(v>0).sum()}, >=50pts {(v>=50).sum()}, "
              f"median {int(np.median(v))}", flush=True)

    pickle.dump({"dotprops": by, "label": lab, "lc": lc, "metu_ref": metu_ref},
                open(os.path.join(OUT, "metu_lc2.pkl"), "wb"))
    print("saved metu_lc2.pkl\nDONE", flush=True)


if __name__ == "__main__":
    main()
