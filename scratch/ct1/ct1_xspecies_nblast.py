"""Cross-species nblast_align: Aedes A/B vs Drosophila CT1 (male-CNS).

A = 648518347638241763, B = 648518347495933144  (putative Aedes CT1)
Fly CT1: male-CNS bodies 10009 (CT1_L), 10157 (CT1_R)

These are huge cells, so each is resampled to a fixed ~POINTS-point dotprops budget;
nblast_align (rigid + scale) handles the cross-species size difference. Positive
control = CT1_L vs CT1_R (same type, same brain). Read-only, throttled CAVE.
"""
from faab.config import REPO
import os
import time
import pickle
import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import navis
from navis.interfaces import neuprint as npn
from fafbseg import flywire

from faab.clients.cave import DATASTACK, latest_roots, throttle
from faab.clients.neuprint import client as npclient, MALE_CNS

OUT = os.path.join(REPO, "figures")
AEDES = {"A": 648518347638241763, "B": 648518347495933144}
FLY = {"CT1_L": 10009, "CT1_R": 10157}
POINTS = 2500


def dp_of(neuron, name):
    rs = max(neuron.cable_length / POINTS, 0.5)
    dp = navis.make_dotprops(neuron, k=5, resample=rs)
    dp.units = "um"
    dp.name = name
    dp.id = name
    return dp


def main():
    # ---- Drosophila CT1 (male-CNS, 8 nm voxels) ----
    c = npclient(MALE_CNS)
    fly = npn.fetch_skeletons(list(FLY.values()), client=c, heal=True)
    flum = {}
    for n in fly:
        n.units = "8 nm"
        flum[int(n.id)] = n.convert_units("um")
    print("fetched fly CT1:", {k: int(flum[v].cable_length) for k, v in FLY.items()}, "um cable", flush=True)

    # ---- Aedes A/B (L2 skeletons, nm -> um) ----
    roots, _ = latest_roots(list(AEDES.values()))
    t = throttle(time.time())
    sk = flywire.get_l2_skeleton(roots, dataset=DATASTACK, max_threads=1,
                                 omit_failures=True, progress=False)
    aed = {}
    for s in sk:
        s2 = s / 1000.0
        s2.units = "um"
        aed[int(s.id)] = s2
    print("fetched aedes:", {k: int(aed[v].cable_length) for k, v in zip("AB", roots)}, "um cable", flush=True)

    order = ["A", "B", "CT1_L", "CT1_R"]
    src = {"A": aed[roots[0]], "B": aed[roots[1]],
           "CT1_L": flum[FLY["CT1_L"]], "CT1_R": flum[FLY["CT1_R"]]}
    dps = []
    for nm in order:
        d = dp_of(src[nm], nm)
        dps.append(d)
        print(f"  {nm}: cable {src[nm].cable_length:8.0f} um  ->  {len(d.points)} dotprops points", flush=True)

    print("running nblast_align (rigid+scale, all-by-all) ...", flush=True)
    nl = navis.NeuronList(dps)
    S = navis.nblast_align(nl, nl, align_method="rigid", two_way_align=False,
                           n_cores=4, progress=False, align_kwargs={"scale": True})
    M = (S.values + S.values.T) / 2.0
    np.fill_diagonal(M, np.nan)

    lab = order
    print("\n=== nblast_align scores (symmetrised; 1.0 = identical) ===")
    print("            " + "".join(f"{n:>9}" for n in lab))
    for i, n in enumerate(lab):
        row = "".join(f"{M[i, j]:>9.3f}" if not np.isnan(M[i, j]) else f"{'-':>9}" for j in range(len(lab)))
        print(f"    {n:<8}{row}")
    print("\nkey cross-species scores:")
    idx = {n: i for i, n in enumerate(lab)}
    for a in ("A", "B"):
        for f in ("CT1_L", "CT1_R"):
            print(f"    {a} vs fly {f}: {M[idx[a], idx[f]]:.3f}")
    print(f"  fly CT1_L vs CT1_R (same-type reference): {M[idx['CT1_L'], idx['CT1_R']]:.3f}")
    print(f"  Aedes A vs B: {M[idx['A'], idx['B']]:.3f}")
    pickle.dump({"labels": lab, "M": M}, open(os.path.join(OUT, "ct1_xspecies_nblast.pkl"), "wb"))
    print("\nsaved ct1_xspecies_nblast.pkl\nDONE", flush=True)


if __name__ == "__main__":
    main()
