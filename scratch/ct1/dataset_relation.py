"""Are optic-lobe:v1.1 and male-CNS:v1.0 distinct volumes, or is OL a subset of mCNS?
Decisive checks: dataset metadata/uuid, ROI inventory, and whether identical bodyIds
carry identical soma coordinates in both."""
import pandas as pd
from neuprint import NeuronCriteria as NC, fetch_neurons
from faab.clients.neuprint import client as npclient, MALE_CNS, OPTIC_LOBE


def main():
    cm = npclient(MALE_CNS)
    co = npclient(OPTIC_LOBE)

    for c, name in [(cm, "male-CNS:v1.0"), (co, "optic-lobe:v1.1")]:
        m = c.meta
        print(f"\n=== {name} ===")
        for k in ("uuid", "dataset", "lastDatabaseEdit", "totalPreCount", "totalPostCount"):
            if k in m:
                print(f"  {k}: {m[k]}")
        desc = str(m.get("info", m.get("description", "")))[:200]
        if desc:
            print(f"  info/description: {desc}")
        rois = m.get("superLevelRois", [])
        print(f"  superLevelRois: n={len(rois)}")
        sides = {"(R)": sum(1 for r in rois if r.endswith("(R)")),
                 "(L)": sum(1 for r in rois if r.endswith("(L)"))}
        print(f"    right-side ROIs {sides['(R)']}, left-side ROIs {sides['(L)']}")
        optic = [r for r in rois if any(r.startswith(p) for p in ("ME", "LO", "LOP", "LA", "AME"))]
        print(f"    optic ROIs: {sorted(optic)}")

    # cross-check: do identical bodyIds carry identical soma coordinates?
    print("\n=== same bodyId -> same soma position? ===")
    om, _ = fetch_neurons(NC(status="Traced", cropped=False), client=co)
    om = om[om.somaLocation.notna()].head(40)
    ids = om.bodyId.tolist()
    mm, _ = fetch_neurons(NC(bodyId=ids), client=cm)
    mm = mm[mm.somaLocation.notna()]
    j = om[["bodyId", "somaLocation", "type"]].merge(
        mm[["bodyId", "somaLocation", "type"]], on="bodyId", suffixes=("_ol", "_mcns"))
    if not len(j):
        print("  no shared bodyIds among the sample -> DISTINCT segmentations")
        return
    same = sum(1 for _, r in j.iterrows() if list(r.somaLocation_ol) == list(r.somaLocation_mcns))
    print(f"  sampled {len(ids)} OL bodyIds -> {len(j)} also exist in male-CNS")
    print(f"  identical soma coordinates: {same}/{len(j)}")
    print(j.head(6).to_string(index=False))
    print("\nDONE")


if __name__ == "__main__":
    main()
