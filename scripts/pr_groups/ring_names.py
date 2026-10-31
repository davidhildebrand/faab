"""Fly names from the compact bundle ALONE, compared with the terminal-plane names.

Input: figures/pr_bundle_track_group<g>.tsv (cutout centroids per slice; the 'cell' column carries the terminal-plane
names and is used only for the comparison). Ring naming uses geometry only: in every complete slice where the six
R1-R6 lie within RING_MAX um of the bundle center, each outer axon's angle around the center is measured relative to
the R7's angle; over the compact stretch the median relative angle ranks the six outer axons around the ring. The two
flanking the R7 (smallest |angle|) are R1/R6, the two farthest from it are R3/R4, the remaining two are R2/R5. Also
reported: the pair gaps (R3/R4 should be the narrow end), R8's radius, and the per-cell agreement with the terminal
names. Usage: python faab_pr_ring_names.py <group> [ring_max_um]"""
from faab.config import REPO
import os
import sys

import numpy as np
import pandas as pd
from faab.config import fig

OUT = os.path.join(REPO, "figures")


def main():
    g = int(sys.argv[1])
    ring_max = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5
    df = pd.read_csv(fig(f"pr_bundle_track_group{g}.tsv"), sep="\t")
    df = df[df.complete]
    outer_roots = sorted(df[~df.cell.isin(["R7", "R8"])].root.unique())
    term_name = dict(zip(df.root, df.cell))
    rows = []
    for s, sub in df.groupby("s_box"):
        out = sub[sub.root.isin(outer_roots)]
        if len(out) != len(outer_roots) or out.radius.max() > ring_max or "R7" not in set(sub.cell):
            continue
        a7 = float(sub[sub.cell == "R7"].angle.iloc[0])
        rel = {int(r): ((a - a7 + 180) % 360) - 180 for r, a in zip(out.root, out.angle)}  # -180..180, 0 = at R7
        rows.append(dict(s=s, r8_radius=float(sub[sub.cell == "R8"].radius.iloc[0]) if "R8" in set(sub.cell) else np.nan,
                         r7_radius=float(sub[sub.cell == "R7"].radius.iloc[0]), **{str(r): v for r, v in rel.items()}))
    R = pd.DataFrame(rows)
    if len(R) < 5:
        sys.exit(f"group {g}: only {len(R)} compact complete slices (need >= 5); loosen ring_max?")
    print(f"group {g}: {len(R)} compact complete slices, s = {R.s.min():.0f} to {R.s.max():.0f} um; "
          f"R8 radius median {R.r8_radius.median():.2f} um, R7 radius {R.r7_radius.median():.2f} um")
    # circular median of each axon's angle relative to R7
    med = {}
    for r in outer_roots:
        a = np.radians(R[str(r)].values)
        med[r] = float(np.degrees(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))))
    order = sorted(outer_roots, key=lambda r: abs(med[r]))
    ring = {}
    for r in order[:2]:
        ring[r] = "R1/R6"
    for r in order[-2:]:
        ring[r] = "R3/R4"
    for r in order[2:4]:
        ring[r] = "R2/R5"
    # sanity: the two R1/R6 should flank R7 on opposite sides, as should R3/R4 and R2/R5
    def side_ok(lab):
        a = [med[r] for r in outer_roots if ring[r] == lab]
        return np.sign(a[0]) != np.sign(a[1])
    print("\ncell (root)            angle vs R7   ring name   terminal name   agree")
    agree = 0
    for r in sorted(outer_roots, key=lambda r: med[r]):
        ok = ring[r] == term_name[r]
        agree += ok
        print(f"{r}   {med[r]:+7.1f} deg   {ring[r]:7s}     {term_name[r]:7s}         {'yes' if ok else 'NO'}")
    gaps = {}
    for lab in ("R1/R6", "R2/R5", "R3/R4"):
        a = [med[r] for r in outer_roots if ring[r] == lab]
        d = abs(a[0] - a[1]) % 360
        gaps[lab] = round(min(d, 360 - d))
    print(f"\npair gaps on the ring (deg): {gaps}  (expect R3/R4 narrow, R1/R6 wide with R7 between, R2/R5 ~180)")
    print(f"members on opposite sides of R7: R1/R6 {side_ok('R1/R6')}, R2/R5 {side_ok('R2/R5')}, R3/R4 {side_ok('R3/R4')}")
    print(f"agreement with the terminal-plane names: {agree}/{len(outer_roots)} cells")


if __name__ == "__main__":
    main()
