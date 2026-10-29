"""Compare bundles_verify.py verdicts before and after a rule change (network-free).

    python verify_diff.py --save [--side left]   copy figures/<pfx>bundles_roots_verified.tsv to a stamped file
    python verify_diff.py [--side left]          diff the current table against the newest stamped copy"""
import glob
import os
import shutil

import pandas as pd

from faab.config import FIGURES
from faab.partners.sheet import utc_stamp
from faab.pr.side import PREFIX, parse as parse_side

side, args = parse_side()
cur = os.path.join(FIGURES, f"{PREFIX[side]}bundles_roots_verified.tsv")
if "--save" in args:
    dst = os.path.join(FIGURES, f"{utc_stamp()}_{PREFIX[side]}bundles_roots_verified_before.tsv")
    shutil.copy(cur, dst)
    print(f"saved {dst}")
    raise SystemExit
old_path = sorted(glob.glob(os.path.join(FIGURES, f"*_{PREFIX[side]}bundles_roots_verified_before.tsv")))[-1]
old = pd.read_csv(old_path, sep="\t", dtype={"root": str})
new = pd.read_csv(cur, sep="\t", dtype={"root": str})
m = old[["root", "cluster", "verdict", "me_frac"]].merge(new[["root", "verdict", "la_frac", "me_frac", "n_pre", "n_post"]],
                                                        on="root", suffixes=("_old", ""))
print(f"{os.path.basename(old_path)} -> {os.path.basename(cur)}: {len(m)} roots")
print(pd.crosstab(m.verdict_old, m.verdict).to_string())
print("\nmedulla share p10/50/90 per new verdict, old -> new:")
for v, s in m[m.n_pre >= 10].groupby("verdict"):
    q = lambda c: s[c].quantile([0.1, 0.5, 0.9]).round(2).tolist()
    print(f"  {v}: {q('me_frac_old')} -> {q('me_frac')}; cells with none: {int((s.me_frac_old == 0).sum())} -> {int((s.me_frac == 0).sum())}")
ch = m[m.verdict_old != m.verdict].sort_values(["verdict_old", "verdict", "cluster"])
pd.set_option("display.width", 250)
print(f"\nchanged ({len(ch)}):")
print(ch.to_string(index=False))
