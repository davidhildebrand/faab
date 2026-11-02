"""Are the existing cutout boxes aligned to a chunk grid? (David 2026-11-01: 'check the existing boxes'.) From
figures/cutouts/cutout_inventory.tsv: each box's lower corner (voxels) modulo common chunk sizes, and the spread of the
remainders (all remainders equal = aligned to that grid). Network-free. Usage: cutouts_alignment.py"""
import json

import numpy as np
import pandas as pd

from faab.config import fig

I = pd.read_csv(fig("cutout_inventory.tsv"), sep="\t")
lo = np.array([json.loads(s) for s in I.lo])
hi = np.array([json.loads(s) for s in I.hi])
print(f"{len(I)} boxes; size (voxels) {np.unique(np.round(hi - lo), axis=0).tolist()}")
print("lower corners, first 8:", np.round(lo[:8], 1).tolist())
frac = lo - np.floor(lo)
print(f"fractional part of the corners: {np.unique(np.round(frac, 1), axis=0).tolist()[:6]}")
for g in (25, 50, 64, 100, 125, 128, 256, 512):
    rem = np.mod(np.round(lo), g)
    print(f"grid {g:4d}: distinct remainders x {len(np.unique(rem[:, 0]))}, y {len(np.unique(rem[:, 1]))}, "
          f"z {len(np.unique(rem[:, 2]))}; e.g. {rem[:5].astype(int).tolist()}")
