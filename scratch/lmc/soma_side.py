"""Where a cell's soma is (2026-11-01, LMC search): for roots with a side (R/L), the nucleus on the root
(figures/glia/nuclei_roots_now.tsv, nuclei_roots.py) and its distance to the lamina and medulla meshes of that side;
soma_by = LA when nearer the lamina, ME when nearer the medulla, '' without a nucleus (fragment or soma cut off),
'merge?' with more than one. Library: soma_table(roots, sides). As a script it calibrates on the known LMCs
(figures/lmc/lmc_known_roots.tsv). Network-free. Usage: soma_side.py"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from faab import ol_meshes
from faab.config import fig

_trees = {}


def tree(m):
    if m not in _trees:
        _trees[m] = cKDTree(ol_meshes.mesh(m)[0] / 1000.0)
    return _trees[m]


def soma_table(roots, sides):
    N = pd.read_csv(fig("nuclei_roots_now.tsv"), sep="\t", dtype={"root": str})
    by = N.groupby("root")
    rows = []
    for r, sd in zip(roots, sides):
        if r not in by.groups:
            rows.append(dict(root=r, soma_by="", soma_d_LA=np.nan, soma_d_ME=np.nan, n_nuclei=0))
            continue
        g = by.get_group(r)
        p = g[["x", "y", "z"]].to_numpy(float)[0]
        dla, dme = tree(f"LA_{sd}").query(p)[0], tree(f"ME_{sd}").query(p)[0]
        rows.append(dict(root=r, soma_by="merge?" if len(g) > 1 else ("LA" if dla < dme else "ME"),
                         soma_d_LA=round(float(dla), 1), soma_d_ME=round(float(dme), 1), n_nuclei=len(g)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    K = pd.read_csv(fig("lmc_known_roots.tsv"), sep="\t", dtype={"root": str}).fillna("")
    S = soma_table(K.root, K.side).merge(K, on="root")
    S["type"] = S.label.str.extract(r"(L[1-5](?:-L5)?)$", expand=False).fillna("LMC")
    print(pd.crosstab(S.type, S.soma_by.replace("", "no nucleus")).to_string())
    print(S.groupby("soma_by")[["soma_d_LA", "soma_d_ME"]].median().round(1).to_string())
