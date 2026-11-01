"""Is a root still current, and if not, what are its latest roots (CAVE chunkedgraph, read-only, one call per root
>= 2 s apart)? Also where the roots appear in our LMC tables (figures/lmc). Usage: tools_root_status.py ROOT [ROOT ...]"""
import sys

import pandas as pd

from faab.clients.cave import client as cave_client, throttle
from faab.config import fig

c, t = cave_client(), 0.0
for r in sys.argv[1:]:
    t = throttle(t)
    latest = bool(c.chunkedgraph.is_latest_roots([int(r)])[0])
    new = [str(x) for x in c.chunkedgraph.get_latest_roots(int(r))] if not latest else [r]
    print(f"{r}: {'current' if latest else 'changed -> ' + ', '.join(new)}")
    for name in ("lmc_cells.tsv", "lmc_typing.tsv", "lmc_candidates.tsv", "lmc_cand_typing.tsv", "lmc_l5_candidates.tsv"):
        try:
            T = pd.read_csv(fig(name), sep="\t", dtype={"root": str})
        except FileNotFoundError:
            continue
        for x in set([r] + new):
            hit = T[T.root == x]
            if len(hit):
                keep = [k for k in ("label", "consensus", "call", "confidence", "n_pr", "notes") if k in hit.columns]
                print(f"  {name}: {x} " + "; ".join(f"{k}={hit.iloc[0][k]}" for k in keep))
