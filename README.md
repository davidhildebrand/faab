# faab

Read-only analysis tools for the FAAB connectome (female adult *Aedes aegypti* brain, CAVE datastack
`wclee_aedes_brain`): photoreceptor (PR) groups and bundles, downstream-partner tables, and plots.

## Layout

| path | contents |
|---|---|
| `src/faab/` | importable library: `config` (paths, datastack), `sources/` (read-only clients for external data: `cave` throttled CAVE helpers, `neuprint`), `morph` (dotprops for NBLAST), `nglink` (Spelunker links), `pr/` (PR group frames, bundles, pairing), `partners/` (partner xyz points, soma checks), `glia/` (shape metrics) |
| `scripts/partners/` | the PR-downstream sheet pipeline (top-20 update, xyz anchor check, glia flags, p_glia, soma column, render) |
| `scripts/pr_groups/` | PR group finding, bundle tracking, two-criterion pairing, ring plots, schematics |
| `scripts/cutouts/`, `scripts/metu/`, `scripts/admin/` | synapse cutouts, PR-to-MeTu bridge analyses, CAVE activity report |
| `scratch/<topic>/` | earlier explorations and superseded versions (Dm2, CT1, MeTu/LC, Mi1, fly references, top-10 era) |
| `data/` | small versioned inputs: `pr_groups.tsv` (canonical PR groups), Drosophila reference CSVs |
| `figures/` | outputs and caches, not versioned except `soma_overrides.tsv` and `glia_logistic_model.json` |

`MOVES.md` maps every file of the old flat layout to its new place.

## Setup

```
python -m venv .venv
.venv/bin/pip install -e .
```

Tokens stay outside the repo: the CAVE token via `caveclient` (`~/.cloudvolume/secrets/`), the neuPrint token in
`~/.neuprint/token.json`. Scripts write to `figures/` and read pasted sheets from `../tmp/` (the project scratch folder
next to the repo).

## Conventions

- CAVE access is read-only and throttled to at most one call every 5 s; no parallel requests.
- A sheet row's xyz point (16 x 16 x 45 nm voxels) is its anchor; root ids are looked up at the point because they change
  with every proofreading edit.
- Live synapse queries (`live_query`, timestamp now) are used for connectivity; the query time is recorded with results.

## Examples

Aligned bundle cross-section for one bundle, from root ids or from the groups file:

```
.venv/bin/python scripts/pr_groups/pr_ring_plot.py --roots <root ids> --r7 <R7 root id> --tag mybundle
.venv/bin/python scripts/pr_groups/pr_ring_plot.py --group 4
```

`pr_ring_plot.py` imports nothing from the package and needs no other file, so it can be copied and run elsewhere.
