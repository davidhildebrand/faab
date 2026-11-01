"""Synapse profile per typed LMC type (figures/lmc/lmc_features.tsv; 2026-11-01, to design the L5 search): p10/p50/p90 of
inputs and outputs in the lamina and medulla. Network-free. Usage: lmc_type_profiles.py"""
import pandas as pd

from faab.config import fig

F = pd.read_csv(fig("lmc_features.tsv"), sep="\t", dtype={"root": str}).fillna({"label": ""})
F = F[F.label.isin(["L1", "L2", "L3", "L4", "L5"])]
q = lambda s: f"{int(s.quantile(0.1))}/{int(s.median())}/{int(s.quantile(0.9))}"
print(F.groupby("label")[["n_in_LA", "n_out_LA", "n_in_ME", "n_out_ME", "LA_beyond_outer_um"]].agg(q).to_string())
