"""(male-CNS) Is Dm2 input MeTu1-specific? And what else inputs MeTu1 with a
Dm2-like 'many-weak' profile?"""
import pandas as pd
from neuprint import NeuronCriteria as NC, fetch_simple_connections
from faab.clients.neuprint import client, MALE_CNS

pd.set_option("display.width", 160)


def main():
    c = client(MALE_CNS)

    # 1) Dm2 -> which MeTu subtypes?
    d = fetch_simple_connections(NC(type="Dm2"), NC(type="MeTu.*", regex=True), client=c)
    g = d.groupby("type_post").agg(syn=("weight", "sum"),
                                   n_conn=("weight", "size"),
                                   n_MeTu=("bodyId_post", "nunique"),
                                   n_Dm2=("bodyId_pre", "nunique")).sort_values("syn", ascending=False)
    print("=== (male-CNS) Dm2 -> MeTu subtypes ===")
    print(g.to_string())
    print(f"  Dm2->MeTu1 share of all Dm2->MeTu synapses: {g.loc['MeTu1','syn']/g['syn'].sum()*100:.0f}%")

    # 2) what inputs MeTu1? (top presynaptic types + their profile)
    u = fetch_simple_connections(None, NC(type="MeTu1"), min_weight=2, client=c)
    prof = u.groupby("type_pre").agg(
        syn=("weight", "sum"),
        n_pre=("bodyId_pre", "nunique"),          # how many cells of this type
        n_MeTu1=("bodyId_post", "nunique"),        # how many MeTu1 they reach
        med_w=("weight", "median")).reset_index()
    prof["per_MeTu1_cells"] = (prof["n_pre"] /  # rough convergence proxy
                               prof["n_MeTu1"].clip(lower=1)).round(1)
    prof = prof.sort_values("syn", ascending=False).head(15)
    prof["type_pre"] = prof["type_pre"].fillna("(untyped)")
    print("\n=== (male-CNS) top inputs to MeTu1 (Dm2-like = many cells, low med_w, broad reach) ===")
    print(prof.to_string(index=False))


if __name__ == "__main__":
    main()
