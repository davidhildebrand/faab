"""Counts of SeaTable aedes_main rows by superclass / class (and side) for the classes matching a word, to see how
photoreceptors and lamina cells are labeled (2026-11-01, LMC search). Network-free.
Usage: seatable_class_counts.py SEATABLE.tsv WORD [WORD ...]"""
import sys

import pandas as pd

S = pd.read_csv(sys.argv[1], sep="\t", dtype=str).fillna("")
words = [w.lower() for w in sys.argv[2:]]
lab = S[["superclass", "class", "subclass", "subsubclass", "type"]].agg(" / ".join, axis=1).str.lower()
hit = S[lab.map(lambda t: any(w in t for w in words))]
print(hit.groupby(["superclass", "class", "subclass", "side"]).size().to_string())
