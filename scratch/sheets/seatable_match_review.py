"""Network-free review of the newest figures/*_seatable_match.tsv: how sheet values compare with SeaTable values
(status crosstab, proofreader and annotation pairs, SeaTable notes), to decide the sync rules with David."""
import glob
import os

import pandas as pd

path = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "..", "figures", "*_seatable_match.tsv")))[-1]
d = pd.read_csv(path, sep="\t", dtype=str).fillna("")
m = d[d.via != ""]
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 70)
print(os.path.basename(path), len(m), "matched rows")
print("\nsheet status x SeaTable status:")
print(pd.crosstab(m.sheet_status.replace("", "(blank)"), m.st_status.replace("", "(blank)")).to_string())
print("\nproofreader pairs (sheet -> SeaTable), top 40:")
print((m.sheet_proofreader.replace("", "(blank)") + "  ->  " + m.st_proofreader.replace("", "(blank)")).value_counts().head(40).to_string())
print("\nSeaTable annotator values:")
print(m.st_annotator.value_counts().head(20).to_string())
print("\nannotation pairs (sheet -> SeaTable label), top 40:")
print((m.sheet_annotation.replace("", "(blank)") + "  ->  " + m.st_label.replace("", "(blank)")).value_counts().head(40).to_string())
print("\nSeaTable notes (non-empty):")
print(m[m.st_notes != ""].st_notes.value_counts().head(30).to_string())
print("\nSeaTable side:", m.st_side.value_counts().to_dict())
print("\nNEW / NEW-DUP rows in SeaTable:")
print(m[m.sheet_status.str.startswith("NEW")][["tab", "row", "root_at_xyz", "sheet_status", "sheet_notes", "st_status", "st_label", "st_proofreader"]].to_string(index=False))
print("\nrows matched only via root_id_curr:")
print(m[m.via == "root_id_curr"][["tab", "row", "root_id_curr", "root_at_xyz", "sheet_status", "st_status", "st_label"]].to_string(index=False))
print("\nmultiple SeaTable rows on one cell:")
print(m[pd.to_numeric(m.st_n) > 1][["tab", "row", "root_at_xyz", "st_serial", "st_status", "st_label"]].to_string(index=False))
print("\nmarked in the SeaTable column but not in SeaTable:")
print(d[(d.via == "") & (d.sheet_SeaTable != "")][["tab", "row", "root_id_curr", "root_at_xyz", "sheet_status", "sheet_SeaTable"]].to_string(index=False))
