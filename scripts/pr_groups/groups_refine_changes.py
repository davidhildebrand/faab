"""What the two-criterion refinement changed in groups 9+, row by row, against the tables handed over on 2026-10-01/02
(figures/pr_bundles_pairs_paste.tsv, _extra.tsv) and the split-off groups. Reads figures/pr_groups9plus_refined_paste.tsv.
Writes figures/pr_groups9plus_changes.txt (' | ' separated, no dash rows) and prints counts. Network-free."""
from faab.config import REPO
import os
from collections import Counter
from faab.config import fig

OUT = os.path.join(REPO, "figures")


def read(fn):
    rows, g, header = {}, None, None
    for ln in open(fig(fn)).read().split("\n"):
        c = ln.split("\t")
        if ln.startswith("group "):
            g = int(c[0].split()[1])
            continue
        if c[0] == "root_id_orig":
            header = c
            continue
        if ln.strip() and header and g is not None:
            d = dict(zip(header, c + [""] * (len(header) - len(c))))
            rows[(g, d["root_id_orig"])] = d
    return rows


old = {}
for fn in ("pr_bundles_pairs_paste.tsv", "pr_bundles_pairs_paste_extra.tsv"):
    old.update(read(fn))
new = read("pr_groups9plus_refined_paste.tsv")
cols = ["group", "root_id_curr", "change", "class_old", "class_new", "fly_pair_old", "partner_old", "fly_pair_new", "partner_new",
        "confidence", "refine note"]
out, kinds = [], Counter()
for (g, ro), d in sorted(new.items()):
    o = old.get((g, ro))
    note = d["notes"].split("refine ")[-1] if "refine " in d["notes"] else ""
    ch = []
    if o is None:
        ch.append("split into this new group" if "split from group" in d["notes"] else "ADDED")
    else:
        if o["class"] != d["class"]:
            if o["class"] == "R7/R8" and d["class"] in ("R7", "R8"):
                ch.append("R7/R8 resolved")
            elif o["class"] == "R1-R6" and d["class"] == "R8":
                ch.append("R1-R6 is the central R8")
            elif o["class"] == "R7/R8" and d["class"] == "R1-R6":
                ch.append("R7/R8 is an outer cell")
            else:
                ch.append(f"class {o['class']} -> {d['class']}")
        if (o.get("partner_root_id_curr", "") or "") != (d.get("partner_root_id_curr", "") or "") and d.get("partner_root_id_curr"):
            ch.append("partner" if o.get("partner_root_id_curr") else "partner (new)")
        elif o.get("partner_root_id_curr") and not d.get("partner_root_id_curr"):
            ch.append("partner removed")
        if (o.get("fly_pair", "") or "") != (d.get("fly_pair", "") or "") and d.get("fly_pair"):
            ch.append("fly_pair")
        if "OFF-BUNDLE" in d["notes"]:
            ch.append("off-bundle")
        if "SECOND BUNDLE" in d["notes"]:
            ch.append("moved to new group")
    if not ch:
        continue
    for k in ch:
        kinds[k] += 1
    out.append([str(g), d["root_id_curr"], ", ".join(ch), o["class"] if o else "", d["class"], o.get("fly_pair", "") if o else "",
                o.get("partner_root_id_curr", "") if o else "", d.get("fly_pair", ""), d.get("partner_root_id_curr", ""),
                d.get("confidence", ""), note[:160]])
lines = [" | ".join(cols)] + [" | ".join(r) for r in out]
open(fig("pr_groups9plus_changes.txt"), "w").write("\n".join(lines) + "\n")
print(f"{len(out)} rows changed in {len({r[0] for r in out})} groups; by kind: {dict(kinds)}")
print(f"unchanged rows: {len(new) - len(out)}")
