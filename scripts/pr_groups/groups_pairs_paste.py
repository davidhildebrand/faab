"""Paste-ready pairing table: same line structure as the user's group paste (a 'group N' line, a header
line, 8 cell rows, blank line), tab-separated, with the pairing columns appended. R7/R8 rows keep blank
pairing columns. Network-free: joins tmp/pr_groups_pasted.tsv with figures/faab_pr_groups_pairs.tsv.
Output figures/faab_pr_groups_pairs_paste.tsv (also printed)."""
from faab.config import REPO
import os

import pandas as pd

HERE = REPO
OUT = os.path.join(HERE, "figures")
NAMES = {"R2/R5": "lateral", "R1/R6": "near7/8", "R3/R4": "opp7/8"}  # cross-section position of the pair


def conf(r):
    m = float(r.mirror_score) - float(r.mirror_next_best)
    tm = float(r.terminal_score) - float(r.terminal_next)
    if m >= 0.10 and tm >= 0.10:
        return "good"
    if m < 0.05 or tm < 0:
        return "weak"
    return "ok"


def main():
    src = pd.read_csv(os.path.join(os.path.dirname(OUT), "..", "tmp", "pr_groups_pasted.tsv"), sep="\t", dtype=str)
    pairs = pd.read_csv(os.path.join(OUT, "faab_pr_groups_pairs.tsv"), sep="\t", dtype=str)
    by_orig = {r.root_id_orig: r for r in pairs.itertuples()}
    used_to_orig = {r.root_used: r.root_id_orig for r in pairs.itertuples()}
    by_used = {r.root_used: r for r in pairs.itertuples()}
    rank = {"weak": 0, "ok": 1, "good": 2}

    def pair_conf(p):  # one confidence per pair: the weaker of its two members
        q = by_used.get(p.mirror_partner)
        return min([conf(p)] + ([conf(q)] if q is not None else []), key=lambda c: rank[c])
    cols = ["root_id_orig", "class", "root_id_curr", "xyz", "pair_type", "fly_pair", "partner_root_id_orig",
            "partner_root_id_curr", "partner_xyz", "mirror_score", "mirror_margin", "terminal_score", "confidence", "root_note"]
    xyz_of = {r.root_used: r.xyz for r in pairs.itertuples()}
    lines = []
    for g, sub in src.groupby("group", sort=False):
        lines.append(f"group {g}" + "\t" * (len(cols) - 1))
        lines.append("\t".join(cols))
        for r in sub.itertuples():
            p = by_orig.get(r.root_id_orig)
            if p is None:  # R7 / R8
                lines.append("\t".join([r.root_id_orig, r._3, r.root_id_curr, r.xyz] + [""] * 10))
                continue
            note = "" if p.match == "ok" else f"root at xyz differs from pasted {r.root_id_curr}"
            lines.append("\t".join([r.root_id_orig, r._3, p.root_used, r.xyz, NAMES[p.fly_pair], p.fly_pair,
                                    used_to_orig.get(p.mirror_partner, ""), p.mirror_partner, xyz_of.get(p.mirror_partner, ""),
                                    f"{float(p.mirror_score):.2f}",
                                    f"{float(p.mirror_score) - float(p.mirror_next_best):+.2f}", f"{float(p.terminal_score):.2f}",
                                    pair_conf(p), note]))
        lines.append("")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "faab_pr_groups_pairs_paste.tsv"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
