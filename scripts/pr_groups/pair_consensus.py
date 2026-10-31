"""Consensus pairing and naming of a group's R1-R6 from TWO independent tests (David, 2026-10-14):
 1. terminal test: mirror plane + NBLAST on the terminal zone (faab_pr_pairing_demo.analyse_group) -> mutual pairs,
    names by |d| (lateral = R2/R5) and the R7's t (near = R1/R6); margin = pair score minus the best score to any
    other cell.
 2. ring test: cutout tracking of the compact bundle (figures/pr_bundle_track_group<g>.tsv, faab_pr_bundle_track.py)
    -> each outer axon's circular-mean angle relative to R7; pairs = cells ranked by |angle| taken two at a time
    (1-2 = R1/R6, 3-4 = R2/R5, 5-6 = R3/R4); margin = the smallest |angle| gap between consecutive ranks that
    separates two pairs (ranks 2/3 and 4/5), in degrees, plus the requirement that partners lie on opposite sides.
Verdict per cell: 'confirmed' when both tests give the same partner and name; otherwise the test with the larger
normalized margin wins and the cell is flagged 'conflict' with both answers shown. Cells without a ring track (not in
the compact stretch) keep the terminal answer, flagged 'terminal only'.
Output: figures/pr_pair_consensus_group<g>.tsv and a printed table. Usage: python faab_pr_pair_consensus.py <group> [...]"""
import os
import sys

import numpy as np
import pandas as pd

from faab.pr.pairing_demo import analyse_group, OUT
from faab.config import fig

RING_MAX = 2.5


def ring_test(g):
    path = fig(f"pr_bundle_track_group{g}.tsv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep="\t")
    df = df[df.complete]
    outer = sorted(df[~df.cell.isin(["R7", "R8"])].root.unique())
    rel = {r: [] for r in outer}
    n = 0
    for s, sub in df.groupby("s_box"):
        out = sub[sub.root.isin(outer)]
        if len(out) != len(outer) or out.radius.max() > RING_MAX or "R7" not in set(sub.cell):
            continue
        n += 1
        a7 = float(sub[sub.cell == "R7"].angle.iloc[0])
        for r, a in zip(out.root, out.angle):
            rel[r].append(np.radians(((a - a7 + 180) % 360) - 180))
    if n < 5:
        return None
    med = {r: float(np.degrees(np.arctan2(np.mean(np.sin(v)), np.mean(np.cos(v))))) for r, v in rel.items()}
    order = sorted(outer, key=lambda r: abs(med[r]))
    pairs = [(order[0], order[1], "R1/R6"), (order[2], order[3], "R2/R5"), (order[4], order[5], "R3/R4")]
    ranks = [abs(med[r]) for r in order]
    margin = min(ranks[2] - ranks[1], ranks[4] - ranks[3])
    sides_ok = all(np.sign(med[a]) != np.sign(med[b]) for a, b, _ in pairs)
    return dict(n_slices=n, angle=med, pairs=pairs, margin_deg=margin, sides_ok=sides_ok)


def main():
    pasted = pd.read_csv(os.path.join(os.path.dirname(OUT), "..", "tmp", "pr_groups_pasted.tsv"), sep="\t", dtype=str)
    import json
    now = json.load(open(fig("pr_footprint_roots.json")))["root_used"]
    for g in [int(a) for a in sys.argv[1:]] or [1, 4]:
        R = analyse_group(g, verbose=False)
        old2new = {r.root_id_curr: int(n) for r, n in zip(pasted.itertuples(), now) if int(r.group) == g}
        outer = R["outer"]
        S, names, pairs = R["S"], R["names"], R["pairs"]
        term = {}
        for i, r in enumerate(outer):
            pr = next((p for p in pairs if i in p), None)
            if pr is None:
                term[r] = dict(partner="", name="", score=np.nan, margin=np.nan)
                continue
            j = pr[1] if pr[0] == i else pr[0]
            oth = [S[i, k] for k in range(len(outer)) if k not in (i, j)]
            term[r] = dict(partner=outer[j], name=names.get(i, ""), score=float(S[i, j]), margin=float(S[i, j] - max(oth)))
        ring = ring_test(g)
        rows = []
        for r in outer:
            cur = old2new.get(r, int(r))
            t = term[r]
            row = dict(group=g, root_2026_09_30=r, root_now=cur, terminal_partner=old2new.get(t["partner"], t["partner"]),
                       terminal_name=t["name"], terminal_margin=round(t["margin"], 2) if not np.isnan(t["margin"]) else "")
            if ring and cur in ring["angle"]:
                rp = next((p for p in ring["pairs"] if cur in p[:2]), None)
                partner = rp[0] if rp[1] == cur else rp[1]
                row.update(ring_partner=partner, ring_name=rp[2], ring_angle_vs_R7=round(ring["angle"][cur], 1),
                           ring_margin_deg=round(ring["margin_deg"], 1), ring_sides_ok=ring["sides_ok"])
                same = (partner == row["terminal_partner"]) and (rp[2] == t["name"])
                if same:
                    verdict = "confirmed"
                else:
                    # normalized margins: terminal 0.10 NBLAST ~ 'good'; ring 20 deg ~ clear
                    tn = (t["margin"] / 0.10) if not np.isnan(t["margin"]) else 0
                    rn = (ring["margin_deg"] / 20.0) * (1 if ring["sides_ok"] else 0.5)
                    verdict = f"conflict -> {'ring' if rn > tn else 'terminal'} ({rn:.1f} vs {tn:.1f})"
                row["final_name"] = rp[2] if verdict.endswith("ring") or verdict.startswith("conflict -> ring") else t["name"]
                row["final_partner"] = partner if row["final_name"] == rp[2] and verdict != "confirmed" else row["terminal_partner"]
                if verdict == "confirmed":
                    row["final_name"], row["final_partner"] = t["name"], row["terminal_partner"]
            else:
                verdict = "terminal only"
                row.update(ring_partner="", ring_name="", ring_angle_vs_R7="", ring_margin_deg="", ring_sides_ok="",
                           final_name=t["name"], final_partner=row["terminal_partner"])
            row["verdict"] = verdict
            rows.append(row)
        tab = pd.DataFrame(rows)
        tab.to_csv(fig(f"pr_pair_consensus_group{g}.tsv"), sep="\t", index=False)
        pd.set_option("display.width", 250)
        print(f"\n=== group {g}: terminal plane {R['th']} deg (mirror {R['score']:.2f}); "
              f"ring {ring['n_slices'] if ring else 0} slices, margin {ring['margin_deg']:.0f} deg, sides ok {ring['sides_ok']}" if ring else
              f"\n=== group {g}: terminal plane {R['th']} deg; no ring track")
        print(tab[["root_now", "terminal_name", "terminal_margin", "ring_name", "ring_angle_vs_R7", "final_name", "final_partner", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
