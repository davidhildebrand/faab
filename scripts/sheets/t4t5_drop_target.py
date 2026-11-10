"""Remove a hop-3 target that is not a T4/T5 from the PRg1toT4T5 tab, with the rows only it pulled in (2026-11-09:
Yerbol Kurmangaliyev found 648518347505602348 is a Tm-like cell, not a T4/T5; 99% of its outputs in the medulla, none
in the lobula plate). In the given workbook (David's latest backup), edited in place of a rebuild:
  - the hop-3 row(s) of the target (root_id_curr = ROOT) are removed;
  - 'more inputs' rows (hop 'in') that feed only this target are removed; rows that feed others too lose it from
    T4T5_root_id and its synapse count from nsyn, T4T5_reached drops by one;
  - nsyn_out of every 'in' row loses its synapses onto the target (weaker contacts included; live synapses now,
    matched by root_id_curr: one read-only CAVE query);
  - hop-1 / hop-2 rows on the target's path (its hop1_root_id / hop2_root_id) get T4T5_reached - 1;
  - the block titles' counts ('20 T4/T5 candidates', '140 further cells', 'LOW PRIORITY ... 2 cells') and the note's
    'all 20 targets' are updated;
  - rows are deleted with faab.partners.reconcile.delete_rows ('<tab> rN' references in notes / tracked_in of every
    tab renumbered; references to removed rows dropped from tracked_in).
Writes figures/sheets/<stamp>_FAABopticlobetargets-<tab>.xlsx per changed tab, the removed rows to
figures/sheets/<stamp>_FAABopticlobetargets-PRg1toT4T5_dropped.tsv (status / proofreader / notes kept for the record),
the full workbook to tmp/ and the change log next to the exports.
Usage: t4t5_drop_target.py BOOK.xlsx ROOT --why TEXT [--tab PRg1toT4T5] [--dry-run]"""
import argparse
import datetime
import os
import re
import sys

import openpyxl
import pandas as pd

from faab.clients.cave import client as cave_client, throttle
from faab.config import TMP, fig
from faab.partners.reconcile import delete_rows
from faab.partners.sheet import cell_text as s, scan_book, utc_stamp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seatable_sync import Book  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("book")
    ap.add_argument("root")
    ap.add_argument("--why", required=True)
    ap.add_argument("--tab", default="PRg1toT4T5")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    stamp, root = utc_stamp(), a.root
    wb = openpyxl.load_workbook(a.book)
    ws = wb[a.tab]
    B = Book(wb)
    rows = [x for x in scan_book(wb) if x["tab"] == a.tab]
    v = lambda x, n: s(x["values"].get(n))
    targets = [x for x in rows if v(x, "hop") == "3" and v(x, "root_id_curr") == root]
    if not targets:
        sys.exit(f"{root} is not a hop-3 target (root_id_curr) in {a.tab}")
    path_ids = set()
    for x in targets:
        for n in ("hop1_root_id", "hop2_root_id"):
            path_ids |= {i.strip() for i in v(x, n).split(",") if i.strip()}
    drop, rec_of, record = {a.tab: set()}, {}, []

    def remove(x, why):
        drop[a.tab].add(x["row"])
        rec_of[(a.tab, x["row"])] = None
        B.log.append(dict(tab=a.tab, row=x["row"], column="row", old=v(x, "root_id_orig"), new="", why=why))
        B.touched.add(a.tab)
        record.append({k: v(x, k) for k in ("hop", "T4T5_root_id", "root_id_orig", "root_id_curr", "xyz", "nsyn", "status",
                                            "pr_date", "proofreader", "checker", "notes", "SeaTable")} | {"why": why})

    for x in targets:
        remove(x, f"not a T4/T5: {a.why}")
    # live synapses onto the target, per presynaptic root (read-only, one query)
    c = cave_client()
    throttle(0)
    now = datetime.datetime.now(datetime.timezone.utc)
    syn = c.materialize.live_query("synapses_v2", timestamp=now, filter_in_dict={"post_pt_root_id": [int(root)]})
    onto = syn[syn.pre_pt_root_id != int(root)].pre_pt_root_id.astype(str).value_counts().to_dict()
    blocks_dropped = {}  # title row -> number of rows removed under it

    def out_minus(x, ns, listed=None):
        """nsyn_out minus the cell's synapses onto the target: the count the row lists for it (listed), else the live
        count matched by root_id_curr (weaker contacts); only ever lowered, not below the per-target counts that remain
        (when those fit), and left alone on rows whose point is off the cell (their root_id_curr is not the row's
        cell: 2026-11-09, a typed ID)"""
        if "nsyn_out" not in x["cols"] or not v(x, "nsyn_out").isdigit() or "xyz off the cell" in v(x, "notes"):
            return
        old = int(v(x, "nsyn_out"))
        n_onto = listed if listed is not None else onto.get(v(x, "root_id_curr"), 0)
        if not n_onto:
            return
        new = max(old - n_onto, min(old, sum(int(n) for n in ns if n.isdigit())))
        if new != old:
            B.put(a.tab, x["row"], x["cols"]["nsyn_out"], new, f"minus synapses onto {root}")

    for x in rows:
        if v(x, "hop") != "in":
            continue
        tg = [i.strip() for i in v(x, "T4T5_root_id").split(",") if i.strip()]
        if root not in tg:
            out_minus(x, [i.strip() for i in v(x, "nsyn").split(",")])
            continue
        if tg == [root]:
            remove(x, f"input only to {root} (not a T4/T5)")
            h = B.header_row(a.tab, x["row"]) - 1  # the block title sits above the header
            blocks_dropped[h] = blocks_dropped.get(h, 0) + 1
            continue
        k = tg.index(root)
        ns = [i.strip() for i in v(x, "nsyn").split(",")]
        tg.pop(k)
        listed = int(ns.pop(k)) if len(ns) > k and ns[k].isdigit() else None
        B.put(a.tab, x["row"], x["cols"]["T4T5_root_id"], ", ".join(tg), f"{root} is not a T4/T5")
        B.put(a.tab, x["row"], x["cols"]["nsyn"], ", ".join(ns), f"{root} is not a T4/T5")
        B.put(a.tab, x["row"], x["cols"]["T4T5_reached"], len(tg), f"{root} is not a T4/T5")
        out_minus(x, ns, listed)
    for x in rows:
        if v(x, "hop") in ("1", "2") and (v(x, "root_id_curr") in path_ids or v(x, "root_id_orig") in path_ids):
            n = v(x, "T4T5_reached")
            if n.isdigit() and int(n) > 0:
                B.put(a.tab, x["row"], x["cols"]["T4T5_reached"], int(n) - 1, f"{root} (on its path) is not a T4/T5")
    # titles and the note
    for r in range(1, ws.max_row + 1):
        t = s(ws.cell(r, 1).value)
        new = t
        if t.startswith("hop-3"):
            new = re.sub(r"(\d+) T4/T5 candidates", lambda m: f"{int(m.group(1)) - len(targets)} T4/T5 candidates", t)
        elif r in blocks_dropped:
            new = re.sub(r"(\d+) (further cells|cells)", lambda m: f"{int(m.group(1)) - blocks_dropped[r]} {m.group(2)}", t, 1)
        elif t.startswith("note: 'more inputs' block"):
            new = re.sub(r"all (\d+) targets", lambda m: f"all {int(m.group(1)) - len(targets)} targets", t)
        if new != t:
            B.put(a.tab, r, 1, new, "counts")
    unmatched = [h for h in blocks_dropped if not re.search(r"\d+ (further cells|cells)", s(ws.cell(h, 1).value))]
    print(f"{root}: {len(targets)} target row(s); 'in' rows removed {sum(blocks_dropped.values())} "
          f"(per block title row {blocks_dropped}); shared 'in' rows edited; path rows {sorted(path_ids)}")
    if unmatched:
        print(f"  WARNING: block titles without a count above rows {unmatched}")
    if a.dry_run:
        print(pd.DataFrame(B.log).to_string())
        return
    delete_rows(wb, B, drop, rec_of)
    full = os.path.join(TMP, f"{stamp}_t4t5_drop_full.xlsx")
    wb.save(full)
    pd.DataFrame(record).to_csv(fig(f"{stamp}_FAABopticlobetargets-{a.tab}_dropped.tsv"), sep="\t", index=False)
    pd.DataFrame(B.log).to_csv(fig(f"{stamp}_FAABopticlobetargets-{a.tab}_drop_changes.tsv"), sep="\t", index=False)
    for tab in sorted(B.touched):
        w2 = openpyxl.load_workbook(full)
        for other in list(w2.worksheets):
            if other.title != tab:
                w2.remove(other)
        out = fig(f"{stamp}_FAABopticlobetargets-{tab}.xlsx")
        w2.save(out)
        print(f"wrote {out}")
    print(f"removed rows: {fig(f'{stamp}_FAABopticlobetargets-{a.tab}_dropped.tsv')}; full workbook {full}")


if __name__ == "__main__":
    main()
