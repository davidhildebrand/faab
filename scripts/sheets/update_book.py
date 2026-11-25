"""One update of the tracking workbook from David's latest backup - the steps that run on EVERY update (David,
2026-10-31: row consolidation and reconciliation every time):
  0. faab.synsnap.refresh()       new synapse checkpoints (David 2026-11-10: always before querying / updating sheets)
  1. seatable_fetch.py          fresh SeaTable snapshot with live roots
  2. seatable_match.py BOOK       every row's live root at its xyz vs SeaTable
  3. cell_consistency.py BOOK     fragment-noted cells sized (L2 chunks) and cells on several rows that disagree
  4. seatable_sync.py BOOK --match <2> --reconcile --sizes <3> [--edits E.json]
                                  SeaTable rules, roots, dates, MISMATCH flags, hand edits, then faab.partners.reconcile
                                  (same-block twins collapsed, self rows removed, one tracking per cell)
Optional step after the fetch: --add-inputs (scripts/partners/target_inputs_block.py adds the 'more inputs' subtable
to PRg1toT4T5 when the backup lacks it) or --add-in2 (scripts/partners/hop_in2_block.py: hop-in-1 / hop-in-2 blocks
first, 2026-11-10); the later steps then run on its full workbook and the sync exports PRg1toT4T5. Then, on every
update unless --no-t4t5-sort, scripts/partners/t4t5_sort.py (2026-11-12): T4T5-1 / T4T5-2 rows ordered by distance from
the PR group 1 column, putative-LMC notes on T4T5-2 rows, the distance column far right. Point lookups share
the supervoxel cache (faab.clients.cave.supervoxels_at_xyz), so only the first step reads new points. After the
T4/T5 sort, --home-sort TAB ... (e.g. T4T5s) and --renumber-prs run scripts/sheets/home_column_sort.py (2026-11-25) with
the lines the sort just saved: those tabs ordered by distance from the PR group 1 column, and the right-side PR groups
nobody has worked on renumbered by their distance from PR group 1 (a one-off, David 2026-11-25). Before the
T4/T5 sort, unless --no-membership, scripts/sheets/membership_check.py --apply (2026-11-21): every block's rule is
applied again to today's connectivity (rows recounted; rows below the criteria moved down and noted, or removed when
nobody worked on them; newly qualifying cells added); the tabs it changes are exported. Each step's output
goes to tmp/<stamp>_update_<step>.log; the summaries and the exports
(figures/sheets/<stamp>_FAABopticlobetargets-<tab>.xlsx) are printed.
Usage: update_book.py BACKUP.xlsx [--edits E.json] [--add-inputs | --add-in2] [--no-t4t5-sort]
       [--home-sort T4T5s ...] [--renumber-prs]"""
import argparse
import os
import re
import subprocess
import sys

from faab import synsnap
from faab.config import TMP
from faab.partners.sheet import utc_stamp

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
NOISE = re.compile(r"Segmentation IDs|it/s\]|s/it\]|^\s*$")


def run(name, args, stamp):
    log = os.path.join(TMP, f"{stamp}_update_{name}.log")
    p = subprocess.run([sys.executable, *args], capture_output=True, text=True)
    out = p.stdout + p.stderr
    open(log, "w", encoding="utf-8").write(out)
    lines = [ln for ln in re.split(r"[\r\n]", out) if not NOISE.search(ln)]
    print(f"\n== {name} (log {log})")
    print("\n".join(lines[-40:]))
    if p.returncode:
        sys.exit(f"{name} failed (exit {p.returncode}); see {log}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("book")
    ap.add_argument("--edits")
    ap.add_argument("--add-inputs", action="store_true")
    ap.add_argument("--add-in2", action="store_true")
    ap.add_argument("--no-t4t5-sort", action="store_true", help="skip scripts/partners/t4t5_sort.py")
    ap.add_argument("--no-membership", action="store_true", help="skip scripts/sheets/membership_check.py --apply")
    ap.add_argument("--export", nargs="+", default=[], help="tabs the sync exports even when it changes nothing there")
    ap.add_argument("--home-sort", nargs="+", default=[], metavar="TAB",
                    help="tabs scripts/sheets/home_column_sort.py orders by distance from the PR group 1 column")
    ap.add_argument("--renumber-prs", action="store_true",
                    help="renumber the right-side PR groups nobody has worked on by distance from PR group 1 (one-off)")
    a = ap.parse_args()
    stamp = utc_stamp()
    book = os.path.abspath(a.book)
    export = ["--export", *a.export] if a.export else []
    # new synapse checkpoints first (David 2026-11-10); the steps' own queries then find them current
    print(f"== synapse snapshot: newest checkpoint {synsnap.refresh(max_age_h=synsnap.AUTO_REFRESH_H)}", flush=True)
    run("seatable_fetch", [os.path.join(HERE, "seatable_fetch.py")], stamp)  # first: the block steps read it too
    for flag, script in ((a.add_inputs, "target_inputs_block.py"), (a.add_in2, "hop_in2_block.py")):
        if flag:
            out = run(script.split("_block")[0], [os.path.join(SCRIPTS, "partners", script), "--book", book], stamp)
            book = re.search(r"full workbook (\S+\.xlsx)", out).group(1)
            export = ["--export", *dict.fromkeys(a.export + ["PRg1toT4T5"])]
    if not a.no_membership:  # every update (David 2026-11-21): block membership follows connectivity
        out = run("membership", [os.path.join(HERE, "membership_check.py"), book, "--apply"], stamp)
        book = re.search(r"full workbook (\S+\.xlsx)", out).group(1)
        tabs_m = re.search(r"changed tabs: ?(.*)", out).group(1).split()
        export = ["--export", *dict.fromkeys(a.export + (export[1:] if export else []) + tabs_m)] if tabs_m or export else export
    if not a.no_t4t5_sort:  # every update (David 2026-11-12): spatial order + putative-LMC notes of the T4T5 blocks
        out = run("t4t5_sort", [os.path.join(SCRIPTS, "partners", "t4t5_sort.py"), "--book", book], stamp)
        book = re.search(r"full workbook (\S+\.xlsx)", out).group(1)
        # the sort marks PR groups too; keep the tabs earlier steps (membership) changed
        export = ["--export", *dict.fromkeys((export[1:] if export else a.export) + ["PRg1toT4T5", "PRs"])]
        lines = re.search(r"home-column lines saved: (\S+\.json)", out)
    if a.home_sort or a.renumber_prs:  # 2026-11-25: tabs by distance from the PR group 1 column; PR group renumbering
        args = [os.path.join(HERE, "home_column_sort.py"), book]
        if a.home_sort:
            args += ["--tabs", *a.home_sort]
        if a.renumber_prs:
            args.append("--renumber-prs")
        if not a.no_t4t5_sort and lines:
            args += ["--lines", lines.group(1)]
        out = run("home_column_sort", args, stamp)
        book = re.search(r"full workbook (\S+\.xlsx)", out).group(1)
        tabs_h = re.search(r"changed tabs: ?(.*)", out).group(1).split()
        export = ["--export", *dict.fromkeys((export[1:] if export else a.export) + tabs_h)] if tabs_h or export else export
    out = run("seatable_match", [os.path.join(HERE, "seatable_match.py"), book], stamp)
    match = re.search(r"written (\S+_seatable_match\.tsv)", out).group(1)
    out = run("cell_consistency", [os.path.join(HERE, "cell_consistency.py"), book], stamp)
    sizes = re.search(r"written (\S+_cell_consistency\.tsv)", out).group(1)
    args = [os.path.join(HERE, "seatable_sync.py"), book, "--match", match, "--reconcile", "--sizes", sizes, *export]
    if a.edits:
        args += ["--edits", a.edits]
    run("seatable_sync", args, stamp)


if __name__ == "__main__":
    main()
