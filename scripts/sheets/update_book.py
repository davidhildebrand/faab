"""One update of the tracking workbook from David's latest backup - the steps that run on EVERY update (David,
2026-10-31: row consolidation and reconciliation every time):
  1. seatable_fetch.py           fresh SeaTable snapshot with live roots
  2. seatable_match.py BOOK       every row's live root at its xyz vs SeaTable
  3. cell_consistency.py BOOK     fragment-noted cells sized (L2 chunks) and cells on several rows that disagree
  4. seatable_sync.py BOOK --match <2> --reconcile --sizes <3> [--edits E.json]
                                  SeaTable rules, roots, dates, MISMATCH flags, hand edits, then faab.partners.reconcile
                                  (same-block twins collapsed, self rows removed, one tracking per cell)
Optional first step --add-inputs: scripts/partners/target_inputs_block.py adds the 'more inputs' subtable to
PRg1toT4T5 when the backup lacks it; the later steps then run on its full workbook. Point lookups share the supervoxel
cache (faab.clients.cave.supervoxels_at_xyz), so only the first step reads new points. Each step's output goes to
tmp/<stamp>_update_<step>.log; the summaries and the exports (figures/sheets/<stamp>_FAABopticlobetargets-<tab>.xlsx)
are printed. Usage: update_book.py BACKUP.xlsx [--edits E.json] [--add-inputs]"""
import argparse
import os
import re
import subprocess
import sys

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
    a = ap.parse_args()
    stamp = utc_stamp()
    book = os.path.abspath(a.book)
    if a.add_inputs:
        out = run("inputs", [os.path.join(SCRIPTS, "partners", "target_inputs_block.py"), "--book", book], stamp)
        book = re.search(r"full workbook (\S+\.xlsx)", out).group(1)
    run("seatable_fetch", [os.path.join(HERE, "seatable_fetch.py")], stamp)
    out = run("seatable_match", [os.path.join(HERE, "seatable_match.py"), book], stamp)
    match = re.search(r"written (\S+_seatable_match\.tsv)", out).group(1)
    out = run("cell_consistency", [os.path.join(HERE, "cell_consistency.py"), book], stamp)
    sizes = re.search(r"written (\S+_cell_consistency\.tsv)", out).group(1)
    args = [os.path.join(HERE, "seatable_sync.py"), book, "--match", match, "--reconcile", "--sizes", sizes]
    if a.edits:
        args += ["--edits", a.edits]
    run("seatable_sync", args, stamp)


if __name__ == "__main__":
    main()
