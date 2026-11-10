"""Clear figures/sheets down to what David uploads (David 2026-11-10: "Any sheets before today can be removed. Also clean
up any of today's that should not be uploaded. It is too confusing to have the accidents with the different time
stamps."). KEEP: the files named with --keep (the tab exports to import, the newest backup) and every UNSTAMPED file
(inputs the code reads: faab_pr_top20_*, soma_overrides.tsv, partner_glia_flags.tsv, ...). Everything else - every
stamped file (old exports, superseded / test exports, backups, SeaTable fetches, match / consistency / change logs,
dated pr_sheet_check outputs) - moves to figures/archive/<YYYYMMDD>_sheets/ with a MANIFEST.tsv (never deleted).
--fix-refs rewrites 'figures/sheets/<name>' (and 'faab/figures/sheets/<name>') in the given files (a folder = its .md
files) to the archived path. Dry run unless --apply.
Usage: archive_sheets.py --keep NAME [NAME ...] [--fix-refs FILE ...] [--apply]"""
import argparse
import datetime
import glob
import os
import re
import shutil

from faab.config import FIGURES

SHEETS = os.path.join(FIGURES, "sheets")
STAMP = re.compile(r"\d{8}(d\d{6}tUTC|d|_\d{4})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", nargs="+", required=True)
    ap.add_argument("--fix-refs", nargs="*", default=[])
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    names = sorted(n for n in os.listdir(SHEETS) if not n.startswith("."))
    missing = [k for k in a.keep if k not in names]
    if missing:
        raise SystemExit(f"--keep names not in figures/sheets: {missing}")
    moves = [n for n in names if n not in a.keep and STAMP.search(n)]
    kept = [n for n in names if n not in moves]
    dest = os.path.join(FIGURES, "archive", datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d") + "_sheets")
    print(f"figures/sheets: {len(names)} files; keep {len(kept)}; archive {len(moves)} -> {os.path.relpath(dest, FIGURES)}")
    print("keep:\n  " + "\n  ".join(kept))
    files = []
    for f in a.fix_refs:
        files += sorted(glob.glob(os.path.join(f, "*.md"))) if os.path.isdir(f) else [f]
    for f in files:
        s = open(f, encoding="utf-8").read()
        t = s
        for n in moves:
            t = re.sub(r"(?<![\w/])((?:faab/)?figures/)sheets/" + re.escape(n) + r"(?![\w]|\.\w)",
                       lambda m, n=n: m.group(1) + f"archive/{os.path.basename(dest)}/{n}", t)
        if t != s:
            print(f"refs rewritten in {os.path.basename(f)}" + ("" if a.apply else " (dry run)"))
            if a.apply:
                open(f, "w", encoding="utf-8").write(t)
    if not a.apply:
        print("dry run; rerun with --apply")
        return
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "MANIFEST.tsv"), "a", encoding="utf-8") as fh:
        for n in moves:
            shutil.move(os.path.join(SHEETS, n), os.path.join(dest, n))
            fh.write(f"sheets/{n}\t{os.path.relpath(os.path.join(dest, n), FIGURES)}\n")
    print(f"done; figures/sheets now holds {len(kept)} files")


if __name__ == "__main__":
    main()
