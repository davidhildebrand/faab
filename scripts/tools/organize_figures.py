"""Sort faab/figures into topic subfolders (David, 2026-10-30: 'figures is still too many files').

Every file at the top level of figures/ moves to figures/<topic>/ with the topic from faab.config.fig_topic (the same
rule faab.config.fig uses, so the code keeps finding its files once its paths go through fig(); see
figures_paths_rewrite.py). Folders (archive, lit, ol_meshes, refine and the topic folders) stay. References
'figures/<name>' and 'faab/figures/<name>' in the --fix-refs files (memory notes, TODO.md) are rewritten to the new
path; figures/archive/organize_<date>.tsv records every move. Dry run unless --apply.
Usage: venv/bin/python faab/scripts/tools/organize_figures.py [--apply] [--fix-refs FILE ...]"""
import argparse
import datetime
import glob
import os
import re
import shutil

from faab.config import FIGURES as FIG, fig_topic


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--fix-refs", nargs="*", default=[])
    a = ap.parse_args()
    names = sorted(n for n in os.listdir(FIG) if not n.startswith(".") and os.path.isfile(os.path.join(FIG, n)))
    moves = [(n, fig_topic(n)) for n in names]
    by = {}
    for n, t in moves:
        by.setdefault(t, []).append(n)
    print(f"figures: {len(names)} files at the top level -> " + ", ".join(f"{t}/ {len(ns)}" for t, ns in sorted(by.items())))
    files = []
    for f in a.fix_refs:
        files += sorted(glob.glob(os.path.join(f, "*.md"))) if os.path.isdir(f) else [f]
    for f in files:
        s = open(f, encoding="utf-8").read()
        t = s
        for n, d in moves:
            t = re.sub(r"(?<![\w/])((?:faab/)?figures/)" + re.escape(n) + r"(?![\w]|\.\w)",
                       lambda m, n=n, d=d: m.group(1) + f"{d}/{n}", t)
        if t != s:
            print(f"  refs rewritten in {os.path.basename(f)}" + ("" if a.apply else " (dry run)"))
            if a.apply:
                open(f, "w", encoding="utf-8").write(t)
    if not a.apply:
        print("dry run; rerun with --apply")
        return
    os.makedirs(os.path.join(FIG, "archive"), exist_ok=True)
    with open(os.path.join(FIG, "archive", f"organize_{datetime.date.today():%Y%m%d}.tsv"), "a", encoding="utf-8") as fh:
        for n, d in moves:
            os.makedirs(os.path.join(FIG, d), exist_ok=True)
            shutil.move(os.path.join(FIG, n), os.path.join(FIG, d, n))
            fh.write(f"{n}\t{d}/{n}\n")
    left = [n for n in os.listdir(FIG) if os.path.isfile(os.path.join(FIG, n))]
    print(f"done; figures top level now holds {len(left)} files and folders {sorted(d for d in os.listdir(FIG) if os.path.isdir(os.path.join(FIG, d)))}")


if __name__ == "__main__":
    main()
