"""Tidy faab/figures: keep what the code reads and the newest of each stamped series, archive the rest (never delete).

KEEP at the top level of figures/:
  - every entry whose name matches a file name in the repo code (src, scripts, scratch), f-string placeholders and
    globs included ('{pfx}bundles_roots.tsv' keeps pr_bundles_roots.tsv and pr_left_bundles_roots.tsv; names with fewer
    than 5 literal characters besides the extension are ignored), and directories named in the code;
  - unstamped files named 'figures/<name>' in the --fix-refs files (deliverables the notes point to);
  - for stamped outputs (a YYYYMMDDdHHMMSStUTC or YYYYMMDDd stamp anywhere in the name): only the newest of each series
    (same name without the stamp), whether or not the code names the series;
  - archive/, dotfiles, --keep NAME.
Everything else moves to figures/archive/<YYYYMMDD>[<suffix>]/{superseded,outputs}/ (superseded = older stamped
versions, outputs = unstamped files no code reads) with a MANIFEST.tsv. --fix-refs rewrites 'figures/<name>' and
'faab/figures/<name>' in the given files (a folder = its .md files) to the archived path. Dry run unless --apply.
Usage: venv/bin/python faab/scripts/tools/tidy_figures.py [--apply] [--keep NAME ...] [--fix-refs FILE ...]"""
import argparse
import datetime
import glob
import os
import re
import shutil

from faab.config import FIG_TOPICS

REPO =os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FAAB = os.path.dirname(REPO)
FIG = os.path.join(REPO, "figures")
EXT = r"(?:tsv|pkl|json|txt|png|html|md|csv|xlsx|npz|svg|pdf|log)"
STAMP = re.compile(r"\d{8}d(?:\d{6}tUTC)?")


def code_patterns():
    text = "\n".join(open(p, encoding="utf-8", errors="ignore").read()
                     for d in ("src", "scripts", "scratch") for p in glob.glob(os.path.join(REPO, d, "**", "*.py"), recursive=True))
    pats, words = set(), set(re.findall(r"[\w\-]+", text))
    for lit in set(re.findall(r"[\w\-{}.*\[\]:%]+\." + EXT + r"\b", text)):
        lit = lit.replace("{{", "{").replace("}}", "}")  # f-string escapes: '{{}}' is a later .format() slot
        if len(re.sub(r"\{[^}]*\}|\*|\." + EXT + "$", "", lit)) < 5:
            continue
        rx = re.escape(lit)
        rx = re.sub(r"\\\{[^}]*\\\}|\\\*", ".*", rx)
        pats.add(re.compile(rx))
    return pats, words


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--keep", nargs="*", default=[])
    ap.add_argument("--fix-refs", nargs="*", default=[])
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    pats, words = code_patterns()
    files = []
    for f in a.fix_refs:
        files += sorted(glob.glob(os.path.join(f, "*.md"))) if os.path.isdir(f) else [f]
    notes = "\n".join(open(f, encoding="utf-8").read() for f in files)
    topics = {t for t, _ in FIG_TOPICS} | {"misc"}
    names = []  # paths relative to figures/: top-level entries, and the files inside the topic folders
    for n in sorted(os.listdir(FIG)):
        if n.startswith(".") or n == "archive":
            continue
        if n in topics and os.path.isdir(os.path.join(FIG, n)):
            names += [f"{n}/{m}" for m in sorted(os.listdir(os.path.join(FIG, n))) if not m.startswith(".")]
        else:
            names.append(n)
    base = lambda rel: rel.split("/")[-1]
    stamped = {n: STAMP.sub("", base(n)) for n in names if STAMP.search(base(n))}
    newest = {}
    for n, series in stamped.items():
        if series not in newest or max(STAMP.findall(base(n))) > max(STAMP.findall(base(newest[series]))):
            newest[series] = n
    keep, moves = [], []
    for n in names:
        path = os.path.join(FIG, n)
        if base(n) in a.keep:
            keep.append(n)
        elif n in stamped:
            if newest[stamped[n]] == n:
                keep.append(n)
            else:
                moves.append((n, "superseded"))
        elif os.path.isdir(path):
            if base(n) in words:
                keep.append(n)
            else:
                moves.append((n, "outputs"))
        elif any(p.fullmatch(base(n)) for p in pats) or re.search(r"figures/" + re.escape(n) + r"(?![\w]|\.\w)", notes):
            keep.append(n)  # read by the code, or named in the notes (deliverables such as html pages)
        else:
            moves.append((n, "outputs"))
    dest = os.path.join(FIG, "archive", datetime.date.today().strftime("%Y%m%d") + a.suffix)
    print(f"figures: {len(names)} entries; keep {len(keep)}; archive {len(moves)} -> {os.path.relpath(dest, REPO)} "
          f"(superseded {sum(k == 'superseded' for _, k in moves)}, outputs {sum(k == 'outputs' for _, k in moves)})")
    for n, k in moves:
        print(f"  {k:10s} {n}")
    for f in files:
        s = open(f, encoding="utf-8").read()
        t = s
        for n, k in moves:
            t = re.sub(r"(?<![\w/])((?:faab/)?figures/)" + re.escape(n) + r"(?![\w]|\.\w)",
                       lambda m, n=n, k=k: m.group(1) + f"archive/{os.path.basename(dest)}/{k}/{base(n)}", t)
        if t != s:
            print(f"  refs rewritten in {os.path.basename(f)}" + ("" if a.apply else " (dry run)"))
            if a.apply:
                open(f, "w", encoding="utf-8").write(t)
    if not a.apply:
        print("dry run; rerun with --apply")
        return
    if moves:
        os.makedirs(dest, exist_ok=True)
        with open(os.path.join(dest, "MANIFEST.tsv"), "a", encoding="utf-8") as fh:
            for n, k in moves:
                os.makedirs(os.path.join(dest, k), exist_ok=True)
                shutil.move(os.path.join(FIG, n), os.path.join(dest, k, base(n)))
                fh.write(f"{n}\t{os.path.relpath(os.path.join(dest, k, base(n)), FIG)}\n")
    print(f"done; figures now holds {len(names) - len(moves)} kept entries")


if __name__ == "__main__":
    main()
