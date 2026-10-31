"""Bring back to tmp/ the archived tmp files that the repo code (src, scripts, scratch) still reads by name - the tmp
tidy used to ignore scratch/ and archived inputs of kept scratch analyses (2026-10-30: t4t5_seatable_20261026.tsv).
References 'tmp/archive/.../<name>' in the --fix-refs files are rewritten back to 'tmp/<name>'. Dry run unless --apply.
Usage: restore_tmp_inputs.py [--apply] [--fix-refs FILE ...]"""
import argparse
import glob
import os
import re
import shutil

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
TMP = os.path.join(os.path.dirname(REPO), "tmp")
ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true")
ap.add_argument("--fix-refs", nargs="*", default=[])
ap.add_argument("--only", nargs="*", help="restore just these names (the name match also hits docstring mentions)")
a = ap.parse_args()
code = "\n".join(open(p, encoding="utf-8", errors="ignore").read()
                 for d in ("src", "scripts", "scratch") for p in glob.glob(os.path.join(REPO, d, "**", "*.py"), recursive=True))
found = {}
for mf in sorted(glob.glob(os.path.join(TMP, "archive", "*", "MANIFEST.tsv"))):
    for line in open(mf, encoding="utf-8"):
        name, rel = line.rstrip("\n").split("\t")
        src = os.path.join(TMP, rel)
        if os.path.exists(src) and not os.path.exists(os.path.join(TMP, name)) and re.search(rf"(?<![\w.-]){re.escape(name)}(?![\w-])", code):
            found[name] = src  # later manifests win: the newest archived copy
if a.only:
    found = {n: p for n, p in found.items() if n in a.only}
print(f"archived tmp inputs the code reads by name: {len(found)}")
for name, src in sorted(found.items()):
    print(f"  {os.path.relpath(src, TMP)} -> tmp/{name}")
files = []
for f in a.fix_refs:
    files += sorted(glob.glob(os.path.join(f, "*.md"))) if os.path.isdir(f) else [f]
for f in files:
    s = open(f, encoding="utf-8").read()
    t = s
    for name, src in found.items():
        t = t.replace("tmp/" + os.path.relpath(src, TMP), "tmp/" + name)
    if t != s:
        print(f"  refs rewritten in {os.path.basename(f)}" + ("" if a.apply else " (dry run)"))
        if a.apply:
            open(f, "w", encoding="utf-8").write(t)
if a.apply:
    for name, src in found.items():
        shutil.move(src, os.path.join(TMP, name))
    print("restored")
else:
    print("dry run; rerun with --apply")
