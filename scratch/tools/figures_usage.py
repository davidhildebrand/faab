"""Network-free: which code files name which figures/ files (literals and f-string patterns, as tidy_figures does), and
the topic folders (organize_figures.TOPICS) those files would go to - to plan moving code-read files into subfolders.
Prints per code file: topics touched and counts; and per topic the code files that touch it.
Usage: run with faab/scripts/tools on the path (it imports tidy_figures and organize_figures)."""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "tools"))
from organize_figures import topic  # noqa: E402
from tidy_figures import EXT, FIG, REPO  # noqa: E402

names = sorted(n for n in os.listdir(FIG) if os.path.isfile(os.path.join(FIG, n)))
per_file = {}
for p in glob.glob(os.path.join(REPO, "*", "**", "*.py"), recursive=True):
    if "/archive/" in p:
        continue
    text = open(p, encoding="utf-8", errors="ignore").read()
    hit = set()
    for lit in set(re.findall(r"[\w\-{}.*\[\]:%]+\." + EXT + r"\b", text)):
        lit = lit.replace("{{", "{").replace("}}", "}")
        if len(re.sub(r"\{[^}]*\}|\*|\." + EXT + "$", "", lit)) < 5:
            continue
        rx = re.compile(re.sub(r"\\\{[^}]*\\\}|\\\*", ".*", re.escape(lit)))
        hit |= {n for n in names if rx.fullmatch(n)}
    if hit:
        per_file[os.path.relpath(p, REPO)] = hit
topics = {}
for f, hit in sorted(per_file.items()):
    ts = sorted({topic(n) for n in hit})
    for t in ts:
        topics.setdefault(t, []).append(f)
    flag = "  MULTI" if len(ts) > 1 else ""
    print(f"{f}: {len(hit)} files, topics {ts}{flag}")
print()
for t, fs in sorted(topics.items()):
    print(f"{t}: {len(fs)} code files")
