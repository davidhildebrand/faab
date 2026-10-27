"""Read-only: for each DOI given, report whether a library digest / extracted text exists (matches the 'doi:' front
matter of */*_digest.md). Usage: lit_have.py DOI [DOI ...]"""
import glob
import re
import sys

want = {d.lower(): None for d in sys.argv[1:]}
for f in glob.glob("/Users/davidh/Sync/Literature/Library/*/*_digest.md"):
    head = open(f, encoding="utf-8", errors="ignore").read(600)
    m = re.search(r"^doi:\s*(\S+)", head, re.M)
    if m and m.group(1).lower() in want:
        want[m.group(1).lower()] = f.split("/Library/")[1]
for d, f in want.items():
    print(f"{d}\t{f or 'NOT IN LIBRARY'}")
