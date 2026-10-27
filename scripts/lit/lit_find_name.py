"""Read-only: list library digests whose file name starts with any given prefix, with their DOI.
Usage: lit_find_name.py PREFIX [PREFIX ...]"""
import glob
import os
import re
import sys

for f in sorted(glob.glob("/Users/davidh/Sync/Literature/Library/*/*_digest.md")):
    if any(os.path.basename(f).startswith(p) for p in sys.argv[1:]):
        m = re.search(r"^doi:\s*(\S+)", open(f, encoding="utf-8", errors="ignore").read(600), re.M)
        print(f"{f.split('/Library/')[1]}\t{m.group(1) if m else '?'}")
