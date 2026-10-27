"""Read-only: search the literature library digests' front matter (title, topics, organism) for keywords; prints
file, year, title, DOI and the matching topics. Usage: lit_topic_search.py KEYWORD [KEYWORD ...] (case-insensitive, any)"""
import glob
import re
import sys

keys = [k.lower() for k in sys.argv[1:]]
for f in sorted(glob.glob("/Users/davidh/Sync/Literature/Library/*/*_digest.md")):
    head = open(f, encoding="utf-8", errors="ignore").read(4000).split("\n---", 1)[0]
    blob = head.lower()
    hit = [k for k in keys if k in blob]
    if not hit:
        continue
    get = lambda name: (re.search(rf"^{name}:\s*(.*)$", head, re.M) or [None, ""])[1].strip().strip('"')
    print(f"{f.split('/Library/')[1]}\n   {get('title')} | doi {get('doi')} | organism {get('organism')} | hits {hit}")
