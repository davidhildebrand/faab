"""Print the member rows of some groups of a PRs tab (row, class, root_id_orig, root_id_curr, xyz, status, notes).
Network-free. Usage: prs_tab_rows.py BOOK G [G ...]"""
import sys

from faab.pr import prs_tab as T

ws = T.load(sys.argv[1])[1]
want = {int(g) for g in sys.argv[2:]}
for b in T.blocks(ws):
    if b.group in want:
        print(f"group {b.group} ({b.section}), title row {b.title_row}; columns {b.cols}")
        for r in b.rows:
            get = lambda k: T.get(ws, b, r, k)
            print(f"  row {r}: {get('class'):9s} {get('root_id_orig')} curr {get('root_id_curr')} xyz {get('xyz')} "
                  f"{get('status')}; {get('notes')[:110]}")
