"""Rows of the tracking tabs that sit on the same live cell but disagree (status, names, dates, notes, SeaTable mark),
and rows whose notes call the cell a fragment / chunk / short portion, checked against the cell's current size
(David, 2026-10-31: 648518347498326095 is noted 'small chunk of a big neuron; in between EM slices are broken' but is
one large continuous neuron now). Read-only. A row's cell = its root_id_curr while that is current, else the root at
its xyz (one point lookup for every row, one is_latest_roots call); backup tabs '*_bYYMMDDd' are skipped. Size =
flywire.get_l2_info (L2 cable length) of every fragment-noted cell, >= 5 s apart. Writes
figures/sheets/<stamp>_cell_consistency.tsv (one line per row of every multi-row or fragment-noted cell) and prints a
summary. Usage: cell_consistency.py BOOK.xlsx"""
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import openpyxl
import pandas as pd

from faab.clients.cave import DATASTACK, client as cave_client, roots_at_xyz, throttle
from faab.config import fig
from faab.partners.sheet import cell_text as s, scan_book, utc_stamp

FRAG = re.compile(r"fragment|chunk|short portion|small portion|small piece|piece of|portion of|part of another|"
                  r"belonging to|broken|orphan|split.?off", re.I)
FIELDS = ("status", "pr_date", "proofreader", "chk_date", "checker", "notes", "SeaTable")
_t = [0.0]


def wait():
    _t[0] = throttle(_t[0])


def initials(name):
    w = re.findall(r"[A-Za-z]+", name)
    return (w[0][0] + w[-1][0]).upper() if len(w) >= 2 else name.strip().upper()


def norm(field, v):
    v = s(v)
    if field == "status":
        return v.lower()
    if field in ("proofreader", "checker"):
        return "; ".join(sorted({initials(x) for x in re.split(r"[;,]", v) if x.strip()}))
    if field == "SeaTable":
        return v.split(":")[0].strip().lower()
    return v


def main():
    book = sys.argv[1]
    stamp, t0 = utc_stamp(), time.time()
    wb = openpyxl.load_workbook(book)
    rows = [x for x in scan_book(wb) if not re.search(r"_b\d{6}d$", x["tab"])]
    for x, r in zip(rows, roots_at_xyz([s(x["values"]["xyz"]) for x in rows])):
        x["at"] = str(r)
    ids = sorted({s(x["values"].get("root_id_curr")) for x in rows if re.fullmatch(r"\d{15,}", s(x["values"].get("root_id_curr")))})
    c = cave_client()
    wait()
    latest = dict(zip(ids, np.asarray(c.chunkedgraph.is_latest_roots([int(i) for i in ids])).astype(bool)))
    for x in rows:
        cu = s(x["values"].get("root_id_curr"))
        x["cell"] = cu if latest.get(cu) else x["at"]
    by = {}
    for x in rows:
        if x["cell"] != "0":
            by.setdefault(x["cell"], []).append(x)
    frag_cells = sorted({x["cell"] for x in rows if FRAG.search(s(x["values"].get("notes"))) and x["cell"] != "0"})
    multi = {k: g for k, g in by.items() if len(g) > 1}
    print(f"{len(rows)} rows on {len(by)} cells; {len(multi)} cells have several rows; {len(frag_cells)} cells have a "
          f"fragment-like note; sizing those ({len(frag_cells)} get_l2_info calls, >= 5 s apart)", flush=True)

    from fafbseg import flywire
    size = {}
    for i, r in enumerate(frag_cells):
        wait()
        try:
            d = flywire.get_l2_info(int(r), progress=False, max_threads=1, dataset=DATASTACK)
            size[r] = (float(d.length_um.iloc[0]), int(d.l2_chunks.iloc[0]))
        except Exception as ex:
            size[r] = (np.nan, 0)
            print(f"  {r}: {str(ex)[:80]}", flush=True)
        if i % 20 == 19:
            print(f"  {i + 1}/{len(frag_cells)}, {time.time() - t0:.0f} s", flush=True)

    out = []
    for cell in sorted(set(multi) | set(frag_cells)):
        g = by.get(cell, [])
        differs = [f for f in FIELDS if len({norm(f, x["values"].get(f)) for x in g}) > 1]
        L, n2 = size.get(cell, (np.nan, ""))
        for x in g:
            v = x["values"]
            out.append(dict(cell=cell, n_rows=len(g), length_um=L if not np.isnan(L) else "", l2_chunks=n2,
                            frag_note=bool(FRAG.search(s(v.get("notes")))), differs=", ".join(differs),
                            tab=x["tab"], row=x["row"], hop=s(v.get("hop")), root_id_orig=s(v.get("root_id_orig")),
                            root_id_curr=s(v.get("root_id_curr")), at_xyz=x["at"], xyz=s(v.get("xyz")),
                            **{f: s(v.get(f)) for f in FIELDS}))
    D = pd.DataFrame(out)
    path = fig(f"{stamp}_cell_consistency.tsv")
    D.to_csv(path, sep="\t", index=False)

    F = D[D.frag_note]
    print(f"\nfragment-like notes: {len(F)} rows on {F.cell.nunique()} cells; statuses {F.status.str.lower().value_counts().to_dict()}")
    for cell, g in F.groupby("cell"):
        allrows = D[D.cell == cell]
        L = g.length_um.iloc[0]
        others = [f"{t} r{r}" for t, r in zip(allrows.tab, allrows.row) if not ((g.tab == t) & (g.row == r)).any()]
        print(f"  {cell}  {L if L == '' else round(L)} um, {g.l2_chunks.iloc[0]} L2 chunks; "
              + "; ".join(f"{t} r{r} [{st}] '{n[:70]}'" for t, r, st, n in zip(g.tab, g.row, g.status, g.notes))
              + (f"; same cell as {', '.join(others)}" if others else ""))
    M = D[(D.n_rows > 1) & (D.differs != "")]
    print(f"\ncells with several rows that disagree: {M.cell.nunique()} of {D[D.n_rows > 1].cell.nunique()}; fields: "
          f"{pd.Series([f for d in M.drop_duplicates('cell').differs for f in d.split(', ')]).value_counts().to_dict()}")
    print(f"written {path}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
