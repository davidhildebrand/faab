"""Reconcile the tracking tabs' rows that sit on the same live cell (David, 2026-10-31).

  - Rows of one cell in the SAME block (same header row) were correctly merged: they collapse into one row. The record
    row (best status, then most nsyn, then the first) stays with the cell's tracking and a live nsyn recount from the
    block's upstream cell (upstream_PR / hop1_root_id); the other rows are deleted. Not collapsed: the PRs tab (groups
    have their own rules), yellow MISMATCH rows, and cells whose notes mention a false merge / merge error.
  - Every other row of the cell (other blocks, other tabs) carries the same tracking: the best status (RANK: checked >
    proofread > complete > adequate > review > to_review > needs_extending > missing soma > in progress > stopped),
    proofreader and checker names merged (each person once; full name over initials), the newest pr_date of the
    best-status rows, the newest chk_date, blank annotation / link filled, one SeaTable mark, and the same notes. Cells
    without any status (NEW / NEW-DUP / candidate only) are left as they are apart from collapsing; yellow MISMATCH rows
    keep their own values.
  - Notes = the union of the rows' notes without pointers to other rows ('duplicate of r12', 'merged into r5', 'merged
    into PRg1hop1 r5'), remarks that only say the row is a repeat ('duplicate', 'same id appeared in multiple column',
    'this ID is repeated throughout, ...'), near-duplicates, fragment wording on cells that are whole now (>= WHOLE_L2
    L2 chunks; 'fragment of a L5 neuron' -> 'L5') and 'please review' on checked cells; row-specific notes ('xyz off the
    cell ...', merge history of the row's own root 'merged to/into/with ...', PRs bundle / pairing notes) stay on their
    own row only.
  - A row whose cell is now the block's own upstream cell (a fragment of that cell, merged into it: e.g. R8 pieces in
    the R8's partner block, David 2026-10-31) is not a partner any more and is removed; references to it are dropped
    from tracked_in. Its synapses would be the cell's synapses onto itself, so nsyn could not be recounted either.
  - Row deletions shift the rows below: 'rN' / '<tab> rN' references in notes and tracked_in are renumbered (a deleted
    row -> its record row) and hyperlinks / row heights follow their rows.
Every change goes through seatable_sync.Book (logged with the reason)."""
import re

from faab.partners.sheet import cell_text as s, scan_book
from faab.pr.prs_tab import as_date

RANK = ["checked", "proofread", "complete", "adequate", "review", "to_review", "needs_extending", "missing soma",
        "in progress", "stopped"]  # complete / to_review / needs_extending / missing soma: SeaTable's words
WHOLE_L2 = 100
NO_COLLAPSE = ("PRs",)
UPSTREAM = ("hop1_root_id", "upstream_PR")
REF_COLS = ("notes", "tracked_in")
FALSE_MERGE = re.compile(r"false(ly)? merge|merge error|wrongly merged", re.I)
POINTER = re.compile(r"^(duplicate( target)? of\b|merged into (r\d+|[A-Za-z0-9_]+ r\d+)\b)", re.I)
REPEAT = re.compile(r"^(duplicate|same (id|neuron) appeared in multiple columns?|multiple ent(e)?ries|this id is repeated "
                    r"(throughout|in the document))[.\s]*$", re.I)
OWN = re.compile(r"^(xyz off the cell|census bundle|second bundle|off-bundle|moved to group|split from group|added by "
                 r"bundle tracking|pair|terminal|names differ|probable merge error|extra axon|check that|"
                 r"(branch |fragment )?merged (to|into|with)\b)", re.I)  # merge history is about the row's own root_id_orig
STALE = re.compile(r"small chunk (of|connecte?d? to) a (big|larger) neuron|in between EM slices are broken|"
                   r"(tiny )?chunk connected to (a )?neuron|small fragment, belonging to a neuron|orphan (fragment|segment)\??",
                   re.I)
FRAG_OF = re.compile(r"^fragment of an? (\S+) neuron$", re.I)
REPEAT_LEAD = re.compile(r"^this id (appears|is repeated)( multiple times| throu?ghout| throoughout| in the document)?[,.]?\s*", re.I)


def rank(st):
    t = s(st).lower()
    if t in RANK:
        return RANK.index(t)
    return RANK.index("review") if "review" in t else len(RANK)  # 'needs review by KT'; NEW / NEW-DUP / candidate last


def parts(text):
    return [p.strip() for p in re.split(r";\s*", s(text)) if p.strip()]


def initials(name):
    w = re.findall(r"[A-Za-z]+", name)
    return (w[0][0] + w[-1][0]).upper() if len(w) >= 2 else name.strip().upper()


def merge_names(values):
    """names of several rows in order, each person once: 'KT' is dropped next to 'Kristo T.' and replaced by it when it
    came first; 'Nandita D' and 'Nandita D.' count as one"""
    out = []
    for v in values:
        for t in (x.strip() for x in re.split(r"[;,]", s(v))):
            if not t:
                continue
            bare = re.fullmatch(r"[A-Z]{2,4}", t) is not None
            same = [i for i, o in enumerate(out) if initials(o) == initials(t) or o.upper() == t.upper()]
            if not same:
                out.append(t)
            elif not bare and re.fullmatch(r"[A-Z]{2,4}", out[same[0]]):
                out[same[0]] = t
    return "; ".join(out)


def shared_part(p, whole, checked, cell=""):
    """the note part as every row of the cell carries it, or None (pointer, repeat remark, row history, stale wording)"""
    if POINTER.search(p) or REPEAT.search(p) or OWN.search(p):
        return None
    q = re.sub(r"\s+", " ", REPEAT_LEAD.sub("", p)).strip(" ,;")
    if q != p and re.fullmatch(r"[-\s]*[A-Z]{2,3}[.\s]*", q):  # 'this ID is repeated throughout - KT' says no more
        return None
    p = q
    if whole:
        m = FRAG_OF.match(p)
        if m:
            return m.group(1)
        p = STALE.sub("", p).strip(" ,.-;")
    if checked:
        p = re.sub(r"^(please )?review$|,?\s*please review$", "", p, flags=re.I).strip(" ,.-;")
    return p or None


def key(p):
    return re.sub(r"\s+", " ", re.sub(r"[\s.]+$", "", p)).lower()


def dedupe(ps):
    """drop repeats and parts that another part already contains ('the backbone seems well, please review' inside
    'the backbone seems well, please review - KT')"""
    out = []
    for p in ps:
        k = re.sub(r"\s*-\s*[a-z]{2,3}$", "", key(p)).strip()  # key() is lowercase
        inside = len(k) >= 15 and any(k in key(q) for q in ps if q is not p and len(key(q)) > len(key(p)))
        if inside or key(p) in map(key, out):
            continue
        out.append(p)
    return out


def first_n(v):
    m = re.match(r"\d+", s(v))
    return int(m.group()) if m else 0


def reconcile(wb, B, rows, cell_of, sizes, yellow, recount=None, latest_of=None, log=print):
    """rows: scan_book rows of the tracking tabs (row numbers before any deletion); cell_of(x) -> live cell id or '';
    sizes: cell -> L2 chunks (cells not listed are never called whole); yellow(x) -> True for MISMATCH rows;
    recount(pairs) -> {(upstream id, cell): live synapses} for [(upstream id as written, cell)], or None (then the
    collapsed rows' nsyn are summed); latest_of(ids) -> {id: [current roots]} for upstream ids as written, or None
    (then rows whose cell became the block's own upstream cell are not looked for)."""
    val = lambda x, n: wb[x["tab"]].cell(x["row"], x["cols"][n]).value if n in x["cols"] else None
    up_of = {}
    for x in rows:
        up = next((n for n in UPSTREAM if n in x["cols"]), None)
        u = s(val(x, up)) if up and x["tab"] not in NO_COLLAPSE else ""
        if re.fullmatch(r"\d{15,}", u):
            up_of[(x["tab"], x["row"])] = u

    # ---- rows whose cell is the block's own upstream cell now (its fragment was merged into it): not a partner
    cur = latest_of(sorted(set(up_of.values()))) if latest_of and up_of else {}
    selfs = [x for x in rows if (x["tab"], x["row"]) in up_of and not yellow(x)
             and re.fullmatch(r"\d+", cell_of(x) or "") and int(cell_of(x)) in cur.get(up_of[(x["tab"], x["row"])], [])]
    self_keys = {(x["tab"], x["row"]) for x in selfs}
    groups = {}
    for x in rows:
        c = cell_of(x)
        if c and c != "0" and "status" in x["cols"] and not yellow(x) and (x["tab"], x["row"]) not in self_keys:
            groups.setdefault(c, []).append(x)
    groups = {c: g for c, g in groups.items() if len(g) > 1}
    stats = dict(cells=len(groups), carried=0, collapsed_rows=0, self_rows=len(selfs))

    # ---- one tracking per cell
    for c, g in groups.items():
        g.sort(key=lambda x: (rank(val(x, "status")), x["tab"], x["row"]))
        best_r = rank(val(g[0], "status"))
        if best_r >= len(RANK):
            continue  # no row has a status: nothing to carry over
        stats["carried"] += 1
        why = f"reconcile: one cell on {len(g)} rows"
        best = [x for x in g if rank(val(x, "status")) == best_r]
        status = s(val(g[0], "status")).lower()
        names = {n: merge_names(val(x, n) for x in g) for n in ("proofreader", "checker")}
        dates = lambda xs, n: [d for d in (val(x, n) if hasattr(val(x, n), "year") else as_date(s(val(x, n))) for x in xs) if d]
        pr_date = max(dates(best, "pr_date") or dates([x for x in g if rank(val(x, "status")) < len(RANK)], "pr_date") or [None])
        chk_date = max(dates(g, "chk_date") or [None])
        ann = next((s(val(x, "annotation")) for x in g if s(val(x, "annotation"))), "")
        link = next((s(val(x, "link")) for x in g if s(val(x, "link"))), "")
        marks = [s(val(x, "SeaTable")) for x in g if s(val(x, "SeaTable"))]
        mark = next((m.split(":")[0].strip() for pref in ("added (", "added", "") for m in marks if m.lower().startswith(pref)), "")
        label = max((m.split(":", 1)[1].strip() for m in marks if ":" in m), key=len, default="")
        sea = mark + (f": {label}" if label and mark else "")
        whole = sizes.get(c, 0) >= WHOLE_L2
        shared, seen = [], set()
        for x in g:
            for p in parts(val(x, "notes")):
                q = shared_part(p, whole, status == "checked", c)
                if q and key(q) not in seen:
                    seen.add(key(q))
                    shared.append(q)
        shared = dedupe(shared)
        for x in g:
            tab, r, cols = x["tab"], x["row"], x["cols"]
            B.put(tab, r, cols["status"], status, why)
            for n in ("proofreader", "checker"):
                if n in cols and names[n]:
                    B.put(tab, r, cols[n], names[n], why)
            if "pr_date" in cols and pr_date:
                B.put(tab, r, cols["pr_date"], pr_date, why)
            if "chk_date" in cols and chk_date:
                B.put(tab, r, cols["chk_date"], chk_date, why)
            if "annotation" in cols and ann and not s(val(x, "annotation")):
                B.put(tab, r, cols["annotation"], ann, why)
            if "link" in cols and link and not s(val(x, "link")):
                B.put(tab, r, cols["link"], link, why)
                wb[tab].cell(r, cols["link"]).hyperlink = link
            if "SeaTable" in cols and sea:
                B.put(tab, r, cols["SeaTable"], sea, why)
            if "notes" in cols:
                own = [p for p in parts(val(x, "notes")) if OWN.search(p) and not POINTER.search(p) and key(p) not in seen]
                B.put(tab, r, cols["notes"], "; ".join(shared + own), why)

    # ---- collapse same-block rows of one cell
    hdr = {}
    for x in rows:
        hdr[(x["tab"], x["row"])] = B.header_row(x["tab"], x["row"])
    blocks = {}
    for c, g in groups.items():
        for x in g:
            if x["tab"] not in NO_COLLAPSE:
                blocks.setdefault((x["tab"], hdr[(x["tab"], x["row"])], c), []).append(x)
    drop, rec_of, todo = {}, {}, []
    for (tab, h, c), g in blocks.items():
        if len(g) < 2 or any(FALSE_MERGE.search(s(val(x, "notes"))) for x in g):
            continue
        g.sort(key=lambda x: (rank(val(x, "status")), s(val(x, "status")).upper() == "NEW-DUP", -first_n(val(x, "nsyn")), x["row"]))
        rec = g[0]
        todo.append((rec, g[1:], c, up_of.get((rec["tab"], rec["row"]), "")))
        for x in g[1:]:
            drop.setdefault(tab, set()).add(x["row"])
            rec_of[(tab, x["row"])] = rec["row"]
    counts = recount([(u, c) for _, _, c, u in todo if u]) if recount and todo else {}
    for rec, others, c, u in todo:
        tab, cols = rec["tab"], rec["cols"]
        if "nsyn" in cols:
            n = counts.get((u, c)) if u else None
            n = n if n else sum(first_n(val(x, "nsyn")) for x in [rec] + others)
            B.put(tab, rec["row"], cols["nsyn"], int(n), f"collapsed {len(others) + 1} rows of one cell: nsyn "
                  + ("live recount" if counts.get((u, c)) else "summed"))
        if "notes" in cols:  # pointers to the twins go with them
            B.put(tab, rec["row"], cols["notes"], "; ".join(p for p in parts(val(rec, "notes")) if not POINTER.search(p)),
                  "collapsed rows of one cell")
        for x in others:
            B.log.append(dict(tab=tab, row=x["row"], column="row", old=s(val(x, "root_id_orig")), new="",
                              why=f"collapsed into r{rec['row']} (same cell {c}, same block)"))
            B.touched.add(tab)
    stats["collapsed_rows"] = sum(len(v) for v in drop.values())
    for x in selfs:
        drop.setdefault(x["tab"], set()).add(x["row"])
        rec_of[(x["tab"], x["row"])] = None  # nothing replaces it: references to it are dropped
        B.log.append(dict(tab=x["tab"], row=x["row"], column="row", old=s(val(x, "root_id_orig")), new="",
                          why=f"removed: its cell {cell_of(x)} is now the block's own upstream cell (a fragment merged "
                              f"into it), not a partner"))
        B.touched.add(x["tab"])

    # ---- delete, then renumber references
    new_of = {}
    for tab, rs in drop.items():
        ws = wb[tab]
        heights = {r: ws.row_dimensions[r].height for r in range(1, ws.max_row + 1) if ws.row_dimensions[r].height}
        for r in sorted(rs, reverse=True):
            ws.delete_rows(r)
        dead = sorted(rs)
        shift = lambda r: r - sum(1 for d in dead if d < r)
        m = {}
        for r in range(1, ws.max_row + len(rs) + 1):
            m[r] = (shift(rec_of[(tab, r)]) if rec_of[(tab, r)] else 0) if r in rs else shift(r)  # 0: removed outright
        new_of[tab] = m
        for row in ws.iter_rows():
            for cell in row:
                if cell.hyperlink is not None:
                    cell.hyperlink.ref = cell.coordinate
        for r in list(ws.row_dimensions):
            ws.row_dimensions[r].height = None
        for r, h in heights.items():
            if r not in rs:
                ws.row_dimensions[m[r]].height = h
    if new_of:
        ref = re.compile(r"\b(?:(" + "|".join(re.escape(w.title) for w in wb.worksheets) + r") )?(r|row )(\d+)\b")
        for x in scan_book(wb):
            if re.search(r"_b\d{6}d$", x["tab"]):
                continue
            for n in REF_COLS:
                if n not in x["cols"]:
                    continue
                cell = wb[x["tab"]].cell(x["row"], x["cols"][n])
                t = s(cell.value)

                def sub(q):
                    m = new_of.get(q[1] or x["tab"], {})
                    k = m.get(int(q[3]), int(q[3]))
                    return f"{q[1] + ' ' if q[1] else ''}{q[2]}{k}" if k else "\0"  # \0: a removed row
                u = ref.sub(sub, t)
                if n == "tracked_in":  # two rows collapsed into one; removed rows dropped
                    u = "; ".join(dict.fromkeys(p for p in u.split("; ") if p and "\0" not in p))
                elif "\0" in u:
                    u = t  # a note naming a removed row stays as written
                if u != t:
                    cell.value = u
                    B.log.append(dict(tab=x["tab"], row=x["row"], column=n, old=t, new=u, why="row references renumbered"))
                    B.touched.add(x["tab"])
    log(f"reconcile: {stats['cells']} cells on several rows; tracking carried over on {stats['carried']}; "
        f"{stats['collapsed_rows']} same-block rows collapsed and {stats['self_rows']} rows removed whose cell is the "
        f"block's own upstream cell ({', '.join(f'{t} {len(v)}' for t, v in drop.items()) or 'none'})")
    return new_of
