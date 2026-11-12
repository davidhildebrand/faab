"""Reconcile the tracking tabs' rows that sit on the same live cell (David, 2026-10-31).

  - Rows of one cell in the SAME block (same header row) were correctly merged: they collapse into one row. The record
    row (best status, then most nsyn, then the first) stays with the cell's tracking and an nsyn recount (synapse
    snapshot caught up to now) from the
    block's upstream cell (upstream_PR / hop1_root_id); the other rows are deleted. Rows that list the cells they FEED
    instead (T4T5-1: T4T5_root_id, T4T5-2: T4T5-1_root_id, with nsyn aligned to that list) are recounted per partner
    (2026-11-11: two T4T5-1 rows of a merged cell got '11, 3' + '4' -> '15', dropping the second target): every partner
    that block lists is counted from the cell, those with >= MIN_SYN are listed strongest first with their nsyn,
    nsyn_out = the sum over all of them and T4T5_reached / T4T5_root_id follow (without a recount: per-partner sums of
    the rows' own lists). cable_um / neuropils come from the collapsed row with the most cable (the merged cell is at
    least that big). Not collapsed: the PRs tab (groups have their own rules), yellow MISMATCH rows, and cells whose
    notes mention a false merge / merge error.
  - Every other row of the cell (other blocks, other tabs) carries the same tracking: the best status (RANK: checked >
    proofread > complete > adequate > review > to_review > needs_extending > missing soma > in progress > stopped),
    proofreader and checker names merged (each person once; full name over initials), the newest pr_date of the
    best-status rows, the newest chk_date, blank annotation / link filled, and one SeaTable mark. Cells
    without any status (NEW / NEW-DUP / candidate only) are left as they are apart from collapsing; yellow MISMATCH rows
    keep their own values.
  - Notes are NOT copied between rows (David 2026-11-08: "only clear notes ... less is more"; until then every row got
    the union, which spread replies such as 'xyz coordinates and link updated'). Each row's own notes are cleaned:
    pointers to other rows ('duplicate of r12', 'merged into r5', 'merged into PRg1hop1 r5'), remarks that only say the
    row is a repeat ('duplicate', 'same id appeared in multiple column', 'this ID is repeated throughout, ...'),
    near-duplicates, fragment wording on cells that are whole now (>= WHOLE_L2 L2 chunks; 'fragment of a L5 neuron'
    -> 'L5') and 'please review' on checked cells go; row-specific notes ('xyz off the cell ...', merge history
    'merged to/into/with ...', PRs bundle / pairing notes) are kept as written.
  - A row whose cell is now the block's own upstream cell (a fragment of that cell, merged into it: e.g. R8 pieces in
    the R8's partner block, David 2026-10-31) is not a partner any more and is removed; references to it are dropped
    from tracked_in. Its synapses would be the cell's synapses onto itself, so nsyn could not be recounted either.
  - Rows are no identifiers (David 2026-11-12: "they move around"): tracked_in names the cell's other tabs (and
    '<tab> (<hop>)' for another block of the same tab), recomputed every update; notes that only point at another row
    by number are cleared ('row N is the cell' names that row's root_id) - no_row_numbers().
  - Row deletions shift the rows below: any 'rN' / '<tab> rN' reference left in notes is renumbered (a deleted row ->
    its record row) and hyperlinks / row heights follow their rows.
Every change goes through seatable_sync.Book (logged with the reason)."""
import re

from faab.partners.sheet import cell_text as s, scan_book
from faab.pr.prs_tab import as_date

RANK = ["checked", "proofread", "complete", "adequate", "review", "to_review", "needs_extending", "missing soma",
        "in progress", "stopped"]  # complete / to_review / needs_extending / missing soma: SeaTable's words
WHOLE_L2 = 100
NO_COLLAPSE = ("PRs",)
UPSTREAM = ("hop1_root_id", "upstream_PR")
DOWNSTREAM = ("T4T5-1_root_id", "T4T5_root_id")  # lists of the cells a row's cell feeds (T4T5-2 / T4T5-1 rows)
MIN_SYN = 3  # the blocks' threshold for listing a partner
IDS = re.compile(r"\d{15,}")
REF_COLS = ("notes", "tracked_in")
FALSE_MERGE = re.compile(r"false(ly)? merge|merge error|wrongly merged", re.I)
POINTER = re.compile(r"^(duplicate( target)? of\b|merged into (r\d+|[A-Za-z0-9_]+ r\d+)\b)", re.I)
REPEAT = re.compile(r"^(duplicate|same (id|neuron) appeared in multiple columns?|multiple ent(e)?ries|this id is repeated "
                    # 'this ID is repeated again later' (2026-11-07: the lead alone was cut, leaving 'again later')
                    r"(throughout|in the document|(again )?(later|below|above)))[.\s]*$", re.I)
OWN = re.compile(r"^(xyz off the cell|census bundle|second bundle|off-bundle|moved to group|split from group|added by "
                 r"bundle tracking|pair|terminal|names differ|probable merge error|extra axon|check that|"
                 r"(branch |fragment )?merged (to|into|with)\b|"
                 # the LMCs tab's typing remarks (scripts/sheets/lmc_tab.py) belong to that tab's row only (2026-11-07);
                 # notes are split at ';', so 'SeaTable says L3; the automatic typing says L2 ...' arrives in two parts
                 r"SeaTable says L[1-5]$|the automatic typing says|no nucleus on this root|several nuclei on this root|"
                 r"medulla part short or missing|features unlike the typed LMCs|left side: the automatic typing|"
                 r"no point found on the cell|flagged as an LMC by David|"
                 # the T4T5s tab's remarks (scripts/sheets/t4t5_tab.py, 2026-11-12)
                 r"about \d+ times the usual T4/T5 size|smaller than the usual T4/T5|only \d+% of its outputs in the "
                 r"lobula plate|shape unlike the typed T4/T5|SeaTable says T[45]$|its inputs say T[45]$)",
                 re.I)  # merge history is about the row's own root_id_orig
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


def retitle(wb, B, rows, hdr, drop):
    """a block title that states the block's size ('101 cells' over 101 rows; title = the row above the header) follows
    the rows dropped from it (2026-11-11: two T4T5-1 pairs merged, the title kept saying 101). Before delete_rows, on
    the original row numbers; a title whose number is not the block's row count is left alone."""
    gone = {}
    for tab, rs in drop.items():
        for r in rs:
            gone[(tab, hdr[(tab, r)])] = gone.get((tab, hdr[(tab, r)]), 0) + 1
    for (tab, h), k in gone.items():
        n = sum(1 for y in rows if y["tab"] == tab and hdr[(tab, y["row"])] == h)
        t = s(wb[tab].cell(h - 1, 1).value) if h > 1 else ""
        if re.search(rf"\b{n} cells\b", t):
            B.put(tab, h - 1, 1, re.sub(rf"\b{n} cells\b", f"{n - k} cells", t, count=1),
                  f"block title: {n} -> {n - k} cells after collapsing")


ROW_POINTER = re.compile(r"^(duplicate( target)? of|merged into)\s+(\w+ )?(r|row )\d+\.?$", re.I)
CELL_AT = re.compile(r"\b(?:row |r)(\d+) is the cell\b")


def no_row_numbers(wb, B, rows, cell_of, yellow, self_keys, val):
    """rows are no identifiers - they move (David 2026-11-12): tracked_in names the OTHER places the cell is on (tab;
    in the same tab, '<tab> (<hop>)' for another block), recomputed every update; notes that only point at another row
    by number ('duplicate of r470' on a NEW-DUP row: its status says it) are cleared, and 'row 71 is the cell' names
    that row's root_id. Yellow MISMATCH rows (cell unknown) keep their tracked_in tabs with the row numbers removed."""
    hdr = {}  # tab -> {row: header row above it}, one pass per tab (B.header_row scans upward cell by cell)

    def header(x):
        if x["tab"] not in hdr:
            m, h = {}, 0
            for i, row in enumerate(wb[x["tab"]].iter_rows(values_only=True), 1):
                names = [s(v) for v in row]
                if "xyz" in names and ("root_id_curr" in names or "root_id_orig" in names):
                    h = i
                m[i] = h or i
            hdr[x["tab"]] = m
        return hdr[x["tab"]].get(x["row"], x["row"])

    live = lambda x: re.search(r"_b\d{6}d$", x["tab"]) is None
    every = {}
    for x in rows:
        c = cell_of(x)
        if c and c != "0" and (x["tab"], x["row"]) not in self_keys and live(x):
            every.setdefault(c, []).append(x)
    at = {(x["tab"], x["row"]): x for x in rows}
    why = "row numbers out: tracked_in names the cell's other tabs; row pointers cleared"
    for x in rows:
        if not live(x):
            continue
        tab, r = x["tab"], x["row"]
        if "tracked_in" in x["cols"]:
            old, c = s(val(x, "tracked_in")), cell_of(x)
            if yellow(x) or not c or c == "0":
                new = "; ".join(dict.fromkeys(re.sub(r"\s+(r|row )\d+\b", "", p).strip() for p in old.split(";") if p.strip()))
            else:
                h = header(x)
                new = "; ".join(dict.fromkeys(
                    (f"{y['tab']} ({s(val(y, 'hop'))})" if y["tab"] == tab and s(val(y, "hop")) else y["tab"])
                    for y in every.get(c, []) if not (y["tab"] == tab and header(y) == h)))
            if new != old:
                B.put(tab, r, x["cols"]["tracked_in"], new, why)
        if "notes" in x["cols"]:
            old = s(val(x, "notes"))
            if not old:
                continue
            keep, hit = [], False
            for p in parts(old):
                if ROW_POINTER.match(p):
                    hit = True
                    continue

                def cell_at(m):
                    y = at.get((tab, int(m[1])))
                    root = (s(val(y, "root_id_curr")) or s(val(y, "root_id_orig"))) if y else ""
                    return f"{root} is the cell" if root else m[0]
                q = CELL_AT.sub(cell_at, p)
                hit |= q != p
                keep.append(q)
            if hit:
                B.put(tab, r, x["cols"]["notes"], "; ".join(keep), why)


def reconcile(wb, B, rows, cell_of, sizes, yellow, recount=None, latest_of=None, recount_out=None, log=print):
    """rows: scan_book rows of the tracking tabs (row numbers before any deletion); cell_of(x) -> live cell id or '';
    sizes: cell -> L2 chunks (cells not listed are never called whole); yellow(x) -> True for MISMATCH rows;
    recount(pairs) -> {(upstream id, cell): synapses now} for [(upstream id as written, cell)], or None (then the
    collapsed rows' nsyn are summed); latest_of(ids) -> {id: [current roots]} for upstream ids as written, or None
    (then rows whose cell became the block's own upstream cell are not looked for); recount_out(cell, ids) -> {id:
    synapses now from cell onto it} for downstream ids as written, or None (then per-partner sums of the rows' lists)."""
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
    no_row_numbers(wb, B, rows, cell_of, yellow, self_keys, val)

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
                # each row keeps its own notes, only cleaned (David 2026-11-08: "only clear notes ... less is more"):
                # pointers, repeat remarks, fragment wording on whole cells and 'please review' on checked cells go
                mine = []
                for p in parts(val(x, "notes")):
                    q = p if OWN.search(p) and not POINTER.search(p) else shared_part(p, whole, status == "checked", c)
                    if q:
                        mine.append(q)
                B.put(tab, r, cols["notes"], "; ".join(dedupe(mine)), why)

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
        u = up_of.get((rec["tab"], rec["row"]), "")
        # no upstream cell: a row that lists the cells it feeds (T4T5-1 / T4T5-2), nsyn aligned to that list
        dn = "" if u else next((n for n in DOWNSTREAM if n in rec["cols"] and any(IDS.search(s(val(x, n))) for x in g)), "")
        todo.append((rec, g[1:], c, u, dn, h))
        for x in g[1:]:
            drop.setdefault(tab, set()).add(x["row"])
            rec_of[(tab, x["row"])] = rec["row"]
    counts = recount([(u, c) for _, _, c, u, _, _ in todo if u]) if recount and todo else {}
    for rec, others, c, u, dn, h in todo:
        tab, cols = rec["tab"], rec["cols"]
        why_c = f"collapsed {len(others) + 1} rows of one cell"
        if dn:
            listed = {}  # partner -> synapses, summed over the rows' own aligned lists
            for x in [rec] + others:
                ids, ns = IDS.findall(s(val(x, dn))), re.findall(r"\d+", s(val(x, "nsyn")))
                for i, p in enumerate(ids):
                    listed[p] = listed.get(p, 0) + (int(ns[i]) if len(ns) == len(ids) else 0)
            # every partner the block lists, so a partner both pieces fed weakly can reach MIN_SYN after the merge
            pool = list(dict.fromkeys([*listed, *(p for y in rows if y["tab"] == tab and hdr[(tab, y["row"])] == h
                                                    for p in IDS.findall(s(val(y, dn))))]))
            live = recount_out(c, pool) if recount_out else {}
            if any(live.values()):
                n_of, how = live, "recount (synapse snapshot, caught up to now)"
                keep = sorted((p for p in pool if live.get(p, 0) >= MIN_SYN), key=lambda p: -live[p])
                total = sum(live.values())
            else:
                n_of, how = listed, "summed per partner"
                keep = sorted(listed, key=lambda p: -listed[p])
                total = sum(first_n(val(x, "nsyn_out")) for x in [rec] + others) if "nsyn_out" in cols else 0
            if keep:
                why = f"{why_c}: per-partner nsyn {how}"
                B.put(tab, rec["row"], cols[dn], ", ".join(keep), why)
                if "nsyn" in cols:
                    B.put(tab, rec["row"], cols["nsyn"], ", ".join(str(n_of[p]) for p in keep), why)
                if "nsyn_out" in cols and total:
                    B.put(tab, rec["row"], cols["nsyn_out"], int(total), why)
                reach = keep
                if dn != "T4T5_root_id" and "T4T5_root_id" in cols:  # T4T5-2: the targets its T4T5-1 partners feed
                    reach = list(dict.fromkeys(p for x in [rec] + others for p in IDS.findall(s(val(x, "T4T5_root_id")))))
                    B.put(tab, rec["row"], cols["T4T5_root_id"], ", ".join(reach), why)
                if "T4T5_reached" in cols:
                    B.put(tab, rec["row"], cols["T4T5_reached"], len(reach), why)
        elif "nsyn" in cols:
            n = counts.get((u, c)) if u else None
            n = n if n else sum(first_n(val(x, "nsyn")) for x in [rec] + others)
            B.put(tab, rec["row"], cols["nsyn"], int(n), f"{why_c}: nsyn "
                  + ("recount (synapse snapshot, caught up to now)" if counts.get((u, c)) else "summed"))
        if "cable_um" in cols:  # the merged cell is at least as big as its biggest piece
            big = max([rec] + others, key=lambda x: first_n(val(x, "cable_um")))
            if big is not rec and first_n(val(big, "cable_um")) > first_n(val(rec, "cable_um")):
                for n in ("cable_um", "neuropils"):
                    if n in cols:
                        B.put(tab, rec["row"], cols[n], val(big, n), f"{why_c}: {n} of its largest piece")
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

    retitle(wb, B, rows, hdr, drop)
    new_of = delete_rows(wb, B, drop, rec_of)
    log(f"reconcile: {stats['cells']} cells on several rows; tracking carried over on {stats['carried']}; "
        f"{stats['collapsed_rows']} same-block rows collapsed and {stats['self_rows']} rows removed whose cell is the "
        f"block's own upstream cell ({', '.join(f'{t} {len(v)}' for t, v in drop.items()) or 'none'})")
    return new_of


def delete_rows(wb, B, drop, rec_of):
    """delete rows {tab: {row, ...}} and renumber 'rN' / '<tab> rN' references in notes and tracked_in of every tab;
    rec_of[(tab, row)] = the row that replaces a deleted one (references move to it) or None (references dropped from
    tracked_in; a note naming it stays as written). Hyperlinks and row heights follow their rows. Returns
    {tab: {old row: new row (0 = removed)}}"""
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
    renumber_refs(wb, B, new_of)
    return new_of


def renumber_refs(wb, B, new_of, why="row references renumbered"):
    """'rN' / '<tab> rN' references in notes and tracked_in of every tab (backup copies skipped) follow new_of
    {tab: {old row: new row (0 = removed)}}: tracked_in drops removed rows; a note naming a removed row stays as
    written. Rows are read at their NEW positions (call after the rows moved)."""
    if new_of:
        tabs = {w.title for w in wb.worksheets}
        ref = re.compile(r"\b(?:(\w+) )?(r|row )(\d+)\b")
        for x in scan_book(wb):
            if re.search(r"_b\d{6}d$", x["tab"]):
                continue
            for n in REF_COLS:
                if n not in x["cols"]:
                    continue
                cell = wb[x["tab"]].cell(x["row"], x["cols"][n])
                t = s(cell.value)

                def sub(q):
                    w = q[1]
                    if w and w not in tabs and re.search(r"[A-Z0-9_]", w):
                        return q[0]  # '<name> rN' with a name that is no tab here ('HomeR7column r5'): left alone
                    tab = w if w in tabs else x["tab"]  # 'r12', 'of r12': this row's own tab
                    k = new_of.get(tab, {}).get(int(q[3]), int(q[3]))
                    lead = f"{w} " if w else ""
                    return f"{lead}{q[2]}{k}" if k else "\0"  # \0: a removed row
                u = ref.sub(sub, t)
                if n == "tracked_in":  # two rows collapsed into one; removed rows dropped
                    u = "; ".join(dict.fromkeys(p for p in u.split("; ") if p and "\0" not in p))
                elif "\0" in u:
                    u = t  # a note naming a removed row stays as written
                if u != t:
                    cell.value = u
                    B.log.append(dict(tab=x["tab"], row=x["row"], column=n, old=t, new=u, why=why))
                    B.touched.add(x["tab"])
