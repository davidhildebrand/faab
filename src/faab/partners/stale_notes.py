"""Notes that are no longer true (David 2026-11-07: proofreaders fix a cell but don't always update the sheet).

Two kinds, checked on every update (scripts/sheets/seatable_sync.py):
  - 'xyz off the cell (on X)' (written by an earlier sync) when the row's point is on its cell again: the flag and the
    proofreader's reply to it ('issue resolved, coordinates updated - KT') are dropped, the yellow highlight cleared;
  - 'missing soma' / 'no soma?' / 'can't find soma' ... when the cell now carries exactly one nucleus of the nucleus
    table (nuclei_v1_aedes mapped to today's roots); our own 'no nucleus on this root ...' when it carries one or more,
    and 'several nuclei on this root: merge?' when it carries at most one.
  - replies and reports of work done ('issue resolved', 'xyz coordinates and link updated', 'xyz is corrected',
    'New ID: ...') once the row's point is checked on its cell, unless the part still asks or flags something;
    'X ..., not merged' once X's rows sit on the same cell (David 2026-11-08: only clear notes, less is more).
Notes are split at ';' and the remaining parts kept verbatim; nothing is ever added. Parts that record finished work ('missing soma
identified', 'axonal segments and soma were missing') are kept. A SeaTable status 'missing soma' is not changed (the
sheet takes SeaTable's status); seatable_sync lists those rows instead."""
import datetime
import os
import re
import time

import numpy as np
import pandas as pd

OFF = re.compile(r"^xyz off the cell \(on \d+\)( - .*)?$", re.I)
# the proofreader's answer to an off-cell flag
REPLY = re.compile(r"^(the |this )?issue (is )?resolved\b", re.I)
# reports of work done / replies (David 2026-11-08: notes flag issues or curiosities; remove replies once checked):
# 'issue resolved, coordinates updated - KT', 'xyz coordinates and link updated', 'xyz is corrected -DR',
# 'updated ID: ... Coordinates: ...', 'New ID: ...', 'the merging is complete' - dropped only when the row's point is on
# its cell, and never when the part still asks or flags something (FLAG)
HISTORY = re.compile(r"\b(issue|problem)s?\b[^;]*\b(resolved|fixed)\b|"
                     r"\b(xyz|coordinates?|root ?ids?|ids?|links?|point)\b[^;]*\b(updated|corrected|fixed)\b|"
                     r"\bupdated (root )?(id|xyz|coordinates?)\b|\bnew id:|have worked on (that|it)\b|"
                     r"merging is complete", re.I)
FLAG = re.compile(r"\?|answer|pending|please|check|unsure|incomplete|missing|"
                  r"not (yet |completely |fully )?(proofread|complete|done|merged)", re.I)
# 'extension of 648518347463102525, not merged - pending Kristo T.': stale once that id's rows sit on this row's cell
NOT_MERGED = re.compile(r"\b(\d{15,})\b[^;]*\bnot merged\b|\bnot merged\b[^;]*\b(\d{15,})\b", re.I)
# a part that says the soma is missing (and nothing that records it was found or attached)
SOMA_MISSING = re.compile(r"^(no soma\??|soma missing\??|missing soma\??|can'?t find (the )?soma|"
                          r"(hard|difficult) to identify (the )?(missing )?soma\b.*)$", re.I)
SOMA_PREFIX = re.compile(r"^missing soma, ", re.I)  # 'missing soma, this ID is repeated again later' -> keep the rest
OUR_NONE = re.compile(r"^no nucleus on this root\b", re.I)
OUR_MANY = re.compile(r"^several nuclei on this root\b", re.I)
DONE = re.compile(r"identified|attached|were missing|was missing|found", re.I)


def parts(notes):
    return [p.strip() for p in notes.split(";") if p.strip()]


def clean(notes, point_on_cell, n_nuclei, at="", cells_of=None):
    """(new notes, [reasons]) for one row; only removes (or shortens 'missing soma, ...'), never adds.
    n_nuclei None = unknown (no soma rule applied); at = the root at the xyz now: when the point is still off the cell,
    older flags naming another segment are dropped (the sync adds the current one); cells_of(id) -> the live cells of
    the sheet rows with that root_id_orig (for 'not merged' flags), or None"""
    out, why = [], []
    off = point_on_cell and any(OFF.match(p) for p in parts(notes))
    for p in parts(notes):
        if off and (OFF.match(p) or REPLY.match(p)):
            why.append("point is on the cell again")
            continue
        if point_on_cell and at and HISTORY.search(p) and not FLAG.search(p):
            why.append("reply / report of work done, point checked on the cell")
            continue
        m = NOT_MERGED.search(p) if point_on_cell and at and cells_of else None
        if m and at in cells_of(m.group(1) or m.group(2)):
            why.append("merged now (same cell)")
            continue
        if not point_on_cell and at and OFF.match(p) and f"(on {at})" not in p:
            why.append("older off-cell flag replaced")
            continue
        if n_nuclei is not None and not DONE.search(p):
            if n_nuclei == 1 and SOMA_MISSING.match(p):
                why.append("the cell has a soma (nucleus) now")
                continue
            if n_nuclei == 1 and SOMA_PREFIX.match(p):
                p = SOMA_PREFIX.sub("", p)
                why.append("the cell has a soma (nucleus) now")
            if n_nuclei >= 1 and OUR_NONE.match(p):
                why.append("the cell has a soma (nucleus) now")
                continue
            if n_nuclei <= 1 and OUR_MANY.match(p):
                why.append("one nucleus on the cell now")
                continue
        if p not in out:  # 'missing soma; missing soma' -> once
            out.append(p)
    return "; ".join(out), list(dict.fromkeys(why))


def nuclei_by_root(client, table_copy, throttle):
    """{root: number of nuclei} now: the nucleus table copy (pt_supervoxel_id) mapped to today's roots, in batches of
    50k (each call >= 2 s after the last; read-only)"""
    N = pd.read_csv(table_copy, sep="\t", usecols=["valid", "pt_supervoxel_id"])
    sv = N[N.valid.astype(str) == "True"].pt_supervoxel_id.astype(np.int64).to_numpy()
    when, t, roots = datetime.datetime.now(datetime.timezone.utc), 0.0, np.zeros(len(sv), np.int64)
    for k in range(0, len(sv), 50000):
        t = throttle(t)
        roots[k:k + 50000] = np.asarray(client.chunkedgraph.get_roots(sv[k:k + 50000], timestamp=when)).astype(np.int64)
        t = time.time()
    return pd.Series(roots.astype(str)).value_counts().to_dict()


def default_table_copy(tmp):
    p = os.path.join(tmp, "nuclei_v1_aedes_v542.tsv")
    return p if os.path.exists(p) else None
