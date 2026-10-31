"""Shorten the notes our scripts wrote into the tracking tabs, leaving the proofreaders' own text verbatim (David,
2026-10-30: notes are for the proofreaders; generated analysis details belong in the output tables, not the sheet).

A notes cell is split into parts at ';' outside parentheses. A part's leading 'refine YYYY-MM-DD: ' is set aside, and
the rest either matches a rule below (generated text: dropped, or replaced by a short form) or the whole part is kept
as written. The PRs-tab flags that the PRs tool reads (prs_tab.ASIDE: OFF-BUNDLE, SECOND BUNDLE) are kept. References
to another row's record ('duplicate target of <id>', 'merged into <id>') become row numbers through the `row_of`
callback (root_id_orig -> 'r117' or 'PRg1hop1 r57'; None keeps the part)."""
import re

ID = r"\d{15,}"
REFINE = re.compile(r"refine \d{4}-\d\d-\d\d:\s*")
DROP = [  # generated analysis details (kept in the census / refine / pairing output tables)
    r"", r"SeaTable R(1-6|7-8)", r"in the bundle core \(.*\)", r"reclassified from .*", r"six outer axons kept .*",
    r"ring angle .*", r"R\d margin .*", r"no mutual mirror partner",
    r"pairing: (pair )?confirmed( \(terminal pair fits the ring order.*\))?",  # the confidence column says it
    r"output synapses (within|beyond) the R1-R6 terminal zone .*",
    r"R\d \((central axon of the bundle|output synapses beyond the lamina .*|listed R\d.*|most central listed .*)\)?",
    r"xyz moved onto the axon .*", r"admitted to complete the ring", r"synapses off the lamina line",
]
SHORT = [  # (pattern, replacement): generated flags worth keeping, in a short form
    # pairing remarks that the pair columns do not hold (David, 2026-10-30: they were missed when dropped)
    (rf"pairing: conflict, ring wins: terminal pair \(({ID})\) breaks the ring order", r"pair conflict: terminal pair \1"),
    (rf"pairing: conflict, ring wins \(.*terminal said ({ID})\)", r"pair conflict: terminal said \1"),
    (rf"pairing: conflict, terminal wins \(.*ring said ({ID})\)", r"pair conflict: ring said \1"),
    (rf"pairing: conflict: terminal pair breaks the ring order \(ring-consistent partner ({ID})\)", r"pair conflict: ring partner \1"),
    (rf"terminal-zone partner differs \(({ID})\)", r"terminal partner \1"),
    (r"pairing: terminal only", "pair from terminals only"),
    (r"pairing: ring only", "pair from the ring only"),
    (r"pairing: pair confirmed, names differ", "pair names differ"),
    (r"LRS: cartridge (\d+)(: .*)?", r"LRS: cartridge \1"),
    (r"census (?:cluster|bundle) (\d+) \(.*?\), \d{4}-\d\d-\d\d: .*", r"census bundle \1"),
    (r"OFF-BUNDLE .*", "OFF-BUNDLE"),
    (r"SECOND BUNDLE: .*", "SECOND BUNDLE"),
    (r"extra lamina-ending axon in the core beyond six: runs closer to the center of group ([\d, ]+)",
     r"extra axon, closer to group \1"),
    (r"extra lamina-ending axon in the core beyond six: .*", "extra axon beyond six"),
    (r"not refined .*", "not refined (bundle not compact)"),
    (r"listed, loosely packed .*", "loosely packed"),
    (r"listed as \S+ but runs at the center of the bundle", "runs at the bundle center"),
    (r"split from group (\d+) .*", r"split from group \1"),
    (r"added \d{4}-\d\d-\d\d by bundle tracking .*", "added by bundle tracking"),
    (r"PROBABLE MERGE ERROR: the same root is the (R\d) of groups (\d+) and (\d+) .*",
     r"PROBABLE MERGE ERROR: same root is the \1 of groups \2 and \3"),
    (r"R7 \(few lamina outputs: .*check that it reaches the medulla\)\)?", "check that the R7 reaches the medulla"),
    (r"neighbor-bundle axon or misassigned \(proofreading lead\)", "neighbor-bundle axon or misassigned?"),
    (r"its axon runs in the (cartridge-\d+) bundle \(census\), listed here", r"axon runs in the \1 bundle"),
    (r"T4/T5 candidate \(line (\w+); morphology negative-like.*\)", r"T4/T5 candidate (line \1; not T4/T5-like by morphology)"),
    (r"T4/T5 candidate \(line (\w+); morphology .*\)", r"T4/T5 candidate (line \1)"),
]
REFS = [  # references to another row's record: (pattern, short form with {row})
    (rf"duplicate target of ({ID}) \(now ({ID})\)", "duplicate of {row}"),
    (rf"merged into ({ID}) \(same cell ({ID})\)", "merged into {row}"),
]


def split_parts(text):
    """'a; b (c; d); e' -> ['a', 'b (c; d)', 'e'] (';' inside parentheses does not split)"""
    parts, depth, cur = [], 0, ""
    for ch in text:
        depth += (ch == "(") - (ch == ")")
        if ch == ";" and depth <= 0:
            parts.append(cur.strip())
            cur, depth = "", 0
        else:
            cur += ch
    parts.append(cur.strip())
    return [p for p in parts if p]


def short_part(p, row_of):
    """the short form of one part ('' = drop), or p itself when it is not generated text"""
    rest = REFINE.sub("", p, count=1) if REFINE.match(p) else p
    if any(re.fullmatch(rx, rest) for rx in DROP):
        return ""
    for rx, rep in SHORT:
        if re.fullmatch(rx, rest):
            return re.sub(rx, rep, rest)
    for rx, fmt in REFS:
        m = re.fullmatch(rx, rest)
        if m:
            row = row_of(m.group(1))
            return fmt.format(row=row) if row else p
    return p


def shorten(text, row_of=lambda orig: None):
    """(new text, [(old part, new part or '')]) for one notes cell; parts the rules do not know stay verbatim"""
    out, changed = [], []
    for p in split_parts(text or ""):
        new = short_part(p, row_of)
        if new != p:
            changed.append((p, new))
            if new and new not in out:  # only generated parts are de-duplicated
                out.append(new)
        else:
            out.append(p)
    return "; ".join(out), changed
