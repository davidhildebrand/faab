"""Second pass over PR partner sheets written by pr_sheet_check.py --write-sheet: David's rules for duplicates, flags and
spelling (2026-10-25). Works on all sheets together because the same cell appears across sheets.

Cell identity = the root at the row's xyz (column `cell` of the sheet's check report; a row added by the writer has
root_id_curr as its cell). Rules:
  MISMATCH rows (root_id_curr hand-set to a current id that is not the root at the xyz): kept and FLAGGED (whole row light
      yellow in the xlsx) with a note; root_id_curr is reset to the root at the xyz, and when the hand-set id receives
      synapses from the block's presynaptic cell a new flagged row for it is inserted right below (xyz = a point on it,
      live nsyn). The two rows together say "split and xyz not updated?".
  Same cell within one block: the record is the row with the largest nsyn; the others become status NEW-DUP with the note
      "duplicate of <orig> (same cell <cell>); was <status> (<names>)". Rows kept, never deleted.
  Same cell across blocks / sheets: tracking unified - status = best (checked > proofread > adequate > review > in progress
      > NEW > stopped), proofreader / checker = every distinct name ("A; B"), dates from the best row (else any), annotation
      and notes = the distinct clauses of all rows. A fragment row whose cell now is a tracked neuron gets "merged into <cell>".
  Spelling: statuses lowercase (NEW / NEW-DUP stay), names mapped to their most common spelling, a sentence sitting in
      the proofreader / checker column moves to notes.
Writes figures/<UTC stamp>_FAABopticlobetargets-<tab>.tsv per sheet, tmp/highlight_<tab>.txt (1-based lines to fill
yellow, for pr_sheet_xlsx.py --highlight) and prints the flagged rows per proofreader. CAVE (read-only, >= 5 s apart) only
for the inserted rows' points and counts.
Usage: pr_sheet_postprocess.py --sheet PRg1hop1=stage1.tsv:report.tsv [--sheet ...] [--pre-points PRg1hop1_export.tsv]"""
import argparse
import os
import re
import time
from collections import Counter, defaultdict

import pandas as pd

from faab.clients.cave import client as cave_client, live_outputs, parse_xyz, throttle
from faab.config import FIGURES, TMP
from faab.partners.sheet import as_int, fmt_vox, norm_xyz, parse_sheet, utc_stamp

RANK = {"checked": 0, "proofread": 1, "adequate": 2, "review": 3, "in progress": 4, "NEW": 5, "stopped": 6, "": 7, "NEW-DUP": 8}
TRACKED = ("status", "pr_date", "proofreader", "chk_date", "checker", "annotation", "notes")
TITLE2 = re.compile(r"^hop-1 (\d{15,}) \(now (\d{15,})\)")
TITLE1 = re.compile(r"^(\d{15,})\s+\[")


def clauses(s):
    return [x.strip() for x in re.split(r";\s*", s or "") if x.strip()]


def join_unique(*texts):
    out = []
    for t in texts:
        for c in clauses(t):
            if c.lower() not in {o.lower() for o in out}:
                out.append(c)
    return "; ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", action="append", required=True, metavar="TAB=STAGE1.tsv:REPORT.tsv")
    ap.add_argument("--out-dir", default=FIGURES)
    a = ap.parse_args()
    stamp = utc_stamp()

    # ---- load every sheet: lines, rows (dicts with cells), cell identity from the report
    sheets = {}
    for spec in a.sheet:
        tab, rest = spec.split("=", 1)
        s1, rep = rest.split(":", 1)
        lines, rows = parse_sheet(s1)
        R = pd.read_csv(rep, sep="\t", dtype=str).fillna("")
        key_col = "hop1_root_id" if "hop1_root_id" in rows[0]["header"] else rows[0]["header"][0]
        rep_by = {}
        for r in R.itertuples():
            rep_by[(r.block_pr, r.root_id_orig, norm_xyz(r.xyz) or "")] = r
        for r in rows:
            c = r["cells"]
            r["tab"], r["block"] = tab, c[key_col].strip()
            k = (r["block"], c["root_id_orig"].strip(), norm_xyz(c["xyz"]) or "")
            rr = rep_by.get(k)
            r["rep"] = rr
            r["cell"] = (rr.cell if rr is not None and rr.cell.isdigit() and rr.cell != "0" else c["root_id_curr"].strip())
            r["verdict"] = rr.verdict if rr is not None else "added"
            r["role"] = rr.role if rr is not None else "main"
            r["flag"], r["insert"] = "", None
        sheets[tab] = dict(lines=[ln.split("\t") for ln in lines], rows=rows, header=rows[0]["header"], key_col=key_col)
        print(f"{tab}: {len(rows)} rows ({sum(r['verdict'] == 'added' for r in rows)} added by the writer, "
              f"{sum(r['verdict'] == 'MISMATCH' for r in rows)} MISMATCH)", flush=True)
    allrows = [r for s in sheets.values() for r in s["rows"]]

    # ---- spelling: canonical names (most common spelling of each name), misplaced sentences -> notes
    names = Counter()
    for r in allrows:
        for col in ("proofreader", "checker"):
            for n in clauses(r["cells"].get(col, "")):
                names[n] += 1
    canon = {}
    for key, grp in pd.Series(list(names)).groupby(lambda i: re.sub(r"[\s.]", "", list(names)[i]).lower()):
        best = max(grp, key=lambda n: names[n])
        for n in grp:
            canon[n] = best
    moved = 0
    for r in allrows:
        c = r["cells"]
        for col in ("proofreader", "checker"):
            parts, keep = clauses(c.get(col, "")), []
            for p in parts:
                if len(p.split()) > 3:  # a sentence, not a name
                    c["notes"] = join_unique(c.get("notes", ""), p)
                    moved += 1
                else:
                    keep.append(canon.get(p, p))
            c[col] = "; ".join(dict.fromkeys(keep))
        st = c.get("status", "").strip()
        c["status"] = st if st.upper() in ("NEW", "NEW-DUP") else st.lower()
        if st.upper() in ("NEW", "NEW-DUP"):
            c["status"] = st.upper()
    print(f"names canonicalized: {sorted({(k, v) for k, v in canon.items() if k != v})}; misplaced notes moved: {moved}", flush=True)

    # ---- MISMATCH rows: flag, reset root_id_curr to the root at the xyz, insert a row for the hand-set id when it has synapses
    mism = [r for r in allrows if r["verdict"] == "MISMATCH"]
    if mism:
        client = cave_client()
        t = throttle(0)
        pre_now = {}  # block key -> current presynaptic root, from the refreshed block titles
        for s in sheets.values():
            for ln in s["lines"]:
                m2 = TITLE2.match(ln[0])
                if m2:
                    pre_now[m2.group(1)] = int(m2.group(2))
        hand = sorted({r["cells"]["root_id_curr"].strip() for r in mism})
        pres = sorted({pre_now[r["block"]] for r in mism if r["block"] in pre_now})
        t = throttle(t)
        syn, ts = live_outputs(pres, client)
        counts = syn.groupby(["pre_pt_root_id", "post_pt_root_id"]).size()
        tot = syn.groupby("pre_pt_root_id").size()
        conv_all = syn.groupby("post_pt_root_id").pre_pt_root_id.nunique()
        from faab.partners.points import cell_points
        t = throttle(t)
        pts, _ = cell_points({int(h) for h in hand}, client, time.time())
        for r in mism:
            c, rep = r["cells"], r["rep"]
            handset, at = c["root_id_curr"].strip(), rep.cell
            pre = pre_now.get(r["block"])
            n_hand = int(counts.get((pre, int(handset)), 0)) if pre else 0
            same_block = [x for x in sheets[r["tab"]]["rows"] if x["block"] == r["block"] and x["cell"] == handset and x is not r]
            c["root_id_curr"] = at
            r["flag"] = "mismatch"
            if same_block:
                c["notes"] = join_unique(c["notes"], f"root_id_curr was hand-set to {handset} but the xyz lies on {at} (still current); "
                                                     f"{handset} is the cell of row root_id_orig {same_block[0]['cells']['root_id_orig']} "
                                                     f"in this block: merged, or mis-edit?")
                same_block[0]["flag"] = "mismatch-partner"
                same_block[0]["cells"]["notes"] = join_unique(same_block[0]["cells"]["notes"],
                                                              f"see flagged row root_id_orig {c['root_id_orig']}: hand-set to this cell")
            else:
                c["notes"] = join_unique(c["notes"], f"split and xyz not updated? root_id_curr was hand-set to {handset} but the xyz "
                                                     f"lies on {at} (still current); row for {handset} added below")
                h = sheets[r["tab"]]["header"]
                new = {k: (c.get(k, "") if k not in ("root_id_orig", "root_id_curr", "xyz", "nsyn", "syn_pct", "h1_conv",
                                                     "PR_conv", "notes") else "") for k in h if k}
                new["root_id_orig"] = new["root_id_curr"] = handset
                new["xyz"] = pts.get(int(handset), ("", ""))[0]
                new["nsyn"] = str(n_hand)
                new["syn_pct"] = f"{round(100 * n_hand / tot.get(pre, 1), 1):g}" if pre else ""
                for cv in ("h1_conv", "PR_conv"):
                    if cv in new:
                        new[cv] = str(int(conv_all.get(int(handset), 0)))
                new["notes"] = (f"split and xyz not updated? added from the hand-set root_id_curr of the row above "
                                f"(root_id_orig {c['root_id_orig']}); xyz = a point on {handset}; {n_hand} syn from the block's "
                                f"presynaptic cell now")
                r["insert"] = dict(tab=r["tab"], block=r["block"], cells=new, cell=handset, verdict="inserted", role="main",
                                   flag="mismatch-new", rep=None, insert=None, header=h)
        print(f"MISMATCH rows handled: {len(mism)}; inserted rows: {sum(1 for r in mism if r['insert'])} "
              f"(live counts at {ts:%Y-%m-%d %H:%M} UTC)", flush=True)
    def rows_of(s):  # the sheet's rows in output order: each original row followed by its inserted row, if any
        for r in s["rows"]:
            yield r
            if r["insert"]:
                yield r["insert"]

    allrows = [r for s in sheets.values() for r in rows_of(s)]

    # ---- same cell within a block: the largest-nsyn row is the record, the rest NEW-DUP
    n_dup = 0
    for s in sheets.values():
        groups = defaultdict(list)
        for r in rows_of(s):
            if r["cell"].isdigit() and r["role"] != "is a PR of this group" and not r["flag"]:
                groups[(r["block"], r["cell"])].append(r)
        for (_, cell), g in groups.items():
            if len(g) < 2:
                continue
            g.sort(key=lambda r: -as_int(r["cells"]["nsyn"], 0))
            main = g[0]
            mc = main["cells"]
            # the record inherits the best status and every name of its duplicates (David: two names are fine)
            real = [r for r in g if r["cells"]["status"] not in ("", "NEW", "NEW-DUP")]
            if real:
                mc["status"] = min(real, key=lambda r: RANK.get(r["cells"]["status"], 7))["cells"]["status"]
            mc["proofreader"] = "; ".join(dict.fromkeys(n for r in g for n in clauses(r["cells"]["proofreader"])))
            mc["checker"] = "; ".join(dict.fromkeys(n for r in g for n in clauses(r["cells"]["checker"])))
            mc["pr_date"] = mc["pr_date"] or next((r["cells"]["pr_date"] for r in g if r["cells"]["pr_date"]), "")
            mc["chk_date"] = mc["chk_date"] or next((r["cells"]["chk_date"] for r in g if r["cells"]["chk_date"]), "")
            mc["annotation"] = join_unique(*(r["cells"]["annotation"] for r in g))
            mc["notes"] = join_unique(*(r["cells"]["notes"] for r in g))
            for r in g[1:]:
                c = r["cells"]
                was = f"{c['status'] or '-'}" + (f" ({c['proofreader']})" if c["proofreader"] else "")
                c["notes"] = join_unique(c["notes"], f"duplicate target of {mc['root_id_orig']} (now {cell}); was {was}")
                c["status"] = "NEW-DUP"
                r["dup"] = True
                n_dup += 1
    print(f"within-block duplicates set to NEW-DUP: {n_dup}", flush=True)

    # ---- same cell across blocks / sheets: unify the tracking columns
    by_cell = defaultdict(list)
    for r in allrows:
        if r["cell"].isdigit() and not r.get("dup") and not r["flag"] and r["role"] != "is a PR of this group":
            by_cell[r["cell"]].append(r)
    n_uni, merged_notes = 0, 0
    for cell, g in by_cell.items():
        if len(g) < 2:
            continue
        g.sort(key=lambda r: (RANK.get(r["cells"]["status"], 7), -as_int(r["cells"]["nsyn"], 0)))
        lead = g[0]["cells"]
        status = lead["status"]
        names_p = list(dict.fromkeys(n for r in g for n in clauses(r["cells"]["proofreader"])))
        names_c = list(dict.fromkeys(n for r in g for n in clauses(r["cells"]["checker"])))
        pr_date = next((r["cells"]["pr_date"] for r in g if r["cells"]["pr_date"]), "")
        chk_date = next((r["cells"]["chk_date"] for r in g if r["cells"]["chk_date"]), "")
        annotation = join_unique(*(r["cells"]["annotation"] for r in g))
        notes = join_unique(*(r["cells"]["notes"] for r in g))
        for r in g:
            c = r["cells"]
            before = dict(c)
            if c["status"] == "stopped" and status not in ("stopped", "", "NEW"):
                notes = join_unique(notes, f"fragment merged into {cell}")
            c.update(status=status, proofreader="; ".join(names_p), checker="; ".join(names_c), pr_date=c["pr_date"] or pr_date,
                     chk_date=c["chk_date"] or chk_date, annotation=annotation, notes=notes)
            if any(before[k] != c[k] for k in TRACKED):
                n_uni += 1
    print(f"rows changed by cross-block unification: {n_uni} (cells with several rows: {sum(len(g) > 1 for g in by_cell.values())})", flush=True)

    # ---- write each sheet: data lines rebuilt from the row dicts (an inserted row right after its parent), other lines verbatim
    flagged = []
    for tab, s in sheets.items():
        h = s["header"]
        width = max(len(ln) for ln in s["lines"])
        by_line = {r["line"]: r for r in s["rows"]}
        out, hl = [], []
        for i, ln in enumerate(s["lines"]):
            r = by_line.get(i)
            if r is None:
                out.append("\t".join(ln + [""] * (width - len(ln))))
                continue
            for rr in (r, r["insert"]):
                if rr is None:
                    continue
                c = rr["cells"]
                out.append("\t".join([c.get(x, "") for x in h] + [""] * (width - len(h))))
                if rr["flag"]:
                    hl.append(len(out))
                    flagged.append((tab, len(out), c["root_id_orig"], rr["flag"], c["status"], c["proofreader"], c["notes"]))
        while out and not out[-1].strip():  # one trailing newline only, like the sheet export
            out.pop()
        path = os.path.join(a.out_dir, f"{stamp}_FAABopticlobetargets-{tab}.tsv")
        open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")
        open(os.path.join(TMP, f"highlight_{tab}.txt"), "w").write("\n".join(map(str, hl)) + "\n")
        print(f"wrote {path} ({len(out)} lines, {len(hl)} flagged rows: {hl})", flush=True)
    print("\nflagged rows (sheet, line in the new file, root_id_orig, kind, status / proofreader, note):")
    for tab, n, orig, kind, st, who, notes in flagged:
        print(f"  {tab} line {n}: {orig} [{kind}] {st} / {who}: {notes[:160]}")


if __name__ == "__main__":
    main()
