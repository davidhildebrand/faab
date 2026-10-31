"""Render figures/faab_pr_top10_xyz.tsv as the fixed-width ASCII table used before,
with the tracking-sheet headers:

  upstream_PR class root_id_orig root_id_curr xyz n_syn pct_of_outp PR_conv annotation
  status pr_date proofreader chk_date checker notes

root_id_orig = the cached id from the 20 Aug connectivity query (row key);
root_id_curr = the current (major) root used for the xyz point - equals root_id_orig
               when the root has not changed.
Row order is forced to match the ORIGINAL top-10 table (figures/faab_pr_top10_original.txt)
rank-for-rank. Writes the ASCII block (faab_pr_top10_xyz_ascii.txt) and a tab-separated
copy with the same headers (faab_pr_top10_sheet.tsv). Local only - no CAVE calls.
"""
from faab.config import REPO
import os
import re
import sys
import pandas as pd
from faab.config import fig

OUT = os.path.join(REPO, "figures")
SRC = fig("faab_pr_top10_xyz.tsv")          # internal working table
ORIG = fig("faab_pr_top10_original.txt")
DST = fig("faab_pr_top10_xyz_ascii.txt")
SHEET = fig("faab_pr_top10_sheet.tsv")

HEADERS = ["upstream_PR", "class", "root_id_orig", "root_id_curr", "xyz", "n_syn",
           "pct_of_outp", "PR_conv", "annotation", "status", "pr_date", "proofreader",
           "chk_date", "checker", "notes"]
ALIGNS = ["<", "<", "<", "<", "<", ">", ">", ">", "<", "<", "<", "<", "<", "<", "<"]

SHORT = {
    "single L2 chunk fragment -> L2 rep coord (no soma/backbone)": "L2 fragment rep coord",
    "no nucleus -> backbone midpoint": "backbone midpoint (no nucleus)",
}


def table(headers, rows, aligns=None):
    cols = len(headers)
    aligns = aligns or ["<"] + [">"] * (cols - 1)
    data = [[str(h) for h in headers]] + [[str(c) for c in r] for r in rows]
    w = [max(len(row[i]) for row in data) for i in range(cols)]
    sep = "-+-".join("-" * x for x in w)
    out = [" | ".join(f"{headers[i]:{aligns[i]}{w[i]}}" for i in range(cols)), sep]
    for r in rows:
        out.append(" | ".join(f"{str(r[i]):{aligns[i]}{w[i]}}" for i in range(cols)))
    return "\n".join(out)


def parse_original(path):
    """{cell: [partner ids in original rank order]}"""
    out, cell = {}, None
    for line in open(path):
        m = re.search(r"=== (\d{15,})", line)
        if m:
            cell = m.group(1)
            out[cell] = []
            continue
        m = re.match(r"\s*\d+\s*\|\s*(\d{15,})\s*\|", line)
        if m and cell is not None:
            out[cell].append(m.group(1))
    return out


def split_notes(note, orig):
    """-> (root_id_curr, short_note)"""
    note = "" if pd.isna(note) else str(note)
    m = re.match(r"root changed; xyz from current root (\d+)(?: \(\+(\d+) more\))?; (.*)", note)
    if m:
        cur, rest = m.group(1), m.group(3)
    else:
        cur, rest = orig, note
    rest = re.sub(r"split into \d+ comparable pieces - this one carries all (\d+) PR synapses, "
                  r"other: (\d+) \((\d+) syn, \d+ chunks\)",
                  r"synapse-bearing half of a split (\1 PR syn; other half \2 has \3)", rest)
    parts = [SHORT.get(p, p.replace(" from soma", "")) for p in rest.split("; ")]
    return cur, "; ".join(parts)


def cell(v):
    return "" if pd.isna(v) else str(v)


def main():
    df = pd.read_csv(SRC, sep="\t", dtype=str)
    if "#" in df.columns:
        df = df.drop(columns=["#"])
    orig = parse_original(ORIG)

    # ---- force original order, rank-for-rank ----
    ordered = []
    for pr, ids in orig.items():
        sub = df[df.upstream_PR == pr]
        for rank, pid in enumerate(ids, 1):
            hit = sub[sub.downstream == pid]
            if len(hit) != 1:
                sys.exit(f"ERROR: PR {pr} partner {pid} found {len(hit)} times in TSV")
            row = hit.iloc[0].to_dict()
            row["#"] = rank
            ordered.append(row)
    if len(ordered) != len(df):
        sys.exit(f"ERROR: {len(ordered)} ordered rows vs {len(df)} TSV rows")
    df = pd.DataFrame(ordered)[["#"] + [c for c in df.columns]]
    # xyz as "x, y, z" (with spaces) so spreadsheets keep it as text, not one number
    df["xyz"] = df["xyz"].map(lambda v: re.sub(r",\s*", ", ", str(v)) if pd.notna(v) else "")
    df.to_csv(SRC, sep="\t", index=False)

    # ---- verify against the original, rank by rank ----
    n_ok = 0
    for pr, ids in orig.items():
        got = list(df.loc[df.upstream_PR == pr, "downstream"])
        n_ok += sum(a == b for a, b in zip(ids, got))
    print(f"VERIFY: {n_ok}/{sum(len(v) for v in orig.values())} rows match the original "
          f"table rank-for-rank\n")

    rows = []
    for _, r in df.iterrows():
        cur, note = split_notes(r["notes"], r["downstream"])
        rows.append([r["upstream_PR"], r["class"], r["downstream"], cur, r["xyz"],
                     r["n_syn"], r["pct_of_outp"], r["PR_conv"], cell(r["annotation"]),
                     cell(r["status"]), cell(r["pr_date"]), cell(r["proofreader"]),
                     "", "", note])
    txt = table(HEADERS, rows, ALIGNS)
    with open(DST, "w") as f:
        f.write(txt + "\n")
    pd.DataFrame(rows, columns=HEADERS).to_csv(SHEET, sep="\t", index=False)
    print(txt)


if __name__ == "__main__":
    main()
