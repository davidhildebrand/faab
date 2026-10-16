"""Render figures/faab_pr_top20_update.tsv as pipe/dash blocks per photoreceptor, with the
sheet's wording: xyz-method text removed, backbone-midpoint rows marked "no soma?" in the
followups column, "new partner" left bare. Rewrites the TSV / blocks files to match."""
from faab.config import REPO
import os
import re
import pandas as pd

OUT = os.path.join(REPO, "figures")
TSV = os.path.join(OUT, "faab_pr_top20_update.tsv")
BLOCKS = os.path.join(OUT, "faab_pr_top20_update_blocks.tsv")
ASCII = os.path.join(OUT, "faab_pr_top20_update_ascii.txt")
COLS = ["upstream_PR", "class", "root_id_orig", "root_id_curr", "xyz", "n_syn", "pct_of_outp", "PR_conv",
        "annotation", "status", "pr_date", "proofreader", "chk_date", "checker", "notes", "followups",
        "SeaTable", "n_syn_prev", "prevrank", "p_glia", "soma_check", "update"]  # p_glia: faab_glia_logistic.py; soma_check: faab_soma_column.py
SHEET = os.path.join(os.path.dirname(OUT), "..", "tmp", "sheet_partners.tsv")


def table(headers, rows):
    aligns = [">" if h in ("n_syn", "pct_of_outp", "PR_conv", "n_syn_prev") else "<" for h in headers]
    data = [[str(h) for h in headers]] + [[str(c) for c in r] for r in rows]
    w = [max(len(row[i]) for row in data) for i in range(len(headers))]
    out = [" | ".join(f"{headers[i]:{aligns[i]}{w[i]}}" for i in range(len(headers))),
           "-+-".join("-" * x for x in w)]
    for r in rows:
        out.append(" | ".join(f"{str(r[i]):{aligns[i]}{w[i]}}" for i in range(len(headers))))
    return "\n".join(out)


def reword(upd, followups):
    parts = [p.strip() for p in str(upd).split(";") if p.strip()]
    keep = []
    for p in parts:
        if p.startswith("no nucleus -> backbone midpoint") or p.startswith("trunk too short"):
            if not followups.strip():
                followups = "no soma?"
            continue
        if p.startswith("primary neurite") or p.startswith("new partner") or p.startswith("single L2 chunk fragment"):
            continue
        keep.append(p)
    return "; ".join(keep), followups


def verify(df):
    """every root_id_orig of the sheet must appear in its PR's block, as a row or in a merge note"""
    sheet = pd.read_csv(os.path.join(os.path.dirname(OUT), "..", "tmp", "sheet_partners.tsv"),
                        sep="\t", dtype=str).fillna("")
    n_ok, n_top, n_kept, missing = 0, 0, 0, []
    for _, s in sheet.iterrows():
        blk = df[df.upstream_PR == s.upstream_PR]
        direct = blk[blk.root_id_orig == s.root_id_orig]
        merged = blk[blk["update"].str.contains(s.root_id_orig, regex=False)]
        if len(direct):
            n_ok += 1
            if str(direct.iloc[0]["update"]).startswith("no synapses") or "dropped" in str(direct.iloc[0]["update"]):
                n_kept += 1
            else:
                n_top += 1
        elif len(merged):
            n_ok += 1
            n_top += 1
        else:
            missing.append((s.upstream_PR, s.root_id_orig))
    print(f"VERIFY: {n_ok}/{len(sheet)} sheet rows accounted for in their own block "
          f"({n_top} in the top 20 incl. merges, {n_kept} kept below the top 20); missing: {missing}\n")


def main():
    df = pd.read_csv(TSV, sep="\t", dtype=str).fillna("")
    for c in COLS:  # p_glia is added by faab_glia_logistic.py; tolerate a table rendered before that step
        if c not in df.columns:
            df[c] = ""
    for i, r in df.iterrows():
        upd, fu = reword(r["update"], r["followups"])
        df.at[i, "update"], df.at[i, "followups"] = upd, fu
    # prevrank = position of the row's root_id_orig in the pasted sheet, within the same PR block
    sheet = pd.read_csv(SHEET, sep="\t", dtype=str).fillna("")
    sheet["prevrank"] = sheet.groupby("upstream_PR").cumcount() + 1
    prev = {(r.upstream_PR, r.root_id_orig): str(r.prevrank) for _, r in sheet.iterrows()}
    df["prevrank"] = [prev.get((r.upstream_PR, r.root_id_orig), "") for _, r in df.iterrows()]
    df.to_csv(TSV, sep="\t", index=False)
    verify(df)

    heads = {  # totals from the live query of 2026-09-18 14:24 UTC; overwritten when the blocks file has them
        "648518347649903570": ("R1–R6", "120", "82"), "648518347649905106": ("R1–R6", "158", "110"),
        "648518347642147360": ("R1–R6", "169", "123"), "648518347626884562": ("R1–R6", "142", "122"),
        "648518347643620818": ("R1–R6", "117", "97"), "648518347567256005": ("R1–R6", "185", "150"),
        "648518347777360890": ("R7", "32", "22"), "648518347626873298": ("R8", "54", "32")}
    live = "2026-09-18"
    meta_path = os.path.join(OUT, "faab_pr_top20_update_meta.json")
    if os.path.exists(meta_path):
        import json
        meta = json.load(open(meta_path))
        live = meta.get("live_utc", live) + " UTC"
        for pr, (tot, npart) in meta.get("totals", {}).items():
            cls = heads.get(pr, (df[df.upstream_PR == pr]["class"].iloc[0],))[0]
            heads[pr] = (cls, str(tot), str(npart))
    for line in open(BLOCKS):
        m = re.search(r"(\d{18})\s+\[([^\]]+)\]\s+(\d+) output synapses across (\d+) partners", line)
        if m and not os.path.exists(meta_path):
            heads[m.group(1)] = (m.group(2), m.group(3), m.group(4))
    txt, tsv = [], []
    for pr in df.upstream_PR.drop_duplicates():
        cls, tot, npart = heads.get(pr, (df[df.upstream_PR == pr]["class"].iloc[0], "?", "?"))
        sub = df[df.upstream_PR == pr]
        head = f"{pr}  [{cls}]  {tot} output synapses across {npart} partners (live {live})"
        txt += [head, table(COLS, sub[COLS].values.tolist()), ""]
        tsv += [head, "\t".join(COLS)] + ["\t".join(str(v) for v in row) for row in sub[COLS].values.tolist()] + [""]
    open(ASCII, "w").write("\n".join(txt))
    open(BLOCKS, "w").write("\n".join(tsv))
    print("\n".join(txt))


if __name__ == "__main__":
    main()
