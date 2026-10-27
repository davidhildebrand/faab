"""Tidy FAAB/tmp: keep only what the repo code reads from tmp, archive everything else (never delete).

KEEP (stays in tmp/): every tmp entry whose name appears in the repo code (faab/src, faab/scripts: e.g.
pr_groups_pasted.tsv, nuclei_v1_aedes_v542.tsv, paths_cache, t4t5_knn.tsv), the newest *_Backup.xlsx workbook,
archive/, dotfiles, and any --keep NAME. Everything else moves to tmp/archive/<YYYYMMDD>[<suffix>]/{scripts,logs,
sheets,data,dirs}/ with a MANIFEST.tsv (name -> new path). With --fix-refs, references 'tmp/<name>' in the given text
files (memory notes, TODO.md) are rewritten to the archived path, so notes keep pointing at the files.
--promote OLD=NEW (repeatable) first moves a tmp file into the repo (e.g. a reusable tool) and rewrites its references.
Default is a dry run; --apply moves. Usage:
  venv/bin/python scripts/tools/tidy_tmp.py [--apply] [--keep NAME ...] [--fix-refs FILE ...] [--promote tmp=repo ...]"""
import argparse
import datetime
import glob
import os
import re
import shutil

FAAB = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
REPO = os.path.join(FAAB, "faab")
TMP = os.path.join(FAAB, "tmp")


def code_text():
    parts = []
    for d in ("src", "scripts"):
        for p in glob.glob(os.path.join(REPO, d, "**", "*.py"), recursive=True):
            parts.append(open(p, encoding="utf-8", errors="ignore").read())
    return "\n".join(parts)


def kind(name, path):
    if os.path.isdir(path):
        return "dirs"
    ext = os.path.splitext(name)[1].lower()
    if ext == ".py":
        return "scripts"
    if ext in (".log", ".err"):
        return "logs"
    if ext == ".xlsx" or "opticlobetargets" in name.lower():
        return "sheets"
    return "data"


def rewrite_refs(files, mapping, apply):
    """'tmp/<old name>' -> new path (relative to FAAB) in each text file"""
    for f in files:
        s = open(f, encoding="utf-8").read()
        t = s
        for old, new in mapping.items():
            t = re.sub(r"(?<![\w/])tmp/" + re.escape(old) + r"(?![\w]|\.\w)", new, t)  # a sentence-ending '.' is fine
        if t != s:
            n = sum(1 for _ in re.finditer("|".join(re.escape("tmp/" + o) for o in mapping), s))
            print(f"  refs in {os.path.basename(f)}: {n} rewritten" + ("" if apply else " (dry run)"))
            if apply:
                open(f, "w", encoding="utf-8").write(t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--keep", nargs="*", default=[])
    ap.add_argument("--fix-refs", nargs="*", default=[])
    ap.add_argument("--promote", nargs="*", default=[], metavar="TMPNAME=REPO_RELATIVE_PATH")
    ap.add_argument("--suffix", default="", help="archive folder suffix, e.g. 'b' for a second tidy on one day")
    ap.add_argument("--manifest", nargs="*", default=[], help="earlier archive MANIFEST.tsv files whose moves should also "
                    "be rewritten in --fix-refs files")
    a = ap.parse_args()
    files = []
    for f in a.fix_refs:  # a folder means every .md file in it (memory notes)
        files += sorted(glob.glob(os.path.join(f, "*.md"))) if os.path.isdir(f) else [f]
    a.fix_refs = files
    refs = {}
    for mf in a.manifest:
        for line in open(mf, encoding="utf-8"):
            name, rel = line.rstrip("\n").split("\t")
            if not os.path.exists(os.path.join(TMP, name)):
                refs[name] = "tmp/" + rel
    for spec in a.promote:
        old, new = spec.split("=", 1)
        src, dst = os.path.join(TMP, old), os.path.join(REPO, new)
        if not os.path.exists(src):
            print(f"promote: {old} not in tmp, skipped")
            continue
        print(f"promote: tmp/{old} -> faab/{new}")
        if a.apply:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
        refs[old] = "faab/" + new
    code = code_text()
    backups = sorted(glob.glob(os.path.join(TMP, "*_Backup.xlsx")))
    keep = {"archive"} | set(a.keep) | ({os.path.basename(backups[-1])} if backups else set())
    keep |= {n for n in os.listdir(TMP) if n in code}
    dest = os.path.join(TMP, "archive", datetime.date.today().strftime("%Y%m%d") + a.suffix)
    moves = [(n, kind(n, os.path.join(TMP, n))) for n in sorted(os.listdir(TMP))
             if n not in keep and not n.startswith(".") and n not in refs]
    print(f"tmp: {len(os.listdir(TMP))} entries; keep {sorted(k for k in keep if os.path.exists(os.path.join(TMP, k)))}")
    print(f"archive -> {os.path.relpath(dest, FAAB)}: {len(moves)} entries "
          f"{ {k: sum(1 for _, x in moves if x == k) for k in sorted({x for _, x in moves})} }")
    for n, k in moves:
        refs[n] = os.path.relpath(os.path.join(dest, k, n), FAAB)
    if a.fix_refs:
        rewrite_refs(a.fix_refs, refs, a.apply)
    if not a.apply:
        print("dry run; rerun with --apply")
        return
    if moves:
        os.makedirs(dest, exist_ok=True)
        with open(os.path.join(dest, "MANIFEST.tsv"), "a", encoding="utf-8") as fh:
            for n, k in moves:
                os.makedirs(os.path.join(dest, k), exist_ok=True)
                shutil.move(os.path.join(TMP, n), os.path.join(dest, k, n))
                fh.write(f"{n}\t{os.path.relpath(os.path.join(dest, k, n), TMP)}\n")
    print(f"done; tmp now holds {len(os.listdir(TMP))} entries")


if __name__ == "__main__":
    main()
