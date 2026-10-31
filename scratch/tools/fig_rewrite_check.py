"""Network-free check after figures_paths_rewrite.py: every .py in src/, scripts/, scratch/ that uses fig( compiles,
imports fig, and does not use it at module level before the import line; faab.config.fig resolves sample names to the
expected topic folders. Usage: fig_rewrite_check.py"""
import ast
import glob
import os
import re

from faab.config import FIGURES, fig

REPO = os.path.dirname(FIGURES)
bad = 0
n = 0
for p in sorted(glob.glob(os.path.join(REPO, "*", "**", "*.py"), recursive=True)):
    rel = os.path.relpath(p, REPO)
    if rel.split(os.sep)[0] not in ("src", "scripts", "scratch"):
        continue
    src = open(p, encoding="utf-8").read()
    if not re.search(r"(?<![\w.])fig\(", src) or rel.endswith("config.py") or "/tools/" in rel:
        continue
    n += 1
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"SYNTAX {rel}: {e}")
        bad += 1
        continue
    imp = [node.lineno for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "faab.config"
           and any(a.name == "fig" for a in node.names)]
    if not imp:
        print(f"NO IMPORT {rel}")
        bad += 1
        continue
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id == "fig" and sub.lineno < imp[0]:
                print(f"USED BEFORE IMPORT {rel}: line {sub.lineno} (import at {imp[0]})")
                bad += 1
print(f"{n} files use fig(); problems: {bad}")
for name in ("pr_bundle_boxes.pkl", "pr_left_bundles_roots.tsv", "20261031d041835tUTC_FAABopticlobetargets-PRs.xlsx",
             "*_seatable_match.tsv", "ol_meshes", "lit", "20261031d_t4t5_query_match", "glia_logistic_model.json",
             "mi1_dotprops.pkl", "cutout_inventory.tsv", "pr_bundles_synapses_raw_{}.tsv"):
    print(f"  fig({name!r}) -> {os.path.relpath(fig(name), FIGURES)}")
