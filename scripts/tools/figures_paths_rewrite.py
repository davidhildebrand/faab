"""Rewrite the code's figures/ paths to go through faab.config.fig (topic subfolders; David, 2026-10-30).

For every .py in src/, scripts/, scratch/ (not this tool, tidy_figures.py, organize_figures.py): calls
os.path.join(B, x) with exactly two arguments, where B is a module-level name bound to the figures folder (OUT, FIG,
FIGURES, ...: assigned from FIGURES or from a join ending in "figures"), become fig(x); 'from faab.config import fig'
is added when missing. Calls with more arguments (folders such as ol_meshes, lit, refine), and other spellings of a
figures path (string concatenation, f"{OUT}/...", literal "figures/..." outside docstrings) are listed for hand edits.
Dry run unless --apply. Usage: venv/bin/python faab/scripts/tools/figures_paths_rewrite.py [--apply]"""
import argparse
import ast
import glob
import os
import re

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SKIP = {"figures_paths_rewrite.py", "tidy_figures.py", "organize_figures.py", "config.py"}  # config.py defines fig()


def figure_bases(tree, src):
    """module-level names bound to the figures folder: FIGURES from faab.config, OUT / FIG imported from a faab module
    (every faab module's OUT is the figures folder), and names assigned from FIGURES or a join ending in "figures"
    (also inside a conditional expression)"""
    names = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("faab"):
            for a in node.names:
                if a.name in ("FIGURES", "OUT", "FIG"):
                    names.add(a.asname or a.name)
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            seg = ast.get_source_segment(src, node.value) or ""
            if re.fullmatch(r"\s*FIGURES\s*", seg) or re.search(r"""os\.path\.join\([^()]*["']figures["']\s*\)""", seg):
                names.add(node.targets[0].id)
    return names


def plan(path):
    src = open(path, encoding="utf-8").read()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return src, [], [f"syntax error: {e}"]
    bases = figure_bases(tree, src)
    edits, notes = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.get_source_segment(src, node.func) in ("os.path.join", "join"):
            if node.args and isinstance(node.args[0], ast.Name) and node.args[0].id in bases:
                if len(node.args) == 2 and not node.keywords:
                    arg = ast.get_source_segment(src, node.args[1])
                    edits.append((node.lineno, node.col_offset, node.end_lineno, node.end_col_offset, f"fig({arg})"))
                else:
                    notes.append(f"line {node.lineno}: join with {len(node.args)} args: {ast.get_source_segment(src, node)[:90]}")
    doc_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) and isinstance(node.value.value, str):
            doc_lines |= set(range(node.lineno, node.end_lineno + 1))
    for i, line in enumerate(src.split("\n"), 1):
        if i in doc_lines or line.lstrip().startswith("#"):
            continue
        for b in bases:
            if re.search(rf"\{{{b}\}}/|\b{b}\s*\+|\+\s*{b}\b", line):
                notes.append(f"line {i}: {line.strip()[:100]}")
        if re.search(r"""["'][^"']*figures/[^"']*\.\w+""", line) and "fig(" not in line:
            notes.append(f"line {i}: literal figures path: {line.strip()[:100]}")
    return src, edits, notes


def apply_edits(src, edits):
    lines = src.split("\n")
    for l0, c0, l1, c1, rep in sorted(edits, reverse=True):
        if l0 == l1:
            s = lines[l0 - 1]
            lines[l0 - 1] = s[:c0] + rep + s[c1:]
        else:
            lines[l0 - 1] = lines[l0 - 1][:c0] + rep + lines[l1 - 1][c1:]
            del lines[l0:l1]
    out = "\n".join(lines)
    if not re.search(r"^from faab\.config import[^\n]*\bfig\b", out, re.M):
        tree = ast.parse(out)
        imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        at = max(n.end_lineno for n in imports) if imports else 0
        ls = out.split("\n")
        ls.insert(at, "from faab.config import fig")
        out = "\n".join(ls)
    ast.parse(out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    n_files = n_edits = 0
    for path in sorted(glob.glob(os.path.join(REPO, "*", "**", "*.py"), recursive=True)):
        rel = os.path.relpath(path, REPO)
        if os.path.basename(path) in SKIP or not rel.split(os.sep)[0] in ("src", "scripts", "scratch"):
            continue
        src, edits, notes = plan(path)
        if not edits and not notes:
            continue
        print(f"{rel}: {len(edits)} joins -> fig()" + "".join(f"\n    {n}" for n in notes))
        if edits:
            n_files, n_edits = n_files + 1, n_edits + len(edits)
            if a.apply:
                open(path, "w", encoding="utf-8").write(apply_edits(src, edits))
    print(f"\n{n_edits} joins in {n_files} files" + ("" if a.apply else " (dry run; rerun with --apply)"))


if __name__ == "__main__":
    main()
