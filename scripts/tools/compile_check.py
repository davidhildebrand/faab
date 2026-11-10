"""Byte-compile the given .py files (a syntax check without running them; allowlist-friendly: no 'python -c').
Usage: venv/bin/python faab/scripts/tools/compile_check.py FILE [FILE ...]"""
import py_compile
import sys

for f in sys.argv[1:]:
    py_compile.compile(f, doraise=True)
    print("ok", f)
