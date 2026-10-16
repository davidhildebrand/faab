"""Paths and constants shared by the faab package and the scripts.

REPO is the repository root, so scripts write to REPO/figures and read the project scratch folder REPO/../tmp no matter
where the script file lives.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../faab (src/faab/config.py)
FIGURES = os.path.join(REPO, "figures")   # outputs and caches (gitignored)
DATA = os.path.join(REPO, "data")         # small versioned inputs
TMP = os.path.normpath(os.path.join(REPO, "..", "tmp"))  # project scratch outside the repo (pasted sheets etc.)

DATASTACK = "wclee_aedes_brain"           # FAAB CAVE datastack (read-only use)
VOX = (16.0, 16.0, 45.0)                  # nm per voxel of sheet / Spelunker xyz
