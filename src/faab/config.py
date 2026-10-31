"""Paths and constants shared by the faab package and the scripts.

REPO is the repository root, so scripts write to REPO/figures and read the project scratch folder REPO/../tmp no matter
where the script file lives. Files in figures/ live in topic subfolders: always build their paths with fig(name).
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../faab (src/faab/config.py)
FIGURES = os.path.join(REPO, "figures")   # outputs and caches (gitignored)
DATA = os.path.join(REPO, "data")         # small versioned inputs
TMP = os.path.normpath(os.path.join(REPO, "..", "tmp"))  # project scratch outside the repo (pasted sheets etc.)

DATASTACK = "wclee_aedes_brain"           # FAAB CAVE datastack (read-only use)
VOX = (16.0, 16.0, 45.0)                  # nm per voxel of sheet / Spelunker xyz

# figures/ is sorted into topic subfolders (David, 2026-10-30: too many files at the top level). The topic comes from
# the file name alone, so a writer and every reader agree; first match wins; stamps are ignored.
FIG_TOPICS = [
    ("sheets", r".*(FAABopticlobetargets|pr_sheet_check|seatable|faab_pr_top|partner_glia|cells_xyz|soma_overrides|no_soma_check).*"),
    ("cutouts", r"cutout.*"),
    ("t4t5", r".*(t4t5|type_match|type_copies|tm546|fragment_search).*"),
    ("paths", r".*(paths_|pr_to_DN|dnp01|lplc2|vpn_|ct1_|seed_|class_paths|state_4684|objects_only|pr777).*"),
    ("metu", r".*(metu|MeTu|ids_LC|bridge_|fly_metu|dm2_|mcns_|fafb_|mirror_|faab_convergence|photoreceptor_).*"),
    ("mi1", r"mi1_.*"),
    ("glia", r".*(glia|nuclei|skel_cache|l2_chunkdata|l2_leaves|dp_cache|fragment_skeletons|candidate_dotprops).*"),
    ("pr_groups", r".*(pr_|faab_pr_groups|lamina|r7_outputs|explain_slab|template_pairwise|align|flip_|cell_\d+).*"),
]


def fig_topic(name):
    """topic subfolder of a figures file name (or glob pattern); 'misc' when no rule matches"""
    import re
    base = re.sub(r"^\d{8}d(?:\d{6}tUTC)?_?", "", name)
    return next((t for t, rx in FIG_TOPICS if re.fullmatch(rx, base) or re.fullmatch(rx, name)), "misc")


def fig(name):
    """path of a figures file: figures/<topic>/<name> (the folder is created). Also for a base name that gets its
    extension later (fig(f"{stamp}_x") + ".tsv"). Existing folders (ol_meshes, lit, refine, archive), names containing
    a '/' and extension-less names of no topic stay relative to figures/ itself. Globs work too:
    glob.glob(fig("*_seatable_match.tsv"))."""
    name = str(name)
    t = fig_topic(name)
    if "/" in name or (not os.path.splitext(name)[1] and (os.path.isdir(os.path.join(FIGURES, name)) or t == "misc")):
        return os.path.join(FIGURES, name)
    d = os.path.join(FIGURES, t)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)
