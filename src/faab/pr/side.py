"""Which eye a photoreceptor-group finder step works on: --side right|left (or --side=left), default right.

Right-side files keep their original names (figures/pr_*); left-side files are figures/pr_left_*. Step 1 differs per
eye (scripts/pr_groups/lattice.py fits the right lattice, scripts/pr_groups/left_sites.py places the left sites); the
census (src/faab/pr/bundles.py), clustering (bundles_cluster.py) and verification (bundles_verify.py) take --side.
"""
import sys

PREFIX = {"right": "pr_", "left": "pr_left_"}
KNOWN_NAME = {"right": "known PRs (8 groups + LRS cartridges)", "left": "known PRs (LRS left cartridges)"}
NEUROPILS = {"right": ("LA_R", "ME_R"), "left": ("LA_L", "ME_L")}  # lamina, medulla meshes (faab.ol_meshes)


def parse(argv=None):
    """(side, remaining args) from --side right|left or --side=right|left; default right"""
    args = list(sys.argv[1:] if argv is None else argv)
    side = "right"
    for i, a in enumerate(args):
        if a == "--side":
            if i + 1 >= len(args):
                raise SystemExit("--side needs a value: right or left")
            side = args[i + 1]
            del args[i:i + 2]
            break
        if a.startswith("--side="):
            side = a.split("=", 1)[1]
            del args[i]
            break
    if side not in PREFIX:
        raise SystemExit(f"--side must be right or left, not {side!r}")
    return side, args
