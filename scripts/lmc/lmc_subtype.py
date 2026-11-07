"""LMC (L1-L5) subtyping of the FAAB Aedes optic lobes, end to end, in the order it was run on 2026-10-31 / 11-01
(David 2026-11-06: "describe clearly what was done for the LMC subtyping, and provide a script for this"). Runs the
step scripts (faab/scratch/lmc/*.py, faab/scripts/sheets/lmc_tab.py) one after another with this python; each step's
docstring has its details. Steps marked CAVE make read-only calls (>= 2 s apart inside each step; the steps run one at a
time, never concurrently); the others are network-free. Outputs go to faab/figures/lmc/, the log to FAAB/tmp/.

Method in brief
  typing set   SeaTable LMCs typed L1..L5 (2026-10-31: 69, all right eye) train; SeaTable 'L1-L5' / untyped LMCs and
               --extra cells are called
  features     per cell, from its L2 skeleton (1 um) and all its synapses: position in the lamina (LA) and medulla (ME)
               of its side as depth 0 (outer face) .. 1 (inner face), how far the cable reaches into ME and past LA's
               outer face, lamina spread, input / output counts per neuropil, inputs per output, synapse depths (37)
  models       LDA (shrinkage), kNN (k=5) and random forest; call = majority of the three (else the forest's top
               type), forest probability; leave-one-out on the typing set (2026-10-31: 97% each). NBLAST (rigid
               align 62%, centroid 51%) is reported beside the call but does not vote
  cartridges   one of each type per LRS cartridge (Hungarian assignment on the forest probabilities, types already
               typed in the cartridge excluded)
  search       more LMCs on both sides: cells with >= 3 synapses from R1-R6 / R8 (both end in the lamina in Aedes) with
               >= 20 LA inputs, >= 20 ME outputs and <= 800 LA inputs, typed with the same models plus checks
               (typical, complete, soma by the lamina); L5s (almost no photoreceptor input) as partners of L4s with
               the L5 synapse profile
  confidence   high = 3/3 models, probability >= 0.6, typical, complete, soma by the lamina; low = probability < 0.4,
               1/3 models, unlike the typed cells, or soma by the medulla (left out as not an LMC); medium = the rest.
               L5 search: high = tight L5 profile + soma by the lamina, medium = loose profile + soma by the lamina,
               low = no nucleus on the root. Left-side calls are flagged (trained on right-eye cells only)
  sheet        the LMCs tab (blocks L1..L5 + 'LMC, type unclear'; annotated cells first, then high / medium / low)

Usage (paths relative to faab/ or absolute):
  lmc_subtype.py --seatable ST.tsv --run 20261101 --fresh-seatable ST2.tsv --book BOOK.xlsx
                 [--extra ROOT:SIDE:NOTE ...] [--from STEP] [--to STEP] [--skip STEP ...] [--dry-run]
  --seatable        seatable_fetch.py TSV for the typing and the search (2026-11-01 run:
                    figures/sheets/20261031d212925tUTC_seatable_aedes_main.tsv)
  --run             PR grouping run folder figures/ommatidia/<RUN> (photoreceptor roots; 2026-11-01 run: 20261101)
  --fresh-seatable  a fresh fetch for the LRS check and the sheet (2026-11-01: 20261101d221232tUTC_seatable_aedes_main.tsv)
  --book            the latest workbook download (2026-11-01: FAAB/tmp/20261031d213400tUTC_synced_full.xlsx)
  --extra           cells to call that SeaTable does not list as LMCs (2026-10-31:
                    648518347544284097:R:PRs-group-19-LMC-per-David)
Steps: run with --dry-run to list them."""
import argparse
import datetime
import glob
import os
import subprocess
import sys
import time

FAAB = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LMC = os.path.join(FAAB, "scratch", "lmc")
TAB = os.path.join(FAAB, "scripts", "sheets", "lmc_tab.py")
TMP = os.path.join(os.path.dirname(FAAB), "tmp")


def newest_tab_tsv():
    found = sorted(glob.glob(os.path.join(FAAB, "figures", "lmc", "*_lmc_tab.tsv")))
    return found[-1] if found else "<the TSV written by tab-build>"


# (name, CAVE?, what it does, script, arguments from the options)
STEPS = [
    ("typing-data", True, "typing set: SeaTable LMCs (+ extras), L2 skeletons, synapses with positions",
     "lmc_data.py", lambda a: [a.seatable] + (["--extra"] + a.extra if a.extra else [])),
    ("typing-features", False, "lamina / medulla position and synapse features", "lmc_features.py", lambda a: []),
    ("typing-nblast", False, "NBLAST matrices (rigid align, centroid) - reported, not voted", "lmc_nblast.py",
     lambda a: []),
    ("typing-classify", False, "LDA / kNN / random forest, leave-one-out, calls -> lmc_typing.tsv", "lmc_classify.py",
     lambda a: []),
    ("typing-cartridges", False, "one of each type per LRS cartridge -> lmc_cartridge_calls.tsv", "lmc_cartridges.py",
     lambda a: []),
    ("search-candidates", True, "targets of R1-R6 / R8 with their lamina / medulla synapse counts", "lmc_candidates.py",
     lambda a: [a.seatable, a.run]),
    ("type-profiles", False, "synapse profile per typed type (sets the L5 search thresholds)", "lmc_type_profiles.py",
     lambda a: []),
    ("search-prep", True, "LMC-like candidates: skeletons and synapses", "lmc_cand_prep.py", lambda a: []),
    ("search-features", False, "the same features for the candidates", "lmc_features.py", lambda a: ["--set", "cand"]),
    ("nuclei", True, "nuclei mapped to today's roots (soma side)", "nuclei_roots.py", lambda a: []),
    ("soma-check", False, "soma side of the known LMCs (all by the lamina)", "soma_side.py", lambda a: []),
    ("search-classify", False, "candidate calls with typical / complete / soma checks -> lmc_cand_typing.tsv",
     "lmc_cand_classify.py", lambda a: []),
    ("l5-search", True, "partners of L4s with the L5 profile -> lmc_l5_candidates.tsv", "lmc_l5_search.py",
     lambda a: [a.seatable]),
    ("search-summary", False, "calls per side, type and confidence", "lmc_search_summary.py", lambda a: []),
    ("lrs-check", False, "LRS's cartridge LMCs in the fresh SeaTable fetch vs our calls", "lrs_check.py",
     lambda a: [a.fresh_seatable]),
    ("sheet-rows", False, "rows of the workbook already annotated as LMCs", "sheet_lmc_rows.py", lambda a: [a.book]),
    ("l5-skeletons", True, "skeletons of the new L5 candidates (for their sheet points)", "lmc_fetch_skeletons.py",
     lambda a: [os.path.join(FAAB, "figures", "lmc", "lmc_l5_candidates.tsv"), "--l5"]),
    ("tab-points", True, "points on the primary neurite of the model-only cells", TAB,
     lambda a: ["points", a.fresh_seatable, a.book]),
    ("tab-build", True, "the LMCs tab table and its block links -> <stamp>_lmc_tab.tsv", TAB,
     lambda a: ["build", a.fresh_seatable, a.book]),
    ("tab-xlsx", False, "one-tab .xlsx to import -> figures/sheets/<stamp>_FAABopticlobetargets-LMCs.xlsx", TAB,
     lambda a: ["xlsx", a.book, newest_tab_tsv()]),
]
NAMES = [s[0] for s in STEPS]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seatable", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--fresh-seatable", required=True)
    ap.add_argument("--book", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--from", dest="start", choices=NAMES, default=NAMES[0])
    ap.add_argument("--to", dest="stop", choices=NAMES, default=NAMES[-1])
    ap.add_argument("--skip", nargs="*", choices=NAMES, default=[])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    for k in ("seatable", "fresh_seatable", "book"):
        p = getattr(a, k)
        setattr(a, k, p if os.path.isabs(p) else os.path.join(FAAB, p))
    todo = STEPS[NAMES.index(a.start):NAMES.index(a.stop) + 1]
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dd%H%M%StUTC")
    log_path = os.path.join(TMP, f"{stamp}_lmc_subtype.log")
    log = None if a.dry_run else open(log_path, "w")
    t_last = 0.0
    for k, (name, cave, what, script, args) in enumerate(todo, 1):
        path = script if os.path.isabs(script) else os.path.join(LMC, script)
        cmd = [sys.executable, path] + args(a)
        head = f"[{k}/{len(todo)}] {name}{' (CAVE, read-only)' if cave else ''}: {what}"
        if name in a.skip:
            print(f"{head}\n  skipped", flush=True)
            continue
        print(f"{head}\n  {' '.join(cmd[1:])}", flush=True)
        if a.dry_run:
            continue
        if cave:  # one CAVE step at a time, >= 2 s after the previous one
            time.sleep(max(0.0, 2.0 - (time.time() - t_last)))
        log.write(f"\n===== {head}\n{' '.join(cmd)}\n")
        t0 = time.time()
        p = subprocess.Popen(cmd, cwd=FAAB, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print("  " + line, end="", flush=True)
            log.write(line)
        p.wait()
        t_last = time.time()
        log.write(f"----- exit {p.returncode}, {t_last - t0:.0f} s\n")
        log.flush()
        if p.returncode:
            print(f"step {name} failed (exit {p.returncode}); rerun from it with --from {name}. Log: {log_path}")
            sys.exit(p.returncode)
    if log:
        log.close()
        print(f"done; log {log_path}")


if __name__ == "__main__":
    main()
