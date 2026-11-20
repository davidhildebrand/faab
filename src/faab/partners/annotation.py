"""Sheet annotation vs SeaTable label (superclass / class / subclass / subsubclass / type), shared by the sheet sync
(scripts/sheets/seatable_sync.py) and the sheet checker (scripts/bot/faab_ol_bot.py).

Rule (David 2026-11-20): the sync never overwrites an annotation the sheet already has. It only fills a BLANK
annotation with SeaTable's label. Every other difference is left for the annotators to clarify: real conflicts are
asked about (messages, the bot's 'annotation' check), and nothing is replaced automatically.

kind(sheet, label) ->
  'same'          the same text (spaces around '/' and case ignored)
  'sheet blank'   the sheet has no annotation, SeaTable has a label (the only case the sync writes)
  'no label'      SeaTable has no label
  'labels differ' the cell has several SeaTable rows with different labels
  'coarser'       the sheet holds an older, coarser copy of SeaTable's label ('ol_intrinsic / LMC / visual' while
                  SeaTable now says 'ol_intrinsic / LMC / visual / L4')
  'short form'    every word of the sheet's text is part of SeaTable's label ('Mi' for 'ol_intrinsic / visual / Mi')
  'finer'         the sheet names a type SeaTable lacks: SeaTable's label is generic ('Tm' vs 'ol_intrinsic /
                  visual') or the sheet names a subtype of SeaTable's class ('L2?' vs 'ol_intrinsic / LMC / visual')
  'conflict'      both name a type and they differ ('L5' vs 'ol_intrinsic / visual / Mi')
"""
import re

# label words that name no cell type (superclasses and the modality): a label made only of these is generic
GENERIC = {"ol_intrinsic", "visual", "ol_sensory", "olsn", "visual_projection", "visual_centrifugal", "cb_intrinsic",
           "central_tbc", "descending_neuron", "ascending_neuron", "sensory", "outer", "inner"}
# a SeaTable class and the start of its subtypes' names: a sheet annotation naming such a subtype is finer
SUBTYPES = {"lmc": r"l[1-5]", "l-like": r"l[1-5]", "tm": r"tm", "mi": r"mi", "t4": r"t4", "t5": r"t5", "dm": r"dm",
            "metu": r"metu"}
CONFLICT = "conflict"


def parts(x):
    return [p.strip() for p in str(x).split("/") if p.strip()]


def norm(x):
    return " / ".join(parts(x)).lower()


def word(w):
    """one label word for comparison: no spaces, no '-like' ('LO ME' = 'LOME', 'CT1' = 'CT1-like')"""
    return re.sub(r"-like$", "", w.lower().replace(" ", ""))


def label_of_mark(seatable_cell):
    """SeaTable's label from the sheet's SeaTable column ('present: <label>', 'added (AK): <label>'); '' when none"""
    m = re.match(r"\s*(?:added|present)[^:]*:\s*(.+)$", str(seatable_cell or ""), re.I)
    return m.group(1).strip() if m else ""


def kind(sheet, label):
    sheet, label = str(sheet or "").strip(), str(label or "").strip()
    if not label:
        return "no label"
    if ";" in label:
        return "labels differ"
    if not sheet:
        return "sheet blank"
    a, l = norm(sheet), norm(label)
    if a == l:
        return "same"
    if l.startswith(a):
        return "coarser"
    lp = [p.lower() for p in parts(label)]
    words = [w.strip().rstrip("?").lower() for w in re.split(r"[;/,]", sheet) if w.strip()]
    lpw = {word(p) for p in lp}
    if all(word(w) in lpw or word(re.sub(r"-[a-z]{2,3}$", "", w)) in lpw for w in words):
        return "short form"  # also 'LO ME, CT1' for 'LOME / CT1-like' and 'L5-ND' (initials) for '.. / L5'
    spec = [p for p in lp if p not in GENERIC]
    if not spec:
        return "finer"
    sub = SUBTYPES.get(spec[-1])
    if sub and all(re.match(sub, w) for w in words if w not in lp):
        return "finer"
    return CONFLICT


def fill_from_seatable(sheet, label):
    """the annotation the sync writes, or None to leave the cell alone: SeaTable's label for a blank annotation only"""
    return str(label).strip() if kind(sheet, label) == "sheet blank" else None
