"""Add a subtable of the other inputs of a path tab's targets to that tab (first use 2026-10-31, David: every cell with
>= 3 synapses onto at least one of the 20 hop-3 T4/T5 candidates of PRg1toT4T5 that the tab does not list yet).

The tab is taken as it is in BOOK (proofreaders' edits, styles and links kept); the subtable goes below its last block,
above the 'note:' lines:
  targets  the tab's hop-3 rows: root_id_curr while it is current (even when the xyz lies off it), else the root at the
           xyz
  inputs   presynaptic cells with >= --thr synapses onto at least one target (local synapse snapshot + a CAVE catch-up
           to now, faab.synsnap; self-contacts removed), minus the targets and the cells the tab already lists
  columns  hop 'in'; T4T5_root_id = the targets the cell feeds with >= thr synapses, strongest first; nsyn = its
           synapses onto each of them, same order; nsyn_out = its synapses onto all the targets (weaker contacts too);
           T4T5_reached = how many targets it feeds with >= thr; cable_um / neuropils from the L2 skeleton
Tracking is carried from the other tabs (rows matched the same way; backup tabs '*_bYYMMDDd' skipped): the best status
(checked > proofread > adequate > review > in progress > stopped), names merged, dates, link and that row's cell styles
(yellow MISMATCH fill dropped), its xyz when the xyz lies on the cell; cells without a status are NEW (this tab is their
primary record, as on the paths). Cells without a usable xyz get one on the primary neurite near the soma (top10_xyz
recipe; no nucleus: the middle of the backbone; one-chunk fragments: their L2 chunk), verified to lie on the cell.
SeaTable (newest figures/sheets/*_seatable_aedes_main.tsv) as in seatable_sync.py: status = SeaTable's unless ours is
checked / proofread, SeaTable proofreaders added as SeaTable writes them, SeaTable column '<added|present>: <label>',
annotation = the sheet's own. Very large cells (>= --large-cable um of cable, or >= --large-outside of it outside the
right optic lobe) go to a LOW PRIORITY block right below, as on the paths. Rows: most targets fed first, then most
synapses. CAVE read-only, >= 5 s between calls. Writes figures/sheets/<stamp>_FAABopticlobetargets-<tab>.xlsx (the
whole tab; import via File > Import > Insert new sheet), the subtable as .tsv, and a short Spelunker link.
Usage: target_inputs_block.py --book BACKUP.xlsx [--tab PRg1toT4T5] [--thr 3]"""
import argparse
import copy
import datetime
import glob
import importlib.util
import os
import pickle
import re
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np

if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import openpyxl
import pandas as pd
from openpyxl.styles import Font, PatternFill

from faab import nglink, ol_meshes, synsnap
from faab.clients.cave import DATASTACK, client as cave_client, roots_at_xyz, throttle
from faab.config import TMP, fig
from faab.partners.sheet import cell_text as s, scan_book, seatable_label, utc_stamp
from faab.pr.prs_tab import DATE_FMT, as_date

_spec = importlib.util.spec_from_file_location(
    "seatable_sync", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sheets", "seatable_sync.py"))
SS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SS)

HL = SS.HL
RANK = ["checked", "proofread", "adequate", "review", "in progress", "stopped"]
STYLE_ALIAS = {"PR_root_id": "upstream_PR", "hop": "PR_root_id", "hop2_root_id": "hop1_root_id",
               "T4T5_root_id": "hop1_root_id", "T4T5-1_root_id": "hop1_root_id", "nsyn_out": "nsyn", "T4T5_reached": "h1_conv", "cable_um": "syn_pct",
               "neuropils": "notes", "tracked_in": "notes"}
OUTSIDE = ("CB", "OL_L", "LA_L", "ME_L", "LO_L", "LOP_L")
SKEL = fig("fragment_skeletons.pkl")
VOX = np.array([16.0, 16.0, 45.0])
_t = [0.0]


def wait():
    _t[0] = throttle(_t[0])


def status_rank(st):
    st = s(st).lower()
    return RANK.index(st) if st in RANK else len(RANK)


def merged(rows, key):
    return "; ".join(dict.fromkeys(t for x in rows for t in re.split(r";\s*", s(x["values"].get(key))) if t))


def carried_notes(rows, keep_dups):
    """notes of the tracked rows; row references made tab-qualified ('duplicate of r12' -> 'PRg1hop2 r12'); notes about
    the other row's own xyz dropped, and 'duplicate' notes too when the cell is NEW here"""
    out = []
    for x in rows:
        for t in re.split(r";\s*", s(x["values"].get("notes"))):
            if not t or t.startswith("xyz off the cell") or (not keep_dups and t.startswith("duplicate")):
                continue
            out.append(re.sub(r"\b(of|into) r(\d+)\b", lambda m: f"{m[1]} {x['tab']} r{m[2]}", t))
    return "; ".join(dict.fromkeys(out))


def src_col(cols, name):
    for nm in (name, STYLE_ALIAS.get(name, name), {"PR_root_id": "upstream_PR"}.get(name, name)):
        if nm in cols:
            return cols[nm]
    return None


def solid(cell):
    return cell.fill is not None and cell.fill.fill_type == "solid"


def live_rows(wb, c):
    """every row with an xyz (backup tabs '*_bYYMMDDd' skipped) -> x['at'] = the live root at its xyz, x['cell'] = its
    root_id_curr while that is current (even when the xyz lies off it), else x['at'] (one point lookup, one
    is_latest_roots call)"""
    rows = [x for x in scan_book(wb) if not re.search(r"_b\d{6}d$", x["tab"])]
    for x, r in zip(rows, roots_at_xyz([s(x["values"]["xyz"]) for x in rows])):
        x["at"] = str(r)
    ids = sorted({s(x["values"].get("root_id_curr")) for x in rows if re.fullmatch(r"\d{15,}", s(x["values"].get("root_id_curr")))})
    wait()
    latest = dict(zip(ids, np.asarray(c.chunkedgraph.is_latest_roots([int(i) for i in ids])).astype(bool)))
    for x in rows:
        cu = s(x["values"].get("root_id_curr"))
        x["cell"] = cu if latest.get(cu) else x["at"]  # a current root_id_curr wins over an xyz that lies off it
    return rows


def tracking(cells, rows, tab):
    """cell -> its tracking carried from the other tabs' rows and SeaTable (see the module docstring)"""
    by_cell = {}
    for x in rows:
        if x["tab"] != tab:
            by_cell.setdefault(x["cell"], []).append(x)
    st_path = sorted(glob.glob(fig("*_seatable_aedes_main.tsv")))[-1]
    sea = {}
    for r in pd.read_csv(st_path, sep="\t", dtype=str).fillna("").to_dict("records"):  # dicts: itertuples renames 'class'
        if r["root_now"]:
            sea.setdefault(r["root_now"], []).append(r)
    rec = {}
    for r in cells:
        rs = by_cell.get(r, [])
        real = [x for x in rs if status_rank(x["values"].get("status")) < len(RANK)]
        best = min(real, key=lambda x: status_rank(x["values"].get("status"))) if real else None
        src = real or rs
        status = s(best["values"]["status"]).lower() if best else "NEW"
        proofreader = merged(src, "proofreader")
        sr = sea.get(r, [])
        seatable = next((s(x["values"].get("SeaTable")) for x in rs if s(x["values"].get("SeaTable"))), "")
        if sr:
            sts = "; ".join(dict.fromkeys(s(x["status"]) for x in sr if s(x["status"])))
            if sts:  # SeaTable's status, always (David 2026-11-18; our checked / proofread were kept until then)
                status = sts
            prs = "; ".join(dict.fromkeys(s(x["proofreader"]) for x in sr if s(x["proofreader"])))
            if prs:
                proofreader = SS.merge_people(proofreader, prs)
            label = "; ".join(dict.fromkeys(seatable_label(x) for x in sr if seatable_label(x)))
            mark = SS.seatable_mark(next((v for v in (s(x["values"].get("SeaTable")) for x in rs) if v.lower().startswith("added")), ""))
            seatable = mark + (f": {label}" if label else "")
        rec[r] = dict(rows=rs, best=best, status=status, proofreader=proofreader, checker=merged(src, "checker"),
                      notes=carried_notes(src, keep_dups=bool(best)), annotation=merged(src, "annotation") or merged(src, "class"),
                      seatable=seatable, in_sea=bool(sr),
                      link=next((s(x["values"].get("link")) for x in ([best] if best else []) + rs if s(x["values"].get("link"))), ""),
                      xyz_src=next((x for x in ([best] if best else []) + rs if x["at"] == r), None),
                      tracked_in="; ".join(dict.fromkeys(x["tab"] for x in rs)))  # tabs, never rows (2026-11-12)
    print(f"tracked in other tabs: {sum(bool(rec[r]['rows']) for r in cells)} (with a status {sum(bool(rec[r]['best']) for r in cells)}); "
          f"in SeaTable ({os.path.basename(st_path)}): {sum(rec[r]['in_sea'] for r in cells)}", flush=True)
    return rec


def geometry(cells, large_cable, large_outside, t0):
    """L2 skeletons (cached in figures/fragment_skeletons.pkl) -> (sk, geo {cell: cable, fr, summ}, large cells)"""
    import navis
    from fafbseg import flywire
    sk = pickle.load(open(SKEL, "rb")) if os.path.exists(SKEL) else {}
    need = [r for r in cells if r not in sk]
    print(f"skeletons: {len(cells) - len(need)} cached, {len(need)} to fetch (throttled)", flush=True)
    for i, r in enumerate(need):
        wait()
        try:
            k = flywire.get_l2_skeleton(int(r), dataset=DATASTACK, progress=False, max_threads=1)
            if k is not None:
                k = k / 1000.0
                k.units = "um"
            sk[r] = k
        except Exception as ex:
            sk[r] = None
            print(f"  skeleton {r} failed: {str(ex)[:80]}", flush=True)
        if i % 10 == 9 or i == len(need) - 1:
            pickle.dump(sk, open(SKEL, "wb"))
            print(f"  {i + 1}/{len(need)} skeletons, {time.time() - t0:.0f} s", flush=True)
    geo = {}
    for r in cells:
        k = sk.get(r)
        geo[r] = dict(cable=np.nan, fr={}, summ="")
        if k is None:
            continue
        try:
            p = navis.resample_skeleton(k, resample_to=1.0).nodes[["x", "y", "z"]].values
        except Exception:
            p = k.nodes[["x", "y", "z"]].values
        fr = ol_meshes.neuropil_fractions(p)
        geo[r] = dict(cable=float(k.cable_length), fr=fr, summ=ol_meshes.summary(fr))
    large = {r for r in cells if geo[r]["cable"] >= large_cable
             or sum(v for kk, v in geo[r]["fr"].items() if kk in OUTSIDE) >= large_outside}
    print(f"very large (low priority): {len(large)}", flush=True)
    return sk, geo, large


def find_xyz(cells, rec, sk, c):
    """cell -> xyz: a tracked row's when it lies on the cell, else a new point verified on the cell (primary neurite
    near the nucleus; no nucleus: the middle of the backbone; one-chunk fragments: their L2 chunks)"""
    from faab.partners.top10_xyz import backbone_midpoint_nm, primary_neurite_point_nm
    xyz = {r: s(rec[r]["xyz_src"]["values"]["xyz"]) for r in cells if rec[r]["xyz_src"]}
    need = [r for r in cells if r not in xyz]
    soma = {}
    if need:
        wait()
        ok = np.asarray(c.chunkedgraph.is_latest_roots([int(r) for r in need])).astype(bool)
        live = [r for r, g in zip(need, ok) if g]
        if len(live) < len(need):
            print(f"  edited since the synapse query (no nucleus lookup): {[r for r in need if r not in live]}", flush=True)
        wait()
        nuc = c.materialize.live_query("nuclei_v1_aedes", timestamp=datetime.datetime.now(datetime.timezone.utc),
                                       filter_in_dict={"pt_root_id": [int(r) for r in live]})
        soma = {str(r): np.asarray(p, float) for r, p in zip(nuc.pt_root_id, nuc.pt_position)}
    cand = {}
    for r in need:
        k, pts = sk.get(r), []
        if k is not None:
            knm = k * 1000.0
            if r in soma:
                try:
                    p, _ = primary_neurite_point_nm(knm, soma[r])
                    if p is not None:
                        pts.append(p)
                except Exception:
                    pass
            pts.append(backbone_midpoint_nm(knm))
            nodes = knm.nodes[["x", "y", "z"]].values.astype(float)
            pts += [nodes[i] for i in np.argsort(((nodes - nodes.mean(0)) ** 2).sum(1))[:3]]
        else:  # one-chunk fragment: its L2 chunks' representative points
            wait()
            l2 = [int(v) for v in c.chunkedgraph.get_leaves(int(r), stop_layer=2)]
            wait()
            d = c.l2cache.get_l2data(l2, attributes=["rep_coord_nm"])
            pts = [np.asarray(v["rep_coord_nm"], float) for v in d.values() if v.get("rep_coord_nm")][:3]
        cand[r] = pts
    for attempt in range(max((len(v) for v in cand.values()), default=0)):
        todo = [r for r in need if r not in xyz and len(cand[r]) > attempt]
        if not todo:
            break
        pts = [np.round(cand[r][attempt] / VOX).astype(int) for r in todo]
        wait()
        for r, p, g in zip(todo, pts, roots_at_xyz(pts)):
            if str(g) == r:
                xyz[r] = f"{p[0]}, {p[1]}, {p[2]}"
    print(f"xyz: {len(cells) - len(need)} from tracked rows; new {sum(r in xyz for r in need)} of {len(need)} verified "
          f"on their cell ({sum(r in soma for r in need)} with a nucleus); without: {[r for r in cells if r not in xyz]}", flush=True)
    return xyz


class Writer:
    """writes titles, header rows and data rows into a tab in the tab's own styles: a data row takes the cell styles of
    its tracked source row (another tab's), else of a plain row of this tab; the yellow MISMATCH fill is dropped"""

    def __init__(self, wb, ws, title_row, hdr, out_names, plain, tab_cols):
        self.wb, self.ws, self.title_row, self.hdr, self.out_names = wb, ws, title_row, hdr, out_names
        self.plain, self.tab_cols = plain, tab_cols
        self.title_style = copy.copy(ws.cell(title_row, 1)._style)
        self.hdr_styles = [copy.copy(ws.cell(hdr, j)._style) for j in range(1, len(out_names) + 1)]
        self.plain_styles = {j: copy.copy(ws.cell(plain, j)._style) for j in range(1, ws.max_column + 1)}

    def title(self, r, text):
        cell = self.ws.cell(r, 1, text)
        cell._style = copy.copy(self.title_style)
        f = cell.font
        cell.font = Font(name=f.name, size=f.size, bold=True, italic=f.italic, color=f.color)

    def header(self, r, names=None):
        for j, n in enumerate(names or self.out_names, 1):
            cell = self.ws.cell(r, j, n or None)
            cell._style = copy.copy(self.hdr_styles[j - 1])

    def row(self, r, d, names=None):
        src = d["_src"]
        for j, n in enumerate(names or self.out_names, 1):
            val = d.get(n)
            if n in ("pr_date", "chk_date"):
                val = val if isinstance(val, datetime.datetime) else (as_date(s(val)) or (s(val) or None))
            elif isinstance(val, (int, np.integer)):
                val = int(val)
            elif n == "nsyn" and re.fullmatch(r"\d+", s(val)):
                val = int(val)
            else:
                val = s(val) or None
            cell = self.ws.cell(r, j, val)
            if src:
                cell._style = copy.copy(self.wb[src["tab"]].cell(src["row"], src_col(src["cols"], n) or 1)._style)
            else:
                cell._style = copy.copy(self.plain_styles[src_col(self.tab_cols, n) or 1])
            if solid(cell) and cell.fill.fgColor.rgb == HL:
                cell.fill = PatternFill(fill_type=None)
            if isinstance(val, datetime.datetime):
                cell.number_format = DATE_FMT
            if n == "link" and isinstance(val, str) and val.startswith("http"):
                cell.hyperlink = val


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", required=True)
    ap.add_argument("--tab", default="PRg1toT4T5")
    ap.add_argument("--thr", type=int, default=3)
    ap.add_argument("--large-cable", type=float, default=1500.0)
    ap.add_argument("--large-outside", type=float, default=0.15)
    a = ap.parse_args()
    t0, stamp = time.time(), utc_stamp()

    # ---- the book: every row with an xyz -> its live cell (one point lookup, one is_latest_roots call)
    wb = openpyxl.load_workbook(a.book)
    c = cave_client()
    rows = live_rows(wb, c)
    tab_rows = [x for x in rows if x["tab"] == a.tab]
    in_tab = {x["cell"] for x in tab_rows}
    hop3 = [x for x in tab_rows if s(x["values"].get("hop")) in ("3", "PR+3") and x["cell"] != "0"]  # PR+3 since 2026-11-10
    targets = sorted({x["cell"] for x in hop3})
    tset = set(targets)
    print(f"{a.tab}: {len(tab_rows)} rows, {len(targets)} targets (hop-3); xyz off the target: "
          f"{[x['row'] for x in hop3 if x['at'] != x['cell']]}; other tabs {len(rows) - len(tab_rows)} rows", flush=True)

    # ---- inputs of the targets
    wait()
    syn = synsnap.Snapshot().synapses(post=[int(t) for t in targets], when="now", client=c)
    when = datetime.datetime.now(datetime.timezone.utc)
    syn = syn[syn.pre_root != syn.post_root]
    e = syn.groupby(["pre_root", "post_root"]).size().rename("n").reset_index()
    e["pre_root"], e["post_root"] = e.pre_root.astype(str), e.post_root.astype(str)
    info, n_thr = {}, 0
    for pre, g in e.groupby("pre_root"):
        g = g.sort_values(["n", "post_root"], ascending=[False, True])
        strong = g[g.n >= a.thr]
        if not len(strong):
            continue
        n_thr += 1
        if pre not in in_tab and pre not in tset:
            info[pre] = dict(tg=list(strong.post_root), n=[int(v) for v in strong.n], total=int(g.n.sum()))
    cells = sorted(info)
    print(f"inputs: {e.pre_root.nunique()} cells with any synapse onto a target; {n_thr} with >= {a.thr} onto one; "
          f"{n_thr - len(cells)} of these already in the tab or targets; {len(cells)} new; {time.time() - t0:.0f} s", flush=True)

    # ---- tracking in the other tabs, SeaTable; skeletons -> cable, neuropils, size; xyz
    rec = tracking(cells, rows, a.tab)
    sk, geo, large = geometry(cells, a.large_cable, a.large_outside, t0)
    xyz = find_xyz(cells, rec, sk, c)

    # ---- rows
    def row(r):
        i, g, R = info[r], geo[r], rec[r]
        b = R["best"]["values"] if R["best"] else {}
        return {"hop": "in", "T4T5_root_id": ", ".join(i["tg"]), "root_id_orig": r, "root_id_curr": r, "xyz": xyz.get(r, ""),
                "nsyn": ", ".join(str(n) for n in i["n"]), "nsyn_out": i["total"], "annotation": R["annotation"],
                "status": R["status"], "pr_date": b.get("pr_date"), "proofreader": R["proofreader"],
                # notes are not copied between rows any more (David 2026-11-08)
                "chk_date": b.get("chk_date"), "checker": R["checker"], "notes": "", "SeaTable": R["seatable"],
                "link": R["link"], "tracked_in": R["tracked_in"], "T4T5_reached": len(i["tg"]),
                "cable_um": "" if np.isnan(g["cable"]) else int(round(g["cable"])), "neuropils": g["summ"],
                "_src": R["best"] or (R["rows"][0] if R["rows"] else None),
                "_key": (-len(i["tg"]), -sum(i["n"]), -i["total"], r)}

    main_rows = sorted((row(r) for r in cells if r not in large), key=lambda d: d["_key"])
    low_rows = sorted((row(r) for r in cells if r in large), key=lambda d: d["_key"])

    # ---- the tab: subtable below the last block, notes moved down
    ws = wb[a.tab]
    hdr_rows = sorted({SS.Book(wb).header_row(a.tab, x["row"]) for x in tab_rows})
    hdr = hdr_rows[-1]
    names = [s(ws.cell(hdr, cc).value) for cc in range(1, ws.max_column + 1)]
    names = names[:max(i for i, n in enumerate(names) if n) + 1]
    width = len(names)
    tab_cols = {n: i + 1 for i, n in enumerate(names) if n}
    col_a = [s(ws.cell(r, 1).value) for r in range(1, ws.max_row + 1)]
    note0 = next(i + 1 for i, t in enumerate(col_a) if t.startswith("note:"))
    filled = lambda r: any(s(ws.cell(r, cc).value) for cc in range(1, width + 1))
    last = max(r for r in range(1, note0) if filled(r))
    blank = last + 1
    title_row = hdr - 1
    plain = next(x["row"] for x in tab_rows if not any(solid(ws.cell(x["row"], cc)) for cc in range(1, width + 1)))
    notes = [(r, [(ws.cell(r, cc).value, copy.copy(ws.cell(r, cc)._style)) for cc in range(1, width + 1)])
             for r in range(note0, ws.max_row + 1) if filled(r)]
    for r, _ in notes:
        for cc in range(1, width + 1):
            ws.cell(r, cc).value = None
            ws.cell(r, cc)._style = copy.copy(ws.cell(blank, cc)._style)
    out_names = ["T4T5_root_id" if n == "hop2_root_id" else n for n in names]
    stamp_when = f"{when:%Y-%m-%d %H:%M} UTC"
    W = Writer(wb, ws, title_row, hdr, out_names, plain, tab_cols)
    put_title, put_header, put_row = W.title, W.header, W.row

    r = last + 2
    first = r
    put_title(r, f"more inputs (hop 'in'): {len(main_rows)} further cells with >= {a.thr} synapses onto at least one hop-3 "
                 f"target (T4/T5 candidate), not listed above (live synapses of {stamp_when}); most targets fed first, then "
                 f"most synapses")
    put_header(r + 1)
    r += 2
    for d in main_rows:
        put_row(r, d)
        r += 1
    if low_rows:
        r += 1
        put_title(r, f"LOW PRIORITY - very large input cells, not Tm-like (>= {a.large_cable / 1000:g} mm of cable, or >= "
                     f"{100 * a.large_outside:.0f}% of it outside the right optic lobe): {len(low_rows)} cells")
        put_header(r + 1)
        r += 2
        for d in low_rows:
            put_row(r, d)
            r += 1
    r += 1
    for _, cells_ in notes:
        for j, (v, sty) in enumerate(cells_, 1):
            ws.cell(r, j).value = v
            ws.cell(r, j)._style = sty
        r += 1
    note_style = notes[0][1][0][1]
    new_note = (f"note: 'more inputs' block (hop 'in', added {when:%Y-%m-%d}): T4T5_root_id = the hop-3 targets the cell "
                f"feeds with >= {a.thr} synapses, strongest first; nsyn = its synapses onto each of them, in that order; "
                f"nsyn_out = its synapses onto all {len(targets)} targets (weaker contacts included); T4T5_reached = targets "
                f"fed with >= {a.thr} synapses. Generated by scripts/partners/target_inputs_block.py ({stamp}).")
    ws.cell(r, 1).value = new_note
    ws.cell(r, 1)._style = copy.copy(note_style)

    # ---- outputs
    base = fig(f"{stamp}_FAABopticlobetargets-{a.tab}")
    with open(base + "_inputs.tsv", "w", encoding="utf-8") as fh:
        fh.write("\t".join(out_names) + "\tblock\n")
        for blk, ds in (("main", main_rows), ("low priority", low_rows)):
            for d in ds:
                fh.write("\t".join(s(d.get(n)) for n in out_names) + f"\t{blk}\n")
    full = os.path.join(TMP, f"{stamp}_inputs_full.xlsx")  # every tab, for the checks that follow
    wb.save(full)
    for other in list(wb.worksheets):
        if other is not ws:
            wb.remove(other)
    wb.save(base + ".xlsx")
    status_counts = pd.Series([d["status"] for d in main_rows + low_rows]).value_counts().to_dict()
    print(f"wrote {base}.xlsx: subtable rows {first}-{r - len(notes) - 2} ({len(main_rows)} + {len(low_rows)} low priority); "
          f"notes from row {r - len(notes)}; statuses {status_counts}; and {os.path.basename(base)}_inputs.tsv; full "
          f"workbook {full}", flush=True)

    # ---- Spelunker link: targets, the new inputs, the optic-lobe meshes
    layers = [{"name": f"hop-3 targets ({len(targets)})", "color": "#7a1010", "ids": targets},
              {"name": f"more inputs ({len(main_rows)})", "color": "#2e86de", "ids": [d["root_id_curr"] for d in main_rows]}]
    if low_rows:
        layers.append({"name": f"more inputs, very large ({len(low_rows)})", "color": "#9467bd",
                       "ids": [d["root_id_curr"] for d in low_rows]})
    spec = {"meshes": [{"kind": "OL", "segments": ["21", "22", "23", "24"], "alpha": 0.15}, {"kind": "neuropil"}],
            "neurons": layers, "view": "anterior", "layout": "3d", "image": True, "select": layers[1]["name"]}
    state = nglink.build(spec)
    open(base + "_inputs_ng.txt", "w").write(nglink.to_url(state) + "\n")
    wait()
    sid = c.state.upload_state_json(state)
    print("short link:", c.state.build_neuroglancer_url(sid, ngl_url="https://spelunker.cave-explorer.org"))
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
