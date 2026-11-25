"""faab-ol-bot: a daily check of the FAAB optic-lobe tracking sheet, reported to Slack (David 2026-11-15: "build
faab-ol-bot that initially just points out common errors such as xyz not matching root_id_curr, mismatches in status
across sheets, two users assigned the same cell on different sheets"). Read-only everywhere: the sheet (an xlsx backup,
or the live Google Sheet exported with a read-only service account), SeaTable (one query, opened read-only), CAVE (the
default view key; 3-4 calls >= 2 s apart, point lookups cached in tmp/sv_at_xyz.json), and Slack only receives
messages, and only with --post.

Checks (every tab except backup copies named '<tab>_bYYMMDDd'):
  xyz_off_cell  root_id_curr is a current root but the xyz is on another cell, or on no cell (stale ids are not
                reported: every sheet update replaces them with the root at the xyz)
  status        one cell on several rows with different statuses, one of them a word the update's ranking does not
                know (a typo, say). Differences between known statuses are not reported: the update copies the best
                status onto every row (faab.partners.reconcile.RANK, checked > proofread > ... > review > ... >
                stopped; David 2026-11-15: "checked > proofread. checked > review."). Blank, NEW, NEW-DUP and
                candidate are ignored
  proofreaders  two rows of one cell each naming someone the other row does not (names compared by initials:
                'Nandita D' = 'ND'); a row listing everyone on another row plus a new name is not reported
  bad_id        a number in root_id_orig / root_id_curr that is not an 18-digit id (a digit missing, 6.48518E+17);
                text rows (block titles, group lines, notes) are skipped
  rounded_id    root_id_orig / root_id_curr stored as a NUMBER: Google Sheets keeps 15 significant digits, so the id
                read back is not the one pasted (the sheet update repairs root_id_curr when the xyz confirms it)
  bad_xyz       a row with an 18-digit root id whose xyz is not three numbers
  annotation    the row's annotation and the SeaTable label in its own SeaTable column ('present: <label>', written
                by the sheet update) name different types ('L5' vs 'ol_intrinsic / visual / Mi'): the annotator is asked
                which is right (David 2026-11-20: conflicts are resolved by people, never overwritten)
  annotation_finer  the annotation names a type SeaTable lacks ('Tm' where SeaTable says 'ol_intrinsic / visual', 'L2?'
                under SeaTable's 'LMC'): add it to SeaTable? An older or shorter copy of SeaTable's label is not
                reported (faab.partners.annotation)
  not_in_seatable  a cell with a done status on the sheet (adequate, complete, proofread, checked; also inside a
                list such as 'needs_extending; adequate') that no SeaTable row is on: no aedes_main row's supervoxel
                resolves to the cell now (David 2026-11-24). A SeaTable row with an old root id, or one whose
                supervoxel is on a split-off piece, does not count, since it shows another cell. A sheet SeaTable
                column that says 'present' / 'added' is quoted (the mark is stale). Needs read access to SeaTable
                (FAAB_SEATABLE_TOKEN, below); without it the check is skipped and the summary says so
A row's cell = its root_id_curr while that is current, else the root at its xyz. Cells are named by full root ids and
tabs, never row numbers (rows move).
New vs old: tmp/faab_ol_bot/state.json keeps the issues already reported; only new ones are listed (the counts always
cover all open ones), and an issue that was fixed and comes back is new again. --all lists every open issue;
--no-state neither reads nor writes the state (for test runs).
Output: tmp/faab_ol_bot/<stamp>_message.md (the Slack text) and <stamp>_issues.tsv. With --post: the summary goes to
the channel and the details into its thread (both as the bot). Without --post nothing leaves this computer.
Config (read from the environment, or from ~/.config/faab-ol-bot/env lines KEY=value; values are never printed):
  SLACK_BOT_TOKEN                 xoxb-... (Slack app 'faab-ol-bot', scope chat:write)        needed for --post
  FAAB_OL_BOT_CHANNEL             channel id, C...                                           needed for --post
  FAAB_SHEET_ID                   the id in the sheet's URL (/spreadsheets/d/<id>/edit)      needed for --live
  GOOGLE_APPLICATION_CREDENTIALS  service-account JSON key (sheet shared with it as Viewer)  needed for --live
  FAAB_OL_BOT_PEOPLE              optional JSON {"initials or name": "Slack user id U..."} for @mentions
  FAAB_SEATABLE_TOKEN             read-only API token of the base holding aedes_main         needed for not_in_seatable
                                  (without it, seaserpent's own SEATABLE_TOKEN is tried, still opened read-only)
  FAAB_SEATABLE_SERVER            optional, default https://flytable.mrc-lmb.cam.ac.uk/
Printed output is counts only (no names, no root ids).
Persistent mode (--serve, David 2026-11-15: run inside screen / tmux on the Ubuntu machine, like banc-bot): the process
stays up and runs the check every day at --at (UTC, default 06:30 = the quietest hour of the CAVE edit log), each run
in a fresh subprocess (so code updated with --pull, git pull --ff-only before each run, is used without a restart). A
failed run is retried after 15, 30 and 60 minutes; tmp/faab_ol_bot/last_run.txt keeps the date of the last run, so a
restart does not repeat that day's post; started after the day's time, it waits for the next day unless --run-now.
Everything it does goes to the screen and to tmp/faab_ol_bot/serve.log.
Time-spent page (David 2026-11-21: "Generate on Ubuntu server and have it host the page"; daily, everyone; nothing
posted): --activity regenerates the proofreaders' editing-time page after each daily check (run_activity: CAVE edit
log, read-only, cached so a day costs a few calls; scripts/admin/user_activity_plot.py --all), --host serves it at
http://127.0.0.1:8765/ (--bind / --port), --activity-now builds it once. Per-person hours are sensitive: keep the
default bind and open it through an SSH tunnel.
Setup walkthrough: faab/docs/faab-ol-bot.md.
Usage: faab_ol_bot.py (--book BACKUP.xlsx | --live) [--post] [--all] [--no-state] [--max-lines 40]
       faab_ol_bot.py --serve [--post] [--pull] [--at 06:30] [--run-now] [--activity] [--host] [--book BACKUP.xlsx]
       faab_ol_bot.py --activity-now"""
import argparse
import collections
import datetime
import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import openpyxl
import pandas as pd

from faab.clients.cave import client as cave_client, roots_at_xyz, throttle
from faab.config import TMP
from faab.partners import annotation as AN
from faab.partners.reconcile import RANK, SEATABLE_STATUSES, rank
from faab.partners.sheet import cell_text as s, utc_stamp

OUT = os.path.join(TMP, "faab_ol_bot")
STATE = os.path.join(OUT, "state.json")
CONFIG = os.path.expanduser("~/.config/faab-ol-bot/env")
BACKUP_TAB = re.compile(r"_b\d{6}d$")
ID = re.compile(r"\d{18}")
XYZ = re.compile(r"-?\d+(?:\.\d+)?")
IGNORED_STATUS = {"", "new", "new-dup", "candidate"}
DONE = set(RANK[:RANK.index("adequate") + 1])  # checked, proofread, complete, adequate: should be in SeaTable
SEATABLE_SERVER = "https://flytable.mrc-lmb.cam.ac.uk/"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
TITLES = {  # plain labels for the proofreaders (no internal jargon)
    "xyz_off_cell": "xyz is not on the row's cell",
    "status": "one cell, different statuses on different rows, one of them not a known status",
    "proofreaders": "one cell, different people on different rows",
    "bad_id": "root id that is not an 18-digit id",
    "rounded_id": "root id stored as a number (Google Sheets keeps only 15 digits)",
    "bad_xyz": "row with a root id but no usable xyz",
    "annotation": "annotation differs from SeaTable's: which one is right?",
    "annotation_finer": "annotation names a type that SeaTable does not have yet: add it to SeaTable?",
    "not_in_seatable": "adequate or proofread on the sheet, but no SeaTable row is on the cell: add it to SeaTable?",
    "pr_groups": "one photoreceptor listed in more than one PR group (a photoreceptor belongs to one group only)",
}


def load_config():
    """KEY=value lines (an 'export ' prefix and quotes allowed) from CONFIG into the environment; existing variables
    win. Nothing is printed."""
    if not os.path.exists(CONFIG):
        return
    for line in open(CONFIG, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def need(*keys):
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        sys.exit(f"missing config: {', '.join(missing)} (environment or {CONFIG}; see faab/docs/faab-ol-bot.md)")


def export_live(path):
    """the live Google Sheet as xlsx, read with the service account (scope drive.readonly: it cannot edit)"""
    need("FAAB_SHEET_ID", "GOOGLE_APPLICATION_CREDENTIALS")
    from google.auth.transport.requests import AuthorizedSession
    from google.oauth2 import service_account
    cr = service_account.Credentials.from_service_account_file(
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"], scopes=["https://www.googleapis.com/auth/drive.readonly"])
    sess, fid = AuthorizedSession(cr), os.environ["FAAB_SHEET_ID"]
    r = sess.get(f"https://www.googleapis.com/drive/v3/files/{fid}/export", params={"mimeType": XLSX}, timeout=180)
    if r.status_code == 403 and "fileNotExportable" in r.text:  # an uploaded .xlsx, not a Google Sheet: download it
        r = sess.get(f"https://www.googleapis.com/drive/v3/files/{fid}", params={"alt": "media"}, timeout=180)
    if r.status_code != 200:
        sys.exit(f"could not read the sheet (HTTP {r.status_code}): {r.text[:300]}")
    open(path, "wb").write(r.content)
    return path


def numlike(v):
    """a value that looks like a number (a root id gone wrong: too few digits, 6.48518E+17, separators), as opposed
    to text such as a block title, a group line ('group 12'), or a note"""
    return bool(re.fullmatch(r"[\d\s.,eE+-]+", v)) and sum(ch.isdigit() for ch in v) >= 6


def scan(wb):
    """every row with a root id (or a number that should be one) in root_id_orig or root_id_curr, read under the
    nearest header row above it (a row naming xyz and a root id column); text rows (block titles, group lines, notes)
    and backup copies skipped"""
    rows = []
    for ws in wb.worksheets:
        if BACKUP_TAB.search(ws.title):
            continue
        cols, grp = None, None
        for r in range(1, ws.max_row + 1):
            vals = [ws.cell(r, c).value for c in range(1, ws.max_column + 1)]
            names = [s(v) for v in vals]
            m = re.fullmatch(r"group (\d+)", names[0]) if ws.title == "PRs" and names else None
            if m:  # PRs tab: the group the following rows belong to
                grp = int(m.group(1))
                continue
            if "xyz" in names and ("root_id_curr" in names or "root_id_orig" in names):
                cols = {n: i for i, n in enumerate(names) if n}
                continue
            if cols is None:
                continue
            v = {n: names[i] if i < len(names) else "" for n, i in cols.items()}
            if any(ID.fullmatch(v.get(k, "")) or numlike(v.get(k, "")) for k in ("root_id_orig", "root_id_curr")):
                # id cells stored as numbers: Google Sheets keeps 15 significant digits, so the id read back is not
                # the one pasted (2026-11-19: 648518347491044000 -> 648518347491043968, another real cell)
                num = [k for k in ("root_id_orig", "root_id_curr") if k in cols and cols[k] < len(vals)
                       and isinstance(vals[cols[k]], (int, float)) and abs(vals[cols[k]]) > 1e15]
                rows.append(dict(tab=ws.title, v=v, num=num, group=grp if ws.title == "PRs" else None))
    return rows


def initials(name):
    w = re.findall(r"[A-Za-zÀ-ÿ]+", name)
    return (w[0][0] + w[-1][0]).upper() if len(w) >= 2 else name.strip().upper()


def people(v):
    return frozenset(initials(x) for x in re.split(r"[;,/&]| and ", v) if x.strip())


def done(status):
    """a done status (DONE), alone or inside a list ('needs_extending; adequate')"""
    return bool({w.strip().lower() for w in re.split(r"[;,]", status)} & DONE)


def seatable_cells(c, t):
    """(the live roots of every SeaTable aedes_main row, note): one read-only SeaTable query (supervoxel ids) and one
    get_roots call; a row without a supervoxel keeps its stored root_id. On no access: (None, why), and the
    not_in_seatable check is skipped"""
    try:
        import seaserpent as ss
        tok = os.environ.get("FAAB_SEATABLE_TOKEN")
        tab = ss.Table("aedes_main", server=os.environ.get("FAAB_SEATABLE_SERVER", SEATABLE_SERVER), read_only=True,
                       progress=False, **({"base_token": tok} if tok else {}))
        recs = tab.query("SELECT supervoxel_id, root_id FROM aedes_main", no_limit=True)
    except Exception as e:  # no token, no network, a changed table: skip this check only
        return None, f"SeaTable could not be read ({type(e).__name__}), so the SeaTable check was skipped"
    # exact integers: one empty cell would make a pandas column float64 and round the 17-18 digit ids
    sv = [int(v) if re.fullmatch(r"\d+", v := str(r.get("supervoxel_id") or "").strip()) and int(v) > 0 else 0
          for r in recs]
    t = throttle(t)
    roots = {str(int(r)) for r in np.asarray(c.chunkedgraph.get_roots(np.array([v for v in sv if v], dtype=np.int64)))}
    roots |= {str(r.get("root_id") or "").strip() for r, v in zip(recs, sv) if not v}
    roots -= {"", "0"}
    return roots, f"{len(recs)} SeaTable rows read"


def check(rows, notes=None):
    """the issues: list of dicts (check, key, cell, tab, text, who); a check that could not run appends a line to
    `notes`"""
    out = []
    for x in rows:
        v = x["v"]
        for k in ("root_id_orig", "root_id_curr"):
            if numlike(v.get(k, "")) and not ID.fullmatch(v[k]):
                ref = v.get("root_id_orig") if k == "root_id_curr" and ID.fullmatch(v.get("root_id_orig", "")) else ""
                out.append(dict(check="bad_id", key=f"bad_id|{x['tab']}|{k}|{v[k]}", cell=ref, tab=x["tab"],
                                who=v.get("proofreader", ""),
                                text=f"{k} is '{v[k]}'" + (f" ({len(v[k])} digits)" if v[k].isdigit() else "")
                                + (f" on the row of {ref}" if ref else "")))
        for k in x.get("num", []):
            out.append(dict(check="rounded_id", key=f"rounded_id|{x['tab']}|{k}|{v[k]}", cell=v[k], tab=x["tab"],
                            who=v.get("proofreader", ""),
                            text=f"{k} reads {v[k]}, but it is stored as a number, so its last digits may have "
                                 "changed. Format the column as plain text and paste the id again"))
        x["ok_xyz"] = len(XYZ.findall(v.get("xyz", ""))) == 3
        ref = next((v[k] for k in ("root_id_curr", "root_id_orig") if ID.fullmatch(v.get(k, ""))), "")
        if not x["ok_xyz"] and ref:
            out.append(dict(check="bad_xyz", key=f"bad_xyz|{x['tab']}|{ref}|{v.get('xyz', '')}", cell=ref,
                            tab=x["tab"], who=v.get("proofreader", ""),
                            text=f"{ref}: xyz is '{v.get('xyz', '')}'"))
        # annotation vs the label in the row's own SeaTable column (written by the sheet update): the update never
        # overwrites an annotation, so differences are asked about here (David 2026-11-20)
        ann, lab = v.get("annotation", "").strip(), AN.label_of_mark(v.get("SeaTable", ""))
        k = AN.kind(ann, lab) if ref and ann and lab else ""
        if k in (AN.CONFLICT, "finer"):
            chk = "annotation" if k == AN.CONFLICT else "annotation_finer"
            out.append(dict(check=chk, key=f"{chk}|{x['tab']}|{ref}|{ann}|{lab}", cell=ref, tab=x["tab"],
                            who=v.get("proofreader", ""),
                            text=f"{ref}: annotation '{ann}', SeaTable '{lab}'"))

    # live roots: is root_id_curr current (1 call), and the root at every xyz (cached points + 1 call)
    c = cave_client()
    ids = sorted({x["v"].get("root_id_curr") for x in rows if ID.fullmatch(x["v"].get("root_id_curr", ""))})
    t = throttle(0.0)
    latest = {}
    for i in range(0, len(ids), 5000):
        part = ids[i:i + 5000]
        latest.update(zip(part, np.asarray(c.chunkedgraph.is_latest_roots([int(r) for r in part])).astype(bool)))
    pts = [x for x in rows if x["ok_xyz"]]
    t = throttle(t)
    for x, r in zip(pts, roots_at_xyz([x["v"]["xyz"] for x in pts], client_=c)):
        x["at"] = str(r)
    by_cell = collections.defaultdict(list)
    for x in rows:
        v, cu, at = x["v"], x["v"].get("root_id_curr", ""), x.get("at", "")
        x["cell"] = cu if latest.get(cu) else (at if at not in ("", "0") else "")
        if x["cell"]:
            by_cell[x["cell"]].append(x)
        if latest.get(cu) and at and at != cu:
            where = f"on {at}" if at != "0" else "on no cell (empty space)"
            out.append(dict(check="xyz_off_cell", key=f"xyz_off_cell|{x['tab']}|{cu}|{at}", cell=cu, tab=x["tab"],
                            who=v.get("proofreader", ""),
                            text=f"{cu}: xyz {v.get('xyz')} is {where}"
                            + (f" (root_id_orig {v['root_id_orig']})" if v.get("root_id_orig") not in ("", cu) else "")))
    for cell, g in by_cell.items():
        if len(g) < 2:
            continue
        st = {x["v"].get("status", "").strip().lower() for x in g} - IGNORED_STATUS
        names = "; ".join(sorted({x["v"]["proofreader"].strip() for x in g if x["v"].get("proofreader", "").strip()}))
        tabs = ", ".join(sorted({x["tab"] for x in g}))
        # known words (the ranking's and SeaTable's): the update settles them (SeaTable's status, else the best)
        unknown = sorted(t for t in st if rank(t) >= len(RANK) and t not in SEATABLE_STATUSES)
        if len(st) > 1 and unknown:
            parts = sorted({f"{x['tab']} {x['v'].get('status', '').strip()}"
                            + (f" ({x['v']['proofreader'].strip()})" if x["v"].get("proofreader", "").strip() else "")
                            for x in g if x["v"].get("status", "").strip().lower() not in IGNORED_STATUS})
            out.append(dict(check="status", key=f"status|{cell}|{'/'.join(sorted(st))}", cell=cell, tab=tabs,
                            who=names, text=f"{cell}: " + " · ".join(parts)
                            + f" (not a known status: {', '.join(repr(t) for t in unknown)})"))
        # two rows each naming someone the other row does not ({A} vs {B}, {A, B} vs {A, C}); a row that lists
        # everyone on another row ({A} vs {A, B}) means that person saw who else is on the cell, so it is not flagged
        sets = list({people(x["v"].get("proofreader", "")) for x in g if x["v"].get("proofreader", "").strip()})
        union = frozenset().union(*sets) if sets else frozenset()
        if any(not (a <= b or b <= a) for i, a in enumerate(sets) for b in sets[i + 1:]):
            parts = sorted({f"{x['tab']} {x['v']['proofreader'].strip()}" for x in g if x["v"].get("proofreader", "").strip()})
            out.append(dict(check="proofreaders", key=f"proofreaders|{cell}|{'/'.join(sorted(union))}", cell=cell,
                            tab=tabs, who=names, text=f"{cell}: " + " · ".join(parts)))

        # one photoreceptor, one group (David 2026-11-25: "different PR groups can't share the same exact PRs")
        grps = sorted({x["group"] for x in g if x.get("group") is not None})
        if len(grps) > 1:
            cls = sorted({f"{x['v'].get('class', '').strip()} in group {x['group']}" for x in g if x.get("group")})
            out.append(dict(check="pr_groups", key=f"pr_groups|{cell}|{'/'.join(map(str, grps))}", cell=cell, tab="PRs",
                            who=names, text=f"{cell}: groups {', '.join(map(str, grps))} ({'; '.join(cls)})"))

    # done on the sheet, but no SeaTable row on the cell (David 2026-11-24); every cell, also one on a single row
    st_cells, note = seatable_cells(c, t)
    if st_cells is None:
        if notes is not None:
            notes.append(note)
        return out
    for cell, g in by_cell.items():
        dn = [x for x in g if done(x["v"].get("status", ""))]
        if not dn or cell in st_cells:
            continue
        parts = sorted({f"{x['tab']} {x['v'].get('status', '').strip()}" for x in dn})
        old = sorted({x["v"].get("root_id_curr", "") for x in g} - {"", cell})
        marks = sorted({x["v"].get("SeaTable", "").strip() for x in g
                        if re.match(r"\s*(?:added|present)", x["v"].get("SeaTable", ""), re.I)})
        xyz = next((x["v"]["xyz"] for x in dn if x["ok_xyz"]), "")
        out.append(dict(check="not_in_seatable", key=f"not_in_seatable|{cell}", cell=cell,
                        tab=", ".join(sorted({x["tab"] for x in g})),
                        who="; ".join(sorted({x["v"]["proofreader"].strip() for x in dn
                                              if x["v"].get("proofreader", "").strip()})),
                        text=f"{cell} (xyz {xyz}): " + " · ".join(parts)
                        + (f"; the sheet still lists {', '.join(old)}" if old else "")
                        + (f"; its SeaTable column says '{marks[0][:60]}', but no SeaTable row is on this cell now"
                           if marks else "")))
    return out


def mention(text, who, people_map):
    """append Slack @mentions for the names in `who` ('; '- or ','-separated) that the people map knows, by exact name
    or by initials"""
    if not people_map:
        return text
    ids = []
    for name in (n.strip() for n in re.split(r"[;,/&]| and ", who)):
        uid = (people_map.get(name) or people_map.get(initials(name))) if name else None
        if uid and uid not in ids:
            ids.append(uid)
    return text + ("  " + " ".join(f"<@{u}>" for u in ids) if ids else "")


def compose(issues, new_keys, source, when, all_=False, max_lines=40, people_map=None, notes=()):
    """(summary, [detail messages]) in Slack mrkdwn; `notes` = checks that could not run"""
    n = collections.Counter(i["check"] for i in issues)
    nn = collections.Counter(i["check"] for i in issues if i["key"] in new_keys)
    head = (f"*faab-ol-bot* sheet check ({source}, {when:%Y-%m-%d %H:%M} UTC): "
            f"{sum(nn.values())} new of {len(issues)} open issue{'s' if len(issues) != 1 else ''}")
    lines = [head] + [f"• {TITLES[k]}: {n[k]}" + (f" ({nn[k]} new)" if nn[k] else "") for k in TITLES if n[k]]
    if not issues:
        lines.append("No open issues. :white_check_mark:")
    lines += [f"• {n}" for n in notes]
    details, chunk = [], []
    for k in TITLES:
        sel = [i for i in issues if i["check"] == k and (all_ or i["key"] in new_keys)]
        if not sel:
            continue
        block = [f"*{TITLES[k]}*" + ("" if all_ else " (new)")]
        for i in sorted(sel, key=lambda i: (i["tab"], i["text"]))[:max_lines]:
            tab = "" if k in ("status", "proofreaders", "not_in_seatable") else f"{i['tab']}: "
            who = f" ({i['who']})" if i["who"] and k not in ("status", "proofreaders") else ""
            block.append(mention(f"• {tab}{i['text']}{who}", i["who"], people_map))
        if len(sel) > max_lines:
            block.append(f"• … and {len(sel) - max_lines} more")
        text = "\n".join(block)
        if chunk and len("\n\n".join(chunk + [text])) > 3500:
            details.append("\n\n".join(chunk))
            chunk = []
        chunk.append(text)
    if chunk:
        details.append("\n\n".join(chunk))
    if details:
        lines.append("Details in the thread.")
    return "\n".join(lines), details


def slack(method, **payload):
    import requests
    r = requests.post(f"https://slack.com/api/{method}", timeout=30, data=json.dumps(payload),
                      headers={"Authorization": f"Bearer {os.environ['SLACK_BOT_TOKEN']}",
                               "Content-Type": "application/json; charset=utf-8"})
    d = r.json()
    if not d.get("ok"):
        sys.exit(f"Slack {method} failed: {d.get('error')}")
    return d


ACTIVITY = os.path.join(OUT, "activity")  # the time-spent page's data (CAVE edit log cache) and the page itself
WWW = os.path.join(ACTIVITY, "www")


def run_activity(log=print):
    """the proofreaders' time-spent page (David 2026-11-21: generated on the Ubuntu machine, hosted there, nothing
    posted): scripts/admin/user_activity_ops.py reads the CAVE edit log since the last run (read-only view key; the
    cache in ACTIVITY makes daily runs a few calls), scripts/admin/user_activity_plot.py --all writes WWW/index.html.
    Names: FAAB_OL_BOT_NAMES or ACTIVITY/cave_user_names.csv (copied from David's cave_user_names.py run; the edit key
    never comes here, so a new account shows as 'user <id>' until that file is refreshed). Returns True when done."""
    import subprocess
    admin = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "admin")
    import glob
    misc = sorted(glob.glob(os.path.join(os.path.dirname(OUT), "..", "faab", "figures", "misc", "*_cave_user_names.csv")))
    cands = [os.environ.get("FAAB_OL_BOT_NAMES"), os.path.join(ACTIVITY, "cave_user_names.csv")] + misc[-1:]
    names = next((c for c in cands if c and os.path.exists(c)), os.path.join(ACTIVITY, "cave_user_names.csv"))
    nm = ["--names", names] if os.path.exists(names) else []
    os.makedirs(WWW, exist_ok=True)
    steps = [("edit log", [sys.executable, os.path.join(admin, "user_activity_ops.py"), "--cache", "faab_ol_bot",
                           "--out-dir", ACTIVITY] + nm),
             ("page", [sys.executable, os.path.join(admin, "user_activity_plot.py"), "--all", "--ops",
                       os.path.join(ACTIVITY, "user_activity_ops.csv"), "--out", os.path.join(WWW, "index.html")] + nm)]
    for what, cmd in steps:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=2 * 3600)
        except subprocess.TimeoutExpired:
            log(f"time-spent page: {what} gave no result after 2 h, stopped")
            return False
        out = [ln for ln in (p.stdout + p.stderr).splitlines() if ln.strip() and not re.search(r"it/s\]|s/it\]", ln)]
        if p.returncode:
            log(f"time-spent page: {what} FAILED (exit {p.returncode}): " + " / ".join(out[-3:]))
            return False
        log(f"time-spent page: {what} done; " + (out[-1] if out else ""))
    if not nm:
        log(f"time-spent page: no names file at {names} - people are shown as 'user <id>' (see the guide)")
    return True


def host(bind, port, log=print):
    """serve WWW (the time-spent page) over HTTP in a background thread; bind 127.0.0.1 (default) = this machine only,
    reach it through an SSH tunnel (see the guide)"""
    import functools
    import http.server
    import threading
    os.makedirs(WWW, exist_ok=True)
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):  # keep the bot's screen quiet
            pass
    srv = http.server.ThreadingHTTPServer((bind, port), functools.partial(Quiet, directory=WWW))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log(f"time-spent page served at http://{bind}:{port}/ (directory {WWW})")
    return srv


def serve(a):
    """persistent mode: the daily check at a.at UTC, each run a fresh subprocess; retries; never twice a day"""
    import subprocess
    import time
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the faab repository
    log_path, last_path = os.path.join(OUT, "serve.log"), os.path.join(OUT, "last_run.txt")
    hh, mm = (int(x) for x in a.at.split(":"))
    utcnow = lambda: datetime.datetime.now(datetime.timezone.utc)

    def log(msg):
        line = f"{utcnow():%Y-%m-%d %H:%M:%S} UTC  {msg}"
        print(line, flush=True)
        open(log_path, "a", encoding="utf-8").write(line + "\n")

    def slot(day):
        return datetime.datetime.combine(day, datetime.time(hh, mm), tzinfo=datetime.timezone.utc)

    def run_once():
        if a.pull:
            p = subprocess.run(["git", "-C", repo, "pull", "--ff-only"], capture_output=True, text=True)
            log("code: " + ((p.stdout or p.stderr).strip().splitlines() or ["?"])[-1])
        cmd = [sys.executable, os.path.abspath(__file__)] + (["--book", a.book] if a.book else ["--live"]) \
            + (["--post"] if a.post else [])
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=3 * 3600)
        except subprocess.TimeoutExpired:
            log("check: no result after 3 h, stopped")
            return False
        for ln in (p.stdout + p.stderr).splitlines():
            if ln.strip() and not re.search(r"it/s\]|s/it\]|Segmentation IDs", ln):
                log("  " + ln)
        log(f"check: {'done' if p.returncode == 0 else f'FAILED (exit {p.returncode})'}")
        return p.returncode == 0

    last = open(last_path).read().strip() if os.path.exists(last_path) else ""
    now = utcnow()
    if a.run_now:
        last = ""
    elif not last and now >= slot(now.date()):
        last = now.date().isoformat()  # first start after today's time: the first run is tomorrow
    if a.host:
        host(a.bind, a.port, log)
    log(f"faab-ol-bot serving: the check runs every day at {a.at} UTC ("
        + ("posts to Slack" if a.post else "dry runs: nothing is posted") + ("; pulls the code first" if a.pull else "")
        + ("; then the time-spent page" if a.activity else "") + f"); log {log_path}; stop with Ctrl+C")
    waits, attempt, retry_at, beat = [15, 30, 60], 0, None, None
    try:
        while True:
            now = utcnow()
            due = (retry_at is not None and now >= retry_at) or (
                retry_at is None and (a.run_now and not last or now >= slot(now.date())) and last != now.date().isoformat())
            if due:
                ok = run_once()
                if a.activity and (ok or attempt >= len(waits)):  # once a day, after the check
                    run_activity(log)
                if ok or attempt >= len(waits):
                    if not ok:
                        log("giving up for today")
                    last, attempt, retry_at = now.date().isoformat(), 0, None
                    open(last_path, "w").write(last)
                else:
                    retry_at = utcnow() + datetime.timedelta(minutes=waits[attempt])
                    attempt += 1
                    log(f"retry {attempt} of {len(waits)} at {retry_at:%H:%M} UTC")
                a.run_now = False
            if beat is None or now - beat >= datetime.timedelta(hours=6):
                nxt = retry_at or (slot(now.date()) if now < slot(now.date()) and last != now.date().isoformat()
                                   else slot(now.date() + datetime.timedelta(days=1)))
                log(f"alive; next check {nxt:%Y-%m-%d %H:%M} UTC")
                beat = now
            time.sleep(30)
    except KeyboardInterrupt:
        log("stopped (Ctrl+C)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--book", help="an xlsx backup of the sheet")
    src.add_argument("--live", action="store_true", help="export the live Google Sheet (service account)")
    ap.add_argument("--post", action="store_true", help="post to Slack (default: write the message to a file only)")
    ap.add_argument("--all", action="store_true", help="list every open issue, not only new ones")
    ap.add_argument("--no-state", action="store_true", help="neither read nor update the state file")
    ap.add_argument("--max-lines", type=int, default=40, help="issues listed per check")
    ap.add_argument("--serve", action="store_true", help="stay running and check every day at --at UTC")
    ap.add_argument("--at", default="06:30", help="daily check time, UTC HH:MM (with --serve)")
    ap.add_argument("--pull", action="store_true", help="git pull --ff-only before each daily check (with --serve)")
    ap.add_argument("--run-now", action="store_true", help="also run once right away (with --serve)")
    ap.add_argument("--activity", action="store_true", help="after each daily check, regenerate the proofreaders' "
                    "time-spent page on this machine (with --serve; nothing is posted)")
    ap.add_argument("--activity-now", action="store_true", help="regenerate the time-spent page once and stop")
    ap.add_argument("--host", action="store_true", help="serve the time-spent page over HTTP (with --serve)")
    ap.add_argument("--bind", default="127.0.0.1", help="address for --host (default 127.0.0.1 = this machine only)")
    ap.add_argument("--port", type=int, default=8765, help="port for --host (default 8765)")
    a = ap.parse_args()
    load_config()
    if a.activity_now:
        os.makedirs(OUT, exist_ok=True)
        return None if run_activity() else sys.exit(1)
    if a.post:
        need("SLACK_BOT_TOKEN", "FAAB_OL_BOT_CHANNEL")
    if a.serve:
        if not a.book:
            need("FAAB_SHEET_ID", "GOOGLE_APPLICATION_CREDENTIALS")
        os.makedirs(OUT, exist_ok=True)
        return serve(a)
    if not (a.book or a.live):
        ap.error("give --book BACKUP.xlsx or --live (or --serve)")
    os.makedirs(OUT, exist_ok=True)
    stamp, when = utc_stamp(), datetime.datetime.now(datetime.timezone.utc)
    if a.live:
        book, source = export_live(os.path.join(OUT, "live.xlsx")), "live sheet"  # one copy, replaced each run
    else:
        book, source = a.book, f"backup {os.path.basename(a.book)[:19]}"
    rows = scan(openpyxl.load_workbook(book))
    print(f"{len(rows)} rows with a root id in {len({x['tab'] for x in rows})} tabs; checking", flush=True)
    notes = []
    issues = check(rows, notes)
    seen = set() if a.no_state or not os.path.exists(STATE) else set(json.load(open(STATE))["open"])
    first = not a.no_state and not os.path.exists(STATE)
    keys = {i["key"] for i in issues}
    new_keys = keys if (first or a.no_state) else keys - seen
    pm = json.load(open(os.environ["FAAB_OL_BOT_PEOPLE"])) if os.environ.get("FAAB_OL_BOT_PEOPLE") else None
    summary, details = compose(issues, new_keys, source, when, all_=a.all, max_lines=a.max_lines, people_map=pm,
                               notes=notes)
    msg_path = os.path.join(OUT, f"{stamp}_message.md")
    open(msg_path, "w", encoding="utf-8").write(summary + "\n\n---\n\n" + "\n\n---\n\n".join(details) + "\n")
    pd.DataFrame(issues, columns=["check", "cell", "tab", "who", "text", "key"]).assign(
        new=lambda d: d.key.isin(new_keys)).to_csv(os.path.join(OUT, f"{stamp}_issues.tsv"), sep="\t", index=False)
    print(summary)  # counts only: names and root ids are in the detail messages, never printed
    print(f"message: {msg_path} ({len(details)} detail message(s))")
    if a.post:
        if new_keys or a.all or first:
            top = slack("chat.postMessage", channel=os.environ["FAAB_OL_BOT_CHANNEL"], text=summary,
                        unfurl_links=False, unfurl_media=False)
            for d in details:
                slack("chat.postMessage", channel=os.environ["FAAB_OL_BOT_CHANNEL"], text=d, thread_ts=top["ts"],
                      unfurl_links=False, unfurl_media=False)
            print(f"posted to Slack: summary + {len(details)} thread message(s)")
        else:
            print("nothing new: not posted")
    if not a.no_state and a.post:  # only a real post moves the 'already reported' line; dry runs leave it
        json.dump({"updated": f"{when:%Y-%m-%d %H:%M} UTC", "open": sorted(keys)}, open(STATE, "w"), indent=1)
        print(f"state: {len(keys)} open issues remembered in {STATE}")


if __name__ == "__main__":
    main()
