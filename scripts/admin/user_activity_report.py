"""Per-user proofreading-activity report from CAVE (aedes / wclee_aedes_brain).

Counts each user's chunkedgraph edit operations (merges/splits) per day, using
cg.get_user_operations(user_id, start, end). Writes a per-user-per-day CSV and
prints per-user totals.

REQUIREMENTS
  - An EDIT-CAPABLE / permitted CAVE token. The read-only view account gets a
    403 FORBIDDEN on get_user_operations (and on auth.get_user_information).
    Either make your permitted token the default, or set CAVE_TOKEN in the env:
        CAVE_TOKEN=xxxxx <venv>/bin/python user_activity_report.py

FINDING YOUR user_id
  - It is the numeric id in a cell's change log:
        cg.get_change_log(<a root you proofread>)["user_info"]  -> {"<user_id>": {...}}
  - Or set USER_IDS below to your team's ids. Leave empty to auto-discover the
    users active in the window (samples recent edits; see DISCOVER_* cost note).

NOTE ON MEANING
  - "work" here = graph edit ops (merges/splits). CAVE logs edits, not viewing
    time or navigation. include_undo=False drops undo ops from the counts.
"""
from faab.config import REPO
import os
import time
import datetime as dt
import numpy as np
import pandas as pd
import caveclient
from faab.sources.cave import DATASTACK, throttle

UTC = dt.timezone.utc
OUT = os.path.join(REPO, "figures")

# ------------------------------- CONFIG -------------------------------
USER_IDS = []                                   # e.g. [5694, 8041, 1019]; [] -> discover
DATE_START = dt.datetime(2026, 7, 1, tzinfo=UTC)
DATE_END = dt.datetime(2026, 8, 1, tzinfo=UTC)  # exclusive-ish upper bound
INCLUDE_UNDO = False                            # False -> exclude undo ops
DISCOVER_IF_EMPTY = True                        # sample recent edits to find users
DISCOVER_SAMPLE = 40                            # #changed-roots to sample (server cost!)
THROTTLE_S = 2.0                                # seconds between CAVE calls
OUT_CSV = os.path.join(OUT, "user_activity_report.csv")
# ----------------------------------------------------------------------


def _client():
    tok = os.environ.get("CAVE_TOKEN")
    return caveclient.CAVEclient(DATASTACK, auth_token=tok) if tok else caveclient.CAVEclient(DATASTACK)


def _timestamps(df):
    """Return a reset-index Series of tz-aware timestamps for the ops DataFrame."""
    for c in df.columns:
        if "time" in c.lower() or c.lower() == "ts":
            try:
                return pd.to_datetime(df[c], utc=True).reset_index(drop=True)
            except Exception:
                pass
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(pd.to_datetime(df.index, utc=True))
    for c in df.columns:                        # last resort: first parseable column
        try:
            s = pd.to_datetime(df[c], utc=True)
            if s.notna().any():
                return s.reset_index(drop=True)
        except Exception:
            continue
    return None


def _is_merge(df):
    """Optional per-op merge flag (True=merge, False=split) if the column exists."""
    for c in df.columns:
        if c.lower() in ("is_merge", "merge"):
            return df[c].astype(bool).reset_index(drop=True)
    return None


def discover_users(cg, start, end, sample, tt):
    print(f"discovering users active {start.date()}..{end.date()} "
          f"(sampling {sample} changed roots; this is the slow part) ...", flush=True)
    tt = throttle(tt, THROTTLE_S)
    _, new = cg.get_delta_roots(start, end)
    if not len(new):
        return [], tt
    idx = np.linspace(0, len(new) - 1, min(sample, len(new))).astype(int)
    users = {}
    for r in new[idx]:
        tt = throttle(tt, THROTTLE_S)
        try:
            for uid, cnt in (cg.get_change_log(int(r)).get("user_info") or {}).items():
                users[int(uid)] = users.get(int(uid), 0) + cnt.get("n_mergers", 0) + cnt.get("n_splits", 0)
        except Exception:
            continue
    print(f"  users seen in sample: {sorted(users.items(), key=lambda kv: -kv[1])}", flush=True)
    return sorted(users), tt


def main():
    client = _client()
    cg = client.chunkedgraph
    tt = throttle(time.time(), THROTTLE_S)

    user_ids = list(USER_IDS)
    if not user_ids and DISCOVER_IF_EMPTY:
        user_ids, tt = discover_users(cg, DATE_START, DATE_END, DISCOVER_SAMPLE, tt)
    if not user_ids:
        print("Set USER_IDS in CONFIG (discovery found none / disabled).")
        return

    names = {}
    try:
        for u in client.auth.get_user_information(user_ids):
            names[int(u.get("id"))] = u.get("name") or u.get("email") or ""
    except Exception as e:
        print(f"(user names unavailable: {str(e)[:60]} -- reporting numeric ids)")

    rows = []
    for uid in user_ids:
        tt = throttle(tt, THROTTLE_S)
        try:
            ops = cg.get_user_operations(uid, DATE_START, include_undo=INCLUDE_UNDO,
                                         timestamp_end=DATE_END)
        except Exception as e:
            print(f"user {uid}: get_user_operations FAILED -> {str(e)[:90]}")
            continue
        ts = _timestamps(ops) if len(ops) else None
        print(f"user {uid} ({names.get(uid, '?')}): {len(ops)} ops"
              + ("" if len(ops) else "  (none in window)"), flush=True)
        if ts is None or not len(ops):
            continue
        merge = _is_merge(ops)
        per = pd.DataFrame({"date": ts.dt.date})
        if merge is not None:
            per["is_merge"] = merge.values
        for d, sub in per.groupby("date"):
            row = {"user_id": uid, "name": names.get(uid, ""), "date": d, "n_ops": len(sub)}
            if merge is not None:
                row["n_merges"] = int(sub["is_merge"].sum())
                row["n_splits"] = int((~sub["is_merge"]).sum())
            rows.append(row)

    if not rows:
        print("\nNo operations returned. Check: token permissions, date range, user ids.")
        return
    rep = pd.DataFrame(rows).sort_values(["date", "user_id"]).reset_index(drop=True)
    rep.to_csv(OUT_CSV, index=False)

    print("\n=== per-user totals in window ===")
    tot = (rep.groupby(["user_id", "name"])
              .agg(total_ops=("n_ops", "sum"), active_days=("date", "nunique")).reset_index())
    tot["ops_per_active_day"] = (tot.total_ops / tot.active_days).round(1)
    print(tot.sort_values("total_ops", ascending=False).to_string(index=False))
    print(f"\nper-user-per-day table -> {OUT_CSV}  ({len(rep)} rows)")


if __name__ == "__main__":
    main()
