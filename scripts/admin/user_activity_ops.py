"""Per-user proofreading activity from CAVE with the default READ-ONLY (view) key (David 2026-11-12).

Why not user_activity_report.py: its call, chunkedgraph user_operations, is gated by the dataset permission
'admin_view' on the server (PyChunkedGraph routes.py: @auth_requires_permission("admin_view")); neither the view key
nor the edit (read-write) key has it -> 403. operation_details only needs 'view' and returns, per edit operation, the
user id, timestamp and the edges added (merge) or removed (split). Operation ids come from one counter per dataset,
so every operation in a time window is found by reading the id range in batches:
  1. a recent operation id: get_delta_roots(last 2 days) -> a changed root -> get_change_log -> its operation ids;
     then step forward to the newest id;
  2. binary search on the id for the first operation at or after START (timestamps rise with the id);
  3. operation_details for every id START..newest, BATCH ids per call, >= 2 s between calls (cached to
     tmp/<run>_ops_raw.parquet; a rerun with --resume RUN continues).
Report per user (ids; names when auth.get_user_information answers): merges, splits, other ops (undo / redo), active
days, work sessions (consecutive ops less than --gap minutes apart; a session = last - first op + 2 min) and the
estimated hours; per user per ISO week. CAVE logs edits only, not viewing or navigation, so hours are a LOWER bound
of time spent and say nothing about review-only work.
Writes figures/misc/<stamp>_user_activity_ops.csv (one row per operation), _user_activity_users.csv, _weeks.csv.
Usage: user_activity_ops.py [--months 4] [--gap 30] [--batch 400] [--resume RUN]"""
import argparse
import datetime as dt
import glob
import os
import time

import numpy as np
import pandas as pd

from faab.clients.cave import client as cave_client, throttle
from faab.config import TMP, fig
from faab.partners.sheet import utc_stamp

UTC = dt.timezone.utc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--months", type=int, default=4)
    ap.add_argument("--gap", type=float, default=30.0, help="minutes between ops that start a new session")
    ap.add_argument("--batch", type=int, default=400)
    ap.add_argument("--resume", help="run stamp of an earlier, interrupted run")
    a = ap.parse_args()
    run = a.resume or utc_stamp()
    raw_path = os.path.join(TMP, f"{run}_ops_raw.parquet")
    end = dt.datetime.now(UTC)
    start = (pd.Timestamp(end) - pd.DateOffset(months=a.months)).to_pydatetime()
    c = cave_client()  # default key = the view-only account
    cg = c.chunkedgraph
    last = [time.time()]

    def call(f, *args, **kw):
        last[0] = throttle(last[0])
        return f(*args, **kw)

    def details(ids):
        d = call(cg.get_operation_details, [int(i) for i in ids]) or {}
        return {int(k): v for k, v in d.items()}

    def ts(v):
        return pd.Timestamp(v["timestamp"]).tz_localize(UTC) if pd.Timestamp(v["timestamp"]).tzinfo is None else pd.Timestamp(v["timestamp"])

    def nearest(i, tries=3):
        """the first existing operation at or after id i (ids can have gaps), or None; one batch per try"""
        for j0 in range(i, i + tries * a.batch, a.batch):
            d = details(range(j0, j0 + a.batch))
            if d:
                k = min(d)
                return k, d[k]
        return None

    # 1. newest operation id
    _, new = call(cg.get_delta_roots, end - dt.timedelta(days=2), end)
    if not len(new):
        raise SystemExit("no edits in the last 2 days: cannot anchor the operation ids")
    ops = []
    for r in new[:: max(1, len(new) // 5)][:5]:
        ops += list(call(cg.get_change_log, int(r)).get("operations_ids") or [])
    hi = max(ops)
    while True:  # step forward to the newest id
        d = details(range(hi + 1, hi + 1 + a.batch))
        if not d:
            break
        hi = max(d)
    print(f"newest operation id {hi}", flush=True)

    # 2. first id at or after START (binary search; gaps resolved by the next existing id)
    lo_id, hi_id = 1, hi
    first = nearest(1)
    if first and ts(first[1]) >= pd.Timestamp(start):
        lo_id = first[0]
    else:
        while hi_id - lo_id > 50:
            mid = (lo_id + hi_id) // 2
            n = nearest(mid)
            if n is None or n[0] > hi_id:
                hi_id = mid
                continue
            if ts(n[1]) < pd.Timestamp(start):
                lo_id = n[0] + 1
            else:
                hi_id = n[0]
    print(f"window {start:%Y-%m-%d %H:%M} .. {end:%Y-%m-%d %H:%M} UTC: operation ids {lo_id} .. {hi} "
          f"(~{hi - lo_id + 1} ids, {(hi - lo_id) // a.batch + 1} calls of {a.batch})", flush=True)

    # 3. read every id in the window (resumable)
    have = pd.read_parquet(raw_path) if os.path.exists(raw_path) else pd.DataFrame(columns=["op", "user", "timestamp", "kind"])
    rows = have.to_dict("records")
    # ids already asked for (ranges 'a-b' per line): ~30% of ids have no operation, so the ops found can't tell
    seen_path = raw_path.replace("_ops_raw.parquet", "_ops_scanned.txt")
    scanned = [tuple(map(int, ln.split("-"))) for ln in open(seen_path)] if os.path.exists(seen_path) else []
    if not scanned and len(have):  # a run from before the ranges file: everything up to its newest op was read
        scanned = [(lo_id, int(have.op.max()))]
    asked = lambda i: any(x <= i <= y for x, y in scanned)
    t0 = time.time()
    batches = list(range(lo_id, hi + 1, a.batch))
    for bi, b0 in enumerate(batches):
        ids = [i for i in range(b0, min(b0 + a.batch, hi + 1)) if not asked(i)]
        if not ids:
            continue
        for k, v in details(ids).items():
            kind = "merge" if v.get("added_edges") else "split" if v.get("removed_edges") else "other"
            rows.append(dict(op=k, user=int(v.get("user", -1)), timestamp=str(ts(v)), kind=kind))
        scanned.append((ids[0], ids[-1]))
        if bi % 25 == 0 or bi == len(batches) - 1:
            pd.DataFrame(rows).to_parquet(raw_path)
            open(seen_path, "w").write("".join(f"{x}-{y}\n" for x, y in scanned))
            el = time.time() - t0
            print(f"  {bi + 1}/{len(batches)} calls, {len(rows)} operations ({el / 60:.1f} min)", flush=True)
    d = pd.DataFrame(rows).drop_duplicates("op")
    d["timestamp"] = pd.to_datetime(d.timestamp, utc=True, format="ISO8601")  # with and without fractional seconds
    d = d[(d.timestamp >= pd.Timestamp(start)) & (d.timestamp <= pd.Timestamp(end))].sort_values(["user", "timestamp"])

    # 4. names (may be refused for the view account) and the report
    names = {}
    known = sorted(glob.glob(fig("*_cave_user_names.csv")))  # from cave_user_names.py (David's key)
    if known:
        names = dict(zip(*pd.read_csv(known[-1])[["user", "name"]].T.values))
    try:
        for u in call(c.auth.get_user_information, sorted(int(x) for x in d.user.unique() if x > 0 and x not in names)):
            names[int(u["id"])] = u.get("name") or ""
    except Exception as ex:
        print(f"(user names unavailable for the view key: {str(ex)[:60]}; {len(names)} from {os.path.basename(known[-1]) if known else 'no names file'}; "
              f"run cave_user_names.py with the edit key for the rest)")
    d["name"] = d.user.map(names).fillna("")
    gap = pd.Timedelta(minutes=a.gap)
    d["session"] = (d.groupby("user").timestamp.diff() > gap).groupby(d.user).cumsum()
    ses = d.groupby(["user", "session"]).timestamp.agg(["min", "max", "size"])
    ses["hours"] = ((ses["max"] - ses["min"]) + pd.Timedelta(minutes=2)).dt.total_seconds() / 3600
    per = d.groupby(["user", "name"]).agg(ops=("op", "size"), merges=("kind", lambda k: int((k == "merge").sum())),
                                          splits=("kind", lambda k: int((k == "split").sum())),
                                          other=("kind", lambda k: int((k == "other").sum())),
                                          active_days=("timestamp", lambda t: t.dt.date.nunique()),
                                          first=("timestamp", "min"), last=("timestamp", "max")).reset_index()
    per["sessions"] = per.user.map(ses.groupby("user").size())
    per["est_hours"] = per.user.map(ses.groupby("user").hours.sum()).round(1)
    per = per.sort_values("ops", ascending=False)
    d["week"] = d.timestamp.dt.strftime("%G-W%V")
    wk = d.groupby(["user", "name", "week"]).size().rename("ops").reset_index()
    wk = wk.merge((ses.reset_index().assign(week=lambda t: t["min"].dt.strftime("%G-W%V"))
                   .groupby(["user", "week"]).hours.sum().round(1).rename("est_hours").reset_index()), on=["user", "week"], how="left")
    stamp = utc_stamp()
    d.drop(columns=["session"]).to_csv(fig(f"{stamp}_user_activity_ops.csv"), index=False)
    per.to_csv(fig(f"{stamp}_user_activity_users.csv"), index=False)
    wk.to_csv(fig(f"{stamp}_user_activity_weeks.csv"), index=False)
    print(f"\n{len(d)} operations by {d.user.nunique()} users, {start:%Y-%m-%d} .. {end:%Y-%m-%d} "
          f"(sessions split at gaps > {a.gap:g} min; hours = edit time only, a lower bound):")
    print(per.drop(columns=["first", "last"]).to_string(index=False))
    print(f"written {fig(f'{stamp}_user_activity_users.csv')} (+ _ops.csv, _weeks.csv)")


if __name__ == "__main__":
    main()
