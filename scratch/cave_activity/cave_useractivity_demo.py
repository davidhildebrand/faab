"""Demo: measure per-user proofreading work with CAVE. Uses get_user_operations
(merges/splits by a user in a time window) and bins by day. Read-only, throttled."""
import time
import datetime as dt
import pandas as pd
import caveclient
from faab.clients.cave import DATASTACK, throttle

UTC = dt.timezone.utc


def main():
    client = caveclient.CAVEclient(DATASTACK)
    cg = client.chunkedgraph
    print("edit history spans back to:", cg.get_oldest_timestamp())

    try:
        info = client.auth.get_user_information([5694, 8041])
        print("\nuser identities (from the change log demo):")
        for u in info:
            print(f"  id {u.get('id')}: {u.get('name')}  ({u.get('email','')})")
    except Exception as e:
        print("user info unavailable:", str(e)[:100])

    start = dt.datetime(2025, 9, 1, tzinfo=UTC)
    end = dt.datetime(2025, 10, 1, tzinfo=UTC)
    t = throttle(time.time())
    for uid in (5694, 8041):
        t = throttle(t)
        try:
            ops = cg.get_user_operations(uid, start, end)
        except Exception as e:
            print(f"\nuser {uid}: get_user_operations failed: {str(e)[:120]}")
            continue
        print(f"\n=== user {uid}: {len(ops)} operations in {start.date()}..{end.date()} ===")
        print("  columns:", list(ops.columns))
        if len(ops):
            tcol = "timestamp" if "timestamp" in ops.columns else ops.columns[-1]
            days = pd.to_datetime(ops[tcol], utc=True).dt.date
            byday = days.value_counts().sort_index()
            print(f"  active days: {byday.shape[0]}, total ops: {len(ops)}, "
                  f"median ops/active-day: {int(byday.median())}")
            print("  ops per day (top 6):",
                  {str(k): int(v) for k, v in byday.sort_values(ascending=False).head(6).items()})
    print("DONE")


if __name__ == "__main__":
    main()
