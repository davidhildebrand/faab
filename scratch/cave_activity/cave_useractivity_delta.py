"""Test whether the accessible endpoints let us reconstruct per-user/day activity:
get_delta_roots (global changed roots in a window) + get_operation_details
(user+timestamp per op). Read-only, throttled."""
import time
import datetime as dt
import caveclient
from faab.clients.cave import DATASTACK, throttle

UTC = dt.timezone.utc


def main():
    client = caveclient.CAVEclient(DATASTACK)
    cg = client.chunkedgraph
    t = throttle(time.time())
    past = dt.datetime(2025, 9, 15, tzinfo=UTC)
    fut = dt.datetime(2025, 9, 16, tzinfo=UTC)
    try:
        old, new = cg.get_delta_roots(past, fut)
        print(f"get_delta_roots {past.date()}..{fut.date()}: "
              f"{len(old)} old / {len(new)} new roots changed  -> ACCESSIBLE")
    except Exception as e:
        print("get_delta_roots FORBIDDEN/failed:", str(e)[:120])
        new = []

    # for a couple of changed roots, show that operation_details gives user+timestamp
    if len(new):
        t = throttle(t)
        try:
            cl = cg.get_change_log(int(new[0]))
            ops = cl.get("operations_ids", [])[:1]
            if ops:
                t = throttle(t)
                det = cg.get_operation_details(ops)
                d = list(det.values())[0]
                print(f"\nsample op on root {int(new[0])}: user={d.get('user')} "
                      f"ts={d.get('timestamp')} type={'merge' if d.get('added_edges') else 'split'}")
        except Exception as e:
            print("detail step failed:", str(e)[:120])
    print("DONE")


if __name__ == "__main__":
    main()
