"""Probe what CAVE (this caveclient + server) exposes for measuring user/proofreading
activity: edit logs, operations, per-user, per-time. Read-only, throttled."""
import inspect
import time
import caveclient
from faab.sources.cave import DATASTACK, throttle


def sig(obj, m):
    try:
        return str(inspect.signature(getattr(obj, m)))
    except Exception:
        return "(...)"


def main():
    print("caveclient version:", caveclient.__version__)
    client = caveclient.CAVEclient(DATASTACK)
    cg = client.chunkedgraph
    kws = ("log", "operation", "user", "change", "edit", "delta", "lineage",
           "timestamp", "past", "root_ts")
    print("\n=== chunkedgraph methods related to edits/activity ===")
    for m in sorted(dir(cg)):
        if not m.startswith("_") and any(k in m.lower() for k in kws):
            print(f"  cg.{m}{sig(cg, m)}")

    for svc in ("materialize", "annotation", "state", "auth", "l2cache"):
        s = getattr(client, svc, None)
        if s is None:
            continue
        hits = [m for m in dir(s) if not m.startswith("_")
                and any(k in m.lower() for k in ("log", "user", "version", "timestamp", "change"))]
        if hits:
            print(f"\n{svc}: {hits}")

    # DEMO 1: full edit history of one cell (who + when)
    sample = 648518347526784128
    t = throttle(time.time())
    try:
        cl = cg.get_change_log(sample)
        print(f"\n=== get_change_log({sample}) ===")
        for k, v in cl.items():
            if isinstance(v, list):
                print(f"  {k}: list[{len(v)}] e.g. {v[:4]}")
            else:
                print(f"  {k}: {v}")
    except Exception as e:
        print("get_change_log failed:", str(e)[:150])
        cl = {}

    # DEMO 2: details (user_id + timestamp + type) for one operation
    opids = cl.get("operations_ids") or cl.get("operation_ids") or []
    if opids:
        t = throttle(t)
        try:
            det = cg.get_operation_details([opids[0]])
            print(f"\n=== get_operation_details([{opids[0]}]) ===")
            print(f"  {det}")
        except Exception as e:
            print("get_operation_details failed:", str(e)[:120])

    # DEMO 3: is there a global changed-roots-in-a-time-window query?
    print("\nget_delta_roots signature:", sig(cg, "get_delta_roots") if hasattr(cg, "get_delta_roots") else "ABSENT")
    print("DONE")


if __name__ == "__main__":
    main()
