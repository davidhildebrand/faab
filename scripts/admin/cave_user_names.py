"""CAVE user id -> name for the ids in a user_activity_ops.py users table (David 2026-11-12). The read-only (view) key
gets 403 on auth.get_user_information, so David runs this with his edit key in CAVE_TOKEN; it only READS user info
(names, no emails written). Writes figures/misc/<stamp>_cave_user_names.csv (user, name), which
user_activity_ops.py picks up (the newest one) on its next run, and adds a name column to the given users table.
Usage: CAVE_TOKEN=... cave_user_names.py [USERS.csv]   (default: the newest figures/misc/*_user_activity_users.csv)"""
import glob
import os
import sys

import caveclient
import pandas as pd

from faab.clients.cave import DATASTACK
from faab.config import fig
from faab.partners.sheet import utc_stamp

path = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob(fig("*_user_activity_users.csv")))
users = pd.read_csv(path)
tok = os.environ.get("CAVE_TOKEN")
if not tok:
    sys.exit("set CAVE_TOKEN (the key that may read user information)")
c = caveclient.CAVEclient(DATASTACK, auth_token=tok)
ids = sorted(int(u) for u in users.user if int(u) > 0)
info = c.auth.get_user_information(ids)
names = pd.DataFrame([{"user": int(u["id"]), "name": u.get("name") or ""} for u in info])
out = fig(f"{utc_stamp()}_cave_user_names.csv")
names.to_csv(out, index=False)
users = users.drop(columns=["name"], errors="ignore").merge(names, on="user", how="left")
users.insert(1, "name", users.pop("name"))
users.to_csv(path, index=False)
print(f"{len(names)} of {len(ids)} names; written {out}; {os.path.basename(path)} now has names:")
print(users[["user", "name", "ops", "active_days", "est_hours"]].to_string(index=False))
