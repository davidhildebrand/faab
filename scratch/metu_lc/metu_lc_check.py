"""Verify how root_id is stored/read for MeTu4/LC rows: dtype, raw values, and
whether my int() conversion corrupts 18-digit ids. Look for alternative id columns.
Read-only (SeaTable only)."""
import pandas as pd
import seaserpent as ss

SERVER = "https://flytable.mrc-lmb.cam.ac.uk/"


def main():
    meta = ss.Table("aedes_main", server=SERVER).to_frame()
    meta.columns = [str(c) for c in meta.columns]
    print(f"total rows (fresh): {len(meta)}")
    print("\nall columns:", list(meta.columns))

    idcols = [c for c in meta.columns if any(k in c.lower()
              for k in ("root", "id", "supervoxel", "svid", "seg", "nucleus", "pt_"))]
    print("\nid-like columns + dtype:")
    for c in idcols:
        print(f"  {c:22} dtype={meta[c].dtype}")

    sub = meta[meta["subclass"].astype(str).isin(["MeTu4", "LC"])]
    print(f"\nMeTu4/LC rows: {len(sub)}")
    print("root_id dtype:", meta["root_id"].dtype)
    print("\nfirst 6 MeTu4/LC root_id RAW values (repr) + int() result:")
    for v in sub["root_id"].head(6):
        try:
            iv = int(str(v))
            same = (str(iv) == str(v).strip())
        except Exception as e:
            iv, same = f"ERR {e}", False
        print(f"  raw={v!r:28} -> int={iv}  exact_match={same}")

    print("\nsample of other id-like columns for MeTu4/LC rows:")
    for c in idcols:
        if c == "root_id":
            continue
        vals = sub[c].dropna().astype(str).head(4).tolist()
        print(f"  {c:22}: {vals}")

    # status / proofreading columns that might indicate the traced neuron
    for c in ("status", "statusLabel", "proofread", "valid", "notes", "comment"):
        if c in meta.columns:
            print(f"\n{c} value counts (MeTu4/LC):", sub[c].astype(str).value_counts().head(6).to_dict())
    print("DONE")


if __name__ == "__main__":
    main()
