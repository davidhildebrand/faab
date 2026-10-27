"""Local aedes synapse snapshots (the flyconnectome 'aedes' R package's synapse snapshots), read from Python.

What the snapshot is (format 2, https://flyconnectome.github.io/aedes/articles/synapse-snapshots.html):
  static.parquet            every synapse of synapses_v2, fixed: id, pre_sv, post_sv, pre_x/y/z, post_x/y/z, size
                            (111,253,775 rows, sorted by id; positions in 16 x 16 x 45 nm VOXELS - x [16, 16, 45]
                            for nm, as live_query returns them)
  <base>/by_pre.parquet     full root table at the base version: pre_root, post_root, id (sorted by pre_root, id)
  <tag>/log.parquet         checkpoint change log: id, t, pre_root, post_root, old_pre, old_post (one side may be null)
  <tag>/meta.json           tag, timestamp, base, parent, kind, version (written last = snapshot complete)
Checkpoints are made at every CAVE materialisation version (v<N>, daily ~14:12 UTC) or at 20:00 UTC on days without
one (r<YYYYMMDDTHHMMSS>); each chains back to its base through 'parent'. The roots at a checkpoint = the base table
with, for each synapse and side, the root of the latest log row (largest t) along the chain.

URL: the published folder is https://flyemdev.mrc-lmb.cam.ac.uk/flyconnectome/aedes/<sha256>/snapshot/, where the
hash is SHA-256 of the newline-joined, sorted L2 ids of root 648518347624785674 (needs CAVE access; one get_leaves
call, cached in <root>/url.txt) - as in the R package's aedes_snapshot_url_cave().

Use:
    from faab import synsnap
    synsnap.download()                       # once (~2.15 GB over HTTP, resumable, md5-checked); later: new checkpoints
    synsnap.refresh()                        # download() only if the last check is > 6 h old
    snap = synsnap.Snapshot()                # local folder (default FAAB/snapshots/aedes_synapses)
    snap.tags()                              # local checkpoints
    e = snap.edges(pre=[...])                # pre, post, n at the newest local checkpoint (self-synapses dropped)
    e = snap.edges(post=[...], when="v540")  # at a version / tag / datetime (newest checkpoint at or before it)
    s = snap.synapses(pre=[...], post=[...], positions=True)   # one row per synapse (+ static columns)
    e = snap.edges(pre=[...], when="now", client=client)       # newest checkpoint + CAVE catch-up (see catch_up)
The first use of a checkpoint builds its root table (~1 min) and caches it in <root>/heads/<tag>/ (two ~0.6 GB files,
sorted by pre_root and by post_root); later loads read those. Query roots must be roots that exist at that time:
an id edited after the checkpoint has no synapses there (use when="now" with current ids).
CAVE use is limited to the URL lookup and catch_up(), always >= the throttle gap apart with an extra pause.
Validated 2026-10-27 (scratch/synsnap/synsnap_validate.py; logic test scratch/synsnap/synsnap_selftest.py): PR group 1 outputs at 'now' = live_query exactly (1013 synapse ids,
0 partner mismatches, 736 edges with identical counts); inputs of 411 T4/T5 candidates = the live edge table (27107
edges) in 0.06 s. Building a checkpoint's root table takes ~3.5 min once; loading it ~10 s."""
import datetime
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from faab.config import REPO

BASE_URL = "https://flyemdev.mrc-lmb.cam.ac.uk/flyconnectome/aedes/"
KEY_ROOT = 648518347624785674
ROOT = os.path.normpath(os.path.join(REPO, "..", "snapshots", "aedes_synapses"))
STATIC_COLS = ["pre_sv", "post_sv", "pre_x", "pre_y", "pre_z", "post_x", "post_y", "post_z", "size"]


def _utc(ts):
    """'2026-10-27 14:12:21.227052 UTC' / datetime / pandas Timestamp -> tz-aware pandas Timestamp (UTC)"""
    if isinstance(ts, str):
        ts = ts.replace(" UTC", "")
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


# ---------------------------------------------------------------- URL and download
def snapshot_url(root=ROOT, client=None):
    """the published snapshot folder (cached in <root>/url.txt; else one CAVE get_leaves call)"""
    cache = os.path.join(root, "url.txt")
    if os.path.exists(cache):
        return open(cache).read().strip()
    if client is None:
        from faab.clients.cave import client as cave_client
        client = cave_client()
    l2 = sorted({int(x) for x in client.chunkedgraph.get_leaves(KEY_ROOT, stop_layer=2)})
    url = f"{BASE_URL}{hashlib.sha256(chr(10).join(map(str, l2)).encode()).hexdigest()}/snapshot/"
    os.makedirs(root, exist_ok=True)
    open(cache, "w").write(url + "\n")
    return url


def _md5(path, block=8 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(block), b""):
            h.update(b)
    return h.hexdigest()


def _fetch(url, dest, size, md5, log=print):
    """download url -> dest with resume (dest.part) and size + md5 check"""
    import requests
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have > size:
        os.remove(part)
        have = 0
    if have < size:
        headers = {"Range": f"bytes={have}-"} if have else {}
        with requests.get(url, headers=headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            mode = "ab" if (have and r.status_code == 206) else "wb"
            t0, done = time.time(), 0
            with open(part, mode) as fh:
                for chunk in r.iter_content(8 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if size > 200e6 and done % (256 << 20) < (8 << 20):
                        log(f"    {os.path.basename(dest)}: {(have + done) / 1e9:.2f} / {size / 1e9:.2f} GB "
                            f"({done / 1e6 / max(time.time() - t0, 1e-3):.0f} MB/s)")
    if os.path.getsize(part) != size:
        raise IOError(f"{dest}: size {os.path.getsize(part)} != {size}; rerun to resume")
    if md5 and _md5(part) != md5:
        os.remove(part)
        raise IOError(f"{dest}: md5 mismatch; removed the partial file, rerun")
    os.replace(part, dest)


def download(root=ROOT, url=None, client=None, log=print):
    """fetch every manifest file not already present (size + md5 checked); meta.json files go last, then the
    manifest. Snapshots never change, so reruns only add new checkpoints. HTTP only (no CAVE beyond the URL)."""
    import requests
    url = url or snapshot_url(root, client)
    man = requests.get(url + "manifest.json", timeout=60).json()
    files = sorted(man["files"], key=lambda f: (f["path"].endswith("meta.json"), f["path"] != "static.parquet", f["path"]))
    order = {s["tag"]: i for i, s in enumerate(man["snapshots"])}
    files.sort(key=lambda f: (f["path"].endswith("meta.json"), order.get(f["path"].split("/")[0], -1), f["path"]))
    todo = [f for f in files if not (os.path.exists(os.path.join(root, f["path"]))
                                     and os.path.getsize(os.path.join(root, f["path"])) == f["size"])]
    log(f"snapshot manifest {man['created']}: latest {man['latest']}; {len(files)} files, {len(todo)} to fetch "
        f"({sum(f['size'] for f in todo) / 1e9:.2f} GB) into {root}")
    for k, f in enumerate(todo, 1):
        _fetch(url + f["path"], os.path.join(root, f["path"]), f["size"], f.get("md5"), log)
        if f["size"] > 50e6 or k % 10 == 0 or k == len(todo):
            log(f"  {k}/{len(todo)} {f['path']} ok")
    json.dump(man, open(os.path.join(root, "manifest.json"), "w"), indent=1)
    return man["latest"]


def refresh(root=ROOT, max_age_h=6.0, log=print):
    """download() new checkpoints if the last check is older than max_age_h (HTTP only; like the R package's
    aedes_snapshot_refresh). The check time is kept in <root>/.last_check.json. Returns the newest local tag."""
    stamp = os.path.join(root, ".last_check.json")
    last = json.load(open(stamp))["t"] if os.path.exists(stamp) else 0.0
    if time.time() - last > max_age_h * 3600:
        try:
            download(root, log=log)
            json.dump({"t": time.time()}, open(stamp, "w"))
        except Exception as e:  # keep working from the local checkpoints
            log(f"snapshot refresh failed ({str(e)[:120]}); using the local checkpoints")
    return Snapshot(root, log=log).tags().tag.iloc[-1]


# ---------------------------------------------------------------- reading
class Snapshot:
    def __init__(self, root=ROOT, log=print):
        self.root, self.log = root, log
        self._loaded = {}
        self._patches = {}  # (tag, time) -> (computed at, catch-up patch): one CAVE catch-up shared by many queries

    def patch(self, tag, when, client=None, **catch):
        """catch_up(tag, when), computed once per (tag, when); for when='now' reused for 60 s"""
        key = (tag, "now" if when == "now" else str(_utc(when)))
        hit = self._patches.get(key)
        if hit is not None and (when != "now" or time.time() - hit[0] < 60):
            return hit[1]
        p = self.catch_up(tag, when, client=client, **catch)
        self._patches[key] = (time.time(), p)
        return p

    # -- checkpoints
    def tags(self):
        rows = []
        for d in sorted(os.listdir(self.root)):
            m = os.path.join(self.root, d, "meta.json")
            if os.path.exists(m):
                j = json.load(open(m))
                j = {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in j.items()}
                rows.append(dict(tag=j.get("tag", d), timestamp=_utc(j["timestamp"]), base=j.get("base") or None,
                                 parent=j.get("parent") or j.get("base") or None, kind=j.get("kind", "full")))
        return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    def at(self, when="latest", tol_s=1.0):
        """tag of the newest local checkpoint at or before `when` ('latest', a tag, a datetime / string)"""
        t = self.tags()
        if when in (None, "latest", "now"):
            return t.tag.iloc[-1]
        if isinstance(when, str) and when in set(t.tag):
            return when
        w = _utc(when)
        ok = t[t.timestamp <= w + pd.Timedelta(seconds=tol_s)]
        if ok.empty:
            raise ValueError(f"no local checkpoint at or before {w}")
        return ok.tag.iloc[-1]

    def chain(self, tag):
        """tags from `tag` back to its base (newest first)"""
        t = self.tags().set_index("tag")
        out, cur = [], tag
        while cur is not None:
            out.append(cur)
            row = t.loc[cur]
            if row.base is None:
                break
            nxt = row.parent if row.parent != cur else None
            if nxt not in t.index:
                raise ValueError(f"checkpoint chain of {tag} is broken at {cur} (missing {nxt})")
            if t.loc[nxt].base not in (None, row.base) and nxt != row.base:
                raise ValueError(f"{nxt} belongs to another base")
            cur = nxt
        return out

    def _latest_log(self, chain):
        """per synapse id and side, the root of the latest log row along the chain.
        Kept in pyarrow / numpy int64 throughout: the root columns have nulls, and pandas turns a nullable int64
        column into float64, which rounds 18-digit root ids (bug found 2026-10-27: 648518347699279442 -> ...488)."""
        import pyarrow.compute as pc
        tabs = [pq.read_table(os.path.join(self.root, c, "log.parquet"), columns=["id", "t", "pre_root", "post_root"])
                for c in chain if os.path.exists(os.path.join(self.root, c, "log.parquet"))]
        if not tabs:
            return {}
        T = pa.concat_tables(tabs)
        ids = np.asarray(T["id"].to_numpy(), dtype=np.int64)
        tt = np.asarray(T["t"].cast(pa.int64()).to_numpy(), dtype=np.int64)
        out = {}
        for side in ("pre_root", "post_root"):
            col = T[side]
            valid = np.asarray(pc.is_valid(col).to_numpy(), dtype=bool)
            vals = np.asarray(col.filter(pc.is_valid(col)).to_numpy(), dtype=np.int64)
            i, t_ = ids[valid], tt[valid]
            o = np.lexsort((t_, i))  # by id, then time
            i, v = i[o], vals[o]
            last = np.r_[i[1:] != i[:-1], True] if len(i) else np.zeros(0, bool)
            out[side] = (i[last], v[last])
        return out

    def _build_head(self, tag):
        """root table at `tag` -> <root>/heads/<tag>/by_pre.parquet + by_post.parquet"""
        chain = self.chain(tag)
        base = chain[-1]
        t0 = time.time()
        b = pq.read_table(os.path.join(self.root, base, "by_pre.parquet"), columns=["id", "pre_root", "post_root"])
        ids = b["id"].to_numpy().astype(np.int64)
        pre = b["pre_root"].to_numpy().astype(np.int64).copy()
        post = b["post_root"].to_numpy().astype(np.int64).copy()
        del b
        if len(chain) > 1:
            lat = self._latest_log(chain[:-1])
            maxid = int(max(ids.max(), max((v[0].max() for v in lat.values() if len(v[0])), default=0)))
            pos = np.full(maxid + 1, -1, np.int64)
            pos[ids] = np.arange(len(ids))
            for side, arr in (("pre_root", pre), ("post_root", post)):
                if side in lat:
                    li, lr = lat[side]
                    rows = pos[li]
                    ok = rows >= 0
                    arr[rows[ok]] = lr[ok]
            del pos
        d = os.path.join(self.root, "heads", tag)
        os.makedirs(d, exist_ok=True)
        for key, arr in (("pre", pre), ("post", post)):
            o = np.lexsort((ids, arr))
            tb = pa.table({"pre_root": pre[o], "post_root": post[o], "id": ids[o]})
            pq.write_table(tb, os.path.join(d, f"by_{key}.parquet.part"), row_group_size=262144, compression="zstd")
            os.replace(os.path.join(d, f"by_{key}.parquet.part"), os.path.join(d, f"by_{key}.parquet"))
        self.log(f"built root table at {tag} ({len(chain) - 1} checkpoint logs on {base}) in {time.time() - t0:.0f} s")

    HEAD_FORMAT = "2"  # bump to rebuild cached root tables (2: exact int64 log replay, 2026-10-27)

    def load(self, tag):
        """sorted arrays of the root table at `tag` (cached in memory and on disk)"""
        if tag in self._loaded:
            return self._loaded[tag]
        d = os.path.join(self.root, "heads", tag)
        fmt = os.path.join(d, "FORMAT")
        if not (all(os.path.exists(os.path.join(d, f"by_{k}.parquet")) for k in ("pre", "post"))
                and os.path.exists(fmt) and open(fmt).read().strip() == self.HEAD_FORMAT):
            self._build_head(tag)
            open(fmt, "w").write(self.HEAD_FORMAT + "\n")
        H = {}
        for key in ("pre", "post"):
            tb = pq.read_table(os.path.join(d, f"by_{key}.parquet"))
            H[key] = {c: tb[c].to_numpy() for c in ("pre_root", "post_root", "id")}
        self._loaded = {tag: H}  # keep one checkpoint in memory (~4.5 GB)
        return H

    @staticmethod
    def _rows(sorted_keys, q):
        q = np.unique(np.asarray(list(q), dtype=np.int64))
        lo = np.searchsorted(sorted_keys, q, "left")
        hi = np.searchsorted(sorted_keys, q, "right")
        n = hi - lo
        if not n.sum():
            return np.empty(0, np.int64)
        return np.repeat(lo - np.cumsum(np.r_[0, n[:-1]]), n) + np.arange(n.sum())

    def synapses(self, pre=None, post=None, when="latest", positions=False, client=None, **catch):
        """one row per synapse: id, pre_root, post_root (+ static columns if positions) with pre_root in `pre` and/or
        post_root in `post`; when='now' (or a time after the newest local checkpoint) adds a CAVE catch-up"""
        if pre is None and post is None:
            raise ValueError("give pre and/or post root ids")
        tag = self.at(when)
        H = self.load(tag)
        side = "pre" if pre is not None and (post is None or len(set(pre)) <= len(set(post))) else "post"
        A = H[side]
        r = self._rows(A[f"{side}_root"], pre if side == "pre" else post)
        df = pd.DataFrame({k: A[k][r] for k in ("id", "pre_root", "post_root")})
        if self._needs_catch_up(tag, when):
            df = self._apply_patch(df, self.patch(tag, when, client=client, **catch), pre, post)
        if pre is not None:
            df = df[df.pre_root.isin(set(int(x) for x in pre))]
        if post is not None:
            df = df[df.post_root.isin(set(int(x) for x in post))]
        df = df.sort_values("id").reset_index(drop=True)
        if positions:
            df = df.merge(self.static(df.id.to_numpy()), on="id", how="left")
        return df

    def edges(self, pre=None, post=None, when="latest", min_syn=1, drop_self=True, client=None, **catch):
        """pre, post, n (synapse counts per connected pair), like faab class_paths edge tables"""
        s = self.synapses(pre, post, when, client=client, **catch)
        if drop_self:
            s = s[s.pre_root != s.post_root]
        e = s.groupby(["pre_root", "post_root"]).size().rename("n").reset_index()
        e.columns = ["pre", "post", "n"]
        return e[e.n >= min_syn].sort_values("n", ascending=False).reset_index(drop=True)

    def static(self, ids, columns=STATIC_COLS):
        """static columns (supervoxels, positions, size) for synapse ids; static.parquet is sorted by id, so the
        id filter reads only the row groups that hold them"""
        ids = np.unique(np.asarray(ids, dtype=np.int64))
        if not len(ids):
            return pd.DataFrame(columns=["id"] + list(columns))
        tb = pq.read_table(os.path.join(self.root, "static.parquet"), columns=["id"] + list(columns),
                           filters=[("id", ">=", int(ids.min())), ("id", "<=", int(ids.max()))])
        df = tb.to_pandas()
        return df[df.id.isin(ids)].reset_index(drop=True)

    # -- catching up to times after the newest local checkpoint (CAVE, slow and spread out)
    def _needs_catch_up(self, tag, when):
        if when == "now":
            return True
        if when in (None, "latest") or (isinstance(when, str) and when in set(self.tags().tag)):
            return False
        return _utc(when) > self.tags().set_index("tag").loc[tag].timestamp + pd.Timedelta(seconds=1)

    def catch_up(self, tag, when="now", client=None, max_interval_s=86400, overlap_s=600, batch=20000, pause_s=10.0):
        """synapses whose roots changed between checkpoint `tag` and `when`, with their roots at `when`:
        1. get_delta_roots over <= 1-day steps from (checkpoint - 10 min) to `when` -> expired (old) roots
        2. rows of the root table with an expired root on either side -> their supervoxels (static.parquet)
        3. get_roots(supervoxels, timestamp=when) in batches of `batch`
        Every CAVE call is at least the throttle gap (5 s) + pause_s apart. Returns a DataFrame id, pre_root,
        post_root of the changed synapses (the full new state of each changed row)."""
        from faab.clients.cave import client as cave_client, throttle
        client = client or cave_client()
        t_tag = self.tags().set_index("tag").loc[tag].timestamp
        t_end = pd.Timestamp.now(tz="UTC") if when == "now" else _utc(when)
        last = [0.0]

        def wait():
            last[0] = throttle(last[0], gap=5.0 + pause_s)

        old = set()
        a = t_tag - pd.Timedelta(seconds=overlap_s)
        while a < t_end:
            b = min(a + pd.Timedelta(seconds=max_interval_s), t_end)
            wait()
            o, _n = client.chunkedgraph.get_delta_roots(a.to_pydatetime(), b.to_pydatetime())
            old |= {int(x) for x in np.atleast_1d(o)}
            a = b
        H = self.load(tag)
        P = H["pre"]
        if not old:
            self.log(f"catch-up {tag} -> {t_end:%Y-%m-%d %H:%M} UTC: no edits")
            return pd.DataFrame(columns=["id", "pre_root", "post_root"])
        oa = np.fromiter(old, np.int64)
        stale = np.isin(P["pre_root"], oa) | np.isin(P["post_root"], oa)
        ch = pd.DataFrame({"id": P["id"][stale], "pre_root": P["pre_root"][stale], "post_root": P["post_root"][stale]})
        st = self.static(ch.id.to_numpy(), columns=["pre_sv", "post_sv"])
        ch = ch.merge(st, on="id")
        need = pd.unique(np.r_[ch.pre_sv[ch.pre_root.isin(old)].to_numpy(), ch.post_sv[ch.post_root.isin(old)].to_numpy()])
        self.log(f"catch-up {tag} -> {t_end:%Y-%m-%d %H:%M} UTC: {len(old)} expired roots, {len(ch)} synapses, "
                 f"{len(need)} supervoxels to look up in {int(np.ceil(len(need) / batch))} calls")
        new = {}
        for i in range(0, len(need), batch):
            part = np.sort(need[i:i + batch]).astype(np.uint64)
            wait()
            r = client.chunkedgraph.get_roots(part, timestamp=t_end.to_pydatetime())
            new.update(zip(part.astype(np.int64).tolist(), np.asarray(r).astype(np.int64).tolist()))
        keys = np.array(sorted(new), dtype=np.int64)
        vals = np.array([new[k] for k in keys], dtype=np.int64)

        def remap(sv):  # exact int64 lookup (no pandas map: a float detour would round 18-digit ids)
            return vals[np.minimum(np.searchsorted(keys, np.asarray(sv, dtype=np.int64)), max(len(keys) - 1, 0))]

        for side in ("pre", "post"):
            stale_side = np.isin(ch[f"{side}_root"].to_numpy(np.int64), oa)
            ch[f"{side}_root"] = np.where(stale_side, remap(ch[f"{side}_sv"]), ch[f"{side}_root"].to_numpy(np.int64))
        return ch[["id", "pre_root", "post_root"]]

    @staticmethod
    def _apply_patch(df, patch, pre, post):
        if patch is None or patch.empty:
            return df
        df = df[~df.id.isin(set(patch.id))]
        keep = pd.Series(True, index=patch.index)
        if pre is not None:
            keep &= patch.pre_root.isin(set(int(x) for x in pre))
        if post is not None:
            keep &= patch.post_root.isin(set(int(x) for x in post))
        return pd.concat([df, patch[keep]], ignore_index=True)
