"""Convex-backed datasets. No runtime reads or writes to a local data directory.

RemotePath provides dataset names/metadata, not filesystem paths. CSV conversion
is explicit through read_csv/write_csv so pandas never creates a local file.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from fnmatch import fnmatch
import gzip
import hashlib
from io import StringIO
import os
from pathlib import PurePosixPath
import threading
import time
from uuid import uuid4
from types import SimpleNamespace

import pandas as pd
import requests

_cache = OrderedDict()
_cache_size = 0
_cache_lock = threading.RLock()
_MAX_CACHE = 128 * 1024 * 1024  # Compressed bytes, RAM only.
_metadata_cache = OrderedDict()
_METADATA_TTL = 10
_download_locks = [threading.Lock() for _ in range(32)]
_lease = ContextVar("cloud_operation_lease", default=None)


def rpc(kind, function, args):
    from .prediction_store import configured, load_environment
    load_environment()
    if not configured():
        raise RuntimeError("Convex data storage is not configured. Set CONVEX_URL and PREDICTION_STORE_SECRET.")
    response = requests.post(os.environ["CONVEX_URL"].rstrip("/") + "/api/" + kind,
        json={"path": "datasets:" + function, "args": {**args, "secret": os.environ["PREDICTION_STORE_SECRET"]}, "format": "json"}, timeout=60)
    response.raise_for_status()
    result = response.json()
    if result.get("status") != "success":
        detail = str(result.get("errorMessage", "Unknown database error")).replace(os.environ["PREDICTION_STORE_SECRET"], "[redacted]")
        raise RuntimeError(f"Convex dataset operation failed: {detail[:600]}. No disk fallback is used.")
    return result["value"]


def _metadata_key(key):
    return (os.environ.get("CONVEX_URL"), os.environ.get("PREDICTION_STORE_SECRET"), key)


def _invalidate_metadata(key):
    with _cache_lock:
        _metadata_cache.pop(_metadata_key(key), None)


def metadata(key):
    from .prediction_store import load_environment
    load_environment()
    identity = _metadata_key(key)
    with _cache_lock:
        cached = _metadata_cache.get(identity)
        if cached and time.monotonic() - cached[0] < _METADATA_TTL:
            return dict(cached[1])
    value = rpc("query", "get", {"key": key})
    if value is not None:
        with _cache_lock:
            _metadata_cache[identity] = (time.monotonic(), dict(value))
            _metadata_cache.move_to_end(identity)
            while len(_metadata_cache) > 4096:
                _metadata_cache.popitem(last=False)
    return value


def list_datasets(prefix=""):
    tables = ["prices", "fundamentals", "news", "companyMetadata", "priceSync", "modelArtifacts", "backtestReports", "paperLedgers", "predictionExports"]
    if prefix.startswith("prices/"):
        tables = ["prices", "priceSync"]
    elif prefix.startswith("fundamentals/"):
        tables = ["fundamentals", "companyMetadata"]
    elif prefix.startswith("news/"):
        tables = ["news"]
    else:
        for directory, table in {"models/": "modelArtifacts", "backtests/": "backtestReports", "paper_trades/": "paperLedgers", "prediction_exports/": "predictionExports"}.items():
            if prefix.startswith(directory):
                tables = [table]
                break
    rows = []
    for table in tables:
        cursor = None
        while True:
            page = rpc("query", "list", {"table": table, "prefix": prefix, "cursor": cursor})
            rows.extend(page["page"])
            if page["isDone"]:
                break
            cursor = page["continueCursor"]
    return rows


def _remember_content(digest, packed):
    global _cache_size
    with _cache_lock:
        if digest not in _cache and len(packed) <= _MAX_CACHE:
            _cache[digest] = packed
            _cache_size += len(packed)
            while _cache_size > _MAX_CACHE:
                _, previous = _cache.popitem(last=False)
                _cache_size -= len(previous)


def read_bytes(key, meta=None):
    meta = metadata(key) if meta is None else meta
    if meta is None:
        raise FileNotFoundError(f"No Convex dataset: {key}")
    digest = meta["contentSha256"]
    # Concurrent stock/model requests share one verified download per content hash.
    with _download_locks[hash(digest) % len(_download_locks)]:
        with _cache_lock:
            cached = _cache.get(digest)
            if cached is not None:
                _cache.move_to_end(digest)
        if cached is not None:
            return gzip.decompress(cached)
        response = requests.get(meta["url"], timeout=90)
        response.raise_for_status()
        packed = response.content
        if hashlib.sha256(packed).hexdigest() != meta["sha256"]:
            raise RuntimeError(f"Convex upload integrity failed: {key}")
        content = gzip.decompress(packed)
        if len(content) != meta["size"] or hashlib.sha256(content).hexdigest() != digest:
            raise RuntimeError(f"Convex data integrity failed: {key}")
        _remember_content(digest, packed)
        return content


def write_bytes(key, content, modified_at=None):
    # Optimistic writes always need fresh metadata, never a cached version.
    _invalidate_metadata(key)
    current = metadata(key)
    digest = hashlib.sha256(content).hexdigest()
    if current and current["contentSha256"] == digest:
        return current
    packed = gzip.compress(content, mtime=0)
    upload = rpc("mutation", "uploadUrl", {})
    response = requests.post(upload, data=packed, headers={"Content-Type": "application/gzip"}, timeout=120)
    response.raise_for_status()
    try:
        committed = rpc("mutation", "commit", {"key": key, "storageId": response.json()["storageId"],
        "sha256": hashlib.sha256(packed).hexdigest(), "contentSha256": digest, "size": len(content),
        "modifiedAt": (time.time() if modified_at is None else modified_at) * 1000,
        "expectedVersion": current["version"] if current else None,
        **({"lease": _lease.get()} if _lease.get() else {})})
    finally:
        _invalidate_metadata(key)
    # Retire superseded versions so daily updates cannot evict the next stock's live history.
    if current and current["contentSha256"] != digest:
        global _cache_size
        with _cache_lock:
            old = _cache.pop(current["contentSha256"], None)
            if old is not None:
                _cache_size -= len(old)
    # commit validates the uploaded compressed checksum before confirming publication.
    _remember_content(digest, packed)
    return committed


@contextmanager
def exclusive_cloud_operation(key):
    lease = {"key": key, "owner": uuid4().hex}
    rpc("mutation", "acquireLock", lease)
    stop = threading.Event()
    errors = []
    def renew():
        while not stop.wait(60):
            try:
                rpc("mutation", "renewLock", lease)
            except Exception as error:
                errors.append(error)
                return
    heartbeat = threading.Thread(target=renew, daemon=True)
    heartbeat.start()
    token = _lease.set(lease)
    try:
        yield
        if errors:
            raise RuntimeError("Could not renew cloud operation lock") from errors[0]
    finally:
        _lease.reset(token)
        stop.set()
        heartbeat.join(timeout=5)
        rpc("mutation", "releaseLock", lease)


@dataclass(frozen=True)
class RemotePath:
    key: str = ""

    def __truediv__(self, name):
        return RemotePath(str(PurePosixPath(self.key) / name))

    @property
    def name(self): return PurePosixPath(self.key).name

    @property
    def stem(self): return PurePosixPath(self.key).stem

    @property
    def parent(self):
        value = str(PurePosixPath(self.key).parent)
        return RemotePath("" if value == "." else value)

    def with_suffix(self, suffix): return RemotePath(str(PurePosixPath(self.key).with_suffix(suffix)))

    def with_name(self, name): return self.parent / name

    def __str__(self): return "convex://datasets/" + self.key

    def __lt__(self, other): return self.key < other.key

    def relative_to(self, _): return self.key

    def mkdir(self, **_): pass  # Convex has no directories to create.

    def exists(self):
        if self.key in {"", "prices", "fundamentals", "news", "models", "backtests", "paper_trades", "prediction_exports"}: return True
        return metadata(self.key) is not None

    def stat(self):
        value = metadata(self.key)
        if value is None: raise FileNotFoundError(str(self))
        return SimpleNamespace(st_mtime=value["modifiedAt"] / 1000, st_size=value["size"])

    def glob(self, pattern):
        prefix = self.key + "/" if self.key else ""
        return [RemotePath(row["key"]) for row in list_datasets(prefix)
                if "/" not in row["key"][len(prefix):] and fnmatch(row["key"][len(prefix):], pattern)]

    def read_bytes(self): return read_bytes(self.key)

    def write_bytes(self, content):
        write_bytes(self.key, content)
        return len(content)

    def read_text(self, encoding="utf-8"): return read_bytes(self.key).decode(encoding)

    def write_text(self, text, encoding="utf-8"):
        write_bytes(self.key, text.encode(encoding))
        return len(text)


def read_csv(path, **kwargs):
    if isinstance(path, RemotePath):
        return pd.read_csv(StringIO(path.read_text()), **kwargs)
    # Explicit paths are used by fixture-based tests,
    # never as a fallback for a missing or unavailable cloud dataset.
    return pd.read_csv(path, **kwargs)


def write_csv(frame, path, **kwargs):
    if isinstance(path, RemotePath):
        path.write_text(frame.to_csv(**kwargs))
    else:
        frame.to_csv(path, **kwargs)
