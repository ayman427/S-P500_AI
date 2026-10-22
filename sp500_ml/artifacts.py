"""Publish immutable training generations through one atomic manifest."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import joblib
from io import BytesIO
from .cloud_data import RemotePath, exclusive_cloud_operation, write_csv


@contextmanager
def exclusive_file(path):
    if isinstance(path, RemotePath):
        with exclusive_cloud_operation(path.key):
            yield
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"Another operation holds {path.name}. Retry after it completes; inspect abandoned locks after a crash.") from error
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


def atomic_csv(frame, path):
    if isinstance(path, RemotePath):
        write_csv(frame, path, index=False)
        return
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_generation(model_dir, prefix):
    manifest_path = model_dir / f"{prefix}_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = manifest["run_id"]
        if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
            raise ValueError("Invalid run identifier")
        folder = model_dir / ".runs" / prefix / run_id
        names = [f"{prefix}_next_close.joblib", f"{prefix}_metrics.json", f"{prefix}_holdout.json", f"{prefix}_final_holdout_trades.csv"]
        paths = [folder / name for name in names]
        for path in paths:
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"][path.name]:
                raise ValueError(f"Checksum mismatch: {path.name}")
        return paths[0], paths[1], paths[3]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError(f"Model artifact integrity check failed for {prefix}: {error}") from error


def publish_generation(model_dir, prefix, bundle, metrics, holdout, trades):
    model_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_file(model_dir / f".{prefix}.publish.lock"):
        run_id = uuid4().hex
        folder = model_dir / ".runs" / prefix / run_id
        folder.mkdir(parents=True)
        bundle = {**bundle, "run_id": run_id}
        metrics = {**metrics, "run_id": run_id}
        holdout = {**holdout, "run_id": run_id}
        model_path = folder / f"{prefix}_next_close.joblib"
        metrics_path = folder / f"{prefix}_metrics.json"
        holdout_path = folder / f"{prefix}_holdout.json"
        trades_path = folder / f"{prefix}_final_holdout_trades.csv"
        if isinstance(model_dir, RemotePath):
            buffer = BytesIO()
            joblib.dump(bundle, buffer)
            model_path.write_bytes(buffer.getvalue())
            metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False))
            holdout_path.write_text(json.dumps(holdout, indent=2, allow_nan=False))
            write_csv(trades, trades_path, index=False, date_format="%Y-%m-%d")
            paths = [model_path, metrics_path, holdout_path, trades_path]
            manifest = {"run_id": run_id, "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
            # Only expose a generation after every uploaded member has been verified.
            (model_dir / f"{prefix}_manifest.json").write_text(json.dumps(manifest, indent=2))
            return model_path, metrics_path, holdout_path, trades_path
        joblib.dump(bundle, model_path)
        metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
        holdout_path.write_text(json.dumps(holdout, indent=2, allow_nan=False), encoding="utf-8")
        trades.to_csv(trades_path, index=False, date_format="%Y-%m-%d")
        paths = [model_path, metrics_path, holdout_path, trades_path]
        manifest = {"run_id": run_id, "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
        temporary = model_dir / f".{prefix}.{run_id}.tmp"
        try:
            temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            os.replace(temporary, model_dir / f"{prefix}_manifest.json")
        finally:
            temporary.unlink(missing_ok=True)
        return model_path, metrics_path, holdout_path, trades_path
