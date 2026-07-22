import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import safetensors.torch as st
from torch import Tensor


def append_metric_record(log_dir: Path | None, step: int, metrics: Mapping[str, Any]) -> None:
    """Append a recoverable local metric record alongside W&B logging."""
    if log_dir is None:
        return
    record = {"_step": step, **metrics}
    with (log_dir / "metrics.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, default=_json_default, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def write_run_metadata(log_dir: Path | None, metadata: Mapping[str, Any]) -> None:
    if log_dir is None:
        return
    path = log_dir / "metadata.json"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=log_dir,
            prefix=".metadata.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            json.dump(metadata, output, default=_json_default, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def save_safetensors_atomic(path: Path, tensors: Mapping[str, Tensor]) -> None:
    """Write safetensors beside its destination, then replace atomically."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=path.suffix,
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
        st.save_file(dict(tensors), str(temporary_path))
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
