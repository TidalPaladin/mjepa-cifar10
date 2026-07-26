from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Final, Iterable


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_key TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    local_path TEXT,
    wandb_id TEXT,
    wandb_url TEXT,
    config_sha256 TEXT,
    config_json TEXT,
    summary_json TEXT,
    history_json TEXT,
    packages_json TEXT,
    checkpoint_available INTEGER NOT NULL,
    checkpoint_bytes INTEGER NOT NULL,
    backbone_available INTEGER NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""
WANDB_HISTORY_KEYS: Final = (
    "_step",
    "probe/validation_accuracy",
    "val/acc",
    "convergence/active_seconds",
    "sft/validation_accuracy",
    "sft/test_accuracy",
)


def _read_json_candidates(paths: Iterable[Path]) -> dict[str, Any]:
    for path in paths:
        if path.is_file():
            try:
                value = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(value, dict):
                return value
    return {}


def _config_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def open_inventory(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(SCHEMA)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
    for column in ("history_json", "packages_json"):
        if column not in columns:
            connection.execute(f"ALTER TABLE runs ADD COLUMN {column} TEXT")
    return connection


def index_local_runs(repo_root: Path, connection: sqlite3.Connection) -> int:
    logs_root = repo_root / "logs"
    count = 0
    if not logs_root.is_dir():
        return count
    for config_path in logs_root.rglob("config.yaml"):
        run_dir = config_path.parent.resolve()
        checkpoint = run_dir / "checkpoint.pt"
        backbone = run_dir / "backbone.safetensors"
        metadata = _read_json_candidates((run_dir / "metadata.json", run_dir / "wandb-metadata.json"))
        if not metadata:
            metadata = _read_json_candidates(run_dir.glob("wandb/**/files/wandb-metadata.json"))
        summary = _read_json_candidates(
            (
                run_dir / "summary.json",
                *run_dir.glob("wandb/**/files/wandb-summary.json"),
            )
        )
        history_path = run_dir / "metrics.jsonl"
        history = history_path.read_text().splitlines() if history_path.is_file() else []
        requirements = next(iter(run_dir.glob("wandb/**/files/requirements.txt")), None)
        packages = requirements.read_text().splitlines() if requirements is not None else []
        source = "managed" if (logs_root / "research").resolve() in run_dir.parents else "legacy"
        connection.execute(
            """
            INSERT INTO runs (
                run_key, source, local_path, wandb_id, wandb_url, config_sha256,
                config_json, summary_json, history_json, packages_json,
                checkpoint_available, checkpoint_bytes, backbone_available, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(run_key) DO UPDATE SET
                source=excluded.source,
                local_path=excluded.local_path,
                wandb_id=excluded.wandb_id,
                wandb_url=excluded.wandb_url,
                config_sha256=excluded.config_sha256,
                config_json=excluded.config_json,
                summary_json=excluded.summary_json,
                history_json=excluded.history_json,
                packages_json=excluded.packages_json,
                checkpoint_available=excluded.checkpoint_available,
                checkpoint_bytes=excluded.checkpoint_bytes,
                backbone_available=excluded.backbone_available,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                str(run_dir),
                source,
                str(run_dir),
                metadata.get("wandb_run_id"),
                metadata.get("wandb_url"),
                _config_hash(config_path),
                json.dumps({"copied_config": str(config_path), "yaml": config_path.read_text()}, sort_keys=True),
                json.dumps(summary, sort_keys=True),
                json.dumps(history),
                json.dumps(packages),
                int(checkpoint.is_file()),
                checkpoint.stat().st_size if checkpoint.is_file() else 0,
                int(backbone.is_file()),
            ),
        )
        count += 1
    connection.commit()
    return count


def index_wandb_runs(entity: str, project: str, connection: sqlite3.Connection) -> int:
    import wandb

    count = 0
    for run in wandb.Api().runs(f"{entity}/{project}"):
        run_key = f"wandb:{entity}/{project}/{run.id}"
        connection.execute(
            """
            INSERT INTO runs (
                run_key, source, local_path, wandb_id, wandb_url, config_sha256,
                config_json, summary_json, history_json, packages_json,
                checkpoint_available, checkpoint_bytes, backbone_available, updated_at
            ) VALUES (?, 'wandb', NULL, ?, ?, NULL, ?, ?, ?, ?, 0, 0, 0, CURRENT_TIMESTAMP)
            ON CONFLICT(run_key) DO UPDATE SET
                wandb_url=excluded.wandb_url,
                config_json=excluded.config_json,
                summary_json=excluded.summary_json,
                history_json=excluded.history_json,
                packages_json=excluded.packages_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                run_key,
                run.id,
                run.url,
                json.dumps(dict(run.config), default=str, sort_keys=True),
                json.dumps(dict(run.summary), default=str, sort_keys=True),
                json.dumps(
                    list(run.scan_history(keys=list(WANDB_HISTORY_KEYS))),
                    default=str,
                ),
                json.dumps(dict(getattr(run, "metadata", {}) or {}), default=str, sort_keys=True),
            ),
        )
        count += 1
    connection.commit()
    return count


def inventory_counts(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        source: count
        for source, count in connection.execute("SELECT source, COUNT(*) FROM runs GROUP BY source").fetchall()
    }
