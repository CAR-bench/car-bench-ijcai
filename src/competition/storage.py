"""Crash-safe filesystem storage for competition runs."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Iterable, TextIO


class RunStorageError(RuntimeError):
    """Raised for incompatible or corrupt persisted run state."""


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON through a same-directory temporary file and atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        try:
            directory_fd = os.open(path.parent, os.O_DIRECTORY)
        except (AttributeError, OSError):  # pragma: no cover - portability fallback
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RunStorageError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RunStorageError(f"Expected a JSON object in {path}")
    return data


class RunLock(AbstractContextManager["RunLock"]):
    """Exclusive non-blocking host lock for one run directory."""

    def __init__(self, run_dir: Path):
        self.path = run_dir / ".run.lock"
        self._handle: TextIO | None = None

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._handle.close()
            self._handle = None
            raise RunStorageError(
                f"Run is already controlled by another process: {self.path.parent}"
            ) from exc
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(str(os.getpid()))
        self._handle.flush()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


class RunStore:
    """Paths and validation for one competition run."""

    def __init__(self, run_dir: str | Path):
        self.root = Path(run_dir).resolve()
        self.manifest_path = self.root / "manifest.json"
        self.progress_path = self.root / "progress.json"
        self.private_dir = self.root / "private"
        self.units_dir = self.private_dir / "units"
        self.attempts_dir = self.private_dir / "attempts"
        self.exports_dir = self.root / "exports"

    def initialize(self, manifest: dict[str, Any], scenario_text: str) -> None:
        if self.root.exists() and any(self.root.iterdir()):
            raise RunStorageError(f"Run directory is not empty: {self.root}")
        for path in (
            self.units_dir,
            self.attempts_dir,
            self.private_dir / "logs",
            self.exports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o700)
        self.private_dir.chmod(0o700)
        self.exports_dir.chmod(0o755)
        atomic_write_json(self.manifest_path, manifest)
        scenario_path = self.root / "scenario.toml"
        scenario_path.write_text(scenario_text, encoding="utf-8")
        self.update_progress(
            {
                "schema_version": 2,
                "status": "created",
                "current_unit": None,
                "completed_units": 0,
                "expected_units": manifest["expected_units"],
                "attempt_budget_start": 0,
                "updated_at": manifest["created_at"],
            }
        )

    def manifest(self) -> dict[str, Any]:
        return load_json(self.manifest_path)

    def progress(self) -> dict[str, Any]:
        return load_json(self.progress_path)

    def update_progress(self, progress: dict[str, Any]) -> None:
        atomic_write_json(self.progress_path, progress)

    @staticmethod
    def _safe_component(value: str) -> str:
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise RunStorageError(f"Unsafe result path component: {value!r}")
        return value

    def unit_path(self, category: str, task_id: str, trial: int) -> Path:
        category = self._safe_component(category)
        task_id = self._safe_component(task_id)
        return self.units_dir / category / task_id / f"trial-{int(trial)}.json"

    def attempt_dir(self, category: str, task_id: str, trial: int) -> Path:
        category = self._safe_component(category)
        task_id = self._safe_component(task_id)
        return self.attempts_dir / category / task_id / f"trial-{int(trial)}"

    def attempt_records(self, category: str, task_id: str, trial: int) -> list[dict[str, Any]]:
        directory = self.attempt_dir(category, task_id, trial)
        if not directory.exists():
            return []
        records = []
        for path in sorted(directory.glob("attempt-*.json")):
            records.append(load_json(path))
        return records

    def next_attempt_number(self, category: str, task_id: str, trial: int) -> int:
        records = self.attempt_records(category, task_id, trial)
        return max((int(row.get("attempt", 0)) for row in records), default=0) + 1

    def write_attempt(self, record: dict[str, Any]) -> Path:
        unit = record["unit"]
        path = self.attempt_dir(
            unit["category"], unit["task_id"], int(unit["trial"])
        ) / f"attempt-{int(record['attempt']):04d}.json"
        if path.exists():
            existing = load_json(path)
            if existing == record:
                return path
            raise RunStorageError(f"Attempt record already exists with different data: {path}")
        atomic_write_json(path, record)
        return path

    def write_unit(self, record: dict[str, Any]) -> Path:
        unit = record["unit"]
        path = self.unit_path(unit["category"], unit["task_id"], int(unit["trial"]))
        self._validate_unit_record(
            record,
            str(unit["category"]),
            str(unit["task_id"]),
            int(unit["trial"]),
            path,
        )
        if path.exists():
            existing = load_json(path)
            if existing == record:
                return path
            raise RunStorageError(f"Completed unit already exists with different data: {path}")
        atomic_write_json(path, record)
        return path

    def load_unit(self, category: str, task_id: str, trial: int) -> dict[str, Any] | None:
        path = self.unit_path(category, task_id, trial)
        if not path.exists():
            return None
        record = load_json(path)
        self._validate_unit_record(record, category, task_id, trial, path)
        return record

    def _validate_unit_record(
        self,
        record: dict[str, Any],
        category: str,
        task_id: str,
        trial: int,
        path: Path,
    ) -> None:
        unit = record.get("unit")
        invalid_identity = (
            record.get("schema_version") != 2
            or record.get("status") != "completed"
            or not isinstance(unit, dict)
            or unit.get("category") != category
            or unit.get("task_id") != task_id
            or unit.get("trial") != int(trial)
        )
        if invalid_identity:
            raise RunStorageError(f"Invalid completed unit record: {path}")
        assert isinstance(unit, dict)
        expected_fingerprint = self.manifest().get("dataset", {}).get("fingerprint")
        required_types: dict[str, type[Any] | tuple[type[Any], ...]] = {
            "attempt_count": int,
            "trial_seed": int,
            "started_at": str,
            "completed_at": str,
            "duration_seconds": (int, float),
            "reward": (int, float),
            "passed": bool,
            "trajectory": list,
            "timing": dict,
            "telemetry": dict,
            "evaluator": dict,
        }
        if unit.get("dataset_fingerprint") != expected_fingerprint:
            raise RunStorageError(f"Completed unit has the wrong dataset fingerprint: {path}")
        if any(
            not isinstance(record.get(key), expected_type)
            for key, expected_type in required_types.items()
        ):
            raise RunStorageError(f"Completed unit is missing required schema fields: {path}")
        if (
            isinstance(record.get("reward"), bool)
            or isinstance(record.get("duration_seconds"), bool)
            or record["attempt_count"] < 1
            or record["duration_seconds"] < 0
            or record.get("error") is not None
        ):
            raise RunStorageError(f"Completed unit has invalid values: {path}")

    def iter_units(self) -> Iterable[dict[str, Any]]:
        if not self.units_dir.exists():
            return []
        records = []
        for path in sorted(self.units_dir.glob("*/*/trial-*.json")):
            record = load_json(path)
            unit = record.get("unit", {})
            expected = self.unit_path(
                str(unit.get("category", "")),
                str(unit.get("task_id", "")),
                int(unit.get("trial", -1)),
            )
            if expected != path:
                raise RunStorageError(f"Unit identity does not match path: {path}")
            self._validate_unit_record(
                record,
                str(unit.get("category", "")),
                str(unit.get("task_id", "")),
                int(unit.get("trial", -1)),
                path,
            )
            records.append(record)
        return records

    def iter_attempts(self) -> Iterable[dict[str, Any]]:
        if not self.attempts_dir.exists():
            return []
        records = []
        for path in sorted(self.attempts_dir.glob("*/*/trial-*/attempt-*.json")):
            record = load_json(path)
            unit = record.get("unit")
            if not isinstance(unit, dict):
                raise RunStorageError(f"Attempt record has no unit identity: {path}")
            expected_dir = self.attempt_dir(
                str(unit.get("category", "")),
                str(unit.get("task_id", "")),
                int(unit.get("trial", -1)),
            )
            if expected_dir != path.parent:
                raise RunStorageError(f"Attempt identity does not match path: {path}")
            records.append(record)
        return records
