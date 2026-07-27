"""Loading and validation for organizer-provided hidden CAR-bench tasks."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


HIDDEN_DATASET_FILES = {
    "base": "hidden_base.csv",
    "hallucination": "hidden_hallucination.csv",
    "disambiguation": "hidden_disambiguation.csv",
}

ALLOWED_TASK_TYPES = {
    "base": {"base"},
    "hallucination": {
        "hallucination_missing_tool",
        "hallucination_missing_tool_parameter",
        "hallucination_missing_tool_response",
    },
    "disambiguation": {
        "disambiguation_internal",
        "disambiguation_user",
    },
}

REQUIRED_COLUMNS = {
    "task_id",
    "calendar_id",
    "actions",
    "persona",
    "instruction",
    "context_init_config",
    "task_type",
}
TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class HiddenDatasetError(ValueError):
    """Raised when the hidden task dataset is absent or invalid."""


def _task_types():
    try:
        from car_bench.types import Action, Task
    except ImportError as exc:  # pragma: no cover - installation diagnostic
        raise HiddenDatasetError(
            "The CAR-bench evaluator dependency is required to validate hidden tasks. "
            "Install the car-bench-evaluator extra first."
        ) from exc
    return Action, Task


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_fingerprint(files: dict[str, dict[str, Any]]) -> str:
    payload = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class HiddenDataset:
    """Validated tasks and a content-addressed dataset manifest."""

    root: Path
    tasks_by_category: dict[str, list[Any]]
    manifest: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        return str(self.manifest["fingerprint"])

    def task_ids(self, category: str) -> list[str]:
        return [str(task.task_id) for task in self.tasks_by_category[category]]

    def index_for(self, category: str, task_id: str) -> int:
        for index, task in enumerate(self.tasks_by_category[category]):
            if str(task.task_id) == task_id:
                return index
        raise HiddenDatasetError(f"Unknown hidden task {category}/{task_id}")


def load_hidden_dataset(root: str | Path) -> HiddenDataset:
    """Load the three hidden CSVs and validate every row as a CAR-bench Task."""

    Action, Task = _task_types()
    dataset_root = Path(root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise HiddenDatasetError(f"Hidden dataset directory not found: {dataset_root}")

    tasks_by_category: dict[str, list[Any]] = {}
    file_manifest: dict[str, dict[str, Any]] = {}
    seen_task_ids: set[str] = set()

    for category, filename in HIDDEN_DATASET_FILES.items():
        path = dataset_root / filename
        if not path.is_file():
            raise HiddenDatasetError(f"Missing hidden dataset file: {path}")
        if path.is_symlink():
            raise HiddenDatasetError(f"Hidden dataset files must not be symlinks: {path}")

        parsed_tasks: list[Any] = []
        line_number = 1
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = set(reader.fieldnames or [])
                missing = REQUIRED_COLUMNS - columns
                if missing:
                    raise HiddenDatasetError(
                        f"{filename} is missing required columns: {sorted(missing)}"
                    )

                for line_number, row in enumerate(reader, start=2):
                    try:
                        task_type = str(row.get("task_type") or "")
                        if task_type not in ALLOWED_TASK_TYPES[category]:
                            raise HiddenDatasetError(
                                f"task_type {task_type!r} does not belong to {category}"
                            )

                        split = row.get("split")
                        if split not in (None, "", "hidden"):
                            raise HiddenDatasetError(
                                f"split must be 'hidden' when present, got {split!r}"
                            )

                        task_id = str(row.get("task_id") or "").strip()
                        if not task_id:
                            raise HiddenDatasetError("task_id must not be empty")
                        if not TASK_ID_RE.fullmatch(task_id):
                            raise HiddenDatasetError(
                                f"task_id {task_id!r} is unsafe for durable result paths"
                            )
                        if task_id in seen_task_ids:
                            raise HiddenDatasetError(f"duplicate task_id {task_id!r}")

                        data = dict(row)
                        data["actions"] = [
                            Action(**action) for action in json.loads(data["actions"])
                        ]
                        data["context_init_config"] = json.loads(
                            data["context_init_config"]
                        )
                        removed_part = data.get("removed_part")
                        data["removed_part"] = (
                            json.loads(removed_part) if removed_part else None
                        )
                        task = Task(**data)
                    except HiddenDatasetError:
                        raise
                    except Exception as exc:
                        raise HiddenDatasetError(str(exc)) from exc

                    seen_task_ids.add(task_id)
                    parsed_tasks.append(task)
        except HiddenDatasetError as exc:
            raise HiddenDatasetError(f"{filename}:{line_number}: {exc}") from exc

        if not parsed_tasks:
            raise HiddenDatasetError(f"{filename} contains no tasks")

        tasks_by_category[category] = parsed_tasks
        file_manifest[category] = {
            "filename": filename,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "row_count": len(parsed_tasks),
            "task_ids": [str(task.task_id) for task in parsed_tasks],
        }

    manifest = {
        "schema_version": 1,
        "files": file_manifest,
    }
    manifest["fingerprint"] = _canonical_fingerprint(file_manifest)
    return HiddenDataset(dataset_root, tasks_by_category, manifest)


_ORIGINAL_TASK_LOADER: Callable[..., list[Any]] | None = None
_INSTALLED_DATASET: HiddenDataset | None = None


def install_hidden_task_loader(dataset: HiddenDataset, *, loader_module=None) -> None:
    """Install a process-local loader hook without changing public HF splits."""

    global _ORIGINAL_TASK_LOADER, _INSTALLED_DATASET
    target_module = loader_module
    if target_module is None:
        from car_bench.envs.car_voice_assistant import env as car_env

        target_module = car_env

    if _ORIGINAL_TASK_LOADER is None:
        _ORIGINAL_TASK_LOADER = target_module._load_tasks
    _INSTALLED_DATASET = dataset

    def load_tasks(task_type: str, task_split: str, repo_id: str | None = None):
        if task_split == "hidden":
            if task_type not in HIDDEN_DATASET_FILES:
                raise HiddenDatasetError(f"Unsupported hidden task category: {task_type}")
            return list(_INSTALLED_DATASET.tasks_by_category[task_type])
        assert _ORIGINAL_TASK_LOADER is not None
        if repo_id is None:
            return _ORIGINAL_TASK_LOADER(task_type, task_split)
        return _ORIGINAL_TASK_LOADER(task_type, task_split, repo_id)

    target_module._load_tasks = load_tasks
