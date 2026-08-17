"""Validate human segmentation reviews and create a deterministic dataset gate."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

from src.data.segmentation_audit import IMAGE_EXTENSIONS, sha256_file

ALLOWED_REVIEW_DECISIONS = {"approved", "exclude", "needs_reannotation"}
ALLOWED_REVIEW_STATUSES = {"pending", "completed"}
LOCK_STATUSES = {"ready_for_split_generation", "blocked_by_manual_review"}
LOCK_MANIFESTS = (
    "consolidation_manifest.csv",
    "included_annotations.csv",
    "excluded_annotations.csv",
    "recovered_annotations.csv",
    "manual_review.csv",
    "mandatory_visual_review.csv",
    "duplicate_groups.csv",
    "image_normalization_manifest.csv",
)
REVIEW_QUEUE_COLUMNS = (
    "review_case_id",
    "source_dataset",
    "filename",
    "original_image_path",
    "original_label_path",
    "original_line_number",
    "original_class_id",
    "original_class_name",
    "reviewer_decision",
    "review_reason",
    "review_status",
)
APPLIED_REVIEW_COLUMNS = (
    "review_key",
    *REVIEW_QUEUE_COLUMNS,
    "origin",
)


class ReviewManifestError(ValueError):
    """Raised when a review manifest contains contradictory or invalid values."""


def read_review_manifest(path: Path) -> list[dict[str, str]]:
    """Read a review CSV while preserving blank human fields."""
    if not path.is_file():
        raise FileNotFoundError(f"Falta el manifiesto de revisión: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "source_dataset",
        "filename",
        "original_line_number",
        "original_class_id",
        "reviewer_decision",
        "review_reason",
        "review_status",
    }
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ReviewManifestError(
            f"{path.name} no contiene columnas requeridas: {sorted(missing)}"
        )
    return rows


def review_key(row: dict[str, str]) -> str:
    """Return the stable identity of one human review case."""
    return "|".join(
        (
            row["source_dataset"].strip(),
            row["filename"].strip(),
            row["original_line_number"].strip(),
            row["original_class_id"].strip(),
        )
    )


def review_case_id(row: dict[str, str]) -> str:
    """Return the public deterministic identifier for one review case."""
    digest = hashlib.sha256(review_key(row).encode("utf-8")).hexdigest()
    return f"review_{digest[:16]}"


def _normalized_review(row: dict[str, str], origin: str) -> dict[str, str]:
    return {
        "review_case_id": review_case_id(row),
        "review_key": review_key(row),
        "source_dataset": row["source_dataset"].strip(),
        "filename": row["filename"].strip(),
        "original_image_path": row.get("original_image_path", "").strip(),
        "original_label_path": row.get("original_label_path", "").strip(),
        "original_line_number": row["original_line_number"].strip(),
        "original_class_id": row["original_class_id"].strip(),
        "original_class_name": row.get("original_class_name", "").strip(),
        "reviewer_decision": row["reviewer_decision"].strip(),
        "review_reason": row["review_reason"].strip(),
        "review_status": row["review_status"].strip(),
        "origin": origin,
    }


def validate_review_manifests(
    mandatory_rows: Sequence[dict[str, str]],
    general_rows: Sequence[dict[str, str]],
) -> dict[str, object]:
    """Validate recorded human decisions without inferring missing values."""
    mandatory = [_normalized_review(row, "mandatory") for row in mandatory_rows]
    general = [_normalized_review(row, "general") for row in general_rows]
    all_rows = [*mandatory, *general]
    invalid: list[dict[str, str]] = []
    for row in all_rows:
        status = row["review_status"]
        decision = row["reviewer_decision"]
        reason = row["review_reason"]
        problems: list[str] = []
        if status not in ALLOWED_REVIEW_STATUSES:
            problems.append(f"review_status inválido: {status!r}")
        if decision and decision not in ALLOWED_REVIEW_DECISIONS:
            problems.append(f"reviewer_decision inválido: {decision!r}")
        if status == "completed" and not decision:
            problems.append("completed requiere reviewer_decision")
        if status == "completed" and not reason:
            problems.append("completed requiere review_reason")
        if decision and status != "completed":
            problems.append("reviewer_decision requiere review_status=completed")
        if problems:
            invalid.append({**row, "problems": "; ".join(problems)})

    by_key: dict[str, list[dict[str, str]]] = {}
    for row in all_rows:
        by_key.setdefault(row["review_key"], []).append(row)
    for key, duplicates in by_key.items():
        completed = [row for row in duplicates if row["review_status"] == "completed"]
        decisions = {
            (row["reviewer_decision"], row["review_reason"]) for row in completed
        }
        if len(decisions) > 1:
            raise ReviewManifestError(f"Decisiones humanas contradictorias para {key}")

    mandatory_by_key = {row["review_key"]: row for row in mandatory}
    unique: dict[str, dict[str, str]] = {}
    for row in general:
        unique[row["review_key"]] = row
    unique.update(mandatory_by_key)
    ordered = [unique[key] for key in sorted(unique)]
    completed = [
        row
        for row in ordered
        if row["review_status"] == "completed"
        and row["reviewer_decision"] in ALLOWED_REVIEW_DECISIONS
    ]
    pending = [row for row in ordered if row["review_status"] != "completed"]
    approved = [row for row in completed if row["reviewer_decision"] == "approved"]
    excluded = [row for row in completed if row["reviewer_decision"] == "exclude"]
    reannotation = [
        row for row in completed if row["reviewer_decision"] == "needs_reannotation"
    ]
    mandatory_pending = [
        row for row in mandatory if row["review_status"] != "completed"
    ]
    return {
        "mandatory_total": len(mandatory),
        "mandatory_completed": len(mandatory) - len(mandatory_pending),
        "mandatory_pending": mandatory_pending,
        "general_total": len(general),
        "general_completed": sum(
            row["review_status"] == "completed" for row in general
        ),
        "unique_total": len(ordered),
        "approved": approved,
        "excluded": excluded,
        "reannotation": reannotation,
        "pending": pending,
        "invalid": invalid,
        "decision_counts": dict(
            sorted(Counter(row["reviewer_decision"] for row in completed).items())
        ),
    }


def dataset_fingerprint(dataset_root: Path) -> dict[str, object]:
    """Hash the candidate pool and review-sensitive manifests deterministically."""
    paths = sorted(
        [
            path
            for path in (dataset_root / "all").rglob("*")
            if path.is_file()
            and (
                path.suffix.lower() in IMAGE_EXTENSIONS
                or path.suffix.lower() == ".txt"
            )
        ]
        + [
            path
            for path in (
                dataset_root / "dataset.yaml",
                *(
                    dataset_root / "manifests" / name
                    for name in LOCK_MANIFESTS
                ),
            )
            if path.is_file()
        ],
        key=lambda path: path.relative_to(dataset_root).as_posix(),
    )
    digest = hashlib.sha256()
    total_bytes = 0
    for path in paths:
        relative = path.relative_to(dataset_root).as_posix()
        file_digest = sha256_file(path)
        total_bytes += path.stat().st_size
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(file_digest.encode())
        digest.update(b"\n")
    return {
        "algorithm": "sha256",
        "file_count": len(paths),
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def count_candidate_dataset(dataset_root: Path) -> tuple[int, int]:
    """Count candidate images and non-empty segmentation rows."""
    images = [
        path
        for path in (dataset_root / "all" / "images").iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    labels = sorted((dataset_root / "all" / "labels").glob("*.txt"))
    if len(images) != len(labels):
        raise ReviewManifestError(
            f"Correspondencia imagen-etiqueta inválida: {len(images)} != {len(labels)}"
        )
    masks = 0
    for label in labels:
        lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line]
        if not lines:
            raise ReviewManifestError(f"Etiqueta vacía en el pool: {label}")
        masks += len(lines)
    return len(images), masks


def _public_review_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    columns = (
        "review_key",
        "source_dataset",
        "filename",
        "original_line_number",
        "original_class_id",
        "original_class_name",
        "reviewer_decision",
        "review_reason",
        "review_status",
        "origin",
    )
    return [{column: row.get(column, "") for column in columns} for row in rows]


def build_dataset_lock(
    dataset_root: Path,
    consolidation_summary: dict[str, object],
    eda_summary: dict[str, object],
    review_summary: dict[str, object],
    *,
    decisions_applied_from_sources: bool,
    lock_date: date | None = None,
) -> dict[str, object]:
    """Build the lock document; readiness requires reviews and a source rebuild."""
    image_count, mask_count = count_candidate_dataset(dataset_root)
    fingerprint = dataset_fingerprint(dataset_root)
    pending = review_summary["pending"]
    invalid = review_summary["invalid"]
    blockers: list[str] = []
    if pending:
        blockers.append(f"manual_reviews_pending={len(pending)}")
    if invalid:
        blockers.append(f"invalid_review_rows={len(invalid)}")
    if not decisions_applied_from_sources:
        blockers.append("decisions_not_rebuilt_from_original_sources")
    status = (
        "ready_for_split_generation"
        if not blockers
        else "blocked_by_manual_review"
    )
    if status not in LOCK_STATUSES:
        raise RuntimeError(f"Estado de lock inesperado: {status}")
    current_date = lock_date or date.today()
    source_after = consolidation_summary["source_fingerprint_after"]
    input_fingerprint = eda_summary["input_fingerprint"]
    return {
        "schema_version": 1,
        "dataset_version": (
            f"leaf-segmentation-{current_date.isoformat()}-{fingerprint['sha256'][:12]}"
        ),
        "fecha": current_date.isoformat(),
        "total_images": image_count,
        "total_masks": mask_count,
        "class_map": {"0": "maize_leaf"},
        "global_fingerprint": fingerprint,
        "source_fingerprints": {
            "source_tree_sha256": source_after["tree_sha256"],
            "source_file_count": source_after["file_count"],
            "source_total_bytes": source_after["total_bytes"],
            "eda_input_global_sha256": input_fingerprint["global_sha256"],
            "eda_input_file_count": input_fingerprint["file_count"],
            "sources_unchanged": bool(
                consolidation_summary["source_files_unchanged"]
                and eda_summary["input_files_unchanged"]
            ),
        },
        "approved_reviews": _public_review_rows(review_summary["approved"]),
        "excluded_reviews": _public_review_rows(review_summary["excluded"]),
        "reannotation_reviews": _public_review_rows(review_summary["reannotation"]),
        "pending_reviews": _public_review_rows(pending),
        "review_summary": {
            "mandatory_total": review_summary["mandatory_total"],
            "mandatory_completed": review_summary["mandatory_completed"],
            "mandatory_pending": len(review_summary["mandatory_pending"]),
            "general_total": review_summary["general_total"],
            "general_completed": review_summary["general_completed"],
            "unique_total": review_summary["unique_total"],
            "invalid_rows": len(invalid),
        },
        "pilot_leakage_count": int(
            consolidation_summary["counts"]["pilot_leakage"]
        ),
        "decisions_applied_from_original_sources": decisions_applied_from_sources,
        "blockers": blockers,
        "status": status,
        "splits_created": False,
        "training_performed": False,
    }


def write_dataset_lock(path: Path, lock: dict[str, object]) -> None:
    """Write a stable lock JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_reannotation_queue(
    path: Path,
    rows: Sequence[dict[str, str]],
) -> None:
    """Write the reannotation queue, including an explicit empty result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_QUEUE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: row.get(column, "") for column in REVIEW_QUEUE_COLUMNS}
            )


def write_applied_review_report(
    path: Path,
    rows: Sequence[dict[str, str]],
) -> None:
    """Write every valid completed human decision with its provenance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=APPLIED_REVIEW_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: row.get(column, "") for column in APPLIED_REVIEW_COLUMNS}
            )
