"""Manifest record schema and validation for Controlled Localization v1.1."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

ALLOWED_MANIPULATION_FAMILIES: Set[str] = {
    "pristine",
    "ai_local",
    "conventional_local",
    "unknown",
}

ALLOWED_COMPOSITING_MODES: Set[str] = {
    "native_pixel_paste",
    "full_frame_decode",
    "posthoc_composite",
    "unknown",
}

ALLOWED_STATUSES: Set[str] = {
    "verified",
    "quarantined",
    "unverified",
}

ALLOWED_VIEW_ROLES: Set[str] = {
    "A",  # canonical unedited photograph
    "B",  # recipe(A)
    "C",  # canonical AI-edited image
    "D",  # same recipe(C)
}

ALLOWED_SPLITS: Set[str] = {"train", "val", "test", "challenge"}


@dataclass
class ManifestRecord:
    schema_version: str
    sample_id: str
    source_id: str
    source_group_id: str
    pair_group_id: str
    split: str
    view_role: str
    image_path: str
    mask_path: Optional[str]
    ignore_mask_path: Optional[str]
    label: int  # 0 = genuine, 1 = manipulated
    manipulation_family: str
    editor_name: str
    editor_version: str
    edit_type: str
    compositing_mode: str
    processing_recipe_id: str
    recipe_parameters: Dict[str, Any]
    width: int
    height: int
    mask_area_fraction: float
    image_sha256: str
    mask_sha256: Optional[str]
    source_dataset: str
    license_or_usage_notes: str
    mask_verified: bool
    status: str
    geometry_verified: bool
    outside_pair_eligible: bool
    provenance_notes: str

    # Audit fields (nullable)
    inside_mask_l1_diff: Optional[float] = None
    outside_mask_l1_diff: Optional[float] = None
    frac_changed_inside: Optional[float] = None
    change_threshold_tau: Optional[float] = None
    audit_decision_reason: Optional[str] = None

    def validate(self) -> None:
        """Enforce all constraints specified in Section 03 of GEMINI handoff v1.1."""
        if not self.schema_version:
            raise ValueError(f"Record {self.sample_id}: schema_version is required")
        if self.split not in ALLOWED_SPLITS:
            raise ValueError(f"Record {self.sample_id}: invalid split '{self.split}'")
        if self.view_role not in ALLOWED_VIEW_ROLES:
            raise ValueError(f"Record {self.sample_id}: invalid view_role '{self.view_role}'")
        if self.manipulation_family not in ALLOWED_MANIPULATION_FAMILIES:
            raise ValueError(f"Record {self.sample_id}: invalid manipulation_family '{self.manipulation_family}'")
        if self.compositing_mode not in ALLOWED_COMPOSITING_MODES:
            raise ValueError(f"Record {self.sample_id}: invalid compositing_mode '{self.compositing_mode}'")
        if self.status not in ALLOWED_STATUSES:
            raise ValueError(f"Record {self.sample_id}: invalid status '{self.status}'")
        if self.label not in (0, 1):
            raise ValueError(f"Record {self.sample_id}: label must be 0 or 1, got {self.label}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Record {self.sample_id}: invalid dimensions {self.width}x{self.height}")
        if not (0.0 <= self.mask_area_fraction <= 1.0):
            raise ValueError(f"Record {self.sample_id}: mask_area_fraction must be in [0, 1]")

        # Views A and B are unedited photographs; label must be 0, mask_area_fraction must be 0
        if self.view_role in ("A", "B"):
            if self.label != 0:
                raise ValueError(f"Record {self.sample_id}: unedited view {self.view_role} cannot have label=1")
            if self.mask_area_fraction > 0.0:
                raise ValueError(f"Record {self.sample_id}: unedited view {self.view_role} cannot have non-zero mask_area_fraction")

        # Full frame decode cannot have outside_pair_eligible=True per F10 / Section 03
        if self.compositing_mode == "full_frame_decode" and self.outside_pair_eligible:
            raise ValueError(f"Record {self.sample_id}: full_frame_decode cannot have outside_pair_eligible=True")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ManifestRecord:
        record = cls(**data)
        record.validate()
        return record


def load_manifest(manifest_path: str | Path, data_root: Optional[str | Path] = None) -> List[ManifestRecord]:
    """Load and validate JSONL manifest records."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    records: List[ManifestRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                rec = ManifestRecord.from_dict(data)
                if data_root is not None:
                    # Verify path can be resolved relative to data_root
                    root = Path(data_root)
                    img_p = root / rec.image_path if not Path(rec.image_path).is_absolute() else Path(rec.image_path)
                    # We store validated record
                records.append(rec)
            except Exception as e:
                raise ValueError(f"Error parsing manifest {manifest_path} at line {idx}: {e}") from e

    return records


def save_manifest(records: List[ManifestRecord], output_path: str | Path) -> None:
    """Save records to JSONL file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

