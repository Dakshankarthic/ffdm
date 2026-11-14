"""Audit data CLI for verifying masks, no-ops, and geometry in Controlled Localization v1.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from PIL import Image

from .data.schema import load_manifest, save_manifest, ManifestRecord
from .data.audit import audit_pair_and_mask
from .data.canonicalize import load_canonical_rgb


def audit_manifest(
    records: List[ManifestRecord],
    out_dir: Path,
    data_root: Optional[Path] = None,
    tau: float = 0.02,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    verified_count = 0
    quarantined_count = 0
    audit_results = []
    updated_records = []

    for rec in records:
        img_p = Path(rec.image_path)
        if not img_p.is_absolute() and data_root is not None:
            img_p = data_root / img_p

        img, _ = load_canonical_rgb(img_p)
        arr_img = np.array(img, dtype=np.float32) / 255.0

        arr_mask = None
        if rec.mask_path:
            m_p = Path(rec.mask_path)
            if not m_p.is_absolute() and data_root is not None:
                m_p = data_root / m_p
            with Image.open(m_p) as m_img:
                m_gray = m_img.convert("L").resize((rec.width, rec.height), resample=Image.NEAREST)
                arr_mask = np.array(m_gray, dtype=np.float32) / 255.0

        res = audit_pair_and_mask(
            canonical_a=arr_img,
            canonical_c=None,
            mask=arr_mask,
            sample_id=rec.sample_id,
            tau=tau,
        )

        rec.inside_mask_l1_diff = res.inside_mask_l1_diff
        rec.outside_mask_l1_diff = res.outside_mask_l1_diff
        rec.frac_changed_inside = res.frac_changed_inside
        rec.change_threshold_tau = res.change_threshold_tau
        rec.audit_decision_reason = res.decision_reason
        rec.status = res.status

        if res.status == "verified":
            verified_count += 1
        else:
            quarantined_count += 1

        audit_results.append({
            "sample_id": rec.sample_id,
            "status": res.status,
            "mask_fraction": res.mask_area_fraction,
            "reason": res.decision_reason,
        })
        updated_records.append(rec)

    # Save audited manifest
    save_manifest(updated_records, out_dir / "audited_manifest.jsonl")

    report = {
        "total_records": len(records),
        "verified_count": verified_count,
        "quarantined_count": quarantined_count,
        "tau": tau,
        "results": audit_results,
    }
    with open(out_dir / "audit_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit data manifest CLI")
    parser.add_argument("--manifest", type=str, required=True, help="Path to JSONL manifest")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--data-root", type=str, default=None, help="Root directory for relative paths")
    parser.add_argument("--tau", type=float, default=0.02, help="Difference threshold tau")
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    data_root = Path(args.data_root) if args.data_root else None
    out_dir = Path(args.out)

    report = audit_manifest(records, out_dir, data_root=data_root, tau=args.tau)
    print(f"Audited {report['total_records']} records: {report['verified_count']} verified, {report['quarantined_count']} quarantined.")


if __name__ == "__main__":
    main()

