#!/usr/bin/env python3
"""Extract frozen V1 CLIP temporal features for video-available manifest rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common.data.wsm_manifest import build_manifest  # noqa: E402
from video.features.clip_video_features import (  # noqa: E402
    PREPROCESSING_VERSION,
    SAMPLING_METHOD,
    ClipVideoFeatureExtractor,
    build_cache_fingerprint,
    save_cache_artifact,
)

PROTECTED_NAMES = {"src", "configs", "docs"}
MANIFEST_COLUMNS = {
    "segment_id", "video_id", "speaker_id", "corpus", "split", "y_depression", "y_parkinson",
    "observed_depression", "observed_parkinson", "audio_available", "video_available",
    "text_available", "description_available",
}


def _refuse_protected(path: Path, option: str) -> None:
    resolved = path.resolve()
    if PROJECT_ROOT in resolved.parents and resolved.relative_to(PROJECT_ROOT).parts and resolved.relative_to(PROJECT_ROOT).parts[0] in PROTECTED_NAMES:
        raise ValueError(f"{option} may not write inside a protected project directory: {resolved}")


def _file_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = MANIFEST_COLUMNS - fields
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        return [dict(row) for row in reader]


def _source_path(data_root: Path, row: dict[str, str]) -> Path:
    corpus, video_id, segment_file = json.loads(row["segment_id"])
    source_split = row["split"]
    # The manifest's segment identity is authoritative; split only locates its source folder.
    return data_root / corpus / f"{source_split}_labels" / video_id / "segments" / segment_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--target-frames", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        _refuse_protected(args.cache_root, "--cache-root")
        _refuse_protected(args.report_output, "--report-output")
        if args.target_frames <= 0 or args.limit is not None and args.limit <= 0:
            raise ValueError("target-frames and limit must be positive")
        if args.manifest_path is None:
            rows, audit = build_manifest(args.data_root)
            manifest_path = Path("<canonical-manifest-built-from-data-root>")
            manifest_fingerprint = audit["manifest_fingerprint"]["value"]
        else:
            manifest_path = args.manifest_path
            rows = _manifest_rows(manifest_path)
            manifest_fingerprint = _file_fingerprint(manifest_path)
        candidates = [row for row in rows if row.get("video_available") is True or str(row.get("video_available", "")).lower() == "true"]
        if args.limit is not None:
            candidates = candidates[:args.limit]
        extractor = ClipVideoFeatureExtractor(
            model_name=args.model_name,
            model_revision=args.model_revision,
            target_frames=args.target_frames,
            device=args.device,
        )
        failures: list[dict[str, Any]] = []
        successes: list[dict[str, Any]] = []
        for row in candidates:
            segment_id = row["segment_id"]
            source = _source_path(args.data_root, row)
            fingerprint = build_cache_fingerprint(
                manifest_fingerprint=manifest_fingerprint,
                segment_id=segment_id,
                source_path=str(source),
                model_name=args.model_name,
                model_revision=args.model_revision,
                target_frames=args.target_frames,
                preprocessing_version=PREPROCESSING_VERSION,
                sampling_method=SAMPLING_METHOD,
            )
            artifact_path = args.cache_root / f"{fingerprint}.pt"
            if artifact_path.exists() and not args.overwrite:
                failures.append({"segment_id": segment_id, "reason": "cache_exists", "cache_path": str(artifact_path)})
                continue
            result = extractor.extract(source)
            if not result.success:
                failures.append({"segment_id": segment_id, "reason": result.error, "source_path": str(source)})
                continue
            save_cache_artifact(
                artifact_path,
                segment_id=segment_id,
                source_path=str(source),
                result=result,
                target_frames=args.target_frames,
                model_name=args.model_name,
                model_revision=args.model_revision,
                cache_fingerprint=fingerprint,
            )
            successes.append({"segment_id": segment_id, "cache_path": str(artifact_path), "cache_fingerprint": fingerprint})
        report = {
            "schema_version": "wsm-video-cache-report-v1",
            "manifest_path": str(manifest_path),
            "manifest_fingerprint": manifest_fingerprint,
            "cache_root": str(args.cache_root),
            "model_name": args.model_name,
            "model_revision": args.model_revision,
            "target_frames": args.target_frames,
            "preprocessing_version": PREPROCESSING_VERSION,
            "sampling_method": SAMPLING_METHOD,
            "limit": args.limit,
            "video_available_candidates": len(candidates),
            "success_count": len(successes),
            "failure_count": len(failures),
            "successes": successes,
            "failures": failures,
            "test_metrics_inspected": False,
            "labels_passed_to_encoder": False,
        }
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"success_count": len(successes), "failure_count": len(failures), "manifest_fingerprint": manifest_fingerprint}, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"extract_clip_video_features: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
