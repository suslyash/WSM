# TASK-002C: Provision the Pinned DEPART YOLOv8 Checkpoint and Run a Limited Real Extraction Audit

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-002c

Do not run full-dataset feature extraction, do not train any model, do not implement V2/prototypes, and do not inspect Test predictions or Test metrics.

## Goal

Provision the exact DEPART-referenced YOLOv8 human-body checkpoint from its pinned upstream source and run a small real-data extraction audit through the accepted TASK-002B pipeline.

This task is the first real end-to-end validation of:

    real video
      -> uniform temporal sampling
      -> YOLOv8 human-body ROI
      -> frozen CLIP
      -> [T,D] temporal cache artifact + validity mask
      -> machine-readable coverage/failure report

The purpose is to verify that the real detector checkpoint, real local videos, CLIP runtime, temporal masking, and cache serialization work together before any full-cache extraction.

## Pinned Detector Source

Use exactly:

- upstream repository: J3lly-Been/YOLOv8-HumanDetection
- upstream commit containing the checkpoint: ce2aae2e821100aee58ce2e7f75994a7e6c2ab9e
- file: best.pt
- Git blob SHA: afa44d4fcd0ff54691912bf8960d4fbb98ae1278

Do not use a different checkpoint and do not use a floating branch as the provenance record.

The checkpoint binary itself must NOT be committed to the WSM repository.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 7-10 and 13-14
3. Stage 2 in docs/PLAN.md
4. docs/PROGRESS_EN.md through TASK-002B
5. docs/NEXT_TASK_EN.md
6. src/video/features/yolov8_body_roi.py
7. src/video/features/clip_video_features.py
8. scripts/video/extract_clip_video_features.py
9. src/common/data/wsm_manifest.py

## Allowed Tracked Files

- scripts/video/audit_depart_real_extraction.py
- docs/PROGRESS_EN.md

No other tracked file may be modified.

Untracked/local artifacts are allowed only under:

    /media/maxim/Programs/Models/WSM/depart_yolov8/
    /tmp/wsm_depart_002c/

Do not commit model weights or cache artifacts.

## Explicitly Authorized External Action

You are authorized in this task to download the pinned upstream best.pt checkpoint from the repository/commit above to:

    /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt

Do not install dependencies.

Do not download any other model weights.

If the file already exists:

- compute its SHA-256;
- verify its provenance if possible;
- do not overwrite it unless it is byte-identical to the pinned source or the task explicitly records why replacement is required.

## Implementation Requirements

### 1. Audit helper

Create scripts/video/audit_depart_real_extraction.py as a small orchestration/audit helper around the accepted extractor.

Required arguments:

    --data-root
    --yolo-weights
    --cache-root
    --report-output
    --per-split

Optional:

    --device
    --model-name
    --model-revision
    --target-frames
    --overwrite

Defaults:

    --per-split 3
    --target-frames 60
    --model-name openai/clip-vit-base-patch32
    --model-revision main

The helper must:

- build the canonical manifest in memory;
- select a deterministic small structural sample from train and dev only;
- select exactly N rows per split when enough eligible rows exist;
- do not use Test rows in this task;
- preserve deterministic ordering, e.g. sorted by segment_id before taking the first N;
- call the real TASK-002B extraction pipeline, not a mock;
- write cache artifacts under the supplied cache root;
- write one JSON audit report.

### 2. Real extraction sample policy

Use:

- train: 3 real video segments;
- dev: 3 real video segments;
- Test: 0.

Total target sample: 6 real segments.

If fewer than 3 eligible rows exist in a split, fail clearly rather than silently changing the sample definition.

No labels, diagnosis values, corpus identity, or task masks may be passed to YOLO or CLIP.

### 3. Checkpoint provenance record

The JSON audit must record:

- upstream repository;
- upstream commit;
- upstream file path;
- Git blob SHA;
- local checkpoint path;
- local checkpoint file size;
- local checkpoint SHA-256;
- ultralytics package version;
- CLIP/transformers package version;
- torch version;
- WSM git commit SHA/dirty flag if obtainable.

Do not claim the local SHA-256 is authoritative until computed from the downloaded file.

### 4. Real cache validation

For every successful segment artifact verify:

- cache file exists;
- features is rank 2;
- valid_mask is bool rank 1;
- features.shape[0] == valid_mask.shape[0];
- temporal positions preserve sampled-frame order;
- invalid masked positions are exactly zero;
- valid positions are finite;
- detector weights SHA-256 in artifact equals the local checkpoint SHA-256;
- detector settings equal conf=0.5, IoU=0.5, imgsz=640;
- target_frames requested is 60;
- cache fingerprint exists and is unique among the sampled segments.

Important preflight:

- if any selected source video yields fewer than 60 sampled temporal positions because the current sampler returns frame_count positions for short videos, record this as a blocker and STOP without modifying the extractor in this task;
- do not silently change sampling semantics here.

### 5. Audit report

The JSON report must contain at least:

- schema/version;
- pinned checkpoint provenance;
- manifest fingerprint;
- deterministic selected segment IDs by split;
- requested target_frames;
- per-segment:
  - split;
  - success/failure;
  - failure category/reason;
  - source frame count;
  - temporal length;
  - valid detection count;
  - detection coverage;
  - cache path if successful;
  - cache fingerprint if successful;
- aggregate:
  - selected count;
  - success count;
  - failure count;
  - no-body-detected count;
  - other extraction failure count;
  - mean/min/max detection coverage over successful segments;
  - all_temporal_lengths_equal_60;
- test_usage:
  - test_rows_processed=false;
  - model_predictions_inspected=false;
  - performance_metrics_inspected=false;
  - selection_or_tuning_performed=false.

### 6. No full extraction

The helper must be structurally incapable of processing the full manifest by default.

This task may process only the fixed small train/dev audit sample.

Do not reuse the generic full extractor with no limit.

## Acceptance Criteria

- pinned best.pt is downloaded or already present locally;
- pinned source repo/commit/path/blob are recorded;
- local SHA-256 is computed and recorded;
- real ultralytics detector runs on real videos;
- real frozen CLIP path runs on real body crops;
- exactly 3 train + 3 dev rows are attempted;
- zero Test rows are processed;
- at least one successful real cache artifact is produced; if zero succeed, task is partial/blocked and must not claim pass;
- every successful artifact passes structural validation;
- any short-video T<60 issue is reported as a blocker, not silently fixed;
- no full-dataset extraction;
- no training/model selection;
- no Test predictions/metrics;
- src/audio unchanged;
- python compilation passes;
- git diff --check passes;
- branch codex/task-002c is committed and pushed;
- main/master is untouched by implementing Codex;
- tracked diff contains only the two allowed paths.

## Exact Verification Commands

Run from repository root.

    python3 -m py_compile scripts/video/audit_depart_real_extraction.py

Provision the pinned checkpoint to:

    /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt

Record:

    sha256sum /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt

Then run only the small real audit:

    rm -rf /tmp/wsm_depart_002c

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
      scripts/video/audit_depart_real_extraction.py \
      --data-root /media/maxim/Databases/WSM_NEW \
      --yolo-weights /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt \
      --cache-root /tmp/wsm_depart_002c/cache \
      --report-output /tmp/wsm_depart_002c/report.json \
      --per-split 3 \
      --target-frames 60 \
      --device cuda

If CUDA is unavailable, rerun with --device cpu and record that deviation.

Validate the report:

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
    import json
    from pathlib import Path

    report = json.loads(Path("/tmp/wsm_depart_002c/report.json").read_text(encoding="utf-8"))

    assert report["selected_count"] == 6
    assert report["selected_by_split"]["train"] == 3
    assert report["selected_by_split"]["dev"] == 3
    assert report["test_usage"]["test_rows_processed"] is False
    assert report["test_usage"]["model_predictions_inspected"] is False
    assert report["test_usage"]["performance_metrics_inspected"] is False
    assert report["test_usage"]["selection_or_tuning_performed"] is False

    assert report["checkpoint"]["upstream_repository"] == "J3lly-Been/YOLOv8-HumanDetection"
    assert report["checkpoint"]["upstream_commit"] == "ce2aae2e821100aee58ce2e7f75994a7e6c2ab9e"
    assert report["checkpoint"]["upstream_path"] == "best.pt"
    assert report["checkpoint"]["git_blob_sha"] == "afa44d4fcd0ff54691912bf8960d4fbb98ae1278"
    assert len(report["checkpoint"]["local_sha256"]) == 64

    if report["success_count"] > 0:
        assert report["all_success_artifacts_valid"] is True

    print("TASK-002C limited real extraction audit assertions passed")
    PY

Then:

    git diff --check
    git diff -- src/audio
    git status --short

Before committing inspect only:

    git diff -- \
      scripts/video/audit_depart_real_extraction.py \
      docs/PROGRESS_EN.md

After commit/push:

    git status --short
    git rev-parse --abbrev-ref HEAD
    git rev-parse HEAD
    git diff --stat origin/main...HEAD
    git diff --name-only origin/main...HEAD

## Required PROGRESS_EN Update

Append TASK-002C facts without erasing previous evidence.

Record:

- branch;
- implementation commit SHA;
- push result;
- pinned upstream checkpoint repository/commit/path/blob;
- local checkpoint SHA-256 and file size;
- whether CUDA or CPU was used;
- exact six selected train/dev segment IDs;
- per-segment success/failure and detection coverage;
- temporal lengths;
- cache artifact structural validation;
- whether any T<60 short-video blocker appeared;
- exact commands/results;
- confirmation no Test rows were processed;
- confirmation no full extraction ran;
- confirmation src/audio unchanged;
- confirmation no training/model selection/Test metrics;
- Stage 2 remains partial;
- recommended next atomic step only.

## Required Handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch codex/task-002c;
- implementation commit SHA;
- pushed-to-origin status;
- main/master untouched;
- diff summary against origin/main;
- local YOLO checkpoint SHA-256;
- six attempted train/dev segments;
- success/failure counts;
- detection coverage summary;
- whether any short-video T<60 blocker appeared;
- src/audio unchanged;
- no Test rows/metrics;
- no full extraction/training.

Stop after this task.
