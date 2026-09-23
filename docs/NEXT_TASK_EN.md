# TASK-002G: Build the Complete Resumable V1 Video Feature Cache for TRAIN+DEV Only

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-002g

This task explicitly authorizes a full V1 video feature extraction over TRAIN+DEV only.

Do not process Test rows, do not train a model, do not select a model/checkpoint, do not implement V2/prototypes, and do not inspect Test predictions or Test metrics.

## Goal

Create the complete reproducible DEPART-like V1 feature cache needed for video-model training:

    canonical TRAIN+DEV rows
      -> deterministic 60-frame sampling
      -> pinned YOLOv8 body ROI
      -> pinned frozen CLIP ViT-B/32
      -> [60,512] temporal features + validity mask
      -> persistent cache + machine-readable extraction index/report

Expected canonical eligible counts under the currently audited manifest:

- train: 6325
- dev: 933
- total TRAIN+DEV: 7258
- Test processed: 0

This task must also make the extraction safely resumable so an interrupted long run does not require recomputing already valid artifacts.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 7-10 and 13-14
3. Stage 2 in docs/PLAN.md
4. docs/PROGRESS_EN.md through TASK-002F
5. docs/NEXT_TASK_EN.md
6. scripts/video/extract_clip_video_features.py
7. scripts/video/audit_depart_real_extraction.py
8. src/video/features/clip_video_features.py
9. src/video/features/yolov8_body_roi.py
10. src/common/data/wsm_manifest.py

## Allowed Tracked Files

- scripts/video/extract_clip_video_features.py
- scripts/video/audit_video_cache.py
- docs/PROGRESS_EN.md

No other tracked file may be modified.

Untracked/local cache/report artifacts are allowed only under:

    /media/maxim/Programs/Features/WSM/video_depart_v1/
    /tmp/wsm_video_002g/

Do not commit feature artifacts, model weights, or generated reports.

## Fixed Runtime Inputs

Pinned YOLO:

- path: /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt
- SHA-256:
  a6aead7bf0eccb35bd56731bfaa6ea19a4645a66150d2d0b19dd3fb1b116ef43

Pinned CLIP:

- repo: openai/clip-vit-base-patch32
- revision: b97b0100e55e367c057773c2a614676470b0d575
- HF_HOME: /media/maxim/Programs/Models/WSM/huggingface
- HF_HUB_CACHE: /media/maxim/Programs/Models/WSM/huggingface/hub

Extraction contract:

- target_frames=60
- YOLO confidence=0.5
- YOLO IoU=0.5
- YOLO imgsz=640
- ROI policy unchanged from TASK-002B
- no labels/task identity passed into YOLO/CLIP

## Implementation Requirements

### 1. Add explicit split filtering

Update scripts/video/extract_clip_video_features.py with:

    --splits train,dev

Requirements:

- parse a comma-separated subset of {train,dev,test};
- reject unknown split names;
- filter canonical rows before source-path resolution, video reading, detector inference, or CLIP inference;
- record requested splits and candidate counts per split in the report;
- preserve deterministic canonical row order;
- this task execution must use exactly train,dev;
- Test rows must never be read or processed in this task.

Default behavior may remain current all-split behavior for backwards compatibility, but the report must always state the resolved split set.

### 2. Add resumable extraction

Add:

    --resume

When --resume is supplied and a cache artifact for the exact expected fingerprint already exists:

- load and validate the artifact structurally;
- require matching segment_id;
- require matching cache_fingerprint;
- require features [60,512] for this task contract;
- require bool valid_mask [60];
- require finite valid features and exact-zero invalid features;
- require pinned detector SHA and extraction settings;
- require pinned CLIP model/revision;
- if valid, count it as reused_success and do not rerun YOLO/CLIP;
- if invalid/corrupt/mismatched, fail clearly for that row or re-extract only if --overwrite is explicitly supplied.

Do not count an existing valid artifact as a failure.

The extraction report must distinguish:

- extracted_success_count
- reused_success_count
- failure_count
- no_body_detected_count
- other_failure_count

### 3. Persistent cache index

Write a machine-readable JSONL or JSON index under the cache root, for example:

    cache_index.jsonl

Each selected TRAIN/DEV segment must have exactly one index record containing at least:

- segment_id
- split
- source_path
- status: extracted | reused | failed
- cache_path or null
- cache_fingerprint or null
- temporal_length
- feature_dim
- valid_detection_count
- detection_coverage
- failure_category/reason when failed

The index must contain no Test row for this task.

Write atomically or safely enough that an interrupted process does not leave a falsely complete record.

### 4. Add independent cache audit helper

Create scripts/video/audit_video_cache.py.

It must accept:

    --data-root
    --cache-root
    --report-output
    --splits train,dev
    --model-name
    --model-revision
    --yolo-weights
    --target-frames

It must not run YOLO/CLIP extraction.

It must:

- rebuild the canonical manifest;
- independently compute the expected selected TRAIN+DEV rows;
- inspect the cache index/artifacts;
- verify no Test row is indexed;
- verify each successful/reused artifact contract;
- verify unique cache fingerprints;
- report expected/success/failed/missing counts per split;
- report detection-coverage statistics by split and overall;
- report failure categories;
- record manifest fingerprint, YOLO SHA, CLIP revision, git state, and package versions;
- set complete_for_requested_splits=true only when every selected row has either:
  - a valid cache artifact, or
  - an explicit extraction-failure record.

The audit may report explicit extraction failures; it must not fabricate features for them.

### 5. Full TRAIN+DEV execution

After implementation verification, run the extractor across exactly:

    train,dev

Use persistent cache root:

    /media/maxim/Programs/Features/WSM/video_depart_v1/cache

Use persistent extraction report:

    /media/maxim/Programs/Features/WSM/video_depart_v1/extraction_report.json

Use resume mode.

Do not use --limit.

Do not process Test.

### 6. Failure handling

Extraction failures are allowed and must be preserved as explicit failure records.

Do not silently retry with:

- raw-frame-debug;
- different YOLO thresholds;
- different detector weights;
- different CLIP revision;
- fewer frames;
- CPU if CUDA failed for a semantic/model reason.

A CUDA availability/runtime issue MAY be rerun on CPU only if it is purely a device/runtime issue and all model/preprocessing parameters remain identical. Record the deviation.

### 7. No training

Do not create:

- video cache DataModule;
- training YAML;
- optimizer/callback changes;
- model training run.

Those are later tasks after cache coverage is audited.

## Acceptance Criteria

Implementation:

- --splits exists and filters before any Test source access;
- --resume validates and reuses exact valid artifacts;
- valid existing cache is not reported as failure;
- persistent index exists;
- independent audit helper exists;
- python compilation passes;
- git diff --check passes;
- src/audio unchanged.

Full extraction:

- requested splits exactly train,dev;
- expected rows train=6325, dev=933, total=7258;
- Test rows processed/indexed=0;
- pinned YOLO SHA unchanged;
- pinned CLIP revision unchanged;
- target_frames=60;
- every selected row is represented in the final index as success/reused/failure;
- successful artifacts validate as [60,512] with bool masks;
- invalid mask positions are exact zero;
- valid features are finite;
- cache fingerprints are unique;
- failures are explicitly categorized;
- independent audit completes successfully;
- complete_for_requested_splits=true;
- no full Test extraction;
- no training/model selection;
- no Test predictions/metrics;
- no dependency installation.

Git:

- branch codex/task-002g;
- implementation committed and pushed;
- main/master untouched;
- tracked diff contains only the three allowed paths.

## Exact Verification Commands

Set environment first:

    export HF_HOME=/media/maxim/Programs/Models/WSM/huggingface
    export HF_HUB_CACHE=/media/maxim/Programs/Models/WSM/huggingface/hub
    export YOLO_AUTOINSTALL=false

Verify pinned YOLO:

    sha256sum /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt

Expected:

    a6aead7bf0eccb35bd56731bfaa6ea19a4645a66150d2d0b19dd3fb1b116ef43

Compile:

    python3 -m py_compile       scripts/video/extract_clip_video_features.py       scripts/video/audit_video_cache.py

Check split-filter surface:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python       scripts/video/extract_clip_video_features.py --help

Run a tiny resumability smoke before the full extraction using a temporary cache and train-only limit:

    rm -rf /tmp/wsm_video_002g

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python       scripts/video/extract_clip_video_features.py       --data-root /media/maxim/Databases/WSM_NEW       --cache-root /tmp/wsm_video_002g/cache       --report-output /tmp/wsm_video_002g/first.json       --splits train       --limit 2       --resume       --yolo-weights /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt       --model-name openai/clip-vit-base-patch32       --model-revision b97b0100e55e367c057773c2a614676470b0d575       --target-frames 60       --device cuda

Run the identical command again with report-output /tmp/wsm_video_002g/second.json.

Assert the second run reports reused_success_count for valid first-run artifacts and does not process Test.

Then run the authorized complete TRAIN+DEV extraction:

    mkdir -p /media/maxim/Programs/Features/WSM/video_depart_v1

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python       scripts/video/extract_clip_video_features.py       --data-root /media/maxim/Databases/WSM_NEW       --cache-root /media/maxim/Programs/Features/WSM/video_depart_v1/cache       --report-output /media/maxim/Programs/Features/WSM/video_depart_v1/extraction_report.json       --splits train,dev       --resume       --yolo-weights /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt       --model-name openai/clip-vit-base-patch32       --model-revision b97b0100e55e367c057773c2a614676470b0d575       --target-frames 60       --device cuda

If CUDA is unavailable, CPU fallback is allowed only as described above.

Run the independent audit:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python       scripts/video/audit_video_cache.py       --data-root /media/maxim/Databases/WSM_NEW       --cache-root /media/maxim/Programs/Features/WSM/video_depart_v1/cache       --report-output /media/maxim/Programs/Features/WSM/video_depart_v1/cache_audit.json       --splits train,dev       --model-name openai/clip-vit-base-patch32       --model-revision b97b0100e55e367c057773c2a614676470b0d575       --yolo-weights /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt       --target-frames 60

Validate the audit:

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
    import json
    from pathlib import Path

    p = Path("/media/maxim/Programs/Features/WSM/video_depart_v1/cache_audit.json")
    r = json.loads(p.read_text(encoding="utf-8"))

    assert r["requested_splits"] == ["train", "dev"]
    assert r["expected_by_split"]["train"] == 6325
    assert r["expected_by_split"]["dev"] == 933
    assert r["expected_total"] == 7258
    assert r["test_rows_indexed"] == 0
    assert r["test_rows_processed"] is False
    assert r["complete_for_requested_splits"] is True
    assert r["missing_record_count"] == 0
    assert r["successful_artifacts_valid"] is True
    assert r["cache_fingerprints_unique"] is True
    assert r["detector_weights_sha256"] == (
        "a6aead7bf0eccb35bd56731bfaa6ea19a4645a66150d2d0b19dd3fb1b116ef43"
    )
    assert r["model_revision"] == "b97b0100e55e367c057773c2a614676470b0d575"
    assert r["target_frames"] == 60
    assert r["test_usage"]["model_predictions_inspected"] is False
    assert r["test_usage"]["performance_metrics_inspected"] is False
    assert r["test_usage"]["selection_or_tuning_performed"] is False

    print("TASK-002G TRAIN+DEV cache audit passed")
    PY

Finally:

    git diff --check
    git diff -- src/audio
    git status --short

Before commit inspect only:

    git diff --       scripts/video/extract_clip_video_features.py       scripts/video/audit_video_cache.py       docs/PROGRESS_EN.md

After commit/push:

    git status --short
    git rev-parse --abbrev-ref HEAD
    git rev-parse HEAD
    git diff --stat origin/main...HEAD
    git diff --name-only origin/main...HEAD

## Required PROGRESS_EN Update

Append TASK-002G facts without erasing prior evidence.

Record:

- branch;
- implementation commit SHA;
- push result;
- split-filter and resume semantics;
- persistent cache/index/report locations;
- pinned YOLO SHA and CLIP revision;
- requested splits;
- expected counts train/dev/total;
- extracted/reused/failure counts by split;
- failure categories;
- detection coverage stats by split and overall;
- cache artifact validation;
- independent audit result;
- confirmation Test rows processed/indexed=0;
- confirmation no dependency install;
- confirmation no training/model selection/Test metrics;
- src/audio unchanged;
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

- branch codex/task-002g;
- implementation commit SHA;
- pushed-to-origin status;
- main/master untouched;
- diff summary;
- expected TRAIN+DEV count;
- extracted/reused/failure counts;
- failure category summary;
- detection coverage summary;
- cache root and audit report path;
- Test rows processed/indexed=0;
- no training/Test metrics;
- src/audio unchanged.

Stop after this task.
