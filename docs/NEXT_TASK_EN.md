# TASK-002H: Recover Short-Video V1 Cache Rows with Variable-Length Temporal Sequences

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-002h

Do not process Test rows, do not train a model, do not create a training config, do not implement V2/prototypes, and do not inspect Test predictions or Test metrics.

## Goal

Apply the manager-approved variable-length V1 cache policy to recover the short-video rows rejected in TASK-002G.

Manager decision:

- real sampled sequence length may be any integer 1 <= T <= 60;
- do not duplicate frames;
- do not interpolate/synthesize temporal positions;
- do not change sampling semantics;
- preserve chronological real frame indices;
- later batching may pad with video_mask=false, but this task must not implement the DataModule yet.

The 96 no_body_detected rows remain genuine extraction failures.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 7-10 and 13-14
3. Stage 2 in docs/PLAN.md
4. docs/PROGRESS_EN.md through TASK-002G and MANAGER-DECISION-003
5. docs/NEXT_TASK_EN.md
6. scripts/video/extract_clip_video_features.py
7. scripts/video/audit_video_cache.py
8. src/video/features/clip_video_features.py
9. src/video/models/depart_v1.py

## Allowed Tracked Files

- scripts/video/extract_clip_video_features.py
- scripts/video/audit_video_cache.py
- docs/PROGRESS_EN.md

No other tracked file may be modified.

Do not modify src/video/features/clip_video_features.py or the V1 model in this task.

## Fixed Runtime Inputs

Pinned YOLO:

- /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt
- SHA-256:
  a6aead7bf0eccb35bd56731bfaa6ea19a4645a66150d2d0b19dd3fb1b116ef43

Pinned CLIP:

- openai/clip-vit-base-patch32
- revision:
  b97b0100e55e367c057773c2a614676470b0d575

Persistent cache root:

    /media/maxim/Programs/Features/WSM/video_depart_v1/cache

Requested splits:

    train,dev

Test rows:

    forbidden

## Implementation Requirements

### 1. Relax cache artifact validation to 1 <= T <= 60

Update both:

- scripts/video/extract_clip_video_features.py
- scripts/video/audit_video_cache.py

Successful artifact validation must require:

- features rank 2 [T,512];
- valid_mask bool [T];
- 1 <= T <= target_frames;
- features.shape[0] == valid_mask.shape[0];
- sampled_frame_indices length == T;
- sampled_frame_indices strictly chronological/nondecreasing according to the existing sampler contract;
- valid features finite;
- invalid positions exact zero;
- model/revision and detector metadata unchanged.

Do not require T == 60 anymore.

### 2. Preserve requested target_frames separately

The artifact preprocessing metadata must continue to record:

    target_frames = 60

This means "maximum/requested temporal positions", not a claim that every source contains 60 real frames.

Do not alter cache fingerprints solely because T is shorter. The expected fingerprint is still determined by the extraction contract and source identity.

### 3. Resume behavior

Use the existing resumable extractor.

When rerun with:

    --resume --overwrite

Behavior must be:

- valid existing 60-frame artifacts are reused;
- valid existing short artifacts under the new 1<=T<=60 contract are reused;
- the stale invalid artifact may be overwritten/re-extracted;
- rows with no existing valid artifact are extracted again;
- no_body_detected remains an explicit failed record;
- Test is not touched.

Do not delete the full cache root.

### 4. Independent audit semantics

Update audit_video_cache.py so it reports:

- success counts;
- failure counts;
- temporal-length distribution;
- min/max/mean T among successful artifacts;
- count of successful artifacts with T < 60;
- count with T == 60;
- no_body_detected failures;
- any other failures;
- Test indexed/processed = 0.

Set:

    complete_for_requested_splits=true

only when every requested TRAIN+DEV row has exactly one valid success/reused artifact or one explicit failure record.

### 5. Full resume execution

Rerun the complete TRAIN+DEV extractor against the existing persistent cache with:

    --splits train,dev
    --resume
    --overwrite

Do not use --limit.

This run is expected to reuse the 7,050 already-valid artifacts, recover the 111 short-video rows if body detection/CLIP succeeds, repair the one stale invalid cache if possible, and retain the 96 no-body rows as failures.

Do not assume the exact final success count in advance; record the measured result.

## Acceptance Criteria

Implementation:

- cache validation accepts [T,512] for every 1<=T<=60;
- bool valid_mask length matches T;
- exact-zero invalid positions preserved;
- no duplicated/synthetic frames added;
- requested target_frames metadata remains 60;
- audit reports temporal-length statistics;
- python compilation passes;
- git diff --check passes;
- src/audio unchanged.

Execution:

- requested splits exactly train,dev;
- Test rows processed/indexed=0;
- all 7,258 rows remain represented in the final index;
- missing_record_count=0;
- successful artifacts all pass variable-length validation;
- fingerprints remain unique;
- no_body failures remain explicit;
- recovered short-video successes are reported;
- stale invalid cache outcome is reported;
- independent audit passes with complete_for_requested_splits=true;
- no dependency install;
- no training/model selection/Test metrics.

Git:

- branch codex/task-002h;
- implementation committed/pushed;
- main/master untouched;
- tracked diff contains only the three allowed paths.

## Exact Verification Commands

Set environment:

    export HF_HOME=/media/maxim/Programs/Models/WSM/huggingface
    export HF_HUB_CACHE=/media/maxim/Programs/Models/WSM/huggingface/hub
    export YOLO_AUTOINSTALL=false

Compile:

    python3 -m py_compile       scripts/video/extract_clip_video_features.py       scripts/video/audit_video_cache.py

Run a synthetic variable-length validator smoke using temporary artifacts of T=17 and T=60, proving both pass and T=0 / T=61 fail.

Then rerun the full TRAIN+DEV cache in place:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python       scripts/video/extract_clip_video_features.py       --data-root /media/maxim/Databases/WSM_NEW       --cache-root /media/maxim/Programs/Features/WSM/video_depart_v1/cache       --report-output /media/maxim/Programs/Features/WSM/video_depart_v1/extraction_report_task002h.json       --splits train,dev       --resume       --overwrite       --yolo-weights /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt       --model-name openai/clip-vit-base-patch32       --model-revision b97b0100e55e367c057773c2a614676470b0d575       --target-frames 60       --device cuda

If CUDA is unavailable, CPU fallback is allowed only for a pure device/runtime issue and must be recorded.

Run independent audit:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python       scripts/video/audit_video_cache.py       --data-root /media/maxim/Databases/WSM_NEW       --cache-root /media/maxim/Programs/Features/WSM/video_depart_v1/cache       --report-output /media/maxim/Programs/Features/WSM/video_depart_v1/cache_audit_task002h.json       --splits train,dev       --model-name openai/clip-vit-base-patch32       --model-revision b97b0100e55e367c057773c2a614676470b0d575       --yolo-weights /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt       --target-frames 60

Validate at least:

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
    import json
    from pathlib import Path

    p = Path("/media/maxim/Programs/Features/WSM/video_depart_v1/cache_audit_task002h.json")
    r = json.loads(p.read_text(encoding="utf-8"))

    assert r["requested_splits"] == ["train", "dev"]
    assert r["expected_total"] == 7258
    assert r["missing_record_count"] == 0
    assert r["test_rows_indexed"] == 0
    assert r["test_rows_processed"] is False
    assert r["complete_for_requested_splits"] is True
    assert r["successful_artifacts_valid"] is True
    assert r["cache_fingerprints_unique"] is True
    assert 1 <= r["temporal_length_stats"]["min"] <= 60
    assert 1 <= r["temporal_length_stats"]["max"] <= 60
    assert r["temporal_length_stats"]["shorter_than_60_count"] >= 1
    assert r["test_usage"]["model_predictions_inspected"] is False
    assert r["test_usage"]["performance_metrics_inspected"] is False
    assert r["test_usage"]["selection_or_tuning_performed"] is False

    print("TASK-002H variable-length TRAIN+DEV cache audit passed")
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

Record:

- branch;
- implementation commit SHA;
- push result;
- manager variable-length decision;
- exact validator changes;
- extracted/reused/failure counts;
- recovered short-video count;
- no_body count;
- other failure count;
- stale-cache outcome;
- temporal-length distribution/stats;
- final success/failure counts by split;
- independent audit result;
- Test rows processed/indexed=0;
- no dependency install;
- no training/model selection/Test metrics;
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

- branch codex/task-002h;
- implementation commit SHA;
- pushed-to-origin status;
- main/master untouched;
- recovered short-video count;
- final success/failure counts;
- no_body count;
- temporal-length stats;
- cache audit path;
- Test rows processed/indexed=0;
- no training/Test metrics;
- src/audio unchanged.

Stop after this task.
