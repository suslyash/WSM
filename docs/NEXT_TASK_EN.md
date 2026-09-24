# TASK-002K2: Rebuild the DEPART-Compatible Full-Coverage Video Cache with Full-Frame Fallback

## Role

You are the implementing Codex. Execute only this corrective task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-002k2

Do not create a training YAML and do not run training in this task.

## Goal

Replace the superseded V1 cache policy with the released DEPART-compatible frame policy:

    sampled frame
      -> valid YOLO body box ? body ROI : full RGB frame
      -> frozen CLIP
      -> temporal feature

A YOLO detection miss is NOT an extraction failure.

The new cache used by future training/evaluation must cover the complete canonical dataset:

- train: 6325
- dev: 933
- test: 1364
- total: 8622

For fair Test comparison, TASK-002K2 passes only if all 1364 canonical Test rows have valid cache artifacts.

Target final coverage:

    train=6325/6325
    dev=933/933
    test=1364/1364
    total=8622/8622

If any Test row remains failed after the fallback, stop and report TASK-002K2 as blocked with the exact segment IDs and non-YOLO failure reasons.

## Authoritative DEPART Compatibility Decision

Released DEPART preprocessing behavior:

- sample up to 60 temporal frames;
- run YOLO body detection;
- when a valid body box exists, use the body crop;
- when no valid box exists, use the full RGB frame;
- do not discard the frame solely because YOLO missed;
- therefore do not discard a segment solely because all sampled frames missed YOLO.

This task changes only the ROI fallback/cache contract.

Keep the currently accepted pinned detector/model parameters unchanged in this corrective task:

- YOLO checkpoint SHA-256:
  a6aead7bf0eccb35bd56731bfaa6ea19a4645a66150d2d0b19dd3fb1b116ef43
- confidence=0.5
- IoU=0.5
- imgsz=640
- CLIP:
  openai/clip-vit-base-patch32
- CLIP revision:
  b97b0100e55e367c057773c2a614676470b0d575
- target_frames=60

Do not silently change detector confidence, tracking/predict mode, CLIP revision, frame sampling, or model architecture here.

Record in PROGRESS_EN.md that the released DEPART repository has additional implementation/config provenance differences that are outside this atomic corrective task.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 7-10, 13-14
3. Stage 2 in docs/PLAN.md
4. docs/PROGRESS_EN.md through MANAGER-DECISION-010
5. docs/NEXT_TASK_EN.md
6. src/video/features/clip_video_features.py
7. src/video/features/yolov8_body_roi.py
8. scripts/video/extract_clip_video_features.py
9. scripts/video/audit_video_cache.py
10. src/video/data/wsm_video_cache_datamodule.py

## Allowed Tracked Files

- src/video/features/clip_video_features.py
- scripts/video/extract_clip_video_features.py
- scripts/video/audit_video_cache.py
- src/video/data/wsm_video_cache_datamodule.py
- docs/PROGRESS_EN.md

No other tracked file may be modified.

Do not modify src/audio.

## New Cache Contract

Use a new cache root so the superseded zero-on-YOLO-miss cache remains preserved:

    /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache

Reports:

    /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/extraction_report_all_task002k2.json

    /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache_audit_all_task002k2.json

Do not overwrite or delete the old cache root:

    /media/maxim/Programs/Features/WSM/video_depart_v1/cache

## Implementation Requirements

### 1. Version the fallback policy

In src/video/features/clip_video_features.py, bump the preprocessing/ROI policy version so the new cache fingerprint is unambiguously distinct from the superseded policy.

The new version must explicitly encode the semantics:

    body ROI if detected, otherwise full RGB frame

The cache fingerprint must include the new preprocessing/ROI policy version as it already includes policy metadata.

### 2. Full-frame fallback per sampled frame

For every sampled frame:

- call the configured YOLO detector;
- if a valid body detection exists:
  - crop that ROI;
  - record selected box/confidence;
  - mark frame source as body_roi;
- otherwise:
  - pass the complete RGB frame to CLIP;
  - selected box/confidence remain null;
  - mark frame source as full_frame_fallback.

All readable sampled frames must produce CLIP features.

Do not skip a sampled frame because YOLO returned no body.

Do not return failure_kind=no_body_detected.

### 3. Temporal mask semantics

Because each readable sampled frame now has either ROI or full-frame CLIP input:

- output features shape remains [T,512], 1<=T<=60;
- valid_mask must be all true for every readable sampled temporal position;
- there must be no zero placeholder solely for a YOLO miss;
- short videos retain their real T without frame duplication;
- padding remains a DataModule/collate responsibility.

A genuine video-read/model failure may still fail extraction.

### 4. Detection/fallback provenance

Artifact/report/index metadata must distinguish detection from feature validity.

Record at least:

- detected_body_count;
- full_frame_fallback_count;
- detection_coverage = detected_body_count / T;
- fallback_coverage = full_frame_fallback_count / T;
- per-frame source list or equivalent auditable representation:
  - body_roi
  - full_frame_fallback

Require:

    detected_body_count + full_frame_fallback_count == T

Do not misuse valid_mask to represent YOLO detection coverage.

### 5. Resume/index semantics

Use the new cache root.

Existing old-cache artifacts must NOT be reused because their preprocessing/fingerprint contract is superseded.

Within the new cache root, --resume must reuse only artifacts matching the new fingerprint/policy.

The final index must contain exactly one record for every canonical segment_id across train, dev, test.

### 6. Rebuild the full canonical cache

Run one full extraction over:

    --splits train,dev,test

with no --limit.

Expected selected counts:

    train=6325
    dev=933
    test=1364
    total=8622

Use the new cache root.

### 7. Strict full-coverage acceptance

The required final outcome is:

    success train = 6325
    success dev   = 933
    success test  = 1364
    success total = 8622

and:

    failure total = 0
    missing total = 0

In particular:

    test success = 1364
    test failure = 0

If a non-YOLO fatal error remains:

- do not fabricate features;
- record the segment ID and exact reason;
- TASK-002K2 is blocked if any Test row fails;
- do not proceed to training integration.

### 8. Update independent audit

scripts/video/audit_video_cache.py must validate the new policy:

For every successful artifact:

- features [T,512], 1<=T<=60;
- valid_mask bool [T] and all true;
- finite features;
- chronological sampled indices;
- new preprocessing/ROI policy version;
- detected_body_count + full_frame_fallback_count == T;
- fallback coverage consistent with counts;
- selected_boxes null exactly where full-frame fallback was used, when applicable;
- pinned model/detector metadata correct;
- fingerprint matches new contract.

Report by split:

- expected;
- success;
- failure;
- body detections;
- full-frame fallbacks;
- number of segments with >=1 fallback;
- mean/min/max detection coverage;
- mean/min/max fallback coverage.

### 9. Repoint the V1 DataModule

Change the default V1 video cache root to:

    /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache

The DataModule must require the new preprocessing/ROI policy metadata.

Under the accepted full cache, expected datasets become:

    train_dataset = 6325
    val_dataset = 933

Raw Test protocol membership stays:

    test_none = 1364
    test_soft = 1208
    test_hard = 1014

Because full Test cache coverage is required, valid dataset lengths must equal those raw protocol counts:

    len(test_none) = 1364
    len(test_soft) = 1208
    len(test_hard) = 1014

Unavailable counts must all be zero:

    train_unavailable = 0
    dev_unavailable = 0
    test_none_unavailable = 0
    test_soft_unavailable = 0
    test_hard_unavailable = 0

### 10. Preserve sparse target semantics

This preprocessing correction must not change:

- two-logit disease target order;
- observed_mask;
- unknown NaN labels;
- split membership;
- soft/hard Test membership;
- DEV selector semantics.

## Acceptance Criteria

Code:

- new versioned ROI/full-frame fallback contract implemented;
- no_body_detected is no longer a failure mode for normal V1 extraction;
- every readable sampled frame is encoded;
- valid_mask is all true for extracted real positions;
- detection and fallback provenance are separately auditable;
- cache fingerprint changes with the new policy;
- old cache is preserved.

Full extraction:

- train expected/success = 6325/6325;
- dev expected/success = 933/933;
- test expected/success = 1364/1364;
- total expected/success = 8622/8622;
- failures = 0;
- missing = 0;
- Test failures = 0.

Audit:

- complete_for_requested_splits=true;
- all artifacts valid;
- fingerprints unique;
- fallback accounting valid;
- Test rows processed/indexed=1364.

DataModule:

- train=6325;
- dev=933;
- test_none=1364;
- test_soft=1208;
- test_hard=1014;
- every unavailable count=0;
- Test streams remain separate from DEV;
- variable-length collate still works.

Safety:

- no training;
- no Test performance metrics/predictions inspected;
- no model/checkpoint selection;
- no dependency install;
- src/audio unchanged;
- python compilation passes;
- git diff --check passes;
- branch codex/task-002k2 committed and pushed;
- main/master untouched;
- tracked diff contains only the five allowed paths.

## Exact Verification Commands

Set environment:

    export HF_HOME=/media/maxim/Programs/Models/WSM/huggingface
    export HF_HUB_CACHE=/media/maxim/Programs/Models/WSM/huggingface/hub
    export YOLO_AUTOINSTALL=false

Compile:

    python3 -m py_compile \
      src/video/features/clip_video_features.py \
      scripts/video/extract_clip_video_features.py \
      scripts/video/audit_video_cache.py \
      src/video/data/wsm_video_cache_datamodule.py

Run a focused fallback smoke with injected detector behavior:

- one frame with detection -> body ROI;
- one frame with no detection -> full RGB frame;
- both temporal positions must produce finite features;
- valid_mask == [true,true];
- detected_body_count == 1;
- full_frame_fallback_count == 1;
- detection_coverage == 0.5;
- fallback_coverage == 0.5;
- no failure.

Fingerprint smoke:

- old ROI-only policy version fingerprint != new fallback-policy fingerprint.

Create the new output root:

    mkdir -p /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback

Run complete extraction:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
      scripts/video/extract_clip_video_features.py \
      --data-root /media/maxim/Databases/WSM_NEW \
      --cache-root /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache \
      --report-output /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/extraction_report_all_task002k2.json \
      --splits train,dev,test \
      --resume \
      --yolo-weights /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt \
      --model-name openai/clip-vit-base-patch32 \
      --model-revision b97b0100e55e367c057773c2a614676470b0d575 \
      --target-frames 60 \
      --device cuda

Run independent audit:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
      scripts/video/audit_video_cache.py \
      --data-root /media/maxim/Databases/WSM_NEW \
      --cache-root /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache \
      --report-output /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache_audit_all_task002k2.json \
      --splits train,dev,test \
      --model-name openai/clip-vit-base-patch32 \
      --model-revision b97b0100e55e367c057773c2a614676470b0d575 \
      --yolo-weights /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt \
      --target-frames 60

Validate strict coverage:

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
    import json
    from pathlib import Path

    p = Path("/media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache_audit_all_task002k2.json")
    r = json.loads(p.read_text(encoding="utf-8"))

    assert r["expected_by_split"] == {"train": 6325, "dev": 933, "test": 1364}
    assert r["expected_total"] == 8622
    assert r["success_by_split"] == {"train": 6325, "dev": 933, "test": 1364}
    assert sum(r["failed_by_split"].values()) == 0
    assert r["missing_record_count"] == 0
    assert r["complete_for_requested_splits"] is True
    assert r["successful_artifacts_valid"] is True
    assert r["cache_fingerprints_unique"] is True
    assert r["test_rows_indexed"] == 1364
    assert r["test_rows_processed"] is True

    print("TASK-002K2 full DEPART-compatible cache coverage passed")
    PY

DataModule strict coverage smoke:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    from video.data.wsm_video_cache_datamodule import WSMVideoCacheDataModule

    dm = WSMVideoCacheDataModule(
        data_root="/media/maxim/Databases/WSM_NEW",
        cache_root="/media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache",
    )

    assert len(dm.train_dataset) == 6325
    assert len(dm.val_dataset) == 933
    assert len(dm.test_dataset["test_none"]) == 1364
    assert len(dm.test_dataset["test_soft"]) == 1208
    assert len(dm.test_dataset["test_hard"]) == 1014

    assert dm.train_unavailable_count == 0
    assert dm.dev_unavailable_count == 0
    assert dm.test_unavailable_count["test_none"] == 0
    assert dm.test_unavailable_count["test_soft"] == 0
    assert dm.test_unavailable_count["test_hard"] == 0

    print("TASK-002K2 full DataModule coverage passed")
    PY

Finally:

    git diff --check
    git diff -- src/audio
    git status --short

Before commit inspect only:

    git diff -- \
      src/video/features/clip_video_features.py \
      scripts/video/extract_clip_video_features.py \
      scripts/video/audit_video_cache.py \
      src/video/data/wsm_video_cache_datamodule.py \
      docs/PROGRESS_EN.md

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
- new preprocessing and ROI policy versions;
- DEPART full-frame fallback semantics;
- new cache root;
- old cache preservation;
- exact train/dev/test expected and success counts;
- exact total success/failure/missing counts;
- Test full coverage result;
- detection/fallback coverage statistics by split;
- number of segments using at least one fallback;
- temporal-length statistics;
- Test protocol dataset counts;
- all unavailable counts;
- DataModule default/root update;
- confirmation no training/Test performance inspection/model selection;
- src/audio unchanged;
- Stage 2 remains partial until the subsequent training-integration gate;
- recommended next atomic step: TASK-002L training config + every-epoch DEV/TEST_NONE/SOFT/HARD wiring.

## Required Handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch codex/task-002k2;
- implementation commit SHA;
- pushed-to-origin status;
- main/master untouched;
- train/dev/test success counts;
- total success/failure count;
- Test success=1364 requirement;
- fallback usage statistics;
- cache audit path;
- DataModule counts;
- no training/Test performance metrics;
- src/audio unchanged.

Stop after this task.
