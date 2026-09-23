# TASK-002K: Build the Full V1 Test Video Cache and Expose test_none/test_soft/test_hard DataModule Streams

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-002k

Do not create a training YAML and do not run training in this task.

This task explicitly authorizes full canonical TEST video feature extraction and structural audit. It does NOT authorize using Test metrics for checkpointing, early stopping, threshold search, hyperparameter/model selection, or architecture decisions.

## Goal

Complete the V1 video data path required by PROJECT_REQUIREMENTS so later training can report, every epoch:

    dev
    test_none
    test_soft
    test_hard

TASK-002K must:

1. correct the extractor/audit Test telemetry so it reports actual Test processing truthfully;
2. build the full canonical TEST V1 video cache using the same pinned YOLO+CLIP contract as TRAIN+DEV;
3. independently audit the Test cache;
4. extend wsm_video_depart_v1_datamodule with separate test_none/test_soft/test_hard datasets;
5. preserve dev as the distinct validation dataset;
6. keep Test protocols separate from val_dataset.

Do not wire the trainer/config to execute all four streams every epoch yet; that is the next atomic integration gate.

## Established Test Protocol Semantics

Use the existing WSM semantics already used by frozen audio/common segment indexing:

- test_none: every canonical row with split == "test";
- test_soft: canonical Test rows whose raw test metadata has soft_filter == 1;
- test_hard: canonical Test rows whose raw test metadata has hard_filter == 1.

Use the same source fields as common.utils.segment_index / frozen audio.

Do not invent new filters.

## Canonical Manifest Constraint

Do NOT add soft_filter or hard_filter to the canonical manifest schema in this task.

Reason:

- the accepted TRAIN+DEV cache fingerprint uses the current canonical manifest serialization/fingerprint;
- changing canonical columns would silently change that fingerprint and invalidate the accepted cache contract.

Instead, join Test protocol membership from the existing raw test metadata / build_wsm_multitask_segment_index by canonical segment identity:

    [corpus/task, video_id, segment_file]

and verify one-to-one matching for canonical Test rows.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 2, 8, 9, 10, 13, 14
3. Stage 2 in docs/PLAN.md
4. docs/PROGRESS_EN.md through TASK-002J and MANAGER-DECISION-009
5. docs/NEXT_TASK_EN.md
6. scripts/video/extract_clip_video_features.py
7. scripts/video/audit_video_cache.py
8. src/video/data/wsm_video_cache_datamodule.py
9. src/common/data/wsm_manifest.py
10. src/common/utils/segment_index.py
11. src/audio/data/wsm_audio_segment_datamodule.py for established Test filter semantics only

## Allowed Tracked Files

- scripts/video/extract_clip_video_features.py
- scripts/video/audit_video_cache.py
- src/video/data/wsm_video_cache_datamodule.py
- docs/PROGRESS_EN.md

No other tracked file may be modified.

Do not modify src/audio.

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

Canonical Test rows expected from the accepted manifest:

    1364

Do not hardcode soft/hard subset counts; compute and record them from raw Test metadata.

## Implementation Requirements

### 1. Correct extraction Test telemetry

Update scripts/video/extract_clip_video_features.py.

Current hard-coded Test flags must become truthful.

Report at least:

- requested_splits;
- selected_count;
- selected counts by split;
- test_rows_selected;
- test_rows_processed;
- test_rows_indexed;
- labels_passed_to_encoder=false;
- test_metrics_inspected=false.

Semantics:

- test_rows_selected = number of selected candidate rows whose split=="test";
- test_rows_processed = test_rows_selected > 0 for a completed extraction attempt;
- test_rows_indexed = number of Test records currently present in the shared cache index.

Do not claim Test was not processed when --splits test is used.

### 2. Generalize independent cache audit

Update scripts/video/audit_video_cache.py so --splits can accept an explicit subset/order from:

    train,dev,test

It must continue to validate TRAIN+DEV correctly and must also support:

    --splits test

For the requested split set:

- independently rebuild canonical expected IDs;
- validate every selected index record/artifact;
- count success/failure/missing;
- validate variable-length [T,512], 1<=T<=60;
- report coverage and temporal-length stats;
- verify unique fingerprints among successful requested rows;
- do not reject legitimate index records belonging to non-requested splits in the shared cache.

For --splits test, report:

- expected Test total;
- valid Test cache artifacts;
- explicit Test failures;
- no_body_detected count;
- other failures;
- missing records;
- complete_for_requested_splits.

Set test_rows_processed=true in the audit report because this task explicitly processed Test features.

Keep:

    test_usage.model_predictions_inspected=false
    test_usage.performance_metrics_inspected=false
    test_usage.selection_or_tuning_performed=false

Feature extraction is not model prediction evaluation.

### 3. Build the full canonical Test cache

Run the existing pinned V1 extractor with:

    --splits test
    --resume
    --overwrite

against the same persistent cache root.

Do not use --limit.

Do not delete existing TRAIN+DEV cache/index rows.

Expected canonical Test row count is 1364.

Every Test row must end with exactly one index record:

- extracted/reused valid artifact; or
- explicit failure record.

No fake feature for no_body_detected.

### 4. Independent Test cache audit

Run audit_video_cache.py with:

    --splits test

and require:

- expected_total == 1364;
- missing_record_count == 0;
- complete_for_requested_splits == true;
- all successful Test artifacts structurally valid;
- successful fingerprints unique;
- pinned YOLO/CLIP metadata correct.

Record the measured success/failure counts.

### 5. Add Test protocol membership join

In src/video/data/wsm_video_cache_datamodule.py, derive Test protocol membership from existing raw Test metadata without changing canonical manifest schema.

Use common.utils.segment_index.build_wsm_multitask_segment_index or equivalent existing helper.

For canonical Test rows:

- join by corpus/task + video_id + segment_file;
- require one raw metadata match per canonical Test segment;
- require soft_filter/hard_filter values to be interpretable as 0/1 or missing according to existing audio semantics;
- test_none membership = all canonical Test rows;
- test_soft membership = soft_filter == 1;
- test_hard membership = hard_filter == 1.

Do not use diagnosis to define filter membership.

### 6. Extend the video DataModule

Preserve:

    train_dataset
    val_dataset

with current accepted semantics.

Add separate:

    test_dataset = {
        "test_none": ...,
        "test_soft": ...,
        "test_hard": ...,
    }

Do NOT merge these datasets into val_dataset.

Each Test dataset must:

- include only rows belonging to the protocol;
- include only valid cached video artifacts;
- exclude no_body_detected rows as unavailable rather than fabricate zero videos;
- retain [depression, parkinson] sparse targets and observed_mask;
- use the same variable-length collate function;
- preserve meta["split"] == "test";
- expose a protocol field in sample metadata, e.g. meta["evaluation_protocol"].

Track unavailable counts separately for test_none/test_soft/test_hard.

### 7. Context contract

describe_context must additionally expose at least:

- data.test_protocols = ["test_none","test_soft","test_hard"];
- data.video_test_none_rows;
- data.video_test_soft_rows;
- data.video_test_hard_rows;
- data.video_test_none_unavailable;
- data.video_test_soft_unavailable;
- data.video_test_hard_unavailable;
- data.video_test_rows_indexed;
- data.video_epoch_test_monitoring_required = true.

Do not expose Test metric values.

### 8. Keep Test separate from selection

This task creates the Test data streams only.

Do not:

- compare Test performance;
- inspect predictions;
- tune thresholds;
- select checkpoints;
- choose hyperparameters;
- alter architecture based on Test.

The next task will wire these datasets into every-epoch evaluation while enforcing dev/mean_score as the only automatic selector.

## Acceptance Criteria

Extractor/audit:

- --splits test runs truthfully with test_rows_processed=true;
- full canonical Test selection count=1364;
- all 1364 Test rows have final index records;
- independent Test cache audit complete=true;
- no missing Test records;
- successful Test artifacts valid [T,512], 1<=T<=60;
- failures explicit;
- no fake features;
- pinned YOLO/CLIP contract unchanged.

DataModule:

- train_dataset remains 6255;
- val_dataset remains 907;
- test_dataset has exactly keys test_none/test_soft/test_hard;
- test_none is based on all canonical Test rows with valid video cache;
- test_soft/hard exactly follow raw soft_filter/hard_filter membership;
- protocol subsets are deterministic;
- Test datasets are not merged into val_dataset;
- no_body rows excluded and counted unavailable;
- sparse targets/observed masks preserved;
- variable-length collate works on Test;
- context exposes protocol counts/unavailable counts;
- no Test metric values computed.

Safety:

- no training;
- no model selection;
- no Test prediction/metric inspection;
- src/audio unchanged;
- python compilation passes;
- git diff --check passes;
- branch codex/task-002k committed and pushed;
- main/master untouched;
- tracked diff contains only the four allowed paths.

## Exact Verification Commands

Set environment:

    export HF_HOME=/media/maxim/Programs/Models/WSM/huggingface
    export HF_HUB_CACHE=/media/maxim/Programs/Models/WSM/huggingface/hub
    export YOLO_AUTOINSTALL=false

Compile:

    python3 -m py_compile \
      scripts/video/extract_clip_video_features.py \
      scripts/video/audit_video_cache.py \
      src/video/data/wsm_video_cache_datamodule.py

Verify pinned YOLO:

    sha256sum /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt

Expected:

    a6aead7bf0eccb35bd56731bfaa6ea19a4645a66150d2d0b19dd3fb1b116ef43

Run full canonical Test extraction:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
      scripts/video/extract_clip_video_features.py \
      --data-root /media/maxim/Databases/WSM_NEW \
      --cache-root /media/maxim/Programs/Features/WSM/video_depart_v1/cache \
      --report-output /media/maxim/Programs/Features/WSM/video_depart_v1/extraction_report_test_task002k.json \
      --splits test \
      --resume \
      --overwrite \
      --yolo-weights /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt \
      --model-name openai/clip-vit-base-patch32 \
      --model-revision b97b0100e55e367c057773c2a614676470b0d575 \
      --target-frames 60 \
      --device cuda

If CUDA is unavailable, CPU fallback is allowed only for a pure device/runtime issue and must be recorded.

Run independent Test audit:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
      scripts/video/audit_video_cache.py \
      --data-root /media/maxim/Databases/WSM_NEW \
      --cache-root /media/maxim/Programs/Features/WSM/video_depart_v1/cache \
      --report-output /media/maxim/Programs/Features/WSM/video_depart_v1/cache_audit_test_task002k.json \
      --splits test \
      --model-name openai/clip-vit-base-patch32 \
      --model-revision b97b0100e55e367c057773c2a614676470b0d575 \
      --yolo-weights /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt \
      --target-frames 60

Validate Test audit:

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
    import json
    from pathlib import Path

    p = Path("/media/maxim/Programs/Features/WSM/video_depart_v1/cache_audit_test_task002k.json")
    r = json.loads(p.read_text(encoding="utf-8"))

    assert r["requested_splits"] == ["test"]
    assert r["expected_total"] == 1364
    assert r["missing_record_count"] == 0
    assert r["complete_for_requested_splits"] is True
    assert r["successful_artifacts_valid"] is True
    assert r["cache_fingerprints_unique"] is True
    assert r["test_rows_processed"] is True
    assert r["test_usage"]["model_predictions_inspected"] is False
    assert r["test_usage"]["performance_metrics_inspected"] is False
    assert r["test_usage"]["selection_or_tuning_performed"] is False

    print("TASK-002K Test cache audit passed")
    PY

DataModule/Test-protocol smoke:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import torch

    from video.data.wsm_video_cache_datamodule import WSMVideoCacheDataModule

    dm = WSMVideoCacheDataModule(
        data_root="/media/maxim/Databases/WSM_NEW",
        cache_root="/media/maxim/Programs/Features/WSM/video_depart_v1/cache",
    )

    assert len(dm.train_dataset) == 6255
    assert len(dm.val_dataset) == 907
    assert set(dm.test_dataset) == {"test_none", "test_soft", "test_hard"}

    none_ds = dm.test_dataset["test_none"]
    soft_ds = dm.test_dataset["test_soft"]
    hard_ds = dm.test_dataset["test_hard"]

    assert len(none_ds) > 0
    assert len(soft_ds) > 0
    assert len(hard_ds) > 0
    assert len(soft_ds) <= len(none_ds)
    assert len(hard_ds) <= len(none_ds)

    for name, ds in dm.test_dataset.items():
        sample = ds[0]
        assert sample["meta"]["split"] == "test"
        assert sample["meta"]["evaluation_protocol"] == name
        assert tuple(sample["targets"].shape) == (2,)
        assert tuple(sample["observed_mask"].shape) == (2,)

    # Test collate must preserve the same variable-length/mask contract.
    samples = [none_ds[0], none_ds[min(1, len(none_ds)-1)]]
    batch = dm.collate_fn(samples)
    assert batch.inputs["video"].ndim == 3
    assert batch.inputs["video"].shape[-1] == 512
    assert batch.get_masks("video_mask").dtype == torch.bool
    assert tuple(batch.targets.shape) == (2,2)
    assert tuple(batch.get_masks("observed_mask").shape) == (2,2)

    # Test streams remain separate from DEV.
    assert not isinstance(dm.val_dataset, dict) or not any(
        key in dm.val_dataset for key in ("test_none","test_soft","test_hard")
    )

    print(
        "TASK-002K DataModule Test streams passed",
        len(none_ds), len(soft_ds), len(hard_ds),
    )
    PY

Add a raw-membership audit in the implementation verification proving every test_soft/test_hard dataset row corresponds exactly to soft_filter==1 / hard_filter==1 from the existing raw Test metadata.

Finally:

    git diff --check
    git diff -- src/audio
    git status --short

Before commit inspect only:

    git diff -- \
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
- corrected Test telemetry semantics;
- full Test cache report/audit paths;
- expected Test count;
- Test extracted/reused/failure counts;
- no_body and other failure counts;
- Test temporal-length and coverage stats;
- raw test_none/test_soft/test_hard membership counts before cache availability filtering;
- valid dataset counts for test_none/test_soft/test_hard;
- unavailable counts for each protocol;
- one-to-one raw membership join evidence;
- confirmation Test datasets remain separate from val_dataset;
- confirmation no Test metrics/predictions were inspected;
- confirmation no training/model selection;
- src/audio unchanged;
- Stage 2 remains partial;
- recommended next atomic step: wire dev/test_none/test_soft/test_hard into every-epoch evaluation with dev/mean_score as the sole automatic selector.

## Required Handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch codex/task-002k;
- implementation commit SHA;
- pushed-to-origin status;
- main/master untouched;
- Test expected/success/failure counts;
- test_none/test_soft/test_hard dataset counts;
- unavailable counts;
- Test cache audit path;
- Test rows processed/indexed;
- confirmation no Test metrics were inspected;
- no training/model selection;
- src/audio unchanged.

Stop after this task.
