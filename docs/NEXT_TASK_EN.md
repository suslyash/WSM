# TASK-002E: Fix Transformers CLIP Output Unwrapping and Clear the Fixed Six-Segment Real Audit

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-002e

Do not run full-dataset feature extraction, do not train any model, do not implement V2/prototypes, and do not inspect Test predictions or Test metrics.

## Goal

Repair only the verified Transformers compatibility defect in the accepted V1 extractor, then rerun the exact fixed six-segment real audit from TASK-002C/TASK-002D.

Current blocker:

- pinned CLIPProcessor loads locally;
- pinned CLIPModel loads locally;
- real YOLO runs;
- all six selected videos preflight to 60 temporal positions;
- extraction fails because current Transformers returns a BaseModelOutputWithPooling from get_image_features / CLIP vision path, while the extractor assumes a tensor and reads .ndim directly.

The fix must make the extractor robust to the actual CLIP output type without changing any preprocessing, sampling, detector, model revision, temporal masking, or cache semantics.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 7-10 and 13-14
3. Stage 2 in docs/PLAN.md
4. docs/PROGRESS_EN.md through TASK-002D
5. docs/NEXT_TASK_EN.md
6. src/video/features/clip_video_features.py
7. scripts/video/audit_depart_real_extraction.py
8. src/video/features/yolov8_body_roi.py

## Allowed Tracked Files

- src/video/features/clip_video_features.py
- docs/PROGRESS_EN.md

No other tracked file may be modified.

## Forbidden Actions

- any change under src/audio;
- changing uniform frame sampling;
- changing target_frames=60;
- changing YOLO weights or thresholds;
- changing body ROI selection;
- changing CLIP repo/revision;
- changing cache fingerprint semantics except if strictly necessary to reflect the compatibility implementation version;
- changing audit sample rows;
- changing train/dev/test splits;
- full extraction;
- training/model selection;
- Test rows, predictions, or metrics;
- dependency installation;
- editing docs/NEXT_TASK_EN.md;
- pushing to main/master;
- opening or merging a PR.

## Fixed Runtime Inputs

Pinned YOLO:

- path: /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt
- SHA-256:
  a6aead7bf0eccb35bd56731bfaa6ea19a4645a66150d2d0b19dd3fb1b116ef43

Pinned CLIP:

- repo: openai/clip-vit-base-patch32
- revision: b97b0100e55e367c057773c2a614676470b0d575
- HF_HOME: /media/maxim/Programs/Models/WSM/huggingface

Environment:

    export HF_HOME=/media/maxim/Programs/Models/WSM/huggingface
    export HF_HUB_CACHE=/media/maxim/Programs/Models/WSM/huggingface/hub
    export YOLO_AUTOINSTALL=false

## Implementation Requirements

### 1. Add one explicit output-normalization helper

In src/video/features/clip_video_features.py, add a small reusable helper for CLIP image-feature outputs.

It must accept:

- a torch Tensor;
- an object exposing .pooler_output;
- an object exposing .image_embeds.

It must return a rank-2 tensor [B,D].

Required precedence:

1. if output is already a Tensor, use it;
2. else if output.image_embeds exists and is a Tensor, use it;
3. else if output.pooler_output exists and is a Tensor, use it;
4. otherwise raise a clear RuntimeError describing the unsupported output type.

Do not silently coerce arbitrary iterables/tuples.

### 2. Use the helper on both CLIP paths

The extractor currently has:

- get_image_features path;
- vision_model(...).pooler_output fallback path.

Normalize the actual returned object through the helper before shape validation.

Do not change the visual projection/model architecture.

If get_image_features returns a pooled/projected BaseModelOutputWithPooling in the installed Transformers version, use the tensor payload exposed by that object. Do not apply an extra learned projection unless the API path demonstrably requires it and existing CLIP output dimensionality proves it.

### 3. Preserve exact feature contract

After the fix:

- valid body crops produce finite rank-2 features;
- feature batch size equals number of valid crops;
- temporal reconstruction remains [60,D];
- invalid temporal positions remain exact zero;
- valid_mask remains bool [60];
- chronological positions remain unchanged.

### 4. Regression smoke

Add no new test file. Use command-line smokes.

Verify the helper with synthetic Tensor and small stand-in objects exposing image_embeds / pooler_output.

Verify unsupported object raises RuntimeError.

### 5. Rerun exact six-segment real audit

Use the existing helper unchanged:

    scripts/video/audit_depart_real_extraction.py

Exact selected rows must remain the TASK-002C/TASK-002D rows.

Zero Test rows.

At least one successful real cache artifact is required for TASK-002E to pass.

If success_count remains zero, stop and report the exact new blocker. Do not broaden the task.

## Acceptance Criteria

- output-normalization helper exists and has the strict precedence above;
- Tensor output works;
- image_embeds output works;
- pooler_output output works;
- unsupported output type raises clearly;
- no double projection is introduced;
- pinned CLIP local-only load remains unchanged;
- pinned YOLO SHA remains unchanged;
- same 3 train + 3 dev segments attempted;
- zero Test rows;
- all preflight temporal lengths remain 60;
- success_count >= 1;
- every successful artifact passes existing structural validation;
- features [60,D], valid_mask bool [60];
- invalid positions exactly zero;
- valid positions finite;
- detection coverage reported;
- no package installation;
- no full extraction;
- no training/model selection;
- no Test predictions/metrics;
- src/audio unchanged;
- python compilation passes;
- git diff --check passes;
- branch codex/task-002e committed and pushed;
- main/master untouched;
- tracked diff contains only the two allowed paths.

## Exact Verification Commands

Run from repository root.

Set environment:

    export HF_HOME=/media/maxim/Programs/Models/WSM/huggingface
    export HF_HUB_CACHE=/media/maxim/Programs/Models/WSM/huggingface/hub
    export YOLO_AUTOINSTALL=false

Compile:

    python3 -m py_compile src/video/features/clip_video_features.py

Run helper regression smoke:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import torch
    from types import SimpleNamespace

    from video.features.clip_video_features import normalize_clip_image_features

    x = torch.randn(3, 512)
    assert normalize_clip_image_features(x) is x

    image = torch.randn(2, 512)
    out = normalize_clip_image_features(SimpleNamespace(image_embeds=image))
    assert out is image

    pooled = torch.randn(4, 768)
    out = normalize_clip_image_features(SimpleNamespace(pooler_output=pooled))
    assert out is pooled

    try:
        normalize_clip_image_features(SimpleNamespace())
    except RuntimeError:
        pass
    else:
        raise AssertionError("unsupported CLIP output must fail clearly")

    print("CLIP output normalization smoke passed")
    PY

Verify local-only CLIP load still passes:

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
    from transformers import CLIPModel, CLIPProcessor

    repo = "openai/clip-vit-base-patch32"
    revision = "b97b0100e55e367c057773c2a614676470b0d575"

    processor = CLIPProcessor.from_pretrained(repo, revision=revision, local_files_only=True)
    model = CLIPModel.from_pretrained(repo, revision=revision, local_files_only=True)
    model.eval()
    print(type(processor).__name__, type(model).__name__)
    PY

Verify YOLO SHA:

    sha256sum /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt

It must remain:

    a6aead7bf0eccb35bd56731bfaa6ea19a4645a66150d2d0b19dd3fb1b116ef43

Run exact fixed audit:

    rm -rf /tmp/wsm_depart_002e

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python       scripts/video/audit_depart_real_extraction.py       --data-root /media/maxim/Databases/WSM_NEW       --yolo-weights /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt       --cache-root /tmp/wsm_depart_002e/cache       --report-output /tmp/wsm_depart_002e/report.json       --per-split 3       --target-frames 60       --model-name openai/clip-vit-base-patch32       --model-revision b97b0100e55e367c057773c2a614676470b0d575       --device cuda

If CUDA is unavailable, use cpu and record it.

Validate:

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
    import json
    from pathlib import Path

    report = json.loads(Path("/tmp/wsm_depart_002e/report.json").read_text(encoding="utf-8"))

    assert report["selected_count"] == 6
    assert report["selected_by_split"]["train"] == [
        '["depression","-7UpRmNVJzQ","-7UpRmNVJzQ_001.mp4"]',
        '["depression","-7UpRmNVJzQ","-7UpRmNVJzQ_002.mp4"]',
        '["depression","-7UpRmNVJzQ","-7UpRmNVJzQ_003.mp4"]',
    ]
    assert report["selected_by_split"]["dev"] == [
        '["depression","3SYkj_mya6A","3SYkj_mya6A_001.mp4"]',
        '["depression","3SYkj_mya6A","3SYkj_mya6A_002.mp4"]',
        '["depression","3SYkj_mya6A","3SYkj_mya6A_003.mp4"]',
    ]

    assert report["test_usage"]["test_rows_processed"] is False
    assert report["test_usage"]["model_predictions_inspected"] is False
    assert report["test_usage"]["performance_metrics_inspected"] is False
    assert report["test_usage"]["selection_or_tuning_performed"] is False

    assert report["requested_target_frames"] == 60
    assert not report["short_video_blockers"]
    assert report["checkpoint"]["local_sha256"] == (
        "a6aead7bf0eccb35bd56731bfaa6ea19a4645a66150d2d0b19dd3fb1b116ef43"
    )
    assert report["success_count"] >= 1
    assert report["all_success_artifacts_valid"] is True
    assert report["cache_fingerprints_unique"] is True

    print("TASK-002E real extraction audit passed")
    PY

Finally:

    git diff --check
    git diff -- src/audio
    git status --short

Before commit inspect only:

    git diff --       src/video/features/clip_video_features.py       docs/PROGRESS_EN.md

After commit/push:

    git status --short
    git rev-parse --abbrev-ref HEAD
    git rev-parse HEAD
    git diff --stat origin/main...HEAD
    git diff --name-only origin/main...HEAD

## Required PROGRESS_EN Update

Append TASK-002E facts without erasing prior evidence.

Record:

- branch;
- implementation commit SHA;
- push result;
- exact compatibility defect;
- exact normalization helper semantics;
- whether get_image_features returned Tensor, image_embeds container, or pooler_output container in the real runtime;
- pinned CLIP revision;
- pinned YOLO SHA;
- same six attempted segment IDs;
- device;
- per-segment success/failure;
- detection coverage;
- temporal lengths;
- feature dimensionality D for successful artifacts;
- artifact structural validation;
- package auto-install status;
- exact commands/results;
- zero Test rows;
- no full extraction;
- src/audio unchanged;
- no training/model selection/Test metrics;
- Stage 2 partial status;
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

- branch codex/task-002e;
- implementation commit SHA;
- pushed-to-origin status;
- main/master untouched;
- diff summary;
- CLIP output type observed in the real runtime;
- six attempted segments;
- success/failure counts;
- detection coverage summary;
- feature dimension;
- YOLO SHA-256;
- package install status;
- src/audio unchanged;
- zero Test rows/metrics;
- no full extraction/training.

Stop after this task.
