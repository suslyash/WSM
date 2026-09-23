# TASK-002A: Define the Reproducible V1 Video Preprocessing and Cache Contract

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-002a

Do not train a model, run Test evaluation, implement V2/prototypes, or begin Stage 3.

## Goal

Start PLAN Stage 2 by implementing the deterministic preprocessing/cache contract for V1 video only.

V1 is DEPART-like in project terms:

- uniform frame sampling;
- quality/failure reporting;
- frozen CLIP-family visual encoder;
- cached temporal video features;
- later projection + temporal Transformer + masked pooling + two independent sigmoid heads.

This task covers only input sampling, frozen feature extraction contract, cache fingerprinting, and extraction-failure accounting. Do not implement or train the classifier yet.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 2–10 and 13–14
3. Stage 2 in docs/PLAN.md
4. docs/PROGRESS_EN.md
5. docs/NEXT_TASK_EN.md
6. src/common/data/wsm_manifest.py
7. src/fusion/data/wsm_manifest_datamodule.py
8. src/common/features/wsm_feature_extractor.py
9. docs/SOTA_REVIEW_EN.md sections relevant to video/DEPART-like baselines

## Allowed Files

- src/video/__init__.py
- src/video/features/__init__.py
- src/video/features/clip_video_features.py
- scripts/video/extract_clip_video_features.py
- docs/PROGRESS_EN.md

Creating the src/video/features and scripts/video directories is allowed.

No other tracked file may be modified.

## Fixed Manager Decisions for V1

Use one frozen CLIP-family encoder only in this task.

Default extractor contract:

- model revision: openai/clip-vit-base-patch32;
- sampling: uniform in time over the full segment;
- default target frames: 32;
- no diagnosis, corpus identity, task label, observed mask, or split label may enter model inputs;
- output is temporal frame embeddings plus a validity mask;
- no learned projection/Transformer/classifier in this task.

If the exact model is unavailable locally without a download during verification, do not install/download silently. The implementation may support the model contract while verification uses a synthetic/mock encoder path.

## Implementation Requirements

### 1. Reusable feature module

Implement src/video/features/clip_video_features.py with:

- deterministic uniform frame-index selection from frame_count and target_frames;
- clear handling of zero/unreadable frames;
- RGB conversion;
- a reusable frozen CLIP extractor class;
- temporal feature output [T,D];
- boolean validity mask [T];
- no fabricated feature for failed extraction: failure must be represented explicitly in result metadata and availability=false.

### 2. Cache fingerprint

Implement a deterministic cache fingerprint including at least:

- manifest fingerprint;
- model name/revision;
- preprocessing implementation version;
- sampling method;
- target frame count;
- image processor/preprocessing identity;
- source video path/segment_id;
- relevant package/version strings when available.

Changing any of the above must change the cache key/fingerprint.

### 3. Cache artifact contract

Each successful cached artifact must include:

- segment_id;
- source path;
- temporal features;
- valid mask;
- model/revision;
- preprocessing parameters;
- cache fingerprint.

Each failed extraction must be recorded in a machine-readable report and must not create a fake successful feature artifact.

### 4. CLI

Implement scripts/video/extract_clip_video_features.py.

Required arguments:

    --data-root
    --cache-root
    --report-output

Optional:

    --manifest-path
    --model-name
    --model-revision
    --target-frames
    --device
    --limit
    --overwrite

Requirements:

- default to canonical manifest if no manifest path;
- process only rows with video_available=true;
- do not modify the source dataset;
- refuse cache/report outputs inside src/, configs/, or docs/;
- write a machine-readable extraction report with success/failure counts and manifest/cache fingerprints;
- --limit is verification/debug only and must be recorded in the report;
- no Test metrics/predictions are computed; structural processing of test video files is allowed.

### 5. No registry/model yet

This task creates preprocessing/features only. Do not add a Chimera model/config/registry entry yet.

## Acceptance Criteria

- deterministic uniform sampler passes edge cases;
- cache fingerprint changes when model revision, target_frames, or manifest fingerprint changes;
- successful mock/synthetic extraction has [T,D] features and [T] bool mask;
- failed read/extraction returns availability=false and is present in report, with no fake artifact;
- CLI dry/small-limit structural run can execute without training;
- no labels/task identity are passed into extractor;
- src/audio unchanged;
- no Test predictions/metrics, training, tuning, or model selection;
- python compilation passes;
- git diff --check passes;
- branch codex/task-002a is committed/pushed;
- diff against origin/main contains only the five allowed paths.

## Exact Verification Commands

Run from repository root:

    python3 -m py_compile       src/video/__init__.py       src/video/features/__init__.py       src/video/features/clip_video_features.py       scripts/video/extract_clip_video_features.py

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    from video.features.clip_video_features import (
        uniform_frame_indices,
        build_cache_fingerprint,
    )

    assert uniform_frame_indices(0, 32) == []
    assert uniform_frame_indices(1, 32) == [0]
    ids = uniform_frame_indices(100, 32)
    assert len(ids) == 32
    assert ids == sorted(ids)
    assert ids[0] == 0
    assert ids[-1] == 99

    a = build_cache_fingerprint(
        manifest_fingerprint="m1",
        segment_id="s1",
        source_path="/tmp/a.mp4",
        model_name="openai/clip-vit-base-patch32",
        model_revision="r1",
        target_frames=32,
        preprocessing_version="v1",
        processor_identity="clip",
    )
    b = build_cache_fingerprint(
        manifest_fingerprint="m1",
        segment_id="s1",
        source_path="/tmp/a.mp4",
        model_name="openai/clip-vit-base-patch32",
        model_revision="r2",
        target_frames=32,
        preprocessing_version="v1",
        processor_identity="clip",
    )
    c = build_cache_fingerprint(
        manifest_fingerprint="m1",
        segment_id="s1",
        source_path="/tmp/a.mp4",
        model_name="openai/clip-vit-base-patch32",
        model_revision="r1",
        target_frames=16,
        preprocessing_version="v1",
        processor_identity="clip",
    )
    assert a != b
    assert a != c
    print("video preprocessing contract assertions passed")
    PY

    git diff --check
    git diff -- src/audio
    git status --short

Before committing inspect only:

    git diff --       src/video/__init__.py       src/video/features/__init__.py       src/video/features/clip_video_features.py       scripts/video/extract_clip_video_features.py       docs/PROGRESS_EN.md

After commit/push:

    git status --short
    git rev-parse --abbrev-ref HEAD
    git rev-parse HEAD
    git diff --stat origin/main...HEAD
    git diff --name-only origin/main...HEAD

## Required PROGRESS_EN Update

Record branch/SHA/push, exact preprocessing contract, cache fingerprint fields, sampler checks, failure semantics, verification commands/results, src/audio unchanged, no Test metrics, no training/model selection, and Stage 2 partial status.

## Required Handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly state branch codex/task-002a, implementation SHA, push status, main/master untouched, diff summary, src/audio unchanged, and that no Test predictions/metrics or training occurred.

Stop after this task.
