# TASK-002B: Integrate the DEPART YOLOv8 Body-ROI Stage into the V1 Temporal CLIP Cache

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-002b

Do not train a video classifier, do not run full-dataset feature extraction, do not implement V2/prototypes, and do not inspect Test predictions or metrics.

## Goal

Extend the accepted TASK-002A temporal CLIP/cache foundation to match the intended DEPART-style visual preprocessing pipeline:

    uniformly sampled temporal frames
        -> YOLOv8 single-class human-body localization
        -> body ROI crop
        -> frozen CLIP frame encoding
        -> ordered temporal feature sequence + validity mask

The DEPART paper applies a YOLOv8 detector trained for human-body localization to every sampled frame before visual encoding. Its documented detector inference settings are confidence=0.5, IoU=0.5, imgsz=640, with body ROI cropping before CLIP. For one-minute segments the paper reports practical operation with N=60 frames. Use N=60 as the Stage 2 V1 default.

This task integrates and verifies that preprocessing contract only. Full feature-cache extraction is a later manager-approved task.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 2-10 and 13-14
3. Stage 2 in docs/PLAN.md
4. docs/PROGRESS_EN.md, especially TASK-002A and MANAGER-DECISION-002
5. docs/NEXT_TASK_EN.md
6. src/video/features/clip_video_features.py
7. scripts/video/extract_clip_video_features.py
8. src/common/data/wsm_manifest.py
9. The DEPART body-region and temporal preprocessing description already recorded by the manager

## Allowed Files

- src/video/features/clip_video_features.py
- src/video/features/yolov8_body_roi.py
- src/video/features/__init__.py
- scripts/video/extract_clip_video_features.py
- pyproject.toml
- docs/PROGRESS_EN.md

No other tracked file may be modified.

## Manager-Fixed DEPART V1 Contract

Use:

- target_frames default: 60;
- detector family: YOLOv8;
- detector semantic class: single human-body class;
- confidence threshold: 0.5;
- IoU threshold: 0.5;
- detector input size: 640;
- detector weights: external/local checkpoint path supplied explicitly by configuration/CLI;
- body crop is computed per sampled frame;
- frozen CLIP visual encoder remains openai/clip-vit-base-patch32 unless a later task explicitly changes it;
- ordered temporal position must be preserved across all 60 sampled positions.

Do not auto-download detector weights.

The referenced DEPART detector checkpoint comes from the single-class human-detection YOLOv8 line. For reproducibility, the project must treat the local weights file and its SHA-256 as part of the cache contract.

## Frame-Level Detection Semantics

Implement the following deterministic policy.

For each sampled frame:

1. run the single-class body detector;
2. discard detections below confidence threshold;
3. if multiple valid detections remain, choose the highest-confidence detection;
4. deterministic tie-break order:
   - larger bounding-box area;
   - then lexicographic coordinates (x1, y1, x2, y2);
5. clip the selected box to image bounds;
6. reject a zero/invalid-area box;
7. crop that body ROI and pass only that crop to CLIP.

If no valid detection exists for a sampled frame:

- preserve its temporal position;
- mark valid_mask=false for that position;
- use a zero tensor only as masked padding after the feature dimension is known;
- never report that padded position as a successful body detection.

If zero frames in the segment have valid body detections:

- the entire segment extraction is success=false;
- do not save a successful cache artifact.

Partial detection coverage is allowed and must be reported machine-readably.

## Implementation Requirements

### 1. YOLOv8 body detector adapter

Create src/video/features/yolov8_body_roi.py.

Provide a reusable adapter that:

- lazily imports the runtime detector dependency;
- loads only an explicitly supplied local weights path;
- never downloads weights;
- freezes/eval-like inference behavior as applicable;
- exposes detector identity and weights SHA-256;
- accepts RGB frames;
- returns selected body boxes/confidences with the deterministic policy above;
- supports injected/mock detector objects for smoke verification without real weights.

If the project uses the Ultralytics YOLO runtime, declare the dependency in pyproject.toml. Do not install it during this task.

### 2. Extend the temporal CLIP extractor

Update ClipVideoFeatureExtractor so it can operate in DEPART ROI mode.

Requirements:

- default target_frames becomes 60;
- accepted TASK-002A uniform temporal sampler remains deterministic;
- when a body detector is configured, CLIP sees only detected body crops;
- returned temporal feature tensor has shape [T,D] where T equals sampled temporal positions;
- returned valid_mask has shape [T];
- invalid detection positions are zero-padded and mask=false;
- valid positions preserve chronological order;
- all-invalid segments fail explicitly and contain no fake features;
- raw-frame CLIP behavior may remain available only as an explicit compatibility/debug mode, not as the default Stage 2 V1 mode.

### 3. Cache fingerprint and artifact

Revise the cache fingerprint for ROI mode to include at least:

- manifest fingerprint;
- segment_id/source path;
- CLIP model/revision;
- preprocessing version;
- sampling method;
- target_frames;
- body detector family/implementation identity;
- detector weights SHA-256;
- confidence threshold;
- IoU threshold;
- detector image size;
- ROI selection policy/version;
- processor identity;
- relevant package versions.

A change to detector weights, confidence, IoU, imgsz, target_frames, or ROI policy must change the fingerprint.

Successful cache artifacts must additionally contain:

- detector identity;
- detector weights SHA-256;
- detection parameters;
- sampled frame indices;
- selected boxes or null per temporal position;
- per-frame detection confidence or null;
- valid_mask;
- valid_detection_count;
- detection_coverage = valid_detection_count / T.

Do not store labels, diagnosis, corpus/task identity, or Test metrics in the feature artifact.

### 4. CLI integration

Update scripts/video/extract_clip_video_features.py.

Add/adjust arguments:

    --yolo-weights PATH
    --yolo-conf FLOAT          default 0.5
    --yolo-iou FLOAT           default 0.5
    --yolo-imgsz INT           default 640
    --target-frames INT        default 60
    --raw-frame-debug          explicit opt-in only

Normal/default V1 execution must require --yolo-weights.

Rules:

- refuse a missing weights file clearly;
- compute weights SHA-256;
- never download weights;
- do not run full dataset in verification;
- preserve --limit;
- report aggregate detection coverage for successful extractions;
- report no-body-detected separately from video-read/model-load failures;
- no Test predictions/metrics.

### 5. Dependency discipline

If ultralytics is required and absent from pyproject.toml:

- add a normal project dependency entry;
- do not run pip/uv/poetry install;
- smoke verification must work through injected mock detector/encoder paths when the runtime package or weights are unavailable.

## Acceptance Criteria

- default V1 target_frames is 60;
- default V1 pipeline requires body ROI detection before CLIP;
- detector settings default to confidence=0.5, IoU=0.5, imgsz=640;
- detector weights are explicit/local and fingerprinted;
- no weight auto-download exists;
- deterministic multiple-box selection passes smoke tests;
- temporal ordering is preserved;
- partial missing detections yield [60,D] plus a correct bool [60] mask;
- masked positions are zero padding only and never counted as detections;
- all-missing detections cause extraction failure and cannot be saved as success;
- cache fingerprint changes with detector weights fingerprint or any detector threshold/size parameter;
- CLI defaults to 60 and YOLO ROI mode;
- no full-dataset extraction occurs;
- no labels/task identity reach YOLO or CLIP;
- no Test predictions/metrics, training, tuning, or model selection;
- src/audio unchanged;
- python compilation passes;
- smoke checks pass;
- git diff --check passes;
- branch codex/task-002b is committed and pushed;
- main/master is untouched by implementing Codex;
- diff against origin/main contains only the six allowed tracked paths.

## Exact Verification Commands

Run from repository root.

    python3 -m py_compile \
      src/video/features/yolov8_body_roi.py \
      src/video/features/clip_video_features.py \
      src/video/features/__init__.py \
      scripts/video/extract_clip_video_features.py

Run a deterministic detector-policy smoke without real YOLO weights:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import numpy as np

    from video.features.yolov8_body_roi import select_body_detection

    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    det = select_body_detection(
        frame,
        boxes=[
            [10, 10, 80, 90],
            [20, 10, 180, 90],
            [0, 0, 50, 50],
        ],
        confidences=[0.8, 0.8, 0.4],
        confidence_threshold=0.5,
    )
    assert det is not None
    # Same confidence: larger valid box wins.
    assert det.box == (20, 10, 180, 90)

    assert select_body_detection(
        frame,
        boxes=[[0, 0, 50, 50]],
        confidences=[0.49],
        confidence_threshold=0.5,
    ) is None

    print("YOLO body-selection policy smoke passed")
    PY

Run the temporal ROI + CLIP contract smoke using injected mock detector/encoder objects. It must prove:

- exactly 60 temporal positions;
- valid and invalid body detections are interleaved without reordering;
- features shape is [60,D];
- valid_mask is bool [60];
- invalid positions are exactly zero;
- all-invalid detection input returns success=false;
- no real model download occurs.

Also verify fingerprint sensitivity with synthetic values:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    from video.features.clip_video_features import build_cache_fingerprint

    common = dict(
        manifest_fingerprint="manifest",
        segment_id="segment",
        source_path="/tmp/video.mp4",
        model_name="openai/clip-vit-base-patch32",
        model_revision="main",
        target_frames=60,
        preprocessing_version="depart-v1-roi",
        processor_identity="clip",
        detector_identity="yolov8-human",
        detector_weights_sha256="a" * 64,
        detector_confidence=0.5,
        detector_iou=0.5,
        detector_imgsz=640,
        roi_policy_version="depart-body-roi-v1",
    )
    base = build_cache_fingerprint(**common)
    for key, value in [
        ("detector_weights_sha256", "b" * 64),
        ("detector_confidence", 0.6),
        ("detector_iou", 0.6),
        ("detector_imgsz", 320),
        ("target_frames", 30),
        ("roi_policy_version", "depart-body-roi-v2"),
    ]:
        changed = dict(common)
        changed[key] = value
        assert build_cache_fingerprint(**changed) != base

    print("DEPART ROI fingerprint sensitivity passed")
    PY

Verify CLI surface:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
      scripts/video/extract_clip_video_features.py --help

Do not execute a full extraction command.

Finally:

    git diff --check
    git diff -- src/audio
    git status --short

Before committing inspect only:

    git diff -- \
      src/video/features/clip_video_features.py \
      src/video/features/yolov8_body_roi.py \
      src/video/features/__init__.py \
      scripts/video/extract_clip_video_features.py \
      pyproject.toml \
      docs/PROGRESS_EN.md

After commit/push:

    git status --short
    git rev-parse --abbrev-ref HEAD
    git rev-parse HEAD
    git diff --stat origin/main...HEAD
    git diff --name-only origin/main...HEAD

## Required PROGRESS_EN Update

Append TASK-002B facts without erasing TASK-002A evidence.

Record:

- branch and implementation commit SHA;
- push result;
- DEPART ROI pipeline settings;
- detector runtime/weights availability;
- whether ultralytics dependency was only declared or already available;
- deterministic body-selection policy;
- temporal [60,D] and mask contract;
- partial/all-missing detection behavior;
- cache fingerprint additions;
- exact verification commands/results;
- confirmation no full-dataset extraction ran;
- confirmation src/audio unchanged;
- confirmation no Test predictions/metrics or training/model selection ran;
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

- branch codex/task-002b;
- implementation commit SHA;
- pushed-to-origin status;
- main/master untouched;
- diff summary against origin/main;
- src/audio unchanged;
- no full feature extraction;
- no Test predictions/metrics;
- whether real YOLO weights/runtime were available.

Stop after this task. Do not start full cache extraction or model training.
