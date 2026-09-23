# TASK-002D: Provision the Pinned CLIP Revision and Rerun the Fixed Six-Segment Real Audit

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-002d

Do not run full-dataset feature extraction, do not train any model, do not implement V2/prototypes, and do not inspect Test predictions or Test metrics.

## Goal

Resolve the TASK-002C blocker by provisioning a pinned local snapshot of the approved frozen CLIP visual encoder and rerunning exactly the same fixed six-segment real DEPART extraction audit.

No preprocessing semantics may change in this task.

The expected end-to-end path is:

    fixed 3 train + 3 dev real segments
      -> uniform 60-frame temporal sampling
      -> pinned YOLOv8 human-body ROI
      -> pinned frozen CLIP ViT-B/32
      -> [60,D] cache artifact + validity mask
      -> machine-readable audit

## Pinned CLIP Source

Use exactly:

- Hugging Face repository: openai/clip-vit-base-patch32
- pinned revision: b97b0100e55e367c057773c2a614676470b0d575
- architecture: CLIP ViT-B/32

Do not use a floating main revision.

The model snapshot must remain outside the WSM repository.

## Existing Pinned YOLO Source

Reuse the already provisioned TASK-002C checkpoint:

- local path: /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt
- expected local SHA-256:
  a6aead7bf0eccb35bd56731bfaa6ea19a4645a66150d2d0b19dd3fb1b116ef43

Do not redownload or replace it unless the file is missing or its SHA-256 differs.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 7-10 and 13-14
3. Stage 2 in docs/PLAN.md
4. docs/PROGRESS_EN.md through TASK-002C
5. docs/NEXT_TASK_EN.md
6. scripts/video/audit_depart_real_extraction.py
7. src/video/features/clip_video_features.py
8. src/video/features/yolov8_body_roi.py

## Allowed Tracked Files

- docs/PROGRESS_EN.md

No source-code modification is expected or authorized in this task.

If the existing audit helper cannot run successfully after correct model provisioning, stop and report the exact code blocker. Do not modify the helper or extractor in this task.

Untracked/local artifacts are allowed only under:

    /media/maxim/Programs/Models/WSM/huggingface/
    /media/maxim/Programs/Models/WSM/depart_yolov8/
    /tmp/wsm_depart_002d/

Do not commit model files or cache artifacts.

## Explicitly Authorized External Action

You are authorized to download/cache the pinned CLIP snapshot from Hugging Face into:

    HF_HOME=/media/maxim/Programs/Models/WSM/huggingface

Do not install dependencies.

Do not download any unrelated model.

Use the pinned revision, not main.

Prefer snapshot_download with allow_patterns limited to the files needed by Transformers CLIPModel/CLIPProcessor, including as applicable:

- config.json
- preprocessor_config.json
- model.safetensors
- pytorch_model.bin
- merges.txt
- vocab.json
- tokenizer.json
- tokenizer_config.json
- special_tokens_map.json

If snapshot_download reports that one of these optional alternatives is absent, that alone is not a failure provided CLIPModel and CLIPProcessor load successfully from the pinned revision with local_files_only=true.

## Implementation / Execution Requirements

### 1. Provision pinned CLIP snapshot

Set:

    export HF_HOME=/media/maxim/Programs/Models/WSM/huggingface

Use huggingface_hub.snapshot_download with:

    repo_id="openai/clip-vit-base-patch32"
    revision="b97b0100e55e367c057773c2a614676470b0d575"

Do not install huggingface_hub; use the environment's existing package.

After provisioning, verify with local-only loads:

    CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32",
        revision="b97b0100e55e367c057773c2a614676470b0d575",
        local_files_only=True,
    )

and:

    CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        revision="b97b0100e55e367c057773c2a614676470b0d575",
        local_files_only=True,
    )

Record the resolved snapshot directory and file list in PROGRESS_EN.md.

### 2. Verify YOLO checkpoint unchanged

Before rerunning the audit:

    sha256sum /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt

It must equal:

    a6aead7bf0eccb35bd56731bfaa6ea19a4645a66150d2d0b19dd3fb1b116ef43

If not, stop and report.

### 3. Rerun the exact TASK-002C fixed audit

Use the existing helper unchanged:

    scripts/video/audit_depart_real_extraction.py

The deterministic selected segment IDs must remain exactly:

Train:
- ["depression","-7UpRmNVJzQ","-7UpRmNVJzQ_001.mp4"]
- ["depression","-7UpRmNVJzQ","-7UpRmNVJzQ_002.mp4"]
- ["depression","-7UpRmNVJzQ","-7UpRmNVJzQ_003.mp4"]

DEV:
- ["depression","3SYkj_mya6A","3SYkj_mya6A_001.mp4"]
- ["depression","3SYkj_mya6A","3SYkj_mya6A_002.mp4"]
- ["depression","3SYkj_mya6A","3SYkj_mya6A_003.mp4"]

Test rows processed: zero.

Run with:

- target_frames=60
- YOLO conf=0.5
- YOLO IoU=0.5
- YOLO imgsz=640
- pinned CLIP revision above
- CUDA if available; otherwise CPU with the deviation recorded.

### 4. Real artifact verification

For every successful cache artifact, require:

- temporal length exactly 60;
- features shape [60,D];
- valid_mask bool [60];
- invalid positions exactly zero;
- valid positions finite;
- selected frame indices chronological;
- detector weights SHA-256 equals the TASK-002C pinned YOLO SHA-256;
- CLIP model name equals openai/clip-vit-base-patch32;
- CLIP model revision equals b97b0100e55e367c057773c2a614676470b0d575;
- detection parameters are conf=0.5, IoU=0.5, imgsz=640;
- unique cache fingerprints across the six attempted rows.

At least one successful cache artifact is required to clear the TASK-002C blocker.

### 5. Runtime side effects

Before launching Ultralytics:

    export YOLO_AUTOINSTALL=false

No package installation is authorized.

If any package auto-install is attempted again, stop and record the exact behavior.

### 6. No full extraction

Only the existing fixed six-row audit is authorized.

Do not run scripts/video/extract_clip_video_features.py over the full manifest.

## Acceptance Criteria

- pinned CLIP revision is cached locally outside the repo;
- local-only CLIPProcessor load succeeds;
- local-only CLIPModel load succeeds;
- pinned YOLO SHA-256 remains unchanged;
- exactly the same 3 train + 3 dev segments are attempted;
- zero Test rows are processed;
- all six temporal preflight lengths remain 60;
- at least one real cache artifact succeeds;
- successful artifacts pass all structural checks;
- detection coverage is reported for successful rows;
- no dependency/package installation occurs;
- no full extraction;
- no training/model selection;
- no Test predictions/metrics;
- src/audio unchanged;
- git diff --check passes;
- branch codex/task-002d is committed and pushed;
- main/master is untouched;
- tracked diff contains only docs/PROGRESS_EN.md.

If success_count remains zero, TASK-002D is blocked and must identify the exact next blocker without changing source code.

## Exact Verification Commands

Run from repository root.

First provision CLIP:

    export HF_HOME=/media/maxim/Programs/Models/WSM/huggingface
    export YOLO_AUTOINSTALL=false

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id="openai/clip-vit-base-patch32",
        revision="b97b0100e55e367c057773c2a614676470b0d575",
        cache_dir="/media/maxim/Programs/Models/WSM/huggingface/hub",
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "model.safetensors",
            "pytorch_model.bin",
            "merges.txt",
            "vocab.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
    )
    print(path)
    PY

Verify local-only load:

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
    from transformers import CLIPModel, CLIPProcessor

    repo = "openai/clip-vit-base-patch32"
    revision = "b97b0100e55e367c057773c2a614676470b0d575"

    processor = CLIPProcessor.from_pretrained(
        repo,
        revision=revision,
        local_files_only=True,
    )
    model = CLIPModel.from_pretrained(
        repo,
        revision=revision,
        local_files_only=True,
    )
    model.eval()

    print(type(processor).__name__)
    print(type(model).__name__)
    PY

Verify YOLO checksum:

    sha256sum /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt

Rerun only the fixed audit:

    rm -rf /tmp/wsm_depart_002d

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python       scripts/video/audit_depart_real_extraction.py       --data-root /media/maxim/Databases/WSM_NEW       --yolo-weights /media/maxim/Programs/Models/WSM/depart_yolov8/best.pt       --cache-root /tmp/wsm_depart_002d/cache       --report-output /tmp/wsm_depart_002d/report.json       --per-split 3       --target-frames 60       --model-name openai/clip-vit-base-patch32       --model-revision b97b0100e55e367c057773c2a614676470b0d575       --device cuda

If CUDA is unavailable, rerun with --device cpu and record the deviation.

Validate:

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
    import json
    from pathlib import Path

    report = json.loads(Path("/tmp/wsm_depart_002d/report.json").read_text(encoding="utf-8"))

    assert report["selected_count"] == 6
    assert report["selected_by_split"]["train"] == 3
    assert report["selected_by_split"]["dev"] == 3

    expected_train = [
        '["depression","-7UpRmNVJzQ","-7UpRmNVJzQ_001.mp4"]',
        '["depression","-7UpRmNVJzQ","-7UpRmNVJzQ_002.mp4"]',
        '["depression","-7UpRmNVJzQ","-7UpRmNVJzQ_003.mp4"]',
    ]
    expected_dev = [
        '["depression","3SYkj_mya6A","3SYkj_mya6A_001.mp4"]',
        '["depression","3SYkj_mya6A","3SYkj_mya6A_002.mp4"]',
        '["depression","3SYkj_mya6A","3SYkj_mya6A_003.mp4"]',
    ]

    assert report["selected_by_split"]["train"] == expected_train
    assert report["selected_by_split"]["dev"] == expected_dev

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

    print("TASK-002D pinned CLIP real extraction audit passed")
    PY

Then:

    git diff --check
    git diff -- src/audio
    git status --short

Before commit, the only tracked diff must be:

    docs/PROGRESS_EN.md

After commit/push:

    git status --short
    git rev-parse --abbrev-ref HEAD
    git rev-parse HEAD
    git diff --stat origin/main...HEAD
    git diff --name-only origin/main...HEAD

## Required PROGRESS_EN Update

Append TASK-002D facts without erasing prior evidence.

Record:

- branch;
- implementation commit SHA;
- push result;
- pinned CLIP repo/revision;
- HF_HOME and resolved snapshot directory;
- cached model file list sufficient for local-only load;
- local-only CLIPProcessor/CLIPModel load result;
- pinned YOLO SHA-256 recheck;
- exact same six selected segment IDs;
- device used;
- per-segment success/failure;
- detection coverage;
- temporal lengths;
- artifact structural validation;
- whether any package auto-install was attempted;
- exact commands/results;
- zero Test rows;
- no full extraction;
- src/audio unchanged;
- no training/model selection/Test metrics;
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

- branch codex/task-002d;
- implementation commit SHA;
- pushed-to-origin status;
- main/master untouched;
- diff summary;
- pinned CLIP revision;
- local-only CLIP load result;
- six attempted segments;
- success/failure counts;
- detection coverage summary;
- YOLO SHA-256;
- whether any package installation occurred;
- src/audio unchanged;
- zero Test rows/metrics;
- no full extraction/training.

Stop after this task.
