# TASK-002I: Implement the V1 Video Cache DataModule and Variable-Length Collate Contract

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-002i

Do not process Test cache rows, do not create a training YAML, do not run training, do not implement V2/prototypes, and do not inspect Test predictions or Test metrics.

## Goal

Implement and register the Stage 2 V1 video cache DataModule for the accepted TRAIN+DEV cache.

The DataModule must consume only valid V1 cache artifacts and produce Chimera Batch objects compatible with:

- wsm_video_depart_v1_model;
- wsm_masked_sparse_loss.

Current accepted cache coverage:

- canonical TRAIN rows: 6325;
- valid TRAIN video artifacts: 6255;
- unavailable TRAIN rows (no_body_detected): 70;
- canonical DEV rows: 933;
- valid DEV video artifacts: 907;
- unavailable DEV rows (no_body_detected): 26;
- total valid TRAIN+DEV artifacts: 7162;
- Test cache rows: 0.

The 96 no_body_detected rows must remain explicitly unavailable. Do not fabricate zero-video samples for them.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 2-10, 13-14
3. Stage 2 in docs/PLAN.md
4. docs/PROGRESS_EN.md through TASK-002H and MANAGER-DECISION-004
5. docs/NEXT_TASK_EN.md
6. src/video/models/depart_v1.py
7. src/common/loss/wsm_masked_sparse_loss.py
8. src/fusion/data/wsm_manifest_datamodule.py
9. src/audio/data/wsm_audio_segment_dataset.py for Chimera Batch/collate conventions only
10. /media/maxim/Programs/Features/WSM/video_depart_v1/cache/cache_index.jsonl structure as needed

## Allowed Tracked Files

- src/video/data/__init__.py
- src/video/data/wsm_video_cache_datamodule.py
- src/chimera_plugin.py
- docs/PROGRESS_EN.md

Creating src/video/data is allowed.

No other tracked file may be modified.

## Fixed Registry Key

Register:

    wsm_video_depart_v1_datamodule

## Data Contract

### Dataset sample

Each valid cached sample must expose enough information to construct:

- inputs["video"]: float tensor [T,512], where 1 <= T <= 60;
- artifact valid_mask: bool [T];
- targets: float tensor [2] ordered [depression, parkinson];
- observed_mask: bool tensor [2];
- metadata:
  - segment_id;
  - video_id;
  - corpus;
  - split;
  - cache_path;
  - cache_fingerprint;
  - temporal_length;
  - detection_coverage.

Unknown disease target must remain NaN with observed_mask=false.

### Collate

Implement a video-local collate function.

Given variable-length samples:

1. batch_max_t = max(real T in batch);
2. create zero-padded video tensor [B,batch_max_t,512];
3. create bool video_mask [B,batch_max_t];
4. copy each artifact feature tensor into [:T];
5. combine the artifact's own valid_mask with batch padding:
   - positions beyond T are false;
   - within T, positions use artifact valid_mask exactly;
6. padded positions must be exact zero;
7. return chimera_ml.core.batch.Batch with:
   - inputs={"video": padded_video};
   - targets=[B,2];
   - masks={
       "video_mask": video_mask,
       "observed_mask": observed_mask,
     };
   - meta containing sample metadata.

Do not replace internal detector-missed positions with true mask values.

### Cache availability policy

Read the persistent cache index:

    /media/maxim/Programs/Features/WSM/video_depart_v1/cache/cache_index.jsonl

For requested TRAIN/DEV rows:

- status extracted/reused with valid artifact -> include in dataset;
- status failed/no_body_detected -> exclude from unimodal video dataset, count as unavailable;
- missing index row -> fail clearly;
- any unexpected failure category -> fail clearly for this accepted cache state;
- Test index rows must not be selected or loaded.

## DataModule Requirements

Constructor parameters must include at least:

    data_root
    cache_root

Optional:

    cache_index_path

Behavior:

- build/load the canonical manifest contract;
- use only train and dev rows;
- do not create or load a Test dataset in this task;
- join canonical rows to cache index by segment_id;
- validate every selected cache artifact before dataset use:
  - features [T,512], 1<=T<=60;
  - bool valid_mask [T];
  - finite valid features;
  - exact-zero invalid features;
  - matching segment_id/fingerprint;
  - pinned CLIP identity/revision;
  - pinned YOLO SHA;
  - target_frames metadata=60;
- train_dataset contains only valid TRAIN artifacts;
- val_dataset contains only valid DEV artifacts;
- test_dataset must be absent, None, or empty and must not cause any Test cache access.

Expected lengths under current accepted cache:

    len(train_dataset) == 6255
    len(val_dataset) == 907

Unavailable counts:

    train_no_body_unavailable == 70
    dev_no_body_unavailable == 26

## Context Description

describe_context must expose at least:

- data.num_tasks = 2;
- data.task_names = ["depression", "parkinson"];
- data.video_feature_dim = 512;
- data.video_sequence_steps = 60;
- data.video_cache_root;
- data.video_train_rows = 6255;
- data.video_dev_rows = 907;
- data.video_train_unavailable = 70;
- data.video_dev_unavailable = 26;
- data.video_cache_success_total = 7162;
- data.video_cache_failure_total = 96;
- data.video_cache_variable_length = true;
- data.test_rows_loaded = 0.

Do not expose Test metrics.

## Chimera Registration

Update src/chimera_plugin.py with explicit import of the new video data registration module.

Project-module import failure must not be hidden as an optional warning.

## Acceptance Criteria

- DATAMODULES contains wsm_video_depart_v1_datamodule;
- plugin import has no project-module warning for video.data.wsm_video_cache_datamodule;
- train_dataset length = 6255;
- val_dataset length = 907;
- unavailable counts train/dev = 70/26;
- no Test cache row is loaded;
- no Test dataset is used for validation;
- a mixed-length collate produces [B,max_T,512] and bool [B,max_T];
- batch max_T is real batch maximum, not forcibly 60;
- padding positions are exact zero and mask=false;
- detector-missed internal positions preserve artifact mask=false;
- targets shape [B,2];
- observed_mask shape [B,2];
- depression/Parkinson ownership masks remain correct;
- masked NaN targets remain unknown;
- V1 model accepts a real cache batch;
- masked sparse loss accepts that batch;
- one real-cache forward/loss/backward smoke passes with finite loss/gradients;
- no training;
- no Test processing/metrics;
- src/audio unchanged;
- python compilation passes;
- registry smoke passes;
- git diff --check passes;
- branch codex/task-002i committed and pushed;
- main/master untouched;
- tracked diff contains only the four allowed paths.

## Exact Verification Commands

Run from repository root.

Compile:

    python3 -m py_compile \
      src/video/data/__init__.py \
      src/video/data/wsm_video_cache_datamodule.py \
      src/chimera_plugin.py

Registry smoke:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import warnings
    from chimera_ml.core.registry import DATAMODULES
    import chimera_plugin

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chimera_plugin.register()

    project_warnings = [
        str(item.message)
        for item in caught
        if "Failed to import" in str(item.message)
        and "video.data.wsm_video_cache_datamodule" in str(item.message)
    ]
    assert not project_warnings, project_warnings
    assert "wsm_video_depart_v1_datamodule" in DATAMODULES.keys()
    print("V1 video datamodule registry smoke passed")
    PY

Real cache/DataModule smoke:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import math
    import torch

    from common.loss.wsm_masked_sparse_loss import WSMMaskedSparseLoss
    from video.data.wsm_video_cache_datamodule import WSMVideoCacheDataModule
    from video.models.depart_v1 import WSMVideoDepartV1Model

    dm = WSMVideoCacheDataModule(
        data_root="/media/maxim/Databases/WSM_NEW",
        cache_root="/media/maxim/Programs/Features/WSM/video_depart_v1/cache",
    )

    assert len(dm.train_dataset) == 6255
    assert len(dm.val_dataset) == 907
    assert dm.train_unavailable_count == 70
    assert dm.dev_unavailable_count == 26
    assert getattr(dm, "test_dataset", None) in (None, {}, [])

    # Find at least one short real sequence and one T=60 sequence.
    short_i = next(i for i in range(len(dm.train_dataset))
                   if dm.train_dataset[i]["inputs"]["video"].shape[0] < 60)
    full_i = next(i for i in range(len(dm.train_dataset))
                  if dm.train_dataset[i]["inputs"]["video"].shape[0] == 60)

    samples = [dm.train_dataset[short_i], dm.train_dataset[full_i]]
    batch = dm.collate_fn(samples)

    video = batch.inputs["video"]
    mask = batch.get_masks("video_mask")
    observed = batch.get_masks("observed_mask")

    assert video.ndim == 3 and video.shape[0] == 2 and video.shape[2] == 512
    assert video.shape[1] == 60
    assert mask.dtype == torch.bool and tuple(mask.shape) == tuple(video.shape[:2])
    assert tuple(batch.targets.shape) == (2, 2)
    assert tuple(observed.shape) == (2, 2)

    short_t = samples[0]["inputs"]["video"].shape[0]
    assert not bool(mask[0, short_t:].any())
    assert torch.equal(video[0, short_t:], torch.zeros_like(video[0, short_t:]))

    # Preserve internal artifact invalid positions exactly.
    original_mask = samples[0]["video_mask"]
    assert torch.equal(mask[0, :short_t], original_mask)

    model = WSMVideoDepartV1Model(
        video_feature_dim=512,
        hidden_dim=64,
        num_layers=1,
        num_heads=4,
        ff_mult=2,
        dropout=0.0,
        sequence_steps=60,
    )
    output = model(batch)
    assert tuple(output.preds.shape) == (2, 2)

    loss = WSMMaskedSparseLoss()(output, batch)
    assert loss.ndim == 0
    assert math.isfinite(float(loss.detach()))
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)

    print("V1 video real-cache datamodule/model/loss smoke passed", float(loss.detach()))
    PY

Test-firewall/source audit:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    from video.data.wsm_video_cache_datamodule import WSMVideoCacheDataModule

    dm = WSMVideoCacheDataModule(
        data_root="/media/maxim/Databases/WSM_NEW",
        cache_root="/media/maxim/Programs/Features/WSM/video_depart_v1/cache",
    )
    assert dm.test_rows_loaded == 0
    assert all(dm.train_dataset[i]["meta"]["split"] == "train" for i in range(min(64, len(dm.train_dataset))))
    assert all(dm.val_dataset[i]["meta"]["split"] == "dev" for i in range(min(64, len(dm.val_dataset))))
    print("video datamodule Test firewall assertions passed")
    PY

Then:

    git diff --check
    git diff -- src/audio
    git status --short

Before commit inspect only:

    git diff -- \
      src/video/data/__init__.py \
      src/video/data/wsm_video_cache_datamodule.py \
      src/chimera_plugin.py \
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
- registry key;
- cache root/index used;
- valid train/dev dataset counts;
- unavailable train/dev counts;
- explicit no-fake-feature policy;
- variable-length padding semantics;
- real cache batch shapes;
- forward/loss/backward smoke result;
- confirmation Test rows loaded=0;
- confirmation no training/Test metrics;
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

- branch codex/task-002i;
- implementation commit SHA;
- pushed-to-origin status;
- main/master untouched;
- train/dev dataset counts;
- unavailable counts;
- real-cache collate shapes;
- forward/loss/backward result;
- Test rows loaded=0;
- no training/Test metrics;
- src/audio unchanged.

Stop after this task.
