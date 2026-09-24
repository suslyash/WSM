# TASK-002L: Create the V1 Video Training Config and Wire Four Epoch-Level Evaluation Streams

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-002l

This is the LAST pre-training gate.

Do not run the real/full V1 training experiment in this task. Only configuration validation and a bounded integration smoke are authorized.

## Goal

Make the accepted V1 video pipeline ready for the first real training run.

The training loop must evaluate and log all four named evaluation streams every epoch:

    dev
    test_none
    test_soft
    test_hard

Automatic selection remains based ONLY on:

    dev/mean_score

The three Test protocol metrics are mandatory epoch-level monitoring outputs but MUST NOT drive checkpointing, early stopping, scheduler selection, threshold search, hyperparameter selection, or architecture decisions.

Use the accepted full-coverage DEPART-compatible cache:

    /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache

## Verified Chimera 0.2.4 Integration Contract

The installed/project-pinned Chimera training path supports this without a custom trainer:

- DataModule.val_dataloader() may return Mapping[str, DataLoader];
- Trainer.fit() normalizes those loaders;
- every epoch it calls _run_epoch(..., split=<mapping key>) for each named validation stream;
- collect_cache=true stores CachedSplitOutputs per split;
- wsm_segment_metrics_callback reads those cached outputs and emits:
  - dev/...
  - test_none/...
  - test_soft/...
  - test_hard/...

Therefore do NOT fork or patch chimera-ml.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 5, 8, 9, 10, 13, 14
3. Stage 2 in docs/PLAN.md
4. docs/PROGRESS_EN.md through TASK-002K2
5. docs/NEXT_TASK_EN.md
6. src/video/data/wsm_video_cache_datamodule.py
7. src/video/models/depart_v1.py
8. src/common/loss/wsm_masked_sparse_loss.py
9. src/common/callbacks/wsm_segment_callback.py
10. src/common/callbacks/wsm_summary_callback.py
11. configs/wsm_mm_pd_dep_v1/audio/00_frozen_baseline.yaml as config-style reference
12. installed Chimera 0.2.4 Trainer.fit/DataModule behavior as verified above

## Allowed Tracked Files

- src/video/data/wsm_video_cache_datamodule.py
- configs/wsm_mm_pd_dep_v1/video/00_depart_v1.yaml
- docs/PROGRESS_EN.md

No other tracked file may be modified.

Do not modify src/audio.
Do not modify chimera-ml.

## 1. Epoch-Level Evaluation Loader Contract

Preserve:

    train_dataset
    val_dataset
    test_dataset

with current meanings:

- train_dataset: 6325 canonical Train rows
- val_dataset: 933 canonical DEV rows
- test_dataset["test_none"]: 1364 rows
- test_dataset["test_soft"]: 1208 rows
- test_dataset["test_hard"]: 1014 rows

Override WSMVideoCacheDataModule.val_dataloader() so it returns exactly a Mapping with keys:

    dev
    test_none
    test_soft
    test_hard

Semantics:

- "dev" loader is built from val_dataset;
- test loaders are built from their corresponding test_dataset entries;
- all four use shuffle=false and drop_last=false;
- all use the accepted variable-length collate;
- val_dataset itself remains DEV only;
- test_dataset remains a separate dict;
- do not alias Test datasets into val_dataset.

Do not change train_dataloader() semantics.

## 2. Training Config

Create:

    configs/wsm_mm_pd_dep_v1/video/00_depart_v1.yaml

It must be self-contained.

Required experiment identity:

    experiment_info.params.experiment_name: wsm_mm_pd_dep_v1

Use run_name:

    depart_v1_clip_yolo_transformer

Required data:

    name: wsm_video_depart_v1_datamodule

with:

- data_root=/media/maxim/Databases/WSM_NEW
- cache_root=/media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache
- batch_size=32
- num_workers=4
- pin_memory=true
- persistent_workers=true
- shuffle_train=true
- drop_last_train=false

Required model:

    name: wsm_video_depart_v1_model

Use the accepted V1 defaults:

- video_feature_dim=512
- hidden_dim=192
- num_layers=2
- num_heads=4
- ff_mult=4
- dropout=0.2
- sequence_steps=60
- num_tasks=2

Required loss:

    name: wsm_masked_sparse_loss

Do not add task-id loss selection.

Required optimizer:

    name: adamw_optimizer

Initial fixed V1 optimizer contract:

- lr=0.0001
- weight_decay=0.01

This is not a sweep.

Required train params:

- epochs=30
- device=cuda
- mixed_precision=true
- grad_clip_norm=0.5
- log_every_steps=25
- collect_cache=true

Do not configure a scheduler in this task unless config validation requires an explicit disabled scheduler.

Do not add generic metrics that cannot handle sparse [B,2] NaN targets.
The authoritative task metrics come from wsm_segment_metrics_callback.

## 3. Required Instrumentation

The YAML MUST include all required project instrumentation:

- checkpoint_callback
- snapshot_callback
- early_stopping_callback
- wsm_summary_callback
- wsm_segment_metrics_callback
- console_file_logger
- mlflow_logger

wsm_segment_metrics_callback:

    splits: [auto]

Checkpoint:

- monitor: dev/mean_score
- mode: max
- save_top_k: 2
- save_last: true

Early stopping:

- monitor: dev/mean_score
- mode: max
- patience: 6
- min_delta: 0.0005

No callback, logger, scheduler, or config field may monitor:

- test_none/*
- test_soft/*
- test_hard/*

Snapshot must save config and source as in the existing project convention.

## 4. Four-Stream Metric Contract

A single epoch-level validation cycle must produce all of:

    dev/depression/uar
    dev/depression/mf1
    dev/depression/score
    dev/parkinson/uar
    dev/parkinson/mf1
    dev/parkinson/score
    dev/mean_score

and equivalent keys under:

    test_none/
    test_soft/
    test_hard/

The Test keys are logged every epoch.

Only dev/mean_score is a selector.

## 5. Config Validation

Run Chimera config validation:

    .venv/bin/chimera-ml validate-config \
      --config-path configs/wsm_mm_pd_dep_v1/video/00_depart_v1.yaml

Then build through the actual Chimera registries/builders without starting the real experiment.

Verify:

- DataModule builds;
- model builds;
- loss builds;
- optimizer builds;
- callbacks build;
- loggers build where practical;
- no project-module warnings;
- train loader exists;
- val_dataloader() keys are exactly:
  dev, test_none, test_soft, test_hard.

## 6. Bounded One-Epoch Integration Smoke

A tiny bounded smoke is authorized ONLY to verify the training/evaluation loop wiring.

Do NOT run the full datasets.

Construct temporary tiny loader mappings from the real accepted datasets, for example using torch.utils.data.Subset:

- train: 4 real samples
- dev: 4 real samples with observed coverage sufficient for both tasks
- test_none: 4 real samples with both task ownerships represented
- test_soft: 4 real samples with both task ownerships represented
- test_hard: 4 real samples with both task ownerships represented

If four samples are insufficient to include both task ownerships in a protocol, use the smallest deterministic number needed. Record the exact smoke counts.

Run exactly one bounded epoch through Chimera Trainer.fit with:

- the real V1 model;
- the real masked sparse loss;
- the real wsm_segment_metrics_callback;
- collect_cache=true;
- no checkpoint writes to production paths;
- no MLflow production run;
- temporary output paths under /tmp/wsm_task002l/.

The smoke must prove that one epoch's callback logs contain:

    dev/mean_score
    test_none/mean_score
    test_soft/mean_score
    test_hard/mean_score

and that all four are finite.

This smoke is NOT a research training result and MUST NOT be reported as model performance.

Delete/ignore temporary smoke checkpoints/logs after verification.

## 7. Selector Firewall Smoke

Programmatically inspect the built config/callbacks and assert:

- checkpoint monitor == "dev/mean_score";
- checkpoint mode == "max";
- early stopping monitor == "dev/mean_score";
- early stopping mode == "max";
- no automatic selector component references test_none, test_soft, or test_hard;
- Test metrics are nevertheless present in one-epoch logs.

## Acceptance Criteria

DataModule:

- train_dataset=6325;
- val_dataset=933;
- test_none=1364;
- test_soft=1208;
- test_hard=1014;
- val_dataloader() returns exactly four named streams;
- DEV remains separate from Test datasets;
- all use correct collate;
- no unavailable rows.

Config:

- self-contained video YAML exists;
- experiment_name correct;
- correct DataModule/model/loss registry keys;
- accepted cache root used;
- required callbacks/loggers all present;
- collect_cache=true;
- checkpoint/early stop use only dev/mean_score max;
- no Test selector reference.

Verification:

- chimera-ml validate-config passes;
- plugin/registry/build smoke passes;
- bounded one-epoch integration smoke passes;
- four finite mean_score keys appear in the same epoch logs;
- forward/loss/backward occurs successfully in smoke;
- no full training experiment is run;
- no hyperparameter/model selection occurs;
- no dependency install;
- src/audio unchanged;
- git diff --check passes;
- branch codex/task-002l committed and pushed;
- main/master untouched;
- tracked diff contains only the three allowed paths.

## Exact Verification Commands

Compile:

    python3 -m py_compile \
      src/video/data/wsm_video_cache_datamodule.py

Validate config:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml validate-config \
      --config-path configs/wsm_mm_pd_dep_v1/video/00_depart_v1.yaml

DataModule stream smoke:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    from video.data.wsm_video_cache_datamodule import WSMVideoCacheDataModule

    dm = WSMVideoCacheDataModule(
        data_root="/media/maxim/Databases/WSM_NEW",
        cache_root="/media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache",
        batch_size=32,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    assert len(dm.train_dataset) == 6325
    assert len(dm.val_dataset) == 933
    assert len(dm.test_dataset["test_none"]) == 1364
    assert len(dm.test_dataset["test_soft"]) == 1208
    assert len(dm.test_dataset["test_hard"]) == 1014

    loaders = dm.val_dataloader()
    assert list(loaders.keys()) == ["dev", "test_none", "test_soft", "test_hard"]

    print("TASK-002L four-stream DataModule contract passed")
    PY

Add a bounded one-epoch Trainer.fit smoke as specified above and assert the four mean_score keys.

Selector firewall:

    grep -RniE 'monitor:.*test_(none|soft|hard)|scheduler_monitor:.*test_(none|soft|hard)' \
      configs/wsm_mm_pd_dep_v1/video/00_depart_v1.yaml && exit 1 || true

    grep -n 'monitor: dev/mean_score' \
      configs/wsm_mm_pd_dep_v1/video/00_depart_v1.yaml

Finally:

    git diff --check
    git diff -- src/audio
    git status --short

Before commit inspect only:

    git diff -- \
      src/video/data/wsm_video_cache_datamodule.py \
      configs/wsm_mm_pd_dep_v1/video/00_depart_v1.yaml \
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
- config path;
- exact fixed V1 model/optimizer/train parameters;
- full DataModule dataset counts;
- val_dataloader four-stream keys;
- required instrumentation presence;
- exact selector monitor/mode;
- config validation result;
- registry/build result;
- bounded smoke sample counts;
- four mean_score smoke values, explicitly labelled structural/non-performance smoke values;
- confirmation Test metrics appeared every epoch in the smoke;
- confirmation Test metrics were not selector inputs;
- confirmation no full training/model-selection run;
- src/audio unchanged;
- Stage 2 is ready for the first real V1 training run if all acceptance criteria pass;
- recommended next atomic step: TASK-002M real V1 video training run.

## Required Handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch codex/task-002l;
- implementation commit SHA;
- pushed-to-origin status;
- main/master untouched;
- config path;
- four validation stream names;
- selector key;
- config validation result;
- bounded integration smoke result;
- no full training;
- src/audio unchanged.

Stop after this task.
