# TASK-004E: Run the Fixed F1 Shared-Representation Sparse A+V MTL Baseline

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-004e

This task explicitly authorizes the real F1 Stage 4 baseline run.

Do not modify source code, the accepted A+V DataModule, F0/F1 model code, frozen audio/video caches, loss, callbacks, or Test protocol definitions.
Do not add pseudo-labeling.
Do not start F2.
Do not start text/description work.

## Goal

Create the fixed F1 production config and run the real seed-42 shared-representation sparse A+V multitask baseline.

Use:

    data: wsm_av_fusion_datamodule
    model: wsm_av_f1_shared_mtl_model
    loss: wsm_masked_sparse_loss

Evaluate every epoch on:

    dev
    test_none
    test_soft
    test_hard

Select the checkpoint ONLY by:

    dev/mean_score

The purpose is to measure the DEV gain/loss of F1 shared representation relative to the already measured F0 baseline before implementing F2.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 5, 8, 9, 10, 13, 14
3. docs/PLAN.md Stage 4
4. docs/PROGRESS_EN.md through MANAGER-DECISION-020
5. docs/NEXT_TASK_EN.md
6. src/fusion/data/wsm_av_fusion_datamodule.py
7. src/fusion/models/av_f0_gated_late.py
8. src/fusion/models/av_f1_shared_mtl.py
9. configs/wsm_mm_pd_dep_v1/fusion/00_f0_gated_late.yaml

## Allowed Tracked Files

- configs/wsm_mm_pd_dep_v1/fusion/01_f1_shared_mtl.yaml
- docs/PROGRESS_EN.md

No other tracked file may be modified.

Generated runtime artifacts under logs/ are expected and must remain untracked unless already tracked by repository policy.

## 1. Create the Fixed F1 Config

Create:

    configs/wsm_mm_pd_dep_v1/fusion/01_f1_shared_mtl.yaml

It must mirror the accepted F0 production config except for the planned F1 model-family changes.

Use exactly:

    seed: 42

Experiment:

    experiment_name: wsm_mm_pd_dep_v1
    run_name: av_f1_shared_mtl

Data:

    name: wsm_av_fusion_datamodule

Data params must match F0 exactly:

    data_root: /media/maxim/Databases/WSM_NEW
    audio_feature_cache_root: /media/maxim/Databases/WSM_NEW/features
    video_cache_root: /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache
    batch_size: 32
    num_workers: 4
    pin_memory: true
    persistent_workers: true
    shuffle_train: true
    drop_last_train: false

Model:

    name: wsm_av_f1_shared_mtl_model

Model params:

    audio_feature_dim: 768
    video_feature_dim: 512
    hidden_dim: 192
    fusion_hidden_dim: 192
    dropout: 0.2
    num_tasks: 2

Loss:

    name: wsm_masked_sparse_loss

Optimizer must match F0 exactly:

    name: adamw_optimizer
    lr: 0.0001
    weight_decay: 0.01

Training must match F0 exactly:

    epochs: 30
    device: cuda
    mixed_precision: true
    grad_clip_norm: 0.5
    log_every_steps: 25
    collect_cache: true

Metrics:

    []

Do not add a scheduler, auxiliary loss, pseudo-labeling, contrastive objective, task weights, or any tuning parameter.

## 2. Required Instrumentation

Use the same instrumentation and durable paths as F0:

- wsm_segment_metrics_callback with splits=[auto]
- checkpoint_callback
- snapshot_callback
- early_stopping_callback
- wsm_summary_callback
- console_file_logger
- mlflow_logger

Checkpoint:

    log_path: logs
    monitor: dev/mean_score
    mode: max
    save_top_k: 2
    save_last: true

Use the same filename_template as F0.

Early stopping:

    monitor: dev/mean_score
    mode: max
    patience: 6
    min_delta: 0.0005

Snapshot:

    log_path: logs
    include:
      - src
      - configs
    save_code_zip: true
    save_config: true

Console:

    log_path: logs
    log_file: train.log
    console_level: INFO
    file_level: INFO

MLflow:

    tracking_uri: sqlite:///logs/mlflow.db

No Test metric may appear in any selector, monitor, scheduler, threshold, or tuning field.

## 3. F0/F1 Config Equivalence Audit

Before training, parse:

    configs/wsm_mm_pd_dep_v1/fusion/00_f0_gated_late.yaml
    configs/wsm_mm_pd_dep_v1/fusion/01_f1_shared_mtl.yaml

Assert all non-model training/data/instrumentation semantics are identical.

The intentional semantic differences are only:

- experiment_info.params.run_name;
- model.name;
- F0 model param gate_hidden_dim is absent in F1;
- F1 model param fusion_hidden_dim=192 is present;
- remaining common model params are identical.

No data, optimizer, training, callback, logger, selector, seed, or loss difference is allowed.

## 4. Pre-Run Validation

Run:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml validate-config \
      --config-path configs/wsm_mm_pd_dep_v1/fusion/01_f1_shared_mtl.yaml

Build through actual Chimera registries and verify:

- DataModule builds;
- F1 model builds;
- masked sparse loss builds;
- optimizer builds;
- callbacks/loggers build;
- no project-module warnings;
- CUDA is available;
- dataset counts:
  - train=6325
  - dev=933
  - test_none=1364
  - test_soft=1208
  - test_hard=1014
- joined_total=8622;
- missing audio/video=0/0;
- val_dataloader keys exactly dev/test_none/test_soft/test_hard;
- checkpoint and early stopping monitor only dev/mean_score.

Run one real production-size batch forward/loss/backward smoke before the full run.

If CUDA is unavailable or config/build/smoke fails, stop as blocked. Do not change source/config to work around it in this task.

## 5. Real F1 Training

Run from repository root:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/fusion/01_f1_shared_mtl.yaml

Do not interrupt a healthy run.

If it fails:

- preserve traceback/logs/artifacts;
- do not modify source or the fixed config;
- record the exact blocker;
- update PROGRESS_EN.md;
- commit/push evidence;
- stop as blocked.

## 6. Epoch-Level Monitoring

For EVERY completed epoch record at minimum:

- train loss;
- DEV depression UAR/MF1/Score;
- DEV Parkinson UAR/MF1/Score;
- DEV Mean_Score;
- TEST_NONE depression/Parkinson UAR/MF1/Score and Mean_Score;
- TEST_SOFT depression/Parkinson UAR/MF1/Score and Mean_Score;
- TEST_HARD depression/Parkinson UAR/MF1/Score and Mean_Score.

All required metrics must be finite.

All four evaluation streams must appear every completed epoch.

Test metrics are monitoring-only.

Do NOT:

- select an epoch from Test;
- tune shared fusion using Test;
- alter threshold using Test;
- manually stop because of Test;
- change any hyperparameter based on Test.

## 7. Best Checkpoint Selection

After training or configured early stopping:

1. choose best epoch ONLY by maximum dev/mean_score;
2. record exact best checkpoint path;
3. record DEV task UAR/MF1/Score and DEV Mean_Score;
4. record Test protocol metrics from that SAME DEV-selected epoch;
5. record top-2 checkpoints and last.pt;
6. record train.log, summary.txt, code.zip;
7. record the exact MLflow experiment name, run ID, and final run status.

Do not omit the MLflow run ID/status.

## 8. F1 Shared-Task Gradient Diagnostic

At the DEV-selected checkpoint, perform one deterministic diagnostic only; do not update model parameters.

Use a deterministic small TRAIN sample containing at least:

- observed depression-positive/negative rows;
- observed Parkinson-positive/negative rows.

Using the selected checkpoint:

1. compute the depression-only BCE-with-logits loss on rows with observed depression labels;
2. compute the Parkinson-only BCE-with-logits loss on rows with observed Parkinson labels;
3. compute gradients of each task loss separately with respect to ONLY the shared_fusion trainable parameters;
4. flatten each gradient vector;
5. report:
   - depression shared-gradient L2 norm;
   - Parkinson shared-gradient L2 norm;
   - cosine similarity between the two shared-gradient vectors.

Requirements:

- gradients finite;
- no optimizer step;
- no parameter update;
- do not use Test data;
- diagnostics must not alter checkpoint/model selection.

If a deterministic 32-row batch does not contain both classes for both observed tasks, use the smallest deterministic sample needed and record its size.

## 9. DEV Comparison

Compare the DEV-selected F1 result to:

F0:
- DEV Mean_Score=0.744211
- depression Score=0.642220
- Parkinson Score=0.846201

Frozen historical audio:
- DEV Mean_Score=0.787827
- depression Score=0.747918
- Parkinson Score=0.827735

Selected video V2:
- DEV Mean_Score=0.706572
- depression Score=0.620101
- Parkinson Score=0.793043

Report:

    F1 - F0 DEV Mean_Score delta
    F1 - audio DEV Mean_Score delta
    F1 - V2 DEV Mean_Score delta

and task-level DEV Score deltas.

The primary Stage 4 F1 conclusion is F1 versus F0 on DEV.

Test deltas may be reported descriptively but MUST NOT determine the conclusion or next architecture.

F2 remains the next planned model regardless of Test behavior.

## Acceptance Criteria

The task passes if:

- F1 YAML is self-contained and valid;
- F0/F1 config equivalence audit passes;
- CUDA gate passes;
- production registry/build/forward/backward smoke passes;
- real run completes or configured early stopping ends it;
- all four streams are evaluated every completed epoch;
- all required metrics are finite;
- best checkpoint selected strictly by dev/mean_score;
- Test metrics never influence selection/tuning;
- exact MLflow run ID/status is recorded;
- selected-checkpoint shared-task gradient norm/cosine diagnostic is finite and recorded;
- F1 versus F0 DEV delta is recorded;
- no source changes;
- no dependency installation;
- no pseudo-labeling;
- no F2 work;
- text/description remains deferred;
- src/audio unchanged;
- src/video unchanged;
- accepted A+V DataModule unchanged;
- F0 model unchanged;
- git diff --check passes;
- tracked diff contains only:
  - configs/wsm_mm_pd_dep_v1/fusion/01_f1_shared_mtl.yaml
  - docs/PROGRESS_EN.md
- branch codex/task-004e committed and pushed;
- main/master untouched.

## Required PROGRESS_EN Update

Append TASK-004E evidence without erasing prior records.

Record:

- branch;
- evidence commit SHA;
- push result;
- config path;
- config equivalence result;
- exact training command;
- GPU/device;
- dataset/join counts;
- epochs completed;
- early-stopping status;
- best epoch;
- best dev/mean_score;
- best DEV depression/Parkinson UAR/MF1/Score;
- same-epoch Test task metrics and Mean_Scores;
- full epoch table;
- top-2/last checkpoint paths;
- run directory, train.log, summary.txt, code.zip;
- MLflow experiment name, run ID, and final status;
- shared_fusion gradient norms and cosine diagnostic plus diagnostic sample size;
- F1 vs F0 DEV deltas;
- F1 vs frozen audio/video V2 DEV deltas;
- explicit Test-selector firewall confirmation;
- confirmation no source changes;
- confirmation F2 not started;
- confirmation text/description deferred;
- src/audio unchanged;
- src/video unchanged;
- Stage 4 status;
- recommended next atomic step: TASK-004F implement/register the fixed F2 task-aware directed fusion baseline, with observed loss only and no pseudo-labeling.

## Required Handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch codex/task-004e;
- evidence commit SHA;
- pushed-to-origin status;
- main/master untouched;
- config path;
- epochs completed;
- best epoch;
- best dev/mean_score;
- DEV depression/Parkinson Scores;
- same-epoch TEST_NONE/SOFT/HARD Mean_Scores;
- F1-F0 DEV delta;
- shared-task gradient norm/cosine diagnostic;
- exact MLflow run ID/status;
- selector firewall confirmation;
- no source changes;
- F2 not started;
- text/description deferred;
- src/audio unchanged;
- src/video unchanged.

Stop after this task.
