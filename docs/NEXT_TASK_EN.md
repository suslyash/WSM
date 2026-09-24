# TASK-004C: Run the Fixed F0 Audio+Video Gated Late-Fusion Baseline

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-004c

This task explicitly authorizes the first real Stage 4 fusion training run.

Do not modify source code, the accepted A+V DataModule, frozen audio/video caches, loss, callbacks, or Test protocol definitions.

## Goal

Create the fixed F0 production config and run the real seed-42 A+V gated late-fusion baseline.

Use:

    data: wsm_av_fusion_datamodule
    model: wsm_av_f0_gated_late_model
    loss: wsm_masked_sparse_loss

Evaluate every epoch on:

    dev
    test_none
    test_soft
    test_hard

Select the checkpoint ONLY by:

    dev/mean_score

Test metrics are mandatory monitoring outputs only.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 5, 8, 9, 10, 13, 14
3. docs/PLAN.md Stage 4
4. docs/PROGRESS_EN.md through MANAGER-DECISION-018
5. docs/NEXT_TASK_EN.md
6. src/fusion/data/wsm_av_fusion_datamodule.py
7. src/fusion/models/av_f0_gated_late.py
8. configs/wsm_mm_pd_dep_v1/video/00_depart_v1.yaml
9. configs/wsm_mm_pd_dep_v1/video/01_depart_v2_prototype.yaml
10. configs/wsm_mm_pd_dep_v1/audio/00_frozen_baseline.yaml

## Allowed Tracked Files

- configs/wsm_mm_pd_dep_v1/fusion/00_f0_gated_late.yaml
- docs/PROGRESS_EN.md

No other tracked file may be modified.

Generated runtime artifacts under logs/ are expected and must remain untracked unless the repository already tracks a specific runtime artifact type.

Do not modify src/audio.
Do not modify src/video.
Do not modify src/fusion source.

## 1. Fixed F0 Config

Create:

    configs/wsm_mm_pd_dep_v1/fusion/00_f0_gated_late.yaml

Use exactly:

    seed: 42

Experiment:

    experiment_name: wsm_mm_pd_dep_v1
    run_name: av_f0_gated_late

Data:

    name: wsm_av_fusion_datamodule

Data params:

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

    name: wsm_av_f0_gated_late_model

Model params:

    audio_feature_dim: 768
    video_feature_dim: 512
    hidden_dim: 192
    gate_hidden_dim: 64
    dropout: 0.2
    num_tasks: 2

Loss:

    name: wsm_masked_sparse_loss

Optimizer:

    name: adamw_optimizer
    lr: 0.0001
    weight_decay: 0.01

Training:

    epochs: 30
    device: cuda
    mixed_precision: true
    grad_clip_norm: 0.5
    log_every_steps: 25
    collect_cache: true

Metrics:

    []

Do not add a scheduler.
Do not add a different loss.
Do not add any pseudo-label, contrastive, prototype, or auxiliary objective.

## 2. Required Instrumentation

Include exactly the accepted project instrumentation:

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

Use the same filename_template as the accepted video configs.

Snapshot:

    log_path: logs
    include:
      - src
      - configs
    save_code_zip: true
    save_config: true

Early stopping:

    monitor: dev/mean_score
    mode: max
    patience: 6
    min_delta: 0.0005

Console:

    log_path: logs
    log_file: train.log
    console_level: INFO
    file_level: INFO

MLflow:

    tracking_uri: sqlite:///logs/mlflow.db

No Test metric may appear in any selector/monitor/scheduler field.

## 3. Pre-Run Validation

Run:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml validate-config \
      --config-path configs/wsm_mm_pd_dep_v1/fusion/00_f0_gated_late.yaml

Build through the actual Chimera registries and verify:

- DataModule builds;
- F0 model builds;
- masked sparse loss builds;
- optimizer builds;
- callbacks/loggers build;
- no project-module warnings;
- CUDA is available;
- dataset counts are exactly:
  - train=6325
  - dev=933
  - test_none=1364
  - test_soft=1208
  - test_hard=1014
- joined_total=8622;
- missing audio/video=0/0;
- val_dataloader keys are exactly:
  - dev
  - test_none
  - test_soft
  - test_hard;
- checkpoint and early stopping monitor only dev/mean_score.

Run one real-batch forward/loss/backward smoke from the production config before the full run.

If CUDA is unavailable, stop as blocked. Do not silently switch the real run to CPU.

If the config/build/forward smoke fails, do not modify source code in this task. Preserve the exact blocker and stop.

## 4. Real F0 Training

Run from repository root:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/fusion/00_f0_gated_late.yaml

Do not interrupt a healthy run.

If the run fails:

- preserve traceback/logs/artifacts;
- do not modify source/config beyond the already-created fixed config;
- record the exact runtime blocker;
- update PROGRESS_EN.md;
- commit/push evidence;
- stop as blocked.

## 5. Epoch-Level Monitoring

For EVERY completed epoch, record at minimum:

DEV:

- depression UAR/MF1/Score;
- Parkinson UAR/MF1/Score;
- Mean_Score.

TEST_NONE:

- depression UAR/MF1/Score;
- Parkinson UAR/MF1/Score;
- Mean_Score.

TEST_SOFT:

- depression UAR/MF1/Score;
- Parkinson UAR/MF1/Score;
- Mean_Score.

TEST_HARD:

- depression UAR/MF1/Score;
- Parkinson UAR/MF1/Score;
- Mean_Score.

Also record:

- train loss;
- DEV loss if emitted;
- epoch duration if available.

All required metrics must be finite.

All four streams must appear on every completed epoch.

Test metrics are monitoring-only.

Do NOT:

- choose an epoch because of Test;
- tune gates using Test;
- change the threshold using Test;
- manually stop because of Test;
- alter any hyperparameter based on Test.

## 6. Best Checkpoint Selection

After training/early stopping:

1. select the best epoch ONLY by maximum dev/mean_score;
2. record best epoch and exact checkpoint path;
3. record DEV depression/Parkinson UAR/MF1/Score and DEV Mean_Score;
4. record TEST_NONE/SOFT/HARD task metrics and Mean_Scores from that SAME DEV-selected epoch;
5. record top-2 checkpoints and last.pt;
6. record train.log, summary.txt, code.zip, and MLflow run ID/status.

Do not select another epoch because its Test values are better.

## 7. F0 Gate Diagnostics

At the DEV-selected checkpoint, run a DEV-only diagnostic pass without changing model parameters.

Report for each task:

- mean audio fusion weight;
- mean video fusion weight;
- standard deviation of audio fusion weight;
- min/max audio fusion weight.

Since current canonical A+V rows have both modalities available, these weights should correspond to:

    [raw_audio_gate, 1 - raw_audio_gate]

Report diagnostics only.

Do not tune or select based on gate statistics.

## 8. Descriptive DEV Comparison

Compare the DEV-selected F0 result to the accepted references:

Frozen historical audio:

- DEV Mean_Score = 0.787827
- depression Score = 0.747918
- Parkinson Score = 0.827735

Selected video V2:

- DEV Mean_Score = 0.706572
- depression Score = 0.620101
- Parkinson Score = 0.793043

Report:

    F0 - audio DEV Mean_Score delta
    F0 - V2 DEV Mean_Score delta

and task-level Score deltas.

This comparison is descriptive Stage 4 evidence.

Do not use Test metrics to decide whether F0 is successful or whether to proceed to F1.

F1 remains the next planned Stage 4 baseline regardless of F0's Test values.

## Acceptance Criteria

The task passes if:

- F0 YAML is self-contained and validates;
- config contains the fixed F0 contract above;
- CUDA gate passes;
- production DataModule/model/loss/optimizer/callback/logger build passes;
- full run completes or ends via configured early stopping;
- all four streams are evaluated every completed epoch;
- all required metrics are finite;
- best checkpoint selected strictly by dev/mean_score;
- Test metrics never drive selection/tuning;
- DEV-selected gate diagnostics are recorded;
- durable checkpoints/log/summary/code archive/MLflow artifacts are retained;
- no source code changes;
- no dependency installation;
- text/description remains deferred;
- src/audio unchanged;
- src/video unchanged;
- A+V DataModule source unchanged;
- git diff --check passes;
- tracked diff contains only:
  - configs/wsm_mm_pd_dep_v1/fusion/00_f0_gated_late.yaml
  - docs/PROGRESS_EN.md
- branch codex/task-004c committed and pushed;
- main/master untouched.

## Required PROGRESS_EN Update

Append TASK-004C evidence without erasing prior records.

Record:

- branch;
- evidence commit SHA;
- push result;
- exact config path;
- exact training command;
- config validation/build results;
- GPU/device;
- dataset/join counts;
- epochs completed;
- early-stopping status;
- best epoch;
- best dev/mean_score;
- best DEV depression/Parkinson UAR/MF1/Score;
- same-epoch Test protocol task metrics and Mean_Scores;
- full epoch table containing at minimum:
  - epoch
  - train loss
  - DEV depression Score
  - DEV Parkinson Score
  - DEV Mean_Score
  - TEST_NONE Mean_Score
  - TEST_SOFT Mean_Score
  - TEST_HARD Mean_Score
- top-2/last checkpoint paths;
- log/summary/code archive/MLflow locations;
- DEV-selected F0 gate statistics;
- F0 vs historical audio DEV deltas;
- F0 vs video V2 DEV deltas;
- explicit Test-selector firewall confirmation;
- confirmation no source changes;
- confirmation text/description remains deferred;
- src/audio unchanged;
- src/video unchanged;
- Stage 4 status;
- recommended next atomic step: implement the F1 sparse shared-representation two-head MTL model contract, without pseudo-labeling and without text/description.

## Required Handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch codex/task-004c;
- evidence commit SHA;
- pushed-to-origin status;
- main/master untouched;
- config path;
- epochs completed;
- best epoch;
- best dev/mean_score;
- DEV depression/Parkinson Scores;
- same-epoch TEST_NONE/SOFT/HARD Mean_Scores;
- DEV gate diagnostics;
- F0 vs audio/V2 DEV deltas;
- selector firewall confirmation;
- no source changes;
- text/description deferred;
- src/audio unchanged;
- src/video unchanged.

Stop after this task.
