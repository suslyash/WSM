# TASK-004G: Run the Fixed F2 Task-Aware Directed A+V Relation-Bank Baseline

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-004g

This task explicitly authorizes the real F2 Stage 4 training run.

Do not modify source code, the accepted A+V DataModule, F0/F1/F2 model code, frozen audio/video caches, loss, callbacks, metric definitions, or Test protocol definitions.
Do not add pseudo-labeling, flow matching, PAGB, auxiliary losses, or text/description.
Do not start Stage 5 RAMPS.

## Critical Evaluation Rule

Training loss is an optimization diagnostic only.

The experiment is evaluated by metrics:

- DEV depression UAR/MF1/Score
- DEV Parkinson UAR/MF1/Score
- DEV Mean_Score
- TEST_NONE depression/Parkinson UAR/MF1/Score and Mean_Score
- TEST_SOFT depression/Parkinson UAR/MF1/Score and Mean_Score
- TEST_HARD depression/Parkinson UAR/MF1/Score and Mean_Score

These four metric streams MUST be produced on every completed epoch.

Checkpoint selection and early stopping MUST use ONLY:

    dev/mean_score

No loss value and no Test metric may select the model.

## Goal

Create the fixed F2 production config and run the real seed-42 task-aware directed relation-bank A+V baseline.

Use:

    data: wsm_av_fusion_datamodule
    model: wsm_av_f2_task_aware_directed_model
    loss: wsm_masked_sparse_loss

The primary Stage 4 question is whether F2 task-aware directed relations improve over the measured F1 shared-representation baseline on DEV.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 5, 8, 9, 10, 13, 14
3. docs/PLAN.md Stage 4
4. docs/PROGRESS_EN.md through MANAGER-DECISION-022
5. docs/NEXT_TASK_EN.md
6. src/fusion/data/wsm_av_fusion_datamodule.py
7. src/fusion/models/av_f2_task_aware_directed.py
8. configs/wsm_mm_pd_dep_v1/fusion/01_f1_shared_mtl.yaml
9. configs/wsm_mm_pd_dep_v1/fusion/00_f0_gated_late.yaml

## Allowed Tracked Files

- configs/wsm_mm_pd_dep_v1/fusion/02_f2_task_aware_directed.yaml
- docs/PROGRESS_EN.md

No other tracked file may be modified.

Generated runtime artifacts under logs/ must remain untracked unless repository policy already tracks them.

## 1. Create the Fixed F2 Config

Create:

    configs/wsm_mm_pd_dep_v1/fusion/02_f2_task_aware_directed.yaml

It must mirror the accepted F1 production config for all non-model training/data/instrumentation semantics.

Use exactly:

    seed: 42

Experiment:

    experiment_name: wsm_mm_pd_dep_v1
    run_name: av_f2_task_aware_directed

Data:

    name: wsm_av_fusion_datamodule

Data params exactly:

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

    name: wsm_av_f2_task_aware_directed_model

Model params exactly:

    audio_feature_dim: 768
    video_feature_dim: 512
    hidden_dim: 192
    fusion_hidden_dim: 192
    relation_hidden_dim: 192
    dropout: 0.2
    num_tasks: 2

Loss:

    name: wsm_masked_sparse_loss

Optimizer exactly as F1:

    name: adamw_optimizer
    lr: 0.0001
    weight_decay: 0.01

Training exactly as F1:

    epochs: 30
    device: cuda
    mixed_precision: true
    grad_clip_norm: 0.5
    log_every_steps: 25
    collect_cache: true

Metrics:

    []

Do not add scheduler, auxiliary loss, pseudo-labeling, task weights, flow matching, PAGB, or tuning parameters.

## 2. Required Instrumentation

Use the same instrumentation and durable paths as F1:

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

Use the same filename_template as F1.

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

No Test metric, train loss, or other quantity may appear in any selector/monitor field.

## 3. F1/F2 Config Equivalence Audit

Before training, parse:

    configs/wsm_mm_pd_dep_v1/fusion/01_f1_shared_mtl.yaml
    configs/wsm_mm_pd_dep_v1/fusion/02_f2_task_aware_directed.yaml

Assert all non-model training/data/instrumentation semantics are identical.

Intentional semantic differences only:

- experiment_info.params.run_name;
- model.name;
- F2 adds relation_hidden_dim=192;
- all common model params stay identical.

No difference is allowed in:

- seed;
- data;
- loss;
- optimizer;
- training;
- metrics;
- callbacks;
- loggers;
- selector/early stopping.

## 4. Pre-Run Validation

Run:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml validate-config \
      --config-path configs/wsm_mm_pd_dep_v1/fusion/02_f2_task_aware_directed.yaml

Build through actual Chimera registries and verify:

- DataModule builds;
- F2 model builds;
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
- val_dataloader keys exactly:
  - dev
  - test_none
  - test_soft
  - test_hard
- checkpoint and early stopping monitor exactly dev/mean_score, mode=max.

Run one real production-size batch forward/loss/backward smoke before the full run. The smoke loss is only a structural check; do not interpret it as model quality.

If CUDA or config/build/smoke fails, stop as blocked. Do not alter source/config to work around a failure.

## 5. Real F2 Training

Run from repository root:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/fusion/02_f2_task_aware_directed.yaml

Do not interrupt a healthy run.

If it fails:

- preserve traceback/logs/artifacts;
- do not modify source or fixed config;
- record the exact blocker;
- update PROGRESS_EN.md;
- commit/push evidence;
- stop as blocked.

## 6. Mandatory Metrics Every Epoch

For EVERY completed epoch record:

- train loss;
- DEV depression UAR;
- DEV depression MF1;
- DEV depression Score;
- DEV Parkinson UAR;
- DEV Parkinson MF1;
- DEV Parkinson Score;
- DEV Mean_Score;
- TEST_NONE depression UAR/MF1/Score;
- TEST_NONE Parkinson UAR/MF1/Score;
- TEST_NONE Mean_Score;
- TEST_SOFT depression UAR/MF1/Score;
- TEST_SOFT Parkinson UAR/MF1/Score;
- TEST_SOFT Mean_Score;
- TEST_HARD depression UAR/MF1/Score;
- TEST_HARD Parkinson UAR/MF1/Score;
- TEST_HARD Mean_Score.

All required metrics must be finite.

All four streams must be present on every completed epoch.

Important:

- training loss is logged but is NOT a model-quality metric;
- best epoch is NOT minimum loss;
- best epoch is maximum dev/mean_score only.

Test metrics are monitoring-only.

Do NOT:

- select by Test;
- select by train loss;
- tune relation_hidden_dim/gates using Test;
- change thresholds using Test;
- manually stop based on Test;
- change hyperparameters after looking at Test.

## 7. Best Checkpoint Selection

After completion/early stopping:

1. identify best epoch ONLY by maximum dev/mean_score;
2. record exact checkpoint path;
3. record DEV task UAR/MF1/Score and DEV Mean_Score;
4. record TEST_NONE/SOFT/HARD task metrics from the SAME DEV-selected epoch;
5. record top-2 checkpoints and last.pt;
6. record train.log, summary.txt, code.zip;
7. record exact MLflow experiment name, run ID, final status.

## 8. DEV Expert-Gate Diagnostic

Using the DEV-selected checkpoint, run one post-hoc evaluation over all 933 DEV samples with no parameter updates.

For each task:

    depression
    parkinson

and each expert:

    audio_to_video
    video_to_audio

record task_expert_weights:

- mean;
- std;
- min;
- max.

Also verify:

- finite values;
- each task's expert weights sum to 1 on all DEV rows;
- no Test rows used;
- diagnostic does not affect checkpoint selection.

Optionally report mean task_relation_features L2 norm per task if it is directly available without source changes. Do not add new code solely for that optional statistic.

## 9. DEV Comparison

Compare the DEV-selected F2 result against:

F1:
- DEV Mean_Score=0.773841
- depression Score=0.689708
- Parkinson Score=0.857973

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

    F2 - F1 DEV Mean_Score delta
    F2 - F0 DEV Mean_Score delta
    F2 - audio DEV Mean_Score delta
    F2 - video V2 DEV Mean_Score delta

Also report per-task DEV Score deltas.

The primary Stage 4 result is F2 versus F1 on DEV.

Test comparisons may be reported descriptively but MUST NOT determine the conclusion.

## Acceptance Criteria

Pass if:

- F2 YAML is self-contained and valid;
- F1/F2 config equivalence audit passes;
- CUDA gate passes;
- real production-size forward/loss/backward smoke passes;
- real training completes or configured early stopping ends it;
- DEV + three Test streams report full task metrics every epoch;
- all required metrics finite;
- best checkpoint selected strictly by dev/mean_score;
- train loss is never used for model selection;
- Test metrics never influence selection/tuning;
- exact MLflow run ID/status recorded;
- all-DEV task/expert gate diagnostics recorded;
- F2 vs F1 DEV deltas recorded;
- no source changes;
- no dependency installation;
- no pseudo-labeling/flow matching/PAGB;
- no Stage 5 work;
- text/description remains deferred;
- src/audio unchanged;
- src/video unchanged;
- accepted A+V DataModule unchanged;
- F0/F1/F2 source unchanged;
- git diff --check passes;
- tracked diff contains only:
  - configs/wsm_mm_pd_dep_v1/fusion/02_f2_task_aware_directed.yaml
  - docs/PROGRESS_EN.md
- branch codex/task-004g committed and pushed;
- main/master untouched.

## Required PROGRESS_EN Update

Append TASK-004G evidence without erasing prior records.

Record:

- branch;
- evidence commit SHA;
- push result;
- config path;
- config equivalence result;
- exact training command;
- GPU/device;
- dataset/join counts;
- production smoke loss, explicitly labeled structural only;
- epochs completed;
- early-stopping status;
- FULL epoch table with train loss plus all DEV/TEST task UAR/MF1/Score and Mean_Score;
- best epoch selected by dev/mean_score;
- best DEV metrics;
- same-epoch Test metrics;
- top-2/last checkpoint paths;
- run directory, train.log, summary.txt, code.zip;
- MLflow experiment/run ID/status;
- DEV gate statistics per task/expert;
- F2 versus F1/F0/audio/video-V2 DEV deltas;
- explicit statement that loss was not used for selection;
- explicit Test-selector firewall confirmation;
- confirmation no source changes;
- confirmation RAMPS not started;
- confirmation text/description deferred;
- src/audio unchanged;
- src/video unchanged;
- Stage 4 status;
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

- branch codex/task-004g;
- evidence commit SHA;
- pushed-to-origin status;
- main/master untouched;
- config path;
- epochs completed;
- best epoch;
- best dev/mean_score;
- DEV depression/Parkinson UAR/MF1/Score;
- same-epoch TEST_NONE/SOFT/HARD task metrics and Mean_Scores;
- F2-F1 DEV delta;
- DEV expert-gate statistics;
- exact MLflow run ID/status;
- explicit statement that train loss was NOT used for selection;
- selector firewall confirmation;
- no source changes;
- RAMPS not started;
- text/description deferred;
- src/audio unchanged;
- src/video unchanged.

Stop after this task.
