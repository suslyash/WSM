# TASK-004I: Run the Fixed Strong-Temporal-Audio F1 Residual A+V Experiment

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-004i

This task explicitly authorizes the real strong-temporal-audio F1 residual training run.

Do not modify source code.
Do not modify src/audio or src/video.
Do not modify the accepted A+V DataModule.
Do not modify F0/F1/F2 or the accepted TASK-004H adapter/model.
Do not tune the historical audio checkpoint.
Do not add RAMPS, pseudo-labeling, F2-temporal directed relations, text, or description.

## Goal

Run the controlled comparison:

    frozen historical strong temporal audio
        vs
    the same frozen strong temporal audio + trainable video residual fusion

The audio model must remain the exact DEV-selected historical epoch-4 checkpoint recovered and reproduced in TASK-004H.

The only trainable components are:

    video_projection
    shared_fusion
    residual_heads

The primary question is whether adding video improves DEV metrics over the exact frozen audio base logits on the same canonical DEV rows.

## Critical Evaluation Rule

Training loss is an optimization diagnostic only.

Checkpoint selection and early stopping use ONLY:

    dev/mean_score

Every completed epoch MUST report:

    dev
    test_none
    test_soft
    test_hard

with the same per-task UAR/MF1/Score and Mean_Score definitions.

Test metrics are monitoring-only and MUST NOT influence:

- epoch selection;
- early stopping;
- architecture;
- hyperparameters;
- thresholding;
- any next-step decision.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 2, 5, 6, 8, 9, 10, 13, 14
3. docs/PLAN.md Stage 4
4. docs/PROGRESS_EN.md through MANAGER-DECISION-025
5. docs/NEXT_TASK_EN.md
6. configs/wsm_mm_pd_dep_v1/audio/00_frozen_baseline.yaml
7. configs/wsm_mm_pd_dep_v1/fusion/01_f1_shared_mtl.yaml
8. src/fusion/models/frozen_audio_temporal_adapter.py
9. src/fusion/models/av_f1_temporal_audio_residual.py
10. src/fusion/data/wsm_av_fusion_datamodule.py
11. src/common/callbacks/wsm_segment_callback.py
12. src/common/loss/wsm_masked_sparse_loss.py

## Allowed Tracked Files

- configs/wsm_mm_pd_dep_v1/fusion/03_f1_temporal_audio_residual.yaml
- docs/PROGRESS_EN.md

No other tracked file may be modified.

Generated runtime artifacts under logs/ must remain untracked unless repository policy already tracks them.

## Fixed Historical Audio Teacher/Base

Exact checkpoint:

    logs/wsm_audio_segment_wavlm_base_l9_pool4/multitask_audio_mamba_2026-08-19_14-12_audio_mamba_segment_model_wsm_audio_models-e0ce-006_ea8c8d77/checkpoints/epoch=4_dev_mean_score=0.7878.pt

Required SHA256:

    0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2

Reproduced canonical DEV reference:

    depression UAR/MF1/Score =
        0.7480392157 / 0.7477975633 / 0.7479183895

    Parkinson UAR/MF1/Score =
        0.8209799862 / 0.8344907407 / 0.8277353635

    DEV Mean_Score =
        0.7878268765

This checkpoint is fixed and non-tunable.

Before training:

- recompute SHA256 and require exact match;
- instantiate the accepted adapter/model;
- verify all audio parameters require_grad=false;
- verify parent model.train() leaves audio_model.training=false;
- verify no audio parameter is present in optimizer param groups.

If any check fails:

    STOP BLOCKED.

Do not repair source code in this task.

## 1. Create the Fixed Training Config

Create:

    configs/wsm_mm_pd_dep_v1/fusion/03_f1_temporal_audio_residual.yaml

Use exactly:

    seed: 42

Experiment:

    experiment_name: wsm_mm_pd_dep_v1
    run_name: av_f1_temporal_audio_residual

Data:

    name: wsm_av_fusion_datamodule

Data params:

    data_root: /media/maxim/Databases/WSM_NEW
    audio_feature_cache_root: /media/maxim/Databases/WSM_NEW/features
    video_cache_root: /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache
    batch_size: 8
    num_workers: 4
    pin_memory: true
    persistent_workers: true
    shuffle_train: true
    drop_last_train: false

Batch size 8 is fixed for this strong-temporal-audio ablation. Do not tune it.

Model:

    name: wsm_av_f1_temporal_audio_residual_model

Model params exactly:

    audio_checkpoint_path: logs/wsm_audio_segment_wavlm_base_l9_pool4/multitask_audio_mamba_2026-08-19_14-12_audio_mamba_segment_model_wsm_audio_models-e0ce-006_ea8c8d77/checkpoints/epoch=4_dev_mean_score=0.7878.pt
    audio_feature_dim: 768
    audio_hidden_dim: 192
    video_feature_dim: 512
    hidden_dim: 192
    fusion_hidden_dim: 192
    dropout: 0.2
    num_tasks: 2

Loss:

    name: wsm_masked_sparse_loss

Do NOT use the historical wsm_audio_loss here.

The historical audio model is already trained and frozen. The new optimization objective applies only to the trainable residual fusion branch under the final independent-logit sparse contract.

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

Do not add:

- scheduler;
- auxiliary loss;
- audio loss;
- pseudo-label loss;
- contrastive loss;
- task weights;
- gradient balancing;
- modality dropout;
- extra regularization;
- threshold tuning.

## 2. Required Instrumentation

Use:

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

Filename template:

    epoch={epoch}_{monitor}={value:.4f}.pt

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

No loss or Test metric may appear in any selector/monitor field.

## 3. Pre-Run Config / Registry / Data Audit

Run:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml validate-config \
      --config-path configs/wsm_mm_pd_dep_v1/fusion/03_f1_temporal_audio_residual.yaml

Verify through actual Chimera registries:

- DataModule builds;
- model builds;
- masked sparse loss builds;
- optimizer builds;
- callbacks/loggers build;
- no project-module warning;
- CUDA available;
- GPU identified;
- counts exactly:
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
- checkpoint/early stopping monitor exactly dev/mean_score max.

## 4. Frozen-Audio and Optimizer Firewall

Before training instantiate the configured model and optimizer.

Assert:

- every parameter under audio_adapter.audio_model has requires_grad=false;
- calling model.train() still leaves:
      model.audio_adapter.audio_model.training == false
- no frozen audio parameter object ID appears in any optimizer param group;
- all trainable model parameters belong only to:
      video_projection
      shared_fusion
      residual_heads
- no task_id/task_ids exists in an incoming real A+V batch.

Record counts:

- frozen audio parameter count;
- trainable video_projection parameter count;
- trainable shared_fusion parameter count;
- trainable residual_heads parameter count;
- total trainable parameter count.

## 5. Pre-Run DEV Base Reproduction

Before training, using the newly constructed configured model:

- iterate ONLY canonical DEV rows;
- collect aux["audio_base_logits"];
- compute sparse two-task DEV metrics.

Require reproduction within 0.0005 of TASK-004H:

    depression Score = 0.7479183895
    Parkinson Score = 0.8277353635
    Mean_Score = 0.7878268765

This verifies the production config points to the exact frozen audio base.

Do not iterate Test for this separate reproduction check.

If reproduction fails:

    STOP BLOCKED.

## 6. Real Production-Batch Smoke

Use one real TRAIN batch of size 8.

Run:

    forward
    wsm_masked_sparse_loss
    backward

Verify:

- output [8,2] unless final short batch semantics are intentionally used for the smoke;
- finite structural loss;
- audio gradients all None;
- finite nonzero gradients in:
  - video_projection
  - shared_fusion
  - depression residual head
  - Parkinson residual head;
- frozen audio stays eval-only.

The smoke loss is structural only and MUST NOT be interpreted as model quality.

Use a fresh model instance for the real training run after smoke if needed so the smoke backward does not contaminate training state.

## 7. Real F1-Temporal Residual Training

Run from repository root:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/fusion/03_f1_temporal_audio_residual.yaml

Do not interrupt a healthy run.

If training fails:

- preserve traceback/logs/artifacts;
- do not modify source/config;
- update PROGRESS_EN.md with the exact blocker;
- commit/push evidence;
- stop blocked.

## 8. Mandatory Metrics Every Epoch

For EVERY completed epoch record at minimum:

- train loss;

DEV:
- depression UAR/MF1/Score;
- Parkinson UAR/MF1/Score;
- Mean_Score;

TEST_NONE:
- depression UAR/MF1/Score;
- Parkinson UAR/MF1/Score;
- Mean_Score;

TEST_SOFT:
- depression UAR/MF1/Score;
- Parkinson UAR/MF1/Score;
- Mean_Score;

TEST_HARD:
- depression UAR/MF1/Score;
- Parkinson UAR/MF1/Score;
- Mean_Score.

All must be finite.

All four evaluation streams must appear on every completed epoch.

Best epoch is:

    argmax dev/mean_score

NOT minimum training loss.

Test remains monitoring-only.

## 9. Best Checkpoint Selection and Artifacts

After completion/early stopping:

1. select best epoch ONLY by dev/mean_score;
2. record exact selected checkpoint path;
3. record DEV task UAR/MF1/Score and DEV Mean_Score;
4. record same-epoch Test task metrics and Mean_Scores;
5. record top-2 checkpoints and last.pt;
6. record run directory, train.log, summary.txt, code.zip, resolved config;
7. record exact MLflow experiment, run ID, final status, artifact URI.

Do not omit MLflow run ID/status.

## 10. Same-DEV-Row Frozen-Base vs Final-Fusion Audit

Using the DEV-selected checkpoint, run one post-hoc pass over all 933 canonical DEV rows.

Collect simultaneously from the SAME forward passes:

    final_logits = output.preds
    base_logits  = output.aux["audio_base_logits"]
    residual_logits = output.aux["residual_logits"]

Use the same targets/observed masks.

Compute sparse metrics separately for:

    frozen audio base logits
    final strong-audio+video logits

The frozen base MUST still reproduce approximately:

    Mean_Score 0.7878268765

Report per task:

- base UAR/MF1/Score;
- final UAR/MF1/Score;
- final minus base Score delta.

Report aggregate:

    final DEV Mean_Score - base DEV Mean_Score

This SAME-row delta is the primary result of TASK-004I.

## 11. DEV Residual Effect Diagnostic

On the same selected-checkpoint DEV pass, for each task record:

Residual distribution over all DEV rows:

- mean;
- std;
- mean absolute value;
- min;
- max.

On rows with an observed label for that task, use threshold logit >= 0 for both base and final.

Record:

- observed sample count;
- base correct count;
- final correct count;
- base wrong -> final correct count ("corrected errors");
- base correct -> final wrong count ("introduced errors");
- prediction sign-flip count;
- prediction sign-flip fraction.

These diagnostics are descriptive only.

Do not use them to select the checkpoint.

Do not tune thresholds.

## 12. DEV Comparisons to Existing Fixed Baselines

Compare selected final DEV result to:

Historical/frozen audio base:
- Mean_Score=0.7878268765
- depression Score=0.7479183895
- Parkinson Score=0.8277353635

Pooled F1:
- Mean_Score=0.773841
- depression Score=0.689708
- Parkinson Score=0.857973

Pooled F2:
- Mean_Score=0.774569
- depression Score=0.697035
- Parkinson Score=0.852104

Report:

    temporal-F1 final - frozen audio base
    temporal-F1 final - pooled F1
    temporal-F1 final - pooled F2

for DEV Mean_Score and task Scores.

The primary comparison remains:

    temporal-F1 final vs same-row frozen audio base

Do not use Test values for this conclusion.

## Acceptance Criteria

The task passes if:

- config is self-contained and valid;
- exact historical checkpoint SHA matches;
- pre-run frozen base DEV reproduction passes;
- only residual fusion components are trainable;
- frozen audio absent from optimizer;
- frozen audio remains eval-only during parent training;
- real batch smoke passes;
- real training completes or configured early stopping ends it;
- all four streams report complete metrics every epoch;
- best checkpoint selected only by dev/mean_score;
- train loss never selects anything;
- Test never selects/tunes anything;
- selected-checkpoint same-row base-vs-final DEV audit is recorded;
- residual corrected-error/introduced-error diagnostics are recorded;
- exact MLflow run provenance is recorded;
- no source changes;
- no audio/video/DataModule changes;
- no RAMPS/F2-temporal/text work;
- git diff --check passes;
- tracked diff contains only:
  - configs/wsm_mm_pd_dep_v1/fusion/03_f1_temporal_audio_residual.yaml
  - docs/PROGRESS_EN.md
- branch codex/task-004i committed and pushed;
- main/master untouched.

## Required PROGRESS_EN Update

Append TASK-004I evidence without erasing prior records.

Record:

- branch;
- evidence commit SHA;
- push result;
- config path;
- exact historical checkpoint path and SHA verification;
- GPU/device;
- dataset/join counts;
- trainable/frozen parameter counts;
- optimizer audio-exclusion proof;
- pre-run DEV base reproduction metrics;
- production smoke loss explicitly labeled structural only;
- epochs completed;
- early-stopping status;
- FULL epoch table with train loss and all DEV/TEST task UAR/MF1/Score/Mean_Score;
- best epoch selected by dev/mean_score;
- best DEV final metrics;
- same-epoch Test metrics;
- top-2/last checkpoint paths;
- run/log/summary/code/config artifact paths;
- MLflow experiment/run ID/status/artifact URI;
- selected-checkpoint same-row frozen-base metrics;
- selected final minus frozen-base DEV deltas;
- residual distribution stats;
- corrected errors, introduced errors, and sign flips per task;
- comparisons versus pooled F1/F2;
- explicit statement loss did not select the model;
- explicit Test-selector firewall confirmation;
- confirmation no source changes;
- confirmation RAMPS remains deferred;
- confirmation F2-temporal not started;
- confirmation text/description deferred;
- src/audio unchanged;
- src/video unchanged;
- Stage 4 reopened-ablation status;
- recommended next atomic step only.

If the run completes without a concrete blocker:

    recommended next = TASK-004J implement/register the analogous strong-temporal-audio F2 directed residual model contract

Do not start TASK-004J inside this task.

## Required Handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch codex/task-004i;
- evidence commit SHA;
- pushed-to-origin status;
- main/master untouched;
- config path;
- exact historical audio checkpoint SHA verification;
- epochs completed;
- best epoch;
- final DEV depression/Parkinson UAR/MF1/Score and Mean_Score;
- SAME-row frozen-base DEV depression/Parkinson Scores and Mean_Score;
- final-minus-base DEV deltas;
- corrected/introduced error counts per task;
- same-epoch TEST_NONE/SOFT/HARD Mean_Scores;
- exact MLflow run ID/status;
- explicit train-loss-not-selector statement;
- Test firewall confirmation;
- no source changes;
- F2-temporal not started;
- RAMPS deferred;
- text/description deferred;
- src/audio unchanged;
- src/video unchanged.

Stop after this task.
