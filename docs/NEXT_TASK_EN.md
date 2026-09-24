# TASK-002O: Run the Fixed Prototype-Aware V2 Video Experiment

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-002o

This task explicitly authorizes the fixed V2 training experiment.

Do not modify V2 source code, preprocessing, cache, DataModule behavior, loss, metrics, callbacks, optimizer family, batch size, seed, epoch budget, or Test protocol definitions.

## Goal

Create a self-contained V2 training config by mirroring the accepted V1 config and changing only the planned model-family fields, then run the full seed-42 V2 experiment.

The controlled comparison is:

    V1: wsm_video_depart_v1_model
    V2: wsm_video_depart_v2_model

Everything else must remain matched unless the V2 model itself requires its fixed prototype parameters.

Use the accepted full-coverage DEPART-compatible cache:

    /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache

Evaluate every epoch on:

    dev
    test_none
    test_soft
    test_hard

Select the V2 checkpoint ONLY by:

    dev/mean_score

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 5, 7, 8, 9, 10, 13, 14
3. docs/PLAN.md Stage 2
4. docs/PROGRESS_EN.md through MANAGER-DECISION-015
5. docs/NEXT_TASK_EN.md
6. configs/wsm_mm_pd_dep_v1/video/00_depart_v1.yaml
7. src/video/models/depart_v2.py

## Allowed Tracked Files

- configs/wsm_mm_pd_dep_v1/video/01_depart_v2_prototype.yaml
- docs/PROGRESS_EN.md

No other tracked file may be modified.

Generated runtime artifacts under logs/ are expected and must remain untracked unless the repository already tracks a specific runtime artifact type.

Do not modify src/audio.

## 1. Create the V2 Config

Create:

    configs/wsm_mm_pd_dep_v1/video/01_depart_v2_prototype.yaml

It must be a self-contained copy of the accepted V1 production config with ONLY these intentional changes:

Run name:

    depart_v2_prototype_gated

Model registry key:

    wsm_video_depart_v2_model

Model params:

    video_feature_dim: 512
    hidden_dim: 192
    num_layers: 2
    num_heads: 4
    ff_mult: 4
    dropout: 0.2
    sequence_steps: 60
    num_tasks: 2
    prototype_scale: 10.0
    gate_hidden_dim: 64

Every non-model setting must match V1 exactly:

- seed=42;
- experiment_name=wsm_mm_pd_dep_v1;
- same DataModule;
- same data_root/cache_root;
- batch_size=32;
- num_workers=4;
- pin_memory=true;
- persistent_workers=true;
- shuffle_train=true;
- drop_last_train=false;
- same masked sparse loss;
- AdamW lr=1e-4;
- weight_decay=0.01;
- epochs=30;
- CUDA;
- mixed precision;
- grad_clip_norm=0.5;
- log_every_steps=25;
- collect_cache=true;
- metrics=[];
- same required callbacks and loggers;
- checkpoint monitor=dev/mean_score mode=max;
- early stopping monitor=dev/mean_score mode=max patience=6 min_delta=0.0005;
- durable logs paths;
- four evaluation streams remain dev/test_none/test_soft/test_hard.

Do not add contrastive loss.
Do not add a scheduler.
Do not add prototype-specific auxiliary loss.
Do not tune prototype_scale or gate_hidden_dim.

## 2. Config Equivalence Audit

Before training, programmatically compare the parsed V1 and V2 YAMLs.

Assert that the only semantic differences are:

- experiment_info.params.run_name;
- model.name;
- model.params.prototype_scale;
- model.params.gate_hidden_dim.

The common V1 model params must be identical.

No optimizer/data/train/callback/logging difference is permitted.

## 3. Pre-Run Validation

Run:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml validate-config \
      --config-path configs/wsm_mm_pd_dep_v1/video/01_depart_v2_prototype.yaml

Verify:

- CUDA available;
- DataModule counts:
  - train=6325;
  - dev=933;
  - test_none=1364;
  - test_soft=1208;
  - test_hard=1014;
- val_dataloader keys exactly:
  - dev
  - test_none
  - test_soft
  - test_hard
- V2 registry builds successfully;
- masked sparse loss builds;
- callbacks/loggers build;
- selector firewall remains DEV-only.

If CUDA is unavailable, stop as blocked. Do not switch the real run to CPU.

## 4. Real V2 Training

Run:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/video/01_depart_v2_prototype.yaml

Do not interrupt a healthy run.

If the run fails:

- preserve traceback/logs/artifacts;
- do not modify source or config in this task;
- record the exact blocker in PROGRESS_EN.md;
- commit/push evidence;
- stop as blocked.

## 5. Epoch-Level Monitoring

For every completed epoch, record at minimum:

- train loss;
- DEV depression UAR/MF1/Score;
- DEV Parkinson UAR/MF1/Score;
- DEV Mean_Score;
- TEST_NONE Mean_Score;
- TEST_SOFT Mean_Score;
- TEST_HARD Mean_Score.

All required metrics must be finite.

All four evaluation streams must appear on every completed epoch.

Test metrics remain monitoring-only.

Do NOT:

- choose an epoch by Test;
- tune prototype parameters using Test;
- manually stop due to Test;
- change any hyperparameter based on Test.

## 6. Best V2 Checkpoint

After training/early stopping:

1. select the best V2 epoch ONLY by maximum dev/mean_score;
2. record best epoch and exact checkpoint path;
3. record DEV depression/Parkinson UAR/MF1/Score and DEV Mean_Score;
4. record TEST_NONE/SOFT/HARD metrics from that SAME DEV-selected epoch;
5. record top-2 checkpoints, last checkpoint, logs, summary, code archive, MLflow run ID.

## 7. Controlled V1 vs V2 Comparison

Use only DEV to compare model families for the Stage 2 decision.

Frozen V1 reference:

- best epoch=8;
- DEV Mean_Score=0.703274;
- DEV depression Score=0.661923;
- DEV Parkinson Score=0.744626;
- same-epoch TEST_NONE=0.732454;
- same-epoch TEST_SOFT=0.742796;
- same-epoch TEST_HARD=0.751864.

Report:

    delta_DEV_Mean_Score = V2_best_DEV_Mean_Score - 0.703274

Also report task-level DEV deltas.

The Test deltas may be listed descriptively but MUST NOT determine the V1/V2 winner.

The V1/V2 Stage 2 family decision is based only on DEV Mean_Score at each family's DEV-selected checkpoint.

Do not start multi-seed confirmation in this task.

## 8. Optional V2 Mechanism Diagnostics

Without modifying source/config, if straightforward from the DEV-selected V2 checkpoint, report descriptive DEV-only diagnostics:

- mean prototype gate for depression;
- mean prototype gate for Parkinson;
- prototype cosine separation:
  - cos(p_D,0, p_D,1)
  - cos(p_P,0, p_P,1).

These diagnostics are optional and must not alter selection.

Do not inspect labels beyond the already-authorized evaluation metrics to tune gates/prototypes.

## Acceptance Criteria

The task passes if:

- V2 YAML is self-contained and valid;
- V1/V2 config equivalence audit passes;
- only intended model-family config differences exist;
- CUDA full run launches;
- run completes or ends via configured early stopping;
- all four streams run every epoch;
- all required metrics finite;
- best V2 checkpoint selected strictly by dev/mean_score;
- Test metrics never influence selection/tuning;
- V1 vs V2 DEV comparison is recorded;
- no source code changes;
- no contrastive loss;
- no dependency install;
- src/audio unchanged;
- git diff --check passes;
- tracked diff contains only:
  - configs/wsm_mm_pd_dep_v1/video/01_depart_v2_prototype.yaml
  - docs/PROGRESS_EN.md
- branch codex/task-002o committed and pushed;
- main/master untouched.

## Required PROGRESS_EN Update

Record:

- branch;
- evidence commit SHA;
- push result;
- exact V2 config path;
- config equivalence audit result;
- training command;
- GPU/device;
- epochs completed;
- early-stopping status;
- best epoch;
- best dev/mean_score;
- best DEV depression/Parkinson UAR/MF1/Score;
- same-epoch Test protocol metrics;
- full epoch table;
- checkpoint/log/summary/MLflow paths;
- optional gate/prototype diagnostics if obtained;
- V1 vs V2 DEV deltas;
- which family is ahead by DEV Mean_Score;
- explicit statement that Test did not determine family selection;
- confirmation no source changes and src/audio unchanged;
- Stage 2 status;
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

- branch codex/task-002o;
- evidence commit SHA;
- pushed-to-origin status;
- main/master untouched;
- V2 config path;
- epochs completed;
- best V2 epoch;
- best V2 dev/mean_score;
- DEV task Scores;
- same-epoch Test mean scores;
- V1 vs V2 DEV delta;
- DEV-based family result;
- selector firewall confirmation;
- no contrastive loss;
- src/audio unchanged.

Stop after this task.
