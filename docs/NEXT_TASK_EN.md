# TASK-002M: Run the First Real V1 Video Training Experiment

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-002m

This task explicitly authorizes the first real/full V1 video training experiment.

Do not change model architecture, preprocessing, DataModule semantics, loss, optimizer, batch size, epoch count, seed, selector, callbacks, or Test protocol definitions in this task.

## Goal

Run the accepted production V1 video training config exactly as frozen after TASK-002L2:

    configs/wsm_mm_pd_dep_v1/video/00_depart_v1.yaml

Use the accepted full-coverage DEPART-compatible cache:

    /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache

Train the real V1 video model on the full TRAIN dataset and evaluate all four named streams every epoch:

    dev
    test_none
    test_soft
    test_hard

The sole model-selection criterion is:

    dev/mean_score

Test metrics are mandatory epoch-level monitoring outputs only.

## Fixed Experiment Contract

Config:

    configs/wsm_mm_pd_dep_v1/video/00_depart_v1.yaml

Seed:

    42

Experiment name:

    wsm_mm_pd_dep_v1

Run name:

    depart_v1_clip_yolo_transformer

Data:

- train = 6325
- dev = 933
- test_none = 1364
- test_soft = 1208
- test_hard = 1014
- unavailable counts = 0 for every stream

Model:

- video_feature_dim=512
- hidden_dim=192
- num_layers=2
- num_heads=4
- ff_mult=4
- dropout=0.2
- sequence_steps=60
- num_tasks=2
- outputs [B,2] independent logits ordered [depression, parkinson]

Loss:

    wsm_masked_sparse_loss

Optimizer:

    AdamW
    lr=1e-4
    weight_decay=0.01

Training:

- epochs=30
- device=cuda
- mixed_precision=true
- grad_clip_norm=0.5
- log_every_steps=25
- collect_cache=true

Automatic selector:

    dev/mean_score
    mode=max

Early stopping:

- monitor=dev/mean_score
- mode=max
- patience=6
- min_delta=0.0005

Checkpoint:

- monitor=dev/mean_score
- mode=max
- save_top_k=2
- save_last=true

Do not change any of the above.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 5, 8, 9, 10, 13, 14
3. docs/PLAN.md Stage 2
4. docs/PROGRESS_EN.md through MANAGER-DECISION-013
5. docs/NEXT_TASK_EN.md
6. configs/wsm_mm_pd_dep_v1/video/00_depart_v1.yaml

## Allowed Tracked Files

- docs/PROGRESS_EN.md

No tracked source or config file may be modified in this task.

Generated runtime artifacts under logs/ are expected and must remain untracked unless the repository already tracks a specific runtime artifact type.

Do not modify src/audio.

## Pre-Run Gate

Before launching the real run, verify:

    git status --short

The task branch must have no unexpected user changes.

Run:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml validate-config \
      --config-path configs/wsm_mm_pd_dep_v1/video/00_depart_v1.yaml

Confirm:

- config valid;
- CUDA available;
- full DataModule counts are exactly accepted;
- val_dataloader() keys exactly:
  dev, test_none, test_soft, test_hard;
- checkpoint and early stopping monitor only dev/mean_score;
- no Test selector reference;
- durable logs/ paths exist or can be created.

If CUDA is unavailable, TASK-002M is blocked. Do not silently switch this full experiment to CPU.

## Real Training Command

Run from repository root:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/video/00_depart_v1.yaml

Do not interrupt a healthy run.

If the run fails:

- preserve logs and traceback;
- do not change code/config in this task;
- classify the exact runtime blocker;
- update PROGRESS_EN.md;
- commit/push the evidence;
- stop as blocked.

## Epoch-Level Monitoring Requirements

For EVERY completed epoch, collect/report at least:

DEV:

- dev/depression/uar
- dev/depression/mf1
- dev/depression/score
- dev/parkinson/uar
- dev/parkinson/mf1
- dev/parkinson/score
- dev/mean_score

TEST_NONE:

- test_none/depression/uar
- test_none/depression/mf1
- test_none/depression/score
- test_none/parkinson/uar
- test_none/parkinson/mf1
- test_none/parkinson/score
- test_none/mean_score

TEST_SOFT:

- test_soft/depression/uar
- test_soft/depression/mf1
- test_soft/depression/score
- test_soft/parkinson/uar
- test_soft/parkinson/mf1
- test_soft/parkinson/score
- test_soft/mean_score

TEST_HARD:

- test_hard/depression/uar
- test_hard/depression/mf1
- test_hard/depression/score
- test_hard/parkinson/uar
- test_hard/parkinson/mf1
- test_hard/parkinson/score
- test_hard/mean_score

Also record:

- train loss;
- validation/evaluation loss if emitted;
- learning rate if emitted;
- epoch duration if available.

Test metrics are monitoring only.

Do NOT:

- pick an epoch because of Test;
- change threshold based on Test;
- stop manually because Test worsens;
- alter any hyperparameter based on Test.

## Best Checkpoint Selection

After training/early stopping completes:

1. determine the best epoch ONLY by maximum dev/mean_score;
2. record the exact best checkpoint path;
3. record the best epoch number;
4. record the best dev task metrics and dev/mean_score;
5. record the test_none/test_soft/test_hard metrics from THAT SAME DEV-selected epoch for comparative monitoring only;
6. do not compare Test epochs to choose a different checkpoint.

Also record:

- final/last epoch;
- whether early stopping triggered;
- total epochs completed;
- top-2 checkpoint paths;
- last checkpoint path;
- snapshot location;
- console log path;
- MLflow experiment/run identifier if available.

## Historical Comparison

For context only, compare the DEV-selected V1 checkpoint to the frozen historical audio reference already recorded in PROGRESS_EN.md:

Audio historical:

- DEV Mean_Score = 0.787827
- depression Score = 0.747918
- Parkinson Score = 0.827735
- TEST_NONE Mean_Score = 0.809486
- TEST_SOFT Mean_Score = 0.815156
- TEST_HARD Mean_Score = 0.828135

This comparison is descriptive only.

Do not select or modify the V1 model based on those Test numbers.

## Required Run Audit

After the run, verify:

- all four evaluation prefixes were present on every completed epoch;
- selector callback used only dev/mean_score;
- checkpoint filenames/metadata correspond to DEV selection;
- no NaN/Inf loss;
- no NaN/Inf required metric;
- no missing task metric;
- Test dataset counts stayed 1364/1208/1014;
- full cache root remained the accepted fullframe-fallback cache;
- no source/config file changed during the run;
- src/audio unchanged.

If any required metric stream disappears on any completed epoch, treat the run as invalid/blocked and report it.

## Acceptance Criteria

The task passes if:

- production config validates;
- CUDA full training launches successfully;
- real training completes or ends via configured early stopping;
- no runtime exception;
- all four streams are evaluated every completed epoch;
- all required metrics are finite;
- best checkpoint is selected strictly by dev/mean_score;
- top-2 + last checkpoints are preserved;
- durable logs/snapshot/MLflow artifacts are preserved;
- Test metrics are reported but never used as selector inputs;
- no architecture/config/hyperparameter change occurred;
- no dependency install;
- src/audio unchanged;
- tracked diff contains only docs/PROGRESS_EN.md;
- branch codex/task-002m committed and pushed;
- main/master untouched.

## Required PROGRESS_EN Update

Append TASK-002M evidence without erasing prior records.

Record:

- branch;
- implementation/evidence commit SHA;
- push result;
- exact training command;
- config path and git starting SHA;
- GPU/device details if available;
- epochs completed;
- whether early stopping triggered;
- best epoch selected by dev/mean_score;
- best checkpoint path;
- top-2 and last checkpoint paths;
- snapshot/log/MLflow locations;
- full epoch-by-epoch table containing at minimum:
  - epoch;
  - dev depression UAR/MF1/Score;
  - dev Parkinson UAR/MF1/Score;
  - dev Mean_Score;
  - test_none Mean_Score;
  - test_soft Mean_Score;
  - test_hard Mean_Score;
  - train loss;
- exact Test task metrics at the DEV-selected best epoch;
- historical audio comparison;
- confirmation every completed epoch contained all four metric streams;
- confirmation Test never drove selection;
- confirmation no source/config changes;
- src/audio unchanged;
- Stage 2 status and recommended next atomic step.

Do not call smoke metrics research results.

## Required Handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch codex/task-002m;
- evidence commit SHA;
- pushed-to-origin status;
- main/master untouched;
- epochs completed;
- best epoch;
- best dev/mean_score;
- best DEV depression/Parkinson Scores;
- same-epoch test_none/test_soft/test_hard Mean_Scores;
- checkpoint/log/MLflow paths;
- early-stopping status;
- comparison with frozen historical audio reference;
- Test selector firewall confirmation;
- src/audio unchanged.

Stop after this task.
