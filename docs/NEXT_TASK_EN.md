# TASK-004I3: Run the Fixed Strong-Temporal-Audio F1 Residual A+V Training Experiment

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-004i3

This task explicitly authorizes the real fixed strong-temporal-audio F1 residual training run.

Do not modify source code.
Do not modify any config.
Do not modify Chimera ML.
Do not modify src/audio or src/video.
Do not modify the accepted A+V DataModule.
Do not modify the frozen temporal-audio adapter/model.
Do not start F2-temporal, RAMPS, text, or description work.

## Goal

Run the already accepted production experiment:

    frozen historical strong temporal audio
        +
    trainable video-conditioned residual fusion

using exactly:

    configs/wsm_mm_pd_dep_v1/fusion/03_f1_temporal_audio_residual.yaml

The config, model, optimizer, historical checkpoint, batch size, seed, selector, and all hyperparameters are frozen by prior manager decisions.

The primary research question is:

    Does adding video improve DEV performance over the exact same frozen strong-audio base logits on the same canonical DEV rows?

## Critical Evaluation Rule

Training loss is an optimization diagnostic only.

The sole automatic selector is:

    dev/mean_score

with:

    mode = max

Every completed epoch MUST report all four streams:

    dev
    test_none
    test_soft
    test_hard

with:

    depression UAR/MF1/Score
    Parkinson UAR/MF1/Score
    Mean_Score

TEST_NONE/SOFT/HARD are mandatory monitoring-only outputs and MUST NOT influence:

- checkpoint selection;
- early stopping;
- thresholding;
- hyperparameter changes;
- architecture changes;
- modality selection;
- any next-step decision.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 5, 6, 8, 9, 13, 14
3. docs/PLAN.md Stage 4
4. docs/PROGRESS_EN.md through MANAGER-DECISION-028
5. docs/NEXT_TASK_EN.md
6. configs/wsm_mm_pd_dep_v1/fusion/03_f1_temporal_audio_residual.yaml
7. src/common/optimizers.py
8. src/fusion/models/frozen_audio_temporal_adapter.py
9. src/fusion/models/av_f1_temporal_audio_residual.py
10. src/fusion/data/wsm_av_fusion_datamodule.py
11. src/common/callbacks/wsm_segment_callback.py
12. src/common/loss/wsm_masked_sparse_loss.py

## Allowed Tracked Files

- docs/PROGRESS_EN.md

No other tracked file may be modified.

Generated runtime artifacts under logs/ are expected and must remain untracked unless repository policy already tracks them.

If any source/config file is dirty at task start, stop and report the exact conflict. Do not overwrite user changes.

## Fixed Production Contract

Config:

    configs/wsm_mm_pd_dep_v1/fusion/03_f1_temporal_audio_residual.yaml

Seed:

    42

Batch size:

    8

Epoch ceiling:

    30

Model:

    wsm_av_f1_temporal_audio_residual_model

Optimizer:

    wsm_trainable_adamw_optimizer

Optimizer semantics:

    AdamW over exactly requires_grad=true parameters

Optimizer hyperparameters:

    lr = 0.0001
    weight_decay = 0.01

Loss:

    wsm_masked_sparse_loss

Selector:

    dev/mean_score
    mode = max

Early stopping:

    patience = 6
    min_delta = 0.0005

Historical audio checkpoint:

    logs/wsm_audio_segment_wavlm_base_l9_pool4/multitask_audio_mamba_2026-08-19_14-12_audio_mamba_segment_model_wsm_audio_models-e0ce-006_ea8c8d77/checkpoints/epoch=4_dev_mean_score=0.7878.pt

Required checkpoint SHA256:

    0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2

Accepted frozen-base canonical DEV reference:

    depression UAR/MF1/Score =
      0.7480392157 / 0.7477975633 / 0.7479183895

    Parkinson UAR/MF1/Score =
      0.8209799862 / 0.8344907407 / 0.8277353635

    Mean_Score =
      0.7878268765

Correct production parameter counts:

    frozen audio = 105 objects / 3,031,880 scalars
    trainable = 22 objects / 286,530 scalars
    video_projection = 99,520
    shared_fusion = 111,744
    residual_heads = 75,266
    optimizer objects = 22

## 1. Start-of-Task Scope Gate

Before any training:

    git fetch origin
    git status --short
    git rev-parse --abbrev-ref HEAD

Require:

- branch is codex/task-004i3;
- working tree starts clean;
- branch started from current origin/main;
- no source/config file is modified.

Do not reuse or reset an existing task branch.

## 2. Final Pre-Run Production Gate

Before invoking chimera-ml train, rerun the bounded production firewall.

Verify:

- config validates;
- checkpoint SHA exact;
- CUDA available;
- GPU identified;
- dataset counts:
  - train=6325
  - dev=933
  - test_none=1364
  - test_soft=1208
  - test_hard=1014
- joined_total=8622;
- missing audio/video=0/0;
- validation keys exactly dev/test_none/test_soft/test_hard;
- frozen audio object/scalar counts exact;
- trainable object/scalar counts exact;
- optimizer_ids == trainable_ids;
- optimizer/frozen intersection empty;
- effective optimizer lr/weight_decay exact;
- parent model train mode leaves frozen audio model in eval mode;
- no incoming task_id/task_ids.

Do not iterate Test streams during this pre-run gate.

If any accepted firewall condition fails:

    STOP BLOCKED.

Do not edit source/config to repair it in this task.

## 3. Real Training Command

Run from repository root exactly:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/fusion/03_f1_temporal_audio_residual.yaml

Do not interrupt a healthy run.

Do not alter the config after seeing any metric.

If the run fails:

- preserve traceback/logs/artifacts;
- do not modify source/config;
- append exact blocker evidence to PROGRESS_EN.md;
- commit/push that evidence;
- stop blocked.

## 4. Mandatory Metrics Every Completed Epoch

For EVERY completed epoch record:

Training:

    train loss

DEV:

    depression UAR
    depression MF1
    depression Score
    Parkinson UAR
    Parkinson MF1
    Parkinson Score
    Mean_Score

TEST_NONE:

    depression UAR/MF1/Score
    Parkinson UAR/MF1/Score
    Mean_Score

TEST_SOFT:

    depression UAR/MF1/Score
    Parkinson UAR/MF1/Score
    Mean_Score

TEST_HARD:

    depression UAR/MF1/Score
    Parkinson UAR/MF1/Score
    Mean_Score

All required metrics must be finite.

All four streams must appear for every completed epoch.

Training loss MUST NOT select the best epoch.

Best epoch MUST be:

    argmax(dev/mean_score)

Test metrics MUST remain monitoring-only.

## 5. Best Checkpoint Selection

After training ends or configured early stopping fires:

1. identify the best epoch only by maximum dev/mean_score;
2. record the exact selected checkpoint path;
3. record DEV task UAR/MF1/Score and DEV Mean_Score;
4. record TEST_NONE/SOFT/HARD task metrics from that SAME DEV-selected epoch;
5. record top-2 checkpoint paths;
6. record last.pt;
7. record run directory;
8. record train.log;
9. record summary.txt;
10. record code.zip;
11. record resolved config artifact if produced;
12. record exact MLflow experiment name, run ID, final status, and artifact URI.

Do not omit MLflow run ID/status.

## 6. Same-DEV-Row Frozen-Base vs Final-Fusion Audit

Using ONLY the DEV-selected checkpoint, run one post-hoc pass over all 933 canonical DEV rows.

From the SAME forward passes collect:

    final_logits = output.preds
    base_logits = output.aux["audio_base_logits"]
    residual_logits = output.aux["residual_logits"]
    targets
    observed_mask

Compute sparse two-task metrics separately for:

    frozen audio base logits
    final strong-audio+video logits

Use the existing:

    compute_sparse_two_task_metrics

Do not tune thresholds.

The frozen base must still reproduce approximately:

    depression Score = 0.7479183895
    Parkinson Score = 0.8277353635
    Mean_Score = 0.7878268765

Acceptance for base reproduction:

    abs(delta) <= 0.0005

Report final-minus-base DEV deltas for:

    depression Score
    Parkinson Score
    Mean_Score

This same-row final-minus-base DEV delta is the PRIMARY TASK-004I3 result.

If final is worse than base, report the negative result plainly. Do not change the experiment.

## 7. DEV Residual Effect Diagnostics

Using the same selected-checkpoint DEV pass, for each task record residual logits:

    mean
    std
    mean absolute value
    min
    max

For rows where that task label is observed, use the fixed decision threshold:

    logit >= 0

for both base and final.

Record per task:

- observed row count;
- base correct count;
- final correct count;
- base wrong -> final correct count:
      corrected_errors
- base correct -> final wrong count:
      introduced_errors
- prediction sign-flip count;
- sign-flip fraction.

Also record:

    corrected_errors - introduced_errors

as a descriptive error-balance count only.

Do not use these diagnostics for selection or tuning.

## 8. DEV Comparison to Existing Baselines

Compare the selected final DEV result descriptively against:

Frozen strong audio:
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

    temporal-F1 final - frozen strong audio
    temporal-F1 final - pooled F1
    temporal-F1 final - pooled F2

for:

    DEV depression Score
    DEV Parkinson Score
    DEV Mean_Score

Primary conclusion remains temporal-F1 final versus SAME-row frozen audio base.

Do not use Test comparisons to determine the conclusion.

## 9. Frozen-Audio Integrity After Training

After training and post-hoc analysis, verify:

- every frozen audio parameter still has requires_grad=false;
- no frozen audio parameter was present in optimizer groups;
- frozen audio model remained eval-only;
- source checkpoint file SHA256 is unchanged;
- src/audio git diff is empty.

If feasible from the selected checkpoint/state payload, also verify no trainable checkpoint state unexpectedly introduces a trainable audio copy. Record the check performed.

## 10. No-Change Boundary

TASK-004I3 MUST NOT modify:

- source code;
- config;
- Chimera ML;
- DataModule;
- audio/video caches;
- checkpoint;
- thresholds.

TASK-004I3 MUST NOT start:

- F2-temporal;
- RAMPS;
- text;
- description.

Only PROGRESS_EN.md may be a tracked change.

## Acceptance Criteria

The task passes if:

- production pre-run firewall still passes;
- real training completes or configured early stopping ends it;
- every completed epoch has DEV + all three Test streams;
- all required metrics are finite;
- best checkpoint is selected only by dev/mean_score;
- train loss is never used for model selection;
- Test metrics never influence selection/tuning;
- exact MLflow run provenance is recorded;
- same-row frozen-base versus final DEV audit is recorded;
- base DEV reproduction remains within tolerance;
- corrected/introduced error diagnostics are recorded;
- frozen audio integrity remains intact;
- no source/config change occurs;
- src/audio unchanged;
- src/video unchanged;
- git diff --check passes;
- tracked diff against origin/main contains only docs/PROGRESS_EN.md;
- branch codex/task-004i3 committed and pushed;
- main/master untouched.

## Required PROGRESS_EN Update

Append TASK-004I3 evidence without erasing prior TASK-004I/TASK-004I2/TASK-004I2B history.

Record:

- branch;
- evidence commit SHA;
- push result;
- exact training command;
- config path;
- GPU/device;
- checkpoint path + SHA;
- production firewall recheck;
- parameter/optimizer counts;
- epochs completed;
- early-stopping status;
- FULL epoch table with:
  - train loss;
  - DEV task UAR/MF1/Score + Mean_Score;
  - TEST_NONE task UAR/MF1/Score + Mean_Score;
  - TEST_SOFT task UAR/MF1/Score + Mean_Score;
  - TEST_HARD task UAR/MF1/Score + Mean_Score;
- best epoch by dev/mean_score;
- selected DEV final metrics;
- same-epoch Test metrics;
- top-2/last checkpoint paths;
- run directory;
- train.log;
- summary.txt;
- code.zip;
- resolved config artifact if present;
- MLflow experiment/run ID/status/artifact URI;
- selected-checkpoint same-row frozen-base DEV metrics;
- final-minus-base DEV task/Mean deltas;
- residual distribution statistics;
- corrected_errors and introduced_errors per task;
- sign-flip counts/fractions;
- descriptive comparisons versus pooled F1/F2;
- frozen-audio post-run integrity evidence;
- explicit statement train loss did not select the model;
- explicit Test-selector firewall confirmation;
- explicit no source/config change;
- confirmation F2-temporal not started;
- confirmation RAMPS remains deferred;
- confirmation text/description deferred;
- src/audio unchanged;
- src/video unchanged;
- Stage 4 status;
- recommended next atomic task only.

If the experiment completes without a concrete implementation blocker:

    recommended next = TASK-004J implement/register the analogous strong-temporal-audio F2 directed residual model contract.

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

- branch codex/task-004i3;
- evidence commit SHA;
- pushed-to-origin status;
- main/master untouched;
- config path;
- epochs completed;
- best epoch;
- final DEV depression/Parkinson UAR/MF1/Score;
- final DEV Mean_Score;
- SAME-row frozen-base depression/Parkinson Scores and Mean_Score;
- final-minus-base DEV deltas;
- corrected/introduced error counts per task;
- same-epoch TEST_NONE/SOFT/HARD Mean_Scores;
- exact MLflow run ID/status;
- explicit train-loss-not-selector statement;
- Test firewall confirmation;
- frozen-audio integrity confirmation;
- no source/config changes;
- F2-temporal not started;
- RAMPS deferred;
- text/description deferred;
- src/audio unchanged;
- src/video unchanged.

Stop after this task.
