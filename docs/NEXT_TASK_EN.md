# TASK-005A2: Calibrate the Frozen Strong-Audio Disease Teachers and Build the Audited RAMPS-R1 TRAIN Missing-Head Target Cache

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-005a2

Do not train a student.
Do not modify the frozen audio checkpoint or src/audio.
Do not modify src/video.
Do not modify the accepted A+V DataModule.
Do not use Candidate B or any Stage 4 fusion model as a task-specific teacher.
Do not iterate or evaluate any Test protocol.
Do not start RAMPS R2/R3/R4.
Do not start text/description work.

## Goal

Start Stage 5 RAMPS R1 from the safe strong-audio anchor.

Use the exact historical DEV-selected frozen temporal-audio checkpoint as the teacher for BOTH disease heads.

For each disease:

1. infer its frozen teacher logits on canonical DEV;
2. fit one temperature using ONLY DEV rows where that disease label is observed;
3. fit separate positive/negative acceptance thresholds using ONLY that same observed-task DEV data;
4. infer all canonical TRAIN rows;
5. create soft calibrated pseudo-targets ONLY for missing task entries;
6. preserve observed truth as authoritative;
7. write an audited, offline, stop-gradient target cache.

This task produces teacher/calibration/target artifacts only.

It MUST NOT apply pseudo-supervision to a trainable student.

## Why the Teacher Is Frozen Strong Audio

Stage 4 found no safe multimodal winner.

Candidate B improved aggregate DEV and Parkinson strongly but failed the predeclared depression negative-transfer gate.

Do NOT cherry-pick Candidate B as a Parkinson-only teacher.

The accepted safe anchor is the exact frozen historical temporal-audio checkpoint:

    DEV depression Score = 0.7479183895
    DEV Parkinson Score = 0.8277353635
    DEV Mean_Score = 0.7878268765

RAMPS R1 therefore starts from one consistent, pre-existing, safe model-selection rule rather than task-specific post-hoc teacher selection.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 2, 5, 6, 8, 9, 10, 11, 12, 13, 14
3. docs/PLAN.md Stage 5
4. docs/PROGRESS_EN.md through MANAGER-DECISION-031
5. docs/NEXT_TASK_EN.md
6. docs/SOTA_REVIEW_EN.md Sections 2, 3, 5, 7, 8, 9, 11
7. src/fusion/models/frozen_audio_temporal_adapter.py
8. src/fusion/data/wsm_av_fusion_datamodule.py
9. src/common/loss/wsm_masked_sparse_loss.py
10. src/common/callbacks/wsm_segment_callback.py

## Allowed Tracked Files

- src/fusion/loss/ramps_r1_teacher.py
- src/fusion/loss/__init__.py
- scripts/common/prepare_ramps_r1_strong_audio_targets.py
- docs/PROGRESS_EN.md

No other tracked file may be modified.

No Chimera registry entry is required in this task because this is an offline teacher-calibration/target-preparation utility, not yet a training-selectable loss.

## Fixed Teacher

Exact checkpoint:

    logs/wsm_audio_segment_wavlm_base_l9_pool4/multitask_audio_mamba_2026-08-19_14-12_audio_mamba_segment_model_wsm_audio_models-e0ce-006_ea8c8d77/checkpoints/epoch=4_dev_mean_score=0.7878.pt

Required SHA256:

    0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2

Teacher adapter:

    FrozenAudioTemporalAdapter

Task order:

    0 = depression
    1 = parkinson

The teacher MUST be:

- loaded from the exact checkpoint;
- checkpoint SHA-verified before inference;
- eval-only;
- requires_grad=false for all audio parameters;
- run under torch.inference_mode() or equivalent;
- never updated;
- never placed in an optimizer.

Cached outputs are therefore detached/offline stop-gradient targets by construction.

## Fixed Data Inputs

Data root:

    /media/maxim/Databases/WSM_NEW

Audio feature cache:

    /media/maxim/Databases/WSM_NEW/features

Video feature cache:

    /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache

Use the accepted:

    WSMAVFusionDataModule

Expected canonical counts:

    train = 6325
    dev = 933
    test_none = 1364
    test_soft = 1208
    test_hard = 1014
    joined_total = 8622
    missing_audio = 0
    missing_video = 0

You MAY instantiate the DataModule even though it defines Test datasets.

You MUST NOT iterate, infer, evaluate, calibrate, threshold-fit, or compute metrics on:

    test_none
    test_soft
    test_hard

Only TRAIN and DEV may be consumed.

## Fixed Output Root

Write generated runtime artifacts outside Git tracking to:

    /media/maxim/Programs/Features/WSM/ramps_r1_strong_audio_teacher_v1

Required artifacts:

    teacher_calibration.json
    train_missing_targets.pt
    audit.json

Do not add these artifacts to git.

The preparation script must refuse to overwrite an existing non-empty output root unless an explicit --overwrite flag is supplied.

Do not use --overwrite in the production run unless the existing directory is first verified to be an incomplete artifact from this same task.

## 1. Reusable Binary Calibration Utilities

Implement in:

    src/fusion/loss/ramps_r1_teacher.py

The module must be deterministic and independent of Test data.

### Binary temperature scaling

For one disease task:

    calibrated_logit = raw_logit / T
    calibrated_prob = sigmoid(calibrated_logit)

with:

    T > 0

Fit one scalar temperature independently for depression and Parkinson using ONLY observed DEV labels for that task.

Use a stable positive parameterization, e.g.:

    T = exp(log_temperature)

Use torch only. Do not install dependencies.

Fit by minimizing binary NLL / BCE-with-logits on that task's observed DEV rows.

Use deterministic optimization.

Allowed deployed temperature interval:

    0.05 <= T <= 20.0

If an unconstrained fit leaves this interval:

- clamp only the final deployed temperature;
- recompute all reported post-calibration metrics using the deployed value;
- record the unclamped and deployed values.

### Calibration metrics

Implement reusable functions for:

- binary NLL/BCE;
- Brier score;
- ECE with 15 equal-width probability bins.

For each disease record BEFORE and AFTER temperature scaling:

- NLL;
- Brier;
- ECE-15.

Also record:

- observed DEV count;
- positive count;
- negative count.

Do not fabricate monotonic ECE/Brier improvement.

The fitted objective is NLL; report all metrics factually.

## 2. Fixed Class-Specific Acceptance Threshold Policy

Fit separate positive and negative thresholds for each disease using ONLY calibrated predictions on that disease's observed DEV labels.

Fixed policy:

    precision_target = 0.90
    min_support = 10
    threshold_step = 0.01

### Positive side

Accept positive pseudo-target if:

    p >= tau_pos

Candidate thresholds:

    0.50, 0.51, ..., 0.99

For each threshold compute precision among accepted predictions.

Choose the LOWEST threshold satisfying:

    support >= 10
    precision >= 0.90

This maximizes accepted coverage under the fixed reliability target.

### Negative side

Accept negative pseudo-target if:

    p <= tau_neg

Candidate thresholds:

    0.01, 0.02, ..., 0.50

Define negative precision as:

    accepted true negatives / accepted negative predictions

Choose the HIGHEST threshold satisfying:

    support >= 10
    precision >= 0.90

This maximizes accepted negative coverage under the fixed reliability target.

### Disabled side

If no candidate satisfies the fixed support/precision requirement:

- enabled=false;
- threshold=null;
- do NOT relax precision;
- do NOT lower min_support;
- do NOT use Test;
- continue the audit.

For each disease/class side record:

- enabled;
- threshold or null;
- accepted DEV support;
- accepted DEV precision;
- accepted DEV coverage/recall;
- precision_target;
- min_support.

## 3. TRAIN Missing-Head Soft Targets

Run the calibrated frozen teacher over ALL canonical TRAIN rows.

For each row/task:

### Observed entry

If:

    observed_mask == true

then:

    pseudo_accept_mask = false
    pseudo_target = NaN
    pseudo_reliability = 0
    pseudo_class = -1

Observed ground truth remains authoritative.

Never duplicate, overwrite, blend, or replace observed supervision in this task.

### Missing entry

If:

    observed_mask == false

compute calibrated probability p.

Positive accepted if:

    positive side enabled
    and p >= tau_pos

Negative accepted if:

    negative side enabled
    and p <= tau_neg

Otherwise rejected.

For accepted missing entries:

    pseudo_target = p

Keep the target SOFT.

Do NOT convert accepted probabilities to hard 0/1 labels.

Base R1 reliability:

    pseudo_reliability = 2 * abs(p - 0.5)

Accepted class:

    1 for accepted positive
    0 for accepted negative

For rejected missing entries:

    pseudo_target = NaN
    pseudo_reliability = 0
    pseudo_class = -1

All saved pseudo tensors must be detached CPU tensors.

R2 will later add uncertainty, modality agreement, and OOD evidence. Do not add those here.

## 4. train_missing_targets.pt Contract

Save a torch artifact containing at least:

    version = "ramps-r1-strong-audio-v1"

    task_names = ["depression", "parkinson"]

    segment_ids
    observed_mask
    observed_targets
    raw_teacher_logits
    calibrated_probs
    pseudo_accept_mask
    pseudo_targets
    pseudo_reliability
    pseudo_class

    temperatures
    thresholds

    teacher_checkpoint_path
    teacher_checkpoint_sha256
    adapter_source = "fusion.models.frozen_audio_temporal_adapter.FrozenAudioTemporalAdapter"

Expected tensor/table contract:

    rows = 6325
    tasks = 2

Requirements:

- exactly 6325 TRAIN rows;
- canonical order retained;
- unique segment_ids;
- observed_mask matches DataModule targets;
- observed targets unchanged;
- pseudo_accept_mask & observed_mask is always false;
- pseudo_targets NaN on every observed entry;
- pseudo_targets NaN on every rejected missing entry;
- accepted pseudo_targets finite and in [0,1];
- accepted pseudo_reliability finite and in [0,1];
- pseudo_reliability exactly zero on observed/rejected entries;
- pseudo_class exactly {-1,0,1} with semantics above.

## 5. teacher_calibration.json Contract

Record at least:

- artifact version;
- UTC generation timestamp;
- exact checkpoint path;
- checkpoint SHA256;
- exact adapter class;
- task order;
- DataModule/cache roots;
- canonical train/dev/join counts;
- exact calibration procedure;
- exact threshold-selection procedure;
- fixed precision_target/min_support/threshold_step/ECE bins;
- per-task temperature;
- per-task raw/calibrated DEV NLL;
- per-task raw/calibrated DEV Brier;
- per-task raw/calibrated DEV ECE-15;
- per-task observed DEV count;
- per-task positive/negative counts;
- per-task positive threshold audit;
- per-task negative threshold audit;
- explicit statement:
  "No Test rows or Test metrics were used."

Do NOT include Test metric values.

## 6. audit.json Contract

For each missing task separately record:

- missing-entry count;
- accepted total;
- rejected total;
- coverage = accepted / missing;
- accepted positive count;
- accepted negative count;
- accepted positive fraction;
- calibrated probability mean/std/min/max among accepted;
- pseudo reliability mean/std/min/max among accepted.

Overall record:

    train_rows = 6325
    train_missing_entries
    accepted_missing_entries
    accepted_overall_coverage
    observed_overwrite_violations = 0
    duplicate_segment_ids = 0
    nonfinite_accepted_targets = 0
    pseudo_values_on_observed_entries = 0
    pseudo_acceptance_on_observed_entries = 0

IMPORTANT:

The truly missing cross-corpus labels are unknown.

Therefore DO NOT report or claim pseudo-label accuracy, recall, F1, correctness, or comorbidity recovery on missing TRAIN entries.

The only precision values are those measured on corresponding observed-task DEV labels during threshold fitting.

## 7. Production Script

Implement:

    scripts/common/prepare_ramps_r1_strong_audio_targets.py

Required script sequence:

1. set deterministic seeds;
2. verify checkpoint SHA256;
3. build accepted A+V DataModule;
4. verify canonical counts;
5. build FrozenAudioTemporalAdapter;
6. verify teacher stop-gradient/eval-only;
7. infer DEV only;
8. fit independent disease temperatures on observed DEV labels;
9. fit class-specific DEV acceptance thresholds;
10. infer TRAIN only;
11. build missing-only soft pseudo-target cache;
12. validate every artifact invariant;
13. write artifacts atomically where practical;
14. print concise calibration + coverage summary.

The script MUST NOT call:

    dm.val_dataloader()

because that exposes Test streams.

Use:

    dm.val_dataset

directly for DEV.

Use TRAIN dataset/loader directly for TRAIN.

The script must not import/iterate test_none/test_soft/test_hard datasets for inference.

## Exact Production Command

Run from repository root:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
      scripts/common/prepare_ramps_r1_strong_audio_targets.py \
      --data-root /media/maxim/Databases/WSM_NEW \
      --audio-feature-cache-root /media/maxim/Databases/WSM_NEW/features \
      --video-cache-root /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache \
      --teacher-checkpoint logs/wsm_audio_segment_wavlm_base_l9_pool4/multitask_audio_mamba_2026-08-19_14-12_audio_mamba_segment_model_wsm_audio_models-e0ce-006_ea8c8d77/checkpoints/epoch=4_dev_mean_score=0.7878.pt \
      --output-root /media/maxim/Programs/Features/WSM/ramps_r1_strong_audio_teacher_v1 \
      --precision-target 0.90 \
      --min-support 10 \
      --threshold-step 0.01 \
      --ece-bins 15 \
      --batch-size 64 \
      --num-workers 4 \
      --device cuda

If CUDA is unavailable:

    STOP BLOCKED.

Do not silently switch production inference to CPU.

## Required Verification

Compile:

    python3 -m py_compile \
      src/fusion/loss/ramps_r1_teacher.py \
      src/fusion/loss/__init__.py \
      scripts/common/prepare_ramps_r1_strong_audio_targets.py

Run focused synthetic checks for:

- finite temperature fit;
- deployed temperature interval;
- NLL/Brier/ECE finite;
- positive threshold selection;
- negative threshold selection;
- threshold-disabled behavior;
- observed entry can never become pseudo entry;
- accepted soft target remains probability;
- reliability in [0,1].

Then run the exact production command.

After production load all three artifacts and assert their complete contracts.

Additionally verify:

Teacher stop-gradient:

- all audio parameters requires_grad=false;
- all teacher parameter grads are None;
- teacher remains eval-only;
- no optimizer instantiated for teacher.

No-Test proof:

- no Test loader is iterated;
- no Test prediction is generated;
- no Test metric value is produced;
- no Test-derived threshold/calibration value exists.

Then:

    git diff --check
    git diff -- src/audio
    git diff -- src/video
    git diff -- src/fusion/models
    git diff -- src/fusion/data
    git status --short

Before commit inspect only:

    git diff -- \
      src/fusion/loss/ramps_r1_teacher.py \
      src/fusion/loss/__init__.py \
      scripts/common/prepare_ramps_r1_strong_audio_targets.py \
      docs/PROGRESS_EN.md

After commit/push:

    git status --short
    git rev-parse --abbrev-ref HEAD
    git rev-parse HEAD
    git diff --stat origin/main...HEAD
    git diff --name-only origin/main...HEAD

## Acceptance Criteria

Pass only if:

- exact checkpoint SHA matches;
- frozen strong-audio teacher loads;
- teacher remains fully stop-gradient/eval-only;
- only observed DEV labels are used for calibration;
- only observed DEV labels are used for threshold fitting;
- no Test row/metric is consumed;
- one temperature is fit per disease;
- positive/negative threshold policy is executed separately per disease;
- a class side is disabled rather than relaxed if fixed reliability cannot be met;
- all 6325 TRAIN rows are inferred exactly once;
- pseudo-targets exist only on missing entries;
- observed truth is never overwritten;
- accepted targets remain soft probabilities;
- reliability is detached and in [0,1];
- calibration metrics are recorded;
- coverage/class balance is recorded separately per missing disease head;
- no missing-label correctness claim is made;
- at least one accepted missing pseudo-target exists for depression;
- at least one accepted missing pseudo-target exists for Parkinson;
- otherwise R1 is blocked rather than thresholds being relaxed;
- no student training;
- no training config;
- no Test evaluation;
- no Candidate B teacher/cherry-picking;
- no R2/R3/R4;
- text/description remains deferred;
- src/audio unchanged;
- src/video unchanged;
- accepted fusion models/DataModule unchanged;
- git diff --check passes;
- tracked diff contains only the four allowed paths;
- branch codex/task-005a2 committed and pushed;
- main/master untouched.

## Required PROGRESS_EN Update

Append TASK-005A2 evidence without erasing prior records.

Record:

- branch;
- implementation commit SHA;
- push result;
- checkpoint path + SHA256;
- adapter class;
- CUDA/device;
- canonical train/dev/join counts;
- output artifact root/filenames;
- per-task observed DEV support and class counts;
- per-task fitted temperature;
- raw/calibrated NLL, Brier, ECE-15;
- positive/negative thresholds or disabled states;
- DEV threshold support/precision/coverage;
- per-missing-head TRAIN accepted/rejected counts;
- accepted positive/negative counts;
- missing-head coverage;
- probability/reliability statistics;
- observed-overwrite audit;
- duplicate/nonfinite audits;
- teacher stop-gradient proof;
- explicit no-Test statement;
- explicit no-Candidate-B-teacher statement;
- explicit no pseudo-label correctness claim;
- explicit no student training;
- confirmation R2/R3/R4 not started;
- confirmation text/description deferred;
- src/audio unchanged;
- src/video unchanged;
- Stage 5 R1 status;
- recommended next atomic task only.

If TASK-005A2 passes:

    recommended next = TASK-005B wire the accepted offline targets into a sparse observed+pseudo RAMPS-R1 loss/data contract and prove a nonzero direct gradient reaches the missing head, without running a full experiment.

If blocked:

    recommend only a narrow correction for the exact failed calibration/coverage gate.

## Required Handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch codex/task-005a2;
- implementation commit SHA;
- pushed-to-origin status;
- main/master untouched;
- checkpoint SHA verification;
- fitted depression/Parkinson temperatures;
- positive/negative thresholds or disabled sides;
- raw/calibrated calibration metrics;
- TRAIN missing-head accepted/rejected/coverage counts for both tasks;
- accepted positive/negative counts;
- artifact paths;
- teacher stop-gradient confirmation;
- no Test use;
- no Candidate B teacher;
- no student training;
- no missing-label correctness claim;
- R2/R3/R4 not started;
- text/description deferred;
- src/audio unchanged;
- src/video unchanged.

Stop after this task.
