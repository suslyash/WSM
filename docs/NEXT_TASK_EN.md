# TASK-005A: Calibrate the Frozen F2 Disease Teachers and Build the Audited RAMPS-R1 TRAIN Missing-Head Target Cache

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-005a

Do not train a student.
Do not modify F0/F1/F2 models.
Do not modify the A+V DataModule.
Do not modify src/audio or src/video.
Do not evaluate or iterate any Test protocol.
Do not start RAMPS R2/R3/R4.
Do not add text/description work.

## Goal

Start Stage 5 RAMPS R1 with an auditable frozen-teacher target artifact.

Use the DEV-selected F2 checkpoint as the frozen teacher, calibrate each disease head independently using ONLY DEV rows where that disease label is observed, fit separate positive/negative acceptance thresholds from that same observed-task DEV data, and then produce soft calibrated pseudo-targets for ONLY the missing task entries of canonical TRAIN rows.

This task produces teacher/calibration/target artifacts only.

It MUST NOT apply pseudo-supervision to a trainable student yet.

## Why This Task Exists

The Stage 5 gate requires a direct gradient to the missing head, but the teacher signal must be frozen, calibrated, auditable, and separated from observed truth before the pseudo-loss is wired.

TASK-005A therefore establishes:

1. frozen stop-gradient teacher evidence;
2. disease-specific temperature calibration;
3. separate positive/negative acceptance thresholds;
4. soft cross-corpus TRAIN pseudo-targets for missing entries only;
5. base confidence reliability weights;
6. coverage/class-balance audit.

TASK-005B will consume this artifact and prove the direct missing-head gradient.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 2, 5, 8, 9, 10, 11, 12, 13, 14
3. docs/PLAN.md Stage 5
4. docs/PROGRESS_EN.md through MANAGER-DECISION-023
5. docs/NEXT_TASK_EN.md
6. docs/SOTA_REVIEW_EN.md Sections 2, 3, 5, 7, 8, 9, 11
7. src/fusion/data/wsm_av_fusion_datamodule.py
8. src/fusion/models/av_f2_task_aware_directed.py
9. configs/wsm_mm_pd_dep_v1/fusion/02_f2_task_aware_directed.yaml
10. src/common/loss/wsm_masked_sparse_loss.py

## Allowed Tracked Files

- src/fusion/loss/ramps_r1_teacher.py
- src/fusion/loss/__init__.py
- scripts/common/prepare_ramps_r1_teacher_targets.py
- docs/PROGRESS_EN.md

No other tracked file may be modified.

No Chimera registry entry is required in this task because this is an offline teacher-calibration/target-preparation utility, not yet a training-selectable loss.

## Fixed Teacher

Teacher architecture/config:

    configs/wsm_mm_pd_dep_v1/fusion/02_f2_task_aware_directed.yaml

Frozen DEV-selected checkpoint:

    logs/wsm_mm_pd_dep_v1/av_f2_task_aware_directed_2026-09-24_18-18_wsm_av_f2_task_aware_directed_model_f1b9902d/checkpoints/epoch=5_dev_mean_score=0.7746.pt

Teacher selected DEV evidence:

    dev/mean_score = 0.774569

Task order:

    0 = depression
    1 = parkinson

The teacher MUST be:

- loaded from the exact selected checkpoint;
- set to eval mode;
- have requires_grad=false for every parameter;
- run under torch.inference_mode() or equivalent;
- never updated;
- never placed in an optimizer.

Cached teacher outputs are therefore detached/offline stop-gradient targets by construction.

## Fixed Data Inputs

Data root:

    /media/maxim/Databases/WSM_NEW

Audio feature cache:

    /media/maxim/Databases/WSM_NEW/features

Video feature cache:

    /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache

Use the accepted:

    wsm_av_fusion_datamodule

Expected counts:

    train = 6325
    dev = 933
    joined_total = 8622
    missing_audio = 0
    missing_video = 0

You MAY instantiate the DataModule even though it defines Test datasets, but this task MUST NOT iterate, infer, evaluate, calibrate, threshold-fit, or compute metrics on:

    test_none
    test_soft
    test_hard

Only TRAIN and DEV may be consumed.

## Fixed Output Root

Write generated runtime artifacts outside Git tracking to:

    /media/maxim/Programs/Features/WSM/ramps_r1_f2_teacher_v1

Required artifacts:

    teacher_calibration.json
    train_missing_targets.pt
    audit.json

Do not add these artifacts to git.

The preparation script must refuse to overwrite an existing non-empty output root unless an explicit --overwrite flag is supplied.

Do not use --overwrite in the production TASK-005A run unless the existing artifact is first verified to be an incomplete artifact produced by this same task.

## 1. Reusable Calibration/Acceptance Utilities

Implement in:

    src/fusion/loss/ramps_r1_teacher.py

The module must be deterministic and independent of Test data.

### Binary temperature scaling

Implement a binary temperature scaler operating on logits.

For one task:

    calibrated_logit = raw_logit / T
    calibrated_prob = sigmoid(calibrated_logit)

with:

    T > 0

Fit one scalar temperature independently for each disease using ONLY observed DEV labels for that disease.

Use a stable positive parameterization, e.g.:

    T = exp(log_temperature)

Use torch optimization only; do not install a dependency.

The fit objective is binary NLL/BCE-with-logits on the corresponding observed DEV task rows.

Use a deterministic optimizer setup.

After fitting, require:

    finite(T)
    0.05 <= T <= 20.0

If unconstrained optimization exits outside the allowed range, clamp only the final deployed temperature to the allowed range and recompute/report calibration metrics with the deployed value.

### Calibration metrics

Implement reusable functions for:

- binary NLL/BCE;
- Brier score;
- ECE with 15 equal-width probability bins.

Report these before and after temperature scaling for each disease.

Also record:

- number of observed DEV examples;
- positive count;
- negative count.

Do not require ECE to monotonically improve; report the actual result.

Temperature fitting is judged primarily by the fitted NLL objective and artifact validity, not by fabricating an ECE improvement.

## 2. Manager-Fixed Class-Specific Threshold Policy

Fit separate acceptance thresholds for positive and negative pseudo-labels for each disease using ONLY that disease's observed DEV labels after temperature calibration.

Fixed parameters:

    precision_target = 0.90
    min_support = 10
    threshold_step = 0.01

Positive acceptance:

    accept positive if p >= tau_pos

Candidate positive thresholds:

    0.50, 0.51, ..., 0.99

For each candidate, compute precision among accepted positive predictions.

Choose the LOWEST candidate threshold satisfying:

    accepted support >= 10
    precision >= 0.90

This maximizes accepted coverage subject to the fixed reliability target.

Negative acceptance:

    accept negative if p <= tau_neg

Candidate negative thresholds:

    0.01, 0.02, ..., 0.50

For each candidate, define negative precision as:

    fraction of accepted rows whose observed label is 0

Choose the HIGHEST candidate threshold satisfying:

    accepted support >= 10
    negative precision >= 0.90

This maximizes accepted negative coverage subject to the fixed reliability target.

If no candidate satisfies a class side:

- mark that side disabled;
- store threshold=null;
- do NOT invent or relax a threshold;
- continue the audit.

Record for each disease/class side:

- selected threshold or null;
- enabled flag;
- accepted DEV support;
- accepted DEV precision;
- accepted DEV recall/coverage;
- precision_target;
- min_support.

Do not use Test to choose thresholds.

## 3. TRAIN Missing-Head Soft Targets

Run the frozen calibrated teacher over all canonical TRAIN rows.

For each row and task:

Observed entry:

    observed_mask=true

Then:

- observed target remains authoritative;
- pseudo_accept_mask=false;
- pseudo_target=NaN;
- pseudo_reliability=0;
- never overwrite or duplicate observed supervision.

Missing entry:

    observed_mask=false

Compute calibrated teacher probability p.

Acceptance:

- positive accepted if positive side enabled and p >= tau_pos;
- negative accepted if negative side enabled and p <= tau_neg;
- otherwise rejected.

For an accepted missing entry:

    pseudo_target = p

The target remains SOFT. Do NOT hard-replace it with 0 or 1.

Base R1 reliability:

    pseudo_reliability = 2 * abs(p - 0.5)

For rejected missing entries:

    pseudo_target = NaN
    pseudo_reliability = 0

All pseudo targets/reliabilities must be detached CPU tensors in the saved artifact.

R2 will later add uncertainty, independent modality agreement, and OOD evidence. Do not add those now.

## 4. train_missing_targets.pt Contract

Save a torch artifact containing at least:

    version = "ramps-r1-f2-teacher-v1"

    task_names = ["depression", "parkinson"]

    segment_ids               # length 6325, canonical order
    observed_mask             # bool [6325,2]
    observed_targets          # float [6325,2], NaN for unknown
    raw_teacher_logits        # float [6325,2]
    calibrated_probs          # float [6325,2]
    pseudo_accept_mask        # bool [6325,2]
    pseudo_targets            # float [6325,2], NaN unless accepted missing
    pseudo_reliability        # float [6325,2], zero unless accepted missing
    pseudo_class              # int8 [6325,2], -1 rejected/not eligible, 0 accepted-negative, 1 accepted-positive

    temperatures
    thresholds
    teacher_checkpoint_path
    teacher_checkpoint_sha256
    teacher_config_path
    teacher_config_sha256

Requirements:

- exactly 6325 rows;
- exactly one canonical TRAIN segment_id per row;
- no duplicate segment_id;
- observed_mask matches the DataModule targets;
- observed target values are unchanged;
- pseudo_accept_mask & observed_mask is always false;
- pseudo_targets are NaN for every observed entry;
- pseudo_targets are NaN for every rejected missing entry;
- accepted pseudo targets are finite and strictly within [0,1];
- accepted pseudo reliability is finite and within [0,1];
- rejected/observed pseudo reliability is exactly zero.

## 5. teacher_calibration.json Contract

Record at least:

- artifact version;
- UTC generation timestamp;
- teacher config path + SHA256;
- teacher checkpoint path + SHA256;
- teacher selected DEV Mean_Score as provenance only;
- task order;
- DataModule/cache roots;
- canonical/join counts;
- exact calibration procedure;
- exact threshold-selection procedure;
- per-task temperature;
- raw/calibrated DEV NLL;
- raw/calibrated DEV Brier;
- raw/calibrated DEV ECE-15;
- DEV observed support/positive/negative counts;
- positive threshold audit;
- negative threshold audit;
- explicit statement:
  "No Test rows or Test metrics were used."

Do not include Test metric values in this artifact.

## 6. audit.json Contract

Audit the generated TRAIN pseudo-target cache.

For each missing head separately:

Depression pseudo-targets on rows where depression is missing.

Parkinson pseudo-targets on rows where Parkinson is missing.

Record:

- missing-entry count;
- accepted total;
- rejected total;
- coverage = accepted/missing;
- accepted positive count;
- accepted negative count;
- accepted positive fraction;
- mean/std/min/max calibrated probability among accepted;
- mean/std/min/max pseudo reliability among accepted.

Also record overall:

- train_rows=6325;
- train_missing_entries;
- accepted_missing_entries;
- accepted overall coverage;
- observed overwrite violations=0;
- duplicate segment ids=0;
- nonfinite accepted targets=0;
- pseudo values on observed entries=0;
- pseudo acceptance on observed entries=0.

IMPORTANT:

The missing cross-corpus labels are unknown.

Therefore DO NOT report or claim pseudo-label accuracy/precision on the missing TRAIN entries.

The only precision numbers are the threshold-fitting precision measured on the corresponding observed-task DEV labels.

## 7. Production Script

Implement:

    scripts/common/prepare_ramps_r1_teacher_targets.py

Required CLI arguments/defaults must support the exact production command below.

The script must:

1. set deterministic seeds;
2. build the accepted A+V DataModule;
3. verify canonical counts;
4. build/load the exact F2 model/checkpoint;
5. freeze teacher;
6. infer DEV only for calibration/threshold fitting;
7. fit the two independent temperatures;
8. fit class-specific thresholds on observed DEV only;
9. infer TRAIN;
10. build missing-only pseudo targets;
11. validate all invariants;
12. write the three artifacts atomically where practical;
13. print a concise calibration + coverage summary.

Do not import or iterate Test loaders.

## Exact Production Command

Run from repository root:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
      scripts/common/prepare_ramps_r1_teacher_targets.py \
      --data-root /media/maxim/Databases/WSM_NEW \
      --audio-feature-cache-root /media/maxim/Databases/WSM_NEW/features \
      --video-cache-root /media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache \
      --teacher-config configs/wsm_mm_pd_dep_v1/fusion/02_f2_task_aware_directed.yaml \
      --teacher-checkpoint logs/wsm_mm_pd_dep_v1/av_f2_task_aware_directed_2026-09-24_18-18_wsm_av_f2_task_aware_directed_model_f1b9902d/checkpoints/epoch=5_dev_mean_score=0.7746.pt \
      --output-root /media/maxim/Programs/Features/WSM/ramps_r1_f2_teacher_v1 \
      --precision-target 0.90 \
      --min-support 10 \
      --threshold-step 0.01 \
      --ece-bins 15 \
      --batch-size 64 \
      --num-workers 4 \
      --device cuda

If CUDA is unavailable, stop as blocked. Do not silently switch the production teacher inference to CPU.

## Required Verification

Compile:

    python3 -m py_compile \
      src/fusion/loss/ramps_r1_teacher.py \
      src/fusion/loss/__init__.py \
      scripts/common/prepare_ramps_r1_teacher_targets.py

Run focused synthetic checks for:

- temperature scaling with finite T;
- ECE/Brier/NLL finite;
- positive threshold selection;
- negative threshold selection;
- threshold-disabled behavior when precision/support cannot be met;
- observed entries can never become pseudo entries;
- soft pseudo targets remain probabilities;
- reliability in [0,1].

Then run the exact production command.

After production, load all three artifacts and assert the full contracts above.

Additionally verify teacher stop-gradient:

- all teacher parameters requires_grad=false;
- no teacher parameter has a gradient;
- no optimizer is instantiated for the teacher.

Verify no Test use by code inspection and runtime evidence:

- script does not call val_dataloader() because that includes Test streams;
- script accesses dm.val_dataset directly for DEV;
- script accesses dm.train_dataset/train loader for TRAIN;
- no test_none/test_soft/test_hard dataset is iterated;
- no Test metric value is produced.

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
      scripts/common/prepare_ramps_r1_teacher_targets.py \
      docs/PROGRESS_EN.md

After commit/push:

    git status --short
    git rev-parse --abbrev-ref HEAD
    git rev-parse HEAD
    git diff --stat origin/main...HEAD
    git diff --name-only origin/main...HEAD

## Acceptance Criteria

The task passes if:

- frozen selected F2 teacher loads successfully;
- teacher is fully stop-gradient;
- only DEV observed labels are used for calibration and threshold fitting;
- no Test row/metric is consumed;
- separate temperature exists for depression and Parkinson;
- separate positive/negative threshold policy is executed for each disease;
- class side is disabled rather than relaxed if the fixed precision/support target cannot be met;
- full TRAIN inference covers exactly 6325 unique rows;
- pseudo targets exist only on missing entries;
- observed truth is never overwritten;
- pseudo targets remain soft calibrated probabilities;
- confidence reliability is detached and in [0,1];
- required calibration metrics are recorded;
- coverage/class balance is recorded separately per missing head;
- cross-corpus missing-target correctness is NOT claimed;
- at least one accepted missing pseudo-target exists for each disease head; otherwise report Stage 5 R1 blocked rather than fabricating acceptance;
- no student training;
- no training config;
- no Test metrics;
- no R2/R3/R4;
- text/description remains deferred;
- src/audio unchanged;
- src/video unchanged;
- F0/F1/F2 source unchanged;
- A+V DataModule unchanged;
- git diff --check passes;
- tracked diff contains only the four allowed paths;
- branch codex/task-005a committed and pushed;
- main/master untouched.

## Required PROGRESS_EN Update

Append TASK-005A evidence without erasing prior records.

Record:

- branch;
- implementation commit SHA;
- push result;
- exact teacher config/checkpoint paths;
- checkpoint/config SHA256;
- output artifact root and filenames;
- CUDA/device;
- train/dev counts and join audit;
- per-task observed DEV support and class counts;
- per-task fitted temperature;
- raw/calibrated NLL, Brier, ECE-15;
- selected positive/negative thresholds or disabled state;
- threshold DEV support/precision/coverage;
- per-missing-head TRAIN accepted/rejected counts;
- accepted positive/negative counts and coverage;
- reliability statistics;
- proof observed truth was not overwritten;
- proof teacher stop-gradient;
- explicit no-Test statement;
- explicit statement that no pseudo-target correctness claim is possible for the genuinely missing TRAIN labels;
- confirmation no student training;
- confirmation R2/R3/R4 not started;
- confirmation text/description deferred;
- src/audio unchanged;
- src/video unchanged;
- Stage 5 R1 status;
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

- branch codex/task-005a;
- implementation commit SHA;
- pushed-to-origin status;
- main/master untouched;
- teacher checkpoint;
- fitted temperatures;
- positive/negative thresholds or disabled sides;
- calibration metrics;
- TRAIN pseudo-target coverage/class counts for both missing heads;
- output artifact paths;
- stop-gradient confirmation;
- no-Test confirmation;
- no student training;
- no cross-corpus pseudo-label accuracy claim;
- text/description deferred;
- src/audio unchanged;
- src/video unchanged.

Stop after this task.
