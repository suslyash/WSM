# TASK-002J: Implement Masked Two-Task DEV Metrics for the V1 Video Pipeline

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-002j

Do not create a training YAML, do not run training, do not process Test, and do not modify src/audio.

## Goal

Implement the training-selection metric contract required by PROJECT_REQUIREMENTS for the new two-logit sparse-label video pipeline.

The V1 video model outputs:

    logits [B,2] ordered [depression, parkinson]

The batch provides:

    targets [B,2]
    observed_mask [B,2]

Unknown disease labels are NaN and MUST be excluded from metrics exactly as they are excluded from the masked sparse loss.

Required metrics for every supported protocol prefix:

    dev
    test_none
    test_soft
    test_hard

For each task t in {depression, parkinson}:

    UAR_t = macro recall over the observed binary labels only
    MF1_t = macro F1 over the observed binary labels only
    Score_t = (UAR_t + MF1_t) / 2

Then for each protocol prefix P:

    P/mean_score = (P/depression/score + P/parkinson/score) / 2

The exact logged selector key remains:

    dev/mean_score

Checkpointing/early stopping in later training configs will monitor only this key in max mode.

The Test protocol metrics are mandatory epoch-level comparative-monitoring outputs. In later training, the callback/evaluation stack must compute dev, test_none, test_soft, and test_hard on every epoch/validation cycle. Test metrics MUST NOT select epochs, thresholds, hyperparameters, architectures, modalities, or ablations.

This task is metric/callback integration only. No training run is authorized, but the callback contract must be ready for all four epoch-level evaluation streams.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 2, 8, 9, 11, 13, 14
3. Stage 2 in docs/PLAN.md
4. docs/PROGRESS_EN.md through TASK-002I
5. docs/NEXT_TASK_EN.md
6. src/common/callbacks/wsm_segment_callback.py
7. src/common/callbacks/wsm_summary_callback.py
8. src/common/loss/wsm_masked_sparse_loss.py
9. src/video/data/wsm_video_cache_datamodule.py
10. src/video/models/depart_v1.py
11. src/chimera_plugin.py

## Allowed Tracked Files

- src/common/callbacks/wsm_segment_callback.py
- src/chimera_plugin.py
- docs/PROGRESS_EN.md

No other tracked file may be modified.

## Fixed Registry Contract

Preserve:

    wsm_segment_metrics_callback

Do not create a second competing callback key unless strictly necessary.

The callback must become compatible with BOTH:

- existing legacy single-task/split metric use where possible;
- the new sparse two-task [B,2] + observed_mask contract.

Do not break existing audio callback registration aliases.

## Required Metric Semantics

### 1. Sparse two-task metric computation

Add a reusable pure helper in wsm_segment_callback.py that accepts:

    logits: Tensor [N,2]
    targets: Tensor [N,2]
    observed_mask: Tensor [N,2]

and returns numeric metrics.

For each task:

- select only positions where observed_mask[:, task_id] is true;
- selected target values must be finite 0/1;
- threshold logits at 0.0, equivalent to sigmoid >= 0.5;
- compute binary confusion counts;
- per-class recall:
    recall_negative = TN / (TN + FP), when denominator > 0;
    recall_positive = TP / (TP + FN), when denominator > 0;
- UAR = mean over classes with defined recall;
- per-class F1:
    F1_negative and F1_positive from the one-vs-rest binary class view;
- MF1 = mean over classes with defined F1;
- Score = (UAR + MF1)/2.

If a DEV task has zero observed samples, raise a clear error. Do not silently report zero.

For every prefix P in {dev,test_none,test_soft,test_hard}, metrics must include at least:

    P/depression/num_samples
    P/depression/uar
    P/depression/mf1
    P/depression/score

    P/parkinson/num_samples
    P/parkinson/uar
    P/parkinson/mf1
    P/parkinson/score

    P/mean_score

Also expose sensible confusion counts for audit if useful.

### 2. Correct masking

Masked NaN targets MUST NOT enter any metric computation.

Changing values in targets where observed_mask=false must not affect any metric.

### 3. Callback cache integration

Update WSMSegmentMetricsCallback so that when cached DEV outputs contain the sparse two-task contract:

- concatenate logits and targets;
- obtain observed_mask from the cached batch/mask data if available in CachedSplitOutputs;
- compute and inject the exact metrics above into logs;
- log them to MLflow when present.

If the current Chimera cache object does not retain masks directly, implement the smallest compatible callback-side collection mechanism needed to collect DEV logits, targets, and observed_mask during the epoch.

Do not read real Test data in this task. The callback/helper implementation must nevertheless support test_none/test_soft/test_hard as mandatory epoch-level evaluation prefixes so that later training can compute all four streams every cycle.

Do not derive selector metrics from legacy task-split naming when the native sparse two-task cache is available.

### 4. Existing callback behavior

Preserve existing confusion-matrix/reporting behavior where compatible.

If the old confusion-matrix drawing cannot represent [N,2] sparse outputs directly, skip only that incompatible panel path for the native two-task cache rather than corrupting metrics.

Do not remove wsm_audio_metrics_callback alias.

### 5. Logging names

The canonical Stage 2 selector name is exactly:

    dev/mean_score

Do not use:

- dev/mean_macro_recall as selector;
- dev/mean_macro_f1 as selector;
- test metrics;
- corpus-specific surrogate metrics.

## Acceptance Criteria

- wsm_segment_metrics_callback remains registered;
- wsm_audio_metrics_callback remains registered;
- pure masked two-task helper exists;
- masked NaNs do not contaminate metrics;
- both tasks' UAR/MF1/Score are computed from observed labels only;
- exact key dev/mean_score is produced;
- helper supports dev, test_none, test_soft, and test_hard prefixes with identical masked-label semantics;
- each protocol mean_score equals arithmetic mean of the two task Scores;
- zero-observation task fails clearly;
- threshold is fixed at logit 0.0;
- Test metrics are supported as mandatory epoch-level comparative-monitoring outputs but are never selector inputs;
- existing project plugin imports without required-module warnings;
- no training run;
- no Test processing/metrics;
- src/audio unchanged;
- python compilation passes;
- synthetic metric smoke passes;
- callback smoke with a native sparse two-task DEV cache or equivalent collection path passes;
- git diff --check passes;
- branch codex/task-002j committed and pushed;
- main/master untouched;
- tracked diff contains only the three allowed paths.

## Exact Verification Commands

Compile:

    python3 -m py_compile \
      src/common/callbacks/wsm_segment_callback.py \
      src/chimera_plugin.py

Registry smoke:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    from chimera_ml.core.registry import CALLBACKS
    import chimera_plugin

    chimera_plugin.register()
    assert "wsm_segment_metrics_callback" in CALLBACKS.keys()
    assert "wsm_audio_metrics_callback" in CALLBACKS.keys()
    print("WSM metric callback registry smoke passed")
    PY

Run exact masked-metric smoke:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import math
    import torch

    from common.callbacks.wsm_segment_callback import compute_sparse_two_task_metrics

    logits = torch.tensor([
        [ 2.0,  999.0],
        [-2.0, -999.0],
        [999.0,  2.0],
        [-999.0, -2.0],
        [ 1.0,  999.0],
        [-1.0, -999.0],
        [999.0,  1.0],
        [-999.0, -1.0],
    ])

    targets = torch.tensor([
        [1.0, float("nan")],
        [0.0, float("nan")],
        [float("nan"), 1.0],
        [float("nan"), 0.0],
        [1.0, float("nan")],
        [0.0, float("nan")],
        [float("nan"), 1.0],
        [float("nan"), 0.0],
    ])

    observed = torch.tensor([
        [True, False],
        [True, False],
        [False, True],
        [False, True],
        [True, False],
        [True, False],
        [False, True],
        [False, True],
    ])

    metrics = compute_sparse_two_task_metrics(
        logits=logits,
        targets=targets,
        observed_mask=observed,
        prefix="dev",
        task_names=("depression", "parkinson"),
    )

    assert metrics["dev/depression/num_samples"] == 4
    assert metrics["dev/parkinson/num_samples"] == 4
    assert math.isclose(metrics["dev/depression/uar"], 1.0)
    assert math.isclose(metrics["dev/depression/mf1"], 1.0)
    assert math.isclose(metrics["dev/depression/score"], 1.0)
    assert math.isclose(metrics["dev/parkinson/uar"], 1.0)
    assert math.isclose(metrics["dev/parkinson/mf1"], 1.0)
    assert math.isclose(metrics["dev/parkinson/score"], 1.0)
    assert math.isclose(metrics["dev/mean_score"], 1.0)

    # Masked target values are irrelevant.
    changed = targets.clone()
    changed[~observed] = 12345.0
    metrics_changed = compute_sparse_two_task_metrics(
        logits=logits,
        targets=changed,
        observed_mask=observed,
        prefix="dev",
        task_names=("depression", "parkinson"),
    )
    assert metrics_changed == metrics

    # A task with zero observed labels must fail.
    bad = observed.clone()
    bad[:, 1] = False
    try:
        compute_sparse_two_task_metrics(
            logits=logits,
            targets=targets,
            observed_mask=bad,
            prefix="dev",
            task_names=("depression", "parkinson"),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("zero-observation task must fail")

    print("masked sparse DEV metric smoke passed")
    PY

Add a proportional callback smoke proving that WSMSegmentMetricsCallback injects the same dev/... keys when fed a native sparse two-task DEV epoch/cache path supported by the implementation.

Also run the pure helper with prefixes test_none, test_soft, and test_hard using synthetic tensors and assert the same key schema/values. Do not invoke real Test data in this task.

Finally:

    git diff --check
    git diff -- src/audio
    git status --short

Before commit inspect only:

    git diff -- \
      src/common/callbacks/wsm_segment_callback.py \
      src/chimera_plugin.py \
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
- exact metric equations/threshold;
- exact logged keys;
- masked-label semantics;
- callback integration path;
- synthetic helper result;
- callback smoke result;
- explicit selector key dev/mean_score;
- confirmation helper/callback metric schema supports dev/test_none/test_soft/test_hard;
- confirmation Test metrics are mandatory every-epoch monitoring outputs but not selectors;
- confirmation no real Test data/metrics were executed in this implementation-only task;
- confirmation no training run;
- src/audio unchanged;
- Stage 2 remains partial;
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

- branch codex/task-002j;
- implementation commit SHA;
- pushed-to-origin status;
- main/master untouched;
- exact selector key;
- synthetic metric result;
- callback integration result;
- no Test metrics;
- no training;
- src/audio unchanged.

Stop after this task.
