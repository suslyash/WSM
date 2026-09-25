# TASK-005C: Implement the Frozen Pseudo-Supervision Warm-Up Contract

## Task Identifier and Title

TASK-005C — Implement the manager-frozen pseudo-scale warm-up callback and config contract without running student training.

Required branch:

    codex/task-005c

Start from the current `origin/main`, which includes:

- PR #39 merge `8a4a7e13de7789a359b3849b65f9af8b39b0cc28`;
- manager acceptance/warm-up freeze commit `5f623660d665afd81c9d5f40dea86a1772936a74`.

Create exactly one task branch from that `origin/main`, implement only this schedule contract, verify it, update `docs/PROGRESS_EN.md`, commit, push, and stop.

## Goal

Implement the Stage-5 warm-up function already frozen by the manager before any student run:

    mu(e) = clamp((e - 3) / 5, 0, 1)

with one-based epochs.

Exact required values:

| Epoch | pseudo scale mu(e) |
|---:|---:|
| 1 | 0.0 |
| 2 | 0.0 |
| 3 | 0.0 |
| 4 | 0.2 |
| 5 | 0.4 |
| 6 | 0.6 |
| 7 | 0.8 |
| 8+ | 1.0 |

This is the fixed schedule for the first bounded R2 student experiment. It is not a search dimension in this task.

Implement one registered callback that updates the existing `WSMRampsObservedPseudoLoss.pseudo_scale` at the start of each one-based training epoch, plus one new config-selectable warm-up contract.

Do not run training.

## Required Reading

Read in this order:

1. AGENTS.md
2. docs/README.md
3. docs/PROJECT_REQUIREMENTS.md, especially Sections 8, 9, 11, 13, 14
4. docs/PLAN.md, especially the pseudo-loss equations and Stage 5
5. docs/PROGRESS_EN.md through MANAGER-DECISION-038
6. docs/NEXT_TASK_EN.md
7. src/common/callbacks/wsm_summary_callback.py
8. src/common/callbacks/wsm_segment_callback.py
9. src/fusion/loss/ramps_observed_pseudo_loss.py
10. src/fusion/data/wsm_ramps_semantic_datamodule.py
11. src/chimera_plugin.py
12. configs/wsm_mm_pd_dep_v1/fusion/03_ramps_r2_pseudo_contract.yaml

Use only active English docs from docs/README.md unless this task explicitly names a research reference. Do not load archived Russian variants.

## Allowed Files

Codex may modify only:

- src/common/callbacks/wsm_pseudo_scale_warmup_callback.py
- src/chimera_plugin.py
- configs/wsm_mm_pd_dep_v1/fusion/04_ramps_r2_warmup_contract.yaml
- docs/PROGRESS_EN.md

No other tracked file may change.

## Forbidden Actions

- Do not modify src/audio.
- Do not modify src/video.
- Do not modify src/fusion/models.
- Do not modify src/fusion/data.
- Do not modify src/fusion/loss.
- Do not modify the frozen pseudo cache.
- Do not modify semantic prompts/rules/thresholds/calibrators.
- Do not change pseudo acceptance or class counts.
- Do not change TASK-005B loss mathematics.
- Do not change the frozen schedule values.
- Do not make warm-up parameters tunable by a sweep in this task.
- Do not run `chimera-ml train`.
- Do not run an epoch training loop.
- Do not execute an optimizer step.
- Do not use DEV metrics to alter the schedule.
- Do not use or inspect Test metrics for any decision.
- Do not iterate Test rows.
- Do not start a bounded student run.
- Do not start R3/R4, Stage 6, Stage 7, or general Text/Description Stage 3.
- Do not make missing-label correctness or comorbidity recovery claims.

## 1. Registered Warm-Up Callback

Implement:

    src/common/callbacks/wsm_pseudo_scale_warmup_callback.py

Registry key:

    wsm_pseudo_scale_warmup_callback

The callback MUST subclass Chimera `BaseCallback`.

Use manager-frozen defaults:

    observed_only_epochs = 3
    ramp_epochs = 5
    final_scale = 1.0

These parameters MAY exist as constructor/config fields for transparent serialization, but the TASK-005C config MUST use exactly `3/5/1.0`. No alternate values may be evaluated in this task.

### Schedule function

Expose a deterministic method/helper such as:

    scale_for_epoch(epoch: int) -> float

Required one-based semantics:

    if epoch <= 3:
        0.0
    else:
        min(1.0, (epoch - 3) / 5)

Equivalent implementation is acceptable.

Require:

- epoch must be an integer >= 1;
- observed_only_epochs must be an integer >= 0;
- ramp_epochs must be an integer >= 1;
- final_scale must be finite and in [0,1].

The exact TASK-005C default/frozen values are `3,5,1.0`.

### Trainer integration

At `on_fit_start(trainer)`:

- require `trainer.loss_fn` to expose a writable numeric `pseudo_scale`;
- require its current value to be finite and in [0,1];
- do not modify model/optimizer/data;
- set/confirm the run begins from the schedule's epoch-1 value `0.0`;
- fail clearly if the active loss is incompatible instead of silently doing nothing.

At `on_epoch_start(trainer, epoch)`:

- compute the exact frozen schedule value;
- assign it to `trainer.loss_fn.pseudo_scale` before the first train batch;
- store the current scale on the callback for audit/logging;
- do not touch logits, pseudo targets, reliability, model parameters, or optimizer state.

At `on_epoch_end(trainer, epoch, logs)`:

- add exactly one scalar diagnostic key:

      train/pseudo_scale

  with the scale used for that epoch;
- if `trainer.mlflow_logger` is available, log the same scalar at `step=epoch`;
- do not consume DEV/Test values to change the scale.

Use the existing callback logging conventions. The module must remain modality-independent and MUST NOT import `fusion`, `audio`, `video`, `text`, or `description`.

## 2. Chimera Registration

Update:

    src/chimera_plugin.py

Add the callback module explicitly.

Required registry key after plugin registration:

    CALLBACKS: wsm_pseudo_scale_warmup_callback

All existing registrations must remain intact.

## 3. New Frozen Warm-Up Contract Config

Add:

    configs/wsm_mm_pd_dep_v1/fusion/04_ramps_r2_warmup_contract.yaml

Start from the accepted TASK-005B contract config:

    configs/wsm_mm_pd_dep_v1/fusion/03_ramps_r2_pseudo_contract.yaml

Preserve:

- seed 42;
- experiment_name `wsm_mm_pd_dep_v1`;
- frozen pseudo cache path;
- `wsm_ramps_semantic_datamodule`;
- existing F2 model and exact F2 architecture dimensions;
- `wsm_ramps_observed_pseudo_loss`;
- AdamW declaration;
- 30-epoch ceiling;
- required callbacks/loggers;
- DEV-only checkpoint/early-stopping monitor;
- separate evaluation streams supplied by the DataModule.

Use:

    run_name: ramps_r2_warmup_contract_smoke

The loss config MUST still initialize:

    pseudo_scale: 0.0

Add the registered warm-up callback with exactly:

    observed_only_epochs: 3
    ramp_epochs: 5
    final_scale: 1.0

Place it before `wsm_summary_callback` so the summary callback can see `train/pseudo_scale` after the warm-up callback appends it at epoch end.

This YAML is a contract scaffold only. Its existence does NOT authorize training.

## 4. Required Pure Schedule Regression

Run a deterministic no-training regression proving the exact schedule.

Required expected vector for epochs 1 through 10:

    [0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0]

Use exact or tight floating-point comparisons.

Also prove:

- epoch 0 is rejected;
- invalid `ramp_epochs=0` is rejected;
- invalid final_scale outside [0,1] is rejected.

## 5. Required Callback/Loss Integration Regression

Without Trainer.fit and without an optimizer step:

1. register project plugins;
2. construct the registered `wsm_ramps_observed_pseudo_loss` with initial `pseudo_scale=0.0`;
3. construct the registered warm-up callback with exact `3/5/1.0`;
4. create a minimal trainer stub containing:
   - `loss_fn`;
   - `logger=None`;
   - `mlflow_logger=None`;
5. call `on_fit_start`;
6. manually invoke `on_epoch_start` for epochs 1, 4, 8, and 30;
7. prove the loss scale becomes exactly `0.0, 0.2, 1.0, 1.0`;
8. invoke `on_epoch_end` and prove `logs["train/pseudo_scale"]` equals the current schedule value.

This is lifecycle-hook simulation, not training.

## 6. Required Direct-Gradient Schedule Regression

Use a tiny synthetic two-task batch and the existing registered loss.

Requirements:

- use identical logits/targets/masks/pseudo tensors for all checks;
- accepted pseudo entries exist;
- observed entries exist;
- manually apply callback epoch values before each loss call;
- retain logit gradients.

At epoch 1 (`mu=0.0`):

- observed direct gradients are non-zero;
- all missing-entry direct gradients are exactly zero.

At epoch 4 (`mu=0.2`):

- accepted missing direct gradients are non-zero;
- rejected missing direct gradients remain zero.

At epoch 8 (`mu=1.0`):

- accepted missing direct gradients are non-zero;
- rejected missing direct gradients remain zero.

For the same accepted missing element and identical logits, prove the epoch-4 pseudo direct gradient is `0.2` times the epoch-8 pseudo direct gradient within tight tolerance.

Pseudo targets and reliability remain detached; no teacher-side gradients may appear.

No optimizer step.

## 7. Required Actual-Cache Compatibility Smoke

Do not rerun a full training epoch.

Instantiate the accepted registered DataModule using the frozen cache and verify:

- cache SHA256 remains `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`;
- accepted counts remain depression `376`, Parkinson `1801`;
- callback can set the registered loss to epoch-1 scale 0.0 and epoch-8 scale 1.0;
- one tiny actual-cache F2 forward/loss/backward at epoch 8 remains finite and produces non-zero direct gradients on accepted missing depression and Parkinson entries and zero gradients on rejected missing entries;
- no optimizer step occurs.

Do not iterate DEV or any Test loader in this task.

## Exact Verification Commands

Run from repository root.

### 1. Compile

    python3 -m py_compile \
      src/common/callbacks/wsm_pseudo_scale_warmup_callback.py \
      src/chimera_plugin.py \
      src/fusion/loss/ramps_observed_pseudo_loss.py \
      src/fusion/data/wsm_ramps_semantic_datamodule.py

### 2. Config validation

    .venv/bin/chimera-ml validate-config -c configs/wsm_mm_pd_dep_v1/fusion/04_ramps_r2_warmup_contract.yaml

### 3. Registry and pure schedule regression

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import math
    import chimera_plugin
    from chimera_ml.core.registry import CALLBACKS

    chimera_plugin.register()
    assert "wsm_pseudo_scale_warmup_callback" in CALLBACKS.keys()

    cb = CALLBACKS.create(
        "wsm_pseudo_scale_warmup_callback",
        observed_only_epochs=3,
        ramp_epochs=5,
        final_scale=1.0,
    )

    expected = [0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0]
    actual = [cb.scale_for_epoch(epoch) for epoch in range(1, 11)]
    assert all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in zip(actual, expected))

    try:
        cb.scale_for_epoch(0)
    except ValueError:
        pass
    else:
        raise AssertionError("epoch 0 must fail")

    for bad in (
        dict(observed_only_epochs=3, ramp_epochs=0, final_scale=1.0),
        dict(observed_only_epochs=3, ramp_epochs=5, final_scale=1.1),
    ):
        try:
            CALLBACKS.create("wsm_pseudo_scale_warmup_callback", **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid schedule accepted: {bad}")

    print("TASK-005C registry/schedule regression: ok", actual)
    PY

### 4. Callback/loss lifecycle and synthetic gradient regression

Implement one inline command that:

- creates the registered loss/callback;
- runs `on_fit_start`;
- manually calls epochs 1,4,8,30 and verifies `0.0/0.2/1.0/1.0`;
- verifies `train/pseudo_scale` on epoch end;
- uses fixed synthetic logits to prove:
  - epoch 1 missing gradients zero;
  - epoch 4 accepted missing gradients non-zero;
  - epoch 8 accepted missing gradients non-zero;
  - rejected missing gradients always zero;
  - epoch-4 accepted pseudo gradient equals 0.2 times epoch-8 accepted pseudo gradient;
  - pseudo target/reliability receive no gradients.

Record the complete exact command in `docs/PROGRESS_EN.md`.

### 5. Actual-cache compatibility smoke

Use the same frozen cache/DataModule and tiny accepted/rejected TRAIN sample selection established by TASK-005B.

Manually invoke the warm-up callback rather than calling `Trainer.fit`.

At minimum assert:

- cache SHA exact;
- accepted counts exact;
- epoch 1 scale = 0.0;
- epoch 8 scale = 1.0;
- one epoch-8 F2 forward/loss/backward is finite;
- accepted missing depression/Parkinson direct gradients non-zero;
- rejected missing direct gradients zero;
- no optimizer step.

Record the complete exact command in `docs/PROGRESS_EN.md`.

### 6. Scope checks

    git diff --check
    git diff origin/main -- src/audio
    git diff origin/main -- src/video
    git diff origin/main -- src/fusion/models
    git diff origin/main -- src/fusion/data
    git diff origin/main -- src/fusion/loss
    git status --short
    git diff --stat origin/main...HEAD
    git log -4 --oneline --decorate

## Acceptance Criteria

TASK-005C passes only if:

- branch is exactly `codex/task-005c` from the manager-updated `origin/main`;
- only the four allowed files change;
- callback is registered as `wsm_pseudo_scale_warmup_callback`;
- callback module imports no modality package;
- exact frozen schedule is `0,0,0,0.2,0.4,0.6,0.8,1,...`;
- epoch numbering is one-based and invalid schedule inputs fail clearly;
- callback updates only the loss `pseudo_scale` lifecycle state and audit/log scalar;
- new YAML validates and uses exactly `3/5/1.0`;
- new YAML initializes loss `pseudo_scale: 0.0`;
- checkpoint and early stopping remain DEV/Mean_Score max-only;
- pure schedule regression passes;
- callback/loss lifecycle regression passes;
- synthetic direct-gradient scaling regression passes;
- pseudo target/reliability remain stop-gradient;
- actual-cache SHA/counts remain unchanged;
- actual-cache epoch-8 forward/loss/backward compatibility smoke passes;
- no optimizer step or training loop occurs;
- no DEV/Test metrics are used to alter schedule;
- no Test rows are iterated;
- no pseudo regeneration/reselection occurs;
- no missing-label correctness/comorbidity claim is made;
- src/audio, src/video, fusion model/data/loss files remain unchanged;
- `git diff --check` passes;
- `docs/PROGRESS_EN.md` records exact commands/results;
- one implementation commit is pushed to origin;
- main/master remains untouched by Codex.

Passing TASK-005C authorizes only manager review of the warm-up contract. It does NOT authorize a student training run. A subsequent manager task must separately freeze and authorize the first bounded student experiment.

## Required Handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch `codex/task-005c`;
- implementation commit SHA;
- pushed-to-origin status;
- main/master untouched;
- manager base/acceptance commits preserved;
- registry key;
- exact schedule formula and epoch 1-10 vector;
- config validation result;
- lifecycle hook regression result;
- synthetic gradient scale-ratio result;
- teacher stop-gradient result;
- actual cache SHA/counts;
- actual-cache compatibility smoke result;
- no optimizer step/training;
- no Test use/iteration;
- no pseudo regeneration/reselection;
- no missing-label correctness/comorbidity claim;
- R3/R4 not started;
- general Text/Description remains deferred;
- src/audio unchanged;
- src/video unchanged;
- fusion models/data/loss unchanged;
- Stage 5 remains active.

If all acceptance criteria pass, recommended next atomic step is manager review followed by a separately authorized first bounded seed-42 R2 student run using this frozen schedule.

Stop after TASK-005C.
