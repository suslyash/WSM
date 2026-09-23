# TASK-001D: Implement the Observed-Label-Only Masked Sparse Loss Contract

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit the implementation on the required task branch, push that branch to origin, then stop. Follow AGENTS.md.

The required branch is:

    codex/task-001d

Do not add pseudo-labeling, reliability weighting, model code, training configs, or any Stage 2 work in this cycle.

## Goal

Implement the required sparse supervised baseline loss for the two-task partial-label contract:

    L_obs = sum(m_it * loss(logit_it, y_it)) / (sum(m_it) + eps)

Only observed labels may contribute to the loss.

Unknown disease targets must never become supervised negatives and must not contaminate gradients even when represented as NaN in the batch.

This task is loss-contract only. Stage 1 remains partial after completion because authoritative speaker identity is unresolved and the Stage 1 gate has not yet been manager-closed.

## Required Reading

Read before editing:

1. AGENTS.md;
2. docs/PROJECT_REQUIREMENTS.md, especially Sections 2, 3, 4, 9, 10, 11, and 13;
3. Stage 1 in docs/PLAN.md;
4. docs/PROGRESS_EN.md, especially TASK-001B and TASK-001C;
5. docs/NEXT_TASK_EN.md;
6. src/fusion/data/wsm_manifest_datamodule.py;
7. src/audio/loss/wsm_audio_loss.py, for Chimera BaseLoss conventions only;
8. src/chimera_plugin.py.

## Allowed Files

- src/common/loss/__init__.py;
- src/common/loss/wsm_masked_sparse_loss.py;
- src/chimera_plugin.py;
- docs/PROGRESS_EN.md.

Creating src/common/loss is allowed if it does not exist.

No other tracked file may be modified.

## Forbidden Actions

- any change under src/audio;
- changing the manifest or datamodule contract;
- changing labels, masks, split semantics, or modality availability;
- converting unknown targets to 0;
- replacing NaN unknowns with supervised numeric negatives;
- pseudo-labeling;
- uncertainty/reliability logic;
- class-threshold tuning;
- adding or modifying models;
- adding a training config;
- training, checkpoint selection, threshold selection, or model selection;
- reading Test predictions or Test performance metrics;
- dependency installation;
- broad refactors;
- editing docs/NEXT_TASK_EN.md;
- pushing implementation directly to main/master;
- opening or merging a PR.

## Implementation Requirements

### 1. Chimera loss

Implement a BaseLoss-compatible loss registered as:

    wsm_masked_sparse_loss

The callable must accept Chimera ModelOutput and Batch.

Expected contract:

- output.preds contains logits of shape [B, 2] ordered [depression, parkinson];
- batch must provide targets of shape [B, 2];
- batch must provide observed_mask of shape [B, 2].

Because the current manifest collate returns a plain dict, the loss must support the actual project batch convention without silently guessing. Use the established Chimera Batch API where available and add only the smallest compatibility helper needed to retrieve:

- targets;
- observed_mask.

Do not modify the datamodule in this task.

### 2. Loss semantics

Use independent binary BCE-with-logits per task.

Requirements:

- logits must be [B,2];
- observed_mask must be bool-compatible [B,2];
- only mask=true elements contribute;
- unknown target values may be NaN, but masked-out NaN values must not propagate into the scalar loss;
- observed targets must be finite and in {0,1};
- if zero observed elements are supplied, fail clearly rather than returning a misleading zero;
- the final loss is the mean over observed task elements only;
- no corpus weighting, class weighting, focal term, label smoothing, pseudo-label term, or task balancing in this task.

### 3. Gradient safety

The implementation must guarantee:

- gradients for masked-out logits are exactly zero in the simple smoke case;
- changing only masked-out target placeholders does not change the loss;
- changing an observed target does change the loss;
- finite forward scalar and finite gradients for valid observed entries.

### 4. Plugin registration

Update src/chimera_plugin.py to import the new registration module explicitly.

Do not change unrelated registrations.

## Acceptance Criteria

- LOSSES registry contains wsm_masked_sparse_loss using LOSSES.keys() or the installed Registry-compatible equivalent;
- plugin import emits no project-module warning for the new loss module;
- loss accepts logits [B,2], targets [B,2], observed_mask [B,2];
- masked-out NaN targets do not make the loss NaN;
- unknown/masked targets never contribute to the result;
- zero-observed batch raises a clear exception;
- masked-out logits receive zero gradient in smoke verification;
- observed logits receive finite gradients;
- no pseudo-labeling or weighting logic exists;
- no Test predictions/performance metrics are inspected;
- no training/model selection occurs;
- no src/audio change;
- python compilation passes;
- registry smoke passes;
- forward/loss/backward smoke passes;
- git diff --check passes;
- task branch is codex/task-001d;
- implementation is committed and pushed to origin;
- main/master is not modified by implementing Codex;
- diff against origin/main contains only the four allowed tracked paths.

## Exact Verification Commands

Run from repository root after creating codex/task-001d from current origin/main.

    python3 -m py_compile       src/common/loss/__init__.py       src/common/loss/wsm_masked_sparse_loss.py       src/chimera_plugin.py

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import warnings
    from chimera_ml.core.registry import LOSSES
    import chimera_plugin

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chimera_plugin.register()

    project_warnings = [
        str(item.message)
        for item in caught
        if "Failed to import" in str(item.message)
        and "common.loss.wsm_masked_sparse_loss" in str(item.message)
    ]
    assert not project_warnings, project_warnings
    assert "wsm_masked_sparse_loss" in LOSSES.keys()
    print("masked loss registry assertions passed")
    PY

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import math
    import torch

    from common.loss.wsm_masked_sparse_loss import WSMMaskedSparseLoss

    loss_fn = WSMMaskedSparseLoss()

    logits = torch.tensor(
        [[0.2, -0.7],
         [1.0,  0.4]],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor(
        [[1.0, float("nan")],
         [float("nan"), 0.0]],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [[True, False],
         [False, True]],
        dtype=torch.bool,
    )

    loss = loss_fn.compute_from_tensors(logits, targets, mask)
    assert loss.ndim == 0
    assert math.isfinite(float(loss.detach()))
    loss.backward()

    assert torch.isfinite(logits.grad[mask]).all()
    assert torch.equal(logits.grad[~mask], torch.zeros_like(logits.grad[~mask]))

    with torch.no_grad():
        targets_alt = targets.clone()
        targets_alt[~mask] = 123.0
        same = loss_fn.compute_from_tensors(logits.detach(), targets_alt, mask)
        assert torch.allclose(loss.detach(), same)

        changed = targets.clone()
        changed[0, 0] = 0.0
        different = loss_fn.compute_from_tensors(logits.detach(), changed, mask)
        assert not torch.allclose(loss.detach(), different)

    try:
        loss_fn.compute_from_tensors(
            torch.zeros(1, 2),
            torch.full((1, 2), float("nan")),
            torch.zeros(1, 2, dtype=torch.bool),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("zero-observed batch must fail")

    print("masked sparse loss forward/backward smoke passed")
    PY

    git diff --check

    git diff -- src/audio

    git status --short

Before committing, inspect:

    git diff --       src/common/loss/__init__.py       src/common/loss/wsm_masked_sparse_loss.py       src/chimera_plugin.py       docs/PROGRESS_EN.md

Then commit and push according to AGENTS.md.

After commit and push, verify:

    git status --short

    git rev-parse --abbrev-ref HEAD

    git rev-parse HEAD

    git diff --stat origin/main...HEAD

    git diff --name-only origin/main...HEAD

The final diff name list must contain only:

    src/common/loss/__init__.py
    src/common/loss/wsm_masked_sparse_loss.py
    src/chimera_plugin.py
    docs/PROGRESS_EN.md

## Required PROGRESS_EN Update

Append TASK-001D facts without erasing prior evidence.

Record:

- exact branch name;
- implementation commit SHA;
- push result;
- registry key;
- exact masking formula/semantics;
- unknown-target representation and proof masked NaNs do not affect loss;
- zero-observed behavior;
- forward/backward smoke result including zero gradient on masked logits;
- exact verification commands/results;
- confirmation src/audio stayed unchanged;
- confirmation no Test predictions/metrics were inspected;
- confirmation no training/model selection ran;
- Stage 1 remains partial;
- recommended next atomic step only.

## Required Handoff

Respond in English using exactly these headings:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly state:

- branch: codex/task-001d;
- implementation commit SHA;
- whether the branch was pushed to origin;
- that implementing Codex did not modify main/master;
- diff summary against origin/main;
- src/audio unchanged;
- Test predictions/metrics not inspected.

Stop after this task. Do not implement pseudo-labeling, model code, training config, or Stage 2 work.
