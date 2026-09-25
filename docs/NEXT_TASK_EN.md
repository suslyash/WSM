# TASK-005B-C1: Enforce Structural Stop-Gradient on Pseudo Targets and Reliability

## Task Identifier and Title

TASK-005B-C1 — Enforce structural stop-gradient on pseudo targets/reliability and reproduce the accepted TASK-005B actual-cache direct-gradient smoke.

This is one narrow corrective task only.

Required branch:

    codex/task-005b

The manager explicitly authorizes reuse of this existing branch for exactly one additional corrective implementation commit. Preserve all existing manager and Codex commits. Do not reset, rebase away, overwrite, or force-push branch history.

## Goal

Fix one remaining loss-contract defect in TASK-005B.

The current implementation detaches the pseudo target inside the BCE call, but the pseudo reliability tensor is still used live in the pseudo-loss numerator:

    weighted = reliability[pseudo_mask] * BCE(...)

The TASK-005B contract requires both teacher-side tensors to be stop-gradient:

- pseudo targets detached;
- pseudo reliability detached.

Make this structural, not accidental. The loss must remain safe even if a caller supplies pseudo targets/reliability with `requires_grad=True`.

Then rerun the already accepted actual-cache TASK-005B smoke unchanged in semantics.

Do not add warm-up scheduling and do not start training.

## Required Reading

Read in this order:

1. AGENTS.md
2. docs/README.md
3. docs/PROJECT_REQUIREMENTS.md, especially Sections 2, 11, 13
4. docs/PLAN.md, especially Stage 5
5. docs/PROGRESS_EN.md through MANAGER-REVIEW-037
6. docs/NEXT_TASK_EN.md
7. src/fusion/loss/ramps_observed_pseudo_loss.py
8. src/fusion/data/wsm_ramps_semantic_datamodule.py
9. configs/wsm_mm_pd_dep_v1/fusion/03_ramps_r2_pseudo_contract.yaml

Use only active English documentation listed in docs/README.md.

## Allowed Files

Codex may modify only:

- src/fusion/loss/ramps_observed_pseudo_loss.py
- docs/PROGRESS_EN.md

No other tracked file may change.

The manager has already modified docs/PROGRESS_EN.md and docs/NEXT_TASK_EN.md on this branch. Preserve those manager commits.

## Forbidden Actions

- Do not modify src/audio.
- Do not modify src/video.
- Do not modify src/fusion/models.
- Do not modify src/fusion/data.
- Do not modify src/chimera_plugin.py.
- Do not modify the TASK-005B YAML.
- Do not modify or regenerate the pseudo cache.
- Do not change pseudo acceptance, thresholds, rules, calibrators, counts, or class balance.
- Do not change the observed-loss formula.
- Do not change the pseudo-loss formula except for explicit stop-gradient handling.
- Do not change pseudo_scale behavior.
- Do not add hidden epoch state or warm-up scheduling.
- Do not run an optimizer step.
- Do not run a training loop.
- Do not use DEV/Test metrics.
- Do not iterate Test rows.
- Do not start TASK-005C, R3, R4, Stage 6, Stage 7, or general Text/Description work.
- Do not make missing-label correctness or comorbidity claims.

## Implementation Requirements

### 1. Explicit teacher-side detach

Inside `WSMRampsObservedPseudoLoss.compute_components()`, make both teacher-side tensors explicitly detached before any validation or loss use that could participate in autograd.

Required semantics:

    pseudo = pseudo_targets.to(device=device, dtype=dtype).detach()
    reliability = pseudo_reliability.to(device=device, dtype=dtype).detach()

Equivalent code is acceptable if it guarantees the same autograd contract.

Do not rely on the frozen cache tensors having `requires_grad=False`.

Do not detach model logits.

Do not detach observed targets/masks beyond ordinary device/dtype conversion.

### 2. Preserve the exact loss

Keep:

    L_obs =
        sum(BCEWithLogits(logit, observed_target))
        / (observed_count + eps)

and:

    pseudo_mask = pseudo_accept_mask & ~observed_mask

    L_pseudo =
        sum(detached_reliability * BCEWithLogits(logit, detached_soft_pseudo_target))
        / (sum(detached_reliability) + eps)

    L = L_obs + pseudo_scale * L_pseudo

All prior validation behavior remains.

No hardening of pseudo targets.

No semantic/video probability becomes a target.

### 3. Synthetic stop-gradient regression

Add no test file. Run an inline smoke using direct `compute_components()` or `__call__`.

The smoke must construct:

- logits with `requires_grad=True`;
- pseudo_targets with `requires_grad=True`;
- pseudo_reliability with `requires_grad=True`;
- valid observed and accepted masks.

After backward with `pseudo_scale=1.0`:

- logits must receive finite non-zero gradients on supervised entries;
- `pseudo_targets.grad` must be `None` or exactly zero;
- `pseudo_reliability.grad` must be `None` or exactly zero.

This must prove structural stop-gradient independent of the actual cache.

### 4. Reproduce the actual-cache gradient smoke

Rerun the TASK-005B actual-cache DataModule/F2/loss smoke after the change.

Required unchanged outcomes:

- cache SHA256 remains:
  `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`;
- accepted counts remain depression 376 and Parkinson 1801;
- `pseudo_scale=1.0`: accepted missing depression/Parkinson direct logit gradients non-zero;
- unaccepted missing direct logit gradients exactly zero;
- `pseudo_scale=0.0`: all missing direct logit gradients zero, observed gradients non-zero;
- both task heads have finite non-zero parameter gradients in the pseudo-on smoke;
- no optimizer step;
- no training;
- no Test use.

## Exact Verification Commands

Run from repository root on `codex/task-005b`.

### 1. Compile

    python3 -m py_compile       src/fusion/loss/ramps_observed_pseudo_loss.py       src/fusion/data/wsm_ramps_semantic_datamodule.py       src/chimera_plugin.py

### 2. Structural stop-gradient smoke

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import torch
    from fusion.loss.ramps_observed_pseudo_loss import WSMRampsObservedPseudoLoss

    logits = torch.tensor(
        [[0.2, -0.3], [0.4, 0.1]],
        dtype=torch.float64,
        requires_grad=True,
    )
    targets = torch.tensor(
        [[1.0, float("nan")], [float("nan"), 0.0]],
        dtype=torch.float64,
    )
    observed = torch.tensor(
        [[True, False], [False, True]],
        dtype=torch.bool,
    )
    accept = torch.tensor(
        [[False, True], [True, False]],
        dtype=torch.bool,
    )

    pseudo_targets = torch.tensor(
        [[float("nan"), 0.8], [0.3, float("nan")]],
        dtype=torch.float64,
        requires_grad=True,
    )
    pseudo_reliability = torch.tensor(
        [[0.0, 0.7], [0.6, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )

    loss_fn = WSMRampsObservedPseudoLoss(pseudo_scale=1.0, eps=1e-8)
    observed_loss, pseudo_loss = loss_fn.compute_components(
        logits,
        targets,
        observed,
        accept,
        pseudo_targets,
        pseudo_reliability,
    )
    total = observed_loss + pseudo_loss
    assert torch.isfinite(total)
    total.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert bool((logits.grad[(observed | accept)].abs() > 0).all())

    assert pseudo_targets.grad is None or bool((pseudo_targets.grad == 0).all())
    assert pseudo_reliability.grad is None or bool((pseudo_reliability.grad == 0).all())

    print("TASK-005B-C1 structural stop-gradient smoke: ok")
    PY

### 3. Config and registry regression

    .venv/bin/chimera-ml validate-config -c configs/wsm_mm_pd_dep_v1/fusion/03_ramps_r2_pseudo_contract.yaml

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import chimera_plugin
    from chimera_ml.core.registry import DATAMODULES, LOSSES, MODELS
    chimera_plugin.register()
    assert "wsm_ramps_semantic_datamodule" in DATAMODULES.keys()
    assert "wsm_ramps_observed_pseudo_loss" in LOSSES.keys()
    assert "wsm_av_f2_task_aware_directed_model" in MODELS.keys()
    print("TASK-005B-C1 registry regression: ok")
    PY

### 4. Actual-cache forward/loss/backward regression

Rerun the exact TASK-005B actual-cache smoke recorded in `docs/PROGRESS_EN.md`.

At minimum record and assert:

- cache SHA256 `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`;
- accepted depression/Parkinson = `376/1801`;
- pseudo_scale=1 accepted missing direct gradients non-zero;
- pseudo_scale=1 unaccepted missing gradients zero;
- pseudo_scale=0 all missing gradients zero;
- observed gradients non-zero;
- finite task-head gradients;
- no optimizer step/training/Test use.

### 5. Scope checks

    git diff --check
    git diff origin/main -- src/audio
    git diff origin/main -- src/video
    git diff origin/main -- src/fusion/models
    git diff origin/main -- src/fusion/data
    git status --short
    git diff --stat origin/main...HEAD
    git log -4 --oneline --decorate

## Acceptance Criteria

TASK-005B-C1 passes only if:

- only `src/fusion/loss/ramps_observed_pseudo_loss.py` and `docs/PROGRESS_EN.md` change after the manager task commit;
- pseudo targets are explicitly detached before pseudo-loss use;
- pseudo reliability is explicitly detached before pseudo-loss use;
- synthetic `requires_grad=True` regression proves zero/None teacher-side gradients;
- model logits still receive expected finite gradients;
- original TASK-005B actual-cache gradient behavior reproduces unchanged;
- cache SHA/counts remain unchanged;
- config still validates;
- registry keys remain present;
- no optimizer step or training occurs;
- no Test rows/metrics are used;
- src/audio/src/video/existing fusion models/data remain unchanged;
- `git diff --check` passes;
- docs/PROGRESS_EN.md records exact commands/results;
- one corrective commit is pushed to origin;
- main/master remains untouched by Codex.

Passing C1 allows manager acceptance/merge of TASK-005B and only then consideration of a separate warm-up-schedule task.

## Required Handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch `codex/task-005b`;
- corrective commit SHA;
- pushed-to-origin status;
- main/master untouched;
- manager commits preserved;
- exact detach implementation;
- synthetic pseudo-target/reliability stop-gradient result;
- actual-cache SHA and counts;
- pseudo_scale=1 actual-cache gradient behavior;
- pseudo_scale=0 actual-cache gradient behavior;
- finite task-head gradients;
- config/registry regression results;
- no optimizer step/training/Test use;
- no pseudo regeneration/reselection;
- no missing-label correctness/comorbidity claim;
- TASK-005C/R3/R4 not started;
- general Text/Description remains deferred;
- src/audio unchanged;
- src/video unchanged;
- existing fusion models/data unchanged;
- Stage 5 remains active.

If all acceptance criteria pass, recommended next atomic step is manager acceptance/merge of TASK-005B followed by a separate warm-up-schedule task.

Stop after TASK-005B-C1.
