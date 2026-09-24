# TASK-005B: Wire the Accepted Semantic R2 Cache into the Registered TRAIN Data/Loss Contract

## Task Identifier and Title

TASK-005B — Wire the accepted `ramps-r2-semantic-v1` TRAIN cache into a registered A+V DataModule/loss contract and prove direct accepted-missing-head gradients without running training.

Required branch:

    codex/task-005b

Start from the current `origin/main`, which includes manager acceptance commit `bd6a09a729e2a7a7bd843b35474674b1e8c2af54` after PR #38 merge.

Execute only this contract/smoke task, update `docs/PROGRESS_EN.md`, commit, push, and stop.

## Goal

Use the frozen accepted artifact:

    /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt

as the only pseudo-target source for TRAIN.

Implement:

1. one registered A+V DataModule that overlays the frozen pseudo-target cache on canonical TRAIN rows while preserving the existing DEV/TEST datasets and observed labels;
2. one registered observed+pseudo sparse loss with a detached reliability weight and an explicit external `pseudo_scale`;
3. one config-selectable smoke contract using the existing F2 A+V model;
4. a bounded actual-cache forward/loss/backward smoke proving that an accepted missing head gets a non-zero direct gradient, while an unaccepted missing head gets zero direct supervision and observed truth always wins.

This task does NOT train a student, choose a checkpoint, compare DEV scores, tune `pseudo_scale`, implement epoch warm-up scheduling, or start R3/R4.

The purpose is to close the direct-gradient/data-loss portion of the Stage-5 gate before any training authorization.

## Required Reading

Read in this order:

1. AGENTS.md
2. docs/README.md
3. docs/PROJECT_REQUIREMENTS.md, especially Sections 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14
4. docs/PLAN.md, especially Stage 5 and the R1/R2 rows
5. docs/PROGRESS_EN.md through MANAGER-DECISION-036
6. docs/NEXT_TASK_EN.md
7. src/fusion/data/wsm_av_fusion_datamodule.py
8. src/common/loss/wsm_masked_sparse_loss.py
9. src/fusion/models/av_f2_task_aware_directed.py
10. src/chimera_plugin.py
11. configs/wsm_mm_pd_dep_v1/fusion/02_f2_task_aware_directed.yaml

Use only active English documents listed in docs/README.md. Do not load archived Russian variants.

## Allowed Files

Codex may modify only:

- src/fusion/data/wsm_ramps_semantic_datamodule.py
- src/fusion/loss/ramps_observed_pseudo_loss.py
- src/chimera_plugin.py
- configs/wsm_mm_pd_dep_v1/fusion/03_ramps_r2_pseudo_contract.yaml
- docs/PROGRESS_EN.md

No other tracked file may change.

## Forbidden Actions

- Do not modify any file under src/audio.
- Do not modify any file under src/video.
- Do not modify existing fusion model implementations.
- Do not modify `src/fusion/data/wsm_av_fusion_datamodule.py`.
- Do not modify the frozen pseudo-target artifact or any Stage-5 semantic audit artifact.
- Do not regenerate or reselect pseudo-targets.
- Do not change semantic prompts, rules, thresholds, calibrators, or acceptance.
- Do not change the accepted depression/Parkinson pseudo counts.
- Do not use Test metrics or Test rows for any selection.
- Do not iterate TEST_NONE, TEST_SOFT, or TEST_HARD in this task.
- Do not call `val_dataloader()`.
- Do not run `chimera-ml train`.
- Do not run any epoch loop or optimizer step.
- Do not tune `pseudo_scale`.
- Do not implement the final warm-up scheduler in this task.
- Do not start TASK-005C, R3, R4, Stage 6, Stage 7, or general Text/Description Stage 3.
- Do not claim missing-label correctness or comorbidity recovery.
- Do not convert the task to three-class softmax.
- Do not silently treat unknown labels as negatives.

## Frozen Cache Contract

Required path:

    /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt

Required identity:

    version = ramps-r2-semantic-v1
    task_names = ["depression", "parkinson"]
    rows = 6325
    clip_model_name = openai/clip-vit-base-patch32
    clip_model_revision = main

Required teacher SHA256:

Audio:

    0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2

Video:

    3e39778126db401a2fe17bb472a172191616e8a7b5265e982233c53dcd4af2f6

Required prompt-bank canonical JSON SHA256:

    19428db58f91f73ca26ce9c4354b5731431e14ec524fc1a47e7070e1b32f447e

Required accepted TRAIN counts:

Depression missing head:
- missing = 2665
- accepted = 376
- accepted positive = 376
- accepted negative = 0

Parkinson missing head:
- missing = 3660
- accepted = 1801
- accepted positive = 212
- accepted negative = 1589

Total accepted missing entries:

    2177

These values are frozen inputs to this task. Do not reselect them.

## 1. Registered Pseudo-Aware DataModule

Implement:

    src/fusion/data/wsm_ramps_semantic_datamodule.py

Registry key:

    wsm_ramps_semantic_datamodule

The implementation SHOULD reuse/subclass `WSMAVFusionDataModule`; do not duplicate the base A+V manifest/cache joining logic.

Required init parameter:

    pseudo_cache_path: str

The DataModule must load the frozen `train_missing_targets.pt` on CPU and validate it before exposing a TRAIN loader.

### Cache-to-canonical identity validation

Require:

- cache version and task_names match exactly;
- cache has 6325 rows and unique segment IDs;
- canonical TRAIN has 6325 rows;
- cache segment IDs match canonical TRAIN segment IDs exactly in order;
- cache `observed_mask` exactly matches canonical TRAIN observed masks;
- cache `observed_targets` exactly matches canonical TRAIN targets, including identical NaN positions;
- CLIP name/revision match the frozen identity;
- audio/video checkpoint SHA fields match the frozen SHA values;
- semantic prompt-bank SHA matches the frozen SHA;
- all pseudo tensor shapes are [6325,2] where task-shaped;
- accepted counts/class counts match the frozen counts above;
- no pseudo acceptance overlaps observed truth;
- accepted targets are finite and in [0,1];
- accepted targets equal `calibrated_audio_probs` exactly;
- pseudo reliability is finite/in [0,1] on accepted entries;
- pseudo reliability is zero on rejected/observed entries;
- pseudo target is NaN on rejected/observed entries;
- pseudo class is -1 on rejected/observed entries and only {-1,0,1} globally.

Compute and record the actual cache-file SHA256 in `self.audit["pseudo_cache_sha256"]` for provenance; do not require a predeclared file hash.

Expose read-only/audit-friendly CPU tensors or properties sufficient for the required smoke, including at least:

    pseudo_accept_mask
    pseudo_targets
    pseudo_reliability
    pseudo_class
    train_segment_ids

Do not mutate the loaded artifact tensors.

### TRAIN sample/batch contract

Only TRAIN rows may receive pseudo fields.

For TRAIN samples/batches expose:

- original observed `batch.targets` unchanged;
- original `observed_mask` unchanged;
- `pseudo_accept_mask` [B,2] bool;
- `pseudo_targets` [B,2];
- `pseudo_reliability` [B,2];
- `pseudo_class` [B,2].

Use a custom collate local to the new module if needed.

Recommended Batch placement:

- `batch.masks["pseudo_accept_mask"]`
- `batch.inputs["pseudo_targets"]`
- `batch.inputs["pseudo_reliability"]`
- `batch.inputs["pseudo_class"]`

Evaluation rows must never receive active pseudo supervision. If the shared collate is used for DEV/Test samples, it must emit:

- pseudo_accept_mask = false;
- pseudo_targets = NaN;
- pseudo_reliability = 0;
- pseudo_class = -1.

Preserve the base DataModule's separate DEV and TEST protocol structure. Do not iterate Test in this task.

The DataModule audit must include frozen accepted/missing/class counts and cache identity/provenance.

## 2. Registered Observed + Pseudo Loss

Implement:

    src/fusion/loss/ramps_observed_pseudo_loss.py

Registry key:

    wsm_ramps_observed_pseudo_loss

It must implement Chimera `BaseLoss` and consume two independent logits [B,2].

Constructor parameters:

    pseudo_scale: float = 0.0
    eps: float = 1e-8

Require:

    0.0 <= pseudo_scale <= 1.0
    eps >= 0

The explicit `pseudo_scale` is a hook for the later warm-up task. Do NOT add hidden epoch state or scheduling here.

### Observed term

Use exactly the existing sparse observed BCE semantics:

    L_obs = sum(BCEWithLogits(logit, target) over observed entries)
            / (observed_count + eps)

Unknown observed targets remain masked and must never enter BCE.

### Pseudo term

Only accepted missing entries are eligible:

    pseudo_mask = pseudo_accept_mask & ~observed_mask

Any overlap between `pseudo_accept_mask` and `observed_mask` is a contract violation: raise instead of silently using pseudo truth.

Detach pseudo targets and reliability before using them.

For accepted entries:

    element = BCEWithLogits(logit, soft_pseudo_target)
    weighted = reliability * element

Define:

    L_pseudo =
        sum(weighted)
        / (sum(reliability over pseudo_mask) + eps)

If no accepted pseudo entry exists in a batch, `L_pseudo` must be an exact differentiable zero and must not create missing-head gradients.

Total:

    L = L_obs + pseudo_scale * L_pseudo

Do not harden soft pseudo targets.

Do not average semantic/video probabilities into the target.

The calibrated strong-audio probability remains the target.

### Loss validation

Require:

- logits/targets/observed/pseudo tensors have [B,2] shape;
- observed targets selected by observed mask are finite binary 0/1;
- accepted pseudo targets are finite in [0,1];
- accepted reliability is finite in [0,1];
- rejected/observed reliability is zero;
- rejected/observed pseudo targets are NaN;
- overlap observed & pseudo_accept is forbidden.

Provide a deterministic helper such as `compute_components(...)` if useful for smoke evidence, but `__call__` must return one scalar Tensor.

## 3. Chimera Plugin Registration

Update:

    src/chimera_plugin.py

Explicitly import/register:

    fusion.data.wsm_ramps_semantic_datamodule
    fusion.loss.ramps_observed_pseudo_loss

Required registry keys after plugin registration:

    DATAMODULES: wsm_ramps_semantic_datamodule
    LOSSES: wsm_ramps_observed_pseudo_loss

Existing registrations must remain intact.

## 4. Config-Selectable Contract

Add:

    configs/wsm_mm_pd_dep_v1/fusion/03_ramps_r2_pseudo_contract.yaml

Requirements:

- experiment_name = `wsm_mm_pd_dep_v1`;
- run_name = `ramps_r2_pseudo_contract_smoke`;
- data = `wsm_ramps_semantic_datamodule`;
- exact canonical A+V roots;
- pseudo_cache_path = frozen semantic cache path;
- model = existing `wsm_av_f2_task_aware_directed_model`;
- preserve the F2 architecture params from `02_f2_task_aware_directed.yaml`;
- loss = `wsm_ramps_observed_pseudo_loss`;
- set `pseudo_scale: 0.0` in the YAML so this contract config cannot accidentally train with full pseudo supervision before a manager-authorized warm-up task;
- optimizer may remain the same AdamW declaration for structural config compatibility;
- seed = 42;
- include all required callbacks/loggers;
- checkpoint and early stopping monitor only `dev/mean_score` in max mode.

This YAML is a contract/smoke scaffold, not authorization to train it.

## 5. Required Actual-Cache Forward/Loss/Backward Smoke

Use the registered components and actual frozen cache.

The smoke must:

1. register plugins;
2. assert the new registry keys exist;
3. instantiate `wsm_ramps_semantic_datamodule` with `num_workers=0`, `persistent_workers=false`, `shuffle_train=false`;
4. deterministically select a tiny TRAIN sample set containing:
   - at least one accepted missing depression entry;
   - at least one accepted missing Parkinson entry;
   - at least one unaccepted missing depression entry;
   - at least one unaccepted missing Parkinson entry;
5. collate those samples with the new DataModule collate;
6. assert observed targets/masks and pseudo fields satisfy the contract;
7. instantiate the existing registered F2 model on CPU with the fixed F2 dimensions and seed 42;
8. run a real model forward;
9. retain gradients on `output.preds`;
10. instantiate the registered pseudo loss with `pseudo_scale=1.0`;
11. compute finite loss and backward;
12. prove:
    - observed entries receive non-zero direct logit gradients;
    - accepted missing depression receives non-zero direct depression-logit gradient;
    - accepted missing Parkinson receives non-zero direct Parkinson-logit gradient;
    - unaccepted missing entries receive exactly zero direct logit gradient;
    - all model parameter gradients that are present are finite;
    - both task heads have at least one non-zero finite parameter gradient;
13. repeat the loss-gradient check with `pseudo_scale=0.0` and prove accepted missing entries now receive zero direct logit gradient while observed entries remain supervised.

No optimizer step.

No training loop.

No DEV/Test metric calculation.

## 6. Exact Verification Commands

Run from repository root.

### Compile

    python3 -m py_compile \
      src/fusion/data/wsm_ramps_semantic_datamodule.py \
      src/fusion/loss/ramps_observed_pseudo_loss.py \
      src/chimera_plugin.py

### Config validation

    .venv/bin/chimera-ml validate-config -c configs/wsm_mm_pd_dep_v1/fusion/03_ramps_r2_pseudo_contract.yaml

### Registry validation

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import chimera_plugin
    from chimera_ml.core.registry import DATAMODULES, LOSSES, MODELS

    chimera_plugin.register()

    assert "wsm_ramps_semantic_datamodule" in DATAMODULES.keys()
    assert "wsm_ramps_observed_pseudo_loss" in LOSSES.keys()
    assert "wsm_av_f2_task_aware_directed_model" in MODELS.keys()
    print("TASK-005B registry validation: ok")
    PY

### Cache/DataModule and forward/loss/backward smoke

Use exactly this invocation shell; implement the assertions inside the heredoc according to the public attributes/helpers created by this task:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import torch
    import chimera_plugin
    from chimera_ml.core.registry import DATAMODULES, LOSSES, MODELS

    torch.manual_seed(42)
    chimera_plugin.register()

    dm = DATAMODULES.create(
        "wsm_ramps_semantic_datamodule",
        data_root="/media/maxim/Databases/WSM_NEW",
        audio_feature_cache_root="/media/maxim/Databases/WSM_NEW/features",
        video_cache_root="/media/maxim/Programs/Features/WSM/video_depart_v1_fullframe_fallback/cache",
        pseudo_cache_path="/media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt",
        batch_size=8,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        shuffle_train=False,
        drop_last_train=False,
    )

    assert len(dm.train_dataset) == 6325
    assert tuple(dm.pseudo_accept_mask.shape) == (6325, 2)
    assert int(dm.pseudo_accept_mask[:, 0].sum()) == 376
    assert int(dm.pseudo_accept_mask[:, 1].sum()) == 1801

    observed = dm.train_observed_mask
    accept = dm.pseudo_accept_mask

    dep_acc = torch.where((~observed[:, 0]) & accept[:, 0])[0][0].item()
    park_acc = torch.where((~observed[:, 1]) & accept[:, 1])[0][0].item()
    dep_rej = torch.where((~observed[:, 0]) & (~accept[:, 0]))[0][0].item()
    park_rej = torch.where((~observed[:, 1]) & (~accept[:, 1]))[0][0].item()

    indices = []
    for idx in (dep_acc, park_acc, dep_rej, park_rej):
        if idx not in indices:
            indices.append(idx)

    samples = [dm.train_dataset[i] for i in indices]
    batch = dm.collate_fn(samples)

    model = MODELS.create(
        "wsm_av_f2_task_aware_directed_model",
        audio_feature_dim=dm.audio_feature_dim,
        video_feature_dim=dm.video_feature_dim,
        hidden_dim=192,
        fusion_hidden_dim=192,
        relation_hidden_dim=192,
        dropout=0.0,
        num_tasks=2,
    )
    model.train()

    loss_on = LOSSES.create("wsm_ramps_observed_pseudo_loss", pseudo_scale=1.0, eps=1e-8)
    output = model(batch)
    assert tuple(output.preds.shape) == (len(indices), 2)
    assert torch.isfinite(output.preds).all()
    output.preds.retain_grad()

    value = loss_on(output, batch)
    assert value.ndim == 0 and torch.isfinite(value)
    value.backward()

    grad = output.preds.grad
    assert grad is not None and torch.isfinite(grad).all()

    obs_b = batch.get_masks("observed_mask").bool()
    acc_b = batch.get_masks("pseudo_accept_mask").bool()
    supervised = obs_b | acc_b
    assert bool((grad[supervised].abs() > 0).all())
    assert bool((grad[~supervised] == 0).all())

    head0 = [p.grad for p in model.task_heads[0].parameters() if p.grad is not None]
    head1 = [p.grad for p in model.task_heads[1].parameters() if p.grad is not None]
    assert head0 and head1
    assert all(torch.isfinite(g).all() for g in head0 + head1)
    assert any(bool((g.abs() > 0).any()) for g in head0)
    assert any(bool((g.abs() > 0).any()) for g in head1)

    model.zero_grad(set_to_none=True)

    loss_off = LOSSES.create("wsm_ramps_observed_pseudo_loss", pseudo_scale=0.0, eps=1e-8)
    output_off = model(batch)
    output_off.preds.retain_grad()
    value_off = loss_off(output_off, batch)
    value_off.backward()
    grad_off = output_off.preds.grad

    assert grad_off is not None and torch.isfinite(grad_off).all()
    assert bool((grad_off[obs_b].abs() > 0).all())
    missing = ~obs_b
    assert bool((grad_off[missing] == 0).all())

    assert all(
        torch.isfinite(g).all()
        for p in model.parameters()
        if (g := p.grad) is not None
    )

    print("TASK-005B actual-cache forward/loss/backward smoke: ok")
    print(dm.audit)
    PY

If the exact public attribute names above need a tiny naming adjustment during implementation, keep their semantics identical and record the final exact smoke command in PROGRESS_EN.md.

### Scope checks

    git diff --check
    git diff -- src/audio
    git diff -- src/video
    git diff -- src/fusion/models
    git status --short
    git diff --stat origin/main...HEAD

No training command is authorized.

## Acceptance Criteria

TASK-005B passes only if:

- the frozen semantic cache validates exactly against canonical TRAIN identity/observed truth;
- accepted counts/classes remain exactly frozen;
- no observed truth is overwritten;
- unknown labels remain unknown unless an accepted pseudo target exists;
- new DataModule is registered as `wsm_ramps_semantic_datamodule`;
- new loss is registered as `wsm_ramps_observed_pseudo_loss`;
- plugin imports both without project-module warnings;
- contract YAML validates and has experiment_name `wsm_mm_pd_dep_v1`;
- YAML monitors only DEV/Mean_Score for checkpoint/early stopping;
- YAML pseudo_scale is 0.0 and no training is run;
- actual-cache F2 forward succeeds with two independent logits;
- observed entries receive non-zero direct gradient;
- accepted missing depression and Parkinson entries receive non-zero direct gradients when pseudo_scale=1.0;
- unaccepted missing entries receive zero direct gradients;
- all missing-entry gradients become zero when pseudo_scale=0.0;
- gradients are finite and both task heads receive finite non-zero parameter gradients in the pseudo-on smoke;
- no optimizer step or epoch training occurs;
- no Test rows or metrics are iterated/inspected;
- no pseudo re-selection/re-generation occurs;
- no missing-label correctness/comorbidity claim is made;
- src/audio and src/video remain unchanged;
- existing fusion models remain unchanged;
- `git diff --check` passes;
- docs/PROGRESS_EN.md is updated with exact commands/results;
- branch `codex/task-005b` is committed and pushed;
- main/master remains untouched by Codex.

Passing TASK-005B closes only the direct pseudo-gradient/data-loss contract portion of the Stage-5 gate. It does not authorize full training. Warm-up scheduling and the next bounded student experiment remain manager-gated.

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
- implementation commit SHA;
- pushed-to-origin status;
- main/master untouched;
- cache path and computed cache-file SHA256;
- cache identity validation;
- exact accepted missing/class counts;
- registry keys;
- config validation result;
- observed+pseudo loss formula actually implemented;
- pseudo_scale=1 and pseudo_scale=0 gradient smoke results;
- proof of non-zero accepted missing-head gradients for both diseases;
- proof of zero unaccepted missing-head gradients;
- finite model/task-head gradients;
- no optimizer step;
- no training;
- no Test use;
- no pseudo regeneration/reselection;
- no missing-label correctness/comorbidity claim;
- TASK-005C/R3/R4 not started;
- general Text/Description remains deferred;
- src/audio unchanged;
- src/video unchanged;
- existing fusion models unchanged;
- Stage 5 remains active.

Recommended next atomic step after success: manager review of the direct-gradient gate and a separate task to add the predeclared pseudo-supervision warm-up schedule before any bounded student training.

Stop after TASK-005B.
