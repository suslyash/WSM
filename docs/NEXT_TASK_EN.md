# TASK-004I2: Repair Strong-Audio Optimizer Filtering and Re-Run All Pre-Training Gates

## Role

You are the implementing Codex. Execute only this corrective task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-004i2

Do not start the real 30-epoch training run.
Do not modify Chimera ML.
Do not modify src/audio or src/video.
Do not modify the accepted A+V DataModule.
Do not modify the frozen audio adapter or F1-temporal model.
Do not start F2-temporal, RAMPS, text, or description work.

## Goal

Repair the exact blocker found in TASK-004I.

Chimera ML v0.2.4 built-in:

    adamw_optimizer

constructs AdamW from:

    model.parameters()

and therefore includes all 105 frozen audio parameter objects even though they have:

    requires_grad = false

The strong-temporal-audio ablation requires frozen audio parameters to be completely absent from optimizer parameter groups.

Implement one project-side Chimera optimizer:

    wsm_trainable_adamw_optimizer

that includes exactly the parameters where:

    parameter.requires_grad == true

Switch only the fixed strong-audio config to that optimizer and rerun every TASK-004I pre-training gate.

Do not train beyond a bounded one-batch forward/loss/backward smoke.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 3, 5, 6, 8, 9, 13, 14
3. docs/PLAN.md Stage 4
4. docs/PROGRESS_EN.md through MANAGER-DECISION-026
5. docs/NEXT_TASK_EN.md
6. configs/wsm_mm_pd_dep_v1/fusion/03_f1_temporal_audio_residual.yaml
7. src/fusion/models/frozen_audio_temporal_adapter.py
8. src/fusion/models/av_f1_temporal_audio_residual.py
9. src/fusion/data/wsm_av_fusion_datamodule.py
10. src/chimera_plugin.py
11. Chimera ML v0.2.4 installed/source behavior for training.optimizers and training.builders

## Confirmed Root Cause

Manager-confirmed Chimera ML v0.2.4 behavior:

    @OPTIMIZERS.register("adamw_optimizer")
    def adamw_optimizer(*, model, lr=..., weight_decay=..., **kwargs):
        return torch.optim.AdamW(
            model.parameters(),
            lr=...,
            weight_decay=...,
            **kwargs,
        )

and build_optimizer injects the model into the registered optimizer factory.

Do not patch or monkey-patch Chimera ML.

## Allowed Tracked Files

- src/common/optimizers.py
- src/chimera_plugin.py
- configs/wsm_mm_pd_dep_v1/fusion/03_f1_temporal_audio_residual.yaml
- docs/PROGRESS_EN.md

No other tracked file may be modified.

## 1. Register the Project Optimizer

Create:

    src/common/optimizers.py

Register exactly:

    wsm_trainable_adamw_optimizer

through:

    chimera_ml.core.registry.OPTIMIZERS

Factory signature must explicitly accept:

    model
    lr
    weight_decay

so Chimera smart injection passes the model.

Required semantics:

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

Then construct:

    torch.optim.AdamW(
        trainable_parameters,
        lr=float(lr),
        weight_decay=float(weight_decay),
        **kwargs,
    )

Requirements:

- preserve normal AdamW kwargs from config;
- raise ValueError if there are zero trainable parameters;
- do not mutate requires_grad;
- do not select parameters by fragile name substring inside the optimizer factory;
- do not special-case the audio model;
- do not include any requires_grad=false parameter;
- do not alter built-in adamw_optimizer.

This optimizer is a generic project utility for models containing intentionally frozen submodules.

## 2. Plugin Registration

Update:

    src/chimera_plugin.py

with explicit required import module:

    common.optimizers

It must register without a project-module warning.

After plugin registration, OPTIMIZERS must contain both:

    adamw_optimizer
    wsm_trainable_adamw_optimizer

Do not replace or shadow the built-in key.

## 3. Switch Only the Strong-Audio Config

Edit:

    configs/wsm_mm_pd_dep_v1/fusion/03_f1_temporal_audio_residual.yaml

Change only:

    optimizer.name

from:

    adamw_optimizer

to:

    wsm_trainable_adamw_optimizer

Preserve exactly:

    lr: 0.0001
    weight_decay: 0.01

Preserve every other accepted TASK-004I config field unchanged.

No scheduler.
No new optimizer hyperparameter.
No training change.

## 4. Compile and Registry Validation

Run:

    python3 -m py_compile \
      src/common/optimizers.py \
      src/chimera_plugin.py

Then:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import warnings
    from chimera_ml.core.registry import OPTIMIZERS
    import chimera_plugin

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chimera_plugin.register()

    project_failures = [
        str(x.message)
        for x in caught
        if "Failed to import" in str(x.message)
        and "common.optimizers" in str(x.message)
    ]
    assert not project_failures, project_failures

    assert "adamw_optimizer" in OPTIMIZERS.keys()
    assert "wsm_trainable_adamw_optimizer" in OPTIMIZERS.keys()

    print("trainable-only optimizer registry smoke passed")
    PY

## 5. Config Validation

Run:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml validate-config \
      --config-path configs/wsm_mm_pd_dep_v1/fusion/03_f1_temporal_audio_residual.yaml

Assert the resolved optimizer is:

    wsm_trainable_adamw_optimizer

with exactly:

    lr = 0.0001
    weight_decay = 0.01

## 6. Checkpoint SHA and Frozen-Audio Gate

Exact historical checkpoint:

    logs/wsm_audio_segment_wavlm_base_l9_pool4/multitask_audio_mamba_2026-08-19_14-12_audio_mamba_segment_model_wsm_audio_models-e0ce-006_ea8c8d77/checkpoints/epoch=4_dev_mean_score=0.7878.pt

Required SHA256:

    0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2

Recompute and assert exact equality.

Build the configured model and assert:

- all 105 audio parameter objects have requires_grad=false;
- frozen audio scalar parameter count remains 3,031,880;
- calling model.train() leaves:
      model.audio_adapter.audio_model.training == false;
- no incoming real A+V batch contains task_id/task_ids.

## 7. Exact Optimizer Firewall

Build the optimizer through the actual Chimera:

    chimera_ml.training.builders.build_optimizer

using the production config and production model.

Define:

    trainable_ids = {
        id(p)
        for p in model.parameters()
        if p.requires_grad
    }

    frozen_ids = {
        id(p)
        for p in model.parameters()
        if not p.requires_grad
    }

    optimizer_ids = {
        id(p)
        for group in optimizer.param_groups
        for p in group["params"]
    }

Require:

    optimizer_ids == trainable_ids
    optimizer_ids.isdisjoint(frozen_ids)

Also require:

    len(frozen_ids) == 105
    len(trainable_ids) == 22
    len(optimizer_ids) == 22

Required scalar parameter counts:

    frozen audio = 3,031,880
    trainable total = 285,442

Trainable parameters must still belong only to:

    video_projection
    shared_fusion
    residual_heads

Record per-block scalar counts:

    video_projection = 99,520
    shared_fusion = 111,744
    residual_heads = 74,178

If any optimizer object/count assertion fails:

    STOP BLOCKED.

Do not start training.

## 8. Data / Evaluation-Stream Gate

Build the accepted A+V DataModule through Chimera.

Require:

    train = 6325
    dev = 933
    test_none = 1364
    test_soft = 1208
    test_hard = 1014
    joined_total = 8622
    missing_audio = 0
    missing_video = 0

Require val_dataloader keys exactly:

    dev
    test_none
    test_soft
    test_hard

Do not iterate any Test loader in TASK-004I2.

It is acceptable to construct the mapping only to verify keys/counts.

## 9. DEV Frozen-Base Reproduction Gate

Before any training, iterate ONLY:

    dm.val_dataset

with a deterministic DEV loader using the accepted fusion collate.

Using the configured strong-audio model, collect only:

    output.aux["audio_base_logits"]

and compute:

    common.callbacks.wsm_segment_callback.compute_sparse_two_task_metrics

Required reproduction:

    depression UAR/MF1/Score =
      0.7480392157 / 0.7477975633 / 0.7479183895

    Parkinson UAR/MF1/Score =
      0.8209799862 / 0.8344907407 / 0.8277353635

    DEV Mean_Score =
      0.7878268765

Required tolerance on each task Score and Mean_Score:

    abs(delta) <= 0.0005

Do not threshold-tune.
Do not iterate Test.

If reproduction fails:

    STOP BLOCKED.

## 10. Real Production-Batch Forward/Loss/Backward Smoke

Use one real TRAIN batch of size 8.

Use a fresh configured model and build the repaired optimizer through actual Chimera.

Run:

    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    loss = wsm_masked_sparse_loss(output, batch)
    loss.backward()

Do NOT call optimizer.step().

Require:

- output.preds shape [8,2];
- finite scalar loss;
- all frozen audio grads are None;
- finite nonzero gradient exists in:
  - video_projection;
  - shared_fusion;
  - depression residual head;
  - Parkinson residual head;
- optimizer ID set still equals exact trainable ID set after backward;
- frozen audio model remains eval-only.

The smoke loss is structural only and must not be interpreted as model quality.

## 11. No-Training Boundary

TASK-004I2 MUST NOT run:

    chimera-ml train

It must not create a new MLflow training run.

It must not produce epoch metrics.

It must not iterate TEST_NONE/SOFT/HARD.

The goal is only to prove that the repaired production optimizer/config passes every pre-run firewall.

## Acceptance Criteria

Pass only if:

- src/common/optimizers.py exists and registers wsm_trainable_adamw_optimizer;
- built-in adamw_optimizer remains registered and untouched;
- project optimizer includes exactly requires_grad=true parameters;
- strong-audio config uses only wsm_trainable_adamw_optimizer as the intentional config change;
- config validates;
- checkpoint SHA matches;
- frozen audio remains 105 objects / 3,031,880 scalar parameters;
- trainable set remains 22 objects / 285,442 scalar parameters;
- optimizer object IDs equal trainable object IDs exactly;
- optimizer contains zero frozen audio parameter objects;
- trainable block scalar counts match expected values;
- DataModule/count/four-stream-key gate passes;
- DEV frozen-base reproduction passes;
- real TRAIN batch forward/loss/backward smoke passes;
- frozen audio gets no gradients;
- trainable residual branch gets finite nonzero gradients;
- no Test loader is iterated;
- no full training/MLflow run occurs;
- src/audio unchanged;
- src/video unchanged;
- accepted fusion model/DataModule source unchanged;
- git diff --check passes;
- tracked diff contains only:
  - src/common/optimizers.py
  - src/chimera_plugin.py
  - configs/wsm_mm_pd_dep_v1/fusion/03_f1_temporal_audio_residual.yaml
  - docs/PROGRESS_EN.md
- branch codex/task-004i2 committed and pushed;
- main/master untouched.

## Required PROGRESS_EN Update

Append TASK-004I2 evidence without erasing blocked TASK-004I.

Record:

- branch;
- implementation commit SHA;
- push result;
- confirmed Chimera v0.2.4 root cause;
- new optimizer registry key;
- exact filtering semantics;
- config optimizer-name correction;
- registry/config validation;
- checkpoint SHA verification;
- frozen/trainable object and scalar parameter counts;
- exact optimizer/trainable ID-set equality result;
- zero frozen-parameter optimizer result;
- DataModule counts and four stream keys;
- DEV base reproduction metrics/deltas;
- real TRAIN batch smoke loss labeled structural only;
- gradient checks;
- explicit no-training statement;
- explicit no-Test-iteration statement;
- confirmation src/audio unchanged;
- confirmation src/video unchanged;
- confirmation F2-temporal/RAMPS/text remain deferred;
- Stage 4 status;
- recommended next atomic task only.

If TASK-004I2 passes:

    recommended next = TASK-004I3 run the fixed seed-42 strong-temporal-audio F1 residual experiment using the repaired optimizer, with no architecture/config changes except the accepted optimizer key.

If TASK-004I2 is blocked:

    recommend only the narrow corrective task for the exact failed pre-run gate.

## Required Handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch codex/task-004i2;
- implementation commit SHA;
- pushed-to-origin status;
- main/master untouched;
- optimizer registry key;
- frozen/trainable/optimizer parameter-object counts;
- frozen/trainable scalar parameter counts;
- exact optimizer ID-set equality result;
- checkpoint SHA verification;
- reproduced DEV depression/Parkinson Scores and Mean_Score;
- real batch smoke loss;
- frozen audio no-gradient result;
- no full training;
- no Test iteration;
- F2-temporal/RAMPS/text deferred;
- src/audio unchanged;
- src/video unchanged.

Stop after this task.
