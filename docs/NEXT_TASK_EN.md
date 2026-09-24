# TASK-004I2B: Re-Run the Strong-Audio Pre-Training Firewall with Corrected Parameter Counts

## Role

You are the implementing Codex. Execute only this evidence-only corrective task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-004i2b

Do not modify source code.
Do not modify any config.
Do not modify Chimera ML.
Do not modify src/audio or src/video.
Do not start the real 30-epoch training run.
Do not iterate any Test loader.
Do not start F2-temporal, RAMPS, text, or description work.

## Goal

Re-run every strong-temporal-audio pre-training firewall after correcting the manager-side parameter-count arithmetic error.

The accepted production implementation is already on main:

    optimizer = wsm_trainable_adamw_optimizer
    model = wsm_av_f1_temporal_audio_residual_model
    config = configs/wsm_mm_pd_dep_v1/fusion/03_f1_temporal_audio_residual.yaml

Do not change any of them.

This task only verifies that the production optimizer/config/model now satisfy all gates required before the real TASK-004I3 training run.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 3, 5, 6, 8, 9, 13, 14
3. docs/PLAN.md Stage 4
4. docs/PROGRESS_EN.md through MANAGER-DECISION-027
5. docs/NEXT_TASK_EN.md
6. src/common/optimizers.py
7. src/chimera_plugin.py
8. configs/wsm_mm_pd_dep_v1/fusion/03_f1_temporal_audio_residual.yaml
9. src/fusion/models/frozen_audio_temporal_adapter.py
10. src/fusion/models/av_f1_temporal_audio_residual.py
11. src/fusion/data/wsm_av_fusion_datamodule.py
12. src/common/callbacks/wsm_segment_callback.py
13. src/common/loss/wsm_masked_sparse_loss.py

## Allowed Tracked Files

- docs/PROGRESS_EN.md

No other tracked file may be modified.

If any source/config file differs from origin/main at task start, stop and report the dirty/conflicting path instead of overwriting it.

## Corrected Fixed Parameter Counts

The accepted residual head is:

    LayerNorm(192)
    Linear(192,192)
    GELU
    Dropout
    Linear(192,1)

Correct scalar arithmetic:

    one LayerNorm(192) = 384
    one Linear(192,192) = 37,056
    one Linear(192,1) = 193
    one residual head = 37,633
    two residual heads = 75,266

Correct production scalar counts:

    frozen audio = 3,031,880
    video_projection = 99,520
    shared_fusion = 111,744
    residual_heads = 75,266
    total trainable = 286,530

Correct parameter-object counts:

    frozen = 105
    trainable = 22
    optimizer = 22

The earlier manager requirements:

    residual_heads = 74,178
    total trainable = 285,442

are revoked and MUST NOT be used.

Do not modify the model to reduce its parameter count.

## 1. Repository / Scope Gate

Start from current origin/main.

Verify:

    git status --short

is clean before task work.

Verify these files are unchanged during the task:

    src/common/optimizers.py
    src/chimera_plugin.py
    configs/wsm_mm_pd_dep_v1/fusion/03_f1_temporal_audio_residual.yaml
    src/fusion/models/frozen_audio_temporal_adapter.py
    src/fusion/models/av_f1_temporal_audio_residual.py
    src/fusion/data/wsm_av_fusion_datamodule.py
    src/audio
    src/video

Only docs/PROGRESS_EN.md may become tracked-dirty.

## 2. Registry / Config Gate

Run:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import warnings
    from chimera_ml.core.registry import OPTIMIZERS, MODELS, DATAMODULES, LOSSES
    import chimera_plugin

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chimera_plugin.register()

    project_failures = [
        str(x.message)
        for x in caught
        if "Failed to import" in str(x.message)
    ]
    assert not project_failures, project_failures

    assert "adamw_optimizer" in OPTIMIZERS.keys()
    assert "wsm_trainable_adamw_optimizer" in OPTIMIZERS.keys()
    assert "wsm_av_f1_temporal_audio_residual_model" in MODELS.keys()
    assert "wsm_av_fusion_datamodule" in DATAMODULES.keys()
    assert "wsm_masked_sparse_loss" in LOSSES.keys()

    print("registry gate passed")
    PY

Run:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml validate-config \
      --config-path configs/wsm_mm_pd_dep_v1/fusion/03_f1_temporal_audio_residual.yaml

Assert the config still resolves:

    optimizer.name = wsm_trainable_adamw_optimizer
    lr = 0.0001
    weight_decay = 0.01
    seed = 42
    batch_size = 8
    epochs = 30
    checkpoint monitor = dev/mean_score
    early stopping monitor = dev/mean_score
    mode = max

No config edit is permitted.

## 3. Historical Checkpoint Gate

Exact checkpoint:

    logs/wsm_audio_segment_wavlm_base_l9_pool4/multitask_audio_mamba_2026-08-19_14-12_audio_mamba_segment_model_wsm_audio_models-e0ce-006_ea8c8d77/checkpoints/epoch=4_dev_mean_score=0.7878.pt

Required SHA256:

    0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2

Recompute and assert exact equality.

## 4. Production Model Parameter Gate

Build the configured model through actual Chimera.

Call:

    model.train()

and require:

    model.audio_adapter.audio_model.training == false

Collect named parameters.

Require:

    frozen object count = 105
    trainable object count = 22

Require scalar counts:

    frozen audio = 3,031,880
    video_projection = 99,520
    shared_fusion = 111,744
    residual_heads = 75,266
    total trainable = 286,530

Require every trainable parameter name to belong to exactly one of:

    video_projection
    shared_fusion
    residual_heads

Require every parameter under:

    audio_adapter.audio_model

to have:

    requires_grad == false

If any corrected count fails:

    STOP BLOCKED.

Do not modify the model.

## 5. Exact Production Optimizer-ID Firewall

Build the optimizer through:

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

Require exactly:

    len(trainable_ids) == 22
    len(frozen_ids) == 105
    len(optimizer_ids) == 22

    optimizer_ids == trainable_ids
    optimizer_ids.isdisjoint(frozen_ids)

Also verify the optimizer instance is:

    torch.optim.AdamW

and its effective hyperparameters are:

    lr = 0.0001
    weight_decay = 0.01

This is the decisive repair gate.

If optimizer_ids differs from trainable_ids in either direction:

    STOP BLOCKED.

## 6. Data / Protocol-Key Gate

Build the production:

    WSMAVFusionDataModule

through the configured Chimera path.

Require:

    train = 6325
    dev = 933
    test_none = 1364
    test_soft = 1208
    test_hard = 1014
    joined_total = 8622
    missing_audio = 0
    missing_video = 0

Construct the validation-loader mapping only far enough to verify keys:

    dev
    test_none
    test_soft
    test_hard

Do NOT iterate:

    test_none
    test_soft
    test_hard

in this task.

## 7. Canonical DEV Frozen-Base Reproduction

Iterate ONLY the canonical DEV dataset:

    dm.val_dataset

with a deterministic DataLoader and the accepted fusion collate.

Use the production configured model.

Collect:

    output.aux["audio_base_logits"]
    targets
    observed_mask

Compute metrics through:

    common.callbacks.wsm_segment_callback.compute_sparse_two_task_metrics

Required reference:

    depression UAR/MF1/Score =
      0.7480392157 / 0.7477975633 / 0.7479183895

    Parkinson UAR/MF1/Score =
      0.8209799862 / 0.8344907407 / 0.8277353635

    DEV Mean_Score =
      0.7878268765

Acceptance:

    abs(delta) <= 0.0005

for depression Score, Parkinson Score, and Mean_Score.

Do not tune a threshold.

Do not iterate Test.

If reproduction fails:

    STOP BLOCKED.

## 8. Real TRAIN Batch Forward/Loss/Backward Gate

Use one real production TRAIN batch with batch size 8.

Use a fresh configured model and build its optimizer through the actual production Chimera path.

Before backward assert again:

    optimizer_ids == trainable_ids
    optimizer_ids.isdisjoint(frozen_ids)

Run only:

    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    loss = wsm_masked_sparse_loss(output, batch)
    loss.backward()

Do NOT call:

    optimizer.step()

Require:

    output.preds shape == [8,2]
    loss finite scalar

Frozen audio:

- every frozen audio parameter grad is None;
- audio model remains eval-only.

Trainable branch:

- video_projection has finite nonzero gradient;
- shared_fusion has finite nonzero gradient;
- depression residual head has finite nonzero gradient;
- Parkinson residual head has finite nonzero gradient;
- all present trainable gradients are finite.

After backward re-check:

    optimizer_ids == trainable_ids
    optimizer_ids.isdisjoint(frozen_ids)

Record smoke loss as structural only.

## 9. No-Training / No-Test Boundary

TASK-004I2B MUST NOT run:

    chimera-ml train

It MUST NOT:

- call optimizer.step();
- create a new training MLflow run;
- produce epoch metrics;
- iterate TEST_NONE;
- iterate TEST_SOFT;
- iterate TEST_HARD;
- modify source/config;
- begin TASK-004I3.

## Acceptance Criteria

Pass only if all are true:

- repository starts clean;
- no source/config change occurs;
- registry/config gates pass;
- checkpoint SHA exact;
- frozen audio = 105 objects / 3,031,880 scalars;
- trainable = 22 objects / 286,530 scalars;
- video_projection = 99,520;
- shared_fusion = 111,744;
- residual_heads = 75,266;
- optimizer = exactly 22 objects;
- optimizer_ids == trainable_ids;
- optimizer/frozen intersection empty;
- AdamW effective lr/weight_decay correct;
- dataset/protocol counts and keys correct;
- canonical DEV frozen-base reproduction passes;
- real TRAIN batch forward/loss/backward passes;
- frozen audio gradients all None;
- trainable gradients finite/nonzero in all required branches;
- no optimizer step;
- no full training;
- no Test iteration;
- src/audio unchanged;
- src/video unchanged;
- git diff --check passes;
- tracked diff against origin/main contains only docs/PROGRESS_EN.md;
- branch codex/task-004i2b committed and pushed;
- main/master untouched.

## Required PROGRESS_EN Update

Append TASK-004I2B evidence without erasing TASK-004I/TASK-004I2 history.

Record:

- branch;
- evidence commit SHA;
- push result;
- corrected parameter-count derivation;
- checkpoint SHA verification;
- registry/config validation;
- frozen object/scalar counts;
- trainable object/scalar counts;
- per-block scalar counts;
- optimizer object count;
- exact optimizer_ids == trainable_ids result;
- zero frozen optimizer intersection;
- optimizer effective lr/weight_decay;
- DataModule counts and protocol keys;
- DEV base reproduction UAR/MF1/Score/Mean_Score and deltas;
- real TRAIN batch smoke loss labeled structural only;
- gradient checks;
- explicit no optimizer.step statement;
- explicit no full training/MLflow run statement;
- explicit no Test iteration statement;
- confirmation no source/config changes;
- src/audio unchanged;
- src/video unchanged;
- confirmation F2-temporal/RAMPS/text remain deferred;
- Stage 4 status;
- recommended next atomic step only.

If every gate passes:

    recommended next = TASK-004I3 run the fixed seed-42 strong-temporal-audio F1 residual training experiment with the already accepted production config and optimizer.

If any gate fails:

    recommend only a narrow correction for that exact failed gate.

## Required Handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch codex/task-004i2b;
- evidence commit SHA;
- pushed-to-origin status;
- main/master untouched;
- corrected residual/trainable counts;
- frozen/trainable/optimizer object counts;
- optimizer_ids == trainable_ids result;
- zero frozen optimizer intersection;
- checkpoint SHA verification;
- reproduced DEV depression/Parkinson Scores and Mean_Score;
- real TRAIN batch smoke loss;
- frozen-audio no-gradient result;
- no optimizer.step;
- no full training;
- no Test iteration;
- no source/config changes;
- F2-temporal/RAMPS/text deferred;
- src/audio unchanged;
- src/video unchanged.

Stop after this task.
