# TASK-004I4: Implement and Register the Zero-Initialized Strong-Audio F1 Residual Control

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-004i4

Do not run a full training experiment.
Do not create or edit a training config.
Do not modify the accepted random-init F1-temporal model.
Do not modify src/audio or src/video.
Do not modify the A+V DataModule.
Do not start F2-temporal, RAMPS, text, or description work.

## Goal

Implement one controlled variant that differs from the accepted TASK-004H/TASK-004I3 strong-audio F1 residual model ONLY by its residual-output initialization.

New registry key:

    wsm_av_f1_temporal_audio_residual_zero_init_model

At construction:

    residual_logits == 0 exactly
    preds == audio_base_logits exactly

for every valid sample, including rows where video is available.

This isolates whether a video residual that starts from the exact frozen-audio baseline can learn useful corrections without the random-initialization confound observed in TASK-004I3.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 3, 5, 6, 8, 9, 13, 14
3. docs/PLAN.md Stage 4
4. docs/PROGRESS_EN.md through MANAGER-DECISION-029
5. docs/NEXT_TASK_EN.md
6. src/fusion/models/av_f1_temporal_audio_residual.py
7. src/fusion/models/frozen_audio_temporal_adapter.py
8. src/common/optimizers.py
9. src/fusion/data/wsm_av_fusion_datamodule.py
10. src/common/loss/wsm_masked_sparse_loss.py
11. src/chimera_plugin.py

## Allowed Tracked Files

- src/fusion/models/av_f1_temporal_audio_residual_zero_init.py
- src/chimera_plugin.py
- docs/PROGRESS_EN.md

No other tracked file may be modified.

The accepted random-init file:

    src/fusion/models/av_f1_temporal_audio_residual.py

MUST remain unchanged.

## 1. Model Contract

Implement:

    src/fusion/models/av_f1_temporal_audio_residual_zero_init.py

Class:

    WSMAVF1TemporalAudioResidualZeroInitModel

The cleanest implementation is to subclass:

    WSMAVF1TemporalAudioResidualModel

Do not copy/rewrite the entire model unless inheritance cannot preserve the exact accepted contract.

Constructor signature/defaults must match the accepted base model:

    audio_checkpoint_path
    audio_feature_dim=768
    audio_hidden_dim=192
    video_feature_dim=512
    hidden_dim=192
    fusion_hidden_dim=192
    dropout=0.2
    num_tasks=2

After super().__init__(...), zero-initialize ONLY the final Linear layer of each residual head:

    final_linear.weight = 0
    final_linear.bias = 0

Do not zero:

- video_projection;
- shared_fusion;
- residual-head hidden Linear layers;
- LayerNorm parameters;
- any frozen audio parameter.

No new trainable parameter is allowed.

## 2. Exact Initialization Semantics

Immediately after model construction, before any optimizer step:

For every valid batch row:

    residual_logits == 0 exactly

and therefore:

    preds == audio_base_logits exactly

for both tasks.

This must hold even when video is available.

Required exact tensor test:

    torch.equal(output.aux["residual_logits"], torch.zeros_like(...))

and:

    torch.equal(output.preds, output.aux["audio_base_logits"])

Do not use an approximate tolerance for these two initialization invariants.

## 3. Registry

Register exactly:

    wsm_av_f1_temporal_audio_residual_zero_init_model

Update:

    src/chimera_plugin.py

with explicit required import:

    fusion.models.av_f1_temporal_audio_residual_zero_init

Preserve all existing registry keys.

Do not replace or alter:

    wsm_av_f1_temporal_audio_residual_model

## 4. Factory Semantics

The new factory must preserve the accepted context-aware checks:

- audio_feature_dim == 768 if supplied by data context;
- video_feature_dim == 512 if supplied;
- num_tasks == 2 if supplied;
- explicit audio_checkpoint_path required;
- same fixed hidden dimensions/defaults.

No guessed checkpoint path.

## 5. Parameter-Count Identity Gate

The zero-init control MUST have the same parameter topology/counts as the accepted random-init model.

Require:

    frozen audio = 105 objects / 3,031,880 scalars
    trainable = 22 objects / 286,530 scalars

Trainable scalar blocks:

    video_projection = 99,520
    shared_fusion = 111,744
    residual_heads = 75,266

No extra parameter object may be introduced.

## 6. Historical Checkpoint Gate

Use the accepted checkpoint:

    logs/wsm_audio_segment_wavlm_base_l9_pool4/multitask_audio_mamba_2026-08-19_14-12_audio_mamba_segment_model_wsm_audio_models-e0ce-006_ea8c8d77/checkpoints/epoch=4_dev_mean_score=0.7878.pt

Required SHA256:

    0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2

Recompute and assert exact match.

Frozen-audio semantics remain unchanged:

- all audio parameters requires_grad=false;
- parent model.train() leaves audio submodel in eval mode;
- no external task_id/task_ids input.

## 7. Exact DEV Initialization Reproduction

Build the accepted A+V DataModule.

Iterate ONLY:

    dm.val_dataset

Do not iterate any Test loader.

With a freshly constructed zero-init model, collect final:

    output.preds

not only aux base logits.

Compute canonical DEV metrics.

Because residual is exactly zero, final initialized model MUST reproduce the frozen historical audio result within the existing metric tolerance:

    depression Score = 0.7479183895
    Parkinson Score = 0.8277353635
    Mean_Score = 0.7878268765

Require:

    abs(delta) <= 0.0005

for the two task Scores and Mean_Score.

Also verify on every DEV batch:

    output.preds == output.aux["audio_base_logits"]

exactly.

No Test iteration.
No threshold tuning.

## 8. First-Backward Expected Gradient Gate

Use one deterministic real TRAIN batch of size 8 containing observed supervision for both tasks.

Use:

    wsm_trainable_adamw_optimizer

through actual Chimera build_optimizer.

Before any optimizer step:

    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    loss = wsm_masked_sparse_loss(output, batch)
    loss.backward()

Require:

- finite loss;
- frozen audio grads all None;
- final Linear of depression residual head gets finite nonzero gradient;
- final Linear of Parkinson residual head gets finite nonzero gradient;
- all present gradients finite.

Important expected zero-init behavior:

Because the final residual weights are exactly zero, upstream residual gradients on this FIRST backward may be zero.

Do NOT treat zero first-backward gradients in:

    video_projection
    shared_fusion
    earlier residual-head layers

as a failure.

Record them explicitly.

## 9. Exactly-One-Step Wake-Up Gate

After the first backward, execute exactly ONE bounded optimizer step:

    optimizer.step()

This is a smoke-only in-memory step, not training.

Then:

    optimizer.zero_grad(set_to_none=True)

Run a second forward/loss/backward on a second deterministic real TRAIN batch that contains observed supervision for both tasks.

Require after this second backward:

- finite scalar loss;
- frozen audio grads all None;
- video_projection finite nonzero gradient;
- shared_fusion finite nonzero gradient;
- depression residual branch finite nonzero gradient;
- Parkinson residual branch finite nonzero gradient;
- all present trainable gradients finite.

Also verify:

- residual_logits are no longer required to be zero after the one optimizer step;
- frozen audio remains eval-only;
- optimizer_ids still equal trainable_ids;
- optimizer/frozen intersection remains empty.

Do NOT execute a second optimizer step.

## 10. Synthetic/Availability Invariants

Verify the inherited accepted semantics remain unchanged:

- output [B,2];
- no external task_id/task_ids;
- audio unavailable raises ValueError;
- audio-only row falls back to exact frozen audio base because video residual is masked out;
- masked audio padding invariance;
- masked video padding invariance;
- unavailable video raw-value invariance.

The only intended behavioral difference from the accepted random-init model is residual-output initialization at construction.

## 11. No-Training Boundary

TASK-004I4 MUST NOT run:

    chimera-ml train

It MUST NOT:

- create a new production training config;
- create an MLflow training run;
- run epoch loops;
- iterate TEST_NONE/SOFT/HARD;
- modify the old random-init model;
- start F2-temporal;
- start RAMPS/text/description.

The single in-memory optimizer.step() required by the wake-up smoke is explicitly allowed and must not save a checkpoint or training artifact.

## Required Verification

Compile:

    python3 -m py_compile \
      src/fusion/models/av_f1_temporal_audio_residual_zero_init.py \
      src/chimera_plugin.py

Registry smoke must verify both keys exist:

    wsm_av_f1_temporal_audio_residual_model
    wsm_av_f1_temporal_audio_residual_zero_init_model

Then run:

1. checkpoint SHA gate;
2. parameter-count identity gate;
3. exact zero residual / preds==base initialization gate;
4. canonical DEV initialized-final reproduction;
5. first-backward gradient gate;
6. exactly-one-step wake-up gate;
7. inherited mask/availability checks.

Finally:

    git diff --check
    git diff -- src/audio
    git diff -- src/video
    git diff -- src/fusion/models/av_f1_temporal_audio_residual.py
    git diff -- src/fusion/data
    git status --short

Before commit inspect only:

    git diff -- \
      src/fusion/models/av_f1_temporal_audio_residual_zero_init.py \
      src/chimera_plugin.py \
      docs/PROGRESS_EN.md

## Acceptance Criteria

Pass only if:

- old random-init model file is unchanged;
- new zero-init registry key exists;
- no extra trainable/frozen parameters introduced;
- exact parameter counts match accepted model;
- checkpoint SHA exact;
- residual final Linear weights/biases are zero at construction;
- residual_logits exactly zero before any step;
- preds exactly equal audio_base_logits before any step;
- canonical DEV initialized final output reproduces frozen audio;
- first backward gives nonzero gradients to both residual output Linear layers;
- exactly one optimizer step is performed;
- second backward produces finite nonzero gradients in video_projection/shared_fusion/both residual branches;
- frozen audio never receives gradients and remains eval-only;
- optimizer contains only trainable parameters;
- inherited mask/availability invariants pass;
- no Test iteration;
- no full training/MLflow training run;
- src/audio unchanged;
- src/video unchanged;
- tracked diff contains only the three allowed paths;
- branch codex/task-004i4 committed and pushed;
- main/master untouched.

## Required PROGRESS_EN Update

Record:

- branch;
- implementation commit SHA;
- push result;
- registry key;
- exact zero-init operation;
- proof old random-init model unchanged;
- checkpoint SHA;
- parameter object/scalar counts;
- zero residual / preds==base exact checks;
- initialized-final canonical DEV metrics;
- first-backward gradient results, including expected upstream zeros;
- exactly-one-step optimizer smoke;
- second-backward wake-up gradient results;
- optimizer firewall result;
- frozen-audio integrity;
- availability/mask checks;
- explicit no-Test statement;
- explicit no-full-training statement;
- confirmation F2-temporal/RAMPS/text remain deferred;
- src/audio unchanged;
- src/video unchanged;
- Stage 4 status;
- recommended next atomic task only.

If all gates pass:

    recommended next = TASK-004I5 run one fixed seed-42 zero-init strong-temporal-audio F1 residual experiment.

If blocked:

    recommend only the narrow correction for the exact failed gate.

## Required Handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch codex/task-004i4;
- implementation commit SHA;
- pushed-to-origin status;
- main/master untouched;
- new registry key;
- exact parameter counts;
- zero residual / preds==base initialization result;
- initialized-final DEV depression/Parkinson Scores and Mean_Score;
- first-backward output-layer gradient result;
- second-backward video/shared/both-head gradient result after exactly one optimizer step;
- frozen-audio no-gradient result;
- no Test iteration;
- no full training;
- old random-init model unchanged;
- F2-temporal/RAMPS/text deferred;
- src/audio unchanged;
- src/video unchanged.

Stop after this task.
