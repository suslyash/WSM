# TASK-004H: Implement the Frozen Strong-Temporal-Audio F1 Residual A+V Contract and Reproduce the Historical Audio DEV Reference

## Role

You are the implementing Codex. Execute only this task, update docs/PROGRESS_EN.md, commit on the required task branch, push to origin, then stop. Follow AGENTS.md.

Required branch:

    codex/task-004h

This task supersedes the previously assigned TASK-005A before execution.

Do not start RAMPS.
Do not create a training config.
Do not run a new fusion training experiment.
Do not retrain or modify the audio baseline.
Do not modify src/audio.
Do not modify the accepted A+V DataModule.
Do not modify existing F0/F1/F2 models.
Do not start text/description work.

## Goal

Remove the main confound in the current fusion ladder.

The historical audio reference uses the full frozen temporal audio model:

    WavLM-base-plus layer 9 / pool 4
    temporal Transformer
    hidden=192
    layers=3
    heads=4
    sequence_steps=128

whereas F0/F1/F2 used only pooled:

    audio_cls [768]

TASK-004H must:

1. uniquely locate and strictly verify the exact historical DEV-selected audio checkpoint;
2. wrap that frozen checkpoint in a fusion-side temporal-audio adapter without editing src/audio;
3. reproduce the historical audio DEV result on the canonical A+V DEV rows;
4. implement/register the first strong-audio fusion contract:
       frozen strong audio base logits
       + shared video-conditioned residual fusion;
5. pass synthetic and real forward/loss/backward smoke;
6. perform NO fusion training yet.

The resulting model is the F1-style strong-audio ablation.

## Research Question

This task prepares the controlled comparison:

    strong temporal audio
        vs
    the same frozen strong temporal audio + video residual fusion

The later fixed training run will test whether video adds value without discarding the strong temporal audio representation.

## Required Reading

1. AGENTS.md
2. docs/PROJECT_REQUIREMENTS.md Sections 2, 5, 6, 7, 8, 9, 10, 13, 14
3. docs/PLAN.md Stage 4
4. docs/PROGRESS_EN.md through MANAGER-DECISION-024
5. docs/NEXT_TASK_EN.md
6. configs/wsm_mm_pd_dep_v1/audio/00_frozen_baseline.yaml
7. src/audio/models/audio_mamba_segment.py — READ ONLY
8. src/fusion/models/av_sync_mamba_segment.py — for the frozen audio model's public temporal blocks only
9. src/fusion/data/wsm_av_fusion_datamodule.py
10. src/fusion/models/av_f1_shared_mtl.py
11. src/common/callbacks/wsm_segment_callback.py
12. src/common/loss/wsm_masked_sparse_loss.py
13. src/chimera_plugin.py

## Allowed Tracked Files

- src/fusion/models/frozen_audio_temporal_adapter.py
- src/fusion/models/av_f1_temporal_audio_residual.py
- src/fusion/models/__init__.py
- src/chimera_plugin.py
- docs/PROGRESS_EN.md

No other tracked file may be modified.

## Registry Key

Register exactly:

    wsm_av_f1_temporal_audio_residual_model

The adapter itself does not need a registry key.

Do not alter existing model registry keys.

## 1. Locate the Exact Historical Audio Checkpoint

Historical accepted reference:

    run = wsm_audio_models-e0ce-006
    selected epoch = 4
    DEV Mean_Score = 0.787827
    DEV depression Score = 0.747918
    DEV Parkinson Score = 0.827735

The exact checkpoint path is not currently recorded in active project docs.

Before implementing the adapter:

1. search local durable run artifacts under the repository/log roots;
2. locate the historical run using the run identifier and/or its recorded selected DEV evidence;
3. identify the epoch-4 selected checkpoint;
4. inspect its checkpoint payload/state-dict structure;
5. compute SHA256 of the checkpoint;
6. verify the checkpoint can be strictly loaded into the exact historical AudioMambaSegmentModel architecture.

Requirements:

- exactly one authoritative checkpoint must be selected;
- do not guess a path from naming alone;
- do not choose another epoch;
- do not retrain audio;
- do not substitute the canonical config's random initialization;
- do not download an unrelated checkpoint;
- do not use Test metrics to choose among candidates.

If the exact selected historical checkpoint cannot be uniquely located:

    STOP BLOCKED.

Update PROGRESS_EN.md with all candidate evidence and do not fabricate the adapter reproduction result.

## 2. Exact Frozen Audio Architecture

Instantiate the historical audio model exactly as:

    AudioMambaSegmentModel(
        audio_feature_dim=768,
        num_tasks=2,
        num_classes=2,
        hidden_dim=192,
        num_layers=3,
        num_heads=4,
        ff_mult=4,
        dropout=0.25,
        encoder_type="transformer",
        sequence_steps=128,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_required=True,
    )

This lives in src/audio and MUST remain unmodified.

Checkpoint loading:

- inspect the actual payload structure first;
- use strict state loading after only the minimal verified extraction/prefix normalization required by that exact checkpoint format;
- missing keys must be empty;
- unexpected keys must be empty;
- record the exact state key/prefix handling used;
- never silently partial-load.

## 3. FrozenAudioTemporalAdapter

Implement:

    src/fusion/models/frozen_audio_temporal_adapter.py

Class:

    FrozenAudioTemporalAdapter

The adapter wraps the exact frozen AudioMambaSegmentModel.

Constructor must require:

    checkpoint_path

and may expose fixed architecture parameters only for validation/provenance.

### Frozen semantics

After checkpoint load:

- all audio-model parameters:
      requires_grad = false
- audio model:
      eval()
- forward:
      under torch.no_grad() or torch.inference_mode()
- audio model is never placed in an optimizer.

Override/guard train-mode propagation as needed so that calling parent fusion_model.train() NEVER turns the frozen audio submodel's dropout/Transformer into training behavior.

After:

    fusion_model.train()

the frozen audio model must still report:

    training == False

### Input

Consume from the accepted A+V Batch:

    batch.inputs["audio"]  -> [B,Ta,768]
    batch.get_masks("audio_mask") -> bool [B,Ta]

Do NOT read task_id/task_ids from the incoming batch.

### Internal task conditioning

The historical model is task-conditioned.

For each input batch, run the frozen audio model internally twice:

    fixed task index 0 = depression
    fixed task index 1 = parkinson

Construct the fixed internal task IDs yourself.

They are architecture-internal constants, not dataset/model inputs.

For each task collect:

    legacy class logits -> [B,2]
    segment feature     -> [B,192]

Stack:

    legacy_class_logits -> [B,2,2]
    task_features       -> [B,2,192]

dimensions of legacy_class_logits:

    batch, task, legacy binary class

Convert the historical two-class disease output into one independent binary logit per task:

    binary_logit_t =
        legacy_class_logits[:,t,1]
        - legacy_class_logits[:,t,0]

Return:

    base_logits   -> [B,2]
    task_features -> [B,2,192]
    legacy_class_logits -> [B,2,2]

This is an equivalent binary decision representation of each historical disease-specific two-class head.

Do not produce a healthy/depression/Parkinson 3-class softmax.

## 4. Frozen Audio DEV Reproduction Gate

Before the new fusion contract can be accepted, reproduce the historical audio result on the accepted canonical A+V DEV dataset.

Use:

    WSMAVFusionDataModule.val_dataset

directly.

Do NOT call val_dataloader() for this reproduction because it also exposes Test streams.

Build a deterministic DEV DataLoader from val_dataset using the accepted fusion collate.

Run ONLY the frozen adapter base_logits.

Collect:

    logits [933,2]
    targets [933,2]
    observed_mask [933,2]

Compute metrics with:

    common.callbacks.wsm_segment_callback.compute_sparse_two_task_metrics

using:

    prefix="dev"

Required reproduction reference:

    dev/depression/score = 0.747918
    dev/parkinson/score  = 0.827735
    dev/mean_score       = 0.787827

Acceptance tolerance for each of these three rounded historical values:

    absolute delta <= 0.0005

Also record reproduced UAR/MF1 values.

If any required Score/Mean_Score differs by more than 0.0005:

    STOP BLOCKED.

Do not compensate with threshold tuning.
Do not inspect Test.
Do not continue to a training run.

A model-contract implementation may remain on the branch for audit, but the handoff must report the reproduction blocker and the next task must not be proposed as a training run.

## 5. Strong-Audio F1 Residual Fusion Model

Implement:

    src/fusion/models/av_f1_temporal_audio_residual.py

Class:

    WSMAVF1TemporalAudioResidualModel

Registry:

    wsm_av_f1_temporal_audio_residual_model

### Fixed constructor defaults

Use:

    audio_feature_dim=768
    audio_hidden_dim=192
    video_feature_dim=512
    hidden_dim=192
    fusion_hidden_dim=192
    dropout=0.2
    num_tasks=2

Require:

    num_tasks == 2
    audio_hidden_dim == 192

Also require:

    audio_checkpoint_path

No default guessed checkpoint path.

### Audio branch

Use FrozenAudioTemporalAdapter.

Its outputs are frozen:

    audio_base_logits   [B,2]
    audio_task_features [B,2,192]

No audio parameter is trainable.

### Video branch

Use the same pooled video input philosophy as F1:

    video [B,Tv,512]
    video_mask [B,Tv]

Masked mean:

    v_raw [B,512]

Projection:

    LayerNorm(512)
    Linear(512,192)
    GELU
    Dropout(0.2)

Output:

    video_features [B,192]

### Availability

This controlled ablation REQUIRES audio to be available.

For every row:

    modality_available[:,0] must be true

If any audio-unavailable row is passed:

    raise ValueError

Video may be unavailable.

Available video requires at least one true video_mask position.

Unavailable video may have all-false video_mask.

For video-unavailable rows:

    effective_video_features = exact zero
    residual_logits = exact zero
    final logits = exact frozen audio_base_logits

Changing raw unavailable-video values must not change output.

### Shared F1-style residual fusion

The historical audio representation is task-conditioned, so preserve that information.

For each task t:

    joined_t =
        concat(
            audio_task_features[:,t],
            effective_video_features
        )                           # [B,384]

Use ONE shared fusion trunk for both tasks:

    LayerNorm(384)
    Linear(384,192)
    GELU
    Dropout(0.2)
    Linear(192,192)
    GELU
    Dropout(0.2)

Apply the same shared_fusion module independently to joined_depression and joined_parkinson.

Result:

    task_fused_features [B,2,192]

Use two independent residual heads:

    LayerNorm(192)
    Linear(192,192)
    GELU
    Dropout(0.2)
    Linear(192,1)

Stack:

    residual_logits [B,2]

Final:

    preds =
        audio_base_logits
        + video_available * residual_logits

where video_available is broadcast to both task logits.

Thus:

- video available:
      trainable video-conditioned residual can modify the strong audio base;
- video unavailable:
      exact fallback to the historical frozen audio base logits.

No sigmoid is applied in forward.

## 6. Important Boundary

TASK-004H MUST NOT add:

- audio fine-tuning;
- src/audio edits;
- new WavLM extraction;
- a new audio architecture;
- external task_id/task_ids input;
- pseudo-labeling;
- RAMPS;
- directed F2 relation experts;
- flow matching;
- PAGB;
- contrastive loss;
- text/description;
- Test-based tuning.

This is F1-temporal only.

The analogous F2-temporal directed model comes later, after the fixed F1-temporal run.

## 7. ModelOutput Contract

Return:

    preds -> [B,2]

aux must contain at least:

    audio_base_logits            -> [B,2]
    legacy_audio_class_logits    -> [B,2,2]
    audio_task_features          -> [B,2,192]
    video_features               -> [B,192]
    effective_video_features     -> [B,192]
    task_fused_features          -> [B,2,192]
    residual_logits              -> [B,2]
    task_logits = {
        "depression": preds[:,0],
        "parkinson": preds[:,1],
    }

Do not expose task IDs as batch/model inputs.

## 8. Context-Aware Factory

Factory must:

- read data.audio_feature_dim if available and require 768;
- read data.video_feature_dim if available and require 512;
- enforce data.num_tasks == 2 if present;
- require explicit audio_checkpoint_path from config/caller;
- default common hidden dimensions as above.

Update src/chimera_plugin.py with explicit required import:

    fusion.models.av_f1_temporal_audio_residual

Do not hide a project import failure as an optional warning.

## 9. Synthetic Contract Smoke

Use the actual discovered frozen audio checkpoint.

Construct a synthetic Batch with:

    audio [4, >=8, 768]
    audio_mask variable
    video [4, <=7, 512]
    video_mask variable
    modality_available:
      at least:
        both available rows
        one audio-only row

No audio-unavailable row is valid in this ablation.

Verify:

- preds [4,2];
- audio_base_logits [4,2];
- legacy_audio_class_logits [4,2,2];
- audio_task_features [4,2,192];
- video_features [4,192];
- task_fused_features [4,2,192];
- residual_logits [4,2];
- no task_id/task_ids in incoming batch;
- final equation exactly matches:
      base + video_available * residual;
- audio-only row preds exactly equal audio_base_logits;
- masked audio padding invariance;
- masked video padding invariance;
- unavailable video raw-value invariance;
- audio-unavailable row raises ValueError.

## 10. Sparse Loss / Gradient Smoke

Use:

    wsm_masked_sparse_loss

with sparse NaN targets.

Verify:

- finite scalar loss;
- backward succeeds;
- every frozen audio parameter has:
      requires_grad == false
      grad is None
- video projection receives finite nonzero gradient;
- shared_fusion receives finite nonzero gradient;
- both residual task heads receive finite nonzero gradient when both tasks have observed labels in the synthetic batch.

Also assert after:

    model.train()

that:

    model.audio_adapter.audio_model.training == False

and the frozen audio parameters remain non-trainable.

## 11. Real A+V Batch Smoke

Build the accepted:

    WSMAVFusionDataModule

Use one real TRAIN batch.

Verify:

- inputs contain temporal audio and video;
- incoming inputs do NOT contain task_id/task_ids;
- model output [B,2];
- finite sparse loss;
- backward succeeds;
- frozen audio still has no gradients;
- trainable video/shared/residual branches get finite gradients.

No full training.

## 12. Required Verification Commands

Compile:

    python3 -m py_compile \
      src/fusion/models/frozen_audio_temporal_adapter.py \
      src/fusion/models/av_f1_temporal_audio_residual.py \
      src/fusion/models/__init__.py \
      src/chimera_plugin.py

Registry smoke:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python - <<'PY'
    import warnings
    from chimera_ml.core.registry import MODELS
    import chimera_plugin

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chimera_plugin.register()

    failures = [
        str(x.message)
        for x in caught
        if "Failed to import" in str(x.message)
        and "fusion.models.av_f1_temporal_audio_residual" in str(x.message)
    ]

    assert not failures, failures
    assert "audio_mamba_segment_model" in MODELS.keys()
    assert "wsm_av_f1_shared_mtl_model" in MODELS.keys()
    assert "wsm_av_f2_task_aware_directed_model" in MODELS.keys()
    assert "wsm_av_f1_temporal_audio_residual_model" in MODELS.keys()

    print("strong temporal-audio F1 registry smoke passed")
    PY

Then run:

1. exact checkpoint discovery/strict-load audit;
2. frozen audio DEV reproduction gate;
3. synthetic contract/loss/backward smoke;
4. real A+V DataModule loss/backward smoke.

Finally:

    git diff --check
    git diff -- src/audio
    git diff -- src/video
    git diff -- src/fusion/data
    git diff -- src/fusion/models/av_f0_gated_late.py
    git diff -- src/fusion/models/av_f1_shared_mtl.py
    git diff -- src/fusion/models/av_f2_task_aware_directed.py
    git status --short

Before commit inspect only:

    git diff -- \
      src/fusion/models/frozen_audio_temporal_adapter.py \
      src/fusion/models/av_f1_temporal_audio_residual.py \
      src/fusion/models/__init__.py \
      src/chimera_plugin.py \
      docs/PROGRESS_EN.md

After commit/push:

    git status --short
    git rev-parse --abbrev-ref HEAD
    git rev-parse HEAD
    git diff --stat origin/main...HEAD
    git diff --name-only origin/main...HEAD

## Acceptance Criteria

The task passes only if:

- exact historical run/checkpoint is uniquely located;
- selected epoch is 4;
- checkpoint SHA256 is recorded;
- checkpoint state structure is documented;
- strict audio state load has zero missing/unexpected keys;
- src/audio remains byte-for-byte unmodified by git diff;
- frozen audio adapter has no trainable parameters;
- parent model train mode never enables audio dropout/training behavior;
- adapter internally evaluates both fixed disease task indices;
- no external task_id/task_ids are consumed;
- base binary logits equal class1_logit-class0_logit;
- canonical DEV reproduction is within 0.0005 of:
  - depression Score 0.747918
  - Parkinson Score 0.827735
  - Mean_Score 0.787827
- F1-temporal registry key exists;
- final output is [B,2];
- audio-only fallback equals frozen audio base logits exactly;
- audio/video mask invariance checks pass;
- sparse forward/loss/backward passes;
- frozen audio receives no gradients;
- video/shared/residual branches receive finite nonzero gradients;
- real A+V batch smoke passes;
- no config or training run is created;
- no Test stream is iterated or metric computed in this task;
- no RAMPS work is performed;
- text/description remains deferred;
- tracked diff contains only the five allowed paths;
- branch codex/task-004h committed and pushed;
- main/master untouched.

## Required PROGRESS_EN Update

Append TASK-004H evidence without erasing prior records.

Record:

- branch;
- implementation commit SHA;
- push result;
- exact historical audio run identifier;
- exact selected checkpoint path;
- checkpoint SHA256;
- checkpoint payload/state key/prefix structure;
- strict-load result;
- frozen audio architecture;
- DEV reproduction UAR/MF1/Score per task and Mean_Score;
- deltas to historical rounded reference;
- adapter internal task-conditioning contract;
- binary-logit conversion equation;
- F1-temporal residual equation;
- output/aux shapes;
- synthetic invariance results;
- synthetic loss/backward result;
- frozen/no-gradient proof;
- real DataModule smoke result;
- no external task_id verification;
- explicit no-Test statement;
- explicit no-training statement;
- confirmation TASK-005A/RAMPS remains deferred;
- confirmation text/description deferred;
- src/audio unchanged;
- src/video unchanged;
- Stage 4 reopened-ablation status;
- recommended next atomic step only.

If DEV reproduction passes:

    recommended next = fixed seed-42 F1-temporal residual training run

If DEV reproduction fails:

    recommended next = narrow checkpoint/reproduction corrective task only

Do not propose F2-temporal or RAMPS before a successful F1-temporal reproduction/run sequence.

## Required Handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch codex/task-004h;
- implementation commit SHA;
- pushed-to-origin status;
- main/master untouched;
- exact audio checkpoint path + SHA256;
- strict-load result;
- reproduced DEV depression/Parkinson Scores and Mean_Score;
- reproduction deltas;
- registry key;
- output/aux shapes;
- frozen-audio no-gradient result;
- synthetic and real DataModule loss/backward result;
- no external task_id;
- no training/Test metrics beyond the authorized DEV reproduction;
- RAMPS deferred;
- text/description deferred;
- src/audio unchanged;
- src/video unchanged.

Stop after this task.
