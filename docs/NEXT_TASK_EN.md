# TASK-005E: Implement the R3 Disease-Query Availability-Aware A+V Model Contract

## Task Identifier and Title

TASK-005E — Implement and register the bounded R3 disease-query, availability-aware A+V model contract with auxiliary unimodal logits, using observed sparse supervision only and no training.

Required branch:

    codex/task-005e

Start from current `origin/main`, which includes:

- PR #41 merge `f102a8190a277fa225b99005d4b20aa651c32fd8`;
- manager decision commit `95f41b0ce3013a2c82fdaaed1febca6bc27255cb`.

Create exactly one task branch from that `origin/main`, implement only this contract, verify it, update `docs/PROGRESS_EN.md`, commit, push, and stop.

## Goal

Implement the next PLAN-authorized Stage-5 hypothesis, R3, as an independent A+V architecture contract after the negative R2 pseudo-supervision result.

R2 direct pseudo-supervision is NOT promoted into this model.

This task must use:

- the ordinary canonical A+V DataModule;
- observed-label-only masked sparse BCE;
- two independent disease logits;
- learned disease-query tokens;
- task-conditioned audio/video modality gates;
- exact availability masking;
- auxiliary audio-only and video-only logits exposed for a later agreement/consistency task.

Do not implement an auxiliary agreement loss and do not train.

## Required Reading

Read in this order:

1. AGENTS.md
2. docs/README.md
3. docs/PROJECT_REQUIREMENTS.md, especially Sections 2, 3, 4, 8, 9, 10, 11, 13, 14
4. docs/PLAN.md, especially Stage 5 R3 and Stage-5 gate
5. docs/PROGRESS_EN.md through MANAGER-DECISION-040
6. docs/NEXT_TASK_EN.md
7. docs/SOTA_REVIEW_EN.md, especially Sections 9–11
8. src/fusion/models/av_f2_task_aware_directed.py
9. src/fusion/data/wsm_av_fusion_datamodule.py
10. src/common/loss/wsm_masked_sparse_loss.py
11. src/chimera_plugin.py
12. configs/wsm_mm_pd_dep_v1/fusion/02_f2_task_aware_directed.yaml

Use active English documentation only. Do not load archived Russian variants.

## Allowed Files

Codex may modify only:

- src/fusion/models/av_r3_disease_query.py
- src/chimera_plugin.py
- configs/wsm_mm_pd_dep_v1/fusion/06_ramps_r3_disease_query_contract.yaml
- docs/PROGRESS_EN.md

No other tracked file may change.

## Forbidden Actions

- Do not modify src/audio.
- Do not modify src/video.
- Do not modify existing fusion models.
- Do not modify fusion/common data or loss code.
- Do not use `wsm_ramps_semantic_datamodule`.
- Do not use the R2 pseudo cache in this task.
- Do not use `wsm_ramps_observed_pseudo_loss`.
- Do not use the pseudo-scale warm-up callback.
- Do not add pseudo targets or teacher signals.
- Do not change F2 or historical baselines.
- Do not add transcript/text/description streams.
- Do not use raw frames.
- Do not run `chimera-ml train`.
- Do not execute an optimizer step.
- Do not use DEV/Test metrics to redesign the model.
- Do not iterate Test rows in the implementation smoke.
- Do not start R4, Stage 6, Stage 7, or Final Test.
- Do not make missing-label correctness or comorbidity claims.

## Frozen R3 Architecture

Implement:

    src/fusion/models/av_r3_disease_query.py

Registry key:

    wsm_av_r3_disease_query_model

Class name SHOULD be:

    WSMAVR3DiseaseQueryModel

The architecture below is frozen for this contract. Do not substitute a Transformer, relation bank, cross-attention block, or a third modality.

### Constructor

Required constructor parameters:

    audio_feature_dim: int = 768
    video_feature_dim: int = 512
    hidden_dim: int = 192
    gate_hidden_dim: int = 192
    dropout: float = 0.2
    num_tasks: int = 2

Require:

- all dimensions positive;
- `num_tasks == 2`;
- `0 <= dropout < 1`.

### Input contract

Consume the same Batch fields as the accepted F2 model:

- `batch.inputs["audio_cls"]`: finite float `[B, audio_feature_dim]`;
- `batch.inputs["video"]`: finite float `[B,T,video_feature_dim]`;
- `batch.masks["video_mask"]`: bool `[B,T]`;
- `batch.masks["modality_available"]`: bool `[B,2]`, ordered `[audio, video]`.

Require at least one available modality per sample.

If video is marked available, require at least one valid video position.

Do not consume corpus identity, disease label, observed mask, or task_id as an input feature.

### Base modality projections

Use exactly:

Audio:

    LayerNorm(audio_feature_dim)
    Linear(audio_feature_dim, hidden_dim)
    GELU
    Dropout(dropout)

Video:

    LayerNorm(video_feature_dim)
    Linear(video_feature_dim, hidden_dim)
    GELU
    Dropout(dropout)

Pool video before projection using the same masked mean semantics as F2.

Zero unavailable raw/projected modality contribution before task fusion.

### Disease-query tokens

Add exactly one learned query vector per disease:

    task_queries: nn.Parameter shape [2, hidden_dim]

Initialize with normal distribution:

    mean = 0.0
    std = 0.02

These are model parameters, not sample-provided task/corpus labels.

### Task-conditioned candidate features

For each disease task `t`:

    q_t = task query
    a_t = LayerNorm_t(audio_feature + q_t)
    v_t = LayerNorm_t(video_feature + q_t)

Use one task-specific `LayerNorm(hidden_dim)` shared between that task's audio/video candidate normalization.

Keep two task-specific candidate norms total.

### Shared query-conditioned gate

Use one shared gate network for both tasks and both modalities:

    LayerNorm(2 * hidden_dim)
    Linear(2 * hidden_dim, gate_hidden_dim)
    GELU
    Dropout(dropout)
    Linear(gate_hidden_dim, 1)

For task `t`, score:

    [q_t || a_t]
    [q_t || v_t]

Mask unavailable modalities before normalization.

Required gate semantics:

- weights are finite;
- weights are in [0,1];
- weights sum to exactly 1 within numerical tolerance across available modalities;
- unavailable modality weight is exactly 0;
- if only audio is available, weights are [1,0];
- if only video is available, weights are [0,1];
- no all-unavailable sample is permitted.

### Task fusion

For each task:

    fused_t =
        LayerNorm_t(
            q_t
            + w_audio * a_t
            + w_video * v_t
        )

Use one task-specific fusion `LayerNorm(hidden_dim)` per task.

### Main task heads

Use two independent main heads, one per disease:

    LayerNorm(hidden_dim)
    Linear(hidden_dim, hidden_dim)
    GELU
    Dropout(dropout)
    Linear(hidden_dim, 1)

Return:

    ModelOutput.preds shape [B,2]

The two outputs remain independent binary logits.

### Auxiliary unimodal heads

Expose auxiliary unimodal logits for later R3 agreement work.

For each task and each modality, add one small auxiliary head:

    LayerNorm(hidden_dim)
    Linear(hidden_dim, 1)

Required outputs:

- `audio_aux_logits`: `[B,2]`;
- `video_aux_logits`: `[B,2]`.

Compute them from `a_t` and `v_t` respectively.

For unavailable modality entries:

- keep auxiliary logits finite;
- expose validity masks;
- do not use them for any loss in TASK-005E.

Required validity outputs:

- `audio_aux_valid`: bool `[B,2]`;
- `video_aux_valid`: bool `[B,2]`.

They are the corresponding modality availability flag broadcast across the two tasks.

### Required ModelOutput.aux

Include at least:

- `features_audio`;
- `features_video`;
- `task_audio_features` with shape `[B,2,H]`;
- `task_video_features` with shape `[B,2,H]`;
- `task_modality_weights` with shape `[B,2,2]`, last axis `[audio,video]`;
- `task_features` with shape `[B,2,H]`;
- `audio_aux_logits` `[B,2]`;
- `video_aux_logits` `[B,2]`;
- `audio_aux_valid` bool `[B,2]`;
- `video_aux_valid` bool `[B,2]`;
- `task_logits` mapping for depression/Parkinson.

All floating outputs must be finite.

## Parameter Budget

The accepted sparse F2 model has:

    736004 trainable parameters

The R3 model MUST have:

    trainable_parameters <= 736004

Do not pad the model merely to equalize parameter count.

Record the exact R3 count and the delta versus F2.

If the frozen architecture as implemented exceeds the cap, stop and report blocked instead of silently shrinking/changing the specified dimensions.

## Chimera Registration

Update:

    src/chimera_plugin.py

Import/register:

    fusion.models.av_r3_disease_query

After plugin registration require:

    MODELS: wsm_av_r3_disease_query_model

All existing registrations must remain intact.

## Contract Config

Add:

    configs/wsm_mm_pd_dep_v1/fusion/06_ramps_r3_disease_query_contract.yaml

Base all non-model semantics on:

    configs/wsm_mm_pd_dep_v1/fusion/02_f2_task_aware_directed.yaml

Required values:

- seed: 42
- experiment_name: `wsm_mm_pd_dep_v1`
- run_name: `ramps_r3_disease_query_contract_smoke`
- data: `wsm_av_fusion_datamodule`
- model: `wsm_av_r3_disease_query_model`
- model params:
  - audio_feature_dim: 768
  - video_feature_dim: 512
  - hidden_dim: 192
  - gate_hidden_dim: 192
  - dropout: 0.2
  - num_tasks: 2
- loss: `wsm_masked_sparse_loss`
- optimizer: same AdamW as F2
- train semantics: same as F2
- required callbacks/loggers: same as F2
- checkpoint and early stopping: `dev/mean_score`, mode=max only.

Do not include pseudo cache, pseudo loss, pseudo callback, teacher path, text, description, or semantic prompt settings.

This config is a structural contract only. It does NOT authorize training.

## Required Synthetic Availability/Query Smoke

Use seed 42 and `model.eval()`.

Construct a synthetic batch containing exactly:

1. one sample with audio+video available;
2. one audio-only sample;
3. one video-only sample.

Verify:

- `preds.shape == [3,2]`;
- all main/aux/features/weights are finite;
- `task_modality_weights.shape == [3,2,2]`;
- each task's weights sum to 1;
- audio-only row has weights exactly `[1,0]` for both tasks;
- video-only row has weights exactly `[0,1]` for both tasks;
- auxiliary valid masks exactly match availability broadcast across tasks.

Availability invariance:

- clone the batch;
- replace the unavailable video tensor of the audio-only sample with large finite noise;
- its main logits must remain identical within `1e-7`;
- replace the unavailable audio tensor of the video-only sample with large finite noise;
- its main logits must remain identical within `1e-7`.

All-unavailable guard:

- a sample with availability `[False,False]` must raise `ValueError`.

## Required Real A+V Forward/Loss/Backward Smoke

Use:

    wsm_av_fusion_datamodule
    wsm_av_r3_disease_query_model
    wsm_masked_sparse_loss

Instantiate the real canonical A+V DataModule with:

- batch_size 4;
- num_workers 0;
- persistent_workers false;
- shuffle_train false.

Deterministically select a four-row TRAIN batch containing:

- at least two depression-owned rows;
- at least two Parkinson-owned rows.

Run on CPU with seed 42.

Require:

- real forward returns finite `[4,2]` logits;
- sparse observed loss is finite;
- backward completes;
- all present parameter gradients are finite;
- both main task heads have at least one non-zero gradient;
- both modality projections have at least one non-zero gradient;
- disease-query parameter receives finite non-zero gradient;
- shared gate network receives finite non-zero gradient.

Because auxiliary logits are not part of TASK-005E loss:

- auxiliary-head parameter gradients MUST be absent/None after the main sparse-loss backward.

This absence is expected and proves TASK-005E did not silently add an auxiliary objective.

No optimizer step.

Do not call `val_dataloader()`.

Do not iterate DEV/Test data.

## Parameter/Registry/Config Verification

Run from repository root.

### 1. Compile

    python3 -m py_compile       src/fusion/models/av_r3_disease_query.py       src/chimera_plugin.py

### 2. Config validation

    .venv/bin/chimera-ml validate-config       -c configs/wsm_mm_pd_dep_v1/fusion/06_ramps_r3_disease_query_contract.yaml

### 3. Registry + parameter cap

Run an inline registry/build command that:

- registers project plugins;
- asserts `wsm_av_r3_disease_query_model` is in MODELS;
- builds F2 and R3 with their exact config dimensions;
- counts trainable parameters;
- asserts F2 count is exactly `736004`;
- asserts R3 count is `<= 736004`;
- prints both counts and delta.

Record the exact command/result in PROGRESS_EN.md.

### 4. Synthetic availability/query smoke

Run the exact required smoke above and record the full command/result in PROGRESS_EN.md.

### 5. Real forward/loss/backward smoke

Run the exact required real TRAIN-only smoke above and record the full command/result in PROGRESS_EN.md.

### 6. Scope checks

    git diff --check
    git diff origin/main -- src/audio
    git diff origin/main -- src/video
    git diff origin/main -- src/fusion/data
    git diff origin/main -- src/fusion/loss
    git diff origin/main -- src/common
    git status --short
    git diff --stat origin/main...HEAD
    git log -4 --oneline --decorate

## Acceptance Criteria

TASK-005E passes only if:

- branch is exactly `codex/task-005e` from current manager-updated `origin/main`;
- only the four allowed files change;
- model registry key is exactly `wsm_av_r3_disease_query_model`;
- model returns exactly two independent logits;
- no corpus/task label is consumed from the batch as an input feature;
- learned disease queries have shape `[2,192]`;
- availability-aware task modality weights satisfy the exact contract;
- unavailable-input invariance passes;
- all-unavailable input is rejected;
- auxiliary unimodal logits/valid masks satisfy the exact shape/finite contract;
- R3 trainable parameter count is no greater than `736004`;
- contract YAML validates;
- YAML uses ordinary `wsm_av_fusion_datamodule` and `wsm_masked_sparse_loss`;
- YAML contains no R2 pseudo path;
- synthetic smoke passes;
- real TRAIN-only forward/loss/backward smoke passes;
- both main heads, both modality projections, disease queries, and shared gate receive finite non-zero gradients;
- auxiliary-head gradients are absent under main-only sparse loss;
- no optimizer step or training occurs;
- no DEV/Test row is iterated;
- src/audio, src/video, fusion data/loss, common code, and existing fusion models remain unchanged;
- `git diff --check` passes;
- docs/PROGRESS_EN.md records exact commands/results and exact parameter count;
- one implementation commit is pushed to origin;
- main/master remains untouched by Codex.

Passing TASK-005E authorizes only manager review of the R3 model contract. It does not authorize R3 training or an auxiliary agreement loss.

## Required Handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch `codex/task-005e`;
- implementation commit SHA;
- pushed-to-origin status;
- main/master untouched;
- manager base commit preserved;
- registry key;
- exact R3 trainable parameter count and delta versus F2 `736004`;
- synthetic availability/query smoke result;
- unavailable-input invariance result;
- real TRAIN-only forward/loss/backward result;
- gradient evidence for main heads, projections, task queries, and shared gate;
- confirmation auxiliary heads received no gradients under main-only loss;
- config validation result;
- no optimizer step/training;
- no DEV/Test iteration;
- no pseudo cache/loss/warm-up use;
- no missing-label correctness/comorbidity claim;
- R2 remains a preserved negative result and is not promoted;
- R4 not started;
- general Text/Description remains deferred;
- src/audio unchanged;
- src/video unchanged;
- fusion data/loss/common/existing fusion models unchanged;
- Stage 5 remains active.

If all acceptance criteria pass, recommended next atomic step is manager review and a separate task that freezes the R3 auxiliary agreement-loss contract before any R3 training.

Stop after TASK-005E.
