# TASK-005H-AA-DQTR-OPTUNA: Search a Novel Audio-Anchored Disease-Query Trust-Region Fusion Model

## Authority and branch

This task supersedes the unexecuted TASK-005H-OPTUNA before any implementation branch was created.

Required branch:

    codex/task-005h-aadqtr-optuna

Start from current `origin/main`, which includes:

- owner/manager optimization override `5c2413038bf6a711a72c21256b5b9c946f29b312`;
- AA-DQTR method override `c1f4492b7e3b8e2609d981e0b5a2a724b84d3401`.

Create exactly one branch from that `origin/main`.

Stage 6 remains paused. This task is one bounded Stage-5 optimization sprint.

## Goal

Implement and optimize one new primary-method candidate:

**AA-DQTR — Audio-Anchored Disease-Query Trust-Region Fusion**

The method must:

- preserve the exact strong historical temporal-audio model as a frozen anchor;
- use disease-specific learned queries to extract task-conditioned evidence from temporal video;
- use multiple shared query-to-video refinement blocks;
- use task-specific reliability gates;
- use task-specific bounded correction scales;
- use a confidence-weighted trust-region loss to reduce negative transfer;
- start exactly at the frozen audio function through zero-initialized correction heads;
- keep two independent binary disease logits;
- remain interpretable and <=1.2M trainable non-audio parameters for every Optuna trial.

Then run a native Chimera Optuna sweep with exactly 24 seed42 trials.

If at least one seed42 trial passes the predeclared safe-audio gate, select the highest DEV Mean among those safe trials and confirm that exact frozen configuration at true seeds43 and 44.

Maximum production training invocations:

    26

## Required reading

Read in this exact order:

1. AGENTS.md
2. docs/README.md
3. docs/PROJECT_REQUIREMENTS.md
4. docs/PLAN.md, especially Stage 4 safe-audio comparison, Stage 5, and Section 7 Promotion Rule
5. docs/PROGRESS_EN.md through OWNER/MANAGER-OVERRIDE-051
6. docs/NEXT_TASK_EN.md
7. docs/SOTA_REVIEW_EN.md relevant fusion/negative-transfer sections
8. src/fusion/models/frozen_audio_temporal_adapter.py
9. src/fusion/models/av_audio_query_temporal_video.py
10. src/fusion/models/av_r3_disease_query.py
11. src/fusion/models/av_audio_confidence_gated.py
12. src/common/optimizers/wsm_trainable_adamw.py
13. src/fusion/loss/wsm_masked_sparse_loss.py
14. src/chimera_plugin.py
15. configs/wsm_mm_pd_dep_v1/fusion/05_audio_query_temporal_video.yaml

Also inspect the installed Chimera ML Optuna implementation:

- docs/en/user-guide/sweeps.md
- src/chimera_ml/training/sweep.py
- src/chimera_ml/cli.py

Do not modify Chimera ML.

## Owner-authorized search exception

For this task only:

- a bounded Optuna architecture/hyperparameter search is authorized;
- the usual no-broad-search restriction is relaxed only for AA-DQTR;
- architectural depth and fusion hyperparameters may be tuned;
- the frozen audio architecture/checkpoint itself may NOT be tuned;
- no other model family may be introduced after seeing results.

All other project invariants remain active.

## Frozen audio anchor

Exact checkpoint:

    logs/wsm_audio_segment_wavlm_base_l9_pool4/multitask_audio_mamba_2026-08-19_14-12_audio_mamba_segment_model_wsm_audio_models-e0ce-006_ea8c8d77/checkpoints/epoch=4_dev_mean_score=0.7878.pt

Required SHA256:

    0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2

Frozen DEV reference:

- depression Score: `0.7479183895`
- Parkinson Score: `0.8277353635`
- Mean: `0.7878268765`

The frozen audio submodel must remain:

- `requires_grad=False`;
- eval-only under parent `train()`;
- executed under `torch.no_grad()`;
- absent from optimizer parameter groups.

## Allowed tracked files

Codex may modify/add only:

- src/fusion/models/av_audio_anchored_disease_query_trust.py
- src/fusion/models/__init__.py
- src/fusion/loss/audio_anchor_trust_loss.py
- src/fusion/loss/__init__.py
- src/chimera_plugin.py
- configs/wsm_mm_pd_dep_v1/fusion/31_aadqtr_optuna_base_seed42.yaml
- configs/wsm_mm_pd_dep_v1/fusion/32_aadqtr_optuna_sweep.yaml
- configs/wsm_mm_pd_dep_v1/fusion/33_aadqtr_optuna_selected_seed42.yaml
- configs/wsm_mm_pd_dep_v1/fusion/34_aadqtr_confirm_seed43.yaml
- configs/wsm_mm_pd_dep_v1/fusion/35_aadqtr_confirm_seed44.yaml
- scripts/common/audit_aadqtr_optuna_sweep.py
- docs/PROGRESS_EN.md

The audit script is optional.

No other tracked file may change.

## Forbidden actions

- Do not modify `src/audio`.
- Do not modify `src/video`.
- Do not modify any existing fusion model.
- Do not modify any DataModule.
- Do not modify any existing loss.
- Do not modify Chimera ML.
- Do not unfreeze/retrain/tune the historical audio model.
- Do not use pseudo labels.
- Do not use corpus ID, split ID, Test protocol ID, observed labels, targets, or external task IDs as model inputs.
- Do not change canonical data/splits/caches.
- Do not use Test metrics as Optuna objectives or for search-space edits, trial selection, safe eligibility, confirmation, stopping, or interpretation.
- Do not modify model semantics or the search space after the first metric-bearing trial starts.
- Do not introduce another model family after seeing results.
- Do not run more than 24 Optuna trials.
- Do not run confirmation seeds unless a seed42 trial passes the frozen safe-audio gate.
- Do not start Stage 6, Stage 7, Text/Description, or Final Test.
- Do not make significance or final-model claims.

## 1. AA-DQTR model

Implement:

    src/fusion/models/av_audio_anchored_disease_query_trust.py

Registry key:

    wsm_av_audio_anchored_disease_query_trust_model

Recommended class:

    WSMAVAudioAnchoredDiseaseQueryTrustModel

### Required constructor parameters

    audio_checkpoint_path: str
    audio_feature_dim: int = 768
    audio_hidden_dim: int = 192
    video_feature_dim: int = 512
    hidden_dim: int = 192
    num_heads: int = 4
    query_layers: int = 2
    temporal_ff_multiplier: int = 3
    residual_hidden_dim: int = 192
    residual_layers: int = 2
    gate_hidden_dim: int = 128
    dropout: float = 0.15
    depression_correction_scale: float = 0.15
    parkinson_correction_scale: float = 1.0
    num_tasks: int = 2

Validate all dimensions/ranges.

### 1.1 Frozen audio outputs

Use `FrozenAudioTemporalAdapter` and obtain:

- `audio_base_logits [B,2]`;
- `audio_task_features [B,2,192]`.

No external task ID may enter the model.

### 1.2 Video token projection

Project temporal video tokens:

    [B,T,512] -> [B,T,H]

Use:

    LayerNorm(512)
    Linear(512,H)
    GELU
    Dropout

Respect the canonical `video_mask`.

Unavailable video must never contribute.

### 1.3 Disease-query initialization

Create exactly two learned disease embeddings:

    disease_queries [2,H]

Initialize:

    Normal(mean=0, std=0.02)

Project each frozen task-specific audio feature:

    a_t = Linear(LayerNorm(audio_task_feature_t)) -> H

Initial query state:

    h_t^0 = LayerNorm(a_t + disease_query_t)

This explicitly anchors every disease query in the frozen task-specific audio representation.

### 1.4 Query-to-video refinement blocks

Implement `query_layers` shared blocks, where `query_layers in {1,2,3}`.

Stack the two query states:

    Hq [B,2,H]

For each block:

1. pre-norm disease queries;
2. multi-head cross-attention:

       Q = disease queries [B,2,H]
       K,V = projected temporal video [B,T,H]

3. residual add;
4. pre-norm position-wise FFN on the disease queries:
   - Linear(H, temporal_ff_multiplier*H)
   - GELU
   - Dropout
   - Linear(...,H)
   - Dropout
5. residual add.

No disease-query self-attention is required or allowed in V1; tasks share block weights but do not directly attend to each other.

All blocks must respect video padding masks.

For unavailable-video rows:

- provide a numerically safe attention mask/input;
- final correction is explicitly zeroed;
- final prediction must be exactly the frozen audio base logits.

### 1.5 Task-specific reliability gate

For each task, compute:

    gate_t = sigmoid(
        MLP_t([
            LayerNorm(final_query_t),
            audio_projected_t,
            abs(audio_base_logit_t)
        ])
    )

Gate MLP:

    Linear(2H+1, gate_hidden_dim)
    GELU
    Dropout
    Linear(gate_hidden_dim, 1)

Gate is learned and task-specific.

The frozen audio base logit may be detached before entering the gate.

### 1.6 Task-specific residual head

For each task, input:

    [LayerNorm(final_query_t), audio_projected_t]

Dimension:

    2H

Build exactly `residual_layers` hidden transformations, each:

    Linear(current_dim, residual_hidden_dim)
    GELU
    Dropout

After the hidden stack:

    Linear(residual_hidden_dim, 1)

The FINAL scalar Linear weight and bias MUST be zero-initialized.

This guarantees exact audio equality at initialization.

### 1.7 Final logits

Fixed config scales:

    s_D = depression_correction_scale
    s_P = parkinson_correction_scale

They are non-trainable scalars.

Final correction:

    delta_t =
        video_available
        * s_t
        * gate_t
        * residual_t

Final logit:

    z_t = audio_base_logit_t + delta_t

Required exact invariants:

- at initialization: `preds == audio_base_logits` bitwise/exactly where dtype permits;
- video unavailable: `preds == audio_base_logits` exactly;
- if `depression_correction_scale == 0`, depression prediction remains exactly frozen audio for the entire run.

### 1.8 Required aux outputs

Expose at least:

    audio_base_logits
    audio_task_features
    projected_audio_task_features
    disease_query_states
    video_sequence
    gate_values
    raw_residual_logits
    correction_logits
    task_logits

All diagnostics must be finite.

## 2. AA-DQTR trust-region loss

Implement:

    src/fusion/loss/audio_anchor_trust_loss.py

Registry key:

    wsm_audio_anchor_trust_loss

Recommended class:

    WSMAudioAnchorTrustLoss

Constructor:

    anchor_weight: float = 0.05
    eps: float = 1e-8

Validate:

    0 <= anchor_weight <= 1.0
    eps > 0

### 2.1 Primary observed sparse BCE

Use only observed labels.

For:

    observed_mask = batch.masks["observed_mask"]

compute standard BCE-with-logits over observed entries only.

Unknown labels MUST remain masked and never become negatives.

Observed targets must be finite binary values.

### 2.2 Confidence-weighted trust penalty

Read:

    base = output.aux["audio_base_logits"]

Detach base before computing confidence/anchor target.

Define:

    confidence =
        2 * abs(sigmoid(base.detach()) - 0.5)

so confidence is in [0,1].

Define correction:

    delta = output.preds - base.detach()

Anchor penalty:

    L_anchor =
        sum(
            observed_mask
            * confidence
            * delta^2
        )
        /
        (observed_count + eps)

Final loss:

    L =
        L_observed_BCE
        + anchor_weight * L_anchor

The regularizer does NOT assert that audio is correct.

It only discourages large video-driven moves where the frozen audio classifier is confident.

No Test quantity enters the loss.

### 2.3 Required diagnostics

Expose or make inspectable:

- observed BCE;
- anchor penalty;
- mean confidence per task;
- mean absolute correction per task.

Do not require a new callback solely for these diagnostics unless necessary.

## 3. Registration

Update:

- `src/fusion/models/__init__.py`
- `src/fusion/loss/__init__.py`
- `src/chimera_plugin.py`

Register/import:

- `wsm_av_audio_anchored_disease_query_trust_model`
- `wsm_audio_anchor_trust_loss`

Preserve all existing registrations.

## 4. Base config

Create:

    configs/wsm_mm_pd_dep_v1/fusion/31_aadqtr_optuna_base_seed42.yaml

Fixed:

- seed 42
- experiment `wsm_mm_pd_dep_v1`
- canonical `wsm_av_fusion_datamodule`
- batch_size 8
- model `wsm_av_audio_anchored_disease_query_trust_model`
- exact frozen audio checkpoint
- num_heads 4
- temporal_ff_multiplier 3
- gate_hidden_dim 128
- loss `wsm_audio_anchor_trust_loss`
- optimizer `wsm_trainable_adamw_optimizer`
- epochs 30
- CUDA
- mixed precision true
- grad clip 0.5
- checkpoint/early stopping only `dev/mean_score`, max
- patience 6
- min_delta 0.0005
- all required instrumentation
- DEV/TEST_NONE/TEST_SOFT/TEST_HARD streams

Use reasonable in-bounds default values for all swept fields.

## 5. Frozen Optuna search space

Create:

    configs/wsm_mm_pd_dep_v1/fusion/32_aadqtr_optuna_sweep.yaml

Use exactly:

    method: optuna
    n_trials: 24
    study_name: wsm-aadqtr-safe-v1
    target:
      monitor: dev/mean_score
      mode: max

Search EXACTLY these 10 parameters:

### 5.1 hidden_dim

    model.params.hidden_dim
    type: categorical
    choices: [128, 160, 192, 224, 256]

### 5.2 query_layers

    model.params.query_layers
    type: int
    low: 1
    high: 3
    step: 1

### 5.3 residual_hidden_dim

    model.params.residual_hidden_dim
    type: categorical
    choices: [128, 192, 256, 320]

### 5.4 residual_layers

    model.params.residual_layers
    type: int
    low: 1
    high: 3
    step: 1

### 5.5 dropout

    model.params.dropout
    type: float
    low: 0.05
    high: 0.30

### 5.6 depression correction scale

    model.params.depression_correction_scale
    type: float
    low: 0.0
    high: 0.50

### 5.7 Parkinson correction scale

    model.params.parkinson_correction_scale
    type: float
    low: 0.25
    high: 1.75

### 5.8 anchor trust weight

    loss.params.anchor_weight
    type: float
    low: 0.0001
    high: 1.0
    log: true

### 5.9 learning rate

    optimizer.params.lr
    type: float
    low: 0.00002
    high: 0.0004
    log: true

### 5.10 weight decay

    optimizer.params.weight_decay
    type: float
    low: 0.00001
    high: 0.05
    log: true

No other parameter may be swept.

Do not edit this search space after the firewall commit.

## 6. Parameter cap

Every possible combination in the search space MUST satisfy:

    trainable non-audio parameters <= 1,200,000

Before production, programmatically instantiate all boundary/worst-case combinations needed to prove this.

Frozen audio parameters do not count toward this cap but MUST remain absent from optimizer groups.

If the specified search space cannot satisfy the cap with the required architecture, stop and report before any production trial. Do not silently shrink the search space.

## 7. Mandatory pre-sweep firewall

Before Optuna trial 1:

1. implement/register model and loss;
2. create base/sweep configs;
3. compile model/loss/plugin;
4. validate base config;
5. run Chimera Optuna `--dry-run`;
6. verify:
   - method optuna;
   - 24 trials;
   - exact 10 search variables;
   - target exactly `dev/mean_score`, max;
7. verify exact audio checkpoint SHA;
8. prove frozen audio absent from optimizer groups;
9. parameter-cap proof for the full search space;
10. synthetic model smoke:
    - exact zero-init equality;
    - exact video-missing fallback;
    - D scale=0 keeps D exact audio after optimizer steps;
    - finite forward/loss/backward;
    - frozen audio gradients None;
11. trust-loss smoke:
    - unknown labels masked;
    - observed BCE correct;
    - anchor penalty exact against direct reference formula;
    - base logits structurally detached;
    - anchor_weight=0 reduces exactly to observed BCE;
12. two-step wake-up smoke:
    - first backward trains final correction layer;
    - after exactly one bounded optimizer step, query/video/gate/residual upstream paths receive finite gradients;
13. canonical DEV initialization reproduction:
    - fresh zero-init AA-DQTR reproduces audio D/P/Mean within 0.0005;
    - DEV only;
    - no Test loader iteration;
14. append complete frozen architecture/search manifest to PROGRESS_EN.md;
15. commit and push one firewall commit.

No metric-bearing Optuna trial may start before the firewall commit exists on origin.

## 8. Exact production sweep

Run exactly once:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml sweep \
      --base-config configs/wsm_mm_pd_dep_v1/fusion/31_aadqtr_optuna_base_seed42.yaml \
      --sweep-config configs/wsm_mm_pd_dep_v1/fusion/32_aadqtr_optuna_sweep.yaml \
      --sweep-name aadqtr-safe-v1 \
      --max-trials 24

Exactly 24 seed42 trial training invocations are authorized.

No manually launched extra seed42 trials.

If a code/config defect aborts the sweep BEFORE the first metric-bearing trial completes, a narrow correction is allowed within scope.

After any metric-bearing trial has completed:

- model semantics are frozen;
- loss semantics are frozen;
- search space is frozen;
- no corrective redesign/tuning is allowed inside this task.

Stop and report a blocker if needed.

## 9. Sweep audit

After completion verify:

- sweep method `optuna`;
- study `wsm-aadqtr-safe-v1`;
- exactly 24 completed trials;
- objective `dev/mean_score`, max;
- no Test objective.

Record every trial:

- trial number/id;
- generated config;
- sampled 10 parameters;
- run name;
- target DEV Mean;
- target epoch;
- trainable parameter count.

Produce top-5 by DEV Mean.

## 10. Frozen safe-audio eligibility

A seed42 trial is SAFE only if all are true:

- DEV Mean > `0.7878268765`;
- depression Score >= `0.7379183895`;
- Parkinson Score >= `0.8177353635`;
- model/audio invariants passed;
- no Test-driven choice.

After all 24 trials:

1. form the set of safe trials using ONLY these frozen DEV conditions;
2. if the set is empty:
   - record `NO SAFE AUDIO-BEATING AA-DQTR TRIAL`;
   - do not run seeds43/44;
3. if safe trials exist:
   - choose the one with highest DEV Mean;
   - ties within `1e-12` are broken by lower trainable parameter count;
   - do not use Test or calibration to break ties.

This selected safe trial is the only one eligible for confirmation.

## 11. Freeze selected safe config

If a safe trial exists, create:

    configs/wsm_mm_pd_dep_v1/fusion/33_aadqtr_optuna_selected_seed42.yaml

This must be an exact tracked copy of its generated trial config.

Record:

- source generated config path;
- config SHA256;
- exact 10 sampled values;
- model trainable parameter count;
- selected checkpoint path/SHA256;
- seed42 DEV D/P UAR/MF1/Score and Mean;
- delta vs frozen audio.

Create:

    configs/wsm_mm_pd_dep_v1/fusion/34_aadqtr_confirm_seed43.yaml
    configs/wsm_mm_pd_dep_v1/fusion/35_aadqtr_confirm_seed44.yaml

They may differ from config 33 ONLY in:

- top-level seed;
- run_name.

All model/loss/optimizer hyperparameters remain exact.

## 12. Conditional true-seed confirmation

Only if Section 10 finds a safe seed42 trial:

1. run seed43 once;
2. run seed44 once.

Use normal `chimera-ml train`.

No sweep and no rerun.

For seeds42/43/44 record:

- selected DEV D/P/Mean;
- checkpoint SHA256;
- trainable parameter count.

Compute three-seed mean/sample std.

### Provisional safe-audio success

AA-DQTR passes the provisional confirmation gate only if:

1. DEV Mean > audio Mean on all three seeds;
2. D Score >= `0.7379183895` on all three seeds;
3. P Score >= `0.8177353635` on all three seeds;
4. three-seed Mean > `0.7878268765`;
5. three-seed D mean >= `0.7379183895`;
6. three-seed P mean >= `0.8177353635`;
7. checkpoint SHA256 values are pairwise distinct;
8. Test did not influence any choice.

Codex MUST NOT declare final promotion.

Manager decides after audit.

## 13. Required diagnostics

For the selected safe seed42 trial, and confirmations if executed, record:

- mean/std gate value by task on DEV;
- mean absolute correction by task on DEV;
- fraction of observed DEV predictions whose sign changes relative to frozen audio, by task;
- count:
  - audio wrong -> AA-DQTR correct;
  - audio correct -> AA-DQTR wrong;
- selected loss observed BCE;
- selected anchor penalty.

These diagnostics are DEV-only and explanatory.

Do not use them for Optuna selection.

## 14. Test firewall

TEST_NONE/SOFT/HARD remain mandatory epoch monitoring only.

They MUST NOT affect:

- Optuna objective;
- sampler;
- architecture;
- loss;
- search space;
- safe eligibility;
- selected safe trial;
- confirmation decision;
- checkpoint selection;
- next action.

## 15. Scope verification

Run:

    git diff --check
    git diff origin/main -- src/audio
    git diff origin/main -- src/video
    git diff origin/main -- src/fusion/data
    git diff origin/main -- src/common
    git status --short
    git diff --stat origin/main...HEAD
    git log -12 --oneline --decorate

Only authorized model/loss/registration/config/audit/PROGRESS files may differ.

## Acceptance criteria

TASK-005H-AA-DQTR-OPTUNA passes only if:

- branch exactly `codex/task-005h-aadqtr-optuna`;
- AA-DQTR implementation matches this frozen contract;
- audio remains exact/frozen;
- zero-init and missing-video exact fallback pass;
- trust loss matches direct reference;
- full 10-variable search space is frozen before trial 1;
- all possible trial architectures respect <=1.2M non-audio trainable parameters;
- native Chimera Optuna is used;
- exactly 24 seed42 trials complete;
- target only `dev/mean_score`, max;
- no Test-driven search;
- safe set is filtered only by the frozen DEV gate;
- selected trial is highest DEV Mean within safe set;
- seed43/44 execute only when a safe seed42 trial exists;
- confirmations reuse exact hyperparameters;
- at most 26 production training invocations;
- no pseudo labels;
- no Stage6/7/Text/Final Test;
- evidence complete;
- branch pushed;
- main/master untouched.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch `codex/task-005h-aadqtr-optuna`;
- firewall commit SHA;
- final evidence commit SHA;
- pushed status;
- main/master untouched;
- frozen audio SHA;
- exact model equation;
- exact trust loss equation;
- exact 10-variable Optuna space;
- production sweep command;
- sweep directory/manifest/study;
- completed trial count;
- top-5 DEV trials;
- number of safe trials;
- selected safe trial or explicit none;
- selected seed42 D/P/Mean and audio deltas if available;
- selected config/checkpoint SHA if available;
- seeds43/44 results if executed;
- three-seed mean/std and provisional gate if executed;
- gate/correction/error-transition diagnostics;
- trainable parameter counts;
- no Test-driven decision;
- no post-hoc search-space change;
- no pseudo labels;
- Stage 5 optimization status;
- no Stage6/7/Text/Final Test.

Stop after TASK-005H-AA-DQTR-OPTUNA.
