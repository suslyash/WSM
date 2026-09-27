# TASK-005H-V2L-OPTUNA: Tune a Compact WSM Adaptation of V2L

## Authority and branch

This task supersedes the unexecuted AA-DQTR task before implementation.

Required branch:

    codex/task-005h-v2l-optuna

Start from current `origin/main`, which includes:

- owner/manager V2L selection commit `c798813684a33d4515f4f3689610874a8e74d8d2`.

Create exactly one branch from that `origin/main`.

Stage 6 remains paused during this optimization sprint.

## Goal

Implement and tune a compact two-view WSM adaptation of:

**V2L — When Semantically Consistent Encoding Meets View-Label Heterogeneity Modeling: A Unified Framework for Incomplete Multi-View Multi-Label Learning (IEEE TPAMI 2026).**

The WSM adaptation MUST preserve V2L's core mechanisms:

1. source-anchored variational proposal clusters;
2. precision-weighted per-view posterior aggregation;
3. product-of-experts joint posterior;
4. reconstruction/information-bottleneck objective;
5. cross-view posterior consistency;
6. intra-cluster posterior coherence;
7. cross-view instance discrimination;
8. hybrid mid-level semantic fusion + late decision fusion;
9. instance-wise and label-wise active view relevance;
10. observed-label-only classification.

Use exactly two views:

- audio;
- video.

Preserve the exact frozen historical temporal-audio model as the audio anchor.

Because WSM has exactly two views, do NOT implement V2L's multi-view random source-perturbation diversity constraint from Eq. 24/25. With two views that diversity condition degenerates. Instead use the paper's direct ordered-pair cross-view posterior consistency surrogate corresponding to Eq. 21.

This is a paper-guided WSM adaptation, not a claim of byte-for-byte reproduction of the original six-view V2L codebase.

## Why V2L was selected

The manager selected V2L from the modern incomplete multi-view/multi-label family because its defining mechanism is instance-wise and label-wise view relevance.

This directly targets the observed WSM failure mode:

- audio is very strong for depression;
- video has shown strong additional Parkinson evidence;
- one global modality weight is therefore inappropriate.

V2L's hybrid representation/decision fusion is intended to preserve shared semantics while allowing different labels on different samples to rely on different views.

## Required reading

Read in this exact order:

1. AGENTS.md
2. docs/README.md
3. docs/PROJECT_REQUIREMENTS.md
4. docs/PLAN.md
5. docs/PROGRESS_EN.md through OWNER/MANAGER-OVERRIDE-052
6. docs/NEXT_TASK_EN.md
7. docs/SOTA_REVIEW_EN.md sections on incomplete multi-view/multi-label learning and V2L
8. src/fusion/models/frozen_audio_temporal_adapter.py
9. src/fusion/models/av_audio_query_temporal_video.py
10. src/fusion/models/av_r3_disease_query.py
11. src/common/optimizers/wsm_trainable_adamw.py
12. src/chimera_plugin.py

Also inspect Chimera ML sweep support:

- docs/en/user-guide/sweeps.md
- src/chimera_ml/training/sweep.py
- src/chimera_ml/cli.py

Do not modify Chimera ML.

## Frozen external method contract

Use the V2L paper as the conceptual source for:

- view-specific Gaussian posterior proposals;
- precision-weighted proposal aggregation;
- PoE mid-level fusion;
- per-view prediction branches;
- active label-wise view relevance;
- relevance target derived from observed-label per-view prediction errors;
- observed-label masking;
- reconstruction / posterior consistency / posterior discrimination terms.

Do not add unrelated mechanisms such as pseudo labels, teacher/student EMA, RA-STCH, or task-relation graphs in this task.

## Frozen audio comparator

Exact historical audio checkpoint:

    logs/wsm_audio_segment_wavlm_base_l9_pool4/multitask_audio_mamba_2026-08-19_14-12_audio_mamba_segment_model_wsm_audio_models-e0ce-006_ea8c8d77/checkpoints/epoch=4_dev_mean_score=0.7878.pt

Required SHA256:

    0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2

Frozen DEV reference:

- depression Score: `0.7479183895`
- Parkinson Score: `0.8277353635`
- Mean: `0.7878268765`

The audio model remains:

- `requires_grad=False`;
- eval-only under parent train mode;
- executed under `torch.no_grad()`;
- absent from optimizer groups.

## Allowed tracked files

Codex may modify/add only:

- src/fusion/models/v2l_wsm.py
- src/fusion/models/__init__.py
- src/fusion/loss/v2l_wsm_loss.py
- src/fusion/loss/__init__.py
- src/chimera_plugin.py
- configs/wsm_mm_pd_dep_v1/fusion/31_v2l_optuna_base_seed42.yaml
- configs/wsm_mm_pd_dep_v1/fusion/32_v2l_optuna_sweep.yaml
- configs/wsm_mm_pd_dep_v1/fusion/33_v2l_selected_seed42.yaml
- configs/wsm_mm_pd_dep_v1/fusion/34_v2l_confirm_seed43.yaml
- configs/wsm_mm_pd_dep_v1/fusion/35_v2l_confirm_seed44.yaml
- scripts/common/audit_v2l_optuna_sweep.py
- docs/PROGRESS_EN.md

The audit script is optional.

No other tracked file may change.

## Forbidden actions

- Do not modify src/audio.
- Do not modify src/video.
- Do not modify existing fusion models.
- Do not modify any DataModule.
- Do not modify any existing loss.
- Do not modify Chimera ML.
- Do not unfreeze or tune the audio model.
- Do not use pseudo labels.
- Do not use Test metrics for model/search selection.
- Do not use corpus ID, split ID, protocol ID, labels, observed-mask state, or external task ID as a model input feature.
- Do not modify the search space after the first metric-bearing trial.
- Do not introduce another architecture after seeing sweep results.
- Do not run more than 24 Optuna trials.
- Do not run seed43/44 unless at least one safe seed42 trial exists.
- Do not start Stage 6/7, Text/Description, or Final Test.

## 1. WSM view construction

### 1.1 Audio view

Use `FrozenAudioTemporalAdapter`.

Construct the deterministic frozen audio source vector from:

- task-specific audio features `[B,2,192]`, flattened to `[B,384]`;
- frozen audio base logits `[B,2]`.

Concatenate:

    x_audio = [flatten(audio_task_features), audio_base_logits]

Dimension:

    386

Detach the complete audio source vector.

This preserves the exact strong temporal-audio semantics while exposing task-specific audio evidence to V2L.

### 1.2 Video view

Use the canonical temporal video tensor:

    [B,T,512]

and video mask.

For this first V2L adaptation, use deterministic masked mean pooling over valid frame features:

    x_video = masked_mean(video)

Dimension:

    512

Do not add a new temporal Transformer/attention family inside V2L. The task is intended to test the V2L fusion/relevance mechanism, not reopen video-family search.

Unavailable video must be masked.

## 2. V2L-WSM model

Implement:

    src/fusion/models/v2l_wsm.py

Registry key:

    wsm_v2l_model

Recommended class:

    WSMV2LModel

Constructor parameters:

    audio_checkpoint_path: str
    audio_view_dim: int = 386
    video_view_dim: int = 512
    latent_dim: int = 192
    encoder_hidden_dim: int = 256
    decoder_hidden_dim: int = 256
    active_hidden_dim: int = 64
    dropout: float = 0.15
    num_labels: int = 2
    num_views: int = 2

Require exactly two labels and two views.

Every floating tensor must remain finite.

### 2.1 Source-anchored proposal clusters

For each source view:

    audio
    video

create exactly two target-aware Gaussian proposal heads:

    source audio -> target audio
    source audio -> target video
    source video -> target audio
    source video -> target video

Each proposal produces:

    mu [B,D]
    logvar [B,D]

Clamp log-variance to a numerically safe frozen interval chosen before the firewall, e.g. [-8,8].

The front-end for each source view may use:

    LayerNorm(input_dim)
    Linear(input_dim, encoder_hidden_dim)
    GELU
    Dropout

and separate proposal output layers.

### 2.2 Precision-weighted per-view posterior aggregation

For a fixed source view v, aggregate its two target-aware proposal Gaussians with a unit Gaussian prior using diagonal precision weighting corresponding to the V2L paper Eq. 13/14.

Produce:

    view_mu[v]
    view_logvar[v]

for audio and video.

Validate finite positive variances.

### 2.3 Joint posterior

Construct the common posterior from available view-specific posteriors plus unit Gaussian prior using product-of-experts precision aggregation corresponding to the paper Eq. 28.

Produce:

    joint_mu
    joint_logvar

During training use the reparameterization trick.

During eval/inference use posterior means for deterministic prediction.

### 2.4 Decoders

Create one decoder for each view posterior.

Reconstruct normalized source-view vectors:

- audio reconstruction target: detached normalized `x_audio`;
- video reconstruction target: normalized `x_video`.

Use MSE reconstruction.

The decoder architecture may be:

    Linear(D, decoder_hidden_dim)
    GELU
    Dropout
    Linear(decoder_hidden_dim, source_dim)

No reconstruction is required at inference.

### 2.5 Mid-level semantic branch

From joint latent z:

    mid_delta = mid_classifier(z)

Shape:

    [B,2]

The final Linear of `mid_classifier` MUST be zero-initialized.

This is the WSM anchoring adaptation.

### 2.6 View-specific decision branches

Audio view logit is the exact frozen audio base logit:

    logits_audio = audio_base_logits

Do not replace it with a trainable scratch classifier.

Video view defines a residual classifier from `z_video`:

    video_delta = video_classifier(z_video)

with zero-initialized final Linear.

Define:

    logits_video = audio_base_logits + video_delta

At initialization:

    logits_audio == logits_video == audio_base_logits

### 2.7 Active instance-wise label-wise view relevance

For each sample i, label j, and view v, compute a relevance score from the corresponding scalar view logit.

Use one shared active-perception MLP applied scalar-wise:

    Linear(1, active_hidden_dim)
    GELU
    Dropout
    Linear(active_hidden_dim, 1)

For each label separately, softmax across available views.

Output:

    relevance_weights [B,2 labels,2 views]

This must be instance-specific and label-specific.

Unavailable views are excluded from normalization.

Late logits:

    late_logits =
        relevance_audio * logits_audio
        + relevance_video * logits_video

### 2.8 Final hybrid prediction

WSM residualized V2L final logits:

    preds =
        late_logits
        + mid_delta

Because mid/video final layers are zero-initialized:

    preds == frozen audio base logits

exactly at initialization.

If video is unavailable:

- video relevance is zero;
- video residual contribution is zero;
- final prediction is frozen audio plus zero-init mid residual initially;
- after training, the mid residual may still operate from audio semantic evidence.

Expose in ModelOutput.aux:

- audio_base_logits
- audio_task_features
- x_audio
- x_video
- proposal_mu
- proposal_logvar
- view_mu
- view_logvar
- joint_mu
- joint_logvar
- mid_delta
- audio_view_logits
- video_view_logits
- relevance_weights
- video_delta
- task_logits

## 3. V2L-WSM loss

Implement:

    src/fusion/loss/v2l_wsm_loss.py

Registry key:

    wsm_v2l_loss

Constructor:

    beta_consistency: float
    sigma_intra: float
    gamma_cross: float
    cross_temperature: float = 0.1
    relevance_weight: float = 1.0
    reconstruction_weight: float = 1.0
    view_aux_weight: float = 1.0
    eps: float = 1e-8

All weights non-negative. Temperature/eps positive.

### 3.1 Observed-label classification

Unknown labels remain masked.

Compute BCE-with-logits on observed entries for:

1. final hybrid `preds`;
2. audio view logits;
3. video view logits.

The audio branch is frozen but its observed BCE participates as a constant reference term; it must not create audio gradients.

Use:

    L_cls =
        BCE_observed(final)
        + view_aux_weight * mean(
            BCE_observed(audio_view),
            BCE_observed(video_view)
        )

No pseudo target is created.

### 3.2 Relevance target and loss

For each observed label:

    E_audio = BCE element from audio view
    E_video = BCE element from video view

Detach these errors when constructing relevance targets.

Target relevance:

    target_B_v =
        softmax_v(-E_v)

over available views.

Do not define target relevance for unknown labels.

Relevance loss:

    mean squared error between predicted relevance and detached target relevance

only over observed labels and available views.

This corresponds to the V2L self-guided relevance principle.

### 3.3 Reconstruction

For each available view, MSE between reconstructed normalized source vector and normalized source target.

Average over available views.

### 3.4 Direct two-view posterior consistency

Because WSM has two views, use direct ordered-pair consistency rather than the paper's multi-view random perturbation mapping.

For target-aware head l in {audio,video}:

    KL(q_audio_to_l || q_video_to_l)
    KL(q_video_to_l || q_audio_to_l)

Average all valid terms.

This is `L_consis`.

### 3.5 Intra-cluster coherence

For each source view and each target-aware proposal:

    KL(proposal || aggregated_view_posterior)

Average.

This is `L_intra`.

### 3.6 Cross-view instance discrimination

Use L2-normalized aggregated posterior means.

For audio/video views of the same sample, treat as positives.

Different samples in batch are negatives.

Use the paper-style cross-view contrastive loss with `cross_temperature`.

This is `L_cross`.

If a batch cannot support at least two paired samples, return an exact differentiable zero for this term.

### 3.7 Total loss

Use:

    L =
        L_cls
        + relevance_weight * L_relevance
        + reconstruction_weight * L_reconstruction
        + beta_consistency * L_consis
        + sigma_intra * L_intra
        + gamma_cross * L_cross

No extra regularizer is allowed.

Expose diagnostics:

- loss_cls
- loss_relevance
- loss_reconstruction
- loss_consistency
- loss_intra
- loss_cross

## 4. Registration

Update:

- src/fusion/models/__init__.py
- src/fusion/loss/__init__.py
- src/chimera_plugin.py

Register:

- wsm_v2l_model
- wsm_v2l_loss

Preserve all existing registrations.

## 5. Base config

Create:

    configs/wsm_mm_pd_dep_v1/fusion/31_v2l_optuna_base_seed42.yaml

Fixed:

- seed 42;
- experiment_name `wsm_mm_pd_dep_v1`;
- canonical `wsm_av_fusion_datamodule`;
- batch size 8;
- exact frozen audio checkpoint;
- model `wsm_v2l_model`;
- loss `wsm_v2l_loss`;
- AdamW trainable-only optimizer;
- epochs 30;
- CUDA;
- mixed precision true;
- grad clip 0.5;
- DEV Mean checkpoint/early stop only;
- patience 6;
- min_delta 0.0005;
- complete required callbacks/loggers;
- DEV/TEST_NONE/TEST_SOFT/TEST_HARD monitoring.

## 6. Frozen Optuna search space

Create:

    configs/wsm_mm_pd_dep_v1/fusion/32_v2l_optuna_sweep.yaml

Required:

    method: optuna
    n_trials: 24
    study_name: wsm-v2l-safe-v1
    target:
      monitor: dev/mean_score
      mode: max

Sweep EXACTLY these 10 variables:

1. `model.params.latent_dim`
   - categorical: [96,128,160,192,256]

2. `model.params.encoder_hidden_dim`
   - categorical: [128,192,256,320]

3. `model.params.decoder_hidden_dim`
   - categorical: [128,192,256,320]

4. `model.params.active_hidden_dim`
   - categorical: [32,64,96,128]

5. `model.params.dropout`
   - float: [0.05,0.30]

6. `loss.params.beta_consistency`
   - log-float: [0.0001,1.0]

7. `loss.params.sigma_intra`
   - log-float: [0.0001,1.0]

8. `loss.params.gamma_cross`
   - log-float: [0.0001,1.0]

9. `optimizer.params.lr`
   - log-float: [0.00002,0.0004]

10. `optimizer.params.weight_decay`
    - log-float: [0.00001,0.05]

Fixed:

- relevance_weight = 1.0
- reconstruction_weight = 1.0
- view_aux_weight = 1.0
- cross_temperature = 0.1

Do not sweep anything else.

Every possible configuration must have <=3,000,000 trainable non-audio parameters.

## 7. Mandatory pre-sweep firewall

Before Optuna trial 1:

1. implement model/loss;
2. create base/sweep configs;
3. compile and register;
4. validate base config;
5. run Chimera sweep dry-run;
6. prove exact 10-variable search and DEV-only target;
7. verify frozen audio SHA;
8. prove frozen audio absent from optimizer groups;
9. instantiate min/max/worst-case parameter combinations and prove <=3M cap;
10. synthetic smoke:
    - exact zero-init equality to frozen audio;
    - finite proposal/posterior variances;
    - relevance weights finite, normalized, [0,1];
    - unknown labels structurally masked;
    - finite forward/loss/backward;
    - frozen audio gradients None;
    - video residual/mid branch wake up after one optimizer step;
11. direct-formula tests:
    - precision aggregation;
    - PoE aggregation;
    - relevance target;
    - consistency KL;
    - intra KL;
    - cross-view contrastive term;
12. canonical DEV initialization reproduction:
    - fresh model reproduces frozen audio D/P/Mean within 0.0005;
    - DEV only;
    - no Test iteration;
13. append frozen V2L adaptation/search manifest to PROGRESS_EN.md;
14. commit and push one firewall commit.

No metric-bearing trial may begin before the firewall commit exists on origin.

## 8. Exact production sweep command

Run exactly once:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml sweep \
      --base-config configs/wsm_mm_pd_dep_v1/fusion/31_v2l_optuna_base_seed42.yaml \
      --sweep-config configs/wsm_mm_pd_dep_v1/fusion/32_v2l_optuna_sweep.yaml \
      --sweep-name v2l-safe-v1 \
      --max-trials 24

Exactly 24 seed42 metric-bearing training trials are authorized.

No manually launched extra seed42 trials.

After the first metric-bearing trial:

- architecture frozen;
- loss semantics frozen;
- search space frozen.

## 9. Sweep audit

After completion record all 24 trials:

- trial number/id;
- generated config;
- exact sampled parameters;
- run name;
- best DEV Mean;
- selected epoch;
- trainable parameter count.

Produce top-5 by DEV Mean.

Test metrics may not be used for ranking.

## 10. Safe trial gate

A seed42 trial is safe only if:

- DEV Mean > `0.7878268765`;
- depression Score >= `0.7379183895`;
- Parkinson Score >= `0.8177353635`;
- frozen audio invariant passed;
- parameter cap passed.

After all 24 trials:

1. filter safe trials only by these DEV conditions;
2. if none, record:
   `NO SAFE AUDIO-BEATING V2L TRIAL`;
3. if one or more, select highest DEV Mean among safe trials;
4. exact tie: choose lower trainable parameter count.

Do not use Test/calibration for safe filtering or tie-break.

## 11. Freeze and confirm the selected safe trial

Only if a safe seed42 trial exists:

Create:

    33_v2l_selected_seed42.yaml
    34_v2l_confirm_seed43.yaml
    35_v2l_confirm_seed44.yaml

Config 33 must be an exact tracked copy of the selected generated trial config.

Configs 34/35 may differ only by:

- seed;
- run_name.

Run seed43 once and seed44 once.

No hyperparameter changes.

Maximum production invocations:

    26

## 12. Three-seed provisional safe-audio gate

The selected V2L configuration passes provisional confirmation only if:

- DEV Mean > frozen audio Mean on seeds42,43,44 individually;
- D Score >= `0.7379183895` on each seed;
- P Score >= `0.8177353635` on each seed;
- three-seed Mean > `0.7878268765`;
- three-seed D mean >= `0.7379183895`;
- three-seed P mean >= `0.8177353635`;
- checkpoint SHA256 values are pairwise distinct;
- Test had no influence.

Codex must not self-promote the model.

## 13. Required V2L diagnostics

For selected seed42 and confirmation runs if executed, record on DEV:

- mean relevance weight for audio/video by disease;
- relevance weight std by disease;
- distribution quantiles (10/50/90%) by disease/view;
- mean absolute video residual by disease;
- mean absolute mid-level residual by disease;
- audio wrong -> V2L correct count;
- audio correct -> V2L wrong count;
- all six loss components.

These diagnostics are explanatory only.

## 14. Test firewall

TEST_NONE/SOFT/HARD remain mandatory monitoring every epoch.

They may not affect:

- Optuna;
- model/search-space design;
- safe filtering;
- trial selection;
- tie-break;
- confirmation;
- checkpoint selection;
- task conclusion.

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

Only authorized V2L model/loss/registration/config/audit/PROGRESS files may differ.

## Acceptance criteria

TASK-005H-V2L-OPTUNA passes only if:

- branch exactly `codex/task-005h-v2l-optuna`;
- implementation preserves the frozen V2L-WSM contract;
- exact audio checkpoint remains frozen;
- zero-init reproduces audio baseline;
- direct two-view consistency is used rather than invalid two-view perturbation diversity;
- unknown labels remain masked;
- no pseudo labels;
- exact 10-variable search frozen before trial 1;
- all possible configurations <=3M trainable non-audio params;
- native Chimera Optuna used;
- exactly 24 seed42 trials complete;
- objective only `dev/mean_score`;
- no Test-driven search;
- safe trial filtering uses only frozen DEV gate;
- confirmation executes only if a safe seed42 trial exists;
- confirmation configs preserve exact selected hyperparameters;
- at most 26 production invocations;
- complete diagnostics/evidence recorded;
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

- branch `codex/task-005h-v2l-optuna`;
- firewall commit SHA;
- final evidence commit SHA;
- pushed status;
- main/master untouched;
- frozen audio SHA;
- exact V2L-WSM equations/architecture summary;
- exact 10-variable search;
- sweep command/directory/manifest/study;
- completed trial count;
- top-5 trials;
- number of safe trials;
- selected safe trial or explicit none;
- selected seed42 D/P/Mean and audio deltas;
- selected config/checkpoint SHA if any;
- seed43/44 results if executed;
- three-seed mean/std and provisional gate if executed;
- relevance/residual/error-transition diagnostics;
- parameter counts;
- no Test-driven decision;
- no post-hoc search-space change;
- no pseudo labels;
- no Stage6/7/Text/Final Test;
- current optimization status.

Stop after TASK-005H-V2L-OPTUNA.
