# TASK-005H-OPTUNA: Bounded Chimera Optuna Search for a Safe Audio-Beating Temporal A+V Model

## Authority and branch

This task is an explicit owner-authorized optimization sprint that supersedes the unexecuted TASK-006A.

Required branch:

    codex/task-005h-optuna

Start from current `origin/main`, which includes:

- PR #44 merge `de05a2233d318add54b71121193e7e412d0b102a`;
- owner/manager Optuna override `5c2413038bf6a711a72c21256b5b9c946f29b312`.

Create exactly one task branch from that `origin/main`.

No `codex/task-006a` branch existed when the override was issued. TASK-006A is superseded before execution, not failed.

## Goal

Use Chimera ML's native Optuna sweep to search one bounded, configurable audio-first temporal-video fusion family for a model that safely beats the frozen historical audio reference.

The search must keep the exact historical temporal-audio model frozen and use video only as a zero-initialized additive correction.

The search is intentionally broader than prior manager tasks. Codex is allowed to choose the exact Optuna search space within manager hard bounds before the first metric-bearing trial.

Exactly 20 seed42 Optuna trials are authorized.

If and only if the highest-DEV-Mean trial passes the frozen safe-audio screen, run that exact configuration at true seeds43 and 44.

Maximum production training invocations:

    22

## Required reading

Read in this order:

1. AGENTS.md
2. docs/README.md
3. docs/PROJECT_REQUIREMENTS.md
4. docs/PLAN.md, especially Stage 4 safe-audio gate and Section 7 Promotion Rule
5. docs/PROGRESS_EN.md through OWNER/MANAGER-OVERRIDE-050
6. docs/NEXT_TASK_EN.md
7. docs/SOTA_REVIEW_EN.md relevant fusion/negative-transfer sections
8. src/fusion/models/frozen_audio_temporal_adapter.py
9. src/fusion/models/av_audio_query_temporal_video.py
10. src/fusion/models/av_audio_confidence_gated.py
11. src/fusion/models/av_audio_first_zero_residual.py
12. src/common/optimizers/wsm_trainable_adamw.py
13. src/chimera_plugin.py
14. configs/wsm_mm_pd_dep_v1/fusion/05_audio_query_temporal_video.yaml

Also read the installed Chimera ML Optuna documentation/source:

- docs/en/user-guide/sweeps.md
- src/chimera_ml/training/sweep.py
- src/chimera_ml/cli.py

Do not modify Chimera ML.

## Owner-authorized rule changes for this task

For this task only:

- a bounded Optuna hyperparameter/architecture search is authorized despite the normal broad-search restriction;
- Stage 6 is paused before execution;
- Stage 5 is reopened for this sprint;
- architectural depth and fusion hyperparameters may be tuned;
- the frozen audio architecture/checkpoint itself MUST NOT be tuned or modified.

All other project invariants remain active.

## Frozen audio anchor and comparator

Exact checkpoint:

    logs/wsm_audio_segment_wavlm_base_l9_pool4/multitask_audio_mamba_2026-08-19_14-12_audio_mamba_segment_model_wsm_audio_models-e0ce-006_ea8c8d77/checkpoints/epoch=4_dev_mean_score=0.7878.pt

Required SHA256:

    0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2

Frozen DEV reference:

- depression Score: `0.7479183895`
- Parkinson Score: `0.8277353635`
- Mean: `0.7878268765`

Historical Candidate-B evidence:

- seed42 DEV Mean `0.809324`;
- depression delta vs audio `-0.038989`;
- Parkinson delta vs audio `+0.081983`.

This motivates task-specific video-correction control but does not authorize manual cherry-picking after the sweep.

## Allowed tracked files

Codex may modify/add only:

- src/fusion/models/av_audio_query_temporal_video_tunable.py
- src/fusion/models/__init__.py
- src/chimera_plugin.py
- configs/wsm_mm_pd_dep_v1/fusion/31_audio_first_optuna_base_seed42.yaml
- configs/wsm_mm_pd_dep_v1/fusion/32_audio_first_optuna_sweep.yaml
- configs/wsm_mm_pd_dep_v1/fusion/33_audio_first_optuna_best_seed42.yaml
- configs/wsm_mm_pd_dep_v1/fusion/34_audio_first_optuna_confirm_seed43.yaml
- configs/wsm_mm_pd_dep_v1/fusion/35_audio_first_optuna_confirm_seed44.yaml
- scripts/common/audit_audio_first_optuna_sweep.py
- docs/PROGRESS_EN.md

No other tracked file may change.

The audit script is optional; if unnecessary, do not create it.

## Forbidden actions

- Do not modify anything under `src/audio`.
- Do not modify anything under `src/video`.
- Do not modify existing accepted fusion models.
- Do not modify any DataModule.
- Do not modify any existing loss.
- Do not modify Chimera ML.
- Do not unfreeze the historical audio model.
- Do not tune the audio architecture/checkpoint.
- Do not use pseudo labels in this search.
- Do not use corpus identity, task_id, observed masks, labels, split identity, or Test protocol identity as model input features.
- Do not change the canonical split/data/cache.
- Do not use TEST_NONE/SOFT/HARD as an Optuna objective or for search-space edits, trial ranking, acceptance, or continuation.
- Do not alter the search space after the first metric-bearing Optuna trial begins.
- Do not add a second model family after seeing results.
- Do not manually choose a lower-ranked trial because its task balance looks nicer.
- Do not run more than 20 Optuna trials.
- Do not run confirmation seeds unless the exact highest-Mean trial passes the safe screen.
- Do not start Stage 6, Stage 7, Text/Description, or Final Test.
- Do not make significance/final-model claims.

## 1. Tunable model family

Implement and register:

    src/fusion/models/av_audio_query_temporal_video_tunable.py

Registry key:

    wsm_av_audio_query_temporal_video_tunable_model

The model is one configurable superset of the accepted Candidate-B mechanism.

### Invariants

- Use `FrozenAudioTemporalAdapter` with the exact frozen checkpoint.
- Frozen audio stays eval-only, `requires_grad=False`, under `torch.no_grad()`, and absent from optimizer groups.
- Consume canonical temporal video features and masks.
- Require audio availability.
- Missing video must produce exact frozen-audio fallback.
- Two independent disease logits.
- No external task/corpus ID input.
- All floating outputs finite.
- Every possible search configuration must have <=1,000,000 trainable fusion parameters excluding frozen audio.

### Architecture

The model MUST support these configurable concepts:

1. video projection from 512 to `hidden_dim`;
2. optional temporal video encoder stack with `temporal_layers` in [0,3];
3. audio-task-feature query projection from frozen audio hidden 192 to `hidden_dim`;
4. one audio-query cross-attention block over temporal video;
5. task-specific residual MLP with configurable depth;
6. optional audio-confidence gate MAY be supported if Codex chooses to include it before the firewall;
7. fixed per-task correction scales:
   - depression scale;
   - Parkinson scale;
8. zero-initialized final correction layer for every task.

Required final equation:

    final_logit_t =
        audio_base_logit_t
        + video_available
        * correction_scale_t
        * optional_gate_t
        * residual_t

where `optional_gate_t = 1` if confidence gating is disabled.

At construction, before any optimizer step:

    preds == audio_base_logits

must hold exactly for every sample because the final correction layer is zero-initialized.

### Temporal encoder contract

If `temporal_layers == 0`, bypass the video self-attention encoder.

If `temporal_layers > 0`, use a mask-aware Transformer/self-attention stack over projected video features. The exact PyTorch block organization is Codex-chosen before the firewall, but MUST:

- use `hidden_dim`;
- use `num_heads`;
- use the frozen `temporal_ff_multiplier`;
- use GELU;
- use configured dropout;
- respect video padding mask;
- keep unavailable video from affecting outputs.

### Residual depth contract

`residual_layers` means 1–3 hidden MLP transformations before the final zero-initialized scalar layer.

Codex may choose the exact repeated block pattern before the firewall, but it MUST be identical across tasks except parameters are task-specific.

### Task-specific correction scale

The scale is a fixed non-trainable config scalar.

Depression scale `0.0` is explicitly legal.

If depression scale is `0.0`, depression output must remain exactly equal to frozen audio throughout training.

This is an intentionally authorized task-conditioned modality-use hypothesis, not a post-hoc manual head replacement.

## 2. Manager hard bounds for Codex-designed search space

Codex has autonomy to select and freeze 7–10 actual Optuna variables from the pool below.

Required variables:

1. `model.params.hidden_dim`
2. `model.params.temporal_layers`
3. `model.params.residual_layers`
4. `model.params.dropout`
5. `optimizer.params.lr`
6. `model.params.depression_correction_scale`
7. `model.params.parkinson_correction_scale`

Optional variables:

- `model.params.num_heads`
- `model.params.temporal_ff_multiplier`
- `model.params.residual_hidden_dim`
- `optimizer.params.weight_decay`
- optional confidence-gate boolean/hidden width if implemented

Hard search bounds:

- hidden_dim choices: subset of `[96,128,160,192,224,256]`
- num_heads choices: subset of `[2,4,8]`
- temporal_layers: integer/categorical in `[0,3]`
- temporal_ff_multiplier: subset of `[2,3,4]`
- residual_hidden_dim: subset of `[64,96,128,160,192,256,320]`
- residual_layers: integer/categorical in `[1,3]`
- dropout: float/categorical entirely inside `[0.05,0.35]`
- depression_correction_scale: entirely inside `[0.0,0.40]`
- Parkinson_correction_scale: entirely inside `[0.25,1.50]`
- AdamW lr: log space entirely inside `[2e-5,4e-4]`
- weight_decay: log space entirely inside `[1e-5,5e-2]`

All hidden_dim choices used in the sweep MUST be divisible by all sampled num_heads choices.

Before the first trial, Codex must record:

- exact selected search variables;
- exact ranges/choices/types;
- rationale;
- parameter-count proof that every possible sampled model is <=1,000,000 trainable fusion parameters.

After this freeze, no search-space edit is allowed.

## 3. Base training config

Create:

    configs/wsm_mm_pd_dep_v1/fusion/31_audio_first_optuna_base_seed42.yaml

Fixed fields:

- seed: 42
- experiment_name: `wsm_mm_pd_dep_v1`
- run_name: `audio_first_optuna_base_seed42`
- data: canonical `wsm_av_fusion_datamodule`
- batch_size: 8
- model: `wsm_av_audio_query_temporal_video_tunable_model`
- exact frozen audio checkpoint
- loss: `wsm_masked_sparse_loss`
- optimizer: `wsm_trainable_adamw_optimizer`
- epochs: 30
- CUDA
- mixed precision true
- grad clip 0.5
- checkpoint monitor only `dev/mean_score`, mode=max
- early stopping only `dev/mean_score`, patience 6, min_delta 0.0005
- required snapshot/summary/segment metrics/console/MLflow instrumentation
- four DEV/Test monitoring streams

Base model/searchable parameter values may be chosen by Codex inside the hard bounds before firewall.

No pseudo DataModule/loss/warm-up.

## 4. Optuna sweep config

Create:

    configs/wsm_mm_pd_dep_v1/fusion/32_audio_first_optuna_sweep.yaml

Required:

    method: optuna
    n_trials: 20
    study_name: wsm-audio-first-safe-v1
    target:
      monitor: dev/mean_score
      mode: max

Use Chimera typed parameter specs.

The sweep target MUST be exactly:

    dev/mean_score

No task-specific or Test objective is permitted.

If storage is configured, use only a project-local SQLite path under `logs/` and `load_if_exists: false`.

## 5. Mandatory firewall before the Optuna sweep

Before the first metric-bearing trial:

1. implement/register the tunable model;
2. create base and sweep configs;
3. freeze exact model block organization;
4. freeze exact 7–10 variable search space;
5. compile new model/plugin;
6. validate the base config;
7. run:
   `chimera-ml sweep ... --dry-run`
   and verify the typed search space/target;
8. strict-load the frozen audio checkpoint and verify SHA;
9. build representative boundary configurations covering:
   - minimum model size;
   - maximum model size;
   - all num_heads/hidden divisibility combinations used;
10. prove every boundary configuration <=1,000,000 trainable fusion parameters;
11. prove frozen audio is absent from optimizer groups;
12. synthetic both/video-missing smokes:
   - exact zero-init equality to audio;
   - exact video-missing fallback;
   - finite forward/loss/backward;
   - after one optimizer step, correction path can wake up;
   - frozen audio gradients remain None;
13. canonical DEV initialization reproduction:
   - fresh zero-init model reproduces frozen audio D/P/Mean within tolerance 0.0005;
   - DEV only, no Test iteration for this gate;
14. append complete frozen search manifest/evidence to PROGRESS_EN.md;
15. commit and push one firewall commit.

No Optuna trial may start before that firewall commit exists on origin.

## 6. Exact Optuna production command

Run exactly once:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml sweep       --base-config configs/wsm_mm_pd_dep_v1/fusion/31_audio_first_optuna_base_seed42.yaml       --sweep-config configs/wsm_mm_pd_dep_v1/fusion/32_audio_first_optuna_sweep.yaml       --sweep-name audio-first-safe-v1       --max-trials 20

Exactly 20 completed trial training invocations are authorized.

Do not manually launch extra trial configs.

If the sweep aborts because of an implementation/configuration defect before a meaningful trial completes, fix only the defect if it remains inside allowed scope and no search-space semantics change. If a metric-bearing trial has completed, do not change the search space or model semantics; stop and report any subsequent blocker.

## 7. Sweep evidence and ranking

After the sweep:

- identify the sweep directory and manifest;
- verify manifest method `optuna`;
- verify exactly 20 completed trials;
- verify objective `dev/mean_score`, mode=max;
- record every trial:
  - trial number/id;
  - exact generated config path;
  - exact sampled params;
  - run name;
  - best target value;
  - target epoch;
  - trainable fusion parameter count if available from snapshot/config reconstruction;
- produce top-5 trial table by DEV Mean.

The ONLY selected candidate is the manifest/study highest-DEV-Mean trial.

Do not substitute a lower-ranked trial.

## 8. Frozen safe-audio screen

Highest-Mean seed42 trial passes only if ALL are true:

- DEV Mean > `0.7878268765`
- depression DEV Score >= `0.7379183895`
- Parkinson DEV Score >= `0.8177353635`
- frozen audio remained exact/frozen
- parameter cap satisfied
- no Test-driven choice occurred

Record deltas versus audio.

If it fails any condition:

- record the search as a valid negative optimization result;
- do NOT create/run seed43/44 confirmation;
- stop after documentation/evidence commit.

## 9. Freeze selected config if seed42 passes

If and only if Section 8 passes:

Create:

    configs/wsm_mm_pd_dep_v1/fusion/33_audio_first_optuna_best_seed42.yaml

It must be an exact tracked copy of the selected generated trial config, preserving all sampled hyperparameters.

Record:

- source trial config path;
- full selected config SHA256;
- selected model trainable parameter count;
- selected checkpoint path/SHA256;
- selected DEV D/P UAR/MF1/Score and Mean.

Then create:

    34_audio_first_optuna_confirm_seed43.yaml
    35_audio_first_optuna_confirm_seed44.yaml

These MUST differ from config 33 only in:

- top-level seed;
- run_name.

Do not change any hyperparameter.

## 10. Conditional confirmation

Only if seed42 passed the safe screen, run exactly:

1. seed43 confirmation;
2. seed44 confirmation.

Use normal `chimera-ml train`, not another sweep.

Checkpoint/early stopping remain DEV-only.

For each seed record selected checkpoint SHA256 and full DEV D/P/Mean.

### Three-seed provisional success gate

The selected Optuna configuration is a provisional safe audio-beating candidate only if:

- safe-audio screen passes separately on seeds42,43,44;
- three-seed DEV Mean > `0.7878268765`;
- three-seed depression mean >= `0.7379183895`;
- three-seed Parkinson mean >= `0.8177353635`;
- selected checkpoint identities are distinct across true seeds;
- no Test metric influenced selection or continuation.

If any fails, preserve the result as non-promoted.

Codex must not declare final promotion; manager decides.

## 11. Test firewall

TEST_NONE/SOFT/HARD remain mandatory epoch-level monitoring.

They MUST NOT be used by:

- Optuna target;
- sampler/search-space changes;
- trial ranking;
- safe screen;
- confirmation decision;
- architecture redesign;
- hyperparameter edits;
- checkpoint selection;
- final task conclusion.

All search/selection evidence is DEV-only.

## 12. Scope checks

Run:

    git diff --check
    git diff origin/main -- src/audio
    git diff origin/main -- src/video
    git diff origin/main -- src/fusion/data
    git diff origin/main -- src/fusion/loss
    git diff origin/main -- src/common
    git status --short
    git diff --stat origin/main...HEAD
    git log -12 --oneline --decorate

Only the allowed new tunable model/registration/config/audit/evidence files may differ.

## Acceptance criteria

TASK-005H-OPTUNA execution passes only if:

- branch is exactly `codex/task-005h-optuna`;
- exact frozen audio checkpoint remains unchanged/frozen;
- model starts exactly at audio function and supports exact video-missing fallback;
- search family obeys manager hard bounds;
- every possible sampled architecture respects <=1M trainable fusion params;
- exact search space is frozen/committed before first trial;
- Chimera native `method: optuna` sweep is used;
- exactly 20 seed42 Optuna trials complete;
- objective is only `dev/mean_score`, max;
- no Test-driven search/tuning;
- no pseudo labels;
- no existing source/model/data/loss modification outside allowed files;
- best trial is selected strictly by highest DEV Mean;
- no lower-ranked safe-looking trial substitution occurs;
- seed43/44 confirmation runs only if the exact best seed42 trial passes the frozen safe screen;
- confirmation configs are exact selected-config copies except seed/run_name;
- at most 22 production invocations occur;
- all evidence is recorded;
- branch is pushed;
- main/master untouched by Codex.

Passing this task gives the manager evidence to decide whether the optimization sprint has produced a safe multimodal model that genuinely beats the frozen audio reference.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch `codex/task-005h-optuna`;
- firewall commit SHA;
- final evidence commit SHA;
- pushed-to-origin status;
- main/master untouched;
- exact frozen audio SHA;
- exact frozen search space and number of variables;
- Chimera sweep command;
- sweep directory/manifest/study name;
- exactly 20 trial count;
- top-5 trials with sampled params and DEV Mean;
- exact best trial and seed42 D/P/Mean;
- audio deltas and safe-screen PASS/FAIL;
- selected config/checkpoint SHA if passed;
- seed43/44 confirmation results if executed;
- three-seed mean/std and safe-screen result if confirmations executed;
- model parameter counts;
- no Test-driven decision;
- no post-hoc search-space change;
- no pseudo labels;
- no Stage6/7/Text/Final Test;
- current Stage 5 optimization status.

Stop after TASK-005H-OPTUNA.
