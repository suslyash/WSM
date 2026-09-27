# TASK-005H-TRIPLE-OPTUNA: Equal-Budget Optuna Search for Candidate B, R3-B, and R4 RA-STCH

## Authority and branch

This task is the active owner-authorized optimization bundle from OWNER/MANAGER-OVERRIDE-054.

Required branch:

    codex/task-005h-triple-optuna

Start from current `origin/main`, including manager authorization commit:

    c118b11dca845165659d8297d483f8d6521b509f

Create exactly one task branch from that `origin/main`.

This task supersedes all earlier unexecuted TASK-005H variants.

## Goal

Run a fair, equal-budget native Chimera Optuna comparison of exactly three already-trained WSM model families:

1. **Candidate B**
   - model: `wsm_av_audio_query_temporal_video_model`
   - strong frozen historical temporal-audio anchor + audio-query temporal-video attention + residual correction.

2. **R3-B**
   - model: `wsm_av_r3_disease_query_model`
   - loss: `wsm_r3_aux_agreement_loss`
   - corrected disease-query multimodal fusion + auxiliary unimodal agreement.

3. **R4 RA-STCH**
   - model: `wsm_av_r3_disease_query_model`
   - data: frozen semantic pseudo cache through `wsm_ramps_semantic_datamodule`
   - loss: `wsm_r4_ramps_balance_loss` with `mode: ra_stch`
   - frozen pseudo warm-up + DEV-only RA-STCH controller.

Run exactly 20 seed42 Optuna trials for each family.

Total production training invocations for full acceptance:

    60

Do NOT run seed43/44 in this task.

The output of this task is search evidence for a later manager family-selection decision, not final promotion.

## Required reading

Read in this exact order:

1. AGENTS.md
2. docs/README.md
3. docs/PROJECT_REQUIREMENTS.md
4. docs/PLAN.md
5. docs/PROGRESS_EN.md through OWNER/MANAGER-OVERRIDE-054
6. docs/NEXT_TASK_EN.md
7. src/fusion/models/frozen_audio_temporal_adapter.py
8. src/fusion/models/av_audio_query_temporal_video.py
9. src/fusion/models/av_r3_disease_query.py
10. src/fusion/loss/r3_aux_agreement_loss.py
11. src/fusion/loss/r4_ramps_balance_loss.py
12. src/common/callbacks/wsm_r4_balance_callback.py
13. src/common/callbacks/wsm_pseudo_scale_warmup_callback.py
14. src/chimera_plugin.py
15. configs/wsm_mm_pd_dep_v1/fusion/05_audio_query_temporal_video.yaml
16. configs/wsm_mm_pd_dep_v1/fusion/08_r3_b_agreement_seed42.yaml
17. configs/wsm_mm_pd_dep_v1/fusion/16_r4_ra_stch_seed42.yaml

Also read installed Chimera ML Optuna support:

- docs/en/user-guide/sweeps.md
- src/chimera_ml/training/sweep.py
- src/chimera_ml/cli.py
- src/chimera_ml/utils/sweep.py

Do not modify Chimera ML.

## Frozen factual baselines

Historical audio reference:

- D Score: `0.7479183895`
- P Score: `0.8277353635`
- Mean: `0.7878268765`
- exact checkpoint SHA256:
  `0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2`

Previously observed family baselines:

Candidate B seed42:
- D `0.708929`
- P `0.909719`
- Mean `0.809324`

Corrected R3-B:
- seed42 D/P/Mean `0.695808/0.893824/0.794816`
- seed43 `0.694349/0.841176/0.767762`
- seed44 `0.698378/0.838989/0.768684`
- three-seed mean `0.696178/0.857996/0.777087`

Corrected R4 RA-STCH seed42:
- D `0.700992`
- P `0.864298`
- Mean `0.782645`

These are context only. They may not be used to modify the frozen search spaces after trial 1.

## Allowed tracked files

Codex may add/modify only:

- configs/wsm_mm_pd_dep_v1/fusion/31_optuna_candidate_b_base.yaml
- configs/wsm_mm_pd_dep_v1/fusion/32_optuna_candidate_b_sweep.yaml
- configs/wsm_mm_pd_dep_v1/fusion/33_optuna_r3b_base.yaml
- configs/wsm_mm_pd_dep_v1/fusion/34_optuna_r3b_sweep.yaml
- configs/wsm_mm_pd_dep_v1/fusion/35_optuna_r4_ra_stch_base.yaml
- configs/wsm_mm_pd_dep_v1/fusion/36_optuna_r4_ra_stch_sweep.yaml
- scripts/common/audit_triple_optuna.py
- docs/PROGRESS_EN.md

The audit script is optional.

No source model/loss/callback code may change.

## Forbidden actions

- Do not modify `src/audio`.
- Do not modify `src/video`.
- Do not modify any model implementation.
- Do not modify any loss implementation.
- Do not modify any DataModule.
- Do not modify any callback.
- Do not modify `src/chimera_plugin.py`.
- Do not modify Chimera ML.
- Do not create a fourth family.
- Do not add V2L, URDF, CTRL, DCSI, TACVI, CDSA, or another literature model.
- Do not unfreeze/tune the frozen historical audio model.
- Do not regenerate or alter the R2 semantic pseudo cache.
- Do not alter pseudo acceptance/reliability fields.
- Do not alter canonical data/splits/caches.
- Do not use task/corpus/split/Test identity as model input.
- Do not use Test metrics as an Optuna target, sampler input, search-space edit, ranking criterion, safe filter, or family-choice criterion.
- Do not run seed43/44.
- Do not run more than 20 Optuna trials per family.
- Do not manually run extra seed42 configs.
- Do not change any search space after the first metric-bearing trial of the whole bundle.
- Do not start Stage 6/7, Text/Description, or Final Test.

## Shared training protocol

All three base configs must preserve the accepted family-specific semantics and use:

- seed: 42
- experiment_name: `wsm_mm_pd_dep_v1`
- epochs: 30
- CUDA
- mixed precision: true
- grad clip: 0.5
- required four DEV/Test monitoring streams
- checkpoint monitor: only `dev/mean_score`, mode=max
- early stopping monitor: only `dev/mean_score`, mode=max
- patience: 6
- min_delta: 0.0005
- required snapshot/summary/segment metrics/console/MLflow instrumentation

Preserve each accepted family's batch size and data path unless explicitly stated below:

- Candidate B: batch_size 8, canonical `wsm_av_fusion_datamodule`;
- R3-B: batch_size 32, canonical `wsm_av_fusion_datamodule`;
- R4 RA-STCH: batch_size 32, `wsm_ramps_semantic_datamodule` with the exact frozen pseudo cache.

## 1. Candidate B base and sweep

Create:

    configs/wsm_mm_pd_dep_v1/fusion/31_optuna_candidate_b_base.yaml
    configs/wsm_mm_pd_dep_v1/fusion/32_optuna_candidate_b_sweep.yaml

Base must mirror accepted config 05 except run_name and searchable base values.

Model remains exactly:

    wsm_av_audio_query_temporal_video_model

Loss remains:

    wsm_masked_sparse_loss

Optimizer remains:

    wsm_trainable_adamw_optimizer

Exact frozen audio checkpoint remains unchanged.

### Candidate B search space — EXACTLY 6 variables

1. `model.params.hidden_dim`
   - categorical: [96, 128, 160, 192, 224, 256]

2. `model.params.num_heads`
   - categorical: [2, 4, 8]

3. `model.params.residual_hidden_dim`
   - categorical: [64, 96, 128, 160, 192, 256]

4. `model.params.dropout`
   - float: [0.05, 0.35]

5. `optimizer.params.lr`
   - log-float: [0.00002, 0.0004]

6. `optimizer.params.weight_decay`
   - log-float: [0.00001, 0.05]

All hidden_dim choices are divisible by all num_heads choices.

Every sampled Candidate B model must have:

    <= 1,000,000

trainable fusion parameters excluding frozen audio.

Sweep required fields:

    method: optuna
    n_trials: 20
    study_name: wsm-candidate-b-optuna-v1
    target:
      monitor: dev/mean_score
      mode: max

## 2. R3-B base and sweep

Create:

    configs/wsm_mm_pd_dep_v1/fusion/33_optuna_r3b_base.yaml
    configs/wsm_mm_pd_dep_v1/fusion/34_optuna_r3b_sweep.yaml

Base must mirror corrected accepted config 08 except run_name and searchable base values.

Model remains exactly:

    wsm_av_r3_disease_query_model

Loss remains exactly:

    wsm_r3_aux_agreement_loss

No pseudo labels.

### R3-B search space — EXACTLY 7 variables

1. `model.params.hidden_dim`
   - categorical: [128, 160, 192, 224, 256]

2. `model.params.gate_hidden_dim`
   - categorical: [96, 128, 160, 192, 256]

3. `model.params.dropout`
   - float: [0.05, 0.35]

4. `loss.params.aux_weight`
   - float: [0.05, 0.50]

5. `loss.params.agreement_weight`
   - log-float: [0.01, 0.50]

6. `optimizer.params.lr`
   - log-float: [0.00002, 0.0004]

7. `optimizer.params.weight_decay`
   - log-float: [0.00001, 0.05]

Keep fixed:

- audio_feature_dim=768
- video_feature_dim=512
- num_tasks=2
- loss eps existing/default semantics

Every sampled R3-B model must have:

    <= 736,004

trainable parameters.

Sweep:

    method: optuna
    n_trials: 20
    study_name: wsm-r3b-optuna-v1
    target:
      monitor: dev/mean_score
      mode: max

## 3. R4 RA-STCH base and sweep

Create:

    configs/wsm_mm_pd_dep_v1/fusion/35_optuna_r4_ra_stch_base.yaml
    configs/wsm_mm_pd_dep_v1/fusion/36_optuna_r4_ra_stch_sweep.yaml

Base must mirror corrected accepted config 16 except run_name and searchable base values.

Data must remain:

    wsm_ramps_semantic_datamodule

Exact pseudo cache:

    /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt

Required SHA256:

    17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945

Model remains:

    wsm_av_r3_disease_query_model

Loss remains:

    wsm_r4_ramps_balance_loss

Mode remains:

    ra_stch

Pseudo warm-up remains exactly:

- observed_only_epochs: 3
- ramp_epochs: 5
- final_scale: 1.0

### R4 RA-STCH search space — EXACTLY 10 variables

1. `model.params.hidden_dim`
   - categorical: [128, 160, 192, 224, 256]

2. `model.params.gate_hidden_dim`
   - categorical: [96, 128, 160, 192, 256]

3. `model.params.dropout`
   - float: [0.05, 0.35]

4. `loss.params.aux_weight`
   - float: [0.05, 0.50]

5. `loss.params.agreement_weight`
   - log-float: [0.01, 0.50]

6. `loss.params.tau`
   - log-float: [0.03, 0.30]

7. `loss.params.progress_temperature`
   - log-float: [0.10, 0.75]

8. `loss.params.controller_ema`
   - float: [0.50, 0.95]

9. `optimizer.params.lr`
   - log-float: [0.00002, 0.0004]

10. `optimizer.params.weight_decay`
    - log-float: [0.00001, 0.05]

Keep fixed exactly:

- mode: `ra_stch`
- pseudo_scale initial: 0.0
- progress_reference_depression: 0.697035
- progress_reference_parkinson: 0.852104
- grad_ema: 0.9
- reliability_ema: 0.9
- weight_min: 0.2
- weight_max: 0.8
- eps: 1e-8
- pseudo warm-up schedule above

Every sampled R4 model must remain:

    <= 736,004

trainable model parameters.

Sweep:

    method: optuna
    n_trials: 20
    study_name: wsm-r4-ra-stch-optuna-v1
    target:
      monitor: dev/mean_score
      mode: max

## 4. One global pre-run firewall

Before ANY metric-bearing trial of ANY family:

1. create all six configs;
2. freeze all three search spaces exactly as above;
3. validate all three base configs;
4. run Chimera sweep `--dry-run` for all three sweep configs;
5. verify each target is exactly `dev/mean_score`, max;
6. verify each sweep has exactly 20 trials;
7. verify variable counts are exactly 6 / 7 / 10;
8. verify Candidate B exact frozen audio checkpoint SHA;
9. verify Candidate B frozen audio remains absent from optimizer groups;
10. verify R4 pseudo-cache exact SHA;
11. prove R3/R4 pseudo/unknown-label contracts remain unchanged;
12. instantiate all necessary boundary combinations and prove:
    - Candidate B <=1M trainable fusion params;
    - R3-B <=736004 trainable params;
    - R4 model <=736004 trainable params;
13. registered TRAIN-only forward/loss/backward smoke for each base:
    - finite outputs/loss/gradients;
    - correct [B,2] output;
    - unknown labels masked;
    - Candidate B frozen audio grads None;
    - R4 accepted pseudo fields detached and observed truth overrides pseudo;
    - R4 controller sees DEV only;
14. append complete firewall/search manifest to PROGRESS_EN.md;
15. commit and push one firewall commit.

All three search spaces are immutable after this firewall.

No first sweep may begin until the firewall commit exists on origin.

## 5. Exact production order and commands

Run sequentially in this exact order.

### A. Candidate B — 20 trials

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml sweep \
      --base-config configs/wsm_mm_pd_dep_v1/fusion/31_optuna_candidate_b_base.yaml \
      --sweep-config configs/wsm_mm_pd_dep_v1/fusion/32_optuna_candidate_b_sweep.yaml \
      --sweep-name candidate-b-optuna-v1 \
      --max-trials 20

### B. R3-B — 20 trials

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml sweep \
      --base-config configs/wsm_mm_pd_dep_v1/fusion/33_optuna_r3b_base.yaml \
      --sweep-config configs/wsm_mm_pd_dep_v1/fusion/34_optuna_r3b_sweep.yaml \
      --sweep-name r3b-optuna-v1 \
      --max-trials 20

### C. R4 RA-STCH — 20 trials

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml sweep \
      --base-config configs/wsm_mm_pd_dep_v1/fusion/35_optuna_r4_ra_stch_base.yaml \
      --sweep-config configs/wsm_mm_pd_dep_v1/fusion/36_optuna_r4_ra_stch_sweep.yaml \
      --sweep-name r4-ra-stch-optuna-v1 \
      --max-trials 20

Exactly 60 completed metric-bearing trials are required for full task acceptance.

Do not run extra trials to replace poor metrics.

If a sweep has a genuine infrastructure/configuration blocker before its first metric-bearing trial, a narrow config-only repair is allowed only if it does not change any search-space semantics.

After the first metric-bearing trial of the bundle, do not change any family search space or accepted model/loss semantics.

If a later blocker prevents exact completion, preserve evidence and stop/report; do not silently substitute runs.

## 6. Frozen per-family reporting rules

For every family, record all 20 trials with:

- Optuna trial number/id;
- generated config path;
- exact sampled values;
- run name;
- selected epoch;
- DEV D Score;
- DEV P Score;
- DEV Mean;
- trainable parameter count;
- selected checkpoint path/SHA256 if available.

Produce top-5 by DEV Mean.

### Objective winner

For each family:

    exact highest DEV Mean trial

is the objective winner.

No lower trial may replace it.

### Safe set

A trial is in the safe set iff all are true:

- DEV Mean > `0.7878268765`;
- D >= `0.7379183895`;
- P >= `0.8177353635`.

For each family report:

- safe trial count;
- safe winner = highest DEV Mean inside the safe set;
- if empty: `NO SAFE TRIAL`.

### Strict non-regression diagnostic

Also count trials satisfying:

- Mean > `0.7878268765`;
- D >= `0.7479183895`;
- P >= `0.8277353635`.

This is diagnostic only.

## 7. Cross-family comparison report

After all three sweeps, produce one frozen DEV-only comparison table with:

- family;
- historical baseline Mean before Optuna;
- Optuna objective-winner D/P/Mean;
- delta D/P/Mean vs frozen audio;
- safe-trial count;
- safe-winner D/P/Mean if any;
- strict non-regression trial count;
- best trial parameter count;
- number of completed trials.

Also record:

- absolute highest DEV Mean across all 60 trials;
- highest safe DEV Mean across all 60 trials, if any;
- whether each family improved its own historical seed42 Mean.

Do NOT choose the winning family.

The manager will make the next family-selection decision after audit.

## 8. Test firewall

TEST_NONE/SOFT/HARD remain mandatory per-epoch monitoring.

They MUST NOT affect:

- Optuna objective;
- sampler behavior;
- search ranges;
- search continuation;
- trial ranking;
- objective winner;
- safe set;
- strict diagnostic;
- family comparison;
- future-family recommendation.

Do not use Test to explain why one family should be chosen.

## 9. Scope verification

Run exactly:

    git diff --check
    git diff origin/main -- src/audio
    git diff origin/main -- src/video
    git diff origin/main -- src/fusion/models
    git diff origin/main -- src/fusion/loss
    git diff origin/main -- src/fusion/data
    git diff origin/main -- src/common
    git diff origin/main -- src/chimera_plugin.py
    git status --short
    git diff --stat origin/main...HEAD
    git log -12 --oneline --decorate

All source diffs above must be empty.

Only the six new configs, optional audit script, and PROGRESS_EN.md may differ.

## Acceptance criteria

TASK-005H-TRIPLE-OPTUNA passes only if:

- branch exactly `codex/task-005h-triple-optuna`;
- all three families are unchanged existing implementations;
- all six configs are frozen before trial 1;
- one global firewall commit is pushed before production;
- Candidate B search has exactly 6 frozen variables;
- R3-B search has exactly 7 frozen variables;
- R4 RA-STCH search has exactly 10 frozen variables;
- native Chimera Optuna is used;
- exactly 20 completed seed42 trials per family;
- exactly 60 production training invocations total;
- no seed43/44;
- objective only `dev/mean_score`, max;
- no Test-driven decision;
- parameter caps hold;
- frozen audio invariant holds;
- R4 pseudo cache and semantic reliability contract remain exact;
- objective winner and safe winner are reported separately;
- no family is promoted/selected by Codex;
- no source file changed;
- no Stage6/7/Text/Final Test;
- complete evidence is committed and branch pushed.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch `codex/task-005h-triple-optuna`;
- firewall commit SHA;
- final evidence commit SHA;
- pushed-to-origin status;
- main/master untouched;
- all source diffs empty;
- frozen audio SHA;
- R4 pseudo-cache SHA;
- exact 6/7/10 search spaces;
- exact three sweep commands;
- sweep directories/manifests/study names;
- 20/20/20 completed counts and 60 total;
- every family's top-5;
- every family's objective winner D/P/Mean;
- every family's safe count and safe winner or none;
- every family's strict non-regression count;
- historical-baseline-versus-Optuna deltas;
- cross-family DEV-only table;
- absolute highest Mean among all 60;
- absolute highest safe Mean among all 60 if any;
- parameter-count proof/results;
- no Test-driven decision;
- no post-hoc search-space change;
- no seed43/44;
- no new family/model;
- no Stage6/7/Text/Final Test;
- current Stage-5 optimization status.

Stop after TASK-005H-TRIPLE-OPTUNA.
