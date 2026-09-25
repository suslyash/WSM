# TASK-005F-BUNDLE: Implement and Evaluate the Frozen R4 Task-Balancing Interaction Bundle

## Authority and branch

Required branch:

    codex/task-005f

Start from current `origin/main`, which includes:

- PR #42 merge `e4b48038b8d502d609a8d0daf11516311f168277`;
- manager R4 freeze commit `899fa8f085e88de6e8faf1b3491691c312689f27`.

Create exactly one task branch from that `origin/main`.

This task is one bounded R4 bundle. It may execute at most six production training invocations under the predeclared tree below. Do not return to the manager between authorized runs. Do not invent another variant.

## Goal

Evaluate whether task-balancing can rescue the full Stage-5 interaction composition after negative standalone R2 and non-promoted R3 results.

All R4 variants MUST use the same frozen full composition:

- exact R3 disease-query model;
- frozen R2 semantic pseudo-aware DataModule/cache;
- frozen R2 pseudo warm-up;
- frozen R3 auxiliary supervision/agreement terms;
- identical optimizer/training/instrumentation.

The only experimental difference across R4 seed42 variants is the task-level scalarization:

1. equal weights;
2. static STCH;
3. progress/PAGB-style comparator;
4. RA-STCH.

RA-STCH may continue to true seeds43/44 only under the frozen DEV rule.

This is an interaction test. It does NOT retroactively promote R2 or R3 as standalone components.

## Required reading

Read in this order:

1. AGENTS.md
2. docs/README.md
3. docs/PROJECT_REQUIREMENTS.md, especially Sections 2, 8, 9, 11, 13, 14
4. docs/PLAN.md, especially the Stage-5 loss equations, R4, experiment matrix, and promotion rule
5. docs/PROGRESS_EN.md through MANAGER-DECISION-044
6. docs/NEXT_TASK_EN.md
7. docs/SOTA_REVIEW_EN.md, especially PASAL/progress balancing
8. src/fusion/models/av_r3_disease_query.py
9. src/fusion/data/wsm_ramps_semantic_datamodule.py
10. src/fusion/loss/ramps_observed_pseudo_loss.py
11. src/fusion/loss/r3_aux_agreement_loss.py
12. src/common/callbacks/wsm_pseudo_scale_warmup_callback.py
13. src/common/callbacks/wsm_segment_callback.py
14. src/chimera_plugin.py
15. configs/wsm_mm_pd_dep_v1/fusion/08_r3_b_agreement_seed42.yaml
16. configs/wsm_mm_pd_dep_v1/fusion/05_ramps_r2_student_seed42.yaml

Use only active English docs/research references named above.

## Allowed tracked files

Codex may modify only:

- src/fusion/loss/r4_ramps_balance_loss.py
- src/common/callbacks/wsm_r4_balance_callback.py
- src/chimera_plugin.py
- configs/wsm_mm_pd_dep_v1/fusion/13_r4_equal_seed42.yaml
- configs/wsm_mm_pd_dep_v1/fusion/14_r4_static_stch_seed42.yaml
- configs/wsm_mm_pd_dep_v1/fusion/15_r4_progress_seed42.yaml
- configs/wsm_mm_pd_dep_v1/fusion/16_r4_ra_stch_seed42.yaml
- configs/wsm_mm_pd_dep_v1/fusion/17_r4_ra_stch_seed43.yaml
- configs/wsm_mm_pd_dep_v1/fusion/18_r4_ra_stch_seed44.yaml
- docs/PROGRESS_EN.md

No other tracked file may change.

## Forbidden actions

- Do not modify src/audio.
- Do not modify src/video.
- Do not modify any model.
- Do not modify any DataModule.
- Do not modify existing R2/R3 loss implementations.
- Do not regenerate/reselect pseudo labels or semantic thresholds.
- Do not change the frozen cache.
- Do not change R3 architecture.
- Do not change auxiliary/agreement coefficients.
- Do not change pseudo warm-up.
- Do not add text/description.
- Do not start Stage 6 or Stage 7.
- Do not run Final Test.
- Do not use TEST_NONE/SOFT/HARD for controller state, variant selection, branching, tuning, or stopping.
- Do not claim dynamic RA-STCH inherits fixed-weight STCH theory.
- Do not make missing-label correctness, comorbidity, significance, or final-model claims.
- Do not tune any R4 constant after the first production run starts.
- Maximum production training invocations: six.

## 1. Full R4 per-task objective

Implement:

    src/fusion/loss/r4_ramps_balance_loss.py

Registry key:

    wsm_r4_ramps_balance_loss

Class SHOULD be:

    WSMR4RampsBalanceLoss

Constructor parameters:

    mode: str
    pseudo_scale: float = 0.0
    aux_weight: float = 0.25
    agreement_weight: float = 0.10
    tau: float = 0.1
    progress_temperature: float = 0.25
    progress_reference_depression: float = 0.697035
    progress_reference_parkinson: float = 0.852104
    controller_ema: float = 0.8
    grad_ema: float = 0.9
    reliability_ema: float = 0.9
    weight_min: float = 0.2
    weight_max: float = 0.8
    eps: float = 1e-8

Allowed modes exactly:

    equal
    stch
    progress
    ra_stch

Validate all numeric ranges and finite values.

The loss MUST expose writable:

    pseudo_scale

so the accepted `wsm_pseudo_scale_warmup_callback` can drive it.

### Task objective

For each task `t in {depression, parkinson}`, compute:

    L_t =
        L_obs_t
        + pseudo_scale * L_pseudo_t
        + 0.25 * L_aux_t
        + 0.10 * L_agree_t

Use the exact frozen coefficients above.

#### Observed term

`L_obs_t`:

- BCE-with-logits mean over `observed_mask[:,t]`;
- unknown labels never enter BCE;
- observed targets must be finite 0/1.

#### Pseudo term

`L_pseudo_t`:

- use `pseudo_accept_mask[:,t] & ~observed_mask[:,t]`;
- pseudo target and pseudo reliability MUST both be detached before any autograd-participating use;
- accepted pseudo targets finite in [0,1];
- reliability finite in [0,1];
- rejected/observed pseudo targets must remain NaN and reliability zero;
- overlap between accepted pseudo and observed truth raises;
- reliability-weighted BCE numerator;
- denominator = detached reliability sum + eps;
- no accepted pseudo => exact differentiable zero.

Observed truth always wins.

#### Auxiliary term

Use R3 output:

    audio_aux_logits
    video_aux_logits
    audio_aux_valid
    video_aux_valid

For task `t`, `L_aux_t` pools audio and video BCE elements where:

    observed_mask[:,t] & modality_valid[:,t]

Normalize by eligible element count + eps.

No eligible element => exact differentiable zero.

#### Agreement term

For task `t`, only where:

    observed_mask[:,t] & audio_aux_valid[:,t] & video_aux_valid[:,t]

use:

    mean((sigmoid(audio_aux)-sigmoid(video_aux))^2)

No eligible element => exact differentiable zero.

### Active-task behavior

A task is active in a batch if it has at least one observed or accepted-pseudo supervision element.

- If both tasks active, apply the selected scalarizer.
- If exactly one task active, return that task objective directly.
- If neither task active, raise ValueError.

Do not fabricate zeros as competing objectives.

## 2. Scalarization modes

### Mode equal

When both tasks active:

    loss = 0.5 * L_D + 0.5 * L_P

Current weights remain exactly [0.5,0.5].

### Mode stch

Frozen constants:

    tau = 0.1
    lambda = [0.5, 0.5]
    z_star = [0.0, 0.0]

Use numerically stable:

    tau * logsumexp(
        lambda_t * (L_t - z_t_star) / tau
    )

Do not adapt lambda.

### Mode progress

Epoch 1 current weights:

    [0.5, 0.5]

The callback supplies previous completed epoch DEV task Scores.

For task `t`:

    p_t = clip(
        (dev_score_t - reference_t) /
        (1 - reference_t + eps),
        -1,
        1,
    )

with frozen references:

    depression = 0.697035
    parkinson = 0.852104

Then:

    raw_t = exp(-p_t / 0.25)

Normalize raw weights to sum 1.

Apply bounded normalization:

1. normalize to sum 1;
2. clamp each component to [0.2,0.8];
3. renormalize to sum 1.

For the next epoch:

    loss = w_D * L_D + w_P * L_P

All weights are detached/non-trainable controller state.

### Mode ra_stch

Epoch 1:

    alpha = [0.5,0.5]

Use the same progress raw values as mode progress.

#### Gradient diagnostics

When both task objectives are active:

- compute `autograd.grad` for each task objective separately with respect to:
  - `output.aux["features_audio"]`;
  - `output.aux["features_video"]`;
- use `retain_graph=True`, `create_graph=False`;
- flatten/concatenate available gradients;
- compute finite task gradient norms and cosine;
- diagnostic gradients/statistics MUST be detached and MUST NOT create second-order optimization.

Maintain batch EMAs:

    grad_norm_ema decay = 0.9
    grad_cos_ema decay = 0.9

If a valid diagnostic cannot be computed for a batch, skip that EMA update rather than inventing a value.

#### Accepted pseudo reliability diagnostics

For each task, on batches containing accepted pseudo entries:

    batch_reliability_t =
        mean(detached pseudo_reliability over accepted missing entries)

Maintain task reliability EMA with decay 0.9.

If a task has no reliability EMA yet at an epoch boundary, use neutral fallback 0.5 for controller calculation and log that fallback state.

#### RA target alpha

At epoch end after DEV metrics exist:

    grad_mean = mean(grad_norm_D, grad_norm_P)

    g_factor_t =
        clip(
            grad_mean / (grad_norm_t + eps),
            0.5,
            2.0,
        ) ** (1 + max(0, -grad_cos))

    r_factor_t =
        0.5 + 0.5 * reliability_ema_t

    raw_alpha_t =
        progress_raw_t * g_factor_t * r_factor_t

Then:

1. normalize raw_alpha to sum 1;
2. clamp each to [0.2,0.8];
3. renormalize;
4. controller EMA:

       alpha_next =
           0.8 * alpha_current
           + 0.2 * alpha_target

5. bounded-normalize again.

All alpha/controller values MUST be detached.

Final loss:

    tau * logsumexp(
        alpha_t * (L_t - 0.0) / tau
    )

with `tau=0.1`.

Do NOT claim theoretical equivalence between this dynamic method and fixed-weight STCH.

## 3. R4 controller callback

Implement:

    src/common/callbacks/wsm_r4_balance_callback.py

Registry key:

    wsm_r4_balance_callback

Subclass Chimera BaseCallback.

The callback MUST remain modality-independent. It may not import audio/video/fusion/text/description packages.

At `on_fit_start`:

- verify `trainer.loss_fn` exposes the R4 controller API;
- verify current mode/weights are valid.

At `on_epoch_end`:

- this callback MUST run after `wsm_segment_metrics_callback`;
- read only:

      dev/depression/score
      dev/parkinson/score

- NEVER read any Test metric for controller state;
- log the weights/alpha that were used for the completed epoch;
- update progress/RA controller state for the next epoch;
- append finite diagnostics into `logs`;
- log the same scalar diagnostics to MLflow when available.

Required log keys when applicable:

    r4/weight_depression
    r4/weight_parkinson
    r4/progress_depression
    r4/progress_parkinson
    r4/grad_norm_depression
    r4/grad_norm_parkinson
    r4/grad_cosine
    r4/reliability_depression
    r4/reliability_parkinson

For equal/static modes, weights may remain fixed and unsupported dynamic diagnostics may be omitted.

## 4. Registration

Update:

    src/chimera_plugin.py

Register:

- `fusion.loss.r4_ramps_balance_loss`;
- `common.callbacks.wsm_r4_balance_callback`.

All existing registrations remain intact.

## 5. Freeze all six production configs before first run

Create before any production training:

    13_r4_equal_seed42.yaml
    14_r4_static_stch_seed42.yaml
    15_r4_progress_seed42.yaml
    16_r4_ra_stch_seed42.yaml
    17_r4_ra_stch_seed43.yaml
    18_r4_ra_stch_seed44.yaml

Common frozen composition:

- experiment_name `wsm_mm_pd_dep_v1`;
- data `wsm_ramps_semantic_datamodule`;
- frozen pseudo cache path;
- exact R3 model `wsm_av_r3_disease_query_model`;
- dimensions `768/512/192/192`, dropout 0.2, two tasks;
- loss `wsm_r4_ramps_balance_loss`;
- `pseudo_scale: 0.0`;
- `aux_weight: 0.25`;
- `agreement_weight: 0.10`;
- all frozen controller constants from this task;
- AdamW lr `1e-4`, weight decay `0.01`;
- batch size 32;
- max epochs 30;
- mixed precision true;
- grad clip 0.5;
- checkpoint/early stopping only `dev/mean_score`, mode max;
- patience 6, min_delta 0.0005;
- required four-stream instrumentation/loggers;
- pseudo warm-up exactly `3/5/1.0`;
- `wsm_r4_balance_callback`;
- no text/description.

Mode/seed:

- 13: mode equal, seed 42
- 14: mode stch, seed 42
- 15: mode progress, seed 42
- 16: mode ra_stch, seed 42
- 17: mode ra_stch, seed 43
- 18: mode ra_stch, seed 44

Run names must encode R4 mode and seed.

Callback ordering MUST ensure:

1. `wsm_segment_metrics_callback` updates DEV task scores;
2. `wsm_r4_balance_callback` reads those DEV scores and writes R4 diagnostics;
3. `wsm_summary_callback` runs after the R4 callback.

Checkpoint/early stopping remain DEV/Mean-only.

## 6. Mandatory firewall commit before production

Before any production run:

1. implement loss and callback;
2. create all six configs;
3. validate all six configs;
4. run all synthetic/lifecycle/actual-cache smokes;
5. update PROGRESS_EN.md with exact frozen formulas/configs/screens;
6. commit and push one firewall commit to `codex/task-005f`.

No production run may start before that commit exists on origin.

## 7. Required pre-run smokes

### A. Compile/registry/config

Compile new modules/plugin.

Register plugins and prove:

    LOSSES: wsm_r4_ramps_balance_loss
    CALLBACKS: wsm_r4_balance_callback
    MODELS: wsm_av_r3_disease_query_model
    DATAMODULES: wsm_ramps_semantic_datamodule

Validate all six YAMLs.

### B. Synthetic task-loss contract

Use a deterministic synthetic batch containing:

- observed D;
- observed P;
- accepted missing D;
- accepted missing P;
- rejected missing entries;
- audio/video validity variation;
- pseudo targets and reliability with `requires_grad=True`.

For every mode:

- finite scalar loss;
- observed supervised gradients non-zero;
- accepted pseudo gradients non-zero when `pseudo_scale=1`;
- rejected missing gradients zero;
- pseudo targets/reliability receive no gradients;
- unknown labels never enter BCE.

For equal mode with both tasks active, prove exact `0.5/0.5` composition.

For static STCH, compare against direct stable reference formula.

### C. Progress callback lifecycle

Without training:

- instantiate progress loss/callback;
- feed fixed DEV Scores for multiple epochs;
- prove exact next-epoch weights from the frozen formula;
- prove no Test key is read or required;
- weights remain finite, bounded, normalized.

### D. RA controller lifecycle/gradient smoke

Use R3 model and synthetic batch.

Prove:

- per-task gradient norms finite/non-zero when both objectives active;
- cosine finite in [-1,1];
- reliability EMA updates only from accepted pseudo reliability;
- alpha remains detached, finite, normalized, bounded;
- epoch-1 alpha exactly [0.5,0.5];
- a fixed synthetic DEV/grad/reliability state produces a deterministic expected alpha update;
- no second-order gradient is created.

### E. Actual-cache TRAIN-only smoke

Use real frozen semantic DataModule and exact R3 model.

Require:

- cache SHA remains `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`;
- accepted counts remain D/P `376/1801`;
- one tiny TRAIN-only forward/loss/backward for each mode;
- missing-head accepted pseudo gradients are non-zero at `pseudo_scale=1`;
- observed gradients remain non-zero;
- both main heads, projections, task queries, and gate receive finite gradients;
- RA diagnostics finite;
- no optimizer step;
- no DEV/Test row iteration.

## 8. Frozen seed42 production sequence

After firewall:

1. equal seed42;
2. static STCH seed42;
3. progress seed42;
4. RA-STCH seed42.

Exactly one production invocation per config.

No rerun to improve metrics.

### Seed42 comparator

Corrected R3-B seed42:

- D `0.695808`
- P `0.893824`
- Mean `0.794816`

RA-STCH seed42 may continue only if ALL are true:

- selected DEV Mean > `0.794816`;
- selected DEV depression Score >= `0.685808`;
- selected DEV Parkinson Score >= `0.883824`;
- RA-STCH selected DEV Mean is strictly greater than equal seed42 Mean;
- strictly greater than static STCH seed42 Mean;
- strictly greater than progress seed42 Mean.

If any condition fails, STOP the production tree after four runs.

No other R4 mode may continue to additional seeds.

## 9. Conditional true seeds43/44

If and only if RA-STCH passes Section 8:

5. run RA-STCH seed43;
6. run RA-STCH seed44.

Compare to corrected same-seed R3-B:

- seed43 R3-B D/P/Mean:
  `0.694349/0.841176/0.767762`
- seed44 R3-B:
  `0.698378/0.838989/0.768684`

Record whether RA DEV Mean beats R3-B on each seed.

Compute RA three-seed mean/sample std for D/P/Mean.

Comparator R3-B three-seed means:

- D `0.696178`
- P `0.857996`
- Mean `0.777087`

Record:

- RA mean deltas versus these comparator means;
- whether RA Mean beats same-seed R3-B on all 42/43/44;
- whether RA three-seed Mean > `0.777087`;
- whether neither RA task mean is more than 0.010000 below R3-B corresponding task mean.

Do NOT self-declare final promotion. Report the frozen checks and let the manager decide.

## 10. Required post-run diagnostics

For every executed run record:

- config;
- exact command;
- seed;
- run directory;
- MLflow run ID/status/artifact URI;
- epochs completed/early stop;
- selected epoch/checkpoint by DEV only;
- selected DEV D/P UAR/MF1/Score and Mean;
- deltas versus corrected R3-B comparator;
- same-epoch TEST_NONE/SOFT/HARD monitoring only after DEV selection is frozen;
- confirmation Test did not influence any decision;
- selected checkpoint SHA256.

For every R4 mode seed42 record epoch table with:

- epoch;
- pseudo_scale;
- task weights/alpha;
- train loss;
- DEV D/P/Mean;
- Test protocol Means as monitoring-only.

For progress/RA record controller diagnostics.

For RA specifically record:

- gradient norm EMAs;
- gradient cosine EMA;
- reliability EMAs;
- progress signals;
- alpha trajectory.

If RA continues to three seeds, perform a post-hoc DEV-only same-row calibration audit for RA and corrected R3-B on each same seed:

- Brier;
- ECE-15;
- observed counts;
- RA minus R3-B delta.

Diagnostics only. Do not recalibrate or rerun.

## 11. Production commands

Use exactly one command per authorized config:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path <authorized-config>

Do not use sweep.

## 12. Scope checks

After implementation and after final evidence:

    git diff --check
    git diff origin/main -- src/audio
    git diff origin/main -- src/video
    git diff origin/main -- src/fusion/models
    git diff origin/main -- src/fusion/data
    git status --short
    git diff --stat origin/main...HEAD
    git log -10 --oneline --decorate

Only the new R4 loss/callback, plugin, six configs, and PROGRESS may differ from the manager base.

## Acceptance criteria

TASK-005F-BUNDLE execution passes only if:

- branch is exactly `codex/task-005f` from manager-updated origin/main;
- only allowed files change;
- all six configs are frozen and committed before first production run;
- full composition uses exact R3 model + frozen R2 pseudo cache/warm-up + frozen R3 aux/agreement;
- no architecture/pseudo threshold/cache change occurs;
- R4 loss implements exact task objectives and all four frozen scalarizers;
- observed truth overrides pseudo supervision;
- pseudo target/reliability are structural stop-gradient;
- progress/RA controller reads DEV task scores only, never Test;
- RA gradient diagnostics are detached/no second-order path;
- all smokes pass;
- at most six production runs occur;
- first four seed42 modes run in fixed order;
- seeds43/44 run only if RA passes every continuation condition;
- selection/checkpoint/early stopping remain DEV-only;
- Test remains monitoring-only;
- no post-hoc tuning occurs;
- selected checkpoint SHA recorded for every run;
- PROGRESS records complete evidence;
- src/audio/src/video/models/data remain unchanged;
- branch is pushed;
- main/master untouched by Codex.

Completing this task does NOT itself authorize Stage 6 or Final Test. The manager will decide whether R4 or any Stage-5 composition is promoted and whether Stage 5 can close.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch `codex/task-005f`;
- firewall commit SHA;
- final evidence commit SHA;
- pushed-to-origin status;
- main/master untouched;
- model/data/cache identities;
- registry keys;
- exact R4 formulas/constants;
- number of production runs executed;
- equal/static/progress/RA seed42 DEV D/P/Mean;
- RA continuation pass/fail and exact reasons;
- true seed43/44 results if executed;
- three-seed RA mean/std and comparator deltas if available;
- selected checkpoint SHA256 for each run;
- calibration diagnostics if RA continued;
- controller diagnostics summary;
- no Test-driven decision;
- no post-hoc tuning;
- R2 and R3 prior standalone conclusions preserved;
- no Text/Description/Stage6/7/Final Test;
- Stage 5 remains active pending manager decision.

Stop after TASK-005F-BUNDLE.
