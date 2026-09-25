# TASK-005G-BUNDLE: Confirm Equal, Static-STCH, and Progress Across True Seeds 42/43/44

## Authority and branch

Required branch:

    codex/task-005g

Start from current `origin/main`, which includes:

- PR #43 merge `d44ccdbb3fad6e4f934bb87aee006560843e5dd5`;
- manager confirmation-freeze commit `92a5d38333adee37cc5c7aa98f6b779aa9a0695c`.

Create exactly one new task branch from that `origin/main`.

This task is experiment-only. It MUST NOT modify source code.

## Goal

Complete the PLAN-required three-seed evidence for the three valid R4 fallback scalarizers:

1. Equal;
2. Static STCH;
3. corrected Progress.

Retain the already-valid seed42 runs and execute only seeds43/44 for each method.

Exactly six new production invocations are authorized.

RA-STCH is already a negative R4 result and MUST NOT be rerun.

## Required reading

Read in this order:

1. AGENTS.md
2. docs/README.md
3. docs/PROJECT_REQUIREMENTS.md
4. docs/PLAN.md, especially Stage 5, Stage 6 boundary, and Section 7 Promotion Rule
5. docs/PROGRESS_EN.md through MANAGER-DECISION-046
6. docs/NEXT_TASK_EN.md
7. configs/wsm_mm_pd_dep_v1/fusion/13_r4_equal_seed42.yaml
8. configs/wsm_mm_pd_dep_v1/fusion/14_r4_static_stch_seed42.yaml
9. configs/wsm_mm_pd_dep_v1/fusion/15_r4_progress_seed42.yaml
10. src/fusion/loss/r4_ramps_balance_loss.py
11. src/common/callbacks/wsm_r4_balance_callback.py

## Allowed tracked files

Codex may add/modify only:

- configs/wsm_mm_pd_dep_v1/fusion/19_r4_equal_seed43.yaml
- configs/wsm_mm_pd_dep_v1/fusion/20_r4_equal_seed44.yaml
- configs/wsm_mm_pd_dep_v1/fusion/21_r4_static_stch_seed43.yaml
- configs/wsm_mm_pd_dep_v1/fusion/22_r4_static_stch_seed44.yaml
- configs/wsm_mm_pd_dep_v1/fusion/23_r4_progress_seed43.yaml
- configs/wsm_mm_pd_dep_v1/fusion/24_r4_progress_seed44.yaml
- docs/PROGRESS_EN.md

No source file may change.

## Forbidden actions

- Do not modify src/audio.
- Do not modify src/video.
- Do not modify src/fusion.
- Do not modify src/common.
- Do not modify src/chimera_plugin.py.
- Do not modify existing configs 13–18.
- Do not change R3 architecture.
- Do not change R2 pseudo cache, thresholds, warm-up, reliability, or pseudo labels.
- Do not change R4 formulas or controller constants.
- Do not rerun seed42.
- Do not run RA-STCH.
- Do not tune after seeing seed43.
- Do not use Test for ranking, branching, or model selection.
- Do not start Stage 6, Stage 7, Text/Description, or Final Test.
- Do not make significance, final-model, missing-label correctness, or comorbidity claims.

## Retained valid seed42 evidence

### Equal seed42

- DEV D/P/Mean:
  `0.712498 / 0.843342 / 0.777920`
- selected checkpoint SHA256:
  `d84d4f107fb5e927491b9f628d6ad9f913c8a7a2dc82689f9149ac32d206f690`

### Static-STCH seed42

- DEV D/P/Mean:
  `0.725077 / 0.849485 / 0.787281`
- selected checkpoint SHA256:
  `a1ad7dc4eb4e19e438ce14854d0981ce175e67c75e454c111cf44994f6fb8373`

### Corrected Progress seed42

- DEV D/P/Mean:
  `0.701245 / 0.873354 / 0.787299`
- selected checkpoint SHA256:
  `e821bc0b2ebe81c7684721a9f05b8e46ea5c3bcdaf60e860d9dc5dbe8673a15d`

Corrected R3-B three-seed comparator:

- seed42: D/P/Mean `0.695808/0.893824/0.794816`
- seed43: `0.694349/0.841176/0.767762`
- seed44: `0.698378/0.838989/0.768684`
- three-seed means:
  - D `0.696178`
  - P `0.857996`
  - Mean `0.777087`

Frozen F2 comparator:

- D `0.697035`
- P `0.852104`
- Mean `0.774569`

## 1. Freeze six confirmation configs before training

Create:

    19_r4_equal_seed43.yaml
    20_r4_equal_seed44.yaml
    21_r4_static_stch_seed43.yaml
    22_r4_static_stch_seed44.yaml
    23_r4_progress_seed43.yaml
    24_r4_progress_seed44.yaml

Each config MUST be an exact copy of its accepted seed42 method config except:

- top-level seed;
- run_name.

Required values:

- 19: Equal, seed 43, run_name `r4_equal_seed43_confirm`
- 20: Equal, seed 44, run_name `r4_equal_seed44_confirm`
- 21: Static STCH, seed 43, run_name `r4_static_stch_seed43_confirm`
- 22: Static STCH, seed 44, run_name `r4_static_stch_seed44_confirm`
- 23: Progress, seed 43, run_name `r4_progress_seed43_confirm`
- 24: Progress, seed 44, run_name `r4_progress_seed44_confirm`

For Progress, copy the corrected config currently in `15_r4_progress_seed42.yaml`.

No other semantic difference is permitted.

## 2. Mandatory pre-run firewall

Before the first production invocation:

### Config equivalence

Programmatically compare:

- 19 vs 13;
- 20 vs 13;
- 21 vs 14;
- 22 vs 14;
- 23 vs 15;
- 24 vs 15.

After normalizing only `seed` and `experiment_info.params.run_name`, each pair MUST be identical.

### Config validation

Run `chimera-ml validate-config` for all six new configs.

### Registry/build

For each unique mode:

- build the frozen `wsm_av_r3_disease_query_model`;
- build `wsm_r4_ramps_balance_loss`;
- build `wsm_ramps_semantic_datamodule`;
- verify R3 trainable parameter count remains exactly `403079`;
- verify pseudo cache path and controller constants match seed42 configs.

No optimizer step.

### Seed identity firewall

For seeds 42, 43, 44:

- call Chimera `define_seed(seed)`;
- instantiate exact R3 model;
- compute deterministic initial-state digest;
- require pairwise-distinct digests across 42/43/44;
- reinstantiate seed42 after resetting and require identical digest;
- record deterministic `torch.randperm(6325)` prefixes for 42/43/44 and require pairwise-distinct prefixes.

This is TRAIN-only/randomness verification. Do not iterate DEV/Test rows.

### Freeze commit

Before any production run:

- append exact firewall evidence to PROGRESS_EN.md;
- commit and push one firewall commit on `codex/task-005g`.

No production run may start before that commit exists on origin.

## 3. Exact production sequence

Run exactly six new production invocations in this fixed order:

1. Equal seed43;
2. Static-STCH seed43;
3. Progress seed43;
4. Equal seed44;
5. Static-STCH seed44;
6. Progress seed44.

Use exactly:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path <authorized-config>

Do not sweep.

Do not stop the sequence based on an intermediate DEV result.

Do not rerun any method to improve metrics.

## 4. Selection firewall

For every run:

- select checkpoint/epoch only by maximum `dev/mean_score`;
- freeze selected epoch/checkpoint before inspecting same-epoch Test monitoring;
- Test cannot alter any run, ranking, or conclusion.

Record selected checkpoint SHA256.

## 5. Three-seed summaries

After all six runs, combine each method with its retained seed42 result.

For Equal, Static, and Progress separately compute:

- seed42/43/44 DEV D Score;
- seed42/43/44 DEV P Score;
- seed42/43/44 DEV Mean;
- arithmetic mean;
- sample standard deviation.

Also record per-seed deltas versus:

- same-seed Equal;
- same-seed corrected R3-B;
- frozen F2 where useful as context.

## 6. Frozen balancing-repeat criterion

For Static and Progress separately, compare against Equal on the SAME seed.

A candidate passes the balancing-repeat criterion only if ALL are true:

1. candidate DEV Mean > Equal DEV Mean on seed42;
2. candidate DEV Mean > Equal DEV Mean on seed43;
3. candidate DEV Mean > Equal DEV Mean on seed44;
4. candidate three-seed Mean > Equal three-seed Mean;
5. candidate three-seed depression mean is not more than `0.010000` below Equal depression mean;
6. candidate three-seed Parkinson mean is not more than `0.010000` below Equal Parkinson mean.

Do not change these conditions after seeing results.

## 7. R-full viability diagnostic

For Static and Progress separately compare the three-seed means against corrected R3-B:

- require/report whether candidate three-seed Mean > `0.777087`;
- require/report whether candidate depression mean >= `0.686178`;
- require/report whether candidate Parkinson mean >= `0.847996`.

These are diagnostics for final Stage-5 composition viability.

They do NOT replace the balancing-repeat criterion.

## 8. DEV-only calibration and negative-transfer audit

After all checkpoint selections are frozen, run a same-row DEV-only audit.

For seeds42/43/44 and methods:

- Equal;
- Static;
- Progress;
- corrected R3-B;

compute for depression and Parkinson:

- observed row count;
- Brier;
- ECE with 15 bins.

Use the same 933 DEV rows/observed masks per seed comparison.

No recalibration.

No threshold change.

No rerun.

Summarize:

- three-seed mean Brier/ECE per task/method;
- Static minus Equal;
- Progress minus Equal;
- Static minus R3-B;
- Progress minus R3-B.

Also report task-score negative-transfer deltas.

## 9. Frozen candidate nomination rule

After all evidence is complete:

1. exclude Static or Progress if it fails the balancing-repeat criterion;
2. if neither remains:
   - record `NO R4 BALANCING CANDIDATE NOMINATED`;
3. if exactly one remains:
   - record that candidate as the provisional Stage-5 composition candidate;
4. if both remain:
   - compare three-seed DEV Mean;
   - if absolute difference > `0.001000`, nominate the higher Mean;
   - if absolute difference <= `0.001000`, treat DEV as practically tied and compare, in order:
     1. worse of the two task-mean deltas versus corrected R3-B;
     2. DEV-only Brier/ECE diagnostics versus Equal and R3-B;
     3. controller complexity/stability;
   - if still tied, nominate Static-STCH as the simpler fixed scalarizer.

This is nomination only.

Codex MUST NOT declare Stage 5 closed or promote a final model.

## 10. Required evidence per new run

Record in PROGRESS_EN.md:

- config;
- exact command;
- seed;
- run directory;
- MLflow run ID/status/artifact URI;
- epochs completed/early-stop status;
- selected epoch/checkpoint;
- selected checkpoint SHA256;
- DEV D UAR/MF1/Score;
- DEV P UAR/MF1/Score;
- DEV Mean;
- same-epoch TEST_NONE/SOFT/HARD monitoring values after DEV freeze;
- confirmation Test did not influence selection.

For Progress additionally record controller-weight/progress trajectory path or full table.

## 11. Scope checks

Run:

    git diff --check
    git diff origin/main -- src
    git status --short
    git diff --stat origin/main...HEAD
    git log -10 --oneline --decorate

The source diff versus origin/main MUST be empty.

## Acceptance criteria

TASK-005G-BUNDLE passes only if:

- branch is exactly `codex/task-005g` from manager-updated origin/main;
- only six new configs plus PROGRESS change;
- all config pairs differ from accepted seed42 templates only by seed/run_name;
- all six configs validate before training;
- seed firewall proves true/distinct 42/43/44 initialization/randomness;
- firewall commit is pushed before production;
- exactly six new production invocations execute in fixed order;
- no seed42 rerun occurs;
- no RA run occurs;
- no source changes occur;
- checkpoint selection remains DEV-only;
- Test remains monitoring-only;
- no post-hoc tuning occurs;
- checkpoint SHA256 is recorded for every new run;
- complete three-seed summaries are recorded;
- balancing-repeat criteria are applied exactly as frozen;
- DEV-only calibration/negative-transfer audit is complete;
- candidate nomination follows the frozen rule;
- branch is pushed;
- main/master remains untouched.

Passing this task gives the manager the evidence needed to decide whether Stage 5 has a promotable final composition or should close with no promoted RAMPS composition. It does not authorize Stage 6 or Final Test.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch `codex/task-005g`;
- firewall commit SHA;
- final evidence commit SHA;
- pushed-to-origin status;
- main/master untouched;
- source diff empty;
- exact config seeds;
- initial-state digests/randperm evidence;
- number of production invocations = 6;
- full Equal/Static/Progress seed42/43/44 D/P/Mean table;
- three-seed mean/std table;
- balancing-repeat PASS/FAIL for Static and Progress;
- R-full viability diagnostics;
- calibration/negative-transfer summary;
- provisional nomination result under the frozen rule;
- no Test-driven decision;
- no post-hoc tuning;
- RA not rerun;
- no Stage6/7/Text/Final Test;
- Stage 5 remains active pending manager decision.

Stop after TASK-005G-BUNDLE.
