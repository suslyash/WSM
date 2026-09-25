# TASK-005G-BUNDLE-EXPANDED: Five-Seed Confirmation of Equal, Static-STCH, and Progress

## Authority and branch

This task is an explicit owner/manager expansion of TASK-005G.

Required branch:

    codex/task-005g

Start from current `origin/main`, which includes:

- PR #43 merge `d44ccdbb3fad6e4f934bb87aee006560843e5dd5`;
- manager R4 confirmation freeze `92a5d38333adee37cc5c7aa98f6b779aa9a0695c`;
- owner/manager expansion commit `661782b6454423b658968973be675a772e1a05a8`.

If `codex/task-005g` already exists from the earlier six-run assignment, reuse is explicitly authorized. Fetch `origin` and merge current `origin/main` into the task branch. Do not reset, rebase away work, force-push, or touch main/master.

This task is experiment-only. Source code MUST remain unchanged.

## Goal

Obtain stronger repeatability evidence for the three valid R4 fallback scalarizers:

1. Equal;
2. Static STCH;
3. corrected Progress.

Retain the already-valid seed42 runs.

Run all three methods on true seeds:

    43, 44, 45, 46

This authorizes exactly:

    12 NEW production invocations

Together with retained seed42, each method will have five total seeds:

    42, 43, 44, 45, 46

RA-STCH is already a negative R4 result and MUST NOT be rerun.

## Required reading

Read in this order:

1. AGENTS.md
2. docs/README.md
3. docs/PROJECT_REQUIREMENTS.md
4. docs/PLAN.md, especially Stage 5, Stage 6 boundary, and Section 7 Promotion Rule
5. docs/PROGRESS_EN.md through OWNER/MANAGER-OVERRIDE-047
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
- configs/wsm_mm_pd_dep_v1/fusion/25_r4_equal_seed45.yaml
- configs/wsm_mm_pd_dep_v1/fusion/26_r4_equal_seed46.yaml
- configs/wsm_mm_pd_dep_v1/fusion/27_r4_static_stch_seed45.yaml
- configs/wsm_mm_pd_dep_v1/fusion/28_r4_static_stch_seed46.yaml
- configs/wsm_mm_pd_dep_v1/fusion/29_r4_progress_seed45.yaml
- configs/wsm_mm_pd_dep_v1/fusion/30_r4_progress_seed46.yaml
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
- Do not change R2 pseudo cache, thresholds, reliability, pseudo targets, or warm-up.
- Do not change R4 formulas or controller constants.
- Do not rerun seed42.
- Do not run RA-STCH.
- Do not tune after seeing any intermediate seed result.
- Do not stop the 12-run sequence based on intermediate DEV outcomes.
- Do not use Test for ranking, selection, branching, tuning, or stopping.
- Do not start Stage 6, Stage 7, Text/Description, or Final Test.
- Do not make significance, final-model, missing-label correctness, or comorbidity claims.

## Retained valid seed42 evidence

### Equal seed42

- DEV D/P/Mean:
  `0.712498 / 0.843342 / 0.777920`
- checkpoint SHA256:
  `d84d4f107fb5e927491b9f628d6ad9f913c8a7a2dc82689f9149ac32d206f690`

### Static-STCH seed42

- DEV D/P/Mean:
  `0.725077 / 0.849485 / 0.787281`
- checkpoint SHA256:
  `a1ad7dc4eb4e19e438ce14854d0981ce175e67c75e454c111cf44994f6fb8373`

### Corrected Progress seed42

- DEV D/P/Mean:
  `0.701245 / 0.873354 / 0.787299`
- checkpoint SHA256:
  `e821bc0b2ebe81c7684721a9f05b8e46ea5c3bcdaf60e860d9dc5dbe8673a15d`

Corrected R3-B contextual comparator:

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

## 1. Freeze all 12 confirmation configs before training

Create/freeze:

### Equal

- 19: seed43, run_name `r4_equal_seed43_confirm`
- 20: seed44, run_name `r4_equal_seed44_confirm`
- 25: seed45, run_name `r4_equal_seed45_confirm`
- 26: seed46, run_name `r4_equal_seed46_confirm`

Each MUST be an exact copy of config 13 except top-level `seed` and `run_name`.

### Static STCH

- 21: seed43, run_name `r4_static_stch_seed43_confirm`
- 22: seed44, run_name `r4_static_stch_seed44_confirm`
- 27: seed45, run_name `r4_static_stch_seed45_confirm`
- 28: seed46, run_name `r4_static_stch_seed46_confirm`

Each MUST be an exact copy of config 14 except `seed` and `run_name`.

### Corrected Progress

- 23: seed43, run_name `r4_progress_seed43_confirm`
- 24: seed44, run_name `r4_progress_seed44_confirm`
- 29: seed45, run_name `r4_progress_seed45_confirm`
- 30: seed46, run_name `r4_progress_seed46_confirm`

Each MUST be an exact copy of corrected config 15 except `seed` and `run_name`.

No semantic difference is allowed.

## 2. Mandatory pre-run firewall

Before the first production invocation:

### Config equivalence

Programmatically compare every new config against its accepted seed42 template.

After normalizing only:

- top-level `seed`;
- `experiment_info.params.run_name`;

the configs MUST be identical.

### Config validation

Run `chimera-ml validate-config` for all 12 configs.

### Registry/build

For each unique mode:

- build `wsm_av_r3_disease_query_model`;
- build `wsm_r4_ramps_balance_loss`;
- build `wsm_ramps_semantic_datamodule`;
- verify R3 trainable parameter count is exactly `403079`;
- verify frozen pseudo cache identity/path;
- verify frozen pseudo warm-up and controller constants.

No optimizer step.

### Seed identity firewall

For seeds:

    42, 43, 44, 45, 46

For each seed:

- call Chimera `define_seed(seed)`;
- instantiate exact R3 model;
- compute deterministic initial-state digest;
- require all five digests pairwise distinct;
- reinstantiate seed42 after resetting and require identical digest;
- record deterministic `torch.randperm(6325)` prefix.

Require pairwise-distinct randperm prefixes across all five seeds.

Do not iterate DEV/Test rows.

### Freeze commit

Before any production run:

- append exact firewall evidence to PROGRESS_EN.md;
- commit and push one firewall commit on `codex/task-005g`.

No production run may start before the firewall commit exists on origin.

## 3. Exact fixed production sequence

Run exactly 12 new production invocations in this order:

1. Equal seed43
2. Static-STCH seed43
3. Progress seed43
4. Equal seed44
5. Static-STCH seed44
6. Progress seed44
7. Equal seed45
8. Static-STCH seed45
9. Progress seed45
10. Equal seed46
11. Static-STCH seed46
12. Progress seed46

Use exactly:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path <authorized-config>

Do not sweep.

Do not stop based on intermediate metrics.

Do not rerun any method to improve results.

## 4. Selection firewall

For every run:

- select checkpoint/epoch only by maximum `dev/mean_score`;
- freeze selected checkpoint before reading same-epoch Test monitoring;
- Test cannot affect any experiment or conclusion.

Record selected checkpoint SHA256.

## 5. Five-seed summaries

For Equal, Static-STCH, and Progress separately, combine retained seed42 with new seeds43/44/45/46.

Record for every seed:

- depression DEV Score;
- Parkinson DEV Score;
- DEV Mean.

Compute:

- five-seed arithmetic mean;
- sample standard deviation;
- median;
- min/max;
- 95% t-based confidence interval for DEV Mean as a descriptive statistic only.

Do not make a significance claim from the CI.

## 6. Frozen five-seed balancing-repeat criterion

For Static and Progress separately compare against Equal on the SAME seed.

A candidate passes only if ALL are true:

1. candidate DEV Mean > Equal DEV Mean on at least 4 of 5 seeds;
2. candidate five-seed DEV Mean > Equal five-seed DEV Mean;
3. candidate five-seed depression mean is not more than `0.010000` below Equal depression mean;
4. candidate five-seed Parkinson mean is not more than `0.010000` below Equal Parkinson mean;
5. the positive aggregate effect is not driven by a single extreme seed.

Operationalize item 5 as:

- recompute candidate-minus-Equal Mean delta after removing the single seed with the largest positive delta;
- the remaining four-seed mean delta must still be strictly positive.

Record:

- wins / ties / losses versus Equal;
- all per-seed Mean deltas;
- leave-one-largest-win-out mean delta.

Do not alter this criterion after seeing results.

## 7. Contextual R-full viability diagnostic

Corrected R3-B exists only for seeds42/43/44.

Therefore:

- compare Static/Progress to R3-B pairwise only on seeds42/43/44;
- report the three-seed paired Mean deltas;
- report whether candidate three-seed Mean exceeds R3-B `0.777087`;
- report whether candidate D mean >= `0.686178`;
- report whether candidate P mean >= `0.847996`.

Do NOT fabricate R3-B seed45/46 comparators.

The primary balancing promotion evidence is the five-seed paired comparison versus Equal.

## 8. DEV-only calibration and negative-transfer audit

After all checkpoint selections are frozen, run a same-row DEV-only audit.

For seeds42/43/44/45/46 and methods:

- Equal;
- Static;
- Progress;

compute, for depression and Parkinson:

- observed row count;
- Brier;
- ECE with 15 bins.

For corrected R3-B, include existing seeds42/43/44 only.

Use identical DEV rows/masks within every same-seed comparison.

No recalibration.

No threshold changes.

No reruns.

Summarize for each method:

- mean Brier/ECE across its available seeds;
- Static minus Equal;
- Progress minus Equal;
- three-seed Static/Progress minus R3-B on seeds42/43/44 only.

Also report task-score negative-transfer deltas.

## 9. Frozen candidate nomination rule

After all 12 new runs and diagnostics:

1. exclude Static or Progress if it fails the five-seed balancing-repeat criterion;
2. if neither remains, record:
   `NO R4 BALANCING CANDIDATE NOMINATED`;
3. if exactly one remains, nominate it provisionally for manager Stage-5 composition review;
4. if both remain:
   - compare five-seed DEV Mean;
   - if absolute difference > `0.001000`, nominate higher Mean;
   - if absolute difference <= `0.001000`, compare in order:
     1. worse task-mean delta versus Equal;
     2. five-seed DEV Brier/ECE versus Equal;
     3. three-seed contextual diagnostics versus corrected R3-B;
     4. controller complexity/stability;
   - if still tied, nominate Static-STCH as simpler.

Nomination is not promotion.

Codex MUST NOT declare Stage 5 closed.

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
- same-epoch TEST_NONE/SOFT/HARD after DEV selection is frozen;
- explicit confirmation Test did not influence selection.

For Progress record controller-weight/progress trajectory path or full table.

## 11. Scope checks

Run:

    git diff --check
    git diff origin/main -- src
    git status --short
    git diff --stat origin/main...HEAD
    git log -12 --oneline --decorate

The source diff versus origin/main MUST be empty.

## Acceptance criteria

TASK-005G-BUNDLE-EXPANDED passes only if:

- branch is `codex/task-005g` from current manager-updated origin/main;
- only 12 confirmation configs plus PROGRESS change;
- all config pairs differ from seed42 templates only by seed/run_name;
- all 12 configs validate before production;
- five-seed initialization/randomness firewall passes;
- firewall commit is pushed before training;
- exactly 12 new production invocations execute in fixed order;
- no seed42 rerun;
- no RA run;
- no source change;
- no post-hoc tuning;
- DEV-only checkpoint selection;
- Test monitoring only;
- checkpoint SHA256 recorded for every new run;
- complete five-seed summaries recorded;
- five-seed balancing-repeat criteria applied exactly;
- DEV-only calibration/negative-transfer audit completed;
- nomination rule applied exactly;
- branch pushed;
- main/master untouched.

Passing this task gives the manager evidence to decide whether Stage 5 has a promotable composition or should close with no promoted RAMPS composition. It does not authorize Stage 6 or Final Test.

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
- initial-state digests/randperm evidence for 42/43/44/45/46;
- number of new production invocations = 12;
- full five-seed Equal/Static/Progress D/P/Mean table;
- five-seed mean/std/median/min/max/CI table;
- balancing-repeat PASS/FAIL for Static and Progress;
- wins/ties/losses versus Equal;
- leave-one-largest-win-out robustness delta;
- contextual R3-B diagnostics on seeds42/43/44;
- calibration/negative-transfer summary;
- provisional nomination result;
- no Test-driven decision;
- no post-hoc tuning;
- RA not rerun;
- no Stage6/7/Text/Final Test;
- Stage 5 remains active pending manager decision.

Stop after TASK-005G-BUNDLE-EXPANDED.
