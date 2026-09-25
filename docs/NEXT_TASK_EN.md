# TASK-005G-BUNDLE: Three-Seed Confirmation of Equal, Static-STCH, and Progress

## Authority and branch

This task supersedes the cancelled five-seed expansion.

Required branch:

    codex/task-005g

Start from current `origin/main`, which includes:

- PR #43 merge `d44ccdbb3fad6e4f934bb87aee006560843e5dd5`;
- manager R4 confirmation freeze `92a5d38333adee37cc5c7aa98f6b779aa9a0695c`;
- cancelled expansion record `661782b6454423b658968973be675a772e1a05a8`;
- owner/manager override restoring three-seed confirmation `b9d1b08d2b6d4ba8555b0e97f0925f0561d6417b`.

If `codex/task-005g` already exists from an earlier TASK-005G assignment, reuse is explicitly authorized. Fetch `origin` and merge current `origin/main` into the task branch. Do not reset, rebase away work, force-push, or touch main/master.

If expansion-only seed45/46 configs were created locally but no production run used them, remove those unneeded task-local files before the firewall commit. No seed45/46 production run is authorized.

This task is experiment-only. Source code MUST remain unchanged.

## Goal

Complete the PLAN-recommended three-seed evidence for the three valid R4 fallback scalarizers:

1. Equal;
2. Static STCH;
3. corrected Progress.

Retain the already-valid seed42 runs and execute only true seeds43/44 for each method.

Exactly six new production invocations are authorized.

Together with retained seed42, each method will have three total seeds:

    42, 43, 44

RA-STCH is already a negative R4 result and MUST NOT be rerun.

## Required reading

Read in this order:

1. AGENTS.md
2. docs/README.md
3. docs/PROJECT_REQUIREMENTS.md
4. docs/PLAN.md, especially Stage 5, Stage 6 boundary, and Section 7 Promotion Rule
5. docs/PROGRESS_EN.md through OWNER/MANAGER-OVERRIDE-048
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

If expansion-only config files 25–30 were already committed on this task branch before this override, deleting those files is additionally authorized. They must not survive the final task diff and must not be used for production.

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
- Do not run seeds45/46.
- Do not run RA-STCH.
- Do not tune after seeing seed43.
- Do not stop the six-run sequence based on intermediate DEV outcomes.
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

- 19: Equal, seed43, run_name `r4_equal_seed43_confirm`
- 20: Equal, seed44, run_name `r4_equal_seed44_confirm`
- 21: Static STCH, seed43, run_name `r4_static_stch_seed43_confirm`
- 22: Static STCH, seed44, run_name `r4_static_stch_seed44_confirm`
- 23: Progress, seed43, run_name `r4_progress_seed43_confirm`
- 24: Progress, seed44, run_name `r4_progress_seed44_confirm`

For Progress, copy the corrected seed42 config currently in `15_r4_progress_seed42.yaml`.

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

After normalizing only:

- top-level `seed`;
- `experiment_info.params.run_name`;

each pair MUST be identical.

### Config validation

Run `chimera-ml validate-config` for all six new configs.

### Registry/build

For each unique mode:

- build the frozen `wsm_av_r3_disease_query_model`;
- build `wsm_r4_ramps_balance_loss`;
- build `wsm_ramps_semantic_datamodule`;
- verify R3 trainable parameter count remains exactly `403079`;
- verify pseudo cache path and all loss/controller constants match the accepted seed42 config.

No optimizer step.

### Seed identity firewall

For seeds42/43/44:

- call Chimera `define_seed(seed)`;
- instantiate the exact R3 model;
- compute deterministic initial-state digest;
- require pairwise-distinct digests across 42/43/44;
- reinstantiate seed42 after resetting and require identical digest;
- record deterministic `torch.randperm(6325)` prefix for each seed;
- require pairwise-distinct prefixes.

Do not iterate DEV/Test rows.

### Freeze commit

Before any production run:

- append exact firewall evidence to PROGRESS_EN.md;
- commit and push one firewall commit on `codex/task-005g`.

No production run may start before that firewall commit exists on origin.

## 3. Exact fixed production sequence

Run exactly six new production invocations in this order:

1. Equal seed43
2. Static-STCH seed43
3. Progress seed43
4. Equal seed44
5. Static-STCH seed44
6. Progress seed44

Use exactly:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path <authorized-config>

Do not sweep.

Do not stop based on intermediate metrics.

Do not rerun any method to improve metrics.

## 4. Selection firewall

For every run:

- select checkpoint/epoch only by maximum `dev/mean_score`;
- freeze selected checkpoint before reading same-epoch Test monitoring;
- Test cannot affect any experiment or conclusion.

Record selected checkpoint SHA256.

## 5. Three-seed summaries

For Equal, Static-STCH, and Progress separately, combine retained seed42 with new seeds43/44.

Record for every seed:

- depression DEV Score;
- Parkinson DEV Score;
- DEV Mean.

Compute:

- arithmetic mean;
- sample standard deviation.

Also record per-seed Mean deltas versus:

- same-seed Equal;
- same-seed corrected R3-B;
- frozen F2 as contextual reference where useful.

## 6. Frozen three-seed balancing-repeat criterion

For Static and Progress separately compare against Equal on the SAME seed.

A candidate passes only if ALL are true:

1. candidate DEV Mean > Equal DEV Mean on seed42;
2. candidate DEV Mean > Equal DEV Mean on seed43;
3. candidate DEV Mean > Equal DEV Mean on seed44;
4. candidate three-seed Mean > Equal three-seed Mean;
5. candidate three-seed depression mean is not more than `0.010000` below Equal depression mean;
6. candidate three-seed Parkinson mean is not more than `0.010000` below Equal Parkinson mean.

Do not alter this criterion after seeing results.

## 7. R-full viability diagnostic

For Static and Progress separately compare against corrected R3-B on seeds42/43/44.

Report:

- per-seed Mean deltas;
- candidate three-seed Mean versus R3-B `0.777087`;
- candidate depression mean versus minimum `0.686178`;
- candidate Parkinson mean versus minimum `0.847996`.

These are Stage-5 composition diagnostics and do not replace the paired Equal balancing-repeat criterion.

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

Use the same 933 DEV rows/observed masks within every same-seed comparison.

No recalibration.

No threshold changes.

No reruns.

Summarize:

- three-seed mean Brier/ECE per task/method;
- Static minus Equal;
- Progress minus Equal;
- Static minus R3-B;
- Progress minus R3-B.

Also report task-score negative-transfer deltas.

## 9. Frozen candidate nomination rule

After all evidence is complete:

1. exclude Static or Progress if it fails the three-seed balancing-repeat criterion;
2. if neither remains:
   - record `NO R4 BALANCING CANDIDATE NOMINATED`;
3. if exactly one remains:
   - nominate it provisionally for manager Stage-5 composition review;
4. if both remain:
   - compare three-seed DEV Mean;
   - if absolute difference > `0.001000`, nominate the higher Mean;
   - if absolute difference <= `0.001000`, compare in order:
     1. worse of the two task-mean deltas versus corrected R3-B;
     2. DEV-only Brier/ECE versus Equal and R3-B;
     3. controller complexity/stability;
   - if still tied, nominate Static-STCH as the simpler fixed scalarizer.

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

- branch is `codex/task-005g` from current manager-updated origin/main;
- only six confirmation configs plus PROGRESS change;
- any obsolete expansion-only configs 25–30 are absent from the final task diff;
- all config pairs differ from seed42 templates only by seed/run_name;
- all six configs validate before production;
- seed42/43/44 initialization/randomness firewall passes;
- firewall commit is pushed before training;
- exactly six new production invocations execute in fixed order;
- no seed42/45/46 rerun occurs;
- no RA run occurs;
- no source changes occur;
- no post-hoc tuning occurs;
- DEV-only checkpoint selection is preserved;
- Test remains monitoring-only;
- checkpoint SHA256 is recorded for every new run;
- complete three-seed summaries are recorded;
- three-seed balancing-repeat criteria are applied exactly;
- DEV-only calibration/negative-transfer audit is complete;
- nomination rule is applied exactly;
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
- initial-state digests/randperm evidence for 42/43/44;
- number of new production invocations = 6;
- full Equal/Static/Progress seed42/43/44 D/P/Mean table;
- three-seed mean/std table;
- balancing-repeat PASS/FAIL for Static and Progress;
- contextual R3-B diagnostics;
- calibration/negative-transfer summary;
- provisional nomination result;
- no Test-driven decision;
- no post-hoc tuning;
- RA not rerun;
- no Stage6/7/Text/Final Test;
- Stage 5 remains active pending manager decision.

Stop after TASK-005G-BUNDLE.
