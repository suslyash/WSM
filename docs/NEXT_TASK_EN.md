# TASK-006A-BUNDLE: Stage-6 Shuffled Pseudo-Target Negative Control on the Equal Composition

## Authority and branch

This task follows **MANAGER-DECISION-059** and resumes the previously paused TASK-006A-BUNDLE after the matched-seed frozen-audio confirmation.

Required branch:

    codex/task-006a

Start from the current manager-updated `origin/main`, which includes the merged TASK-006-PRE-AUDIO-CONFIRM evidence and MANAGER-DECISION-059.

Create exactly one new task branch from that `origin/main`.

Stage-5 optimization is CLOSED. The prior negative robustness interpretation was superseded by matched-seed audio confirmation; R4 trial 012 is retained as the leading matched-seed candidate for Stage-6 audit. Do not reopen Stage-5 tuning.

## Goal

Execute the PLAN-required shuffled/mismatched pseudo-target negative control while preserving every non-semantic factor as closely as possible.

Use the accepted three-seed Equal full composition as the matched-pseudo reference because:

- it has no dynamic/fixed balancing-controller comparison confound;
- it uses the exact R3 model;
- it uses the frozen R2 pseudo path and warm-up;
- it uses the frozen R3 auxiliary/agreement terms;
- it has valid matched-pseudo results for seeds42/43/44.

Build ONE deterministic negative-control pseudo cache by permuting the complete pseudo-supervision tuple among missing rows within each task while leaving canonical rows and observed truth fixed.

Then run the Equal composition at seeds42/43/44 using ONLY that derived shuffled cache.

Exactly three production training invocations are authorized.

This task is an ablation/claims audit. It is NOT a tuning task and cannot reopen Stage 5.

## Required reading

Read in this exact order:

1. AGENTS.md
2. docs/README.md
3. docs/PROJECT_REQUIREMENTS.md
4. docs/PROGRESS_EN.md
5. docs/plan/STAGE_6.md
6. docs/NEXT_TASK_EN.md
7. src/fusion/data/wsm_ramps_semantic_datamodule.py
8. src/fusion/models/av_r3_disease_query.py
9. src/fusion/loss/r4_ramps_balance_loss.py
10. configs/wsm_mm_pd_dep_v1/fusion/13_r4_equal_seed42.yaml
11. configs/wsm_mm_pd_dep_v1/fusion/19_r4_equal_seed43.yaml
12. configs/wsm_mm_pd_dep_v1/fusion/20_r4_equal_seed44.yaml

Do not read the historical progress archive or closed-stage plan files unless a concrete verification question requires them.

## Allowed tracked files

Codex may modify/add only:

- scripts/common/build_ramps_shuffled_pseudo_negative_control.py
- configs/wsm_mm_pd_dep_v1/ablations/31_shuffled_pseudo_equal_seed42.yaml
- configs/wsm_mm_pd_dep_v1/ablations/32_shuffled_pseudo_equal_seed43.yaml
- configs/wsm_mm_pd_dep_v1/ablations/33_shuffled_pseudo_equal_seed44.yaml
- docs/PROGRESS_EN.md

No source file may change.

## Forbidden actions

- Do not modify src/audio.
- Do not modify src/video.
- Do not modify src/fusion.
- Do not modify src/common.
- Do not modify src/chimera_plugin.py.
- Do not modify the original semantic pseudo cache.
- Do not regenerate teachers, semantic embeddings, thresholds, calibration, or acceptance rules.
- Do not change accepted counts, pseudo class balance, or the multiset of pseudo target/reliability values.
- Do not change R3 architecture.
- Do not change Equal loss/formula, auxiliary/agreement coefficients, optimizer, warm-up, or instrumentation.
- Do not run Static, Progress, RA-STCH, Candidate B, or another Stage-5 model.
- Do not tune based on negative-control results.
- Do not use Test for selection, ranking, interpretation, or branching.
- Do not start another Stage-6 ablation.
- Do not start Stage 7, Text/Description, or Final Test.
- Do not claim missing-label correctness, comorbidity, significance, or final-model superiority.
- Do not reopen Stage-5 optimization or tune against confirmation seeds.

## Frozen source cache

Source:

    /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt

Required SHA256:

    17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945

Required rows:

    6325

Required accepted counts D/P:

    376 / 1801

Required accepted positive/negative counts:

- depression: 376 / 0
- Parkinson: 212 / 1589

The source cache is immutable.

## 1. Negative-control cache builder

Implement:

    scripts/common/build_ramps_shuffled_pseudo_negative_control.py

This is an executable/reproducibility helper, not reusable training code.

CLI arguments:

    --source
    --output
    --shuffle-seed

Frozen invocation:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 \
      scripts/common/build_ramps_shuffled_pseudo_negative_control.py \
      --source /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt \
      --output /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_negative_control/shuffled_missing_targets_seed6001.pt \
      --shuffle-seed 6001

The script MUST fail rather than overwrite an existing output path.

### Frozen permutation contract

Load the source cache read-only on CPU.

For each task independently:

1. identify rows where `observed_mask[:, task] == False`;
2. use a deterministic `torch.Generator`:
   - depression seed = `6001`;
   - Parkinson seed = `6002`;
3. generate a random ordering of the missing row indices;
4. create a derangement by circularly shifting that random ordering by exactly one position;
5. every missing destination row receives the pseudo tuple from a DIFFERENT missing source row;
6. observed rows are never sources or destinations.

Permute the following fields together as one aligned tuple for that task:

- `pseudo_accept_mask`;
- `pseudo_targets`;
- `pseudo_reliability`;
- `pseudo_class`;
- `calibrated_audio_probs`.

Do not permute:

- `segment_ids`;
- `observed_mask`;
- `observed_targets`;
- corpus/split identity;
- teacher/prompt/cache provenance metadata.

The derived cache MUST retain the original required `version`, task names, teacher identities, prompt SHA, and other fields needed by `wsm_ramps_semantic_datamodule`.

Add an extra metadata mapping that does not alter training, for example:

    negative_control:
      type: within_task_missing_row_tuple_derangement
      source_cache_sha256: <exact source SHA>
      shuffle_seed: 6001
      depression_permutation_sha256: ...
      parkinson_permutation_sha256: ...
      permuted_fields: [...]

Do not change the canonical cache version solely for this negative control.

## 2. Mandatory cache invariant audit

Before saving, and again after loading the derived file, prove:

- source SHA256 is exactly the frozen SHA;
- `segment_ids` unchanged exactly;
- `observed_mask` unchanged exactly;
- `observed_targets` unchanged exactly including NaN pattern/finite values;
- no observed row has pseudo acceptance;
- each task permutation has zero fixed points among missing-row source/destination identities;
- missing counts remain `2665/3660`;
- accepted counts remain `376/1801`;
- accepted class balance remains D `376/0`, P `212/1589`;
- pseudo target/reliability/class/calibrated-probability tuple multiset over missing rows is preserved exactly per task;
- rejected entries still have pseudo_target NaN, reliability 0, pseudo_class -1;
- accepted pseudo targets remain equal to calibrated_audio_probs at accepted entries;
- accepted reliability remains finite in [0,1];
- source file SHA is unchanged after generation;
- derived cache SHA256 is recorded;
- derived cache loads successfully through the existing `wsm_ramps_semantic_datamodule`.

Record per-task permutation SHA256 and the number of rows whose acceptance assignment differs from the original.

The purpose is to break sample-to-pseudo alignment while preserving coverage/class/reliability distributions.

## 3. Frozen matched reference

Use these accepted matched-pseudo Equal results:

| Seed | D Score | P Score | Mean |
|---|---:|---:|---:|
| 42 | 0.712498 | 0.843342 | 0.777920 |
| 43 | 0.703299 | 0.856043 | 0.779671 |
| 44 | 0.707977 | 0.851878 | 0.779927 |

Matched three-seed means:

- D `0.707925`
- P `0.850421`
- Mean `0.779173`

Matched three-seed sample std:

- D `0.004600`
- P `0.006475`
- Mean `0.001092`

## 4. Freeze three negative-control configs before training

Create:

    configs/wsm_mm_pd_dep_v1/ablations/31_shuffled_pseudo_equal_seed42.yaml
    configs/wsm_mm_pd_dep_v1/ablations/32_shuffled_pseudo_equal_seed43.yaml
    configs/wsm_mm_pd_dep_v1/ablations/33_shuffled_pseudo_equal_seed44.yaml

For each seed, copy the corresponding matched Equal config semantics exactly:

- seed42 from config 13;
- seed43 from config 19;
- seed44 from config 20.

The ONLY semantic changes are:

1. `data.params.pseudo_cache_path` points to the derived shuffled cache;
2. `run_name` identifies the Stage-6 shuffled negative control.

Required run names:

- `stage6_shuffled_pseudo_equal_seed42`
- `stage6_shuffled_pseudo_equal_seed43`
- `stage6_shuffled_pseudo_equal_seed44`

All other data/model/loss/optimizer/train/callback/logging settings MUST be identical to the matched Equal config for that seed.

## 5. Mandatory pre-run firewall commit

Before any production run:

1. build and audit the derived negative-control cache;
2. validate all three configs;
3. programmatically prove config equivalence to matched Equal except cache path/run_name;
4. build DataModule/model/loss for each seed config;
5. verify R3 trainable parameter count `403079`;
6. verify derived cache accepted counts/class balance;
7. run a tiny TRAIN-only forward/loss/backward at `pseudo_scale=1.0`:
   - finite loss;
   - observed gradients non-zero;
   - accepted missing-head pseudo gradients non-zero;
   - pseudo target/reliability receive no gradients;
   - main heads, projections, task queries, and gate have finite gradients;
   - no optimizer step;
   - no DEV/Test loader iteration;
8. append exact evidence to PROGRESS_EN.md;
9. commit and push one firewall commit.

No production run may begin before that firewall commit exists on origin.

## 6. Exact production sequence

Run exactly three new production invocations:

1. shuffled Equal seed42;
2. shuffled Equal seed43;
3. shuffled Equal seed44.

Use exactly:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path <authorized-config>

No sweep.

No rerun for metric improvement.

## 7. Selection and Test firewall

For each run:

- checkpoint/epoch selected only by maximum `dev/mean_score`;
- freeze selected checkpoint before reading same-epoch Test monitoring;
- TEST_NONE/SOFT/HARD remain mandatory monitoring only;
- Test must not affect interpretation of the negative control.

Record selected checkpoint SHA256.

## 8. Frozen negative-control interpretation

After all three shuffled runs, compute:

- shuffled D/P/Mean per seed;
- shuffled three-seed mean/sample std;
- same-seed `matched - shuffled` deltas;
- three-seed matched-minus-shuffled deltas.

Evidence SUPPORTS sample-specific pseudo alignment only if BOTH are true:

1. matched Equal DEV Mean > shuffled Equal DEV Mean on at least 2 of 3 seeds;
2. matched Equal three-seed Mean > shuffled Equal three-seed Mean.

If either condition fails, record exactly:

    SAMPLE-SPECIFIC PSEUDO ALIGNMENT CLAIM NOT SUPPORTED BY THIS NEGATIVE CONTROL

This interpretation is about sample-specific semantic alignment only.

Do NOT claim correctness of the missing disease labels.

## 9. DEV-only calibration audit

After all selected checkpoints are frozen, compare matched Equal vs shuffled Equal on identical DEV rows/masks for each seed.

For D/P record:

- observed count;
- Brier;
- ECE-15.

Compute three-seed mean calibration deltas:

    matched - shuffled

Diagnostics only.

No recalibration, threshold changes, or reruns.

## 10. Required evidence

Record in PROGRESS_EN.md:

- source cache SHA;
- derived cache absolute path/SHA;
- negative-control metadata;
- permutation hashes;
- zero-fixed-point proof;
- original-vs-derived invariant table;
- acceptance-assignment Hamming differences;
- exact three configs and validation;
- firewall command/results;
- exact three production commands;
- run directories;
- MLflow IDs/status/artifact URI;
- epochs/early stopping;
- selected DEV-only epoch/checkpoint;
- checkpoint SHA256;
- full DEV D/P UAR/MF1/Score and Mean;
- same-epoch TEST_NONE/SOFT/HARD monitoring only after DEV freeze;
- three-seed matched/shuffled tables;
- frozen negative-control interpretation;
- calibration audit;
- explicit no-Test-driven-decision statement.

## 11. Scope checks

Run exactly:

    git diff --check
    git diff origin/main -- src
    git status --short
    git diff --stat origin/main...HEAD
    git log -10 --oneline --decorate

The source diff MUST be empty.

The generated negative-control cache is an external artifact and MUST NOT be committed to Git.

## Acceptance criteria

TASK-006A-BUNDLE passes only if:

- branch is exactly `codex/task-006a` from current manager-updated `origin/main`;
- only the script, three ablation configs, and PROGRESS change;
- original cache remains byte-identical;
- derived cache is deterministic and clearly marked as a negative control;
- row/observed-truth identity is unchanged;
- pseudo tuple is deranged only within each task's missing rows;
- coverage/class balance/value multisets are preserved;
- existing semantic DataModule accepts the derived cache;
- all three configs differ from matched Equal only by cache path/run_name;
- firewall commit exists before production;
- exactly three production runs occur;
- no source changes;
- no tuning/reruns;
- DEV-only selection;
- Test monitoring only;
- selected checkpoint SHA256 recorded;
- three-seed negative-control interpretation applied exactly;
- DEV-only calibration audit complete;
- branch pushed;
- main/master untouched.

Passing TASK-006A-BUNDLE closes Stage-6 negative-control item 8 for the matched Equal reference. It does not authorize another Stage-6 task.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch `codex/task-006a`;
- firewall commit SHA;
- final evidence commit SHA;
- pushed-to-origin status;
- main/master untouched;
- source diff empty;
- source and derived cache SHA256;
- permutation hashes/fixed-point counts;
- preserved coverage/class balance;
- acceptance-assignment Hamming differences;
- number of production runs = 3;
- matched vs shuffled seed42/43/44 D/P/Mean table;
- three-seed mean/std and deltas;
- frozen sample-specific-pseudo interpretation result;
- calibration summary;
- no Test-driven decision;
- no post-hoc tuning;
- no missing-label correctness/comorbidity claim;
- Stage 6 active;
- Stage-5 optimization closed; prior negative robustness interpretation superseded by matched-seed audio confirmation;
- no Stage7/Text/Final Test.

For section 6 write only:

    Manager review of TASK-006A-BUNDLE; do not start another task.

Stop after TASK-006A-BUNDLE.
