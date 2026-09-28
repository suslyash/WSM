# TASK-006A-FIX1: Exact Shuffled-Pseudo Derangement Correction and Three-Seed Rerun

## Authority and branch

This corrective task follows **MANAGER-DECISION-060**.

Required branch:

    codex/task-006a-fix1

Start from the current manager-updated `origin/main`.

Create exactly one fresh task branch from that `origin/main`.

Do **not** reuse, reset, overwrite, force-push, merge, or rebase the rejected branch `codex/task-006a`. It remains preserved as rejected evidence.

Stage-5 optimization is CLOSED. Stage 6 remains ACTIVE. This is a narrow reproducibility correction to TASK-006A, not a tuning task.

## Why FIX1 is required

Manager review rejected `codex/task-006a` despite its apparently strong matched-vs-shuffled result.

The rejected builder did:

    random_order = missing[randperm(...)]
    source_rows = roll(random_order, 1)
    derived[field][missing, task] = source[field][source_rows, task]

This is **not** the frozen mapping.

The required mapping is:

    order = missing[randperm(...)]
    source_order = roll(order, 1)
    derived[field][order, task] = source[field][source_order, task]

The rejected builder also included a conditional fixed-point repair. That repair is forbidden. A one-position circular shift of a unique `order` already guarantees zero destination/source fixed points for these task sizes.

The rejected branch also reported calibration deltas as `shuffled - matched` and omitted the required three-seed mean `matched - shuffled` calibration deltas.

Therefore the rejected cache and its three runs are superseded diagnostic evidence only and MUST NOT be used to close the claim gate.

## Goal

Re-execute the Stage-6 shuffled/mismatched pseudo-target negative control using the **exact frozen within-task circular-shift derangement**.

Preserve every non-semantic factor:

- canonical rows;
- observed truth;
- accepted counts;
- accepted class balance;
- pseudo target/reliability/class/calibrated-probability tuple multiset;
- R3 architecture;
- Equal loss;
- optimizer;
- warm-up;
- seeds.

Run exactly three new corrective production trainings: seeds42,43,44.

No sweep. No tuning. No rerun for metric improvement.

## Required reading

Read in this exact order:

1. `AGENTS.md`
2. `docs/README.md`
3. `docs/PROJECT_REQUIREMENTS.md`
4. `docs/PROGRESS_EN.md`
5. `docs/plan/STAGE_6.md`
6. `docs/NEXT_TASK_EN.md`
7. `src/fusion/data/wsm_ramps_semantic_datamodule.py`
8. `src/fusion/models/av_r3_disease_query.py`
9. `src/fusion/loss/r4_ramps_balance_loss.py`
10. `configs/wsm_mm_pd_dep_v1/fusion/13_r4_equal_seed42.yaml`
11. `configs/wsm_mm_pd_dep_v1/fusion/19_r4_equal_seed43.yaml`
12. `configs/wsm_mm_pd_dep_v1/fusion/20_r4_equal_seed44.yaml`

Do not load the historical archive or closed-stage plans unless a concrete verification issue requires them.

You MAY inspect the rejected `codex/task-006a` builder only to verify the manager-identified defect. Do not copy its mapping or its rejected result into the accepted evidence.

## Allowed tracked files

Codex may add/modify only:

- `scripts/common/build_ramps_shuffled_pseudo_negative_control.py`
- `configs/wsm_mm_pd_dep_v1/ablations/31_shuffled_pseudo_equal_seed42.yaml`
- `configs/wsm_mm_pd_dep_v1/ablations/32_shuffled_pseudo_equal_seed43.yaml`
- `configs/wsm_mm_pd_dep_v1/ablations/33_shuffled_pseudo_equal_seed44.yaml`
- `docs/PROGRESS_EN.md`

No source file may change.

## Frozen source cache

Source:

    /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt

Required source SHA256:

    17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945

Rows:

    6325

Missing D/P:

    2665 / 3660

Accepted D/P:

    376 / 1801

Accepted class counts:

- D positive/negative: `376 / 0`
- P positive/negative: `212 / 1589`

The source cache is immutable.

## Fresh corrective external cache

The rejected external cache must remain untouched.

Use this new output path:

    /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_negative_control/shuffled_missing_targets_seed6001_exact_fix1.pt

The builder MUST fail rather than overwrite an existing output path.

Frozen corrective invocation:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 \
      scripts/common/build_ramps_shuffled_pseudo_negative_control.py \
      --source /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt \
      --output /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_negative_control/shuffled_missing_targets_seed6001_exact_fix1.pt \
      --shuffle-seed 6001

If this exact FIX1 path already exists before the task begins, stop and report the conflict. Do not delete or overwrite it.

## Exact permutation contract — MUST be implemented literally

For each task independently:

1. compute the canonical missing-row IDs:

       missing = (~observed_mask[:, task]).nonzero(...).flatten()

2. construct a CPU `torch.Generator`:
   - D seed = `6001`
   - P seed = `6002`

3. construct exactly:

       order = missing[torch.randperm(missing.numel(), generator=generator)]

4. construct exactly:

       source_order = torch.roll(order, shifts=1, dims=0)

5. assert:

       torch.all(order != source_order)

6. for EACH field below, assign exactly:

       derived[field][order, task] = source[field][source_order, task].clone()

Permuted fields:

- `pseudo_accept_mask`
- `pseudo_targets`
- `pseudo_reliability`
- `pseudo_class`
- `calibrated_audio_probs`

Forbidden mapping changes:

- do not assign shuffled sources to canonical `missing` destinations;
- do not sort `order` before assignment;
- do not repair, swap, rotate, or modify fixed points after the one-position roll;
- do not add any conditional permutation repair;
- do not perform rejection sampling;
- do not change the per-task seeds.

Observed rows must never be a source or destination.

## Permutation audit

For each task retain and audit both tensors:

- `destination_order = order`;
- `source_order = roll(order, 1)`.

Record a permutation SHA256 over the exact ordered int64 pair matrix:

    stack([destination_order, source_order], dim=1)

Record:

- destination/source fixed points = exactly 0;
- destination set equals the complete task missing-row set;
- source set equals the complete task missing-row set;
- source_order equals `roll(destination_order, 1)` exactly;
- no repair path exists or executes.

Add negative-control metadata including:

- type;
- source cache SHA256;
- shuffle seed;
- D/P permutation SHA256;
- permuted fields;
- fixed-point counts;
- acceptance-assignment Hamming differences;
- explicit `mapping: destination_order_to_roll1_source_order`.

Do not change the canonical cache version.

## Mandatory invariant audit — before save and after reload

Prove all of the following:

- source SHA before generation equals the frozen SHA;
- source SHA **after generation** still equals the frozen SHA;
- `segment_ids` unchanged exactly;
- `observed_mask` unchanged exactly;
- `observed_targets` unchanged exactly, including NaN pattern and finite values;
- explicitly assert that no observed row has pseudo acceptance;
- missing counts remain D/P `2665/3660`;
- accepted counts remain D/P `376/1801`;
- class counts remain D `376/0`, P `212/1589`;
- complete missing-row tuple multiset is identical per task;
- rejected entries have target NaN, reliability0, class -1;
- accepted target equals calibrated audio probability;
- accepted reliability is finite and in [0,1];
- exact roll-1 permutation relation holds;
- zero destination/source fixed points;
- derived cache SHA256 is recorded;
- derived cache loads through the existing semantic DataModule.

Record D/P acceptance-assignment Hamming differences.

## Frozen matched Equal reference

| Seed | D Score | P Score | Mean |
|---:|---:|---:|---:|
| 42 | 0.712498 | 0.843342 | 0.777920 |
| 43 | 0.703299 | 0.856043 | 0.779671 |
| 44 | 0.707977 | 0.851878 | 0.779927 |

Three-seed matched:

- D mean `0.707925`
- P mean `0.850421`
- Mean `0.7791726667`
- Mean sample std `0.0010923664`

Do not use the rejected shuffled results from `codex/task-006a` for the claim decision.

## Corrective configs

Create the same tracked config paths:

    configs/wsm_mm_pd_dep_v1/ablations/31_shuffled_pseudo_equal_seed42.yaml
    configs/wsm_mm_pd_dep_v1/ablations/32_shuffled_pseudo_equal_seed43.yaml
    configs/wsm_mm_pd_dep_v1/ablations/33_shuffled_pseudo_equal_seed44.yaml

Each must copy the corresponding matched Equal config semantics exactly.

Allowed semantic differences only:

1. `data.params.pseudo_cache_path` points to the fresh FIX1 cache;
2. `run_name` uses the distinct FIX1 name below.

Required run names:

- `stage6_shuffled_pseudo_equal_seed42_exact_fix1`
- `stage6_shuffled_pseudo_equal_seed43_exact_fix1`
- `stage6_shuffled_pseudo_equal_seed44_exact_fix1`

Everything else must remain identical to matched Equal for the same seed.

## Mandatory FIX1 pre-run firewall commit

Before any corrective production run:

1. build the fresh FIX1 cache;
2. perform all pre-save/post-load invariants above;
3. validate all three configs;
4. programmatically prove config equivalence except cache path/run_name;
5. build DataModule/model/loss for each config;
6. verify R3 trainable params = `403079`;
7. run TRAIN-only forward/loss/backward at `pseudo_scale=1.0`;
8. explicitly verify:
   - finite loss;
   - at least one observed-supervision contribution has non-zero model gradient;
   - at least one accepted missing-head pseudo-supervision contribution has non-zero model gradient;
   - pseudo target/reliability tensors have no gradients;
   - main heads have finite gradients;
   - projections have finite gradients;
   - task queries have finite gradients;
   - gate has finite gradients;
   - no optimizer step;
   - no DEV/Test loader iteration;
9. run `git diff --check`;
10. run `git diff origin/main -- src` and require empty output;
11. append exact firewall evidence to PROGRESS;
12. commit and push one firewall commit to `origin/codex/task-006a-fix1`.

No corrective production run may start before the firewall commit is visible on origin.

## Exactly three corrective production runs

Run exactly, in order:

1. FIX1 seed42;
2. FIX1 seed43;
3. FIX1 seed44.

Commands:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/ablations/31_shuffled_pseudo_equal_seed42.yaml

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/ablations/32_shuffled_pseudo_equal_seed43.yaml

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/ablations/33_shuffled_pseudo_equal_seed44.yaml

The three rejected runs on `codex/task-006a` are not counted as accepted TASK-006A evidence; they must not be repeated, reused, averaged, or selected among.

No sweep, extra seed, retry for metric improvement, or post-hoc config change.

## Checkpoint/Test firewall

For each FIX1 run:

- checkpoint/epoch selected only by maximum `dev/mean_score`;
- freeze selected checkpoint identity before reading same-epoch Test monitoring;
- record checkpoint SHA256;
- TEST_NONE/SOFT/HARD are monitoring only and may not affect the claim interpretation.

## Frozen claim rule

Using only matched Equal and the three **FIX1** shuffled results, compute:

- FIX1 shuffled D/P/Mean per seed;
- FIX1 shuffled three-seed mean and sample std;
- same-seed `matched - FIX1_shuffled` D/P/Mean deltas;
- aggregate `matched - FIX1_shuffled` deltas.

Sample-specific pseudo alignment is supported only if BOTH:

1. matched DEV Mean > FIX1 shuffled DEV Mean on at least 2/3 seeds;
2. matched three-seed DEV Mean > FIX1 shuffled three-seed DEV Mean.

If either fails, record exactly:

    SAMPLE-SPECIFIC PSEUDO ALIGNMENT CLAIM NOT SUPPORTED BY THIS NEGATIVE CONTROL

If both pass, report support narrowly as sample-specific pseudo-alignment evidence only.

Do not claim missing-label correctness, comorbidity recovery, significance, or final promotion.

## DEV-only calibration audit — corrected direction

Use identical DEV rows/masks.

For each seed and D/P report:

- observed count;
- matched Brier;
- FIX1 shuffled Brier;
- `matched - FIX1_shuffled` Brier delta;
- matched ECE-15;
- FIX1 shuffled ECE-15;
- `matched - FIX1_shuffled` ECE-15 delta.

Then compute and record **three-seed mean matched-minus-shuffled deltas** for:

- D Brier;
- D ECE-15;
- P Brier;
- P ECE-15.

This is diagnostic only. No recalibration or rerun.

## Required PROGRESS evidence

Record:

- rejected `codex/task-006a` status = NOT MERGED / superseded diagnostic evidence;
- FIX1 task/branch;
- source SHA before and after generation;
- fresh FIX1 cache path/SHA;
- exact destination/source mapping contract;
- D/P permutation hashes;
- fixed points;
- destination/source set equality;
- acceptance Hamming;
- all invariant results;
- config-equivalence proof;
- firewall commands/results and firewall commit SHA;
- exact three corrective commands;
- production count = exactly 3 corrective runs;
- run directories / MLflow IDs/status/artifact URI;
- selected epochs/checkpoints/SHA256;
- full DEV D/P UAR/MF1/Score/Mean;
- same-epoch Test monitoring only after freeze;
- matched-vs-FIX1 shuffled tables;
- three-seed means/std and matched-minus-shuffled deltas;
- frozen claim interpretation;
- corrected calibration table/deltas and three-seed mean calibration deltas;
- no Test decision;
- no tuning/rerun;
- no `src` changes.

## Final scope checks

Run exactly:

    git diff --check
    git diff origin/main -- src
    git status --short
    git diff --stat origin/main...HEAD
    git log -10 --oneline --decorate

Only the builder, three ablation configs, and PROGRESS may differ from origin/main.

## Acceptance criteria

FIX1 passes only if:

- branch exactly `codex/task-006a-fix1` from current manager main;
- rejected branch is untouched and unmerged;
- exact destination-order -> roll1-source-order mapping is implemented literally;
- no fixed-point repair exists;
- fresh external FIX1 cache is used;
- source remains byte-identical;
- all tuple/count/class/observed invariants pass;
- source SHA is rechecked after generation;
- semantic DataModule loads the FIX1 cache;
- configs differ only cache path/run_name;
- explicit gradient firewall evidence passes;
- firewall commit precedes all three corrective runs;
- exactly three corrective production runs occur;
- no source changes;
- DEV-only selection;
- Test monitoring only;
- claim rule uses only matched Equal vs FIX1 runs;
- calibration deltas use `matched - FIX1_shuffled` and include three-seed mean deltas;
- branch pushed;
- main/master untouched.

Passing FIX1 closes only Stage-6 shuffled/mismatched negative-control item 8. It authorizes no next task.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch `codex/task-006a-fix1`;
- rejected branch `codex/task-006a` untouched/unmerged;
- firewall SHA;
- final evidence SHA;
- pushed status;
- main/master untouched;
- source diff empty;
- source SHA before/after;
- fresh FIX1 derived SHA;
- exact mapping statement;
- permutation hashes/fixed points;
- acceptance Hamming;
- preserved coverage/class balance;
- production runs = exactly 3 corrective runs;
- matched vs FIX1 shuffled seed42/43/44 D/P/Mean;
- three-seed mean/std and matched-minus-shuffled deltas;
- frozen claim result;
- calibration matched-minus-shuffled per-seed and mean summary;
- no Test-driven decision;
- no tuning/rerun;
- no missing-label correctness/comorbidity/significance/final-promotion claim;
- Stage 6 active;
- Stage-5 optimization closed;
- no Stage7/Text/Final Test.

For section 6 write only:

    Manager review of TASK-006A-FIX1; do not start another task.

Stop.
