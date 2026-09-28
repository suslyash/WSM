# TASK-006A-FIX2: Bind Configs to the Verified Exact Cache and Rerun

## Authority and branch

This task follows **MANAGER-DECISION-062**.

Required branch:

    codex/task-006a

This is the same logical TASK-006A branch. Do not create another Codex branch.

First synchronize the existing branch:

    git fetch origin
    git checkout codex/task-006a
    git merge --no-edit origin/main

This merge of current `origin/main` into the existing task branch is manager-authorized.

Do not reset, rebase, overwrite, delete, or force-push the branch.

## Why FIX2 is required

FIX1 corrected the builder and generated the exact cache, but the three production configs still referenced the old rejected cache.

At both FIX1 firewall commit `096b720c875b427fc6d17d1dd04b65a67614b802` and final commit `d5fbbc5b9ac0fb7136996caf4090fd5d309f5fe7`, configs31/32/33 contain:

    pseudo_cache_path: /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_negative_control/shuffled_missing_targets_seed6001.pt

That path is WRONG for accepted TASK-006A evidence.

The required verified exact cache is:

    /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_negative_control/shuffled_missing_targets_seed6001_exact_fix1.pt

Required SHA256:

    e4baf2a2eea9dc7703e73b7956971b0d571eedda55fc8e6984750270712cf0f9

The FIX1 production metrics/checkpoint hashes exactly reproduced the earlier rejected runs and are superseded evidence only.

## Goal

Perform only the cache-binding correction:

1. verify the existing exact cache read-only;
2. point configs31/32/33 to that exact cache;
3. assign distinct FIX2 run names;
4. firewall the resolved configs/DataModule;
5. execute exactly three new production runs seeds42/43/44;
6. recompute the frozen negative-control claim and calibration audit using only these FIX2 runs.

No tuning, no new cache generation, no architecture/loss/optimizer/warm-up change.

## Required reading

Read in this exact order:

1. `AGENTS.md`
2. `docs/README.md`
3. `docs/PROJECT_REQUIREMENTS.md`
4. `docs/PROGRESS_EN.md`
5. `docs/plan/STAGE_6.md`
6. `docs/NEXT_TASK_EN.md`
7. `scripts/common/build_ramps_shuffled_pseudo_negative_control.py`
8. `src/fusion/data/wsm_ramps_semantic_datamodule.py`
9. `configs/wsm_mm_pd_dep_v1/fusion/13_r4_equal_seed42.yaml`
10. `configs/wsm_mm_pd_dep_v1/fusion/19_r4_equal_seed43.yaml`
11. `configs/wsm_mm_pd_dep_v1/fusion/20_r4_equal_seed44.yaml`
12. current configs31/32/33

Do not load historical archives or unrelated closed-stage plans.

## Accepted FIX1 artifact facts

The corrected builder already generated the exact external cache. Do NOT rebuild it and do NOT overwrite it.

Exact cache path:

    /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_negative_control/shuffled_missing_targets_seed6001_exact_fix1.pt

Exact cache SHA256:

    e4baf2a2eea9dc7703e73b7956971b0d571eedda55fc8e6984750270712cf0f9

Frozen source SHA256:

    17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945

Permutation SHA256:

- D: `1dcfde36463f21d2d2d525d70f6642c6b599cc23534f3cb47ffc41953ec4d4b8`
- P: `847f8e2cf156cd0d4ecb28f5269d5cc79fd5943c6605f44c57c78528ecc6d767`

Fixed points D/P:

    0 / 0

Acceptance-assignment Hamming D/P:

    646 / 1788

Missing counts D/P:

    2665 / 3660

Accepted counts D/P:

    376 / 1801

Accepted class counts:

- D positive/negative `376 / 0`
- P positive/negative `212 / 1589`

Mapping:

    destination_order = order
    source_order = roll(order, 1)
    derived[field][order, task] = source[field][source_order, task]

## Read-only exact-cache revalidation

Before modifying configs or running production, verify the existing exact cache without rewriting it:

- path exists;
- SHA256 exactly `e4baf2a2eea9dc7703e73b7956971b0d571eedda55fc8e6984750270712cf0f9`;
- metadata source SHA matches frozen source;
- metadata mapping is `destination_order_to_roll1_source_order`;
- recorded permutation hashes match the frozen D/P hashes above;
- fixed points are `0/0`;
- accepted counts/class balance match above;
- source cache SHA remains frozen;
- existing semantic DataModule can load this exact cache.

If exact-cache SHA or metadata does not match, STOP and report. Do not regenerate or overwrite it.

## Allowed tracked modifications in FIX2

Modify only:

- `configs/wsm_mm_pd_dep_v1/ablations/31_shuffled_pseudo_equal_seed42.yaml`
- `configs/wsm_mm_pd_dep_v1/ablations/32_shuffled_pseudo_equal_seed43.yaml`
- `configs/wsm_mm_pd_dep_v1/ablations/33_shuffled_pseudo_equal_seed44.yaml`
- `docs/PROGRESS_EN.md`

Do not modify the corrected builder further.
Do not modify any `src/*` file.

## Exact config correction

For each config, copy the corresponding matched Equal semantics exactly.

The ONLY semantic differences from matched Equal must be:

1. `data.params.pseudo_cache_path` =
   `/media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_negative_control/shuffled_missing_targets_seed6001_exact_fix1.pt`
2. `experiment_info.params.run_name` as below.

Required FIX2 run names:

- seed42: `stage6_shuffled_pseudo_equal_seed42_exact_fix2`
- seed43: `stage6_shuffled_pseudo_equal_seed43_exact_fix2`
- seed44: `stage6_shuffled_pseudo_equal_seed44_exact_fix2`

Everything else must be identical to configs13/19/20 respectively.

## Mandatory FIX2 firewall

Before any new production run:

1. verify exact cache read-only as above;
2. validate all three corrected configs;
3. programmatically prove each config differs from matched Equal only by cache path/run_name;
4. instantiate DataModule/model/loss for each;
5. verify the instantiated DataModule actually resolved the exact cache path ending in `shuffled_missing_targets_seed6001_exact_fix1.pt`;
6. verify loaded cache SHA is `e4baf2a2...`;
7. verify R3 trainable parameters `403079`;
8. run TRAIN-only forward/loss/backward at `pseudo_scale=1.0` with:
   - finite loss;
   - nonzero observed-supervision model gradient;
   - nonzero accepted missing-head pseudo-supervision model gradient;
   - no pseudo-target/reliability gradients;
   - finite heads/projections/task-query/gate gradients;
   - no optimizer step;
   - no DEV/Test iteration;
9. `git diff --check`;
10. `git diff origin/main -- src` MUST be empty;
11. append firewall evidence to PROGRESS;
12. commit and push one NEW firewall commit on `codex/task-006a`.

No FIX2 production run may begin before the new firewall commit is visible on origin.

## Exactly three NEW FIX2 production runs

Run exactly in order:

1. seed42;
2. seed43;
3. seed44.

Commands:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/ablations/31_shuffled_pseudo_equal_seed42.yaml

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/ablations/32_shuffled_pseudo_equal_seed43.yaml

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/ablations/33_shuffled_pseudo_equal_seed44.yaml

The six earlier shuffled runs (initial rejected + FIX1 wrong-binding) are superseded and may not be used, averaged, selected among, or counted.

No sweep, extra seed, retry for metric improvement, or post-hoc change.

## Selection/Test firewall

For each FIX2 run:

- select only maximum `dev/mean_score`;
- freeze epoch/checkpoint before reading same-epoch Test monitoring;
- record checkpoint SHA256;
- Test is monitoring only.

## Frozen matched reference and claim rule

Matched Equal:

| Seed | D | P | Mean |
|---:|---:|---:|---:|
| 42 | 0.712498 | 0.843342 | 0.777920 |
| 43 | 0.703299 | 0.856043 | 0.779671 |
| 44 | 0.707977 | 0.851878 | 0.779927 |

Matched three-seed Mean:

    0.7791726667

Using only matched Equal vs the three NEW FIX2 runs, compute:

- shuffled D/P/Mean each seed;
- shuffled three-seed means/sample std;
- same-seed `matched - FIX2` D/P/Mean;
- aggregate `matched - FIX2` deltas.

Support sample-specific pseudo alignment only if BOTH:

1. matched Mean > FIX2 Mean on at least 2/3 seeds;
2. matched three-seed Mean > FIX2 three-seed Mean.

If either fails, record exactly:

    SAMPLE-SPECIFIC PSEUDO ALIGNMENT CLAIM NOT SUPPORTED BY THIS NEGATIVE CONTROL

If both pass, report support only for sample-specific pseudo alignment.

No missing-label correctness, comorbidity, significance, or final-promotion claim.

## DEV-only calibration audit

On identical DEV rows/masks, for each seed/task report matched, FIX2, and `matched - FIX2` for:

- Brier;
- ECE-15.

Record observed counts and three-seed mean `matched - FIX2` deltas for D/P Brier/ECE.

No recalibration or rerun.

## Required evidence

PROGRESS must explicitly record:

- FIX1 production rejected because configs pointed to old cache;
- exact old wrong path and required exact path;
- exact cache SHA and read-only verification;
- corrected config paths/run names;
- resolved DataModule exact-cache path for all three configs;
- new firewall commit SHA;
- exactly three NEW FIX2 production commands/runs;
- run directories and MLflow identities;
- selected DEV epochs/checkpoints/SHA256;
- full DEV UAR/MF1/Score/Mean;
- Test monitoring after freeze only;
- matched-vs-FIX2 table/deltas;
- frozen claim result;
- calibration table and three-seed mean deltas;
- no tuning/rerun;
- no source changes.

## Final scope checks

Run:

    git diff --check
    git diff origin/main -- src
    git status --short
    git diff --stat origin/main...HEAD
    git log -15 --oneline --decorate

Source diff MUST be empty.

## Acceptance criteria

FIX2 passes only if:

- branch remains `codex/task-006a`;
- current manager main was merged before FIX2 work;
- corrected builder/history remains preserved;
- fresh exact cache is verified read-only with SHA `e4baf2a2...`;
- all three configs point to `shuffled_missing_targets_seed6001_exact_fix1.pt`;
- all three configs use distinct FIX2 run names;
- config equivalence otherwise passes;
- DataModule resolves the exact cache path;
- new firewall commit precedes all FIX2 runs;
- exactly three NEW runs occur;
- no source changes;
- DEV-only selection;
- claim/calibration use only FIX2 runs;
- branch pushed;
- main/master untouched.

Passing FIX2 closes only Stage-6 shuffled/mismatched negative-control item 8. It authorizes no next task.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Include branch, manager-main merge, new firewall/final SHA, exact cache path/SHA, config resolved paths, run count = 3 NEW FIX2 runs, metrics/deltas, calibration, no Test decision, no tuning/rerun, no source diff.

For section 6 write only:

    Manager review of TASK-006A-FIX2 on codex/task-006a; do not start another task.

Stop.
