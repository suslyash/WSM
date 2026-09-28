# TASK-006E: Uncertainty/Reliability Ablation on Optimized R4 Trial-012

## Authority and branch

This task follows **MANAGER-DECISION-066**.

Required branch:

    codex/task-006e

Start from current manager-updated `origin/main`.

Create exactly one new task branch from that main.

This is a Stage-6 component ablation. Stage-5 optimization remains closed.

## Scientific question

Does the **graded uncertainty/reliability signal** contribute repeatably to the leading optimized R4 trial-012 composition?

This task preserves the semantic pseudo subset and pseudo targets, but replaces accepted reliability values with uniform 1.0.

That removes graded reliability from BOTH:

1. reliability-weighted direct pseudo BCE;
2. reliability EMA used by the RA controller.

It does NOT remove:

- pseudo acceptance;
- pseudo targets;
- pseudo classes;
- calibrated audio probabilities;
- direct pseudo supervision itself;
- semantic evidence deciding which rows are accepted;
- architecture;
- RA-STCH controller mechanics;
- auxiliary/agreement losses;
- optimizer;
- warm-up;
- any optimized trial-012 hyperparameter.

Therefore this is the Stage-6 **uncertainty/reliability** ablation, not a semantic-evidence ablation.

## Required reading

Read in this exact order:

1. `AGENTS.md`
2. `docs/README.md`
3. `docs/PROJECT_REQUIREMENTS.md`
4. `docs/PROGRESS_EN.md`
5. `docs/plan/STAGE_6.md`
6. `docs/NEXT_TASK_EN.md`
7. `src/fusion/data/wsm_ramps_semantic_datamodule.py`
8. `src/fusion/loss/r4_ramps_balance_loss.py`
9. `src/common/callbacks/wsm_r4_balance_callback.py`
10. configs37/38/39.

## Allowed tracked files

Codex may add/modify only:

- `scripts/common/build_ramps_uniform_reliability_ablation.py`
- `configs/wsm_mm_pd_dep_v1/ablations/43_uniform_reliability_seed42.yaml`
- `configs/wsm_mm_pd_dep_v1/ablations/44_uniform_reliability_seed43.yaml`
- `configs/wsm_mm_pd_dep_v1/ablations/45_uniform_reliability_seed44.yaml`
- `docs/PROGRESS_EN.md`

No `src/*` file may change.
No existing config may change.

## Frozen source pseudo cache

Path:

    /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt

SHA256:

    17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945

Rows:

    6325

Missing D/P:

    2665 / 3660

Accepted D/P:

    376 / 1801

Accepted classes:

- D positive/negative: `376 / 0`
- P positive/negative: `212 / 1589`

The source cache is immutable.

## Derived uniform-reliability cache

Create exactly one external derived cache:

    /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_reliability_ablation/uniform_accepted_reliability_v1.pt

The builder MUST fail rather than overwrite an existing path.

Required CLI:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python       scripts/common/build_ramps_uniform_reliability_ablation.py       --source /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt       --output /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_reliability_ablation/uniform_accepted_reliability_v1.pt

## Exact derived-cache transformation

Load the source cache read-only.

Copy all fields.

Modify ONLY `pseudo_reliability`:

- where `pseudo_accept_mask == True`, set reliability to exactly `1.0`;
- everywhere else, set/retain reliability exactly `0.0`.

Do not modify:

- `version`;
- `task_names`;
- `segment_ids`;
- `observed_mask`;
- `observed_targets`;
- `pseudo_accept_mask`;
- `pseudo_targets`;
- `pseudo_class`;
- `calibrated_audio_probs`;
- teacher identities;
- CLIP identity;
- prompt-bank identity;
- any threshold/calibration/acceptance metadata.

You MAY add one top-level metadata mapping:

    reliability_ablation

containing:

- type = `uniform_accepted_reliability`;
- source_cache_sha256;
- accepted_value = 1.0;
- rejected_or_observed_value = 0.0;
- source accepted reliability summary by task;
- derived cache SHA256 recorded externally in PROGRESS after save.

Do not change the canonical cache version.

## Mandatory cache invariants

Before save and after reload verify:

- source SHA equals frozen SHA before generation;
- source SHA still equals frozen SHA after generation;
- all non-reliability top-level tensor/list fields above are exactly unchanged;
- segment IDs identical and unique;
- observed truth identical including NaN patterns;
- acceptance mask identical;
- pseudo targets identical including NaN patterns;
- pseudo classes identical;
- calibrated audio probabilities identical;
- accepted counts D/P `376/1801`;
- accepted classes D `376/0`, P `212/1589`;
- accepted reliability exactly 1.0 for every accepted entry;
- rejected/observed reliability exactly 0.0;
- accepted pseudo targets still equal calibrated audio probabilities;
- no observed row is pseudo-accepted;
- existing semantic DataModule loads the derived cache successfully.

Record source reliability min/mean/max per task over accepted rows before replacement.

Record the number of accepted reliability entries that changed from source to derived per task.

Record derived cache SHA256.

## Frozen full reference

Full optimized trial-012:

| Seed | D | P | Mean |
|---:|---:|---:|---:|
| 42 | 0.759025 | 0.879259 | 0.8191424538 |
| 43 | 0.718527 | 0.804363 | 0.7614450000 |
| 44 | 0.728217 | 0.832060 | 0.7801390000 |

Three-seed full:

- D mean/std `0.7352563333/0.0211467766`;
- P mean/std `0.8385606667/0.0378688091`;
- Mean `0.7869088179/0.0294384420`.

Checkpoint SHA256:

- seed42 `104b79e409832839502b4d20a98c054f5efda633930e313ce74f34327e668b2a`;
- seed43 `6af4a4ed21040aff4db2906b0adc8aa5fd27a4b72c375fbb67f68de04ab8446e`;
- seed44 `b06fec6912b27cd19d811b72b2c3c08cd1b0a6c826b637f87a314e3976b88c17`.

Verify these files/hashes before production.

## Exact ablation configs

Create configs43/44/45 from full configs37/38/39 respectively.

Allowed semantic differences only:

1. `experiment_info.params.run_name`;
2. `data.params.pseudo_cache_path` points to the derived uniform-reliability cache.

Required run names:

- seed42: `stage6_uniform_reliability_trial012_seed42`
- seed43: `stage6_uniform_reliability_trial012_seed43`
- seed44: `stage6_uniform_reliability_trial012_seed44`

Everything else must be identical to the corresponding full reference config.

## Isolation contract

The ablation must preserve the exact accepted pseudo rows and exact pseudo targets.

Direct pseudo supervision remains enabled with the original warm-up schedule ending at scale 1.0.

Only reliability magnitudes are neutralized to 1.0 for accepted rows.

Because the same reliability field also feeds the RA reliability EMA, this task tests the **combined graded reliability contribution** in the current implemented method.

Do not claim that it isolates BCE weighting separately from controller reliability. It does not.

## Mandatory pre-run firewall

Before any production run:

1. build and audit the derived cache;
2. verify source and derived SHA values;
3. validate configs43/44/45;
4. prove each differs from configs37/38/39 only by run_name/cache path;
5. prove seeds remain 42/43/44;
6. instantiate DataModule/model/loss/callbacks;
7. verify the DataModule resolved the derived cache path/SHA;
8. verify trainable params match full trial-012;
9. verify accepted masks/targets/classes/counts are unchanged;
10. verify accepted reliability is exactly 1.0 and rejected/observed is 0.0;
11. run TRAIN-only forward/loss/backward with accepted pseudo rows present;
12. verify finite nonzero observed and pseudo-supervision model gradients;
13. verify pseudo targets/reliability are detached;
14. verify direct pseudo loss remains active when pseudo_scale > 0;
15. verify RA reliability EMA updates to exactly 1.0 after an accepted batch for the corresponding active task(s);
16. verify controller diagnostics finite;
17. no optimizer step;
18. no DEV/Test loader iteration;
19. `git diff --check`;
20. `git diff origin/main -- src` empty;
21. append firewall evidence to PROGRESS;
22. commit and push one firewall commit.

No production run before the firewall commit exists on origin.

## Exactly three production runs

Run exactly in order:

1. seed42;
2. seed43;
3. seed44.

Commands:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train       --config-path configs/wsm_mm_pd_dep_v1/ablations/43_uniform_reliability_seed42.yaml

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train       --config-path configs/wsm_mm_pd_dep_v1/ablations/44_uniform_reliability_seed43.yaml

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train       --config-path configs/wsm_mm_pd_dep_v1/ablations/45_uniform_reliability_seed44.yaml

No sweep.
No retry for metric improvement.
No extra seed.
No post-firewall config/cache change.

## Selection/Test firewall

For each run:

- select checkpoint only by max `dev/mean_score`;
- freeze checkpoint identity before reading same-epoch Test monitoring;
- record checkpoint SHA256;
- Test cannot affect interpretation.

## Three-seed comparison

Compute uniform-reliability D/P/Mean by seed and three-seed mean/sample std/range.

Compute same-seed:

    full trial012 - uniform-reliability

for D, P, Mean.

Compute aggregate full-minus-ablation D/P/Mean.

## Frozen claim rule

Record exactly:

    UNCERTAINTY/RELIABILITY CONTRIBUTION SUPPORTED

only if ALL conditions hold:

1. full DEV Mean > uniform-reliability DEV Mean on at least 2/3 seeds;
2. full three-seed Mean > uniform-reliability three-seed Mean;
3. full three-seed D mean is not more than `0.010000` below uniform-reliability D mean;
4. full three-seed P mean is not more than `0.010000` below uniform-reliability P mean.

Otherwise record exactly:

    UNCERTAINTY/RELIABILITY CONTRIBUTION NOT SUPPORTED

This claim concerns the implemented graded reliability signal as a whole.

Do not claim pseudo-label correctness, semantic acceptance correctness, missing-label recovery, comorbidity recovery, significance, or final promotion.

## Controller diagnostics

At each DEV-selected epoch record:

- pseudo scale;
- alpha D/P;
- progress D/P;
- grad-norm EMA D/P;
- grad-cosine EMA;
- reliability EMA D/P.

For the ablation, reliability EMA is expected to be 1.0 once updated from accepted entries. Record whether this occurs.

## DEV-only calibration/gate audit

After checkpoint freeze, evaluate DEV only.

For each seed/task report:

- observed count;
- Brier;
- ECE-15;
- task-modality audio gate mean/std.

Report three-seed means.

Compare full vs uniform-reliability in `full - ablation` direction.

No recalibration.
No threshold search.
No Test rows.

## Required PROGRESS evidence

Record:

- task/branch;
- builder path;
- source and derived cache paths/SHA;
- cache invariant audit;
- source accepted reliability min/mean/max;
- changed accepted-reliability counts D/P;
- configs43/44/45;
- config equivalence proof;
- full checkpoint SHAs;
- firewall evidence/SHA;
- exact three production commands;
- production count exactly 3;
- run dirs / MLflow IDs / status / artifact URI;
- selected epochs/checkpoints/SHA;
- DEV D/P UAR/MF1/Score/Mean;
- Test monitoring only after freeze;
- three-seed ablation mean/std/range;
- full-minus-ablation per-seed/aggregate deltas;
- exact frozen claim string;
- controller diagnostics;
- DEV calibration/gate audit;
- no Test decision;
- no tuning/rerun;
- no source changes.

## Final scope checks

Run:

    git diff --check
    git diff origin/main -- src
    git status --short
    git diff --stat origin/main...HEAD
    git log -12 --oneline --decorate

Only builder, configs43/44/45, and PROGRESS may differ from origin/main.

The derived cache remains external and uncommitted.

## Acceptance criteria

TASK-006E passes only if:

- branch exactly `codex/task-006e` from current manager main;
- source cache unchanged;
- derived cache changes only reliability plus explicit ablation metadata;
- accepted rows/targets/classes/calibrated probabilities unchanged;
- accepted reliability exactly 1.0;
- rejected/observed reliability exactly 0.0;
- DataModule accepts derived cache;
- configs differ only run_name/cache path;
- firewall pushed before production;
- exactly three runs seeds42/43/44;
- DEV-only selection;
- Test monitoring only;
- frozen claim applied exactly;
- controller/calibration/gate diagnostics recorded;
- no tuning/sweep/retry;
- no source changes;
- branch pushed;
- main/master untouched.

Passing TASK-006E closes only Stage-6 uncertainty/reliability item 4. It authorizes no next task.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch `codex/task-006e`;
- firewall SHA;
- final evidence SHA;
- pushed status;
- main/master untouched;
- source diff empty;
- source/derived cache SHA;
- accepted reliability transformation/invariants;
- exactly three production runs;
- per-seed D/P/Mean;
- three-seed mean/std;
- full-minus-ablation deltas;
- exact frozen claim string;
- controller reliability EMA summary;
- calibration/gate summary;
- no pseudo correctness/semantic correctness/comorbidity/significance/final-promotion claim;
- no Test-driven decision;
- Stage 6 active;
- Stage-5 optimization closed;
- no Stage7/Text/Final Test.

For section 6 write only:

    Manager review of TASK-006E; do not start another task.

Stop.
