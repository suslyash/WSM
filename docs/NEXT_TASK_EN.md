# TASK-006D: Direct Pseudo-Supervision Ablation on Optimized R4 Trial-012

## Authority and branch

This task follows **MANAGER-DECISION-065**.

Required branch:

    codex/task-006d

Start from current manager-updated `origin/main`.

Create exactly one new task branch from that main.

This is a Stage-6 component ablation. Stage-5 optimization remains closed.

## Scientific question

Does the **direct accepted-pseudo BCE supervision term** contribute repeatably to the leading optimized R4 trial-012 composition?

This task removes only direct pseudo loss by keeping `pseudo_scale=0` for the entire training run.

It intentionally RETAINS:

- the exact semantic pseudo cache;
- pseudo accept masks;
- pseudo reliability values;
- RA reliability EMA;
- RA controller logic;
- architecture;
- auxiliary/agreement terms;
- optimizer;
- all optimized trial-012 hyperparameters.

Therefore TASK-006D does **not** test reliability contribution or semantic-acceptance contribution. Those remain separate future Stage-6 claims.

## Required reading

Read in this exact order:

1. `AGENTS.md`
2. `docs/README.md`
3. `docs/PROJECT_REQUIREMENTS.md`
4. `docs/PROGRESS_EN.md`
5. `docs/plan/STAGE_6.md`
6. `docs/NEXT_TASK_EN.md`
7. `src/fusion/loss/r4_ramps_balance_loss.py`
8. `src/common/callbacks/wsm_pseudo_scale_warmup_callback.py`
9. `src/common/callbacks/wsm_r4_balance_callback.py`
10. `src/fusion/data/wsm_ramps_semantic_datamodule.py`
11. `src/fusion/models/av_r3_disease_query.py`
12. configs37/38/39.

## Allowed tracked files

Codex may add/modify only:

- `configs/wsm_mm_pd_dep_v1/ablations/40_no_direct_pseudo_seed42.yaml`
- `configs/wsm_mm_pd_dep_v1/ablations/41_no_direct_pseudo_seed43.yaml`
- `configs/wsm_mm_pd_dep_v1/ablations/42_no_direct_pseudo_seed44.yaml`
- `docs/PROGRESS_EN.md`

No `src/*` file may change.
No existing config may change.
No script may change.

## Frozen full reference

Use configs37/38/39 as the full-method reference.

### Seed42

Config:

    configs/wsm_mm_pd_dep_v1/fusion/37_r4_ra_stch_optuna_selected_seed42.yaml

DEV D/P/Mean:

    0.759025 / 0.879259 / 0.8191424538

Checkpoint SHA256:

    104b79e409832839502b4d20a98c054f5efda633930e313ce74f34327e668b2a

### Seed43

Config:

    configs/wsm_mm_pd_dep_v1/fusion/38_r4_ra_stch_optuna_confirm_seed43.yaml

DEV D/P/Mean:

    0.718527 / 0.804363 / 0.7614450000

Checkpoint SHA256:

    6af4a4ed21040aff4db2906b0adc8aa5fd27a4b72c375fbb67f68de04ab8446e

### Seed44

Config:

    configs/wsm_mm_pd_dep_v1/fusion/39_r4_ra_stch_optuna_confirm_seed44.yaml

DEV D/P/Mean:

    0.728217 / 0.832060 / 0.7801390000

Checkpoint SHA256:

    b06fec6912b27cd19d811b72b2c3c08cd1b0a6c826b637f87a314e3976b88c17

Full three-seed means/sample std:

- D mean `0.7352563333`, std `0.0211467766`;
- P mean `0.8385606667`, std `0.0378688091`;
- Mean `0.7869088179`, std `0.0294384420`.

## Frozen pseudo cache

Path:

    /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt

SHA256:

    17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945

Accepted missing D/P:

    376 / 1801

Accepted class counts:

- D positive/negative: `376 / 0`
- P positive/negative: `212 / 1589`

Observed truth continues to override pseudo supervision.

## Exact ablation configs

Create configs40/41/42 by copying configs37/38/39 respectively.

Allowed semantic differences from the corresponding full config are exactly:

1. `experiment_info.params.run_name`;
2. `wsm_pseudo_scale_warmup_callback.params.final_scale: 0.0`.

Required run names:

- seed42: `stage6_no_direct_pseudo_trial012_seed42`
- seed43: `stage6_no_direct_pseudo_trial012_seed43`
- seed44: `stage6_no_direct_pseudo_trial012_seed44`

Keep:

- `loss.params.pseudo_scale: 0.0`;
- observed-only epochs = 3;
- ramp epochs = 5;
- the warm-up callback present;
- every other config field byte/semantic-equivalent to its full reference except run_name/final_scale.

With `final_scale=0.0`, callback `scale_for_epoch` must return exactly zero at every epoch.

Do not remove the pseudo cache or acceptance masks.

## Isolation contract

This is specifically a **direct pseudo-loss** ablation.

The accepted pseudo mask continues to affect task activity exactly as in the current loss implementation.

Pseudo reliability continues to update the RA reliability EMA/controller exactly as in the full method.

Do not describe this variant as “no pseudo information” or “no semantic pseudo path.”

Describe it only as:

    no direct pseudo-supervision loss

## Mandatory pre-run firewall

Before any production run:

1. verify full-reference checkpoint SHAs above;
2. validate configs40/41/42;
3. programmatically prove each new config differs from its corresponding full config only by run_name and warm-up final_scale;
4. prove seeds remain exactly 42/43/44;
5. instantiate DataModule/model/loss/callbacks for all three configs;
6. verify trainable params match the optimized trial-012 model;
7. verify pseudo cache path/SHA/counts/class balance;
8. prove warm-up `scale_for_epoch(e) == 0.0` for every epoch 1..30;
9. run TRAIN-only forward/loss/backward with accepted missing pseudo rows present;
10. verify finite nonzero observed/aux/agreement model gradients;
11. verify the direct pseudo-loss contribution is zero:
    - loss `pseudo_scale == 0.0`;
    - perturb accepted `pseudo_targets` deterministically while keeping accept/reliability/observed fields fixed;
    - total loss and model gradients must remain identical within strict numerical tolerance because the pseudo BCE is multiplied by zero;
12. verify pseudo targets/reliability remain detached;
13. verify RA reliability EMA still updates from accepted reliability entries;
14. verify controller diagnostics finite;
15. no optimizer step;
16. no DEV/Test loader iteration;
17. `git diff --check`;
18. `git diff origin/main -- src` empty;
19. append firewall evidence to PROGRESS;
20. commit and push one firewall commit.

No production run before the firewall commit is visible on origin.

## Exactly three production runs

Run exactly in order:

1. seed42;
2. seed43;
3. seed44.

Commands:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train       --config-path configs/wsm_mm_pd_dep_v1/ablations/40_no_direct_pseudo_seed42.yaml

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train       --config-path configs/wsm_mm_pd_dep_v1/ablations/41_no_direct_pseudo_seed43.yaml

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train       --config-path configs/wsm_mm_pd_dep_v1/ablations/42_no_direct_pseudo_seed44.yaml

No sweep.
No retry for metric improvement.
No extra seed.
No config modification after the firewall.

## Selection/Test firewall

For each run:

- checkpoint selection only by maximum `dev/mean_score`;
- freeze selected checkpoint identity before reading same-epoch Test monitoring;
- record checkpoint SHA256;
- Test cannot affect interpretation or follow-up.

## Three-seed comparison

Compute no-direct-pseudo D/P/Mean by seed and three-seed mean/sample std/range.

Compute same-seed:

    full trial012 - no-direct-pseudo

for D, P, and Mean.

Compute aggregate full-minus-ablation D/P/Mean deltas.

## Frozen claim rule

Record exactly:

    DIRECT PSEUDO-SUPERVISION CONTRIBUTION SUPPORTED

only if ALL conditions hold:

1. full trial012 DEV Mean > no-direct-pseudo DEV Mean on at least 2/3 seeds;
2. full trial012 three-seed Mean > no-direct-pseudo three-seed Mean;
3. full trial012 three-seed D mean is not more than `0.010000` below no-direct-pseudo D mean;
4. full trial012 three-seed P mean is not more than `0.010000` below no-direct-pseudo P mean.

Otherwise record exactly:

    DIRECT PSEUDO-SUPERVISION CONTRIBUTION NOT SUPPORTED

This rule addresses only the direct pseudo BCE term.

Even if supported, do not claim pseudo-label correctness, missing-label recovery, comorbidity recovery, or significance.

## Controller diagnostics

At each DEV-selected epoch record for the no-direct-pseudo runs:

- pseudo scale — MUST be `0.0`;
- alpha D/P;
- progress D/P;
- grad-norm EMA D/P;
- grad-cosine EMA;
- reliability EMA D/P.

Explicitly record that reliability/controller information remains present despite zero direct pseudo loss.

## DEV-only calibration/gate audit

After checkpoint freeze, evaluate DEV only.

For each seed/task report:

- observed count;
- Brier;
- ECE-15;
- task-modality audio gate mean/std.

Report three-seed means.

Compare full trial012 vs no-direct-pseudo using `full - ablation` direction for Brier/ECE and gate means.

Do not recalibrate or threshold-search.
No Test row in post-hoc diagnostics.

## Negative-transfer and pseudo diagnostics

Record:

- accepted pseudo counts/class balance remain frozen;
- no-direct-pseudo vs full three-seed D/P/Mean;
- no-direct-pseudo vs matched audio three-seed contextual means:
  - audio D `0.7303377965`;
  - audio P `0.8162961212`;
  - audio Mean `0.7733169588`.

Contextual audio comparison does not replace the full-vs-ablation claim gate.

## Required PROGRESS evidence

Record:

- task/branch;
- configs40/41/42;
- full-reference checkpoint SHAs;
- config-equivalence proof;
- pseudo cache SHA/counts/classes;
- firewall proof including pseudo-target perturbation invariance;
- firewall commit SHA;
- exact three production commands;
- production count = exactly 3;
- run dirs / MLflow IDs / status / artifact URIs;
- selected epochs/checkpoints/SHA256;
- full DEV D/P UAR/MF1/Score/Mean;
- same-epoch Test monitoring after freeze only;
- three-seed ablation mean/std/range;
- full-minus-ablation per-seed and aggregate deltas;
- exact frozen claim string;
- controller diagnostics with pseudo_scale=0;
- DEV-only calibration/gate audit;
- no Test-driven decision;
- no tuning/rerun;
- no source changes.

## Final scope checks

Run:

    git diff --check
    git diff origin/main -- src
    git status --short
    git diff --stat origin/main...HEAD
    git log -12 --oneline --decorate

Only configs40/41/42 and PROGRESS may differ from origin/main.

## Acceptance criteria

TASK-006D passes only if:

- branch exactly `codex/task-006d` from current manager main;
- no source changes;
- configs differ only by run_name/final_scale;
- direct pseudo scale is exactly zero for every epoch;
- pseudo acceptance/reliability/controller information remains otherwise intact;
- firewall pushed before production;
- exactly three runs seeds42/43/44;
- DEV-only selection;
- Test monitoring only;
- claim rule applied exactly;
- calibration/gate/controller diagnostics recorded;
- no tuning/sweep/retry;
- branch pushed;
- main/master untouched.

Passing TASK-006D closes only Stage-6 direct pseudo-supervision item 3. It authorizes no next task.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch `codex/task-006d`;
- firewall SHA;
- final evidence SHA;
- pushed status;
- main/master untouched;
- source diff empty;
- config equivalence;
- exactly three production runs;
- per-seed no-direct-pseudo D/P/Mean;
- three-seed mean/std;
- full-minus-ablation deltas;
- exact frozen claim string;
- controller diagnostics and pseudo_scale=0;
- calibration/gate summary;
- explicit statement that reliability/controller information was retained;
- no pseudo correctness/comorbidity/significance claim;
- no Test-driven decision;
- Stage 6 active;
- Stage-5 optimization closed;
- no Stage7/Text/Final Test.

For section 6 write only:

    Manager review of TASK-006D; do not start another task.

Stop.
