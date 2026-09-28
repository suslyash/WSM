# TASK-006F: Semantic-Evidence Contribution via Depression Pseudo Acceptance

## Authority and branch

This task follows **MANAGER-DECISION-067**.

Required branch:

    codex/task-006f

Start from current manager-updated `origin/main`.

Create exactly one new task branch from that main.

This is a Stage-6 component ablation. Stage-5 optimization remains closed.

## Scientific question

Does the **semantic-evidence-enabled depression pseudo pathway** contribute repeatably to the leading optimized R4 trial-012 composition?

Historical frozen evidence defines the ablation:

- before semantic evidence, the audio+video reliability audit had **no deployable depression side** at the unchanged precision>=0.90/support>=10 gate;
- the Parkinson audio+video rules were already valid and frozen;
- semantic evidence was introduced only to recover the blocked depression side;
- the final semantic cache has accepted D/P counts `376/1801`.

Therefore the no-semantic comparator must:

- remove all 376 depression accepted pseudo entries;
- preserve all 1801 Parkinson accepted pseudo entries exactly;
- preserve every observed label and canonical row;
- preserve the full optimized R4 architecture/loss/optimizer/controller/warm-up.

This task tests the contribution of the **semantic-enabled depression pseudo acceptance path**. It does not prove semantic-label correctness or causal disease semantics.

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
9. `scripts/common/prepare_ramps_r2_av_targets.py`
10. `scripts/common/prepare_ramps_r2_semantic_targets.py`
11. configs37/38/39.

You MAY read the Stage-5 historical archive only to verify the frozen facts that pre-semantic depression was blocked and Parkinson rules were deployable.

## Allowed tracked files

Codex may add/modify only:

- `src/fusion/data/wsm_ramps_semantic_datamodule.py`
- `scripts/common/build_ramps_no_semantic_depression_ablation.py`
- `configs/wsm_mm_pd_dep_v1/ablations/46_no_semantic_depression_seed42.yaml`
- `configs/wsm_mm_pd_dep_v1/ablations/47_no_semantic_depression_seed43.yaml`
- `configs/wsm_mm_pd_dep_v1/ablations/48_no_semantic_depression_seed44.yaml`
- `docs/PROGRESS_EN.md`

Forbidden:
- all model/loss/callback source changes;
- all `src/audio/*` and `src/video/*` changes;
- changes to existing configs;
- changes to semantic rule/search code.

## Frozen source cache

Path:

    /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt

SHA256:

    17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945

Full accepted counts:

- D: `376`, classes `376/0`;
- P: `1801`, classes `212/1589`.

The source cache is immutable.

## Minimal DataModule generalization

Modify `WSMRampsSemanticDataModule` only to allow explicit expected pseudo-count contracts for ablation caches.

Add optional constructor parameters with frozen defaults:

    expected_accepted_counts = (376, 1801)
    expected_positive_counts = (376, 212)
    expected_negative_counts = (0, 1589)

Requirements:

- omitted parameters preserve current behavior exactly;
- validate each supplied contract as exactly two non-negative integers;
- keep expected missing counts fixed at `2665/3660`;
- use the resolved expected values in count/class validation and audit/context reporting;
- do not relax any other cache invariant;
- do not change version/teacher/CLIP/prompt-bank/observed-truth validation;
- do not change dataset/collate semantics.

Mandatory backward-compatibility proof:
- instantiate full configs37/38/39 with no new parameters;
- verify audit accepted/class counts remain exactly the frozen full values;
- verify a canonical TRAIN batch is tensor-identical before vs after the source edit under the full cache, using a clean `origin/main` implementation comparison or an equivalent isolated reference import;
- trainable model parameters and loss behavior must be unchanged.

This source change is infrastructure-only for the ablation and must not alter full-method semantics.

## Derived no-semantic-depression cache

Create exactly:

    /media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1_semantic_ablation/no_semantic_depression_v1.pt

Builder:

    scripts/common/build_ramps_no_semantic_depression_ablation.py

Builder MUST fail rather than overwrite an existing output.

Required transformation:

- deep-copy the full source cache;
- depression task index 0:
  - set every `pseudo_accept_mask[:,0]` to False;
  - set every `pseudo_targets[:,0]` to NaN;
  - set every `pseudo_reliability[:,0]` to 0.0;
  - set every `pseudo_class[:,0]` to -1;
- Parkinson task index 1:
  - preserve `pseudo_accept_mask[:,1]` exactly;
  - preserve `pseudo_targets[:,1]` exactly including NaNs;
  - preserve `pseudo_reliability[:,1]` exactly;
  - preserve `pseudo_class[:,1]` exactly;
- preserve `calibrated_audio_probs`, raw teacher outputs/features, observed truth, semantic fields, teacher identities, prompt-bank identity, and every other source field exactly.

You MAY add one top-level metadata mapping:

    semantic_ablation

with:
- type = `remove_semantic_enabled_depression_pseudo_path`;
- source_cache_sha256;
- removed_depression_accepted_count = 376;
- retained_parkinson_accepted_count = 1801;
- historical_basis = `pre_semantic_depression_not_deployable_parkinson_frozen`.

Do not change cache version.

Semantic metadata may remain in the cache for identity/audit compatibility; it must have no training effect because all depression pseudo fields are neutralized.

## Derived cache invariants

Before save and after reload verify:

- source SHA before/after equals frozen SHA;
- rows/segment IDs exactly unchanged;
- observed mask/targets exactly unchanged including NaNs;
- calibrated audio probabilities exactly unchanged;
- all non-pseudo fields exactly unchanged except allowed `semantic_ablation` metadata;
- depression accepted count = 0;
- depression pseudo targets all NaN;
- depression reliability all 0;
- depression pseudo classes all -1;
- Parkinson accepted count = 1801;
- Parkinson accept mask/targets/reliability/classes are tensor-identical to source;
- Parkinson classes = `212/1589`;
- no observed pseudo acceptance;
- accepted P targets equal calibrated audio probabilities;
- DataModule loads the derived cache when supplied expected contracts:
  - accepted `[0,1801]`;
  - positive `[0,212]`;
  - negative `[0,1589]`.

Record derived cache SHA256.

## Frozen full reference

Full optimized trial-012:

| Seed | D | P | Mean |
|---:|---:|---:|---:|
| 42 | 0.759025 | 0.879259 | 0.8191424538 |
| 43 | 0.718527 | 0.804363 | 0.7614450000 |
| 44 | 0.728217 | 0.832060 | 0.7801390000 |

Full three-seed:

- D mean/std `0.7352563333/0.0211467766`;
- P mean/std `0.8385606667/0.0378688091`;
- Mean `0.7869088179/0.0294384420`.

Checkpoint SHA256:

- seed42 `104b79e409832839502b4d20a98c054f5efda633930e313ce74f34327e668b2a`;
- seed43 `6af4a4ed21040aff4db2906b0adc8aa5fd27a4b72c375fbb67f68de04ab8446e`;
- seed44 `b06fec6912b27cd19d811b72b2c3c08cd1b0a6c826b637f87a314e3976b88c17`.

Verify checkpoint paths/SHA before production.

## Exact ablation configs

Create configs46/47/48 from configs37/38/39 respectively.

Allowed semantic differences only:

1. `experiment_info.params.run_name`;
2. `data.params.pseudo_cache_path`;
3. the three explicit expected-count parameters required for the ablation cache.

Required data params:

    expected_accepted_counts: [0, 1801]
    expected_positive_counts: [0, 212]
    expected_negative_counts: [0, 1589]

Required run names:

- seed42 `stage6_no_semantic_depression_trial012_seed42`
- seed43 `stage6_no_semantic_depression_trial012_seed43`
- seed44 `stage6_no_semantic_depression_trial012_seed44`

Everything else must be identical to full refs37/38/39.

## Isolation contract

This task removes only the training contribution that exists because semantic evidence recovered a depression pseudo subset.

Parkinson pseudo supervision remains exactly the source semantic-cache Parkinson path.

Direct pseudo supervision, graded reliability, RA controller, architecture, aux/agreement losses, optimizer, and warm-up remain unchanged.

Do not describe this as:
- no pseudo supervision;
- no semantic metadata;
- audio-only;
- no reliability.

Describe it as:

    no semantic-enabled depression pseudo path

## Mandatory pre-run firewall

Before production:

1. implement and audit the minimal DataModule generalization;
2. prove full-cache backward compatibility as above;
3. build/audit derived cache;
4. verify source/derived cache SHA;
5. validate configs46/47/48;
6. prove config equivalence except the four allowed difference categories;
7. verify full checkpoint SHA values;
8. instantiate DataModule/model/loss/callbacks for all ablation configs;
9. verify trainable params match full trial-012 (`295239`);
10. verify D accepted count 0 and P accepted count 1801;
11. verify P pseudo fields exactly match source;
12. run TRAIN-only forward/loss/backward using batches that include accepted Parkinson pseudo rows;
13. verify finite nonzero observed gradients and finite nonzero Parkinson pseudo-supervision gradients once pseudo_scale=1;
14. verify there is no depression accepted-pseudo gradient contribution;
15. verify pseudo targets/reliability detached;
16. verify controller diagnostics finite; record the D reliability-EMA state/fallback semantics and P reliability EMA;
17. no optimizer step;
18. no DEV/Test loader iteration;
19. `git diff --check`;
20. verify no diffs under `src/audio`, `src/video`, model/loss/callback source;
21. append firewall evidence to PROGRESS;
22. commit and push one firewall commit.

No production run before firewall is visible on origin.

## Exactly three production runs

Run exactly:

1. seed42;
2. seed43;
3. seed44.

Commands:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train       --config-path configs/wsm_mm_pd_dep_v1/ablations/46_no_semantic_depression_seed42.yaml

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train       --config-path configs/wsm_mm_pd_dep_v1/ablations/47_no_semantic_depression_seed43.yaml

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train       --config-path configs/wsm_mm_pd_dep_v1/ablations/48_no_semantic_depression_seed44.yaml

No sweep, retry for metric improvement, extra seed, or post-firewall config/cache change.

## Selection/Test firewall

For every run:

- select checkpoint only by max `dev/mean_score`;
- freeze checkpoint identity before same-epoch Test monitoring;
- Test cannot influence interpretation or follow-up.

## Three-seed comparison

Compute ablation D/P/Mean by seed and three-seed mean/sample std/range.

Compute same-seed:

    full trial012 - no-semantic-depression

for D/P/Mean.

Compute aggregate full-minus-ablation D/P/Mean.

## Frozen claim rule

Record exactly:

    SEMANTIC-EVIDENCE CONTRIBUTION VIA DEPRESSION PSEUDO ACCEPTANCE SUPPORTED

only if ALL hold:

1. full DEV Mean > ablation DEV Mean on at least 2/3 seeds;
2. full DEV D > ablation DEV D on at least 2/3 seeds;
3. full three-seed Mean > ablation three-seed Mean;
4. full three-seed D > ablation three-seed D;
5. full three-seed P is not more than `0.010000` below ablation P.

Otherwise record exactly:

    SEMANTIC-EVIDENCE CONTRIBUTION VIA DEPRESSION PSEUDO ACCEPTANCE NOT SUPPORTED

This claim is deliberately narrow.

Even if supported, do not claim:
- semantic evidence is clinically correct;
- pseudo depression labels are correct;
- missing-label/comorbidity recovery;
- statistical significance;
- final promotion.

## Controller / coverage diagnostics

At selected epoch record:

- pseudo scale;
- alpha D/P;
- progress D/P;
- grad-norm EMA D/P;
- grad-cosine EMA;
- reliability EMA D/P or uninitialized state;
- P accepted coverage/classes;
- D accepted count = 0.

Explain the D reliability fallback used by the current RA loss when its EMA has not been updated.

## DEV-only calibration/gate audit

After checkpoint freeze, use DEV only.

For each seed/task:
- observed count;
- Brier;
- ECE-15;
- audio gate mean/std.

Report three-seed means and full-minus-ablation deltas.

No recalibration or threshold search.
No Test rows.

## Required PROGRESS evidence

Record:
- task/branch;
- exact historical basis for D=0/P=frozen comparator;
- DataModule source change and backward-compatibility proof;
- builder path;
- source/derived cache path/SHA;
- cache invariants;
- configs46/47/48 and equivalence proof;
- full checkpoint SHAs;
- firewall SHA;
- exact three commands and production count=3;
- run/MLflow identities;
- selected epochs/checkpoint SHA;
- DEV D/P UAR/MF1/Score/Mean;
- Test monitoring after freeze only;
- three-seed aggregate;
- full-minus-ablation deltas;
- exact claim string;
- controller/coverage diagnostics;
- DEV calibration/gate audit;
- no correctness/significance/promotion claim;
- no tuning/rerun.

## Final scope checks

Run:

    git diff --check
    git diff origin/main -- src/audio
    git diff origin/main -- src/video
    git diff origin/main -- src/fusion/models
    git diff origin/main -- src/fusion/loss
    git diff origin/main -- src/common/callbacks
    git status --short
    git diff --stat origin/main...HEAD
    git log -12 --oneline --decorate

Only the authorized DataModule source file, builder, configs46/47/48, and PROGRESS may differ.

## Acceptance criteria

TASK-006F passes only if:
- branch exactly `codex/task-006f`;
- DataModule default full behavior is proven unchanged;
- derived cache removes exactly 376 D pseudo entries and preserves P exactly;
- source cache remains unchanged;
- configs differ only as authorized;
- firewall pushed before production;
- exactly three runs seeds42/43/44;
- DEV-only selection;
- Test monitoring only;
- claim gate applied exactly;
- controller/coverage/calibration/gate diagnostics recorded;
- no tuning/sweep/retry;
- branch pushed;
- main/master untouched.

Passing TASK-006F closes only the Stage-6 semantic-evidence item. It authorizes no next task.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Include:
- branch;
- firewall/final SHA;
- pushed status;
- main/master untouched;
- DataModule backward compatibility;
- source/derived cache SHA;
- D removed count and exact P preservation;
- exactly three runs;
- per-seed D/P/Mean;
- aggregate mean/std;
- full-minus-ablation deltas;
- exact frozen claim;
- controller/coverage summary;
- calibration/gate summary;
- no Test-driven decision;
- no correctness/comorbidity/significance/promotion claim;
- Stage 6 active;
- Stage-5 closed;
- Stage7/Text/Final Test locked.

For section 6 write only:

    Manager review of TASK-006F; do not start another task.

Stop.
