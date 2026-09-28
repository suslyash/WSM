# TASK-006G: No-Video Modality-Removal Ablation on Optimized R4 Trial-012

## Authority and branch

This task follows **MANAGER-DECISION-068**.

Required branch:

    codex/task-006g

Start from current manager-updated `origin/main`.

Create exactly one new task branch from that main.

Stage-5 optimization remains closed. This is a Stage-6 ablation only.

## Scientific question

Does the **video modality** contribute repeatably to the leading optimized R4 trial-012 composition?

Remove video through the model's existing `modality_available` mask, not by deleting parameters or changing architecture.

The no-video variant must retain:

- the same model class and all parameters;
- the same hidden/gate dimensions and dropout;
- the same semantic pseudo cache;
- direct pseudo supervision;
- graded reliability;
- RA controller;
- aux/agreement loss implementation;
- optimizer;
- warm-up;
- all optimized trial-012 hyperparameters.

Only modality availability changes: audio=true, video=false for every sample on every split.

## Required reading

Read in order:

1. `AGENTS.md`
2. `docs/README.md`
3. `docs/PROJECT_REQUIREMENTS.md`
4. `docs/PROGRESS_EN.md`
5. `docs/plan/STAGE_6.md`
6. `docs/NEXT_TASK_EN.md`
7. `src/fusion/data/wsm_av_fusion_datamodule.py`
8. `src/fusion/data/wsm_ramps_semantic_datamodule.py`
9. `src/fusion/models/av_r3_disease_query.py`
10. `src/fusion/loss/r4_ramps_balance_loss.py`
11. configs37/38/39.

## Allowed tracked files

Only:

- `src/fusion/data/wsm_ramps_semantic_datamodule.py`
- `configs/wsm_mm_pd_dep_v1/ablations/49_no_video_seed42.yaml`
- `configs/wsm_mm_pd_dep_v1/ablations/50_no_video_seed43.yaml`
- `configs/wsm_mm_pd_dep_v1/ablations/51_no_video_seed44.yaml`
- `docs/PROGRESS_EN.md`

Forbidden:

- all model/loss/callback changes;
- all `src/audio/*` and `src/video/*` changes;
- all existing config changes;
- pseudo cache changes;
- tuning.

## Minimal DataModule generalization

Extend the semantic DataModule/collate path with one optional fixed availability contract.

Recommended API:

    modality_available_override: null | [bool, bool]

Default `null` MUST preserve existing behavior exactly.

Requirements:

- validate supplied value as exactly two booleans;
- reject `[false,false]`;
- when provided, every collated batch gets that exact availability vector for every row;
- do not alter raw audio/video tensors, masks, pseudo fields, targets, IDs, or split membership;
- all existing full configs with omitted override behave tensor-identically to current `origin/main`.

For TASK-006G configs use exactly:

    modality_available_override: [true, false]

Do not zero or rewrite cached video features in storage. Availability masking is the ablation mechanism.

## Frozen full reference

Full trial-012 DEV:

| Seed | D | P | Mean |
|---:|---:|---:|---:|
| 42 | 0.759025 | 0.879259 | 0.8191424538 |
| 43 | 0.718527 | 0.804363 | 0.7614450000 |
| 44 | 0.728217 | 0.832060 | 0.7801390000 |

Three-seed full:

- D mean/std `0.7352563333/0.0211467766`;
- P mean/std `0.8385606667/0.0378688091`;
- Mean `0.7869088179/0.0294384420`.

Full checkpoint SHA256:

- seed42 `104b79e409832839502b4d20a98c054f5efda633930e313ce74f34327e668b2a`;
- seed43 `6af4a4ed21040aff4db2906b0adc8aa5fd27a4b72c375fbb67f68de04ab8446e`;
- seed44 `b06fec6912b27cd19d811b72b2c3c08cd1b0a6c826b637f87a314e3976b88c17`.

Verify these before production.

## Exact ablation configs

Create configs49/50/51 from full refs37/38/39 respectively.

Allowed semantic differences only:

1. run_name;
2. `data.params.modality_available_override: [true, false]`.

Required run names:

- seed42: `stage6_no_video_trial012_seed42`
- seed43: `stage6_no_video_trial012_seed43`
- seed44: `stage6_no_video_trial012_seed44`

Everything else must match the corresponding full config.

## Mandatory pre-run firewall

Before any production run:

1. implement the minimal availability override;
2. prove full-config backward compatibility with override omitted:
   - canonical TRAIN batch tensor/mask/meta fields identical to clean `origin/main`;
   - full `modality_available` remains all true;
3. validate configs49/50/51;
4. prove config equivalence except run_name/availability override;
5. verify frozen full checkpoint SHAs;
6. instantiate DataModule/model/loss/callbacks for each ablation config;
7. verify trainable parameter count remains `295239`;
8. verify pseudo cache path/SHA/counts/classes remain full frozen values;
9. on TRAIN batch verify `modality_available == [true,false]` for every row;
10. forward invariants:
    - audio modality weight exactly 1;
    - video modality weight exactly 0;
    - `features_video` exactly zero;
    - `video_aux_valid` all false;
    - `audio_aux_valid` all true;
11. deterministic isolation test:
    - perturb the raw video tensor strongly while availability remains false;
    - main logits, total loss, and non-video parameter gradients remain identical within strict numerical tolerance;
12. verify video-only parameter gradients are zero/None as expected;
13. verify observed and accepted pseudo supervision still produce finite nonzero gradients through the active path at pseudo_scale=1;
14. pseudo target/reliability detached;
15. controller diagnostics finite;
16. no optimizer step;
17. no DEV/Test iteration;
18. `git diff --check`;
19. forbidden source scopes unchanged;
20. append firewall evidence to PROGRESS;
21. commit and push one firewall commit.

No production run before firewall is visible on origin.

## Exactly three production runs

Run exactly in order:

1. seed42;
2. seed43;
3. seed44.

Commands:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/ablations/49_no_video_seed42.yaml

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/ablations/50_no_video_seed43.yaml

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/ablations/51_no_video_seed44.yaml

No sweep, retry for metric improvement, extra seed, or post-firewall change.

## Selection/Test firewall

For each run:

- select by maximum `dev/mean_score` only;
- freeze selected checkpoint before reading same-epoch Test monitoring;
- Test cannot influence any claim or next-step decision.

## Three-seed comparison

Compute no-video D/P/Mean by seed and three-seed mean/sample std/range.

Compute same-seed and aggregate:

    full trial012 - no-video

for D/P/Mean.

Also report contextual comparison of no-video against the frozen matched temporal-audio reference, but do not treat those as architecture-equivalent models.

## Frozen claim rule

Record exactly:

    VIDEO MODALITY CONTRIBUTION SUPPORTED

only if ALL hold:

1. full DEV Mean > no-video DEV Mean on at least 2/3 seeds;
2. full three-seed Mean > no-video three-seed Mean;
3. full three-seed D is not more than `0.010000` below no-video D;
4. full three-seed P is not more than `0.010000` below no-video P.

Otherwise record exactly:

    VIDEO MODALITY CONTRIBUTION NOT SUPPORTED

No significance or causal disease-content claim is authorized.

## Diagnostics

At selected epoch record:

- pseudo scale;
- alpha D/P;
- progress D/P;
- grad-norm EMA D/P;
- grad-cosine EMA;
- reliability EMA D/P;
- task modality weights — must be audio=1/video=0.

DEV-only post-freeze audit:

- observed counts;
- Brier;
- ECE-15;
- audio gate mean/std (expected exactly 1/0 std under forced availability);
- full-minus-ablation calibration deltas.

No recalibration/threshold search.
No Test rows.

## Required PROGRESS evidence

Record:

- task/branch;
- DataModule change and backward-compatibility proof;
- configs49/50/51 equivalence;
- full checkpoint SHAs;
- pseudo-cache identity;
- firewall evidence/SHA;
- exact three production commands;
- exactly three run/MLflow identities;
- selected epochs/checkpoint SHA;
- DEV D/P UAR/MF1/Score/Mean;
- same-epoch Test monitoring after freeze only;
- three-seed no-video aggregate;
- full-minus-no-video deltas;
- exact frozen claim string;
- controller/gate/calibration diagnostics;
- contextual frozen-audio comparison;
- no tuning/retry/Test-driven decision.

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

Only the authorized semantic DataModule source file, configs49/50/51, and PROGRESS may differ.

## Acceptance criteria

TASK-006G passes only if:

- branch exactly `codex/task-006g`;
- full default DataModule behavior is unchanged;
- no-video availability is exactly [true,false];
- architecture/parameter count unchanged;
- pseudo path unchanged;
- video perturbation has no effect under unavailable mask;
- firewall pushed before production;
- exactly three runs seeds42/43/44;
- DEV-only selection;
- Test monitoring only;
- frozen claim rule applied exactly;
- diagnostics recorded;
- no tuning/sweep/retry;
- branch pushed;
- main/master untouched.

Passing TASK-006G closes only the video-removal half of Stage-6 modality-removal item 7. It authorizes no next task.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Include branch, firewall/final SHA, pushed status, main/master untouched, backward compatibility, parameter count, exactly three runs, per-seed/aggregate metrics, full-minus-ablation deltas, exact claim string, availability/gate/controller/calibration summary, no Test-driven decision, Stage 6 active, Stage-5 closed, Stage7/Text/Final Test locked.

For section 6 write only:

    Manager review of TASK-006G; do not start another task.

Stop.
