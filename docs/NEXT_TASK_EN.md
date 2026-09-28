# TASK-006H: No-Audio Online-Input Modality-Removal Ablation on Optimized R4 Trial-012

## Authority and branch

This task follows **MANAGER-DECISION-069**.

Required branch:

    codex/task-006h

Start from current manager-updated `origin/main`.

Create exactly one new task branch from that main.

Stage-5 optimization remains closed. This is a Stage-6 ablation only.

## Scientific question

Does the **online audio input branch** contribute repeatably to the leading optimized R4 trial-012 composition when all other training semantics are held fixed?

Remove audio through the existing `modality_available` mask.

Do not delete audio parameters or change architecture.

The no-audio variant must retain:

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

Only online modality availability changes:

    audio = false
    video = true

## Critical interpretation boundary

The frozen pseudo cache was constructed from historical audio/video teacher evidence and stores calibrated audio probabilities as pseudo targets.

Therefore TASK-006H does NOT remove every audio-derived signal from the whole training pipeline.

It removes the **online audio feature branch** from the student while leaving the frozen pseudo-training path unchanged.

Any accepted claim must be phrased only as an online-audio-input contribution under the current pipeline.

Do not describe the ablation as:

- no audio information;
- video-only in a fully information-pure sense;
- removal of audio-derived pseudo supervision.

## Required reading

Read in order:

1. `AGENTS.md`
2. `docs/README.md`
3. `docs/PROJECT_REQUIREMENTS.md`
4. `docs/PROGRESS_EN.md`
5. `docs/plan/STAGE_6.md`
6. `docs/NEXT_TASK_EN.md`
7. `src/fusion/data/wsm_ramps_semantic_datamodule.py`
8. `src/fusion/models/av_r3_disease_query.py`
9. `src/fusion/loss/r4_ramps_balance_loss.py`
10. configs37/38/39;
11. configs49/50/51 only to verify the already-merged availability mechanism, not as performance comparators.

## Allowed tracked files

Only:

- `configs/wsm_mm_pd_dep_v1/ablations/52_no_audio_seed42.yaml`
- `configs/wsm_mm_pd_dep_v1/ablations/53_no_audio_seed43.yaml`
- `configs/wsm_mm_pd_dep_v1/ablations/54_no_audio_seed44.yaml`
- `docs/PROGRESS_EN.md`

No source change is authorized.

Forbidden:

- all `src/*` changes;
- all existing config changes;
- pseudo-cache changes;
- tuning.

## Frozen availability implementation

Reuse the already-merged semantic DataModule parameter:

    modality_available_override

TASK-006H must use exactly:

    modality_available_override: [false, true]

Do not edit its implementation.

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

Create configs52/53/54 by copying refs37/38/39 respectively.

Allowed semantic differences only:

1. run_name;
2. `data.params.modality_available_override: [false, true]`.

Required run names:

- seed42: `stage6_no_audio_trial012_seed42`
- seed43: `stage6_no_audio_trial012_seed43`
- seed44: `stage6_no_audio_trial012_seed44`

Everything else must match the corresponding full config exactly.

## Mandatory pre-run firewall

Before any production run:

1. validate configs52/53/54;
2. programmatically prove each config differs from its full ref only by run_name and availability override;
3. verify seeds remain 42/43/44;
4. verify full checkpoint paths/SHA values;
5. instantiate DataModule/model/loss/callbacks for each config;
6. verify trainable parameter count remains `295239`;
7. verify pseudo cache path/SHA/counts/classes remain full frozen values;
8. on TRAIN batch verify `modality_available == [false,true]` for every row;
9. forward invariants:
   - audio modality weight exactly 0;
   - video modality weight exactly 1;
   - `features_audio` exactly zero;
   - `audio_aux_valid` all false;
   - `video_aux_valid` all true;
10. deterministic isolation test:
    - strongly perturb `audio_cls` and the collated raw/temporal audio tensor while availability remains false;
    - main logits, total loss, and non-audio-specific parameter gradients remain identical within strict numerical tolerance;
11. verify audio-specific parameter gradients are zero/None:
    - audio projection;
    - audio auxiliary heads;
12. verify active video path has finite nonzero gradients;
13. verify observed supervision and accepted pseudo supervision remain finite/nonzero at pseudo_scale=1;
14. verify pseudo targets/reliability detached;
15. verify controller diagnostics finite;
16. no optimizer step;
17. no DEV/Test loader iteration;
18. `git diff --check`;
19. `git diff origin/main -- src` must be empty;
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
      --config-path configs/wsm_mm_pd_dep_v1/ablations/52_no_audio_seed42.yaml

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/ablations/53_no_audio_seed43.yaml

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/ablations/54_no_audio_seed44.yaml

No sweep, retry for metric improvement, extra seed, or post-firewall change.

## Selection/Test firewall

For every run:

- checkpoint selection only by maximum `dev/mean_score`;
- freeze checkpoint before reading same-epoch Test monitoring;
- Test cannot affect the claim or next-step decision.

## Three-seed comparison

Compute no-audio D/P/Mean by seed and three-seed mean/sample std/range.

Compute same-seed and aggregate:

    full trial012 - no-audio

for D/P/Mean.

Contextually compare no-audio against the frozen Stage-2 V2 video reference:

    D 0.6201013364
    P 0.7930427585
    Mean 0.7065720475

This contextual comparison is not architecture-equivalent and does not replace the full-vs-ablation gate.

## Frozen claim rule

Record exactly:

    ONLINE AUDIO INPUT MODALITY CONTRIBUTION SUPPORTED

only if ALL hold:

1. full DEV Mean > no-audio DEV Mean on at least 2/3 seeds;
2. full three-seed Mean > no-audio three-seed Mean;
3. full three-seed D is not more than `0.010000` below no-audio D;
4. full three-seed P is not more than `0.010000` below no-audio P.

Otherwise record exactly:

    ONLINE AUDIO INPUT MODALITY CONTRIBUTION NOT SUPPORTED

This claim concerns only the online student audio branch under the frozen pseudo-training pipeline.

No claim of total audio-information removal is authorized.

## Diagnostics

At selected epoch record:

- pseudo scale;
- alpha D/P;
- progress D/P;
- grad-norm EMA D/P;
- grad-cosine EMA;
- reliability EMA D/P;
- task modality weights — must be audio=0/video=1.

DEV-only post-freeze audit:

- observed counts;
- Brier;
- ECE-15;
- audio gate mean/std — expected exactly `0.0/0.0`;
- video gate mean/std — expected exactly `1.0/0.0`;
- full-minus-ablation calibration deltas.

No recalibration or threshold search.
No Test rows.

## Required PROGRESS evidence

Record:

- task/branch;
- configs52/53/54 and exact equivalence proof;
- availability implementation reused unchanged;
- full checkpoint SHAs;
- pseudo-cache identity and the explicit audio-derived-pseudo interpretation caveat;
- firewall evidence/SHA;
- exact three production commands;
- exactly three run/MLflow identities;
- selected epochs/checkpoint SHA;
- DEV D/P UAR/MF1/Score/Mean;
- Test monitoring after freeze only;
- three-seed no-audio aggregate;
- full-minus-no-audio deltas;
- exact frozen claim string;
- controller/gate/calibration diagnostics;
- contextual Stage-2 video comparison;
- no tuning/retry/Test-driven decision;
- no source changes.

## Final scope checks

Run:

    git diff --check
    git diff origin/main -- src
    git status --short
    git diff --stat origin/main...HEAD
    git log -12 --oneline --decorate

Only configs52/53/54 and PROGRESS may differ.

## Acceptance criteria

TASK-006H passes only if:

- branch exactly `codex/task-006h`;
- no source changes;
- configs differ only run_name/availability override;
- availability exactly `[false,true]`;
- architecture/parameter count unchanged;
- pseudo path unchanged;
- audio perturbation has no effect under unavailable mask;
- firewall pushed before production;
- exactly three runs seeds42/43/44;
- DEV-only selection;
- Test monitoring only;
- frozen claim applied exactly;
- interpretation is explicitly limited to online audio input;
- diagnostics recorded;
- no tuning/sweep/retry;
- branch pushed;
- main/master untouched.

Passing TASK-006H closes the audio-removal half, and therefore completes Stage-6 modality-removal item 7. It authorizes no next task.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Include branch, firewall/final SHA, pushed status, main/master untouched, source diff empty, parameter count, pseudo-path caveat, exactly three runs, per-seed/aggregate metrics, full-minus-ablation deltas, exact claim string, availability/gate/controller/calibration summary, no Test-driven decision, Stage 6 active, Stage-5 closed, Stage7/Text/Final Test locked.

For section 6 write only:

    Manager review of TASK-006H; do not start another task.

Stop.
