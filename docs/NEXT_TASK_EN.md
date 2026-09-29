# TASK-003C: Frozen T1 Three-Seed Production Evaluation

## Authority and branch

This task follows **MANAGER-DECISION-075**.

Required branch:

    codex/task-003c

Start from current manager-updated `origin/main`.

This task is production evidence only. The T1 family is frozen.

## Frozen T1 identity

Encoder:

    FacebookAI/xlm-roberta-base

Requested revision:

    e73636d

Resolved commit:

    e73636d4f797dec63c3081bb6ed5c7b0bb3f2089

Model weight SHA256:

    6fd4797bc397c3b8b55d6bb5740366b57e6a3ce91c04c77f22aafc0c128e6feb

Cache root:

    /media/maxim/Programs/Features/WSM/text_t1_xlmr_v1

Cache index SHA256:

    4f276b60e8e423a1eab2d69a02782daa274fb183ae028a525f16df3adcf83ea8

Frozen downstream trainable parameter count:

    447746

TRAIN contract:

- 553 unique video-level transcript units;
- depression 287;
- Parkinson 266;
- no segment replication in TRAIN.

Evaluation contract:

- DEV = 933 canonical segment rows;
- TEST_NONE = 1364;
- TEST_SOFT = 1208;
- TEST_HARD = 1014;
- each row loads its parent video's frozen transcript feature sequence.

## Allowed tracked files

Only:

- `docs/PROGRESS_EN.md`

No source changes.
No config changes.
No cache changes.
No dependency changes.
No new scripts.

If an implementation/config/cache defect is discovered, STOP and report the blocker. Do not patch it inside TASK-003C.

An environment-only interruption before any optimizer step may be repeated with the exact same command only if documented. No metric-driven retry is allowed.

## Mandatory pre-run firewall

Before seed42:

1. verify branch/base;
2. verify working tree clean;
3. verify cache index SHA256 exactly;
4. verify all 754 artifact files referenced by the index exist;
5. verify model/cache identities from TASK-003B;
6. validate all three frozen configs;
7. programmatically prove configs differ only by seed/run_name;
8. instantiate all three configs;
9. verify trainable parameter count = `447746`;
10. verify TRAIN unit counts D287/P266/total553;
11. verify evaluation membership counts 933/1364/1208/1014;
12. one shape-only TRAIN batch and one shape-only DEV batch;
13. no DEV/Test performance computation;
14. `git diff --check`;
15. `git diff origin/main -- src configs scripts pyproject.toml` must be empty;
16. append firewall evidence to PROGRESS;
17. commit and PUSH the firewall.

No production run before the firewall commit is visible on origin.

## Exactly three production runs

Run exactly in this order:

1. seed42;
2. seed43;
3. seed44.

Commands:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/text/00_t1_xlmr_seed42.yaml

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/text/01_t1_xlmr_seed43.yaml

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/text/02_t1_xlmr_seed44.yaml

No sweep.
No hyperparameter change.
No architecture change.
No cache change.
No extra seed.
No retry for metric improvement.

## DEV selection / Test firewall

For each seed:

1. determine the selected checkpoint solely by maximum `dev/mean_score`;
2. record selected epoch, checkpoint path, checkpoint SHA256, MLflow/run identity;
3. freeze that checkpoint identity;
4. only after freeze, read the same-epoch TEST_NONE/SOFT/HARD monitoring metrics.

Test metrics may be reported for traceability but MUST NOT influence:

- checkpoint selection;
- model-family judgment;
- T2 design;
- thresholds;
- follow-up.

## Required primary DEV evidence

For each seed at the selected checkpoint record:

Depression:
- UAR;
- MF1;
- Score.

Parkinson:
- UAR;
- MF1;
- Score.

Aggregate:
- Mean_Score.

Compute across seeds42/43/44:

- D Score mean / sample std / range;
- P Score mean / sample std / range;
- Mean_Score mean / sample std / range.

No significance claim is authorized from three seeds.

## Secondary DEV video-unit audit

Because T1 is trained on one transcript per video while the primary protocol is segment weighted, perform one post-freeze DEV diagnostic for each selected checkpoint:

- deduplicate DEV by canonical `(corpus, video_id)`;
- evaluate each unique video's prediction exactly once;
- use the same observed-task mask and UAR/MF1/Score definitions;
- report D/P/Mean at unique-video level.

Do not use video-level DEV metrics to select checkpoints.

Report segment-level minus unique-video-level differences so transcript broadcast weighting is transparent.

Do not compute a Test video-level diagnostic.

## DEV calibration audit

After checkpoint freeze, DEV only:

For each seed/task report:

- observed sample count;
- Brier score;
- ECE-15.

Primary calibration is segment-level to match the primary protocol.

No threshold fitting/recalibration.

## Contextual frozen comparisons

Compare T1 three-seed DEV means descriptively against:

Matched temporal audio three-seed:
- D `0.7303377965`;
- P `0.8162961212`;
- Mean `0.7733169588`;
- Mean std `0.0138512531`.

Full R4 trial-012 three-seed:
- D `0.7352563333`;
- P `0.8385606667`;
- Mean `0.7869088179`;
- Mean std `0.0294384420`.

Accepted Stage-2 V2 video reference, contextual single-run only:
- D `0.6201013364`;
- P `0.7930427585`;
- Mean `0.7065720475`.

These are not architecture-equivalent comparisons.

Do NOT claim that text “adds” to A+V from T1 alone. T1 is a standalone text-stream baseline.

## TASK-003C conclusion

Do not invent a component-contribution gate.

Record exactly:

    T1 THREE-SEED TEXT BASELINE COMPLETE

if all three frozen runs complete and all required evidence is traceable.

Otherwise record exactly:

    T1 THREE-SEED TEXT BASELINE INCOMPLETE

and name the concrete missing evidence.

The metric level does not determine task completion; execution integrity does.

## Required PROGRESS evidence

Record:

- branch;
- firewall SHA;
- final SHA;
- cache/index identity;
- exactly three production commands/run identities;
- selected epochs/checkpoint paths/SHA256;
- per-seed primary DEV D/P/Mean evidence;
- three-seed mean/std/range;
- unique-video DEV secondary metrics and segment-minus-video differences;
- DEV Brier/ECE-15;
- same-epoch Test monitoring only after checkpoint freeze;
- contextual audio/R4/video comparison;
- exact T1 completion string;
- no config/source/cache changes;
- no tuning/retry;
- Stage 3 active;
- Stage 7 and Final Test locked.

## Final scope checks

Run:

    git diff --check
    git diff origin/main -- src
    git diff origin/main -- configs
    git diff origin/main -- scripts
    git diff origin/main -- pyproject.toml
    git status --short
    git diff --stat origin/main...HEAD
    git log -10 --oneline --decorate

Only PROGRESS may differ.

## Acceptance criteria

TASK-003C passes only if:

- branch exactly `codex/task-003c`;
- frozen cache/config/model identities match TASK-003B;
- no source/config/cache changes;
- firewall pushed before production;
- exactly three completed production runs in seed order 42/43/44;
- DEV/Mean_Score is the sole selector;
- Test monitoring is inspected only after selected checkpoint freeze and is not decision-driving;
- primary segment metrics, secondary unique-video DEV metrics, and DEV calibration are recorded;
- no tuning, sweep, threshold search, or extra seed;
- exact T1 completion string is correct;
- branch pushed;
- main/master untouched.

Passing TASK-003C closes only the T1 standalone baseline. It authorizes no T2 implementation automatically.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Include branch, firewall/final SHA, pushed status, main/master untouched, run count/order, per-seed and aggregate primary DEV metrics, unique-video DEV diagnostics, DEV calibration summary, selected checkpoint identities, Test-monitoring boundary, contextual comparator summary, exact T1 completion string, no tuning/source/config/cache changes, Stage 3 active, Stage 7/Final Test locked.

For section 6 write only:

    Manager review of TASK-003C; do not start another task.

Stop.
