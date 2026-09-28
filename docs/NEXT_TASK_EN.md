# TASK-006C: Isolated Three-Seed RA-STCH Balancing Contribution Audit

## Authority and branch

This task follows **MANAGER-DECISION-064**.

Required branch:

    codex/task-006c

Start from current manager-updated `origin/main`.

Create exactly one new task branch from that main.

This is a Stage-6 ablation/claims audit, not Stage-5 optimization.

## Goal

Complete the original **fixed-composition RA-STCH** three-seed evidence by running only the two frozen, previously unrun RA-STCH configs for seeds43 and44.

Then compare RA-STCH against the same-seed frozen controls:

- Equal weighting;
- Static-STCH;
- corrected Progress.

The purpose is to isolate the balancing/controller contribution with architecture, pseudo supervision, optimizer, warm-up, and all non-balancing factors held fixed.

Do **not** use the optimized trial-012 R4 configs37/38/39 as the RA ablation result. Those differ in model/loss/optimizer hyperparameters and are not an isolated balancing comparison.

## Required reading

Read in this exact order:

1. `AGENTS.md`
2. `docs/README.md`
3. `docs/PROJECT_REQUIREMENTS.md`
4. `docs/PROGRESS_EN.md`
5. `docs/plan/STAGE_6.md`
6. `docs/NEXT_TASK_EN.md`
7. `src/fusion/models/av_r3_disease_query.py`
8. `src/fusion/loss/r4_ramps_balance_loss.py`
9. `src/common/callbacks/wsm_r4_balance_callback.py`
10. `src/fusion/data/wsm_ramps_semantic_datamodule.py`
11. configs `13_r4_equal_seed42.yaml` through `24_r4_progress_seed44.yaml`

The Stage-5 historical archive may be read only to verify the retained selected checkpoints/results listed below.

## No implementation changes

This task is experiment/evidence only.

Allowed tracked modification:

- `docs/PROGRESS_EN.md`

Forbidden tracked modifications:

- every `src/*` file;
- every config;
- every script;
- `src/audio`, `src/video`, common training code, plugin registration.

If an existing frozen config or source implementation cannot run as-is, STOP and report the blocker. Do not patch it inside TASK-006C.

## Frozen RA-STCH composition

Use configs exactly as committed:

- seed42: `configs/wsm_mm_pd_dep_v1/fusion/16_r4_ra_stch_seed42.yaml` — retained, DO NOT rerun;
- seed43: `configs/wsm_mm_pd_dep_v1/fusion/17_r4_ra_stch_seed43.yaml`;
- seed44: `configs/wsm_mm_pd_dep_v1/fusion/18_r4_ra_stch_seed44.yaml`.

All three must be semantically identical except top-level `seed` and `run_name`.

Frozen composition:

- model `wsm_av_r3_disease_query_model`;
- audio/video dims `768/512`;
- hidden/gate hidden `192/192`;
- dropout `0.2`;
- trainable params `403079`;
- semantic pseudo cache:
  `/media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt`;
- pseudo cache SHA256:
  `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`;
- loss `wsm_r4_ramps_balance_loss`, mode `ra_stch`;
- aux weight `0.25`;
- agreement weight `0.10`;
- tau `0.1`;
- progress temperature `0.25`;
- progress references D/P `0.697035/0.852104`;
- controller EMA `0.8`;
- grad EMA `0.9`;
- reliability EMA `0.9`;
- weight min/max `0.2/0.8`;
- AdamW lr `1e-4`, weight decay `0.01`;
- pseudo warm-up: 3 observed-only epochs, 5 ramp epochs, final scale 1.0.

## Retained valid RA seed42

Do not rerun seed42.

Run:

    logs/wsm_mm_pd_dep_v1/r4_ra_stch_seed42_c1_2026-09-25_16-51_wsm_av_r3_disease_query_model_c350ca36

Selected epoch:

    13

Checkpoint:

    logs/wsm_mm_pd_dep_v1/r4_ra_stch_seed42_c1_2026-09-25_16-51_wsm_av_r3_disease_query_model_c350ca36/checkpoints/epoch=13_dev_mean_score=0.7826.pt

Checkpoint SHA256:

    53397e3bbcbfc95abd30bdf63fec018a28f0effc2d92d66231061fb2d051254a

DEV:

- D `0.700992`
- P `0.864298`
- Mean `0.782645`

Selected-epoch controller diagnostics:

- alpha D/P `0.269192/0.730808` in the accepted Stage-5 record;
- final logged controller trajectory alpha `0.256828/0.743172`;
- progress `0.020070/-0.151804`;
- grad-norm EMA D/P `0.248995/0.121096`;
- grad-cosine EMA `0.009814`;
- reliability EMA D/P `0.671974/0.773563`.

Verify the retained checkpoint file and SHA before production.

## Frozen three-seed controls

### Equal

| Seed | D | P | Mean |
|---:|---:|---:|---:|
| 42 | 0.712498 | 0.843342 | 0.777920 |
| 43 | 0.703299 | 0.856043 | 0.779671 |
| 44 | 0.707977 | 0.851878 | 0.779927 |

Three-seed means:

- D `0.707925`
- P `0.850421`
- Mean `0.7791726667`

### Static-STCH

| Seed | D | P | Mean |
|---:|---:|---:|---:|
| 42 | 0.725077 | 0.849485 | 0.787281 |
| 43 | 0.698945 | 0.853184 | 0.776064 |
| 44 | 0.702072 | 0.853622 | 0.777847 |

Three-seed means:

- D `0.708698`
- P `0.852097`
- Mean `0.7803973333`

### Corrected Progress

| Seed | D | P | Mean |
|---:|---:|---:|---:|
| 42 | 0.701245 | 0.873354 | 0.787299 |
| 43 | 0.721597 | 0.863131 | 0.792364 |
| 44 | 0.701971 | 0.847208 | 0.774590 |

Three-seed means:

- D `0.708271`
- P `0.861231`
- Mean `0.7847510000`

## Mandatory pre-run firewall

Before either production run:

1. validate configs16/17/18;
2. programmatically prove 17 and18 are semantically identical to16 except `seed` and `run_name`;
3. prove seeds are exactly 42/43/44;
4. verify retained seed42 checkpoint path/SHA;
5. instantiate DataModule/model/loss for configs17/18;
6. verify trainable params `403079`;
7. verify semantic cache path/SHA and accepted counts D/P `376/1801`;
8. run one TRAIN-only forward/loss/backward smoke for each of configs17/18;
9. verify finite loss and non-zero finite gradients for main model under observed+pseudo supervision;
10. verify RA controller diagnostics are finite;
11. verify pseudo target/reliability are detached;
12. no optimizer step;
13. no DEV/Test loader iteration;
14. `git diff --check`;
15. `git diff origin/main -- src` is empty;
16. `git diff origin/main -- configs` is empty;
17. append firewall evidence to PROGRESS;
18. commit and push one firewall commit.

No production run before the firewall commit exists on origin.

## Exactly two production runs

Run exactly in this order:

1. RA-STCH seed43;
2. RA-STCH seed44.

Commands:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train       --config-path configs/wsm_mm_pd_dep_v1/fusion/17_r4_ra_stch_seed43.yaml

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train       --config-path configs/wsm_mm_pd_dep_v1/fusion/18_r4_ra_stch_seed44.yaml

No seed42 rerun.
No retry for metric improvement.
No sweep.
No config change between runs.
No optimized trial-012 run.

## Selection/Test firewall

For each new run:

- select checkpoint only by maximum `dev/mean_score`;
- freeze selected epoch/checkpoint before using same-epoch Test monitoring;
- record checkpoint SHA256;
- Test remains mandatory epoch-level monitoring but cannot enter any Stage-6 conclusion.

## Three-seed RA evidence

Combine retained valid seed42 with new seeds43/44.

Compute RA-STCH:

- D/P/Mean by seed;
- three-seed arithmetic mean;
- sample std;
- min/max/range.

Compute same-seed RA minus Equal D/P/Mean deltas.

Compute same-seed RA minus Static D/P/Mean deltas.

Compute same-seed RA minus Progress D/P/Mean deltas.

For a same-seed “best simpler balancing” comparator use:

    max(Static Mean, Progress Mean)

and report:

    RA Mean - best_simple Mean

for each seed.

## Frozen claim rule A — contribution over Equal

Record exactly:

    RA-STCH BALANCING CONTRIBUTION OVER EQUAL SUPPORTED

only if ALL hold:

1. RA DEV Mean > Equal DEV Mean on at least 2/3 seeds;
2. RA three-seed Mean > Equal three-seed Mean;
3. RA three-seed D mean is not more than `0.010000` below Equal D mean;
4. RA three-seed P mean is not more than `0.010000` below Equal P mean.

Otherwise record exactly:

    RA-STCH BALANCING CONTRIBUTION OVER EQUAL NOT SUPPORTED

## Frozen claim rule B — advantage beyond simpler balancing

Record exactly:

    RA-STCH ADVANTAGE OVER SIMPLER BALANCING SUPPORTED

only if ALL hold:

1. RA Mean > same-seed `max(Static, Progress)` on at least 2/3 seeds;
2. RA three-seed Mean > both Static three-seed Mean and Progress three-seed Mean;
3. relative to corrected Progress three-seed task means, neither RA D mean nor RA P mean is lower by more than `0.010000`.

Otherwise record exactly:

    RA-STCH ADVANTAGE OVER SIMPLER BALANCING NOT SUPPORTED

The two claim rules are independent. Report both.

## Controller/mechanism diagnostics

For each RA seed at the DEV-selected epoch, record:

- alpha D/P;
- progress D/P;
- grad-norm EMA D/P;
- grad-cosine EMA;
- reliability EMA D/P;
- pseudo scale.

Also report across the three seeds:

- whether all values are finite;
- whether alpha remains within the frozen `[0.2,0.8]` bounds;
- whether alpha values are identical across all seeds or show seed-specific adaptation;
- sign/range of grad-cosine EMA;
- D/P reliability EMA ranges.

These are diagnostics only. They cannot rescue a failed DEV claim gate.

## DEV-only calibration and gate audit

After selected checkpoints are frozen, evaluate **DEV only** for RA seeds42/43/44.

Record for each seed/task:

- observed row count;
- Brier;
- ECE-15.

Record three-seed mean Brier/ECE for D/P.

Compare RA three-seed calibration against Equal and corrected Progress using the already-frozen comparator means where available.

Also report DEV task-modality audio gate mean and sample std for each task and seed from `task_modality_weights`.

Do not recalibrate, threshold-search, or rerun.

No Test row may be used in this post-hoc audit.

## Pseudo/negative-transfer diagnostics

Record:

- frozen pseudo accepted counts D/P `376/1801`;
- frozen pseudo class counts D `376/0`, P `212/1589`;
- RA minus Equal three-seed D/P/Mean;
- RA minus corrected R3-B contextual three-seed means:
  - R3-B D `0.696178`;
  - R3-B P `0.857996`;
  - R3-B Mean `0.777087`.

This is diagnostic only and does not reopen Stage 5.

## Required PROGRESS evidence

Record:

- task/branch;
- retained seed42 checkpoint path/SHA and DEV metrics;
- config equivalence proof;
- firewall commit SHA;
- exact two production commands;
- production count = exactly 2;
- seed43/44 run dirs, MLflow IDs/status/artifact URIs;
- selected epochs/checkpoints/SHA256;
- DEV D/P UAR/MF1/Score/Mean;
- same-epoch Test monitoring only after freeze;
- full three-seed RA mean/std/range;
- all paired comparator deltas;
- both exact frozen claim strings;
- controller diagnostics;
- DEV-only calibration and gate audit;
- pseudo coverage/class counts;
- no Test-driven decision;
- no tuning/rerun;
- no source/config changes.

## Final scope checks

Run:

    git diff --check
    git diff origin/main -- src
    git diff origin/main -- configs
    git status --short
    git diff --stat origin/main...HEAD
    git log -12 --oneline --decorate

Only PROGRESS may differ from origin/main.

## Acceptance criteria

TASK-006C passes only if:

- branch exactly `codex/task-006c` from current manager main;
- no source/config/script changes;
- retained seed42 checkpoint is verified;
- configs17/18 remain frozen;
- firewall pushed before production;
- exactly two production runs occur, seed43 then44;
- seed42 is not rerun;
- DEV-only selection;
- Test is monitoring-only;
- three-seed RA evidence uses retained42 + new43/44 only;
- both frozen claim rules are applied exactly;
- controller, calibration, gate, pseudo, and negative-transfer diagnostics are recorded;
- no tuning, sweep, retry, or Stage-5 reopening;
- branch pushed;
- main/master untouched.

Passing TASK-006C closes only the detailed Stage-6 RA-STCH balancing contribution item. It authorizes no next task.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch `codex/task-006c`;
- firewall SHA;
- final evidence SHA;
- pushed status;
- main/master untouched;
- source/config diff empty;
- retained seed42 checkpoint/SHA;
- seed43/44 run identities/checkpoint SHA;
- exactly two new production runs;
- three-seed RA D/P/Mean and std;
- same-seed RA-minus-Equal;
- same-seed RA-minus-best-simple;
- both exact frozen claim strings;
- controller diagnostics summary;
- calibration/gate summary;
- no Test-driven decision;
- no tuning/rerun;
- Stage 6 active;
- Stage-5 optimization closed;
- no Stage7/Text/Final Test.

For section 6 write only:

    Manager review of TASK-006C; do not start another task.

Stop.
