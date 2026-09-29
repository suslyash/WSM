# TASK-007B: Missing Final Seeds and Five-Seed DEV Checkpoint Freeze

## Authority and branch

This task follows **MANAGER-DECISION-079**.

Required branch:

    codex/task-007b

Start from current `origin/main`.

This task is the final **DEV-only production evidence** task before Final Test.

Final Test remains locked.

## Frozen compared methods and seeds

Methods are fixed:

1. frozen temporal audio;
2. full R4 trial012;
3. equal-parameter shared fusion.

Seeds are fixed:

    42, 43, 44, 45, 46

Accepted seeds42-44 MUST NOT be retrained.

Do not add/drop/replace a method after seeing seed45/46.

## Allowed tracked files

Only:

- `docs/STAGE7_FINAL_DEV_EVIDENCE_EN.md`
- `docs/PROGRESS_EN.md`

No source changes.
No config changes.
No script changes.
No cache changes.
No dependency changes.
No existing evidence edits.

All six production configs already exist in `origin/main`.

## Frozen six-run order

Execute exactly:

1. audio seed45;
2. full R4 seed45;
3. shared fusion seed45;
4. audio seed46;
5. full R4 seed46;
6. shared fusion seed46.

Configs:

- `configs/wsm_mm_pd_dep_v1/audio/03_final_seed45.yaml`
- `configs/wsm_mm_pd_dep_v1/fusion/40_r4_trial012_final_seed45.yaml`
- `configs/wsm_mm_pd_dep_v1/fusion/42_shared_fusion_final_seed45.yaml`
- `configs/wsm_mm_pd_dep_v1/audio/04_final_seed46.yaml`
- `configs/wsm_mm_pd_dep_v1/fusion/41_r4_trial012_final_seed46.yaml`
- `configs/wsm_mm_pd_dep_v1/fusion/43_shared_fusion_final_seed46.yaml`

No retry for any failed production invocation inside TASK-007B.

If any production command exits nonzero, STOP immediately, record which invocation failed and whether any optimizer step occurred if determinable without Test inspection, push an incomplete evidence update, and hand back to manager. Do not rerun it and do not continue to later jobs.

Poor DEV performance is NOT a failure and must not stop the sequence.

## Mandatory pre-run firewall

Before run 1:

1. verify branch/base and clean working tree;
2. verify `docs/STAGE7_FINAL_FREEZE_EN.md` is present and unchanged;
3. verify the six config file SHA256 values and semantic identities against their references;
4. validate all six configs;
5. verify trainable parameter counts:
   - audio `3033416`;
   - full R4 `295239`;
   - shared `295239`;
6. verify semantic pseudo cache SHA256:
   `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`;
7. verify canonical audio/video feature/cache identities are unchanged;
8. verify seed/run names and exact frozen order;
9. verify checkpoint + early-stopping selectors are `dev/mean_score`, mode max;
10. verify no tracked diff outside PROGRESS;
11. do not iterate any Test loader;
12. do not compute any metric;
13. append firewall evidence to PROGRESS;
14. `git diff --check`;
15. commit and PUSH firewall.

No production command before firewall commit is visible on origin.

## Test-value blindness during training

The existing training stack may compute mandatory epoch-level TEST_NONE/SOFT/HARD monitoring.

During TASK-007B those values are forbidden evidence.

To minimize accidental exposure:

- run every training command with stdout/stderr redirected to an external temporary file under `/tmp`;
- do not open/cat/tail/search the raw redirected stdout/stderr after a successful run;
- do not open `summary.txt` or raw training logs if they contain Test values;
- retrieve production evidence only from exact run/checkpoint metadata and MLflow queries restricted to `dev/` metric keys;
- never enumerate, query, fetch, print, copy, compare, or summarize a Test metric key/value;
- existing historical Test values are also irrelevant to this task.

The fact that the stack computed monitoring does not authorize looking at it.

## Exact production commands

Use:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/audio/03_final_seed45.yaml \
      >/tmp/wsm_stage7_audio45.stdout 2>&1

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/fusion/40_r4_trial012_final_seed45.yaml \
      >/tmp/wsm_stage7_r4_45.stdout 2>&1

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/fusion/42_shared_fusion_final_seed45.yaml \
      >/tmp/wsm_stage7_shared45.stdout 2>&1

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/audio/04_final_seed46.yaml \
      >/tmp/wsm_stage7_audio46.stdout 2>&1

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/fusion/41_r4_trial012_final_seed46.yaml \
      >/tmp/wsm_stage7_r4_46.stdout 2>&1

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/fusion/43_shared_fusion_final_seed46.yaml \
      >/tmp/wsm_stage7_shared46.stdout 2>&1

After each command, inspect only its exit status and filesystem/run identity. Do not read the redirected output on success.

No sweep.
No extra seed.
No config/source/cache change.
No retry.

## New seed45/46 checkpoint selection

For each successful new run:

1. identify the exact run/MLflow identity by the frozen run_name;
2. query only metric keys under `dev/`;
3. select/freeze checkpoint exactly by maximum `dev/mean_score` under the existing checkpoint callback;
4. record selected epoch;
5. record full checkpoint path;
6. compute checkpoint SHA256;
7. record DEV D/P UAR/MF1/Score and Mean_Score at that selected epoch;
8. do not query or inspect any Test key/value.

Do not choose a different epoch based on calibration or any other diagnostic.

## Freeze all fifteen checkpoint identities

After all six jobs complete, assemble the final 15-checkpoint ledger:

- audio seeds42-46;
- full R4 seeds42-46;
- shared fusion seeds42-46.

For existing seeds42-44, use accepted checkpoint identities from `docs/STAGE7_FINAL_FREEZE_EN.md` and committed evidence.

Resolve and verify the full local checkpoint path for every historical checkpoint by SHA256/run identity without rerunning training.

For every one of the 15 record:

- method;
- seed;
- config path;
- run/MLflow identity when available;
- selected epoch;
- checkpoint path;
- checkpoint SHA256.

If an accepted historical MLflow ID was never recorded, do not invent one; retain the exact accepted run identity and mark MLflow ID as not recorded.

All checkpoint paths must exist and every SHA must match.

## Unified DEV-only evaluation for all 15 checkpoints

After all 15 identities are frozen, perform exactly one DEV-only evaluation pass per checkpoint using the current frozen evaluation code.

Never call/iterate a Test loader.

Use the canonical DEV set and the same binary metric definition used by `wsm_segment_metrics_callback`.

For every checkpoint/task record:

- observed sample count;
- UAR;
- MF1;
- Score.

Record Mean_Score for every checkpoint.

For historical seeds42-44, verify reproduced Score values against accepted evidence. Small floating serialization noise only is acceptable; investigate any material mismatch without touching Test.

## DEV calibration — frozen procedure

During the same DEV-only pass, compute probability = sigmoid(logit) on observed target rows only.

Brier:

    mean((probability - target)^2)

ECE-15:

- 15 equal-width confidence bins on [0,1];
- bins 0-13 are left-closed/right-open;
- final bin includes probability 1.0;
- for each nonempty bin:
  - confidence = mean(probability);
  - accuracy = mean(binary target);
  - contribution = (bin_count / N) * abs(accuracy - confidence);
- ECE is the sum of contributions.

No threshold fitting.
No recalibration.
No class balancing in calibration.

For every method/seed/task record Brier and ECE-15.

## Frozen five-seed statistics

Use exactly the TASK-007A plan and multiplier:

    2.7764451051977987

For each method and each of D Score, P Score, Mean_Score over seeds42-46 report:

- five seed values;
- arithmetic mean;
- sample std (`ddof=1`);
- min;
- max;
- range;
- two-sided 95% Student-t CI:
  `mean ± 2.7764451051977987 * s / sqrt(5)`.

For DEV Brier and ECE-15, per method/task report:

- five values;
- mean;
- sample std;
- same 95% CI.

Do not replace the CI method after seeing results.

## Frozen paired comparisons

Same-seed deltas for D/P/Mean:

1. full R4 minus audio;
2. shared fusion minus audio;
3. shared fusion minus full R4.

For every pair/metric report:

- five seed deltas in seed order;
- mean delta;
- sample std;
- same two-sided 95% Student-t CI.

For Mean additionally report wins/ties/losses using exact numerical comparison of the frozen Score values.

These are descriptive paired comparisons. Do not change the finalist set regardless of outcome.

## Required evidence document

Create:

    docs/STAGE7_FINAL_DEV_EVIDENCE_EN.md

It must contain:

- frozen methods/seeds;
- six new production run identities;
- final 15-checkpoint ledger;
- per-seed DEV D/P UAR/MF1/Score and Mean for all methods;
- five-seed summaries/95% CIs;
- paired delta tables/95% CIs;
- DEV calibration tables and summaries;
- explicit confirmation that Test values were not inspected;
- explicit confirmation that no method/config/threshold changed after TASK-007A.

Record exactly on full success:

    STAGE-7 FIVE-SEED DEV CHECKPOINT FREEZE COMPLETE

If any required run/checkpoint/evidence is missing, record:

    STAGE-7 FIVE-SEED DEV CHECKPOINT FREEZE INCOMPLETE

and name the exact missing evidence.

## PROGRESS update

Record:

- branch;
- firewall SHA;
- final SHA;
- exact six production commands and completion order;
- six new run identities;
- seed45/46 selected epoch/checkpoint path/SHA and DEV metrics;
- all 15 frozen checkpoint identities;
- five-seed DEV statistics;
- paired comparisons;
- DEV Brier/ECE-15;
- exact completion string;
- no retry/tuning/change;
- Test values not inspected;
- Final Test still locked;
- Stage 7 active.

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

Only `docs/STAGE7_FINAL_DEV_EVIDENCE_EN.md` and `docs/PROGRESS_EN.md` may differ from origin/main.

## Acceptance criteria

TASK-007B passes only if:

- branch exactly `codex/task-007b`;
- firewall pushed before production;
- exactly six successful production invocations in frozen order;
- no retry or extra seed;
- no source/config/cache/dependency change;
- selected checkpoints use DEV/Mean_Score only;
- all fifteen checkpoint paths/SHA identities are frozen and verified;
- unified DEV-only metrics/calibration are complete;
- frozen five-seed CI/paired plans are applied exactly;
- Test metric values are not inspected/transcribed/queried;
- exact completion string is correct;
- branch pushed;
- main/master untouched.

Passing TASK-007B does NOT itself authorize reading Final Test. Manager must review and assign a separate TASK-007C.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Include branch, firewall/final SHA, pushed status, main/master untouched, six-run completion/order, new run/checkpoint identities, fifteen-checkpoint freeze status, five-seed DEV summaries/CIs, paired Mean comparison summary, DEV calibration summary, exact completion string, Test inspection=false, no retry/tuning/config/source/cache changes, Stage 7 active, Final Test locked.

For section 6 write only:

    Manager review of TASK-007B; do not start another task.

Stop.
