# TASK-006-PRE-AUDIO-CONFIRM: Frozen Audio Robustness Confirmation on Seeds43/44

## Authority and branch

This task follows **MANAGER-OVERRIDE-058**.

Required branch:

    codex/task-006-pre-audio-confirm

Start from the current manager-updated `origin/main`.

Create exactly one new task branch from that `origin/main`.

The previously assigned `TASK-006A-BUNDLE` is PAUSED / UNSTARTED. Do not execute any shuffled-pseudo work in this task.

Stage 5 remains CLOSED NEGATIVE as a historical decision. This task audits the robustness of the comparator; it does not tune or promote a model.

## Goal

Run the **exact frozen temporal-audio method** on two additional random seeds, 43 and44, so the project has matched-seed DEV evidence against the already frozen R4 RA-STCH confirmation seeds42/43/44.

This task answers one narrow question:

> Is the frozen audio baseline itself at least as seed-stable as the multimodal R4 confirmation, or does audio show comparable/greater seed degradation?

Exactly **two** new production training invocations are authorized.

No sweep. No hyperparameter search. No rerun for metric improvement.

## Required reading

Read in this exact order:

1. `AGENTS.md`
2. `docs/README.md`
3. `docs/PROJECT_REQUIREMENTS.md`
4. `docs/PROGRESS_EN.md`
5. `docs/NEXT_TASK_EN.md`
6. `configs/wsm_mm_pd_dep_v1/audio/00_frozen_baseline.yaml`
7. only the audio source files needed to instantiate/verify the already registered frozen method; read-only

Do not read historical archives or closed-stage plans unless a concrete verification issue requires them.

## Frozen reference

Canonical config:

    configs/wsm_mm_pd_dep_v1/audio/00_frozen_baseline.yaml

Historical seed42 reference:

- seed: `42`
- run: `wsm_audio_models-e0ce-006`
- selected epoch: `4`
- checkpoint SHA256: `0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2`
- DEV D Score: `0.7479183895`
- DEV P Score: `0.8277353635`
- DEV Mean: `0.7878268765`

Frozen R4 RA-STCH confirmation comparator:

| Seed | D Score | P Score | Mean |
|---:|---:|---:|---:|
| 42 | 0.759025 | 0.879259 | 0.8191424538 |
| 43 | 0.718527 | 0.804363 | 0.7614450000 |
| 44 | 0.728217 | 0.832060 | 0.7801390000 |

R4 three-seed aggregate already frozen:

- D mean/std: `0.7352563333 / 0.0211467766`
- P mean/std: `0.8385606667 / 0.0378688091`
- Mean mean/std: `0.7869088179 / 0.0294384420`

## Allowed tracked files

Codex may add/modify only:

- `configs/wsm_mm_pd_dep_v1/audio/01_frozen_baseline_seed43.yaml`
- `configs/wsm_mm_pd_dep_v1/audio/02_frozen_baseline_seed44.yaml`
- `docs/PROGRESS_EN.md`

No source file may change.

## Config freeze

Create the two configs from `00_frozen_baseline.yaml`.

Seed43 config:

    configs/wsm_mm_pd_dep_v1/audio/01_frozen_baseline_seed43.yaml

Required differences from config00:

- `seed: 43`
- `experiment_info.params.run_name: frozen_audio_wavlm_l9_pool4_seed43`

Seed44 config:

    configs/wsm_mm_pd_dep_v1/audio/02_frozen_baseline_seed44.yaml

Required differences from config00:

- `seed: 44`
- `experiment_info.params.run_name: frozen_audio_wavlm_l9_pool4_seed44`

There must be **no other semantic difference** from config00.

In particular, do not change:

- data root, task, cache, split/test filters, batch size, workers;
- WavLM model/layer or temporal pooling;
- model architecture or dropout;
- loss or its coefficients;
- optimizer, lr, weight decay;
- epochs, mixed precision, grad clipping;
- metrics;
- checkpoint/early-stopping monitors;
- callbacks/loggers/instrumentation.

## Forbidden actions

- Do not modify anything under `src/audio`.
- Do not modify any other `src/*` file.
- Do not modify `src/chimera_plugin.py`.
- Do not tune architecture, loss, optimizer, learning rate, regularization, patience, batch size, feature layer, temporal pool, or threshold.
- Do not change dataset membership or feature cache.
- Do not run seed42 again.
- Do not run any seed other than43 and44.
- Do not run R4 or any multimodal system.
- Do not perform a sweep.
- Do not rerun a failed/weak seed for metric improvement.
- Do not use Test metrics for checkpoint selection, comparison, ranking, interpretation, or deciding follow-up work.
- Do not resume `TASK-006A-BUNDLE`.
- Do not start another Stage-6 task.
- Do not start Text/Description, Stage 7, or Final Test.

## Mandatory pre-run firewall

Before either production run:

1. validate both new configs with Chimera;
2. programmatically prove each resolved config is identical to config00 except the authorized `seed` and `run_name`;
3. instantiate the existing DataModule/model/loss from each config;
4. verify the model/loss build and a tiny TRAIN-only forward/loss/backward smoke test succeeds with finite loss and finite/non-zero trainable gradients;
5. perform no optimizer step;
6. do not iterate DEV or Test during this smoke;
7. run:
   - `git diff --check`
   - `git diff origin/main -- src` — MUST be empty;
8. append the firewall evidence to `docs/PROGRESS_EN.md`;
9. commit and push a **firewall commit** to `origin/codex/task-006-pre-audio-confirm`.

No production training may begin before that firewall commit exists on origin.

## Exact production sequence

Run exactly:

1. seed43;
2. seed44.

Commands:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/audio/01_frozen_baseline_seed43.yaml

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/audio/02_frozen_baseline_seed44.yaml

No third production invocation.

## Checkpoint selection and Test firewall

For each run:

- select the checkpoint/epoch only by maximum `dev/mean_score`;
- freeze the selected epoch/checkpoint identity before inspecting its same-epoch Test monitoring values;
- record selected checkpoint SHA256;
- TEST_NONE/SOFT/HARD remain mandatory monitoring outputs because the project stack emits them, but they must not affect any conclusion in this task.

All seed-robustness comparisons in this task use **DEV only**.

## Required DEV evidence per new seed

For seed43 and seed44 record:

- selected epoch;
- checkpoint path and SHA256;
- DEV depression UAR, MF1, Score;
- DEV Parkinson UAR, MF1, Score;
- DEV Mean_Score;
- training stop reason / total epochs;
- run directory;
- MLflow run ID/status/artifact URI;
- resolved config / commit identity.

After checkpoint freeze, record same-epoch TEST_NONE/SOFT/HARD monitoring separately and explicitly label it non-decision evidence.

## Frozen matched-seed analysis

Using historical audio seed42 plus new audio seeds43/44, compute:

### Audio three-seed table

For each seed42/43/44:

- D Score;
- P Score;
- Mean.

### Audio aggregate

Compute sample mean and sample standard deviation across seeds42/43/44 for:

- D Score;
- P Score;
- Mean.

Also report min, max, and range for DEV Mean.

### Matched R4-vs-audio deltas

For each seed42/43/44 compute:

    R4 - audio

for:

- D Score;
- P Score;
- Mean.

Count how many of the three seeds have:

    R4 DEV Mean > audio DEV Mean

### Variability comparison

Place the frozen R4 and new audio three-seed statistics side by side:

- D mean/std;
- P mean/std;
- Mean mean/std;
- Mean min/max/range.

Do not perform statistical significance testing from only three seeds.

Do not alter, tune, rerun, or choose a model based on these results.

## Interpretation boundary

Codex must report the facts, but **must not**:

- reopen Stage 5;
- promote R4;
- demote audio;
- choose the next Stage-6 ablation;
- change a config after seeing seed43/44;
- turn seed43/44 into tuning targets.

The manager will decide after reviewing the branch whether the phrase "strongest robust audio reference" remains justified and how the matched-seed evidence affects Stage-6 comparisons.

## Required evidence in PROGRESS_EN.md

Record:

- task ID and branch;
- config paths;
- exact proof that configs differ from config00 only by seed/run_name;
- firewall commands/results;
- firewall commit SHA and pushed status;
- exact two production commands;
- run directories / MLflow identities;
- selected DEV-only epochs/checkpoints and checkpoint SHA256;
- complete DEV metrics;
- same-epoch Test monitoring only after freeze;
- audio seed42/43/44 table;
- audio three-seed mean/std and Mean range;
- matched R4-audio deltas;
- R4-vs-audio variability table;
- production invocation count = exactly 2;
- explicit statement: no Test-driven decision, no tuning, no rerun;
- explicit statement: `src/audio` and all `src/*` unchanged;
- `TASK-006A-BUNDLE` remains paused/unstarted.

## Final scope checks

Run exactly:

    git diff --check
    git diff origin/main -- src
    git status --short
    git diff --stat origin/main...HEAD
    git log -10 --oneline --decorate

The source diff MUST be empty.

Only the two authorized configs and `docs/PROGRESS_EN.md` may differ from `origin/main`.

## Acceptance criteria

This task passes only if:

- branch is exactly `codex/task-006-pre-audio-confirm` from current manager-updated `origin/main`;
- only the two authorized audio configs and PROGRESS change;
- config equivalence proof passes;
- firewall commit is pushed before production;
- exactly two production runs occur, in order seed43 then seed44;
- no source changes;
- no tuning/sweep/rerun;
- DEV-only checkpoint selection;
- Test monitoring only after checkpoint freeze and never used in interpretation;
- checkpoint SHA256 values recorded;
- complete three-seed audio aggregate computed using historical seed42 + new seeds43/44;
- matched-seed R4-audio deltas and variability comparison recorded;
- branch pushed;
- main/master untouched by Codex.

## Required handoff

Respond in English using exactly these six sections:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch `codex/task-006-pre-audio-confirm`;
- firewall commit SHA;
- final evidence commit SHA;
- pushed-to-origin status;
- main/master untouched;
- source diff empty;
- production runs = exactly 2;
- seed43 and seed44 selected checkpoint SHA256;
- audio seed42/43/44 D/P/Mean table;
- audio three-seed mean/std and Mean range;
- frozen R4 three-seed mean/std;
- matched R4-audio deltas for each seed;
- count of seeds where R4 Mean > matched audio Mean;
- no Test-driven decision;
- no post-hoc tuning/rerun;
- TASK-006A-BUNDLE still paused/unstarted;
- Stage 6 active;
- Stage 5 closed negative historically;
- no Stage7/Text/Final Test.

For section 6 write only:

    Manager review of TASK-006-PRE-AUDIO-CONFIRM; do not start another task.

Stop after this task.
