# TASK-006J: Sparse-MTL Joint-Training Contribution Audit

## Authority and branch

This task follows **MANAGER-DECISION-071**.

Required branch:

    codex/task-006j

Start from current manager-updated `origin/main`.

Create exactly one new task branch from that main.

Stage-5 optimization remains closed. This is a Stage-6 component ablation only.

## Scientific question

Does **joint sparse multi-task training** contribute repeatably to the leading optimized R4 trial-012 composition compared with task-isolated training?

For every seed, train two independent copies of the same trial-012 model:

1. depression-only optimization;
2. Parkinson-only optimization.

The paired task-isolated comparator uses each model only for its active task.

This task does NOT test task-aware fusion; that claim is already closed negative by TASK-006I.

## Interpretation contract

The full method is one jointly trained two-task model.

The no-MTL comparator is a **pair of separately trained task-isolated models**. Each individual model has the same architecture and trainable parameter count as full, but the pair is not a single equal-deployment-size model.

Therefore:

- a positive full result supports positive transfer / efficiency from joint sparse-MTL optimization under the current pipeline;
- a negative result means the joint sparse-MTL contribution is not supported by this paired task-isolated control;
- do not describe the comparator as equal total deployment parameter count.

## Required reading

Read in order:

1. `AGENTS.md`
2. `docs/README.md`
3. `docs/PROJECT_REQUIREMENTS.md`
4. `docs/PROGRESS_EN.md`
5. `docs/plan/STAGE_6.md`
6. `docs/NEXT_TASK_EN.md`
7. `src/fusion/loss/r4_ramps_balance_loss.py`
8. `src/fusion/models/av_r3_disease_query.py`
9. `src/fusion/data/wsm_ramps_semantic_datamodule.py`
10. configs37/38/39.

Historical F0/F1/F2 evidence is context only and must not substitute for this trial-012 matched-seed audit.

## Allowed tracked files

Only:

- `src/fusion/loss/r4_ramps_balance_loss.py`
- `configs/wsm_mm_pd_dep_v1/ablations/58_d_only_seed42.yaml`
- `configs/wsm_mm_pd_dep_v1/ablations/59_p_only_seed42.yaml`
- `configs/wsm_mm_pd_dep_v1/ablations/60_d_only_seed43.yaml`
- `configs/wsm_mm_pd_dep_v1/ablations/61_p_only_seed43.yaml`
- `configs/wsm_mm_pd_dep_v1/ablations/62_d_only_seed44.yaml`
- `configs/wsm_mm_pd_dep_v1/ablations/63_p_only_seed44.yaml`
- `docs/PROGRESS_EN.md`

Forbidden:

- model source changes;
- DataModule changes;
- callback changes;
- `src/audio/*` / `src/video/*` changes;
- existing config changes;
- pseudo-cache changes;
- tuning.

## Minimal backward-compatible loss switch

Extend `WSMR4RampsBalanceLoss` with:

    training_task: str = "both"

Allowed values exactly:

- `both`
- `depression`
- `parkinson`

Reject anything else.

### Default/full behavior

When `training_task="both"`, behavior, numerical ordering, loss value, gradients, diagnostics, and controller interaction must remain tensor/numerically identical to current `origin/main`.

Do not refactor the default path beyond what is necessary to add the switch.

### Depression-only behavior

When `training_task="depression"`:

- compute and optimize only task index 0 objective;
- task index 1 must not contribute to total loss;
- task index 1 observed/pseudo/aux/agreement terms must not be computed into the training objective;
- depression observed/pseudo/aux/agreement formulation remains exactly the current one;
- `pseudo_scale` warm-up remains active;
- RA task-balancing weights are mathematically irrelevant because only one task objective is returned;
- controller callback may remain configured for instrumentation compatibility, but its weights must not affect training;
- inactive Parkinson-specific model parameters must receive zero/None gradients;
- shared projections and shared gate may receive depression gradients as normal.

### Parkinson-only behavior

Symmetric for task index 1.

### Diagnostics

For single-task mode:

- diagnostics for the active task remain valid;
- inactive-task grad norm/cosine/reliability training diagnostics should be absent/uninitialized or explicit neutral fallback;
- no inactive-task reliability update may influence active training;
- default `both` diagnostics must remain unchanged.

## Frozen full reference

Full trial-012 DEV:

| Seed | D | P | Mean |
|---:|---:|---:|---:|
| 42 | 0.759025 | 0.879259 | 0.8191424538 |
| 43 | 0.718527 | 0.804363 | 0.7614450000 |
| 44 | 0.728217 | 0.832060 | 0.7801390000 |

Full three-seed:

- D mean `0.7352563333`;
- P mean `0.8385606667`;
- Mean `0.7869088179`.

Checkpoint SHA256:

- seed42 `104b79e409832839502b4d20a98c054f5efda633930e313ce74f34327e668b2a`;
- seed43 `6af4a4ed21040aff4db2906b0adc8aa5fd27a4b72c375fbb67f68de04ab8446e`;
- seed44 `b06fec6912b27cd19d811b72b2c3c08cd1b0a6c826b637f87a314e3976b88c17`.

Verify these before production.

## Exact task-isolated configs

For each seed create one D-only and one P-only config from the corresponding full ref37/38/39.

### D-only allowed semantic differences

Only:

1. run_name;
2. `loss.params.training_task: depression`;
3. checkpoint callback monitor = `dev/depression/score`;
4. early-stopping callback monitor = `dev/depression/score`;
5. sweep-target callback monitor = `dev/depression/score`.

### P-only allowed semantic differences

Only:

1. run_name;
2. `loss.params.training_task: parkinson`;
3. checkpoint callback monitor = `dev/parkinson/score`;
4. early-stopping callback monitor = `dev/parkinson/score`;
5. sweep-target callback monitor = `dev/parkinson/score`.

All modes remain `max`.
All patience/min_delta/save_top_k settings remain unchanged.
Everything else must match the corresponding full reference.

Required run names:

- 58: `stage6_d_only_trial012_seed42`
- 59: `stage6_p_only_trial012_seed42`
- 60: `stage6_d_only_trial012_seed43`
- 61: `stage6_p_only_trial012_seed43`
- 62: `stage6_d_only_trial012_seed44`
- 63: `stage6_p_only_trial012_seed44`

## Mandatory pre-run firewall

Before any production run:

1. implement the loss switch;
2. prove default `training_task=both` backward compatibility against clean `origin/main`:
   - identical loss value;
   - identical model gradients;
   - identical loss diagnostic/controller state after the same deterministic TRAIN batch;
3. validate configs58..63;
4. prove config equivalence to refs37/38/39 except the allowed task/run/monitor differences;
5. verify seeds exactly 42/42/43/43/44/44;
6. verify full checkpoint paths/SHA;
7. instantiate DataModule/model/loss/callbacks for all six configs;
8. verify every model has exactly `295239` trainable parameters;
9. verify pseudo cache SHA/counts/classes unchanged;
10. D-only isolation:
    - total loss equals D objective;
    - deterministic perturbation of Parkinson logits and P aux logits leaves total loss and all active gradients unchanged;
    - D main/query/candidate/fusion/aux plus shared projections/gate receive finite gradients;
    - P main/query/candidate/fusion/aux parameters receive zero/None gradients;
11. P-only isolation: symmetric;
12. at pseudo_scale=1 verify active-task accepted pseudo supervision produces finite nonzero model gradients;
13. inactive-task accepted pseudo entries do not affect total loss or gradients;
14. pseudo target/reliability tensors detached;
15. active task reliability diagnostics finite when accepted pseudo rows occur;
16. controller/balancing weights cannot change the single-task total loss;
17. no optimizer step;
18. no DEV/Test loader iteration;
19. `git diff --check`;
20. `git diff origin/main -- src/fusion/models src/fusion/data src/common/callbacks src/audio src/video` empty;
21. append firewall evidence to PROGRESS;
22. commit and push one firewall commit.

No production run before the firewall commit is visible on origin.

## Exactly six production runs

Run exactly in this order:

1. D-only seed42;
2. P-only seed42;
3. D-only seed43;
4. P-only seed43;
5. D-only seed44;
6. P-only seed44.

Commands:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/ablations/58_d_only_seed42.yaml

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/ablations/59_p_only_seed42.yaml

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/ablations/60_d_only_seed43.yaml

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/ablations/61_p_only_seed43.yaml

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/ablations/62_d_only_seed44.yaml

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path configs/wsm_mm_pd_dep_v1/ablations/63_p_only_seed44.yaml

No sweep.
No retry for metric improvement.
No extra seed.
No config change after firewall.

## Selection/Test firewall

D-only:

- select checkpoint only by maximum `dev/depression/score`.

P-only:

- select checkpoint only by maximum `dev/parkinson/score`.

Freeze each checkpoint before reading same-epoch Test monitoring.

Inactive-task DEV metrics are diagnostic only and MUST NOT affect selection.

Test cannot affect selection, claim, or follow-up.

## Paired no-MTL comparator

For each seed define:

    single_D(seed) = DEV depression Score from selected D-only checkpoint
    single_P(seed) = DEV Parkinson Score from selected P-only checkpoint
    single_pair_mean(seed) = 0.5 * (single_D(seed) + single_P(seed))

Ignore P from D-only and D from P-only for the primary comparator.

Compute:

- single_D mean/std/range across seeds;
- single_P mean/std/range;
- paired Mean mean/std/range;
- same-seed `full_D - single_D`;
- same-seed `full_P - single_P`;
- same-seed `full_Mean - single_pair_mean`;
- aggregate deltas.

These task-specific deltas are the Stage-6 negative-transfer diagnostic:

- positive full-minus-single = joint-training gain;
- negative full-minus-single = task-specific negative transfer relative to isolated optimization.

## Frozen claim rule

Record exactly:

    SPARSE MTL JOINT-TRAINING CONTRIBUTION SUPPORTED

only if ALL hold:

1. full paired DEV Mean > task-isolated paired Mean on at least 2/3 seeds;
2. full three-seed Mean > task-isolated paired three-seed Mean;
3. full three-seed D is not more than `0.010000` below D-only three-seed D;
4. full three-seed P is not more than `0.010000` below P-only three-seed P.

Otherwise record exactly:

    SPARSE MTL JOINT-TRAINING CONTRIBUTION NOT SUPPORTED

This claim concerns joint sparse-MTL optimization only.

## Diagnostics

For each selected single-task checkpoint record:

- selected active-task Score/UAR/MF1;
- inactive-task DEV Score as diagnostic only;
- pseudo scale;
- active shared-gradient norm on a deterministic TRAIN diagnostic batch;
- active task modality gate mean/std on DEV;
- Brier/ECE-15 for the active task on DEV;
- accepted pseudo coverage/classes for the active task.

For the paired comparator report:

- D-only and P-only active calibration three-seed means;
- full-minus-single active calibration deltas;
- task-specific full-minus-single DEV Score deltas and their signs.

RA controller weights from single-task runs are instrumentation only and cannot be interpreted as task balancing because only one task objective enters training.

## Required PROGRESS evidence

Record:

- task/branch;
- loss source change and default backward-compatibility proof;
- exact single-task routing semantics;
- configs58..63 equivalence;
- full checkpoint SHAs;
- pseudo-cache identity;
- firewall evidence/SHA;
- exact six production commands;
- production count exactly 6;
- six run/MLflow identities;
- active-task selected epoch/checkpoint/SHA;
- active DEV UAR/MF1/Score;
- inactive DEV Score diagnostic;
- Test monitoring after freeze only;
- paired no-MTL per-seed/aggregate results;
- full-minus-single task/Mean deltas;
- exact frozen claim string;
- calibration/gate/gradient/coverage diagnostics;
- explicit statement that each task-isolated model has `295239` parameters but the pair is not an equal-total-deployment-size comparator;
- no tuning/retry/Test-driven decision.

## Final scope checks

Run:

    git diff --check
    git diff origin/main -- src/fusion/models
    git diff origin/main -- src/fusion/data
    git diff origin/main -- src/common/callbacks
    git diff origin/main -- src/audio
    git diff origin/main -- src/video
    git status --short
    git diff --stat origin/main...HEAD
    git log -14 --oneline --decorate

Only the authorized R4 loss file, configs58..63, and PROGRESS may differ.

## Acceptance criteria

TASK-006J passes only if:

- branch exactly `codex/task-006j`;
- default both-task loss behavior is unchanged;
- single-task loss routing is isolated as specified;
- each model remains `295239` trainable parameters;
- configs differ only as authorized;
- pseudo path unchanged;
- firewall pushed before production;
- exactly six runs in fixed order;
- active-task DEV-only selection;
- Test monitoring only;
- frozen claim applied exactly;
- negative-transfer/calibration/gate/coverage diagnostics recorded;
- no tuning/sweep/retry;
- branch pushed;
- main/master untouched.

Passing TASK-006J closes the remaining Stage-6 sparse-MTL item. It authorizes no Stage-7 action by itself; manager must first review whether Stage 6 is complete and freeze the supported/unsupported claim ledger.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Include branch, firewall/final SHA, pushed status, main/master untouched, default backward compatibility, exactly six runs, per-seed task-isolated D/P/paired Mean, three-seed aggregates, full-minus-single deltas, exact claim string, negative-transfer/calibration/gate/coverage summary, parameter-count caveat, no Test-driven decision, Stage 6 active, Stage-5 closed, Stage7/Text/Final Test locked.

For section 6 write only:

    Manager review of TASK-006J; do not start another task.

Stop.
