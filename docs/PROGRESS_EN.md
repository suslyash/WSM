# WSM Project Progress

## 1. Current State

Plan initialized: 2026-09-23.

Current stage: **Stage 6 — ACTIVE (ablations and claims audit)**.

Stage 5: **CLOSED NEGATIVE**.

Stage 7: **LOCKED**.

Final Test authorized: **no**.

Current atomic task: **TASK-006A-BUNDLE — Stage-6 shuffled pseudo-target negative control on the Equal composition**.

Expected Codex branch: `codex/task-006a`.

At the time of this documentation refactor, that branch had not yet been created.

## 2. Default Context Policy

This file is the compact authoritative status/roadmap entrypoint.

Default agents should read:

1. `AGENTS.md`;
2. `docs/README.md`;
3. `docs/PROJECT_REQUIREMENTS.md`;
4. this file;
5. `docs/NEXT_TASK_EN.md`;
6. only the active stage plan and source/config files explicitly named by the task.

Do **not** load closed-stage plans or the historical progress archive unless the active task or manager explicitly requires them.

Optional historical evidence:

- [archive/PROGRESS_EN_THROUGH_STAGE5_2026-09-28.md](archive/PROGRESS_EN_THROUGH_STAGE5_2026-09-28.md) — complete former progress ledger through Stage-5 closure;
- [plan/README.md](plan/README.md) — stage-plan index;
- [PLAN.md](PLAN.md) — cross-stage formulation, architecture, experiment matrix, promotion rules, risks, and orchestration details.

## 3. High-Level Project Roadmap

| Stage | Status | High-level goal | Detailed plan |
|---|---|---|---|
| 0. Reproducible base | COMPLETE | Freeze and validate the historical audio baseline and instrumentation contract. | [plan/STAGE_0.md](plan/STAGE_0.md) |
| 1. Manifest / partial labels | COMPLETE | Canonical sparse-label manifest, split contract, masking, and evaluation separation. | [plan/STAGE_1.md](plan/STAGE_1.md) |
| 2. Video | COMPLETE | Compare the bounded video families and establish the accepted V2 video reference. | [plan/STAGE_2.md](plan/STAGE_2.md) |
| 3. Text / description | DEFERRED | Bounded text/description study after the core Stage-6 research cycle. | [plan/STAGE_3.md](plan/STAGE_3.md) |
| 4. Fusion baselines | COMPLETE | Honest A+V baselines and strong-audio fusion search; no safe robust winner. | [plan/STAGE_4.md](plan/STAGE_4.md) |
| 5. RAMPS | CLOSED NEGATIVE | Reliability/pseudo-label, disease-query, balancing, Optuna, and true-seed confirmation cycle. No robust promoted multimodal method. | [plan/STAGE_5.md](plan/STAGE_5.md) |
| 6. Ablations / claims audit | ACTIVE | Determine which component/causal claims are supported; retain negative controls. | [plan/STAGE_6.md](plan/STAGE_6.md) |
| 7. Final evaluation | LOCKED | Final multi-seed evaluation only after configuration/claim freeze and manager authorization. | [plan/STAGE_7.md](plan/STAGE_7.md) |

## 4. Frozen Scientific References

### Strongest robust reference — frozen temporal audio

- run: `wsm_audio_models-e0ce-006`;
- selected epoch: 4;
- checkpoint SHA256: `0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2`;
- DEV depression Score: `0.7479183895`;
- DEV Parkinson Score: `0.8277353635`;
- DEV Mean_Score: `0.7878268765`.

`src/audio` remains frozen.

### Accepted Stage-2 video reference

- model: V2 prototype-aware video;
- DEV depression Score: `0.6201013364`;
- DEV Parkinson Score: `0.7930427585`;
- DEV Mean_Score: `0.7065720475`.

### Frozen semantic pseudo cache

- path: `/media/maxim/Programs/Features/WSM/ramps_r2_semantic_v1/train_missing_targets.pt`;
- SHA256: `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`;
- TRAIN rows: `6325`;
- accepted missing depression: `376/2665`, all positive;
- accepted missing Parkinson: `1801/3660`, positive/negative `212/1589`;
- observed truth always overrides pseudo supervision;
- no missing-label correctness or comorbidity claim is authorized.

## 5. Stage-5 Closure — Authoritative Summary

Stage 5 is closed negative by **MANAGER-DECISION-057**.

The final optimized R4 RA-STCH trial `18c6-012` was selected by DEV only and then frozen for true-seed confirmation.

| Seed | DEV D | DEV P | DEV Mean | Relaxed safe | Exact D/P non-regression |
|---:|---:|---:|---:|---|---|
| 42 | 0.759025 | 0.879259 | 0.8191424538 | yes | yes |
| 43 | 0.718527 | 0.804363 | 0.7614450000 | no | no |
| 44 | 0.728217 | 0.832060 | 0.7801390000 | no | no |

Three-seed aggregates:

- D mean/std: `0.7352563333 / 0.0211467766`;
- P mean/std: `0.8385606667 / 0.0378688091`;
- Mean mean/std: `0.7869088179 / 0.0294384420`;
- seeds beating frozen audio Mean: `1/3`;
- relaxed-safe seeds: `1/3`;
- exact D/P non-regression seeds: `1/3`;
- checkpoint SHA256 values: pairwise distinct.

Frozen decision: **ROBUST THREE-SEED CONFIRMATION FAIL**.

Consequences:

- no currently trained multimodal/RAMPS family robustly exceeds the frozen audio reference;
- Candidate B and corrected R3-B are not substitutes: their 20-trial Optuna searches produced zero relaxed-safe trials;
- Stage 5 must not be reopened by tuning against seeds43/44;
- seeds43/44 are confirmation evidence, not future tuning targets;
- frozen audio remains the strongest robust reference.

Full Stage-5 chronology, Optuna ledger, corrective reviews, and checkpoint provenance are preserved in the optional historical archive.

## 6. Stage-6 General Plan

Stage 6 is a **claims/ablation audit, not a tuning stage**.

The detailed contract is in [plan/STAGE_6.md](plan/STAGE_6.md). At high level it covers:

1. sparse MTL / task-aware fusion contribution;
2. direct pseudo-supervision contribution;
3. uncertainty/reliability contribution;
4. semantic-evidence contribution;
5. balancing/RA-STCH contribution;
6. modality removals;
7. shuffled/mismatched pseudo-target negative control;
8. corpus probe;
9. equal-parameter control if model size becomes a confound;
10. calibration, gradient, negative-transfer, coverage, and gate diagnostics.

Each scientific claim must map to an ablation or negative control. Negative results are retained. Test protocols remain monitoring-only and cannot drive decisions.

## 7. Active Task — TASK-006A-BUNDLE

Purpose: test the **sample-specific pseudo alignment** claim without changing coverage, class balance, pseudo-value/reliability distributions, architecture, loss, optimizer, or warm-up.

Frozen reference: the three-seed **Equal** full composition.

Matched Equal DEV results:

| Seed | D Score | P Score | Mean |
|---:|---:|---:|---:|
| 42 | 0.712498 | 0.843342 | 0.777920 |
| 43 | 0.703299 | 0.856043 | 0.779671 |
| 44 | 0.707977 | 0.851878 | 0.779927 |

Matched three-seed means:

- D `0.707925`;
- P `0.850421`;
- Mean `0.779173`.

The task must:

- derive one deterministic within-task missing-row tuple derangement from the frozen semantic cache;
- preserve canonical rows, observed truth, accepted counts, class balance, and pseudo tuple multisets;
- run exactly three Equal-composition production runs at seeds42/43/44;
- select checkpoints only by `dev/mean_score`;
- perform the frozen matched-vs-shuffled interpretation and DEV-only calibration audit;
- make no tuning decision from the result;
- stop after manager handoff.

Full executable requirements are authoritative in [NEXT_TASK_EN.md](NEXT_TASK_EN.md).

## 8. Non-Negotiable Current Boundaries

- Do not tune or modify `src/audio`.
- Do not reopen Stage 5.
- Seeds43/44 from R4 confirmation are not tuning targets.
- Unknown labels remain masked, never converted to negative.
- Observed ground truth overrides pseudo labels.
- Test metrics cannot drive epoch, architecture, hyperparameter, threshold, ablation, or follow-up selection.
- General Text/Description remains deferred.
- Stage 7 and Final Test remain locked.
- Do not claim verified recovery of genuinely missing disease labels or comorbidity without dual-annotated evidence.

## 9. Recent Authoritative Manager Decisions

Only recent state-changing decisions are repeated here; older decisions are in the archive.

### MANAGER-DECISION-056 — Triple Optuna accepted; R4 trial 012 selected for confirmation

- Candidate B objective winner: Mean `0.8165907903`, but 0/20 relaxed-safe.
- corrected R3-B objective winner: Mean `0.8159243728`, but 0/20 relaxed-safe.
- corrected R4 RA-STCH objective winner `18c6-012`: D/P/Mean `0.759025/0.879259/0.8191424538`;
- R4 had 4/20 relaxed-safe trials; trial 012 was the only objective winner with exact audio D/P non-regression;
- this was DEV-only family selection, not final promotion.

### MANAGER-DECISION-057 — R4 confirmation failure accepted; Stage 5 closed negative; Stage 6 activated

- true seed43 and seed44 confirmations failed the frozen robust gate;
- three-seed Mean `0.7869088179` remained below frozen audio Mean `0.7878268765`;
- Stage 5 closed negative;
- Stage 6 activated;
- no tuning against confirmation seeds is allowed.

### MANAGER ASSIGNMENT — TASK-006A-BUNDLE

- first Stage-6 task is the shuffled/mismatched pseudo-target negative control on matched Equal;
- exactly three production runs are authorized;
- task is claims audit only, not tuning;
- passing the task closes only this Stage-6 negative-control item and does not authorize the next task.

## 10. Historical Evidence

The full former 5,000+ line ledger is intentionally no longer part of default context.

Use it only when a historical implementation detail, exact old command, old PR/commit, or prior corrective review is material:

[archive/PROGRESS_EN_THROUGH_STAGE5_2026-09-28.md](archive/PROGRESS_EN_THROUGH_STAGE5_2026-09-28.md)
