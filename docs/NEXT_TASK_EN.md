# TASK-005F-BUNDLE-C1: Correct Progress Scalarization and RA Gradient Diagnostics, Then Rerun Only Affected R4 Variants

## Authority and branch

This is one manager-authorized corrective bounded task.

Required branch:

    codex/task-005f

Reuse of the existing branch is explicitly authorized. Preserve all prior Codex/manager commits and all original R4 run evidence. Do not reset, rebase away history, force-push, or modify main/master.

The original Equal and Static-STCH seed42 runs remain valid and MUST NOT be rerun.

The original Progress and RA-STCH seed42 runs are preserved as invalid/exploratory evidence because the implemented scalarizer/controller deviated from the frozen contract.

Maximum NEW production training invocations in C1: 4.

## Goal

Correct exactly two implementation defects:

1. `mode="progress"` must use the frozen linear weighted sum:
   `w_D * L_D + w_P * L_P`.
2. RA gradient norm/cosine EMA updates must occur only on batches where BOTH task objectives are active.

Then:

- rerun corrected Progress seed42 once;
- rerun corrected RA-STCH seed42 once;
- evaluate the ORIGINAL frozen RA continuation gate using:
  - retained valid Equal seed42;
  - retained valid Static-STCH seed42;
  - corrected Progress seed42;
  - corrected RA seed42;
- conditionally run corrected RA-STCH true seeds43/44 only if every original continuation condition passes.

No other experiment is authorized.

## Required reading

Read in this order:

1. AGENTS.md
2. docs/README.md
3. docs/PROJECT_REQUIREMENTS.md
4. docs/PLAN.md, especially Stage 5 R4 and promotion rule
5. docs/PROGRESS_EN.md through MANAGER-REVIEW-045
6. docs/NEXT_TASK_EN.md
7. src/fusion/loss/r4_ramps_balance_loss.py
8. src/common/callbacks/wsm_r4_balance_callback.py
9. configs/wsm_mm_pd_dep_v1/fusion/13_r4_equal_seed42.yaml
10. configs/wsm_mm_pd_dep_v1/fusion/14_r4_static_stch_seed42.yaml
11. configs/wsm_mm_pd_dep_v1/fusion/15_r4_progress_seed42.yaml
12. configs/wsm_mm_pd_dep_v1/fusion/16_r4_ra_stch_seed42.yaml
13. configs/wsm_mm_pd_dep_v1/fusion/17_r4_ra_stch_seed43.yaml
14. configs/wsm_mm_pd_dep_v1/fusion/18_r4_ra_stch_seed44.yaml

## Allowed tracked files

Codex may modify only:

- src/fusion/loss/r4_ramps_balance_loss.py
- configs/wsm_mm_pd_dep_v1/fusion/15_r4_progress_seed42.yaml
- configs/wsm_mm_pd_dep_v1/fusion/16_r4_ra_stch_seed42.yaml
- configs/wsm_mm_pd_dep_v1/fusion/17_r4_ra_stch_seed43.yaml
- configs/wsm_mm_pd_dep_v1/fusion/18_r4_ra_stch_seed44.yaml
- docs/PROGRESS_EN.md

No other tracked file may change.

Do NOT modify:

- src/common/callbacks/wsm_r4_balance_callback.py;
- src/chimera_plugin.py;
- any model;
- any DataModule;
- any existing R2/R3 loss;
- Equal/Static configs 13/14;
- src/audio;
- src/video;
- src/common;
- PLAN/PROJECT_REQUIREMENTS.

If a new blocker outside this exact scope appears, stop and report it.

## 1. Correct the progress scalarizer

Current invalid behavior routes both `progress` and `ra_stch` through weighted STCH.

Required exact both-active behavior:

    if mode == "equal":
        total = 0.5 * L_D + 0.5 * L_P
    elif mode == "stch":
        total = tau * logsumexp(
            stack([0.5 * L_D / tau, 0.5 * L_P / tau])
        )
    elif mode == "progress":
        w = detached current weights
        total = w[0] * L_D + w[1] * L_P
    elif mode == "ra_stch":
        alpha = detached current weights
        total = tau * logsumexp(
            alpha * stack([L_D, L_P]) / tau
        )

Single-active-task behavior remains unchanged: return the only active task objective directly.

Do not change any task-objective formula, pseudo logic, coefficients, STCH tau, progress weights, or controller update formula.

## 2. Correct RA gradient diagnostic gating

Gradient norms/cosine used by RA MUST update only when both tasks are active in the current batch.

Required behavior:

- if both tasks are active and mode is `ra_stch`:
  - compute detached per-task gradients against `features_audio/features_video`;
  - update grad-norm EMAs;
  - update grad-cosine EMA when valid.
- if fewer than two tasks are active:
  - do NOT update either task grad-norm EMA;
  - do NOT update grad-cosine EMA;
  - set only per-batch diagnostic fields to None if useful;
  - preserve existing EMA state unchanged.
- accepted-pseudo reliability EMA remains independent of this condition:
  - it MAY update per task whenever that task has accepted pseudo entries;
  - it remains detached.
- non-RA modes do not need RA gradient diagnostic updates.

No second-order graph is allowed.

## 3. Required deterministic regression before production

### A. Progress exact-formula regression

Construct fixed scalar task objectives and fixed controller weights, e.g.:

    L_D = 0.7
    L_P = 1.3
    weights = [0.25, 0.75]

Prove exactly/tightly:

    progress_total == 0.25*0.7 + 0.75*1.3

Also prove it is NOT the weighted-STCH reference for the same values.

Then repeat through a real synthetic `ModelOutput + Batch` call so the public loss path is tested, not only a helper.

### B. RA single-active EMA regression

1. create RA loss;
2. run one both-active synthetic batch and capture:
   - grad_norm_ema;
   - grad_cos_ema;
3. run a deterministic single-active-task batch;
4. assert:
   - grad_norm_ema is bitwise/numerically unchanged;
   - grad_cos_ema is unchanged;
   - no invalid/NaN controller state appears.
5. separately verify reliability EMA still updates from accepted pseudo reliability when applicable.

### C. Original contract regressions

Repeat:

- structural detach for pseudo target/reliability;
- equal exact 0.5/0.5;
- static STCH stable reference;
- progress controller lifecycle exact formula;
- RA alpha lifecycle exact formula;
- actual frozen-cache TRAIN-only forward/loss/backward for corrected progress and RA;
- cache SHA/counts unchanged;
- no optimizer step;
- no DEV/Test loader access in smokes.

## 4. Corrected run identity

Before production, change ONLY run_name in configs 15–18 to distinguish corrected runs:

- `r4_progress_seed42_c1`
- `r4_ra_stch_seed42_c1`
- `r4_ra_stch_seed43_c1`
- `r4_ra_stch_seed44_c1`

All seeds/method/data/model/loss/controller constants remain exactly frozen.

Validate configs 15–18.

## 5. Mandatory C1 firewall commit

Before any NEW production run:

1. implement both corrections;
2. apply run_name-only changes to configs 15–18;
3. run all C1 regressions/smokes;
4. update PROGRESS_EN.md with exact commands/results;
5. commit and push one C1 firewall commit.

No corrected production run may start before that firewall commit exists on origin.

## 6. Retained valid controls

Do NOT rerun these:

### Equal seed42

- D `0.712498`
- P `0.843342`
- Mean `0.777920`
- checkpoint SHA `d84d4f107fb5e927491b9f628d6ad9f913c8a7a2dc82689f9149ac32d206f690`

### Static STCH seed42

- D `0.725077`
- P `0.849485`
- Mean `0.787281`
- checkpoint SHA `a1ad7dc4eb4e19e438ce14854d0981ce175e67c75e454c111cf44994f6fb8373`

These remain part of the frozen R4 comparison.

Original Progress/RA seed42 results remain recorded but MUST be labelled invalid/exploratory and excluded from branching.

## 7. Corrected production sequence

After the C1 firewall:

1. run corrected Progress seed42 exactly once;
2. run corrected RA-STCH seed42 exactly once.

Use:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/chimera-ml train \
      --config-path <authorized-config>

No sweep and no rerun for metric improvement.

## 8. Original frozen RA continuation gate

Use corrected RA seed42 and corrected Progress seed42 plus retained Equal/Static controls.

Corrected RA may continue only if ALL are true:

- RA DEV Mean > corrected R3-B seed42 `0.794816`;
- RA depression Score >= `0.685808`;
- RA Parkinson Score >= `0.883824`;
- RA DEV Mean > retained Equal Mean `0.777920`;
- RA DEV Mean > retained Static-STCH Mean `0.787281`;
- RA DEV Mean > corrected Progress Mean.

No Test metric may enter this decision.

If any condition fails: STOP after the two corrected seed42 runs.

## 9. Conditional true RA seeds43/44

Only if Section 8 fully passes:

3. run corrected RA seed43 exactly once;
4. run corrected RA seed44 exactly once.

Use existing same-seed corrected R3-B comparators:

- R3-B seed43 D/P/Mean `0.694349/0.841176/0.767762`
- R3-B seed44 `0.698378/0.838989/0.768684`

Record:

- RA same-seed deltas;
- RA three-seed D/P/Mean mean and sample std;
- three-seed deltas vs R3-B means `0.696178/0.857996/0.777087`;
- whether RA beats same-seed R3-B Mean on all 42/43/44;
- whether RA three-seed Mean > `0.777087`;
- whether neither task mean is >0.010 below the corresponding R3-B mean.

Do not self-promote. Manager decides later.

## 10. Required production evidence

For each NEW corrected run record:

- config and exact command;
- seed;
- run directory;
- MLflow run ID/status/artifact URI;
- epochs completed/early stopping;
- selected epoch/checkpoint based only on DEV;
- full DEV D/P UAR/MF1/Score and Mean;
- same-epoch TEST_NONE/SOFT/HARD monitoring only after DEV selection;
- selected checkpoint SHA256;
- confirmation Test did not affect selection/branching.

For corrected Progress:

- complete controller-weight/progress trajectory or exact path to complete summary;
- verify loss is linear weighted sum in the snapshot/code.

For corrected RA:

- complete alpha/progress/grad-norm/cosine/reliability trajectory or exact path;
- explicitly report number/fraction of training batches with both tasks active if available from added in-memory diagnostic counters ONLY if this can be done without new tracked code outside the loss;
- report that single-active batches did not update grad EMAs.

If RA continues, perform the already-frozen DEV-only calibration audit against same-seed R3-B. Diagnostics only.

## 11. Scope checks

Run:

    git diff --check
    git diff origin/main -- src/audio
    git diff origin/main -- src/video
    git diff origin/main -- src/fusion/models
    git diff origin/main -- src/fusion/data
    git diff origin/main -- src/common
    git status --short
    git diff --stat origin/main...HEAD
    git log -10 --oneline --decorate

For C1 itself, changes after the manager task commit must be limited to the six allowed paths.

## Acceptance criteria

C1 passes only if:

- exact progress linear scalarizer is implemented;
- weighted STCH remains exclusive to `stch` and `ra_stch` as specified;
- RA grad-norm/cosine EMAs do not update on single-active batches;
- reliability EMA behavior remains correct;
- no other objective/controller constants change;
- corrected regression suite passes;
- configs 15–18 differ only by corrected `run_name` from their frozen method settings;
- C1 firewall commit is pushed before corrected production;
- exactly two corrected seed42 runs execute initially;
- corrected RA seeds43/44 execute only if the original continuation gate fully passes;
- at most four NEW production invocations occur;
- Equal/Static are not rerun;
- Test remains monitoring-only;
- no post-hoc tuning;
- selected checkpoint SHA256 is recorded for each new run;
- scope checks pass;
- branch is pushed;
- main/master remains untouched.

Passing C1 closes the implementation validity of the R4 bundle. It does NOT itself authorize Stage 6 or Final Test.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch `codex/task-005f`;
- C1 firewall commit SHA;
- final C1 evidence commit SHA;
- pushed-to-origin status;
- main/master untouched;
- exact code corrections;
- retained Equal/Static metrics and checkpoint SHA;
- corrected Progress seed42 D/P/Mean + checkpoint SHA;
- corrected RA seed42 D/P/Mean + checkpoint SHA;
- RA continuation pass/fail with every condition;
- RA true seeds43/44 results if executed;
- corrected three-seed RA mean/std if available;
- controller diagnostic summary;
- no Test-driven decision;
- no post-hoc tuning;
- R2/R3 prior conclusions preserved;
- no Text/Description/Stage6/7/Final Test;
- Stage 5 remains active pending manager decision.

Stop after C1.
