# TASK-005H-TRIPLE-OPTUNA-C1: Complete the 60-Trial Evidence Ledger and Correct the Handoff

## Authority and branch

This is a narrow evidence-only corrective task after MANAGER-REVIEW-055.

Explicit manager authorization:

- reuse the existing branch:
  `codex/task-005h-triple-optuna`;
- do NOT create a new branch;
- do NOT reset/rebase/force-push;
- preserve existing commits and runtime artifacts.

Required manager base state for this correction is current `origin/main`, containing:

- task definition `9e8dcb3ddd03b65a4b8605752a832a13c450b9ae`;
- manager review `a18b0055dacb4a4125227cd3ebf0dc96f885c1c7`.

The existing task branch already contains:

- firewall `cbfdad1f1d3e8838975b716589e8d168eb677e87`;
- production/evidence `9633fec7ad725abdb095e73cfc6795dec87d4103`;
- top-three summary `c61a587e11318e24b59975d84f4c867582de1009`.

Do not alter those historical commits. Append exactly one corrective evidence commit.

## Goal

Make TASK-005H-TRIPLE-OPTUNA fully auditable in tracked repository evidence without rerunning any experiment.

Use the existing three completed sweep manifests and run artifacts to append the complete 60-trial ledger to `docs/PROGRESS_EN.md`, correct the unauthorized manager wording/status, and return the exact required handoff.

No training or evaluation is authorized.

## Required reading

Read in this exact order:

1. AGENTS.md
2. docs/README.md
3. docs/PROJECT_REQUIREMENTS.md
4. docs/PLAN.md
5. docs/PROGRESS_EN.md through MANAGER-REVIEW-055
6. docs/NEXT_TASK_EN.md

Then inspect read-only:

- existing branch commits:
  - `cbfdad1f1d3e8838975b716589e8d168eb677e87`
  - `9633fec7ad725abdb095e73cfc6795dec87d4103`
  - `c61a587e11318e24b59975d84f4c867582de1009`
- the three existing local sweep manifests:
  - `logs/wsm_mm_pd_dep_v1/_sweeps/candidate-b-optuna-v1-260927-2014-bd81/manifest.yaml`
  - `logs/wsm_mm_pd_dep_v1/_sweeps/r3b-optuna-v1-260927-2256-5990/manifest.yaml`
  - `logs/wsm_mm_pd_dep_v1/_sweeps/r4-ra-stch-optuna-v1-260928-0047-18c6/manifest.yaml`
- their referenced resolved configs/checkpoints/run summaries as needed.

## Allowed tracked files

Modify only:

- `docs/PROGRESS_EN.md`

No other tracked file may change.

## Forbidden actions

- Do not run `chimera-ml train`.
- Do not run `chimera-ml sweep`.
- Do not run any DEV or Test dataloader/evaluation.
- Do not run optimizer steps.
- Do not create new checkpoints.
- Do not modify configs 31–36.
- Do not modify any source file.
- Do not modify search spaces.
- Do not recompute or replace objective winners.
- Do not choose a different trial.
- Do not run seed43/44.
- Do not inspect Test metrics for comparison/selection.
- Do not start Stage6/7, Text/Description, or Final Test.
- Do not claim final promotion/generalization/significance.

Read-only parsing, hashing existing files, and arithmetic consistency checks are allowed.

## 1. Correct the unauthorized wording

In `docs/PROGRESS_EN.md`:

- the existing Codex-authored sentence beginning with
  `Manager conclusion: nominate corrected R4 RA-STCH trial 012...`
  must NOT remain presented as a manager decision;
- either relabel it explicitly as
  `Codex observation (non-authoritative): ...`
  or remove that sentence from the Codex evidence entry;
- do not edit MANAGER-REVIEW-055;
- do not rewrite historical measured metrics.

Also correct the task-status wording so that the authoritative plan state is:

- Stage 5: active/reopened for optimization;
- Stage 6: paused/not started;
- Stage 7: locked;
- Final Test: locked.

Do not falsely state that Stage 5 is complete.

## 2. Append the complete 60-trial tracked ledger

Append one new section:

    ### TASK-005H-TRIPLE-OPTUNA-C1 — complete 60-trial evidence ledger

For every one of the 60 completed trials, record one table row.

Required columns:

- family;
- Optuna trial number/id;
- run name;
- generated/resolved config path;
- exact sampled parameter values;
- selected target epoch;
- DEV depression Score;
- DEV Parkinson Score;
- DEV Mean_Score;
- trainable parameter count;
- selected checkpoint path;
- selected checkpoint SHA256.

If a selected checkpoint SHA/path genuinely cannot be recovered from existing artifacts:

- write `UNAVAILABLE` for that exact field;
- state the precise reason;
- do not fabricate it;
- report how many rows have unavailable checkpoint provenance.

Do not omit lower-ranked trials.

The ledger must contain exactly:

- 20 Candidate B rows;
- 20 corrected R3-B rows;
- 20 corrected R4 RA-STCH rows;
- 60 rows total.

Use the existing manifest trial order/IDs exactly.

## 3. Recompute only read-only audit summaries

From the 60 recorded rows, independently verify and record:

### Candidate B

- completed count = 20;
- objective winner exact trial and D/P/Mean;
- top-5 trial IDs/Means;
- safe count using:
  - Mean > `0.7878268765`
  - D >= `0.7379183895`
  - P >= `0.8177353635`
- safe winner or none;
- strict non-regression count using:
  - Mean > `0.7878268765`
  - D >= `0.7479183895`
  - P >= `0.8277353635`.

### corrected R3-B

Same six checks.

### corrected R4 RA-STCH

Same six checks.

These must reproduce the already recorded results, not redefine them.

Expected frozen summary to verify:

- Candidate B objective winner `bd81-003`:
  D/P/Mean `0.730860 / 0.902321 / 0.8165907903`;
  safe count `0/20`;
  strict count `0/20`.
- R3-B objective winner `5990-018`:
  D/P/Mean `0.732198 / 0.899651 / 0.8159243728`;
  safe count `0/20`;
  strict count `0/20`.
- R4 RA-STCH objective winner `18c6-012`:
  D/P/Mean `0.759025 / 0.879259 / 0.8191424538`;
  safe count `4/20`;
  strict count `1/20`.

If the complete manifest ledger contradicts any expected summary, STOP and report the exact discrepancy. Do not rewrite selection rules.

## 4. Verify artifact identities

Record read-only SHA256 values for:

- frozen audio checkpoint:
  `0873c7cb5e32d415cdd301058949f5dd140c16b874cc5230d027a4ee33e3daf2`;
- R4 pseudo cache:
  `17cf5e67e8c244d81b7c21f0842988966a23d83d69e296f7c5a5181376d6b945`.

For the 60 selected checkpoints, hash existing checkpoint files when available.

Do not load Test metrics.

## 5. Verify frozen search provenance

Record that the search spaces remain exactly:

- Candidate B: 6 variables;
- R3-B: 7 variables;
- R4 RA-STCH: 10 variables.

Record:

- firewall commit:
  `cbfdad1f1d3e8838975b716589e8d168eb677e87`;
- production evidence commit:
  `9633fec7ad725abdb095e73cfc6795dec87d4103`;
- prior summary commit:
  `c61a587e11318e24b59975d84f4c867582de1009`.

Verify by read-only git history that firewall precedes production.

## 6. Scope verification

Run:

    git diff --check
    git diff origin/main -- src/audio
    git diff origin/main -- src/video
    git diff origin/main -- src/fusion/models
    git diff origin/main -- src/fusion/loss
    git diff origin/main -- src/fusion/data
    git diff origin/main -- src/common
    git diff origin/main -- src/chimera_plugin.py
    git diff origin/main -- configs/wsm_mm_pd_dep_v1/fusion/31_optuna_candidate_b_base.yaml
    git diff origin/main -- configs/wsm_mm_pd_dep_v1/fusion/32_optuna_candidate_b_sweep.yaml
    git diff origin/main -- configs/wsm_mm_pd_dep_v1/fusion/33_optuna_r3b_base.yaml
    git diff origin/main -- configs/wsm_mm_pd_dep_v1/fusion/34_optuna_r3b_sweep.yaml
    git diff origin/main -- configs/wsm_mm_pd_dep_v1/fusion/35_optuna_r4_ra_stch_base.yaml
    git diff origin/main -- configs/wsm_mm_pd_dep_v1/fusion/36_optuna_r4_ra_stch_sweep.yaml
    git status --short
    git diff --stat origin/main...HEAD
    git log -12 --oneline --decorate

Expected:

- no source changes beyond the already existing authorized branch history;
- C1 itself changes only `docs/PROGRESS_EN.md`;
- configs 31–36 remain byte-identical to the existing firewall commit;
- no uncommitted tracked changes at handoff.

## 7. Commit and push

Create exactly one new corrective commit on the same branch.

Commit message:

    TASK-005H-TRIPLE-OPTUNA-C1: complete trial evidence ledger

Push normally to:

    origin/codex/task-005h-triple-optuna

No force push.

## Acceptance criteria

C1 passes only if:

- no training/evaluation/sweep ran;
- only PROGRESS_EN.md changed in the C1 commit;
- complete 60-row ledger is tracked;
- exact 20/20/20 counts are visible in tracked evidence;
- all available selected checkpoint paths/SHA256 values are recorded;
- missing checkpoint provenance, if any, is explicitly counted and explained;
- objective/safe/strict summaries independently reproduce the frozen results;
- unauthorized Codex `Manager conclusion` wording is corrected;
- Stage 5 status is corrected to active/reopened;
- source/config/search-space semantics remain unchanged;
- exact required six-section handoff is complete;
- branch is pushed;
- main/master remain untouched by Codex.

## Required handoff

Respond in English using exactly:

1. Outcome
2. Changed files
3. Verification
4. Plan status
5. Blockers and risks
6. Next atomic step

Explicitly include:

- branch `codex/task-005h-triple-optuna`;
- original firewall SHA `cbfdad1f1d3e8838975b716589e8d168eb677e87`;
- production SHA `9633fec7ad725abdb095e73cfc6795dec87d4103`;
- previous summary SHA `c61a587e11318e24b59975d84f4c867582de1009`;
- new C1 evidence commit SHA;
- pushed status;
- main/master untouched;
- C1 changed only PROGRESS_EN.md;
- source diffs empty;
- config 31–36 unchanged;
- exact 20/20/20 and 60 total ledger counts;
- number of checkpoint paths/SHA256 recovered vs unavailable;
- frozen audio SHA;
- pseudo-cache SHA;
- exact 6/7/10 search-space counts;
- objective winners with D/P/Mean;
- safe counts/winners;
- strict counts;
- top-5 IDs/Means per family;
- no Test inspection;
- no training/sweep/evaluation;
- no seed43/44;
- no post-hoc search-space/model/loss changes;
- Stage 5 active/reopened;
- Stage 6 paused/not started;
- Stage 7 and Final Test locked.

For Next atomic step write only:

    Manager review of TASK-005H-TRIPLE-OPTUNA-C1; do not start another task.

Stop after C1.
